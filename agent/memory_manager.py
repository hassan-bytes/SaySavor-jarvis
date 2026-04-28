"""
agent/memory_manager.py
========================
Conversation memory and session state manager for SaySavor Jarvis.

Responsibilities
----------------
- Maintain a sliding window of the last N conversation turns to stay within
  Groq's token context limit (32k for Llama-3.3-70B, 6k for Whisper).
- Track session metadata: session_id, partner_id, user_id, and cart state.
- Provide a clean `get_context_messages()` interface for LLM context assembly.
- Inject dynamic runtime context (cart, time, user name) into the system prompt
  so the LLM always has up-to-date situational awareness.

Design decisions
----------------
- Pure Python dataclass — no external DB calls in this class.  Persistence to
  Supabase is handled separately by a future `supabase_store.py` module.
- Thread-safe via `asyncio.Lock` so multiple coroutines can call `add_message`
  concurrently without race conditions (relevant if tool-use callbacks fire in
  parallel with the main pipeline).
- Cart is a dict[str, int] mapping item name → quantity for simplicity.
  Callers can replace this with a richer CartItem dataclass in Phase 3.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("saysavor.memory_manager")


# ==============================================================================
# Type aliases
# ==============================================================================

# A single LLM-ready message dict  {"role": "user"|"assistant"|"system", "content": str}
MessageDict = dict[str, str]

# Cart: item_name → quantity
CartDict = dict[str, int]


# ==============================================================================
# CartItem — Rich cart entry (used in Phase 3 billing/order flow)
# ==============================================================================

@dataclass
class CartItem:
    """
    Represents a single line item in the customer's cart.

    Attributes
    ----------
    name : str
        Human-readable item name, e.g. "Chicken Karahi".
    quantity : int
        Number of units.
    unit_price : float
        Price per unit in PKR.
    notes : str
        Special instructions, e.g. "extra spicy".
    """
    name:       str
    quantity:   int   = 1
    unit_price: float = 0.0
    notes:      str   = ""

    @property
    def subtotal(self) -> float:
        """Total price for this line item."""
        return self.quantity * self.unit_price

    def __str__(self) -> str:
        note_suffix = f" ({self.notes})" if self.notes else ""
        return f"{self.quantity}x {self.name}{note_suffix} @ PKR {self.unit_price:.0f}"


# ==============================================================================
# SessionMemory — Core Memory Manager
# ==============================================================================

class SessionMemory:
    """
    Manages per-session conversation history and state for Jarvis.

    Parameters
    ----------
    max_turns : int
        Maximum number of conversation turns (user + assistant pairs) to retain
        in the sliding window.  Default 15 ≈ ~3k tokens, well within Groq's
        32k context limit for Llama-3.3-70B.
    session_id : str | None
        Unique identifier for this session.  Auto-generated as UUID4 if None.
    partner_id : str | None
        Restaurant/partner identifier.  Set from Supabase lookup on session start.
    user_id : str | None
        Customer identifier.  Set after user authentication or Supabase lookup.
    user_name : str | None
        Display name of the authenticated user.

    Thread safety
    -------------
    `add_message` and `clear_history` acquire `_lock` (asyncio.Lock) to prevent
    race conditions when multiple coroutines write concurrently.
    """

    def __init__(
        self,
        *,
        max_turns:  int           = 15,
        session_id: Optional[str] = None,
        partner_id: Optional[str] = None,
        user_id:    Optional[str] = None,
        user_name:  Optional[str] = None,
    ) -> None:
        # Unique session identifier
        self.session_id: str = session_id or str(uuid.uuid4())

        # SaaS / multi-tenant identifiers
        self.partner_id: Optional[str] = partner_id   # Restaurant owner
        self.user_id:    Optional[str] = user_id       # Customer
        self.user_name:  Optional[str] = user_name     # Display name

        # Cart: simple name → quantity mapping (upgraded to CartItem in Phase 3)
        self.cart: CartDict = {}

        # Rich cart items (parallel structure; used for pricing)
        self._cart_items: dict[str, CartItem] = {}

        # Sliding window of raw message dicts (not including the system prompt)
        # deque automatically evicts oldest messages when maxlen is exceeded,
        # providing O(1) append and eviction with zero manual management.
        self._history: deque[MessageDict] = deque(maxlen=max_turns * 2)
        # × 2 because each turn = 1 user message + 1 assistant message

        # Async lock for thread-safe history writes
        self._lock: asyncio.Lock = asyncio.Lock()

        # Session start timestamp (UTC)
        self._started_at: datetime = datetime.now(tz=timezone.utc)

        # Arbitrary key-value store for plugin-injected metadata
        # e.g. {"detected_language": "ur", "table_number": "7"}
        self.metadata: dict[str, Any] = {}

        logger.info(
            "SessionMemory created | session_id=%s partner=%s user=%s max_turns=%d",
            self.session_id,
            self.partner_id or "—",
            self.user_id    or "—",
            max_turns,
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    async def add_message(self, role: str, content: str) -> None:
        """
        Append a message to the conversation history.

        Parameters
        ----------
        role : str
            "user" | "assistant" | "tool"
        content : str
            The message text.

        Notes
        -----
        The deque enforces the max-turn window automatically.
        Only `user` and `assistant` roles increment the turn count.
        """
        if role not in ("user", "assistant", "system", "tool"):
            logger.warning("Unexpected message role '%s' — appending anyway.", role)

        message: MessageDict = {"role": role, "content": content}

        async with self._lock:
            self._history.append(message)
            logger.debug(
                "Message added | role=%s | len=%d | text=%.50s…",
                role,
                len(self._history),
                content,
            )

    def add_message_sync(self, role: str, content: str) -> None:
        """
        Synchronous variant of `add_message` for use in non-async callbacks.

        Warning: not coroutine-safe.  Prefer `add_message` wherever possible.
        """
        message: MessageDict = {"role": role, "content": content}
        self._history.append(message)

    async def clear_history(self) -> None:
        """Remove all conversation history (keeps session metadata and cart)."""
        async with self._lock:
            self._history.clear()
            logger.info("Conversation history cleared for session %s", self.session_id)

    # ------------------------------------------------------------------
    # LLM context assembly
    # ------------------------------------------------------------------

    def get_context_messages(self) -> list[MessageDict]:
        """
        Return the conversation history as a list of dicts suitable for the LLM.

        The list is a copy of the internal deque — callers can safely mutate it
        without affecting stored state.

        Returns
        -------
        list[dict[str, str]]
            e.g. [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}, …]
        """
        return list(self._history)

    def get_recent_turns(self, n: int) -> list[MessageDict]:
        """Return the last n messages from history (all roles).

        Used to inject conversation context into the LLM prompt without
        including the full history.

        Parameters
        ----------
        n : int
            Maximum number of messages to return (not turn pairs, individual messages).

        Returns
        -------
        list[dict[str, str]]
            The n most recent messages in chronological order.
        """
        msgs = list(self._history)
        return msgs[-n:] if len(msgs) > n else msgs

    # ------------------------------------------------------------------
    # Cart management
    # ------------------------------------------------------------------

    def add_to_cart(
        self,
        item_name: str,
        quantity:  int   = 1,
        unit_price: float = 0.0,
        notes:     str   = "",
    ) -> None:
        """
        Add or increment an item in the cart.

        If the item already exists, its quantity is incremented and any new
        `notes` or `unit_price` are applied.
        """
        if item_name in self._cart_items:
            self._cart_items[item_name].quantity += quantity
            if notes:
                self._cart_items[item_name].notes = notes
        else:
            self._cart_items[item_name] = CartItem(
                name=item_name,
                quantity=quantity,
                unit_price=unit_price,
                notes=notes,
            )

        # Keep the simple cart dict in sync for backward compatibility
        self.cart[item_name] = self._cart_items[item_name].quantity
        logger.info("Cart updated: %s", self.get_cart_summary())

    def remove_from_cart(self, item_name: str, quantity: int = 1) -> bool:
        """
        Decrement or remove an item from the cart.

        Returns True if the item was found and modified, False otherwise.
        """
        if item_name not in self._cart_items:
            logger.warning("Tried to remove '%s' but it's not in the cart.", item_name)
            return False

        self._cart_items[item_name].quantity -= quantity
        if self._cart_items[item_name].quantity <= 0:
            del self._cart_items[item_name]
            self.cart.pop(item_name, None)
        else:
            self.cart[item_name] = self._cart_items[item_name].quantity

        logger.info("Cart after removal: %s", self.get_cart_summary())
        return True

    def clear_cart(self) -> None:
        """Empty the cart entirely."""
        self._cart_items.clear()
        self.cart.clear()
        logger.info("Cart cleared for session %s", self.session_id)

    def get_cart_total(self) -> float:
        """Return the total cart value in PKR."""
        return sum(item.subtotal for item in self._cart_items.values())

    def get_cart_summary(self) -> str:
        """
        Return a human-readable, LLM-injectable cart summary string.

        Example output:
            "2x Chicken Karahi @ PKR 650, 1x Naan @ PKR 30 | Total: PKR 1,330"
        """
        if not self._cart_items:
            return "Cart is empty."

        line_items = ", ".join(str(item) for item in self._cart_items.values())
        total = self.get_cart_total()
        total_str = f" | Total: PKR {total:,.0f}" if total > 0 else ""
        return f"{line_items}{total_str}"

    # ------------------------------------------------------------------
    # Dynamic context injection for system prompt
    # ------------------------------------------------------------------

    def build_dynamic_context(self) -> str:
        """
        Compose a compact runtime-context string to be injected into the
        system prompt at each LLM call.

        This gives the LLM live situational awareness without storing it in
        the chat history (which would bloat the token count).

        Returns
        -------
        str
            A multi-line string block ready to be appended to the system prompt.

        Example output
        --------------
            --- Live Session Context ---
            Session ID : abc123
            User       : Ahmed (ID: usr_456)
            Restaurant : partner_789
            Time       : 8:15 PM PKT
            Cart       : 2x Chicken Karahi @ PKR 650 | Total: PKR 1,300
            ---
        """
        now_pkt = datetime.now(tz=timezone.utc).astimezone(
            # Pakistan Standard Time (UTC+5) — no DST
            timezone.utc.__class__.utc  # fallback; replace with zoneinfo in Python 3.9+
        )
        # Format time as human-readable 12-hour clock
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            pkt = ZoneInfo("Asia/Karachi")
            now_pkt = datetime.now(tz=pkt)
        except (ImportError, KeyError):
            # Fallback to UTC if zoneinfo not available
            now_pkt = datetime.now(tz=timezone.utc)

        # Windows-compatible strftime (no %-I directive)
        try:
            time_str = now_pkt.strftime("%I:%M %p PKT").lstrip("0")
        except Exception:
            time_str = "N/A"

        user_str = (
            f"{self.user_name} (ID: {self.user_id})"
            if self.user_name and self.user_id
            else self.user_id or "Guest"
        )

        cart_str = self.get_cart_summary()

        lines = [
            "--- Live Session Context ---",
            f"Session ID : {self.session_id}",
            f"User       : {user_str}",
            f"Restaurant : {self.partner_id or 'Not set'}",
            f"Time       : {time_str}",
            f"Cart       : {cart_str}",
        ]

        # Inject any extra plugin metadata (e.g. table number, loyalty points)
        for key, value in self.metadata.items():
            lines.append(f"{key:<11}: {value}")

        lines.append("---")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Repr / debugging helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SessionMemory(session_id={self.session_id!r}, "
            f"turns={len(self._history)}, "
            f"cart_items={len(self._cart_items)})"
        )

    @property
    def turn_count(self) -> int:
        """Number of individual messages (not pairs) in history."""
        return len(self._history)
