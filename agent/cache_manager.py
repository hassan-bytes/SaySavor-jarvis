"""
In-memory per-restaurant context cache.
Eliminates repeated Supabase round-trips for stable data (menu items, tables).
TTL = 5 minutes. Write-through on ADD_MENU_ITEM. Invalidated on mutations.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

CACHE_TTL_SECONDS = 300  # 5 minutes


@dataclass
class _RestaurantSnapshot:
    restaurant_id: str
    context: dict = field(default_factory=dict)    # raw dict from get_partner_context
    menu_items: list = field(default_factory=list)  # full menu list (with category flattened)
    last_sync: Optional[datetime] = None

    def is_stale(self) -> bool:
        if not self.last_sync:
            return True
        return (datetime.now(timezone.utc) - self.last_sync).total_seconds() > CACHE_TTL_SECONDS

    def age_seconds(self) -> float:
        if not self.last_sync:
            return float("inf")
        return (datetime.now(timezone.utc) - self.last_sync).total_seconds()

    def load(self, ctx: dict):
        self.context = ctx
        self.menu_items = ctx.get("menu_items", [])
        self.last_sync = datetime.now(timezone.utc)


class CacheManager:
    """Thread-safe (asyncio.Lock) per-restaurant in-memory cache.

    Usage:
        ctx = await cache_manager.get_context(restaurant_id, get_partner_context)
        items = cache_manager.get_menu_items(restaurant_id)  # None = cache miss, call DB
        cache_manager.on_menu_item_added(restaurant_id, new_item)
        cache_manager.invalidate(restaurant_id)
    """

    def __init__(self):
        self._store: dict[str, _RestaurantSnapshot] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, rid: str) -> asyncio.Lock:
        if rid not in self._locks:
            self._locks[rid] = asyncio.Lock()
        return self._locks[rid]

    async def get_context(
        self,
        restaurant_id: str,
        loader: Callable[[str], Coroutine[Any, Any, dict]],
    ) -> dict:
        """Return cached context; reload from DB via `loader` if stale (> 5 min)."""
        snap = self._store.get(restaurant_id)
        if snap and not snap.is_stale():
            age = int(snap.age_seconds())
            print(f"[Cache] HIT {restaurant_id[:8]} — {len(snap.menu_items)} menu items, age={age}s")
            return snap.context

        async with self._lock(restaurant_id):
            # Double-check after acquiring lock (another coroutine may have refreshed)
            snap = self._store.get(restaurant_id)
            if snap and not snap.is_stale():
                return snap.context

            print(f"[Cache] MISS {restaurant_id[:8]} — loading from Supabase…")
            ctx = await loader(restaurant_id)
            if not snap:
                snap = _RestaurantSnapshot(restaurant_id=restaurant_id)
                self._store[restaurant_id] = snap
            snap.load(ctx)
            print(
                f"[Cache] STORED {restaurant_id[:8]} — "
                f"{len(snap.menu_items)} items, {ctx.get('total_tables', 0)} tables"
            )
            return ctx

    def get_menu_items(self, restaurant_id: str) -> Optional[list]:
        """Return cached menu items list, or None if stale/missing (caller must hit DB)."""
        snap = self._store.get(restaurant_id)
        if snap and not snap.is_stale() and snap.menu_items:
            return snap.menu_items
        return None

    def invalidate(self, restaurant_id: str):
        """Force the next get_context call to reload from DB."""
        if restaurant_id in self._store:
            self._store[restaurant_id].last_sync = None
            print(f"[Cache] INVALIDATED {restaurant_id[:8]}")

    def invalidate_all(self):
        for snap in self._store.values():
            snap.last_sync = None

    # ── Write-through helpers ──────────────────────────────────────────────────

    def on_menu_item_added(self, restaurant_id: str, item: dict):
        """Append new item to cache instantly after a successful DB insert."""
        snap = self._store.get(restaurant_id)
        if snap and not snap.is_stale():
            snap.menu_items.append(item)
            snap.context["menu_items"] = snap.menu_items
            snap.context["total_menu_items"] = len(snap.menu_items)
            print(f"[Cache] Write-through: '{item.get('name')}' added to {restaurant_id[:8]}")

    def stats(self) -> dict:
        return {
            rid: {
                "menu_items": len(s.menu_items),
                "age_s": int(s.age_seconds()),
                "stale": s.is_stale(),
            }
            for rid, s in self._store.items()
        }


# Module-level singleton — import this everywhere
cache_manager = CacheManager()
