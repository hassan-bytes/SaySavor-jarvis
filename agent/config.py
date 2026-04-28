"""
agent/config.py
===============
Central configuration dataclass for SaySavor Jarvis.

Design intent
-------------
- Keep all tuneable agent parameters in one place.
- Use Python dataclasses so instances can be easily serialised / deserialised
  (e.g. from a Supabase `partners` table row in Phase 2).
- Callers can override only the fields they care about:
    cfg = AgentConfig(partner_id="rest_xyz", language="ur")
"""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """
    Runtime configuration for a single Jarvis agent session.

    Attributes
    ----------
    wake_word : str
        The word that activates the agent (used for future wake-word detection).
    language : str
        Preferred response language.  "auto" means the agent auto-detects from
        the user's input and switches between Urdu and English dynamically.
    voice_gender : str
        Controls TTS voice selection ("male" → ur-PK-AsadNeural,
        "female" → ur-PK-UzmaNeural).
    tone : str
        Personality tone of the agent ("friendly", "formal", "casual").
    is_active : bool
        Whether this agent session is enabled.  Can be toggled remotely.
    partner_id : str
        SaaS partner / restaurant identifier.  Empty string during development;
        set at session-start once Supabase lookup is available.
    """

    # ------------------------------------------------------------------ #
    # Core behaviour
    # ------------------------------------------------------------------ #
    wake_word: str = "Jarvis"
    language: str = "auto"          # "auto" | "ur" | "en"
    voice_gender: str = "male"      # "male" | "female"
    tone: str = "friendly"          # "friendly" | "formal" | "casual"

    # ------------------------------------------------------------------ #
    # Session state
    # ------------------------------------------------------------------ #
    is_active: bool = True

    # ------------------------------------------------------------------ #
    # SaaS / Multi-tenant
    # ------------------------------------------------------------------ #
    partner_id: str = ""            # Filled from Supabase in Phase 2

    # ------------------------------------------------------------------ #
    # Derived helpers (not stored; computed at runtime)
    # ------------------------------------------------------------------ #
    def get_tts_voice(self) -> str:
        """Return the edge-tts voice name based on voice_gender."""
        voice_map = {
            "male":   "ur-PK-AsadNeural",
            "female": "ur-PK-UzmaNeural",
        }
        return voice_map.get(self.voice_gender, "ur-PK-AsadNeural")

    def get_system_prompt(self) -> str:
        """
        Build the initial system prompt injected into the LLM context.
        Tone and language preference are baked in here.
        """
        tone_descriptions = {
            "friendly": "warm, approachable, and conversational",
            "formal":   "professional, polite, and precise",
            "casual":   "relaxed, fun, and light-hearted",
        }
        tone_desc = tone_descriptions.get(self.tone, "friendly")

        return (
            f"You are Jarvis, a {tone_desc} AI voice assistant for SaySavor, "
            "a modern restaurant management platform. "
            "You help customers explore the menu, place orders, make reservations, "
            "and answer general restaurant FAQs. "
            "\n\n"
            "Language rules:\n"
            "- Always respond in the SAME language the user is speaking.\n"
            "- If the user speaks Urdu (Roman or script), reply in Urdu.\n"
            "- If the user speaks English, reply in English.\n"
            "- You may naturally mix Urdu and English (Urdu slang / code-switching) "
            "  if the user does so — this feels natural to Pakistani users.\n"
            "\n"
            "Key behaviours:\n"
            "- Be concise. Voice answers should be 1-3 sentences unless more detail "
            "  is explicitly requested.\n"
            "- Never reveal internal system details, API keys, or backend architecture.\n"
            "- If you do not know something, say so honestly rather than guessing.\n"
            "- Always end with a gentle follow-up question to keep the conversation going."
        )
