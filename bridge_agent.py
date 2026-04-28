import os
import json
import asyncio
import tempfile
import uuid
from urllib.parse import quote
from fastapi import FastAPI, UploadFile, File, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from deepgram import DeepgramClient
from google import genai
import groq as groq_lib
import edge_tts
from dotenv import load_dotenv

from agent.prompts import build_partner_prompt, build_customer_prompt
from agent.memory_manager import SessionMemory
from agent.config import AgentConfig
from agent.partner_tools import execute_partner_tool, get_partner_context
from agent.customer_tools import execute_customer_tool, get_customer_context
from agent.cache_manager import cache_manager

load_dotenv()

if not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
    print(
        "\n[WARNING] ══════════════════════════════════════════════════════\n"
        "[WARNING] SUPABASE_SERVICE_ROLE_KEY is NOT set in .env\n"
        "[WARNING] orders + restaurant_tables will return 0 rows (RLS blocks anon key)\n"
        "[WARNING] Fix: Supabase Dashboard → Project Settings → API → service_role → paste in .env\n"
        "[WARNING] ══════════════════════════════════════════════════════\n"
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dg_client    = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client   = groq_lib.Groq(api_key=os.getenv("GROQ_API_KEY"))

# Valid Gemini models — tried in order, skipped on 429/404.
# gemini-1.5-* models return 404 with the google.genai v1beta SDK — do not add them back.
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# Groq models — fallback when all Gemini quotas are exhausted.
# DECOMMISSIONED (do NOT add back): llama3-8b-8192, mixtral-8x7b-32768, gemma2-9b-it,
#   llama3-groq-70b-8192-tool-use-preview, deepseek-r1-distill-llama-70b, qwen-qwq-32b
GROQ_MODELS = [
    "llama-3.3-70b-versatile",                       # best quality — often rate-limited on free tier
    "meta-llama/llama-4-scout-17b-16e-instruct",      # Llama 4 medium — fast + smart
    "llama-3.1-70b-versatile",                        # older 70B — independent daily limit
    "llama-3.1-8b-instant",                           # always-on fallback
]

# Models that support response_format={"type":"json_object"} on Groq
JSON_MODE_MODELS = {
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
}

agent_config = AgentConfig(
    partner_id=os.getenv("PARTNER_ID", "default"),
    language="auto",
    tone="friendly"
)

# Per-session memory: keyed on restaurant_id (partner) or user_id (customer)
# Each restaurant/user gets their own conversation history
_session_memories: dict[str, SessionMemory] = {}


def _get_session_memory(key: str) -> SessionMemory:
    if key not in _session_memories:
        _session_memories[key] = SessionMemory(partner_id=key)
    return _session_memories[key]


VOICE_MAP = {
    "ur": {"female": "ur-PK-UzmaNeural", "male": "ur-PK-AsadNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "ar": {"female": "ar-SA-ZariyahNeural", "male": "ar-SA-HamedNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "tr": {"female": "tr-TR-EmelNeural", "male": "tr-TR-AhmetNeural"},
    "it": {"female": "it-IT-ElsaNeural", "male": "it-IT-DiegoNeural"},
}


def _strip_json_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    elif s.startswith("```"):
        s = s.split("```", 1)[1].rsplit("```", 1)[0].strip()
    return s


def _select_tts_voice(lang_code: str, tts_text: str, gender: str) -> str:
    """Choose the Edge-TTS voice. If tts_text contains Urdu/Arabic script
    we always force an Urdu voice — passing Urdu script to a Hindi neural
    model returns silent/empty audio."""
    effective_lang = lang_code
    if any('؀' <= c <= 'ۿ' for c in tts_text):
        effective_lang = "ur"
    lang_voices = VOICE_MAP.get(effective_lang, VOICE_MAP["en"])
    return lang_voices.get(gender, lang_voices["female"])


def _build_data_summary(action: str, data: dict, lang_code: str) -> str:
    """Convert raw tool result data into compact, human-readable text for the follow-up LLM call.
    Keeps token count low and responses focused on actual numbers/names."""
    lang = lang_code if lang_code in ("ur", "en") else "ur"

    if action == "GET_MENU":
        # Partner GET_MENU returns data.menu_items
        items = data.get("menu_items", [])
        count = len(items)
        if not items:
            return "No menu items found." if lang == "en" else "Koi menu item nahi mila."
        available = [i for i in items if i.get("is_available", True) or i.get("available", True)]
        unavailable = [i for i in items if not (i.get("is_available", True) or i.get("available", True))]
        lines = []
        for item in items[:20]:
            name = item.get("name", "?")
            current_price = item.get("price", 0)
            original_price = item.get("original_price")
            disc = float(item.get("discount_percentage") or 0)
            cat = item.get("category", "")
            avail_mark = "" if (item.get("is_available", True) or item.get("available", True)) else " [unavailable]"
            if disc > 0 and original_price:
                lines.append(f"  {name} ({cat}): Rs.{original_price} → Rs.{current_price} ({int(disc)}% OFF){avail_mark}")
            else:
                lines.append(f"  {name} ({cat}): Rs.{current_price}{avail_mark}")
        tail = f"\n  ...and {count - 20} more" if count > 20 else ""
        return (f"Menu — {count} total items ({len(available)} available, {len(unavailable)} unavailable):\n"
                + "\n".join(lines) + tail)

    elif action == "GET_REVENUE":
        rev = data.get("revenue", 0)
        period = data.get("period", "today")
        return f"Revenue ({period}): Rs.{rev}"

    elif action == "GET_SUMMARY":
        rev = data.get("today_revenue", 0)
        orders = data.get("today_orders", 0)
        active = data.get("active_orders", 0)
        return f"Today summary — Orders: {orders}, Revenue: Rs.{rev}, Active right now: {active}"

    elif action in ("GET_ACTIVE_ORDERS", "LIST_ORDERS"):
        orders = data.get("orders", [])
        count = data.get("count", len(orders))
        if not orders:
            return "No orders." if lang == "en" else "Koi order nahi."
        lines = []
        for o in orders[:12]:
            oid = (o.get("id") or "")[-6:]
            status = o.get("status", "")
            total = float(o.get("total_amount") or 0)
            table = o.get("table_number", "?")
            name = o.get("customer_name") or ""
            lines.append(f"  #{oid} Table-{table}{' '+name if name else ''}: {status}, Rs.{total}")
        tail = f"\n  ...and {count - 12} more" if count > 12 else ""
        return f"{count} orders:\n" + "\n".join(lines) + tail

    elif action == "GET_TOP_ITEMS":
        top = data.get("top_items", [])
        if not top:
            return "No top items data."
        lines = [f"  {i+1}. {t['name']}: {t['count']} orders" for i, t in enumerate(top[:10])]
        return "Top selling items:\n" + "\n".join(lines)

    elif action == "GET_PEAK":
        peaks = data.get("peak_hours", [])
        if not peaks:
            return "No peak hours data."
        lines = [f"  {p['hour']}: {p['orders']} orders" for p in peaks[:5]]
        return "Busiest hours:\n" + "\n".join(lines)

    elif action == "GET_INSIGHTS":
        avg = data.get("avg_order_value", 0)
        unique = data.get("unique_customers", 0)
        total = data.get("total_orders", 0)
        return f"This week: {total} orders, {unique} unique customers, average order Rs.{avg}"

    elif action in ("GET_CHART", "GET_VOLUME"):
        trend = data.get("trend") or data.get("volume", [])
        if not trend:
            return "No trend data."
        lines = [f"  {t.get('date','?')}: {t.get('revenue') or t.get('orders','?')}" for t in trend[-7:]]
        return ("Revenue trend (last 7 days):" if action == "GET_CHART" else "Order volume (last 7 days):") \
               + "\n" + "\n".join(lines)

    elif action == "GET_TABLES":
        tables = data.get("tables", [])
        count = len(tables)
        return f"QR Tables: {count} tables — Table numbers: {', '.join(str(t) for t in tables)}"

    elif action in ("ADD_TABLE", "DELETE_TABLE"):
        return data.get("summary", json.dumps(data, ensure_ascii=False)[:200])

    elif action in ("ADD_MENU_ITEM", "UPDATE_MENU_BY_NAME", "RENAME_ITEM", "TOGGLE_ITEM", "APPLY_DISCOUNT"):
        name = data.get("name", "item")
        if action == "ADD_MENU_ITEM":
            return f"New item '{name}' added at Rs.{data.get('price', 0)}"
        if action == "APPLY_DISCOUNT":
            return (f"'{name}' — {data.get('discount_pct', 0)}% discount applied: "
                    f"Rs.{data.get('original_price',0)} → Rs.{data.get('discounted_price',0)}")
        if action == "TOGGLE_ITEM":
            status = "available" if data.get("is_available") else "unavailable"
            return f"'{name}' is now {status}"
        field = data.get("field", "")
        value = data.get("value", "")
        return f"'{name}' updated: {field} = {value}"

    elif action == "ADD_CART":
        name = data.get("item_name") or data.get("name", "item")
        price = data.get("price", 0)
        qty = data.get("qty", 1)
        if data.get("error"):
            return f"Item not found: {data['error']}"
        return f"{qty}x '{name}' (Rs.{int(float(price))}) cart mein add ho gaya. Ab bolo: koi aur item chahiye, ya order confirm karoon?"

    elif action == "UPDATE_ORDER":
        oid = (data.get("order_id") or "")[-6:]
        status = data.get("new_status", "")
        return f"Order #{oid} status → {status}"

    elif action == "SEARCH_MENU":
        items = data.get("items", data.get("menu_items", []))
        query = data.get("query", "")
        count = len(items)
        if not items:
            return f"No items found for '{query}'." if lang == "en" else f"'{query}' nahi mila menu mein."
        lines = []
        for item in items[:15]:
            name = item.get("name", "?")
            current_price = item.get("price", 0)
            original_price = item.get("original_price")
            disc = float(item.get("discount_percentage") or 0)
            cat = item.get("category", "")
            avail = "" if item.get("is_available", True) else " [unavailable]"
            if disc > 0 and original_price:
                lines.append(f"  {name} ({cat}): Rs.{original_price} → Rs.{current_price} ({int(disc)}% OFF){avail}")
            else:
                lines.append(f"  {name} ({cat}): Rs.{current_price}{avail}")
        tail = f"\n  ...and {count - 15} more" if count > 15 else ""
        return f"Search results for '{query}' — {count} item(s):\n" + "\n".join(lines) + tail

    elif action == "GET_DEALS":
        items = data.get("deals", data.get("items", []))
        count = len(items)
        if not items:
            return "No active deals." if lang == "en" else "Abhi koi deal nahi hai."
        lines = []
        for item in items[:10]:
            name = item.get("name", "?")
            price = item.get("price", 0)
            original = item.get("original_price")
            disc = float(item.get("discount_percentage") or 0)
            label = item.get("offer_name") or f"{int(disc)}% OFF"
            if original:
                lines.append(f"  {name}: Rs.{original} → Rs.{price} — {label}")
            else:
                lines.append(f"  {name}: Rs.{price} — {label}")
        tail = f"\n  ...and {count - 10} more deals" if count > 10 else ""
        return f"Best deals — {count} offers:\n" + "\n".join(lines) + tail

    elif action == "BUDGET_SUGGEST":
        budget = data.get("budget", 0)
        total = data.get("total_in_budget", 0)
        suggestions = data.get("suggestions", [])
        if not suggestions:
            return f"No items found within Rs.{budget} budget." if lang == "en" else f"Rs.{budget} mein koi item nahi mila."
        lines = []
        for item in suggestions[:8]:
            name = item.get("name", "?")
            price = item.get("price", 0)
            disc = float(item.get("discount_percentage") or 0)
            tag = f" ({int(disc)}% OFF)" if disc > 0 else ""
            lines.append(f"  {name}: Rs.{price}{tag}")
        return (f"Budget Rs.{budget} — {total} item(s) available:\n" + "\n".join(lines))

    # Fallback: compact JSON
    return json.dumps(data, ensure_ascii=False)[:400]


async def _gemini_generate(prompt: str) -> str:
    """Try Gemini models in order; skip on 429 quota or 404 not-found errors."""
    last_err: Exception | None = None
    for model in GEMINI_MODELS:
        try:
            response = gemini_client.models.generate_content(model=model, contents=prompt)
            if model != GEMINI_MODELS[0]:
                print(f"[Gemini] Using fallback model: {model}")
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if any(x in err_str for x in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND")):
                print(f"[Gemini] {model} unavailable ({err_str[:60]}), trying next...")
                last_err = e
                continue
            raise
    raise last_err or Exception("All Gemini models exhausted")


async def _groq_generate(prompt: str) -> str:
    """Call Groq as fallback when all Gemini quota is exhausted."""
    last_err: Exception | None = None
    for model in GROQ_MODELS:
        try:
            kwargs: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }
            if model in JSON_MODE_MODELS:
                kwargs["response_format"] = {"type": "json_object"}
            resp = await asyncio.to_thread(groq_client.chat.completions.create, **kwargs)
            print(f"[Groq] Using model: {model}")
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            # Skip rate-limited or decommissioned models
            if any(x in err_str for x in ("429", "rate_limit", "400", "decommissioned", "deprecated")):
                print(f"[Groq] {model} skipped ({err_str[:80]}), trying next...")
            else:
                print(f"[Groq] {model} error: {err_str[:80]}, trying next...")
            last_err = e
            continue
    raise last_err or Exception("All Groq models failed")


async def _llm_generate(prompt: str) -> str:
    """Gemini first → Groq fallback. Guarantees a response as long as one LLM has quota."""
    try:
        return await _gemini_generate(prompt)
    except Exception as gemini_err:
        print(f"[LLM] All Gemini models failed ({str(gemini_err)[:80]}), switching to Groq...")
        return await _groq_generate(prompt)


async def _tts_bytes(text: str, voice: str) -> bytes:
    """Generate TTS audio bytes with a multi-level fallback.

    Level 1: Requested text + voice
    Level 2: Shorter Urdu fallback text + Urdu voice (when Urdu script fails)
    Level 3: English fallback text + English voice
    Returns b"" only if all three levels fail.
    """
    # Build a progressive fallback: try original voice, then alternate Urdu voice,
    # then English voice with English text (most reliable as last resort).
    alt_ur_voice = "ur-PK-AsadNeural" if voice == "ur-PK-UzmaNeural" else "ur-PK-UzmaNeural"
    fallback_levels = [
        (text, voice),
        (text, alt_ur_voice),                           # same text, alternate Urdu voice
        ("معاف کیجیے۔", "ur-PK-AsadNeural"),            # very short Urdu, reliable male voice
        ("Sorry, please try again.", "en-US-JennyNeural"),  # English always works
    ]
    for tts_text, tts_voice in fallback_levels:
        tmp = os.path.join(tempfile.gettempdir(), f"jarvis_{uuid.uuid4().hex}.mp3")
        try:
            communicate = edge_tts.Communicate(tts_text, tts_voice)
            await communicate.save(tmp)
            with open(tmp, "rb") as f:
                data = f.read()
            if data:
                return data
            print(f"[TTS] Empty audio from {tts_voice}, trying fallback...")
        except Exception as e:
            print(f"[TTS Error] {tts_voice}: {e}, trying fallback...")
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
    return b""


def _format_partner_context(ctx: dict) -> str:
    """Convert the raw context dict into readable text the LLM can directly cite.
    Groq and Gemini both handle natural-language facts much better than raw JSON."""
    if not ctx or ctx.get("error"):
        return ""

    tables = ctx.get("tables", [])
    table_str = (", ".join(f"Table {t}" for t in tables)) if tables else "koi table nahi"
    active = ctx.get("active_orders_summary", [])
    active_str = (", ".join(active)) if active else "koi active order nahi"
    sample = ctx.get("sample_menu", [])
    menu_str = (", ".join(sample)) if sample else ""
    status = "OPEN (Online)" if ctx.get("is_open", True) else "CLOSED (Offline)"

    # Detect likely RLS block: menu works (public) but orders+tables return 0
    rls_suspected = (
        ctx.get("total_menu_items", 0) > 0
        and ctx.get("today_orders", 0) == 0
        and ctx.get("total_tables", 0) == 0
    )

    city = ctx.get("city", "")
    opens = ctx.get("opens_at", "")
    closes = ctx.get("closes_at", "")
    timing = f" | Hours: {opens}–{closes}" if opens and closes else ""
    currency = ctx.get("currency", "PKR")
    delivery = "Yes" if ctx.get("is_delivery", True) else "No"
    rating = ctx.get("rating", "N/A")
    min_order = ctx.get("min_order", 0)
    delivery_fee = ctx.get("delivery_fee", 0)
    tax_pct = ctx.get("tax_percent", 0)

    lines = [
        f"\n\n══ LIVE RESTAURANT DATA ══",
        f"Restaurant: {ctx.get('restaurant_name', 'N/A')} | Status: {status}"
        + (f" | City: {city}" if city else "")
        + timing,
        f"Currency: {currency} | Min Order: {currency} {min_order} | Delivery Fee: {currency} {delivery_fee}"
        + (f" | Tax: {tax_pct}%" if tax_pct else "")
        + f" | Delivery: {delivery} | Rating: {rating}",
    ]
    if rls_suspected:
        lines.append(
            "⚠ DATA NOTE: Orders and tables data is currently unavailable "
            "(database access issue on backend). "
            "Menu data is available. "
            "For orders/tables questions → say 'iska data abhi load nahi hua, thodi der mein try karein' "
            "— do NOT say '0 tables' or '0 orders'."
        )
    else:
        lines += [
            f"Today: {ctx.get('today_orders', 0)} orders | Revenue: {currency} {ctx.get('today_revenue', 0)} | "
            f"Pending/Active: {ctx.get('pending_count', 0)} | Completed: {ctx.get('completed_orders', 0)}",
            f"Active orders: {active_str}",
        ]
    lines.append(f"Menu: {ctx.get('total_menu_items', 0)} items total ({ctx.get('available_menu_count', 0)} available)")
    cats = ctx.get("categories", [])
    if cats:
        lines.append(f"Menu categories: {', '.join(cats)}")
    lines.append(f"QR Tables: {ctx.get('total_tables', 0)} tables — {table_str}")
    if menu_str:
        lines.append(f"Sample menu items: {menu_str}")
    lines.append(
        "RULE: These numbers are LIVE — cite them directly. "
        "Combined requests (open page + tell me data) → emit BOTH actions [NAVIGATE + data action or NONE]. "
        "NEVER say 'main check kar raha hun'."
    )
    lines.append("══════════════════════════════════")
    return "\n".join(lines)


def _format_customer_context(ctx: dict, restaurant_id: str, cart_items: list | None = None) -> str:
    """Convert raw customer context dict into readable text for the LLM.
    cart_items: real-time cart from React CartContext (sent in every request).
    Never bail out early — even if ctx is empty, we may still have live cart data.
    """
    has_ctx = bool(ctx and not ctx.get("error"))
    has_cart = bool(cart_items)

    if not has_ctx and not has_cart:
        return ""

    lines = ["\n\n══ LIVE CUSTOMER SESSION DATA ══"]

    # ── Customer identity ─────────────────────────────────────────────────────
    if has_ctx:
        profile = ctx.get("customer_profile", {})
        if profile:
            name = profile.get("full_name") or "Guest"
            email = profile.get("email") or ""
            phone = profile.get("phone") or ""
            lines.append(
                f"Customer: {name}"
                + (f" | Email: {email}" if email else "")
                + (f" | Phone: {phone}" if phone else "")
            )
        prefs = ctx.get("preferences", {})
        if prefs:
            avg = prefs.get("avg_order_value")
            fav_cuisines = prefs.get("favorite_cuisines") or {}
            if avg:
                lines.append(f"Customer avg order: Rs.{avg}"
                             + (f" | Fav cuisines: {', '.join(list(fav_cuisines.keys())[:3])}" if fav_cuisines else ""))

    if has_ctx:
        # ── Single restaurant view ────────────────────────────────────────────
        rest = ctx.get("restaurant", {})
        if rest:
            status = "OPEN" if rest.get("is_open", True) else "CLOSED"
            currency = rest.get("currency", "PKR")
            min_ord = rest.get("min_order") or rest.get("min_order_price", 0)
            delivery = "Delivery available" if rest.get("is_delivery", True) else "Dine-in only"
            lines.append(
                f"Restaurant: {rest.get('name', 'N/A')} | {status} | {delivery}"
                + (f" | Min order: {currency} {min_ord}" if min_ord else "")
                + (f" | Rating: {rest.get('rating', 'N/A')}" if rest.get('rating') else "")
            )

        menu_items = ctx.get("menu_items", [])
        total_items = len(menu_items)
        cats: list[str] = []
        seen_cats: set = set()
        for m in menu_items:
            c = m.get("category") or ""
            if c and c not in seen_cats:
                cats.append(c)
                seen_cats.add(c)
        if total_items:
            lines.append(f"Menu: {total_items} items available | Categories: {', '.join(cats[:8]) or 'N/A'}")

        top_deals = ctx.get("top_deals", [])
        if top_deals:
            deal_strs = []
            for d in top_deals[:5]:
                disc = int(d.get("discount_percentage") or 0)
                deal_strs.append(f"{d.get('name')} Rs.{d.get('price')} ({disc}% OFF)")
            lines.append(f"Active deals: {' | '.join(deal_strs)}")

        # ── Home page view (no specific restaurant) ───────────────────────────
        all_restaurants = ctx.get("all_restaurants", [])
        if all_restaurants:
            rest_strs = []
            for r in all_restaurants[:10]:
                cuisines = r.get("cuisine_types") or r.get("cuisine_type") or []
                cuisine_str = (", ".join(cuisines[:2]) if isinstance(cuisines, list) else str(cuisines)) if cuisines else ""
                rest_strs.append(
                    f"{r['name']} ({r.get('city', 'N/A')})"
                    + (f" [{cuisine_str}]" if cuisine_str else "")
                    + (f" ★{r.get('rating', '')}" if r.get("rating") else "")
                    + (" 🚚" if r.get("is_delivery") else "")
                )
            lines.append(f"Open restaurants ({len(all_restaurants)} total):")
            lines.append("  " + " | ".join(rest_strs))

        all_deals = ctx.get("all_deals", [])
        if all_deals:
            deal_strs = [
                f"{d['name']} Rs.{d['price']} ({int(d.get('discount_percentage') or 0)}% OFF)"
                for d in all_deals[:8]
            ]
            lines.append(f"Top deals across all restaurants: {' | '.join(deal_strs)}")

        all_promos = ctx.get("all_promotions", [])
        if all_promos:
            promo_strs = [
                f"{p['code']} ({p.get('discount_type','')}: {p.get('discount_value','')} off)"
                for p in all_promos[:5]
            ]
            lines.append(f"Active promo codes: {', '.join(promo_strs)}")

    # Use real-time cart from frontend (CartContext) — always current, never stale
    cart = cart_items or []
    if cart:
        cart_total = 0.0
        cart_lines = []
        for item in cart[:8]:
            mi = item.get("menuItem") or {}
            name = mi.get("name", "?")
            price = float(mi.get("price") or 0)
            qty = int(item.get("quantity") or 1)
            subtotal = price * qty
            cart_total += subtotal
            cart_lines.append(f"{qty}x {name} Rs.{int(price)}")
        tail = f" (+{len(cart) - 8} more)" if len(cart) > 8 else ""
        lines.append(
            f"Cart ({len(cart)} items): {', '.join(cart_lines)}{tail} | "
            f"Total: Rs.{int(cart_total)}"
        )
    else:
        lines.append("Cart: empty")

    if has_ctx:
        recent = ctx.get("recent_orders", [])
        if recent:
            o = recent[0]
            lines.append(
                f"Last order: #{str(o.get('id',''))[-6:]} — {o.get('status','?')} — Rs.{o.get('total_amount',0)}"
                + (f" ({o.get('order_type','')})" if o.get('order_type') else "")
            )
            if len(recent) > 1:
                lines.append(f"Total past orders (loaded): {len(recent)}")

    lines.append(
        "RULE: Cite LIVE DATA directly. Cart details above are REAL-TIME from the user's browser. "
        "For cart total / cart contents questions → use NONE action (answer directly from Cart above). "
        "When 'Open restaurants' list is shown above → answer restaurant queries from that list (use NEARBY only if more detail needed). "
        "When 'Top deals' list is shown above → answer deals/discount queries from that list (use GET_DEALS for full list). "
        "'biryani dhundo' → SEARCH_MENU (searches ALL restaurants). 'best deals' → GET_DEALS. 'budget X' → BUDGET_SUGGEST."
    )
    lines.append("══════════════════════════════════")
    return "\n".join(lines)


@app.post("/process-voice")
async def process_voice(
    file: UploadFile = File(...),
    agent_type: str = Form("partner"),
    restaurant_id: str = Form(""),
    user_id: str = Form(""),
    preferred_lang: str = Form("auto"),    # "auto" | "ur" | "en"
    preferred_gender: str = Form("auto"),  # "auto" | "male" | "female"
    cart_data: str = Form(""),             # JSON array of CartItem objects from frontend
    user_lat: str = Form(""),              # customer GPS latitude (optional)
    user_lng: str = Form(""),             # customer GPS longitude (optional)
):
    print(f"[Jarvis] agent_type={agent_type} restaurant_id={restaurant_id or '(none)'} user_id={user_id[:12] if user_id else '(none)'} lang={preferred_lang}")

    # Parse cart from frontend — React CartContext sends this on every request
    cart_items_from_frontend: list = []
    if cart_data:
        try:
            cart_items_from_frontend = json.loads(cart_data)
        except Exception:
            pass

    # ── 1. STT — explicit language, no detect_language hallucination ──────────
    audio_data = await file.read()
    try:
        # Force explicit language: Deepgram detect_language hallucinates id/ja/tr for Urdu.
        stt_lang = "en" if preferred_lang == "en" else "ur"
        # nova-2 does not support Urdu — use nova-3 which supports ur + en both.
        # keywords boost common food/restaurant terms that Urdu STT mishears as native words
        # (e.g. "burgers" → "برکت", "pizza" → native phonetics). Boost them so STT keeps English.
        stt_model = "nova-2" if stt_lang == "en" else "nova-3"
        # nova-3 keyterm = plain word list; nova-2 keywords = "word:boost" format
        if stt_model == "nova-3":
            boost_param = "keyterm"
            food_keywords = [
                "burger", "burgers", "pizza", "biryani", "karahi",
                "fries", "paratha", "naan", "deal", "combo",
                "shawarma", "roll", "sandwich", "pasta", "nuggets",
                "tikka", "barbeque", "bbq", "zinger", "steak",
                "menu", "order", "discount",
            ]
        else:
            boost_param = "keywords"
            food_keywords = [
                "burger:3", "burgers:3", "pizza:3", "biryani:2", "karahi:2",
                "fries:2", "paratha:2", "naan:2", "deal:2", "combo:2",
                "shawarma:2", "roll:2", "sandwich:2", "pasta:2", "nuggets:2",
                "tikka:2", "bbq:2", "zinger:2", "steak:2", "menu:2",
            ]
        options = {
            "model": stt_model,
            "language": stt_lang,
            "smart_format": True,
            "punctuate": True,
            boost_param: food_keywords,
        }
        stt_res = await asyncio.to_thread(
            dg_client.listen.v1.media.transcribe_file,
            request=audio_data, **options
        )
        user_text = stt_res.results.channels[0].alternatives[0].transcript
        lang_code = stt_lang
        print(f"[STT] lang={stt_lang}: {user_text}")
    except Exception as e:
        print(f"[STT Error] {e}")
        return Response(content=json.dumps({"error": "STT failed"}), status_code=500)

    if not user_text:
        return Response(content=b"", status_code=204)

    # Drop clips that are too short to be real speech (mic noise, breath, random chars)
    if len(user_text.strip()) < 5:
        print(f"[STT] Skipping noise clip: '{user_text}'")
        return Response(content=b"", status_code=204)

    # ── 2. Session memory — remember conversation history ────────────────────
    # Partner: keyed on restaurant_id (one memory per restaurant).
    # Customer: keyed on user_id / guest UUID (one memory per person, regardless of which restaurant they browse).
    session_key = (restaurant_id if agent_type == "partner" else None) or user_id or restaurant_id or "default"
    session_mem = _get_session_memory(session_key)
    await session_mem.add_message("user", user_text)

    # ── 3. Pre-fetch live context from Supabase ───────────────────────────────
    context_block = ""
    raw_ctx: dict = {}
    try:
        if agent_type == "partner" and restaurant_id:
            raw_ctx = await cache_manager.get_context(restaurant_id, get_partner_context)
            context_block = _format_partner_context(raw_ctx)
            print(f"[Context] tables={raw_ctx.get('total_tables')} menu={raw_ctx.get('total_menu_items')} "
                  f"orders={raw_ctx.get('today_orders')} revenue={raw_ctx.get('today_revenue')}")
        elif agent_type == "customer":
            raw_ctx = await get_customer_context(user_id, restaurant_id)
            context_block = _format_customer_context(raw_ctx, restaurant_id, cart_items_from_frontend)
    except Exception as e:
        print(f"[Context fetch error] {e}")

    # ── 4. Build system prompt ────────────────────────────────────────────────
    effective_lang = preferred_lang if preferred_lang != "auto" else lang_code
    if agent_type == "partner":
        system_prompt = build_partner_prompt(context_block, preferred_lang=effective_lang)
    else:
        system_prompt = build_customer_prompt(context_block, preferred_lang=effective_lang)

    lang_instruction = (
        "Reply in Roman Urdu (English letters) in 'reply'. 'tts_text' MUST be Urdu script (اردو)."
        if effective_lang == "ur" else
        "Reply in English in 'reply'. 'tts_text' also in English."
        if effective_lang == "en" else
        "Match the user's language. If Urdu speaker, 'reply'=Roman Urdu, 'tts_text'=Urdu script (اردو)."
    )

    # Inject conversation history (last 10 messages, excluding the one just added)
    recent = session_mem.get_recent_turns(10)
    history_section = ""
    if len(recent) > 1:
        history_block = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in recent[:-1]  # exclude the current user message we just added
        )
        history_section = f"\n\nCONVERSATION HISTORY (most recent first):\n{history_block}\n"

    prompt = (
        f"{system_prompt}\n"
        f"{history_section}\n"
        f"LANGUAGE: {lang_instruction}\n\n"
        f"User said: \"{user_text}\"\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Choose the correct action(s) from the AVAILABLE ACTIONS list.\n"
        f"2. ACTION RULES — pick the most specific action:\n"
        f"   • Navigation request (open/show/jaao) → NAVIGATE with the correct target.\n"
        f"   • Data question needing a FULL LIST (all orders, full menu, top items, charts) → use the matching GET_* action.\n"
        f"   • 'Show me all [category] items / kitne [category] hain / [category] ki list' → SEARCH_MENU with query=[category word only, e.g. 'burger']. ONE action, not multiple.\n"
        f"   • Data question answerable from the LIVE DATA block (table count, revenue, order count, status) → include ACTUAL NUMBER in 'reply' AND use NONE ONLY IF there is no navigation request; if there IS a navigation request, add NAVIGATE as a second action.\n"
        f"   • Combined request (open page AND tell me data) → return BOTH actions: [NAVIGATE, ...] — NEVER collapse to a single NONE.\n"
        f"   • Pure greeting / small talk → NONE.\n"
        f"   ⚠ SEARCH_MENU query MUST be a dish name or category (burger, pizza, biryani…). NEVER use meta-words like 'price', 'name', 'items', 'list', 'all' as the query.\n"
        f"3. Reply MUST be specific — never say 'main check kar raha hun' or 'main bata raha hun'. Use the ACTUAL numbers from LIVE DATA.\n"
        f"4. Respond ONLY with valid JSON — no markdown, no extra text:\n"
        f"{{\"reply\":\"...\",\"tts_text\":\"...\",\"gender\":\"male|female\","
        f"\"actions\":[{{\"action\":\"ACTION_CODE\",\"target\":\"...\",\"params\":{{}}}}]}}"
    )

    # ── 5. LLM — Gemini with automatic model fallback on quota errors ─────────
    try:
        raw_text = await _llm_generate(prompt)
        parsed = json.loads(_strip_json_fences(raw_text))

        # Groq sometimes returns a bare JSON array [{"action":...}] instead of an object.
        # Wrap it so the rest of the code always sees a dict.
        if isinstance(parsed, list):
            ai_data = {
                "reply": "",
                "tts_text": "",
                "gender": "female",
                "actions": parsed,
            }
        else:
            ai_data = parsed

        # Normalize: support both old flat format {action,target,params} and new {actions:[...]}
        if "actions" not in ai_data:
            if "action" in ai_data:
                ai_data["actions"] = [{
                    "action": ai_data["action"],
                    "target": ai_data.get("target", ""),
                    "params": ai_data.get("params", {}),
                }]
            else:
                ai_data["actions"] = [{"action": "NONE", "target": "", "params": {}}]

        print(f"[LLM] actions={[(a.get('action'), a.get('target','')[:15]) for a in ai_data.get('actions', [])]} reply={ai_data.get('reply', '')[:60]}")
    except Exception as e:
        print(f"[LLM Error] {e}")
        if lang_code == "en":
            fb_reply = "Sorry, I couldn't process that. Please try again."
            fb_tts   = "Sorry, I couldn't process that. Please try again."
        else:
            fb_reply = "Maaf kijiyega, main samajh nahi saka. Phir se batayein?"
            fb_tts   = "معاف کیجیے گا، میں سمجھ نہیں سکا، پھر سے بتائیں۔"
        ai_data = {
            "reply":   fb_reply,
            "tts_text": fb_tts,
            "gender":  "female",
            "actions": [{"action": "NONE", "target": "", "params": {}}],
        }

    # ── 6. Execute tools for ALL actions (multi-intent support) ──────────────
    all_tool_results: list[dict] = []
    # VIEW_CART / CART_TOTAL are answered directly from the LIVE CUSTOMER SESSION DATA context
    # (cart sent by frontend in every request) — no stub tool needed.
    SKIP_ACTIONS = {"NONE", "NAVIGATE", "LOGOUT", "AUTH", "STRIPE_PAY", "TOGGLE_STATUS",
                    "VIEW_CART", "CART_TOTAL"}
    for act_obj in ai_data.get("actions", []):
        action = act_obj.get("action", "NONE")
        target = act_obj.get("target", "")
        params = act_obj.get("params", {})

        if action in SKIP_ACTIONS:
            print(f"[Tool] Skipping {action} target='{target}' (frontend action)")
            continue

        print(f"[Tool] Executing {action} target='{target}' params={params}")
        try:
            if agent_type == "partner":
                tool_result = await execute_partner_tool(action, target, params, restaurant_id)
            else:
                # Inject GPS coords for NEARBY so it can sort by distance
                if action == "NEARBY" and user_lat and user_lng:
                    params = {**(params or {}), "lat": user_lat, "lng": user_lng}
                tool_result = await execute_customer_tool(action, target, params, user_id, restaurant_id)

            if tool_result:
                has_data = bool(tool_result.get("data"))
                err = tool_result.get("error", "")
                print(f"[Tool] {action} → data={has_data} summary='{tool_result.get('summary', '')[:80]}'"
                      + (f" ERROR={err}" if err else ""))
                act_obj["tool_result"] = tool_result  # attach to its action
                all_tool_results.append(tool_result)
            else:
                print(f"[Tool] {action} returned None (not in _TOOL_MAP?)")
        except Exception as e:
            print(f"[Tool Error] {action}: {e}")

    # ── Cache maintenance after mutations ────────────────────────────────────
    if agent_type == "partner" and restaurant_id:
        for act_obj in ai_data.get("actions", []):
            action = act_obj.get("action", "")
            tool_res = act_obj.get("tool_result", {})
            if action == "ADD_MENU_ITEM" and tool_res.get("data"):
                # Write-through: add new item to cache without a DB round-trip
                new_item = tool_res["data"].get("item", {})
                if new_item:
                    cache_manager.on_menu_item_added(restaurant_id, new_item)
            elif action in (
                "UPDATE_MENU_BY_NAME", "RENAME_ITEM", "TOGGLE_ITEM", "APPLY_DISCOUNT",
                "UPDATE_ORDER", "ADD_TABLE", "DELETE_TABLE",
            ):
                # Mutations that change DB state — invalidate so next request refreshes
                cache_manager.invalidate(restaurant_id)

    # Generate natural reply from tool data — use attached tool_result, not zip (zip misaligns
    # when NAVIGATE/NONE are skipped and all_tool_results has fewer entries than actions)
    action_results = [
        (act_obj.get("action", ""), act_obj["tool_result"])
        for act_obj in ai_data.get("actions", [])
        if act_obj.get("tool_result") and act_obj["tool_result"].get("data")
    ]
    if action_results:
        print(f"[LLM-2] Follow-up call with data from: {[a for a, _ in action_results]}")
        summaries = [
            _build_data_summary(action, result["data"], lang_code)
            for action, result in action_results
        ]
        combined = "\n\n".join(f"[{a}]\n{s}" for (a, _), s in zip(action_results, summaries))

        lang_instr = (
            "Reply in Roman Urdu (English letters). 'tts_text' MUST be Urdu script (اردو)."
            if effective_lang != "en" else
            "Reply in English. 'tts_text' also in English."
        )
        follow_up = (
            f"User asked: \"{user_text}\"\n\n"
            f"Data retrieved:\n{combined}\n\n"
            f"Write a FRIENDLY, NATURAL voice response (2-3 sentences max). "
            f"Include specific numbers and names from the data. {lang_instr} "
            f"Return ONLY JSON: {{\"reply\": \"...\", \"tts_text\": \"...\", \"gender\": \"male|female\"}}"
        )
        try:
            follow_raw = await _llm_generate(follow_up)
            follow_data = json.loads(_strip_json_fences(follow_raw))
            ai_data["reply"]    = follow_data.get("reply", ai_data["reply"])
            ai_data["tts_text"] = follow_data.get("tts_text", ai_data.get("tts_text", ai_data["reply"]))
            ai_data["gender"]   = follow_data.get("gender", ai_data.get("gender", "female"))
        except Exception:
            # Fallback: use the pre-built summary as reply
            fallback_summary = summaries[0] if summaries else ""
            if fallback_summary:
                ai_data["reply"] = fallback_summary[:200]

    # Add tool results to response (backward-compat: keep top-level tool_result too)
    if all_tool_results:
        ai_data["tool_result"]  = all_tool_results[0]
        ai_data["tool_results"] = all_tool_results

    # Save assistant reply to memory for next turn
    await session_mem.add_message("assistant", ai_data["reply"])

    # ── 7. TTS — Edge TTS with multi-level fallback ───────────────────────────
    if preferred_gender in ("male", "female"):
        gender = preferred_gender
    else:
        gender = ai_data.get("gender", "female")
    speech_text = ai_data.get("tts_text") or ai_data.get("reply", "")
    selected_voice = _select_tts_voice(lang_code, speech_text, gender)
    print(f"[TTS] voice={selected_voice}")

    audio_bytes = await _tts_bytes(speech_text, selected_voice)

    # ── 8. Respond ────────────────────────────────────────────────────────────
    encoded_data = quote(json.dumps(ai_data, ensure_ascii=False), safe='')
    headers = {
        "X-Jarvis-Data": encoded_data,
        "Access-Control-Expose-Headers": "X-Jarvis-Data",
    }
    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)


@app.post("/speak")
async def speak(
    text: str = Form(...),
    lang: str = Form("ur"),
    gender: str = Form("female"),
):
    """Lightweight TTS-only endpoint — no STT, no Gemini. Used for proactive voice alerts."""
    voice = _select_tts_voice(lang, text, gender)
    audio_bytes = await _tts_bytes(text, voice)
    if not audio_bytes:
        return Response(content=b"", status_code=500)
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/webhook/cache-invalidate")
async def webhook_cache_invalidate(request: Request):
    """Supabase webhook — POST here when orders/menu change to keep cache fresh.
    Configure in Supabase: Table → Webhooks → POST http://your-host:8001/webhook/cache-invalidate
    Body shape: {"record": {"restaurant_id": "..."}} or {"restaurant_id": "..."}
    """
    try:
        body = await request.json()
        rid = (body.get("record") or {}).get("restaurant_id") or body.get("restaurant_id", "")
        if rid:
            cache_manager.invalidate(rid)
        return {"ok": True, "invalidated": rid or "none"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/cache/stats")
async def cache_stats():
    """Dev endpoint — see what's in the in-memory cache."""
    return cache_manager.stats()


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "SaySavor Jarvis v2"}


if __name__ == "__main__":
    import uvicorn
    print("SaySavor Jarvis starting on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
