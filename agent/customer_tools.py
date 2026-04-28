"""
Customer Tools — voice AI agent tools for the SaySavor customer experience.
Cart mutations return signals (frontend executes via CartContext).
Data queries return Supabase results for LLM-2 narration.
"""

import os
from supabase import create_client, Client

_supa: Client | None = None


def _db() -> Client:
    global _supa
    if _supa is None:
        url = os.getenv("SUPABASE_URL", "")
        key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or
               os.getenv("SUPABASE_ANON_KEY") or
               os.getenv("SUPABASE_KEY", ""))
        _supa = create_client(url, key)
    return _supa


def _cat_name(item: dict) -> str:
    cats = item.get("categories")
    if isinstance(cats, dict):
        return cats.get("name", "") or ""
    return ""


# ── Context snapshot ──────────────────────────────────────────────────────────

async def get_customer_context(user_id: str, restaurant_id: str) -> dict:
    ctx: dict = {}
    try:
        db = _db()

        # ── Customer identity — who is talking to Jarvis right now ────────────
        if user_id and not user_id.startswith("guest_"):
            try:
                profile_res = (
                    db.table("profiles")
                    .select("full_name, email, phone, avatar_url, role")
                    .eq("id", user_id)
                    .limit(1)
                    .execute()
                )
                if profile_res.data:
                    ctx["customer_profile"] = profile_res.data[0]
            except Exception:
                pass

            try:
                pref_res = (
                    db.table("user_preferences")
                    .select("favorite_items, favorite_cuisines, avg_order_value, last_order_date")
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                if pref_res.data:
                    ctx["preferences"] = pref_res.data[0]
            except Exception:
                pass

        if user_id:
            recent_res = (
                db.table("orders")
                .select("id, status, total_amount, created_at, restaurant_id, order_type")
                .eq("customer_id", user_id)
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
            ctx["recent_orders"] = recent_res.data or []

        if restaurant_id:
            # ── Single restaurant mode — customer is inside a specific restaurant ──

            # Active promotions for this restaurant
            try:
                promo_res = (
                    db.table("promotions")
                    .select("code, discount_type, discount_value, min_order")
                    .eq("restaurant_id", restaurant_id)
                    .eq("is_active", True)
                    .limit(5)
                    .execute()
                )
                ctx["active_promotions"] = promo_res.data or []
            except Exception:
                pass

            # Full menu with discount info
            try:
                menu_res = (
                    db.table("menu_items")
                    .select(
                        "id, name, price, is_available, description, "
                        "discount_percentage, original_price, offer_name, "
                        "categories(name)"
                    )
                    .eq("restaurant_id", restaurant_id)
                    .eq("is_available", True)
                    .execute()
                )
                menu_items = menu_res.data or []
                for item in menu_items:
                    item["category"] = _cat_name(item)

                deals = sorted(
                    [i for i in menu_items if float(i.get("discount_percentage") or 0) > 0],
                    key=lambda x: float(x.get("discount_percentage") or 0),
                    reverse=True,
                )
                ctx["menu_items"] = menu_items
                ctx["total_menu"] = len(menu_items)
                ctx["deal_count"] = len(deals)
                ctx["top_deals"] = [
                    {
                        "name": d["name"],
                        "price": d.get("price"),
                        "original_price": d.get("original_price"),
                        "discount_percentage": d.get("discount_percentage"),
                        "offer_name": d.get("offer_name", ""),
                    }
                    for d in deals[:5]
                ]
            except Exception:
                pass

            # Restaurant basic info
            try:
                rest_res = (
                    db.table("restaurants")
                    .select("name, is_open, min_order, min_order_price, delivery_fee, currency, city, opens_at, closes_at, rating, is_delivery")
                    .eq("id", restaurant_id)
                    .limit(1)
                    .execute()
                )
                if rest_res.data:
                    ctx["restaurant"] = rest_res.data[0]
            except Exception:
                pass

        else:
            # ── Home page mode — show all open restaurants + cross-restaurant deals ──

            # All open restaurants (for "kaunse restaurants hain" queries)
            try:
                rest_res = (
                    db.table("restaurants")
                    .select("id, name, city, cuisine_type, cuisine_types, rating, is_open, is_delivery, min_order, delivery_fee, delivery_time_min")
                    .eq("is_open", True)
                    .limit(20)
                    .execute()
                )
                ctx["all_restaurants"] = rest_res.data or []
            except Exception:
                pass

            # Top deals across ALL restaurants
            try:
                deals_res = (
                    db.table("menu_items")
                    .select("id, name, price, original_price, discount_percentage, offer_name, restaurant_id")
                    .eq("is_available", True)
                    .gt("discount_percentage", 0)
                    .order("discount_percentage", desc=True)
                    .limit(10)
                    .execute()
                )
                ctx["all_deals"] = deals_res.data or []
            except Exception:
                pass

            # Active promos across all restaurants
            try:
                promo_res = (
                    db.table("promotions")
                    .select("code, discount_type, discount_value, min_order, restaurant_id")
                    .eq("is_active", True)
                    .limit(10)
                    .execute()
                )
                ctx["all_promotions"] = promo_res.data or []
            except Exception:
                pass

    except Exception as e:
        ctx["error"] = str(e)
    return ctx


# ── Tool Functions ─────────────────────────────────────────────────────────────

async def handle_auth(user_id: str, restaurant_id: str, params: dict) -> dict:
    mode = params.get("mode", "login")
    return {"signal": "AUTH", "data": {"mode": mode}, "summary": f"Auth: {mode}"}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def get_nearby_restaurants(user_id: str, restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        res = (
            db.table("restaurants")
            .select("id, name, logo_url, cuisine_type, cuisine_types, rating, city, delivery_time_min, is_delivery, latitude, longitude, min_order, delivery_fee")
            .eq("is_open", True)
            .limit(30)
            .execute()
        )
        restaurants = res.data or []

        # Sort by distance if customer location provided
        user_lat = params.get("lat")
        user_lng = params.get("lng")
        if user_lat is not None and user_lng is not None:
            try:
                ulat, ulng = float(user_lat), float(user_lng)
                for r in restaurants:
                    rlat = r.get("latitude")
                    rlng = r.get("longitude")
                    if rlat and rlng:
                        r["distance_km"] = round(_haversine_km(ulat, ulng, float(rlat), float(rlng)), 1)
                    else:
                        r["distance_km"] = 999
                restaurants.sort(key=lambda r: r.get("distance_km", 999))
                # Filter within 25km radius
                restaurants = [r for r in restaurants if r.get("distance_km", 999) <= 25]
            except Exception:
                pass

        return {
            "signal": "NAVIGATE",
            "data": {
                "restaurants": restaurants[:12],
                "navigate_to": "/foodie/home",
                "total": len(restaurants),
            },
            "summary": (
                f"{len(restaurants)} restaurants found"
                + (f" within 25km" if user_lat else "")
                + (f" — nearest: {restaurants[0]['name']} ({restaurants[0].get('distance_km','?')}km)" if restaurants and user_lat else "")
            ),
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def search_restaurants(user_id: str, restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        query_text = params.get("query") or params.get("target", "")
        res = (
            db.table("restaurants")
            .select("id, name, logo_url, cuisine_type")
            .ilike("name", f"%{query_text}%")
            .eq("is_open", True)
            .limit(10)
            .execute()
        )
        restaurants = res.data or []
        return {
            "signal": "NAVIGATE",
            "data": {
                "restaurants": restaurants,
                "query": query_text,
                "navigate_to": f"/foodie/home?q={query_text}",
            },
            "summary": f"{len(restaurants)} results for '{query_text}'",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def search_menu_items(user_id: str, restaurant_id: str, params: dict) -> dict:
    """Search menu items by name/category. When inside a restaurant, filters to that restaurant.
    When on home page (no restaurant_id), searches across ALL restaurants."""
    query = (params.get("query") or params.get("item_name") or params.get("target", "")).strip()
    rid = restaurant_id or params.get("restaurant_id", "")

    try:
        db = _db()
        q = (
            db.table("menu_items")
            .select(
                "id, name, price, is_available, description, "
                "discount_percentage, original_price, offer_name, "
                "restaurant_id, categories(name)"
            )
            .eq("is_available", True)
        )
        if rid:
            q = q.eq("restaurant_id", rid)

        res = q.execute()
        items = res.data or []
        for item in items:
            item["category"] = _cat_name(item)

        q_lower = query.lower()
        if q_lower:
            items = [
                i for i in items
                if q_lower in (i.get("name") or "").lower()
                or q_lower in (i.get("category") or "").lower()
                or q_lower in (i.get("description") or "").lower()
            ]

        navigate_to = (
            f"/foodie/restaurant/{rid}?search={query}" if rid
            else f"/foodie/home?q={query}"
        )
        return {
            "signal": "SEARCH_MENU",
            "data": {
                "query": query,
                "items": items[:20],
                "count": len(items),
                "restaurant_id": rid,
                "navigate_to": navigate_to,
            },
            "summary": (
                f"{len(items)} items matching '{query}'"
                + (f": {', '.join(i['name'] for i in items[:3])}" if items else " — koi item nahi mila")
            ),
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_best_deals(user_id: str, restaurant_id: str, params: dict) -> dict:
    """Fetch discounted items, sorted best discount first."""
    rid = restaurant_id or params.get("restaurant_id", "")
    try:
        db = _db()
        q = (
            db.table("menu_items")
            .select(
                "id, name, price, is_available, discount_percentage, "
                "original_price, offer_name, categories(name)"
            )
            .eq("is_available", True)
        )
        if rid:
            q = q.eq("restaurant_id", rid)
        res = q.execute()
        all_items = res.data or []
        for item in all_items:
            item["category"] = _cat_name(item)

        deals = sorted(
            [i for i in all_items if float(i.get("discount_percentage") or 0) > 0],
            key=lambda x: float(x.get("discount_percentage") or 0),
            reverse=True,
        )

        deal_list = [
            {
                "id": d.get("id"),
                "name": d["name"],
                "original_price": d.get("original_price") or d.get("price"),
                "price": d.get("price"),
                "discount_percentage": float(d.get("discount_percentage") or 0),
                "offer_name": d.get("offer_name", ""),
                "category": d.get("category", ""),
            }
            for d in deals[:8]
        ]

        return {
            "signal": "SHOW_DEALS",
            "data": {
                "deals": deal_list,
                "count": len(deals),
                "restaurant_id": rid,
                "navigate_to": f"/foodie/restaurant/{rid}" if rid else "/foodie/home",
            },
            "summary": (
                f"{len(deals)} deals available"
                + (
                    f" — best: {deals[0]['name']} "
                    f"{float(deals[0].get('discount_percentage', 0)):.0f}% off"
                    if deals else ""
                )
            ),
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_budget_recommendations(user_id: str, restaurant_id: str, params: dict) -> dict:
    """Suggest items within the customer's stated budget.
    Optionally filter by dish name/category via params['query'].
    Prioritises deals first, then cheapest regular items."""
    try:
        budget = float(
            params.get("budget") or params.get("max_price") or params.get("target") or 500
        )
    except (ValueError, TypeError):
        budget = 500.0

    # Optional item/category filter — "500 mein burgers" passes query="burger"
    query = (params.get("query") or params.get("category") or "").strip().lower()

    rid = restaurant_id or params.get("restaurant_id", "")
    try:
        db = _db()
        q = (
            db.table("menu_items")
            .select(
                "id, name, price, is_available, discount_percentage, "
                "original_price, offer_name, restaurant_id, categories(name)"
            )
            .eq("is_available", True)
            .lte("price", budget)
        )
        if rid:
            q = q.eq("restaurant_id", rid)
        res = q.order("price").execute()
        items = res.data or []
        for item in items:
            item["category"] = _cat_name(item)

        # Filter by item name or category when user specifies a dish type
        if query:
            items = [
                i for i in items
                if query in i.get("name", "").lower()
                or query in (i.get("category") or "").lower()
            ]

        deals_in_budget = sorted(
            [i for i in items if float(i.get("discount_percentage") or 0) > 0],
            key=lambda x: float(x.get("discount_percentage") or 0),
            reverse=True,
        )
        regular_in_budget = [i for i in items if not float(i.get("discount_percentage") or 0)]

        # When filtered by category, return all matches (up to 10); else top 5 picks
        limit = 10 if query else 5
        suggestions = (deals_in_budget + regular_in_budget)[:limit]

        def _fmt(s: dict) -> str:
            disc = float(s.get("discount_percentage") or 0)
            note = f" ({int(disc)}% off!)" if disc > 0 else ""
            return f"{s['name']} Rs.{s['price']}{note}"

        query_note = f" '{query}' mein" if query else ""
        return {
            "signal": "BUDGET_SUGGEST",
            "data": {
                "budget": budget,
                "query": query,
                "suggestions": suggestions,
                "deals_in_budget": len(deals_in_budget),
                "total_in_budget": len(items),
            },
            "summary": (
                f"Rs.{int(budget)} budget{query_note} mein {len(items)} items available"
                + (f", {len(deals_in_budget)} deals bhi hain" if deals_in_budget else "")
                + ". Top picks: "
                + ", ".join(_fmt(s) for s in suggestions[:4])
            ),
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_active_promotions(user_id: str, restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        rid = restaurant_id or params.get("restaurant_id", "")
        q = (
            db.table("promotions")
            .select("code, discount_type, discount_value, description, min_order")
            .eq("is_active", True)
        )
        if rid:
            q = q.eq("restaurant_id", rid)
        res = q.limit(10).execute()
        promos = res.data or []
        return {"data": {"promotions": promos}, "summary": f"{len(promos)} active promotions"}
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_restaurant_menu(user_id: str, restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        rid = restaurant_id or params.get("restaurant_id", "")
        if not rid:
            return {"data": None, "error": "restaurant_id required"}
        res = (
            db.table("menu_items")
            .select(
                "id, name, price, category_id, description, is_available, image_url, "
                "discount_percentage, original_price, offer_name, categories(name)"
            )
            .eq("restaurant_id", rid)
            .eq("is_available", True)
            .execute()
        )
        items = res.data or []
        for item in items:
            item["category"] = _cat_name(item)
        return {
            "signal": "NAVIGATE",
            "data": {
                "menu_items": items,
                "navigate_to": f"/foodie/restaurant/{rid}",
            },
            "summary": f"{len(items)} menu items",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def add_to_cart(user_id: str, restaurant_id: str, params: dict) -> dict:
    """Look up item by name in this restaurant's menu, then return data for frontend to add to cart."""
    item_name = params.get("item_name") or params.get("name") or params.get("target", "")
    item_id = params.get("item_id", "")
    qty = int(params.get("qty", 1))
    price = float(params.get("price", 0))
    category = ""

    if item_name:
        try:
            db = _db()
            q = (
                db.table("menu_items")
                .select("id, name, price, restaurant_id, categories(name)")
                .ilike("name", f"%{item_name}%")
                .eq("is_available", True)
                .limit(1)
            )
            if restaurant_id:
                q = q.eq("restaurant_id", restaurant_id)
            res = q.execute()
            if res.data:
                item = res.data[0]
                item_id = item["id"]
                price = float(item.get("price", price))
                item_name = item["name"]
                restaurant_id = item.get("restaurant_id", restaurant_id)
                category = _cat_name(item)
            else:
                return {
                    "signal": "ADD_CART_ERROR",
                    "data": {"error": f"'{item_name}' is restaurant ke menu mein nahi mila"},
                    "summary": f"'{item_name}' menu mein available nahi hai. Doosra item try karein.",
                }
        except Exception as e:
            return {"data": None, "error": str(e)}

    return {
        "signal": "ADD_CART",
        "data": {
            "item_id": item_id,
            "item_name": item_name,
            "name": item_name,
            "qty": qty,
            "price": price,
            "category": category,
            "restaurant_id": restaurant_id,
        },
        "summary": f"{qty}x '{item_name}' (Rs.{int(price)}) cart mein add ho gaya",
    }


async def view_cart(user_id: str, restaurant_id: str, params: dict) -> dict:
    return {"signal": "VIEW_CART", "data": {"navigate_to": "/foodie/cart"}, "summary": "Show cart"}


async def update_cart_item(user_id: str, restaurant_id: str, params: dict) -> dict:
    index = params.get("index", 0)
    qty = int(params.get("qty", 1))
    return {
        "signal": "UPDATE_CART",
        "data": {"index": index, "qty": qty},
        "summary": f"Update cart item {index} to qty {qty}",
    }


async def remove_from_cart(user_id: str, restaurant_id: str, params: dict) -> dict:
    index = params.get("index", 0)
    item_name = params.get("item_name") or params.get("target", "")
    return {
        "signal": "REMOVE_CART",
        "data": {"index": index, "item_name": item_name},
        "summary": f"Remove {item_name} from cart",
    }


async def get_cart_totals(user_id: str, restaurant_id: str, params: dict) -> dict:
    return {
        "signal": "VIEW_CART",
        "data": {"navigate_to": "/foodie/cart", "show_totals": True},
        "summary": "Show cart totals",
    }


async def apply_promo_code(user_id: str, restaurant_id: str, params: dict) -> dict:
    code = params.get("code") or params.get("target", "")
    try:
        db = _db()
        res = (
            db.table("promotions")
            .select("id, code, discount_type, discount_value, min_order")
            .eq("code", code.upper())
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        promo = res.data[0] if res.data else None
        if promo:
            return {
                "signal": "APPLY_PROMO",
                "data": {"promo": promo},
                "summary": f"Promo {code} valid: {promo.get('discount_value')} off",
            }
        return {"data": {"valid": False}, "summary": f"Promo code '{code}' not valid"}
    except Exception as e:
        return {"data": None, "error": str(e)}


async def set_checkout_details(user_id: str, restaurant_id: str, params: dict) -> dict:
    address = params.get("address", "")
    phone = params.get("phone", "")
    return {
        "signal": "SET_CHECKOUT",
        "data": {"address": address, "phone": phone},
        "summary": "Checkout details set",
    }


async def set_payment_method(user_id: str, restaurant_id: str, params: dict) -> dict:
    method = params.get("method") or params.get("target", "cash")
    return {"signal": "SET_PAYMENT", "data": {"method": method}, "summary": f"Payment: {method}"}


async def initiate_stripe_payment(user_id: str, restaurant_id: str, params: dict) -> dict:
    return {
        "signal": "STRIPE_PAY",
        "data": {"navigate_to": "/foodie/checkout?payment=stripe"},
        "summary": "Redirect to Stripe",
    }


async def place_order(user_id: str, restaurant_id: str, params: dict) -> dict:
    return {
        "signal": "PLACE_ORDER",
        "data": {"navigate_to": "/foodie/checkout"},
        "summary": "Proceed to checkout",
    }


async def get_order_status(user_id: str, restaurant_id: str, params: dict) -> dict:
    order_id = params.get("order_id") or params.get("target", "")
    if not order_id and user_id:
        try:
            db = _db()
            res = (
                db.table("orders")
                .select("id, status, total_amount, created_at, table_number")
                .eq("customer_id", user_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            order = res.data[0] if res.data else None
            if order:
                return {
                    "signal": "NAVIGATE",
                    "data": {"order": order, "navigate_to": f"/foodie/track/{order['id']}"},
                    "summary": f"Last order status: {order.get('status')}",
                }
        except Exception as e:
            return {"data": None, "error": str(e)}
    return {
        "signal": "NAVIGATE",
        "data": {"navigate_to": f"/foodie/track/{order_id}"},
        "summary": "Order tracker",
    }


async def get_order_history(user_id: str, restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        res = (
            db.table("orders")
            .select("id, status, total_amount, created_at, restaurant_id")
            .eq("customer_id", user_id)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )
        orders = res.data or []
        return {
            "signal": "NAVIGATE",
            "data": {"orders": orders, "navigate_to": "/foodie/profile"},
            "summary": f"{len(orders)} past orders",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def update_customer_profile(user_id: str, restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        field = params.get("field", "")
        value = params.get("value")
        if not field or not user_id:
            return {"data": None, "error": "field and user_id required"}
        db.table("customers").update({field: value}).eq("id", user_id).execute()
        return {
            "signal": "NAVIGATE",
            "data": {"updated": True, "field": field, "navigate_to": "/foodie/profile"},
            "summary": f"Profile {field} updated",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def manage_addresses(user_id: str, restaurant_id: str, params: dict) -> dict:
    action = params.get("action", "list")
    address = params.get("address", "")
    return {
        "signal": "NAVIGATE",
        "data": {"action": action, "address": address, "navigate_to": "/foodie/profile"},
        "summary": f"Address {action}",
    }


async def update_alert_settings(user_id: str, restaurant_id: str, params: dict) -> dict:
    setting = params.get("setting", "")
    value = params.get("value", True)
    return {
        "signal": "NAVIGATE",
        "data": {"setting": setting, "value": value, "navigate_to": "/foodie/profile"},
        "summary": f"Alert '{setting}' → {value}",
    }


# ── Router ────────────────────────────────────────────────────────────────────

_TOOL_MAP = {
    "AUTH":           handle_auth,
    "NEARBY":         get_nearby_restaurants,
    "SEARCH":         search_restaurants,
    "SEARCH_MENU":    search_menu_items,
    "GET_DEALS":      get_best_deals,
    "BUDGET_SUGGEST": get_budget_recommendations,
    "PROMOTIONS":     get_active_promotions,
    "GET_MENU":       get_restaurant_menu,
    "ADD_CART":       add_to_cart,
    "VIEW_CART":      view_cart,
    "UPDATE_CART":    update_cart_item,
    "REMOVE_CART":    remove_from_cart,
    "CART_TOTAL":     get_cart_totals,
    "PROMO_CODE":     apply_promo_code,
    "SET_CHECKOUT":   set_checkout_details,
    "SET_PAYMENT":    set_payment_method,
    "STRIPE_PAY":     initiate_stripe_payment,
    "PLACE_ORDER":    place_order,
    "ORDER_STATUS":   get_order_status,
    "ORDER_HISTORY":  get_order_history,
    "UPDATE_PROFILE": update_customer_profile,
    "MANAGE_ADDRESS": manage_addresses,
    "ALERT_SETTINGS": update_alert_settings,
}


async def execute_customer_tool(
    action: str, target: str, params: dict, user_id: str, restaurant_id: str
) -> dict | None:
    fn = _TOOL_MAP.get(action)
    if fn is None:
        return None
    merged_params = {**(params or {}), "target": target}
    return await fn(user_id, restaurant_id, merged_params)
