"""
Partner Tools — Supabase functions for the Partner Jarvis agent.
Each function returns {"data": ..., "summary": "..."} or {"signal": ..., "summary": "..."}.

Column safety: orders table may use 'total_amount' or 'total' — we try both via helper.
"""

import os
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from agent.cache_manager import cache_manager

_supa: Client | None = None


def _db() -> Client:
    global _supa
    if _supa is None:
        url = os.getenv("SUPABASE_URL", "")
        # Service role key bypasses RLS — required for backend reads on orders/restaurant_tables.
        # Anon key falls back when service key is not configured.
        key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or
               os.getenv("SUPABASE_ANON_KEY") or
               os.getenv("SUPABASE_KEY", ""))
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        key_type = "service_role" if os.getenv("SUPABASE_SERVICE_ROLE_KEY") else "anon"
        print(f"[DB] Supabase client initialized — project: {url.split('.')[0].split('/')[-1]} key={key_type}")
        _supa = create_client(url, key)
    return _supa


def _today_range():
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(), now.isoformat()


def _week_range():
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.isoformat(), now.isoformat()


def _date_range_from_params(params: dict):
    """Return (start_iso, end_iso) from flexible params.
    Supports: start_date/end_date strings, days=N, date='YYYY-MM-DD', or defaults to today."""
    if params.get("start_date") and params.get("end_date"):
        return params["start_date"], params["end_date"]
    if params.get("days"):
        days = int(params["days"])
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), now.isoformat()
    if params.get("date"):
        # specific single date
        d = params["date"]
        start = f"{d}T00:00:00+00:00"
        end   = f"{d}T23:59:59+00:00"
        return start, end
    # keyword support
    kw = (params.get("range") or params.get("period") or "today").lower()
    now = datetime.now(timezone.utc)
    if "week" in kw or "hafta" in kw:
        return _week_range()
    if "month" in kw or "mahina" in kw:
        start = (now - timedelta(days=29)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), now.isoformat()
    if "yesterday" in kw or "kal" in kw:
        yesterday = (now - timedelta(days=1))
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
        return start.isoformat(), end.isoformat()
    return _today_range()


def _order_total(order: dict) -> float:
    """Extract order total — uses total_amount (the real column name)."""
    return float(order.get("total_amount") or 0)


def _cat_name(item: dict) -> str:
    """Safely extract category name from a menu_item row that may have a categories join."""
    cats = item.get("categories")
    if isinstance(cats, dict):
        return cats.get("name") or "Uncategorized"
    if isinstance(cats, list) and cats:
        return cats[0].get("name") or "Uncategorized"
    return "Uncategorized"


# Columns that actually exist in each table (ground-truth from schema + migrations)
# orders:     id, restaurant_id, customer_name, customer_phone, table_number,
#             order_type, status, payment_status, total_amount, created_at, updated_at
# menu_items: id, restaurant_id, name, price, description, category_id, is_available,
#             offer_name, discount_percentage, original_price, offer_expires_at
#             NOTE: price = current price (discounted when deal active)
#                   original_price = pre-discount price, discount_percentage = % off
#             (no 'category' text column — category name comes from categories(name) join)
# restaurant_tables: restaurant_id, table_number, capacity


# ── Context snapshot (pre-fetched before every LLM call) ─────────────────────

async def get_partner_context(restaurant_id: str) -> dict:
    """Rich context snapshot — pre-fetched before every LLM call.
    Includes actual names/numbers so the LLM can answer basic questions
    directly from context without an extra tool call."""
    if not restaurant_id:
        return {}
    try:
        db = _db()
        start_today, _ = _today_range()

        # Today's orders — only select columns that actually exist
        orders_res = (
            db.table("orders")
            .select("id, status, total_amount, table_number, created_at")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start_today)
            .order("created_at", desc=True)
            .execute()
        )
        orders = orders_res.data or []
        total_revenue = sum(_order_total(o) for o in orders)
        pending   = [o for o in orders if o.get("status") in ("pending", "accepted", "cooking", "ready")]
        completed = sum(1 for o in orders if o.get("status") in ("completed", "delivered"))

        # Menu items — full detail for caching (category name via join)
        menu_res = (
            db.table("menu_items")
            .select(
                "id, name, price, category_id, is_available, description, "
                "offer_name, discount_percentage, original_price, offer_expires_at, "
                "categories(name)"
            )
            .eq("restaurant_id", restaurant_id)
            .order("name")
            .execute()
        )
        menu_items = menu_res.data or []
        for item in menu_items:
            item["category"] = _cat_name(item)
        available_items = [i for i in menu_items if i.get("is_available", True)]

        # QR tables — actual numbers
        tables_res = (
            db.table("restaurant_tables")
            .select("table_number")
            .eq("restaurant_id", restaurant_id)
            .order("table_number")
            .execute()
        )
        table_numbers = [t["table_number"] for t in (tables_res.data or [])]

        # Restaurant basic info
        rest_res = (
            db.table("restaurants")
            .select("name, is_open, min_order, delivery_fee, tax_percent, currency, city, opens_at, closes_at, is_delivery, rating")
            .eq("id", restaurant_id)
            .limit(1)
            .execute()
        )
        rest = (rest_res.data or [{}])[0]

        # Active order quick summary (table + status) for top 5
        active_summary = [
            f"Table {o.get('table_number','?')} [{o.get('status','?')}]"
            for o in pending[:5]
        ]

        # Top menu names for quick reference
        sample_menu = [
            f"{i['name']} Rs.{i['price']}"
            for i in available_items[:8]
        ]

        # Unique category names — extracted from the existing join, no extra query
        category_names = sorted({
            _cat_name(i) for i in menu_items
            if _cat_name(i) not in ("Uncategorized", "")
        })

        result = {
            "restaurant_name": rest.get("name", "Restaurant"),
            "is_open": rest.get("is_open", True),
            "min_order": rest.get("min_order") or rest.get("min_order_price", 0),
            "delivery_fee": rest.get("delivery_fee", 0),
            "tax_percent": rest.get("tax_percent", 0),
            "currency": rest.get("currency", "PKR"),
            "city": rest.get("city", ""),
            "opens_at": rest.get("opens_at", ""),
            "closes_at": rest.get("closes_at", ""),
            "is_delivery": rest.get("is_delivery", True),
            "rating": rest.get("rating", 4.5),
            "today_orders": len(orders),
            "today_revenue": round(total_revenue, 2),
            "pending_count": len(pending),
            "completed_orders": completed,
            "active_orders_summary": active_summary,
            "total_menu_items": len(menu_items),
            "available_menu_count": len(available_items),
            "sample_menu": sample_menu,
            "total_tables": len(table_numbers),
            "tables": table_numbers,
            "categories": category_names,
            "menu_items": menu_items,
        }
        # Detect RLS block: menu works (public) but orders+tables returned 0
        if menu_items and not orders and not table_numbers:
            print(
                "[RLS ALERT] menu has items but orders=0 and tables=0 — "
                "anon key is blocked by Row Level Security. "
                "FIX: add SUPABASE_SERVICE_ROLE_KEY to .env and restart."
            )
        return result
    except Exception as e:
        print(f"[Context Error] {e}")
        return {"error": str(e)}


# ── Tool Functions ─────────────────────────────────────────────────────────────

async def get_total_revenue(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, end = _date_range_from_params(params)
        res = (
            db.table("orders")
            .select("total_amount")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .lte("created_at", end)
            .execute()
        )
        total = sum(_order_total(r) for r in (res.data or []))
        period_label = f"{start[:10]}"
        if start[:10] == end[:10]:
            period_label = "today" if start[:10] == _today_range()[0][:10] else start[:10]
        else:
            period_label = f"{start[:10]} to {end[:10]}"
        return {
            "data": {"revenue": round(total, 2), "period": period_label,
                     "start": start[:10], "end": end[:10]},
            "summary": f"Revenue ({period_label}): Rs.{round(total, 2)}"
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_total_orders(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, end = _date_range_from_params(params)
        res = (
            db.table("orders")
            .select("id, status", count="exact")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .execute()
        )
        count = res.count or len(res.data or [])
        return {"data": {"total_orders": count}, "summary": f"Total orders: {count}"}
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_active_orders(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        res = (
            db.table("orders")
            .select("id, status, total_amount, table_number, customer_name, created_at")
            .eq("restaurant_id", restaurant_id)
            .in_("status", ["pending", "accepted", "cooking", "ready"])
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
        orders = res.data or []
        return {
            "data": {"orders": orders, "count": len(orders)},
            "summary": f"{len(orders)} active orders"
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def filter_by_time_range(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, end = _date_range_from_params(params)
        res = (
            db.table("orders")
            .select("id, status, total_amount, created_at, table_number")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .lte("created_at", end)
            .order("created_at", desc=True)
            .execute()
        )
        orders = res.data or []
        revenue = sum(_order_total(o) for o in orders)
        return {
            "data": {"orders": orders[:20], "count": len(orders), "revenue": round(revenue, 2),
                     "start": start[:10], "end": end[:10]},
            "summary": f"{len(orders)} orders, revenue Rs.{round(revenue, 2)}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_revenue_trend_chart(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, end = _week_range()
        res = (
            db.table("orders")
            .select("total_amount, created_at")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .execute()
        )
        daily: dict = {}
        for o in (res.data or []):
            day = (o.get("created_at") or "")[:10]
            daily[day] = round(daily.get(day, 0) + _order_total(o), 2)
        trend = [{"date": k, "revenue": v} for k, v in sorted(daily.items())]
        total = sum(t["revenue"] for t in trend)
        return {
            "data": {"trend": trend, "total_7d": round(total, 2)},
            "summary": f"7-day revenue: Rs.{round(total, 2)}"
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_order_volume_data(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, _ = _week_range()
        res = (
            db.table("orders")
            .select("created_at, status")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .execute()
        )
        daily: dict = {}
        for o in (res.data or []):
            day = (o.get("created_at") or "")[:10]
            daily[day] = daily.get(day, 0) + 1
        volume = [{"date": k, "orders": v} for k, v in sorted(daily.items())]
        total = sum(v["orders"] for v in volume)
        return {
            "data": {"volume": volume, "total_7d": total},
            "summary": f"7-day total orders: {total}"
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_peak_order_times(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, _ = _week_range()
        res = (
            db.table("orders")
            .select("created_at")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .execute()
        )
        hourly: dict = {}
        for o in (res.data or []):
            ts = o.get("created_at", "")
            if len(ts) >= 13:
                hour = ts[11:13]
                hourly[hour] = hourly.get(hour, 0) + 1
        peak = sorted(hourly.items(), key=lambda x: -x[1])[:5]
        return {
            "data": {"peak_hours": [{"hour": f"{h}:00", "orders": c} for h, c in peak]},
            "summary": f"Peak: {peak[0][0]}:00 ({peak[0][1]} orders)" if peak else "No peak data",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_top_items(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, _ = _week_range()
        # Try order_items join first, fallback to orders.items JSON
        res = (
            db.table("order_items")
            .select("item_name, quantity, orders!inner(restaurant_id, created_at)")
            .eq("orders.restaurant_id", restaurant_id)
            .gte("orders.created_at", start)
            .execute()
        )
        counts: dict = {}
        for item in (res.data or []):
            name = item.get("item_name") or "Unknown"
            counts[name] = counts.get(name, 0) + (item.get("quantity") or 1)
        top = sorted(counts.items(), key=lambda x: -x[1])[:10]
        return {
            "data": {"top_items": [{"name": n, "count": c} for n, c in top]},
            "summary": f"Top: {top[0][0]} x{top[0][1]}" if top else "No data",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_today_summary(restaurant_id: str, params: dict) -> dict:
    try:
        rev = await get_total_revenue(restaurant_id, {})
        orders = await get_total_orders(restaurant_id, {})
        active = await get_active_orders(restaurant_id, {})
        revenue_val = (rev.get("data") or {}).get("revenue", 0)
        orders_val  = (orders.get("data") or {}).get("total_orders", 0)
        active_val  = (active.get("data") or {}).get("count", 0)
        return {
            "data": {
                "today_revenue": revenue_val,
                "today_orders": orders_val,
                "active_orders": active_val,
            },
            "summary": f"Today: {orders_val} orders, Rs.{revenue_val} revenue, {active_val} active",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_customer_insights(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        start, _ = _week_range()
        res = (
            db.table("orders")
            .select("customer_id, total_amount")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .execute()
        )
        orders = res.data or []
        if not orders:
            return {"data": {"message": "No orders this week"}, "summary": "No data"}
        totals = [_order_total(o) for o in orders]
        avg = round(sum(totals) / len(totals), 2)
        customers = [o.get("customer_id") for o in orders if o.get("customer_id")]
        unique = len(set(customers))
        return {
            "data": {"avg_order_value": avg, "unique_customers": unique, "total_orders": len(orders)},
            "summary": f"{unique} unique customers, avg order Rs.{avg}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def list_orders(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        status = params.get("status")
        start, end = _date_range_from_params(params)
        query = (
            db.table("orders")
            .select("id, status, total_amount, table_number, customer_name, created_at")
            .eq("restaurant_id", restaurant_id)
            .gte("created_at", start)
            .order("created_at", desc=True)
            .limit(25)
        )
        if status:
            query = query.eq("status", status)
        res = query.execute()
        orders = res.data or []
        return {"data": {"orders": orders, "count": len(orders)}, "summary": f"{len(orders)} orders"}
    except Exception as e:
        return {"data": None, "error": str(e)}


async def update_order_status(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        order_id = params.get("order_id") or params.get("target", "")
        new_status = params.get("status") or params.get("new_status", "")
        if not order_id or not new_status:
            return {"data": None, "error": "order_id and status required"}
        db.table("orders").update({"status": new_status}).eq("id", order_id).eq("restaurant_id", restaurant_id).execute()
        return {
            "data": {"updated": True, "order_id": order_id, "new_status": new_status},
            "summary": f"Order {order_id[-6:]} → {new_status}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def get_menu_items(restaurant_id: str, params: dict) -> dict:
    """Fetch ALL menu items. Serves from in-memory cache when fresh; falls back to Supabase."""
    # ── Try cache first ──────────────────────────────────────────────────────
    cached = cache_manager.get_menu_items(restaurant_id)
    if cached is not None:
        items = cached
        print(f"[Cache] GET_MENU served {len(items)} items from cache")
    else:
        # ── Fall back to Supabase ────────────────────────────────────────────
        try:
            db = _db()
            res = (
                db.table("menu_items")
                .select(
                    "id, name, price, category_id, is_available, description, "
                    "offer_name, discount_percentage, original_price, offer_expires_at, "
                    "categories(name)"
                )
                .eq("restaurant_id", restaurant_id)
                .order("name")
                .execute()
            )
            items = res.data or []
            for item in items:
                item["category"] = _cat_name(item)
        except Exception as e:
            return {"data": None, "error": str(e)}

    cat_groups: dict = {}
    for item in items:
        cat_groups.setdefault(item.get("category", "Uncategorized"), []).append(item)

    available  = [i for i in items if i.get("is_available", True)]
    discounted = [i for i in items if float(i.get("discount_percentage") or 0) > 0]

    return {
        "data": {
            "menu_items": items,
            "count": len(items),
            "available_count": len(available),
            "unavailable_count": len(items) - len(available),
            "discounted_count": len(discounted),
            "categories": list(cat_groups.keys()),
            "category_counts": {k: len(v) for k, v in cat_groups.items()},
        },
        "summary": f"{len(items)} menu items in {len(cat_groups)} categories",
    }


async def search_menu_items(restaurant_id: str, params: dict) -> dict:
    """Search menu by name/category/description. Filters in-memory cache; falls back to Supabase."""
    query_str = (params.get("query") or params.get("target") or "").strip().lower()

    # ── Try cache first ──────────────────────────────────────────────────────
    cached = cache_manager.get_menu_items(restaurant_id)
    if cached is not None:
        items = cached
        print(f"[Cache] SEARCH_MENU filtering {len(items)} cached items for '{query_str}'")
    else:
        # ── Fall back to Supabase ────────────────────────────────────────────
        try:
            db = _db()
            res = (
                db.table("menu_items")
                .select(
                    "id, name, price, category_id, is_available, "
                    "discount_percentage, original_price, offer_name, description, "
                    "categories(name)"
                )
                .eq("restaurant_id", restaurant_id)
                .execute()
            )
            items = res.data or []
            for item in items:
                item["category"] = _cat_name(item)
        except Exception as e:
            return {"data": None, "error": str(e)}

    if query_str:
        items = [
            i for i in items
            if query_str in (i.get("name") or "").lower()
            or query_str in (i.get("category") or "").lower()
            or query_str in (i.get("description") or "").lower()
        ]
    return {
        "data": {"menu_items": items, "count": len(items), "query": query_str},
        "summary": f"{len(items)} items matching '{query_str}'",
    }


async def add_menu_item(restaurant_id: str, params: dict) -> dict:
    """Insert a new menu item. LLM passes category name; we resolve to category_id.
    If the category doesn't match, we auto-pick the best existing one by name similarity."""
    try:
        db = _db()
        name        = (params.get("name") or params.get("target") or "New Dish").strip()
        price       = float(params.get("price") or 0)
        category    = (params.get("category") or "").strip()
        description = (params.get("description") or "").strip()
        # deal / offer fields (optional)
        offer_name     = (params.get("offer_name") or params.get("deal_name") or "").strip()
        offer_discount = float(params.get("offer_discount") or params.get("discount_pct") or 0)

        # ── Resolve category_id ──────────────────────────────────────────────
        category_id: str | None = None
        # 1. Try exact/fuzzy match on what LLM provided
        if category:
            try:
                cat_res = (
                    db.table("categories")
                    .select("id, name")
                    .eq("restaurant_id", restaurant_id)
                    .ilike("name", f"%{category}%")
                    .limit(1)
                    .execute()
                )
                if cat_res.data:
                    category_id = cat_res.data[0]["id"]
                    category    = cat_res.data[0]["name"]  # normalise to DB name
            except Exception:
                pass

        # 2. If still no match, fetch all categories and pick best by item-name similarity
        if not category_id:
            try:
                all_cats = (
                    db.table("categories")
                    .select("id, name")
                    .eq("restaurant_id", restaurant_id)
                    .execute()
                )
                cat_list = all_cats.data or []
                if cat_list:
                    name_lower = name.lower()
                    # Score each category: how many of its words appear in the item name
                    def _score(cat: dict) -> int:
                        return sum(
                            1 for w in cat["name"].lower().split()
                            if len(w) > 2 and w in name_lower
                        )
                    best = max(cat_list, key=_score)
                    category_id = best["id"]
                    category    = best["name"]
            except Exception:
                pass

        # ── Build insert payload ─────────────────────────────────────────────
        new_item: dict = {
            "restaurant_id": restaurant_id,
            "name":          name,
            "price":         price,
            "description":   description,
            "is_available":  True,
        }
        if category_id:
            new_item["category_id"] = category_id
        if offer_name:
            new_item["offer_name"] = offer_name
        discounted_price: float | None = None
        if offer_discount > 0:
            discounted_price = round(price * (1 - offer_discount / 100), 2)
            new_item["original_price"]      = price           # save the full price
            new_item["price"]               = discounted_price # update price to discounted
            new_item["discount_percentage"] = offer_discount

        res = db.table("menu_items").insert(new_item).execute()
        inserted = (res.data or [{}])[0]
        # Flatten category so the cached item has it immediately
        inserted["category"] = category or "Uncategorized"
        summary = f"'{name}' added — Rs.{discounted_price if discounted_price else price}"
        if offer_discount:
            summary += f" with {offer_discount}% off"
        summary += f" in '{category}'" if category else ""
        return {
            "data": {
                "added": True,
                "item": inserted,
                "name": name,
                "price": discounted_price if discounted_price else price,
                "original_price": price if discounted_price else None,
                "category": category, "category_id": category_id,
                "discount_percentage": offer_discount,
            },
            "summary": summary,
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def update_menu_item_by_name(restaurant_id: str, params: dict) -> dict:
    """Update a menu item field by dish name. Params: item_name, field, value."""
    try:
        db = _db()
        item_name = params.get("item_name") or params.get("name") or params.get("target", "")
        field     = params.get("field", "")
        value     = params.get("value")

        if not item_name or not field:
            return {"data": None, "error": "item_name and field required"}

        search = (
            db.table("menu_items")
            .select("id, name")
            .eq("restaurant_id", restaurant_id)
            .ilike("name", f"%{item_name}%")
            .limit(1)
            .execute()
        )
        if not search.data:
            return {"data": None, "error": f"Item '{item_name}' not found", "summary": f"'{item_name}' nahi mila"}

        item_id    = search.data[0]["id"]
        found_name = search.data[0]["name"]

        # If updating category, resolve category name → category_id
        update_field = field
        update_value = value
        if field == "category":
            try:
                cat_res = (
                    db.table("categories")
                    .select("id")
                    .eq("restaurant_id", restaurant_id)
                    .ilike("name", f"%{value}%")
                    .limit(1)
                    .execute()
                )
                if cat_res.data:
                    update_field = "category_id"
                    update_value = cat_res.data[0]["id"]
                else:
                    return {"data": None, "error": f"Category '{value}' not found"}
            except Exception:
                return {"data": None, "error": "Category lookup failed"}

        db.table("menu_items").update({update_field: update_value}).eq("id", item_id).execute()
        return {
            "data": {"updated": True, "name": found_name, "field": field, "value": value},
            "summary": f"'{found_name}' {field} → {value}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def toggle_menu_availability(restaurant_id: str, params: dict) -> dict:
    """Toggle is_available on a menu item by name."""
    try:
        db = _db()
        item_name = params.get("item_name") or params.get("name") or params.get("target", "")
        available = params.get("available")

        search = (
            db.table("menu_items")
            .select("id, name, is_available")
            .eq("restaurant_id", restaurant_id)
            .ilike("name", f"%{item_name}%")
            .limit(1)
            .execute()
        )
        if not search.data:
            return {"data": None, "error": f"Item '{item_name}' not found"}

        item = search.data[0]
        new_val = (not item.get("is_available")) if available is None else bool(available)
        db.table("menu_items").update({"is_available": new_val}).eq("id", item["id"]).execute()

        status_word = "available" if new_val else "unavailable"
        return {
            "data": {"updated": True, "name": item["name"], "is_available": new_val},
            "summary": f"'{item['name']}' ab {status_word}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def apply_item_discount(restaurant_id: str, params: dict) -> dict:
    """Apply a discount % to a menu item by name. Params: item_name, discount_pct, offer_name."""
    try:
        db = _db()
        item_name    = params.get("item_name") or params.get("name") or params.get("target", "")
        discount_pct = float(params.get("discount_pct") or params.get("discount") or 0)
        offer_name   = params.get("offer_name") or f"{int(discount_pct)}% OFF"

        search = (
            db.table("menu_items")
            .select("id, name, price")
            .eq("restaurant_id", restaurant_id)
            .ilike("name", f"%{item_name}%")
            .limit(1)
            .execute()
        )
        if not search.data:
            return {"data": None, "error": f"Item '{item_name}' not found"}

        item           = search.data[0]
        original_price = float(item.get("price") or 0)
        discounted_price = round(original_price * (1 - discount_pct / 100), 2)
        expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        db.table("menu_items").update({
            "offer_name":          offer_name,
            "discount_percentage": discount_pct,
            "original_price":      original_price,   # save pre-discount price
            "price":               discounted_price, # update price to discounted value
            "offer_expires_at":    expires,
        }).eq("id", item["id"]).execute()

        return {
            "data": {
                "updated": True, "name": item["name"],
                "original_price": original_price,
                "discounted_price": discounted_price,
                "discount_pct": discount_pct,
            },
            "summary": f"'{item['name']}' {int(discount_pct)}% OFF: Rs.{original_price} → Rs.{discounted_price}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


# ── QR Table Management ───────────────────────────────────────────────────────

async def get_qr_tables(restaurant_id: str, params: dict) -> dict:
    """List all QR code tables for the restaurant."""
    try:
        db = _db()
        res = (
            db.table("restaurant_tables")
            .select("table_number, capacity")
            .eq("restaurant_id", restaurant_id)
            .order("table_number", desc=False)
            .execute()
        )
        tables = [t["table_number"] for t in (res.data or [])]
        return {
            "data": {"tables": tables, "count": len(tables)},
            "summary": f"{len(tables)} tables: {', '.join('Table ' + str(t) for t in tables)}" if tables else "No tables configured"
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def add_qr_table(restaurant_id: str, params: dict) -> dict:
    """Add a new table for QR code generation."""
    try:
        db = _db()
        table_no = int(params.get("target") or params.get("table_no") or 0)
        if not table_no:
            return {"data": None, "error": "Table number required"}
        db.table("restaurant_tables").upsert({
            "restaurant_id": restaurant_id,
            "table_number": table_no,
            "capacity": 4,
        }, on_conflict="restaurant_id,table_number").execute()
        return {
            "data": {"added": True, "table_no": table_no},
            "summary": f"Table {table_no} QR add ho gaya",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def delete_qr_table(restaurant_id: str, params: dict) -> dict:
    """Delete a table's QR code."""
    try:
        db = _db()
        table_no = int(params.get("target") or params.get("table_no") or 0)
        if not table_no:
            return {"data": None, "error": "Table number required"}
        db.table("restaurant_tables").delete().eq("restaurant_id", restaurant_id).eq("table_number", table_no).execute()
        return {
            "data": {"deleted": True, "table_no": table_no},
            "summary": f"Table {table_no} delete ho gaya",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def generate_qr_code(restaurant_id: str, params: dict) -> dict:
    table_no = params.get("table_no") or params.get("target", "")
    base_url = os.getenv("VITE_APP_URL", "http://localhost:5173")
    qr_url   = f"{base_url}/menu/{restaurant_id}?table={table_no}"
    return {
        "signal": "NAVIGATE",
        "data":   {"qr_url": qr_url, "table_no": table_no},
        "summary": f"QR for table {table_no}",
    }


async def show_quick_actions(restaurant_id: str, params: dict) -> dict:
    actions = [
        "Aaj ki revenue batao",
        "Is hafte ka revenue batao",
        "Active orders dikhao",
        "Pury menu ki list do with prices",
        "Pizza items search karo",
        "Table 5 ka QR banao",
        "Kitne tables hain?",
        "Order X ka status update karo",
        "Top selling items batao",
        "Peak hours batao",
        "Biryani par 20% discount lagao",
        "Restaurant online karo",
    ]
    return {"data": {"commands": actions}, "summary": "Quick voice commands list"}


async def update_restaurant_settings(restaurant_id: str, params: dict) -> dict:
    try:
        db = _db()
        field = params.get("field", "")
        value = params.get("value")
        if not field:
            return {"data": None, "error": "field required"}
        db.table("restaurants").update({field: value}).eq("id", restaurant_id).execute()
        return {
            "data": {"updated": True, "field": field, "value": value},
            "summary": f"Restaurant {field} updated to {value}",
        }
    except Exception as e:
        return {"data": None, "error": str(e)}


async def perform_logout(restaurant_id: str, params: dict) -> dict:
    return {"signal": "LOGOUT", "data": {}, "summary": "Logout signal sent"}


# ── Router ────────────────────────────────────────────────────────────────────

_TOOL_MAP = {
    # Dashboard data
    "GET_REVENUE":      get_total_revenue,
    "GET_ORDERS":       get_total_orders,
    "GET_ACTIVE_ORDERS": get_active_orders,
    "FILTER_TIME":      filter_by_time_range,
    "GET_CHART":        get_revenue_trend_chart,
    "GET_VOLUME":       get_order_volume_data,
    "GET_PEAK":         get_peak_order_times,
    "GET_TOP_ITEMS":    get_top_items,
    "GET_SUMMARY":      get_today_summary,
    "GET_INSIGHTS":     get_customer_insights,
    # Orders
    "LIST_ORDERS":      list_orders,
    "UPDATE_ORDER":     update_order_status,
    # Menu
    "GET_MENU":         get_menu_items,
    "SEARCH_MENU":      search_menu_items,
    "ADD_MENU_ITEM":    add_menu_item,
    "UPDATE_MENU":      update_menu_item_by_name,
    "UPDATE_MENU_BY_NAME": update_menu_item_by_name,
    "RENAME_ITEM":      update_menu_item_by_name,
    "TOGGLE_ITEM":      toggle_menu_availability,
    "APPLY_DISCOUNT":   apply_item_discount,
    # QR tables
    "GET_TABLES":       get_qr_tables,
    "ADD_TABLE":        add_qr_table,
    "DELETE_TABLE":     delete_qr_table,
    "GEN_QR":           generate_qr_code,
    # Other
    "SHOW_ACTIONS":     show_quick_actions,
    "UPDATE_SETTINGS":  update_restaurant_settings,
    "LOGOUT":           perform_logout,
}


# Common LLM typos / alternate spellings → canonical action code
_ACTION_ALIASES: dict[str, str] = {
    # Orders
    "GET_ORDER": "GET_ORDERS",
    "GET_ALL_ORDERS": "LIST_ORDERS",
    "ORDERS": "LIST_ORDERS",
    "ACTIVE_ORDERS": "GET_ACTIVE_ORDERS",
    "GET_PENDING_ORDERS": "GET_ACTIVE_ORDERS",
    "PENDING_ORDERS": "GET_ACTIVE_ORDERS",
    "UPDATE_ORDER_STATUS": "UPDATE_ORDER",
    "CHANGE_ORDER_STATUS": "UPDATE_ORDER",
    # Revenue
    "REVENUE": "GET_REVENUE",
    "GET_SALES": "GET_REVENUE",
    "SALES": "GET_REVENUE",
    "KAMAI": "GET_REVENUE",
    # Menu
    "MENU": "GET_MENU",
    "GET_MENU_ITEMS": "GET_MENU",
    "LIST_MENU": "GET_MENU",
    "SEARCH": "SEARCH_MENU",
    "SEARCH_ITEM": "SEARCH_MENU",
    "MENU_SEARCH": "SEARCH_MENU",
    "ADD_ITEM": "ADD_MENU_ITEM",
    "ADD_DISH": "ADD_MENU_ITEM",
    "UPDATE_ITEM": "UPDATE_MENU_BY_NAME",
    "UPDATE_PRICE": "UPDATE_MENU_BY_NAME",
    "TOGGLE_AVAILABILITY": "TOGGLE_ITEM",
    "TOGGLE_MENU_ITEM": "TOGGLE_ITEM",
    "DISCOUNT": "APPLY_DISCOUNT",
    "SET_DISCOUNT": "APPLY_DISCOUNT",
    # QR / Tables
    "QR": "GEN_QR",
    "QR_CODE": "GEN_QR",
    "GENERATE_QR": "GEN_QR",
    "TABLES": "GET_TABLES",
    "LIST_TABLES": "GET_TABLES",
    "TABLE_LIST": "GET_TABLES",
    "ADD_QR_TABLE": "ADD_TABLE",
    "DELETE_QR_TABLE": "DELETE_TABLE",
    "REMOVE_TABLE": "DELETE_TABLE",
    # Summary / insights
    "SUMMARY": "GET_SUMMARY",
    "TODAY_SUMMARY": "GET_SUMMARY",
    "DASHBOARD_SUMMARY": "GET_SUMMARY",
    "INSIGHTS": "GET_INSIGHTS",
    "CUSTOMER_INSIGHTS": "GET_INSIGHTS",
    "PEAK": "GET_PEAK",
    "PEAK_HOURS": "GET_PEAK",
    "TOP_ITEMS": "GET_TOP_ITEMS",
    "TOP_DISHES": "GET_TOP_ITEMS",
    "CHART": "GET_CHART",
    "REVENUE_CHART": "GET_CHART",
    "VOLUME": "GET_VOLUME",
    "ORDER_VOLUME": "GET_VOLUME",
    # Settings
    "SETTINGS": "UPDATE_SETTINGS",
    "RESTAURANT_SETTINGS": "UPDATE_SETTINGS",
    # Auth
    "SIGN_OUT": "LOGOUT",
    "LOG_OUT": "LOGOUT",
}


async def execute_partner_tool(action: str, target: str, params: dict, restaurant_id: str) -> dict | None:
    # Guard: partner tools require a valid restaurant_id UUID.
    # Empty string causes Supabase to throw "invalid input syntax for type uuid: ''"
    if not restaurant_id or not restaurant_id.strip():
        print(f"[Tool] Skipping {action} — restaurant_id is empty (partner not logged in?)")
        return {
            "data": None,
            "error": "restaurant_id missing — please log in as a partner first",
            "summary": "Login required",
        }

    # Normalize: try exact match first, then alias table, then case-insensitive
    canonical = _TOOL_MAP.get(action) and action  # exact match
    if canonical is None:
        canonical = _ACTION_ALIASES.get(action)  # alias
    if canonical is None:
        canonical = _ACTION_ALIASES.get(action.upper())  # uppercase alias
    resolved = canonical or action

    fn = _TOOL_MAP.get(resolved)
    if fn is None:
        print(f"[Tool] Unknown action: '{action}' (resolved='{resolved}') — not in _TOOL_MAP")
        return None
    merged_params = {**(params or {}), "target": target}
    return await fn(restaurant_id, merged_params)
