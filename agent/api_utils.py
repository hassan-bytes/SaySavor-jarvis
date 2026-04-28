import httpx
import os
import logging
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Logger set karein taake errors ka pata chalay
logger = logging.getLogger("api_utils")

# Aapki website ka local ya live address
# Agar local test kar rahe hain toh: http://localhost:3000
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
# Woh secret key jo dono side (Python aur Next.js) par same honi chahiye
AI_SECRET = os.getenv("AI_SECRET") or os.getenv("INTERNAL_AI_SECRET")

if not AI_SECRET:
    raise ValueError("ERROR: AI_SECRET ya INTERNAL_AI_SECRET .env file mein nahi mila!")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
if not supabase_url or "your-project-ref" in supabase_url:
    # Fallback to extracting from BACKEND_URL if possible
    if "supabase.co" in BACKEND_URL:
        supabase_url = "https://" + BACKEND_URL.split("/")[2]

supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("ERROR: SUPABASE_URL ya SUPABASE_KEY .env file mein nahi milay!")

supabase: Client = create_client(supabase_url, supabase_key)


async def get_restaurant_context(restaurant_id: str):
    """
    Next.js Backend se restaurant aur menu ka saara data fetch karna.
    """
    # Endpoint URL
    endpoint = f"{BACKEND_URL}"
    # Anon Key aur Secret dono bhejna zaroori hain
    headers = {
        "Authorization": f"Bearer {os.getenv('SUPABASE_ANON_KEY')}",
        "x-ai-secret": AI_SECRET,
        "Content-Type": "application/json"
    }
    # Body mein ID bhej rahe hain
    payload = {"res_id": restaurant_id}

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Fetching menu for ID: {restaurant_id} via POST with Auth")
            # POST request bhej rahe hain
            response = await client.post(endpoint, headers=headers, json=payload, timeout=10.0)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Agar error aaye toh log mein show hoga
                logger.error(f"Backend API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Backend se connection nahi ho saka: {e}")
            return None

async def create_order_in_db(res_id, items_text, price, table, name="Guest", address="Dine-in", payment="CASH"):
    """
    AI ke text ko aik proper list mein badalna zaroori hai taake woh database ki umeedon par poora utray.
    Ab hum is mein customer name, address aur payment method bhi shamil kar rahe hain.
    """
    endpoint = os.getenv("BACKEND_URL", "").replace("get-jarvis-menu", "create-order")
    
    payload = {
        "res_id": res_id,
        "table_no": table,
        "total_price": price,
        "customer_name": name,
        "customer_address": address,
        "payment_method": payment,
        "items": [
            {
                "name": items_text, # Maslan: "1 Zinger Burger"
                "quantity": 1, 
                "price": price
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('SUPABASE_ANON_KEY')}",
        "x-ai-secret": AI_SECRET,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Placing structured order for ID: {res_id}")
            response = await client.post(endpoint, headers=headers, json=payload, timeout=15.0)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.error(f"Order connection failed: {e}")
            return None

async def get_active_order(res_id, query):
    try:
        # 'or' filter use karke sab columns dhoondein
        response = supabase.table("orders") \
            .select("*") \
            .eq("restaurant_id", res_id) \
            .or_(f"customer_name.ilike.%{query}%,customer_address.ilike.%{query}%,table_number.eq.{query}") \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
            
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error(f"Failed to get active order: {e}")
        return None

async def process_cancellation(res_id, query, reason):
    try:
        # 1. Pehle order dhoondo (Latest Order)
        response = supabase.table("orders") \
            .select("id, created_at") \
            .eq("restaurant_id", res_id) \
            .or_(f"customer_name.ilike.%{query}%,table_number.eq.{query}") \
            .order("created_at", desc=True).limit(1).execute()

        if not response.data:
            return {"status": "not_found"}

        order = response.data[0]
        # 'created_at' format is ISO string, might have 'Z' or +00:00
        created_at_str = order['created_at'].replace('Z', '+00:00')
        # Ensure it has timezone info, if not assume UTC
        created_at = datetime.fromisoformat(created_at_str)
        now = datetime.now(timezone.utc)

        # 2. Time Difference calculate karein (5 minute rule)
        if now - created_at > timedelta(minutes=5):
            return {"status": "too_late"}

        # 3. Agar 5 min ke andar hai toh cancel kar dein
        supabase.table("orders").update({
            "status": "cancelled", 
            "notes": f"Reason: {reason}" # Notes mein wajah save karein
        }).eq("id", order['id']).execute()

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to process cancellation: {e}")
        return {"status": "error"}
