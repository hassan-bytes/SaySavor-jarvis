import google.generativeai as genai

from config import Config


genai.configure(api_key=Config.GEMINI_KEY)

# System Instruction: Jarvis ko driver banana
SYSTEM_PROMPT = """
Aap SaySavor ke AI Agent hain. Aap Roman Urdu/Hinglish mein baat karte hain.
Aap ka kaam user ki help karna aur website ko control karna hai.
Hamesha JSON format mein reply dain:
{
  "reply": "User ko jo bolna hai",
  "action": "SEARCH | NAVIGATE | ADD_TO_CART | NONE",
  "target": "Action ki detail (e.g. 'Pizza' ya 'Orders')"
}
"""

model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction=SYSTEM_PROMPT,
)


async def get_jarvis_response(user_text: str) -> str:
    chat = model.start_chat()
    response = chat.send_message(user_text)
    return response.text  # Ye JSON string hogi
