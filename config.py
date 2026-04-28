import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    PORT = int(os.getenv("PORT", 8001))
