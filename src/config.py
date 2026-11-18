import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
FAQ_JSON_PATH = BASE_DIR / "data" / "faq_dataset.json"

FAISS_INDEX_DIRECTORY = os.getenv("FAISS_INDEX_DIRECTORY", "./faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
