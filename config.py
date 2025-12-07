# config.py - improved, drop-in replacement

from pathlib import Path
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

# --- Base paths ---
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploaded_docs"

# Ensure directories exist (create parents where necessary)
for d in (DATA_DIR, PERSIST_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- API Keys / tokens ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")           # optional if only using HF embeddings
HF_TOKEN = os.getenv("HF_TOKEN")                   # required for HuggingFaceInferenceAPIEmbeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")       # optional if you later use OpenAI

# --- Model configs (override with env vars if you like) ---
EMBEDDING_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() in ("1", "true", "yes")

# Model configs
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
NORMALIZE_EMBEDDINGS = True
LLM_MODEL = "llama-3.3-70b-versatile" 
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# --- RAG configs ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
SEPARATORS = ["\n\n", "\n", " ", ""]
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 8))
SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

# --- Database / limits ---
SQLITE_DB_PATH = DATA_DIR / os.getenv("SQLITE_DB_NAME", "chatbot.db")
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", 20))

# --- Logging (Rotating for safety) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "chatbot.log"

logger = logging.getLogger("rag_chatbot")
logger.setLevel(LOG_LEVEL)

# Add rotating file handler + console handler (avoid duplicate handlers on re-import)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    logger.addHandler(ch)

# SYSTEM_PROMPT ___

SYSTEM_PROMPT = """You are a helpful AI assistant with access to specific documents AND general knowledge.

When answering:
1. If context is relevant: Use it and cite it
2. You may supplement with general knowledge
3. If using general knowledge, say: "(Not from your documents)"
4. Always be helpful

Context: {context}
Question: {question}
Answer:"""


def get_user_upload_dir(user_id: str) -> Path:
    """
    Returns a directory where the user's uploaded documents are stored.
    """
    user_dir = UPLOAD_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_persist_dir(user_id: str) -> Path:
    """
    Returns a directory where the user's Chroma embeddings are stored.
    """
    user_dir = PERSIST_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

# --- Validation helper (non-fatal by default) ---
def validate_config(require_groq=False, require_hf=False):
    """
    Validate presence of API keys. By default this only logs warnings.
    Set require_groq=True to raise if GROQ_API_KEY missing, or require_hf=True to require HF_TOKEN.
    """
    missing = []
    if require_groq and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if require_hf and not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")
    # warn if neither is present (likely misconfiguration)
    if not GROQ_API_KEY and not HF_TOKEN and not OPENAI_API_KEY:
        logger.warning("No model API keys found (GROQ/HF/OPENAI). You may be unable to run LLMs or embeddings.")
    return True

# Try a soft validate (will not raise); if you want hard errors, call validate_config(require_groq=True) at runtime.
try:
    validate_config()
    logger.info("Configuration loaded successfully.")
except Exception as e:
    logger.error(f"Configuration validation failed: {e}")
    # don't re-raise here to allow dev flow; let runtime code decide

