import os

CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
COLLECTION = "knowledge_base"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 150))
TOP_K = int(os.environ.get("TOP_K", 5))