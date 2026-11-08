from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL = "http://localhost:11434"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"

CHROMA_PATH = str(BASE_DIR / "data" / "chroma")
PDF_FOLDER = str(BASE_DIR / "data" / "pdfs")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 15
