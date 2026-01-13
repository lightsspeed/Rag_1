import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File tracking
PROCESSED_FILES_JSON = DATA_DIR / "processed_files.json"

# Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
VISION_MODEL = "llava:13b"

# ChromaDB settings
COLLECTION_NAME = "pdf_knowledge_base"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# Processing settings
CHUNK_SIZE = 500  # characters per text chunk
MIN_IMAGE_SIZE = 100  # minimum image dimension in pixels
MAX_CHAT_HISTORY = 6  # messages to keep in context

# Web search settings
WEB_SEARCH_ENABLED = True
SIMILARITY_THRESHOLD = 0.7  # Above this: PDF only, Below: add web search

# API Keys - Add your keys here
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")  # Get from https://brave.com/search/api/
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")  # Get from https://tavily.com/

# You can also set these in environment variables:
# export BRAVE_API_KEY="your_key_here"
# export TAVILY_API_KEY="your_key_here"

# Web search provider: "brave" or "tavily"
WEB_SEARCH_PROVIDER = "brave" if BRAVE_API_KEY else ("tavily" if TAVILY_API_KEY else None)

# Logging
LOG_LEVEL = "INFO"