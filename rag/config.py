import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    embedding_model: str = os.getenv("EMBEDDING_MODEL",
    "gemini-embedding-001")
    chat_model: str = os.getenv("CHAT_MODEL",
    "gemini-3.8-flash")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    collection: str = "documents"
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    # Cosine similarity (1 - cosine distance). Below this, we refuse.
    min_relevance: float = float(os.getenv("MIN_RELEVANCE", "0.30"))


CONFIG = Config()
