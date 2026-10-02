from collections import Counter

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .chunking import Chunk
from .config import CONFIG, Config


def get_store(cfg: Config = CONFIG) -> Chroma:
    return Chroma(
        collection_name=cfg.collection,
        embedding_function=GoogleGenerativeAIEmbeddings(
            model=cfg.embedding_model
        ),
        persist_directory=cfg.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(store: Chroma, chunks: list[Chunk], source: str, batch: int = 100) -> int:
    store.delete(where={"source": source})  # re-ingesting a changed file leaves no stale chunks
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        store.add_texts(
            texts=[c.text for c in part],
            metadatas=[c.metadata for c in part],
            ids=[c.id for c in part],
        )
    return len(chunks)


def list_sources(store: Chroma) -> dict[str, int]:
    metas = store.get(include=["metadatas"])["metadatas"] or []
    return dict(Counter(m["source"] for m in metas))


def delete_source(store: Chroma, source: str) -> None:
    store.delete(where={"source": source})
