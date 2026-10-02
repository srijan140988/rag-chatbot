"""Type-aware chunking.

Prose (PDF/TXT): token-based recursive splitting with overlap, so chunk size matches
what the embedding model actually sees.
CSV: row-atomic. A row is one chunk and is NEVER split or merged: splitting a row
separates a value from its column name and yields chunks that are unanswerable.
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CONFIG, Config
from .loaders import RawDoc
from .tokens import count_tokens


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def chunk_documents(docs: list[RawDoc], cfg: Config = CONFIG) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_tokens,
        chunk_overlap=cfg.chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for doc in docs:
        md = doc.metadata
        if md["type"] == "csv":
            chunks.append(Chunk(f"{md['source']}:row{md['row']}", doc.text, dict(md, chunk_index=0)))
            continue
        page_tag = f":p{md['page']}" if "page" in md else ""
        for i, text in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(f"{md['source']}{page_tag}:c{i}", text, dict(md, chunk_index=i, tokens=count_tokens(text)))
            )
    return chunks


def location_label(md: dict) -> str:
    """Human-readable pointer used in citations."""
    if md.get("type") == "csv":
        return f"{md['source']}, row {md['row']}"
    if md.get("type") == "pdf":
        return f"{md['source']}, page {md['page']}, chunk {md['chunk_index']}"
    return f"{md['source']}, chunk {md['chunk_index']}"
