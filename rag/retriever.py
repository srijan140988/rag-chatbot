from dataclasses import dataclass

from .config import CONFIG


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity in [-1, 1]; higher = more relevant


def retrieve(store, question: str, k: int = CONFIG.top_k) -> list[Hit]:
    """Top-k hits with cosine similarity. Thresholding happens in the pipeline so the
    eval harness can see raw scores and calibrate the cutoff."""
    results = store.similarity_search_with_score(question, k=k)
    hits = [
        Hit(id=(d.id or ""), text=d.page_content, metadata=d.metadata, score=round(1.0 - float(dist), 4))
        for d, dist in results
    ]
    return sorted(hits, key=lambda h: h.score, reverse=True)


def apply_threshold(hits: list[Hit], min_relevance: float) -> list[Hit]:
    return [h for h in hits if h.score >= min_relevance]
