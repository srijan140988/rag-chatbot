"""Offline tests (no API key needed): chunking invariants, citation parsing, refusal gate."""
from pathlib import Path

from rag.chunking import chunk_documents
from rag.config import Config
from rag.generator import parse_citations
from rag.loaders import RawDoc, load_file
from rag.pipeline import RAGPipeline
from rag.tokens import count_tokens

SAMPLE = Path(__file__).parent.parent / "data" / "sample"
CFG = Config(chunk_tokens=60, chunk_overlap=10)


def test_csv_is_row_atomic():
    docs = load_file(SAMPLE / "products.csv")
    chunks = chunk_documents(docs, CFG)
    assert len(chunks) == 6 == len(docs)  # one chunk per row, never split or merged
    assert all("price_inr:" in c.text and "name:" in c.text for c in chunks)  # header travels with every value
    assert chunks[0].metadata["row"] == 2


def test_csv_long_row_stays_atomic():
    long_row = RawDoc("note: " + "word " * 500, {"source": "x.csv", "type": "csv", "row": 2})
    assert len(chunk_documents([long_row], CFG)) == 1


def test_prose_respects_token_budget_and_ids_unique():
    chunks = chunk_documents(load_file(SAMPLE / "handbook.txt"), CFG)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= CFG.chunk_tokens + 5 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_parse_citations_flags_hallucinated_numbers():
    valid, invalid = parse_citations("Leave is 24 days [1][3]. Also [9].", n_hits=3)
    assert valid == [1, 3] and invalid == [9]


class _FakeStore:
    def similarity_search_with_score(self, q, k=5):
        from langchain_core.documents import Document

        return [(Document(page_content="irrelevant", metadata={"source": "a.txt", "type": "txt", "chunk_index": 0}), 0.9)]  # sim 0.1


def test_refuses_below_threshold_without_calling_llm():
    class Boom:
        def invoke(self, *_):
            raise AssertionError("LLM must not be called when retrieval is below threshold")

    ans = RAGPipeline(Config(min_relevance=0.3), store=_FakeStore(), llm=Boom()).ask("anything")
    assert ans.refused and ans.reason == "below_relevance_threshold"
