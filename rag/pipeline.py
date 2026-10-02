from dataclasses import dataclass, field, replace
from pathlib import Path

from .chunking import chunk_documents, location_label
from .config import CONFIG, Config
from .generator import NOT_FOUND, generate, parse_citations
from .loaders import load_file
from .retriever import Hit, apply_threshold, retrieve

REFUSAL_MESSAGE = "I couldn't find this in your documents, so I won't guess."


@dataclass
class Citation:
    n: int
    label: str  # e.g. "handbook.txt, chunk 2"
    source: str
    score: float
    text: str


@dataclass
class Answer:
    answer: str
    refused: bool
    reason: str = ""  # below_relevance_threshold | model_declined | no_valid_citations | ""
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)  # everything fetched, pre-threshold
    invalid_citations: list[int] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, cfg: Config = CONFIG, store=None, llm=None):
        self.cfg = cfg
        self._store = store
        self._llm = llm

    @property
    def store(self):
        if self._store is None:
            from .store import get_store

            self._store = get_store(self.cfg)
        return self._store

    def with_threshold(self, min_relevance: float) -> "RAGPipeline":
        return RAGPipeline(replace(self.cfg, min_relevance=min_relevance), self._store, self._llm)

    def ingest_file(self, path: str | Path) -> dict:
        from .store import ingest_chunks

        path = Path(path)
        chunks = chunk_documents(load_file(path), self.cfg)
        n = ingest_chunks(self.store, chunks, source=path.name)
        return {"source": path.name, "chunks": n}

    def ask(self, question: str) -> Answer:
        hits = retrieve(self.store, question, self.cfg.top_k)
        kept = apply_threshold(hits, self.cfg.min_relevance)

        # Gate 1: nothing relevant enough -> refuse WITHOUT calling the LLM.
        if not kept:
            return Answer(REFUSAL_MESSAGE, True, "below_relevance_threshold", retrieved=hits)

        raw = generate(question, kept, self._llm)

        # Gate 2: the model itself says the context doesn't contain the answer.
        if raw.strip().upper().startswith(NOT_FOUND):
            return Answer(REFUSAL_MESSAGE, True, "model_declined", retrieved=hits)

        # Gate 3: an answer with no verifiable citation is treated as ungrounded.
        valid, invalid = parse_citations(raw, len(kept))
        if not valid:
            return Answer(REFUSAL_MESSAGE, True, "no_valid_citations", retrieved=hits, invalid_citations=invalid)

        cites = [
            Citation(n, location_label(kept[n - 1].metadata), kept[n - 1].metadata["source"], kept[n - 1].score, kept[n - 1].text)
            for n in valid
        ]
        return Answer(raw, False, "", cites, hits, invalid)
