"""File loaders. Each returns RawDocs that keep the location info needed for citations."""
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".csv"}


@dataclass
class RawDoc:
    text: str
    metadata: dict = field(default_factory=dict)


def load_file(path: str | Path) -> list[RawDoc]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext} (supported: {sorted(SUPPORTED)})")
    return {".pdf": _load_pdf, ".txt": _load_txt, ".csv": _load_csv}[ext](path)


def _load_pdf(path: Path) -> list[RawDoc]:
    from pypdf import PdfReader

    docs = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # skips blank / image-only pages
            docs.append(RawDoc(text, {"source": path.name, "type": "pdf", "page": i}))
    return docs


def _load_txt(path: Path) -> list[RawDoc]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [RawDoc(text, {"source": path.name, "type": "txt"})] if text else []


def _load_csv(path: Path) -> list[RawDoc]:
    import pandas as pd

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="latin-1")
    docs = []
    for i, row in df.iterrows():
        # "column: value" pairs keep each row self-describing for the embedder.
        text = " | ".join(f"{c}: {v.strip()}" for c, v in row.items() if str(v).strip())
        if text:
            # row = spreadsheet row number (header is row 1)
            docs.append(RawDoc(text, {"source": path.name, "type": "csv", "row": int(i) + 2}))
    return docs
