# 📄 RAG Chatbot for Documents

Chat with your own **PDFs, TXT and CSVs**. Every answer cites the exact chunk of the exact file it came from — and when the documents don't contain the answer, the bot **refuses instead of hallucinating**. Eval metrics prove it.

**Stack:** Python · LangChain · ChromaDB · OpenAI API · Streamlit

> **Resume line:** Built a type-aware chunking pipeline (token-based for prose, row-atomic for CSV) with a relevance-threshold retriever that refuses to answer instead of hallucinating — plus eval metrics to prove it.

## Architecture

```
 upload ─► loaders ─► type-aware chunking ─► OpenAI embeddings ─► ChromaDB (cosine)
           pdf: per page    prose: token-based + overlap
           txt: whole       csv:   1 row = 1 chunk (atomic)

 question ─► embed ─► top-k ─► GATE 1: similarity >= threshold?  no ─► refuse (LLM never called)
                                   │ yes
                                   ▼
                      LLM answers ONLY from numbered excerpts, cites [n]
                                   │
                      GATE 2: model says NOT_FOUND?              ─► refuse
                      GATE 3: no valid [n] citation in answer?   ─► refuse
                                   ▼
                      answer + citations (file, page/row, chunk, similarity, text)
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY
python cli.py ingest data/sample/*
python cli.py ask "How many days of paid annual leave do employees get?"
python cli.py ask "Who won the 2018 World Cup?"      # -> refuses
streamlit run app.py            # UI with upload, citations, threshold slider
pytest                          # offline unit tests (no API key needed)
```

## Design decisions (good interview material)

| Decision | Why |
|---|---|
| **Token-based chunking** for prose | Embedding/LLM limits are in tokens, not characters; recursive splitter keeps paragraphs/sentences intact, with overlap so facts on a boundary survive. |
| **Row-atomic chunks** for CSV | Splitting a row detaches values from column names. Each row is one chunk written as `col: value \| col: value`, so it is self-describing. Tested, even for very long rows. |
| **Cosine space + similarity threshold** | Scores become interpretable (`1 - distance`), so "how relevant is relevant enough" is a number you can calibrate, not a guess. |
| **Three refusal gates** | Retrieval gate saves cost and blocks off-topic questions; model gate handles near-topic questions; citation gate rejects answers that can't be traced to a chunk. |
| **Citations validated in code** | `[n]` numbers are checked against the excerpts actually sent; invented numbers are caught. |
| **Delete-by-source before re-ingest** | Editing a file doesn't leave stale chunks that produce outdated answers. |

## Evaluation

```bash
python -m eval.run_eval --mode retrieval   # embeddings only, cheap: hit@k, MRR, threshold sweep
python -m eval.run_eval --mode full        # end-to-end with the LLM
```

| Metric | Meaning |
|---|---|
| hit@k, MRR | Right file **and** chunk containing the expected fact appears in top-k |
| Correct-refusal rate | Unanswerable questions that were refused (= not hallucinated) |
| False-refusal rate | Answerable questions wrongly refused |
| Answer accuracy | Answer contains the expected fact |
| Citation validity / grounding | Cited numbers exist / cited chunk really contains the fact |

The **threshold sweep** is how `MIN_RELEVANCE` gets chosen: pick the value with high correct-refusal and ~0% false-refusal, then set it in `.env`. Add your own questions to `eval/eval_set.json`, especially *near-topic* unanswerable ones (e.g. "parental leave policy" against a handbook that never mentions it) — those are the hard cases.

**After you run it, replace this with your real numbers** for the resume/README, e.g. *"hit@5 X%, Y% correct refusals at Z% false refusals on N questions."* Don't claim figures you haven't measured.

## Project layout

```
rag/        config · tokens · loaders · chunking · store · retriever · generator · pipeline
app.py      Streamlit UI        cli.py   command-line interface
eval/       eval_set.json + run_eval.py
tests/      offline unit tests  data/sample/  demo handbook + product CSV
```

## Known limitations / next steps
- Scanned (image-only) PDFs need OCR; they are skipped today.
- Semantic-only retrieval; add hybrid BM25 + reranker for exact-ID lookups.
- Single global threshold; per-file-type thresholds (CSV rows vs prose) may do better.
- Add an LLM-as-judge faithfulness metric to `--mode full`.
