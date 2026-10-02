# 📄 RAG Chatbot 

A document-grounded **Retrieval-Augmented Generation (RAG) chatbot**. Chat with your own **PDF, TXT, and CSV** files: every answer cites the exact chunk of the exact file it came from, and when the documents don't contain the answer, the bot **refuses instead of hallucinating**. An evaluation harness measures how well it does both.

---

## 🚀 Features

- 📄 Supports **PDF, TXT, and CSV** documents
- 🧩 **Type-aware chunking**: token-based for prose, row-atomic for CSV
- 🔎 Semantic retrieval with cosine similarity
- 🛑 **Relevance-threshold retriever** that refuses instead of guessing
- 📌 **Source citations** (file, page/row, chunk, similarity score) validated against the retrieved chunks
- 🧠 Gemini-powered embeddings and answer generation
- 🗄️ ChromaDB vector database
- 📊 **Eval harness**: hit@k, MRR, refusal rates, citation validity, threshold sweep
- 💻 Command-line interface and Streamlit UI
- ✅ Offline unit tests (no API key needed)

---

## 🛠️ Tech Stack

**Python** · **LangChain** · **Google Gemini API** · **ChromaDB** · **Streamlit**

| Component | Choice |
|---|---|
| Embedding model | `gemini-embedding-001` |
| Chat model | `gemini-3.8-flash` |
| Vector store | ChromaDB (cosine space) |

---

## 🏗️ Architecture

```text
                      DOCUMENT INGESTION
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
           PDF               TXT               CSV
        (per page)       (whole file)      (per row)
            └─────────────────┼─────────────────┘
                              ▼
                     Type-Aware Chunking
                  ┌───────────┴───────────┐
                  ▼                       ▼
             PDF / TXT                   CSV
          Token-based chunks        Row-atomic chunks
          (with overlap)            (1 row = 1 chunk)
                  └───────────┬───────────┘
                              ▼
                      Gemini Embeddings
                              ▼
                      ChromaDB (cosine)


                        USER QUESTION
                              ▼
                      Question Embedding
                              ▼
                        Top-K Retrieval
                              ▼
              GATE 1: similarity >= MIN_RELEVANCE ?
                  ┌───────────┴───────────┐
                  ▼ no                    ▼ yes
               REFUSE            Gemini answers ONLY from
        (LLM is never called)    numbered excerpts, cites [n]
                                          ▼
                          GATE 2: model replied NOT_FOUND? ─► REFUSE
                          GATE 3: no valid [n] citation?   ─► REFUSE
                                          ▼
                              Answer + Validated Citations
```

---

## 🧠 How the Pipeline Works

### 1. Document ingestion
Each file type has its own loader. PDFs are read page by page (blank/image-only pages are skipped), TXT files are read whole, and CSVs are read row by row.

### 2. Type-aware chunking

**PDF / TXT**: token-based recursive splitting with overlap, so chunk size matches what the embedding model actually sees:

```env
CHUNK_TOKENS=400
CHUNK_OVERLAP=60
```

**CSV**: each row is one atomic chunk, written as self-describing `column: value` pairs, so a value is never separated from its column name:

```text
sku: SKU-101 | name: AeroDesk Standing Desk | category: Furniture | price_inr: 24999 | stock: 12
```

### 3. Embeddings and vector storage
Chunks are embedded with `gemini-embedding-001` and stored in ChromaDB using **cosine** space, so similarity is simply `1 - distance` and a threshold is an interpretable number.

### 4. Retrieval
The question is embedded and the top-K most similar chunks are fetched:

```env
TOP_K=5
```

### 5. Relevance threshold (refusal gate 1)
If no retrieved chunk reaches the threshold, the system refuses **without calling the LLM**. This saves cost and stops unrelated context from reaching the model.

```env
MIN_RELEVANCE=0.30
```

### 6. Answer generation (refusal gate 2)
Chunks that pass are given to Gemini as numbered excerpts:

```text
[1] (handbook.txt, chunk 0)
Full-time employees receive 24 days of paid annual leave...

[2] (products.csv, row 2)
sku: SKU-101 | name: AeroDesk Standing Desk...
```

Gemini is instructed to use only these excerpts, cite every claim as `[n]`, and reply `NOT_FOUND` if the excerpts don't contain the answer. That handles near-topic questions that pass the similarity threshold but still can't be answered.

### 7. Citation validation (refusal gate 3)
Citation numbers (`[1]`, `[2]`, `[1][3]`) are checked against the excerpts actually sent to the model. Invented numbers are flagged, and an answer with **no** valid citation is treated as ungrounded and refused.

---

## 💡 Example

**Question:** `What is the price of the AeroDesk Standing Desk?`

**Answer:**

```text
The price of the AeroDesk Standing Desk is 24,999 INR [1].

[1] products.csv, row 2 (sim=0.7135)
```

**Unsupported question:** `What is the weather in Delhi today?`

```text
I couldn't find this in your documents, so I won't guess.
```

---

## ⚙️ Installation

```bash
git clone https://github.com/srijan140988/rag-chatbot.git
cd rag-chatbot
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key

EMBEDDING_MODEL=gemini-embedding-001
CHAT_MODEL=gemini-3.8-flash

CHUNK_TOKENS=400
CHUNK_OVERLAP=60
TOP_K=5
MIN_RELEVANCE=0.30
```

> ⚠️ Never commit your `.env` file or API key. Also keep any personal documents (e.g. transcripts or ID scans) out of the repo: don't put them in `data/`, or add their paths to `.gitignore`.

---

## ▶️ Usage

### Ingest documents

```bash
# Sample files
python cli.py ingest data/sample/handbook.txt data/sample/products.csv

# Your own PDF
python cli.py ingest path/to/your-document.pdf

# Everything in a folder
python cli.py ingest data/sample/*
```

### Ask questions (CLI)

```bash
python cli.py ask "How many days of paid annual leave do employees get?"
```

```text
Full-time employees receive 24 days of paid annual leave per calendar year [1].

[1] handbook.txt, chunk 0 (sim=0.6876)
```

### Web UI

```bash
streamlit run app.py
```

Upload files, chat, expand each citation to see the source chunk, and tune the relevance threshold with a slider.

### Run the tests (no API key needed)

```bash
pytest
```

---

## 📊 Evaluation

Claiming the bot "doesn't hallucinate" means little without measurements, so the project includes an evaluation harness over `eval/eval_set.json`, which mixes **answerable** and **unanswerable** questions. The unanswerable set includes near-topic questions, such as asking for a parental-leave policy from a handbook that never mentions it. Those are the hard cases.

```bash
python -m eval.run_eval --mode retrieval   # embeddings only (cheap): hit@k, MRR, threshold sweep
python -m eval.run_eval --mode full        # end-to-end with the LLM
```

| Metric | What it measures |
|---|---|
| **hit@k / MRR** | The right file *and* a chunk containing the expected fact appears in the top-K |
| **Correct-refusal rate** | Unanswerable questions that were refused (higher = fewer hallucinations) |
| **False-refusal rate** | Answerable questions wrongly refused (lower = still useful) |
| **Answer accuracy** | The answer contains the expected fact |
| **Citation validity** | Every cited `[n]` refers to a chunk that was actually retrieved |
| **Citation grounding** | The cited chunk really contains the expected fact |

The **threshold sweep** is how `MIN_RELEVANCE` is chosen: pick the value with a high correct-refusal rate and a near-zero false-refusal rate, then set it in `.env`.

### Results

> Fill in after running `python -m eval.run_eval` on your setup.

| Metric | Result |
|---|---|
| hit@5 | _TBD_ |
| MRR | _TBD_ |
| Correct-refusal rate (at `MIN_RELEVANCE` = _TBD_) | _TBD_ |
| False-refusal rate | _TBD_ |
| Citation validity | _TBD_ |
| Eval set size | _TBD_ questions |

---

## 🎯 Key Engineering Decisions

| Decision | Why |
|---|---|
| **Type-aware chunking** | Prose and tabular data break in different ways, so each gets its own strategy. |
| **Token-based prose chunks** | Model limits are measured in tokens, and overlap keeps facts that straddle a boundary retrievable. |
| **Row-atomic CSV chunks** | Splitting a row detaches values from column names and makes chunks unanswerable. |
| **Cosine similarity + threshold** | Makes "relevant enough" a number you can calibrate with the eval sweep, not a guess. |
| **Three refusal gates** | Retrieval gate (off-topic), model gate (near-topic), citation gate (ungrounded answers). |
| **Citation validation in code** | Citation numbers are verified against the retrieved excerpts rather than trusted blindly. |
| **Source-based re-ingestion** | A file's old chunks are deleted before re-ingesting, so edited documents never leave stale answers behind. |

---

## 📁 Project Structure

```text
rag-chatbot/
├── rag/
│   ├── config.py
│   ├── tokens.py
│   ├── loaders.py
│   ├── chunking.py
│   ├── store.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
├── data/
│   └── sample/
│       ├── handbook.txt
│       └── products.csv
├── eval/
│   ├── eval_set.json
│   └── run_eval.py
├── tests/
│   └── test_core.py
├── app.py
├── cli.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚠️ Limitations and Next Steps

- Scanned (image-only) PDFs need OCR and are currently skipped.
- Retrieval is semantic only; hybrid BM25 + a reranker would help exact-ID lookups.
- A single global threshold is used; per-type thresholds (CSV rows vs prose) may perform better.
- Add an LLM-as-judge faithfulness metric to the full eval.

---

## 👨‍💻 Author

**Srijan Verma**
B.Tech Computer Science and Engineering, VIT Bhopal
GitHub: [github.com/srijan140988](https://github.com/srijan140988)

---

## ⭐ Summary

A document-grounded RAG chatbot built with Python, LangChain, Gemini, and ChromaDB. It supports PDF, TXT, and CSV with type-aware chunking (token-based for prose, row-atomic for CSV), a relevance-threshold retriever that refuses instead of hallucinating, validated source citations, and an evaluation harness that measures retrieval quality and refusal behavior.
