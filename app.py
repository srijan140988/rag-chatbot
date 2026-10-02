"""Streamlit UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from rag.config import CONFIG
from rag.loaders import SUPPORTED
from rag.pipeline import RAGPipeline
from rag.store import delete_source, list_sources

st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")
st.title("📄 Chat with your documents")
st.caption("Every answer cites the exact chunk it came from — or the bot declines to answer.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Documents")
    threshold = st.slider("Relevance threshold", 0.0, 0.9, CONFIG.min_relevance, 0.01,
                          help="Minimum cosine similarity for a chunk to be used. Higher = stricter, more refusals.")
    rag = RAGPipeline().with_threshold(threshold)

    files = st.file_uploader("Upload PDF / TXT / CSV", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True)
    if files and st.button("Ingest files", type="primary"):
        tmp = Path(tempfile.mkdtemp())
        for f in files:
            dest = tmp / f.name  # keep the original filename: it is what citations show
            dest.write_bytes(f.getbuffer())
            with st.spinner(f"Indexing {f.name}..."):
                info = rag.ingest_file(dest)
            st.success(f"{info['source']}: {info['chunks']} chunks")

    st.subheader("Indexed")
    sources = list_sources(rag.store)
    if not sources:
        st.info("Nothing indexed yet.")
    for name, n in sources.items():
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{name}** · {n} chunks")
        if c2.button("🗑", key=f"del-{name}"):
            delete_source(rag.store, name)
            st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for c in msg.get("citations", []):
            with st.expander(f"[{c.n}] {c.label} · similarity {c.score:.2f}"):
                st.text(c.text)

if q := st.chat_input("Ask a question about your documents"):
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            ans = rag.ask(q)
        if ans.refused:
            st.warning(ans.answer)
            top = ans.retrieved[0].score if ans.retrieved else None
            st.caption(f"Refusal reason: `{ans.reason}`" + (f" · best match similarity {top:.2f} (threshold {threshold:.2f})" if top is not None else ""))
        else:
            st.markdown(ans.answer)
            for c in ans.citations:
                with st.expander(f"[{c.n}] {c.label} · similarity {c.score:.2f}"):
                    st.text(c.text)
    st.session_state.history.append({"role": "assistant", "content": ans.answer, "citations": ans.citations})
