import os
import json
from pathlib import Path
import streamlit as st
from math import ceil

# Whoosh imports
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser

# FAISS + embeddings
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# OpenAI
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIG ---
DATA_DIR = Path("./data/json_files")  # JSON files folder
INDEX_DIR = "whoosh_index"
EMBED_DIM = 384
EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-3.5-turbo"
BASE_PDF_URL = "https://www.archives.gov/files/research/jfk/releases/2025/0318"

# --- RESOURCE LOADER ---
@st.cache_resource
def load_resources():
    # Load documents
    docs = []
    for fp in DATA_DIR.glob("*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        docs.append(data)

    # Build or open Whoosh index
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            summary=TEXT(stored=True),
            key_findings=TEXT(stored=True),
            keywords=TEXT(stored=True)
        )
        ix = create_in(INDEX_DIR, schema)
        writer = ix.writer()
        for doc in docs:
            writer.add_document(
                doc_id=doc.get("doc_id", ""),
                summary=doc.get("summary", ""),
                key_findings=" | ".join(doc.get("key_findings", [])),
                keywords=" ".join(doc.get("keywords", []))
            )
        writer.commit()
    ix = open_dir(INDEX_DIR)

    # FAISS index + embedder
    texts = [f"{d.get('summary','')} {' | '.join(d.get('key_findings',[]))}" for d in docs]
    embedder = SentenceTransformer(EMBED_MODEL)
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embs)
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    faiss_index.add(embs)

    return docs, ix, embedder, faiss_index

# Cache resources
@st.cache_resource
def get_resources():
    return load_resources()

# Set OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY", "")
if not openai.api_key:
    st.error("🔑 Please set the OPENAI_API_KEY environment variable.")

# --- APP SETUP ---
st.set_page_config(page_title="📚 JFK Archive Search & RAG", layout="wide")
docs, ix, embedder, faiss_idx = get_resources()
id_map = {i: docs[i] for i in range(len(docs))}

mode = st.sidebar.radio(
    "Select view",
    ["🔍 Traditional Search", "🤖 RAG Search", "📑 Browse All PDFs"]
)

# Custom CSS to justify text
# This will apply to all paragraphs in the app
st.markdown(
    """
    <style>
      /* Make every <p> tag justified */
      p {
        text-align: justify;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Traditional Full-Text Search ---
if mode == "🔍 Traditional Search":
    st.header("🔍 Traditional Document Search")
    query = st.sidebar.text_input("Enter keyword(s)")
    limit = st.sidebar.slider("Max results", 1, 100, 10)

    if query:
        parser = MultifieldParser(["summary", "key_findings", "keywords"], schema=ix.schema)
        with ix.searcher() as searcher:
            results = searcher.search(parser.parse(query), limit=limit)
            st.write(f"Found {len(results)} documents.")
            for r in results:
                st.subheader(r['doc_id'])
                st.markdown(f"**Summary:** {r['summary']}")
                st.markdown("**Key Findings:**")
                for kf in r['key_findings'].split(" | "):
                    st.markdown(f"- {kf}")
                url = f"{BASE_PDF_URL}/{r['doc_id']}"
                preview_path = f"./data/preview/{r['doc_id']}.png"
                if os.path.exists(preview_path):
                    st.image(preview_path, width=350)
                st.markdown(f"[📥 Download PDF]({url})")
                st.markdown("---")
    else:
        st.info("Enter a search query in the sidebar to begin.")

# --- RAG Semantic Search + LLM ---
elif mode == "🤖 RAG Search":
    st.header("🤖 RAG Search & Answer Generation")
    query = st.sidebar.text_input("Enter your question")
    top_k = st.sidebar.slider("Number of documents to retrieve", 1, 15, 5)

    if query:
        q_emb = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = faiss_idx.search(q_emb, top_k)

        context = []
        for idx in ids[0]:
            doc = id_map[idx]
            context.append(
                f"Document: {doc.get('doc_id','')}\n"
                f"Summary: {doc.get('summary','')}\n"
                f"Findings: {' | '.join(doc.get('key_findings',[]))}"
            )

        prompt = (
            "You are an analyst researcher. Use the following context to answer the question.\n"
            f"Question: {query}\nContext:\n" + "\n---\n".join(context) + "\nAnswer:"
        )

        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        answer = response.choices[0].message.content.strip()

        st.subheader("Answer:")
        st.write(answer)
        st.markdown("**Sources:**")
        for idx in ids[0]:
            st.markdown(f"- {id_map[idx].get('doc_id','')}")
            url = f"{BASE_PDF_URL}/{id_map[idx].get('doc_id','')}"
            preview_path = f"./data/preview/{id_map[idx].get('doc_id','')}.png"
            if os.path.exists(preview_path):
                    st.image(preview_path, width=500)
            st.markdown(f"[📥 Download PDF]({url})")

    else:
        st.info("Enter a question in the sidebar to begin.")

# --- Browse PDFs View with Pagination ---
else:
    st.header("📑 Browse All PDFs")
    st.write("Below is a list of all available documents. Use the sidebar to navigate pages.")

    # Pagination setup
    docs_per_page = 10
    total_docs = len(docs)
    total_pages = ceil(total_docs / docs_per_page)
    page = st.sidebar.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1
    )

    start_idx = (page - 1) * docs_per_page
    end_idx = start_idx + docs_per_page
    page_docs = docs[start_idx:end_idx]

    st.sidebar.write(f"Showing {start_idx + 1}–{min(end_idx, total_docs)} of {total_docs}")
    st.sidebar.write(f"Page {page} of {total_pages}")

    for doc in page_docs:
        doc_id = doc.get('doc_id', '')
        url = f"{BASE_PDF_URL}/{doc_id}"
        st.subheader(doc_id)
        st.markdown(f"[📥 Download PDF]({url})")
        preview_path = f"./data/preview/{doc_id}.png"
        if os.path.exists(preview_path):
            st.image(preview_path, width=400)
        summary = doc.get('summary', '')
        if summary:
            st.markdown(f"**Summary:** {summary}")
        key_findings = doc.get('key_findings', [])
        if key_findings:
            st.markdown("**Key Findings:**")
            for kf in key_findings:
                st.markdown(f"- {kf}")
        st.markdown("---")
