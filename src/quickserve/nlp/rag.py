import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = None
_index = None
_documents: list[dict] = []

EMBED_MODEL = "all-MiniLM-L6-v2"


def _load(index_path: str, kb_path: str):
    global _model, _index, _documents
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        if os.path.exists(index_path):
            _index = faiss.read_index(index_path)
            with open(kb_path, "r", encoding="utf-8") as f:
                _documents = json.load(f)
        else:
            # Build index on the fly from the KB JSON if FAISS file not found
            with open(kb_path, "r", encoding="utf-8") as f:
                _documents = json.load(f)
            texts = [d["question"] + " " + d["answer"] for d in _documents]
            embeddings = _model.encode(texts, convert_to_numpy=True).astype("float32")
            _index = faiss.IndexFlatL2(embeddings.shape[1])
            _index.add(embeddings)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Return the top_k most relevant FAQ entries for a query."""
    index_path = os.getenv("FAQ_INDEX_PATH", "models/faq_index.faiss")
    kb_path = os.getenv("FAQ_KB_PATH", "data/faq_kb.json")
    _load(index_path, kb_path)

    q_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(_documents):
            results.append({**_documents[idx], "score": float(dist)})
    return results


def build_and_save_index(kb_path: str, index_path: str):
    """One-time call to build and persist the FAISS index."""
    global _model, _index, _documents
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    with open(kb_path, "r", encoding="utf-8") as f:
        _documents = json.load(f)
    texts = [d["question"] + " " + d["answer"] for d in _documents]
    embeddings = _model.encode(texts, convert_to_numpy=True).astype("float32")
    _index = faiss.IndexFlatL2(embeddings.shape[1])
    _index.add(embeddings)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(_index, index_path)
    print(f"Index saved to {index_path} ({len(_documents)} documents)")
