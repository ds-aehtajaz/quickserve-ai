"""Build and save the FAISS index for FAQ retrieval. Run after seeding."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quickserve.nlp.rag import build_and_save_index

build_and_save_index(
    kb_path='data/faq_kb.json',
    index_path='models/faq_index.faiss',
)
