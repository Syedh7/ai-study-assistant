# config.py
# Purpose: Central configuration — uses Groq for LLM + lightweight embeddings

import os
import sys
from dotenv import load_dotenv

# ── Load API Key ───────────────────────────────────────────────────────────
GROQ_API_KEY = None

try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    pass

if not GROQ_API_KEY:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY not found! "
        "Add it to your .env file or Streamlit Cloud secrets."
    )

# ── Model Settings ─────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
FAISS_INDEX_PATH = "faiss_index"


def get_embeddings():
    """
    Uses TF-IDF style fake embeddings via a lightweight approach.
    No heavy ML models needed — works instantly on cloud!
    """
    from langchain_community.embeddings import FakeEmbeddings
    return FakeEmbeddings(size=384)


def get_llm(temperature=0.3):
    """Returns Groq LLM — fast and free."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature
    )
