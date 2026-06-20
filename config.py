# config.py
# Purpose: Central configuration file — load API key ONCE, use everywhere
#
# WHY THIS FILE EXISTS (Important lesson!):
# Your original code had this repeated in chatbot.py, summarizer.py,
# quiz_generator.py, and vector_store.py:
#
#   config = dotenv_values(".env")
#   GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
#   embeddings = GoogleGenerativeAIEmbeddings(...)
#
# This violates the DRY principle: Don't Repeat Yourself.
# If you ever change the embedding model, you'd have to edit 4 files.
# With this config.py, you change it in ONE place.

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ── Load API Key ───────────────────────────────────────────────────────────
# Strategy: Try Streamlit secrets first (works on Streamlit Cloud)
# If not found, fall back to .env file (works locally)

GOOGLE_API_KEY = None

# First try: Streamlit secrets (used when deployed on Streamlit Cloud)
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
except Exception:
    pass  # Not running in Streamlit context, try .env next

# Second try: .env file (used during local development)
if not GOOGLE_API_KEY:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# If still not found, raise a clear error
if not GOOGLE_API_KEY:
    raise ValueError(
        "❌ GOOGLE_API_KEY not found! "
        "For local use: add it to your .env file. "
        "For deployment: add it to Streamlit Cloud secrets."
    )

# ── Embedding Model ────────────────────────────────────────────────────────
# Used to convert text into numerical vectors (numbers that represent meaning)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── LLM Model ─────────────────────────────────────────────────────────────
# Used for generating answers, summaries, and quizzes
# gemini-2.5-flash is fast, free-tier friendly, and very capable
LLM_MODEL = "gemini-2.5-flash"

# ── FAISS Index Path ───────────────────────────────────────────────────────
# Where the vector database is saved on disk
FAISS_INDEX_PATH = "faiss_index"


def get_embeddings():
    """Returns a configured embeddings object. Call this instead of re-creating it."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def get_llm(temperature=0.3):
    """
    Returns a configured LLM object.

    Args:
        temperature (float): 0 = deterministic/factual, 1 = creative
                             Use 0.3 for factual answers, 0.7 for creative quizzes
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature
    )