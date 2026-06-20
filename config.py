# config.py
# Purpose: Central configuration file
# Updated to use Groq API instead of Google Gemini
# Groq provides free access to Llama 3.3, Gemma 2, and Mixtral models

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ── Load API Keys ──────────────────────────────────────────────────────────
GROQ_API_KEY = None
GOOGLE_API_KEY = None

# Try Streamlit secrets first (Streamlit Cloud deployment)
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", None)
except Exception:
    pass

# Fall back to .env file (local development)
if not GROQ_API_KEY:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "❌ GROQ_API_KEY not found! "
        "Add it to your .env file as: GROQ_API_KEY=gsk_your_key_here"
    )

# ── Embedding Model ────────────────────────────────────────────────────────
# Still using Google embeddings — these work with the AQ. key too
# BUT if Google embeddings fail, we use a free local alternative
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ── LLM Model ─────────────────────────────────────────────────────────────
# Groq free models — llama-3.3-70b is very powerful and completely free
LLM_MODEL = "llama-3.3-70b-versatile"

# ── FAISS Index Path ───────────────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index"


def get_embeddings():
    """
    Returns embeddings — uses HuggingFace local embeddings (completely free,
    no API key needed, runs on your machine)
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def get_llm(temperature=0.3):
    """
    Returns Groq LLM — fast, free, powerful.
    Uses Llama 3.3 70B model.
    """
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature
    )
