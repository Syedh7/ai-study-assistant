# config.py
# Purpose: Central configuration — uses Groq for LLM + custom lightweight embeddings

import os
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

LLM_MODEL = "llama-3.3-70b-versatile"
FAISS_INDEX_PATH = "faiss_index"


class SimpleEmbeddings:
    """
    Lightweight embeddings using Python's built-in hashlib.
    No external ML libraries needed — works on any platform instantly.
    Creates consistent 384-dimensional vectors from text.
    """

    def __init__(self):
        self.size = 384

    def _text_to_vector(self, text):
        import hashlib
        import math

        # Create a deterministic vector from text using hashing
        vector = []
        text = text.lower().strip()

        for i in range(self.size):
            # Use different hash seeds for each dimension
            seed = f"{i}:{text}"
            hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
            # Normalize to [-1, 1] range
            val = (hash_val % 10000) / 5000.0 - 1.0
            vector.append(val)

        # Normalize the vector
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector

    def embed_documents(self, texts):
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text):
        return self._text_to_vector(text)


def get_embeddings():
    """Returns lightweight embeddings that work everywhere."""
    return SimpleEmbeddings()


def get_llm(temperature=0.3):
    """Returns Groq LLM — fast and free."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=temperature
    )
