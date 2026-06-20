# vector_store.py
# Purpose: Convert text chunks into vectors and store them in FAISS
#
# WHY FAISS?
# After chunking, we have 50-200 text pieces. When a user asks a question,
# we can't send ALL of them to Gemini (too expensive, too slow).
# Instead, FAISS finds the TOP 3-5 chunks most SIMILAR to the question.
# This is called "semantic search" — searching by meaning, not keywords.
#
# HOW IT WORKS:
# Text chunk → Embedding model → [0.23, -0.87, 0.45, ...] (768 numbers)
# FAISS stores all these number-lists and finds the closest ones to your query.

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from config import get_embeddings, FAISS_INDEX_PATH


def create_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks and saves it to disk.

    Args:
        text_chunks (list[str]): List of text chunks from the PDF

    Returns:
        FAISS: The vector store object, or None if creation fails
    """

    if not text_chunks:
        print("❌ No text chunks provided.")
        return None

    try:
        print(f"🔄 Creating embeddings for {len(text_chunks)} chunks...")

        embeddings = get_embeddings()  # Get from config — not recreated each time

        # FAISS.from_texts does two things:
        # 1. Calls the embedding model for each chunk (API calls happen here)
        # 2. Stores all vectors in a FAISS index in memory
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )

        # Save the index to disk so we don't re-embed every time the app reruns
        # This creates two files: faiss_index.faiss and faiss_index.pkl
        vector_store.save_local(FAISS_INDEX_PATH)

        print(f"✅ FAISS index saved to '{FAISS_INDEX_PATH}/'")
        return vector_store

    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        st.error(f"Vector DB Error: {str(e)}")  # Shows error in UI
        return None


def load_vector_store():
    """
    Loads a previously saved FAISS index from disk.

    Returns:
        FAISS: The loaded vector store, or None if not found
    """

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"❌ FAISS index not found at '{FAISS_INDEX_PATH}'. Upload a PDF first.")
        return None

    try:
        embeddings = get_embeddings()

        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
            # allow_dangerous_deserialization=True is needed because FAISS
            # uses Python's pickle format. It's safe when YOU created the file.
        )

        print("✅ FAISS index loaded from disk.")
        return vector_store

    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        return None


def get_relevant_chunks(query, k=5):
    """
    Retrieves the top-k most relevant chunks for a given query.
    This is the core of RAG — the Retrieval step.

    Args:
        query (str): The user's question or search query
        k (int): Number of chunks to retrieve (default: 5)

    Returns:
        str: Combined context text from top-k relevant chunks
    """

    vector_store = load_vector_store()

    if vector_store is None:
        return ""

    # similarity_search finds chunks whose embedding is closest to the query embedding
    docs = vector_store.similarity_search(query, k=k)

    # Join the chunks with clear separators for the LLM
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    return context
