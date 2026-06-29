# vector_store.py
import os
import numpy as np
import pickle
import faiss
from config import get_embeddings, FAISS_INDEX_PATH


def create_vector_store(text_chunks):
    """Creates FAISS vector store directly without langchain wrapper."""
    if not text_chunks:
        return None
    try:
        embeddings = get_embeddings()

        # Generate embeddings for all chunks
        vectors = embeddings.embed_documents(text_chunks)
        vectors_np = np.array(vectors, dtype=np.float32)

        # Create FAISS index
        dimension = vectors_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors_np)

        # Save index and chunks together
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        faiss.write_index(index, f"{FAISS_INDEX_PATH}/index.faiss")
        with open(f"{FAISS_INDEX_PATH}/chunks.pkl", "wb") as f:
            pickle.dump(text_chunks, f)

        print(f"✅ FAISS index saved with {len(text_chunks)} chunks")
        return True

    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None


def get_relevant_chunks(query, k=5):
    """Gets top-k relevant chunks for a query."""
    try:
        index_path = f"{FAISS_INDEX_PATH}/index.faiss"
        chunks_path = f"{FAISS_INDEX_PATH}/chunks.pkl"

        if not os.path.exists(index_path):
            return ""

        # Load index and chunks
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            text_chunks = pickle.load(f)

        # Embed the query
        embeddings = get_embeddings()
        query_vector = np.array(
            [embeddings.embed_query(query)],
            dtype=np.float32
        )

        # Search
        k = min(k, len(text_chunks))
        distances, indices = index.search(query_vector, k)

        # Return relevant chunks
        relevant = [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
        return "\n\n---\n\n".join(relevant)

    except Exception as e:
        print(f"❌ Error searching: {e}")
        return ""


def load_vector_store():
    """Check if vector store exists."""
    return os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss")
