# text_chunker.py
# Purpose: Split large text into smaller overlapping chunks
#
# Why do we chunk?
# LLMs have a "context window" limit — they can't read 100 pages at once.
# We split text into chunks of ~1000 characters so the model only reads
# the RELEVANT parts when answering a question.
#
# Why overlap?
# If a sentence is cut across two chunks, the overlap (200 chars)
# ensures meaning is not lost at the boundary.

from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks for vector storage.

    Args:
        text (str): Full document text
        chunk_size (int): Max characters per chunk (default: 1000)
        chunk_overlap (int): Characters shared between consecutive chunks (default: 200)

    Returns:
        list[str]: List of text chunks
    """

    if not text or not text.strip():
        print("⚠️ Empty text passed to chunker.")
        return []

    # RecursiveCharacterTextSplitter tries to split on:
    # paragraphs → sentences → words → characters (in that order)
    # This keeps semantic meaning intact as much as possible
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order — paragraph breaks first
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    print(f"✅ Created {len(chunks)} chunks from {len(text)} characters")
    return chunks
