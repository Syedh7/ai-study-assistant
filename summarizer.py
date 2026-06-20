# summarizer.py
# Purpose: Generate a structured summary of the entire uploaded document
#
# IMPROVEMENT OVER YOUR VERSION:
# Your version searched for "Give me an overview" — but that only returns
# 10 similar chunks, potentially missing important parts of the document.
#
# Better approach: retrieve chunks spread across the whole document,
# then ask Gemini to produce a structured academic summary.

from config import get_llm
from vector_store import get_relevant_chunks


def generate_summary():
    """
    Generates a comprehensive summary of the uploaded document.

    Returns:
        str: Structured summary with key topics, concepts, and takeaways
    """

    # Use a broad query to retrieve diverse, representative chunks
    # k=10 gets a wide sample of the document's content
    context = get_relevant_chunks(
        "main topics key concepts important points overview introduction conclusion",
        k=10
    )

    if not context:
        return "❌ No document found. Please upload a PDF first."

    prompt = f"""You are an expert academic summarizer for engineering students.

Based on the provided document content, create a well-structured summary.

Your summary MUST include:
1. **Document Overview** (2-3 sentences about what this document is about)
2. **Key Topics Covered** (bullet list of main topics)
3. **Important Concepts** (explain 3-5 key concepts in simple terms)
4. **Key Takeaways** (what a student should remember after reading this)

Document Content:
{context}

Provide the summary in clear, student-friendly language:"""

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"
