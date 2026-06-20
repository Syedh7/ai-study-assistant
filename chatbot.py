# chatbot.py
# Purpose: Answer user questions using RAG (Retrieval-Augmented Generation)
#
# THE RAG PIPELINE (This is the heart of the project):
# 1. User asks: "What is a transformer in deep learning?"
# 2. We embed the question → get a vector
# 3. FAISS finds the 5 most similar chunks from the PDF
# 4. We send: [context chunks] + [question] → Gemini
# 5. Gemini answers using ONLY the provided context
#
# WHY "ONLY the context"?
# This prevents hallucination. Without context, Gemini might make up an answer.
# With context, it can only use what's in your PDF. This is grounding.

from config import get_llm
from vector_store import get_relevant_chunks


def answer_question(user_question):
    """
    Answers a question using RAG — retrieves relevant PDF chunks, then asks Gemini.

    Args:
        user_question (str): The student's question

    Returns:
        str: AI-generated answer grounded in the PDF content
    """

    if not user_question or not user_question.strip():
        return "Please enter a valid question."

    # Step 1: Retrieve relevant chunks from FAISS
    context = get_relevant_chunks(user_question, k=5)

    if not context:
        return (
            "❌ No document found. Please upload a PDF first, "
            "then ask your question."
        )

    # Step 2: Build a clear, structured prompt
    # A good prompt = better answers. This is called "prompt engineering."
    prompt = f"""You are a helpful AI study assistant for engineering students.
Your job is to answer questions clearly and accurately based ONLY on the provided context.

Rules:
- Answer ONLY from the given context
- If the answer is not in the context, say "This topic is not covered in the uploaded document."
- Use simple language suitable for students
- If relevant, use bullet points or numbered steps
- Keep the answer focused and concise

Context from the document:
{context}

Student's Question:
{user_question}

Answer:"""

    # Step 3: Send to Gemini and get the answer
    try:
        llm = get_llm(temperature=0.2)  # Low temperature = more factual answers
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"
