# quiz_generator.py
# Purpose: Generate multiple-choice quiz questions from the document
#
# IMPROVEMENT OVER YOUR VERSION:
# Your prompt was minimal and unstructured. The output would be inconsistent.
# This version uses a structured prompt with clear formatting rules,
# making the output predictable and easy to display in the UI.

from config import get_llm
from vector_store import get_relevant_chunks


def generate_quiz(num_questions=5):
    """
    Generates multiple-choice quiz questions from the uploaded document.

    Args:
        num_questions (int): Number of questions to generate (default: 5)

    Returns:
        str: Formatted quiz with questions, options, and answers
    """

    # Retrieve varied content from across the document for diverse questions
    context = get_relevant_chunks(
        "important concepts definitions examples problems solutions theory",
        k=10
    )

    if not context:
        return "❌ No document found. Please upload a PDF first."

    prompt = f"""You are an expert professor creating a quiz for engineering students.

Create exactly {num_questions} multiple-choice questions based ONLY on the provided content.

STRICT FORMAT — follow this exactly for each question:

Q1. [Question text here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
✅ Correct Answer: [A/B/C/D] - [Brief explanation why]

Q2. [Question text here]
...and so on.

Rules:
- Questions must be based ONLY on the provided content
- Each question must have exactly 4 options
- Include one clearly correct answer per question
- Make wrong options plausible (not obviously wrong)
- Vary difficulty: mix easy, medium, and hard questions
- At the end, add: "--- END OF QUIZ ---"

Document Content:
{context}

Quiz:"""

    try:
        llm = get_llm(temperature=0.5)  # Slightly higher temp for varied questions
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ Error generating quiz: {str(e)}"
