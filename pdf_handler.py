# pdf_handler.py
# Purpose: Extract text from uploaded PDF files
# Why: Before we can do anything with AI, we need raw text from the PDF

from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    """
    Extracts all text from a PDF file.

    Args:
        pdf_file: File object (from Streamlit uploader or file path)

    Returns:
        str: Extracted text, or empty string if extraction fails
    """

    text = ""

    try:
        reader = PdfReader(pdf_file)

        # Check if PDF has pages
        if len(reader.pages) == 0:
            print("⚠️ PDF has no pages.")
            return ""

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()

            if page_text:
                # Add page separator so we know where each page starts
                text += f"\n--- Page {page_number + 1} ---\n"
                text += page_text + "\n"

        # If text is still empty, PDF might be scanned (image-based)
        if not text.strip():
            print("⚠️ No text found. PDF might be a scanned image.")
            return ""

        return text.strip()

    except Exception as e:
        # Catch any unexpected errors (corrupted PDF, password-protected, etc.)
        print(f"❌ Error reading PDF: {e}")
        return ""
