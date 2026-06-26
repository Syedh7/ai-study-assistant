# pdf_handler.py
# Purpose: Extract text from PDFs — supports both normal and scanned/image PDFs
#
# Smart fallback system:
# - Local (Windows): Uses OCR with Windows paths
# - Cloud (Linux): Uses OCR with Linux paths (installed via packages.txt)
# - If OCR fails anywhere: Falls back to normal extraction gracefully

import os
import sys
from pypdf import PdfReader
import io

# Detect if running on Streamlit Cloud (Linux) or local Windows
IS_CLOUD = sys.platform != "win32"

# Try to import OCR libraries
try:
    import pytesseract
    from pdf2image import convert_from_bytes

    if IS_CLOUD:
        # Linux (Streamlit Cloud) — tesseract installed via packages.txt
        POPPLER_PATH = None  # Linux finds it automatically
        print("✅ OCR ready (Cloud/Linux mode)")
    else:
        # Windows (Local development)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        POPPLER_PATH = r"C:\Users\hashi\OneDrive\Desktop\poppler\poppler-26.02.0\Library\bin"
        print("✅ OCR ready (Windows/Local mode)")

    OCR_AVAILABLE = True

except ImportError:
    OCR_AVAILABLE = False
    POPPLER_PATH = None
    print("⚠️ OCR not available — normal extraction only")


def extract_text_from_pdf(pdf_file):
    """
    Extracts text from any PDF — digital or scanned.
    Auto-detects PDF type and uses best extraction method.

    Args:
        pdf_file: File object from Streamlit uploader

    Returns:
        tuple: (text: str, ocr_used: bool)
    """

    pdf_bytes = pdf_file.read()

    # ── Method 1: Normal PyPDF Extraction ─────────────────────────────────
    text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_number + 1} ---\n"
                text += page_text + "\n"
    except Exception as e:
        print(f"⚠️ Normal extraction error: {e}")

    # Return if enough text found
    if len(text.strip()) > 100:
        print(f"✅ Normal extraction: {len(text)} chars")
        return text.strip(), False

    # ── Method 2: OCR Extraction ───────────────────────────────────────────
    if OCR_AVAILABLE:
        print("🔍 Switching to OCR mode...")
        try:
            # Convert PDF pages to images
            convert_kwargs = {"dpi": 300}
            if POPPLER_PATH:
                convert_kwargs["poppler_path"] = POPPLER_PATH

            pages = convert_from_bytes(pdf_bytes, **convert_kwargs)

            ocr_text = ""
            for page_number, page_image in enumerate(pages):
                print(f"🔍 OCR page {page_number + 1}/{len(pages)}...")
                page_text = pytesseract.image_to_string(
                    page_image, lang="eng"
                )
                if page_text.strip():
                    ocr_text += f"\n--- Page {page_number + 1} ---\n"
                    ocr_text += page_text + "\n"

            if ocr_text.strip():
                print(f"✅ OCR successful: {len(ocr_text)} chars")
                return ocr_text.strip(), True

        except Exception as e:
            print(f"❌ OCR failed: {e}")

    return text.strip() if text.strip() else "", False
