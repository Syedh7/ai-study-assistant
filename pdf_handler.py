# pdf_handler.py
# Purpose: Extract text from PDFs — supports both normal and scanned/image PDFs
#
# HOW IT WORKS:
# 1. First tries normal text extraction (PyPDF) — fast, works for digital PDFs
# 2. If text is too short or empty → automatically switches to OCR
# 3. OCR converts each page to image → Tesseract reads the text
# 4. User never needs to worry about PDF type — it just works!

import pytesseract
from pdf2image import convert_from_bytes
from pypdf import PdfReader
import io

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path to poppler bin folder (required by pdf2image)
POPPLER_PATH = r"C:\Users\hashi\OneDrive\Desktop\poppler\poppler-26.02.0\Library\bin"


def extract_text_from_pdf(pdf_file):
    """
    Extracts text from any PDF — digital or scanned.
    Automatically uses OCR if normal extraction fails.

    Args:
        pdf_file: File object from Streamlit uploader

    Returns:
        tuple: (text: str, ocr_used: bool)
    """

    # Read file bytes — needed for both methods
    pdf_bytes = pdf_file.read()

    # ── Method 1: Normal Text Extraction (PyPDF) ───────────────────────────
    text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_number + 1} ---\n"
                text += page_text + "\n"

    except Exception as e:
        print(f"⚠️ Normal extraction failed: {e}")

    # If we got enough text — return directly (no OCR needed)
    if len(text.strip()) > 100:
        print(f"✅ Normal extraction: {len(text)} characters")
        return text.strip(), False

    # ── Method 2: OCR Extraction (Tesseract) ──────────────────────────────
    print("⚠️ Switching to OCR mode...")

    try:
        pages = convert_from_bytes(
            pdf_bytes,
            dpi=300,
            poppler_path=POPPLER_PATH
        )

        ocr_text = ""
        for page_number, page_image in enumerate(pages):
            print(f"🔍 OCR page {page_number + 1}/{len(pages)}...")

            page_text = pytesseract.image_to_string(
                page_image,
                lang="eng"
            )

            if page_text.strip():
                ocr_text += f"\n--- Page {page_number + 1} ---\n"
                ocr_text += page_text + "\n"

        if ocr_text.strip():
            print(f"✅ OCR successful: {len(ocr_text)} characters")
            return ocr_text.strip(), True
        else:
            return "", False

    except Exception as e:
        print(f"❌ OCR failed: {e}")
        return "", False
