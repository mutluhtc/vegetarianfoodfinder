import requests
from bs4 import BeautifulSoup
import pdfplumber
import io

def get_pdf_text(pdf_bytes):
    """Extract text from a PDF (matches notebook behavior)."""
    try:
        full_text = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text.append(text)
        return "\n".join(full_text) if full_text else "ERROR: No text extracted from PDF"
    except Exception as e:
        return f"ERROR extracting PDF: {e}"

def get_website_text(url):
    """Fetch website content (HTML or PDF) and return raw text (matches notebook)."""
    try:
        if not url or not isinstance(url, str) or url.strip() == "":
            return "ERROR: No URL"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        print(f"Fetching content from: {url}")
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            print("📄 Detected PDF, extracting text...")
            return get_pdf_text(resp.content)
        else:
            print("🌐 Detected HTML, extracting text...")
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            return text if text else "ERROR: No text extracted from HTML"
    except Exception as e:
        return f"ERROR: {e}"
