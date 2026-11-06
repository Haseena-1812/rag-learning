import io
import re
import requests
from pypdf import PdfReader
from bs4 import BeautifulSoup


def load_pdf(data: bytes):
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def load_text(data: bytes):
    return [{"page": 0, "text": data.decode("utf-8", errors="ignore")}]


def load_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [{"page": 0, "text": text}]


def load_url(url: str):
    r = requests.get(url, timeout=20, headers={"User-Agent": "rag-learning/0.1"})
    r.raise_for_status()
    return load_html(r.text)


def load_file(filename: str, data: bytes):
    """Returns (doc_type, pages)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return "pdf", load_pdf(data)
    if ext in ("md", "markdown", "txt"):
        return ("md" if ext != "txt" else "txt"), load_text(data)
    if ext in ("html", "htm"):
        return "html", load_html(data.decode("utf-8", errors="ignore"))
    raise ValueError(f"Unsupported file type: .{ext}")