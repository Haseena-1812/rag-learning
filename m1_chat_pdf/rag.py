import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")

_model = None

def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

# 1. LOAD: extract text page by page so we can cite pages
def load_pdf(file):
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages

# 2. CHUNK: fixed-size pieces with overlap
def chunk_pages(pages, size=800, overlap=150):
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": p["page"], "text": piece})
            start += size - overlap
    return chunks

# 3. EMBED + STORE: turn text into vectors and index them
def build_index(chunks):
    vecs = get_model().encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
    )
    vecs = np.asarray(vecs, dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index

# 4. RETRIEVE: find the k closest chunks
def retrieve(query, index, chunks, k=4):
    q = get_model().encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, k)
    return [
        {**chunks[i], "score": float(s)}
        for s, i in zip(scores[0], ids[0]) if i != -1
    ]

# 5. AUGMENT + GENERATE
SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "If the context does not contain the answer, say \"I don't know based on this document.\" "
    "Cite page numbers like (p. 3) after each claim."
)

def answer(query, hits):
    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
    client = Groq()  # reads GROQ_API_KEY from the environment
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content