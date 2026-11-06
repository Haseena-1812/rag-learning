import re
import hashlib
import datetime
import chromadb
from sentence_transformers import SentenceTransformer
from . import config

_model = None
_col = None


def model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def collection():
    global _col
    if _col is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _col = client.get_or_create_collection(
            config.COLLECTION, metadata={"hnsw:space": "cosine"}
        )
    return _col


def embed(texts):
    return model().encode(texts, batch_size=32, normalize_embeddings=True).tolist()


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split on sentences/paragraphs, pack up to `size` chars, carry `overlap` chars."""
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if u.strip()]
    pieces = []
    for u in units:
        while len(u) > size:
            pieces.append(u[:size])
            u = u[size - overlap:]
        pieces.append(u)

    chunks, cur, cur_len = [], [], 0
    for u in pieces:
        if cur_len + len(u) + 1 > size and cur:
            chunks.append(" ".join(cur))
            carry, n = [], 0
            for prev in reversed(cur):
                if n >= overlap:
                    break
                carry.insert(0, prev)
                n += len(prev) + 1
            cur, cur_len = carry, n
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def add_document(source, doc_type, pages, topic="general"):
    """Re-ingesting a source replaces its old chunks (simple change handling)."""
    col = collection()
    col.delete(where={"source": source})

    today = int(datetime.date.today().strftime("%Y%m%d"))
    items = {}  # id -> (text, metadata); dict removes duplicate chunks
    for p in pages:
        for c in chunk_text(p["text"]):
            if len(c) < 50:
                continue
            cid = hashlib.sha1(f"{source}|{p['page']}|{c}".encode()).hexdigest()
            items[cid] = (c, {
                "source": source,
                "doc_type": doc_type,
                "topic": topic,
                "page": p["page"],
                "added": today,
            })

    ids = list(items.keys())
    for i in range(0, len(ids), 64):
        batch = ids[i:i + 64]
        texts = [items[x][0] for x in batch]
        col.upsert(
            ids=batch,
            documents=texts,
            embeddings=embed(texts),
            metadatas=[items[x][1] for x in batch],
        )
    return len(ids)


def build_where(source=None, doc_type=None, topic=None, since=None):
    conds = []
    if source:
        conds.append({"source": source})
    if doc_type:
        conds.append({"doc_type": doc_type})
    if topic:
        conds.append({"topic": topic})
    if since:
        conds.append({"added": {"$gte": int(since)}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def search(query, k=config.TOP_K, where=None):
    res = collection().query(
        query_embeddings=embed([query]), n_results=k, where=where
    )
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({**meta, "text": doc, "score": round(1 - dist, 3)})
    return hits


def list_sources():
    metas = collection().get(include=["metadatas"])["metadatas"]
    out = {}
    for m in metas:
        s = out.setdefault(m["source"], {
            "source": m["source"], "doc_type": m["doc_type"],
            "topic": m["topic"], "added": m["added"], "chunks": 0,
        })
        s["chunks"] += 1
    return list(out.values())


def delete_source(source):
    collection().delete(where={"source": source})