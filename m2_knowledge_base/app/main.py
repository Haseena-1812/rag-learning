import json
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from . import store, rag, loaders, config


@asynccontextmanager
async def lifespan(app):
    store.model()        # load the embedding model at startup, not on first request
    store.collection()
    yield


app = FastAPI(title="RAG Knowledge Base", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    source: str | None = None
    doc_type: str | None = None   # pdf | md | txt | html
    topic: str | None = None
    since: int | None = None      # YYYYMMDD, only docs added on/after
    k: int = config.TOP_K


class UrlRequest(BaseModel):
    url: str
    topic: str = "general"


@app.get("/health")
def health():
    return {"status": "ok", "chunks": store.collection().count()}


@app.post("/ingest/file")
def ingest_file(file: UploadFile = File(...), topic: str = Form("general")):
    try:
        doc_type, pages = loaders.load_file(file.filename, file.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    n = store.add_document(file.filename, doc_type, pages, topic)
    return {"source": file.filename, "chunks": n}


@app.post("/ingest/url")
def ingest_url(req: UrlRequest):
    try:
        pages = loaders.load_url(req.url)
    except Exception as e:
        raise HTTPException(400, f"Could not fetch URL: {e}")
    n = store.add_document(req.url, "html", pages, req.topic)
    return {"source": req.url, "chunks": n}


@app.get("/sources")
def sources():
    return store.list_sources()


@app.delete("/sources")
def delete_source(name: str):
    store.delete_source(name)
    return {"deleted": name}


@app.post("/chat")
def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    where = store.build_where(req.source, req.doc_type, req.topic, req.since)

    def gen():
        for event in rag.chat_stream(sid, req.question, where, req.k):
            yield json.dumps(event) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")