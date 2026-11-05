import io
import streamlit as st
from rag import load_pdf, chunk_pages, build_index, retrieve, answer

st.title("Chat with your PDF")

@st.cache_resource
def prepare(name, data):
    pages = load_pdf(io.BytesIO(data))
    chunks = chunk_pages(pages)
    return chunks, build_index(chunks)

up = st.file_uploader("Upload a PDF", type="pdf")
if up:
    chunks, index = prepare(up.name, up.getvalue())
    st.caption(f"{len(chunks)} chunks indexed")
    q = st.text_input("Ask a question about the document")
    if q:
        hits = retrieve(q, index, chunks)
        st.subheader("Answer")
        st.write(answer(q, hits))
        st.subheader("Retrieved chunks")
        for h in hits:
            with st.expander(f"Page {h['page']} (score {h['score']:.2f})"):
                st.write(h["text"])