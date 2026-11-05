from rag import *

pages = load_pdf("sample.pdf")
chunks = chunk_pages(pages)
print(f"{len(pages)} pages -> {len(chunks)} chunks")

index = build_index(chunks)
q = input("Question: ")
hits = retrieve(q, index, chunks)
for h in hits:
    print(f"\n--- page {h['page']} (score {h['score']:.2f}) ---\n{h['text'][:200]}")
print("\nANSWER:\n", answer(q, hits))