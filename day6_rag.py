import os
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# 1. Load PDF
pdf_path = "test.pdf"
if not os.path.exists(pdf_path):
    print("No test.pdf, creating dummy...")
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, "RAG is Retrieval Augmented Generation.")
    c.drawString(100, 730, "It has 2 steps: 1) Retrieve relevant docs from vector DB")
    c.drawString(100, 710, "2) Generate answer using LLM with retrieved context.")
    c.drawString(100, 690, "This avoids hallucination because it uses your own data, not just LLM memory.")
    c.drawString(100, 670, "ChromaDB stores embeddings for fast similarity search.")
    c.save()
    print("Created test.pdf")

reader = PdfReader(pdf_path)
raw_text = ""
for page in reader.pages:
    if page.extract_text():
        raw_text += page.extract_text() + "\n"

print(f"Loaded PDF: {len(raw_text)} chars")

# 2. Manual chunk (no library needed)
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

text_chunks = chunk_text(raw_text)[:20]  # only first 20 for test
print(f"Using {len(text_chunks)} for quick test (full PDF = 3253 would be slow)")
docs = [Document(page_content=chunk) for chunk in text_chunks]
print(f"Chunked into {len(docs)} chunks")

# 3. Embed + Store in Chroma
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="day6_rag",
    host="localhost",
    port=8000
)
print("Stored in Chroma ✅")

# 4. Query
query = "What is RAG and why use it?"
results = vectorstore.similarity_search(query, k=2)
print(f"\nQuery: {query}\n")
for i, r in enumerate(results):
    print(f"Result {i+1}: {r.page_content[:300]}...\n")

print("Day 6 RAG DONE! 🚀")
