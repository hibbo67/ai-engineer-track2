from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("Day 8: Indexing FULL PDF...")
loader = PyPDFLoader("test.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks (was 20, now full book)")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vs = Chroma(
    collection_name="day7_full",
    embedding_function=embeddings,
    host="localhost",
    port=8000
)

print("Embedding 3253 chunks -> ~15 min...")
vs.add_documents(chunks)
print(f"DONE! Count: {vs._collection.count()}")
