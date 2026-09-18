import warnings
warnings.filterwarnings("ignore")
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

print("Loading FULL 1997...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="day7_full",
    embedding_function=embeddings,
    host="localhost",
    port=8000
)

count = vectorstore._collection.count()
print(f"FULL count: {count}")

llm = ChatOllama(model="llama3.2", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
Context from book (1.4M chars):
{context}

Question: {question}
Answer in detail using ONLY context. If not found, say Not in book.
""")

def rag(q, k=5):
    docs = vectorstore.similarity_search(q, k=k)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    resp = (prompt | llm).invoke({"context": context, "question": q})
    return resp.content, docs

tests = [
    "Explain Byte-Pair Encoding in detail with steps",
    "Explain Flash Attention tiling and why it saves memory",
    "What is the difference between character, word, and subword tokenization?"
]

for q in tests:
    print(f"\n{'='*70}\nQ: {q}\n")
    ans, src = rag(q, k=5)
    print(ans)
    print(f"\n[Used {len(src)} chunks from full {count}]")

print("\nDay 8 FULL RAG DONE")
