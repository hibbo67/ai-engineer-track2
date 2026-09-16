from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Load existing vectorstore from Day 6
print("Loading Chroma Day 6...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    collection_name="day6_rag",
    embedding_function=embeddings,
    host="localhost",
    port=8000
)

# 2. Check count
count = vectorstore._collection.count()
print(f"Chroma has {count} chunks (Day 6 = 20, full = 3253)")
print(f"Full PDF is 3253 - we will keep 20 for fast Day 7, index full later")

# 3. Setup LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# 4. RAG Prompt - anti-hallucination
prompt = ChatPromptTemplate.from_template("""
You are a RAG assistant. Use ONLY the context below to answer.
If not in context, say "Not found in PDF".

Context:
{context}

Question: {question}
Answer concisely:
""")

# 5. Full RAG Chain
def rag_answer(question):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, docs

# Test it
questions = [
    "What is Byte-Pair Encoding?",
    "Why not use characters or words for tokenization?",
    "What is Flash Attention?"
]

for q in questions:
    print(f"\n{'='*60}\nQ: {q}")
    answer, sources = rag_answer(q)
    print(f"A: {answer}")
    print(f"Sources used: {len(sources)} chunks")

print("\nDay 7 RAG + LLM DONE! 🚀")
