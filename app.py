import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Day 9 - Book RAG 1997 chunks", layout="wide")
st.title("📚 Day 9 - Full Book RAG Chat (1997 chunks)")
st.caption("Local RAG: Chroma + nomic-embed-text + llama3.2 | No API key")

@st.cache_resource
def load_rag():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(collection_name="day7_full", embedding_function=embeddings, host="localhost", port=8000)
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    prompt = ChatPromptTemplate.from_template("""
Context from 1.4M book (1997 chunks):
{context}

Question: {question}
Answer in detail using ONLY context. Cite page if possible. If not found, say Not in book.
""")
    return vs, llm, prompt

vs, llm, prompt = load_rag()
count = vs._collection.count()
st.sidebar.metric("Indexed Chunks", count)
st.sidebar.metric("Milestone", "30% → 35% today")
k = st.sidebar.slider("k (retrieved chunks)", 1, 10, 5)
show_sources = st.sidebar.checkbox("Show sources", True)

def rag_answer(q, k=5):
    docs = vs.similarity_search(q, k=k)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    resp = (prompt | llm).invoke({"context": context, "question": q})
    return resp.content, docs

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

q = st.chat_input("Ask your book anything (BPE, Flash Attention, tokenization...)")

if q:
    with st.spinner(f"Searching {count} chunks + generating..."):
        ans, docs = rag_answer(q, k=k)
        st.session_state.history.append((q, ans, docs))

for q, ans, docs in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(ans)
        if show_sources:
            with st.expander(f"Sources ({len(docs)} chunks)"):
                for i, d in enumerate(docs):
                    st.code(d.page_content[:600], language="text")

st.divider()
st.subheader("Day 7 vs Day 8 Evaluation")
st.write("Day 7: 20 chunks → 1 sentence | Day 8: 1997 chunks → full paragraph")
if st.button("Run 3 eval questions"):
    tests = [
        "Explain Byte-Pair Encoding in detail with steps",
        "Explain Flash Attention tiling and why it saves memory",
        "What is difference between character, word, subword tokenization?"
    ]
    for tq in tests:
        ans, docs = rag_answer(tq, k=k)
        st.markdown(f"**Q: {tq}**\n\n{ans}\n\n*Used {len(docs)} chunks*")
        st.divider()

