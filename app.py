import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Day 10 - RAG v2 Anti-Hallucination", layout="wide")
st.title("📚 Day 10 - RAG v2 (1997 chunks) Anti-Hallucination")
st.caption("k=7 | temp=0 | Strict context-only | No API key")

@st.cache_resource
def load_rag():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(collection_name="day7_full", embedding_function=embeddings, host="localhost", port=8000)
    llm = ChatOllama(model="llama3.2", temperature=0.0)
    prompt = ChatPromptTemplate.from_template("""
You are a book-only assistant. Use ONLY the context below. Do NOT add outside knowledge.
If context does not contain answer, say "Not in provided context".

Context (from book, {k} chunks):
{context}

Rules:
- Answer in 4-6 bullet steps if explaining algorithm
- Include (Page X) citations when present in context
- Never mix BPE with PagedAttention/blocks — they are separate
- Be concise, technical, no fluff

Question: {question}
Answer:
""")
    return vs, llm, prompt

vs, llm, prompt = load_rag()
count = vs._collection.count()
st.sidebar.metric("Chunks", count)
st.sidebar.metric("Progress", "35% → 40% today")
k = st.sidebar.slider("k", 3, 10, 7)
st.sidebar.write("Day 7: 20 chunks → 1 line\nDay 9: 1997 chunks → detailed\nDay 10: fix hallucination")

def rag(q, k=7):
    docs = vs.similarity_search(q, k=k)
    context = "\n\n---\n\n".join([f"[Chunk {i}] {d.page_content}" for i,d in enumerate(docs)])
    resp = (prompt | llm).invoke({"context": context, "question": q, "k": k})
    return resp.content, docs

if "hist" not in st.session_state:
    st.session_state.hist = []

q = st.chat_input("Ask again: Explain BPE / Flash Attention / tokenization types...")

if q:
    with st.spinner(f"Searching {count} + LLM temp=0..."):
        ans, docs = rag(q, k=k)
        st.session_state.hist.append((q, ans, docs))

for q, ans, docs in reversed(st.session_state.hist):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(ans)
        with st.expander(f"Sources {len(docs)}"):
            for d in docs:
                st.code(d.page_content[:700])

st.divider()
st.subheader("Day 10 Eval — Hallucination Check")
if st.button("Run eval 3 Qs"):
    for tq in ["Explain Byte-Pair Encoding steps with vocab init", "Explain Flash Attention tiling memory saving", "Difference between char, word, subword tokenization"]:
        ans, docs = rag(tq, k=k)
        st.markdown(f"**Q: {tq}**\n\n{ans}\n\n*chunks:{len(docs)}*")
        # simple hallucination check
        if "block table" in ans.lower() and "bpe" in tq.lower():
            st.error("⚠️ Still mixing BPE + block table!")
        else:
            st.success("✅ No mix")
        st.divider()
