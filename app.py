import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Day 11 - Agentic RAG fixed", layout="wide")
st.title("🤖 Day 11 - Agentic RAG v3.1 (Rerank + Memory)")
st.caption("Rewrite = paraphrase only → Retrieve 12 → CrossEncoder rerank → Top 5")

@st.cache_resource
def load_rag():
    emb = OllamaEmbeddings(model="nomic-embed-text")
    vs = Chroma(collection_name="day7_full", embedding_function=emb, host="localhost", port=8000)
    llm = ChatOllama(model="llama3.2", temperature=0.0)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return vs, llm, reranker

vs, llm, reranker = load_rag()
count = vs._collection.count()
st.sidebar.metric("Chunks", count)
st.sidebar.metric("Progress", "40% → 45% today")

if "hist" not in st.session_state:
    st.session_state.hist = []
if "chat_mem" not in st.session_state:
    st.session_state.chat_mem = []

def agentic_rag(question, k_retrieve=12, k_final=5):
    # FIXED rewrite: only paraphrase, no explanation
    rewrite_prompt = ChatPromptTemplate.from_template("You are search query rewriter. Paraphrase the user question into 2 short search queries. No explanation, no steps. Only queries.\nQuestion: {q}\nQueries:")
    rewrites_raw = (rewrite_prompt | llm).invoke({"q": question}).content
    rewrites = [r.strip("- 1234567890. ") for r in rewrites_raw.split("\n") if len(r.strip())>5][:2]

    all_docs = []
    for rq in [question] + rewrites:
        if rq.strip():
            all_docs.extend(vs.similarity_search(rq.strip(), k=4))

    uniq = list({d.page_content: d for d in all_docs}.values())[:k_retrieve]
    pairs = [[question, d.page_content] for d in uniq]
    scores = reranker.predict(pairs)
    scored = sorted(zip(uniq, scores), key=lambda x: x[1], reverse=True)
    top_docs = [d for d,s in scored[:k_final]]

    mem = "\n".join([f"{r}: {c[:200]}" for r,c in st.session_state.chat_mem[-4:]])
    prompt = ChatPromptTemplate.from_template("""
Use ONLY context. History: {history}
Context: {context}
Question: {question}
Answer with 4-6 bullets, Page citations, no mix of concepts.
""")
    context = "\n---\n".join([d.page_content for d in top_docs])
    ans = (prompt | llm).invoke({"context": context, "question": question, "history": mem}).content
    return ans, top_docs, rewrites

q = st.chat_input("Ask: Explain Byte-Pair Encoding steps")
if q:
    with st.spinner(f"Agentic searching {count}..."):
        ans, docs, rewrites = agentic_rag(q)
        st.session_state.hist.append((q, ans, docs, rewrites))
        st.session_state.chat_mem.append(("user", q))
        st.session_state.chat_mem.append(("assistant", ans))

for q, ans, docs, rewrites in reversed(st.session_state.hist):
    with st.chat_message("user"):
        st.write(q)
        st.caption(f"Search queries: {rewrites}")
    with st.chat_message("assistant"):
        st.write(ans)
        with st.expander(f"Reranked Sources {len(docs)}"):
            for d in docs:
                st.code(d.page_content[:700])

if st.button("Clear"):
    st.session_state.hist=[]; st.session_state.chat_mem=[]; st.rerun()
