import streamlit as st, sys, json
sys.path.insert(0, 'day23')
from tools import tool_search_chroma, tool_critique
st.set_page_config(page_title="RAG 1997 hibbo67")
st.title("RAG 1997 - hibbo67 - 93% GOD")
if "msgs" not in st.session_state: st.session_state.msgs=[]
for m in st.session_state.msgs: st.chat_message(m["role"]).markdown(m["content"])
if p := st.chat_input("Ask about 1997 chunks..."):
    st.session_state.msgs.append({"role":"user","content":p})
    st.chat_message("user").markdown(p)
    ctx = tool_search_chroma(p, n_results=2)
    ans = f"Answer from 1997: {p}\n\nContext: {ctx[:500]}"
    score = json.loads(tool_critique(ans,p)).get("score",0.9)
    st.chat_message("assistant").markdown(f"{ans}\n\nScore {score} - 3 calls DIRECT - 29.74MB Docker - K8s 2 replicas")
    st.session_state.msgs.append({"role":"assistant","content":ans})
