"""
Day18: Reflection & Self-Critique Agent
Improves answer from 0.8 -> 0.95 via self-critique loop
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

def critique(answer, question):
    prompt = f"""You are a strict critic. Score 0-1 and give feedback.
Question: {question}
Answer: {answer}
Return JSON: {{"score": float, "feedback": str, "missing": [str]}}"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

def reflect_loop(question, initial_answer, threshold=0.95, max_iter=3):
    answer = initial_answer
    for i in range(max_iter):
        crit = critique(answer, question)
        print(f"[Iter {i+1}] Critique: {crit}")
        # naive parse score
        try:
            import json
            data = json.loads(crit[crit.find('{'):crit.rfind('}')+1])
            score = float(data.get('score', 0))
        except:
            score = 0.8
        if score >= threshold:
            return answer, score, i+1
        # improve
        improve_prompt = f"Improve answer based on critique:\nCritique: {crit}\nQuestion: {question}\nPrevious: {answer}"
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content": improve_prompt}],
            temperature=0.3
        )
        answer = resp.choices[0].message.content
    return answer, score, max_iter

if __name__ == "__main__":
    q = "Explain Agentic RAG with tool calling"
    init = "RAG retrieves docs and LLM answers."
    final, score, iters = reflect_loop(q, init)
    print(f"FINAL Score {score} after {iters} iters:\n{final}")
