import re
def validate_input(q):
    if not q or len(q.strip())<5: return False, "too short"
    if len(q)>2000: return False, "too long"
    for pat in [r"ignore previous instructions", r"delete.*chunks", r"drop table"]:
        if re.search(pat, q.lower()): return False, f"blocked {pat}"
    return True, "ok"
def filter_output(a):
    if not a or len(a.strip())<20: return False, a
    a = re.sub(r"GROQ_API_KEY\s*=\s*\S+", "[REDACTED]", a)
    return True, a
def score_guardrail(q,a,s):
    checks = {
        "input_valid": validate_input(q)[0],
        "output_valid": filter_output(a)[0],
        "score_pass": s>=0.85,
        "has_citations": "Chunk" in a or "Doc" in a or "1997" in a,
        "has_stateful": "stateful" in a.lower(),
        "length_ok": 300<=len(a)<=5000
    }
    passed = sum(checks.values())
    return {"checks": checks, "passed": f"{passed}/{len(checks)}", "all_pass": passed==len(checks), "guardrail_score": passed/len(checks)}
