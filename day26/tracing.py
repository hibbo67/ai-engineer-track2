import os, json, time
from datetime import datetime
class Tracer:
    def __init__(self):
        self.log_file = "day26/traces.jsonl"
        os.makedirs("day26", exist_ok=True)
    def start_trace(self, query, thread_id):
        return {"trace_id": f"trace_{int(time.time()*1000)}", "query": query, "thread_id": thread_id, "start_ts": datetime.now().isoformat(), "spans": [], "model": "openai/gpt-oss-20b", "chunks": 1997}
    def add_span(self, trace, name, duration_ms, data):
        trace["spans"].append({"name": name, "duration_ms": duration_ms, "timestamp": datetime.now().isoformat(), "data": data})
    def end_trace(self, trace, answer, score):
        trace["end_ts"] = datetime.now().isoformat()
        trace["answer_len"] = len(answer)
        trace["final_score"] = score
        trace["total_duration_ms"] = sum(s["duration_ms"] for s in trace["spans"])
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(trace)+"\n")
        except: pass
        return trace
tracer = Tracer()
