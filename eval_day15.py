import os, requests, time
KEY = os.getenv("GROQ_API_KEY")
if not KEY:
    print("Set env: export GROQ_API_KEY=gsk_REDACTED

"); exit(1)
# rest of your working eval code (no hardcoded gsk_ inside)
