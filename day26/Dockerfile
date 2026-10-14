FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY day26/requirements.txt./day26/requirements.txt
COPY day23/requirements.txt./day23/requirements.txt
RUN pip install --no-cache-dir -r day26/requirements.txt || pip install mcp chromadb sentence-transformers groq python-dotenv fastapi uvicorn langsmith

# Copy code
COPY day23/./day23/
COPY day25/./day25/
COPY day26/./day26/
COPY chroma_db/./chroma_db/

# Env
ENV PYTHONPATH=/app
ENV GROQ_API_KEY=""
ENV LANGSMITH_TRACING=false

EXPOSE 8000 8001

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "day26/app.py", "--host", "0.0.0.0", "--port", "8000"]
