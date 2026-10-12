FROM python:3.12-slim

WORKDIR /app

# System deps for chroma + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY day24/requirements.txt ./day24/
COPY day23/requirements.txt ./day23/ 2>/dev/null || echo "groq\nchromadb\nsentence-transformers\npython-dotenv\nfastapi\nuvicorn\npydantic" > ./day23/requirements.txt

RUN pip install --no-cache-dir -r day24/requirements.txt
RUN pip install --no-cache-dir groq chromadb sentence-transformers python-dotenv fastapi uvicorn pydantic

# Copy code
COPY day23/ ./day23/
COPY day24/ ./day24/
COPY chroma_db/ ./chroma_db/ 2>/dev/null || mkdir -p ./chroma_db

ENV PYTHONPATH=/app
ENV GROQ_API_KEY=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "day24.api:app", "--host", "0.0.0.0", "--port", "8000"]
