FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use the minimal deps inside the container
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project (code, etc.)
COPY . .

ENV DB_NAME=dge \
    DB_HOST=host.docker.internal \
    DB_PORT=5432 \
    DB_USER=aakashwalavalkar \
    DB_PASSWORD=aakash1234 \
    BGE_LOCAL_DIR=/models/bge-large-en-v1.5 \
    BGE_RERANKER_DIR=/models/bge-reranker-v2-m3 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    PHOENIX_ENDPOINT=http://host.docker.internal:6006/v1/traces \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "crew_server:app", "--host", "0.0.0.0", "--port", "8000"]