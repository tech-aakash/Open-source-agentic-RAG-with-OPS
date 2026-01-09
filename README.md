# DGE Agentic RAG Server

An open‑source **Abu Dhabi DGE procurement assistant** built on:

- **FastAPI** (REST API server)
- **LlamaIndex** + **PGVector** (RAG over DGE documents)
- **HuggingFace embeddings** (BGE models)
- **Ollama** (local LLM – `qwen2.5:1.5b`)
- **Arize Phoenix** + OpenTelemetry (tracing & observability)
- Simple **long‑term conversation memory** + **RAGAS logging**
- **CrewAI tools** wrapping the pipeline

This README explains, step by step, how to:

1. Clone this repository
2. Set up all external dependencies (Postgres, Ollama, models, Phoenix)
3. Build the Docker image
4. Run the container exactly as you have it now
5. Call the API locally

---

## 0. Repository Layout (high‑level)

Your project should look roughly like this:

```text
project-root/
├─ crew_server.py              # The FastAPI app (code you pasted)
├─ requirements.txt
├─ Dockerfile
├─ models/                     # (On host) Folder with HF models
│   ├─ bge-large-en-v1.5/
│   └─ bge-reranker-v2-m3/
├─ conversation_memory.json    # Persisted memory (mounted into container)
├─ ragas_eval_data.json        # RAGAS log (mounted into container)
└─ README.md                   # This file
```

> If your Python file is named differently, make sure the Docker `CMD` matches it (`uvicorn your_file:app`).

---

## 1. Prerequisites

### 1.1. System requirements

- OS: macOS, Linux, or WSL2 on Windows
- RAM: ideally **16 GB+**
- Disk space: at least **10 GB** free (models + DB)

### 1.2. Install Git

If you do not have Git:

- **macOS:** Install Xcode command‑line tools or download Git from git-scm.
- **Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y git
```

### 1.3. Install Docker

Install Docker Desktop from the official Docker website and ensure the `docker` CLI works:

```bash
docker --version
```

You should see a version string, e.g. `Docker version 27.0.0, build ...`.

### 1.4. Install PostgreSQL + pgvector

You need a local PostgreSQL instance with the **pgvector** extension.

#### 1.4.1. Install Postgres

- **macOS (Homebrew):**

```bash
brew install postgresql
brew services start postgresql
```

- **Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
sudo service postgresql start
```

#### 1.4.2. Create the database

Open a `psql` shell:

```bash
psql -U postgres
```

Then in psql, create the DB (name matches your env variable `DB_NAME=dge`):

```sql
CREATE DATABASE dge;
\c dge;
```

#### 1.4.3. Install and enable pgvector

How you install `pgvector` may depend on OS. On many systems:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If this fails, install `pgvector` via your package manager (e.g. `brew install pgvector` on macOS, or use Postgres extension installation instructions), then run the `CREATE EXTENSION` command again.

> The `PGVectorStore` in your code will create the table (`rag_chunks`) automatically if it doesn’t exist, as long as the `vector` extension is enabled and `embed_dim` matches.

### 1.5. Install and start Ollama

Ollama is used to run the local LLM `qwen2.5:1.5b`.

1. Install Ollama from the official website.
2. Start Ollama (usually by launching the app or running `ollama serve`).
3. Pull the model:

```bash
ollama pull qwen2.5:1.5b
```

Make sure it’s running on your host **outside Docker** at:

```text
http://localhost:11434
```

Your container will access it via `http://host.docker.internal:11434`.

### 1.6. Download HuggingFace models (BGE)

You need two local models:

- `bge-large-en-v1.5` (embedding model)
- `bge-reranker-v2-m3` (reranker model)

Download them (e.g. with `git lfs` or `huggingface-cli`) into a folder on your host machine. For example:

```bash
mkdir -p /Users/<your_user>/Open-source-agentic-RAG-with-OPS/models
cd /Users/<your_user>/Open-source-agentic-RAG-with-OPS/models

# Example using git (exact repo names may differ – use the HF pages for precise commands)
git clone https://huggingface.co/BAAI/bge-large-en-v1.5 bge-large-en-v1.5
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3 bge-reranker-v2-m3
```

> The **host path** to this folder is what you mount as `/models` into the container.

### 1.7. (Optional) Install Arize Phoenix

If you want tracing:

- Run Phoenix following its docs (often via Docker).
- Ensure it listens at:

```text
http://localhost:6006/v1/traces
```

The container will reach it via `http://host.docker.internal:6006/v1/traces`.

---

## 2. Clone the repository

Replace `<your-github-username>` and `<repo-name>` with your actual GitHub values.

```bash
cd /path/where/you/want/the/project
git clone https://github.com/<your-github-username>/<repo-name>.git
cd <repo-name>
```

This directory is your **project root**.

---

## 3. (Optional) Run locally without Docker

If you want to run the server directly on your machine first:

### 3.1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3.2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Set environment variables

You can export them in your shell or create a `.env` (and load with `python-dotenv` if you like). For now, export them directly:

```bash
export DB_NAME=dge
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=aakashwalavalkar   # or your own Postgres user
export DB_PASSWORD=aakash1234     # or your own password
export BGE_LOCAL_DIR=/absolute/path/to/models/bge-large-en-v1.5
export BGE_RERANKER_DIR=/absolute/path/to/models/bge-reranker-v2-m3
export OLLAMA_BASE_URL=http://localhost:11434
export PHOENIX_ENDPOINT=http://localhost:6006/v1/traces
export PYTHONUNBUFFERED=1
```

### 3.4. Run the FastAPI app

Assuming your main file is `crew_server.py` and the `FastAPI` instance is `app`:

```bash
uvicorn crew_server:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- Swagger UI: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json

If this works, you’re ready for Docker.

---

## 4. Build the Docker image

From the **project root** (where `Dockerfile` and `requirements.txt` live):

```bash
docker build -t dge-rag-server .
```

What the Dockerfile does:

- Uses `python:3.11-slim`
- Installs system deps for PostgreSQL (`libpq-dev`, `build-essential`)
- Installs Python deps from `requirements.txt`
- Copies the whole project into `/app`
- Sets default environment variables (you can override them at `docker run`)
- Exposes port `8000`
- Runs:

```bash
uvicorn crew_server:app --host 0.0.0.0 --port 8000
```

---

## 5. Prepare host files & folders

On your **host machine**, make sure you have:

1. **Models folder** (used as `/models` inside container):

   ```bash
   /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/models
   ├─ bge-large-en-v1.5/
   └─ bge-reranker-v2-m3/
   ```

2. **Memory and RAGAS files** (if they don’t exist, create empty JSON files):

   ```bash
   touch /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/conversation_memory.json
   touch /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/ragas_eval_data.json
   ```

   Initialize them with empty lists to avoid JSON parse errors:

   ```bash
   echo "[]" > /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/ragas_eval_data.json
   echo '{"user_message_count": 0, "pending": [], "summaries": []}'      > /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/conversation_memory.json
   ```

---

## 6. Run the Docker container

Use the same startup command you shared (from the project root or any directory):

```bash
docker run --rm   -p 8000:8000   -e DB_NAME=dge   -e DB_HOST=host.docker.internal   -e DB_PORT=5432   -e DB_USER=aakashwalavalkar   -e DB_PASSWORD=aakash1234   -e BGE_LOCAL_DIR=/models/bge-large-en-v1.5   -e BGE_RERANKER_DIR=/models/bge-reranker-v2-m3   -e OLLAMA_BASE_URL=http://host.docker.internal:11434   -e PHOENIX_ENDPOINT=http://host.docker.internal:6006/v1/traces   -v /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/models:/models   -v /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/conversation_memory.json:/app/conversation_memory.json   -v /Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/ragas_eval_data.json:/app/ragas_eval_data.json   dge-rag-server
```

**Explanation of key pieces:**

- `-p 8000:8000`  
  Maps container port 8000 → host port 8000 (so you can hit `http://localhost:8000`).

- `DB_HOST=host.docker.internal`  
  Inside the container, this points back to your host machine’s `localhost`. This is crucial so the container can reach your host Postgres and Ollama.

- `BGE_LOCAL_DIR=/models/bge-large-en-v1.5`  
  Inside the container, the embedding model lives at `/models/...`. The host model directory is mounted to `/models` using `-v`.

- `-v <host_path>:/models`  
  Makes the host `models/` directory visible inside the container at `/models`.

- `-v <host_json>:/app/conversation_memory.json` and `-v <host_json>:/app/ragas_eval_data.json`  
  Persist conversation memory and RAGAS logs on your host between container runs.

If the container starts successfully, you should see logs similar to:

```text
[INIT] Loading embeddings...
[INIT] Connecting to PGVector...
[INIT] Building index...
[INIT] Loading Ollama model...
[INIT] Loading BGE reranker v2-m3...
[INIT] Creating query engine with reranker...
[MEMORY] Loaded memory from conversation_memory.json
[RAGAS_LOG] Loaded ... examples from ragas_eval_data.json
```

---

## 7. Test the running API

Once the container is running, open:

- Swagger UI: http://localhost:8000/docs
- Models endpoint: http://localhost:8000/v1/models

You should see something like:

```json
{
  "object": "list",
  "data": [
    {
      "id": "dge-rag",
      "object": "model",
      "owned_by": "crew-backend"
    }
  ]
}
```

### 7.1. Example: call `/v1/chat/completions` via curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "dge-rag",
    "messages": [
      { "role": "user", "content": "Hi" }
    ]
  }'
```

You should receive a response like:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "dge-rag",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Now try a procurement‑related question that triggers the RAG pipeline:

```bash
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "dge-rag",
    "messages": [
      { "role": "user", "content": "What is the procurement threshold for direct purchase?" }
    ]
  }'
```

If your PGVector store is populated with DGE documents, you should see a detailed answer plus `Sources:` in the text.

---

## 8. How to rebuild the project on another machine (summary checklist)

On **any new machine**, to fully reproduce your setup:

1. **Install prerequisites**
   - Git
   - Docker
   - PostgreSQL + pgvector
   - Ollama + `qwen2.5:1.5b`
   - (Optional) Arize Phoenix

2. **Clone the repo**

   ```bash
   git clone https://github.com/<your-github-username>/<repo-name>.git
   cd <repo-name>
   ```

3. **Download HuggingFace models** into a host `models` folder  
   (`bge-large-en-v1.5`, `bge-reranker-v2-m3`).

4. **Create Postgres DB** `dge` and enable `CREATE EXTENSION vector;`.

5. **Build Docker image**

   ```bash
   docker build -t dge-rag-server .
   ```

6. **Create memory + RAGAS JSON files** on the host if needed.

7. **Run container** with:

   - Port mapping `-p 8000:8000`
   - Env vars for DB & OLLAMA
   - Volume mounts for `/models` and JSON files.

8. **Open `http://localhost:8000/docs`** and test `/v1/chat/completions`.

Once these steps are followed, anyone can rebuild and run the **exact same project** on their own machine.

---

## 9. Troubleshooting

- **Error: cannot connect to Postgres**

  - Check Postgres is running on the host.
  - From host: `psql -h localhost -U <user> -d dge`.
  - Inside container logs, confirm `DB_HOST=host.docker.internal`, `DB_PORT=5432`.

- **Error: cannot connect to Ollama**

  - Confirm Ollama is running on host: `curl http://localhost:11434/api/tags`.
  - Ensure container uses `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

- **Model loading errors**

  - Verify that `/models` inside the container contains:
    - `/models/bge-large-en-v1.5`
    - `/models/bge-reranker-v2-m3`
  - Check the host path in your `-v` mount is correct.

- **Phoenix not reachable**

  - If you don’t need tracing, you can change `PHOENIX_ENDPOINT` to a dummy value or disable Phoenix in code.
  - Otherwise, ensure Phoenix is running at `http://localhost:6006/v1/traces` on the host.

