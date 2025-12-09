from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Literal
import json
import time

from crewai.tools import tool

# LlamaIndex RAG components
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# ======================================================
#           JSON HELPERS (NO REGEX)
# ======================================================

def extract_json(text: str) -> str:
    """
    Extract a JSON object substring from an LLM output that may contain
    code fences and extra text. Returns best-effort JSON string.
    """
    s = text.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1].strip()
    return s


# ======================================================
#                Pydantic Output Schemas
# ======================================================

class GuardrailOutput(BaseModel):
    is_abusive: bool
    explanation: str


class RouterOutput(BaseModel):
    route: Literal["greeting", "rag", "other"]
    reason: str


class ValidatorOutput(BaseModel):
    is_valid: bool
    feedback: str


# ======================================================
#              FastAPI + RAG Initialization
# ======================================================

DB_NAME = "dge"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "aakashwalavalkar"
DB_PASSWORD = "aakash1234"
TABLE_NAME = "rag_chunks"

BGE_LOCAL_DIR = (
    "/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/"
    "models/bge-large-en-v1.5"
)

EMBED_DIM = 1024
OLLAMA_MODEL = "qwen2.5:1.5b"
TOP_K = 5

app = FastAPI(
    title="CrewAI Agentic RAG Server",
    version="2.1.0",
    description="Abu Dhabi DGE RAG with CrewAI tools, routing, guardrails, and validation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

print("[INIT] Loading embeddings...")
embed_model = HuggingFaceEmbedding(model_name=BGE_LOCAL_DIR, normalize=True)

print("[INIT] Connecting to PGVector...")
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    table_name=TABLE_NAME,
    embed_dim=EMBED_DIM,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("[INIT] Building index...")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("[INIT] Loading Ollama model...")
llm = Ollama(
    model=OLLAMA_MODEL,
    system_prompt="You are an Abu Dhabi DGE procurement assistant."
)

print("[INIT] Creating query engine...")
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    llm=llm,
    response_mode="tree_summarize",
)


# ======================================================
#              CORE LOGIC FUNCTIONS
# ======================================================

def guardrail_logic(question: str) -> GuardrailOutput:
    """Run abusive-language guardrail and return GuardrailOutput."""
    print(f"[GUARDRAIL] Input: {question!r}")

    prompt = f"""
You are a content safety classifier.

Classify the user message as either SAFE or ABUSIVE.

Return ONLY a JSON object with exactly these fields:
- "is_abusive": true or false
- "explanation": a very short explanation

Example:
{{
  "is_abusive": false,
  "explanation": "Just a normal greeting."
}}

User message:
"{question}"
"""

    raw = str(llm.complete(prompt)).strip()
    print("[GUARDRAIL] Raw LLM output:", raw)

    clean = extract_json(raw)
    print("[GUARDRAIL] Extracted JSON candidate:", clean)

    try:
        data = json.loads(clean)
        result = GuardrailOutput(**data)
        print("[GUARDRAIL] Parsed GuardrailOutput:", result)
        return result
    except Exception as e:
        print("[GUARDRAIL] JSON parse/validation error:", e)
        # default to safe
        return GuardrailOutput(
            is_abusive=False,
            explanation="Failed to parse model output; treating as safe.",
        )


def router_logic(question: str) -> RouterOutput:
    """Route query intent to greeting / rag / other."""
    print(f"[ROUTER] Input: {question!r}")

    prompt = f"""
You are a router for an Abu Dhabi DGE assistant.

Classify the intent of the user message into EXACTLY one category:

- "greeting": simple greeting or smalltalk (hi, hello, good morning, etc.).
- "rag": any question that should use the DGE procurement RAG system.
- "other": anything else.

Return ONLY a JSON object with exactly:
- "route": "greeting" | "rag" | "other"
- "reason": short explanation

Example:
{{
  "route": "greeting",
  "reason": "The message is just 'Hi'."
}}

User message:
"{question}"
"""

    raw = str(llm.complete(prompt)).strip()
    print("[ROUTER] Raw LLM output:", raw)

    clean = extract_json(raw)
    print("[ROUTER] Extracted JSON candidate:", clean)

    try:
        data = json.loads(clean)
        result = RouterOutput(**data)
        print("[ROUTER] Parsed RouterOutput:", result)
        return result
    except Exception as e:
        print("[ROUTER] JSON parse/validation error:", e)
        # default to RAG if router fails
        return RouterOutput(
            route="rag",
            reason="Failed to parse model output; defaulting to rag.",
        )


def greeting_logic(question: str) -> str:
    """Produce a short, warm DGE-branded greeting."""
    print(f"[GREETING] Input: {question!r}")

    prompt = """
The user has greeted you.

Respond as an Abu Dhabi DGE procurement assistant:
- Warm and polite
- 1–2 short sentences
- Optional: invite them to ask a procurement-related question
"""

    out = str(llm.complete(prompt)).strip()
    print("[GREETING] Output:", out)
    return out


def rag_logic(question: str) -> dict:
    """Run the RAG engine and return {'answer': str, 'sources': list[dict]}."""
    print(f"[RAG] Input: {question!r}")

    resp = query_engine.query(question)
    answer = str(resp)
    print("[RAG] Answer:", answer)

    sources = []
    for sn in resp.source_nodes:
        meta = sn.node.metadata or {}
        src = {
            "score": float(sn.score or 0.0),
            "file_name": meta.get("file_name", ""),
            "snippet": sn.node.get_content()[:500],
        }
        sources.append(src)
        print("[RAG] Source:", src)

    return {"answer": answer, "sources": sources}


def validator_logic(question: str, sources_json: str, answer: str) -> ValidatorOutput:
    """Validate RAG answer vs question & docs."""
    print("[VALIDATOR] Running...")
    print("[VALIDATOR] Question:", question)
    print("[VALIDATOR] Sources_json:", sources_json)
    print("[VALIDATOR] Answer:", answer)

    prompt = f"""
You are a validator for a RAG system.

Check whether the answer is:
- Relevant to the user's question.
- Consistent with the documents (if any).
- Free of obvious hallucinations or unsupported details.

Return ONLY a JSON object with:
- "is_valid": true or false
- "feedback": very short explanation

Example:
{{
  "is_valid": true,
  "feedback": "Relevant and consistent with the documents."
}}

QUESTION:
{question}

DOCUMENTS (JSON):
{sources_json}

ANSWER:
{answer}
"""

    raw = str(llm.complete(prompt)).strip()
    print("[VALIDATOR] Raw LLM output:", raw)

    clean = extract_json(raw)
    print("[VALIDATOR] Extracted JSON candidate:", clean)

    try:
        data = json.loads(clean)
        result = ValidatorOutput(**data)
        print("[VALIDATOR] Parsed ValidatorOutput:", result)
        return result
    except Exception as e:
        print("[VALIDATOR] JSON parse/validation error:", e)
        return ValidatorOutput(
            is_valid=True,
            feedback="Failed to parse validator output; treating as valid.",
        )


# ======================================================
#              CREWAI TOOL WRAPPERS
# ======================================================

@tool("dge_guardrail_tool")
def t_guardrail(question: str) -> str:
    """CrewAI tool: run guardrail over the question and return JSON."""
    result = guardrail_logic(question)
    return result.json()


@tool("dge_router_tool")
def t_router(question: str) -> str:
    """CrewAI tool: decide if query should go to greeting or rag."""
    result = router_logic(question)
    return result.json()


@tool("dge_greeting_tool")
def t_greeting(question: str) -> str:
    """CrewAI tool: answer greetings directly using the LLM."""
    return greeting_logic(question)


@tool("dge_rag_tool")
def t_rag(question: str) -> str:
    """CrewAI tool: run RAG over DGE documents and return JSON."""
    return json.dumps(rag_logic(question))


@tool("dge_validator_tool")
def t_validator(payload: str) -> str:
    """
    CrewAI tool: validate a RAG answer.

    Payload must be JSON:
    {
      "question": "...",
      "sources_json": "...",
      "answer": "..."
    }
    """
    data = json.loads(payload)
    result = validator_logic(
        data["question"],
        data["sources_json"],
        data["answer"],
    )
    return result.json()


# ======================================================
#        OpenAI-Compatible Chat Completion Endpoint
# ======================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


def build_response(model: str, content: str) -> ChatCompletionResponse:
    now = int(time.time())
    return ChatCompletionResponse(
        id=f"chatcmpl-{now}",
        object="chat.completion",
        created=now,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(),
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat(req: ChatCompletionRequest):
    user_msg = req.messages[-1].content
    print("[CHAT] User:", user_msg)

    # 1) Guardrail
    guard = guardrail_logic(user_msg)
    if guard.is_abusive:
        return build_response(
            req.model,
            "Please avoid abusive language. I can help with Abu Dhabi DGE procurement-related questions.",
        )

    # 2) Router
    router = router_logic(user_msg)

    if router.route == "greeting":
        return build_response(req.model, greeting_logic(user_msg))

    if router.route == "rag":
        rag = rag_logic(user_msg)
        validator = validator_logic(user_msg, json.dumps(rag["sources"]), rag["answer"])

        answer = rag["answer"]
        if not validator.is_valid:
            answer += f"\n\n[Validator note] {validator.feedback}"

        if rag["sources"]:
            src_lines = [
                f"- {s.get('file_name', 'Unknown')} (score={s.get('score',0.0):.3f})"
                for s in rag["sources"]
            ]
            answer += "\n\nSources:\n" + "\n".join(src_lines)

        return build_response(req.model, answer)

    # route == "other"
    return build_response(
        req.model,
        "I mainly help with Abu Dhabi DGE procurement policies and processes. "
        "Please ask a procurement-related question.",
    )


# ======================================================
#                MODELS LISTING
# ======================================================

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": "dge-rag", "object": "model", "owned_by": "crew-backend"}],
    }


# ======================================================
#                CUSTOM OPENAPI
# ======================================================

def custom_openapi():
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["servers"] = [{"url": "http://localhost:8000"}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi