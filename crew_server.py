from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Literal
import json
import time
import os
from threading import Lock

from crewai.tools import tool

# LlamaIndex RAG components
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import SentenceTransformerRerank
# ========= Arize Phoenix / OpenInference Tracing =========
from phoenix.otel import register as phoenix_register
from opentelemetry import trace
# =========================================================


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

# NEW: local BGE reranker path
BGE_RERANKER_DIR = (
    "/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/"
    "models/bge-reranker-v2-m3"
)

EMBED_DIM = 1024
OLLAMA_MODEL = "qwen2.5:1.5b"
TOP_K = 5  # final number of docs LLM will see
RERANK_RETRIEVAL_K = TOP_K * 2  # how many docs to fetch before reranking

app = FastAPI(
    title="CrewAI Agentic RAG Server",
    version="2.2.0",
    description="Abu Dhabi DGE RAG with CrewAI tools, routing, guardrails, validation, and memory.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ======================================================
#          Arize Phoenix Tracing Setup
# ======================================================

# Configure Phoenix OTEL and enable automatic instrumentation (LlamaIndex, etc.)
tracer_provider = phoenix_register(
    project_name="dge-rag-pipeline",
    endpoint="http://localhost:6006/v1/traces",
    auto_instrument=True,   # auto-trace supported AI libs
    batch=True,             # batch span export
)

# Phoenix-aware tracer used for decorators (chain/llm/etc.)
tracer = tracer_provider.get_tracer("dge-rag-app")


# ======================================================
#          LlamaIndex / RAG components
# ======================================================

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

# NEW: BGE reranker
print("[INIT] Loading BGE reranker v2-m3...")
reranker = SentenceTransformerRerank(
    model=BGE_RERANKER_DIR,  # local folder
    top_n=TOP_K,             # keep TOP_K docs after reranking
)

print("[INIT] Creating query engine with reranker...")
query_engine = index.as_query_engine(
    similarity_top_k=RERANK_RETRIEVAL_K,   # fetch more from PGVector
    node_postprocessors=[reranker],        # then rerank and trim
    llm=llm,
    response_mode="tree_summarize",
)


# ======================================================
#              MEMORY STATE & HELPERS
# ======================================================

MEMORY_FILE = "conversation_memory.json"
memory_lock = Lock()

conversation_state = {
    "user_message_count": 0,
    "pending": [],
    "summaries": [],
}


def load_memory():
    global conversation_state
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                conversation_state = json.load(f)
            print(f"[MEMORY] Loaded memory from {MEMORY_FILE}")
        except Exception as e:
            print(f"[MEMORY] Failed to load memory, starting fresh. Error: {e}")
    else:
        print("[MEMORY] No existing memory file, starting fresh.")


def save_memory():
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(conversation_state, f, indent=2)
        print(f"[MEMORY] Saved memory to {MEMORY_FILE}")
    except Exception as e:
        print(f"[MEMORY] Failed to save memory: {e}")


@tracer.chain
def summarize_window(interactions: List[dict]) -> str:
    """
    Summarize a window of 5 user-assistant interactions into a compact memory.
    """
    span = trace.get_current_span()
    span.set_attribute("dge.component", "summarize_window")
    span.set_attribute("dge.num_interactions", len(interactions))

    convo_text = ""
    for turn in interactions:
        convo_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    prompt = f"""
You are summarizing a short conversation between a user and an Abu Dhabi DGE procurement assistant.

Summarize the key points of this conversation in 3–5 bullet points:
- Focus on what the user asked or cares about
- Important policies / thresholds / rules mentioned
- Any clear preferences or constraints from the user

Conversation:
{convo_text}

Return ONLY the summary text (bullets or a short paragraph).
"""
    out = llm.complete(prompt)
    summary = str(out).strip()
    print("[MEMORY] Summary for last 5 messages:", summary)
    span.set_attribute("dge.memory.summary", summary)
    return summary


def update_memory(user_msg: str, assistant_msg: str):
    """
    Append the latest interaction and every 5th user message:
      - Summarize the last 5 exchanges
      - Store summary
      - Clear detailed pending history
    """
    with memory_lock:
        st = conversation_state
        st["user_message_count"] += 1
        st["pending"].append(
            {
                "user": user_msg,
                "assistant": assistant_msg,
            }
        )
        print(f"[MEMORY] Recorded interaction #{st['user_message_count']}")

        # When we reach a multiple of 5 user messages → summarize & clear
        if st["user_message_count"] % 5 == 0:
            print("[MEMORY] Reached 5-user-message window. Summarizing...")
            summary = summarize_window(st["pending"])
            window_index = len(st["summaries"]) + 1
            st["summaries"].append(
                {
                    "window_index": window_index,
                    "timestamp": int(time.time()),
                    "summary": summary,
                }
            )
            st["pending"] = []
            print(f"[MEMORY] Stored summary for window {window_index} and cleared pending turns.")

        save_memory()


def get_memory_summaries_text(max_summaries: int = 5) -> str:
    """
    Returns a concatenated text of recent conversation summaries.
    """
    with memory_lock:
        summaries = conversation_state.get("summaries", [])
        if not summaries:
            return ""

        recent = summaries[-max_summaries:]
        lines = []
        for s in recent:
            idx = s.get("window_index", "?")
            lines.append(f"Window {idx}:\n{s.get('summary', '')}")
        memory_text = "\n\n".join(lines)
        print("[MEMORY] Using the following summaries for context:\n", memory_text)
        return memory_text


@tracer.chain
def enhance_with_memory(question: str, rag_answer: str) -> str:
    """
    Take the RAG answer and refine it using stored conversation memory (summaries).
    If no memory is available, return rag_answer unchanged.
    """
    span = trace.get_current_span()
    span.set_attribute("dge.component", "memory_enhance")
    span.set_attribute("dge.user_question", question)

    mem_text = get_memory_summaries_text()
    if not mem_text.strip():
        span.set_attribute("dge.memory.has_memory", False)
        return rag_answer

    span.set_attribute("dge.memory.has_memory", True)
    span.set_attribute("dge.rag_answer_before", rag_answer)

    prompt = f"""
You are an Abu Dhabi DGE procurement assistant.

You have a draft answer based on current procurement documents:

DRAFT ANSWER:
{rag_answer}

You also have summaries of this user's previous conversation with you:

CONVERSATION MEMORY:
{mem_text}

Task:
- If the memory is relevant to the current question, slightly refine or extend the draft answer
  to maintain continuity, recall their previous questions, or clarify related policies.
- Do NOT change factual details that come from the procurement standards.
- If the memory is not relevant, just return the draft answer as-is.

Return ONLY the final answer text.
"""

    out = llm.complete(prompt)
    final_answer = str(out).strip()
    print("[MEMORY] Enhanced answer with memory.")
    span.set_attribute("dge.rag_answer_after", final_answer)
    return final_answer


# Load existing memory once on startup
load_memory()


# ======================================================
#              CORE LOGIC FUNCTIONS
# ======================================================

@tracer.chain
def guardrail_logic(question: str) -> GuardrailOutput:
    """Run abusive-language guardrail and return GuardrailOutput."""
    span = trace.get_current_span()
    span.set_attribute("dge.component", "guardrail")
    span.set_attribute("dge.user_question", question)

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
        span.set_attribute("dge.guardrail.is_abusive", result.is_abusive)
        span.set_attribute("dge.guardrail.explanation", result.explanation)
        return result
    except Exception as e:
        print("[GUARDRAIL] JSON parse/validation error:", e)
        span.set_attribute("dge.guardrail.error", str(e))
        return GuardrailOutput(
            is_abusive=False,
            explanation="Failed to parse model output; treating as safe.",
        )


@tracer.chain
def router_logic(question: str) -> RouterOutput:
    """Route query intent to greeting / rag / other."""
    span = trace.get_current_span()
    span.set_attribute("dge.component", "router")
    span.set_attribute("dge.user_question", question)

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
        span.set_attribute("dge.router.route", result.route)
        span.set_attribute("dge.router.reason", result.reason)
        return result
    except Exception as e:
        print("[ROUTER] JSON parse/validation error:", e)
        span.set_attribute("dge.router.error", str(e))
        return RouterOutput(
            route="rag",
            reason="Failed to parse model output; defaulting to rag.",
        )


@tracer.chain
def greeting_logic(question: str) -> str:
    """Produce a short, warm DGE-branded greeting."""
    span = trace.get_current_span()
    span.set_attribute("dge.component", "greeting")
    span.set_attribute("dge.user_question", question)

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
    span.set_attribute("dge.greeting.answer", out)
    return out


@tracer.chain
def rag_logic(question: str) -> dict:
    """Run the RAG engine and return {'answer': str, 'sources': list[dict]}."""
    span = trace.get_current_span()
    span.set_attribute("dge.component", "rag")
    span.set_attribute("dge.user_question", question)

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

    span.set_attribute("dge.rag.answer", answer)
    span.set_attribute("dge.rag.num_sources", len(sources))
    span.set_attribute(
        "dge.rag.sources",
        json.dumps(
            [
                {"file_name": s["file_name"], "score": s["score"]}
                for s in sources
            ]
        ),
    )

    return {"answer": answer, "sources": sources}


@tracer.chain
def validator_logic(question: str, sources_json: str, answer: str) -> ValidatorOutput:
    """Validate RAG answer vs question & docs."""
    span = trace.get_current_span()
    span.set_attribute("dge.component", "validator")
    span.set_attribute("dge.user_question", question)

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
        span.set_attribute("dge.validator.is_valid", result.is_valid)
        span.set_attribute("dge.validator.feedback", result.feedback)
        return result
    except Exception as e:
        print("[VALIDATOR] JSON parse/validation error:", e)
        span.set_attribute("dge.validator.error", str(e))
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


@tracer.chain
def chat_pipeline(user_msg: str, model: str) -> str:
    """
    Core chat pipeline used by the FastAPI endpoint.
    This is the span where you'll see:
    - Input: user_msg, model
    - Output: final_answer
    """
    span = trace.get_current_span()
    span.set_attribute("dge.component", "chat_pipeline")
    span.set_attribute("dge.user_question", user_msg)
    span.set_attribute("dge.model", model)

    # 1) Guardrail
    guard = guardrail_logic(user_msg)
    span.set_attribute("dge.guardrail.is_abusive", guard.is_abusive)

    if guard.is_abusive:
        assistant_answer = (
            "Please avoid abusive language. I can help with Abu Dhabi DGE procurement-related questions."
        )
        span.set_attribute("dge.route", "blocked_by_guardrail")
        span.set_attribute("dge.final_answer", assistant_answer)
        update_memory(user_msg, assistant_answer)
        return assistant_answer

    # 2) Router
    router = router_logic(user_msg)
    span.set_attribute("dge.route", router.route)
    span.set_attribute("dge.router.reason", router.reason)

    if router.route == "greeting":
        assistant_answer = greeting_logic(user_msg)
        span.set_attribute("dge.final_answer", assistant_answer)
        update_memory(user_msg, assistant_answer)
        return assistant_answer

    if router.route == "rag":
        # Run RAG
        rag = rag_logic(user_msg)

        # Enhance with memory summaries, if any
        enriched_answer = enhance_with_memory(user_msg, rag["answer"])

        # Validate the enriched answer
        validator = validator_logic(user_msg, json.dumps(rag["sources"]), enriched_answer)
        span.set_attribute("dge.validator.is_valid", validator.is_valid)
        span.set_attribute("dge.validator.feedback", validator.feedback)

        final_answer = enriched_answer
        if not validator.is_valid:
            final_answer += f"\n\n[Validator note] {validator.feedback}"

        if rag["sources"]:
            src_lines = [
                f"- {s.get('file_name', 'Unknown')} (score={s.get('score',0.0):.3f})"
                for s in rag["sources"]
            ]
            final_answer += "\n\nSources:\n" + "\n".join(src_lines)

        span.set_attribute("dge.final_answer", final_answer)
        update_memory(user_msg, final_answer)
        return final_answer

    # route == "other"
    assistant_answer = (
        "I mainly help with Abu Dhabi DGE procurement policies and processes. "
        "Please ask a procurement-related question."
    )
    span.set_attribute("dge.final_answer", assistant_answer)
    update_memory(user_msg, assistant_answer)
    return assistant_answer


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat(req: ChatCompletionRequest):
    user_msg = req.messages[-1].content
    print("[CHAT] User:", user_msg)

    final_answer = chat_pipeline(user_msg=user_msg, model=req.model)
    return build_response(req.model, final_answer)


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