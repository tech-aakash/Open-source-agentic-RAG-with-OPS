from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Literal, Optional
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
    # includes follow_up route
    route: Literal["greeting", "rag", "follow_up", "other"]
    reason: str


class ValidatorOutput(BaseModel):
    is_valid: bool
    feedback: str


class RewriterOutput(BaseModel):
    is_follow_up: bool
    rewritten_question: str
    reason: str


# ======================================================
#              FastAPI + RAG Initialization
# ======================================================

DB_NAME = os.getenv("DB_NAME", "dge")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "aakashwalavalkar")
DB_PASSWORD = os.getenv("DB_PASSWORD", "aakash1234")
TABLE_NAME = "rag_chunks"

# model folders will be mounted to /models in the container
# These defaults are for when running INSIDE docker
BGE_LOCAL_DIR = os.getenv(
    "BGE_LOCAL_DIR",
    "/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/models/bge-large-en-v1.5"
)

BGE_RERANKER_DIR = os.getenv(
    "BGE_RERANKER_DIR",
    "/Users/aakashwalavalkar/Desktop/Open-source-agentic-RAG-with-OPS/models/bge-reranker-v2-m3"
)

EMBED_DIM = 1024
OLLAMA_MODEL = "qwen2.5:1.5b"
TOP_K = 5
RERANK_RETRIEVAL_K = TOP_K * 2

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

PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")

tracer_provider = phoenix_register(
    project_name="dge-rag-pipeline",
    endpoint=PHOENIX_ENDPOINT,
    auto_instrument=True,
    batch=True,
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

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print("[INIT] Loading Ollama model...")
llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    system_prompt="You are an Abu Dhabi DGE procurement assistant."
)

# BGE reranker
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

# Last RAG Q&A kept in memory for follow-up routing/rewriting
last_rag_question: Optional[str] = None
last_rag_answer: Optional[str] = None


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


# Load existing memory once on startup
load_memory()


# ======================================================
#              RAGAS LOGGING HELPERS
# ======================================================

RAGAS_LOG_FILE = "ragas_eval_data.json"
ragas_log_lock = Lock()
ragas_log: List[dict] = []


def load_ragas_log():
    """Load existing RAGAS log from disk if present."""
    global ragas_log
    if os.path.exists(RAGAS_LOG_FILE):
        try:
            with open(RAGAS_LOG_FILE, "r") as f:
                ragas_log = json.load(f)
            print(f"[RAGAS_LOG] Loaded {len(ragas_log)} examples from {RAGAS_LOG_FILE}")
        except Exception as e:
            print(f"[RAGAS_LOG] Failed to load RAGAS log, starting fresh. Error: {e}")
            ragas_log = []
    else:
        print("[RAGAS_LOG] No existing RAGAS log file, starting fresh.")


def save_ragas_log():
    """Persist the RAGAS log to disk."""
    try:
        with open(RAGAS_LOG_FILE, "w") as f:
            json.dump(ragas_log, f, indent=2)
        print(f"[RAGAS_LOG] Saved {len(ragas_log)} examples to {RAGAS_LOG_FILE}")
    except Exception as e:
        print(f"[RAGAS_LOG] Failed to save RAGAS log: {e}")


def append_ragas_example(question: str, answer: str, sources: List[dict]):
    """
    Append one example in the format RAGAS expects:

    {
      "question": str,
      "answer": str,
      "contexts": List[str],
      "ground_truth": ""   # you will manually fill later
    }
    """
    contexts = [s.get("snippet", "") for s in sources]

    example = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ""  # you will manually fill this later
    }

    with ragas_log_lock:
        ragas_log.append(example)
        save_ragas_log()


# Load existing RAGAS log once on startup
load_ragas_log()


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
    """
    Route query intent to greeting / rag / follow_up / other.

    Router is aware of the LAST RAG question and answer to better detect follow-ups.
    """
    global last_rag_question, last_rag_answer

    span = trace.get_current_span()
    span.set_attribute("dge.component", "router")
    span.set_attribute("dge.user_question", question)

    print(f"[ROUTER] Input: {question!r}")
    print("[ROUTER] Last RAG question:", repr(last_rag_question))
    print("[ROUTER] Last RAG answer:", repr(last_rag_answer))

    last_q = last_rag_question or ""
    last_a = last_rag_answer or ""

    prompt = f"""
You are a router for an Abu Dhabi DGE assistant.

You must classify the CURRENT user message into EXACTLY one category:

- "greeting": simple greeting or smalltalk (hi, hello, good morning, etc.).
- "follow_up": a short or vague message that clearly depends on the previous procurement-related Q&A
  (for example, uses pronouns like "it", "this", "that", or phrases like "tell me more", "can you explain more",
   and is about the same topic as the last RAG question and answer).
- "rag": a new, self-contained question that should use the DGE procurement / HR by Laws / Information security /
  Procurement manual (Ariba Aligned or Business Process) RAG system.
  It mentions its topic explicitly (e.g., thresholds, exceptions, conflicts of interest, annexes, articles, etc.).
- "other": anything else that is not a greeting and not about the DGE RAG content.

You are given:
- CURRENT_MESSAGE: what the user just wrote
- LAST_RAG_QUESTION: the last procurement question that was sent to the RAG system (may be empty)
- LAST_RAG_ANSWER: the last answer from the RAG system (may be empty)

If the CURRENT_MESSAGE is clearly referring back to the LAST_RAG_QUESTION/ANSWER (e.g., "can you tell me more about it?",
"what about the thresholds you mentioned?", "and what are the risks?"), classify as "follow_up".

Return ONLY a JSON object with exactly:
- "route": "greeting" | "rag" | "follow_up" | "other"
- "reason": short explanation

Example for follow_up:
{{
  "route": "follow_up",
  "reason": "The message 'can you tell me more about this?' refers back to the previous question about exceptions."
}}

CURRENT_MESSAGE:
"{question}"

LAST_RAG_QUESTION:
"{last_q}"

LAST_RAG_ANSWER:
"{last_a}"
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
        # Fallback: default to "rag"
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


@tracer.chain
def rewriter_logic(current_question: str, last_question: str, last_answer: str) -> RewriterOutput:
    """
    Decide if the current question is a follow-up to the last RAG question/answer.
    If it is, rewrite it into a fully self-contained question that includes context.
    """
    span = trace.get_current_span()
    span.set_attribute("dge.component", "rewriter")
    span.set_attribute("dge.current_question", current_question)
    span.set_attribute("dge.last_rag_question", last_question)

    print("[REWRITER] Current question:", current_question)
    print("[REWRITER] Last RAG question:", last_question)
    print("[REWRITER] Last RAG answer:", last_answer)

    prompt = f"""
You are a query rewriter for a procurement RAG system.

You are given:
- The user's CURRENT message
- The LAST procurement-related user question
- The assistant's LAST answer

Your tasks:
1. Decide if the CURRENT message is a follow-up that relies on the previous Q&A
   (e.g., uses pronouns like "it", "that", "this", "more details", "tell me more", or clearly refers to the same topic).
2. If it IS a follow-up, rewrite the CURRENT message into a fully self-contained question
   that includes the necessary context from the previous Q&A.
3. If it is NOT a follow-up, simply copy the CURRENT message as-is.

Return ONLY a JSON object with:
- "is_follow_up": true or false
- "rewritten_question": the full, explicit question to send to RAG
- "reason": a very short explanation

Example when it IS a follow-up:
{{
  "is_follow_up": true,
  "rewritten_question": "Can you provide more details about the definition of 'Critical Entity' as provided in Annex F of the UAE IA Standards?",
  "reason": "The user said 'tell me more about it', referring to the last question."
}}

Example when it is NOT a follow-up:
{{
  "is_follow_up": false,
  "rewritten_question": "What are the KPIs for procurement performance?",
  "reason": "The user asked about a new topic unrelated to the last Q&A."
}}

CURRENT message:
"{current_question}"

LAST user question:
"{last_question}"

LAST assistant answer:
"{last_answer}"
"""

    raw = str(llm.complete(prompt)).strip()
    print("[REWRITER] Raw LLM output:", raw)

    clean = extract_json(raw)
    print("[REWRITER] Extracted JSON candidate:", clean)

    try:
        data = json.loads(clean)
        result = RewriterOutput(**data)
        print("[REWRITER] Parsed RewriterOutput:", result)
        span.set_attribute("dge.rewriter.is_follow_up", result.is_follow_up)
        span.set_attribute("dge.rewriter.reason", result.reason)
        span.set_attribute("dge.rewriter.rewritten_question", result.rewritten_question)
        return result
    except Exception as e:
        print("[REWRITER] JSON parse/validation error:", e)
        span.set_attribute("dge.rewriter.error", str(e))
        # Fallback: not a follow-up, keep original question
        return RewriterOutput(
            is_follow_up=False,
            rewritten_question=current_question,
            reason="Failed to parse rewriter output; treating as not a follow-up.",
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
    """CrewAI tool: decide if query should go to greeting, rag, follow_up, or other."""
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


@tool("dge_rewriter_tool")
def t_rewriter(payload: str) -> str:
    """
    CrewAI tool: rewrite a follow-up question into a full question if needed.

    Payload must be JSON:
    {
      "current_question": "...",
      "last_question": "...",
      "last_answer": "..."
    }
    """
    data = json.loads(payload)
    result = rewriter_logic(
        data["current_question"],
        data["last_question"],
        data["last_answer"],
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

    Routing logic:

    - greeting  -> greeting_logic only
    - rag       -> direct RAG (no rewriter)
    - follow_up -> rewriter + RAG
    - other     -> generic domain message
    """
    global last_rag_question, last_rag_answer

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
        # NOTE: we don't log abusive cases to RAGAS
        return assistant_answer

    # 2) Router
    router = router_logic(user_msg)
    span.set_attribute("dge.route", router.route)
    span.set_attribute("dge.router.reason", router.reason)

    # 3) Greeting route -> greeting only
    if router.route == "greeting":
        assistant_answer = greeting_logic(user_msg)
        span.set_attribute("dge.final_answer", assistant_answer)
        update_memory(user_msg, assistant_answer)
        # We also don't log pure greetings to RAGAS
        return assistant_answer

    # Prepare question that will go to RAG
    question_for_rag = user_msg

    # 4) Follow-up route -> rewriter + RAG
    if router.route == "follow_up":
        if last_rag_question and last_rag_answer:
            rew = rewriter_logic(user_msg, last_rag_question, last_rag_answer)
            span.set_attribute("dge.rewriter.is_follow_up", rew.is_follow_up)
            span.set_attribute("dge.rewriter.reason", rew.reason)
            span.set_attribute("dge.rewriter.rewritten_question", rew.rewritten_question)

            if rew.is_follow_up:
                question_for_rag = rew.rewritten_question
            else:
                # Rewriter decided it's not really a follow-up; treat as direct RAG query
                question_for_rag = user_msg
        else:
            # No previous RAG context; treat as direct RAG
            question_for_rag = user_msg

    # 5) Direct RAG route -> RAG only (no rewriter)
    if router.route == "rag":
        question_for_rag = user_msg

    # If router.route is "rag" or "follow_up", run the RAG pipeline
    if router.route in ("rag", "follow_up"):
        # Run RAG
        rag = rag_logic(question_for_rag)
        answer_text = rag["answer"]

        # Validate the answer
        validator = validator_logic(
            question_for_rag,
            json.dumps(rag["sources"]),
            answer_text,
        )
        span.set_attribute("dge.validator.is_valid", validator.is_valid)
        span.set_attribute("dge.validator.feedback", validator.feedback)

        final_answer = answer_text
        if not validator.is_valid:
            final_answer += f"\n\n[Validator note] {validator.feedback}"

        if rag["sources"]:
            src_lines = [
                f"- {s.get('file_name', 'Unknown')} (score={s.get('score',0.0):.3f})"
                for s in rag["sources"]
            ]
            final_answer += "\n\nSources:\n" + "\n".join(src_lines)

        span.set_attribute("dge.final_answer", final_answer)

        # >>> RAGAS LOGGING: only for RAG-related routes <<<
        append_ragas_example(
            question=question_for_rag,
            answer=final_answer,
            sources=rag["sources"],
        )

        update_memory(user_msg, final_answer)

        # Update last RAG Q&A for future follow-up detection
        last_rag_question = question_for_rag
        last_rag_answer = final_answer

        return final_answer

    # 6) route == "other" -> generic domain message
    assistant_answer = (
        "I mainly help with Abu Dhabi DGE procurement policies and processes. "
        "Please ask a procurement-related question."
    )
    span.set_attribute("dge.final_answer", assistant_answer)
    update_memory(user_msg, assistant_answer)
    # not logged to RAGAS
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