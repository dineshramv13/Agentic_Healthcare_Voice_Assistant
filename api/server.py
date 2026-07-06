"""
api/server.py

FastAPI application exposing the agent over HTTP.

Endpoints:
    POST /chat            — text input -> agent response (the main endpoint)
    POST /voice            — audio file upload -> transcribed text + agent response
    GET  /session/{id}    — view a session's conversation history
    DELETE /session/{id}  — clear a session's history
    GET  /health          — system health check (checks LLM key present, ChromaDB reachable)
    GET  /traces          — last N trace records (observability)

Run with:
    uvicorn api.server:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for the auto-generated Swagger UI —
every endpoint is testable interactively from the browser, no extra
frontend code needed. A future frontend (Streamlit or otherwise) would
call these same JSON endpoints.

Note on /voice: this endpoint accepts an uploaded audio FILE (not a live
microphone stream) — the server has no microphone of its own. For an
interactive live-microphone demo, use scripts/demo.py instead, which uses
voice/pipeline.py's listen_and_respond() directly on your local machine's mic.
"""

import logging
import os
import tempfile
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from config.settings import settings
from agent.graph import get_graph
from memory.session import SessionMemory
from observability.tracer import tracer

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Local",
    description="Local, zero-cost healthcare voice reception agent — RAG + LangGraph + voice.",
    version="0.1.0",
)

# Module-level singletons — built once at server startup, reused across requests.
_graph = None
_memory = None
_voice_pipeline = None


def get_memory() -> SessionMemory:
    global _memory
    if _memory is None:
        _memory = SessionMemory()
    return _memory


def get_compiled_graph():
    global _graph
    if _graph is None:
        _graph = get_graph()
    return _graph


def get_voice_pipeline():
    """
    Lazily builds the VoicePipeline (which loads Whisper + pyttsx3) only
    when /voice is actually called — so a server running text-chat-only
    never pays the Whisper model load cost, and never requires
    openai-whisper/pyttsx3/sounddevice to be installed unless /voice is used.
    """
    global _voice_pipeline
    if _voice_pipeline is None:
        try:
            from voice.pipeline import VoicePipeline
            _voice_pipeline = VoicePipeline(memory=get_memory())
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Voice dependencies are not installed. Run "
                    "'pip install openai-whisper pyttsx3 sounddevice' and ensure "
                    f"ffmpeg is installed on your system. Original error: {e}"
                ),
            )
    return _voice_pipeline


# --- Request / Response schemas ---

class ChatRequest(BaseModel):
    message: str = Field(..., description="The patient's message", min_length=1)
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn memory. Omit to start a new session.",
    )


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    intent: Optional[str] = None
    response: str
    verified: Optional[bool] = None
    retry_count: int


class VoiceResponse(BaseModel):
    session_id: str
    transcribed_text: str
    response: str
    intent: Optional[str] = None
    verified: Optional[bool] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: list


class HealthResponse(BaseModel):
    status: str
    llm_key_configured: bool
    chroma_collection_populated: bool
    details: dict


# --- Routes ---

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """
    Main conversational endpoint. Runs the full LangGraph pipeline:
    safety -> intent -> router -> (retriever -> generator -> verifier) | emergency | fallback

    Loads prior conversation history for the session (if any), invokes the
    graph, then persists both the user's message and the assistant's reply
    back to session memory.
    """
    session_id = request.session_id or str(uuid.uuid4())
    trace_id = str(uuid.uuid4())

    memory = get_memory()
    graph = get_compiled_graph()

    history_text = memory.get_history_as_text(session_id, last_n=6)

    initial_state = {
        "user_message": request.message,
        "session_id": session_id,
        "trace_id": trace_id,
        "conversation_history": history_text,
        "retry_count": 0,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        logger.error("Graph invocation failed for session '%s': %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {e}")

    final_response = result.get("final_response") or result.get("response") or (
        "Sorry, something went wrong and I couldn't generate a response."
    )

    # Persist this turn to memory (user message first, then assistant reply,
    # matching add_turn's expectation that the user row is added first so
    # turn_number increments correctly — see memory/session.py)
    memory.add_turn(session_id, role="user", content=request.message, intent=result.get("intent"))
    memory.add_turn(session_id, role="assistant", content=final_response)

    return ChatResponse(
        session_id=session_id,
        trace_id=trace_id,
        intent=result.get("intent"),
        response=final_response,
        verified=result.get("verified"),
        retry_count=result.get("retry_count", 0),
    )


@app.post("/voice", response_model=VoiceResponse, tags=["chat"])
async def voice(
    audio_file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/etc.) containing the patient's spoken message"),
    session_id: Optional[str] = Form(default=None, description="Session ID for multi-turn memory. Omit to start a new session."),
):
    """
    Voice endpoint: accepts an uploaded audio file, transcribes it locally
    with Whisper, runs the SAME agent graph used by /chat, and returns the
    text response (does not return synthesized audio — see voice/tts.py if
    you want the response spoken back; that's done client-side or in the
    interactive CLI demo, see scripts/demo.py).

    Requires voice dependencies (openai-whisper, ffmpeg) to be installed —
    see README Phase 4 setup instructions. Returns 503 with install
    instructions if they're missing, rather than a raw stack trace.
    """
    pipeline = get_voice_pipeline()  # raises HTTPException(503) if deps missing
    session_id = session_id or str(uuid.uuid4())

    # Whisper needs a real file path (or numpy array) to read from — write
    # the uploaded bytes to a temp file, since UploadFile gives us a stream.
    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        contents = await audio_file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # NOTE: transcribe_file_to_response() is a synchronous, CPU-bound call
    # (Whisper inference) and will block the event loop for its duration.
    # Fine at this project's single-user demo scale; in production this
    # would run in a thread pool (e.g. FastAPI's `run_in_threadpool`) or a
    # background worker so concurrent requests aren't blocked on it.
    try:
        result = pipeline.transcribe_file_to_response(tmp_path, session_id=session_id)
    except Exception as e:
        logger.error("Voice pipeline failed for session '%s': %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Voice pipeline failed: {e}")
    finally:
        os.unlink(tmp_path)  # always clean up the temp file, even on failure

    return VoiceResponse(
        session_id=session_id,
        transcribed_text=result["transcribed_text"],
        response=result["response_text"],
        intent=result.get("intent"),
        verified=result.get("verified"),
    )


@app.get("/session/{session_id}", response_model=SessionHistoryResponse, tags=["session"])
def get_session(session_id: str):
    """Returns the full conversation history for a session."""
    memory = get_memory()
    if not memory.session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    turns = memory.get_history(session_id, last_n=1000)
    return SessionHistoryResponse(session_id=session_id, turns=turns)


@app.delete("/session/{session_id}", tags=["session"])
def delete_session(session_id: str):
    """Clears a session's conversation history."""
    memory = get_memory()
    deleted = memory.clear_session(session_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"session_id": session_id, "rows_deleted": deleted}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """
    Health check. Verifies the LLM API key is configured and ChromaDB has
    been populated (i.e. scripts/ingest.py has been run).
    """
    llm_key_configured = bool(settings.openrouter_api_key)

    chroma_populated = False
    chroma_error = None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection = client.get_or_create_collection(settings.chroma_collection_name)
        chroma_populated = collection.count() > 0
    except Exception as e:
        chroma_error = str(e)

    status = "ok" if (llm_key_configured and chroma_populated) else "degraded"

    return HealthResponse(
        status=status,
        llm_key_configured=llm_key_configured,
        chroma_collection_populated=chroma_populated,
        details={
            "chroma_error": chroma_error,
            "model_name": settings.model_name,
            "note": "Run 'python scripts/ingest.py' if chroma_collection_populated is false.",
        },
    )


@app.get("/traces", tags=["system"])
def get_traces(n: int = 20, trace_id: Optional[str] = None):
    """Returns the last N trace records (or all records for a specific trace_id)."""
    records = tracer.read_traces(n=n, trace_id=trace_id)
    return {"count": len(records), "traces": records}
