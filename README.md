# AI-Local

A local, zero-cost healthcare voice reception agent — RAG + LangGraph agent +
voice pipeline + evaluation harness. Built to mirror a production AI
receptionist system, as a hands-on project.

> **Build status: ** All 5 phases — RAG, LLM agent, memory +
> observability + API, voice pipeline, evaluation harness + tests.

---

## What this project is

AI-Local is a healthcare practice receptionist assistant that:
- Takes patient input via **text or voice** (local Whisper STT)
- Classifies intent (appointment / prescription / emergency / info / out-of-scope)
- Retrieves relevant policy information using **hybrid RAG** (BM25 + dense vectors + RRF fusion + cross-encoder reranking, with HyDE query transformation)
- Generates a grounded response via a free LLM (OpenRouter)
- **Verifies its own response isn't hallucinated** before showing it to the patient, retrying up to twice if not
- **Hardcodes emergency redirects** — no LLM involved, instant, deterministic
- Remembers conversation history across turns (SQLite)
- Logs full observability traces (JSONL, LangSmith-style) for every node execution
- Speaks responses aloud (local TTS) and can be evaluated against a 50-query golden set with RAGAS-style metrics

Zero cloud services. Zero paid APIs. Runs entirely on your machine once set up.

---

## Architecture at a glance

```
User (text or voice)
        ↓
  Safety Node (injection detection + emergency keyword scan — no LLM)
        ↓
  Intent Classifier (few-shot LLM call)
        ↓
     Router
   ├── emergency  → hardcoded 999 redirect (no LLM) → END
   ├── out_of_scope / unsafe → fallback → END
   └── appointment / prescription / info →
            Retriever (HyDE + hybrid RAG: BM25 + dense + RRF + rerank)
                ↓
            Generator (grounded LLM response, versioned prompt)
                ↓
            Verifier (second LLM call: is this grounded in context?)
                ├── verified        → END
                ├── unverified, retries left → back to Retriever (max 2)
                └── retries exhausted → fallback (human callback) → END
```

Every node writes a trace record; every turn is saved to session memory;
the exact same graph powers both the text API and the voice pipeline.

---

## Project structure (complete)

```
AI-local/
├── README.md
├── requirements.txt
├── .env.example
├── conftest.py
│
├── config/
│   └── settings.py              — central Pydantic config
│
├── docs/                        — NHS-style knowledge base (RAG source documents)
│   ├── appointment_policy.md
│   ├── prescriptions.md
│   ├── surgery_info.md
│   ├── emergencies.md
│   └── services.md
│
├── rag/
│   ├── embeddings.py             — local sentence-transformers embedding model
│   ├── ingestion.py               — chunking + ChromaDB ingestion
│   ├── retriever.py               — hybrid retrieval: BM25 + dense + RRF + rerank
│   └── query_transform.py         — HyDE query transformation
│
├── llm/
│   └── client.py                  — OpenRouter client, retry + fallback model
│
├── prompts/
│   └── registry.yaml              — every prompt, versioned (v1/v2 A/B-test ready)
├── prompts_manager/
│   └── registry.py                — loads & formats prompts by name+version
│
├── agent/
│   ├── state.py                   — shared AgentState TypedDict
│   ├── graph.py                   — THE CORE FILE: LangGraph state machine
│   ├── router.py                  — pure-Python conditional routing logic
│   └── nodes/
│       ├── safety.py              — injection detection + emergency keyword scan
│       ├── intent.py              — few-shot LLM intent classification
│       ├── retriever.py           — calls FAQTool (HyDE + hybrid retrieval)
│       ├── generator.py           — grounded prompt + LLM call + appointment tool trigger
│       ├── verifier.py            — self-corrective grounding check + retry loop
│       ├── emergency.py           — hardcoded, no-LLM 999 redirect
│       └── fallback.py            — graceful out-of-scope / exhausted-retry handling
│
├── tools/
│   ├── faq_tool.py                — wraps RAG retrieval as a named tool
│   ├── appointment_tool.py        — simulated booking action
│   ├── callback_tool.py           — simulated human-callback request
│   └── escalation_tool.py         — emergency escalation audit logging
│
├── memory/
│   └── session.py                 — SQLite multi-turn conversation history
│
├── observability/
│   └── tracer.py                  — JSONL trace logger + @traced decorator (LangSmith-style)
│
├── api/
│   └── server.py                  — FastAPI: /chat, /voice, /session, /health, /traces
│
├── voice/
│   ├── stt.py                     — local Whisper speech-to-text
│   ├── tts.py                     — local pyttsx3 text-to-speech
│   ├── vad.py                     — energy-threshold voice activity detection
│   └── pipeline.py                — orchestrates VAD → STT → agent graph → TTS
│
├── eval/
│   ├── golden_set.json            — 50 patient queries with expected intent/themes
│   ├── metrics.py                 — RAGAS-style metrics (faithfulness, relevance, precision) + LLM-as-judge
│   └── run_eval.py                — eval runner, supports A/B testing prompt versions
│
├── tests/
│   ├── test_safety.py             — injection detection + emergency keyword tests
│   ├── test_agent.py              — routing logic tests (the retry loop, emergency priority)
│   ├── test_rag.py                — RRF fusion + tokenizer + chunking tests
│   ├── test_voice.py               — STT/TTS tests with mocked audio dependencies
│   ├── test_prompts.py             — PromptRegistry tests
│   └── test_eval_metrics.py        — eval metric-parsing tests
│
├── scripts/
│   ├── ingest.py                   — populate ChromaDB from docs/
│   └── demo.py                     — interactive CLI demo (--mode text|voice)
│
├── chroma_db/          ← created by scripts/ingest.py
├── traces/             ← created automatically, JSONL trace files
├── eval_results/       ← created by eval/run_eval.py
└── sessions.db         ← created automatically on first /chat, /voice, or demo.py call
```

---

## Full setup guide (from scratch)

### 1. Check your Python version

```bash
python3 --version
```

Use **Python 3.10, 3.11, or 3.12**. Whisper (Phase 4's voice pipeline) does not support Python 3.13+ at the time of writing.

### 2. System-level dependencies (install these BEFORE pip install)

These are the only non-Python dependencies in the whole project, all needed for voice:

```bash
# ffmpeg — required by Whisper to decode audio
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg portaudio19-dev espeak
# macOS:
brew install ffmpeg portaudio
# Windows:
choco install ffmpeg
# (PortAudio is bundled in the sounddevice wheel on Windows — no extra step)
```

If you only want **text chat** (skip voice entirely), you can skip this step — everything else works without it.

### 3. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything across all 5 phases in one go. First run of various components will download (once, then cached and fully offline forever after):
- `sentence-transformers/all-MiniLM-L6-v2` (~90MB) — embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~135MB) — reranker
- Whisper `base` model (~150MB) — speech-to-text

### 5. Set up environment variables

```bash
cp .env.example .env
```

Get a **free** OpenRouter API key (no card required):
1. Go to **https://openrouter.ai/keys**
2. Sign up, create a key
3. In `.env`, set: `OPENROUTER_API_KEY=sk-or-v1-your-key-here`

Default model is `mistralai/mistral-7b-instruct:free`. If rate-limited, the client auto-falls-back to `meta-llama/llama-3.1-8b-instruct:free`, or change `MODEL_NAME` in `.env`.

### 6. Populate the knowledge base

```bash
python scripts/ingest.py
```

Expected output: `Ingested <N> chunks into ChromaDB at './chroma_db'.`

### 7. Run the test suite

```bash
pytest tests/ -v
```

These tests use mocking for external dependencies (LLM calls, audio hardware) where needed, so they run fast and free — no API calls, no microphone required.

### 8. Run the evaluation harness

```bash
python eval/run_eval.py
```

Runs all 50 golden-set queries through the full pipeline, scores each with RAGAS-style metrics (faithfulness, answer relevance, context precision) plus an LLM-as-judge quality score, and prints a regression summary. Takes a few minutes (each query makes 2-5 LLM calls). For a quick smoke test first:

```bash
python eval/run_eval.py --limit 5
```

To A/B test the two system prompt versions:

```bash
python eval/run_eval.py --prompt-version v1
python eval/run_eval.py --prompt-version v2
```

Compare the two `eval_results/report_*.json` files.

### 9. Try the text demo

```bash
python scripts/demo.py --mode text
```

### 10. Try the voice demo (needs system deps from step 2)

```bash
python scripts/demo.py --mode voice
```

Speak after "Listening..." appears. Try:
- *"I need to book an appointment"*
- *"How do I get a repeat prescription?"*
- *"I think I'm having a heart attack"* — should respond almost instantly (no LLM call), spoken aloud

### 11. Run the full API server

```bash
uvicorn api.server:app --reload --port 8000
```

Open **http://127.0.0.1:8000/docs** — interactive Swagger UI, generated automatically from `api/server.py`. Try `POST /chat`, then `POST /voice` (upload a short audio file), `GET /session/{id}`, `GET /traces`, `GET /health`.

---

## A note on RAGAS

We use **RAGAS-style metrics implemented ourselves** (`eval/metrics.py`), not the `ragas` pip package. The real package defaults to needing an OpenAI key for its judge model and has a habit of pulling in conflicting dependency versions. Our version computes the same three metrics — faithfulness, answer relevance, context precision — using our own free OpenRouter LLM as judge.

## A note on LangSmith

We use a self-built `observability/tracer.py` instead of the real LangSmith service. It mimics LangSmith's trace structure (node name, timing, input, output, written as JSONL) without needing an external account or API key.
## Known, intentional trade-offs 

- **`/voice`'s transcription call blocks FastAPI's event loop** (CPU-bound Whisper inference inside an `async def` route). Fine at single-user demo scale; production fix is a thread pool or background worker.
- **VAD is energy-threshold based**, not ML-based (Silero/WebRTC VAD). Good enough to demo, clearly explainable, with a known upgrade path.
- **pyttsx3 sounds robotic** compared to cloud TTS. Free upgrade path is `edge-tts` (free, but requires network — not used here to preserve "zero network at runtime").
- **Appointment booking is simulated**, not connected to a real scheduling system — this project's scope is the AI layer, not backend integration; the swap-in point (`tools/appointment_tool.py`) is clearly isolated.

---

## Q&A map (the short version)

Every file in this repo answers at least one likely question. The pattern for any answer: 

| Topic | Key files |
|---|---|
| Voice agent pipeline | `voice/stt.py`, `voice/tts.py`, `voice/vad.py`, `voice/pipeline.py` |
| RAG / hybrid retrieval | `rag/ingestion.py`, `rag/retriever.py`, `rag/embeddings.py`, `rag/query_transform.py` |
| LLM integration & cost control | `llm/client.py`, `config/settings.py` |
| Prompt engineering & injection defense | `prompts/registry.yaml`, `prompts_manager/registry.py`, `agent/nodes/safety.py` |
| Agentic AI / self-correction | `agent/graph.py`, `agent/router.py`, `agent/nodes/verifier.py` |
| Tool calling | `tools/faq_tool.py`, `tools/appointment_tool.py`, `tools/callback_tool.py`, `tools/escalation_tool.py` |
| Multi-turn memory | `memory/session.py` |
| Observability | `observability/tracer.py` |
| Evaluation | `eval/golden_set.json`, `eval/metrics.py`, `eval/run_eval.py` |
| System design / API | `api/server.py` |
| Testing | `tests/*.py` |

---

## Cost summary

**$0.** Everything runs on free tiers or local/open-source software:
- LLM calls: OpenRouter free models (`:free` suffix), no card on file
- Embeddings + reranker: local sentence-transformers models
- Vector store: local ChromaDB
- STT: local Whisper
- TTS: local pyttsx3
- Memory: local SQLite
- Observability: local JSONL files
- Evaluation judge: same free OpenRouter LLM, no separate judge API
