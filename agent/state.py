"""
agent/state.py

AgentState: the shared state object that flows through every node in the
LangGraph graph (agent/graph.py). Each node reads what it needs from state
and returns a dict of fields to update — LangGraph merges these automatically.

This is the single contract every node file in agent/nodes/ must respect.
If you add a new field here, every node that should populate or read it
must be updated — this is the most common source of mismatches across a
phased build, so this file should be treated as load-bearing and not
casually edited later without checking every node.
"""

from typing import TypedDict, List, Dict, Optional, Literal

IntentLabel = Literal["appointment", "prescription", "emergency", "info", "out_of_scope"]


class RetrievedChunk(TypedDict):
    text: str
    source: str
    score: float


class AgentState(TypedDict, total=False):
    # --- Input ---
    session_id: str
    trace_id: str
    user_message: str          # raw input for this turn (text, or transcribed voice)

    # --- Safety node output ---
    is_safe: bool
    safety_reason: Optional[str]
    is_emergency_flagged: bool  # quick keyword/pattern flag, separate from intent classifier

    # --- Intent node output ---
    intent: Optional[IntentLabel]

    # --- Query transform + retrieval ---
    transformed_query: Optional[str]   # HyDE output, or raw query if HyDE skipped/failed
    retrieved_chunks: List[RetrievedChunk]

    # --- Memory ---
    conversation_history: str   # formatted last-N-turns string injected into prompts

    # --- Generation ---
    prompt_version_used: Optional[str]  # which system_prompt version (v1/v2) was used
    response: Optional[str]

    # --- Verification / self-correction ---
    verified: Optional[bool]
    verification_reason: Optional[str]
    retry_count: int

    # --- Routing / control flow ---
    next_node: Optional[str]
    used_fallback: bool   # True if the fallback node had to handle this turn

    # --- Final output (set right before END) ---
    final_response: Optional[str]
