"""
agent/router.py

Pure Python routing functions — no LLM calls. These are LangGraph
"conditional edge" functions: given the current state, return the name
of the next node to execute.

Three routing decision points in this graph:
    1. route_after_safety_and_intent — decides emergency / out_of_scope / RAG path
    2. route_after_verifier          — decides END / retry-via-retriever / fallback

Both are pure functions of state, easy to unit test in isolation
(see tests/test_agent.py in Phase 5) without any mocking of LLM calls.
"""

import logging
from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # must match VerifierNode.max_retries — see agent/nodes/verifier.py


def route_after_safety_and_intent(state: AgentState) -> str:
    """
    Decision point after safety_check -> intent_classify have both run.

    Emergency routing uses OR logic across two independent signals:
        - SafetyNode's fast keyword scan (is_emergency_flagged)
        - IntentClassifierNode's LLM-based classification (intent == "emergency")
    Either one firing is sufficient to route to the emergency node. This
    redundancy is deliberate: a keyword scan can miss phrasing it doesn't
    recognize, and an LLM classifier can occasionally mis-classify under
    rare/adversarial inputs. Two independent, differently-failing signals
    catch more real emergencies than either alone.

    Unsafe input (prompt injection detected) is also redirected to fallback
    rather than ever reaching the generator/LLM with attacker-controlled text.
    """
    if state.get("is_emergency_flagged") or state.get("intent") == "emergency":
        return "emergency"

    if not state.get("is_safe", True):
        logger.warning("Routing unsafe input to fallback (reason: %s)", state.get("safety_reason"))
        return "fallback"

    if state.get("intent") == "out_of_scope":
        return "fallback"

    # appointment, prescription, info all go through the same RAG path —
    # they differ only in which tool the generator invokes afterward
    # (see agent/nodes/generator.py's appointment-intent branch).
    return "retriever"


def route_after_verifier(state: AgentState) -> str:
    """
    Decision point after the verifier node runs.

    - verified=True              -> END (return the response as-is)
    - verified=False, retries left -> back to retriever (retry_count already
      incremented by VerifierNode; transformed_query already modified)
    - verified=False, retries exhausted -> fallback (honest "couldn't verify"
      message + human callback, rather than ever showing a possibly-
      hallucinated answer to the patient)
    """
    verified = state.get("verified", True)
    retry_count = state.get("retry_count", 0)

    if verified:
        return "end"

    if retry_count < MAX_RETRIES:
        return "retriever"

    logger.warning("Verifier retries exhausted (retry_count=%d) — routing to fallback", retry_count)
    return "fallback"
