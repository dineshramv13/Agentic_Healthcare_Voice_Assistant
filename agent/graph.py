"""
agent/graph.py

THE CORE FILE. Defines the LangGraph state machine wiring every node
together via agent/router.py's conditional edges.

Graph shape:

    safety_check -> intent_classify -> [router] -> emergency -> END
                                              |
                                              +--> fallback -> END
                                              |
                                              +--> retriever -> generator -> verifier -> [router]
                                                       ^                                    |
                                                       |____________(retry, max 2)__________|
                                                                                             |
                                                                                       (verified) -> END
                                                                                       (exhausted) -> fallback -> END

Input:  AgentState (at minimum: user_message, session_id, trace_id)
Output: AgentState with final_response populated

Usage:
    from agent.graph import build_graph
    graph = build_graph()
    result = graph.invoke({
        "user_message": "I need to book an appointment",
        "session_id": "abc123",
        "trace_id": "trace-001",
        "retry_count": 0,
    })
    print(result["final_response"])
"""

import logging
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.router import route_after_safety_and_intent, route_after_verifier
from agent.nodes.safety import SafetyNode
from agent.nodes.intent import IntentClassifierNode
from agent.nodes.retriever import RetrieverNode
from agent.nodes.generator import GeneratorNode
from agent.nodes.verifier import VerifierNode
from agent.nodes.emergency import EmergencyNode
from agent.nodes.fallback import FallbackNode

logger = logging.getLogger(__name__)


def _set_final_response(state: AgentState) -> dict:
    """
    Tiny terminal node: copies `response` into `final_response` for the
    success path (END after verified=True). The emergency and fallback
    nodes already set final_response themselves, so this only fires on
    the "verified" success path.
    """
    return {"final_response": state.get("response")}


def build_graph(
    safety_node: SafetyNode | None = None,
    intent_node: IntentClassifierNode | None = None,
    retriever_node: RetrieverNode | None = None,
    generator_node: GeneratorNode | None = None,
    verifier_node: VerifierNode | None = None,
    emergency_node: EmergencyNode | None = None,
    fallback_node: FallbackNode | None = None,
):
    """
    Builds and compiles the LangGraph state machine.

    All node instances are optional constructor args so tests (Phase 5)
    can inject mocked nodes without touching this file — this function
    just wires whatever node instances it's given (or defaults).
    """
    safety_node = safety_node or SafetyNode()
    intent_node = intent_node or IntentClassifierNode()
    retriever_node = retriever_node or RetrieverNode()
    generator_node = generator_node or GeneratorNode()
    verifier_node = verifier_node or VerifierNode()
    emergency_node = emergency_node or EmergencyNode()
    fallback_node = fallback_node or FallbackNode()

    graph = StateGraph(AgentState)

    # --- Register nodes ---
    graph.add_node("safety_check", safety_node)
    graph.add_node("intent_classify", intent_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", _set_final_response)

    # --- Entry point ---
    graph.set_entry_point("safety_check")

    # --- Linear edges ---
    graph.add_edge("safety_check", "intent_classify")

    # --- Conditional edge #1: after intent classification ---
    graph.add_conditional_edges(
        "intent_classify",
        route_after_safety_and_intent,
        {
            "emergency": "emergency",
            "fallback": "fallback",
            "retriever": "retriever",
        },
    )

    # --- RAG path linear edges ---
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "verifier")

    # --- Conditional edge #2: after verification (the self-correction loop) ---
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "end": "finalize",
            "retriever": "retriever",   # retry loop
            "fallback": "fallback",     # retries exhausted
        },
    )

    # --- Terminal edges ---
    graph.add_edge("emergency", END)
    graph.add_edge("fallback", END)
    graph.add_edge("finalize", END)

    compiled = graph.compile()
    logger.info("LangGraph agent graph compiled successfully")
    return compiled


# Module-level singleton, built lazily on first import use.
# api/server.py (Phase 3) will import and call build_graph() once at startup.
_graph_instance = None


def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance
