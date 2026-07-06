"""
agent/nodes/retriever.py

RetrieverNode: calls FAQTool (HyDE + hybrid retrieval) and stores the
resulting chunks in state. Handles the empty-retrieval case explicitly —
if nothing relevant is found, the generator node still runs but receives
an empty context, and its prompt instructs it to say so honestly rather
than hallucinate.

Input:  state["user_message"] (first attempt) or state["transformed_query"]
        (on a verifier-triggered retry — see agent/graph.py retry edge)
Output: state updates: retrieved_chunks, transformed_query
"""

import logging
from typing import Dict

from agent.state import AgentState
from tools.faq_tool import FAQTool
from observability.tracer import traced

logger = logging.getLogger(__name__)


class RetrieverNode:
    def __init__(self, faq_tool: FAQTool | None = None):
        self.faq_tool = faq_tool or FAQTool()

    @traced("retriever")
    def __call__(self, state: AgentState) -> Dict:
        # On a retry (verifier said "unsupported"), the verifier node will
        # have already set a modified transformed_query in state — prefer
        # that over re-deriving from the raw user_message.
        query = state.get("transformed_query") or state.get("user_message", "")

        chunks = self.faq_tool.run(query)

        if not chunks:
            logger.warning(
                "RetrieverNode found no chunks for query '%s' — generator will "
                "receive empty context and should say it doesn't have the info.",
                query[:60],
            )

        return {
            "retrieved_chunks": chunks,
            "transformed_query": query,
        }
