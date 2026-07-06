"""
agent/nodes/generator.py

GeneratorNode: builds the final prompt from the prompt registry (system
prompt v1 or v2), injecting retrieved context + conversation history +
the user's question, then calls the LLM to produce a response.

Also triggers the actionable tools (AppointmentTool for booking-flavored
appointment intents) — informational answers always go through the
RAG-grounded path regardless of intent, but appointment intent ALSO logs
a simulated booking action so it's visible in traces.

Input:  state (intent, retrieved_chunks, conversation_history, user_message)
Output: state updates: response, prompt_version_used
"""

import logging
from typing import Dict

from agent.state import AgentState
from llm.client import LLMClient
from prompts_manager.registry import prompt_registry
from tools.appointment_tool import AppointmentTool
from observability.tracer import traced

logger = logging.getLogger(__name__)

# Which system_prompt version is active by default. Change this single
# constant to "v2" to A/B test — the eval runner (Phase 5) can run the
# golden set against both and compare RAGAS scores.
ACTIVE_SYSTEM_PROMPT_VERSION = "v1"


def _format_chunks_as_context(chunks) -> str:
    if not chunks:
        return "(No relevant information was found in the knowledge base for this question.)"
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)


class GeneratorNode:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        appointment_tool: AppointmentTool | None = None,
        prompt_version: str = ACTIVE_SYSTEM_PROMPT_VERSION,
    ):
        self.llm_client = llm_client or LLMClient()
        self.appointment_tool = appointment_tool or AppointmentTool()
        self.prompt_version = prompt_version

    @traced("generator")
    def __call__(self, state: AgentState) -> Dict:
        context = _format_chunks_as_context(state.get("retrieved_chunks", []))
        history = state.get("conversation_history", "") or "(no prior turns)"
        question = state.get("user_message", "")

        prompt = prompt_registry.format(
            "system_prompt",
            self.prompt_version,
            context=context,
            history=history,
            question=question,
        )

        try:
            response = self.llm_client.generate(prompt)
        except Exception as e:
            logger.error("GeneratorNode LLM call failed: %s", e)
            response = (
                "I'm having trouble generating a response right now. "
                "Please contact the surgery directly, or try again shortly."
            )

        # Appointment intent also triggers the simulated booking tool, purely
        # for audit/demo purposes — the spoken/text response itself still
        # comes from the grounded generation above.
        if state.get("intent") == "appointment":
            self.appointment_tool.simulate_booking(
                patient_message=state.get("user_message", ""),
                session_id=state.get("session_id", "unknown"),
            )

        logger.info("GeneratorNode produced response using prompt version '%s'", self.prompt_version)

        return {
            "response": response,
            "prompt_version_used": self.prompt_version,
        }
