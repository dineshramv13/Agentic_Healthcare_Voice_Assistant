"""
agent/nodes/fallback.py

FallbackNode: handles two distinct cases gracefully, without an LLM call:
    1. "out_of_scope" intent — a polite, hardcoded redirect.
    2. Verification exhausted retries and still failed — rather than risk
       showing a potentially unsupported/hallucinated answer to the
       patient, this node requests a human callback and tells the patient
       honestly that it couldn't fully verify the answer.

Input:  state["intent"], state["verified"], state["session_id"]
Output: state updates: response, final_response
"""

import logging
from typing import Dict

from agent.state import AgentState
from tools.callback_tool import CallbackTool
from observability.tracer import traced

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_RESPONSE = (
    "I'm AI, the practice's assistant, and I can only help with questions "
    "about appointments, prescriptions, and surgery services. For anything "
    "else, I'm afraid I can't help — is there something practice-related I "
    "can assist with instead?"
)

VERIFICATION_FAILED_RESPONSE = (
    "I want to make sure I give you accurate information, and I wasn't able "
    "to fully confirm my answer against our records on this occasion. "
    "I've asked a member of the team to call you back to make sure you get "
    "the right answer. In the meantime, you're welcome to call the surgery "
    "directly."
)


class FallbackNode:
    def __init__(self, callback_tool: CallbackTool | None = None):
        self.callback_tool = callback_tool or CallbackTool()

    @traced("fallback")
    def __call__(self, state: AgentState) -> Dict:
        intent = state.get("intent")
        verified = state.get("verified")

        if intent == "out_of_scope":
            logger.info("FallbackNode handling out_of_scope intent")
            response = OUT_OF_SCOPE_RESPONSE
        elif verified is False:
            logger.warning(
                "FallbackNode handling verification-exhausted case for session '%s'",
                state.get("session_id", "unknown"),
            )
            self.callback_tool.request_callback(
                reason="Grounding verification failed after max retries",
                session_id=state.get("session_id", "unknown"),
            )
            response = VERIFICATION_FAILED_RESPONSE
        else:
            # Generic safety net — should rarely be hit, but ensures the
            # graph never ends without SOME response if routing logic
            # sends an unexpected state here in the future.
            logger.warning("FallbackNode hit generic case — intent=%s, verified=%s", intent, verified)
            response = (
                "I'm sorry, I wasn't able to process that properly. "
                "Please contact the surgery directly for assistance."
            )

        return {
            "response": response,
            "final_response": response,
        }
