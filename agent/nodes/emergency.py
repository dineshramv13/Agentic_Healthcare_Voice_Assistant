"""
agent/nodes/emergency.py

EmergencyNode: NO LLM call. Hardcoded, deterministic safety response.
This is intentional and important for clinical safety — an emergency
redirect must never depend on an LLM being available, correctly prompted,
or behaving as expected. It's also the fastest possible path (no network
call to OpenRouter at all), which matters when seconds count.

Input:  state["user_message"], state["session_id"]
Output: state updates: response, final_response
"""

import logging
from typing import Dict

from agent.state import AgentState
from tools.escalation_tool import EscalationTool
from observability.tracer import traced

logger = logging.getLogger(__name__)

HARDCODED_EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please call 999 "
    "(or 112) right now, or go to your nearest A&E immediately. If you are "
    "with someone else, ask them to call for you. Stay on the line with the "
    "999 operator and follow their instructions. This assistant cannot help "
    "with emergencies — please contact emergency services now."
)


class EmergencyNode:
    def __init__(self, escalation_tool: EscalationTool | None = None):
        self.escalation_tool = escalation_tool or EscalationTool()

    @traced("emergency")
    def __call__(self, state: AgentState) -> Dict:
        self.escalation_tool.log_escalation(
            user_message=state.get("user_message", ""),
            session_id=state.get("session_id", "unknown"),
        )
        logger.warning("EmergencyNode fired — hardcoded 999 redirect, no LLM involved.")

        return {
            "response": HARDCODED_EMERGENCY_RESPONSE,
            "final_response": HARDCODED_EMERGENCY_RESPONSE,
            "verified": True,  # hardcoded response is trivially "grounded" — skip verifier
        }
