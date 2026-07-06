"""
agent/nodes/intent.py

IntentClassifierNode: classifies the user's message into one of five
intents using a few-shot LLM prompt (intent_classifier_prompt in the
registry). This determines which branch the router sends the state to.

Input:  state["user_message"]
Output: state update: intent

Note: this node ALSO re-checks for emergency via the LLM (the prompt
includes emergency as a class), independent of SafetyNode's keyword scan.
The router (agent/router.py) treats EITHER signal as sufficient to route
to the emergency node — see router.py for that OR logic.
"""

import logging
from typing import Dict

from agent.state import AgentState
from llm.client import LLMClient
from prompts_manager.registry import prompt_registry
from observability.tracer import traced

logger = logging.getLogger(__name__)

VALID_INTENTS = {"appointment", "prescription", "emergency", "info", "out_of_scope"}


class IntentClassifierNode:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    @traced("intent_classify")
    def __call__(self, state: AgentState) -> Dict:
        message = state.get("user_message", "")

        prompt = prompt_registry.format("intent_classifier_prompt", "v1", message=message)

        try:
            raw_output = self.llm_client.generate(prompt, temperature=0.0)
            intent = raw_output.strip().lower()
            # Defensive cleanup: LLM sometimes wraps the label in punctuation/quotes
            intent = intent.strip(' .\n"\'')
        except Exception as e:
            logger.error("Intent classification LLM call failed (%s); defaulting to 'info'", e)
            intent = "info"

        if intent not in VALID_INTENTS:
            logger.warning(
                "Intent classifier returned unrecognized label '%s'; defaulting to 'info'", intent
            )
            intent = "info"

        logger.info("Classified intent: '%s' for message: '%s'", intent, message[:60])
        return {"intent": intent}
