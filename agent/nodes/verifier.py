"""
agent/nodes/verifier.py

VerifierNode: a second LLM call that checks whether the generator's
response is actually grounded in the retrieved context (hallucination
mitigation). If not, it triggers a retry: modifies the query slightly and
sends control back to the retriever node, up to max_retries (enforced via
state["retry_count"], checked here AND in the router's conditional edge).

This is the "self-corrective agent" feature — the single most impressive piece of this project's design.

Input:  state["response"], state["retrieved_chunks"], state["retry_count"]
Output: state updates: verified, verification_reason, retry_count,
        transformed_query (modified, only set if retrying)
"""

import logging
from typing import Dict

from agent.state import AgentState
from llm.client import LLMClient
from prompts_manager.registry import prompt_registry
from observability.tracer import traced

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _format_chunks_as_context(chunks) -> str:
    if not chunks:
        return "(No context was retrieved.)"
    return "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)


class VerifierNode:
    def __init__(self, llm_client: LLMClient | None = None, max_retries: int = MAX_RETRIES):
        self.llm_client = llm_client or LLMClient()
        self.max_retries = max_retries

    def _parse_verdict(self, raw_output: str) -> tuple[bool, str]:
        """
        Parses the strict VERDICT/REASON format from grounding_verifier_prompt.
        Defaults to "supported" on any parse failure — verification is a
        safety NET, not the sole gate; defaulting to fail-open here is a
        deliberate choice to avoid infinite retry loops on parse errors
        (max_retries is the hard backstop regardless).
        """
        lowered = raw_output.lower()
        reason = ""
        if "reason:" in lowered:
            reason = raw_output.split("REASON:", 1)[-1].strip() if "REASON:" in raw_output else ""

        if "verdict: unsupported" in lowered:
            return False, reason or "Marked unsupported by verifier."
        if "verdict: supported" in lowered:
            return True, reason or "Marked supported by verifier."

        logger.warning("Verifier output didn't match expected format: '%s'", raw_output[:120])
        return True, "Could not parse verifier output; defaulting to supported."

    @traced("verifier")
    def __call__(self, state: AgentState) -> Dict:
        retry_count = state.get("retry_count", 0)
        context = _format_chunks_as_context(state.get("retrieved_chunks", []))
        response = state.get("response", "")

        prompt = prompt_registry.format(
            "grounding_verifier_prompt", "v1", context=context, response=response
        )

        try:
            raw_output = self.llm_client.generate(prompt, temperature=0.0)
            verified, reason = self._parse_verdict(raw_output)
        except Exception as e:
            logger.error("VerifierNode LLM call failed (%s); defaulting to verified=True", e)
            verified, reason = True, f"Verifier call failed: {e}"

        updates: Dict = {
            "verified": verified,
            "verification_reason": reason,
        }

        if not verified and retry_count < self.max_retries:
            new_retry_count = retry_count + 1
            modified_query = f"{state.get('user_message', '')} (additional detail, attempt {new_retry_count})"
            logger.info(
                "Verifier rejected response (reason: %s). Triggering retry %d/%d.",
                reason, new_retry_count, self.max_retries,
            )
            updates["retry_count"] = new_retry_count
            updates["transformed_query"] = modified_query
        elif not verified:
            logger.warning(
                "Verifier rejected response but max_retries (%d) reached. Passing through anyway.",
                self.max_retries,
            )
            # retry_count stays as-is; router will see verified=False AND
            # retry_count >= max_retries and route to END (or fallback)
            # instead of looping forever.

        return updates
