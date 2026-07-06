"""
agent/nodes/safety.py

SafetyNode: the first node in the graph. Runs BEFORE any LLM call.
Two jobs:
    1. Detect likely prompt injection attempts (pattern-based, no LLM —
       fast and avoids letting an injection attempt reach the LLM at all).
    2. Quick keyword-based emergency flag — a fast pre-check that runs
       alongside the (LLM-based) intent classifier as a safety net. If
       either this node OR the intent classifier says "emergency", the
       router sends the user to the hardcoded emergency node.

Input:  state["user_message"]
Output: state updates: is_safe, safety_reason, is_emergency_flagged
"""

import re
import logging
from typing import Dict

from agent.state import AgentState
from observability.tracer import traced

logger = logging.getLogger(__name__)

# Patterns that suggest an attempt to override system instructions.
# Pattern-based and intentionally simple — a real production system would
# layer an LLM-based or ML classifier on top, but a fast regex pre-filter
# is the right first line of defense: it costs ~0ms and catches the most
# common, unsophisticated injection attempts before they reach the LLM.
INJECTION_PATTERNS = [
    r"ignore (all |previous |above )*instructions",
    r"disregard (all|previous|above)",
    r"you are now",
    r"forget (everything|all) (you|that)",
    r"system prompt",
    r"reveal your (instructions|prompt)",
    r"act as (if|a)",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"developer mode",
]

# Keyword-based emergency pre-check — deliberately broad/over-sensitive.
# False positives here just mean the LLM-based intent classifier (which also
# checks for emergency) gets a second vote; false negatives here are the
# dangerous failure mode, so we bias toward catching too much rather than
# too little.
EMERGENCY_KEYWORDS = [
    "can't breathe", "cant breathe", "chest pain", "heart attack",
    "stroke", "unconscious", "unresponsive", "severe bleeding",
    "bleeding heavily", "choking", "anaphylaxis", "allergic reaction",
    "seizure", "overdose", "suicidal", "kill myself", "want to die",
    "self harm", "self-harm", "not breathing", "collapsed",
]


class SafetyNode:
    """Stateless — pure pattern matching, no LLM call, no external dependency."""

    def __init__(self):
        self._injection_regex = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

    def check_injection(self, message: str) -> tuple[bool, str | None]:
        """Returns (is_safe, reason). is_safe=False means injection-like pattern matched."""
        match = self._injection_regex.search(message)
        if match:
            return False, f"Potential prompt injection pattern detected: '{match.group(0)}'"
        return True, None

    def check_emergency_keywords(self, message: str) -> bool:
        """Fast keyword scan. Returns True if any emergency keyword is present."""
        lowered = message.lower()
        return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)

    @traced("safety_check")
    def __call__(self, state: AgentState) -> Dict:
        """
        LangGraph node entrypoint. Takes full state, returns a dict of
        fields to merge back into state.
        """
        message = state.get("user_message", "")

        is_safe, reason = self.check_injection(message)
        is_emergency_flagged = self.check_emergency_keywords(message)

        if not is_safe:
            logger.warning("Safety node flagged message as unsafe: %s", reason)
        if is_emergency_flagged:
            logger.info("Safety node keyword-flagged this message as a possible emergency")

        return {
            "is_safe": is_safe,
            "safety_reason": reason,
            "is_emergency_flagged": is_emergency_flagged,
        }
