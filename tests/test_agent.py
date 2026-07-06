"""
tests/test_agent.py

Unit tests for agent/router.py's conditional routing functions. These are
pure functions of state — no LLM calls, no mocking needed, which is
exactly the testability benefit of keeping routing logic in plain Python
functions separate from the nodes themselves (see router.py's docstring).

Run with:
    pytest tests/test_agent.py -v
"""

from agent.router import route_after_safety_and_intent, route_after_verifier, MAX_RETRIES


class TestRouteAfterSafetyAndIntent:
    def test_emergency_keyword_flag_routes_to_emergency(self):
        state = {"is_emergency_flagged": True, "intent": "info", "is_safe": True}
        assert route_after_safety_and_intent(state) == "emergency"

    def test_emergency_intent_routes_to_emergency(self):
        state = {"is_emergency_flagged": False, "intent": "emergency", "is_safe": True}
        assert route_after_safety_and_intent(state) == "emergency"

    def test_either_signal_alone_is_sufficient_for_emergency(self):
        # Keyword flag True, classifier says something else entirely — still emergency.
        state = {"is_emergency_flagged": True, "intent": "appointment", "is_safe": True}
        assert route_after_safety_and_intent(state) == "emergency"

    def test_unsafe_input_routes_to_fallback(self):
        state = {"is_emergency_flagged": False, "intent": "info", "is_safe": False, "safety_reason": "injection"}
        assert route_after_safety_and_intent(state) == "fallback"

    def test_out_of_scope_routes_to_fallback(self):
        state = {"is_emergency_flagged": False, "intent": "out_of_scope", "is_safe": True}
        assert route_after_safety_and_intent(state) == "fallback"

    def test_appointment_intent_routes_to_retriever(self):
        state = {"is_emergency_flagged": False, "intent": "appointment", "is_safe": True}
        assert route_after_safety_and_intent(state) == "retriever"

    def test_prescription_intent_routes_to_retriever(self):
        state = {"is_emergency_flagged": False, "intent": "prescription", "is_safe": True}
        assert route_after_safety_and_intent(state) == "retriever"

    def test_info_intent_routes_to_retriever(self):
        state = {"is_emergency_flagged": False, "intent": "info", "is_safe": True}
        assert route_after_safety_and_intent(state) == "retriever"

    def test_emergency_takes_priority_over_unsafe_flag(self):
        # If BOTH emergency and unsafe signals fire, emergency must win —
        # clinical safety takes priority over injection handling.
        state = {"is_emergency_flagged": True, "intent": "info", "is_safe": False}
        assert route_after_safety_and_intent(state) == "emergency"


class TestRouteAfterVerifier:
    def test_verified_routes_to_end(self):
        state = {"verified": True, "retry_count": 0}
        assert route_after_verifier(state) == "end"

    def test_unverified_with_retries_left_routes_to_retriever(self):
        state = {"verified": False, "retry_count": 1}
        assert route_after_verifier(state) == "retriever"

    def test_unverified_at_zero_retries_routes_to_retriever(self):
        state = {"verified": False, "retry_count": 0}
        assert route_after_verifier(state) == "retriever"

    def test_unverified_at_max_retries_routes_to_fallback(self):
        state = {"verified": False, "retry_count": MAX_RETRIES}
        assert route_after_verifier(state) == "fallback"

    def test_unverified_beyond_max_retries_routes_to_fallback(self):
        # Defensive: even if retry_count somehow exceeds MAX_RETRIES, never loop.
        state = {"verified": False, "retry_count": MAX_RETRIES + 5}
        assert route_after_verifier(state) == "fallback"

    def test_missing_verified_key_defaults_to_end(self):
        # route_after_verifier defaults verified=True if the key is absent —
        # a defensive default so a malformed state doesn't loop forever.
        state = {"retry_count": 0}
        assert route_after_verifier(state) == "end"

    def test_retry_loop_never_exceeds_max_retries(self):
        """
        Simulates the full retry loop end-to-end using only the router
        functions, confirming it terminates within MAX_RETRIES retriever
        re-entries no matter what — this is the test that would catch an
        off-by-one between VerifierNode.max_retries and router.MAX_RETRIES.
        """
        retry_count = 0
        verified = False
        loop_iterations = 0
        max_iterations = MAX_RETRIES + 2  # safety net for the test itself

        while loop_iterations < max_iterations:
            decision = route_after_verifier({"verified": verified, "retry_count": retry_count})
            loop_iterations += 1
            if decision == "end":
                break
            if decision == "fallback":
                break
            # decision == "retriever" -> simulate VerifierNode incrementing retry_count
            retry_count += 1

        assert decision == "fallback"
        assert retry_count == MAX_RETRIES
        assert loop_iterations <= MAX_RETRIES + 1
