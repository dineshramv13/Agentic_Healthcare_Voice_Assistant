"""
tests/test_prompts.py

Unit tests for prompts_manager/registry.py. This module has minimal
dependencies (only pyyaml) and no config.settings dependency, so these
tests run in nearly any environment without needing the full project's
dependency stack installed.

Run with:
    pytest tests/test_prompts.py -v
"""

import pytest
from prompts_manager.registry import PromptRegistry, PromptNotFoundError


@pytest.fixture
def registry():
    return PromptRegistry()


class TestPromptRegistry:
    def test_loads_all_expected_prompt_names(self, registry):
        expected_names = {
            "system_prompt",
            "intent_classifier_prompt",
            "grounding_verifier_prompt",
            "hyde_prompt",
            "emergency_detection_prompt",
        }
        assert expected_names.issubset(set(registry._registry.keys()))

    def test_system_prompt_has_both_v1_and_v2(self, registry):
        # This is the literal claim from the README: "A/B-test ready".
        versions = registry.list_versions("system_prompt")
        assert "v1" in versions
        assert "v2" in versions

    def test_format_substitutes_variables_correctly(self, registry):
        formatted = registry.format(
            "intent_classifier_prompt", "v1", message="How do I book an appointment?"
        )
        assert "How do I book an appointment?" in formatted
        assert "{message}" not in formatted  # placeholder must be fully substituted

    def test_format_with_multiple_variables(self, registry):
        formatted = registry.format(
            "system_prompt", "v1",
            context="Some context here",
            history="(no prior turns)",
            question="What are your hours?",
        )
        assert "Some context here" in formatted
        assert "What are your hours?" in formatted
        assert "{context}" not in formatted
        assert "{question}" not in formatted

    def test_missing_variable_raises_clear_error(self, registry):
        with pytest.raises(PromptNotFoundError):
            registry.format("intent_classifier_prompt", "v1")  # missing 'message'

    def test_nonexistent_prompt_name_raises(self, registry):
        with pytest.raises(PromptNotFoundError):
            registry.format("totally_made_up_prompt", "v1", foo="bar")

    def test_nonexistent_version_raises(self, registry):
        with pytest.raises(PromptNotFoundError):
            registry.format("system_prompt", "v99", context="", history="", question="")

    def test_list_versions_for_nonexistent_prompt_raises(self, registry):
        with pytest.raises(PromptNotFoundError):
            registry.list_versions("not_a_real_prompt")

    def test_get_raw_template_returns_unformatted_string(self, registry):
        raw = registry.get_raw_template("hyde_prompt", "v1")
        assert "{query}" in raw  # placeholder still present, not yet formatted
