"""
prompts_manager/registry.py

PromptRegistry: loads prompts/registry.yaml once and provides a clean
`.get(name, version, **kwargs)` interface for formatting prompts with
variables. Also returns which version was used so callers can log it to
the tracer (observability/tracer.py, built later this phase) — this is
what makes the registry "A/B test ready": system_prompt has both v1 and v2,
and switching is a one-line change at the call site.

Input:  prompt name + version + template variables
Output: formatted prompt string
"""

import os
import logging
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "registry.yaml")


class PromptNotFoundError(Exception):
    pass


class PromptRegistry:
    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        self.registry_path = registry_path
        self._registry: Dict[str, Any] = {}
        self._load()

    def _load(self):
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self._registry = yaml.safe_load(f)
        logger.info("Loaded prompt registry from '%s' (%d prompt names)", self.registry_path, len(self._registry))

    def get_raw_template(self, name: str, version: str = "v1") -> str:
        """Returns the unformatted template string for a given prompt name/version."""
        if name not in self._registry:
            raise PromptNotFoundError(f"No prompt named '{name}' in registry")
        if version not in self._registry[name]:
            raise PromptNotFoundError(f"Prompt '{name}' has no version '{version}'")
        return self._registry[name][version]["template"]

    def format(self, name: str, version: str = "v1", **kwargs) -> str:
        """
        Loads the prompt template by name+version and formats it with kwargs.
        Missing variables raise a clear error rather than silently producing
        a broken prompt with literal "{variable}" left in it.
        """
        template = self.get_raw_template(name, version)
        try:
            formatted = template.format(**kwargs)
        except KeyError as e:
            raise PromptNotFoundError(
                f"Missing variable {e} when formatting prompt '{name}' version '{version}'. "
                f"Expected variables: {self._registry[name][version].get('variables')}"
            ) from e
        return formatted

    def list_versions(self, name: str) -> list:
        """Returns all available versions for a prompt name — useful for A/B test setup."""
        if name not in self._registry:
            raise PromptNotFoundError(f"No prompt named '{name}' in registry")
        return list(self._registry[name].keys())


# Singleton instance — import this everywhere a prompt is needed:
#   from prompts_manager.registry import prompt_registry
prompt_registry = PromptRegistry()
