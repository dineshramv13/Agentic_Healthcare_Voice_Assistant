"""
llm/client.py

LLMClient: wraps OpenRouter's free-tier models via their OpenAI-compatible
chat completions API. Handles retries with backoff, timeouts, and falls back
to a secondary free model if the primary one is rate-limited or down.

Input:  prompt string (+ optional system prompt, message history)
Output: response text string

No paid services. OpenRouter's free models (suffixed ":free") require only
a free API key — no card on file, no usage charges.

Satisfies the `GeneratesText` protocol expected by rag/query_transform.py:
    def generate(self, prompt: str, system_prompt: str | None = None) -> str
"""

import time
import logging
from typing import List, Dict, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMClientError(Exception):
    """Raised when the LLM call fails after all retries and fallback attempts."""
    pass


class LLMClient:
    """
    Thin client over OpenRouter's chat completions endpoint.

    Usage:
        client = LLMClient()
        text = client.generate("What are your opening hours?")

        # with conversation history:
        text = client.generate_from_messages([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self.api_key = api_key or settings.openrouter_api_key
        self.model_name = model_name or settings.model_name
        self.fallback_model_name = fallback_model_name or settings.fallback_model_name
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout_seconds
        self.max_retries = max_retries or settings.llm_max_retries

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY is not set. LLM calls will fail until it's "
                "configured in your .env file. Get a free key at https://openrouter.ai/keys"
            )

    def _call_api(self, messages: List[Dict[str, str]], model_name: str, temperature: Optional[float] = None) -> str:
        """Single HTTP call to OpenRouter. Raises on non-200 or malformed response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout
        )

        if response.status_code != 200:
            raise LLMClientError(
                f"OpenRouter returned status {response.status_code} for model "
                f"'{model_name}': {response.text[:300]}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMClientError(f"Unexpected OpenRouter response shape: {data}") from e

    def generate_from_messages(
        self, messages: List[Dict[str, str]], temperature: Optional[float] = None
    ) -> str:
        """
        Calls the LLM with a full message history.
        Retries the primary model with exponential backoff, then tries the
        fallback model once before giving up entirely.
        """
        last_error: Optional[Exception] = None

        # --- Attempt 1: primary model, with retries ---
        for attempt in range(self.max_retries):
            try:
                return self._call_api(messages, self.model_name, temperature)
            except Exception as e:
                last_error = e
                wait = 2 ** attempt  # 1s, 2s, 4s...
                logger.warning(
                    "LLM call failed (attempt %d/%d) on model '%s': %s. Retrying in %ds.",
                    attempt + 1, self.max_retries, self.model_name, e, wait,
                )
                time.sleep(wait)

        # --- Attempt 2: fallback model, single try ---
        logger.warning(
            "Primary model '%s' exhausted retries. Trying fallback model '%s'.",
            self.model_name, self.fallback_model_name,
        )
        try:
            return self._call_api(messages, self.fallback_model_name, temperature)
        except Exception as e:
            last_error = e

        raise LLMClientError(
            f"LLM call failed on both primary ('{self.model_name}') and fallback "
            f"('{self.fallback_model_name}') models. Last error: {last_error}"
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """
        Convenience method for a single-turn prompt (no conversation history).
        This is the method rag/query_transform.py's HyDE step calls.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_from_messages(messages, temperature=temperature)
