"""
rag/query_transform.py

QueryTransformer: implements HyDE (Hypothetical Document Embeddings).

Instead of embedding the user's raw, often short/awkward query, we ask an LLM
to write a hypothetical answer to the query first, then embed THAT instead.
A hypothetical answer is much closer in style and vocabulary to the actual
document chunks we're searching over, which improves dense retrieval recall
for short or vaguely-worded patient queries (e.g. "can I get my pills early"
-> hypothetical answer uses words like "prescription", "repeat", "early
reissue" that are much closer to the actual docs).

Input:  raw query string
Output: transformed query string (the hypothetical answer text itself —
        rag/retriever.py embeds this directly in place of the raw query)

Deps: any object implementing `.generate(prompt: str) -> str`
      (the real implementation is llm/client.py:LLMClient, built in Phase 2;
      this file is written against a minimal protocol so it works standalone
      today and needs zero changes once LLMClient exists)
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

HYDE_PROMPT_TEMPLATE = """You are a medical receptionist assistant. Write a short, \
plausible-sounding answer (2-4 sentences) to the following patient question, as if it \
came from an NHS practice's policy documents. Do not say you are unsure. Do not add \
disclaimers. Just write the hypothetical answer text directly, in a neutral, factual tone.

Patient question: {query}

Hypothetical answer:"""


class GeneratesText(Protocol):
    """
    Minimal structural protocol any LLM client must satisfy to be used here.
    llm/client.py:LLMClient (Phase 2) implements this exact method signature,
    so QueryTransformer(llm_client=LLMClient()) will work with no changes here.
    """

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        ...


class QueryTransformer:
    """
    Wraps any LLM client implementing `.generate()` to perform HyDE transformation.
    """

    def __init__(self, llm_client: "GeneratesText"):
        self.llm_client = llm_client

    def hyde_transform(self, query: str) -> str:
        """
        Generates a hypothetical answer to `query` and returns it as the new
        search string. Falls back to the original query on any LLM failure
        so retrieval never hard-fails because of this optional enhancement step.
        """
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        try:
            hypothetical_answer = self.llm_client.generate(prompt)
            hypothetical_answer = hypothetical_answer.strip()
            if not hypothetical_answer:
                logger.warning("HyDE returned empty output, falling back to raw query")
                return query
            logger.info("HyDE transformed query '%s' -> '%s'", query[:50], hypothetical_answer[:80])
            return hypothetical_answer
        except Exception as e:
            logger.warning("HyDE transformation failed (%s), falling back to raw query", e)
            return query
