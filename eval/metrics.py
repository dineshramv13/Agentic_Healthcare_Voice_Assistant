"""
eval/metrics.py

RAGASMetrics: implements RAGAS-style evaluation metrics — faithfulness,
answer relevance, and context precision — using OUR OWN free OpenRouter
LLM as the judge, rather than the `ragas` pip package (which defaults to
needing an OpenAI key for its judge model and has a habit of pulling in
conflicting dependency versions). Same metrics, zero extra cost or dependency risk.

LLMJudge: a separate, more holistic 1-5 quality score ("LLM-as-judge"),
distinct from the three RAGAS-style metrics — covers things like tone,
clarity, and completeness that faithfulness/relevance/precision don't
directly capture.

Metrics implemented:
    - faithfulness:        does the response avoid claims NOT supported
                            by the retrieved context? (0.0-1.0)
    - answer_relevance:     does the response actually address the
                            question asked? (0.0-1.0)
    - context_precision:    of the retrieved chunks, what fraction were
                            actually relevant to the query? (0.0-1.0)
    - llm_judge_score:      holistic quality score (1-5 integer)

Input:  query, retrieved_chunks, response (per evaluated turn)
Output: a dict of scores for that turn
"""

import logging
import re
from typing import List, Dict

from llm.client import LLMClient

logger = logging.getLogger(__name__)


FAITHFULNESS_PROMPT = """You are evaluating whether a RESPONSE is faithful to (fully \
supported by) the given CONTEXT — i.e. it does not invent facts that aren't in the context.

CONTEXT:
{context}

RESPONSE:
{response}

Count the factual claims in the RESPONSE. Count how many of those claims are directly \
supported by the CONTEXT. A response that correctly says "I don't have that information" \
when the context lacks the answer counts as fully faithful.

Respond with ONLY a single decimal number between 0.0 and 1.0, representing the fraction \
of claims that are supported (1.0 = fully faithful, 0.0 = entirely unsupported). \
No explanation, just the number."""


ANSWER_RELEVANCE_PROMPT = """You are evaluating whether a RESPONSE actually addresses \
the QUESTION that was asked, regardless of whether the response is factually correct.

QUESTION:
{question}

RESPONSE:
{response}

A response that is evasive, off-topic, or answers a different question than the one \
asked should score low. A response that directly addresses what was asked should score high.

Respond with ONLY a single decimal number between 0.0 and 1.0 (1.0 = fully relevant, \
0.0 = completely irrelevant). No explanation, just the number."""


CONTEXT_PRECISION_PROMPT = """You are evaluating retrieved CONTEXT CHUNKS for relevance \
to a QUESTION.

QUESTION:
{question}

CONTEXT CHUNKS:
{chunks}

For each chunk, decide if it is relevant (contains information useful for answering the \
question) or irrelevant (off-topic, not useful for this specific question).

Respond with ONLY a single decimal number between 0.0 and 1.0, representing the fraction \
of chunks that are relevant. If there are no chunks, respond with 0.0. No explanation, \
just the number."""


LLM_JUDGE_PROMPT = """You are a quality reviewer for an AI healthcare receptionist's responses.

QUESTION:
{question}

RESPONSE:
{response}

Rate the RESPONSE on a scale of 1 to 5, considering:
- Clarity (is it easy to understand?)
- Completeness (does it fully answer the question?)
- Tone (is it warm, professional, and appropriate for a healthcare setting?)
- Safety (does it avoid giving medical advice or diagnoses?)

Respond with ONLY a single integer from 1 to 5. No explanation, just the number."""


def _extract_float(raw_output: str, default: float = 0.0) -> float:
    """Pulls the first decimal/integer number out of an LLM response, clamped to [0,1]."""
    match = re.search(r"(\d+\.?\d*)", raw_output)
    if not match:
        logger.warning("Could not parse a number from judge output: '%s'", raw_output[:100])
        return default
    value = float(match.group(1))
    return max(0.0, min(1.0, value))


def _extract_int(raw_output: str, default: int = 3, min_val: int = 1, max_val: int = 5) -> int:
    """Pulls the first integer out of an LLM response, clamped to [min_val, max_val]."""
    match = re.search(r"(\d+)", raw_output)
    if not match:
        logger.warning("Could not parse an integer from judge output: '%s'", raw_output[:100])
        return default
    value = int(match.group(1))
    return max(min_val, min(max_val, value))


def _format_chunks(chunks: List[Dict]) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(f"[Chunk {i+1} — source: {c.get('source', 'unknown')}]\n{c.get('text', '')}" for i, c in enumerate(chunks))


class RAGASMetrics:
    """Computes faithfulness, answer_relevance, and context_precision for a single turn."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def faithfulness(self, context: str, response: str) -> float:
        prompt = FAITHFULNESS_PROMPT.format(context=context or "(no context)", response=response)
        try:
            raw = self.llm_client.generate(prompt, temperature=0.0)
            return _extract_float(raw)
        except Exception as e:
            logger.error("Faithfulness scoring failed: %s", e)
            return 0.0

    def answer_relevance(self, question: str, response: str) -> float:
        prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, response=response)
        try:
            raw = self.llm_client.generate(prompt, temperature=0.0)
            return _extract_float(raw)
        except Exception as e:
            logger.error("Answer relevance scoring failed: %s", e)
            return 0.0

    def context_precision(self, question: str, chunks: List[Dict]) -> float:
        if not chunks:
            return 0.0
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, chunks=_format_chunks(chunks))
        try:
            raw = self.llm_client.generate(prompt, temperature=0.0)
            return _extract_float(raw)
        except Exception as e:
            logger.error("Context precision scoring failed: %s", e)
            return 0.0

    def score_turn(self, question: str, chunks: List[Dict], response: str) -> Dict[str, float]:
        """Computes all three RAGAS-style metrics for one query/response pair."""
        context_text = _format_chunks(chunks)
        return {
            "faithfulness": self.faithfulness(context_text, response),
            "answer_relevance": self.answer_relevance(question, response),
            "context_precision": self.context_precision(question, chunks),
        }


class LLMJudge:
    """Holistic 1-5 quality score, separate from the RAGAS-style metrics above."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def score(self, question: str, response: str) -> int:
        prompt = LLM_JUDGE_PROMPT.format(question=question, response=response)
        try:
            raw = self.llm_client.generate(prompt, temperature=0.0)
            return _extract_int(raw)
        except Exception as e:
            logger.error("LLM-judge scoring failed: %s", e)
            return 3  # neutral default on failure, not 1 or 5, to avoid skewing aggregates
