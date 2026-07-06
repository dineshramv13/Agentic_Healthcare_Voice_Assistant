"""
tools/faq_tool.py

FAQTool: the tool the RetrieverNode calls to answer "info" and similar
intents. Wraps HybridRetriever + QueryTransformer (HyDE) behind a single
clean interface. This is the canonical "tool calling" example in the
project — when asked "what is tool calling in your agent", this file is
the answer: the agent doesn't query ChromaDB directly, it calls a named
tool that encapsulates that capability.

Input:  query string
Output: list of retrieved chunks (text, source, score)
"""

import logging
from typing import List

from agent.state import RetrievedChunk
from rag.retriever import HybridRetriever
from rag.query_transform import QueryTransformer
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class FAQTool:
    """
    Tool name: "faq_lookup"
    Description: Searches the practice's knowledge base (appointments,
    prescriptions, surgery info, services, emergency guidance) for
    information relevant to a patient query, using hybrid retrieval
    with HyDE query transformation.
    """

    name = "faq_lookup"
    description = (
        "Searches the NHS-style knowledge base for information about "
        "appointments, prescriptions, surgery details, and services."
    )

    def __init__(self, retriever: HybridRetriever | None = None, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()
        self.retriever = retriever or HybridRetriever()
        self.query_transformer = QueryTransformer(llm_client=self.llm_client)

    def run(self, query: str, use_hyde: bool = True) -> List[RetrievedChunk]:
        """
        Executes the tool: optionally HyDE-transforms the query, then runs
        hybrid retrieval (BM25 + dense + RRF + rerank).
        """
        search_query = self.query_transformer.hyde_transform(query) if use_hyde else query
        results = self.retriever.retrieve(search_query)
        logger.info("FAQTool returned %d chunks for query '%s'", len(results), query[:60])
        return results  # type: ignore[return-value]
