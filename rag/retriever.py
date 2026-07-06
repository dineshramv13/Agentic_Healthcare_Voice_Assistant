"""
rag/retriever.py

HybridRetriever: combines BM25 (keyword) and dense vector retrieval using
Reciprocal Rank Fusion (RRF), then re-ranks the fused results with a local
cross-encoder for a final precision pass.

Input:  query string, top_k
Output: list of {"text", "source", "score"} dicts, best chunks first

Why hybrid: BM25 catches exact clinical/policy terms (e.g. "999", "NHS 111",
"fit note") that dense embeddings can blur into nearby concepts. Dense vectors
catch paraphrases and semantic meaning BM25 would miss entirely. RRF fuses
both rankings without needing to normalize incomparable score scales.

Deps: chromadb, rank_bm25, sentence-transformers (CrossEncoder)
"""

import logging
from typing import List, Dict

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config.settings import settings
from rag.embeddings import EmbeddingModel
from rag.ingestion import DocumentIngester

logger = logging.getLogger(__name__)

# Local cross-encoder reranker — small, fast, free, runs on CPU
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _simple_tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for BM25 — lowercase, split on whitespace/punctuation."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Loads every chunk from the ChromaDB collection once at init time
    (cheap at this project's scale — a few hundred chunks) to build a
    parallel BM25 index in memory alongside the existing dense vector index.
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None, ingester: DocumentIngester | None = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.ingester = ingester or DocumentIngester(embedding_model=self.embedding_model)
        self.collection = self.ingester.get_collection()
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        self._load_bm25_index()

    def _load_bm25_index(self):
        """Pulls all chunks out of ChromaDB and builds an in-memory BM25 index."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids: List[str] = all_data["ids"]
        self.doc_texts: List[str] = all_data["documents"]
        self.doc_metadatas: List[dict] = all_data["metadatas"]

        if not self.doc_texts:
            logger.warning(
                "ChromaDB collection is empty. Run 'python scripts/ingest.py' first."
            )
            self.bm25 = None
            return

        tokenized_corpus = [_simple_tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index over %d chunks", len(self.doc_texts))

    def _dense_search(self, query: str, top_k: int) -> List[str]:
        """Returns a ranked list of chunk ids from dense vector search."""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(len(self.doc_texts), 1)),
        )
        return results["ids"][0] if results["ids"] else []

    def _bm25_search(self, query: str, top_k: int) -> List[str]:
        """Returns a ranked list of chunk ids from BM25 keyword search."""
        if self.bm25 is None:
            return []
        tokenized_query = _simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.doc_ids[i] for i in ranked_indices[:top_k]]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[str]], k: int = 60
    ) -> List[str]:
        """
        Reciprocal Rank Fusion, implemented from scratch.

        For each id, score = sum over all ranked lists of 1 / (k + rank).
        k=60 is the standard RRF constant from the original paper — it dampens
        the impact of very high ranks so one list doesn't dominate completely.

        Returns ids sorted by fused score, descending.
        """
        fused_scores: Dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                fused_scores.setdefault(doc_id, 0.0)
                fused_scores[doc_id] += 1.0 / (k + rank + 1)  # rank+1 so top item isn't divided by k+0

        return sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)

    def _rerank(self, query: str, candidate_ids: List[str], top_n: int) -> List[Dict]:
        """
        Cross-encoder reranking: scores each (query, chunk) pair directly
        (more accurate than embedding similarity, but too slow to run on the
        whole corpus — hence only applied to the small RRF-fused candidate set).
        """
        id_to_text = dict(zip(self.doc_ids, self.doc_texts))
        id_to_meta = dict(zip(self.doc_ids, self.doc_metadatas))

        pairs = [(query, id_to_text[cid]) for cid in candidate_ids if cid in id_to_text]
        valid_ids = [cid for cid in candidate_ids if cid in id_to_text]

        if not pairs:
            return []

        scores = self.reranker.predict(pairs)

        scored = list(zip(valid_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored[:top_n]:
            results.append(
                {
                    "text": id_to_text[doc_id],
                    "source": id_to_meta[doc_id].get("source", "unknown"),
                    "score": float(score),
                }
            )
        return results

    def retrieve(self, query: str, top_k: int | None = None, top_n: int | None = None) -> List[Dict]:
        """
        Full hybrid retrieval pipeline:
            BM25 (top_k) + Dense (top_k) -> RRF fusion -> cross-encoder rerank -> top_n

        Args:
            query: the (possibly HyDE-transformed) search query
            top_k: how many candidates to pull from EACH of BM25/dense before fusion
            top_n: how many final chunks to return after reranking

        Returns:
            list of {"text", "source", "score"}, best first. Empty list if the
            collection has no data — caller (agent/nodes/retriever.py) must handle this.
        """
        top_k = top_k or settings.retrieval_top_k
        top_n = top_n or settings.rerank_top_n

        if not self.doc_texts:
            return []

        dense_ids = self._dense_search(query, top_k)
        bm25_ids = self._bm25_search(query, top_k)

        fused_ids = self._reciprocal_rank_fusion([dense_ids, bm25_ids])

        # Rerank only the fused candidate set (cheap — at most 2*top_k items)
        candidate_pool = fused_ids[: top_k * 2]
        results = self._rerank(query, candidate_pool, top_n)

        logger.info(
            "Retrieved %d final chunks for query '%s' (dense=%d, bm25=%d, fused_pool=%d)",
            len(results), query[:60], len(dense_ids), len(bm25_ids), len(candidate_pool),
        )
        return results
