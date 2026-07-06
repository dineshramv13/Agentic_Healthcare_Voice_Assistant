"""
rag/embeddings.py

Wraps a local, free sentence-transformers embedding model.
No API calls, no cost — runs entirely on CPU.

Input:  list[str] of text chunks (or a single query string)
Output: list[list[float]] embedding vectors (384-dim for all-MiniLM-L6-v2)

Used by:
    - rag/ingestion.py        (embed chunks before storing in ChromaDB)
    - rag/retriever.py        (embed queries at search time)
    - rag/query_transform.py  (embed the HyDE hypothetical answer)
"""

from typing import List
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.
    Loaded once and reused — model loading is the expensive part,
    so this class should be instantiated once and shared (see ingestion.py
    and retriever.py, which both take an EmbeddingModel instance).
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        # trust_remote_code=False is the safe default for a known public model
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of document chunks. Used during ingestion.
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-similarity ready
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string. Used at retrieval time.
        """
        embedding = self._model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
