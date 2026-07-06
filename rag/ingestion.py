"""
rag/ingestion.py

DocumentIngester: reads markdown/text docs from a folder, splits them into
overlapping chunks, embeds them, and stores them in a local persistent
ChromaDB collection.

Input:  path to a docs/ folder containing .md / .txt files
Output: a populated ChromaDB collection on disk at settings.chroma_persist_dir

Deps: langchain-text-splitters, chromadb, sentence-transformers (via embeddings.py)

Run via: python scripts/ingest.py
"""

import os
import glob
import logging
from typing import List, Dict

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Handles the offline ingestion pipeline:
        docs/*.md  ->  chunks  ->  embeddings  ->  ChromaDB (persisted locally)
    """

    def __init__(self, embedding_model: EmbeddingModel | None = None):
        self.embedding_model = embedding_model or EmbeddingModel()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )

        # Local persistent client — no server, no cloud, just a folder on disk
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def _load_documents(self, docs_dir: str) -> List[Dict[str, str]]:
        """
        Reads every .md / .txt file in docs_dir.
        Returns a list of {"source": filename, "text": full_text}.
        """
        paths = sorted(
            glob.glob(os.path.join(docs_dir, "*.md"))
            + glob.glob(os.path.join(docs_dir, "*.txt"))
        )
        if not paths:
            raise FileNotFoundError(
                f"No .md or .txt files found in '{docs_dir}'. "
                "Add documents before running ingestion."
            )

        documents = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"source": os.path.basename(path), "text": text})
            logger.info("Loaded document: %s (%d chars)", os.path.basename(path), len(text))
        return documents

    def _chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Splits each document into overlapping chunks.
        Returns a flat list of {"source", "chunk_index", "text"}.
        """
        chunks = []
        for doc in documents:
            pieces = self.splitter.split_text(doc["text"])
            for i, piece in enumerate(pieces):
                chunks.append(
                    {
                        "source": doc["source"],
                        "chunk_index": i,
                        "text": piece,
                    }
                )
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
        return chunks

    def ingest(self, docs_dir: str | None = None, reset: bool = True) -> int:
        """
        Full ingestion pipeline. Returns number of chunks ingested.

        Args:
            docs_dir: folder to read from (defaults to settings.docs_dir)
            reset: if True, deletes any existing collection first so re-running
                   ingestion doesn't duplicate or stack chunks
        """
        docs_dir = docs_dir or settings.docs_dir

        if reset:
            try:
                self.client.delete_collection(settings.chroma_collection_name)
                logger.info("Deleted existing collection '%s'", settings.chroma_collection_name)
            except Exception:
                pass  # collection didn't exist yet — fine

        collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        documents = self._load_documents(docs_dir)
        chunks = self._chunk_documents(documents)

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        ids = [f"{c['source']}::chunk_{c['chunk_index']}" for c in chunks]
        metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            "Ingested %d chunks into ChromaDB collection '%s' at '%s'",
            len(chunks),
            settings.chroma_collection_name,
            settings.chroma_persist_dir,
        )
        return len(chunks)

    def get_collection(self):
        """Returns the ChromaDB collection handle (used by retriever.py)."""
        return self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
