"""
scripts/ingest.py

Run this once (and again any time docs/ changes) to populate the local
ChromaDB collection from the markdown files in docs/.

Usage:
    python scripts/ingest.py
"""

import logging
import sys
import os

# Allow running this script directly from the scripts/ folder or project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from rag.ingestion import DocumentIngester

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    print(f"Ingesting documents from '{settings.docs_dir}' ...")
    ingester = DocumentIngester()
    num_chunks = ingester.ingest(reset=True)
    print(f"Done. Ingested {num_chunks} chunks into ChromaDB at '{settings.chroma_persist_dir}'.")
    print(f"Collection name: '{settings.chroma_collection_name}'")


if __name__ == "__main__":
    main()
