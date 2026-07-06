"""
config/settings.py

Central configuration for the entire AI-Local project.
Every other module imports `settings` from here instead of reading
os.environ directly. This is the single source of truth for config.

Input:  environment variables (loaded from .env via python-dotenv)
Output: a singleton `settings` object used everywhere else in the codebase
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- LLM Provider ---
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    model_name: str = Field(default="openrouter/owl-alpha", alias="MODEL_NAME")
    fallback_model_name: str = Field(
        default="google/gemma-4-31b-it:free", alias="FALLBACK_MODEL_NAME"
    )

    # --- RAG / Vector store ---
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="AI_nhs_docs", alias="CHROMA_COLLECTION_NAME")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(default=3, alias="RERANK_TOP_N")

    # --- Docs ---
    docs_dir: str = Field(default="./docs", alias="DOCS_DIR")

    # --- General ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    session_db_path: str = Field(default="./sessions.db", alias="SESSION_DB_PATH")
    trace_dir: str = Field(default="./traces", alias="TRACE_DIR")

    # --- LLM behavior ---
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    class Config:
        env_file = ".env"
        populate_by_name = True
        extra = "ignore"


# Singleton instance — import this everywhere:
#   from config.settings import settings
settings = Settings()
