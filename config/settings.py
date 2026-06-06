"""
Central configuration for the AI Knowledge Base / FAQ Assistant.
All paths, model names, thresholds, and constants live here.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EVAL_DIR: Path = DATA_DIR / "eval"
    EVAL_RESULTS_DIR: Path = EVAL_DIR / "eval_results"

    # --- LLM ---
    LLM_PROVIDER: str = "ollama"  # ollama | groq | deepseek

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"

    # Groq (free tier)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # DeepSeek (paid)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # --- Embedding ---
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL_LOCAL: str = str(
        Path.home() / ".cache/modelscope/hub/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DIM: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # --- Reranker ---
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

    # --- Chunking ---
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # --- Retrieval ---
    BM25_TOP_K: int = 20
    DENSE_TOP_K: int = 20
    RRF_K: int = 60
    RERANK_TOP_K: int = 5
    FINAL_TOP_K: int = 5

    # --- Clarification ---
    DOMAIN_RATIO_THRESHOLD: float = 0.80
    SCORE_GAP_THRESHOLD: float = 0.12
    CLARIFY_TOP_K: int = 15

    # --- Vector Store ---
    VECTOR_PERSIST_DIR: str = str(PROJECT_ROOT / ".chroma")

    # --- Server ---
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # --- API Security ---
    API_KEY: str = ""
    ALLOWED_ORIGINS: str = "http://localhost:8501"

    # --- Feishu ---
    FEISHU_APP_ID: str = ""
    FEISHU_APP_SECRET: str = ""
    FEISHU_VERIFICATION_TOKEN: str = ""

    # --- Eval ---
    EVAL_TEST_SET_SIZE: int = 80

    model_config = {
        "env_file": str(Path.home() / ".ai_bot_env"),
        "env_file_encoding": "utf-8",
    }


settings = Settings()
