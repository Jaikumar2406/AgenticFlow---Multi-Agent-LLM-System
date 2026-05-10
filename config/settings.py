import os
from typing import Optional

class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://agentic:agentic@postgres:5432/agenticflow")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # LLM Configuration (Groq)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Web Search (Tavily)
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Worker Configuration
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "4"))

    # Context Budget
    DEFAULT_CONTEXT_BUDGET: int = int(os.getenv("DEFAULT_CONTEXT_BUDGET", "100000"))
    MAX_CONTEXT_BUDGET: int = int(os.getenv("MAX_CONTEXT_BUDGET", "200000"))

    # Tool Configuration
    CODE_EXECUTION_TIMEOUT: int = int(os.getenv("CODE_EXECUTION_TIMEOUT", "30"))
    WEB_SEARCH_TIMEOUT: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "10"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Evaluation
    EVAL_RETRIES: int = int(os.getenv("EVAL_RETRIES", "2"))

settings = Settings()