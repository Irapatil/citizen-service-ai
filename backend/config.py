from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute path so the server finds .env regardless of working directory
_ENV_FILE = str(Path(__file__).parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    # ------------------------------------------------------------------
    # LLM Provider Selection  (claude | openai | azure | gemini)
    # ------------------------------------------------------------------
    llm_provider: str = "claude"

    # ------------------------------------------------------------------
    # Anthropic Claude
    # ------------------------------------------------------------------
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # ------------------------------------------------------------------
    # Azure OpenAI
    # ------------------------------------------------------------------
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"
    azure_openai_deployment_name: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/citizen_service"
    database_url_sync: str = "postgresql://user:password@localhost:5432/citizen_service"

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "citizen_knowledge"

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_env: str = "development"
    secret_key: str = "changeme"
    cors_origins: str = "http://localhost:3000"

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    session_ttl_seconds: int = 3600
    max_history_turns: int = 10

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
