"""
JARVIS OMEGA - Configuration Management
Environment configuration using Pydantic Settings with validation and type safety.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Application
    APP_NAME: str = "JARVIS OMEGA"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Keys
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    DEEPGRAM_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None
    HUGGINGFACE_API_KEY: str | None = None

    # Database
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "jarvis"
    POSTGRES_USER: str = "jarvis"
    POSTGRES_PASSWORD: str = "jarvis_secure_password"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Neo4j
    NEO4J_URI: str = "bolt://neo4j:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "jarvis_neo4j_password"
    NEO4J_DATABASE: str = "neo4j"

    # ChromaDB
    CHROMADB_HOST: str = "chromadb"
    CHROMADB_PORT: int = 8000
    CHROMADB_COLLECTION: str = "jarvis_memory"

    # Ollama
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_MODELS: list[str] = Field(
        default_factory=lambda: [
            "llama3.1:70b",
            "llama3.1:8b",
            "mistral:7b",
            "codellama:34b",
            "mistral-nemo:12b",
            "phi3:14b",
            "gemma2:27b",
        ]
    )
    OLLAMA_TIMEOUT: int = 300
    OLLAMA_NUM_CTX: int = 32768

    # LLM Providers
    DEFAULT_LLM_PROVIDER: Literal["openai", "anthropic", "ollama"] = "ollama"
    DEFAULT_MODEL: str = "llama3.1:70b"
    FALLBACK_MODEL: str = "llama3.1:8b"

    # API Keys
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    DEEPGRAM_API_KEY: str | None = None
    ELEVENLABS_API_KEY: str | None = None

    # Voice Settings
    WAKE_WORD: str = "jarvis"
    WAKE_WORD_SENSITIVITY: float = 0.5
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_DURATION: int = 250
    VAD_MAX_SPEECH_DURATION: int = 30000
    STT_PROVIDER: Literal["whisper", "deepgram", "whisper_local"] = "whisper_local"
    STT_MODEL: str = "large-v3"
    STT_LANGUAGE: str = "en"
    TTS_PROVIDER: Literal["edge", "elevenlabs", "piper", "coqui"] = "edge"
    TTS_VOICE: str = "en-US-AriaNeural"
    TTS_SPEED: float = 1.0
    TTS_PITCH: float = 0.0

    # Vision
    YOLO_MODEL: str = "yolov8x.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU: float = 0.45
    FACE_RECOGNITION_MODEL: str = "Facenet512"
    FACE_RECOGNITION_THRESHOLD: float = 0.6
    GESTURE_RECOGNITION_ENABLED: bool = True

    # Screen Capture
    SCREEN_CAPTURE_FPS: int = 10
    SCREEN_OCR_ENABLED: bool = True
    SCREEN_OCR_LANGUAGES: list[str] = ["eng"]

    # Memory
    MEMORY_SHORT_TERM_TURNS: int = 20
    MEMORY_WORKING_ITEMS: int = 50
    MEM0_API_KEY: str | None = None
    MEM0_LOCAL_MODE: bool = True
    CHROMADB_PERSIST_DIR: str = "/app/data/chromadb"
    GRAPHRAG_ENABLED: bool = True
    GRAPHRAG_COMMUNITY_LEVEL: int = 2

    # Agent Settings
    AGENT_TIMEOUT: int = 300
    AGENT_MAX_RETRIES: int = 3
    AGENT_MAX_DELEGATION_DEPTH: int = 3
    TOOL_EXECUTION_TIMEOUT: int = 120
    TOOL_MAX_RETRIES: int = 2

    # Sandbox
    SANDBOX_PROVIDER: Literal["docker", "e2b", "local"] = "docker"
    E2B_API_KEY: str | None = None
    SANDBOX_TIMEOUT: int = 300
    SANDBOX_MEMORY_LIMIT: str = "2g"
    SANDBOX_CPU_LIMIT: float = 2.0
    SANDBOX_NETWORK_ENABLED: bool = False

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 300
    CELERY_TASK_SOFT_TIME_LIMIT: int = 270

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    GRAFANA_ENABLED: bool = True
    GRAFANA_PORT: int = 3001
    GRAFANA_USER: str = "admin"
    GRAFANA_PASSWORD: str = "jarvis_grafana_password"

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"
    VITE_API_URL: str = "http://localhost:8000"
    VITE_WS_URL: str = "ws://localhost:8000"

    # Paths
    MODELS_DIR: Path = Path("/app/models")
    DATA_DIR: Path = Path("/app/data")
    LOGS_DIR: Path = Path("/app/logs")
    CACHE_DIR: Path = Path("/app/cache")
    TEMP_DIR: Path = Path("/tmp/jarvis")

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # WebSocket
    WS_PING_INTERVAL: int = 20
    WS_PING_TIMEOUT: int = 30
    WS_MAX_CONNECTIONS: int = 1000
    WS_MESSAGE_QUEUE_SIZE: int = 1000

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    @field_validator("MODELS_DIR", "DATA_DIR", "LOGS_DIR", "CACHE_DIR", "TEMP_DIR", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("OLLAMA_MODELS", mode="before")
    @classmethod
    def parse_ollama_models(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [model.strip() for model in v.split(",")]
        return v

    @field_validator("SCREEN_OCR_LANGUAGES", mode="before")
    @classmethod
    def parse_languages(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()

# Convenience exports
__all__ = ["Settings", "settings", "get_settings"]