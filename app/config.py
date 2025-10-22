import os
from typing import Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    app_name: str = Field(default="Voice Assistant AI Backend", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    version: str = Field(default="1.0.0", env="VERSION")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database settings
    database_url: str = Field(..., env="DATABASE_URL")
    database_host: str = Field(default="localhost", env="DB_HOST")
    database_port: int = Field(default=5432, env="DB_PORT")
    database_name: str = Field(default="voice_ai_db", env="DB_NAME")
    database_user: str = Field(default="postgres", env="DB_USER")
    database_password: str = Field(default="root", env="DB_PASSWORD")
    
    # Connection pool settings for optimal performance
    database_min_connections: int = Field(default=5, env="DB_MIN_CONNECTIONS")
    database_max_connections: int = Field(default=20, env="DB_MAX_CONNECTIONS")
    database_pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    database_pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    
    # Redis settings (for caching and session management)
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # OpenAI settings
    openai_api_key: str = Field(default="your_openai_api_key_here", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=150, env="OPENAI_MAX_TOKENS")
    
    # Whisper settings
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")
    
    # ElevenLabs settings
    elevenlabs_api_key: str = Field(default="your_elevenlabs_api_key_here", env="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", env="ELEVENLABS_VOICE_ID")
    elevenlabs_model: str = Field(default="eleven_turbo_v2", env="ELEVENLABS_MODEL")
    
    # WebSocket settings
    websocket_timeout: int = Field(default=60, env="WEBSOCKET_TIMEOUT")
    websocket_max_connections: int = Field(default=100, env="WEBSOCKET_MAX_CONNECTIONS")
    
    # Security settings
    secret_key: str = Field(default="your_super_secret_key_here_change_this_in_production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    
    # CORS settings - stored as string for environment variable compatibility
    allowed_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        env="ALLOWED_ORIGINS",
        description="Comma-separated list of allowed origins for CORS"
    )
    
    @property
    def allowed_origins(self) -> list[str]:
        """Get the list of allowed origins."""
        return [origin.strip() for origin in self.allowed_origins_str.split(",") if origin.strip()]
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    @property
    def async_database_url(self) -> str:
        """Get async PostgreSQL URL for asyncpg."""
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        return self.database_url
    
    @property
    def sync_database_url(self) -> str:
        """Get sync PostgreSQL URL for migrations."""
        if self.database_url.startswith("postgresql+asyncpg://"):
            return self.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return self.database_url
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env file


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()