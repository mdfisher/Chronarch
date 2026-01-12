"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Chronarch"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "chronarch"
    postgres_password: str = Field(default="chronarch")
    postgres_db: str = "chronarch"

    @computed_field
    @property
    def database_url(self) -> str:
        """Construct the database URL from components."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Construct the sync database URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Binance API
    binance_base_url: str = "https://api.binance.com"
    binance_api_key: str | None = None
    binance_api_secret: str | None = None

    # Data ingestion
    default_symbol: str = "BTCUSDT"
    candle_intervals: list[str] = Field(default=["1m", "5m", "1h", "1d"])
    historical_days: int = 365  # Days of historical data to fetch on startup

    # ML model
    model_dir: Path = Path("models")
    prediction_horizon_hours: int = 1
    direction_threshold: float = 0.001  # 0.1% price change for up/down classification

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default=["*"])


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
