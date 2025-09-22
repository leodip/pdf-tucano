from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application configuration stored in environment variables."""

    database_url: str = Field(..., env="DATABASE_URL")
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_model: str = Field("google/gemini-2.0-flash-001", env="OPENROUTER_MODEL")

    # Processing configuration
    page_image_width: int = Field(1500, env="PAGE_IMAGE_WIDTH")
    page_image_dpi: int = Field(300, env="PAGE_IMAGE_DPI")
    max_concurrent_pages: int = Field(8, env="MAX_CONCURRENT_PAGES")
    job_poll_interval_seconds: float = Field(2.0, env="JOB_POLL_INTERVAL_SECONDS")
    job_idle_sleep_seconds: float = Field(5.0, env="JOB_IDLE_SLEEP_SECONDS")
    page_heading_label: str = Field("", env="PAGE_HEADING_LABEL")

    storage_root: Path = Field(Path("/data"), env="STORAGE_ROOT")
    pdf_subdir: str = Field("pdfs", env="PDF_SUBDIR")
    cleanup_completed_after_hours: int = Field(72, env="CLEANUP_COMPLETED_AFTER_HOURS")

    # Debugging / observability
    generation_stats_probe_enabled: bool = Field(False, env="GENERATION_STATS_PROBE_ENABLED")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("storage_root", pre=True)
    def _expand_storage_root(cls, value: Optional[str] | Path) -> Path:
        if value is None:
            return Path("/data")
        return Path(value).expanduser().resolve()


@lru_cache()
def get_settings() -> Settings:
    return Settings()
