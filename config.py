from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "sqlite+aiosqlite:///./pycrossfade.db"
    media_dir: Path = Path("media")
    uploads_dir: Path = Path("media/uploads")
    mixes_dir: Path = Path("media/mixes")
    analysis_cache_dir: Path = Path("media/cache")
    max_upload_size_bytes: int = 250 * 1024 * 1024
    tracks_per_page: int = 10


settings = Settings()
