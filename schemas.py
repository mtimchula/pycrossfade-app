from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TrackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    content_hash: str
    original_filename: str
    media_type: str | None
    size_bytes: int
    status: str
    duration_seconds: float | None
    bpm: float | None
    key: str | None
    key_scale: str | None
    key_strength: float | None
    loudness_db: float | None
    energy_start: float | None
    energy_end: float | None
    brightness: float | None
    beat_count: int | None
    downbeat_count: int | None
    analysis_error: str | None
    created_at: datetime


class PaginatedTracksResponse(BaseModel):
    tracks: list[TrackResponse]
    total: int
    skip: int
    limit: int
    has_more: bool


class MixCreate(BaseModel):
    track_ids: list[int] = Field(min_length=2)
    auto_order: bool = True
    horn_rough_transitions: bool = False


class MixResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    output_filename: str | None
    duration_seconds: float | None
    bpm: float | None
    horn_rough_transitions: bool
    generation_error: str | None
    created_at: datetime
