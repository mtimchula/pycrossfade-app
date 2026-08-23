from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    key: Mapped[str | None] = mapped_column(String(4), nullable=True)
    key_scale: Mapped[str | None] = mapped_column(String(10), nullable=True)
    key_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    loudness_db: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    brightness: Mapped[float | None] = mapped_column(Float, nullable=True)
    waveform_peaks: Mapped[str | None] = mapped_column(Text, nullable=True)
    beat_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    downbeat_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    analysis_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    mix_tracks: Mapped[list[MixTrack]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
    )


class Mix(Base):
    __tablename__ = "mixes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    output_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    bpm: Mapped[float | None] = mapped_column(Float, nullable=True)
    horn_rough_transitions: Mapped[bool] = mapped_column(Boolean, default=False)
    generation_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    transition_manifest: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    tracks: Mapped[list[MixTrack]] = relationship(
        back_populates="mix",
        cascade="all, delete-orphan",
    )


class MixTrack(Base):
    __tablename__ = "mix_tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mix_id: Mapped[int] = mapped_column(ForeignKey("mixes.id"), index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    mix: Mapped[Mix] = relationship(back_populates="tracks")
    track: Mapped[Track] = relationship(back_populates="mix_tracks")
