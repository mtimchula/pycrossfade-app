from contextlib import asynccontextmanager
from statistics import median

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload

import models  # noqa: F401 - registers SQLAlchemy models with Base metadata
from config import settings
from database import AsyncSessionLocal, Base, engine
from pycrossfade.utils import load_cached_beats
from routers import mixes, tracks


async def upgrade_sqlite_schema():
    """Apply additive columns needed by existing local SQLite libraries."""
    async with engine.begin() as connection:
        additions = {
            "tracks": {
                "bpm": "FLOAT", "key": "VARCHAR(4)", "key_scale": "VARCHAR(10)",
                "key_strength": "FLOAT", "loudness_db": "FLOAT",
                "energy_start": "FLOAT", "energy_end": "FLOAT", "brightness": "FLOAT",
                "waveform_peaks": "TEXT",
            },
            "mixes": {"bpm": "FLOAT", "horn_rough_transitions": "BOOLEAN DEFAULT 0"},
        }
        for table, wanted_columns in additions.items():
            columns = {
                row[1]
                for row in (await connection.execute(text(f"PRAGMA table_info({table})"))).all()
            }
            for column, column_type in wanted_columns.items():
                if column not in columns:
                    await connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"))


def bpm_from_tracks(mix: models.Mix) -> float | None:
    values = []
    for item in mix.tracks:
        if not item.track:
            continue
        if item.track.bpm is not None:
            values.append(item.track.bpm)
            continue
        beats = load_cached_beats(item.track.content_hash, settings.analysis_cache_dir)
        if beats is None or len(beats) < 2:
            continue
        intervals = [float(right[0] - left[0]) for left, right in zip(beats, beats[1:])]
        valid_intervals = [interval for interval in intervals if 0.2 <= interval <= 1.5]
        if valid_intervals:
            item.track.bpm = round(60 / median(valid_intervals), 1)
            values.append(item.track.bpm)
    return round(median(values), 1) if values else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.mixes_dir.mkdir(parents=True, exist_ok=True)
    settings.analysis_cache_dir.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    await upgrade_sqlite_schema()
    yield
    await engine.dispose()


app = FastAPI(title="FastAPI pyCrossfade", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/media", StaticFiles(directory="media"), name="media")

templates = Jinja2Templates(directory="templates")
app.include_router(tracks.router, prefix="/api/tracks", tags=["tracks"])
app.include_router(mixes.router, prefix="/api/mixes", tags=["mixes"])


@app.get("/", include_in_schema=False, name="home")
async def home(request: Request):
    async with AsyncSessionLocal() as session:
        track_result = await session.execute(
            select(models.Track).order_by(models.Track.created_at.desc()),
        )
        mix_result = await session.execute(
            select(models.Mix)
            .options(selectinload(models.Mix.tracks).selectinload(models.MixTrack.track))
            .order_by(models.Mix.created_at.desc()),
        )
        tracks_for_page = track_result.scalars().all()
        mixes_for_page = mix_result.scalars().all()
        for mix in mixes_for_page:
            if mix.bpm is None:
                mix.bpm = bpm_from_tracks(mix)
        await session.commit()

    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "title": "Music library",
            "tracks": tracks_for_page,
            "total": len(tracks_for_page),
            "mixes": mixes_for_page,
        },
    )
