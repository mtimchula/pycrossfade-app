import asyncio
import hashlib
import json
from statistics import median
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

import models
from config import settings
from database import AsyncSessionLocal, get_db
from pycrossfade.song import Song
from pycrossfade.analysis import analyze_mix_features, peak_envelope
from pycrossfade.utils import load_audio
from pycrossfade.utils import cache_paths
from schemas import PaginatedTracksResponse, TrackResponse


router = APIRouter()
ALLOWED_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
CHUNK_SIZE = 1024 * 1024
ANALYSIS_CONCURRENCY = 1
analysis_slots = asyncio.Semaphore(ANALYSIS_CONCURRENCY)


@router.get("", response_model=PaginatedTracksResponse)
async def get_tracks(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = settings.tracks_per_page,
):
    count_result = await db.execute(select(func.count()).select_from(models.Track))
    total = count_result.scalar() or 0
    result = await db.execute(
        select(models.Track)
        .order_by(models.Track.created_at.desc())
        .offset(skip)
        .limit(limit),
    )
    tracks = result.scalars().all()
    return PaginatedTracksResponse(
        tracks=[TrackResponse.model_validate(track) for track in tracks],
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + len(tracks) < total,
    )


@router.get("/{track_id}", response_model=TrackResponse)
async def get_track(track_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(models.Track).where(models.Track.id == track_id))
    track = result.scalars().first()
    if not track:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track not found")
    return track


@router.get("/{track_id}/waveform")
async def get_track_waveform(
    track_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(select(models.Track).where(models.Track.id == track_id))
    track = result.scalars().first()
    if not track or track.status != "ready":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track is not ready")
    if not track.waveform_peaks:
        async with analysis_slots:
            audio = await run_in_threadpool(
                load_audio, str(settings.uploads_dir / track.stored_filename)
            )
            track.waveform_peaks = json.dumps(peak_envelope(audio, 180))
            await db.commit()
    return {"peaks": json.loads(track.waveform_peaks), "bpm": track.bpm}


@router.delete("/{track_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_track(track_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(
        select(models.Track)
        .options(selectinload(models.Track.mix_tracks))
        .where(models.Track.id == track_id)
    )
    track = result.scalars().first()
    if not track:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Track not found")
    if track.status in {"queued", "processing"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Wait for track analysis to finish before deleting it.",
        )
    if track.mix_tracks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Delete mixes containing this track before deleting the track.",
        )

    audio_path = settings.uploads_dir / track.stored_filename
    beat_data_path, beat_metadata_path = cache_paths(
        track.content_hash, settings.analysis_cache_dir
    )
    await db.delete(track)
    await db.commit()
    audio_path.unlink(missing_ok=True)
    beat_data_path.unlink(missing_ok=True)
    beat_metadata_path.unlink(missing_ok=True)


@router.post("", response_model=TrackResponse, status_code=status.HTTP_201_CREATED)
async def upload_track(
    background_tasks: BackgroundTasks,
    response: Response,
    file: UploadFile,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    original_filename = Path(file.filename or "upload").name
    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported audio files are MP3, WAV, FLAC, M4A, and OGG.",
        )

    temporary_path, content_hash, size_bytes = await save_upload(file)
    try:
        result = await db.execute(
            select(models.Track).where(models.Track.content_hash == content_hash)
        )
        existing_track = result.scalars().first()
        if existing_track:
            temporary_path.unlink(missing_ok=True)
            response.status_code = status.HTTP_200_OK
            return existing_track

        stored_filename = f"{content_hash}{suffix}"
        destination = settings.uploads_dir / stored_filename
        temporary_path.replace(destination)
        track = models.Track(
            content_hash=content_hash,
            original_filename=original_filename,
            stored_filename=stored_filename,
            media_type=file.content_type,
            size_bytes=size_bytes,
            status="queued",
        )
        db.add(track)
        await db.commit()
        await db.refresh(track)
        background_tasks.add_task(analyze_track, track.id)
        return track
    finally:
        await file.close()


async def save_upload(file: UploadFile) -> tuple[Path, str, int]:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    with NamedTemporaryFile(dir=settings.uploads_dir, delete=False, suffix=".part") as temporary_file:
        temporary_path = Path(temporary_file.name)
        while chunk := await file.read(CHUNK_SIZE):
            size_bytes += len(chunk)
            if size_bytes > settings.max_upload_size_bytes:
                temporary_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Upload exceeds the configured size limit.",
                )
            digest.update(chunk)
            temporary_file.write(chunk)
    return temporary_path, digest.hexdigest(), size_bytes


async def analyze_track(track_id: int):
    # Beat/key extraction is CPU intensive and the shared model does not gain
    # throughput from competing jobs. Keep later uploads queued instead.
    async with analysis_slots:
        await _analyze_track(track_id)


async def _analyze_track(track_id: int):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(models.Track).where(models.Track.id == track_id))
        track = result.scalars().first()
        if not track:
            return

        track.status = "processing"
        await db.commit()
        try:
            song = await run_in_threadpool(
                Song,
                str(settings.uploads_dir / track.stored_filename),
                settings.analysis_cache_dir,
            )
            track.duration_seconds = round(song.audio.size / song.sample_rate, 3)
            track.beat_count = len(song.beats)
            track.downbeat_count = len(song.get_downbeats())
            track.bpm = estimate_bpm(song.beats)
            for name, value in analyze_mix_features(song.audio, song.sample_rate).items():
                setattr(track, name, value)
            track.waveform_peaks = json.dumps(peak_envelope(song.audio, 180))
            track.status = "ready"
            track.analysis_error = None
        except Exception as error:
            track.status = "failed"
            track.analysis_error = str(error)[:500]
        await db.commit()


def estimate_bpm(beats) -> float | None:
    beat_times = [float(beat[0]) for beat in beats]
    if len(beat_times) < 2:
        return None
    intervals = [right - left for left, right in zip(beat_times, beat_times[1:])]
    valid_intervals = [interval for interval in intervals if 0.2 <= interval <= 1.5]
    if not valid_intervals:
        return None
    return round(60 / median(valid_intervals), 1)
