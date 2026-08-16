from pathlib import Path
from statistics import median
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
import numpy as np
from scipy.io import wavfile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

import models
from config import settings
from database import AsyncSessionLocal, get_db
from pycrossfade.song import Song
from pycrossfade.analysis import peak_envelope
from pycrossfade.transition import crossfade_multiple
from pycrossfade.sequencing import optimize_track_order
from pycrossfade.utils import save_audio
from schemas import MixCreate, MixResponse


router = APIRouter()


@router.post("", response_model=MixResponse, status_code=status.HTTP_201_CREATED)
async def create_mix(
    mix_data: MixCreate,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if len(set(mix_data.track_ids)) != len(mix_data.track_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Select each track once.")

    result = await db.execute(
        select(models.Track).where(models.Track.id.in_(mix_data.track_ids))
    )
    tracks_by_id = {track.id: track for track in result.scalars().all()}
    if len(tracks_by_id) != len(mix_data.track_ids):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more tracks were not found.")
    if any(tracks_by_id[track_id].status != "ready" for track_id in mix_data.track_ids):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Wait until every selected track has finished analysis.",
        )

    selected_tracks = [tracks_by_id[track_id] for track_id in mix_data.track_ids]
    if mix_data.auto_order:
        selected_tracks = optimize_track_order(selected_tracks)

    mix = models.Mix(
        status="queued",
        horn_rough_transitions=mix_data.horn_rough_transitions,
    )
    mix.tracks = [
        models.MixTrack(track_id=track.id, position=position)
        for position, track in enumerate(selected_tracks)
    ]
    db.add(mix)
    await db.commit()
    await db.refresh(mix)
    background_tasks.add_task(generate_mix, mix.id)
    return mix


@router.get("/{mix_id}", response_model=MixResponse)
async def get_mix(mix_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(models.Mix).where(models.Mix.id == mix_id))
    mix = result.scalars().first()
    if not mix:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mix not found")
    return mix


@router.delete("/{mix_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_mix(mix_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(
        select(models.Mix)
        .options(selectinload(models.Mix.tracks))
        .where(models.Mix.id == mix_id)
    )
    mix = result.scalars().first()
    if not mix:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mix not found")
    if mix.status in {"queued", "processing"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Wait for mix generation to finish before deleting it.",
        )

    output_path = settings.mixes_dir / mix.output_filename if mix.output_filename else None
    await db.delete(mix)
    await db.commit()
    if output_path:
        output_path.unlink(missing_ok=True)


@router.get("/{mix_id}/download", response_class=FileResponse)
async def download_mix(mix_id: int, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(models.Mix).where(models.Mix.id == mix_id))
    mix = result.scalars().first()
    if not mix or mix.status != "ready" or not mix.output_filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mix is not ready")

    output_path = settings.mixes_dir / mix.output_filename
    if not output_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mix file is missing")
    return FileResponse(output_path, media_type="audio/wav", filename=mix.output_filename)


@router.get("/{mix_id}/waveform")
async def get_mix_waveform(
    mix_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    samples: Annotated[int, Query(ge=48, le=600)] = 180,
):
    """Return a compact, normalized peak envelope for the browser player."""
    result = await db.execute(select(models.Mix).where(models.Mix.id == mix_id))
    mix = result.scalars().first()
    if not mix or mix.status != "ready" or not mix.output_filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mix is not ready")

    output_path = settings.mixes_dir / mix.output_filename
    if not output_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mix file is missing")
    _, audio = await run_in_threadpool(wavfile.read, output_path)
    return {"peaks": peak_envelope(audio, samples), "bpm": mix.bpm}


async def generate_mix(mix_id: int):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(models.Mix)
            .options(selectinload(models.Mix.tracks).selectinload(models.MixTrack.track))
            .where(models.Mix.id == mix_id),
        )
        mix = result.scalars().first()
        if not mix:
            return

        mix.status = "processing"
        await db.commit()
        try:
            ordered_tracks = sorted(mix.tracks, key=lambda item: item.position)
            output_filename, duration_seconds = await run_in_threadpool(
                build_mix,
                [track_item.track.stored_filename for track_item in ordered_tracks],
                mix.horn_rough_transitions,
            )
            mix.output_filename = output_filename
            mix.duration_seconds = duration_seconds
            bpm_values = [item.track.bpm for item in ordered_tracks if item.track.bpm]
            mix.bpm = round(median(bpm_values), 1) if bpm_values else None
            mix.status = "ready"
            mix.generation_error = None
        except Exception as error:
            mix.status = "failed"
            mix.generation_error = str(error)[:500]
        await db.commit()


def build_mix(stored_filenames: list[str], horn_rough_transitions=False) -> tuple[str, float]:
    songs = [
        Song(
            str(settings.uploads_dir / stored_filename),
            cache_dir=settings.analysis_cache_dir,
        )
        for stored_filename in stored_filenames
    ]
    for song in songs:
        song.audio = normalize_track_loudness(song.audio)
    audio = crossfade_multiple(
        songs,
        len_crossfade=8,
        len_time_stretch=8,
        overlay_rough_transitions=horn_rough_transitions,
    )
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.80:
        audio = audio * (0.80 / peak)
    settings.mixes_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"mix-{uuid4().hex}.wav"
    save_audio(audio, str(settings.mixes_dir / output_filename))
    return output_filename, round(audio.size / 44100, 3)


def normalize_track_loudness(audio, target_rms: float = 0.18, peak_ceiling: float = 0.80):
    """Match source levels before mixing while retaining each song's dynamics."""
    values = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(values)) + 1e-12))
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if not rms or not peak:
        return values
    gain = min(target_rms / rms, peak_ceiling / peak)
    return values * gain
