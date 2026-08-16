# FastAPI pyCrossfade

FastAPI pyCrossfade is a music-library application built around beat-matched crossfades. It preserves the original pyCrossfade audio-processing library while evolving it toward a local FastAPI application for uploading tracks, storing analysis metadata, generating mixes, playing them in the browser, and downloading the resulting WAV files.

## Fork and attribution

This repository is a fork of [Oğuzhan Yılmaz’s pyCrossfade](https://github.com/oguzhan-yilmaz/pyCrossfade). The original project introduced the beat-matched crossfade implementation that this work builds on.

The original MIT license and copyright notice are retained in [LICENSE](LICENSE).

## What changed in this fork

- Modern Python 3.14 project management with `uv` and a committed lockfile.
- A content-addressed beat-analysis cache: identical audio reuses analysis regardless of filename or location.
- Beat This! replaces the unmaintained Madmom dependency for beat and downbeat detection.
- Safer transition planning that avoids extreme tempo ramps by selecting a compatible incoming-song window when necessary.
- FastAPI, SQLAlchemy, and SQLite foundations for a persistent song library.
- Track and mix data models keyed by a song content hash, including analysis and generation status.
- Static templates, JavaScript, and CSS for a Cover-style upload interface, mix generation, an in-page audio player, and downloads.

## Requirements

Install the system tools required by audio processing:

```bash
# macOS
brew install libsndfile rubberband ffmpeg

# Debian/Ubuntu
sudo apt-get install libsndfile1 rubberband-cli ffmpeg
```

Install [uv](https://docs.astral.sh/uv/), then create the environment:

```bash
uv python install 3.14
uv sync
```

## Run the application

```bash
uv run fastapi dev main.py
```

Open `http://127.0.0.1:8000`.

The intended application workflow is:

1. Upload audio tracks.
2. Wait for beat analysis to complete.
3. Select two or more ready tracks in order.
4. Generate a mix.
5. Play or download the completed WAV from the mix list.

Uploads and generated files belong in `media/`, which is intentionally ignored by Git. `pyproject.toml` and `uv.lock` are the reproducible dependency source of truth.

## Library usage

The original Python API remains available through the `pycrossfade` import package:

```python
from pycrossfade.song import Song
from pycrossfade.transition import crossfade_multiple
from pycrossfade.utils import save_audio

songs = [
    Song("/path/to/first.mp3"),
    Song("/path/to/second.mp3"),
    Song("/path/to/third.mp3"),
]

audio = crossfade_multiple(songs, len_crossfade=8, len_time_stretch=8)
save_audio(audio, "/path/to/mix.wav")
```

Set `PYCROSSFADE_CACHE_DIR` or pass `cache_dir` to `Song` to choose the analysis-cache location. Cache entries are safe to delete and will be recreated as needed.

## Development

```bash
uv run python -m unittest discover -s tests -v
uv build
```
