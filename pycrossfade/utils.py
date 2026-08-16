"""Small utilities shared by the audio-processing modules."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


CACHE_FORMAT_VERSION = 1
BEAT_ANALYSIS_VERSION = 2
CACHE_ENV_VAR = 'PYCROSSFADE_CACHE_DIR'


def time_stretch(audio, factor, sample_rate=44100):
    import pyrubberband as pyrb
    return pyrb.time_stretch(audio, sample_rate, factor)


def load_audio(filepath):
    # returns loaded mono audio.
    from essentia.standard import MonoLoader
    return MonoLoader(filename=filepath)()


def save_audio(audio, filename, file_format='wav', bit_rate=320):
    from essentia.standard import MonoWriter
    MonoWriter(filename=filename, bitrate=bit_rate, format=file_format)(audio)


def content_hash(filepath, chunk_size=1024 * 1024):
    """Return a SHA-256 digest for the exact bytes in an audio file.

    A content hash deliberately does not include the filename or location, so
    copied or re-uploaded audio reuses the same analysis.
    """
    digest = hashlib.sha256()
    with open(filepath, 'rb') as audio_file:
        for chunk in iter(lambda: audio_file.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_beat_annotations(beats, downbeats, match_tolerance=0.05):
    """Return legacy ``[time_seconds, beat_number]`` annotations.

    Beat This! returns separate arrays for beat and downbeat times. The rest of
    pyCrossfade expects a two-column array where downbeats have beat number 1,
    so this adapter preserves that public representation.
    """
    beat_times = np.asarray(beats, dtype=float).reshape(-1)
    downbeat_times = np.asarray(downbeats, dtype=float).reshape(-1)
    if beat_times.size == 0:
        return np.empty((0, 2), dtype=float)

    beat_numbers = np.zeros(beat_times.size, dtype=int)
    for downbeat_time in downbeat_times:
        closest_index = np.abs(beat_times - downbeat_time).argmin()
        if abs(beat_times[closest_index] - downbeat_time) <= match_tolerance:
            beat_numbers[closest_index] = 1
    return np.column_stack((beat_times, beat_numbers))


def cache_directory(cache_dir=None):
    """Return the directory used for beat-analysis cache files.

    ``cache_dir`` is useful to applications that want to manage their own
    storage. The environment variable provides the same control for scripts.
    """
    configured_dir = cache_dir or os.environ.get(CACHE_ENV_VAR)
    if configured_dir:
        return Path(configured_dir).expanduser()
    return Path.home() / '.cache' / 'pycrossfade'


def cache_paths(cache_key, cache_dir=None):
    """Return data and metadata paths for a content-addressed cache entry."""
    directory = cache_directory(cache_dir)
    return directory / (cache_key + '.npz'), directory / (cache_key + '.json')


def load_cached_beats(cache_key, cache_dir=None):
    """Load cached beat data, or return ``None`` for a missing/invalid entry."""
    data_path, metadata_path = cache_paths(cache_key, cache_dir)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
            metadata = json.load(metadata_file)
        if (metadata.get('cache_format_version') != CACHE_FORMAT_VERSION or
                metadata.get('beat_analysis_version') != BEAT_ANALYSIS_VERSION):
            return None
        with np.load(data_path, allow_pickle=False) as cache_data:
            beats = cache_data['beats']
        if beats.ndim != 2 or beats.shape[1] != 2:
            return None
        return beats
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_cached_beats(cache_key, beats, filepath, cache_dir=None):
    """Atomically save beat data and descriptive metadata for a cache key."""
    data_path, metadata_path = cache_paths(cache_key, cache_dir)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    beats = np.asarray(beats)
    if beats.ndim != 2 or beats.shape[1] != 2:
        raise ValueError('Beat annotations must be a two-column array.')

    metadata = {
        'cache_format_version': CACHE_FORMAT_VERSION,
        'beat_analysis_version': BEAT_ANALYSIS_VERSION,
        'content_hash': cache_key,
        'source_filename': Path(filepath).name,
        'analysis': 'beat_this_downbeat_tracking',
    }
    _atomic_save_npz(data_path, beats)
    _atomic_write_json(metadata_path, metadata)


def _atomic_save_npz(destination, beats):
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=str(destination.parent), suffix='.npz'
    )
    try:
        with os.fdopen(file_descriptor, 'wb') as temporary_file:
            np.savez_compressed(temporary_file, beats=beats)
        os.replace(temporary_path, destination)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(destination, value):
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=str(destination.parent), suffix='.json'
    )
    try:
        with os.fdopen(file_descriptor, 'w', encoding='utf-8') as temporary_file:
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write('\n')
        os.replace(temporary_path, destination)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def linear_fade_volume(audio, start_volume=0.0, end_volume=1.0):
    import numpy as np

    if start_volume == end_volume:
        return audio

    length = audio.size
    profile = np.sqrt(np.linspace(start_volume, end_volume, length))
    return audio * profile


def linear_fade_filter(audio, filter_type, start_volume=0.0, end_volume=1.0):
    from yodel.filter import Biquad
    import numpy as np
    from scipy.signal import lfilter

    if start_volume == end_volume:
        return audio

    SAMPLE_RATE = 44100
    LOW_CUTOFF = 70
    MID_CENTER = 1000
    HIGH_CUTOFF = 13000
    Q = 1.0 / np.sqrt(2)
    NUM_STEPS = 20 if start_volume != end_volume else 1

    bquad_filter = Biquad()
    length = audio.size  # Assumes mono audio

    profile = np.linspace(start_volume, end_volume, NUM_STEPS)
    output_audio = np.zeros(audio.shape)

    for i in range(NUM_STEPS):
        start_idx = int((i / float(NUM_STEPS)) * length)
        end_idx = int(((i + 1) / float(NUM_STEPS)) * length)
        if filter_type == 'low_shelf':
            bquad_filter.low_shelf(SAMPLE_RATE, LOW_CUTOFF, Q, -int(26 * (1.0 - profile[i])))
        elif filter_type == 'high_shelf':
            bquad_filter.high_shelf(SAMPLE_RATE, HIGH_CUTOFF, Q, -int(26 * (1.0 - profile[i])))
        else:
            raise Exception('Unknown filter type: ' + filter_type)
        # ~ bquad_filter.process(audio[start_idx : end_idx], output_audio[start_idx : end_idx]) # This was too slow, code beneath is faster!
        b = bquad_filter._b_coeffs
        a = bquad_filter._a_coeffs
        a[
            0] = 1.0  # Normalizing the coefficients is already done in the yodel object, but a[0] is never reset to 1.0 after division!
        output_audio[start_idx: end_idx] = lfilter(b, a, audio[start_idx: end_idx]).astype('float32')

    return output_audio
