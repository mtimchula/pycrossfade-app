"""Lightweight musical features used to plan transitions between tracks."""

from __future__ import annotations

import numpy as np


def analyze_mix_features(audio, sample_rate: int = 44100) -> dict[str, float | str | None]:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not values.size:
        return {
            "key": None, "key_scale": None, "key_strength": None,
            "loudness_db": -120.0, "energy_start": 0.0, "energy_end": 0.0,
            "brightness": 0.0,
        }

    peak = float(np.max(np.abs(values))) or 1.0
    normalized = values / peak
    frame_length = max(1, sample_rate // 2)
    usable = normalized[: (normalized.size // frame_length) * frame_length]
    if usable.size:
        frames = usable.reshape(-1, frame_length)
        rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    else:
        rms = np.array([np.sqrt(np.mean(np.square(normalized)) + 1e-12)])
    edge_frames = max(1, min(len(rms), int(30 / 0.5)))
    loudness_db = float(20 * np.log10(np.sqrt(np.mean(np.square(normalized))) + 1e-12))

    # A few short FFT windows provide a stable brightness descriptor cheaply.
    window_size = min(values.size, sample_rate * 12)
    starts = np.linspace(0, max(0, values.size - window_size), min(5, max(1, values.size // window_size)), dtype=int)
    brightness_values = []
    frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
    for start in starts:
        spectrum = np.abs(np.fft.rfft(normalized[start:start + window_size] * np.hanning(window_size)))
        brightness_values.append(float(np.sum(frequencies * spectrum) / (np.sum(spectrum) + 1e-12)))

    key = scale = None
    key_strength = None
    try:
        from essentia.standard import KeyExtractor
        key, scale, key_strength = KeyExtractor()(normalized)
    except Exception:
        pass

    return {
        "key": key,
        "key_scale": scale,
        "key_strength": round(float(key_strength), 4) if key_strength is not None else None,
        "loudness_db": round(loudness_db, 3),
        "energy_start": round(float(np.mean(rms[:edge_frames])), 5),
        "energy_end": round(float(np.mean(rms[-edge_frames:])), 5),
        "brightness": round(float(np.mean(brightness_values)), 1),
    }


def peak_envelope(audio, sample_count: int) -> list[float]:
    """Reduce mono or stereo audio to display-ready normalized peak amplitudes."""
    values = np.asarray(audio, dtype=float)
    if values.ndim > 1:
        values = np.max(np.abs(values), axis=1)
    else:
        values = np.abs(values)
    if not values.size:
        return [0.0] * sample_count

    boundaries = np.linspace(0, values.size, sample_count + 1, dtype=int)
    peaks = np.array([
        values[left:right].max() if right > left else 0.0
        for left, right in zip(boundaries, boundaries[1:])
    ])
    scale = np.percentile(peaks, 98) or peaks.max() or 1.0
    return np.clip(peaks / scale, 0.03, 1.0).round(3).tolist()
