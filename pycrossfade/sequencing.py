"""Directed track compatibility and mix-order optimization."""

from __future__ import annotations

import math


PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def harmonic_compatibility(first, second) -> float:
    if first.key not in PITCH_CLASS or second.key not in PITCH_CLASS:
        return 0.5
    a, b = PITCH_CLASS[first.key], PITCH_CLASS[second.key]
    distance = min((a - b) % 12, (b - a) % 12)
    same_scale = first.key_scale == second.key_scale
    if distance == 0:
        return 1.0 if same_scale else 0.86
    if distance == 7 and same_scale:  # adjacent on the circle of fifths
        return 0.9
    if distance in (3, 4) and not same_scale:  # relative major/minor neighborhood
        return 0.88
    return max(0.0, 0.72 - distance * 0.12)


def tempo_compatibility(first_bpm, second_bpm) -> float:
    if not first_bpm or not second_bpm:
        return 0.5
    ratios = [second_bpm / first_bpm, (second_bpm * 2) / first_bpm, (second_bpm / 2) / first_bpm]
    distance = min(abs(math.log2(ratio)) for ratio in ratios if ratio > 0)
    return max(0.0, 1.0 - distance / math.log2(1.12))


def transition_score(first, second) -> float:
    """Score a directional A -> B join on a 0–100 scale."""
    tempo = tempo_compatibility(first.bpm, second.bpm)
    harmony = harmonic_compatibility(first, second)
    outgoing = first.energy_end if first.energy_end is not None else 0.2
    incoming = second.energy_start if second.energy_start is not None else 0.2
    energy = max(0.0, 1.0 - abs(outgoing - incoming) / max(outgoing, incoming, 0.08))
    loudness = max(0.0, 1.0 - abs((first.loudness_db or -14) - (second.loudness_db or -14)) / 12)
    brightness = max(0.0, 1.0 - abs((first.brightness or 2500) - (second.brightness or 2500)) / 6000)
    # Slight upward-energy preference gives a natural set arc without forcing it.
    direction = 1.0 if incoming >= outgoing * 0.82 else 0.65
    return round(100 * (0.34 * tempo + 0.24 * harmony + 0.20 * energy + 0.10 * loudness + 0.07 * brightness + 0.05 * direction), 2)


def optimize_track_order(tracks, beam_width: int = 256):
    """Find a strong complete directed route while keeping runtime bounded."""
    tracks = list(tracks)
    if len(tracks) < 3:
        return tracks
    by_id = {track.id: track for track in tracks}
    scores = {(a.id, b.id): transition_score(a, b) for a in tracks for b in tracks if a.id != b.id}
    # Prefer a lower-energy opener, then explore best continuations.
    starts = sorted(tracks, key=lambda t: (t.energy_start if t.energy_start is not None else 0.2, t.id))
    beams = [([track.id], 0.0) for track in starts]
    while len(beams[0][0]) < len(tracks):
        candidates = []
        for route, total in beams:
            remaining = by_id.keys() - route
            for track_id in remaining:
                candidates.append((route + [track_id], total + scores[(route[-1], track_id)]))
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_width]
    return [by_id[track_id] for track_id in beams[0][0]]
