import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pycrossfade import utils
from pycrossfade.song import Song
from pycrossfade.transition import (
    air_horn,
    fallback_crossfade,
    find_compatible_slave_start,
    find_outro_crossfade_start,
    safe_tempo_window,
)
from pycrossfade.sequencing import optimize_track_order, transition_score
from pycrossfade.analysis import peak_envelope
from routers.mixes import normalize_track_loudness
from types import SimpleNamespace


class ContentAddressedCacheTests(unittest.TestCase):
    def test_outro_crossfade_skips_a_silent_tail(self):
        audio = np.concatenate((np.ones(800), np.zeros(200))).astype(np.float32)
        downbeats = np.arange(0, 1000, 100)
        start = find_outro_crossfade_start(
            audio, downbeats, fade_length=200, sample_rate=100,
            max_search_seconds=10,
        )
        self.assertEqual(600, start)

    def test_tempo_window_rejects_outlier_hidden_by_safe_median(self):
        ratios = [0.504, 1.0, 1.131, 8.625, 7.667, 1.15, 1.0, 0.504]
        self.assertFalse(safe_tempo_window(ratios))

    def test_tempo_window_accepts_small_per_bar_corrections(self):
        self.assertTrue(safe_tempo_window([.98, 1.01, 1.0, 1.04]))

    def test_unsafe_tempo_fallback_does_not_change_total_duration(self):
        master = Song()
        slave = Song()
        master.audio = np.zeros(1000, dtype=np.float32)
        slave.audio = np.zeros(1200, dtype=np.float32)
        master.downbeats = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800])
        slave.downbeats = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400])
        result = fallback_crossfade(master, slave, 4, 0, True, False, 2.0)
        self.assertEqual(len(master.audio) + len(slave.audio) - 400, len(result))

    def test_air_horn_is_bounded_and_fades(self):
        horn = air_horn(44100)
        self.assertEqual(44100, len(horn))
        self.assertLessEqual(float(np.max(np.abs(horn))), .2)
        self.assertLess(abs(float(horn[-1])), .03)

    def test_loudness_normalization_respects_peak_ceiling(self):
        audio = np.array([-.8, -.1, .1, .8], dtype=np.float32)
        normalized = normalize_track_loudness(audio, target_rms=.5, peak_ceiling=.9)
        self.assertLessEqual(float(np.max(np.abs(normalized))), .90001)

    def make_track(self, track_id, bpm, key, scale, start, end, loudness=-14, brightness=2500):
        return SimpleNamespace(
            id=track_id, bpm=bpm, key=key, key_scale=scale,
            energy_start=start, energy_end=end, loudness_db=loudness,
            brightness=brightness,
        )

    def test_directional_score_rewards_smooth_tempo_key_and_energy(self):
        first = self.make_track(1, 174, "A", "minor", .20, .32)
        smooth = self.make_track(2, 176, "E", "minor", .31, .45)
        disruptive = self.make_track(3, 128, "C#", "major", .75, .50, -5, 7000)
        self.assertGreater(transition_score(first, smooth), transition_score(first, disruptive))

    def test_optimizer_chooses_the_smoother_complete_route(self):
        low = self.make_track(1, 174, "A", "minor", .12, .22)
        middle = self.make_track(2, 175, "E", "minor", .21, .34)
        peak = self.make_track(3, 176, "B", "minor", .33, .52)
        self.assertEqual([1, 2, 3], [track.id for track in optimize_track_order([peak, low, middle])])

    def test_peak_envelope_is_normalized_and_has_requested_size(self):
        audio = np.array([0, 1, -2, 4, -8, 2, 1, 0], dtype=float)
        peaks = peak_envelope(audio, 4)
        self.assertEqual(4, len(peaks))
        self.assertTrue(all(0.0 <= peak <= 1.0 for peak in peaks))
        self.assertEqual(1.0, max(peaks))

    def test_peak_envelope_accepts_stereo_audio(self):
        audio = np.array([[0, 0], [1, -3], [2, 1], [0, 0]], dtype=float)
        self.assertEqual(2, len(peak_envelope(audio, 2)))

    def test_transition_uses_a_stable_later_slave_window_for_extreme_tempo_gap(self):
        master_dbeats = np.array([0, 138, 276, 414, 552])
        slave_dbeats = np.array([
            0, 34, 136, 172, 274, 310, 412, 550, 688, 826, 964,
        ])
        self.assertEqual(
            6,
            find_compatible_slave_start(master_dbeats, slave_dbeats, len_crossfade=4),
        )

    def test_transition_keeps_compatible_song_opening(self):
        master_dbeats = np.array([0, 138, 276, 414, 552])
        slave_dbeats = np.array([0, 140, 280, 420, 560])
        self.assertEqual(
            0,
            find_compatible_slave_start(master_dbeats, slave_dbeats, len_crossfade=4),
        )

    def test_song_name_parsing_preserves_periods_in_the_title(self):
        song = Song()
        song.filepath = '/music/DJ Example feat. Artist - Track v1.2.mp3'
        self.assertEqual(
            ('DJ Example feat. Artist - Track v1.2', 'mp3'),
            song.get_song_name_and_format(),
        )

    def test_beat_this_output_adapts_to_legacy_annotation_format(self):
        annotations = utils.make_beat_annotations(
            beats=[0.0, 0.5, 1.0, 1.5], downbeats=[0.0, 1.0]
        )
        np.testing.assert_array_equal(
            annotations,
            np.array([[0.0, 1.0], [0.5, 0.0], [1.0, 1.0], [1.5, 0.0]]),
        )

    def test_same_bytes_have_same_key_and_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_file = root / 'first-name.mp3'
            second_file = root / 'other-name.wav'
            first_file.write_bytes(b'audio bytes')
            second_file.write_bytes(b'audio bytes')

            cache_key = utils.content_hash(first_file)
            self.assertEqual(cache_key, utils.content_hash(second_file))

            beats = np.array([[0.0, 1.0], [0.5, 2.0]])
            utils.save_cached_beats(cache_key, beats, first_file, root / 'cache')

            np.testing.assert_array_equal(
                beats, utils.load_cached_beats(cache_key, root / 'cache')
            )
            _, metadata_path = utils.cache_paths(cache_key, root / 'cache')
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual('first-name.mp3', metadata['source_filename'])

    def test_changed_bytes_have_a_different_key(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_file = Path(temporary_directory) / 'song.mp3'
            audio_file.write_bytes(b'first version')
            first_key = utils.content_hash(audio_file)
            audio_file.write_bytes(b'second version')
            self.assertNotEqual(first_key, utils.content_hash(audio_file))


if __name__ == '__main__':
    unittest.main()
