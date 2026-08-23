import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from pycrossfade import utils
from routers.mixes import build_mix, normalize_track_loudness
from routers.tracks import estimate_bpm
from pycrossfade.song import Song
from pycrossfade.transition import (
    crossfade_multiple,
    find_compatible_slave_start,
    time_stretch_gradually_in_downbeats,
    validate_song_for_transition,
)


class AudioQualityInvariantTests(unittest.TestCase):
    def test_loudness_normalization_leaves_silence_finite(self):
        audio = np.zeros(64, dtype=np.float32)

        normalized = normalize_track_loudness(audio)

        np.testing.assert_array_equal(audio, normalized)
        self.assertTrue(np.isfinite(normalized).all())

    def test_loudness_normalization_preserves_stereo_shape(self):
        audio = np.array([[-0.5, 0.25], [0.5, -0.25]], dtype=np.float32)

        normalized = normalize_track_loudness(audio, target_rms=0.2, peak_ceiling=0.4)

        self.assertEqual(audio.shape, normalized.shape)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 0.40001)

    def test_loudness_normalization_rejects_nonfinite_audio(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_track_loudness(np.array([0.0, np.nan], dtype=np.float32))

    def test_transition_validation_rejects_insufficient_downbeats(self):
        song = Song()
        song.audio = np.zeros(1000, dtype=np.float32)
        song.downbeats = np.array([0])

        with self.assertRaisesRegex(ValueError, "needs at least 9 downbeats"):
            validate_song_for_transition(song, 9, "Incoming")

    def test_multi_mix_requires_two_songs(self):
        with self.assertRaisesRegex(ValueError, "at least two songs"):
            crossfade_multiple([Song()], 8, 8)

    def test_candidate_search_rejects_median_safe_opening_with_local_outlier(self):
        master = np.array([0, 100, 200, 300, 400])
        slave = np.array([0, 100, 200, 210, 400, 500, 600, 700, 800])

        self.assertEqual(4, find_compatible_slave_start(master, slave, 4))

    def test_gradual_stretch_processes_trailing_audio_interval(self):
        song = Song()
        song.audio = np.arange(1000, dtype=np.float32)
        song.downbeats = np.array([0, 250, 500, 750])

        with patch("pycrossfade.transition.time_stretch", side_effect=lambda audio, factor: audio):
            result = time_stretch_gradually_in_downbeats(song, 1.1)

        np.testing.assert_array_equal(song.audio, result)

    def test_guarded_stereo_mix_returns_transition_manifest(self):
        master = Song()
        slave = Song()
        master.audio = np.ones((3000, 2), dtype=np.float32) * 0.1
        slave.audio = np.ones((3000, 2), dtype=np.float32) * 0.2
        master.downbeats = np.arange(0, 2100, 100)
        slave.downbeats = np.arange(0, 1050, 50)

        audio, manifest = crossfade_multiple(
            [master, slave],
            len_crossfade=8,
            len_time_stretch=8,
            return_manifest=True,
        )

        self.assertEqual(2, audio.shape[1])
        self.assertTrue(np.isfinite(audio).all())
        self.assertEqual(1, len(manifest))
        self.assertTrue(manifest[0]["rough_transition"])
        self.assertEqual(
            "no_candidate_with_all_bar_ratios_between_0.80_and_1.25",
            manifest[0]["fallback_reason"],
        )
        self.assertLess(manifest[0]["output_start_sample"], manifest[0]["output_end_sample"])

    def test_audio_io_preserves_stereo_wav(self):
        audio = np.column_stack((
            np.linspace(-0.5, 0.5, 1000),
            np.linspace(0.5, -0.5, 1000),
        )).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "stereo.wav"
            utils.save_audio(audio, path)
            loaded = utils.load_audio(path, mono=False)
            sample_rate, encoded = wavfile.read(path)

        self.assertEqual(44100, sample_rate)
        self.assertEqual((1000, 2), encoded.shape)
        self.assertEqual((1000, 2), loaded.shape)
        self.assertTrue(np.isfinite(loaded).all())

    def test_bpm_estimation_ignores_implausible_intervals(self):
        beats = np.array([
            [0.0, 1.0],
            [0.5, 0.0],
            [0.55, 0.0],
            [1.05, 1.0],
            [3.05, 0.0],
        ])

        self.assertEqual(120.0, estimate_bpm(beats))

    def test_bpm_estimation_requires_two_valid_beats(self):
        self.assertIsNone(estimate_bpm(np.array([[0.0, 1.0]])))

    def test_build_mix_limits_output_and_reports_duration(self):
        source_audio = np.ones((44100, 2), dtype=np.float32) * 0.5
        mixed_audio = np.ones((88200, 2), dtype=np.float32) * 2.0
        song = SimpleNamespace(audio=np.mean(source_audio, axis=1))
        manifest = [{"position": 0}]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            uploads = root / "uploads"
            mixes = root / "mixes"
            uploads.mkdir()
            saved = {}

            def capture_save(audio, output_path):
                saved["audio"] = np.asarray(audio)
                saved["path"] = Path(output_path)

            with (
                patch("routers.mixes.settings.uploads_dir", uploads),
                patch("routers.mixes.settings.mixes_dir", mixes),
                patch("routers.mixes.Song", return_value=song) as song_factory,
                patch("routers.mixes.load_audio", return_value=source_audio),
                patch(
                    "routers.mixes.crossfade_multiple",
                    return_value=(mixed_audio, manifest),
                ),
                patch("routers.mixes.save_audio", side_effect=capture_save),
            ):
                filename, duration, transitions = build_mix(["first.wav", "second.wav"])

        self.assertEqual(2.0, duration)
        self.assertTrue(filename.startswith("mix-"))
        self.assertTrue(filename.endswith(".wav"))
        self.assertEqual(2, song_factory.call_count)
        self.assertEqual(saved["path"].name, filename)
        self.assertEqual((88200, 2), saved["audio"].shape)
        self.assertLessEqual(float(np.max(np.abs(saved["audio"]))), 0.80001)
        self.assertTrue(np.isfinite(saved["audio"]).all())
        self.assertEqual("first.wav", transitions[0]["outgoing_filename"])
        self.assertEqual("second.wav", transitions[0]["incoming_filename"])


if __name__ == "__main__":
    unittest.main()
