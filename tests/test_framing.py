"""Hợp đồng 25/10 ms và đối chiếu lịch sử 20/10 bằng fixture tính tay."""
import contextlib
import io
import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from algorithms import tt2_histogram as tt2
from app.pipeline import context_benchmark, noisy_copy, prepare_records, run_experiment
from core.features import compute_features
from core.postprocess import fill_short_internal_silences


def audio_record(name="fixture", fs=1000, samples=None):
    """Fixture thật: 100 ms có speech giữa 30 và 70 ms."""
    samples = samples if samples is not None else [0.0] * 30 + [1.0] * 40 + [0.0] * 30
    return dict(name=name, samples=samples, sample_rate=fs,
                duration=len(samples) / fs,
                intervals=[(0.0, .03, "sil"), (.03, .07, "v"), (.07, len(samples) / fs, "sil")])


def two_impulse_record():
    """Hai speech gốc; padding cho [15,65] và [87,137], gap support 195 ms."""
    values = [0.0] * 160
    values[40] = values[112] = 1.0
    return {"features": {"energy": values, "centroid": list(values),
                         "frame_ms": 25.0, "hop_ms": 10.0,
                         "starts": [i / 100 for i in range(160)],
                         "ends": [(i * 10 + 25) / 1000 for i in range(160)]},
            "labels": [0] * 15 + [1] * 123 + [0] * 22, "duration": 1.62}


class FramingTests(unittest.TestCase):
    def test_default_full_frames_have_25ms_support_and_10ms_steps(self):
        # Đổi default về 20 ms hoặc thêm tail padding sẽ sai 8 khung này.
        features = compute_features([1.0] * 100, 1000)
        self.assertEqual(features["starts"], [0, .01, .02, .03, .04, .05, .06, .07])
        self.assertEqual(features["ends"], [.025, .035, .045, .055, .065, .075, .085, .095])
        self.assertEqual(features["ste"], [25.0] * 8)
        self.assertEqual((features["frame_size"], features["hop_size"]), (25, 10))
        self.assertEqual((features["frame_ms"], features["hop_ms"]), (25.0, 10.0))

    def test_16khz_support_count_uses_400_and_160_samples(self):
        features = compute_features([1.0] * 1600, 16000)
        self.assertEqual(len(features["starts"]), 8)
        self.assertEqual((features["frame_size"], features["hop_size"]), (400, 160))
        self.assertEqual(features["ends"][-1], .095)

    def test_44100hz_metadata_describes_actual_rounded_sample_support(self):
        features = compute_features([1.0] * 4410, 44100)
        self.assertEqual((features.get("frame_size"), features.get("hop_size")), (1102, 441))
        self.assertEqual(len(features["starts"]), 8)
        self.assertEqual(features["frame_ms"], 1102 / 44100 * 1000)
        self.assertEqual(features["ends"][-1], 4189 / 44100)
        self.assertEqual((features["requested_frame_ms"], features["requested_hop_ms"]), (25.0, 10.0))

    def test_common_and_histogram_preparation_share_energy_only_supports(self):
        common = prepare_records([audio_record()])[0]
        source = prepare_records([audio_record()], source_histogram=True)[0]
        self.assertEqual(common["features"]["ends"], [.025, .035, .045, .055, .065, .075, .085, .095])
        self.assertEqual(source["features"]["ends"], common["features"]["ends"])
        self.assertEqual(source["labels"], common["labels"])
        self.assertNotIn("centroid", common["features"])
        self.assertNotIn("centroid", source["features"])
        self.assertEqual(source["features"]["energy"], common["features"]["energy"])

    def test_noisy_preparation_keeps_the_current_frame_support(self):
        noisy = noisy_copy(audio_record(), 10, 123)
        for source_histogram in (False, True):
            features = prepare_records([noisy], source_histogram)[0]["features"]
            self.assertEqual(features["ends"][-1], .095)
            self.assertEqual(len(features["starts"]), 8)
            self.assertEqual((features["frame_size"], features["hop_size"]), (25, 10))

    def test_25ms_cleanup_retains_exact_200ms_and_longer_support_gaps(self):
        # Speech [10,35] ms, next speech starts 230/235/240 ms.
        # Gap 195/200/205 ms; shifted latter starts permit exact 200 ms.
        for speech_start, should_fill in ((.230, True), (.235, False), (.240, False)):
            starts = [0, .01, .02, speech_start, speech_start + .01]
            ends = [.025, .035, .045, speech_start + .025, speech_start + .035]
            original = [0, 1, 0, 1, 0]
            actual = fill_short_internal_silences(original, starts, .3, frame_ends=ends)
            self.assertEqual(actual, [0, 1, int(should_fill), 1, 0])
            self.assertEqual(original, [0, 1, 0, 1, 0])


class HistogramFramingTests(unittest.TestCase):
    def test_source_fit_selects_W_after_250ms_padding_and_support_cleanup(self):
        # Padding 5 frames ou bỏ cleanup làm F1 < 1: kỳ vọng độc lập tính tay.
        record = two_impulse_record()
        try:
            model = tt2.fit([record])
        except ValueError as error:
            self.fail(f"25/10 source record must be accepted: {error}")
        self.assertEqual(model["train_frame_f1"], 1.0)
        self.assertEqual([score["train_frame_f1"] for score in model["selection_scores"]], [1.0] * 5)
        self.assertEqual(model["W"], 1.0)
        self.assertEqual((model["padding_ms"], model["padding_frames"]), (250.0, 25))
        mask, diagnostic = tt2.predict(record["features"], model)
        self.assertEqual(mask, [0] * 15 + [1] * 51 + [0] * 21 + [1] * 51 + [0] * 22)
        self.assertEqual((diagnostic["padding_ms"], diagnostic["padding_frames"]), (250.0, 25))
        self.assertEqual(record["features"]["energy"].count(1.0), 2)

    def test_default_padding_expands_250ms_each_side_without_cascading(self):
        raw = [0] * 80
        raw[40] = 1
        self.assertEqual(tt2.pad_speech(raw), [0] * 15 + [1] * 51 + [0] * 14)
        self.assertEqual(sum(raw), 1)

    def test_source_fit_and_predict_accept_real_25ms_features(self):
        samples = [0.0] * 500 + [math.sin(2 * math.pi * 200 * i / 1000) for i in range(500)] + [0.0] * 500
        record = audio_record(samples=samples)
        record["intervals"] = [(0, .5, "sil"), (.5, 1, "v"), (1, 1.5, "sil")]
        prepared = prepare_records([record], True)[0]
        model = tt2.fit([prepared])
        mask, _ = tt2.predict(prepared["features"], model)
        self.assertEqual(len(mask), 148)
        self.assertTrue(any(mask))
        self.assertEqual((model["frame_ms"], model["hop_ms"]), (25.0, 10.0))

    def test_source_guard_rejects_inconsistent_actual_timing(self):
        features = compute_features([0.0] * 100, 1000, frame_ms=25, hop_ms=10, include_centroid=True)
        params = dict(variant="source", bins=64, smooth_radius=2, W=1.0)
        for key, value in (("frame_ms", 50.0), ("hop_ms", 50.0), ("frame_size", 20),
                           ("hop_size", 50), ("ends", [.020 + i * .01 for i in range(8)])):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    tt2.predict(dict(features, **{key: value}), params)

    def test_source_guard_accepts_rounding_at_44100hz(self):
        features = compute_features([1.0] * 4410, 44100, include_centroid=True)
        mask, _ = tt2.predict(features, dict(variant="source", bins=64, smooth_radius=2, W=1.0))
        self.assertEqual(mask, [0] * 8)

    def test_feature_only_records_use_declared_default_25ms(self):
        features = {"energy": [0.0] * 40 + [1.0] + [0.0] * 39,
                    "centroid": [0.0] * 40 + [1.0] + [0.0] * 39}
        mask, _ = tt2.predict(features, dict(variant="source", bins=64, smooth_radius=2, W=1.0))
        self.assertEqual(mask, [0] * 15 + [1] * 51 + [0] * 14)

    def test_missing_nominal_metadata_cannot_hide_50ms_actual_supports(self):
        # Thực có starts/ends: không được đi qua nhánh synthetic mặc định.
        features = {"energy": [0, 1, 0], "centroid": [0, 1, 0],
                    "starts": [0, .05, .10], "ends": [.05, .10, .15]}
        with self.assertRaises(ValueError):
            tt2.predict(features, dict(variant="source", bins=64, smooth_radius=2, W=1))

    def test_feature_only_records_reject_declared_50ms_model(self):
        features = {"energy": [0, 1, 0], "centroid": [0, 1, 0]}
        with self.assertRaises(ValueError):
            tt2.predict(features, dict(variant="source", bins=64, smooth_radius=2, W=1,
                                       frame_ms=50.0, hop_ms=50.0))

    def test_fit_rejects_inconsistent_record_level_timing(self):
        record = {"features": {"energy": [0, 1, 0], "centroid": [0, 1, 0]},
                  "labels": [0, 1, 0], "duration": .15,
                  "starts": [0, .05, .10], "ends": [.05, .10, .15]}
        with self.assertRaises(ValueError):
            tt2.fit([record])

    def test_fit_keeps_feature_only_fallback_explicit_without_inventing_timing(self):
        record = two_impulse_record()
        feature_only = {"features": {key: record["features"][key] for key in ("energy", "centroid")},
                        "labels": record["labels"]}
        fitted = tt2.fit([feature_only])
        self.assertEqual(fitted["full_pipeline_records"], 0)
        self.assertEqual(fitted["predict_only_fallback_records"], 1)
        # 102 padded positives / 123 speech labels => F1 = 204 / 225.
        self.assertAlmostEqual(fitted["train_frame_f1"], 204 / 225)

    def test_current_context_model_uses_25ms_without_padding(self):
        model = tt2.fit([], variant="context")
        self.assertEqual((model["frame_ms"], model["hop_ms"]), (25.0, 10.0))
        self.assertEqual((model["padding_ms"], model["padding_frames"]), (0.0, 0))


class HistoricalBenchmarkTests(unittest.TestCase):
    def test_benchmark_recomputes_historical_20ms_features(self):
        # 20/10 cho last END=.100, còn current 25/10 chỉ đến .095.
        records = [audio_record(name, samples=[1.0] * 100)
                   for name in ("phone_F2", "phone_M2", "studio_F2", "studio_M2")]
        current = prepare_records(records)
        rows = context_benchmark(current)
        fixed = [row for row in rows if row["method"] == "report_fixed_0.0025"]
        self.assertEqual([row["recomputed_end_s"] for row in fixed], [.100] * 4)
        self.assertTrue(all(row.get("historical_comparison") for row in rows))
        self.assertTrue(all((row["frame_ms"], row["hop_ms"]) == (20.0, 10.0) for row in rows))
        self.assertEqual(current[0]["features"]["ends"][-1], .095)


class PipelineMetadataTests(unittest.TestCase):
    def test_real_run_declares_current_framing_in_models_and_run_config(self):
        # Tiêu thụ pipeline thật và đọc artifacts: bắt metadata cũ hardcoded.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, test, output = root / "train", root / "test", root / "output"
            for destination in (train, test):
                destination.mkdir()
                for name in ("phone_F2", "phone_M2", "studio_F2", "studio_M2"):
                    samples = [0] * 30 + [int(12000 * math.sin(2 * math.pi * i / 5)) for i in range(40)] + [0] * 30
                    with wave.open(str(destination / f"{name}.wav"), "wb") as handle:
                        handle.setnchannels(1)
                        handle.setsampwidth(2)
                        handle.setframerate(1000)
                        handle.writeframes(struct.pack("<100h", *samples))
                    (destination / f"{name}.lab").write_text("0 .03 sil\n.03 .07 v\n.07 .1 sil\n", encoding="utf-8")
            args = SimpleNamespace(algorithm="all", compare_context=True, snr_study=False,
                                   file=None, no_show=True, show_seconds=0)
            with patch.multiple("app.pipeline", TRAIN_DIR=train, TEST_DIR=test, OUTPUT_DIR=output), \
                    patch("app.weight_selection.TRAIN_DIR", train), contextlib.redirect_stdout(io.StringIO()):
                run_experiment(args)
            for name in ("tt1", "tt2", "tt3", "tt2-context"):
                model = json.loads((output / "endpoint_modes" / "enhanced" / "models" / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual((model["frame_ms"], model["hop_ms"]), (25.0, 10.0))
                self.assertIn("ties_to_even", model.get("sample_rounding", ""))
                if name == 'tt2':
                    self.assertEqual(model['W'], 1., 'Equal TRAIN final scores must use the smallest W')
            config = json.loads((output / "endpoint_modes" / "enhanced" / "tables" / "all_with_context" / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual((config.get("frame_ms"), config.get("hop_ms")), (25.0, 10.0))
            self.assertIn("ties_to_even", config.get("sample_rounding", ""))


if __name__ == "__main__":
    unittest.main()
