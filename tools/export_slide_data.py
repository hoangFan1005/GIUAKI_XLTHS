"""Export current slide evidence without retraining or running demo modes.

Run with the project's Python environment:
    python tools/export_slide_data.py
The saved models and original four test WAV/LAB pairs remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import TEST_DIR
from app.pipeline import load_audio_folder, prepare_records, predict_and_score


def dataset_hashes() -> dict[str, str]:
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "data").rglob("*") if p.is_file()}


def numeric_parameters(model: dict) -> dict:
    """Keep numerical facts for English notes, without copying model prose."""
    if model.get("schema_version") != 2:
        raise ValueError("Unsupported model schema: regenerate schema 2 models before export")
    if (model.get("parameter_selection_set") != "train" or
            model.get("evaluation_protocol") != "train_selected_reused_test" or
            model.get("historical_test_exposure") is not True or "W_selection_test" in model):
        raise ValueError("Unsupported schema 2 selection provenance")
    fields = ("threshold", "frame_ms", "hop_ms", "iterations", "muSil",
              "stdSil", "muSp", "stdSp", "silence_count", "speech_count",
              "bins", "smooth_radius", "padding_ms", "W",
              "schema_version", "metrics_schema_version", "parameter_selection_set",
              "evaluation_protocol", "historical_test_exposure", "training_files",
              "finalW", "tie_preference_W", "candidate_frame_selected_W",
              "candidate_frame_f1", "candidate_frame_selection_scores",
              "candidate_cleanup_records", "candidate_frame_selection_applicable",
              "minimum_internal_silence_ms", "minimum_speech_ms")
    out = {key: model[key] for key in fields if key in model}
    out["endpoint_noise"] = {key: value for key, value in model["endpoint_noise"].items()
                             if isinstance(value, (int, float))}
    selection = model.get("W_selection_train")
    if "W" in model and selection is None:
        raise ValueError("Unsupported schema 2 histogram model: missing TRAIN selection")
    if selection:
        if (selection.get("selection_set") != "train" or
                selection.get("evaluation_protocol") != "train_final_calibration"):
            raise ValueError("Unsupported schema 2 TRAIN calibration provenance")
        out["selection"] = dict(selection)
    return out


def check_saved_metrics(key: str, files: list[dict]) -> None:
    """Require the rendered numbers to agree with the current batch CSV."""
    source = ROOT / "outputs" / "tables" / key / "test_metrics.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["file"]: row for row in csv.DictReader(handle)}
    if set(rows) != {entry["name"] for entry in files}:
        raise ValueError(f"{source}: expected the same four test files")
    for entry in files:
        for field, csv_field in (("mae", "mae_ms"), ("rmse", "rmse_ms")):
            if not math.isclose(entry[field], float(rows[entry["name"]][csv_field]),
                                rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"{key}/{entry['name']}: {field} differs from saved CSV")


def export_data(destination: Path) -> None:
    before = dataset_hashes()
    models = {f"tt{n}": json.loads((ROOT / "outputs" / "models" / f"tt{n}.json").read_text(encoding="utf-8"))
              for n in range(1, 4)}
    parameters = {key: numeric_parameters(model) for key, model in models.items()}
    audio = load_audio_folder(TEST_DIR)
    data = {}
    for number in range(1, 4):
        key = f"tt{number}"
        model = models[key]
        if model["frame_ms"] != 25 or model["hop_ms"] != 10:
            raise ValueError(f"{key}: the slide framing requires 25/10 ms")
        files = []
        for record in prepare_records(audio, number == 2):
            result = predict_and_score(key, record, model)
            features = record["features"]
            stride = max(1, len(record["samples"]) // 1800)
            maximum = max(map(abs, record["samples"])) or 1
            mae, rmse = result["metrics"]["mae_ms"], result["metrics"]["rmse_ms"]
            if mae is None or rmse is None:
                raise ValueError(f"{key}/{record['name']}: undefined FINAL endpoint metric")
            files.append({
                "name": record["name"], "duration": record["duration"],
                "wave_x": [i / record["sample_rate"]
                           for i in range(0, len(record["samples"]), stride)],
                "wave_y": [value / maximum for value in record["samples"][::stride]],
                "centers": features["centers"], "ste": features["ste_norm"],
                "gt": result["ground_truth_boundaries"],
                "pred": result["predicted_boundaries"],
                "mae": mae, "rmse": rmse,
                "low": result["diagnostic"]["low_ste_threshold"],
                "high": result["diagnostic"]["high_ste_threshold"],
                "region_count": len(result["final_regions"]),
                "evaluation_protocol": result["metrics"]["evaluation_protocol"],
            })
        check_saved_metrics(key, files)
        data[key] = {"files": files, "threshold": model.get("threshold", 0),
                     "mean": sum(entry["mae"] for entry in files) / len(files),
                     "parameters": parameters[key]}
        print(f"{key}: four current test files, mean FINAL MAE {data[key]['mean']:.2f} ms")
    if before != dataset_hashes():
        raise RuntimeError("Original dataset changed during evidence export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    destination.with_name("dataset_hashes.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8")
    print(f"Slide evidence exported to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_build" / "slides" / "deck_data.json")
    export_data(parser.parse_args().output.resolve())
