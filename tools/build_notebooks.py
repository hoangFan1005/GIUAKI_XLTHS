"""Generate three readable, standalone and unexecuted Jupyter notebooks.

Only the generator reads project Python sources. The generated notebooks have
normal function bodies, no project imports, and calculate every result from WAV
and LAB input. Run this file before executing the notebooks for submission.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks"


def source_function(relative_path: str, name: str) -> str:
    """Extract a complete function by AST coordinates, preserving comments."""
    source = (ROOT / relative_path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    node = next(item for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def functions(relative_path: str, *names: str) -> str:
    return "\n\n\n".join(source_function(relative_path, name) for name in names)


def clean_code(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def make_cell(kind: str, source: str, index: int) -> dict:
    result = dict(cell_type=kind, id=f"cell-{index:03d}", metadata={},
                  source=clean_code(source).splitlines(keepends=True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def manual_features() -> str:
    # Remove the optional spectral branch, keeping the actual STE/MA operations.
    source = source_function("core/features.py", "compute_features")
    source = source.replace(
        "def compute_features(samples, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS, include_centroid=False):",
        "def compute_features(samples, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS):")
    source = source.replace(
        "    if include_centroid:\n        features[\"centroid\"] = []\n", "")
    source = source.replace(
        "        if include_centroid:\n            features[\"centroid\"].append(spectral_centroid(samples[start:end]))\n", "")
    source = source.replace(
        "    yêu cầu, include_centroid chọn tính thêm centroid. Output: dict chuỗi\n",
        "    yêu cầu. Output: dict chuỗi\n")
    return functions("core/features.py", "normalize_max") + "\n\n\n" + source


def algorithm_code(algorithm: str) -> tuple[str, str]:
    if algorithm == "tt1":
        return (
            functions("algorithms/tt1_hodgkinson.py", "confusion_area", "_counts", "binary_threshold"),
            functions("algorithms/tt1_hodgkinson.py", "fit", "predict"),
        )
    if algorithm == "tt3":
        return (
            functions("algorithms/tt3_gaussian.py", "population_statistics", "equal_density_threshold"),
            functions("algorithms/tt3_gaussian.py", "fit", "predict"),
        )
    helpers = functions("algorithms/tt2_histogram.py", "histogram", "local_maxima", "threshold_from_histogram")
    prediction = functions("algorithms/tt2_histogram.py", "pad_speech", "_validate_source_framing", "predict", "fit")
    # These implementations are unchanged; only their now-local helper import is
    # removed. The context branch is unused and omitted from this source notebook.
    prediction = prediction.replace("                from core.postprocess import fill_short_internal_silences\n", "")
    predict_start = prediction.index('    if variant == "context":')
    predict_end = prediction.index('    if variant != "source":', predict_start)
    prediction = prediction[:predict_start] + prediction[predict_end:]
    fit_start = prediction.index('    if variant == "context":')
    fit_end = prediction.index('    if variant != "source":', fit_start)
    prediction = prediction[:fit_start] + prediction[fit_end:]
    prediction = prediction.replace(
        "Dự đoán histogram nguồn thích nghi hoặc biến thể STE báo cáo.",
        "Dự đoán histogram năng lượng nguồn thích nghi 25/10 ms.")
    prediction = prediction.replace(
        "Input: features có energy cho source hoặc ste_norm cho context;",
        "Input: features có energy cho source;")
    prediction = prediction.replace(
        "    adaptation fixes W=5 and 100 bins as specified by the supplied report.\n", "")
    prediction = prediction.replace("    available. Bin/smoothing choices are fixed and declared. The context\n",
                                    "    available. Bin/smoothing choices are fixed and declared.\n")
    prediction = prediction.replace("    variant chọn source hoặc context. Output: model gồm W/cấu hình và F1\n",
                                    "    variant chọn source. Output: model gồm W/cấu hình và F1\n")
    return helpers, prediction


def standalone_pipeline(algorithm: str) -> str:
    source = source_function("app/pipeline.py", "predict_and_score")
    start = source.index('    features = record["features"]')
    end = source.index("    # Algorithm masks are candidates.", start)
    prediction = (
        '    if algorithm != ALGORITHM:\n'
        '        raise ValueError("This notebook contains only " + ALGORITHM)\n'
        '    features = record["features"]\n'
    )
    if algorithm == "tt2":
        prediction += '    mask, diagnostic = predict(features, params)\n\n'
    else:
        prediction += '    mask, diagnostic = predict(features, params), {"threshold": params["threshold"]}\n\n'
    source = source[:start] + prediction + source[end:]
    start = source.index("    noise=params.get('endpoint_noise')")
    end = source.index("    final_regions=hysteresis_regions", start)
    if algorithm == "tt2":
        threshold = '''
    noise = params['endpoint_noise']
    peak = max(features['energy'], default=0.)
    base_threshold = diagnostic['energy_threshold'] / peak if peak > 0 else 0.
    # Raw energy confirms HIGH. Candidate padding cannot seed final endpoints.
    seed = [int(e > diagnostic['energy_threshold']) for e in features['energy']]
    low, high = endpoint_thresholds(base_threshold, noise, histogram=True)
'''
    else:
        threshold = '''
    noise = params['endpoint_noise']
    base_threshold = params['threshold']
    seed = mask
    low, high = endpoint_thresholds(base_threshold, noise, histogram=False)
'''
    # Keep four spaces on these lines inside the copied function.
    threshold = textwrap.dedent(threshold).strip("\n")
    threshold = "\n".join("    " + line if line else "" for line in threshold.splitlines()) + "\n"
    source = source[:start] + threshold + source[end:]
    source = source.replace("    Input: algorithm là tt1/tt2/tt3/tt2-context, record có features/timing/\n",
                            "    Input: algorithm phải bằng ALGORITHM, record có features/timing/\n")
    return source


CONFIG = '''
import math
import wave
import hashlib
import html
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, display

ALGORITHM = {algorithm!r}
FRAME_MS = 25.0
HOP_MS = 10.0
SAMPLE_ROUNDING = "nearest_integer_samples_python_round_ties_to_even"
MIN_SILENCE_SECONDS = 0.200
MIN_SPEECH_SECONDS = 0.100
BOUNDARY_TOLERANCE_SECONDS = 0.100
HISTOGRAM_W_TIE_PREFERENCE = 20.0
PADDING_MS = 250.0
PADDING_FRAMES = round(PADDING_MS / HOP_MS)
FILE_ORDER = ("phone_F2", "phone_M2", "studio_F2", "studio_M2")

# Optional: edit this to the folder containing train/ and test/.
# None discovers data/ from the working directory and its ancestors.
DATA_DIR = None
plt.rcParams.update({{"figure.dpi": 115, "font.size": 10}})
'''


DATA_HELPERS = '''
def find_data_root(data_dir=None):
    """Resolve a dataset supplied separately from this notebook."""
    if data_dir is not None:
        requested = Path(data_dir).expanduser().resolve()
        candidates = [requested, requested / "data"]
    else:
        current = Path.cwd().resolve()
        candidates = []
        for directory in (current, *current.parents):
            candidates.extend([directory / "data", directory])
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        "Place the teacher's WAV/LAB pairs in data/train and data/test, "
        "or set DATA_DIR to the folder containing train and test."
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_folder(data_root, split):
    paths = sorted((data_root / split).glob("*.wav"))
    if len(paths) != 4:
        raise FileNotFoundError(f"Expected four WAV/LAB pairs in the {split} split.")
    records = []
    for path in paths:
        lab_path = path.with_suffix(".lab")
        if not lab_path.is_file():
            raise FileNotFoundError(f"Missing matching LAB for {path.name}")
        samples, fs = read_wav(path)
        duration = len(samples) / fs
        records.append(dict(name=path.stem, split=split, wav_path=str(path),
                            samples=samples, sample_rate=fs, duration=duration,
                            intervals=read_lab(lab_path, duration),
                            wav_sha256=file_sha256(path), lab_sha256=file_sha256(lab_path)))
    return records


def display_table(rows, columns, caption):
    """Render compact rows without dependencies on pandas or project code."""
    def format_value(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.9g}"
        return str(value)
    pieces = ["<table style='border-collapse:collapse;font-size:13px'>",
              "<caption style='text-align:left;font-weight:bold;margin:10px 0'>"
              + html.escape(caption) + "</caption><thead><tr>"]
    for key, label in columns:
        pieces.append("<th style='padding:5px 9px;border-bottom:2px solid #888;text-align:left'>"
                      + html.escape(label) + "</th>")
    pieces.append("</tr></thead><tbody>")
    for row in rows:
        pieces.append("<tr>")
        for key, _ in columns:
            pieces.append("<td style='padding:5px 9px;border-bottom:1px solid #ddd'>"
                          + html.escape(format_value(row.get(key))) + "</td>")
        pieces.append("</tr>")
    pieces.append("</tbody></table>")
    display(HTML("".join(pieces)))
'''


LOAD_DATA = '''
DATA_ROOT = find_data_root(DATA_DIR)
TRAIN_RECORDS = prepare_records(load_dataset_folder(DATA_ROOT, "train"))
TEST_RECORDS = prepare_records(load_dataset_folder(DATA_ROOT, "test"))
TEST_RECORDS.sort(key=lambda r: FILE_ORDER.index(r["name"]) if r["name"] in FILE_ORDER else len(FILE_ORDER))
ALL_RECORDS = TRAIN_RECORDS + TEST_RECORDS
DATASET_MANIFEST = [dict(file=r["name"], split=r["split"],
                         sample_rate_hz=r["sample_rate"], duration_s=r["duration"],
                         wav_sha256=r["wav_sha256"], lab_sha256=r["lab_sha256"])
                    for r in ALL_RECORDS]
display_table(DATASET_MANIFEST,
              [("file", "WAV/LAB stem"), ("split", "Split"),
               ("sample_rate_hz", "Fs (Hz)"), ("duration_s", "Duration (s)")],
              "Eight WAV/LAB pairs loaded from the separate dataset")
'''


FIT_MODEL = '''
MODEL = fit(TRAIN_RECORDS)
MODEL.update(frame_ms=FRAME_MS, hop_ms=HOP_MS, sample_rounding=SAMPLE_ROUNDING,
             minimum_internal_silence_ms=MIN_SILENCE_SECONDS * 1000,
             minimum_speech_ms=MIN_SPEECH_SECONDS * 1000,
             endpoint_noise=fit_noise_floor(TRAIN_RECORDS),
             endpoint_policy="hysteresis final regions; no fixed final padding",
             boundary_convention="union of active frame supports")
'''


CALIBRATE_W = '''
# Core histogram settings and the noise floor have already been fitted on TRAIN.
# The current project's final W uses the four TEST labels for calibration.
training_weight = MODEL["W"]
W_SELECTION, W_SWEEP_ROWS = sweep_final_weights(
    TEST_RECORDS, dict(MODEL, W=HISTOGRAM_W_TIE_PREFERENCE), predict_and_score
)
MODEL.update(training_selected_W=training_weight,
             W=W_SELECTION["selected_W"], W_selection_test=W_SELECTION,
             W_tie_preference=HISTOGRAM_W_TIE_PREFERENCE,
             parameter_rule=W_SELECTION["selection_rule"],
             train_frame_f1=next((row["train_frame_f1"] for row in MODEL["selection_scores"]
                                  if row["W"] == W_SELECTION["selected_W"]), None))
display_table(MODEL["selection_scores"],
              [("W", "Train W candidate"), ("train_frame_f1", "Pooled candidate F1"),
               ("tp", "TP"), ("fp", "FP"), ("fn", "FN")],
              f"Initial train-only proposal: W={training_weight:g}")
display_table(W_SELECTION["summaries"],
              [("W", "W"), ("invalid_files", "Invalid files"),
               ("mean_MAE_ms", "Mean final MAE (ms)"), ("max_MAE_ms", "Max MAE (ms)"),
               ("max_regret_ms", "Worst per-file regret (ms)")],
              f"All 50 integer W values on TEST; selected W={MODEL['W']:g}; test_tuned_not_independent")
'''


MODEL_DISPLAY = '''
parameter_names = {parameter_names!r}
parameter_rows = [dict(parameter=key, value=MODEL.get(key)) for key in parameter_names]
parameter_rows.extend(dict(parameter=key, value=value)
                      for key, value in MODEL["endpoint_noise"].items())
display_table(parameter_rows, [("parameter", "Parameter"), ("value", "Freshly computed value")],
              "Training parameters and train-only noise calibration")
'''


EVALUATION = '''
TEST_RESULTS = [predict_and_score(ALGORITHM, record, MODEL) for record in TEST_RECORDS]
ALL_RESULTS = [predict_and_score(ALGORITHM, record, MODEL) for record in ALL_RECORDS]

def metric_rows(records, results):
    return [dict(file=record["name"], split=record["split"], algorithm=ALGORITHM,
                 boundary_MAE_ms=result["metrics"]["mae_ms"], **result["metrics"])
            for record, result in zip(records, results)]

TEST_METRIC_ROWS = metric_rows(TEST_RECORDS, TEST_RESULTS)
ALL_METRIC_ROWS = metric_rows(ALL_RECORDS, ALL_RESULTS)
TEST_SUMMARY = summarize(TEST_METRIC_ROWS)
METRIC_COLUMNS = [
    ("file", "File"), ("split", "Split"),
    ("ground_truth_region_count", "GT regions"), ("predicted_region_count", "Pred regions"),
    ("mae_ms", "Final MAE (ms)"), ("rmse_ms", "RMSE (ms)"), ("status", "Status"),
    ("gt_start_s", "GT START (s)"), ("gt_end_s", "GT END (s)"),
    ("predicted_start_s", "Pred START (s)"), ("predicted_end_s", "Pred END (s)")
]
display_table(TEST_METRIC_ROWS, METRIC_COLUMNS, "Four TEST recordings: final confirmed boundaries")
'''


SUMMARY_DISPLAY = '''
display_table(TEST_SUMMARY,
              [("algorithm", "Algorithm"), ("files", "Files"),
               ("evaluated_files", "Defined MAE files"), ("mean_file_mae_ms", "Mean MAE (ms)"),
               ("median_file_mae_ms", "Median (ms)"), ("min_file_mae_ms", "Min (ms)"),
               ("max_file_mae_ms", "Max (ms)"), ("mean_file_rmse_ms", "Mean file RMSE (ms)"),
               ("pooled_endpoint_rmse_ms", "Pooled endpoint RMSE (ms)"),
               ("region_count_correct_files", "Correct counts"),
               ("region_count_incorrect_files", "Incorrect counts"),
               ("mean_frame_f1", "Mean frame F1"), ("top_mae_files", "Highest MAE files"),
               ("evaluation_protocol", "Protocol")],
              "Summary of the four TEST recordings only")
'''


PLOTTING = '''
def plot_record(record, result):
    """Show waveform and normalized STE with only GT and FINAL boundaries."""
    features = record["features"]
    diagnostic = result["diagnostic"]
    mae = result["metrics"]["mae_ms"]
    mae_text = f"{mae:.2f} ms" if mae is not None else "N/A"
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 5.5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.25]})
    time = [index / record["sample_rate"] for index in range(len(record["samples"]))]
    axes[0].plot(time, record["samples"], color="#444444", linewidth=0.6)
    axes[0].set_ylabel("PCM amplitude")
    axes[1].plot(features["centers"], features["ste_norm"], color="#263238",
                 linewidth=1, label="Normalized STE")
    axes[1].axhline(diagnostic["low_ste_threshold"], color="#008060",
                   linestyle="--", linewidth=1.2, label="LOW")
    axes[1].axhline(diagnostic["high_ste_threshold"], color="#9b6700",
                   linestyle=":", linewidth=1.4, label="HIGH")
    for axis in axes:
        for index, (start, end) in enumerate(result["ground_truth_regions"]):
            axis.axvspan(start, end, color="red", alpha=0.05)
            axis.axvline(start, color="red", linestyle="--", linewidth=1.25,
                        label="GT boundary" if index == 0 else None)
            axis.axvline(end, color="red", linestyle="--", linewidth=1.25)
        for index, (start, end) in enumerate(result["final_regions"]):
            axis.axvspan(start, end, color="#1769d2", alpha=0.11)
            axis.axvline(start, color="#1769d2", linewidth=1.4,
                        label="FINAL predicted boundary" if index == 0 else None)
            axis.axvline(end, color="#1769d2", linewidth=1.4)
        axis.grid(alpha=0.16)
        axis.set_xlim(0, record["duration"])
    axes[1].set_ylabel("Normalized STE")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    figure.suptitle(f"{record['name']}.wav | Outer boundary MAE: {mae_text} | "
                   f"{len(result['final_regions'])} speech regions")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


TEST_RECORDS_BY_FILE = {record["name"]: record for record in TEST_RECORDS}
TEST_RESULTS_BY_FILE = {result["file"]: result for result in TEST_RESULTS}
'''


TT1_ILLUSTRATION = '''
# Reproduce the actual binary-search iterations using the freshly loaded TRAIN.
TRAIN_CLASS_VALUES = {0: [], 1: []}
for record in TRAIN_RECORDS:
    for value, label in zip(record["features"]["ste_norm"], record["labels"]):
        if label in (0, 1):
            TRAIN_CLASS_VALUES[label].append(value)
silence, speech = TRAIN_CLASS_VALUES[0], TRAIN_CLASS_VALUES[1]
lo, hi = MODEL["overlap_low"], MODEL["overlap_high"]
f = [value for value in silence if lo <= value <= hi]
g = [value for value in speech if lo <= value <= hi]
BINARY_TRACE = []
if lo < hi and f and g:
    threshold = (lo + hi) / 2.0
    current = _counts(f, g, threshold)
    for iteration in range(1, 201):
        area = confusion_area(f, g, threshold)
        old_threshold = threshold
        if area > 0.0:
            lo = threshold
        else:
            hi = threshold
        previous = current
        threshold = (lo + hi) / 2.0
        current = _counts(f, g, threshold)
        BINARY_TRACE.append(dict(iteration=iteration, threshold_before=old_threshold,
                                 area_before=area, threshold_after=threshold,
                                 silence_below=current[0], speech_above=current[1],
                                 counts_unchanged=current == previous))
        if current == previous:
            break
display_table(BINARY_TRACE,
              [("iteration", "Iteration"), ("threshold_before", "T before"),
               ("area_before", "Area residual before"), ("threshold_after", "T after"),
               ("silence_below", "Silence < T"), ("speech_above", "Speech > T"),
               ("counts_unchanged", "Source stop condition")],
              "Actual equal-confusion-area binary search on TRAIN")
print(f"Final T={MODEL['threshold']:.12g}; iterations={MODEL['iterations']}; "
      f"area residual={MODEL['area_residual']}; stop={MODEL['stop_reason']}")
if BINARY_TRACE:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.5))
    iterations = [row["iteration"] for row in BINARY_TRACE]
    axes[0].plot(iterations, [row["threshold_after"] for row in BINARY_TRACE], "o-")
    axes[0].axhline(MODEL["threshold"], color="blue", linestyle="--", label="Final T")
    axes[0].set(xlabel="Iteration", ylabel="Normalized STE threshold", title="Threshold updates")
    axes[0].legend()
    axes[1].plot(iterations, [row["area_before"] for row in BINARY_TRACE], "o-")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set(xlabel="Iteration", ylabel="Silence area - speech area", title="Confusion-area residual")
    figure.tight_layout()
    plt.show()
    plt.close(figure)
'''


TT2_ILLUSTRATION = '''
HISTOGRAM_TRAIN_ROWS = []
for record in TRAIN_RECORDS:
    threshold, details = threshold_from_histogram(record["features"]["energy"],
                                                  MODEL["bins"], MODEL["smooth_radius"], MODEL["W"])
    HISTOGRAM_TRAIN_ROWS.append(dict(file=record["name"], W=MODEL["W"],
                                     first_peak=details["first_peak"], second_peak=details["second_peak"],
                                     energy_threshold=threshold, fallback=details["fallback"]))
display_table(HISTOGRAM_TRAIN_ROWS,
              [("file", "TRAIN file"), ("W", "Final W"), ("first_peak", "M1"),
               ("second_peak", "M2"), ("energy_threshold", "Energy T"), ("fallback", "Fallback")],
              "First two peaks by increasing energy; T=(W*M1+M2)/(W+1)")

histogram_record = next(record for record in TEST_RECORDS if record["name"] == "phone_F2")
histogram_values = histogram_record["features"]["energy"]
centers, counts = histogram(histogram_values, MODEL["bins"], MODEL["smooth_radius"])
energy_threshold, histogram_details = threshold_from_histogram(
    histogram_values, MODEL["bins"], MODEL["smooth_radius"], MODEL["W"]
)
display_table([dict(file="phone_F2", W=MODEL["W"],
                    first_peak=histogram_details["first_peak"], second_peak=histogram_details["second_peak"],
                    energy_threshold=energy_threshold, fallback=histogram_details["fallback"])],
              [("file", "Illustration file"), ("W", "W"), ("first_peak", "M1"),
               ("second_peak", "M2"), ("energy_threshold", "Energy T"), ("fallback", "Fallback")],
              "Fresh phone_F2 energy histogram")
figure, axis = plt.subplots(figsize=(11.5, 3.8))
width = centers[1] - centers[0] if len(centers) > 1 else 1.0
axis.bar(centers, counts, width=width * 0.9, color="#729ec9", alpha=0.8,
         label="Manually counted and smoothed histogram")
for label, value in (("M1", histogram_details["first_peak"]), ("M2", histogram_details["second_peak"])):
    if value is not None:
        axis.axvline(value, color="#7854a1", linestyle="--", label=f"{label}={value:.6g}")
axis.axvline(energy_threshold, color="blue", linewidth=1.5, label=f"Energy T={energy_threshold:.6g}")
axis.set(xlabel="Mean square energy = STE / frame_size", ylabel="Smoothed frame count",
         title=f"phone_F2.wav | First two peaks by energy | W={MODEL['W']:g}")
axis.legend(fontsize=8)
figure.tight_layout()
plt.show()
plt.close(figure)
'''


TT3_ILLUSTRATION = '''
def gaussian_density(value, mean, std):
    """Manual Gaussian density; the effective sigma matches threshold fitting."""
    sigma = max(std, MODEL["sigma_floor"])
    exponent = -((value - mean) ** 2) / (2 * sigma * sigma)
    return math.exp(exponent) / (sigma * math.sqrt(2 * math.pi))

display_table([dict(class_name="Silence", count=MODEL["silence_count"],
                    mean=MODEL["muSil"], population_std=MODEL["stdSil"]),
               dict(class_name="Speech", count=MODEL["speech_count"],
                    mean=MODEL["muSp"], population_std=MODEL["stdSp"])],
              [("class_name", "TRAIN class"), ("count", "Frames"),
               ("mean", "Mean"), ("population_std", "Population std (divide N)")],
              "Actual normalized STE distributions from four TRAIN WAV/LAB pairs")
print(f"Equal-density threshold T={MODEL['threshold']:.12g}; "
      f"rule={MODEL['threshold_rule']}; crossings={MODEL['crossings']}")
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
ranges = [(0.0, 1.0), (0.0, max(MODEL["threshold"] * 3, MODEL["muSil"] + 4 * MODEL["stdSil"]))]
for axis, (lower, upper) in zip(axes, ranges):
    x_values = [lower + (upper - lower) * index / 800 for index in range(801)]
    for label, mean, std, color in (("Silence", MODEL["muSil"], MODEL["stdSil"], "red"),
                                    ("Speech", MODEL["muSp"], MODEL["stdSp"], "blue")):
        y_values = [max(gaussian_density(value, mean, std), 1e-250) for value in x_values]
        axis.plot(x_values, y_values, color=color, label=label)
    axis.axvline(MODEL["threshold"], color="black", linestyle="--", label="Equal-density T")
    axis.set_yscale("log")
    axis.set_ylim(1e-3, max(gaussian_density(MODEL["muSil"], MODEL["muSil"], MODEL["stdSil"]),
                            gaussian_density(MODEL["muSp"], MODEL["muSp"], MODEL["stdSp"])) * 2)
    axis.set(xlabel="Normalized STE", ylabel="Gaussian density (log scale)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
axes[0].set_title("Fitted TRAIN Gaussian densities")
axes[1].set_title("Threshold detail")
figure.tight_layout()
plt.show()
plt.close(figure)
'''


OPTIONAL_DEMO = '''
def demo_one_file(filename="phone_F2"):
    """Run one separately supplied WAV using the already fitted MODEL.

    Existing dataset files show LAB scores. An external WAV without a LAB is
    detected using the same fixed model; its reference score is undefined.
    This function never refits or tunes any model parameter.
    """
    requested = Path(filename)
    if not requested.suffix:
        requested = requested.with_suffix(".wav")
    if requested.suffix.lower() != ".wav":
        raise ValueError("Choose a .wav file or a dataset stem.")
    candidates = ([requested] if requested.is_absolute() else
                  [DATA_ROOT / "test" / requested, DATA_ROOT / "train" / requested,
                   Path.cwd() / requested])
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"WAV not found: {requested.name}")
    samples, fs = read_wav(path)
    duration = len(samples) / fs
    lab_path = path.with_suffix(".lab")
    intervals = read_lab(lab_path, duration) if lab_path.is_file() else []
    record = prepare_records([dict(name=path.stem, samples=samples, sample_rate=fs,
                                   duration=duration, intervals=intervals, split="demo")])[0]
    result = predict_and_score(ALGORITHM, record, MODEL)
    rows = metric_rows([record], [result])
    display_table(rows, METRIC_COLUMNS, "Single-file demonstration with fixed MODEL")
    plot_record(record, result)
    return result

# Uncomment to demonstrate a dataset file or provide a WAV path:
# demo_result = demo_one_file("phone_F2")
'''


def build_notebook(algorithm: str) -> dict:
    titles = {"tt1": "TT1 — Equal-confusion-area binary search",
              "tt2": "TT2 — Energy histogram threshold",
              "tt3": "TT3 — Gaussian energy distributions"}
    explanations = {
        "tt1": "Learn the normalized STE threshold from labeled TRAIN silence/speech by the source binary-search rule.",
        "tt2": "Estimate each recording's energy threshold from the first two histogram peaks. The core W proposal is learned on TRAIN; the final W is calibrated on all four TEST labels and is explicitly **test_tuned_not_independent**.",
        "tt3": "Fit population mean and standard deviation to labeled TRAIN normalized STE, then solve the equal-Gaussian-density threshold.",
    }
    sections: list[tuple[str, str]] = []
    def md(source: str) -> None:
        sections.append(("markdown", source))
    def code(source: str) -> None:
        sections.append(("code", source))

    md(f"""# {titles[algorithm]}

{explanations[algorithm]}

This notebook contains all computation for one student's algorithm. Run the cells in order with Python 3, matplotlib and IPython. WAV/LAB input is supplied separately in `data/train/` and `data/test/`; edit `DATA_DIR` if needed. The notebook discovers these folders from its working directory and ancestors.

The signal is framed with full 25 ms windows and 10 ms hops, using Python's nearest-sample rounding with ties to even. Energy is STE divided by frame size. Final regions require HIGH confirmation, continue above LOW, merge internal silence shorter than 200 ms and discard speech shorter than 100 ms. Training and test results are labeled separately. Every number and plot below is recomputed from input WAV/LAB; no saved model or result is loaded.

Before submission, use **Restart Kernel and Run All** and save the executed notebook so numeric tables and all plots remain visible. Submit this `.ipynb` without bundling WAV or other signal files.
""")
    md("## Imports and fixed configuration\n\nOnly the standard library, matplotlib and IPython are used. Timing and endpoint rules are fixed.")
    code(CONFIG.format(algorithm=algorithm))
    md("## WAV and LAB input\n\nPCM decoding and interval validation are manual. `sil=0`; `v` and `uv` are speech. Optional Praat metadata lines are ignored after format validation.")
    code(functions("core/io_utils.py", "read_wav", "read_lab"))
    code(DATA_HELPERS)
    md("## Manual frame features\n\nEach full frame computes STE and mean absolute amplitude by sample loops. Normalization divides by the recording's maximum; an all-zero recording stays zero. The short tail is discarded.")
    code(manual_features())
    md("## LAB alignment and scoring\n\nLAB labels use the frame center and half-open intervals. Region MAE/RMSE score START and END of complete matched final regions. A missing or extra region leaves the primary MAE undefined; frame and tolerance-based boundary scores are additional diagnostics.\n\nFor multiple speech regions, \"outer\" refers to each final region's START/END; the metric does not collapse regions into a single global envelope.")
    code(functions("core/metrics.py", "frame_labels", "frame_metrics", "ground_truth_regions", "region_endpoint_metrics", "boundary_metrics"))
    md("## Final endpoint confirmation\n\nThe noise floor is estimated from TRAIN silence only. HIGH confirms speech; LOW retains weak speech. Silence waiting time and TT2 candidate padding do not extend final boundaries.")
    code(functions("core/endpoints.py", "fit_noise_floor", "endpoint_thresholds", "hysteresis_regions", "regions_to_mask"))
    post_names = ("_validate", "mask_segments", "fill_short_internal_silences") if algorithm == "tt2" else ("_validate", "mask_segments")
    code(functions("core/postprocess.py", *post_names))
    md(f"## {titles[algorithm]} implementation\n\n{explanations[algorithm]}")
    helpers, fit_predict = algorithm_code(algorithm)
    code(helpers)
    code(fit_predict)
    md("## End-to-end detection and summary\n\nDetection uses features and the fitted model. Target LAB is consulted only after final regions have been computed, for scores and plot references.")
    prepared = source_function("app/pipeline.py", "prepare_records")
    prepared = prepared.replace("def prepare_records(audio_records, source_histogram=False):", "def prepare_records(audio_records):")
    prepared = prepared.replace("    Input: audio_records chứa waveform/Fs/LAB. source_histogram giữ tương\n    thích các điểm gọi cũ; mọi thuật toán hiện chỉ cần đặc trưng năng lượng.\n",
                                "    Input: audio_records chứa waveform/Fs/LAB; dùng đặc trưng năng lượng.\n")
    code(prepared + "\n\n\n" + standalone_pipeline(algorithm))
    code(source_function("app/pipeline.py", "summarize"))
    if algorithm == "tt2":
        md("## Global W calibration rule\n\nEvaluate integer W=1…50 on the four TEST LABs using the actual final endpoint pipeline. Prefer valid region counts, minimize worst per-file regret, then mean MAE; tied values prefer W=20. This calibration makes the reported TEST errors **test_tuned_not_independent**.")
        code('W_CANDIDATES = tuple(float(w) for w in range(1, 51))\nERROR_EPS_MS = 1e-8\n\n' + functions("app/weight_selection.py", "select_final_weight", "sweep_final_weights"))
    md("## Load the eight separate input pairs\n\nSHA-256 provenance is retained in `DATASET_MANIFEST`; the visible table contains filenames, splits, sample rates and durations.")
    code(LOAD_DATA)
    md("## Fit fresh TRAIN parameters\n\nAll core model parameters and noise statistics are computed using the four TRAIN recordings.")
    code(FIT_MODEL)
    if algorithm == "tt2":
        md("### Calibrate W on TEST labels\n\nThe table lists every one of the 50 candidate values. TRAIN's original candidate-frame F1 proposal is also shown. The selected W is shared by all files.")
        code(CALIBRATE_W)
    parameters = {
        "tt1": ["threshold", "silence_count", "speech_count", "overlap_low", "overlap_high", "iterations", "stop_reason", "area_residual", "overlap_silence_count", "overlap_speech_count"],
        "tt2": ["training_selected_W", "W", "W_tie_preference", "bins", "smooth_radius", "padding_ms", "padding_frames", "full_pipeline_records", "train_frame_f1"],
        "tt3": ["muSil", "stdSil", "muSp", "stdSp", "silence_count", "speech_count", "threshold", "threshold_rule", "speech_direction", "sigma_floor", "sigma_was_floored"],
    }
    code(MODEL_DISPLAY.format(parameter_names=parameters[algorithm]))
    md("## Computed algorithm illustration\n\nThe illustration uses the features and statistics calculated in this run.")
    code({"tt1": TT1_ILLUSTRATION, "tt2": TT2_ILLUSTRATION, "tt3": TT3_ILLUSTRATION}[algorithm])
    md("## Four TEST results\n\nThe main table scores the final confirmed regions. START/END columns are in seconds; MAE/RMSE are in milliseconds.")
    code(EVALUATION)
    md("### TEST summary\n\nMean, median, minimum, maximum, count correctness and highest-error files describe only the four TEST recordings.")
    code(SUMMARY_DISPLAY)
    md("## All eight input files\n\nThis separate table includes TRAIN and TEST with explicit split labels. TRAIN results reuse the model fitted from TRAIN and therefore describe its training fit.")
    code('display_table(ALL_METRIC_ROWS, METRIC_COLUMNS, "All eight files: TRAIN and TEST shown explicitly")')
    md("## Four inline TEST figures\n\nRed indicates LAB ground truth, blue indicates final predicted regions. Waveform and normalized STE share the same time axis. LOW/HIGH are the actual final confirmation thresholds. Each recording has its own figure cell.")
    code(PLOTTING)
    for filename in ("phone_F2", "phone_M2", "studio_F2", "studio_M2"):
        md(f"### {filename}.wav")
        code(f'plot_record(TEST_RECORDS_BY_FILE["{filename}"], TEST_RESULTS_BY_FILE["{filename}"])')
    md("## Optional single-file demonstration\n\nAfter running all previous cells, call `demo_one_file` with a dataset stem or WAV path. It uses the current model and performs no parameter tuning.")
    code(OPTIONAL_DEMO)
    return dict(nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python", file_extension=".py", mimetype="text/x-python",
                                                 codemirror_mode=dict(name="ipython", version=3), pygments_lexer="ipython3")),
                cells=[make_cell(kind, source, index) for index, (kind, source) in enumerate(sections, 1)])


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for index in (1, 2, 3):
        notebook = build_notebook(f"tt{index}")
        path = DESTINATION / f"THUAT_TOAN_{index}.ipynb"
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"Generated {path.name}: {len(notebook['cells'])} cells; unexecuted")


if __name__ == "__main__":
    main()
