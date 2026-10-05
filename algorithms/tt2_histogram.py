"""Energy-only histogram VAD, as requested by the teacher on 05/10/2026.

Reference: Giannakopoulos PDF pp.1–2 supplies the histogram threshold method.
This assignment uses energy only, not the reference's two-feature AND rule.
Assignment adaptation: 25/10 ms framing; retain 250 ms candidate padding each
side as 25 hop steps. FINAL endpoints use raw STE hysteresis without padding.
The paper does not fix bin count, smoothing window, W or fallback behavior.
These engineering choices are exposed as parameters and diagnostics.
"""
from __future__ import annotations

import math

from app.config import FRAME_MS, HOP_MS, MIN_SILENCE_SECONDS, SAMPLE_ROUNDING

PADDING_MS = 250.0
PADDING_FRAMES = round(PADDING_MS / HOP_MS)


def histogram(values: list[float], bins: int = 64, radius: int = 2,
              value_range: tuple[float, float] | None = None) -> tuple[list[float], list[float]]:
    """Tự đếm histogram bin đều và làm trơn bằng cửa sổ 2*radius+1 bin.

    Input: values là quan sát, bins là số bin, radius là bán kính làm trơn,
    value_range là miền cố định hoặc None để lấy min/max. Output: tuple
    (bin centers, counts làm trơn); chuỗi rỗng trả hai list rỗng.
    """
    if bins < 1 or radius < 0:
        raise ValueError("bins must be positive; radius must be nonnegative")
    if not values:
        return [], []
    # Histogram nguồn dùng min/max bản ghi; adapter báo cáo cố định [0,1].
    lower, upper = value_range if value_range is not None else (min(values), max(values))
    if upper <= lower:
        return [lower], [float(len(values))]
    width = (upper - lower) / bins
    counts = [0.0] * bins
    # Chặn giá trị cực đại vào bin cuối, kể cả giá trị bằng đúng upper.
    for value in values:
        index = max(0, min(bins - 1, int((value - lower) / width)))
        counts[index] += 1.0
    smoothed = []
    # Trung bình trượt tự cài đặt; ở mép chỉ chia số bin thật trong cửa sổ.
    for index in range(bins):
        left, right = max(0, index - radius), min(bins, index + radius + 1)
        total = 0.0
        for position in range(left, right):
            total += counts[position]
        smoothed.append(total / (right - left))
    centers = [lower + (index + 0.5) * width for index in range(bins)]
    return centers, smoothed


def local_maxima(counts: list[float]) -> list[int]:
    """Find positive plateau peaks, including either endpoint.

    A plateau is one candidate, represented by its middle bin. A flat
    positive histogram has one peak; an all-zero histogram has none.
    Input: counts là độ cao histogram đã làm trơn. Output: list chỉ số đỉnh
    tăng dần theo trục bin, plateau dùng bin giữa; không xếp theo độ cao.
    """
    peaks = []
    start = 0
    while start < len(counts):
        # Gom plateau bằng nhau thành một ứng viên; không tạo nhiều đỉnh giả.
        end = start
        while end + 1 < len(counts) and counts[end + 1] == counts[start]:
            end += 1
        left_lower = start == 0 or counts[start] > counts[start - 1]
        right_lower = end + 1 == len(counts) or counts[end] > counts[end + 1]
        if counts[start] > 0.0 and left_lower and right_lower:
            peaks.append((start + end) // 2)
        start = end + 1
    return peaks


def threshold_from_histogram(values: list[float], bins: int, radius: int,
                             weight: float, fixed_range: bool = False) -> tuple[float, dict]:
    """Lấy ngưỡng từ hai đỉnh đầu theo trục giá trị và công khai fallback.

    Input: values là chuỗi đặc trưng, bins/radius cấu hình histogram,
    weight là W không âm, fixed_range chọn [0,1]. Output: (threshold,
    diagnostic) với ngưỡng (W*M1+M2)/(W+1); thiếu đỉnh dùng midpoint miền.
    """
    if weight < 0.0:
        raise ValueError("W must be nonnegative")
    centers, counts = histogram(values, bins, radius, (0.0, 1.0) if fixed_range else None)
    # Đỉnh trả về theo thứ tự trục đặc trưng tăng, không xếp theo độ cao.
    indices = local_maxima(counts)
    peak_values = [centers[index] for index in indices]
    fallback = None
    if not values:
        threshold = 0.0
        fallback = "empty_sequence"
    elif min(values) == max(values):
        threshold = values[0]
        fallback = "constant_sequence"
    elif len(indices) >= 2:
        threshold = (weight * peak_values[0] + peak_values[1]) / (weight + 1.0)
    else:
        # Bài gốc không nêu trường hợp thiếu đỉnh; midpoint là lựa chọn công khai.
        threshold = (min(values) + max(values)) / 2.0
        fallback = "fewer_than_two_peaks_range_midpoint"
    return threshold, {"threshold": threshold, "peak_indices": indices,
                       "peak_values": peak_values, "first_peak": peak_values[0] if peak_values else None,
                       "second_peak": peak_values[1] if len(peak_values) >= 2 else None,
                       "fallback": fallback, "bins": bins, "smooth_radius": radius,
                       "histogram_range": [0.0, 1.0] if fixed_range else ([min(values), max(values)] if values else []),
                       "W": weight}


def pad_speech(mask: list[int], padding_frames: int = PADDING_FRAMES) -> list[int]:
    """Mở rộng speech gốc hai phía; input mask 0/1, padding_frames là số hop.

    Output: mask mới cùng độ dài; mặc định 25 hop = 250 ms với hop 10 ms.
    Đọc mask gốc nên padding không lan tiếp theo các nhãn vừa được thêm.
    """
    result = list(mask)
    # Đọc mask gốc để padding mới không tự lan tiếp sang toàn bộ bản ghi.
    for index, value in enumerate(mask):
        if value == 1:
            for position in range(max(0, index - padding_frames), min(len(mask), index + padding_frames + 1)):
                result[position] = 1
    return result


def _validate_source_framing(features: dict, params: dict) -> float:
    """Kiểm timing TT2 thích nghi 25/10; trả hop thực (ms) để diagnostic.

    Input: features có metadata/support nếu là âm thanh thật, params có
    framing nominal. Output: hop sau làm tròn mẫu; báo ValueError khi các
    metadata có mặt mâu thuẫn. Chuỗi synthetic không timing dùng 25/10.
    """
    error = "Source histogram assignment adaptation requires consistent 25/10 ms framing"
    for mapping, frame_key, hop_key in ((params, "frame_ms", "hop_ms"),
                                        (features, "requested_frame_ms", "requested_hop_ms")):
        if mapping.get(frame_key, FRAME_MS) != FRAME_MS or mapping.get(hop_key, HOP_MS) != HOP_MS:
            raise ValueError(error)
    fs = features.get("sample_rate_hz")
    if fs is not None:
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError(error)
        frame_size, hop_size = max(1, round(fs * FRAME_MS / 1000)), max(1, round(fs * HOP_MS / 1000))
        frame_ms, hop_ms = frame_size / fs * 1000, hop_size / fs * 1000
        if features.get("frame_size", frame_size) != frame_size or features.get("hop_size", hop_size) != hop_size:
            raise ValueError(error)
    else:
        # Số mẫu không thể xác minh nếu thiếu Fs; không giả định là 50/50.
        if "frame_size" in features or "hop_size" in features:
            raise ValueError(error)
        frame_ms, hop_ms = FRAME_MS, HOP_MS
    for key, expected in (("frame_ms", frame_ms), ("hop_ms", hop_ms)):
        if key in features and not math.isclose(features[key], expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(error)
    starts, ends = features.get("starts"), features.get("ends")
    if ends is not None and starts is None:
        raise ValueError(error)
    if starts is not None:
        if len(starts) != len(features["energy"]) or (starts and abs(starts[0]) > 1e-12):
            raise ValueError(error)
        for index, start in enumerate(starts):
            # Dùng chỉ số*hop thay vì cộng dồn số thực; starts phải khớp mẫu.
            expected = index * hop_size / fs if fs is not None else index * hop_ms / 1000
            if not math.isclose(start, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(error)
        if ends is not None:
            if len(ends) != len(starts):
                raise ValueError(error)
            for index, end in enumerate(ends):
                expected = (index * hop_size + frame_size) / fs if fs is not None else starts[index] + frame_ms / 1000
                if not math.isclose(end, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(error)
    return hop_ms


def predict(features: dict, params: dict) -> tuple[list[int], dict]:
    """Dự đoán histogram nguồn thích nghi hoặc biến thể STE báo cáo.

    Input: features có energy cho source hoặc ste_norm cho context;
    params có variant, bins, smooth_radius, W và padding. Output: (mask 0/1,
    diagnostic ngưỡng/padding); source dùng energy > ngưỡng và 250 ms
    padding hai phía. Cleanup silence 200 ms thực hiện ở pipeline chung.
    """
    variant = params.get("variant", "source")
    bins, radius, weight = params["bins"], params["smooth_radius"], params["W"]
    if variant == "context":
        # Adapter theo báo cáo chỉ dùng STE và không kéo dài đoạn tiếng nói.
        values = features["ste_norm"]
        threshold, details = threshold_from_histogram(values, bins, radius, weight, True)
        mask = [1 if value > threshold else 0 for value in values]
        return mask, {"variant": "context", "energy_threshold": threshold,
                      "energy": details,
                      "padding_ms": 0.0, "padding_frames": 0, "raw_speech_frames": sum(mask),
                      "padded_speech_frames": sum(mask), "feature": "ste_norm",
                      "source_status": "energy-only adaptation in context report; not original paper"}
    if variant != "source":
        raise ValueError(f"Unknown histogram variant: {variant}")
    energy = features["energy"]
    actual_hop_ms = _validate_source_framing(features, params)
    energy_threshold, energy_details = threshold_from_histogram(energy, bins, radius, weight)
    raw = [1 if e > energy_threshold else 0 for e in energy]
    # Energy only theo yêu cầu thầy; padding vẫn chỉ thuộc candidate stage.
    padding = params.get("padding_frames", PADDING_FRAMES)
    mask = pad_speech(raw, padding)
    return mask, {"variant": "source", "energy_threshold": energy_threshold,
                  "energy": energy_details, "padding_frames": padding,
                  "padding_ms": params.get("padding_ms", PADDING_MS),
                  "effective_padding_ms": padding * actual_hop_ms,
                  "raw_speech_frames": sum(raw), "padded_speech_frames": sum(mask),
                  "feature": "energy",
                  "source_status": "Teacher-requested energy-only histogram; 25/10 ms; 250 ms candidate padding retained as 25 hops"}


def fit(records: list[dict], variant: str = "source") -> dict:
    """Initial train-only W proposal; thresholds remain per-recording.

    Source W candidates are selected by pooled speech-frame F1 after source
    padding and the shared 200 ms internal-silence cleanup when timing is
    available. Bin/smoothing choices are fixed and declared. The context
    adaptation fixes W=5 and 100 bins as specified by the supplied report.
    No TEST labels enter this core fit. Its TRAIN candidate frame-F1 proposal
    is separate from the downstream app TRAIN FINAL-region MAE calibration
    of global W. TEST is historically exposed and reused only for scoring.
    Input: records là features/labels train, có timing để cleanup nếu có;
    variant chọn source hoặc context. Output: model gồm W/cấu hình và F1
    các ứng viên; synthetic không timing được đếm fallback rõ ràng.
    """
    if variant == "context":
        # Các hằng số này là cấu hình báo cáo, không phải ngưỡng benchmark.
        return {"variant": "context", "frame_ms": FRAME_MS, "hop_ms": HOP_MS,
                "sample_rounding": SAMPLE_ROUNDING,
                "bins": 100, "smooth_radius": 2, "W": 5.0, "padding_ms": 0.0, "padding_frames": 0,
                "parameter_rule": "context report bins/smoothing/W; assignment 25/10 ms framing",
                "framing_adaptation": "context report 20/10 ms to assignment 25/10 ms",
                "source": "BAO_CAO_NGU_CANH section 3.3 energy-only adaptation; current framing follows updated assignment"}
    if variant != "source":
        raise ValueError(f"Unknown histogram variant: {variant}")
    if not records:
        raise ValueError("Histogram W selection requires training records")
    candidates = (1.0, 3.0, 5.0, 10.0, 20.0)
    # Chỉ dò W trên tập train; ngưỡng histogram vẫn tính riêng mỗi file.
    best_score = -1.0
    best_weight = candidates[0]
    scores = []
    base = {"variant": "source", "frame_ms": FRAME_MS, "hop_ms": HOP_MS,
            "feature": "energy", "decision_rule": "energy > histogram threshold; no spectral centroid",
            "sample_rounding": SAMPLE_ROUNDING, "padding_ms": PADDING_MS,
            "bins": 64, "smooth_radius": 2, "padding_frames": PADDING_FRAMES,
            "framing_adaptation": "paper 50/50 ms to assignment 25/10 ms; preserve 250 ms padding as 25 hops"}
    full_pipeline_records, fallback_records = 0, 0
    # Kiểm tra metadata thời gian một lần; bộ huấn luyện thật luôn có đủ.
    # Synthetic unit tests có thể chỉ cung cấp chuỗi đặc trưng và nhãn.
    prepared = []
    for record in records:
        features = dict(record["features"])
        # Nhận timing ở cấp record nếu có; không để thiếu metadata che 50/50.
        for key in ("starts", "ends", "frame_ms", "hop_ms", "requested_frame_ms",
                    "requested_hop_ms", "frame_size", "hop_size", "sample_rate_hz"):
            if key in record:
                if key in features and features[key] != record[key]:
                    raise ValueError(f"Conflicting record and feature timing: {key}")
                features[key] = record[key]
        if "sample_rate" in record:
            if "sample_rate_hz" in features and features["sample_rate_hz"] != record["sample_rate"]:
                raise ValueError("Conflicting record and feature sample rates")
            features["sample_rate_hz"] = record["sample_rate"]
        _validate_source_framing(features, base)
        prepared.append(dict(record, features=features))
        starts = features.get("starts")
        duration = record.get("duration", features.get("duration"))
        if starts is not None and duration is not None:
            full_pipeline_records += 1
        else:
            fallback_records += 1
    for weight in candidates:
        params = dict(base, W=weight)
        tp, fp, fn = 0, 0, 0
        for record in prepared:
            mask, _ = predict(record["features"], params)
            # Initial W proposal retains the source candidate-stage F1 criterion.
            # The app's FINAL-region W sweep is a separate, explicit step.
            features = record["features"]
            starts = features.get("starts")
            duration = record.get("duration", features.get("duration"))
            if starts is not None and duration is not None:
                from core.postprocess import fill_short_internal_silences
                mask = fill_short_internal_silences(mask, starts, duration,
                                                   min_silence=MIN_SILENCE_SECONDS, frame_ends=features.get("ends"))
            labels = record["labels"]
            if len(mask) != len(labels):
                raise ValueError("Training features and LAB labels have unequal lengths")
            for label, prediction in zip(labels, mask):
                # Đếm TP/FP/FN cho speech; nhãn None không ảnh hưởng các tổng.
                if label == 1 and prediction == 1:
                    tp += 1
                elif label == 0 and prediction == 1:
                    fp += 1
                elif label == 1 and prediction == 0:
                    fn += 1
        denominator = 2 * tp + fp + fn
        score = 2.0 * tp / denominator if denominator else 0.0
        scores.append({"W": weight, "train_frame_f1": score, "tp": tp, "fp": fp, "fn": fn})
        # Nếu F1 bằng nhau, thứ tự duyệt chọn W nhỏ nhất một cách xác định.
        if score > best_score:
            best_score, best_weight = score, weight
    return dict(base, W=best_weight, train_frame_f1=best_score, selection_scores=scores,
                full_pipeline_records=full_pipeline_records, predict_only_fallback_records=fallback_records,
                parameter_rule="training pooled frame F1 after padding and 200ms cleanup when timing exists; smallest W breaks ties",
                source="Energy-only adaptation requested by teacher; histogram threshold from Giannakopoulos pp.1-2; 25/10 ms and candidate padding are declared choices")
