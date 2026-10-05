"""CS425 (Hodgkinson), PDF pp. 38–39, printed pp. 37–38, equation (2.8).

The course demonstrates MA; this assignment applies the same equal-confusion
area method to normalized STE. Only LAB-labeled training frames are pooled.
"""
from __future__ import annotations


def confusion_area(silence: list[float], speech: list[float], threshold: float) -> float:
    """Tính hiệu diện tích nhầm lẫn (2.8), giảm theo threshold.

    Input: silence/speech là quan sát hai lớp, threshold là ngưỡng thử.
    Output: trung bình khoảng cách silence trên ngưỡng trừ speech dưới
    ngưỡng; hai lớp phải có dữ liệu.
    """
    if not silence or not speech:
        raise ValueError("Equation (2.8) needs both classes")
    # Tổng khoảng cách phía sai ngưỡng của silence và speech trong (2.8).
    sil_area = 0.0
    sp_area = 0.0
    for value in silence:
        if value > threshold:
            sil_area += value - threshold
    for value in speech:
        if value < threshold:
            sp_area += threshold - value
    return sil_area / len(silence) - sp_area / len(speech)


def _counts(silence: list[float], speech: list[float], threshold: float) -> tuple[int, int]:
    """Đếm điều kiện dừng TT1 bằng dấu < và > đúng tài liệu nguồn.

    Input: silence/speech là hai lớp trong overlap, threshold là ngưỡng.
    Output: tuple số silence < threshold và số speech > threshold.
    """
    # Dùng đúng dấu < và > của sách cho điều kiện dừng, không dùng <=/>=.
    i, p = 0, 0
    for value in silence:
        if value < threshold:
            i += 1
    for value in speech:
        if value > threshold:
            p += 1
    return i, p


def binary_threshold(silence: list[float], speech: list[float], max_iterations: int = 200) -> dict:
    """Execute source steps 1–10, including unchanged-count stopping.

    Repeated counts are the source stopping criterion, not a guarantee that
    the continuous area residual is exactly zero. Diagnostics expose that
    distinction. An iteration cap only guards pathological inputs.
    Input: silence/speech là STE chuẩn hóa của hai lớp train; max_iterations
    giới hạn vòng lặp. Output: dict threshold, overlap và lý do dừng/counts;
    không có overlap dùng midpoint được ghi rõ trong diagnostic.
    """
    if not silence or not speech:
        raise ValueError("TT1 training needs labeled silence and speech")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    # Tính đoạn giao của hai miền giá trị; ngoài đoạn này không có overlap.
    lo = max(min(silence), min(speech))
    hi = min(max(silence), max(speech))
    if lo >= hi:
        threshold = (max(silence) + min(speech)) / 2.0
        return {"threshold": threshold, "overlap_low": lo, "overlap_high": hi,
                "iterations": 0, "stop_reason": "no_overlap_midpoint",
                "area_residual": None, "overlap_silence_count": 0,
                "overlap_speech_count": 0}
    # Bước 1: chỉ giữ các quan sát nằm trong đoạn chồng lấn chung.
    f = [value for value in silence if lo <= value <= hi]
    g = [value for value in speech if lo <= value <= hi]
    threshold = (lo + hi) / 2.0
    if not f or not g:
        # Miền min/max giao nhau nhưng có thể không có mẫu thực bên trong.
        return {"threshold": threshold, "overlap_low": lo, "overlap_high": hi,
                "iterations": 0, "stop_reason": "empty_discrete_overlap_midpoint",
                "area_residual": None, "overlap_silence_count": len(f),
                "overlap_speech_count": len(g)}
    current = _counts(f, g, threshold)
    initial_low, initial_high = lo, hi
    reason = "iteration_cap"
    iterations = 0
    # Bước 7–10: cập nhật một nửa miền theo dấu diện tích rồi đếm lại.
    for iterations in range(1, max_iterations + 1):
        area = confusion_area(f, g, threshold)
        if area > 0.0:
            lo = threshold
        else:
            hi = threshold
        previous = current
        threshold = (lo + hi) / 2.0
        current = _counts(f, g, threshold)
        if current == previous:
            reason = "unchanged_counts_source_rule"
            break
    # Trả midpoint vừa đếm; không dịch ngưỡng thêm sau khi điều kiện dừng đạt.
    return {"threshold": threshold, "overlap_low": initial_low,
            "overlap_high": initial_high, "iterations": iterations,
            "stop_reason": reason, "area_residual": confusion_area(f, g, threshold),
            "silence_below_threshold": current[0], "speech_above_threshold": current[1],
            "overlap_silence_count": len(f), "overlap_speech_count": len(g)}


def fit(records: list[dict]) -> dict:
    """Học một ngưỡng TT1 từ STE train chuẩn hóa theo từng WAV.

    Input: records chứa features.ste_norm và labels 0/1/None từ LAB.
    Output: dict ngưỡng binary search, số quan sát mỗi lớp và provenance;
    nhãn None không tham gia học.
    """
    silence, speech = [], []
    # Gom dữ liệu huấn luyện theo nhãn LAB; bỏ các khung ngoài vùng được gán.
    for record in records:
        values, labels = record["features"]["ste_norm"], record["labels"]
        if len(values) != len(labels):
            raise ValueError("Training features and LAB labels have unequal lengths")
        for value, label in zip(values, labels):
            if label == 0:
                silence.append(value)
            elif label == 1:
                speech.append(value)
    params = binary_threshold(silence, speech)
    # Lưu provenance và số lượng mẫu để báo cáo có thể kiểm tra cách học.
    params.update({"feature": "ste_norm", "normalization": "per_wav_max",
                   "silence_count": len(silence), "speech_count": len(speech),
                   "source": "CS425 PDF pp.38-39, equation (2.8) and steps 1-10"})
    return params


def predict(features: dict, params: dict) -> list[int]:
    """Phân lớp STE >= ngưỡng đã học thành speech trước cleanup chung.

    Input: features.ste_norm là chuỗi STE, params.threshold là ngưỡng train.
    Output: list nhãn 0/1 cùng độ dài; hàm không đọc nhãn LAB.
    """
    return [1 if value >= params["threshold"] else 0 for value in features["ste_norm"]]
