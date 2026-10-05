"""STE, MA và trọng tâm phổ được tính bằng vòng lặp và FFT tự viết."""

import math

from app.config import FRAME_MS, HOP_MS, SAMPLE_ROUNDING


def fft_radix2(values):
    """FFT Cooley–Tukey radix 2, biến đổi thuận không chia cho N.

    Đầu vào phải có độ dài là lũy thừa của 2; không sửa dữ liệu đầu vào.
    Đảo bit đưa các phần tử về thứ tự phù hợp với các tầng butterfly.
    Mỗi tầng ghép hai phổ con bằng hệ số quay exp(-j*2*pi*k/span).
    Input: values là chuỗi mẫu thực/phức dài 2^k. Output: list phổ phức
    cùng độ dài, theo thứ tự bin từ DC; không thay đổi values.
    """
    size = len(values)
    if size == 0 or size & (size - 1):
        raise ValueError("FFT radix 2 cần độ dài là lũy thừa dương của 2")
    result, reversed_index = list(values), 0
    # Đảo bit bằng phép dịch/xor, không dùng FFT hay phép sắp phổ thư viện.
    for index in range(1, size):
        bit = size >> 1
        while reversed_index & bit:
            reversed_index ^= bit
            bit >>= 1
        reversed_index ^= bit
        if index < reversed_index:
            result[index], result[reversed_index] = result[reversed_index], result[index]
    span = 2
    while span <= size:
        # Hệ số quay của tầng hiện tại; mỗi butterfly dùng một cặp phổ con.
        angle = -2 * math.pi / span
        step = complex(math.cos(angle), math.sin(angle))
        half = span // 2
        for block in range(0, size, span):
            rotation = 1 + 0j
            for offset in range(half):
                # Nhánh trên cộng, nhánh dưới trừ; cập nhật hệ số quay kế tiếp.
                even = result[block + offset]
                odd = result[block + offset + half] * rotation
                result[block + offset] = even + odd
                result[block + offset + half] = even - odd
                rotation *= step
        span *= 2
    return result


def spectral_centroid(frame):
    """Trọng tâm biên độ phổ một phía, chia cho tần số Nyquist.

    Chỉ bổ sung số 0 cho kích thước FFT, không bổ sung cho STE hoặc MA.
    Phổ magnitude một phía, có DC/Nyquist và zero padding FFT là quy ước
    cài đặt công bố ở đây; bài Giannakopoulos không ấn định các chi tiết này.
    Trả về 0 cho khung bằng 0; khung khác cho giá trị trong [0, 1].
    Input: frame là các mẫu thực của một khung. Output: centroid vô hướng
    chuẩn hóa theo Nyquist; không dùng Fs vì tỷ số bin đã bỏ đơn vị Hz.
    """
    if not frame:
        return 0.0
    size = 1
    while size < len(frame):
        size *= 2
    if size == 1:
        return 0.0
    spectrum = fft_radix2([complex(value) for value in frame] + [0j] * (size - len(frame)))
    # Tử số là tổng bin*biên độ, mẫu số là tổng biên độ của phổ một phía.
    numerator, denominator = 0.0, 0.0
    half = size // 2
    for index in range(half + 1):
        magnitude = abs(spectrum[index])
        numerator += index * magnitude
        denominator += magnitude
    return numerator / denominator / half if denominator > 0 else 0.0


def normalize_max(values):
    """Chuẩn hóa chuỗi không âm theo đỉnh của chính bản ghi.

    Input: values là năng lượng/biên độ không âm. Output: list cùng độ dài
    đã chia max; chuỗi không có đỉnh dương trả toàn 0, không sửa input.
    """
    peak = 0.0
    for value in values:
        if value > peak:
            peak = value
    return [value / peak for value in values] if peak > 0 else [0.0 for _ in values]


def compute_features(samples, fs, frame_ms=FRAME_MS, hop_ms=HOP_MS, include_centroid=False):
    """Chia khung đầy đủ theo số mẫu, bỏ đuôi ngắn hơn một khung.

    starts/ends là miền hỗ trợ khung; centers là tâm miền hỗ trợ đó.
    Chỉ giữ start + frame_size <= len(samples), không thêm 0 ở đuôi WAV.
    Tín hiệu ngắn hơn một khung bị từ chối để tránh lệch thống kê năng lượng.
    STE là tổng bình phương, energy là trung bình bình phương, MA là meanabs.
    Các đặc trưng normalized dùng đỉnh riêng của từng bản ghi WAV.

    Input: samples là waveform mono, fs là Hz; frame_ms/hop_ms là thời gian
    yêu cầu, include_centroid chọn tính thêm centroid. Output: dict chuỗi
    đặc trưng và timing; frame_ms/hop_ms phản ánh số mẫu thật đã làm tròn,
    requested_* giữ thời gian yêu cầu, frame_size/hop_size là số mẫu nguyên.
    round(fs*ms/1000) chọn mẫu gần nhất, tie .5 dùng số chẵn của Python.
    """
    if fs <= 0 or frame_ms <= 0 or hop_ms <= 0:
        raise ValueError("Tần số mẫu, độ dài khung và bước khung phải dương")
    frame_size = max(1, int(round(fs * frame_ms / 1000)))
    hop_size = max(1, int(round(fs * hop_ms / 1000)))
    if hop_size > frame_size:
        raise ValueError("Bước khung không được lớn hơn độ dài khung")
    if 0 < len(samples) < frame_size:
        raise ValueError("Tín hiệu ngắn hơn một khung đầy đủ")
    features = {key: [] for key in ("starts", "centers", "ends", "ste", "energy", "ma")}
    if include_centroid:
        features["centroid"] = []
    for start in range(0, len(samples) - frame_size + 1, hop_size):
        # Tính trực tiếp trên các mẫu thật, với hai tổng độc lập STE và MA.
        end = start + frame_size
        squared, absolute = 0.0, 0.0
        for index in range(start, end):
            value = samples[index]
            squared += value * value
            absolute += abs(value)
        # Đổi chỉ số mẫu sang giây sau khi tính, tránh tích lũy sai số bước nhảy.
        features["starts"].append(start / fs)
        features["ends"].append(end / fs)
        features["centers"].append((start + end) / (2 * fs))
        features["ste"].append(squared)
        features["energy"].append(squared / (end - start))
        features["ma"].append(absolute / (end - start))
        if include_centroid:
            features["centroid"].append(spectral_centroid(samples[start:end]))
    # Giữ raw và normalized song song để plot và từng thuật toán dùng đúng đơn vị.
    features["ste_norm"] = normalize_max(features["ste"])
    features["ma_norm"] = normalize_max(features["ma"])
    features.update(frame_ms=frame_size / fs * 1000, hop_ms=hop_size / fs * 1000,
                    requested_frame_ms=float(frame_ms), requested_hop_ms=float(hop_ms),
                    frame_size=frame_size, hop_size=hop_size, sample_rate_hz=fs,
                    sample_rounding=SAMPLE_ROUNDING)
    return features
