"""Đọc WAV PCM và nhãn LAB, không dùng thư viện xử lý tín hiệu."""

import math
import wave
from pathlib import Path


def read_wav(path):
    """Đọc WAV PCM nguyên và chuyển các kênh thành waveform mono.

    Đầu vào: path là đường dẫn WAV PCM 8/16/24/32 bit không nén.
    Đầu ra: Tuple (samples, fs): list float chia toàn thang PCM và tần số mẫu Hz từ header.

    Các mẫu trả về được chia cho toàn thang PCM, không chuẩn hóa theo đỉnh.
    Với PCM 8 bit, dữ liệu không dấu được dịch về tâm 0 trước khi chia.
    Tần số mẫu lấy trực tiếp từ header; WAV nén bị từ chối rõ ràng.
    """
    # Header xác định Fs, số kênh và số byte mỗi mẫu.
    # Chỉ nhận PCM nguyên không nén để giải mã từng byte rõ ràng.
    # PCM 8 bit không dấu được dịch về tâm 0; các độ rộng khác có dấu.
    # Chia toàn thang PCM rồi lấy trung bình các kênh tại cùng thời điểm.
    # Không chuẩn hóa theo đỉnh waveform; STE được chuẩn hóa ở features.
    with wave.open(str(path), "rb") as source:
        channels, width = source.getnchannels(), source.getsampwidth()
        fs = source.getframerate()
        if source.getcomptype() != "NONE" or width not in (1, 2, 3, 4):
            raise ValueError(f"WAV cần PCM nguyên 8/16/24/32 bit: {path}")
        raw = source.readframes(source.getnframes())
    samples = []
    scale = float(1 << (width * 8 - 1))
    # Một frame WAV chứa lần lượt một mẫu của mỗi kênh.
    # Chuyển PCM sang float trước khi trộn để giữ toàn thang [-1, 1).
    for offset in range(0, len(raw), width * channels):
        total = 0.0
        for channel in range(channels):
            index = offset + channel * width
            block = raw[index:index + width]
            value = block[0] - 128 if width == 1 else int.from_bytes(block, "little", signed=True)
            total += value / scale
        samples.append(total / channels)
    return samples, fs


def read_lab(path, duration):
    """Đọc, kiểm tra và chặn các khoảng nhãn theo thời lượng WAV.

    Đầu vào: path là đường dẫn LAB; duration là thời lượng WAV dương, hữu hạn theo giây.
    Đầu ra: List (start, end, label) theo giây với label sil/v/uv; lỗi ghi rõ dòng nguồn.

    Nhãn hợp lệ là sil, v, uv; lỗi dữ liệu báo kèm số dòng, không bị nuốt.
    Kiểm tra thứ tự và đoạn chồng lấn trước khi chặn biên theo WAV.
    Sai lệch cuối tối đa 10 ms được chấp nhận do LAB làm tròn 2 chữ số.
    Khoảng âm thanh ngoài nhãn không tự được gán thành khoảng lặng.
    """
    # Thời lượng WAV phải hợp lệ trước khi kiểm tra nhãn.
    # Bỏ dòng trắng và chỉ hai dạng thống kê F0 được quy định.
    # Mỗi đoạn phải có đủ start/end/nhãn và thứ tự không chồng lấn.
    # LAB có thể làm tròn đuôi trong 10 ms, sau đó chặn END theo WAV.
    # Khoảng không được LAB phủ vẫn thiếu nhãn, không tự thêm silence.
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Thời lượng WAV phải dương và hữu hạn")
    path = Path(path)
    intervals, previous_end = [], 0.0
    # Không suy luận nhãn từ dòng lỗi: mỗi lỗi phải được phát hiện tại nguồn.
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            columns = line.split()
            if not columns:
                continue
            error = f"{path.name}, dòng {number}: LAB không hợp lệ"
            if columns[0] in ("F0mean", "F0std"):
                # Hai dòng F0 là thống kê Praat; kiểm tra định dạng rồi bỏ qua.
                if len(columns) != 2:
                    raise ValueError(error)
                try:
                    value = float(columns[1])
                except ValueError as exc:
                    raise ValueError(error) from exc
                if not math.isfinite(value) or value < 0:
                    raise ValueError(error)
                continue
            if len(columns) != 3 or columns[2] not in ("sil", "v", "uv"):
                raise ValueError(error)
            try:
                start, end = float(columns[0]), float(columns[1])
            except ValueError as exc:
                raise ValueError(error) from exc
            # Kiểm tra số, thứ tự và thời lượng trước khi chặn sai lệch làm tròn.
            if (not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start or start < previous_end - 1e-9
                    or end > duration + 0.010001):
                raise ValueError(error)
            previous_end = end
            end = min(end, duration)
            if start < end:
                intervals.append((start, end, columns[2]))
    if not intervals:
        raise ValueError(f"{path.name}: không có đoạn nhãn")
    return intervals
