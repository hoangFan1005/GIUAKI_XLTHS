"""Chuyển nhãn khung thành các đoạn thời gian và lọc khoảng lặng nội bộ."""

import math


def _validate(mask, starts, duration):
    """Kiểm tra độ dài, nhãn và trật tự thời điểm trước hậu xử lý.

    Đầu vào: mask là list 0/1; starts là các đầu khung theo giây; duration là thời lượng WAV.
    Đầu ra: None nếu hợp lệ; ValueError nếu nhãn/timing không đáp ứng các bất biến.
    """
    # Mask và starts phải có cùng số phần tử để ghép đúng khung.
    # Thời lượng WAV không âm và hữu hạn là miền timing hợp lệ.
    # Mỗi nhãn chỉ nhận 0/1; starts phải tăng nghiêm ngặt.
    # Mọi đầu khung nằm trong WAV và khung đầu bắt đầu tại 0.
    # Kiểm tra dùng chung giúp các hàm sau tránh hiểu sai input.
    if len(mask) != len(starts):
        raise ValueError("Số nhãn phải bằng số thời điểm bắt đầu khung")
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Thời lượng phải không âm và hữu hạn")
    previous = -1.0
    for label, start in zip(mask, starts):
        if label not in (0, 1) or not math.isfinite(start) or start < 0 or start <= previous or start >= duration:
            raise ValueError("Nhãn phải là 0/1; starts phải tăng và nằm trong WAV")
        previous = start
    if starts and abs(starts[0]) > 1e-12:
        raise ValueError("Khung đầu phải bắt đầu tại 0")


def mask_segments(mask, starts, duration, frame_ends=None):
    """Chuyển mask thành các đoạn speech/silence theo thời gian.

    Đầu vào: mask, starts, duration mô tả WAV; frame_ends tùy chọn là END thực của từng khung.
    Đầu ra: List (start, end, 0/1) theo giây; có frame_ends thì speech là hợp miền hỗ trợ.

    Khi thiếu frame_ends, biên đổi nhãn là starts của khung mới, không phải tâm.
    Khi có frame_ends, speech là hợp miền hỗ trợ thực của các khung speech.
    Silence là phần bù; các miền speech chồng lấn được gộp thành một đoạn.
    Không có frame_ends thì dùng khoảng bước khung, cuối tại duration.
    """
    # Kiểm tra mask/timing trước khi ghép miền thời gian.
    # Nếu có END thực, speech được lấy theo toàn miền hỗ trợ khung.
    # Hợp nhất speech giao nhau rồi lấy phần bù làm silence.
    # Đuôi WAV không có khung speech phủ được thể hiện là silence.
    # Nhánh thiếu END dùng khoảng bước khung để hỗ trợ input đơn giản.
    _validate(mask, starts, duration)
    if not mask:
        return []
    if frame_ends is not None:
        # Ưu tiên miền hỗ trợ của speech nếu nó chồng lên khung silence kế tiếp.
        if len(frame_ends) != len(mask):
            raise ValueError("Số biên kết thúc khung phải bằng số nhãn")
        speech = []
        for label, start, end in zip(mask, starts, frame_ends):
            if not math.isfinite(end) or not start < end <= duration + 1e-12:
                raise ValueError("Miền hỗ trợ khung phải nằm trong WAV và có độ dài dương")
            end = min(end, duration)
            if label:
                if speech and start <= speech[-1][1] + 1e-12:
                    speech[-1] = (speech[-1][0], max(end, speech[-1][1]))
                else:
                    speech.append((start, end))
        segments, cursor = [], 0.0
        # Chèn khoảng lặng là phần bù giữa các miền speech đã hợp nhất.
        for start, end in speech:
            if start > cursor + 1e-12:
                segments.append((cursor, start, 0))
            segments.append((start, end, 1))
            cursor = end
        if cursor < duration - 1e-12:
            segments.append((cursor, duration, 0))
        return segments
    segments, first = [], 0
    for index in range(1, len(mask)):
        if mask[index] != mask[first]:
            segments.append((starts[first], starts[index], int(mask[first])))
            first = index
    segments.append((starts[first], duration, int(mask[first])))
    return segments


def fill_short_internal_silences(mask, starts, duration, min_silence=0.2, frame_ends=None):
    """Nối hai vùng speech khi gap nội bộ ngắn hơn ngưỡng.

    Đầu vào: mask, starts, duration; min_silence theo giây; frame_ends tùy chọn cho support END.
    Đầu ra: List mask mới, giữ silence đầu/cuối và gap đúng ngưỡng; không sửa mask đầu vào.

    Khoảng lặng ở đầu và cuối bản ghi được giữ nguyên dù rất ngắn.
    Có frame_ends thì khoảng lặng tính từ support END đến support START.
    Ngưỡng áp dụng theo thời gian thật, không đếm khung bằng hằng số 50 ms.
    Khoảng đúng 200 ms được giữ lại; sai số làm tròn số thực có dung sai nhỏ.
    """
    # Sao chép mask trước khi đổi nhãn để giữ nguyên input.
    # Quét từng run silence và tìm speech ở cả hai phía.
    # Có frame_ends thì gap đo từ END speech trước đến START speech sau.
    # Chỉ nối gap nhỏ hơn min_silence; dung sai giữ đúng trường hợp bằng ngưỡng.
    # Silence ở hai mép và speech island ngắn không bị xóa bởi thao tác này.
    _validate(mask, starts, duration)
    if frame_ends is not None:
        mask_segments(mask, starts, duration, frame_ends=frame_ends)
    if min_silence < 0:
        raise ValueError("Thời lượng khoảng lặng tối thiểu phải không âm")
    result = list(mask)
    index = 0
    # Quét từng run silence; chỉ run có speech hai đầu mới đủ điều kiện nối.
    while index < len(mask):
        if mask[index] == 1:
            index += 1
            continue
        first = index
        while index < len(mask) and mask[index] == 0:
            index += 1
        # Dùng END của speech trước nếu cung cấp miền hỗ trợ khung.
        gap_start = frame_ends[first - 1] if frame_ends is not None and first > 0 else starts[first]
        if first > 0 and index < len(mask) and starts[index] - gap_start < min_silence - 1e-12:
            for offset in range(first, index):
                result[offset] = 1
    return result


def speech_envelope(mask, starts, duration, frame_ends=None):
    """Lấy hai biên ngoài cùng của toàn bộ speech dự đoán.

    Đầu vào: mask, starts, duration và frame_ends tùy chọn có cùng nghĩa như mask_segments().
    Đầu ra: Tuple (START, END) theo giây hoặc None khi không phát hiện speech.
    """
    # Lấy segment speech sau khi hợp miền hỗ trợ khung.
    # Biên ngoài cùng bao cả nhiều đoạn speech; không có speech trả None.
    speech = [segment for segment in mask_segments(mask, starts, duration, frame_ends=frame_ends) if segment[2] == 1]
    return (speech[0][0], speech[-1][1]) if speech else None


def boundaries(mask, starts, duration, frame_ends=None):
    """Lấy tất cả biên chuyển speech/silence nội bộ của WAV.

    Đầu vào: mask, starts, duration và frame_ends tùy chọn mô tả nhãn và timing của khung.
    Đầu ra: List thời điểm biên theo giây, theo thứ tự tăng; không thêm hai mép WAV.
    """
    # Kiểm tra nhãn và starts bằng quy tắc dùng chung.
    # Có END thực thì lấy biên từ các segment đã hợp miền hỗ trợ.
    # Không có END thực thì biên là starts nơi mask chuyển nhãn.
    # Không thêm 0 hoặc duration vào list biên nội bộ.
    # Danh sách này dùng metric phụ; START/END ngoài cùng lấy bằng envelope.
    _validate(mask, starts, duration)
    if frame_ends is not None:
        segments = mask_segments(mask, starts, duration, frame_ends=frame_ends)
        return [segment[0] for segment in segments[1:]]
    return [starts[index] for index in range(1, len(mask)) if mask[index] != mask[index - 1]]
