"""Đánh giá đầu/cuối speech và các biên nội bộ bằng phép tính tự cài."""

import math

def ground_truth_regions(intervals):
    """Đầu vào: LAB intervals. Đầu ra: vùng nói chuẩn; chỉ gộp v/uv tiếp giáp.

    Silence được ghi rõ trong LAB vẫn là khoảng phân cách; không xoá nhãn chuẩn.
    """
    regions=[]
    for start,end,label in intervals:
        if label not in ('v','uv'):continue
        if regions and abs(start-regions[-1][1])<=1e-9:
            regions[-1]=(regions[-1][0],end)
        else:regions.append((start,end))
    return regions

def region_endpoint_metrics(reference,predicted):
    """Đầu vào: các FINAL vùng chuẩn/dự đoán tăng dần, đơn vị giây.

    Đầu ra: MAE/RMSE trên start/end của các cặp vùng, counts và status.
    DP ghép theo thời gian, ưu tiên số cặp overlap rồi tổng lỗi nhỏ nhất.
    Cùng số vùng: ghép theo thứ tự và tính cả lỗi không overlap.
    Thiếu/thừa vùng: MAE chính=None; matched MAE riêng để không che vùng sai.
    """
    for regions in (reference,predicted):
        previous=-1.
        for start,end in regions:
            if not(math.isfinite(start) and math.isfinite(end) and 0<=start<end and start>=previous):
                raise ValueError('Regions must be finite, ordered and nonoverlapping')
            previous=end
    nr,np=len(reference),len(predicted)
    score=[[(0,0.) for _ in range(np+1)] for _ in range(nr+1)]
    action=[[0]*np for _ in range(nr)]
    # Match complete regions, never individual starts to ends or debug candidates.
    for i in range(nr-1,-1,-1):
        for j in range(np-1,-1,-1):
            best,act=score[i+1][j],1
            other=score[i][j+1]
            if other[0]>best[0] or (other[0]==best[0] and other[1]<best[1]):best,act=other,2
            r,p=reference[i],predicted[j]
            overlap=min(r[1],p[1])-max(r[0],p[0])
            if overlap>0 or nr==np:
                tail=score[i+1][j+1]
                candidate=(tail[0]+1,tail[1]+abs(p[0]-r[0])+abs(p[1]-r[1]))
                if candidate[0]>best[0] or (candidate[0]==best[0] and candidate[1]<=best[1]):best,act=candidate,3
            score[i][j],action[i][j]=best,act
    pairs=[];i=j=0
    while i<nr and j<np:
        if action[i][j]==3:pairs.append((i,j));i+=1;j+=1
        elif action[i][j]==1:i+=1
        else:j+=1
    errors=[]
    for i,j in pairs:
        errors.extend([(predicted[j][0]-reference[i][0])*1000,(predicted[j][1]-reference[i][1])*1000])
    absolute=squared=0.
    for error in errors:absolute+=abs(error);squared+=error*error
    matched_mae=absolute/len(errors) if errors else None
    matched_rmse=math.sqrt(squared/len(errors)) if errors else None
    complete=len(pairs)==nr==np and nr>0
    status='ok' if complete else 'both_no_speech' if nr==np==0 else 'region_count_or_matching_mismatch'
    missing,extra=nr-len(pairs),np-len(pairs)
    return dict(mae_ms=matched_mae if complete else None,rmse_ms=matched_rmse if complete else None,
                matched_boundary_mae_ms=matched_mae,matched_boundary_rmse_ms=matched_rmse,
                ground_truth_region_count=nr,predicted_region_count=np,matched_region_count=len(pairs),
                missing_region_count=missing,extra_region_count=extra,region_pairs=pairs,
                signed_endpoint_errors_ms=errors,status=status,
                start_error_ms=errors[0] if complete and nr==1 else None,
                end_error_ms=errors[1] if complete and nr==1 else None,
                start_abs_error_ms=abs(errors[0]) if complete and nr==1 else None,
                end_abs_error_ms=abs(errors[1]) if complete and nr==1 else None)


def ground_truth_speech_envelope(intervals):
    """Gộp v/uv thành speech và lấy hai biên ngoài cùng của LAB.

    Đầu vào: intervals là list (start, end, label) đã đọc và sắp thời gian.
    Đầu ra: Tuple (START, END) theo giây hoặc None nếu LAB không có speech.
    """
    # Vô thanh uv và hữu thanh v cùng thuộc lớp speech.
    # Lấy hai mép ngoài của speech để so START/END với dự đoán.
    speech = [interval for interval in intervals if interval[2] in ("v", "uv")]
    return (speech[0][0], speech[-1][1]) if speech else None


def ground_truth_boundaries(intervals):
    """Lấy biên VAD chuẩn giữa các đoạn LAB tiếp giáp.

    Đầu vào: intervals là list (start, end, sil/v/uv) theo thứ tự thời gian.
    Đầu ra: List thời điểm chuyển sil↔speech theo giây; đổi v↔uv và khoảng thiếu nhãn không tạo biên.

    Khoảng không được gán nhãn không được tự coi là silence.
    Do đó chỉ lấy chuyển nhãn ở hai đoạn thực sự tiếp giáp nhau.
    """
    # Chỉ xét từng cặp đoạn LAB kế tiếp theo thời gian.
    # Hai đoạn phải tiếp giáp nhau trong dung sai số thực.
    # So sánh sil với speech để gộp v và uv cùng một lớp.
    # Khoảng thiếu nhãn không được suy diễn thành silence.
    # Các biên được giữ thứ tự để ghép một lần trong metric phụ.
    result = []
    for previous, current in zip(intervals, intervals[1:]):
        if abs(previous[1] - current[0]) <= 1e-9 and (previous[2] == "sil") != (current[2] == "sil"):
            result.append(current[0])
    return result


def endpoint_metrics(predicted, reference):
    """Tính sai số hai biên START/END làm metric chính.

    Đầu vào: predicted/reference là tuple (START, END) theo giây hoặc None khi thiếu speech.
    Đầu ra: Dict MAE/RMSE/lỗi từng đầu theo ms và status; lỗi lớn vẫn tính, thiếu speech ghi None.

    Mọi sai số đều tham gia phép tính, kể cả lớn hơn dung sai ghép biên.
    Thiếu speech trả về None cùng trạng thái để không bị hiểu là lỗi 0.
    Đồng thời lưu lỗi có dấu giúp phân biệt phát hiện sớm và muộn.
    """
    # Kiểm tra thiếu speech trước khi làm phép trừ biên.
    # Thiếu dự đoán/tham chiếu được lưu status và None, không ghi lỗi 0.
    # Lỗi có dấu bằng predicted trừ reference để đọc sớm/muộn.
    # Đổi giây sang ms trước khi tính MAE và RMSE hai đầu.
    # Không áp dụng dung sai ghép biên ở metric chính này.
    if predicted is None or reference is None:
        status = "both_no_speech" if predicted is None and reference is None else "missing_prediction" if predicted is None else "missing_reference"
        return {"mae_ms": None, "rmse_ms": None, "start_error_ms": None, "end_error_ms": None,
                "start_abs_error_ms": None, "end_abs_error_ms": None, "status": status}
    start_error = (predicted[0] - reference[0]) * 1000
    end_error = (predicted[1] - reference[1]) * 1000
    # Trung bình trên đúng hai biên của từng file, không dùng dung sai ghép biên.
    return {"mae_ms": (abs(start_error) + abs(end_error)) / 2,
            "rmse_ms": math.sqrt((start_error * start_error + end_error * end_error) / 2),
            "start_error_ms": start_error, "end_error_ms": end_error,
            "start_abs_error_ms": abs(start_error), "end_abs_error_ms": abs(end_error), "status": "ok"}


def frame_labels(centers, intervals):
    """Gán nhãn chuẩn cho mỗi tâm khung bằng các khoảng LAB.

    Đầu vào: centers là tâm khung theo giây tăng dần; intervals là LAB theo thứ tự thời gian.
    Đầu ra: List 0 cho sil, 1 cho v/uv, None ngoài nhãn; dùng khoảng nửa mở [start, end).

    sil cho 0, v/uv cho 1; ngoài phạm vi nhãn cho None.
    Các tâm và đoạn phải theo thứ tự thời gian để quét tuyến tính.
    """
    # Tâm khung và LAB được duyệt theo thứ tự thời gian.
    # Con trỏ chỉ tiến qua các đoạn đã kết thúc để giữ chi phí tuyến tính.
    # Khoảng [start,end) làm tâm đúng END thuộc đoạn tiếp theo.
    # Silence nhận 0, v/uv nhận 1 vì đều là speech.
    # Tâm ngoài LAB nhận None để không đưa nhãn giả vào train/metric.
    labels, cursor = [], 0
    # Con trỏ chỉ đi tới: bỏ các đoạn đã hết trước khi gán nhãn cho tâm khung.
    for center in centers:
        while cursor < len(intervals) and center >= intervals[cursor][1]:
            cursor += 1
        if cursor < len(intervals) and intervals[cursor][0] <= center < intervals[cursor][1]:
            labels.append(0 if intervals[cursor][2] == "sil" else 1)
        else:
            labels.append(None)
    return labels


def frame_metrics(labels, mask):
    """Tính confusion matrix và accuracy/precision/recall/F1 theo khung.

    Đầu vào: labels là list 0/1/None; mask là list 0/1 dự đoán cùng độ dài.
    Đầu ra: Dict TP/TN/FP/FN, ignored/evaluated và các tỷ lệ; bỏ None khỏi mẫu số.
    """
    # Độ dài phải khớp để mỗi prediction có một nhãn chuẩn.
    # Kiểm tra miền nhãn trước khi tính confusion matrix.
    # None tăng ignored và không tham gia các mẫu số.
    # TP/FP dựa vào dự đoán speech; TN/FN dựa vào dự đoán silence.
    # Mẫu số precision/recall bằng 0 cho tỷ lệ 0; không có nhãn cho accuracy None.
    if len(labels) != len(mask):
        raise ValueError("Số nhãn chuẩn và dự đoán phải bằng nhau")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "ignored": 0}
    # None không tham gia mẫu số; frame có nhãn phải được tính dù dự đoán sai.
    for label, predicted in zip(labels, mask):
        if predicted not in (0, 1) or label not in (0, 1, None):
            raise ValueError("Nhãn phải là 0, 1 hoặc None cho LAB ngoài phạm vi")
        if label is None:
            counts["ignored"] += 1
            continue
        key = ("t" if label == predicted else "f") + ("p" if predicted else "n")
        counts[key] += 1
    total = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    # Precision lấy trên dự đoán speech, recall lấy trên speech của LAB.
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    return {**counts, "evaluated": total, "accuracy": (counts["tp"] + counts["tn"]) / total if total else None,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def boundary_metrics(reference, predicted, tolerance=0.1):
    """Ghép biên chuẩn/dự đoán một lần theo thứ tự bằng quy hoạch động.

    Đầu vào: reference/predicted là list biên hữu hạn theo giây; tolerance là dung sai giây không âm.
    Đầu ra: Dict cặp ghép, MAE/RMSE có điều kiện theo ms, biên thừa/bỏ sót và precision/recall/F1.

    Ưu tiên số cặp hợp lệ lớn nhất, sau đó tổng sai số tuyệt đối nhỏ nhất.
    MAE/RMSE ở đây chỉ mô tả các cặp trong dung sai, không phải metric chính.
    Biên bỏ sót và biên thừa được lưu riêng để tránh che giấu lỗi lớn.
    """
    # Input biên phải hữu hạn và tăng theo thời gian.
    # Mỗi ô DP lưu số cặp ghép và tổng sai số tuyệt đối.
    # Ưu tiên nhiều cặp, rồi ưu tiên tổng sai số nhỏ hơn.
    # Truy vết theo thứ tự bảo đảm mỗi biên được dùng tối đa một lần.
    # Chỉ tính lỗi cặp trong dung sai; biên thừa/bỏ sót vẫn được lưu riêng.
    if tolerance < 0:
        raise ValueError("Dung sai không được âm")
    for values in (reference, predicted):
        if any(not math.isfinite(value) for value in values) or any(values[i] > values[i + 1] for i in range(len(values) - 1)):
            raise ValueError("Biên phải hữu hạn và được sắp theo thời gian")
    rows, columns = len(reference), len(predicted)
    scores = [[(0, 0.0) for _ in range(columns + 1)] for _ in range(rows + 1)]
    actions = [[0 for _ in range(columns)] for _ in range(rows)]
    # Điền từ cuối về đầu; mỗi ô lưu số cặp, tổng lỗi và thao tác truy vết.
    for row in range(rows - 1, -1, -1):
        for column in range(columns - 1, -1, -1):
            score, action = scores[row + 1][column], 1
            other = scores[row][column + 1]
            if other[0] > score[0] or (other[0] == score[0] and other[1] < score[1]):
                score, action = other, 2
            # Ghép hai biên chỉ khi trong dung sai; điểm tốt hơn có nhiều cặp hơn.
            error = abs(reference[row] - predicted[column])
            if error <= tolerance + 1e-12:
                tail = scores[row + 1][column + 1]
                candidate = (tail[0] + 1, tail[1] + error)
                if candidate[0] > score[0] or (candidate[0] == score[0] and candidate[1] <= score[1]):
                    score, action = candidate, 3
            scores[row][column], actions[row][column] = score, action
    row, column, pairs = 0, 0, []
    # Truy vết từng cặp trong thứ tự thời gian, không tái sử dụng biên.
    while row < rows and column < columns:
        action = actions[row][column]
        if action == 3:
            pairs.append((reference[row], predicted[column]))
            row += 1
            column += 1
        elif action == 1:
            row += 1
        else:
            column += 1
    count = len(pairs)
    # Các count cho biết phần metric có điều kiện đã bỏ sót bao nhiêu biên.
    precision = count / columns if columns else (1.0 if not rows else 0.0)
    recall = count / rows if rows else (1.0 if not columns else 0.0)
    absolute, squared = 0.0, 0.0
    # Chuyển lỗi sang ms trước khi tổng hợp để mọi trường CSV cùng đơn vị.
    for expected, detected in pairs:
        error = (detected - expected) * 1000
        absolute += abs(error)
        squared += error * error
    return {"mae_ms": absolute / count if count else None, "rmse_ms": math.sqrt(squared / count) if count else None,
            "matched": count, "false_positive": columns - count, "missed": rows - count,
            "reference_count": rows, "predicted_count": columns, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "pairs": pairs, "tolerance_ms": tolerance * 1000}
