# Kết quả hiện tại — 05/10/2026

## 1. Phiên bản đang dùng

- TT1: Binary Search cân bằng diện tích nhầm lẫn trên normalized STE.
- TT2: Histogram **chỉ Energy** theo yêu cầu mới của thầy; W = 20 chung, 64 bins, smoothing radius 2.
- TT3: thống kê Gaussian population trên normalized STE, ngưỡng giao hai mật độ.
- Chung: frame 25 ms, hop 10 ms; HIGH/LOW hysteresis; nối silence dưới 200 ms; loại speech dưới 100 ms. Không tính F0.

Tất cả endpoint chính lấy từ FINAL speech regions. Candidate/debug và padding 250 ms của TT2 không được dùng làm biên cuối. Dataset gốc có một vùng nói trong mỗi WAV; code hỗ trợ nhiều vùng khi có khoảng lặng đủ dài.

## 2. Bốn file test — MAE biên ngoài (ms)

| File | TT1 | TT2 | TT3 |
|---|---:|---:|---:|
| phone_F2 | 57.50 | 32.50 | 27.50 |
| phone_M2 | 12.50 | 7.50 | 7.50 |
| studio_F2 | 7.51 | 7.51 | 7.51 |
| studio_M2 | 2.49 | 7.49 | 7.49 |
| Trung bình | 20.00 | 13.75 | 12.50 |

Mean per-file RMSE lần lượt: TT1 **24.69 ms**, TT2 **15.81 ms**, TT3 **14.08 ms**. Đây là trung bình RMSE từng file; khác pooled RMSE trên toàn bộ endpoint.

## 3. Toàn bộ 8 WAV — MAE (ms)

| File | Split | TT1 | TT2 | TT3 |
|---|---|---:|---:|---:|
| phone_F1 | train | 57.50 | 2.50 | 2.50 |
| phone_F2 | test | 57.50 | 32.50 | 27.50 |
| phone_M1 | train | 77.50 | 12.50 | 7.50 |
| phone_M2 | test | 12.50 | 7.50 | 7.50 |
| studio_F1 | train | 17.49 | 17.49 | 17.49 |
| studio_F2 | test | 7.51 | 7.51 | 7.51 |
| studio_M1 | train | 22.49 | 22.49 | 22.49 |
| studio_M2 | test | 2.49 | 7.49 | 7.49 |

| Thuật toán | Mean | Median | Min | Max | Đúng số vùng |
|---|---:|---:|---:|---:|---|
| TT1 | 31.87 | 19.99 | 2.49 | 77.50 | 8/8 |
| TT2 | 13.75 | 10.00 | 2.50 | 32.50 | 8/8 |
| TT3 | 12.50 | 7.50 | 2.50 | 27.50 | 8/8 |

Cả 24 trường hợp đúng số vùng. Bốn file train tham gia học nên bảng 8 WAV là kiểm tra toàn dataset, không phải 8 test độc lập.

## 4. Bất thường còn lại

Các biên dưới đây nằm trong silence của LAB và cách biên chuẩn quá một frame. MAE thấp không có nghĩa tất cả biên đều chính xác.

| File | Thuật toán | Status | Lỗi END có dấu (ms) |
|---|---|---|---:|
| phone_F1 | TT1 | boundary inside silence | +115.00 |
| phone_M1 | TT1 | boundary inside silence | +145.00 |
| phone_F2 | TT1 | boundary inside silence | +105.00 |
| phone_F2 | TT2 | boundary inside silence | +55.00 |
| phone_F2 | TT3 | boundary inside silence | +45.00 |

`phone_F2`: TT2 dự đoán [1.01, 4.095] s, GT [1.02, 4.04] s. START sớm 10 ms, END muộn 55 ms, MAE = (10 + 55)/2 = **32.50 ms**. Bỏ Centroid tăng raw speech frames từ 161 lên 224 nhưng không đổi biên cuối trên dataset này; LOW STE vẫn quyết định hỗ trợ cuối.

## 5. Ngưỡng và cách chọn W

Noise floor học từ 497 khung silence TRAIN: mean ≈ 0.0003871108, population std ≈ 0.0007092855, Q95 ≈ 0.0012140145, U = mean + 3std ≈ 0.0025149672.

TT1 có T ≈ 0.0010293375. TT3 có T ≈ 0.0028777337. TT2 tính ngưỡng Energy riêng từng WAV bằng `(W*M1 + M2)/(W+1)`.

TT2 đề xuất candidate **W1** bằng TRAIN frame F1 (0.898698, không phải F1 của FINAL W20). Sau đó [fit_training_model()](../app/pipeline.py#L229) khảo sát W nguyên **1–50** trên FINAL regions của `phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`: 200 dòng TRAIN. Kết quả lưu hiện tại: cả 50 W đồng tối ưu; mean FINAL MAE TRAIN **13.747165532879801 ms**. Chọn **W20** bằng `tie_preference_W=20` cố định. Đây là kết quả của TRAIN hiện tại, không phải giả định mọi dataset đều hòa. Manifest `W_selection_train` ghi `selection_set=train`, `evaluation_protocol=train_final_calibration` và tên TRAIN đã chấm.

`candidate_frame_selected_W`, `candidate_frame_f1`, `candidate_frame_selection_scores`, `candidate_cleanup_records` chỉ mô tả bước candidate. `W`/`finalW` và manifest TRAIN mô tả lựa chọn FINAL; không gán candidate F1 cho W20.

Mọi model, W và `endpoint_noise` được fit từ bốn TRAIN trước khi đọc TEST. Ba thuật toán dùng `schema_version=2`, `parameter_selection_set=train`, `evaluation_protocol=train_selected_reused_test`, `historical_test_exposure=true`. TEST đã được xem trong phát triển trước đây và được dùng lại để chấm; không gọi đây là holdout mới hoặc độc lập. Chạy một file (`--file`) vẫn fit TRAIN trước, không hiệu chỉnh bằng TEST.

## 6. Cách đọc metric

Với một vùng nói, lỗi có dấu là `1000*(predicted - ground_truth)`; âm là sớm, dương là muộn.

`MAE = (abs(start_error) + abs(end_error))/2`.

`RMSE = sqrt((start_error² + end_error²)/2)`.

Nhiều vùng được ghép theo thời gian. Thiếu/thừa vùng làm MAE chính không xác định. Metric chính trong schema 2 là `mae_ms` / `rmse_ms`. Diagnostic sự kiện trong dung sai 100 ms dùng `tolerance_boundary_*`; `matched_boundary_mae_ms` chấm các vùng ghép theo thứ tự. Các diagnostic này không thay thế MAE chính khi thiếu/thừa vùng hoặc biên sai xa.

## 7. Bằng chứng và tài liệu

- [4 test](../outputs/tables/all/test_metrics.csv)
- [8 WAV](../outputs/tables/all_all_dataset/test_metrics.csv)
- [Summary 8 WAV](../outputs/tables/all_all_dataset/summary.csv)
- [200 trường hợp W](../outputs/tables/tt2_w_selection/sweep.csv)
- [Model TT2](../outputs/models/tt2.json)
- [Giải thích TT1](TT1_BINARY_SEARCH_GIAI_THICH.md)
- [Giải thích TT2](TT2_HISTOGRAM_GIAI_THICH.md)
- [Giải thích TT3](TT3_GAUSSIAN_GIAI_THICH.md)

Các bảng là kết quả pipeline hiện hành đã tính sau chuyển TT2 Energy-only. Việc dọn thư mục, tách tài liệu và đổi ngôn ngữ slide không thay đổi code thuật toán.

## 8. Stage1: thay đổi và bằng chứng hồi quy

Stage1 tách detector khỏi chấm LAB, fit toàn bộ TRAIN trước TEST, hiệu chỉnh W trên TRAIN FINAL, khóa mô hình khi đánh giá và chuyển sang schema 2. Các main Python hiện hành đã đồng bộ.

[Bảng before/after](../outputs/tables/all_all_dataset/regression_before_after.csv) lưu đối chiếu 24 file/thuật toán. Lượt tái tạo Python do tác vụ tích hợp chạy xác nhận FINAL regions, LOW/HIGH, MAE/RMSE, frame F1, số vùng và status không đổi so với baseline. Mean TEST vẫn 20.00 / 13.75 / 12.50 ms; năm trường hợp END muộn trong mục 4 vẫn còn. Thay đổi giao thức không tạo tuyên bố cải thiện độ chính xác.

Đã hoàn tất Stage1 và Stage4 bắt buộc ngày 05/10/2026 tại thư mục chính `H:/GIUAKI_XLTHS`. Bộ test cuối đạt **98/98 trong 22.091 s**. Ba notebook đã chạy bằng **ba kernel mới**, mỗi notebook khớp đầy đủ model/metric/FINAL/ngưỡng với Python ở 8/8 WAV, tổng **24/24**; TT2 kiểm tra 200 dòng TRAIN và digest model đã khóa. Auditor xác nhận saved outputs hợp lệ.

Các lệnh nghiệm thu đã chạy bằng Python trong `.venv/Scripts/python.exe`: `-m unittest discover -s tests -v`; `tools/build_notebooks.py`; `tools/run_notebooks.py`; `tools/run_notebooks.py --validate-only`. Các lệnh trên đều hoàn tất thành công; đối chiếu hồi quy, hash input và manifest ZIP được kiểm tra riêng trong tác vụ tích hợp.

Cả **15 PNG nhúng** đã được xem trực quan. [ZIP CODE](../submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip) chứa đúng **ba notebook đã thực thi**, byte trùng các file ngoài ZIP, không WAV/LAB/audio; xem [hướng dẫn notebook](../notebooks/README.md). Trước đó, cả ba notebook cũng chạy bằng kernel mới trong ba thư mục độc lập, mỗi thư mục chỉ có notebook tương ứng và data, không có module Python của dự án. Runtime thực tế là **Python 3.14**; source cells chỉ được kiểm tra cú pháp theo Python 3.10, chưa xác minh runtime 3.10.

[Bảng hồi quy](../outputs/tables/all_all_dataset/regression_before_after.csv) xác nhận 24/24 trường hợp giữ nguyên so với `111ab37`, **0 regression**; đối chiếu SHA256 xác nhận **28 file data/Source** không đổi. W20 vẫn là lựa chọn theo hòa của toàn bộ W1…50 trên TRAIN FINAL, khác candidate W1/F1 ≈ 0.898698. Năm trường hợp END muộn ở mục 4 vẫn còn.

Tích hợp cục bộ trên branch `codex/endpoint-train-calibration`, giữ `main`, chưa push GitHub. Giai đoạn 2/3/5 hoãn. Cả chín binary PPTX/PDF và `THUAT_TOAN_1/2/3.zip` giữ nguyên byte, là **previous version trước Stage1**, chưa dùng làm gói nộp hiện hành. STT nhóm/họ tên/MSSV còn placeholder vì người dùng chưa cung cấp.
