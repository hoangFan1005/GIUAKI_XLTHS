# Kết quả hiện tại — cập nhật 07/10/2026

## Nghiệm thu core/enhanced ngày 07/10/2026

[Báo cáo nghiệm thu tích hợp](KET_QUA_NGHIEM_THU_2026_10_07.md): fresh **173 tests PASS trong15.200s**, enhanced24/24numeric exact so baseline,48core/enhanced cases;6TRAIN models khóa trước TEST,3notebooks fresh/15PNG và CODE ZIP đúng3executednotebook bytes. **124protected hashes** trong148baseline unchanged; core TT1/TT2 mean TEST undefined do counts sai, không thay bằng valid-only mean. [Final verification](../outputs/tables/endpoint_modes_oct07/final_verification.json), [before/after](../outputs/tables/endpoint_modes_oct07/final_regression.csv), [mode comparison](../outputs/tables/endpoint_modes_oct07/final_core_enhanced.csv).

Energy thực là STE/N (mean squared sample amplitude); production labels/models/notebooks đã sửa đúng. Timing research giữ study metadata lịch sử47cc00b/provenance; physical silence200/210ms còn có thể bị nối, LOW=T1 cófold+205ms nên chưa đổi default. Python3.10 chưa kiểm. Root đang hoàn tất review độc lập/visual và archive [verification_logs.zip](../outputs/tables/endpoint_modes_oct07/verification_logs.zip). Các nghiệm thu133/137 và số liệu bên dưới là lịch sử, không phải gate mới.

## Cập nhật sửa lỗi ngày 06/10, nghiệm thu ngày 07/10/2026

[Báo cáo sửa lỗi chi tiết](KET_QUA_SUA_LOI_2026_10_06.md) ghi root cause, hàm đã sửa, TRAIN ablation, synthetic timing và nghiệm thu notebook. Solver TT3 đã dịch tâm/chia scale, kiểm tra log-density và sửa ca gần biên phụ thuộc thứ tự lớp; FINAL từ chối rõ hướng Gaussian `low`/không hợp lệ, chưa hỗ trợ hướng đó ở HIGH/LOW. Nghiệm thu lại ngày 07/10 đạt **133/133 tests trong 13.862 s**; ba notebook đã chạy bằng kernel mới, khớp 24/24 WAV, lưu 15 hình và ZIP CODE đã kiểm tra nguồn/bytes. Review độc lập cuối đã hoàn tất, không còn finding trọng yếu trong phạm vi sửa; bằng chứng tại [verification.json](../outputs/tables/hotfix_oct06/verification.json).

[Before/after 24 trường hợp](../outputs/tables/hotfix_oct06/regression.csv) có **0 regression**: MAE/biên/số vùng đều không đổi, mean bốn TEST vẫn 20.00/13.75/12.50 ms. Năm cờ END muộn trong mục 4 vẫn còn. Có thêm 48 ca waveform tổng hợp và 48 dòng ablation chỉ TRAIN; giữ tham số production.

Điều kiện 200 ms hiện đo **gap support ước lượng**, không bảo đảm giữ mọi silence vật lý ≥200 ms: silence thực 200/210 ms vẫn có thể bị gộp do frame chồng lấn. 100 ms lọc span support sau merge, không phải thời lượng voiced samples. Giới hạn này đã được kiểm tra/công bố, chưa sửa bằng estimator mới. Frame 25/hop 10 ms được giữ; Fs44,1k dùng 1102/441 samples, frame thực 24.988662 ms. Slide/PDF và ba ZIP Python cũ giữ nguyên theo yêu cầu, là previous version; notebook ZIP là gói CODE được làm mới. Các đoạn ngày 05/10 bên dưới là lịch sử, bao gồm trạng thái Git ở thời điểm đó.

## 1. Phiên bản đang dùng

Baseline cho đợt đánh giá thuật toán/endpoint ngày 07/10 được khóa tại [baseline.json](../outputs/tables/endpoint_modes_oct07/baseline.json): **ba model enhanced hiện tại và 24 kết quả (8 WAV × 3 thuật toán)**, gồm FINAL regions, full masks, metrics/status, T/W/LOW/HIGH, record/features, SHA256 input/artifacts và versions môi trường. Đối chiếu trước sửa tài liệu khớp đủ 24 diagnostics/metric rows hiện hành và FINAL/MAE/status của nghiệm thu 07/10; mean TEST vẫn 20.00/13.75/12.50 ms. Đây là capture bằng model đã lưu, không fit lại hoặc tái tạo số liệu. Model core riêng chưa được tạo trong đợt khóa baseline.

- TT1: Binary Search cân bằng diện tích nhầm lẫn trên normalized STE.
- TT2: Histogram **chỉ Energy** theo yêu cầu mới của thầy; W được hiệu chỉnh riêng theo mode bằng TRAIN, 64 bins, smoothing radius 2. Enhanced chọn W=1 khi toàn bộ W=1…50 hòa; core chọn W=10 do objective số vùng hợp lệ.
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

TT2 đề xuất candidate **W1** bằng TRAIN frame F1 (0.898698; không dùng F1 để chọn FINAL). Sau đó [fit_training_model()](../app/pipeline.py#L229) khảo sát W nguyên **1–50** trên FINAL regions của `phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`: 200 dòng TRAIN. Enhanced có cả 50 W đồng tối ưu, mean FINAL MAE TRAIN **13.747165532879801 ms**, và chọn **W1** theo tie-break tổng quát “W nhỏ nhất”. Không còn `tie_preference_W=20`. Core chọn W10 vì tiêu chí số vùng hợp lệ cho kết quả khác. Manifest `W_selection_train` ghi `selection_set=train`, `evaluation_protocol=train_final_calibration` và tên TRAIN đã chấm.

`candidate_frame_selected_W`, `candidate_frame_f1`, `candidate_frame_selection_scores`, `candidate_cleanup_records` chỉ mô tả bước candidate. `W`/`finalW` và manifest TRAIN mô tả lựa chọn FINAL; không gán candidate F1 cho W1.

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

## 8. Lịch sử 05/10/2026 — Stage1 và bằng chứng hồi quy

Mục này giữ nguyên bằng chứng Stage1/Stage4 ngày 05/10; số 98 tests và branch/trạng thái push dưới đây chỉ áp dụng thời điểm đó. Nghiệm thu hiện hành 07/10 là 133 tests trong mục cập nhật đầu báo cáo.

Stage1 tách detector khỏi chấm LAB, fit toàn bộ TRAIN trước TEST, hiệu chỉnh W trên TRAIN FINAL, khóa mô hình khi đánh giá và chuyển sang schema 2. Các main Python hiện hành đã đồng bộ.

[Bảng before/after](../outputs/tables/all_all_dataset/regression_before_after.csv) lưu đối chiếu 24 file/thuật toán. Lượt tái tạo Python do tác vụ tích hợp chạy xác nhận FINAL regions, LOW/HIGH, MAE/RMSE, frame F1, số vùng và status không đổi so với baseline. Mean TEST vẫn 20.00 / 13.75 / 12.50 ms; năm trường hợp END muộn trong mục 4 vẫn còn. Thay đổi giao thức không tạo tuyên bố cải thiện độ chính xác.

Đã hoàn tất Stage1 và Stage4 bắt buộc ngày 05/10/2026 tại thư mục chính `H:/GIUAKI_XLTHS`. Bộ test cuối khi tiếp tục công việc đạt **98/98 trong 21.788 s**. Ba notebook đã chạy bằng **ba kernel mới**, mỗi notebook khớp đầy đủ model/metric/FINAL/ngưỡng với Python ở 8/8 WAV, tổng **24/24**; TT2 kiểm tra 200 dòng TRAIN và digest model đã khóa. Auditor xác nhận saved outputs hợp lệ.

Các lệnh nghiệm thu đã chạy bằng Python trong `.venv/Scripts/python.exe`: `-m unittest discover -s tests -v`; `tools/build_notebooks.py`; `tools/run_notebooks.py`; `tools/run_notebooks.py --validate-only`. Các lệnh trên đều hoàn tất thành công; đối chiếu hồi quy, hash input và manifest ZIP được kiểm tra riêng trong tác vụ tích hợp.

Cả **15 PNG nhúng** đã được xem trực quan. [ZIP CODE](../submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip) chứa đúng **ba notebook đã thực thi**, byte trùng các file ngoài ZIP, không WAV/LAB/audio; xem [hướng dẫn notebook](../notebooks/README.md). Trước đó, cả ba notebook cũng chạy bằng kernel mới trong ba thư mục độc lập, mỗi thư mục chỉ có notebook tương ứng và data, không có module Python của dự án. Runtime thực tế là **Python 3.14**; source cells chỉ được kiểm tra cú pháp theo Python 3.10, chưa xác minh runtime 3.10.

[Bảng hồi quy](../outputs/tables/all_all_dataset/regression_before_after.csv) xác nhận 24/24 trường hợp giữ nguyên so với `111ab37`, **0 regression**; đối chiếu SHA256 xác nhận **28 file data/Source** không đổi. Trong bản trước khi sửa tie-break, W20 được chọn theo cấu hình khi toàn bộ W hòa; hiện enhanced chọn W1 theo tie-break nhỏ nhất, còn core chọn W10. Cả hai kết quả được fit riêng từ TRAIN. Candidate frame-F1 cũng đề xuất W1 nhưng là tiêu chí khác. Năm trường hợp END muộn ở mục 4 vẫn còn.

Tích hợp cục bộ trên branch `codex/endpoint-train-calibration`, giữ `main`, chưa push GitHub. Giai đoạn 2/3/5 hoãn. Cả chín binary PPTX/PDF và `THUAT_TOAN_1/2/3.zip` giữ nguyên byte, là **previous version trước Stage1**, chưa dùng làm gói nộp hiện hành. STT nhóm/họ tên/MSSV còn placeholder vì người dùng chưa cung cấp.

**Review cuối đã hoàn tất:** subagent đã review độc lập toàn nhánh từ `111ab37` đến `2629237`, đọc source diff, code/metadata notebook thật, hình xuất và bằng chứng nghiệm thu. Không phát hiện lỗi Critical, Important hoặc Minor; phần tài liệu bàn giao đạt specification/quality PASS. Không cần sửa code thêm.

Sau khi tiếp tục công việc bị dừng do credit, đã chạy lại đủ 98 test và kiểm tra hồi quy 24 trường hợp; kết quả vẫn giữ nguyên, không file nào xấu đi. Kiểm tra saved outputs và SHA256 xác nhận ba notebook/ZIP vẫn trùng artifacts đã thực thi; 28 file data/Source không đổi. Không chạy lại kernel vì code và artifacts không thay đổi. Bằng chứng tạm được dọn sau khi chốt báo cáo.

Kết luận review chỉ áp dụng cho Stage1 và Stage4 đã duyệt. Các ablation giảm MAE, guard mở rộng của Stage3, slide chung/refresh binary cũ, runtime Python 3.10 và đánh giá trên holdout chưa từng được xem vẫn chưa được xác minh trong đợt này. Thông tin nhóm/thành viên tiếp tục chờ người dùng cung cấp.
