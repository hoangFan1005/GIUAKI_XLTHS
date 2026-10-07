# Kết quả hiện hành — nghiệm thu 07/10/2026

## Hai chế độ endpoint (07/10/2026)

Mặc định `enhanced`: ngưỡng native tạo seed; noise TRAIN và HIGH/LOW xác nhận, giữ speech; nối gap support dưới 200 ms và lọc span dưới 100 ms. `core` giữ quyết định native và chỉ nối gap support dưới 200 ms, không lọc duration hoặc padding FINAL; LOW/HIGH là null. Gap ước lượng đúng 200 ms được giữ riêng. TT1 dùng STE ≥ T, TT2 dùng Energy > T, TT3 giữ hướng high/low của Gaussian native. Đây là hai chế độ endpoint của cùng ba thuật toán; enhanced là phần bổ sung của dự án.

```powershell
python main_tt1.py --endpoint-mode core
python main.py --evaluate-all --compare-endpoint-modes
```

Mỗi mode fit và khóa model độc lập trên bốn TRAIN trước đọc TEST. Kết quả mới ở `outputs/endpoint_modes/core/` và `outputs/endpoint_modes/enhanced/`; bảng so sánh ở `outputs/endpoint_modes/comparison/tables/`. Model schema 3 giữ policy/digest, metric schema 2 tiếp tục `mae_ms`/`rmse_ms`. Thiếu/thừa vùng làm primary mean toàn bộ file undefined; `valid_files_*` là thống kê subset riêng. TEST có `historical_test_exposure=true`, không phải holdout độc lập. Không chọn mode tốt hơn riêng cho từng WAV.

Ba notebook đã tái tạo và thực thi mới, dùng `MODE="enhanced"` mặc định cho bốn TEST figure và một illustration (5 PNG mỗi notebook); luôn fit/khóa `MODELS` cả hai mode trước TEST, xuất 16 dòng mode-file trên tám WAV (8 dòng mỗi mode) và bảng bốn TEST/mode. TT2 có 200 TRAIN sweep rows mỗi mode. Tổng 48 trường hợp và 15 PNG; [CODE ZIP hiện hành](../submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip) có đúng ba notebook byte trùng bản ngoài ZIP. Core plot native T; Energy TT2 được chuẩn hóa chỉ ở bước hiển thị. Source digest và model lock digests độc lập kiểm tra nguồn/outputs và model; Restart Kernel → Run All → Save vẫn chạy độc lập.

Nghiệm thu hiện hành: **177/177 tests trong 19.208 s** (21.030 s wall), exit 0; [báo cáo đầy đủ](../reports/KET_QUA_NGHIEM_THU_2026_10_07.md) và [final_verification.json](tables/endpoint_modes_oct07/final_verification.json) ghi số thực tế, 24 enhanced exact và 48 trường hợp hai mode. Sáu model hiện hành ở `endpoint_modes/{core,enhanced}/models/`; các bảng mode dùng `tables/all/` (bốn TEST) và `tables/all_all_dataset/` (tám WAV). `dataset_statistics.csv` mỗi mode giữ tám nguồn duy nhất, bốn TRAIN/bốn TEST.

`endpoint_modes/comparison/artifact_audit.json` là snapshot lịch sử sau sửa units Task5, trước sửa routing final review; full-file source hashes trong snapshot không mô tả writer hiện hành. Audit hiện hành nằm trong `final_verification.json` nêu trên; source digest notebook, saved output và CODE ZIP vẫn khớp vì writer không được nhúng vào notebook.

`python main.py --evaluate-all --compare-endpoint-modes` ghi bốn bảng benchmark đầy đủ ở `endpoint_modes/comparison/tables/{all_metrics,all_summary,test_metrics,test_summary}.csv`. Lượt partial giữ bảng riêng tại `comparison/tables/<run_name>/`: mặc định bốn TEST là `all/`; thuật toán cố định đánh giá tám WAV là `ttN_all_dataset/`; `--file` thêm `single/<inputstem>/` dưới `all/` hoặc `ttN/`. Các lượt partial giữ nguyên bốn bảng benchmark đầy đủ. Thống kê một file nằm trong thư mục run mode tương ứng, chỉ ghi nguồn đã chọn.

PPTX/PDF và ZIP Python cũ giữ nguyên previous version. Các bảng, noise và ví dụ HIGH/LOW bên dưới mô tả enhanced; không phải công thức quyết định FINAL của core. Xem [báo cáo hai mode](../reports/KET_QUA_ENDPOINT_MODES_2026_10_07.md).


## Lịch sử thư mục canonical / hotfix trước hai mode

Danh sách dưới đây mô tả outputs canonical cũ, được giữ nguyên byte làm evidence lịch sử. Các thư mục hiện hành là `endpoint_modes/` nêu trên.

- `models/`: ba model enhanced TT1/TT2/TT3 lịch sử, fit TRAIN và khóa trước TEST; TT2 Energy-only, W = 20. Khi đó chưa có model core riêng.
- `tables/all/`: 4 test × 3 thuật toán.
- `tables/all_all_dataset/`: 8 WAV × 3 thuật toán, gồm cả train nên không phải 8 test độc lập.
- `tables/tt1/`, `tt2/`, `tt3/`: batch riêng.
- `tables/tt2_w_selection/`: W nguyên 1–50 trên đủ 4 TRAIN (`phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`), 200 trường hợp; chấm FINAL, chọn W20 bằng tie preference 20 khi hòa.
- `tables/training_frames.csv`, `dataset_statistics.csv`: dữ liệu thống kê.
- `figures/`: 4 test, hình so sánh và phân phối Gaussian train.
- `diagnostics/`: candidate/final regions, ngưỡng và fallback của từng WAV.
- `predictions/`: các đoạn speech/sil được xuất cho từng WAV.

Metric schema 2 dùng MAE/RMSE chính `mae_ms` / `rmse_ms`. Thiếu/thừa vùng làm metric chính không xác định. `matched_boundary_mae_ms` chấm vùng ghép theo thứ tự; `tolerance_boundary_*` là diagnostic sự kiện trong dung sai 100 ms, không thay metric chính. Đọc [báo cáo kết quả](../reports/KET_QUA_HIEN_TAI.md) để phân biệt TRAIN/TEST, counts và bất thường.

Model schema 3 với metric schema 2 ghi `parameter_selection_set=train`, `evaluation_protocol=train_selected_reused_test`, `historical_test_exposure=true`. TEST đã được xem trong phát triển và được dùng lại; không phải holdout mới độc lập. Model legacy hiệu chỉnh bằng TEST vẫn phải giữ provenance cũ.

Nghiệm thu hotfix / baseline lịch sử 07/10: **133/133 tests**, 24/24 trường hợp giữ nguyên và 0 regression; mean bốn TEST TT1/TT2/TT3 **20.00/13.75/12.50 ms**. Xem [bằng chứng hotfix lịch sử](tables/hotfix_oct06/verification.json). [Baseline khóa trước sửa tài liệu](tables/endpoint_modes_oct07/baseline.json) giữ ba model, 24 FINAL/masks/metrics/T/W/LOW/HIGH, đặc trưng, hashes và môi trường. Đợt khóa baseline khi đó không tái tạo bảng/model hoặc tuyên bố cải thiện số liệu; năm cờ END muộn vẫn còn.

Chạy lại main cập nhật kết quả tương ứng. Các lượt `--file`, `--snr-study` hoặc `--compare-context` sinh thêm thư mục/bảng mới; bản dọn hiện tại chỉ giữ các batch chính.
