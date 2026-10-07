# Kết quả hiện hành — nghiệm thu 07/10/2026

## Hai chế độ endpoint (07/10/2026)

Mặc định `enhanced`: ngưỡng native tạo seed; noise TRAIN và HIGH/LOW xác nhận, giữ speech; nối gap support dưới 200 ms và lọc span dưới 100 ms. `core` giữ quyết định native và chỉ nối gap support dưới 200 ms, không lọc duration hoặc padding FINAL; LOW/HIGH là null. Gap ước lượng đúng 200 ms được giữ riêng. TT1 dùng STE ≥ T, TT2 dùng Energy > T, TT3 giữ hướng high/low của Gaussian native. Đây là hai chế độ endpoint của cùng ba thuật toán; enhanced là phần bổ sung của dự án.

```powershell
python main_tt1.py --endpoint-mode core
python main.py --evaluate-all --compare-endpoint-modes
```

Mỗi mode fit và khóa model độc lập trên bốn TRAIN trước đọc TEST. Kết quả mới ở `outputs/endpoint_modes/core/` và `outputs/endpoint_modes/enhanced/`; bảng so sánh ở `outputs/endpoint_modes/comparison/tables/`. Model schema 3 giữ policy/digest, metric schema 2 tiếp tục `mae_ms`/`rmse_ms`. Thiếu/thừa vùng làm primary mean toàn bộ file undefined; `valid_files_*` là thống kê subset riêng. TEST có `historical_test_exposure=true`, không phải holdout độc lập. Không chọn mode tốt hơn riêng cho từng WAV.

Notebook dùng `MODE="enhanced"` mặc định cho bốn TEST figure và một illustration; luôn fit/khóa `MODELS` cả hai mode trước TEST, xuất 16 dòng tám WAV/mode và bảng bốn TEST/mode. TT2 có 200 TRAIN sweep rows mỗi mode. Core plot native T; Energy TT2 được chuẩn hóa chỉ ở bước hiển thị. Source digest và model lock digests độc lập kiểm tra nguồn/outputs và model; Restart Kernel → Run All → Save vẫn chạy độc lập.

PPTX/PDF và ZIP Python cũ giữ nguyên previous version. Các bảng, noise và ví dụ HIGH/LOW bên dưới mô tả enhanced; không phải công thức quyết định FINAL của core. Xem [báo cáo hai mode](../reports/KET_QUA_ENDPOINT_MODES_2026_10_07.md).


- `models/`: ba model enhanced TT1/TT2/TT3 hiện tại, fit TRAIN và khóa trước TEST; TT2 Energy-only, W = 20. Chưa có model core riêng.
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

Nghiệm thu 07/10: **133/133 tests**, 24/24 trường hợp giữ nguyên và 0 regression; mean bốn TEST TT1/TT2/TT3 vẫn **20.00/13.75/12.50 ms**. Xem [bằng chứng nghiệm thu](tables/hotfix_oct06/verification.json). [Baseline khóa trước sửa tài liệu](tables/endpoint_modes_oct07/baseline.json) giữ ba model, 24 FINAL/masks/metrics/T/W/LOW/HIGH, đặc trưng, hashes và môi trường. Đợt khóa baseline không tái tạo bảng/model hoặc tuyên bố cải thiện số liệu; năm cờ END muộn vẫn còn.

Chạy lại main cập nhật kết quả tương ứng. Các lượt `--file`, `--snr-study` hoặc `--compare-context` sinh thêm thư mục/bảng mới; bản dọn hiện tại chỉ giữ các batch chính.
