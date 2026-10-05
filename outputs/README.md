# Kết quả hiện hành

- `models/`: model TT1/TT2/TT3; TT2 Energy-only, W = 20.
- `tables/all/`: 4 test × 3 thuật toán.
- `tables/all_all_dataset/`: 8 WAV × 3 thuật toán, gồm cả train nên không phải 8 test độc lập.
- `tables/tt1/`, `tt2/`, `tt3/`: batch riêng.
- `tables/tt2_w_selection/`: W nguyên 1–50 trên đủ 4 test, 200 trường hợp.
- `tables/training_frames.csv`, `dataset_statistics.csv`: dữ liệu thống kê.
- `figures/`: 4 test, hình so sánh và phân phối Gaussian train.
- `diagnostics/`: candidate/final regions, ngưỡng và fallback của từng WAV.
- `predictions/`: các đoạn speech/sil được xuất cho từng WAV.

MAE chính là `mae_ms`/`boundary_MAE_ms`. Cột phụ `boundary_mae_ms` chỉ ghép biên trong dung sai 100 ms; không dùng thay metric chính. Đọc [báo cáo kết quả](../reports/KET_QUA_HIEN_TAI.md) để phân biệt train/test, counts và bất thường.

Chạy lại main cập nhật kết quả tương ứng. Các lượt `--file`, `--snr-study` hoặc `--compare-context` sinh thêm thư mục/bảng mới; bản dọn hiện tại chỉ giữ các batch chính.
