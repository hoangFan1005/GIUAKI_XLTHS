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

Train-F1 proposal của TT2 là **W = 1**. Bước chọn W cuối khảo sát mọi W nguyên **1–50** trên 4 test bằng FINAL-region MAE; cả 50 giá trị đồng tối ưu, giữ **W = 20** đã chọn trước đó khi đồng hạng. W20 chưa phải nghiệm tối ưu duy nhất.

LAB test tham gia chọn W theo yêu cầu của người dùng. Do đó, kết quả TT2 được ghi `test_tuned_not_independent`, không phải đánh giá độc lập. TT1/TT3 và noise statistics chỉ học TRAIN. Dự đoán với model cố định không dùng GT để sửa biên và không có rule riêng theo filename.

## 6. Cách đọc metric

Với một vùng nói, lỗi có dấu là `1000*(predicted - ground_truth)`; âm là sớm, dương là muộn.

`MAE = (abs(start_error) + abs(end_error))/2`.

`RMSE = sqrt((start_error² + end_error²)/2)`.

Nhiều vùng được ghép theo thời gian. Thiếu/thừa vùng làm MAE chính không xác định; matched-only MAE được ghi riêng. Trong CSV, `mae_ms` và `boundary_MAE_ms` là metric chính. Cột phụ `boundary_mae_ms` chỉ ghép trong dung sai 100 ms và có thể bỏ biên sai quá xa; không dùng thay MAE chính.

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
