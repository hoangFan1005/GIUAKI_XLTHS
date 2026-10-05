# Gói CODE chính và artifacts trước Stage1

[JUPYTER_NOTEBOOKS_CODE_ONLY.zip](JUPYTER_NOTEBOOKS_CODE_ONLY.zip) là gói CODE chính đã tái tạo và kiểm tra ngày **05/10/2026**: đúng ba notebook đã chạy, byte trùng file ngoài ZIP, không WAV/LAB/audio/module Python/slide. Xem [hướng dẫn notebook](../notebooks/README.md).

Ba notebook chạy bằng ba kernel mới tại thư mục chính, khớp đầy đủ model/metric/FINAL/ngưỡng với Python ở **24/24** trường hợp trên tám WAV. TT2 có 200 dòng TRAIN, kiểm tra digest khóa model; saved outputs hợp lệ, tổng 15 PNG đã được xem trực quan. Trước đó ba notebook cũng chạy độc lập, mỗi thư mục chỉ có notebook riêng và data. Runtime thực tế **Python 3.14**; Python 3.10 mới được kiểm tra cú pháp, chưa chạy runtime.

Bộ test cuối **98/98 đạt trong 22.091 s**; 24/24 trường hợp giữ nguyên so với `111ab37`, 0 regression, mean TEST 20.00/13.75/12.50 ms. SHA256 của 28 file data/Source không đổi. Xem [bảng hồi quy](../outputs/tables/all_all_dataset/regression_before_after.csv) và [kết quả hiện tại](../reports/KET_QUA_HIEN_TAI.md).

`THUAT_TOAN_1.zip`, `THUAT_TOAN_2.zip`, `THUAT_TOAN_3.zip` và PPTX/PDF hiện có là **previous version (trước Stage1)**, cả chín binary giữ nguyên byte, chưa đồng bộ giao thức TRAIN/schema 2; không dùng chúng như gói nộp hiện hành. Các main Python trong dự án đã đồng bộ Stage1.

Model/noise/W fit TRAIN trước TEST; TT2 W1–50 hòa trên TRAIN FINAL, chọn W20 bằng tie preference 20. Candidate W1/F1 ≈ 0.898698 là diagnostic riêng. TEST đã được xem trong phát triển, protocol `train_selected_reused_test`, `historical_test_exposure=true`. CLI chấm cần WAV/LAB cùng tên. Họ tên/MSSV và STT nhóm thật còn placeholder, chờ thông tin người dùng theo hướng dẫn notebook.

Đã hoàn tất Stage1 và Stage4 bắt buộc; Stage2/3/5 hoãn. Branch tích hợp cục bộ `codex/endpoint-train-calibration` giữ `main`; chưa push GitHub. Năm trường hợp END muộn đã công bố vẫn còn; đợt này không có tuyên bố cải thiện MAE.
