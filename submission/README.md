# Gói CODE chính — nghiệm thu 07/10/2026

[JUPYTER_NOTEBOOKS_CODE_ONLY.zip](JUPYTER_NOTEBOOKS_CODE_ONLY.zip) là gói CODE hiện hành ngày **07/10/2026**: đúng ba notebook đã tái tạo và thực thi mới cho `core` và `enhanced`, byte trùng file ngoài ZIP, không WAV/LAB/audio/module Python/slide. Mỗi notebook lưu 16 dòng mode-file trên tám WAV (8 dòng mỗi mode) và 5 PNG, tổng 48 trường hợp thuật toán-mode-file và 15 PNG. Xem [hướng dẫn notebook](../notebooks/README.md), [báo cáo nghiệm thu hiện hành](../reports/KET_QUA_NGHIEM_THU_2026_10_07.md) và [final_verification.json](../outputs/tables/endpoint_modes_oct07/final_verification.json).

Gate suite mới sau sửa final review đạt **177/177 tests trong 19.208 s** (21.030 s wall), exit 0; số thực tế và bằng chứng nằm trong báo cáo/JSON nghiệm thu trên. Enhanced giữ **24/24** trường hợp exact, 0 numeric regression; mean bốn TEST TT1/TT2/TT3 vẫn **20.00/13.75/12.50 ms**. Core giữ các lỗi count/primary undefined trong đủ 48 trường hợp; không có tuyên bố cải thiện MAE cho mọi file.

Sáu model (ba thuật toán × hai mode) dùng model schema 3 và metric schema 2: fit và khóa TRAIN trước đọc TEST. TT2 W1–50 chấm FINAL trên bốn TRAIN, **200 dòng mỗi mode**, enhanced chọn W20 bằng tie preference 20 khi hòa. Candidate W1/F1 ≈ 0.898698 là diagnostic riêng. Metric chính là `mae_ms` / `rmse_ms`; thiếu/thừa vùng làm primary undefined. TEST đã được xem trong phát triển: `train_selected_reused_test`, `historical_test_exposure=true`; đổi protocol không tạo holdout độc lập.

`THUAT_TOAN_1.zip`, `THUAT_TOAN_2.zip`, `THUAT_TOAN_3.zip` và PPTX/PDF là **previous version (trước Stage1)**, cả chín binary giữ nguyên byte, chưa đồng bộ giao thức TRAIN/schema 2. Runtime đã chạy là Python 3.14; Python 3.10 chỉ kiểm tra cú pháp. CLI chấm cần WAV/LAB cùng tên. STT nhóm/họ tên/MSSV còn placeholder. Các trạng thái branch/push ngày 05/10 dưới đây là lịch sử, không mô tả trạng thái Git hiện tại.

## Lịch sử hotfix / khóa baseline 07/10/2026

Gate trước triển khai hai mode đạt **133/133 tests trong 13.862 s**, 24/24 trường hợp, ba model enhanced schema 2; mean TEST **20.00/13.75/12.50 ms**. Đợt sửa tài liệu và [khóa baseline](../outputs/tables/endpoint_modes_oct07/baseline.json) khi đó không tái tạo notebook/ZIP. Đây là bằng chứng lịch sử; CODE hiện hành đã được tái tạo cho hai mode. Xem [hotfix verification](../outputs/tables/hotfix_oct06/verification.json). Năm trường hợp END muộn vẫn còn.

## Lịch sử 05/10/2026

Các số test, thời gian chạy và trạng thái tích hợp trong mục này là bằng chứng lịch sử Stage1/Stage4.

[JUPYTER_NOTEBOOKS_CODE_ONLY.zip](JUPYTER_NOTEBOOKS_CODE_ONLY.zip) là gói CODE chính đã tái tạo và kiểm tra ngày **05/10/2026**: đúng ba notebook đã chạy, byte trùng file ngoài ZIP, không WAV/LAB/audio/module Python/slide. Xem [hướng dẫn notebook](../notebooks/README.md).

Ba notebook chạy bằng ba kernel mới tại thư mục chính, khớp đầy đủ model/metric/FINAL/ngưỡng với Python ở **24/24** trường hợp trên tám WAV. TT2 có 200 dòng TRAIN, kiểm tra digest khóa model; saved outputs hợp lệ, tổng 15 PNG đã được xem trực quan. Trước đó ba notebook cũng chạy độc lập, mỗi thư mục chỉ có notebook riêng và data. Runtime thực tế **Python 3.14**; Python 3.10 mới được kiểm tra cú pháp, chưa chạy runtime.

Bộ test cuối **98/98 đạt trong 22.091 s**; 24/24 trường hợp giữ nguyên so với `111ab37`, 0 regression, mean TEST 20.00/13.75/12.50 ms. SHA256 của 28 file data/Source không đổi. Xem [bảng hồi quy](../outputs/tables/all_all_dataset/regression_before_after.csv) và [kết quả hiện tại](../reports/KET_QUA_HIEN_TAI.md).

`THUAT_TOAN_1.zip`, `THUAT_TOAN_2.zip`, `THUAT_TOAN_3.zip` và PPTX/PDF hiện có là **previous version (trước Stage1)**, cả chín binary giữ nguyên byte, chưa đồng bộ giao thức TRAIN/schema 2; không dùng chúng như gói nộp hiện hành. Các main Python trong dự án đã đồng bộ Stage1.

Model/noise/W fit TRAIN trước TEST; TT2 W1–50 hòa trên TRAIN FINAL, chọn W20 bằng tie preference 20. Candidate W1/F1 ≈ 0.898698 là diagnostic riêng. TEST đã được xem trong phát triển, protocol `train_selected_reused_test`, `historical_test_exposure=true`. CLI chấm cần WAV/LAB cùng tên. Họ tên/MSSV và STT nhóm thật còn placeholder, chờ thông tin người dùng theo hướng dẫn notebook.

Đã hoàn tất Stage1 và Stage4 bắt buộc; Stage2/3/5 hoãn. Branch tích hợp cục bộ `codex/endpoint-train-calibration` giữ `main`; chưa push GitHub. Năm trường hợp END muộn đã công bố vẫn còn; đợt này không có tuyên bố cải thiện MAE.
