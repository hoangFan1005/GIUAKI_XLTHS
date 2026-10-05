# Gói CODE chính và artifacts trước Stage1

[JUPYTER_NOTEBOOKS_CODE_ONLY.zip](JUPYTER_NOTEBOOKS_CODE_ONLY.zip) là gói CODE chính: chỉ ba notebook đã chạy, không WAV/LAB/module Python/slide. Xem [hướng dẫn notebook](../notebooks/README.md). Tác vụ tích hợp tái tạo và kiểm tra kernel mới trước khi chốt gói.

`THUAT_TOAN_1.zip`, `THUAT_TOAN_2.zip`, `THUAT_TOAN_3.zip` và PPTX/PDF hiện có là **previous version (trước Stage1)**, chưa đồng bộ giao thức TRAIN/schema 2; không dùng chúng như gói nộp hiện hành. Các main Python trong dự án đã đồng bộ Stage1.

Model/noise/W fit TRAIN trước TEST; TT2 W1–50 chọn theo TRAIN FINAL với tie preference 20. TEST đã được xem trong phát triển, protocol `train_selected_reused_test`, `historical_test_exposure=true`. CLI chấm cần WAV/LAB cùng tên. Họ tên/MSSV và STT nhóm thật còn chờ bổ sung theo hướng dẫn notebook.
