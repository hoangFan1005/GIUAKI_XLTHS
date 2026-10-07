# Mã nguồn nộp bài — Jupyter Notebook

Theo yêu cầu bổ sung của thầy, mỗi sinh viên có một `.ipynb` riêng. Ba notebook chứa toàn bộ hàm tính toán cần thiết, chạy độc lập với các module `.py` của dự án:

| Thành viên phụ trách | Notebook |
|---|---|
| TT1 — Binary Search | [THUAT_TOAN_1.ipynb](THUAT_TOAN_1.ipynb) |
| TT2 — Histogram chỉ Energy | [THUAT_TOAN_2.ipynb](THUAT_TOAN_2.ipynb) |
| TT3 — Thống kê Gaussian | [THUAT_TOAN_3.ipynb](THUAT_TOAN_3.ipynb) |

## Xem kết quả đã lưu

Mở notebook bằng JupyterLab, VS Code hoặc trình xem notebook. Số liệu và ảnh kết quả được lưu trong chính notebook; không cần WAV/LAB để **xem** chúng. Giữ các outputs này khi nộp; không chọn “Clear All Outputs”.

Notebook có các cell lần lượt: cấu hình → đọc WAV/LAB → chia khung/tính năng lượng thủ công → thuật toán riêng → High/Low và vùng nói cuối → học trên train → đánh giá → bảng và đồ thị. Bảng 4 test và bảng 8 file train + test được ghi rõ riêng.

TT2 chỉ Energy, không F0/Centroid. Notebook fit mọi model/noise/W từ TRAIN trước TEST, chọn W1–50 trên TRAIN FINAL với `tie_preference_W=20`. TRAIN hiện tại hòa cả 50 W, chọn W20; candidate frame-F1 đề xuất W1 riêng (0.898698). Model schema 2 ghi `train_selected_reused_test`, `historical_test_exposure=true`; TEST được dùng lại sau lịch sử phát triển, không khẳng định độc lập. Model digest được khóa trước TEST và kiểm tra lại sau chấm. `--file` demo dùng model đã khóa. Metric chính `mae_ms`/`rmse_ms`, calibration `final_region_mae_ms`, diagnostic `tolerance_boundary_*` và `matched_boundary_mae_ms`.

TT3 giải equal-density trong tọa độ dịch tâm/chia scale, với sigma floor `1e-9` và kiểm tra log-density. Gaussian thô `fit`/`predict` hỗ trợ hướng `low`; FINAL energy HIGH/LOW chỉ hỗ trợ `speech_direction='high'`, từ chối rõ ràng `low` hoặc hướng không hợp lệ lúc fit model FINAL và detect. Quy tắc 200 ms đo khoảng trống ước lượng giữa support khung hoạt động; 100 ms là heuristic lọc span support, không bảo đảm khoảng lặng vật lý hay độ dài voiced audio vì frame chồng lấn.

## Chạy lại trên máy này

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

Chọn notebook và dùng **Restart Kernel and Run All Cells**, sau đó **Save** để xem và lưu kết quả mới. Notebook chạy độc lập, không cần module dự án hoặc script builder/runner. Người bảo trì dự án dùng runner bên dưới để làm mới evidence và gói CODE ZIP đã audit; yêu cầu đó dành cho audit/đóng gói của dự án. Môi trường `.venv` đã được chuẩn bị riêng cho dự án.

Chạy lại cần dữ liệu gốc ở `data/train/` và `data/test/`, mỗi thư mục 4 cặp WAV/LAB. Notebook tìm thư mục `data` từ thư mục đang chạy và các thư mục cha; có thể sửa cấu hình đường dẫn trong cell đầu. Không đọc model, CSV hoặc PNG đã xuất của chương trình Python để dựng kết quả.

Trên máy khác:

```powershell
python -m pip install matplotlib jupyterlab
python -m jupyterlab
```

Đặt các WAV/LAB gốc trên máy đó và chọn đường dẫn trong notebook để chạy lại. Chỉ xem kết quả thì không cần chuyển dữ liệu âm thanh.

## Gói mã nguồn để nộp

[JUPYTER_NOTEBOOKS_CODE_ONLY.zip](../submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip) chỉ chứa **3 notebook đã chạy**, không có WAV/LAB, module Python hay slide. Đồ thị được nhúng trong notebook là kết quả tính toán, không phải file tín hiệu âm thanh.

Trong ZIP, thư mục `STTNhom-PhanDoanTiengNoiKhoangLang` dùng placeholder cho STT nhóm. Đổi `STTNhom` thành số thứ tự nhóm thật theo quy định `STTnhom-TenDeTai` trước khi nộp. Các yêu cầu về slide/tên thành viên khác nằm ngoài đợt chuyển mã nguồn này.

Ba ZIP `THUAT_TOAN_N.zip` cũ là gói Python trước đây; **gói notebook mới** ở liên kết trên là gói mã nguồn cho yêu cầu bổ sung.

## Người bảo trì: tạo lại và kiểm tra gói notebook của dự án

```powershell
.\.venv\Scripts\python.exe tools/build_notebooks.py
.\.venv\Scripts\python.exe tools/run_notebooks.py
```

Lệnh đầu tạo lại notebook và xóa outputs cũ; luôn chạy lệnh thứ hai sau đó. Lệnh thứ hai chạy kernel mới, lưu outputs thật, đối chiếu 24 kết quả với pipeline Python và cập nhật ZIP. Runner bỏ cell audit tạm trước khi lưu SHA-256 của nguồn các code cell theo thứ tự vào `endpoint_execution.code_source_sha256`. Digest thiếu hoặc source/comment bị sửa sau execution sẽ yêu cầu chạy runner lại. Model digest được giữ riêng để kiểm tra model khóa; source digest chỉ phát hiện sai lệch vô tình giữa source và outputs, không phải chữ ký xác thực. Hai lệnh không chạy unit test của dự án.

Sau rebuild và fresh execution, kiểm tra source-bound outputs đã lưu và đóng gói lại:

```powershell
.\.venv\Scripts\python.exe tools/run_notebooks.py --validate-only
.\.venv\Scripts\python.exe tools/run_notebooks.py --validate-only --group-number 1
```

Lệnh cuối dùng số nhóm minh họa, đổi `1` thành số nhóm thật. Nếu tạo môi trường mới cho cả bộ công cụ, dùng `python -m pip install -r requirements-notebooks.txt`.
