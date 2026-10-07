# Mã nguồn nộp bài — Jupyter Notebook

## Hai chế độ endpoint (07/10/2026)

Mặc định `enhanced`: ngưỡng native tạo seed; noise TRAIN và HIGH/LOW xác nhận, giữ speech; nối gap support dưới 200 ms và lọc span dưới 100 ms. `core` giữ quyết định native và chỉ nối gap support dưới 200 ms, không lọc duration hoặc padding FINAL; LOW/HIGH là null. Gap ước lượng đúng 200 ms được giữ riêng. TT1 dùng STE ≥ T, TT2 dùng Energy > T, TT3 giữ hướng high/low của Gaussian native. Đây là hai chế độ endpoint của cùng ba thuật toán; enhanced là phần bổ sung của dự án.

```powershell
python main_tt1.py --endpoint-mode core
python main.py --evaluate-all --compare-endpoint-modes
```

Mỗi mode fit và khóa model độc lập trên bốn TRAIN trước đọc TEST. Kết quả mới ở `outputs/endpoint_modes/core/` và `outputs/endpoint_modes/enhanced/`; bảng so sánh ở `outputs/endpoint_modes/comparison/tables/`. Model schema 3 giữ policy/digest, metric schema 2 tiếp tục `mae_ms`/`rmse_ms`. Thiếu/thừa vùng làm primary mean toàn bộ file undefined; `valid_files_*` là thống kê subset riêng. TEST có `historical_test_exposure=true`, không phải holdout độc lập. Không chọn mode tốt hơn riêng cho từng WAV.

Notebook dùng `MODE="enhanced"` mặc định cho bốn TEST figure và một illustration; luôn fit/khóa `MODELS` cả hai mode trước TEST, xuất 16 dòng tám WAV/mode và bảng bốn TEST/mode. TT2 có 200 TRAIN sweep rows mỗi mode. Core plot native T; Energy TT2 được chuẩn hóa chỉ ở bước hiển thị. Source digest và model lock digests độc lập kiểm tra nguồn/outputs và model; Restart Kernel → Run All → Save vẫn chạy độc lập.

PPTX/PDF và ZIP Python cũ giữ nguyên previous version. Các bảng, noise và ví dụ HIGH/LOW bên dưới mô tả enhanced; không phải công thức quyết định FINAL của core. Xem [báo cáo hai mode](../reports/KET_QUA_ENDPOINT_MODES_2026_10_07.md).


Theo yêu cầu bổ sung của thầy, mỗi sinh viên có một `.ipynb` riêng. Ba notebook chứa toàn bộ hàm tính toán cần thiết, chạy độc lập với các module `.py` của dự án:

| Thành viên phụ trách | Notebook |
|---|---|
| TT1 — Binary Search | [THUAT_TOAN_1.ipynb](THUAT_TOAN_1.ipynb) |
| TT2 — Histogram chỉ Energy | [THUAT_TOAN_2.ipynb](THUAT_TOAN_2.ipynb) |
| TT3 — Thống kê Gaussian | [THUAT_TOAN_3.ipynb](THUAT_TOAN_3.ipynb) |

## Xem kết quả đã lưu

Mở notebook bằng JupyterLab, VS Code hoặc trình xem notebook. Số liệu và ảnh kết quả được lưu trong chính notebook; không cần WAV/LAB để **xem** chúng. Giữ các outputs này khi nộp; không chọn “Clear All Outputs”.

Notebook có các cell lần lượt: cấu hình → đọc WAV/LAB → chia khung/tính năng lượng thủ công → thuật toán riêng → High/Low và vùng nói cuối → học trên train → đánh giá → bảng và đồ thị. Bảng 4 test và bảng 8 file train + test được ghi rõ riêng.

TT2 chỉ Energy, không F0/Centroid. Notebook fit model/W từ TRAIN trước TEST, khảo sát W nguyên 1–50 theo FINAL. Khi toàn bộ objective hòa, chọn W nhỏ nhất: enhanced hiện chọn **W1** (cả 50 W hòa; mean TRAIN FINAL MAE 13.747165532879801 ms); core chọn W10 vì objective ưu tiên số vùng hợp lệ, khác với enhanced. Candidate frame-F1 cũng chọn W1 (0.898698), nhưng không phải lý do hay metric để chọn FINAL W. Model schema 3 và metric schema 2 ghi `train_selected_reused_test`, `historical_test_exposure=true`; TEST đã có lịch sử được xem, không khẳng định độc lập. Model digest được khóa trước TEST và kiểm tra lại sau chấm. `--file` demo dùng model đã khóa. Metric chính `mae_ms`/`rmse_ms`, calibration `final_region_mae_ms`, diagnostic `tolerance_boundary_*` và `matched_boundary_mae_ms`.

Notebook có bảng và biểu đồ SNR proxy theo từng WAV và nhóm phone/studio, tách TRAIN/TEST. Proxy tính `10 log10(Pspeech+noise / Psilence)` từ khoảng LAB; không có tín hiệu sạch tham chiếu, mỗi nhóm/split chỉ có hai file nên trung bình mang tính mô tả. Một cell riêng khảo sát nhiễu Gaussian tổng hợp ở 20/10/0 dB trên TEST bằng model đã khóa. Đây là độ bền với nhiễu thêm, không phải phép đo SNR môi trường và không được dùng để fit/tune.

TT3 giải equal-density trong tọa độ dịch tâm/chia scale, với sigma floor `1e-9` và kiểm tra log-density. Gaussian thô `fit`/`predict` hỗ trợ hướng `low`; Enhanced FINAL energy HIGH/LOW chỉ hỗ trợ `speech_direction='high'`, từ chối rõ ràng `low` hoặc hướng không hợp lệ lúc fit model FINAL và detect. Quy tắc 200 ms đo khoảng trống ước lượng giữa support khung hoạt động; 100 ms là heuristic lọc span support, không bảo đảm khoảng lặng vật lý hay độ dài voiced audio vì frame chồng lấn.

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
