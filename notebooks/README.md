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

TT2 chỉ dùng Energy. W vẫn chọn một giá trị chung bằng khảo sát 1–50 trên 4 test, giữ W20 khi hòa; notebook ghi rõ đây là `test_tuned_not_independent`. Không tính F0 hoặc Spectral Centroid.

## Chạy lại trên máy này

Từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

Chọn notebook và dùng **Restart Kernel and Run All Cells**, sau đó **Save**. Môi trường `.venv` đã được chuẩn bị riêng cho dự án.

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

## Tạo lại và kiểm tra notebook

```powershell
.\.venv\Scripts\python.exe tools/build_notebooks.py
.\.venv\Scripts\python.exe tools/run_notebooks.py
```

Lệnh đầu tạo lại notebook và xóa outputs cũ; luôn chạy lệnh thứ hai sau đó. Lệnh thứ hai chạy kernel mới, lưu outputs thật, đối chiếu 24 kết quả với pipeline Python và cập nhật ZIP. Không chạy unit test của dự án.

Chỉ kiểm tra outputs đã lưu và đóng gói lại:

```powershell
.\.venv\Scripts\python.exe tools/run_notebooks.py --validate-only
.\.venv\Scripts\python.exe tools/run_notebooks.py --validate-only --group-number 1
```

Lệnh cuối dùng số nhóm minh họa, đổi `1` thành số nhóm thật. Nếu tạo môi trường mới cho cả bộ công cụ, dùng `python -m pip install -r requirements-notebooks.txt`.
