# Công cụ tạo lại artifacts

Các main chạy bài tập không phụ thuộc thư mục này. Công cụ này giúp cập nhật slide và ZIP sau khi có kết quả mới hoặc bổ sung họ tên/MSSV.

## Mã nguồn Jupyter Notebook

```powershell
.\.venv\Scripts\python.exe tools/build_notebooks.py
.\.venv\Scripts\python.exe tools/run_notebooks.py
```

Builder đưa hàm tính toán vào ba notebook độc lập. Runner thực thi bằng kernel mới, lưu số liệu/đồ thị vào `.ipynb`, đối chiếu biên/MAE trên cả 8 WAV với pipeline Python và tạo `submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip` không chứa WAV/LAB. Runner dùng Python benchmark schema 2 đã tái tạo để đối chiếu toàn bộ model, metric, LOW/HIGH, FINAL, 200 dòng TRAIN W và digest. Mọi model được fit TRAIN trước TEST; TEST có historical exposure. Metric chính mae_ms/rmse_ms; calibration final_region_mae_ms; tolerance_boundary_* và matched_boundary_mae_ms là diagnostic.

`--validate-only` chỉ kiểm tra outputs đã lưu và đóng gói lại. `--group-number N` đặt STT nhóm thật trong tên thư mục ZIP. Xem [notebooks/README.md](../notebooks/README.md).

## Slide — binaries previous version trước Stage1

1. Cập nhật kết quả bằng main cần thiết.
2. `python tools/export_slide_data.py` xuất dữ liệu từ WAV và model hiện hành.
3. Chạy `tools/build_slides.mjs` bằng Node có `@oai/artifact-tool` và runtime authoring của Codex; xem các biến cấu hình ở đầu script.
4. `python tools/export_slide_pdf.py` tạo PDF từ hình slide đã render; cần reportlab, Pillow và pypdf.

PPTX/PDF cuối nằm trong `slides/THUAT_TOAN_N/`. `_build/` và `.chart-data-*` là file tạm của quá trình tạo slide, có thể dọn sau khi xem kết quả. Không cần cài các thư viện authoring này trên máy chỉ chạy main Python.

## ZIP Python — previous version trước Stage1

```powershell
python tools/package_submission.py
```

Lệnh cập nhật ba ZIP trong `submission/` trực tiếp, không tạo bản giải nén trùng. ZIP lấy code, tài liệu thuật toán tương ứng, kết quả 4 test và slide tiếng Anh. README trong ZIP lấy W, candidate F1, TRAIN names, tie preference và shared optima từ model schema 2; model cũ bị từ chối. Slide đóng kèm phải xem là previous version trừ khi được tạo lại/kiểm tra riêng. Công cụ giữ nguyên submission/README.md và gói CODE chính JUPYTER_NOTEBOOKS_CODE_ONLY.zip. Thiết kế lại deck tiếng Anh được hoãn; không coi ZIP Python hiện có là gói nộp đã đồng bộ.
