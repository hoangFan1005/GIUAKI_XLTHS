# Công cụ tạo lại artifacts

Các main chạy bài tập không phụ thuộc thư mục này. Công cụ này giúp cập nhật slide và ZIP sau khi có kết quả mới hoặc bổ sung họ tên/MSSV.

## Slide

1. Cập nhật kết quả bằng main cần thiết.
2. `python tools/export_slide_data.py` xuất dữ liệu từ WAV và model hiện hành.
3. Chạy `tools/build_slides.mjs` bằng Node có `@oai/artifact-tool` và runtime authoring của Codex; xem các biến cấu hình ở đầu script.
4. `python tools/export_slide_pdf.py` tạo PDF từ hình slide đã render; cần reportlab, Pillow và pypdf.

PPTX/PDF cuối nằm trong `slides/THUAT_TOAN_N/`. `_build/` và `.chart-data-*` là file tạm của quá trình tạo slide, có thể dọn sau khi xem kết quả. Không cần cài các thư viện authoring này trên máy chỉ chạy main Python.

## ZIP

```powershell
python tools/package_submission.py
```

Lệnh cập nhật ba ZIP trong `submission/` trực tiếp, không tạo bản giải nén trùng. ZIP lấy code, tài liệu thuật toán tương ứng, kết quả 4 test và slide tiếng Anh. Chạy sau khi cập nhật các file đó.
