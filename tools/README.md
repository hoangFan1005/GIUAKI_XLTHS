# Công cụ tạo lại artifacts

Các main chạy bài tập không phụ thuộc thư mục này. Công cụ này giúp cập nhật slide và ZIP sau khi có kết quả mới hoặc bổ sung họ tên/MSSV.

## Mã nguồn Jupyter Notebook

Các lệnh dưới đây dành cho người bảo trì dự án làm mới evidence và gói CODE ZIP đã audit. Notebook độc lập vẫn chạy bằng **Restart Kernel and Run All**, rồi **Save**, không cần module dự án hay builder/runner; yêu cầu runner chỉ thuộc audit/đóng gói của dự án.

```powershell
.\.venv\Scripts\python.exe tools/build_notebooks.py
.\.venv\Scripts\python.exe tools/run_notebooks.py
```

Builder đưa hàm tính toán vào ba notebook độc lập. Runner thực thi bằng kernel mới, lưu số liệu/đồ thị vào `.ipynb`, đối chiếu biên/MAE trên cả 8 WAV với pipeline Python và tạo `submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip` không chứa WAV/LAB. Runner dùng Python benchmark schema 2 đã tái tạo để đối chiếu toàn bộ model, metric, LOW/HIGH, FINAL, 200 dòng TRAIN W và digest. Mọi model được fit TRAIN trước TEST; TEST có historical exposure. Metric chính mae_ms/rmse_ms; calibration final_region_mae_ms; tolerance_boundary_* và matched_boundary_mae_ms là diagnostic.

Quy trình là rebuild → fresh execution → audit nguồn đã chạy: sau hai lệnh trên, chạy `tools/run_notebooks.py --validate-only` để kiểm tra outputs đã lưu và đóng gói lại. Runner xóa cell audit tạm rồi lưu SHA-256 của nguồn code cell theo thứ tự ở `endpoint_execution.code_source_sha256`; digest thiếu hoặc source/comment thay đổi sẽ yêu cầu chạy runner lại. Đây là kiểm tra sai lệch vô tình source/outputs, không phải xác thực/chữ ký. Model digest vẫn khóa model trước TEST riêng. `--group-number N` đặt STT nhóm thật trong tên thư mục ZIP. Xem [notebooks/README.md](../notebooks/README.md).

Builder thêm comment theo từng bước xử lý ở tọa độ AST sau hai phép `ast.unparse`, giữ nguyên AST tính toán và docstring của nguồn đã biến đổi. TT3 copied definitions dùng nghiệm Gaussian dịch tâm/chia scale và sigma floor `1e-9`; Gaussian thô `fit`/`predict` vẫn hỗ trợ `low`, còn FINAL HIGH/LOW yêu cầu `speech_direction='high'` và báo lỗi rõ ràng cho `low` hoặc hướng không hợp lệ. Quy tắc 200 ms đo khoảng trống ước lượng giữa support khung hoạt động; 100 ms lọc span support theo heuristic. Frame chồng lấn có thể nối qua khoảng lặng vật lý hoặc giữ xung ngắn, nên hai con số không bảo đảm thời lượng khoảng lặng/voiced audio thực.

## Kiểm tra khoảng lặng và ablation trên TRAIN

```powershell
.\.venv\Scripts\python.exe tools/check_endpoint_robustness.py
```

Công cụ chỉ đọc bốn TRAIN, fit/khóa model một lần rồi chấm bốn nhánh: ngưỡng gốc hoặc HIGH/LOW, mỗi nhánh có/không lọc span 100 ms; tất cả giữ quy tắc gap support 200 ms. TT2 raw không dùng candidate padding. Công cụ không chọn lại W/ngưỡng hoặc sửa model production. Nó xuất 48 ca WAV tổng hợp về silence 190/200/210/250 ms ở hai Fs và sáu phase, hai ví dụ burst 75 ms, 48 dòng TRAIN ablation, summary và manifest tại `outputs/tables/endpoint_robustness/`. MAE chính khi thiếu/thừa vùng là `null`; mean của toàn nhánh cũng `null` nếu có file không chấm được, mean chỉ các dòng hợp lệ được ghi riêng. Các ca true 200/210 ms bị gộp mang nhãn `known_limitation`, không phải pass bảo đảm khoảng lặng vật lý. Xem [báo cáo sửa lỗi](../reports/KET_QUA_SUA_LOI_2026_10_06.md).

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
