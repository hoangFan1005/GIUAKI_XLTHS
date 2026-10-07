# Công cụ tạo lại artifacts

## Hai chế độ endpoint (07/10/2026)

Mặc định `enhanced`: ngưỡng native tạo seed; noise TRAIN và HIGH/LOW xác nhận, giữ speech; nối gap support dưới 200 ms và lọc span dưới 100 ms. `core` giữ quyết định native và chỉ nối gap support dưới 200 ms, không lọc duration hoặc padding FINAL; LOW/HIGH là null. Gap ước lượng đúng 200 ms được giữ riêng. TT1 dùng STE ≥ T, TT2 dùng Energy > T, TT3 giữ hướng high/low của Gaussian native. Đây là hai chế độ endpoint của cùng ba thuật toán; enhanced là phần bổ sung của dự án.

```powershell
python main_tt1.py --endpoint-mode core
python main.py --evaluate-all --compare-endpoint-modes
```

Mỗi mode fit và khóa model độc lập trên bốn TRAIN trước đọc TEST. Kết quả mới ở `outputs/endpoint_modes/core/` và `outputs/endpoint_modes/enhanced/`; bảng so sánh ở `outputs/endpoint_modes/comparison/tables/`. Model schema 3 giữ policy/digest, metric schema 2 tiếp tục `mae_ms`/`rmse_ms`. Thiếu/thừa vùng làm primary mean toàn bộ file undefined; `valid_files_*` là thống kê subset riêng. TEST có `historical_test_exposure=true`, không phải holdout độc lập. Không chọn mode tốt hơn riêng cho từng WAV.

Notebook dùng `MODE="enhanced"` mặc định cho bốn TEST figure và một illustration; luôn fit/khóa `MODELS` cả hai mode trước TEST, xuất 16 dòng tám WAV/mode và bảng bốn TEST/mode. TT2 có 200 TRAIN sweep rows mỗi mode. Core plot native T; Energy TT2 được chuẩn hóa chỉ ở bước hiển thị. Source digest và model lock digests độc lập kiểm tra nguồn/outputs và model; Restart Kernel → Run All → Save vẫn chạy độc lập.

PPTX/PDF và ZIP Python cũ giữ nguyên previous version. Các bảng, noise và ví dụ HIGH/LOW bên dưới mô tả enhanced; không phải công thức quyết định FINAL của core. Xem [báo cáo hai mode](../reports/KET_QUA_ENDPOINT_MODES_2026_10_07.md).


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
