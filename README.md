# Giữa kỳ XLTHS — phân đoạn tiếng nói và khoảng lặng

## Hai chế độ endpoint (07/10/2026)

Mặc định `enhanced`: ngưỡng native tạo seed; noise TRAIN và HIGH/LOW xác nhận, giữ speech; nối gap support dưới 200 ms và lọc span dưới 100 ms. `core` giữ quyết định native và chỉ nối gap support dưới 200 ms, không lọc duration hoặc padding FINAL; LOW/HIGH là null. Gap ước lượng đúng 200 ms được giữ riêng. TT1 dùng STE ≥ T, TT2 dùng Energy > T, TT3 giữ hướng high/low của Gaussian native. Đây là hai chế độ endpoint của cùng ba thuật toán; enhanced là phần bổ sung của dự án.

```powershell
python main_tt1.py --endpoint-mode core
python main.py --evaluate-all --compare-endpoint-modes
```

Mỗi mode fit và khóa model độc lập trên bốn TRAIN trước đọc TEST. Kết quả mới ở `outputs/endpoint_modes/core/` và `outputs/endpoint_modes/enhanced/`; bảng so sánh ở `outputs/endpoint_modes/comparison/tables/`. Model schema 3 giữ policy/digest, metric schema 2 tiếp tục `mae_ms`/`rmse_ms`. Thiếu/thừa vùng làm primary mean toàn bộ file undefined; `valid_files_*` là thống kê subset riêng. TEST có `historical_test_exposure=true`, không phải holdout độc lập. Không chọn mode tốt hơn riêng cho từng WAV.

Notebook dùng `MODE="enhanced"` mặc định cho bốn TEST figure và một illustration; luôn fit/khóa `MODELS` cả hai mode trước TEST, xuất 16 dòng tám WAV/mode và bảng bốn TEST/mode. TT2 có 200 TRAIN sweep rows mỗi mode. Core plot native T; Energy TT2 được chuẩn hóa chỉ ở bước hiển thị. Source digest và model lock digests độc lập kiểm tra nguồn/outputs và model; Restart Kernel → Run All → Save vẫn chạy độc lập.

PPTX/PDF và ZIP Python cũ giữ nguyên previous version. Các bảng, noise và ví dụ HIGH/LOW bên dưới mô tả enhanced; không phải công thức quyết định FINAL của core. Xem [báo cáo hai mode](reports/KET_QUA_ENDPOINT_MODES_2026_10_07.md).


Ba thuật toán Python: **TT1 Binary Search**, **TT2 Histogram chỉ Energy**, **TT3 Gaussian**. Đặc trưng, ngưỡng và metrics được tự tính; Matplotlib dùng để vẽ. Cấu hình chung: frame **25 ms**, hop **10 ms**; **200 ms** đo khoảng trống ước lượng giữa các support khung hoạt động; **100 ms** chỉ lọc theo span vùng support trong enhanced. Hai quy tắc này là heuristic, không bảo đảm khoảng lặng vật lý 200 ms hoặc tiếng nói thực 100 ms vì các khung có thể chồng lấn. Không tính F0.

## Mã nguồn Jupyter theo yêu cầu bổ sung

Mỗi sinh viên có một notebook riêng, chứa toàn bộ code và kết quả đã chạy:

- [TT1 — Binary Search](notebooks/THUAT_TOAN_1.ipynb)
- [TT2 — Histogram Energy](notebooks/THUAT_TOAN_2.ipynb)
- [TT3 — Gaussian](notebooks/THUAT_TOAN_3.ipynb)

[Hướng dẫn mở/chạy notebook](notebooks/README.md) · [ZIP chỉ chứa 3 notebook](submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip).

Người bảo trì dự án tạo lại gói CODE bằng `tools/build_notebooks.py` → `tools/run_notebooks.py` bằng kernel mới → `tools/run_notebooks.py --validate-only`. Runner lưu SHA-256 của nguồn các code cell theo thứ tự trong `endpoint_execution.code_source_sha256`. Thiếu digest hoặc sửa code/comment sau khi chạy sẽ yêu cầu thực thi lại trước audit/đóng gói của dự án; đây là kiểm tra sai lệch vô tình giữa source và outputs, không phải chữ ký xác thực. Model digest vẫn kiểm tra model khóa trước TEST riêng. Notebook độc lập vẫn chạy bằng **Restart Kernel and Run All**, rồi **Save**, không cần module dự án hay builder/runner.

Kết quả số và đồ thị nằm trong `.ipynb`; xem không cần âm thanh. Gói mã nguồn notebook không chứa WAV/LAB. Mở trên máy này bằng `.\.venv\Scripts\python.exe -m jupyterlab notebooks` từ thư mục dự án. Các lệnh main Python dưới đây đã đồng bộ Stage1/schema 2. PPTX/PDF và ba ZIP Python là **previous version (trước Stage1)**; gói CODE chính hiện dùng notebook ZIP ở trên.

## 1. Chạy chương trình

Yêu cầu Python ≥ 3.10, có Tkinter để mở cửa sổ biểu đồ.

```powershell
python -m pip install -r requirements.txt
python main_tt1.py
python main_tt2.py
python main_tt3.py
```

Chọn một main tương ứng thuật toán mình phụ trách. Mặc định xử lý 4 WAV test và mở 4 cửa sổ ở bốn góc. `main.py` so sánh cả 3 thuật toán.

Chạy một file hoặc chỉ xuất kết quả:

```powershell
python main_tt2.py --file phone_F2.wav
python main_tt2.py --no-show
python main.py --evaluate-all
```

Thay tên WAV trong lệnh để thử file khác trong `data/test/` hoặc `data/train/`; có thể truyền đường dẫn WAV ngoài dự án nếu có LAB cùng tên. `--evaluate-all` đánh giá 8 WAV train + test và tự chạy headless. Dùng main riêng để chỉ chạy một thuật toán. `python main.py --help` liệt kê các cờ khác.

Khi chuyển máy, sao chép main tương ứng, `app/`, `algorithms/`, `core/`, `data/` và `requirements.txt`. Một file main riêng cần các module chung để chạy.

## 2. Ba tài liệu giải thích riêng

Mỗi tài liệu gồm ý tưởng, ví dụ tính tay, công thức, các hàm Python, High/Low, biên cuối và MAE:

| Thuật toán | Tài liệu |
|---|---|
| TT1 Binary Search | [Giải thích TT1](reports/TT1_BINARY_SEARCH_GIAI_THICH.md) |
| TT2 Histogram Energy-only | [Giải thích TT2](reports/TT2_HISTOGRAM_GIAI_THICH.md) |
| TT3 Gaussian | [Giải thích TT3](reports/TT3_GAUSSIAN_GIAI_THICH.md) |

[Kết quả hiện tại và giới hạn](reports/KET_QUA_HIEN_TAI.md) tổng hợp 4 test và toàn bộ 8 WAV. [Báo cáo sửa lỗi ngày 06/10, nghiệm thu 07/10](reports/KET_QUA_SUA_LOI_2026_10_06.md) ghi hai sửa TT3, khảo sát timing/TRAIN, 133 tests và hồi quy 24 trường hợp không đổi.

TT2 chỉ dùng **Energy**, không Centroid/F0. Trong enhanced, W20 chọn trên bốn TRAIN FINAL, khảo sát W1–50 với tie preference cố định 20; tất cả 50 W hòa trong TRAIN hiện tại (mean 13.747165532879801 ms). Candidate frame-F1 đề xuất W1 riêng. Mọi model/noise/W fit TRAIN trước TEST. Model schema 3 và metric schema 2 ghi `train_selected_reused_test` và `historical_test_exposure=true`: TEST có lịch sử được xem, không khẳng định độc lập.

TT3 giải giao điểm Gaussian bằng tọa độ được dịch tâm/chia scale, sigma floor `1e-9`, nghiệm quadratic ổn định và kiểm tra log-density. Hàm Gaussian thô `fit`/`predict` vẫn hỗ trợ hướng `low`; Enhanced FINAL HIGH/LOW yêu cầu `speech_direction='high'`; core giữ hướng native. Enhanced từ chối model `low` hoặc hướng không hợp lệ khi fit/detect FINAL; core dùng hướng high/low native đã học.

## 3. Bố trí thư mục

```text
GIUAKI_XLTHS/
  main.py, main_tt1.py, main_tt2.py, main_tt3.py
  README.md, requirements.txt, .gitignore
  algorithms/       Ba thuật toán
  app/              CLI, pipeline, vẽ, chọn W
  core/             WAV/LAB, đặc trưng, endpoints, metrics
  data/             Dữ liệu train/test gốc
  Source/           Đề, hướng dẫn, slide môn học, tài liệu tham khảo
  reports/          Ba hướng dẫn riêng và kết quả hiện tại
  outputs/          Model, bảng batch, hình và prediction
  slides/           Ba bộ slide tiếng Anh, PPTX/PDF
  submission/       ZIP notebook CODE chính và ZIP Python cũ
  tests/            Bộ kiểm tra hiện có
  tools/            Công cụ tái tạo slide và ZIP
  notebooks/        Ba notebook độc lập, kèm outputs đã chạy
```

Đã dọn snapshot code cũ, báo cáo debug lịch sử, bản giải nén ZIP, cache, hình render kiểm tra và kết quả `single/` cũ. Chạy chương trình có thể sinh lại `__pycache__` và kết quả của lượt chạy mới.

## 4. Đọc kết quả

| Đường dẫn trong `outputs/` | Nội dung |
|---|---|
| `models/` | Model TT1/TT2/TT3 |
| `tables/all/` | So sánh 3 thuật toán trên 4 test |
| `tables/all_all_dataset/` | 24 trường hợp trên 8 WAV; counts, status, MAE |
| `tables/tt1/`, `tt2/`, `tt3/` | Batch riêng của mỗi thuật toán |
| `tables/tt2_w_selection/` | 200 kết quả khảo sát W |
| `tables/training_frames.csv` | STE chuẩn hóa và nhãn 1291 khung train |
| `tables/dataset_statistics.csv` | Thời lượng, Fs và thống kê dữ liệu |
| `figures/` | Hình 4 test và hình so sánh |
| `diagnostics/`, `predictions/` | Ngưỡng, candidate/final và nhãn của 8 WAV |

MAE chính chỉ dùng START/END của FINAL speech regions. Candidate/debug không tham gia metric chính. HIGH/LOW thực sự tham gia detection. GT đỏ, prediction xanh, STE chuẩn hóa cam. Không smoothing STE theo thời gian; histogram TT2 được làm trơn theo bins.

## 5. Slide tiếng Anh và ZIP Python — previous version

| Thuật toán | PowerPoint | PDF | ZIP |
|---|---|---|---|
| TT1 | [PPTX](slides/THUAT_TOAN_1/THUAT_TOAN_1.pptx) | [PDF](slides/THUAT_TOAN_1/THUAT_TOAN_1.pdf) | [ZIP](submission/THUAT_TOAN_1.zip) |
| TT2 | [PPTX](slides/THUAT_TOAN_2/THUAT_TOAN_2.pptx) | [PDF](slides/THUAT_TOAN_2/THUAT_TOAN_2.pdf) | [ZIP](submission/THUAT_TOAN_2.zip) |
| TT3 | [PPTX](slides/THUAT_TOAN_3/THUAT_TOAN_3.pptx) | [PDF](slides/THUAT_TOAN_3/THUAT_TOAN_3.pdf) | [ZIP](submission/THUAT_TOAN_3.zip) |

Các PPTX/PDF và THUAT_TOAN_1/2/3.zip dưới đây là **previous version, trước Stage1**, chưa được tái tạo với giao thức mới. Gói CODE chính là `JUPYTER_NOTEBOOKS_CODE_ONLY.zip`. Mỗi bộ 7 slide; PPTX giữ biểu đồ/bảng chỉnh sửa được. Nội dung slide và speaker notes bằng tiếng Anh. Tài liệu giải thích code bằng tiếng Việt.

ZIP có `main.py` cố định thuật toán, module chung, hướng dẫn tương ứng, kết quả và slide; không có WAV. Giải nén rồi thêm 4 cặp WAV/LAB vào mỗi thư mục `data/train/`, `data/test/` để demo. Họ tên/MSSV bổ sung sau; trước nộp đổi tên thư mục thành `MaTheSV-HoTen`.

Các công cụ trong `tools/` dùng khi cần tái tạo slide/ZIP; xem [tools/README.md](tools/README.md). Chạy main không cần Node hoặc thư viện tạo PowerPoint.

Detector công khai `detect_regions` không nhận LAB/nhãn/filename và yêu cầu `endpoint_noise` fit sẵn; CLI chấm vẫn cần WAV/LAB. `--file` fit TRAIN trước, không hiệu chỉnh TEST. Metric chính schema 2: `mae_ms`/`rmse_ms`; calibration: `final_region_mae_ms`; diagnostic: `tolerance_boundary_*` và `matched_boundary_mae_ms`.
