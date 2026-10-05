# Giữa kỳ XLTHS — phân đoạn tiếng nói và khoảng lặng

Ba thuật toán Python: **TT1 Binary Search**, **TT2 Histogram chỉ Energy**, **TT3 Gaussian**. Đặc trưng, ngưỡng và metrics được tự tính; Matplotlib dùng để vẽ. Cấu hình chung: frame **25 ms**, hop **10 ms**, khoảng lặng nội bộ tối thiểu **200 ms**, vùng nói tối thiểu **100 ms**. Không tính F0.

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

[Kết quả hiện tại và giới hạn](reports/KET_QUA_HIEN_TAI.md) tổng hợp 4 test và toàn bộ 8 WAV.

TT2 hiện chỉ dùng **Energy** theo yêu cầu mới của thầy; pipeline không tính hoặc sử dụng Spectral Centroid. W = 20 dùng chung cho mọi file. Chương trình khảo sát W nguyên 1–50 trên 4 test có LAB, giữ 20 khi đồng hạng. LAB test tham gia chọn W nên kết quả TT2 ghi `test_tuned_not_independent`. TT1/TT3 và noise statistics học từ TRAIN.

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
  submission/       Ba ZIP nộp bài độc lập
  tests/            Bộ kiểm tra hiện có
  tools/            Công cụ tái tạo slide và ZIP
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

## 5. Slide tiếng Anh và gói nộp

| Thuật toán | PowerPoint | PDF | ZIP |
|---|---|---|---|
| TT1 | [PPTX](slides/THUAT_TOAN_1/THUAT_TOAN_1.pptx) | [PDF](slides/THUAT_TOAN_1/THUAT_TOAN_1.pdf) | [ZIP](submission/THUAT_TOAN_1.zip) |
| TT2 | [PPTX](slides/THUAT_TOAN_2/THUAT_TOAN_2.pptx) | [PDF](slides/THUAT_TOAN_2/THUAT_TOAN_2.pdf) | [ZIP](submission/THUAT_TOAN_2.zip) |
| TT3 | [PPTX](slides/THUAT_TOAN_3/THUAT_TOAN_3.pptx) | [PDF](slides/THUAT_TOAN_3/THUAT_TOAN_3.pdf) | [ZIP](submission/THUAT_TOAN_3.zip) |

Mỗi bộ 7 slide; PPTX giữ biểu đồ/bảng chỉnh sửa được. Nội dung slide và speaker notes bằng tiếng Anh. Tài liệu giải thích code bằng tiếng Việt.

ZIP có `main.py` cố định thuật toán, module chung, hướng dẫn tương ứng, kết quả và slide; không có WAV. Giải nén rồi thêm 4 cặp WAV/LAB vào mỗi thư mục `data/train/`, `data/test/` để demo. Họ tên/MSSV bổ sung sau; trước nộp đổi tên thư mục thành `MaTheSV-HoTen`.

Các công cụ trong `tools/` dùng khi cần tái tạo slide/ZIP; xem [tools/README.md](tools/README.md). Chạy main không cần Node hoặc thư viện tạo PowerPoint.
