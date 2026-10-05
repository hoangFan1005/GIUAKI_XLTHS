# TT1 — Tìm ngưỡng bằng Binary Search: giải thích thuật toán và code

Cập nhật theo code hiện tại ngày 05/10/2026. Tài liệu này giải thích riêng TT1 để thành viên phụ trách có thể đọc, chạy và thuyết trình độc lập.

## Giao thức Stage1 và schema 2

Mọi model, W và `endpoint_noise` được fit từ bốn TRAIN trước khi đọc TEST. Ba thuật toán dùng `schema_version=2`, `parameter_selection_set=train`, `evaluation_protocol=train_selected_reused_test`, `historical_test_exposure=true`. TEST đã được xem trong phát triển trước đây và được dùng lại để chấm; không gọi đây là holdout mới hoặc độc lập. Chạy một file (`--file`) vẫn fit TRAIN trước, không hiệu chỉnh bằng TEST.

Hàm công khai [detect_regions()](../app/pipeline.py#L132) nhận `(algorithm, features, duration, params)`, không nhận LAB, nhãn hay filename; `endpoint_noise` phải được fit sẵn. [predict_and_score()](../app/pipeline.py#L174) gọi detector rồi chấm với LAB. CLI xuất metric vẫn yêu cầu WAV/LAB cùng tên.

Schema 2 dùng `mae_ms`/`rmse_ms` làm metric chính. `final_region_mae_ms` là MAE của mỗi dòng calibration W trên TRAIN. Các sự kiện ghép trong dung sai dùng `tolerance_boundary_*`; `matched_boundary_mae_ms` là diagnostic ghép vùng theo thứ tự. Alias cũ `boundary_MAE_ms` đã bỏ để tránh hai cột chỉ khác hoa/thường.

## 1. TT1 giải quyết việc gì?

TT1 học một ngưỡng năng lượng để phân biệt tiếng nói và khoảng lặng. Khung có năng lượng chuẩn hóa cao hơn ngưỡng được xem là ứng viên tiếng nói.

**Binary Search tìm một giá trị ngưỡng, không tìm trực tiếp vị trí START/END trong âm thanh.** Sau khi học ngưỡng, code áp dụng ngưỡng lên từng khung, rồi dùng xử lý chung để tạo các vùng nói cuối cùng.

Nguồn của phương pháp là tài liệu CS425 của Hodgkinson, trang PDF 38–39, phương trình (2.8) và các bước 1–10. Tài liệu nguồn minh họa trên MA; project áp dụng cách cân bằng diện tích nhầm lẫn lên normalized STE theo bài tập. Cần nêu sự thích nghi này khi trình bày.

Luồng chạy:

```text
4 WAV train + LAB
    → chia khung → normalized STE → gom hai lớp sil/speech
    → Binary Search → ngưỡng T1

WAV cần phân đoạn
    → chia khung → normalized STE
    → ứng viên z >= T1
    → High/Low + thời lượng
    → final speech regions → START/END → so với LAB để chấm điểm
```

## 2. Đọc WAV và tính năng lượng

[read_wav()](../core/io_utils.py#L8) đọc WAV PCM 8/16/24/32 bit bằng `wave`, chia mẫu cho toàn thang PCM và lấy trung bình các kênh để tạo tín hiệu mono. Ví dụ PCM 16 bit: mẫu 16384 trở thành 0.5 sau khi chia 32768. Đây chưa phải chuẩn hóa theo đỉnh của file.

[compute_features()](../core/features.py#L90) dùng:

\[
L=\operatorname{round}(F_s\times0.025),\qquad
H=\operatorname{round}(F_s\times0.010)
\]

Trong đó `L` là độ dài khung, `H` là bước dịch, `Fs` là tần số lấy mẫu. Với 16 kHz: L = 400 mẫu, H = 160 mẫu. Hai khung liên tiếp chồng nhau 15 ms.

Code chỉ dùng khung đầy đủ, bỏ phần đuôi thiếu một khung. Với 44.1 kHz, độ dài khung là 1102 mẫu do quy tắc `round()` của Python; thời gian thực hơi khác 25 ms và được lưu trong metadata. Hiện không nhân cửa sổ Hann/Hamming và không làm trơn STE theo thời gian.

Năng lượng khung thứ i:

\[
STE_i=\sum_{n=0}^{L-1}x_i[n]^2
\]

[normalize_max()](../core/features.py#L77) chia mỗi STE cho đỉnh của chính WAV:

\[
z_i=\frac{STE_i}{\max_j STE_j}
\]

Ví dụ các STE là `[2, 8, 4]` thì normalized STE là `[0.25, 1, 0.5]`. Nếu đỉnh bằng 0, code trả toàn bộ 0 để tránh chia cho 0. Mỗi WAV được chuẩn hóa riêng trước khi gom dữ liệu train.

Code còn tính `energy = STE/L` và `MA = mean(abs(x))`, nhưng **TT1 hiện dùng `ste_norm`, không dùng Energy hoặc MA**. MA ở project là mean absolute amplitude, không phải moving average.

## 3. LAB được sử dụng như thế nào?

[frame_labels()](../core/metrics.py#L135) lấy nhãn tại tâm của khung:

\[
center_i=\frac{start_i+end_i}{2}
\]

- `sil` → nhãn 0.
- `v` và `uv` → nhãn 1, đều thuộc tiếng nói.
- Tâm khung nằm ngoài vùng LAB → `None`, bỏ khỏi việc học và chấm nhãn khung.

Khoảng LAB dùng quy ước `[start, end)`: tâm bằng END thuộc đoạn sau nếu có.

TT1 học từ bốn file `phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`. Dataset hiện có 497 khung silence và 794 khung speech có nhãn hợp lệ.

**LAB test không đi vào việc học T1 hoặc quyết định biên.** Nó được đọc để đánh giá kết quả sau khi đã tạo final regions. Bộ chạy bài tập hiện yêu cầu WAV có LAB cùng tên để xuất đầy đủ metric.

## 4. Thuật toán tìm ngưỡng T1

### 4.1. Tìm miền chồng lấn

Gọi S là các giá trị silence và P là các giá trị speech:

\[
a=\max(\min S,\min P),\qquad
b=\min(\max S,\max P)
\]

Code chỉ giữ các giá trị trong `[a,b]`, tạo hai danh sách `f` và `g`. Những quan sát ngoài miền chồng lấn không tham gia phương trình diện tích của Binary Search.

Nếu `a >= b`, code dùng midpoint `(max(S)+min(P))/2` và lưu lý do `no_overlap_midpoint`. Nếu miền min/max giao nhau nhưng một lớp không có quan sát thực trong miền này, code dùng midpoint `(a+b)/2` và lưu fallback riêng.

### 4.2. Tính diện tích nhầm lẫn

[confusion_area()](../algorithms/tt1_hodgkinson.py#L9) tính:

\[
A(T)=\frac{\sum_{s\in f}\max(s-T,0)}{|f|}
-\frac{\sum_{p\in g}\max(T-p,0)}{|g|}
\]

Phần đầu đo các giá trị silence nằm sai phía trên ngưỡng. Phần sau đo các giá trị speech nằm sai phía dưới ngưỡng. Code cộng khoảng cách bằng vòng lặp rồi chia cho số quan sát mỗi lớp.

- `A(T) > 0`: phần silence sai phía lớn hơn, cần tăng T.
- `A(T) <= 0`: cần giảm T theo quy tắc cập nhật của tài liệu.

Đây là **cân bằng khoảng cách/diện tích**, không phải chọn ngưỡng có accuracy cao nhất hoặc số khung sai bằng nhau.

### 4.3. Chia đôi và điều kiện dừng

[binary_threshold()](../algorithms/tt1_hodgkinson.py#L47) khởi tạo `T=(a+b)/2`, sau đó:

```python
area = confusion_area(f, g, threshold)
if area > 0:
    lo = threshold
else:
    hi = threshold
threshold = (lo + hi) / 2
```

[_counts()](../algorithms/tt1_hodgkinson.py#L30) đếm:

\[
I(T)=\#\{s\in f:s<T\},\qquad J(T)=\#\{p\in g:p>T\}
\]

Dừng khi **cả hai số đếm không thay đổi sau một lần cập nhật**. Code dùng đúng dấu `<` và `>` cho hai số đếm; tối đa 200 vòng để chặn trường hợp bất thường.

Điều kiện này không bảo đảm nghiệm diện tích chính xác bằng 0. Ví dụ minh họa: S = `[0.001, 0.004]`, P = `[0.002, 0.006]`. Miền chồng lấn là `[0.002,0.004]`, nên `f=[0.004]`, `g=[0.002]`. T ban đầu bằng 0.003, A(T)=0. Sau cập nhật nhánh `else`, T=0.0025. Hai số đếm vẫn là `(0,0)` nên code dừng, dù A(0.0025)=0.001. Đây là hệ quả của điều kiện dừng hiện được giữ theo nguồn.

### 4.4. Giá trị thực tế đã lưu

| Đại lượng | Giá trị hiện tại |
|---|---:|
| Overlap low | 0.0000542396991 |
| Overlap high | 0.0113686891974 |
| Quan sát silence trong overlap | 253 |
| Quan sát speech trong overlap | 178 |
| Số vòng | 11 |
| T1 | **0.0010293375196** |
| A(T1) còn dư | khoảng 0.0000002448924 |

Các giá trị lấy từ [model TT1](../outputs/models/tt1.json), không phải hằng ngưỡng gán sẵn trong code. Chạy lại trên cùng dữ liệu sẽ học lại mô hình.

[predict()](../algorithms/tt1_hodgkinson.py#L131) tạo nhãn ứng viên `1 if z >= T1 else 0`. Nó không đọc LAB.

## 5. Từ ứng viên đến vùng nói cuối cùng

Phần này nằm ngoài lõi Binary Search. Nó được dùng chung để tránh nhiễu làm xuất hiện nhiều endpoint và hỗ trợ nhiều phát ngôn trong một WAV.

### 5.1. Học nền nhiễu từ train

[fit_noise_floor()](../core/endpoints.py#L4) lấy 497 khung `sil` của train, tự tính mean, population std và Q95:

\[
\mu_n=0.0003871107966,\quad \sigma_n=0.0007092854828
\]

\[
Q_{95}=0.0012140144555,\qquad
U=\mu_n+3\sigma_n=0.0025149672449
\]

Q95 dùng nearest-rank: sắp xếp tăng dần và lấy phần tử thứ `ceil(0.95*N)`, tính từ 1. Không gọi percentile/mean/std của thư viện thống kê.

### 5.2. Hai ngưỡng thực sự tham gia detection

[endpoint_thresholds()](../core/endpoints.py#L28) tính cho TT1:

\[
LOW=\max(Q_{95},T1,10^{-12})
\]

\[
HIGH=\max(1.5\,LOW,U,T1)
\]

Hiện LOW = 0.0012140144555, HIGH = 0.0025149672449. **T1 đang nhỏ hơn Q95 nên ngưỡng LOW/HIGH cuối bị noise floor chi phối.** Không nên trình bày rằng mọi biên cuối chỉ do T1 tạo ra.

### 5.3. Hysteresis và thời lượng

[hysteresis_regions()](../core/endpoints.py#L42) duyệt thời gian theo thứ tự:

1. Khi chưa có speech, STE đạt LOW thì nhớ đầu chuỗi ứng viên LOW liên tục.
2. Khi STE đạt HIGH và nhãn ứng viên TT1 bằng 1, xác nhận speech.
3. START lùi về đầu chuỗi LOW vừa nhớ để giữ phần mở đầu yếu hơn HIGH.
4. Khi đang speech, khung đạt LOW cập nhật điểm hỗ trợ cuối.
5. Nếu khoảng từ END của khung hỗ trợ cuối tới START của khung tiếp theo đạt **200 ms**, chốt vùng. Gap ngắn hơn 200 ms được giữ trong cùng vùng.
6. Sau khi nối gap, loại vùng có tổng thời lượng dưới **100 ms**.

Ví dụ LOW=0.002, HIGH=0.006: chuỗi `[0.0005, 0.003, 0.007, ...]` chỉ xác nhận tại 0.007, nhưng START có thể lùi về khung 0.003. Một tiếng động 0.007 chỉ dài 40 ms có thể xác nhận ứng viên nhưng bị loại bởi min speech 100 ms.

START là đầu khung bắt đầu vùng. END là cuối khung hỗ trợ cuối đạt LOW. **Không cộng 200 ms thời gian chờ vào END.** Code hỗ trợ nhiều vùng; không lấy một first/last chung để nuốt mọi khoảng lặng dài.

200 ms là yêu cầu của đề. 100 ms và hệ số HIGH 1.5 là lựa chọn bổ sung của project, dùng chung mọi file, không đặt riêng theo filename.

## 6. Đọc code theo thứ tự nào?

| Hàm / file | Vai trò |
|---|---|
| [main_tt1.py](../main_tt1.py) | Điểm chạy riêng, cố định TT1 |
| [run_cli()](../app/cli.py#L67) | Đọc cờ, chọn backend vẽ và báo lỗi |
| [run_experiment()](../app/pipeline.py#L417) | Đọc train/test, học mô hình, chạy và xuất kết quả |
| [prepare_records()](../app/pipeline.py#L114) | Tính đặc trưng và nhãn tâm khung |
| [fit()](../algorithms/tt1_hodgkinson.py#L105) | Gom STE hai lớp, gọi Binary Search |
| `confusion_area`, `_counts`, `binary_threshold` | Lõi học ngưỡng TT1 |
| `predict` | Tạo mask ứng viên từ ngưỡng train |
| [predict_and_score()](../app/pipeline.py#L174) | Gọi detector LAB-free rồi chấm FINAL với LAB |
| [regions_to_mask()](../core/endpoints.py#L78) | Đổi vùng cuối sang nhãn khung để chấm frame metric |

Các phép tổng bình phương, diện tích và thống kê được viết bằng vòng lặp. Các hàm cơ bản như `len`, `min`, `max`, `range`, `round` và toán học căn bậc hai phục vụ triển khai; không gọi VAD hoặc Binary Search có sẵn từ thư viện.

## 7. Cách chấm biên cuối và đọc kết quả

[ground_truth_regions()](../core/metrics.py#L5) gộp các đoạn v/uv tiếp giáp. Khoảng sil được LAB ghi rõ vẫn chia các vùng tham chiếu.

[region_endpoint_metrics()](../core/metrics.py#L18) chỉ dùng final regions. Với một vùng:

\[
e_s=1000(pred_s-gt_s),\quad e_e=1000(pred_e-gt_e)
\]

\[
MAE=\frac{|e_s|+|e_e|}{2},\qquad
RMSE=\sqrt{\frac{e_s^2+e_e^2}{2}}
\]

Âm nghĩa là sớm, dương nghĩa là muộn. Nhiều vùng có cùng số lượng được ghép theo thứ tự và tính tất cả 2N lỗi. Thiếu/thừa vùng làm MAE chính thành `None`; matched-only MAE được lưu riêng, không giả là kết quả tốt.

Ví dụ `phone_F2`: GT `[1.02,4.04]`, prediction `[1.01,4.145]`. Lỗi START = −10 ms, END = +105 ms, MAE = 57.50 ms. END vẫn muộn; không phải do cộng 200 ms chờ.

| Test WAV | MAE TT1 hiện tại |
|---|---:|
| phone_F2 | 57.50 ms |
| phone_M2 | 12.50 ms |
| studio_F2 | 7.51 ms |
| studio_M2 | 2.49 ms |
| Trung bình bốn file | **20.00 ms** |

Metric chính trong schema 2 là `mae_ms` / `rmse_ms`. Diagnostic sự kiện trong dung sai 100 ms dùng `tolerance_boundary_*`; `matched_boundary_mae_ms` chấm các vùng ghép theo thứ tự. Các diagnostic này không thay thế MAE chính khi thiếu/thừa vùng hoặc biên sai xa.

## 8. Chạy và xem file nào?

Mở terminal tại thư mục gốc của dự án:

```powershell
python main_tt1.py
python main_tt1.py --file phone_F2.wav
python main_tt1.py --no-show
python main_tt1.py --evaluate-all
```

Mặc định chạy bốn test. `--file` chọn một WAV có LAB cùng tên. `--no-show` chỉ xuất file; `--evaluate-all` chấm cả tám WAV train/test và tự tắt cửa sổ.

- [Model học được](../outputs/models/tt1.json): ngưỡng, overlap, số vòng, noise.
- [Bảng bốn test](../outputs/tables/tt1/test_metrics.csv): MAE, RMSE, số vùng và status.
- [Ảnh phone_F2](../outputs/figures/tt1/phone_F2.png): waveform, STE, High/Low, GT và final prediction.
- [Diagnostic phone_F2](../outputs/diagnostics/tt1/phone_F2.json): candidate và final được lưu riêng.

## 9. Cách giải thích ngắn khi thuyết trình

“Em học một ngưỡng normalized STE từ các khung train có nhãn. Binary Search thu hẹp miền chồng lấn theo dấu của hiệu diện tích nhầm lẫn, dừng khi hai số đếm không đổi. Khi chạy test, ngưỡng tạo ứng viên; High/Low xác nhận và duy trì tiếng nói, nối gap dưới 200 ms, loại vùng dưới 100 ms. Biên cuối lấy theo support khung và MAE chỉ tính trên các vùng đã xác nhận.”

Giới hạn cần hiểu: nguồn dùng MA còn project dùng STE; điều kiện dừng không giải diện tích chính xác; chuẩn hóa theo đỉnh mỗi WAV và noise train có thể hạn chế khả năng áp dụng sang môi trường nhiễu khác. Với bộ hiện tại, noise floor quyết định nhiều phần của biên cuối TT1.
