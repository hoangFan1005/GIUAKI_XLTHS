# TT2 — Histogram chỉ dùng Energy: giải thích thuật toán và code

Cập nhật theo code hiện tại ngày 05/10/2026. TT2 đã bỏ Spectral Centroid theo yêu cầu của thầy. Ba main hiện không tính FFT/Centroid trong đường chạy thực nghiệm.

## 1. Ý tưởng chính

Histogram mô tả các mức năng lượng xuất hiện thường xuyên trong một WAV. Code tìm hai đỉnh đầu của histogram, sau đó đặt ngưỡng gần đỉnh năng lượng thấp bằng trọng số W.

Khung có Energy vượt ngưỡng được xem là ứng viên speech. High/Low và thời lượng tạo các vùng nói cuối cùng.

```text
WAV → khung 25 ms / hop 10 ms → Energy
    → histogram 64 bin → làm trơn histogram
    → hai đỉnh đầu M1, M2 → ngưỡng TE
    → ứng viên Energy > TE
    → High/Low STE → nối gap < 200 ms → loại vùng < 100 ms
    → final regions → START/END → chấm MAE
```

Ngưỡng TE **được tính lại riêng cho từng WAV**. W là một giá trị chung cho tất cả WAV, hiện bằng 20.

Phương pháp histogram tham khảo bài Giannakopoulos “A method for silence removal and segmentation of speech signals”, trang 1–2. Bài gốc dùng Energy và Spectral Centroid với điều kiện AND. **Bản hiện tại là sự thích nghi chỉ dùng Energy theo yêu cầu thầy**, không phải bản sao đầy đủ hai đặc trưng của bài báo. Tên `variant="source"` còn trong code để tương thích, không có nghĩa Centroid vẫn được sử dụng.

## 2. Đọc âm thanh, chia khung và tính Energy

[read_wav()](../core/io_utils.py#L8) đọc PCM bằng `wave`, chia toàn thang PCM và lấy trung bình các kênh thành mono. Ví dụ PCM 16 bit có mẫu 16384 → 0.5 sau khi chia 32768.

[compute_features()](../core/features.py#L90) dùng:

\[
L=\operatorname{round}(F_s\times0.025),\qquad
H=\operatorname{round}(F_s\times0.010)
\]

Ở 16 kHz: L=400 mẫu, H=160 mẫu. Chỉ dùng khung đầy đủ, bỏ phần đuôi thiếu khung. Không nhân Hann/Hamming và không làm trơn STE theo thời gian.

Với khung i:

\[
STE_i=\sum_{n=0}^{L-1}x_i[n]^2,\qquad
E_i=\frac{STE_i}{L}
\]

STE là tổng bình phương, Energy là trung bình bình phương. Ví dụ một khung minh họa có `[0.1,-0.2,0.3,-0.4]` thì STE=0.30, E=0.075. Khung bốn mẫu chỉ để tính tay; code thật dùng độ dài 25 ms.

Đồng thời code tính:

\[
z_i=\frac{STE_i}{\max_j STE_j}
\]

Max bằng 0 thì trả toàn bộ z=0. Với cùng kích thước khung trong một WAV:

\[
z_i=\frac{E_i}{\max_j E_j}
\]

**Histogram chính dùng E chưa chia theo đỉnh; hysteresis dùng z trong [0,1].** Không được so trực tiếp ngưỡng Energy với z khi chưa chuyển đơn vị.

MA vẫn được tính như đặc trưng chung nhưng không tham gia TT2. Các hàm FFT/Centroid cũ còn như tiện ích tùy chọn trong `core/features.py`; [prepare_records()](../app/pipeline.py#L110) không gọi chúng cho ba main.

## 3. Histogram được tạo như thế nào?

### 3.1. Chia miền giá trị thành 64 bin

[histogram()](../algorithms/tt2_histogram.py#L20) lấy:

\[
E_{min}=\min E_i,\quad E_{max}=\max E_i,\quad
\Delta=\frac{E_{max}-E_{min}}{64}
\]

Khung có Energy E đi vào bin:

\[
j=\operatorname{int}\left(\frac{E-E_{min}}{\Delta}\right)
\]

Code chặn j trong `[0,63]`, nên giá trị đúng bằng Emax vẫn vào bin cuối. Tâm bin j:

\[
center_j=E_{min}+(j+0.5)\Delta
\]

Code tăng `counts[j]` bằng vòng lặp, không gọi hàm histogram của thư viện.

### 3.2. Làm trơn số đếm

Với `smooth_radius=2`, mỗi bin được thay bởi trung bình các bin từ j−2 đến j+2. Bin ở mép chỉ chia cho số bin thực có trong cửa sổ.

\[
\widetilde h_j=\frac{\sum_{k=\max(0,j-2)}^{\min(63,j+2)}h_k}
{\min(63,j+2)-\max(0,j-2)+1}
\]

**Đây là smoothing trên trục histogram; không phải moving average làm mượt STE theo thời gian.**

### 3.3. Tìm hai đỉnh

[local_maxima()](../algorithms/tt2_histogram.py#L54) quét bin từ trái sang phải:

- Đỉnh phải có số đếm dương và cao hơn hàng xóm ngoài vùng bằng nhau.
- Một plateau gồm các bin cao bằng nhau chỉ được xem là một đỉnh, chọn bin giữa.
- Hai đầu histogram cũng có thể là đỉnh.
- Danh sách đỉnh theo chiều giá trị Energy tăng, **không xếp theo chiều cao số đếm**.

M1 và M2 là **tâm của hai đỉnh đầu tiên**, không phải hai bin có số lượng lớn nhất. Histogram phẳng dương có một đỉnh; toàn 0 thì không có đỉnh.

## 4. Từ M1, M2 tới ngưỡng TE

[threshold_from_histogram()](../algorithms/tt2_histogram.py#L77) tính:

\[
T_E=\frac{WM_1+M_2}{W+1}
\]

Với W=20:

\[
T_E=\frac{20M_1+M_2}{21}
\]

M1 có trọng số lớn hơn nên TE gần M1. W tăng thì TE giảm về M1 nếu M2>M1, tạo thêm ứng viên speech. Điều đó không bảo đảm biên cuối hay MAE thay đổi.

Ví dụ tính tay: M1=0.001, M2=0.022:

- W=1 → TE=0.0115.
- W=20 → TE=(0.020+0.022)/21=0.002.

Với cùng Energy 0.005, khung bị loại ở W=1 nhưng được chọn ở W=20.

Fallback công khai trong code:

| Tình huống | Ngưỡng |
|---|---|
| Chuỗi rỗng | 0 |
| Mọi giá trị bằng nhau | Chính giá trị hằng |
| Có từ hai đỉnh | Công thức W ở trên |
| Thiếu hai đỉnh, chuỗi không hằng | `(min(E)+max(E))/2` |

Số bin 64, bán kính 2, W và fallback là các lựa chọn cài đặt được công bố; bài báo không cố định tất cả các tham số này.

### Ví dụ thực tế `phone_F2`

| Đại lượng | Giá trị |
|---|---:|
| Min Energy | 0.0000006162352 |
| Max Energy | 0.0091230984428 |
| M1 | 0.0000718856274 |
| M2 | 0.0007845795499 |
| W | 20 |
| TE | **0.0001058234333** |

[predict()](../algorithms/tt2_histogram.py#L176) tạo mask raw:

```python
raw = [1 if e > energy_threshold else 0 for e in energy]
```

Code dùng dấu **`>`**, không phải `>=`. Không còn điều kiện Centroid. Với phone_F2 ở W20, mask raw hiện có 224 khung speech.

## 5. W hiện được chọn như thế nào?

### 5.1. Đề xuất ban đầu từ train

[fit()](../algorithms/tt2_histogram.py#L214) thử W=`1,3,5,10,20` trên bốn file F1/M1 train. Mỗi W được chấm pooled frame F1 sau candidate padding và cleanup silence 200 ms:

\[
F1=\frac{2TP}{2TP+FP+FN}
\]

TP: speech dự đoán đúng; FP: silence bị nhận thành speech; FN: speech bị bỏ sót. Khung LAB=None không tham gia. Điểm bằng nhau thì lần duyệt đầu, tức W nhỏ hơn, được giữ.

| W | Train frame F1 sau khi bỏ Centroid |
|---|---:|
| 1 | **0.898698** |
| 3 | 0.896669 |
| 5 | 0.895657 |
| 10 | 0.894144 |
| 20 | 0.894144 |

Vì vậy **train đề xuất W1**, khác bản Energy+Centroid cũ đề xuất W20.

### 5.2. Chọn W chung theo final MAE trên bốn test

Theo yêu cầu trước đó của người dùng, pipeline tiếp tục thử mọi W nguyên từ **1 đến 50** trên `phone_F2`, `phone_M2`, `studio_F2`, `studio_M2`. Tổng cộng 200 trường hợp.

[sweep_final_weights()](../app/weight_selection.py#L67) chạy pipeline thật cho mỗi W. [select_final_weight()](../app/weight_selection.py#L11) ưu tiên:

1. Ít file có số vùng sai hoặc MAE không xác định nhất.
2. Nhỏ nhất mức thiệt hại tệ nhất so với W tốt nhất riêng của từng file.
3. Nhỏ nhất MAE trung bình các file hợp lệ.
4. Nếu vẫn hòa, giữ W đang ưu tiên từ cấu hình; nếu không có thì chọn W nhỏ nhất.

“Thiệt hại” ở đây là `MAE(file,W) - min_W MAE(file,W)`, không phải một công thức ngưỡng mới.

Tất cả W1–50 hiện cho cùng final regions và MAE trên từng test. Cấu hình `HISTOGRAM_W_TIE_PREFERENCE=20` giữ W20. Model phân biệt rõ:

```text
training_selected_W = 1   # đề xuất train-F1
W_tie_preference    = 20  # ưu tiên khi kết quả final hòa
W                  = 20  # giá trị thực tế chạy
```

Không có W riêng cho từng filename. **W20 chưa phải tối ưu duy nhất, và chưa khảo sát mọi số thực ngoài miền 1–50.**

W đã được chọn có sử dụng LAB test nên CSV ghi `test_tuned_not_independent`. Bốn test này không còn là đánh giá độc lập về khả năng tổng quát hóa của lựa chọn W. Chạy `--file` vẫn hiệu chỉnh một W chung trên đủ bốn test trước khi demo file được chọn.

## 6. Padding ứng viên và High/Low tạo biên cuối

### 6.1. Padding không phải endpoint cuối

[pad_speech()](../algorithms/tt2_histogram.py#L112) mở rộng mask ứng viên 25 hop ở mỗi phía, tức 250 ms. Nó đọc mask gốc để padding không tự lan tiếp.

Mask này còn phục vụ đề xuất W bằng train-F1 và diagnostic. Nhưng [predict_and_score()](../app/pipeline.py#L128) **tạo lại seed bằng raw `E > TE`**, không dùng mask padding để quyết định START/END cuối.

Phone_F2 hiện có candidate padded region `[0.8,4.265]`, còn final region `[1.01,4.095]`. Nếu lấy candidate để chấm endpoint sẽ sai bản chất.

### 6.2. Đổi TE sang thang normalized STE

\[
B=\frac{T_E}{\max_i E_i}
\]

Max E=0 thì code dùng B=0. Với phone_F2, B≈0.0115995058.

[fit_noise_floor()](../core/endpoints.py#L4) học noise chỉ từ khung sil của train:

\[
Q_{95}=0.0012140144555,\quad U=\mu_n+3\sigma_n=0.0025149672449
\]

Mean/std được cộng bằng vòng lặp, phương sai chia N. Q95 dùng nearest-rank trên danh sách đã sắp xếp.

[endpoint_thresholds()](../core/endpoints.py#L28) tính riêng kiểu histogram:

\[
LOW=\max(Q_{95},\min(B,U),10^{-12})
\]

\[
HIGH=\max(1.5\,LOW,U,B)
\]

Phone_F2: LOW≈0.0025149672, HIGH≈0.0115995058. B được chặn bởi U trước khi áp dụng sàn Q95 để giữ các phần speech yếu; HIGH vẫn phải đạt ngưỡng xác nhận.

### 6.3. Logic tạo final regions

[hysteresis_regions()](../core/endpoints.py#L42):

1. STE đạt LOW → nhớ đầu chuỗi LOW liên tục.
2. STE đạt HIGH **và raw Energy seed bằng 1** → xác nhận speech, START lùi về đầu chuỗi LOW đó.
3. Đang speech: khung đạt LOW cập nhật hỗ trợ cuối; không bắt buộc khung này vượt TE.
4. Gap từ cuối support gần nhất tới đầu khung mới <200 ms → giữ cùng vùng.
5. Gap ≥200 ms → chốt vùng trước khi đọc khung mới; vùng sau phải xác nhận lại.
6. Loại vùng <100 ms sau khi nối gap.

END là cuối khung hỗ trợ cuối đạt LOW. Không cộng 200 ms thời gian chờ, không thêm padding 250 ms. Nhiều gap dài tạo nhiều vùng nói, không ép mọi file có một vùng.

**W chủ yếu thay ngưỡng xác nhận; LOW giữ và kết thúc vùng.** Với dataset hiện tại, W thay đổi vẫn xác nhận cùng các vùng, rồi LOW support dẫn đến cùng endpoint. Điều này giải thích plateau W1–50.

## 7. Bản đồ code cần đọc

| Hàm / file | Vai trò |
|---|---|
| [main_tt2.py](../main_tt2.py) | Điểm chạy riêng TT2 |
| [run_experiment()](../app/pipeline.py#L336) | Đọc dữ liệu, fit train, khảo sát W, chạy và xuất |
| `histogram` | Tự đếm bin và làm trơn số đếm |
| `local_maxima` | Tìm đỉnh/plateau theo thứ tự trục giá trị |
| `threshold_from_histogram` | Công thức TE và fallback |
| `_validate_source_framing` | Kiểm tra metadata frame/hop thực thống nhất 25/10 |
| `predict` | Tính TE cho WAV, tạo raw và padded mask |
| `fit` | Đề xuất W bằng train frame F1 |
| `sweep_final_weights`, `select_final_weight` | Chọn một W chung bằng final-region metric |
| `predict_and_score` | Seed raw → hysteresis → final → metric |

Các phép đếm histogram, smoothing, tìm đỉnh, tính Energy và MAE tự viết. Không gọi `numpy.histogram`, FFT thư viện hoặc thuật toán VAD có sẵn.

Biến thể phụ `tt2-context` vẫn dùng histogram normalized STE trên [0,1], 100 bin, W5, không padding. **Đây không phải TT2 chính do `main_tt2.py` chạy.** Khi báo cáo bản hiện tại dùng thông số 64 bin, W20 và Energy.

## 8. Chấm MAE và đọc số liệu

[ground_truth_regions()](../core/metrics.py#L5) gộp các đoạn v/uv tiếp giáp, giữ sil làm phân cách. [region_endpoint_metrics()](../core/metrics.py#L18) chấm START/END của final regions:

\[
e_s=1000(pred_s-gt_s),\quad e_e=1000(pred_e-gt_e)
\]

\[
MAE=\frac{|e_s|+|e_e|}{2},\quad
RMSE=\sqrt{\frac{e_s^2+e_e^2}{2}}
\]

Nhiều vùng cùng số lượng: ghép theo thứ tự, chấm tất cả 2N lỗi. Thiếu/thừa vùng: MAE chính `None`, matched-only MAE lưu riêng. Candidate/padding không đi vào MAE chính.

Phone_F2 có GT `[1.02,4.04]`, final `[1.01,4.095]`: START −10 ms, END +55 ms, MAE=32.50 ms. END còn muộn và status vẫn có `boundary inside silence`; không nên xem kết quả này là hoàn hảo.

| Test WAV | MAE TT2 Energy-only |
|---|---:|
| phone_F2 | 32.50 ms |
| phone_M2 | 7.50 ms |
| studio_F2 | 7.51 ms |
| studio_M2 | 7.49 ms |
| Trung bình | **13.75 ms** |

Metric chính trong CSV là `mae_ms` / `boundary_MAE_ms`. `boundary_mae_ms` là metric sự kiện phụ trong dung sai 100 ms; nó có thể loại biên sai xa và không thay thế MAE chính.

## 9. Chạy và xem kết quả

Tại thư mục gốc của dự án:

```powershell
python main_tt2.py
python main_tt2.py --file phone_F2.wav
python main_tt2.py --no-show
python main_tt2.py --evaluate-all
```

Mặc định bốn test; `--file` chọn một WAV có LAB cùng tên; `--evaluate-all` chấm tám WAV và không mở cửa sổ. Do quy trình chọn W hiện tại, phải giữ đủ bốn test calibration kể cả khi demo một file.

- [Model TT2](../outputs/models/tt2.json): W train, W cuối, cấu hình histogram, noise.
- [Bảng metric](../outputs/tables/tt2/test_metrics.csv).
- [Khảo sát W](../outputs/tables/tt2_w_selection/sweep.csv): 200 lần chấm final regions.
- [Diagnostic phone_F2](../outputs/diagnostics/tt2/phone_F2.json): M1/M2, TE, High/Low, candidate/final.
- [Ảnh phone_F2](../outputs/figures/tt2/phone_F2.png).

## 10. Cách giải thích ngắn và giới hạn

“Em tính Energy từng khung 25 ms, tạo histogram 64 bin, làm trơn số đếm bằng tối đa 5 bin và lấy hai đỉnh đầu theo chiều năng lượng tăng. Ngưỡng là `(20*M1+M2)/21`. Bản này chỉ dùng Energy theo yêu cầu thầy. Sau khi tạo ứng viên, em dùng High/Low STE và điều kiện 200/100 ms để tạo final regions; chỉ các biên cuối mới được chấm MAE.”

Energy-only nhẹ và dễ giải thích hơn, nhưng không có đặc trưng phổ để loại một số loại nhiễu. Histogram một đỉnh phải dùng fallback; hai đỉnh đầu có thể không đại diện đúng hai lớp. W20 không duy nhất trong khảo sát hiện tại. High/Low là phần cải tiến chung, còn 100 ms là tham số bổ sung; MAE test có dùng dữ liệu test để chọn W và cần trình bày rõ.
