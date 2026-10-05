# TT3 — Thống kê phân phối chuẩn Gaussian: giải thích thuật toán và code

Cập nhật theo code hiện tại ngày 05/10/2026. Tài liệu này mô tả TT3 đang chạy: thống kê trên normalized STE tuyến tính, tìm giao điểm hai mật độ Gaussian, sau đó lấy biên bằng xử lý chung.

## 1. Ý tưởng dễ hiểu

Silence thường có năng lượng nhỏ; speech thường lớn hơn và phân tán hơn. TT3 dùng dữ liệu train có nhãn để ước lượng hai cặp mean/std, xem mỗi lớp như một phân phối chuẩn.

Ngưỡng được chọn tại vị trí mà hai mật độ bằng nhau. Ngưỡng này tạo ứng viên speech; High/Low và thời lượng tạo final speech regions.

```text
4 WAV train + LAB → khung → normalized STE
    → tập silence và tập speech
    → mean/std hai lớp → p_sil(T) = p_speech(T) → T3

WAV cần phân đoạn → normalized STE → ứng viên theo T3
    → High/Low → nối gap <200 ms → loại vùng <100 ms
    → final START/END → MAE/RMSE so với LAB
```

**Gaussian là giả định mô hình, không phải kết luận đã chứng minh rằng dữ liệu năng lượng có phân phối chuẩn.**

## 2. Âm thanh và đặc trưng đầu vào

[read_wav()](../core/io_utils.py#L8) đọc PCM nguyên 8/16/24/32 bit bằng `wave`, chia toàn thang PCM và trung bình các kênh thành mono. Với PCM 16 bit, 16384/32768=0.5.

[compute_features()](../core/features.py#L90) dùng:

\[
L=\operatorname{round}(F_s\times0.025),\qquad
H=\operatorname{round}(F_s\times0.010)
\]

Ở 16 kHz: L=400, H=160 mẫu. Frame 25 ms, hop 10 ms. Khung phải đầy đủ; phần đuôi thiếu khung được bỏ. Hiện không nhân Hann/Hamming và không làm trơn STE theo thời gian.

\[
STE_i=\sum_{n=0}^{L-1}x_i[n]^2
\]

[normalize_max()](../core/features.py#L77):

\[
z_i=\frac{STE_i}{\max_j STE_j}
\]

Ví dụ STE `[1,4,2]` → z=`[0.25,1,0.5]`. Đỉnh bằng 0 → toàn z=0. Mỗi WAV chuẩn hóa bằng đỉnh riêng rồi mới gom để học.

**TT3 dùng z tuyến tính trong [0,1], không dùng `log(z)`, dB, MA hay Spectral Centroid.** Chuyển sang log-STE để học sẽ là thay đổi thuật toán và phải học lại thống kê/ngưỡng.

Thời gian support khung:

\[
start_i=\frac{iH}{F_s},\quad end_i=\frac{iH+L}{F_s}
\]

Ở 44.1 kHz, L=1102 mẫu theo `round()` nên thời gian thực hơi khác 25 ms; metadata và biên được tính từ số mẫu thật, không cộng thời gian bằng số thực lặp lại.

## 3. Gom hai lớp bằng LAB train

[frame_labels()](../core/metrics.py#L135) lấy nhãn tại tâm khung và dùng khoảng LAB nửa mở `[start,end)`:

- `sil` → 0.
- `v`, `uv` → 1.
- Ngoài vùng được gán nhãn → `None`, bỏ khỏi thống kê.

[fit()](../algorithms/tt3_gaussian.py#L79) gom z của bốn file F1/M1: `phone_F1`, `phone_M1`, `studio_F1`, `studio_M1`. Dataset hiện có 497 giá trị silence và 794 giá trị speech.

LAB test chỉ được dùng để chấm kết quả sau detection. TT3 không học T3 hoặc noise floor từ LAB test. Bộ chạy bài tập hiện yêu cầu WAV/LAB cùng tên để xuất đầy đủ bảng metric.

## 4. Mean, variance và standard deviation

[population_statistics()](../algorithms/tt3_gaussian.py#L7) tính bằng hai vòng lặp:

\[
\mu=\frac{1}{N}\sum_{i=1}^{N}z_i
\]

\[
\sigma^2=\frac{1}{N}\sum_{i=1}^{N}(z_i-\mu)^2,\qquad
\sigma=\sqrt{\sigma^2}
\]

Code dùng **population variance chia N**, không dùng sample variance chia N−1. `math.sqrt` chỉ thực hiện căn bậc hai; mean/variance không lấy từ thư viện thống kê.

Ví dụ tính tay với `[0,0.01,0.02]`:

1. Mean=(0+0.01+0.02)/3=0.01.
2. Tổng bình phương sai lệch=0.0001+0+0.0001=0.0002.
3. Variance=0.0002/3≈0.0000666667.
4. Std≈0.0081649658.

Ví dụ nhỏ này giải thích phép tính, không phải thống kê thật của dataset.

Thống kê hiện tại:

| Đại lượng | Silence | Speech |
|---|---:|---:|
| Số khung | 497 | 794 |
| Mean | 0.0003871107966 | 0.2026488981676 |
| Population std | 0.0007092854828 | 0.2356261327740 |

Độ phân tán speech lớn hơn nhiều do các khung âm mạnh/yếu khác nhau.

## 5. Tìm ngưỡng giao hai mật độ Gaussian

### 5.1. Mật độ mỗi lớp

\[
p(z\mid class)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left[-\frac{(z-\mu)^2}{2\sigma^2}\right]
\]

[equal_density_threshold()](../algorithms/tt3_gaussian.py#L27) tìm:

\[
p(T\mid silence)=p(T\mid speech)
\]

Code **không nhân thêm xác suất tiên nghiệm** 497/(497+794) hoặc 794/(497+794). Vì vậy không nên gọi đây là ngưỡng posterior có xét tần suất lớp.

### 5.2. Đổi thành phương trình bậc hai

Lấy log hai vế để bỏ hàm exp:

\[
\frac{(T-\mu_s)^2}{2v_s}
-\frac{(T-\mu_p)^2}{2v_p}
+\ln\frac{\sigma_s}{\sigma_p}=0
\]

Với s=silence, p=speech, `vs=σs²`, `vp=σp²`. Nhân cả phương trình với `2*vs*vp`:

\[
aT^2+bT+c=0
\]

\[
a=v_p-v_s
\]

\[
b=2(\mu_pv_s-\mu_sv_p)
\]

\[
c=\mu_s^2v_p-\mu_p^2v_s+2v_sv_p\ln(\sigma_s/\sigma_p)
\]

Code dùng chính ba hệ số này. Không dò T trên một grid và không sử dụng solver thống kê có sẵn.

### 5.3. Giải nghiệm ổn định

Trước khi tính, sigma nhỏ hơn `1e-9` được nâng lên `1e-9` để tránh chia/log với 0. Model lưu cả sigma gốc, sigma hiệu dụng và cờ floor.

- Nếu hai phương sai gần bằng nhau theo tolerance tương đối `1e-12`, và hai mean khác nhau: nghiệm là midpoint `(μs+μp)/2`.
- Nếu phương sai khác nhau: tính discriminant `D=b*b-4*a*c`.
- D âm: không có nghiệm thực, chuyển fallback.
- D không âm: dùng công thức q để giảm mất chữ số khi trừ hai số gần nhau:

```python
q = -0.5 * (b + (sqrt_D if b >= 0 else -sqrt_D))
roots = [q / a, c / q]
```

Nếu q=0, code dùng `-b/(2*a)`.

### 5.4. Chọn nghiệm nào?

Chỉ giữ nghiệm hữu hạn nằm giữa hai mean. Có nhiều nghiệm phù hợp thì chọn nghiệm gần midpoint nhất. Không có nghiệm trong miền thì dùng midpoint và lưu `threshold_rule` là fallback.

Với dataset hiện tại, hai nghiệm là:

\[
T_a=0.0028777336852,\qquad T_b=-0.0021071776746
\]

Ta nằm giữa mean silence và mean speech, Tb không nằm trong miền đó, nên:

\[
\boxed{T3=0.0028777336852}
\]

[Model TT3](../outputs/models/tt3.json) lưu ngưỡng và hai nghiệm này. Không có sigma nào bị floor trên dữ liệu hiện tại.

## 6. Tạo nhãn ứng viên

[predict()](../algorithms/tt3_gaussian.py#L109) xem hướng của hai mean:

- Mean speech ≥ mean silence → speech nếu `z >= T3`.
- Mean speech < mean silence → nhánh lõi dùng `z <= T3`.

Dataset hiện tại dùng hướng **high**. Ví dụ z=0.001 → silence ứng viên, z=0.01 → speech ứng viên.

Nhánh lõi hỗ trợ mean đảo hướng, nhưng **hysteresis chung hiện là logic năng lượng cao xác nhận speech**. Vì vậy không nên khẳng định toàn pipeline đã phù hợp mọi dataset đảo hướng chỉ vì `predict()` có nhánh `low`.

Một ngưỡng không tự tạo final region: các khung có thể dao động quanh T3 và tạo nhiều đoạn nhỏ. Bước xử lý chung dưới đây làm nhiệm vụ đó.

## 7. Noise floor, High/Low và vùng nói cuối

Đây là phần cải tiến dùng chung, tách khỏi việc tính ngưỡng Gaussian.

### 7.1. Thống kê nền nhiễu

[fit_noise_floor()](../core/endpoints.py#L4) dùng các khung sil của train:

\[
Q_{95}=0.0012140144555
\]

\[
U=\mu_n+3\sigma_n=0.0025149672449
\]

Mean/std noise chính là mean/std lớp silence đã nêu, vì cùng lấy từ tập silence train. Q95 là nearest-rank: sắp xếp tăng dần, lấy phần tử thứ `ceil(0.95*N)` tính từ 1.

### 7.2. Hai ngưỡng cuối

[endpoint_thresholds()](../core/endpoints.py#L28):

\[
LOW=\max(Q_{95},T3,10^{-12})
\]

\[
HIGH=\max(1.5\,LOW,U,T3)
\]

Vì T3>Q95 và 1.5*T3>U:

\[
LOW=0.0028777336852,\qquad HIGH=0.0043166005278
\]

HIGH xác nhận speech; LOW giữ phần speech yếu hơn. Cả hai ngưỡng thực sự tham gia logic detection, không chỉ vẽ lên hình.

### 7.3. Duyệt thời gian bằng hysteresis

[hysteresis_regions()](../core/endpoints.py#L42):

1. Đang silence: z đạt LOW thì nhớ đầu chuỗi LOW liên tục.
2. z đạt HIGH và nhãn ứng viên TT3=1 thì xác nhận speech.
3. START lùi về đầu chuỗi LOW đã nhớ.
4. Đang speech: z đạt LOW thì cập nhật support cuối.
5. Gap giữa END support cuối và START khung mới <200 ms được giữ trong cùng vùng.
6. Gap ≥200 ms kết thúc vùng; vùng kế tiếp phải được xác nhận lại.
7. Loại vùng dưới 100 ms sau khi nối gap.

Ví dụ với ngưỡng TT3 hiện tại: z=`[0.001,0.003,0.006,...]`. Khung 0.003 đạt LOW nhưng chưa đạt HIGH; khung 0.006 xác nhận speech, START có thể lùi về khung 0.003.

START/END lấy theo support mẫu thật: START ở đầu khung bắt đầu, END ở cuối khung hỗ trợ cuối. **Không cộng 200 ms thời gian chờ vào END.** Không cố định mỗi WAV có một vùng; các gap đủ dài giữ thành nhiều regions.

200 ms theo đề; 100 ms và hệ số 1.5 là lựa chọn thêm của project, áp dụng chung mọi filename.

## 8. Bản đồ code để tự học

| Hàm / file | Vai trò |
|---|---|
| [main_tt3.py](../main_tt3.py) | Điểm chạy riêng TT3 |
| [run_experiment()](../app/pipeline.py#L336) | Đọc train/test, fit, xuất model/ảnh/bảng |
| [prepare_records()](../app/pipeline.py#L110) | STE và nhãn tâm khung |
| `population_statistics` | Tự tính mean/std chia N |
| `equal_density_threshold` | Floor sigma, lập hệ số, giải và chọn nghiệm |
| `fit` | Gom sil/speech train và lưu provenance/hướng |
| `predict` | Áp dụng ngưỡng để tạo candidate mask |
| [predict_and_score()](../app/pipeline.py#L128) | Candidate → High/Low → final → metric |
| [regions_to_mask()](../core/endpoints.py#L78) | Đổi final regions thành nhãn khung để chấm frame metric |

Tổng, phương sai, mean, nghiệm và metric được triển khai trực tiếp. `math.sqrt` và `math.log` là phép toán cơ bản; code không gọi `numpy.mean`, `numpy.std`, Gaussian fitting hoặc VAD có sẵn.

## 9. MAE/RMSE được tính thế nào?

[ground_truth_regions()](../core/metrics.py#L5) gộp v/uv tiếp giáp, giữ sil làm phân cách. [region_endpoint_metrics()](../core/metrics.py#L18) chỉ chấm biên của final regions:

\[
e_s=1000(pred_s-gt_s),\quad e_e=1000(pred_e-gt_e)
\]

\[
MAE=\frac{|e_s|+|e_e|}{2},\qquad
RMSE=\sqrt{\frac{e_s^2+e_e^2}{2}}
\]

Âm là sớm; dương là muộn. Nhiều vùng cùng số lượng được ghép theo thứ tự và chấm toàn bộ 2N endpoint. Thiếu/thừa vùng làm MAE chính `None`; matched-only MAE lưu riêng. Lỗi lớn vẫn được tính nếu ghép đầy đủ, không cắt theo dung sai 100 ms.

Phone_F2 có GT `[1.02,4.04]`, final TT3 `[1.01,4.085]`:

- START −10 ms.
- END +45 ms.
- MAE=(10+45)/2=27.50 ms.
- RMSE≈32.60 ms.

END vẫn muộn và status có `boundary inside silence`. Gaussian + hysteresis hiện chưa xử lý hoàn hảo phần đuôi trên file này.

| Test WAV | MAE TT3 hiện tại |
|---|---:|
| phone_F2 | 27.50 ms |
| phone_M2 | 7.50 ms |
| studio_F2 | 7.51 ms |
| studio_M2 | 7.49 ms |
| Trung bình | **12.50 ms** |

Trong CSV, **`mae_ms` / `boundary_MAE_ms` là metric chính**. Cột chữ thường `boundary_mae_ms` là metric sự kiện phụ chỉ ghép biên trong dung sai 100 ms; không dùng nó để che lỗi endpoint lớn.

## 10. Chạy và xem output

Tại thư mục gốc của dự án:

```powershell
python main_tt3.py
python main_tt3.py --file phone_F2.wav
python main_tt3.py --no-show
python main_tt3.py --evaluate-all
```

Mặc định chạy bốn test; `--file` chọn một WAV có LAB cùng tên; `--no-show` chỉ xuất file; `--evaluate-all` chấm cả tám WAV và không mở cửa sổ.

- [Model TT3](../outputs/models/tt3.json): mean/std, nghiệm, sigma floor, T3, noise.
- [Bảng metric TT3](../outputs/tables/tt3/test_metrics.csv).
- [Ảnh phân phối train](../outputs/figures/training/gaussian_distributions.png): minh họa mật độ từ tham số đã học, không chứng minh phân phối thực đúng Gaussian.
- [Diagnostic phone_F2](../outputs/diagnostics/tt3/phone_F2.json).
- [Ảnh kết quả phone_F2](../outputs/figures/tt3/phone_F2.png).

## 11. Cách giải thích ngắn và giới hạn

“Em gom normalized STE của silence và speech trên train, tự tính mean và population standard deviation. Em giả định hai lớp có mật độ Gaussian, giải phương trình hai mật độ bằng nhau và chọn nghiệm giữa hai mean. Ngưỡng tạo ứng viên; High/Low cùng điều kiện thời lượng tạo final speech regions. MAE chỉ chấm START/END của các vùng cuối.”

Giới hạn cần nêu: dữ liệu STE không được kiểm định Gaussian; normalized STE bị giới hạn [0,1] còn Gaussian trải trên toàn trục thực. Code dùng equal density không xét class priors. Một ngưỡng cố định học từ train có thể nhạy khi noise hoặc cách thu thay đổi. Hysteresis là phần bổ sung chung nên kết quả cuối không thể quy hoàn toàn cho một ngưỡng T3.
