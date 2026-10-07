# Kiểm tra bản chất thuật toán và yêu cầu sau hot fix — 06/10/2026

## 1. Kết luận

**Đường chạy hiện tại đáp ứng thông số khung/hop, tách TRAIN–TEST và cách chấm FINAL endpoints trên dataset được cấp. Tuy nhiên, chưa thể kết luận dự án không còn sai sót hoặc đã sẵn sàng nộp.**

Phát hiện mới có thể tái hiện:

1. **TT3 công bố hỗ trợ `speech_direction='low'`, nhưng tầng FINAL dùng điều kiện năng lượng cao, khiến vùng nói của hướng này biến mất.** Đây là lỗi logic giữa hai tầng, có thể xảy ra với model do chính hàm fit tạo ra.
2. **Bộ giải ngưỡng Gaussian có thể mất chính xác khi hai độ lệch chuẩn rất nhỏ ở nền mean khác 0**, rồi ghi nhận một điểm không phải giao mật độ là nghiệm giao. Công thức đại số đúng; cách tính hệ số/discriminant chưa ổn định đủ.
3. **Điều kiện 200 ms áp dụng cho khoảng lặng ước lượng từ support khung.** Khoảng lặng thật đúng 200 ms có thể bị cửa sổ chồng lấn đo thành khoảng 165 ms rồi nối lại. Đây là giới hạn ước lượng biên cần kiểm chứng thêm, không phải nhầm đơn vị hay phép so sánh `<`/`>=`.
4. **Slide và hướng dẫn slide còn nội dung giao thức cũ**, đặc biệt câu chọn W bằng bốn TEST LAB. Code sau hot fix đã chuyển sang TRAIN. Thông tin thành viên vẫn chờ người dùng cung cấp.

Hai lỗi TT3 trên **không xuất hiện ở model và tám WAV hiện tại**, do model có `speech_direction='high'` và sigma không gần floor. Chúng **không giải thích MAE đang còn cao ở các file phone**. Những biên END muộn hiện tại vẫn tồn tại sau hot fix; hot fix vừa qua sửa giao thức và tính minh bạch, chưa cải thiện độ chính xác.

### Phạm vi và bằng chứng

- Checkout kiểm tra: `codex/endpoint-train-calibration`, HEAD `d035dcb4d12e5f01abe7b33b6bd812a806193f1e`.
- Đọc bản so sánh người dùng cung cấp; đối chiếu lại với đề gốc, code Python, model/CSV, notebook, ZIP và nội dung slide hiện hành. Nhận xét trong bản so sánh là thông tin để kiểm chứng, không tự trở thành yêu cầu của thầy.
- Ba subagent độc lập kiểm tra: công thức thuật toán; framing/timing/gap; yêu cầu/giao thức/metrics/notebook. Agent chính tái hiện độc lập các lỗi TT3 và ca khoảng lặng 200 ms.
- Chạy mới **98/98 tests đạt**, 10,645 giây, bằng Python của dự án.
- Fit lại model từ bốn TRAIN trước khi đọc TEST; tính lại **24 trường hợp = 8 WAV × 3 thuật toán** trong bộ nhớ. Ba model khớp toàn bộ JSON đã lưu; biên, MAE/RMSE, số vùng và status khớp các bảng hiện hành.
- Kiểm tra SHA256 xác nhận **188 file Git đang theo dõi không đổi** trong lượt audit. Không sửa thuật toán, dữ liệu, model, notebook, slide hoặc kết quả chính; chỉ thêm báo cáo này. Không commit/push trong lượt kiểm tra.
- Lượt này kiểm tra source và saved outputs notebook; **không chạy lại kernel notebook** hoặc mở bốn cửa sổ GUI để kiểm chứng bố trí màn hình.

## 2. Đối chiếu yêu cầu chính

| Yêu cầu | Kết luận | Ghi chú |
|---|---|---|
| Ba thuật toán cho nhóm ba thành viên | Đạt | TT1 Binary Search, TT2 Energy Histogram, TT3 Gaussian; có ba main riêng và ba notebook riêng. |
| Frame danh nghĩa 25 ms, hop danh nghĩa 10 ms | Đạt | Dùng số mẫu nguyên gần nhất; timing phản ánh số mẫu thực. Chi tiết mục 3. |
| Nối khoảng lặng nội bộ dưới 200 ms | Đúng trên khoảng lặng ước lượng | Giữ gap support đúng 200 ms; chưa bảo đảm mọi khoảng lặng thật từ 200 ms đều được giữ. Mục 5. |
| Tự tính xử lý tín hiệu/thống kê/metric | Đạt ở phần tính toán | Không dùng NumPy/SciPy detector, histogram, mean/std hoặc FFT thư viện. Có vòng lặp tự viết; thư viện I/O/vẽ được người dùng cho phép. |
| Học ngưỡng/tham số bằng TRAIN | Đạt trong đường chạy hiện tại | Noise model, TT1/TT3 và chọn FINAL W đều dùng TRAIN trước khi đọc TEST. |
| Đánh giá TEST độc lập hoàn toàn | Chưa thể tuyên bố | TEST từng được xem và dùng trong phát triển trước hot fix; code công khai `historical_test_exposure=true`. Không thể xóa lịch sử đó bằng đổi tên protocol. |
| TT3 dùng mean/std normalized STE của các khung TRAIN có nhãn | Đạt | Dùng population variance chia N; không có yêu cầu bắt buộc N−1. Hai ca biên TT3 vẫn có lỗi tại mục 4. |
| MAE/RMSE từ biên FINAL, đơn vị ms | Đạt | Candidate/padding/debug không tham gia MAE chính. Không bỏ lỗi lớn bằng dung sai 100 ms. |
| Tự phát hiện nhiều vùng nói | Có logic hỗ trợ, dữ liệu chưa bao phủ | Tám LAB hiện tại đều chỉ có một speech region sau khi ghép v/uv; phải thêm ca nhiều vùng để chứng minh độ ổn định. |
| Một lần chạy xuất bốn TEST figures, GT đỏ/prediction xanh | Đạt về code và saved outputs | Không nghiệm thu GUI bốn góc màn hình trong lượt này. |
| Không cần F0 và Spectral Centroid | Đúng yêu cầu người dùng | Không xem việc bỏ chúng là thiếu sót. Thuật toán STE/ZCR thứ tư không bắt buộc cho nhóm ba người. |
| Chú thích code theo block 5–10 dòng | Chưa đồng đều | Python có docstrings/comments; một số notebook functions sinh qua AST mất block comments. |
| Slide/PDF và bộ nộp thống nhất với phiên bản code mới | Chưa hoàn tất | Có mô tả chọn W bằng TEST của bản cũ; còn placeholder họ tên/MSSV/STT nhóm. |

Đề gốc cập nhật yêu cầu mỗi thành viên báo cáo riêng và nhóm tổng hợp so sánh N thuật toán. **Không thấy trong tài liệu gốc truy cập được tại H: một câu bắt buộc gộp thành đúng một deck/PDF chung.** Bản so sánh có nhắc nguồn `F:/.../yeucau.md` nhưng nguồn đó không nằm trong bộ tài liệu đã truy cập. Vì vậy không kết luận thiếu một deck chung là vi phạm đã xác minh. Bảng so sánh nhóm hiện đã có trong CSV và báo cáo; tính cập nhật của slide vẫn là vấn đề thật.

## 3. Frame, hop, STE và quy ước thời gian

Vị trí: [config](H:/GIUAKI_XLTHS/app/config.py:8), [compute_features](H:/GIUAKI_XLTHS/core/features.py:90), [prepare_records](H:/GIUAKI_XLTHS/app/pipeline.py:113).

Với Fs là Hz, L là số mẫu mono:

```text
N = round(Fs × 25 / 1000)      # mẫu/khung
H = round(Fs × 10 / 1000)      # mẫu/bước dịch
frame_count = 1 + floor((L − N) / H), khi L ≥ N
frame k lấy mẫu [kH, kH + N)
start = kH / Fs
end = (kH + N) / Fs
center = (start + end) / 2
```

`round` dùng quy ước `.5` về số chẵn của Python. Chỉ giữ khung đủ mẫu, không zero-pad STE ở cuối WAV.

| Nhóm WAV thực tế | Fs | N | H | Frame thực tế | Hop thực tế |
|---|---:|---:|---:|---:|---:|
| phone | 16.000 Hz | 400 | 160 | 25,000000 ms | 10,000000 ms |
| studio | 44.100 Hz | 1.102 | 441 | 24,988662 ms | 10,000000 ms |

Ở 44,1 kHz, 25 ms tương ứng **1102,5 mẫu**, nên không thể dùng chính xác 25 ms bằng một số mẫu nguyên cố định. Sai lệch −0,011338 ms của quy ước hiện tại là lượng tử hóa hợp lý, được công bố trong metadata. **Nó không giải thích lỗi biên hàng chục/hàng trăm ms.** Không có hop drift trên hai Fs của dataset hiện tại.

| File | Split | Số mẫu | Số khung đủ | Đuôi chưa phân tích (ms) |
|---|---|---:|---:|---:|
| phone_F1 | TRAIN | 51.840 | 322 | 5,000000 |
| phone_M1 | TRAIN | 66.560 | 414 | 5,000000 |
| studio_F1 | TRAIN | 126.267 | 284 | 8,208617 |
| studio_M1 | TRAIN | 120.400 | 271 | 5,170068 |
| phone_F2 | TEST | 76.800 | 478 | 5,000000 |
| phone_M2 | TEST | 44.800 | 278 | 5,000000 |
| studio_F2 | TEST | 138.892 | 313 | 4,489796 |
| studio_M2 | TEST | 105.040 | 236 | 6,870748 |

Đuôi thực bị bỏ nhỏ hơn một hop, không phải mặc định mất cả 25 ms. Tất cả GT speech END của tám file nằm trước END khung đầy đủ cuối, nên các đuôi này không bỏ mất speech đã được gán nhãn trong dataset hiện tại.

### Phép tính đặc trưng

```text
STE[k] = tổng x[n]² trong khung k
Energy[k] = STE[k] / N
Normalized_STE[k] = STE[k] / max(STE của chính WAV)
```

Max bằng 0 trả chuỗi zero, không chia cho 0. Dùng cửa sổ chữ nhật, không Hann/Hamming. Đề không bắt buộc taper; người dùng trước đó cũng chỉ nêu có thể dùng nếu phù hợp. Vì vậy thiếu Hann/Hamming không phải lỗi yêu cầu.

TT2 source dùng **mean-square Energy**, trong khi tầng FINAL dùng normalized STE. `detect_regions()` đổi ngưỡng Energy bằng cách chia peak Energy; do N cố định trong một WAV, hai thang normalized tương đương. Không thấy thiếu/thừa hệ số N.

Raw `ma` hiện là **mean absolute magnitude**, khác `sum(abs(x))` trong công thức MA của tài liệu. Docstring đã nói rõ; `ma_norm` giống normalized summed MA do N cố định. Ba detector hiện dùng STE/Energy, nên khác biệt này không gây lỗi endpoint hiện tại. Khi trình bày raw MA, phải ghi đúng tên hoặc bổ sung trường tổng riêng.

### Nhãn tâm khung và biên support

- LAB gán nhãn bằng **center**, khoảng `[start,end)`; v và uv đều là speech.
- Biên prediction là hợp của **support đầy đủ** các khung active, không phải hop cells tại tâm.
- Ca nhãn lý tưởng speech `[0,300; 0,700)` giây ở Fs=1000 khi dựng lại từ các khung có tâm nằm trong speech cho support `[0,290; 0,705)` giây: START sớm 10 ms, END muộn 5 ms, dù nhãn tâm hoàn hảo.
- So với cell dài một hop, support END của active run có thể dài hơn `frame − hop ≈ 15 ms`. Đây là lựa chọn quy ước biên; đề không ấn định một quy ước cụ thể.

Nên kiểm chứng quy ước này bằng tín hiệu tổng hợp có biên đã biết trước khi đổi; không lấy quy ước của bài bạn để thay máy móc.

## 4. Hai lỗi TT3 mới xác minh

### 4.1. P2 — Hướng speech thấp mâu thuẫn với xác nhận HIGH

Vị trí:

- [tt3.fit/predict](H:/GIUAKI_XLTHS/algorithms/tt3_gaussian.py:104) sinh `speech_direction='low'` nếu mean speech thấp hơn mean silence, và dùng `z <= T` để tạo candidate.
- [detect_regions](H:/GIUAKI_XLTHS/app/pipeline.py:156) đưa candidate đó vào `seed_mask`.
- [endpoint_thresholds](H:/GIUAKI_XLTHS/core/endpoints.py:38) cho LOW ≥ T và HIGH ≥ 1,5 × LOW theo default.
- [hysteresis_regions](H:/GIUAKI_XLTHS/core/endpoints.py:69) đòi đồng thời `z >= HIGH` và seed speech.

Với T dương, seed yêu cầu `z <= T`, nhưng xác nhận yêu cầu `z > T`: không khung nào thỏa đồng thời.

**Repro bằng hàm thực, không giả lập lại thuật toán:**

```python
from app.pipeline import detect_regions, fit_training_model

values = [.9, 1.] + [.1]*10 + [.9, 1.]
features = {
    'ste_norm': values,
    'starts': [i*.01 for i in range(14)],
    'ends': [i*.01+.025 for i in range(14)],
}
params = {
    'threshold': .5, 'speech_direction': 'low',
    'endpoint_noise': {
        'noise_q95': .01, 'noise_upper': .02, 'high_multiplier': 1.5,
    },
}
result = detect_regions('tt3', features, .155, params)
# candidate_regions = [(0.02, 0.135)]  -> dài 115 ms
# LOW = .5, HIGH = .75
# final_regions = []
```

Hướng này còn reachable qua fit thật: TRAIN `ste_norm=[.9,1.,.1,.2]`, labels `[0,0,1,1]`, `split='train'` cho μSil=0,95, μSp=0,15, T=0,55, `speech_direction='low'`. Chạy FINAL vẫn rỗng. Vì vậy đây không chỉ là một model người dùng tự khai sai.

**Ảnh hưởng hiện tại:** model gốc và cả notebook đã lưu đều có `speech_direction='high'`, nên 24 kết quả hiện tại không bị lỗi này. Notebook dùng cùng logic FINAL nên vẫn mang lỗi tiềm ẩn.

**Sửa tối thiểu đề xuất:** thống nhất direction giữa candidate và hysteresis; hoặc từ chối model low rõ ràng tại API FINAL và công bố giới hạn chỉ speech-high. Không nên công bố hỗ trợ low rồi âm thầm trả zero speech. Chưa triển khai trong lượt audit.

### 4.2. P2 — Hệ số Gaussian mất chính xác gần sigma floor

Vị trí: [equal_density_threshold](H:/GIUAKI_XLTHS/algorithms/tt3_gaussian.py:27), đặc biệt dòng 44–55.

Phương trình đúng với equal priors:

```text
(T−μSil)²/σSil² − (T−μSp)²/σSp² + 2 ln(σSil/σSp) = 0
```

**Dấu log và hệ số đại số hiện tại đúng.** Lỗi ở cách khai triển hệ số quanh 0: khi μ gần 0,5 nhưng sigma gần 1e−9, các số lớn tương đối triệt tiêu trong c/discriminant. Công thức roots dùng q ổn định không khôi phục thông tin đã mất ở bước tính discriminant.

```python
from algorithms.tt3_gaussian import equal_density_threshold
equal_density_threshold(.5, 1e-9, .50000001, 2e-9)
```

| Đại lượng | Giá trị tái hiện |
|---|---:|
| T code trả | 0.5000000052831314 |
| Quy tắc code ghi | `equal_density_between_means` |
| `ln pSil(T) − ln pSp(T)` | −10.481485131520571 |
| T oracle độc lập, tọa độ đã center/scale, Decimal 60 digits | 0.5000000034705506 |
| Residual oracle sau làm tròn về float | 8.65e−8 |

Residual −10,48 không gần 0, nên T code trả không phải giao mật độ. Input nằm ngay tại sigma floor công bố, không bị từ chối bởi contract.

Ca thứ hai: `equal_density_threshold(.5,1e-9,.5,2e-9)` trả roots `[.5,.5]` và gọi midpoint là giao mật độ, trong khi density ratio tại đó là 2. True roots là `.5 ± 1.3595559868917453e-9`; không root nào nằm giữa hai mean trùng nhau. Midpoint fallback được phép theo policy công bố, nhưng ghi nhầm nó là nghiệm giao là lỗi.

**Ảnh hưởng hiện tại:** với model TRAIN đang lưu, T=0,0028777336852328582 có residual độc lập khoảng **1,78e−15**, đúng trong độ chính xác số thực. Lỗi này chưa làm sai kết quả dataset hiện tại.

**Sửa tối thiểu đề xuất:** center/scale biến trước khi tạo polynomial; kiểm tra residual phương trình log-density của root được nhận; giữ fallback minh bạch nếu không có giao hợp lệ. Thêm regression hai ca trên và xác nhận không đổi model/dataset hiện tại. Chưa triển khai trong lượt audit.

## 5. Khoảng lặng 200 ms và lọc speech 100 ms

### 5.1. Comparator hiện tại đúng, nhưng khoảng lặng ước lượng có sai số

[Cleanup](H:/GIUAKI_XLTHS/core/postprocess.py:114) và [FINAL hysteresis](H:/GIUAKI_XLTHS/core/endpoints.py:63) đo:

```text
gap = start của speech tiếp theo − end support active trước
nối khi gap < 0,200 giây; giữ riêng khi gap ≥ 0,200 giây
```

Probe trực tiếp giữ nguyên gap support đúng 200/210 ms; 190 ms nối. Sai số dung sai so sánh chỉ 1e−12 giây, không thể giải thích hàng ms.

Với q khung zero giữa hai active runs trên lưới 25/10:

```text
gap support = (q+1)×10 − 25 = q×10 − 15 ms
```

Do đó 20 khung zero không đồng nghĩa support silence 200 ms: chỉ 185 ms. 21 khung cho 195 ms, 22 khung cho 205 ms. Không được dùng việc 20 khung vẫn merge để kết luận code dùng sai ngưỡng 200 ms.

Tuy nhiên, waveform tổng hợp gồm speech 300 ms → zero silence 200 ms → speech 300 ms, LOW=.1/HIGH=.5, tái hiện:

| Fs | Khoảng zero thật | Khoảng support ước lượng | FINAL |
|---|---:|---:|---|
| 16.000 Hz | 200 ms | 165,000000 ms | Một region, bị nối |
| 44.100 Hz | 200 ms | 165,011338 ms | Một region, bị nối |

Cửa sổ có một phần speech vẫn có STE trên LOW: speech trái kéo đến khoảng .315 s, speech phải bắt đầu khoảng .480 s. Đây là **window spill + quy ước support**, không phải nhầm Fs, hop hay phép so sánh.

**Kết luận giới hạn:** code thực hiện đúng minimum-silence trên các vùng đã ước lượng; hiện chưa chứng minh bảo toàn mọi khoảng lặng thật ≥200 ms. Cần thêm probes 190/200/210/250 ms, nhiều phát ngôn, nhiều pha đặt biên so với hop, trước khi chọn quy ước biên phù hợp hơn. Không sửa bằng cộng/trừ thời gian tùy file hoặc nhìn GT để kéo biên.

### 5.2. Speech tối thiểu 100 ms là heuristic, không phải số bắt buộc của thầy

Người dùng trước đó có yêu cầu lọc region quá ngắn; default100 ms là lựa chọn hiện hành. Đề chỉ bắt buộc silence200 ms. Vì vậy việc có minimum speech không tự là vi phạm, nhưng không nên ghi 100 ms là yêu cầu gốc.

- 8 khung HIGH liên tiếp có span95 ms bị xóa; 9 khung có span105 ms được giữ.
- Filter chạy sau merge và đo cả span, bao gồm gap được nối. Hai burst25 ms cách nhau support gap175 ms có thể merge thành span225 ms rồi được giữ, dù tổng active support chỉ50 ms.
- Heuristic có thể xóa utterance ngắn hợp lệ; không bảo đảm100 ms năng lượng active.
- GT ngắn nhất hiện tại1190 ms, FINAL ngắn nhất khoảng1154,99 ms. Tám WAV không bao phủ speech ngắn nên chưa chứng minh lựa chọn này an toàn trên miền đó.

## 6. Các điểm không phát hiện lỗi bản chất

### TT1 Binary Search

Đối chiếu CS425, PDF trang38 công thức2.8 và trang39 bước2–10:

```text
A(T) = sum(max(f−T,0))/Nf − sum(max(T−g,0))/Ng
A(T)>0 -> tăng cận dưới; ngược lại -> giảm cận trên
dừng khi hai số đếm strict dưới/trên ngưỡng không đổi
```

Code dùng đúng hai mẫu số lớp riêng, đúng chiều binary search và đúng dừng theo count của nguồn. Không yêu cầu `A(T)=0` tuyệt đối khi dừng. Ca f/g overlap `[.2,.4,.6]` có thể dừngT=.35, residual .05; ép hội tụroot .4 sẽ là đổi policy, không phải sửa một lỗi đối chiếu nguồn. Dùng normalized STE thay MA là lựa chọn đặc trưng được đề cho phép.

### TT2 Energy Histogram

- Energy = mean-square, đúng nguồn Giannakopoulos.
- Hai maxima đầu theo trục Energy thấp → cao, không phải hai cột cao nhất.
- Ngưỡng `(W×M1+M2)/(W+1)` gắn W với peak đầu, đúng.
- 64 bins, radius2, edge/fallback là các lựa chọn công bố, paper không ấn định chi tiết đó.
- Framing gốc50/50 ms đổi thành25/10 ms theo đề cập nhật là đúng ưu tiên.
- Energy-only đúng yêu cầu bỏ Spectral Centroid của người dùng.
- Padding250 ms chỉ ở candidate/debug, không kéo final END/START hay chấm MAE chính.
- Sau hot fix, W khảo sát1–50 trên bốn TRAIN FINAL,200 trường hợp; **cả50 W đồng tối ưu**. W20 được chọn theo tie preference cố định, không chứng minh W20 duy nhất tối ưu.

Tầng HIGH/LOW chung có thể chi phối biên FINAL và làm nhiều W cho cùng kết quả. Vì vậy các kết quả đang so sánh **ngưỡng riêng + hậu xử lý chung**, không phải ba ngưỡng thuần độc lập. Đây là giới hạn diễn giải/thiết kế đã công bố; cần ablation nếu muốn chứng minh đóng góp riêng của W hoặc thuật toán, không tự đổi code theo bài bạn.

### TT3 Gaussian ngoài hai lỗi mục4

- Mean và population std tính đúng, variance chiaN phù hợp estimate Gaussian; N−1 không bắt buộc bởi đề.
- Equal-density equation và dấu logarithm đúng.
- Model một ngưỡng, equal priors, chọn giao giữa mean, midpoint khi không có giao phù hợp: phải mô tả đúng phạm vi. Không phải full Bayes hai đuôi với prior thực nghiệm.
- Model dataset hiện tại có root còn lại âm, ngoài miền normalized STE không âm. Không phát hiện lỗi chọnroot hiện tại.

### Metric, candidate và leakage sau hot fix

- MAE chính lấy outer START/END của **từng FINAL region**, rồi ghép theo thời gian; không lấy tất cả candidate hoặc điểm binary-search debug.
- GT[(1,2)] vs prediction[(3,4)] vẫn có MAE **2000 ms**, không bị loại do quá dung sai.
- Thiếu/thừa region làm MAE chính không xác định; `matched_boundary_mae_ms` và `tolerance_boundary_*` là diagnostic riêng, không thay thế kết quả chính.
- Không còn hai tên cột MAE chỉ khác hoa/thường ở schema hiện hành.
- Không thấy code hiện tại chọn W/noise theo TEST, hard-code theo filename hoặc sửa prediction theo GT. Nhận xét trong bản so sánh về TEST-tuned W mô tả phiên bản trước hot fix.

## 7. Dataset hiện tại — kết quả kiểm tra lại

MAE ms, tính mới trong bộ nhớ, khớp các giá trị lưu trước lượt audit:

| File | Split | GT regions | TT1 MAE | TT2 MAE | TT3 MAE | Pred regions mỗi thuật toán |
|---|---|---:|---:|---:|---:|---:|
| phone_F1 | TRAIN | 1 | 57.50 | 2.50 | 2.50 | 1 |
| phone_M1 | TRAIN | 1 | 77.50 | 12.50 | 7.50 | 1 |
| studio_F1 | TRAIN | 1 | 17.49 | 17.49 | 17.49 | 1 |
| studio_M1 | TRAIN | 1 | 22.49 | 22.49 | 22.49 | 1 |
| phone_F2 | TEST | 1 | 57.50 | 32.50 | 27.50 | 1 |
| phone_M2 | TEST | 1 | 12.50 | 7.50 | 7.50 | 1 |
| studio_F2 | TEST | 1 | 7.51 | 7.51 | 7.51 | 1 |
| studio_M2 | TEST | 1 | 2.49 | 7.49 | 7.49 | 1 |
| **Mean bốn TEST** | | | **20.00** | **13.75** | **12.50** | |

Các kết quả khớp không có nghĩa tất cả biên đúng. Năm trường hợp vẫn gắn status `boundary inside silence` vì END nằm trong LAB silence và cách biên chuẩn quá một frame:

| File | Thuật toán | END error có dấu |
|---|---|---:|
| phone_F1 | TT1 | +115 ms |
| phone_M1 | TT1 | +145 ms |
| phone_F2 | TT1 | +105 ms |
| phone_F2 | TT2 | +55 ms |
| phone_F2 | TT3 | +45 ms |

Đây là sai số phát hiện biên còn tồn tại, không phải MAE tính từ candidate hay sai do làm tròn khung44,1 kHz. LOW continuation, đuôi năng lượng/noise và support convention là các yếu tố cần tách bằng probes/ablation trước khi đề xuất cải thiện. Chưa đủ bằng chứng để quy tất cả lỗi phone cho một nguyên nhân duy nhất.

Không có file nào tốt lên/xấu đi trong lượt audit: không sửa detector, và cả24 kết quả tính mới khớp kết quả trước. TRAIN nằm trong bảng8 WAV để kiểm tra toàn dataset, không được gọi8 WAV là8 TEST độc lập.

## 8. Notebook, slide và bộ nộp

### Notebook/ZIP

- Source cells và Markdown của ba notebook khớp generator hiện tại.
- Saved execution counts liên tiếp; mỗi notebook5 PNG, không saved error; model digest đúng. Năm PNG gồm bốn TEST figures và một figure giải thích.
- ZIP CODE chứa đúng ba notebook, byte-identical với bản ngoài ZIP, không WAV/LAB/audio.
- Có ba main Python riêng cho sinh viên chạy thuật toán phụ trách.
- Runtime đã nghiệm thu trước đây là Python3.14; **chưa nghiệm thu runtime3.10**. Lượt này chạy98 tests bằng môi trường dự án, không mở kernel mới hoặc tự nâng mức xác minh runtime.

### Slide và tài liệu

- TT2 slide workflow còn câu **“Select one W using four test LABs”**; ghi chú kết quả còn câu TEST LABs tune W. Đây là mô tả nhị phân slide cũ, khác currentTRAIN pipeline.
- [slides/README.md](H:/GIUAKI_XLTHS/slides/README.md:11) cũng lặp câu TEST-tuned mà không đặt nhãn phiên bản cũ ngay tại chỗ. README cấp trên có cảnh báo nhưng người mở riêng slides vẫn có thể hiểu sai.
- Các cover còn `Name / Student ID: pending`; group ZIP còn `STTNhom`. Người dùng đã nói sẽ gửi danh tính sau, nên đây là hạng mục chờ dữ liệu, không cần hỏi lại trong audit.
- Comment theo block5–10 dòng chưa đồng đều trong notebook do AST unparse bỏ comments. Cần sửa generator/source khi hoàn thiện bộ nộp, không chỉ thêm lời giải thích ngoài notebook.
- Yêu cầu nhận xét nhiễu môi trường/SNR của tài liệu trước có phần tính proxy và đối chiếu phone/studio. Proxy là observed speech-plus-noise/silence power, không phải SNR theo tín hiệu sạch đã biết. Notebook saved outputs hiện tại không tự chứng minh độ bền với nhiều mức noise tổng hợp.
- Các PPTX/PDF và ZIP Python cũ đã được cảnh báo là previous-version ở cấp trên; cần cập nhật nội dung giao thức và đóng gói đúng phiên bản trước khi nộp. Không nên chỉ sửa câuREADME rồi đểslidePDF cũ.

## 9. Thứ tự cải tiến tối thiểu đề xuất

**Cập nhật sau khi người dùng đồng ý sửa:** xem [báo cáo triển khai ngày06/10](KET_QUA_SUA_LOI_2026_10_06.md). Hai lỗi TT3 đã sửa, notebook đã chạy lại, bổ sung timing/ablation; slide được loại khỏi phạm vi theo yêu cầu người dùng. Các nhận xét bên dưới mô tả trạng thái tại lúc audit, trước đợt triển khai này.

**Chưa triển khai các thay đổi dưới đây; đây là kết quả đánh giá để quyết định bước sửa tiếp.**

1. Sửa hai lỗi TT3 tại mục4, đồng bộ source/notebook, thêm regression targeted. Chọn rõ phạm vi speech-high hay hỗ trợlow đầy đủ; không mở rộng model một cách ngầm định.
2. Bổ sung ca nhiều speech regions, silence190/200/210/250 ms, utterance dưới100 ms, biên lệch pha hop và Fs16k/44,1k. Chốt ý nghĩa200 ms cùng quy ước biên dựa trên bằng chứng; không rewrite framing chỉ để giống code bạn.
3. Làm ablation trên TRAIN cho threshold riêng, HIGH/LOW và minimum speech để biết vì sao FINAL ít nhạy vớiW và vì sao END phone kéo muộn. Khóa quy tắc rồi chạy toàn8 WAV; báo cáo mọi regression, không tune riêngphone_F2.
4. Cập nhật slide/PDF/README/ZIP và comments theo giao thứcTRAIN hiện hành. Giữ disclosureTEST từng được dùng; điền danh tính khi người dùng cung cấp.

Nghiệm thu sau sửa cần cả tests target lẫn98 tests hiện hành, bảng24 trường hợp before/after, mới kernel cho ba notebook, model/metric digest vàZIP byte checks. Test đạt vẫn chưa chứng minh tổng quát hóa trên holdout chưa từng xem.

## 10. Nguồn và bằng chứng hiện hành

- [Bản so sánh người dùng](C:/Users/Admin/Downloads/SO_SANH_BT1_VA_GIUAKI_XLTHS.md).
- [Đề cập nhật nhóm3–4](<H:/GIUAKI_XLTHS/Source/assignment/Hướng dẫn BT thi GK nhóm 3-4 SV_Phân đoạn tín hiệu thành tiếng nói và khoảng lặng_XLTHS_GK 2026.docx>): đoạn2 TRAIN/TEST, đoạn3 N thành viên, đoạn5 frame25/hop10, đoạn6 silence200, đoạn14 báo cáo riêng và tổng hợp nhóm, đoạn18–19 figures/metric, đoạn22 Gaussian.
- [Đề trước](<H:/GIUAKI_XLTHS/Source/assignment/Hướng dẫn BT - Phân đoạn tín hiệu thành tiếng nói và khoảng lặng_XLTHS_GK 2026.docx>): phần nhiễu/SNR; dùng đề cập nhật để giải quyết khác biệt số thuật toán.
- [Hướng dẫn trình bày/nộp](<H:/GIUAKI_XLTHS/Source/assignment/Hướng dẫn trình bày slide và nộp bài thi.pdf>): một trang; đã đọc và xem bản render, đặc biệt main/functions/comments/4figures/PDF và tênfolder.
- [CS425 Hodgkinson2012](<H:/GIUAKI_XLTHS/Source/references/CS425 Audio and Speech Processing_Hodgkinson_2012.pdf>): PDF trang34–39, đã xem trực quan công thức2.8 và bước2–10 tại trang38–39 (số trang in37–38).
- [Giannakopoulos2014](<H:/GIUAKI_XLTHS/Source/references/A method for silence removal and segmentation of speech signals_Giannakopoulos_2014.pdf>): trang1–2, đặc biệt mean-square Energy, hai maxima đầu, W và padding; trang2 đã xem trực quan.
- [Chapter6](<H:/GIUAKI_XLTHS/Source/course_slides/Chapter6_AUDIO-SPEECH SIGNAL PROCESSING.pdf>): phần đặc trưng/thống kê25; đối chiếu với yêu cầu Gaussian củaDOCX.
- [Kết quả hiện hành](H:/GIUAKI_XLTHS/reports/KET_QUA_HIEN_TAI.md), [modelTT3](H:/GIUAKI_XLTHS/outputs/models/tt3.json), [24 trường hợp](H:/GIUAKI_XLTHS/outputs/tables/all_all_dataset/test_metrics.csv), [TEST](H:/GIUAKI_XLTHS/outputs/tables/all/test_metrics.csv), [TRAIN W sweep](H:/GIUAKI_XLTHS/outputs/tables/tt2_w_selection/sweep.csv).

Các phép tính/probes ở mục4–5 được tái hiện bằng hàm production thật và oracle độc lập trong lượt audit; script tạm không được đưa vào bộ nộp. Báo cáo này lưu input/output đủ để chạy lại các ca mới mà không cần giữ rác trung gian.
