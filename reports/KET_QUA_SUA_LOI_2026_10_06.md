# Kết quả sửa lỗi sau audit — 06/10/2026

**Nghiệm thu cuối: 07/10/2026.**

Triển khai theo [đánh giá sau hot fix](KIEM_TRA_SAU_HOTFIX_2026_10_06.md), với phạm vi người dùng chấp thuận: sửa code, tests, notebook và tài liệu code; **không sửa slide**. Baseline trước đợt sửa là commit `d035dcb4d12e5f01abe7b33b6bd812a806193f1e`, branch `codex/endpoint-train-calibration`. Baseline được tính lại và lưu trước khi sửa, không lấy ảnh tham khảo làm ground truth.

## 1. Các lỗi đã sửa

### TT3: mất chính xác khi giải giao hai Gaussian

- Hàm gây lỗi: `algorithms/tt3_gaussian.py::equal_density_threshold()`.
- Trước: khai triển hệ số quanh 0 làm các bình phương mean gần nhau triệt tiêu khi sigma gần floor `1e-9`. Có ca trả điểm không phải giao mật độ nhưng vẫn ghi `equal_density_between_means`.
- Sau: dịch tọa độ về mean nhỏ hơn, chia scale trước khi lập hệ số; tính discriminant đã phân tích nhân tử, giải bằng `q/a` và `c/q`, kiểm tra phương trình log-density của nghiệm float trước khi nhận. Sigma bằng nhau thật dùng midpoint; sigma gần bằng nhau vẫn được giải, không coi chúng tự động bằng nhau.
- Giữ population variance chia N, equal priors, sigma floor và chính sách chọn giao giữa hai mean. Khi không có giao hợp lệ, midpoint được ghi rõ `no_between_means_crossing_midpoint`.
- Reviewer phát hiện thêm ca nghiệm vừa ngoài hai mean phụ thuộc thứ tự lớp do phép trừ hai logarithm. Đã sửa bằng `log1p` khi sigma gần nhau, `log(sigma_sil/sigma_sp)` ở nhánh còn lại, và thêm regression cả hai thứ tự lớp.

| Ca số học | Trước | Sau |
|---|---|---|
| μSil=.5, σSil=1e-9, μSp=.50000001, σSp=2e-9 | T=.5000000052831314; log-density residual khoảng −10.48 | T=.5000000034705506, khớp oracle Decimal; residual khoảng 8.65e-8 do làm tròn float T |
| Hai mean=.5, sigma 1e-9 và 2e-9 | Ghi midpoint là giao mật độ | T=.5 nhưng ghi đúng **fallback**, không gọi là giao |
| Mean/std TRAIN hiện tại | T=.0028777336852328582 | T giữ nguyên chính xác |

Residual được kiểm tra theo sai số số học và một ulp của T, không dùng một epsilon STE tùy ý. `math.fsum` chỉ cộng các hạng hệ số; không gọi hàm fit Gaussian, mean/std hoặc solver thư viện. Model TT3 xuất lại từ TRAIN chỉ đổi diagnostics `crossings` (thứ tự và một ulp của root ngoài miền); ngưỡng chọn và các trường khác không đổi.

### TT3: hướng phân lớp không tương thích với tầng FINAL

- Hàm gây lỗi: `app/pipeline.py::detect_regions()`, cùng đường `fit_training_model()`.
- Trước: Gaussian thô có thể trả `speech_direction='low'`, nhưng FINAL xác nhận bằng năng lượng HIGH; vùng nói có thể biến mất âm thầm.
- Sau: model FINAL phải là hướng `high`; `low` hoặc hướng không hợp lệ bị từ chối bằng `ValueError` rõ ràng. Fit model FINAL cũng từ chối hướng `low` ngay khi phát hiện.
- Đây là **giới hạn API được kiểm tra rõ**, chưa triển khai hysteresis cho tiếng nói có năng lượng thấp hơn silence. Hàm Gaussian thô `fit()`/`predict()` vẫn giữ khả năng hướng `low`. Model cũ không có trường direction vẫn mặc định `high` để tương thích.

Hai lỗi TT3 này không xuất hiện trong tám WAV hiện tại. Vì vậy sửa chúng không làm MAE phone giảm; không dùng chúng để giải thích sai số END còn tồn tại.

## 2. Framing và giới hạn 200/100 ms

Cấu hình vẫn là **frame 25 ms, hop 10 ms**. Số mẫu dùng `round(fs*ms/1000)` theo nearest/ties-to-even; không đổi sang khung 20/30 ms:

| Fs | Frame samples | Frame thực | Hop samples | Hop thực |
|---|---:|---:|---:|---:|
| 16.000 Hz | 400 | 25 ms | 160 | 10 ms |
| 44.100 Hz | 1102 | 24.988662 ms | 441 | 10 ms |

STE vẫn tự cộng `sample²`, Energy=STE/frame_size; normalized STE chia max từng WAV, max=0 trả toàn 0. Dùng khung chữ nhật, bỏ đuôi không đủ khung, không thêm window hoặc smoothing mới. Tại Fs44,1k, 1102.5 mẫu không thể biểu diễn nguyên; số mẫu gần nhất dùng số chẵn 1102 và thời gian thực được công bố.

Các tests mới kiểm tra nhiều vùng, gap support 190/200/210/250 ms, phase lệch hop, Fs16k/44,1k, zero signal, span dưới/đúng/vượt 100 ms, merge trước filter và MAE không xác định khi thiếu/thừa vùng. Có thêm **48 WAV tổng hợp**: hai burst dài 300 ms, bốn gap vật lý, hai Fs, sáu phase; STE đối chiếu oracle độc lập đếm sample overlap.

| Gap vật lý giữa hai burst | Số ca | Kết quả detector giữ nguyên |
|---|---:|---|
| 190 ms | 12 | Gộp thành một vùng; đây là khoảng nghỉ ngắn theo policy |
| 200 ms | 12 | Vẫn gộp; **giới hạn ước lượng đã xác nhận** |
| 210 ms | 12 | Vẫn gộp; **giới hạn ước lượng đã xác nhận** |
| 250 ms | 12 | Giữ hai vùng riêng |

**Quy tắc 200 ms hiện đo gap giữa support khung hoạt động ước lượng.** Trong 12 ca có silence thực 200 ms, gap support chỉ khoảng 155–175.011 ms; phase 0 cho 165 ms ở Fs16k và 165.011 ms ở Fs44,1k. Khung chồng lấn lan sang hai phía khoảng lặng. So sánh `<200`/`≥200` trên support đúng, nhưng không bảo đảm giữ mọi silence vật lý ≥200 ms. Đợt sửa này giữ estimator và công bố giới hạn, chưa khắc phục hoàn toàn sai lệch giữa silence vật lý và silence ước lượng. Thay riêng support bằng ô hop quanh tâm cũng chưa bảo đảm điều đó, nên không thay hình học biên thiếu căn cứ.

**100 ms là heuristic lọc span support sau merge**, không phải tổng thời lượng voiced samples. Hai ví dụ burst thực khoảng 75 ms tạo span support 115 ms (16k) /114.989 ms (44,1k) và được giữ. Không tuyên bố bộ lọc loại mọi xung vật lý dưới 100 ms.

Bảng [waveform characterization](../outputs/tables/endpoint_robustness/waveform_characterization.csv) giữ GT hai vùng, thiếu vùng và primary MAE=`null` ở ca bị gộp; các ca silence thực 200/210 ms ghi `known_limitation`. `expected_short_gap_merge` ở silence thực 190 ms cũng không phải khớp GT hai vùng. [Burst ngắn](../outputs/tables/endpoint_robustness/short_burst_characterization.csv) ghi cả thời lượng mẫu thật và span ước lượng.

## 3. Ablation trên bốn TRAIN

[Công cụ tái hiện](../tools/check_endpoint_robustness.py) chỉ mở TRAIN WAV/LAB, fit model/noise/W một lần, khóa rồi so sánh bốn nhánh. Mọi nhánh giữ merge gap support 200 ms; TT2 raw dùng quyết định Energy nghiêm ngặt `E>T`, **không candidate padding**. Bỏ/giữ bộ lọc 100 ms là chẩn đoán, không chọn policy từ TEST.

| Thuật toán | Raw, min span0 | Raw, min span100 | HIGH/LOW, min span0 | HIGH/LOW, min span100 (production) |
|---|---:|---:|---:|---:|
| TT1 mean MAE TRAIN | 140.00 ms | 140.00 ms | 43.75 ms | 43.75 ms |
| TT2 mean MAE TRAIN | Không xác định: 1/4 file thừa vùng | Không xác định: 1/4 file thừa vùng | 13.75 ms | 13.75 ms |
| TT3 mean MAE TRAIN | 12.50 ms | 12.50 ms | 12.50 ms | 12.50 ms |

TT2 raw có mean chỉ ba dòng hợp lệ 12.503779 ms nhưng **không coi đó là mean đầy đủ** hoặc dùng để kết luận raw tốt hơn. Cả 12 cấu hình file/algorithm của nhánh production `hysteresis100` khớp `detect_regions()` chính xác. Bỏ filter 100 ms không thay kết quả trên bốn TRAIN này; dữ liệu chưa cho thấy nên bỏ filter, và các ca tổng hợp vẫn cho thấy ý nghĩa span của nó.

Một số bằng chứng END trên TRAIN:

- TT1 `phone_F1`: HIGH+seed cuối ở 2.745 s, FINAL END 2.865 s; còn đảo LOW cuối toàn WAV tới 3.235 s nhưng không được HIGH xác nhận thành vùng cuối. Raw END 3.235 s. Tầng HIGH/LOW đã loại một phần đuôi/noise nhưng LOW vẫn duy trì vùng đã xác nhận thêm 120 ms so với HIGH cuối.
- TT1 `phone_M1`: HIGH+seed cuối 3.545 s, LOW/FINAL END 3.665 s, raw END 4.055 s. Ngưỡng TT1 gốc≈.00102934 thấp hơn LOW≈.00121401 và HIGH≈.00251497.
- TT2 `phone_M1`: raw/HIGH cuối 3.525 s, LOW/FINAL END 3.545 s; LOW continuation kéo thêm 20 ms. Raw `phone_F1` tách hai vùng, còn hysteresis giữ một vùng đúng số lượng.
- TT3 raw/hysteresis đồng nhất ở TRAIN này. Không có bằng chứng trong bốn TRAIN rằng đổi hysteresis sẽ cải thiện TT3.

W FINAL vẫn 20 theo hòa của W1…50 trên TRAIN; W1 candidate frame-F1 là diagnostic riêng. Các HIGH/LOW và noise floor có thể làm nhiều W dẫn tới cùng FINAL, nhưng ablation này khóa W20, **không tự chứng minh nguyên nhân hòa của toàn bộ 50 W**. Không chọn ngưỡng/GAP mới, không hard-code tên file hay chỉnh prediction bằng GT. Xem [48 dòng TRAIN](../outputs/tables/endpoint_robustness/train_ablation.csv), [summary có valid/invalid counts](../outputs/tables/endpoint_robustness/train_ablation_summary.csv), [manifest/hashes/tham số](../outputs/tables/endpoint_robustness/manifest.json).

`status=ok` trong bảng ablation là trạng thái **ghép vùng của region metric**, không có nghĩa biên chắc chắn nằm trong speech theo LAB. Các cờ END nằm trong silence được giữ riêng ở bảng production bên dưới.

## 4. Hồi quy toàn bộ dataset

Fit lại mọi model từ TRAIN trước TEST; đối chiếu 24 trường hợp với baseline đã đóng băng. **FINAL regions, masks, metrics và số vùng không đổi; 0 regression.** Mọi file có một vùng GT và một vùng prediction. Bảng đầy đủ có before/after regions, count và status: [regression.csv](../outputs/tables/hotfix_oct06/regression.csv).

Mỗi ô ghi **MAE trước → sau, ms**:

| File | Split | TT1 | TT2 | TT3 |
|---|---|---:|---:|---:|
| phone_F1 | TRAIN | 57.50 → 57.50 | 2.50 → 2.50 | 2.50 → 2.50 |
| phone_M1 | TRAIN | 77.50 → 77.50 | 12.50 → 12.50 | 7.50 → 7.50 |
| studio_F1 | TRAIN | 17.49 → 17.49 | 17.49 → 17.49 | 17.49 → 17.49 |
| studio_M1 | TRAIN | 22.49 → 22.49 | 22.49 → 22.49 | 22.49 → 22.49 |
| phone_F2 | TEST | 57.50 → 57.50 | 32.50 → 32.50 | 27.50 → 27.50 |
| phone_M2 | TEST | 12.50 → 12.50 | 7.50 → 7.50 | 7.50 → 7.50 |
| studio_F2 | TEST | 7.51 → 7.51 | 7.51 → 7.51 | 7.51 → 7.51 |
| studio_M2 | TEST | 2.49 → 2.49 | 7.49 → 7.49 | 7.49 → 7.49 |
| **Mean bốn TEST** | | **20.00 → 20.00** | **13.75 → 13.75** | **12.50 → 12.50** |

Năm cờ `boundary inside silence` vẫn còn: TT1 phone_F1 END +115 ms, phone_M1 +145 ms, phone_F2 +105 ms; TT2 phone_F2 +55 ms; TT3 phone_F2 +45 ms. Không có file tốt hơn hoặc xấu đi trong đợt này. MAE chỉ chấm biên các FINAL regions, không chấm candidate/debug. Tám LAB đều chỉ có một vùng nên ca nhiều vùng được kiểm tra bổ sung bằng dữ liệu tổng hợp; chưa chứng minh ổn định trên các WAV thật nhiều phát ngôn.

## 5. Notebook, comment và đóng gói

- Generator thêm comment theo bước xử lý sau `ast.unparse` của TT2 và pipeline; tests xác nhận AST tính toán/docstring không đổi và anchor không bị mơ hồ. TT3 trong notebook dùng đúng solver mới và FINAL direction guard.
- Runner thêm `endpoint_execution.code_source_sha256`: hash code-cell source theo thứ tự, không tính cell audit tạm. Sửa source/comment sau execution hoặc thiếu hash làm audit từ chối, yêu cầu kernel mới. Đây là phát hiện sai lệch vô tình source/output, không phải chữ ký xác thực. Model digest khóa TRAIN trước TEST vẫn độc lập.
- Người dùng notebook vẫn **Restart Kernel and Run All → Save**, không cần module dự án hoặc builder/runner. Runner chỉ phục vụ audit/đóng gói của repository. Notebook không nạp số liệu lưu/model/CSV để dựng prediction.
- Đã rebuild rồi chạy **ba kernel mới**, tổng 24/24 WAV khớp model/biên/ngưỡng/metrics Python; mỗi notebook 5 PNG, tổng 15 PNG đã xem trực quan. TT2 vẫn tính 200 dòng TRAIN W trong kernel. Saved outputs/source digest được audit lại thành công.
- [ZIP CODE](../submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip) chứa đúng ba `.ipynb` đã thực thi, byte trùng file notebook ngoài ZIP; không WAV/LAB/audio. STT nhóm/tên thành viên còn placeholder theo dữ liệu chưa được cung cấp.

Lần kernel đầu thừa kế `MPLBACKEND=Agg` từ shell test headless nên không nhúng PNG và audit đã từ chối. Đã bỏ biến môi trường này và chạy lại toàn bộ ba kernel với backend inline; không sửa logic hoặc bỏ yêu cầu hình để vượt audit.

## 6. Nghiệm thu và phạm vi còn lại

- Baseline trước sửa: 98 tests OK. Nghiệm thu lại ngày 07/10: **133/133 tests OK trong 13.862 s** bằng Python 3.14 của `.venv`; chưa nghiệm thu runtime 3.10. [verification.json](../outputs/tables/hotfix_oct06/verification.json) lưu command/exit code, output suite, model/source/notebook/ZIP digests và hash files đã bảo toàn.
- Tests mới có RED trước sửa và GREEN sau sửa; numerical, direction, timing/TRAIN ablation và notebook protocol đều được kiểm tra.
- [Acceptance 24 cases](../outputs/tables/hotfix_oct06/acceptance.json) đối chiếu biên/metric và SHA256 **39 file** thuộc data, Source, slides và ZIP Python cũ: tất cả không đổi. Kiểm tra thêm tám file TT1/TT2, features/endpoints/config và slide builders/exporters cũng không đổi, tổng 47 file bảo toàn. Không thêm F0/Centroid vào detector.
- **Review độc lập cuối ngày 07/10 đã hoàn tất: không còn finding trọng yếu trong phạm vi sửa đã duyệt.** Reviewer kiểm tra lại các ca TT3, AST solver của notebook so với production, ba source digests, số PNG/error, ZIP bytes và bảng diagnostic; đọc bằng chứng nghiệm thu 133 tests/24 cases của tác vụ chính, không tự chạy lại suite/kernel. Kết luận review được lưu trong `verification.json`; không đồng nghĩa các giới hạn detector hoặc slide cũ đã được giải quyết.

Đợt này sửa hai lỗi TT3 tiềm ẩn và đồng bộ notebook/source, bổ sung bằng chứng về các giới hạn của FINAL. **Chưa khắc phục giới hạn silence vật lý 200 ms, các END phone muộn, hoặc xác minh độ bền với noise/holdout mới.** TRAIN metrics là fit trên TRAIN; TEST có lịch sử đã được xem và được dùng lại để đánh giá (`historical_test_exposure=true`), không tuyên bố generalization.

Slide/PPTX/PDF và ZIP Python trước Stage1 được giữ nguyên theo yêu cầu. Một số slide cũ vẫn mô tả TEST-tuned W khác code TRAIN hiện tại, nên chưa thể coi toàn bộ bộ nộp gồm slide là đã đồng bộ; người dùng đã loại việc sửa slide khỏi đợt này. Báo cáo/code hiện tại ghi rõ giao thức TRAIN, thời gian mẫu thực và giới hạn của estimator.
