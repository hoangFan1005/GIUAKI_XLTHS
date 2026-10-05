# Kế hoạch cải tiến sau bản so sánh BT1 và GIUAKI_XLTHS

**Ngày:** 05/10/2026. **Trạng thái:** người dùng đã duyệt và đã hoàn tất giai đoạn 1 kèm giai đoạn 4 bắt buộc; giai đoạn 2, 3 và 5 hoãn ngoài phạm vi đợt này.

**Dự án của chúng ta:** phần được gọi là **“bạn kia”** trong `SO_SANH_BT1_VA_GIUAKI_XLTHS.md`. BT1 là dự án của bạn người dùng.

**Mốc đối chiếu:** commit `111ab37628072cf7ed9d6fbece3bbe2a8458ac1a`. Nguồn đánh giá: toàn bộ bản so sánh, DOCX đề nhóm 3–4 SV, code Python, code sinh notebook và các CSV/JSON hiện tại. Các đường dẫn BT1 ở ổ F: trong báo cáo không phải nguồn đã được truy cập trong lượt này.

**Mục tiêu:** sửa các vấn đề đã có bằng chứng về quy trình và báo cáo; tìm cách cải thiện endpoint bằng thí nghiệm có kiểm soát trên TRAIN, giữ đúng ba thuật toán và cập nhật notebook đồng bộ.

**Cách thực hiện:** thay đổi từng phần trong cấu trúc hiện có. Python chuẩn và phép tính tự cài; Matplotlib/IPython phục vụ vẽ, hiển thị. Người dùng đã duyệt thực thi và commit cục bộ giai đoạn 1 cùng bàn giao giai đoạn 4; không tự động push GitHub.

## 1. Những nhận xét đã được xác nhận

Các nhận xét và đường dẫn/dòng code dưới đây là quan sát baseline tại commit `111ab37628072cf7ed9d6fbece3bbe2a8458ac1a`, được giữ để đối chiếu lịch sử; không mô tả mọi trạng thái của code sau Stage1. Bản so sánh gốc cũng được giữ như bằng chứng baseline.

| Nhận xét | Kết luận và bằng chứng |
|---|---|
| Histogram dùng nhãn TEST để chọn W | Đúng: `app/pipeline.py:388`–`404` và `app/weight_selection.py:64`. Đây là vấn đề quy trình so với đề yêu cầu TRAIN xác định tham số. Việc này trước đây được thực hiện theo yêu cầu khảo sát W trên TEST của người dùng và đã được ghi rõ; hiện đề xuất thay đổi để tuân thủ đề. |
| Hai cột MAE chỉ khác hoa/thường | Đúng: `boundary_MAE_ms` là alias MAE chính, `boundary_mae_ms` là MAE phụ trong dung sai. Xuất hiện đồng thời trong CSV, gây khó đọc và xung đột với công cụ không phân biệt hoa/thường. |
| MAE chính bị tính sai | Chưa có bằng chứng. `core/metrics.py:18` tính START/END của vùng FINAL; lỗi lớn không bị loại khỏi MAE chính. Khi thiếu/thừa vùng, MAE chính không được tự cho bằng 0. |
| Gaussian bị sai dấu/công thức | Chưa thấy sai trong phương trình hoặc cách tính population mean/std hiện tại. Nghiệm ổn định và các fallback được công bố trong `algorithms/tt3_gaussian.py:27`. |
| HIGH/LOW chung ảnh hưởng biên cuối | Đúng: `app/pipeline.py:143`–`160`, `core/endpoints.py:28`. Kết quả hiện là kết quả của cả pipeline: thuật toán tạo ngưỡng/seed, rồi HIGH/LOW và hậu xử lý tạo FINAL. |
| W20 là nghiệm tối ưu duy nhất trên bốn TEST | Không đúng theo dữ liệu đã lưu: 200 dòng khảo sát W1…50 có cùng FINAL theo từng file; cả 50 giá trị có mean MAE 13,75 ms. W20 thắng quy tắc ưu tiên khi hòa. Không suy ra W luôn vô tác dụng trên dữ liệu khác. |
| Padding 250 ms làm END cuối trễ cố định | Không đúng trong pipeline FINAL: seed Histogram được lấy lại từ raw energy, không lấy padding candidate làm endpoint. |
| Cùng 25/10 ms phải có cùng timestamp | Không đúng. Cửa sổ phân tích, pha cửa sổ, gán nhãn và quy ước support/cell khác nhau có thể tạo kết quả khác nhau. |
| STE dạng tổng so với mean energy gây khác biệt chính | Với khung hiện tại dài bằng nhau, hai chuỗi cho cùng giá trị sau max-normalize. Không đổi SUM thành MEAN để kỳ vọng tự giảm MAE. |
| Minimum speech 100 ms là yêu cầu đề | Không phải yêu cầu bắt buộc trong DOCX đã đọc. Đây là heuristic bổ sung, khác yêu cầu silence tối thiểu 200 ms. |

Trong lượt đánh giá baseline, hai subagent đã kiểm tra riêng quy ước endpoint/Binary/Gaussian và Histogram/quy trình chọn W; lượt đó chỉ đọc nguồn và kết quả lưu, không chạy lại thuật toán, notebook hoặc unit test. Không coi số lượng test đạt hoặc kết quả chạy lại của BT1 trong bản so sánh là kết quả tự tái lập của dự án này. Bằng chứng chạy mới sau Stage1 được ghi ở mục 5 và `KET_QUA_HIEN_TAI.md` mục 8.

### Kết quả gốc để đối chiếu

| TEST | TT1 Binary | TT2 Histogram | TT3 Gaussian |
|---|---:|---:|---:|
| phone_F2 | 57,50 ms | 32,50 ms | 27,50 ms |
| phone_M2 | 12,50 ms | 7,50 ms | 7,50 ms |
| studio_F2 | 7,505668934 ms | 7,505668934 ms | 7,505668934 ms |
| studio_M2 | 2,494331066 ms | 7,494331066 ms | 7,494331066 ms |
| Mean theo file | 20,00 ms | 13,75 ms | 12,50 ms |

Nguồn: `outputs/tables/all/test_metrics.csv` và `summary.csv`. Khi so với BT1, dùng đúng bốn TEST và cùng định nghĩa mean theo file. Tám WAV gồm bốn TRAIN và bốn TEST; không gọi cả tám là tập kiểm thử độc lập.

Trên phone_F2, Binary có START=1,010 s, END=4,145 s so với LAB 1,020/4,040 s. MAE chính 57,5 ms là đúng: `(10 + 105)/2`. Riêng quy ước cuối cửa sổ 25 ms không giải thích được toàn bộ lỗi END 105 ms; cần kiểm tra thêm các khung vẫn được LOW giữ sau vùng LAB. Chưa tách được mức đóng góp của từng thành phần bằng thí nghiệm trong lượt đánh giá này.

## 2. Ràng buộc giữ nguyên

- Ba thuật toán: Binary Search, Histogram **chỉ Energy**, Gaussian; không thêm F0, Spectral Centroid hoặc tự thêm thuật toán thứ tư.
- Frame 25 ms, hop 10 ms; số mẫu thật và quy tắc làm tròn được ghi rõ.
- Chỉ nối khoảng lặng nội bộ **nhỏ hơn** 200 ms; khoảng đúng 200 ms phải giữ tách biệt.
- Nhiều vùng speech được giữ khi cần. Main figure và MAE chính chỉ dùng FINAL START/END.
- Không đặt quy tắc theo tên WAV, không chỉnh mốc theo LAB, không snap prediction về mốc GT.
- HIGH/LOW vẫn phải thực sự tham gia detection của cấu hình chính, theo yêu cầu trước của người dùng.
- Không đổi thuật toán đồng loạt hoặc sao chép cấu hình BT1 chỉ vì bảng TEST của BT1 thấp hơn.
- Khi sửa logic, Python và notebook phải cùng một cấu hình/thuật toán. Ba notebook tiếp tục độc lập và lưu output thật.

## 3. Thứ tự triển khai và phạm vi đã duyệt

### Giai đoạn 1 — Quy trình TRAIN/TEST và tên metric

Đã hoàn tất ngày 05/10/2026. Giữ cửa sổ, Gaussian, tiêu chí dừng Binary và HIGH/LOW hiện hành.

**1A. Chọn W bằng kết quả FINAL trên TRAIN**

- [x] Khóa trước khảo sát: W nguyên 1…50, histogram 64 bins/radius 2, detector hiện tại, quy tắc thời gian hiện tại và ưu tiên W20 khi thật sự hòa.
- [x] Học noise từ silence TRAIN và model nền từ TRAIN.
- [x] Khảo sát W trên bốn TRAIN bằng **cùng hàm tạo FINAL và metric chính** sẽ dùng khi suy luận. Dùng thứ tự mục tiêu hiện tại: ít file thiếu/thừa vùng nhất, nhỏ nhất worst per-file regret, rồi mean MAE; hòa mới xét W20 và W nhỏ nhất.
- [x] Ghi `W_selection_train`, `tie_preference_W`, file TRAIN dùng chọn W và mục tiêu chọn. Đề xuất W theo candidate-frame F1 cũ, nếu còn giữ, phải được đặt tên là diagnostic candidate, không gọi là lựa chọn theo FINAL.
- [x] Khóa model trước đánh giá TEST. Demo một WAV cũng không gọi lại quy trình chọn W trên TEST.
- [x] Tách phần phát hiện hiện có thành `detect_regions(algorithm, features, duration, params) -> dict` trong `app/pipeline.py`, trả `final_regions`, `mask` và `diagnostic`. Hàm này không nhận tên WAV, LAB hoặc nhãn. `predict_and_score` gọi hàm đó rồi mới chấm điểm theo LAB; giữ cùng thân logic detector, không viết lại thuật toán.
- [x] Model dùng cho FINAL phải có noise statistics đã học; đường tương thích model cũ, nếu giữ, phải được tách rõ khỏi luồng đánh giá chính.

Kết quả khảo sát mới: cả 50 W từ 1…50 hòa theo FINAL trên TRAIN; W20 được chọn bằng `tie_preference_W=20`. Candidate-frame riêng chọn W1, F1 ≈ 0.898698. Ưu tiên W20 được công bố cho lần khảo sát mới, không phải bằng chứng W20 đã được đăng ký trước lần xem TEST cũ.

**Mã liên quan:** `app/pipeline.py`, `app/weight_selection.py`, `app/config.py`, `algorithms/tt2_histogram.py` nếu đổi metadata của diagnostic candidate.

**1B. Một MAE chính, tên metric phụ rõ ràng**

- [x] Giữ `mae_ms`/`rmse_ms` làm key MAE/RMSE chính để giảm thay đổi các hàm hiện có.
- [x] Bỏ alias `boundary_MAE_ms` khỏi bảng chính. Với bảng khảo sát W, dùng tên riêng `final_region_mae_ms` thay vì tên chỉ khác chữ hoa/thường với metric phụ.
- [x] Đổi prefix của nhóm metric phụ trong dung sai từ `boundary_` sang `tolerance_boundary_`, gồm MAE/RMSE, matched/missed/false positive, precision/recall/F1 và tolerance.
- [x] Giữ `matched_boundary_mae_ms` của ghép **vùng** ở namespace riêng; mô tả rõ nó không phải MAE trong dung sai.
- [x] Bảng chính hiển thị MAE FINAL, lỗi START/END có dấu, counts, status và split. Không lấy metric phụ để thay MAE chính khi lỗi lớn.
- [x] Thêm schema/version hoặc ghi chú chuyển tên cho CSV mới; không sửa các file cũ chỉ bằng việc đổi header rồi coi đó là kết quả một lần chạy mới.

**Mã liên quan:** `app/pipeline.py`, `app/weight_selection.py`, `tools/build_notebooks.py`, `tools/run_notebooks.py` và các consumer được tìm bằng `rg`.

**Tiêu chí hoàn thành giai đoạn 1:** sửa nhãn TEST không được làm đổi W/noise/model; inference không cần nhãn; tên cột không trùng khi chuyển về chữ thường; đổi tên metric không làm đổi số liệu. Cập nhật protocol và kết quả bằng một lượt chạy mới sau khi code được duyệt.

**Lưu ý lịch sử TEST:** bộ TEST này đã được xem và đã được dùng chọn W. Sửa code dừng việc dùng TEST trong lần chọn tham số mới, nhưng không biến bốn file cũ thành dữ liệu chưa từng được xem. Báo cáo mới ghi `parameter_selection_set=train`, `evaluation_protocol=train_selected_reused_test` và `historical_test_exposure=true`. Manifest `W_selection_train` ghi riêng `selection_set=train`/`evaluation_protocol=train_final_calibration`. Cần dữ liệu mới nếu muốn đánh giá tổng quát hóa trên một holdout hoàn toàn mới.

### Giai đoạn 2 — Thí nghiệm tách từng nguyên nhân trên TRAIN (hoãn)

Chỉ triển khai khi giai đoạn 1 đã rõ. Không mặc định rằng làm giống BT1 sẽ tốt hơn.

**Quy trình lựa chọn:** dùng bốn vòng giữ lại một file TRAIN. Trong mỗi vòng, học T/noise từ ba file còn lại; đánh giá file TRAIN giữ lại. Không học noise bằng nhãn của file đang giữ lại. Sau lựa chọn, học lại trên cả bốn TRAIN rồi khóa cấu hình để báo cáo TEST.

Nếu cấu hình detector Histogram thay đổi, chọn lại W trên TRAIN bằng chính cấu hình mới. Không dùng W được tối ưu cho detector cũ để tuyên bố detector mới đã được tối ưu. Không khảo sát tích Descartes mọi tùy chọn; thử từng yếu tố với số phương án nhỏ đã khai báo trước.

| Thí nghiệm | So sánh và yêu cầu |
|---|---|
| T gốc so với tầng FINAL | Ghi T gốc, LOW/HIGH, seed/raw mask, FINAL và lỗi có dấu. Một nhánh threshold trực tiếp + nối gap 200 ms chỉ phục vụ ablation được gắn nhãn riêng; cấu hình chính vẫn có HIGH/LOW. Candidate/debug không được đưa vào MAE chính. |
| LOW bị noise cap chi phối | Với Histogram, đối chiếu công thức hiện tại với một phương án: `LOW=max(Q95,0.5*T_hist_normalized,1e-12)`; `HIGH=max(T_hist_normalized,1.5*LOW,U)`. Hệ số 0.5 là giả thuyết toàn cục để thử trên TRAIN, chưa phải giá trị được chứng minh tối ưu. Chấm cả speech yếu bị mất lẫn speech/noise đuôi bị giữ; không sửa LOW chỉ nhằm làm W có ảnh hưởng. |
| Support timestamp và decision-cell timestamp | Giữ nguyên đặc trưng và tâm LAB để thử riêng quy ước thời gian. Phương án thử: `decision_start=max(0,center-hop_actual/2)`, `decision_end=min(duration,center+hop_actual/2)`. Phải dùng cùng quy ước START/END, gap, minimum speech và mask; giữ `starts/ends` làm support phân tích 25 ms và thêm riêng `decision_starts/decision_ends`, không đổi metadata 25 ms thành 10 ms. Không chỉ trừ cố định 15 ms khỏi END. Nếu đổi pha cửa sổ hoặc tâm LAB, đó là thí nghiệm riêng và phải học lại toàn bộ thống kê. |
| Minimum speech | Thử 0 và 100 ms trên TRAIN, giữ nguyên silence 200 ms; công bố rõ đây là heuristic. Không loại âm ngắn mạnh chỉ vì muốn số vùng của TEST đẹp hơn. |
| Median nhẹ cho Binary | Sau các thí nghiệm trên, có thể thử bậc 1/3/5 trên normalized STE, tự tính bằng phép cơ bản; không mặc định bậc 15 của BT1. Dùng cửa sổ centered với các phần tử thật có ở mép, không normalize lần hai sau lọc. Học lại T và noise từ đúng chuỗi detection đã lọc; plot raw STE và ghi rõ smoothing dùng cho detection. Giữ nguyên quy tắc dừng theo counts của Binary. |
| Histogram bins/smoothing | Ưu tiên sau LOW/timing. Có thể thử 64/128 bins hoặc radius 1/2, từng yếu tố một trên TRAIN. Không đổi raw energy sang normalized STE chỉ vì khác tên; thay miền bins/fallback phải được định nghĩa và công bố riêng. |

**Mã có thể liên quan:** `core/features.py`, `core/endpoints.py`, `core/postprocess.py`, `app/pipeline.py`, `algorithms/tt1_hodgkinson.py`, `algorithms/tt2_histogram.py`. Gaussian giữ nguyên công thức đang đúng.

**Cổng lựa chọn:** ưu tiên đúng số vùng và không có biên ngoài WAV; sau đó MAE/RMSE và lỗi START/END. Tính cả số file MAE không xác định, không loại chúng khỏi quyết định rồi chỉ báo mean đẹp. Nếu không có cải thiện ổn định trên TRAIN, giữ cấu hình hiện hành và ghi kết luận “chưa có bằng chứng để đổi”. Không chọn lại cấu hình sau khi thấy bảng TEST thấp/cao.

### Giai đoạn 3 — Kiểm tra biên và hợp đồng thuật toán (hoãn)

- [ ] Silence toàn bộ, âm thanh ngắn, toàn speech, speech chạm đầu/cuối WAV và nhiều speech regions.
- [ ] Gap 190/200/210 ms: nối khoảng ngắn, giữ tách tại đúng 200 ms trở lên theo cùng quy ước thời gian.
- [ ] Speech mạnh ngắn hơn 100 ms, burst nhiễu và LOW dao động: chứng minh tác động của minimum speech/HIGH confirmation.
- [ ] Histogram thiếu đỉnh, một plateau, năng lượng hằng/0 và scale consistency; không tự đổi fallback midpoint vì chưa có bằng chứng nó gây sai hiện tại.
- [ ] Ghép nhiều vùng, thiếu/thừa vùng và lỗi lớn hơn 100 ms: MAE chính không bị kiểm duyệt bởi metric trong dung sai.
- [ ] Gaussian `speech_direction=low`: model fit/predict hiện có nhánh này nhưng FINAL chỉ hỗ trợ HIGH theo chiều năng lượng tăng. Đề xuất tối thiểu là báo lỗi rõ với model chưa được FINAL hỗ trợ. Không tự thêm detector đảo chiều trong đợt sửa này. Đây là lỗi hợp đồng tiềm ẩn, **không phải nguyên nhân MAE hiện tại**, vì model hiện dùng direction=high.

Checklist mở rộng này chưa được triển khai trong phạm vi được duyệt. Bộ test hiện có đã được cập nhật cho quy trình TRAIN của Stage1; việc một số ca biên đã có test không có nghĩa toàn bộ giai đoạn 3 đã hoàn tất.

### Giai đoạn 4 — Đồng bộ Python, notebook và báo cáo (đã hoàn tất)

Đây là bước bàn giao bắt buộc sau mỗi giai đoạn có sửa logic, **kể cả chỉ triển khai giai đoạn 1**; không đợi người dùng duyệt toàn bộ ablation của giai đoạn 2 mới cập nhật notebook. Các kiểm tra quy trình/names ở giai đoạn 1 cũng phải được hoàn tất trong cùng đợt bàn giao đó.

- [x] Sửa cell hiệu chỉnh W trong `tools/build_notebooks.py`: TRAIN calibration; TEST evaluation sau khóa model. Sửa caption, protocol, metadata và bảng tên metric đồng bộ.
- [x] Sửa `tools/run_notebooks.py`: auditor baseline từng **bắt buộc** `W_selection_test` và bốn tên TEST. Đã đổi thành kiểm tra nguồn TRAIN, model bị khóa và schema mới; giữ kiểm tra bốn figure riêng, bảng tổng hợp, input hashes và không có audio payload.
- [x] Tạo benchmark Python mới cho cấu hình được chọn, rồi mới dùng benchmark đó đối chiếu notebook mới. Không bỏ đối chiếu chỉ vì kết quả mới khác commit cũ.
- [x] Tạo lại và thực thi ba notebook bằng kernel mới, lưu output thật; đối chiếu tất cả 24 trường hợp Python/notebook trên tám WAV, đồng thời tách bảng bốn TEST để báo cáo hiệu suất.
- [x] Tạo lại `JUPYTER_NOTEBOOKS_CODE_ONLY.zip`: đúng ba notebook đã chạy, byte trùng file ngoài, không WAV/LAB.
- [x] Cập nhật README, ba tài liệu thuật toán và `KET_QUA_HIEN_TAI.md`. Tên metric/protocol mới phải được cập nhật trong `export_slide_data.py`/`package_submission.py` nếu còn được dùng.

Kết quả trước/sau lấy baseline từ Git commit hiện tại, không nhân thêm các bản sao dự án hoặc thư mục lịch sử. Bảng ablation gộp vào một khu vực kết quả có tên rõ; không tạo hàng loạt hình và notebook phụ.

### Giai đoạn 5 — Bộ trình chiếu chung, chờ mở lại phạm vi

Phần này có trong nhận xét nhưng yêu cầu trước của người dùng đang ưu tiên mã nguồn, tạm hoãn các phần khác. Chỉ đưa vào bước thực thi khi người dùng muốn làm tiếp:

- Ghép nội dung cần thiết thành **một deck tiếng Anh/PDF chung** có bảng so sánh ba thuật toán trên bốn TEST, dùng MAE cùng định nghĩa và cùng protocol.
- Giữ ba notebook riêng cho ba thành viên.
- Điền STT nhóm, họ tên, MSSV và phân công thật; hiện người dùng chưa cung cấp.
- Không dùng tên nhóm 07 hoặc tên thành viên của BT1 cho dự án mình.

## 4. Bảng và tiêu chí nghiệm thu sau sửa

Mỗi file/thuật toán cần có: split, GT/pred region count, START/END error có dấu, MAE/RMSE chính, status, T gốc/LOW/HIGH, cấu hình detector, W và nguồn chọn tham số. Với nhiều vùng, lưu danh sách vùng và lỗi của các cặp; không ép tất cả thành một envelope.

Bảng trước/sau phải ghi cả delta theo từng file, mean/median/min/max, số file đúng/sai số vùng và các file xấu đi. Không lấy mean chung đẹp hơn để che một file studio bị hỏng.

Giai đoạn 1 sửa quy trình/names có thể không giảm MAE. Giai đoạn 2 cũng không có cam kết đạt MAE của BT1: chưa có source BT1 để tự chạy lại, và các thay đổi thành phần chưa được kiểm chứng riêng. Nếu TEST sau khóa cấu hình có regression, báo cáo nguyên trạng và điều tra nguyên nhân; không quay lại tune cho TEST. Nếu phải tiếp tục dùng TEST để phát triển, ghi rõ nó đang đóng vai trò dữ liệu phát triển và cần holdout mới cho kết luận độc lập.

## 5. Phê duyệt, nghiệm thu và các quyết định bàn giao

**Phạm vi đã duyệt và hoàn tất ngày 05/10/2026:** giai đoạn 1 kèm kiểm chứng và đồng bộ notebook/báo cáo ở giai đoạn 4. Giai đoạn 2, 3 và 5 vẫn hoãn. Công thức Gaussian, thuật toán Binary, framing và HIGH/LOW được giữ theo cấu hình baseline.

Bằng chứng nghiệm thu: 98/98 test đạt trong 22.091 s; ba notebook trong thư mục chính chạy bằng ba kernel mới, khớp Python ở đủ 24/24 trường hợp về model, metric và FINAL/ngưỡng. TT2 có 200 dòng khảo sát TRAIN và kiểm tra digest khóa model. Có 15 PNG nhúng đã được xem trực quan; ZIP CODE chứa đúng ba notebook đã chạy, byte trùng file ngoài, không WAV/LAB/audio. Ba notebook cũng đã chạy độc lập trước đó, mỗi thư mục chỉ có notebook riêng và data. Runtime thực tế là Python 3.14; Python 3.10 chỉ được kiểm tra cú pháp, chưa chạy runtime.

Đối chiếu commit baseline `111ab37`: đủ 24 trường hợp giữ nguyên FINAL regions, HIGH/LOW và metric chính/phụ; 0 regression, mean TEST TT1/TT2/TT3 vẫn 20.00/13.75/12.50 ms. Cả 28 file gốc trong `data`/`Source` có SHA256 không đổi. Năm trường hợp END muộn đã ghi trong báo cáo vẫn còn; Stage1 cải thiện giao thức và tên metric, chưa chứng minh giảm MAE.

Ba quyết định bàn giao:

- Tích hợp vào thư mục chính của người dùng trên branch cục bộ `codex/endpoint-train-calibration`, giữ `main`; chưa push GitHub.
- Model legacy nhập vào vẫn giữ metadata TEST đúng lịch sử. Exporter/auditor mới yêu cầu schema 2, nên artifacts legacy cần được tái tạo để dùng với công cụ mới; không đổi nhãn thành TRAIN để vượt kiểm tra.
- PPTX/PDF và ba ZIP Python `THUAT_TOAN_1/2/3.zip` hiện có là previous version trước Stage1; chi phí cập nhật chúng đang chờ mở lại phạm vi. ZIP CODE notebook hiện hành đã được tái tạo. STT nhóm/họ tên/MSSV vẫn là placeholder do chưa có thông tin người dùng.

Bằng chứng lưu trong [bảng hồi quy 24 trường hợp](../outputs/tables/all_all_dataset/regression_before_after.csv), [ba notebook hiện hành](../notebooks/README.md) và [ZIP CODE](../submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip). Cả chín binary PPTX/PDF và ZIP Python cũ giữ nguyên byte.

Các lệnh đã chạy bằng `.venv/Scripts/python.exe`: `-m unittest discover -s tests -v`, `tools/build_notebooks.py`, `tools/run_notebooks.py`, `tools/run_notebooks.py --validate-only`; tất cả hoàn tất thành công. Kết quả kiểm tra hồi quy/hash/manifest riêng đã được ghi ở trên.
