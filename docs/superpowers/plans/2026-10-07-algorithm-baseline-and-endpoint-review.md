# Kiểm chứng phản biện và tách kết quả thuật toán — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Báo cáo riêng kết quả lõi của từng thuật toán và kết quả cải tiến HIGH/LOW, sửa tài liệu còn cũ, kiểm tra thời gian biên bằng bằng chứng TRAIN; không thay thuật toán để chạy theo MAE của BT1.

**Architecture:** Giữ cấu trúc Python hiện có. Tạo hai chính sách endpoint có tên `core` và `enhanced`, khóa chính sách trong model trước TEST; chung feature extraction và metric, khác phần quyết định/hậu xử lý được công bố rõ. Giữ quy ước support hiện tại trong đợt tích hợp hai nhánh; khảo sát quy ước thời gian khác thành một bước nghiên cứu riêng, không tự đưa vào production.

**Tech Stack:** Python chuẩn, các hàm tính toán tự cài; Matplotlib/IPython để trình bày, Jupyter để bàn giao notebook. Không thêm thư viện fit, VAD, histogram, thống kê hoặc DSP thay cho thuật toán tự viết.

**Spec:** [Đề cập nhật nhóm 3–4](<H:/GIUAKI_XLTHS/Source/assignment/Hướng dẫn BT thi GK nhóm 3-4 SV_Phân đoạn tín hiệu thành tiếng nói và khoảng lặng_XLTHS_GK 2026.docx>), [phản biện mới](<C:/Users/Admin/Downloads/XAC_THUC_SO_SANH_VA_DE_XUAT_SUA_CODE (1).md>), [kế hoạch trước](H:/GIUAKI_XLTHS/reports/KE_HOACH_CAI_TIEN_ENDPOINT.md), các ràng buộc người dùng và phần thiết kế dưới đây. Phản biện là nguồn cần kiểm chứng; các khẳng định của tác giả không tự trở thành yêu cầu của thầy.

**Trạng thái thực thi 07/10/2026:** Người dùng đã duyệt và implementation Tasks1–5 đã thực hiện trên local `codex/endpoint-modes`; backup remote `codex/backup-before-endpoint-modes-20261007` tại `5ceb289c51079d53b977f5316bcc70971d119c47`. Task6 gate sau tích hợp/sửa writer+units:173testsPASS,24enhancednumeric exact,48core/enhanced cases,124protected hashes và3freshnotebooks/CODEZIP. Review độc lập cuối/visual và archive do root hoàn tất trước bàn giao; không push/merge implementation. Chi tiết: [báo cáo nghiệm thu](../../../reports/KET_QUA_NGHIEM_THU_2026_10_07.md).

**Ghi chép lập kế hoạch lịch sử (trước duyệt, không phải trạng thái hiện tại):** CHỜ NGƯỜI DÙNG DUYỆT. Ngày 07/10/2026. Lượt này chỉ đọc code, DOCX, artifacts và chạy probes đọc/số học nhỏ; chưa sửa code, chạy lại suite/kernel hay tái tạo artifacts. HEAD hiện là `d035dcb4d12e5f01abe7b33b6bd812a806193f1e`, nhưng working tree có các sửa Gaussian/notebook đã hoàn tất trước đó, chưa commit; đây mới là trạng thái phải giữ khi triển khai tiếp. Bằng chứng nghiệm thu trước là 133 tests, 24 trường hợp, không phải tests mới của lượt lập kế hoạch này.

## A. Kết quả kiểm chứng tài liệu

Ba subagent kiểm tra độc lập: giao thức/metrics/notebook; vai trò ngưỡng ba thuật toán; quy ước biên và 200/100 ms. Không subagent nào sửa file hoặc tái tạo kết quả.

| Nhận xét trong file | Kết luận với code hiện tại | Bằng chứng |
|---|---|---|
| TT2 chọn W từ TEST | **Đã sửa trong đợt trước.** Nhận xét mô tả phiên bản cũ. | `app/pipeline.py:245–290` và `:445–451`; `app/weight_selection.py:75–77` yêu cầu TRAIN. Hai model/schema và notebook hiện ghi TRAIN. |
| MAE có hai tên chỉ khác hoa/thường | **Đã sửa.** Không cần đổi tên metric lần nữa. | Primary=`mae_ms`/`rmse_ms`; diagnostic=`tolerance_boundary_*`, `matched_boundary_mae_ms`; CSV writer chặn casefold collision. Năm CSV metrics hiện tại đọc được bằng PowerShell Import-Csv. |
| Ngưỡng TT1 mất vai trò ở FINAL | **Đúng với model hiện tại.** Không phải quy luật của mọi TRAIN/model. | T1=.0010293375 < Q95=.0012140145; LOW=Q95, HIGH=.0025149672. Mọi HIGH frame tự thỏa seed z≥T1, nên T1 chỉ còn ảnh hưởng candidate/debug trong cấu hình này. |
| TT2 cũng bỏ hoàn toàn ngưỡng histogram/W | **Khẳng định quá mức.** FINAL hiện bất biến theo W không có nghĩa W không tham gia. | HIGH trên bốn TEST bằng normalized histogram threshold. TRAIN sweep có 50 ngưỡng khác nhau mỗi file, raw frame counts thay đổi, nhưng FINAL giữ cùng biên. |
| TT3 bị noise floor đè mất T3 | **Không đúng với model hiện tại.** | T3=.0028777337 > noise upper=.0025149672; LOW=T3, HIGH=1.5T3=.0043166005. |
| 100 ms không phải yêu cầu đề | **Đúng.** Đây là filter bổ sung, đã công bố nhưng chưa có nhánh baseline production riêng. | `app/config.py:13–14`, `core/endpoints.py:76`; đề chỉ quy định minimum silence 200 ms. Filter đo span sau merge, không đo thời lượng voiced samples. |
| Shared hysteresis làm bảng hiện tại khó đại diện thuật toán gốc | **Có cơ sở.** Các số hiện là hiệu suất cả pipeline, không thể gán toàn bộ cho ngưỡng riêng. | Mọi thuật toán đi qua `detect_regions()`/HIGH–LOW. Đã có TRAIN ablation, chưa có chế độ lõi được hiệu chỉnh W riêng và báo cáo đủ TEST/notebook. |
| Lấy END theo support luôn sai15 ms | **Không đúng.** Support là quy ước hợp lệ đã công bố; offset so với quy ước khác không bằng sai số so với GT. | `core/features.py:125–127`, `core/endpoints.py:64,75`; đề không quy định END phải theo cạnh hop. |
| Lùi END sẽ giảm MAE cả ba10–15 ms | **Sai qua phản ví dụ đọc dữ liệu hiện tại.** | studio_F2 đang END sớm≈5.011 ms; lùi15 ms làm MAE7.506→15 ms. Chỉ đổi một END 15 ms thì MAE hai biên giảm tối đa7.5 ms. |
| Gaussian xấp xỉ chưa khớp STE | **Hạn chế có bằng chứng, không phải cài sai yêu cầu.** Đề chủ động yêu cầu giả định Gaussian. | TRAIN skewness speech≈1.38155, silence≈8.37357; chưa fit lognormal/gamma để kết luận mô hình đó tốt hơn. Solver hiện tại đã được sửa và không cần rewrite. |
| Sigma speech lớn là nguyên nhân duy nhất T3>T1 | **Giải thích thiếu căn cứ.** | TT1 dùng253 silence/178 speech trong overlap và objective diện tích; TT3 dùng497/794 toàn bộ khung có nhãn và 4 thống kê. T theo sigma speech không đơn điệu trong probe giữ các thống kê khác. |
| Chưa có bảng chung/main4 figures | **Bảng report và demo đã có.** Deck chung chưa có, nhưng slide đang ngoài phạm vi. | `reports/KET_QUA_HIEN_TAI.md`, `main_tt1/2/3.py`, `app/plotting.py`. Lượt này chưa nghiệm thu GUI chạy thật. |

### Phản ví dụ về timestamp

Với N=frame samples, H=hop samples và j=khung speech cuối, bốn timestamp dưới đây khác nhau:

| Quy ước END | Công thức | Lùi so với support ở Fs16k |
|---|---|---:|
| Hiện tại | (jH+N)/Fs | 0 ms |
| Công thức jH trong phản biện | jH/Fs | 25 ms; đây là đầu khung j |
| Cạnh phải ô hop | (j+1)H/Fs | 15 ms |
| Tâm khung + nửa hop | (jH+(N+H)/2)/Fs | 7.5 ms |

Đọc24 kết quả lưu, chỉ đổi END trong bộ nhớ, giữ START/khung được chọn/merge/count:

| Mean MAE bốn TEST, ms | Support hiện tại | jH | (j+1)H | Center+H/2, chỉ đổi END |
|---|---:|---:|---:|---:|
| TT1 | 20.00 | 21.25 | 18.75 | 18.7514 |
| TT2 | 13.75 | 17.50 | 15.00 | 13.1264 |
| TT3 | 12.50 | 16.25 | 13.75 | 11.8764 |

Đây là counterfactual số học để bác lời hứa cải thiện mọi file, **không phải kết quả của detector đã sửa**, và không dùng bảng TEST này để chọn quy ước mới. Ví dụ TT1 phone_M2 đổi END về 2.520 s vẫn còn START−10 ms, nên MAE 5 ms, không phải 0.

### Tài liệu còn cũ thực sự cần sửa

- `outputs/README.md:7` vẫn gọi200 dòng W là TEST; `:13` mô tả các MAE alias đã bỏ.
- `submission/README.md:3,7` còn trình bày ngày05/10 và98 tests như thông tin gói CODE hiện tại. Cần đưa phiên bản07/10/133tests lên phần hiện hành, giữ lịch sử riêng.
- Slide/README cũ có mô tả TEST-tuned W; slide vẫn được giữ nguyên theo yêu cầu trước của người dùng. Không coi tài liệu cũ là bằng chứng đường chạy mới còn leakage.

## B. Thiết kế đề xuất để duyệt

### Hai nhánh có vai trò rõ

| Chính sách | `core` — kết quả lõi phục vụ đối chiếu thuật toán | `enhanced` — pipeline hiện tại |
|---|---|---|
| TT1 | z≥T1 do Binary học TRAIN | Seed TT1 rồi HIGH/LOW/noise |
| TT2 | Energy>T_hist(W), chỉ Energy | Raw Energy seed rồi HIGH/LOW/noise |
| TT3 | z≥T3 hoặc z≤T3 theo direction | HIGH/LOW; chỉ direction high, giữ guard hiện tại |
| Silence nội bộ | Nối gap ước lượng<200 ms | Giữ quy tắc hiện tại<200 ms |
| Speech minimum | 0 ms; không thêm filter 100 | 100 ms như hiện tại, ghi rõ heuristic |
| TT2 padding 250 ms | Chỉ diagnostic, không dùng FINAL | Giữ loại khỏi FINAL như hiện tại |
| Endpoint geometry đợt này | Cùng support hiện tại để tách riêng tác dụng detector | Support hiện tại |

Tên đầy đủ của baseline là **lõi ngưỡng theo adaptation của bài**: TT2 đã bỏ Centroid theo chỉ đạo của thầy, và framing 25/10 ms theo đề. Không gọi nó là bản tái lập đầy đủ paper Histogram có Centroid/padding.

**Đề xuất giữ `enhanced` làm mặc định CLI/demo**, để bảo toàn yêu cầu trước của người dùng rằng HIGH/LOW phải thực sự tham gia detection. Bổ sung `--endpoint-mode core` để mỗi sinh viên chạy kết quả lõi. Report và notebook trình bày cả hai bảng, không gọi MAE enhanced là MAE thuật toán ngưỡng đơn. Một lần chạy main riêng vẫn mở đúng 4 figure của mode được chọn.

W phải fit **riêng theo từng policy trên TRAIN**, bằng chính nhánh sẽ dùng suy luận. Không lấy candidate W1 hoặc enhanced W20 làm W core đã tối ưu. Giữ rule chọn hiện tại: ít file không chấm được nhất, worst regret, mean hợp lệ, hòa mới xét preference20. Khi không có W cho4/4 region counts đúng, ghi tình trạng đó; không ép mỗi WAV thành một vùng hoặc cho primary MAE=0.

Không đổi LOW TT1 thành T1 ngay: LOW thấp hơn có thể kéo thêm noise/END. Có thể đưa biến thể đó vào khảo sát TRAIN ở Task4, nhưng không tự đưa vào default enhanced. Giữ toàn bộ số học Gaussian đã sửa.

### Global Constraints

- Giữ3 thuật toán, Python tự tính; TT2 chỉ Energy, không F0/Centroid, không thêm ZCR/thuật toán thứ 4.
- Analysis frame 25 ms, hop 10 ms, nearest samples/ties-to-even; Fs44.1k dùng1102/441samples, thời gian thực được báo cáo.
- Minimum internal silence 200 ms; exact estimated200 ms phải giữ, <200 ms mới nối. Silence đầu/cuối không ép kéo dài200 ms.
- Mỗi FINAL speech region có START/END; không candidate/debug trong primary figure/MAE.
- TRAIN chọn tham số và khóa model trước mở TEST; target LAB chỉ dùng scoring/plot. Không rule filename, GT snapping, TEST tuning hoặc bỏ lỗi lớn.
- Giữ `historical_test_exposure=true`; legacy TEST-calibrated model phải giữ provenance cũ. Đổi protocol không tạo holdout độc lập.
- Metric schema 2 tiếp tục `mae_ms`/`rmse_ms`; thiếu/thừa vùng làm primary undefined, không dùng mean subset thay mean đầy đủ.
- Không sửa slides/PPTX/PDF, dữ liệu Source/audio hoặc ZIP Python cũ. CODE notebook ZIP được cập nhật khi source notebook thay đổi.
- Không hứa MAE core thấp hơn enhanced hoặc timestamp mới làm mọi file tốt lên.

### Review Focus

1. Legacy model thiếu mode phải resolve enhanced; baseline không tự đòi noise/direction-high và enhanced vẫn từ chối low.
2. TT2 ở peak energy0; WAV im lặng/không frame; threshold equality và native Energy units khác normalized STE.
3. Gap190/200/210 ms, nhiều regions, short voiced islands, mép WAV và phần đuôi không đủ analysis frame.
4. Đổi nhãn TEST hoặc đảo thứ tự file không làm W/model/threshold thay đổi; mode của model phải khớp detector/selector/auditor.
5. Missing/extra regions, mỗi thuật toán có W/model provenance riêng, và notebook source/output/ZIP bị lệch sau chỉnh code.

## C. Công việc chi tiết

### Task1 — Khóa baseline hiện tại, sửa diễn giải còn cũ

**Files:** sửa `outputs/README.md`, `submission/README.md`, `reports/KET_QUA_HIEN_TAI.md`; thêm tests metadata/documentation phù hợp trong `tests/test_training_protocol.py`/`tests/test_notebook_protocol.py`. Dùng một thư mục evidence của đợt thực thi trong outputs, không lưu nhiều snapshot project.

**Interfaces:** baseline capture gồm **3 model enhanced hiện tại và 24 kết quả** (8 WAV × 3 thuật toán), input/artifact hashes và versions. Core models mới chỉ có từ Task3.

- [ ] Viết assertion cho W sweep current có TRAIN names, casefold-unique metric headers và README không gọi bảng current là TEST.
- [ ] Chạy test RED cho hai README stale; khóa bảng trước sửa gồm FINAL/masks/metrics/T/W/LOW/HIGH và hash Source/data/slides/ZIP cũ.
- [ ] Sửa đúng mô tả current TRAIN/schema 2; đưa thông tin05/10 về lịch sử, giữ historical exposure. Không đổi metric names/công thức.
- [ ] Chạy tests mục tiêu và suite hiện có. Gate: số trước sửa khớp nghiệm thu07/10, docs phản ánh current artifacts; không regenerate số liệu chỉ vì đổi prose.

### Task2 — Tách quyết định lõi và FINAL policy

**Files:** sửa `app/pipeline.py`, `core/postprocess.py` nếu cần helper vùng trực tiếp; tạo `tests/test_endpoint_modes.py`. Giữ `algorithms/tt1_hodgkinson.py`, số học TT3 và `core/features.py`. Chỉ sửa TT2 helper nếu cần public API unpadded, không rewrite histogram.

**Interfaces:**
- Thêm `algorithm_decision(algorithm: str, features: dict, params: dict) -> tuple[list[int], dict]` trong pipeline: trả native raw decision và threshold diagnostics, không cần LAB/name/duration.
- Giữ `detect_regions(algorithm, features, duration, params) -> dict`; đọc `params['endpoint_mode']`, thiếu mode resolve enhanced. Policy ngoài `{core,enhanced}` báo lỗi.
- Output hiện có mask/final_regions/predicted_boundaries/diagnostic giữ nguyên; thêm mode, native threshold/units, geometry, min speech và padding stage để không nhập nhằng. Ở `core`, `low_ste_threshold`/`high_ste_threshold`/`endpoint_noise` là `None` (JSON null, CSV ô trống), không tạo HIGH/LOW giả; threshold trực tiếp vẫn có trong native threshold/units. Sweep/export/auditor/plot phải đọc theo mode và chấp nhận các trường không áp dụng này.

- [ ] Viết tests: TT1/TT3 equality làspeech; TT2 Energy equality làsilence theo dấu `>`; TT3 core-low giữ đúng raw inequality; core chạy không có endpoint_noise; enhanced vẫn reject low/unknown; legacy thiếu mode bằng enhanced.
- [ ] Viết tests: estimated gap 190→gộp,200/210/250→giữ2; core vùng support80ms vẫn giữ, enhanced loại; không đổi input. WAV zero với model TRAIN hợp lệ không tạo vùng giả; empty features cho emptyregions; TT2 peak0 tránh divide-by-zero.
- [ ] Chạy RED; thêm helper và branch core=raw decisions→merge internal gap 200→FINAL regions, không filter 100. Enhanced dùng logic hiện tại và noise guard của riêng branch đó.
- [ ] Chạy tests mục tiêu: `python -m unittest tests.test_endpoint_modes tests.test_endpoints tests.test_endpoint_robustness tests.test_gaussian_direction_contract -v`. Gate: raw phù hợp inequalities, final không chứa padding/debug, enhanced giữ numeric baseline24 cases.

### Task3 — Model và W phải khóa cùng policy

**Files:** sửa `app/pipeline.py`, `app/weight_selection.py`, `tests/test_training_protocol.py`, `tests/test_weight_selection.py`.

**Interfaces:**
- `fit_training_model(algorithm, train_records, *, endpoint_mode='enhanced') -> tuple[dict,list[dict]]`.
- Giữ `sweep_final_weights(records, model, predictor, ...)` nhưng model mang policy và mỗi row/selection mang mode; predictor đi đúng policy đó. Row/manifest khóa cả endpoint_mode, boundary_convention, minimum_speech_ms, minimum_silence_ms và padding_stage; LOW/HIGH nullable theo contract Task2, không index rồi ép float cho core.
- Model mới schema 3, metric schema 2; thêm endpoint_mode, feature/decision rule, minimum_speech_ms, boundary_convention và padding_stage. Model schema1/2 import resolve enhanced, không đổi provenance hoặc numeric params ngầm.

- [ ] Viết RED: calibrator dùng đúng core/enhanced scorer; TEST/external records bị từ chối; TT1/TT3 core không phụ thuộc noise-floor fit; W candidate F1 không tự trở thành FINAL W; core LOW/HIGH null đi qua sweep/export/audit/plot không lỗi hoặc hiện ngưỡng giả.
- [ ] Fit T1/T3 từ TRAIN như hiện tại, TT2 grid1…50 riêng mỗi policy; core min speech 0, enhanced100. Với enhanced, selectedW/T/noise và kết quả phải giữ như baseline. TT3 core-low được fit; enhanced vẫn reject ngay.
- [ ] Khóa tất cả model/mode/W/noise trước TEST; lưu digest từng mode và mode-specific selection manifests. Primary mean undefined nếu count mismatch; báo invalid count rõ.
- [ ] Chạy tests training/weight/modes. Gate: thay target TEST LAB không đổi detection/calibration; expected200 row TRAIN sweep mỗi mode TT2; không gọi W20 hoặc W1 là optimum duy nhất khi hòa.

### Task4 — Thời gian biên và 200/100 ms: khảo sát TRAIN, không đổi default bằng TEST

**Files:** mở rộng `tools/check_endpoint_robustness.py`, `tests/test_endpoint_robustness.py`; nếu cần helper dùng riêng thí nghiệm thì thêm `core/decision_timing.py`, `tests/test_decision_timing.py`. Không nối helper thử nghiệm vào detector default trong task này.

**Interfaces:** `decision_cells(features: dict, duration: float) -> tuple[list[float],list[float]]` dùng tâm và hop thực. Các cạnh nội bộ là midpoint giữa hai centers liên tiếp; cạnh ngoài 0/duration theo nearest-frame convention được công bố. Trả một cell không chồng lấn cho mỗi frame; không thay analysis window 25 ms. Chuỗi không frame trả([],[]).

- [ ] Viết RED cho cell coverage/order/clipping và nhiều regions ở hai Fs; gap decision cells đúng190/200/210 ms; mép0/duration; chỉ một frame và đuôi ngắn. Không dùng `regions_to_mask` full 25ms support để chấm cell-basedregions.
- [ ] Thực hiện helper và diagnostic branch dùng cells **nhất quán** cho START/END/gap/span/mask membership; không chỉ trừ15 ms END. Không coi jH là END của chính khungj.
- [ ] Khảo sát tối đa các yếu tố đã khai báo: support hiện tại vs centered cells; min speech 0 vs100; riêng enhancedTT1 LOW gốc vsLOW=T1/HIGH=max(1.5T1,noise_upper). Thử từng yếu tố, không quét toàn bộ tổ hợp. Học trên 3 TRAIN, kiểm tra file TRAIN thứ 4 trong4 folds. Với TT2, **hiệu chỉnh lại W1…50 riêng cho từng nhánh thử nghiệm bằng chính geometry/cleanup/scorer của nhánh đó trên 3 fit files**, rồi khóa W để chấm file TRAIN giữ lại. Manifest ghi trial_id và geometry/filter/mode dùng cho chọn W; không dùng W đã chọn theo support-enhanced để gọi nhánh cells hoặc core là đã tối ưu.
- [ ] Xuất START/END errors, regioncounts, full/valid-only MAE/RMSE và config provenance. Gate numeric: enhanced current unchanged; báo mọi fold regression. Chưa chọn convention từ TEST counterfactual ở phầnA.
- [ ] Reuse48 waveforms physical190/200/210/250, hai Fs/sáu phase; thêm burst 75/95/100/105ms và audioedges/weak tails. Phân biệt tests duration-rule trên nhãn với characterization duration thực.

**Giới hạn bắt buộc:** centered cells **không đủ** bảo đảm silence vật lý200 ms được giữ; window có speech một phần vẫn có thể bị phân loại speech. Không biến24 ca physical200/210 hiện fail thành pass bằng đổi GT/giảm ngưỡng 200. Đề yêu cầu lọc khoảng lặng ước lượng dưới200, không nêu convention END hay bảo đảm detector biết mọi silence thật. Nếu cần sửa estimator để giữ đúng silence vật lý quanh200, cần thiết kế refinement riêng dựa trên TRAIN/waveform, không chỉ đổi timestamp. Task này bàn giao khảo sát và khuyến nghị; áp dụng production convention/LOW mới cần trình người dùng kết quả cụ thể để quyết định, giữ default hiện tại trong lúc đó.

### Task5 — Chạy hai mode, bảng/figure và notebook đồng bộ

**Files:** sửa `app/cli.py`, `app/pipeline.py`, `app/plotting.py`, `tools/build_notebooks.py`, `tools/run_notebooks.py`, các README/hướng dẫn3 thuật toán và `tests/test_notebook_protocol.py`. Main1/2/3 giữ riêng. Output mới dưới thư mục mode, không ghi đè evidence/legacy outputs cũ; cập nhật CODE ZIP sau fresh execution.

**Interfaces:** CLI `--endpoint-mode {enhanced,core}`, defaultenhanced; thêm `--compare-endpoint-modes` xuất bảng headless2 modes, fit tất cả TRAIN trước đọc TEST. Notebook giữ một config mode chọn4 figure; tính/khóa `MODELS={'core':..., 'enhanced':...}` trước TEST, lưu model digests theo mode và source digest chung.

- [ ] Viết RED: đường chạy mặc định vẫn 4 figure/một main call; figure metric FINAL của mode chọn; core plot Tnative và không giả vờ HIGH/LOW quyết định; enhanced plot HIGH/LOW thực.
- [ ] Bảng gồm filename/split/algorithm/mode/GT-count/pred-count/START/END/primaryMAE/RMSE/status và các trường timing/filter/padding đã khóa; summarize group theo algorithm+mode, không gom hai mode thành một mean. Core LOW/HIGH null phải được export/audit đúng và không vẽ như threshold có tham gia detector.
- [ ] Report so sánh bốn TEST riêng, támWAV riêng; raw/core và enhanced không gọi cùng là thuật toán gốc. Dùng `FINAL-region endpoint MAE` nếu muốn làm rõ nhiều regions; metric chính vẫn không bỏ lỗi lớn.
- [ ] Notebook mỗi sinh viên chứa code riêng, bảng 8 WAV/mode, so sánh 4 TEST và threshold diagnostics; mỗi mode TT2 có 200 TRAIN rows. Generator phải nhúng cả `algorithm_decision` và các helper core được Task2 thêm vào, không chỉ ba hàm pipeline cũ. Giữ 4 TEST plots + 1 illustration của mode được chọn; runner kiểm tra models/mode/sweep/metrics/source digests đúng mode. RunAllSave độc lập vẫn được.
- [ ] Rebuild→ba kernel mới→validate savedoutputs→ZIP exact3 notebooks byteequal. Lưu24 kết quả/mode (48 cho2 modes); diễn giải Gaussian là xấp xỉ theo đề, không chuyển fit sang gamma/lognormal. T1/T3 objectives khác nhau, không gán T3>T1 cho sigma đơn lẻ.

### Task6 — Nghiệm thu và báo cáo để bàn giao

**Files:** tạo một report kết quả thực thi trong `reports/`, một bảng before/after enhanced và một bảng core/enhanced trong outputs; cập nhật currentreport với link, không sửa lịch sử thành dữ liệu mới.

- [ ] Chạy tests mục tiêu sau mỗi thay đổi lớn, toàn suite sau tích hợp; unit tests phải có RED/GREEN khi thêm hành vi. Số133 là baseline, không ấn định sốtests tương lai phải bằng133.
- [ ] Enhanced: mọi24 FINAL/count/mask/quantitative metric/status trước–sau phải giữ; schema/provenance mới được đổi có chủ đích. Mọi thay đổi numeric bất ngờ là regression cần xử lý trước bàn giao.
- [ ] Core: báo đúng số vùng, thiếu/thừa, MAE undefined; có thể cao hơn enhanced. Không chọn nhánh báo cáo theo file để chỉ lấy điểm tốt.
- [ ] Chạy4 TEST sau khóa model, đánh giá8 WAV để regression; mean/median/min/max, counts và worstfiles riêng từng mode/algorithm. TEST đã được nhìn ở nhiều lượt phát triển, giữ disclosure.
- [ ] Kiểm tra notebook độc lập/fresh source–output digests, model lock, ZIP không WAV/LAB; hash Source/data/slides/ZIP Python giữ nguyên. Dọn scratch do tác vụ tạo sau khi lưu evidence cần thiết.
- [ ] Reviewer độc lập kiểm code+spec+actualartifacts; report rõ hàm sửa, before/after, mọi file xấu đi, giới hạn physical200/shortspeech và runtime3.10 chưa kiểm nếu không có môi trường.

## D. Phân công và thứ tự

1. Root giữ baseline, design/mode contract, tích hợp và báo cáo.
2. SubagentA: Task2 quyết định lõi/policy +tests; reviewer riêng sau khi xong.
3. SubagentB: Task4 timing/TRAIN/synthetic đọc contract của Task2; không sửa pipeline hoặc đổi default.
4. Sau Task2, root hoặc subagentA làm Task3 calibration; Task5 notebook/CLI dùng interface đã ổn định, giao subagentC và review riêng.
5. Task1 docs có thể song song nếu fileownership rõ; Task6 sau tất cả task bắt buộc. Không giao hai agent sửa cùng file cùng lúc.

Task4 là nghiên cứu tách biệt; Task2/3/5 không đợi một timing mới được chọn. Nhờ vậy vẫn có kết quả lõi/evidence đúng bản chất dù nghiên cứu chưa tìm được estimator tốt hơn.

## E. Các quyết định xin duyệt

- Thêm `core` để báo cáo đúng vai trò của3 ngưỡng, giữ `enhanced`/HIGH–LOW và defaultdemo hiện tại.
- Core không filter 100, chỉ mergeinternalgap<200; enhanced giữ100 được ghi rõ bổ sung.
- Chọn W riêng từng mode trên TRAIN, giữ historical exposure và metric schema 2.
- Timing/LOW=T1 là khảo sát có kiểm soát, **chưa thay production**; không lùiEND 15 ms hoặc épW1 theo file phản biện.
- Đồng bộ CLI/notebook/reports và sửa2README stale; slide giữ nguyên.

**Sau khi người dùng duyệt mới triển khai.** Plan này nối tiếp các hạng baseline/ablation trước đây đang hoãn; không tự coi phê duyệt Stage1/Stage4 cũ là phê duyệt thay policy mới. Phần kết quả thực nghiệm timing sẽ được trình riêng nếu cần thay default, vì chưa có bằng chứng để quyết định trước.
