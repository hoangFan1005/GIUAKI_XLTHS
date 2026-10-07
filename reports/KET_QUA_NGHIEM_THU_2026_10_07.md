# Nghiệm thu endpoint core/enhanced — 07/10/2026

Gate tích hợp sau sửa final review đã PASS; bàn giao cuối chờ root hoàn tất scoped re-review, commit và archive. Không push/merge implementation.

## Phạm vi và các hàm đã thay đổi

Ba thuật toán giữ nguyên: TT1 Binary Search cân bằng diện tích nhầm lẫn trên normalized STE; TT2 Histogram chỉ Energy; TT3 Gaussian equal-density xấp xỉ theo đề. Không thêm F0/Centroid/ZCR hay thuật toán thứ tư.

`app.pipeline.algorithm_decision` tách quyết định native; `detect_regions` chọn policy khóa trong model: core quyết định native, merge internal support gap <200 ms, không lọc minimum speech/padding FINAL; enhanced giữ noise/HIGH–LOW hysteresis và 100 ms span filter. `fit_training_model` fit riêng từng mode trên TRAIN, chọn FINAL W theo mode và khóa schema3/calibration digest trước TEST. `app.weight_selection.sweep_final_weights`/`select_final_weight` giữ primary undefined nếu count sai; `summarize` nhóm theo algorithm+mode. `run_experiment`, CLI/config/output routing, plotting và notebook builder/runner xử lý cả hai mode và audit provenance. `core.decision_timing.decision_cells` cùng `tools.check_endpoint_robustness` chỉ phục vụ nghiên cứu opt-in, chưa đổi timing/LOW production.

Đối chiếu Task3 `47cc00b`: 7 module tính toán có AST không đổi (TT2 chỉ sửa module docstring), 3 API `algorithm_decision`/`detect_regions`/`fit_training_model` có mọi AST node không đổi ngoài đúng hai string metadata: `sum of squared samples` → `mean squared sample amplitude`, và `Energy (sum of squared samples)` → `Energy (mean squared sample amplitude)`. Occurrence mỗi replacement phải đúng1; không dùng generic ignore AST. Task5 nâng phần consumer/metadata, không đổi native arithmetic. Initial byte gate TT2 fail do docstring đã được duyệt; diff thật được kiểm tra, gate sửa thành AST bỏ module docstring. Log fail được giữ, không che lỗi.

## Gate thực thi và môi trường

Fresh suite sau final-review-fix STABLE: **177 tests PASS**, 19.208 s unittest, 21.030 s wall, exit0. Baseline133, Task1 gate137 và pre-final-review-fix gate173 là số lịch sử; số mới lấy từ output thật. Acceptance numeric/audit: 1.414 s.

Môi trường: Python **3.14.7**, Windows11; numpy 2.5.2, matplotlib 3.11.1, nbformat 5.11.1, nbclient 0.11.0. Python3.10 **chưa kiểm** vì không có môi trường3.10; không suy diễn từ3.14.

Commands: `.venv/Scripts/python.exe -m unittest discover -s tests -v`; `.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-algorithm-baseline-and-endpoint-review/task6-acceptance.py`. Sau cleanup, scratch script và full raw command logs nằm trong [verification_logs.zip](../outputs/tables/endpoint_modes_oct07/verification_logs.zip); archive entry names giữ nguyên đường dẫn gốc. Root lưu review/rulings trước cleanup.

## Enhanced trước/sau baseline

**24/24 exact, 0 numeric regression**: full masks, FINAL regions, predicted/GT boundaries, region pairs, mọi metric cũ trừ intentional `model_schema_version` 2→3, status và mọi diagnostic cũ. T/W/noise/candidate W/F1 giữ nguyên. Bốn TEST enhanced mean MAE TT1/TT2/TT3 vẫn20.00/13.75/12.50 ms. Metadata schema/provenance mới không được dùng làm cớ để đổi số.

Chi tiết before/after đầy đủ: [final_regression.csv](../outputs/tables/endpoint_modes_oct07/final_regression.csv); [baseline.json](../outputs/tables/endpoint_modes_oct07/baseline.json); [final_verification.json](../outputs/tables/endpoint_modes_oct07/final_verification.json).

## Core/enhanced: toàn bộ bốn TEST và tám WAV

Primary mean chỉ defined khi mọi file đúng count/matching. Median/min/max dưới đây là trên file có primary defined, n được báo rõ; không thay primary mean bằng valid-only mean. RMSE trong evidence là mean per-file RMSE, khác pooled boundary RMSE.

| Scope | Mode | Algorithm | Defined/total | Primary mean MAE ms | Valid-only mean | Median defined | Min defined | Max defined | Extra regions | Worst defined file |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| all | core | tt1 | 7/8 | undefined | 83.212666 | 12.500000 | 2.494331 | 297.500000 | 2 | train/phone_M1 |
| test | core | tt1 | 3/4 | undefined | 7.500000 | 7.505669 | 2.494331 | 12.500000 | 2 | test/phone_M2 |
| all | core | tt2 | 6/8 | undefined | 10.837113 | 7.505669 | 2.500000 | 27.505669 | 2 | train/studio_M1 |
| test | core | tt2 | 3/4 | undefined | 9.170446 | 7.505669 | 2.500000 | 17.505669 | 1 | test/studio_F2 |
| all | core | tt3 | 8/8 | 12.498583 | 12.498583 | 7.502834 | 2.500000 | 27.500000 | 0 | test/phone_F2 |
| test | core | tt3 | 4/4 | 12.500000 | 12.500000 | 7.502834 | 7.494331 | 27.500000 | 0 | test/phone_F2 |
| all | enhanced | tt1 | 8/8 | 31.873583 | 31.873583 | 19.994331 | 2.494331 | 77.500000 | 0 | train/phone_M1 |
| test | enhanced | tt1 | 4/4 | 20.000000 | 20.000000 | 10.002834 | 2.494331 | 57.500000 | 0 | test/phone_F2 |
| all | enhanced | tt2 | 8/8 | 13.748583 | 13.748583 | 10.002834 | 2.500000 | 32.500000 | 0 | test/phone_F2 |
| test | enhanced | tt2 | 4/4 | 13.750000 | 13.750000 | 7.502834 | 7.494331 | 32.500000 | 0 | test/phone_F2 |
| all | enhanced | tt3 | 8/8 | 12.498583 | 12.498583 | 7.502834 | 2.500000 | 27.500000 | 0 | test/phone_F2 |
| test | enhanced | tt3 | 4/4 | 12.500000 | 12.500000 | 7.502834 | 7.494331 | 27.500000 | 0 | test/phone_F2 |

Core count failures: TT1 test/phone_F2 GT1/pred3 (extra2), TT2 train/phone_F1 và test/phone_F2 GT1/pred2 (extra1 mỗi file); MAE/RMSE đều undefined. Không bỏ lỗi lớn: core TT1 phone_M1=297.5 ms, phone_F1=242.5 ms. Core TT3/endpoints trùng enhanced TT3 trên dataset này, không cam kết với dữ liệu khác.

Toàn48 kết quả/allregions/status/counts: [final_core_enhanced.csv](../outputs/tables/endpoint_modes_oct07/final_core_enhanced.csv); thống kê: [final_summary.csv](../outputs/tables/endpoint_modes_oct07/final_summary.csv). [Báo cáo Task5](KET_QUA_ENDPOINT_MODES_2026_10_07.md) giữ bảng48file/mode chi tiết.

Các file core xấu hơn enhanced (mode comparison, không phải regression enhanced):

| WAV | Algorithm | Enhanced MAE ms | Core MAE ms | Lý do |
|---|---|---:|---:|---|
| train/phone_F1 | tt1 | 57.500000 | 242.500000 | suspiciously high MAE |
| train/phone_M1 | tt1 | 77.500000 | 297.500000 | suspiciously high MAE; boundary inside silence |
| test/phone_F2 | tt1 | 57.500000 | undefined | extra predicted regions; boundary inside silence |
| train/phone_F1 | tt2 | 2.500000 | undefined | extra predicted regions |
| train/studio_M1 | tt2 | 22.494331 | 27.505669 | OK |
| test/phone_F2 | tt2 | 32.500000 | undefined | extra predicted regions |
| test/studio_F2 | tt2 | 7.505669 | 17.505669 | OK |
| test/studio_M2 | tt2 | 7.494331 | 7.505669 | OK |

## Khóa model, notebook và bảo toàn

Fit cả6 model riêng core/enhanced trước lần load TEST đầu tiên; fresh calibration exact với6 saved models, tự recompute calibration digest (loại digest self-reference), model không đổi sau inference48cases. TRAIN_files đúng4; `parameter_selection_set=train`, `evaluation_protocol=train_selected_reused_test`, `historical_test_exposure=true`. TEST đã được xem nhiều lượt trước; đổi protocol không tạo holdout độc lập. Legacy TEST-calibrated provenance được giữ trong canonical old models.

Baseline148 SHA256; **124 protected files exact** gồm Source/data/slides/3ZIP Python cũ/toàn canonical outputs models/tables/diagnostics/predictions. Changed paths hợp lệ (source/docs/notebooks/CODEZIP) có before/after digests trong JSON. Không sửa slides/PDF/PPTX/data/LAB hay legacy canonical outputs.

Ba notebook chạy kernel fresh độc lập, code sources không import project runtime; ordered source digest khớp saved execution evidence. External audit kiểm core+enhanced model khóa, results/provenance/diagnostics và TT2 hai TRAIN sweeps200rows; mỗi notebook5PNG (4TEST enhanced+1illustration), tổng15PNG. CODE ZIP có đúng3executedipynb, từng entry byteexact với notebook hiện tại, không WAV/LAB/audio/widget payload. Root phụ trách visual15PNG và review độc lập.

## Timing research, mọi fold xấu đi và giới hạn vật lý

4TRAIN folds:6commontrials×3algorithms +1TT1 LOW=T1 trial =76heldout rows. TT2 mỗi branch W1..50 trên3fitfiles:3600calibration rows,24heldout TT2 scores. 19synthetic branch configs resolve1216waveform row SHA; cả76heldout SHA resolve vềfull locked models. Study được fit bởi source47cc00b trước sửa units label; manifest/native/synthetic models và CSV diagnostics lịch sử có thể còn ghi sum, nhưng phép tính thực là Energy=STE/N (mean squared sample amplitude). Giữ exact study bytes/provenance, không viết lại evidence thành dữ liệu mới; production6models/notebooks/plots hiện đã sửa label đúng. P3 provenance thiếu19branch manifest đã được sửa bằng deterministic reconstruction từ6native locked models+trials, không refit/re-score;7CSV numerical bytes giữ nguyên. Full model hashes và source-native links được kiểm fresh.

Không đẩy cells/LOW vào default. Estimated support gap exact200ms giữ, <200ms mới nối; 100ms enhanced filter đo span sau merge. Frame25/hop10 nearest samples/ties-to-even:16k400/160samples;44.1k1102/441samples, actualframe24.988662ms. Edges đầu/cuối không ép200ms; cells có thể extend nearest decision ra0/duration và incomplete tail.

Physical GT không được đổi. Support oracle gộp cả24ca silence vật lý200/210ms (MAE undefined); cells oracle vẫn gộp23/24, chỉ1ca210ms giữ2vùng. Tất cả19locked branches gộp cả24ca200/210ms/branch. Có8shortbursts75/95/100/105ms và8edge/weak-tail cases trong characterization. Quy tắc200ms đúng với estimated support, không bảo đảm mọi physical silence≥200ms vì cửa sổ25ms overlap speech. Cần estimator mới có bằng chứng riêng nếu muốn sửa giới hạn vật lý.

LOW=T1 TT1 phone_M1 fold xấu thêm **+205ms**, fullmean54.997→106.247ms; không thay noise/hysteresis LOW production. TT1 enhanced ở model hiện tại vẫn chịu noise policy, sửa đợt này là báo thêm faithful core. Các timing trials không tự làm mọi MAE tốt hơn.

| Trial | Algorithm | Held-out TRAIN WAV | Delta MAE ms |
|---|---|---|---:|
| core_cells_0 | tt3 | phone_F1 | +7.500000 |
| enhanced_cells_100 | tt2 | phone_F1 | +7.500000 |
| enhanced_cells_100 | tt3 | phone_F1 | +7.500000 |
| enhanced_tt1_t1_support_100 | tt1 | phone_M1 | +205.000000 |
| core_cells_0 | tt1 | studio_M1 | +2.505669 |
| core_cells_0 | tt2 | studio_M1 | +7.494331 |
| core_cells_0 | tt3 | studio_M1 | +7.494331 |
| enhanced_cells_100 | tt1 | studio_M1 | +2.505669 |
| enhanced_cells_100 | tt2 | studio_M1 | +2.505669 |
| enhanced_cells_100 | tt3 | studio_M1 | +7.494331 |

Tổng10MAE fold regressions,52alternative comparisons; count regression không xảy ra. Undefined core means được giữ. Bằng chứng: [fold_comparisons.csv](../outputs/tables/endpoint_modes_oct07/timing_research/fold_comparisons.csv), [heldout_summary.csv](../outputs/tables/endpoint_modes_oct07/timing_research/heldout_summary.csv), [manifest.json](../outputs/tables/endpoint_modes_oct07/timing_research/manifest.json), [physical waveforms](../outputs/tables/endpoint_modes_oct07/timing_research/locked_model_waveforms.csv).

## Coordination, review và archive

Ruling cho Task4/5 chạy song song vì research không chặn integration, native/model interfaces đã ổn định và file ownership tách rõ. Task4 sở hữu decision_timing/check_endpoint_robustness/research/focusedtests; Task5 sở hữu CLI/config/run_experiment/summary/plotting/notebooktools/docs/geometry audit guard. Cost nếu interface drift: focused integration fixes và rerun complete regression. Gate fullsuite/regression chỉ chạy sau root STABLE; không stage/commit cùng lúc. Không production timing/default change được phép. Reviewer tìm P2: `main_tt1 --evaluate-all` có thể ghi đè shared `tables/all` bằng subset. Writer nay chỉ ghi shared benchmark khi selected chứa đủ3algorithms; demo1algorithm giữ run riêng, regression test kiểm cả3fixed-entrypoints và shared bytes. Root còn sửa label units từ sum thành mean-square đúng với Energy=STE/N; số học không đổi,6models/digests/plots và3freshkernel notebooks+ZIP đã regenerate. Gate171tests trướcfix được giữ riêng; saufix freshsuite173PASS và acceptance48casesPASS. Đây là rerun có lý do source mới, không lặp suite không cần thiết.

Final review phát hiện shared comparison bị partial run ghi đè, entrypoint docs cũ và ba link guide thiếu file. Routing nay giữ canonical comparison chỉ cho đủ ba thuật toán/evaluate-all/no-file; partial xuất `comparison/tables/<run_name>[/single/<inputstem>]`, không bỏ kết quả. Bốn comparison CSV và shared TEST CSV mỗi mode được kiểm byte qua fixed-all-dataset, default TEST-only và all/fixed single-file; full command vẫn cập nhật benchmark. Ba guide dùng shared `tables/all/test_metrics.csv` hiện có và lọc cột algorithm.

Ruling single-file statistics: chỉ ghi evaluated WAV trong dataset_statistics.csv scoped của run. Full mode statistics/training_frames đã có TRAIN; không lặp TRAIN trong single-file. Dedup theo resolved wav_path, không theo filename; nguồn external trùng basename TRAIN vẫn phân biệt. Cost là focused fixture tests và một refresh metadata-only bằng run_experiment thật; không đổi arithmetic/tuning hoặc chạy lại notebook kernels. Full mode statistics sửa 12 dòng trùng/all_dataset thành 8 nguồn unique với 4TRAIN/4TEST; hai CSV này là intentional deltas, 195/197 fresh artifacts còn lại byte exact. Old canonical CSV và 124 protected hashes giữ nguyên.

Final fix RED: ba regression tests ghi 11 failures cho partial routing/statistics/guide links; doc RED riêng fail1. GREEN14 integration tests; một fresh fullsuite sau source STABLE. Gate173 trước fix và acceptance trước fix được giữ thành before-final-review-fix evidence. Saved audit cả ba notebook/15PNG, exact ba CODE ZIP entries và source digests PASS; run_experiment writer không được nhúng, notebook/ZIP bytes không đổi. `outputs/endpoint_modes/comparison/artifact_audit.json` là snapshot lịch sử Task5 units/pre-final-routing; source toàn file trong snapshot không được tuyên bố hiện hành. Xem final_verification.json cho arithmetic/native/model/notebook audit hiện hành và final-fix-metadata-refresh.json trong verification archive cho hashes hai intentional CSV.

Root lưu final independent review và artifact visual acceptance ở mục bàn giao dưới đây trước cleanup. RED/GREEN các Task1–5, final-fix evidence, reviewer packages/reports và intermediate taskN logs được bảo toàn exact bytes/digests trong [verification_logs.zip](../outputs/tables/endpoint_modes_oct07/verification_logs.zip). Standalone retained: baseline.json/final_verification.json/final_regression.csv/final_core_enhanced.csv/final_summary.csv/timing_research CSV+JSON và báo cáo này. Không xóa protected hoặc unowned artifacts.
