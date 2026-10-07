# Kết quả endpoint core và enhanced — 07/10/2026

Cùng ba thuật toán và hai policy được fit TRAIN độc lập trước TEST. Không chọn mode theo từng bản ghi. TEST từng được xem trong lịch sử; protocol `train_selected_reused_test`, `historical_test_exposure=true`. Gaussian vẫn là xấp xỉ theo đề. TT1 cân bằng diện tích nhầm lẫn, TT3 giải equal-density; sigma đơn lẻ không suy ra T3 > T1.

## Bốn TEST (12 kết quả mỗi mode)

| Algorithm | Mode | Mean FINAL-region endpoint MAE ms | Valid-only mean ms | Counts correct | Max defined MAE ms |
|---|---|---|---|---|---|
| tt1 | core | undefined | 7.500000 | 3/4 | 12.500000 |
| tt2 | core | undefined | 9.170446 | 3/4 | 17.505669 |
| tt3 | core | 12.500000 | 12.500000 | 4/4 | 27.500000 |
| tt1 | enhanced | 20.000000 | 20.000000 | 4/4 | 57.500000 |
| tt2 | enhanced | 13.750000 | 13.750000 | 4/4 | 32.500000 |
| tt3 | enhanced | 12.500000 | 12.500000 | 4/4 | 27.500000 |

## Tám WAV TRAIN + TEST (24 kết quả mỗi mode)

| Algorithm | Mode | Mean FINAL-region endpoint MAE ms | Valid-only mean ms | Counts correct | Max defined MAE ms |
|---|---|---|---|---|---|
| tt1 | core | undefined | 83.212666 | 7/8 | 297.500000 |
| tt2 | core | undefined | 10.837113 | 6/8 | 27.505669 |
| tt3 | core | 12.498583 | 12.498583 | 8/8 | 27.500000 |
| tt1 | enhanced | 31.873583 | 31.873583 | 8/8 | 77.500000 |
| tt2 | enhanced | 13.748583 | 13.748583 | 8/8 | 32.500000 |
| tt3 | enhanced | 12.498583 | 12.498583 | 8/8 | 27.500000 |

Primary MAE/RMSE cần đúng số vùng trên mọi file. Mean subset chỉ nằm ở valid-only; lỗi lớn vẫn được giữ. Counts sai của core TT1/TT2 không được thay bằng metric các vùng đã ghép. Core TT3 và enhanced TT3 cho cùng endpoints trên dataset này; đây không phải bảo đảm cho dữ liệu khác.

## Từng WAV và từng mode

| Split | WAV | Algorithm | Mode | GT / prediction regions | START s | END s | MAE ms | RMSE ms | Status |
|---|---|---|---|---|---|---|---|---|---|
| train | phone_F1 | tt1 | core | 1 / 1 | 0.53 | 3.235 | 242.49999999999994 | 342.9467888754755 | suspiciously high MAE |
| train | phone_M1 | tt1 | core | 1 / 1 | 0.4 | 4.055 | 297.49999999999983 | 380.67374482619607 | suspiciously high MAE; boundary inside silence |
| train | studio_F1 | tt1 | core | 1 / 1 | 0.66 | 2.164988662131519 | 17.494331065759596 | 17.67286044324514 | OK |
| train | studio_M1 | tt1 | core | 1 / 1 | 0.87 | 2.0649886621315194 | 2.4943310657596918 | 3.5275168222458926 | OK |
| test | phone_F2 | tt1 | core | 1 / 3 | 0.61 | 4.795 | undefined | undefined | extra predicted regions; boundary inside silence |
| test | phone_M2 | tt1 | core | 1 / 1 | 0.52 | 2.535 | 12.500000000000068 | 12.747548783982039 | OK |
| test | studio_F2 | tt1 | core | 1 / 1 | 0.76 | 2.3649886621315193 | 7.505668934240428 | 7.909282749784279 | OK |
| test | studio_M2 | tt1 | core | 1 / 1 | 0.45 | 1.9349886621315193 | 2.4943310657596918 | 3.5275168222458926 | OK |
| train | phone_F1 | tt2 | core | 1 / 2 | 0.54 | 2.745 | undefined | undefined | extra predicted regions |
| train | phone_M1 | tt2 | core | 1 / 1 | 0.46 | 3.525 | 2.4999999999999467 | 3.5355339059326623 | OK |
| train | studio_F1 | tt2 | core | 1 / 1 | 0.67 | 2.144988662131519 | 7.505668934240428 | 7.909282749784279 | OK |
| train | studio_M1 | tt2 | core | 1 / 1 | 0.91 | 2.0449886621315194 | 27.505668934240333 | 30.21043085261847 | OK |
| test | phone_F2 | tt2 | core | 1 / 2 | 1.05 | 4.015 | undefined | undefined | extra predicted regions |
| test | phone_M2 | tt2 | core | 1 / 1 | 0.53 | 2.515 | 2.4999999999999467 | 3.5355339059326623 | OK |
| test | studio_F2 | tt2 | core | 1 / 1 | 0.76 | 2.3449886621315192 | 17.505668934240436 | 19.046876672715992 | OK |
| test | studio_M2 | tt2 | core | 1 / 1 | 0.46 | 1.9249886621315193 | 7.505668934240317 | 7.909282749784209 | OK |
| train | phone_F1 | tt3 | core | 1 / 1 | 0.53 | 2.745 | 2.4999999999999467 | 3.5355339059326623 | OK |
| train | phone_M1 | tt3 | core | 1 / 1 | 0.46 | 3.535 | 7.500000000000062 | 10.6066017177983 | OK |
| train | studio_F1 | tt3 | core | 1 / 1 | 0.66 | 2.164988662131519 | 17.494331065759596 | 17.67286044324514 | OK |
| train | studio_M1 | tt3 | core | 1 / 1 | 0.91 | 2.0649886621315194 | 22.49433106575971 | 28.503392340758833 | OK |
| test | phone_F2 | tt3 | core | 1 / 1 | 1.01 | 4.085 | 27.499999999999968 | 32.59601202601319 | boundary inside silence |
| test | phone_M2 | tt3 | core | 1 / 1 | 0.52 | 2.525 | 7.499999999999951 | 7.905694150420921 | OK |
| test | studio_F2 | tt3 | core | 1 / 1 | 0.76 | 2.3649886621315193 | 7.505668934240428 | 7.909282749784279 | OK |
| test | studio_M2 | tt3 | core | 1 / 1 | 0.46 | 1.9349886621315193 | 7.494331065759696 | 7.902112055091845 | OK |
| train | phone_F1 | tt1 | enhanced | 1 / 1 | 0.53 | 2.865 | 57.50000000000011 | 81.31727983645311 | boundary inside silence |
| train | phone_M1 | tt1 | enhanced | 1 / 1 | 0.45 | 3.665 | 77.50000000000001 | 102.77402395547234 | boundary inside silence |
| train | studio_F1 | tt1 | enhanced | 1 / 1 | 0.66 | 2.164988662131519 | 17.494331065759596 | 17.67286044324514 | OK |
| train | studio_M1 | tt1 | enhanced | 1 / 1 | 0.91 | 2.0649886621315194 | 22.49433106575971 | 28.503392340758833 | OK |
| test | phone_F2 | tt1 | enhanced | 1 / 1 | 1.01 | 4.145 | 57.49999999999978 | 74.58216945088117 | boundary inside silence |
| test | phone_M2 | tt1 | enhanced | 1 / 1 | 0.52 | 2.535 | 12.500000000000068 | 12.747548783982039 | OK |
| test | studio_F2 | tt1 | enhanced | 1 / 1 | 0.76 | 2.3649886621315193 | 7.505668934240428 | 7.909282749784279 | OK |
| test | studio_M2 | tt1 | enhanced | 1 / 1 | 0.45 | 1.9349886621315193 | 2.4943310657596918 | 3.5275168222458926 | OK |
| train | phone_F1 | tt2 | enhanced | 1 / 1 | 0.53 | 2.745 | 2.4999999999999467 | 3.5355339059326623 | OK |
| train | phone_M1 | tt2 | enhanced | 1 / 1 | 0.46 | 3.545 | 12.499999999999956 | 17.677669529663625 | OK |
| train | studio_F1 | tt2 | enhanced | 1 / 1 | 0.66 | 2.164988662131519 | 17.494331065759596 | 17.67286044324514 | OK |
| train | studio_M1 | tt2 | enhanced | 1 / 1 | 0.91 | 2.0649886621315194 | 22.49433106575971 | 28.503392340758833 | OK |
| test | phone_F2 | tt2 | enhanced | 1 / 1 | 1.01 | 4.095 | 32.499999999999865 | 39.528470752104546 | boundary inside silence |
| test | phone_M2 | tt2 | enhanced | 1 / 1 | 0.52 | 2.525 | 7.499999999999951 | 7.905694150420921 | OK |
| test | studio_F2 | tt2 | enhanced | 1 / 1 | 0.76 | 2.3649886621315193 | 7.505668934240428 | 7.909282749784279 | OK |
| test | studio_M2 | tt2 | enhanced | 1 / 1 | 0.46 | 1.9349886621315193 | 7.494331065759696 | 7.902112055091845 | OK |
| train | phone_F1 | tt3 | enhanced | 1 / 1 | 0.53 | 2.745 | 2.4999999999999467 | 3.5355339059326623 | OK |
| train | phone_M1 | tt3 | enhanced | 1 / 1 | 0.46 | 3.535 | 7.500000000000062 | 10.6066017177983 | OK |
| train | studio_F1 | tt3 | enhanced | 1 / 1 | 0.66 | 2.164988662131519 | 17.494331065759596 | 17.67286044324514 | OK |
| train | studio_M1 | tt3 | enhanced | 1 / 1 | 0.91 | 2.0649886621315194 | 22.49433106575971 | 28.503392340758833 | OK |
| test | phone_F2 | tt3 | enhanced | 1 / 1 | 1.01 | 4.085 | 27.499999999999968 | 32.59601202601319 | boundary inside silence |
| test | phone_M2 | tt3 | enhanced | 1 / 1 | 0.52 | 2.525 | 7.499999999999951 | 7.905694150420921 | OK |
| test | studio_F2 | tt3 | enhanced | 1 / 1 | 0.76 | 2.3649886621315193 | 7.505668934240428 | 7.909282749784279 | OK |
| test | studio_M2 | tt3 | enhanced | 1 / 1 | 0.46 | 1.9349886621315193 | 7.494331065759696 | 7.902112055091845 | OK |

Các START/END scalar là envelope hỗ trợ đọc nhanh; khi nhiều vùng, diagnostic JSON chứa toàn bộ FINAL regions và primary metric chấm mọi cặp START/END theo thứ tự. Core LOW/HIGH null; native T nằm trong threshold diagnostics. Timing 25/10 ms dùng nearest samples/ties-to-even (44.1 kHz: 1102/441 samples); actual timing có trong metrics. Gap support <200 ms mới nối; exact200 ms giữ. Slides và ZIP Python cũ là previous version và được giữ nguyên.
