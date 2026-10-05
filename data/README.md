# Dữ liệu thực nghiệm

- `train/`: 4 WAV và 4 LAB gốc; học ngưỡng TT1/TT3, noise floor và đề xuất W ban đầu của TT2.
- `test/`: 4 WAV và 4 LAB gốc; dùng đánh giá. Theo yêu cầu chọn W trước đó, TT2 còn dùng LAB của 4 test để chọn một W chung.
- `train/README.txt`: mô tả LAB gốc của giảng viên.

Phone có Fs 16 kHz; studio 44.1 kHz; cả 8 WAV là mono PCM 16-bit. Mỗi cặp WAV/LAB có cùng tên. LAB chứa `start end sil|v|uv` theo giây; `v` và `uv` đều thuộc speech. F0mean/F0std không dùng.

WAV/LAB gốc được giữ nguyên. TT2 hiện Energy-only, W = 20 chung, ghi `test_tuned_not_independent`. Model cố định không dùng LAB target để nắn biên. Không nộp WAV trong ZIP theo hướng dẫn giảng viên.
