# Tài liệu nguồn

| Thư mục | Nội dung |
|---|---|
| `assignment/` | Đề hiện hành, đề cũ để đối chiếu và hướng dẫn trình bày/nộp/demo |
| `references/` | CS425 Hodgkinson và bài Giannakopoulos |
| `course_slides/` | Slide Chapter 6 của môn học |
| `examples/` | Hình mẫu BMP/EPS của giảng viên |
| `context/` | Báo cáo ngữ cảnh tham khảo |

Giữ nguyên các tài liệu gốc. Bản DOCX trùng ở thư mục gốc dự án đã được bỏ; bản chính nằm trong `assignment/`.

[Đề hiện hành](<assignment/Hướng dẫn BT thi GK nhóm 3-4 SV_Phân đoạn tín hiệu thành tiếng nói và khoảng lặng_XLTHS_GK 2026.docx>) yêu cầu frame 25 ms, hop 10 ms và khoảng lặng nội bộ tối thiểu 200 ms. [Hướng dẫn nộp/demo](<assignment/Hướng dẫn trình bày slide và nộp bài thi.pdf>) quy định slide PDF, thư mục `MaTheSV-HoTen` và không upload WAV.

Theo yêu cầu bổ sung được người dùng xác nhận ngày 05/10/2026, TT2 chỉ dùng Histogram Energy, không Spectral Centroid. Bài Giannakopoulos là nguồn tham khảo cách đặt ngưỡng histogram; bản bài tập là thích nghi Energy-only với frame/hop 25/10 ms. Padding 250 ms chỉ ở candidate stage; FINAL endpoints dùng High/Low STE.

Đề và yêu cầu mới của thầy được ưu tiên so với cấu hình trong tài liệu tham khảo. Báo cáo ngữ cảnh có cấu hình lịch sử 20/10 ms; đường `--compare-context` giữ riêng và có thể tái tạo bằng CLI khi cần.
