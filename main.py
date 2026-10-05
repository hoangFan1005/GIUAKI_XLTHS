"""Main tổng hợp để so sánh thuật toán; sinh viên có main_tt1/tt2/tt3 riêng."""
from app.cli import main, parse_args, run_cli


# Đây là điểm chạy tổng hợp; argparse mặc định so sánh TT1/TT2/TT3.
# Các main_ttN.py chuyển đến cùng CLI nhưng cố định một thuật toán.
# Chỉ khởi chạy khi file được Run trực tiếp, không chạy khi được import.
# Exit code từ run_cli được chuyển cho terminal hoặc IDE bằng SystemExit.
# Train/test và các artifacts được xử lý trong app.pipeline.
if __name__ == "__main__":
    raise SystemExit(run_cli())
