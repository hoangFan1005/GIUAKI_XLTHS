"""Sinh viên TT3: thống kê phân phối chuẩn và ngưỡng Gaussian.

Run: python main_tt3.py
Một file: python main_tt3.py --file phone_F2.wav
Code thuật toán: algorithms/tt3_gaussian.py.
"""
from app.cli import run_cli


def main(argv=None):
    """Điểm chạy riêng cố định thuật toán TT3.

    Đầu vào: argv là list cờ hoặc None để argparse đọc tham số terminal.
    Đầu ra: Exit code 0/1 từ run_cli; mặc định chạy bốn test, --file chọn một WAV có LAB.
    """
    # Chuyển cờ CLI đến bộ chạy dùng chung của dự án.
    # Cố định TT3 để Run trong IDE luôn chọn đúng thuật toán.
    return run_cli(argv, fixed_algorithm="tt3")


if __name__ == "__main__":
    raise SystemExit(main())
