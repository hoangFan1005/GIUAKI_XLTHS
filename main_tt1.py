"""Sinh viên TT1: binary search Hodgkinson.

Run: python main_tt1.py
Một file: python main_tt1.py --file phone_F2.wav
Code thuật toán: algorithms/tt1_hodgkinson.py.
"""
from app.cli import run_cli


def main(argv=None):
    """Điểm chạy riêng cố định thuật toán TT1.

    Đầu vào: argv là list cờ hoặc None để argparse đọc tham số terminal.
    Đầu ra: Exit code 0/1 từ run_cli; mặc định chạy bốn test, --file chọn một WAV có LAB.
    """
    # Chuyển cờ CLI đến bộ chạy dùng chung của dự án.
    # Cố định TT1 để Run trong IDE luôn chọn đúng thuật toán.
    return run_cli(argv, fixed_algorithm="tt1")


if __name__ == "__main__":
    raise SystemExit(main())
