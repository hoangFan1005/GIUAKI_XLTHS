"""CLI dùng chung cho main tổng hợp và ba main cố định từng thuật toán."""
import argparse
import sys


def parse_args(argv=None, fixed_algorithm=None):
    """Đọc và kiểm tra lựa chọn chạy cho main tổng hợp hoặc main riêng.

    Đầu vào: argv là list cờ hoặc None để đọc terminal; fixed_algorithm là None/tt1/tt2/tt3.
    Đầu ra: argparse.Namespace chứa các cờ hợp lệ; argparse báo lỗi cờ với exit code 2.
    """
    # Main riêng cố định thuật toán để sinh viên chạy đúng phần phụ trách.
    # Main tổng hợp cho phép chọn ba thuật toán hoặc biến thể ngữ cảnh.
    # Các cờ được gom vào một Namespace trước khi đọc bất kỳ WAV nào.
    # Benchmark cần đủ bốn test; --file chỉ chọn một bản ghi để đánh giá.
    # Argparse xử lý cờ sai ngay tại CLI để tránh sinh bộ kết quả không hợp lệ.
    if fixed_algorithm not in (None, "tt1", "tt2", "tt3"):
        raise ValueError("Main riêng phải là tt1, tt2 hoặc tt3")
    parser = argparse.ArgumentParser(description="Chạy endpoint detection; học ngưỡng/noise và chọn TT2 W bằng FINAL MAE trên TRAIN, sau đó chấm điểm TEST đã dùng trong lịch sử.")
    if fixed_algorithm is None:
        parser.add_argument("--algorithm", choices=("all", "tt1", "tt2", "tt3", "tt2-context"), default="all",
                            help="all so sánh ba thuật toán; mỗi tt chạy thuật toán tương ứng.")
    else:
        parser.set_defaults(algorithm=fixed_algorithm)
    parser.add_argument("--file", metavar="WAV", default=None,
                        help="Tên WAV trong data/test hoặc data/train, hoặc đường dẫn WAV có LAB cùng tên.")
    parser.add_argument("--no-show", action="store_true", help="Chỉ xuất ảnh/CSV; không mở cửa sổ.")
    parser.add_argument('--evaluate-all',action='store_true',help='Chạy cả 8 WAV train+test, headless. Mọi tham số và TT2 W chọn trên TRAIN; chấm điểm TRAIN/TEST bằng model đã khóa.')
    parser.add_argument("--snr-study", action="store_true", help="Thêm khảo sát nhiễu tổng hợp 20/10/0 dB.")
    # Benchmark ngữ cảnh cần đủ 4 test, còn main riêng chỉ chạy thuật toán được giao.
    if fixed_algorithm is None:
        parser.add_argument("--compare-context", action="store_true",
                            help="Xuất thêm biến thể histogram STE và kiểm benchmark 4 WAV test.")
    else:
        parser.set_defaults(compare_context=False)
    parser.add_argument("--show-seconds", type=float, default=None,
                        help="Tự đóng cửa sổ sau số giây chỉ định khi kiểm tra demo.")
    args = parser.parse_args(argv)
    if args.evaluate_all and (args.file or args.compare_context):
        parser.error('--evaluate-all không kết hợp --file hoặc --compare-context.')
    if args.evaluate_all:args.no_show=True
    if args.file and args.compare_context:
        parser.error("--compare-context cần đủ 4 WAV test; hãy bỏ --file.")
    if args.show_seconds is not None and args.show_seconds <= 0:
        parser.error("--show-seconds phải lớn hơn 0.")
    return args


def main(argv=None, fixed_algorithm=None):
    """Chọn backend Matplotlib rồi điều phối toàn bộ thực nghiệm.

    Đầu vào: argv là list cờ hoặc None; fixed_algorithm xác định main riêng nếu có.
    Đầu ra: None; tác dụng phụ là fit model, ghi JSON/CSV/PNG và mở cửa sổ khi được yêu cầu.
    """
    # Backend phải được chọn trước khi import pipeline có pyplot.
    # Agg chỉ ghi ảnh cho chế độ --no-show; TkAgg tạo cửa sổ demo.
    # Mọi main dùng chung pipeline để train, timing và metric nhất quán.
    # Namespace đã kiểm tra được truyền nguyên vào run_experiment.
    # Hàm điều phối tạo artifacts; không trả model hay prediction trực tiếp.
    args = parse_args(argv, fixed_algorithm)
    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    from app.pipeline import run_experiment
    run_experiment(args)


def run_cli(argv=None, fixed_algorithm=None):
    """Đổi lỗi dữ liệu/phụ thuộc thành thông báo và mã kết thúc dễ hiểu.

    Đầu vào: argv và fixed_algorithm có cùng ý nghĩa như main().
    Đầu ra: Số nguyên 0 khi hoàn tất, 1 khi gặp lỗi được bắt; lỗi argparse vẫn trả 2.
    """
    # Chạy toàn bộ thực nghiệm trong khối bắt lỗi dự kiến.
    # Lỗi file, dữ liệu hoặc import được in ra stderr cho terminal/IDE.
    # Trả 1 giúp bên gọi nhận biết chạy thất bại.
    # Lỗi argparse tự kết thúc bằng SystemExit với mã 2.
    # Các lỗi lập trình khác được giữ traceback để tìm nguyên nhân.
    try:
        main(argv, fixed_algorithm)
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 1
    return 0
