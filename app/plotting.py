"""Vẽ kết quả/đặc trưng và bố trí bốn cửa sổ demo Matplotlib."""
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from app.config import ALGORITHM_TITLES


def _plot_axis(axis, record, result):
    """Vẽ waveform, normalized STE và biên chuẩn/dự đoán trên một hàng.

    Đầu vào: axis là trục Matplotlib; record chứa waveform/features; result chứa metrics và biên.
    Đầu ra: None; cập nhật axis và tạo trục STE phụ, không thay đổi dữ liệu tính toán.
    """
    # Waveform được thưa mẫu chỉ ở bước vẽ để cửa sổ mở nhanh.
    # STE dùng tâm khung và trục phụ vì khác đơn vị với biên độ.
    # Biên chuẩn đỏ; biên dự đoán xanh chỉ lấy từ FINAL speech regions.
    # Tiêu đề dùng MAE các cặp START/END vùng đã ghép trong pipeline.
    # Nhãn legend được dựng riêng để vẫn đủ khi không phát hiện speech.
    # Thưa mẫu chỉ để hiển thị nhanh; phép tính vẫn dùng toàn bộ samples.
    samples, fs = record["samples"], record["sample_rate"]
    stride = max(1, len(samples) // 12000)
    axis.plot([i / fs for i in range(0, len(samples), stride)], samples[::stride],
              color="#555555", linewidth=.55, alpha=.4)
    energy_axis = axis.twinx()
    energy_axis.plot(record["features"]["centers"], record["features"]["ste_norm"],
                     color="#dd8b12", linewidth=1.6, alpha=.95)
    # These are the exact thresholds used by detection, not plot-only lines.
    low=result['diagnostic']['low_ste_threshold']
    high=result['diagnostic']['high_ste_threshold']
    energy_axis.axhline(high,color='#8E44AD',linestyle='-.',linewidth=1.2)
    energy_axis.axhline(low,color='#148F77',linestyle=':',linewidth=1.2)
    energy_axis.set_ylim(-.05, 1.1)
    energy_axis.set_ylabel("Normalized STE", color="#a2670a", fontsize=9)
    axis.set_ylabel("Amplitude", fontsize=9)
    axis.set_xlim(0, record["duration"])
    axis.grid(alpha=.18)

    # Candidate/debug points không được vẽ hoặc đưa vào metric chính.
    for boundary in result["ground_truth_boundaries"]:
        axis.axvline(boundary, color="#d62728", linewidth=1.45, alpha=.85)
    for boundary in result["predicted_boundaries"]:
        axis.axvline(boundary, color="#1766bd", linestyle="--", linewidth=1.35)
    mae = result["metrics"]["mae_ms"]
    suffix = "Outer boundary MAE: N/A" if mae is None else f"Outer boundary MAE: {mae:.2f} ms"
    axis.set_title(f"{record['name']} | {suffix} | {len(result['final_regions'])} speech region(s)", fontsize=10)
    # Legend đủ nhãn kể cả không tìm được speech hoặc biên nội bộ.
    axis.legend(handles=[Line2D([], [], color="#555555", label="Waveform"),
                         Line2D([], [], color="#dd8b12", label="Normalized STE"),
                         Line2D([], [], color='#8E44AD',linestyle='-.',label=f'High STE: {high:.5g}'),
                         Line2D([], [], color='#148F77',linestyle=':',label=f'Low STE: {low:.5g}'),
                         Line2D([], [], color="#d62728", label="Ground truth"),
                         Line2D([], [], color="#1766bd", linestyle="--", label="Prediction")],
                fontsize=8, ncol=3, loc="upper right")


def make_file_figure(name, record_results, save_path):
    """Tạo và lưu một figure cho WAV với một hàng mỗi thuật toán.

    Đầu vào: name là tên WAV; record_results chứa cặp (record, result); save_path là đường dẫn PNG.
    Đầu ra: Figure Matplotlib để bố trí/hiển thị; ảnh PNG được ghi vào save_path.
    """
    # Mỗi cặp record/result được đặt lên một hàng độc lập.
    # Trục thời gian dùng chung để dễ so sánh biên giữa các thuật toán.
    # Số hàng quyết định chiều cao ảnh, tránh dồn ba kết quả vào một trục.
    # Lưu ảnh trước khi đưa Figure cho phần bố trí cửa sổ.
    # Đường dẫn theo stem WAV có thể bị cập nhật khi chạy --file cùng tên.
    rows = len(record_results)
    fig, axes = plt.subplots(rows, 1, figsize=(12, max(4.5, 3.5 * rows)),
                             squeeze=False, sharex=True)
    for axis_row, (record, result) in zip(axes, record_results):
        _plot_axis(axis_row[0], record, result)
    axes[-1][0].set_xlabel("Time (s)")
    fig.suptitle(name, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(save_path, dpi=200)
    manager = fig.canvas.manager
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(f"XLTHS - {name}")
    return fig


def arrange_four_windows(figures):
    """Bố trí tối đa bốn cửa sổ demo tại bốn góc màn hình.

    Đầu vào: figures là list Figure theo thứ tự phone_F2/M2 rồi studio_F2/M2.
    Đầu ra: List dict vị trí/kích thước cửa sổ đã bố trí; backend thiếu window được bỏ qua.
    """
    # Chỉ bố trí bốn Figure đầu tương ứng thứ tự batch test.
    # Backend Agg không có cửa sổ nên không tạo vị trí giả.
    # Tk lấy kích thước màn hình, Qt lấy vùng màn hình khả dụng.
    # Chỉ cập nhật geometry của cửa sổ, không chỉnh dữ liệu của Figure.
    # Vị trí đã đặt được trả về để pipeline ghi diagnostic demo.
    positions = []
    for index, fig in enumerate(figures[:4]):
        window = getattr(fig.canvas.manager, "window", None)
        if window is None:
            continue
        # Python Windows thường có TkAgg; hỗ trợ cả Qt nếu dùng backend đó.
        if hasattr(window, "winfo_screenwidth"):
            window.update_idletasks()
            width = window.winfo_screenwidth() // 2
            height = max(300, (window.winfo_screenheight() - 70) // 2)
            x, y = index % 2 * width, index // 2 * height
            window.wm_geometry(f"{width}x{height}+{x}+{y}")
            positions.append(dict(index=index, x=x, y=y, width=width, height=height))
        elif hasattr(window, "setGeometry"):
            geometry = window.screen().availableGeometry()
            width, height = geometry.width() // 2, geometry.height() // 2
            x, y = geometry.x() + index % 2 * width, geometry.y() + index // 2 * height
            window.setGeometry(x, y, width, height)
            positions.append(dict(index=index, x=x, y=y, width=width, height=height))
    return positions


def show_figures(figures, auto_close_seconds=None):
    """Hiển thị các figure và có thể tự đóng phục vụ demo thử.

    Đầu vào: figures là list Figure; auto_close_seconds là None hoặc thời gian tự đóng theo giây.
    Đầu ra: List vị trí từ arrange_four_windows; plt.show() chờ đến khi cửa sổ đóng.
    """
    # Bố trí cửa sổ trước khi bắt đầu vòng sự kiện GUI.
    # Timer chỉ dùng khi có thời gian tự đóng và ít nhất một Figure.
    # Giữ tham chiếu timer đến hết plt.show() để callback còn hiệu lực.
    # Callback đóng toàn bộ cửa sổ của lượt demo khi hết thời gian.
    # Không có timer thì người dùng đóng cửa sổ để kết thúc show().
    positions = arrange_four_windows(figures)
    timer = None
    if auto_close_seconds is not None and figures:
        timer = figures[0].canvas.new_timer(interval=max(1, int(auto_close_seconds * 1000)))
        timer.add_callback(lambda: plt.close("all"))
        timer.single_shot = True
        timer.start()
    plt.show()
    return positions


def plot_gaussian_training(values_by_class, model, output_path):
    """Minh họa phân bố STE train và hai mật độ Gaussian đã học.

    Đầu vào: values_by_class ánh xạ 0/1 sang STE; model có mean/std/threshold; output_path là PNG.
    Đầu ra: None; lưu PNG rồi đóng figure. Histogram/mật độ phục vụ minh họa, không học ngưỡng.
    """
    # Histogram minh họa được đếm bằng vòng lặp trên STE của từng lớp.
    # Chia số đếm cho số quan sát và độ rộng bin để có mật độ.
    # Đường Gaussian sử dụng mean/std đã học, không điều chỉnh ngưỡng.
    # Sàn std ở bước vẽ tránh chia 0 khi minh họa model suy biến.
    # Trục log giúp quan sát hai lớp có mức mật độ rất khác nhau.
    fig, axis = plt.subplots(figsize=(10, 5))
    bins, width = 100, .01
    for label, color in ((0, "#1766bd"), (1, "#d98b12")):
        values, counts = values_by_class[label], [0] * bins
        for value in values:
            counts[min(bins - 1, max(0, int(value / width)))] += 1
        density = [count / max(1, len(values)) / width for count in counts]
        axis.step([(i + .5) * width for i in range(bins)], density, color=color,
                  alpha=.5, label="Observed silence" if label == 0 else "Observed speech")
    x_values = [i / 2000 for i in range(2001)]
    for mean, std, color, label in ((model["muSil"], model["stdSil"], "#1766bd", "Gaussian silence"),
                                   (model["muSp"], model["stdSp"], "#d98b12", "Gaussian speech")):
        safe_std = max(std, 1e-12)
        density = [math.exp(-.5 * ((x - mean) / safe_std) ** 2) / (safe_std * math.sqrt(2 * math.pi)) for x in x_values]
        axis.plot(x_values, density, color=color, label=label)
    axis.axvline(model["threshold"], color="#d62728", linestyle="--", label=f"Threshold={model['threshold']:.6g}")
    axis.set_yscale("log")
    axis.set_ylim(1e-3, 1e4)
    axis.set_xlabel("Normalized STE")
    axis.set_ylabel("Density (log scale)")
    axis.set_title("Training distributions: Gaussian is a modeling assumption")
    axis.legend(fontsize=9)
    axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
