"""Assignment TT3: population Gaussian statistics on labeled normalized STE."""
from __future__ import annotations

import math


def population_statistics(values: list[float]) -> tuple[float, float]:
    """Tự tính mean và độ lệch chuẩn population (phương sai chia N).

    Input: values là chuỗi quan sát không rỗng. Output: tuple (mean, std)
    cùng đơn vị với values; báo lỗi khi không có quan sát.
    """
    if not values:
        raise ValueError("Gaussian statistics need at least one observation")
    # Ước lượng population mean; không gọi mean/std từ thư viện thống kê.
    total = 0.0
    for value in values:
        total += value
    mean = total / len(values)
    # Phương sai population chia N; đây không phải sample variance chia N-1.
    squared = 0.0
    for value in values:
        squared += (value - mean) * (value - mean)
    return mean, math.sqrt(squared / len(values))


def equal_density_threshold(mu_sil: float, std_sil: float, mu_sp: float, std_sp: float) -> dict:
    """Solve p_sil(T)=p_sp(T); choose a real crossing between the class means.

    Tiny standard deviations are floored to 1e-9 on normalized STE. If no
    crossing lies between the means, their midpoint is an explicit fallback.
    The quadratic is centered/scaled before solving; rounded roots must also
    satisfy log-density equality within arithmetic and output-rounding error.
    Input: mu_sil/std_sil và mu_sp/std_sp là mean/std của silence/speech.
    Output: dict threshold và nghiệm/diagnostic; sigma nhỏ bị floor rõ ràng,
    không có nghiệm giữa hai mean dùng midpoint đã công bố.
    """
    floor = 1e-9
    # Chặn sigma rất nhỏ trên thang STE chuẩn hóa để tránh chia cho 0.
    sigma_sil, sigma_sp = max(std_sil, floor), max(std_sp, floor)
    lower, upper = min(mu_sil, mu_sp), max(mu_sil, mu_sp)
    midpoint = lower + (upper - lower) / 2.0
    # Input: means/sigmas theo STE. Output: hệ số theo x=(T-lower)/scale;
    # một mean bằng 0 nên không trừ các bình phương mean lớn gần bằng nhau.
    scale = max(upper - lower, sigma_sil, sigma_sp)
    ms, mp = (mu_sil - lower) / scale, (mu_sp - lower) / scale
    ss, sp = sigma_sil / scale, sigma_sp / scale
    vs, vp = ss * ss, sp * sp
    sigma_difference = sigma_sil - sigma_sp
    # log1p giữ chính xác khi hai sigma gần bằng nhau. Trên normalized STE
    # với sigma floor, ratio hữu hạn; log(ratio) tránh trừ hai log lớn gần nhau.
    if abs(sigma_difference) <= 0.5 * sigma_sp:
        log_ratio = math.log1p(sigma_difference / sigma_sp)
    else:
        log_ratio = math.log(sigma_sil / sigma_sp)
    a = ((sigma_sp - sigma_sil) / scale) * (sp + ss)
    b = 2.0 * (mp * vs - ms * vp)
    c = math.fsum([ms * ms * vp, -mp * mp * vs, 2.0 * vs * vp * log_ratio])
    roots = []
    # Hai phương sai bằng nhau cho nghiệm midpoint; trường hợp khác dùng q
    # và c/q để giảm mất chữ số do trừ hai số gần bằng nhau.
    if sigma_sil == sigma_sp:
        if mu_sil != mu_sp:
            roots.append(midpoint)
    else:
        # D=4*vs*vp*((mp-ms)^2-2*a*log_ratio). Hai hạng trong ngoặc
        # không triệt tiêu vì a và log_ratio trái dấu; tránh b²-4ac.
        root_disc = 2.0 * ss * sp * math.sqrt((mp - ms) ** 2 - 2.0 * a * log_ratio)
        q = -0.5 * (b + math.copysign(root_disc, b))
        if q != 0.0:
            roots.extend([lower + scale * (q / a), lower + scale * (c / q)])
        else:
            roots.append(lower + scale * (-b / (2.0 * a)))

    def matches_log_density(value: float) -> bool:
        """Input: rounded STE root. Output: equality within its rounding bound."""
        zs, zp = (value - mu_sil) / sigma_sil, (value - mu_sp) / sigma_sp
        sil_term, sp_term = 0.5 * zs * zs, 0.5 * zp * zp
        gap = math.fsum([-log_ratio, -sil_term, sp_term])
        # Sai số số học phụ thuộc độ lớn log-density, không dùng epsilon STE.
        arithmetic_error = 32.0 * math.ulp(1.0) * (1.0 + abs(log_ratio) + sil_term + sp_term)
        # Một ulp T bao phủ làm tròn đổi tọa độ về float STE. Taylor bound
        # giữ residual ~1e-7 hợp lệ khi sigma=1e-9 và T gần .5.
        rounding_step = math.ulp(value)
        ds, dp = rounding_step / sigma_sil, rounding_step / sigma_sp
        rounding_error = abs(zs) * ds + abs(zp) * dp + 0.5 * (ds * ds + dp * dp)
        return abs(gap) <= arithmetic_error + rounding_error

    candidates = [value for value in roots if math.isfinite(value)
                  and lower <= value <= upper and matches_log_density(value)]
    # Chỉ nhận nghiệm giữa hai mean đã kiểm tra log-density; fallback rõ ràng.
    if candidates:
        # Nếu có hai nghiệm trong miền, chọn nghiệm gần midpoint và lưu cả hai.
        threshold = min(candidates, key=lambda value: abs(value - midpoint))
        rule = "equal_density_between_means"
    else:
        threshold = midpoint
        rule = "no_between_means_crossing_midpoint"
    return {"threshold": threshold, "crossings": roots, "threshold_rule": rule,
            "sigma_floor": floor, "sigma_sil_effective": sigma_sil,
            "sigma_sp_effective": sigma_sp,
            "sigma_was_floored": std_sil < floor or std_sp < floor}


def fit(records: list[dict]) -> dict:
    """Học hai phân bố Gaussian từ khung train có nhãn LAB.

    Input: records chứa features.ste_norm và labels 0/1/None.
    Output: dict mean/std hai lớp, ngưỡng equal-density và hướng speech;
    loại nhãn None trước thống kê, không sử dụng test.
    """
    silence, speech = [], []
    # Nhãn sil=0, v/uv=1 do core cung cấp; None không tham gia thống kê.
    for record in records:
        values, labels = record["features"]["ste_norm"], record["labels"]
        if len(values) != len(labels):
            raise ValueError("Training features and LAB labels have unequal lengths")
        for value, label in zip(values, labels):
            if label == 0:
                silence.append(value)
            elif label == 1:
                speech.append(value)
    mu_sil, std_sil = population_statistics(silence)
    mu_sp, std_sp = population_statistics(speech)
    params = equal_density_threshold(mu_sil, std_sil, mu_sp, std_sp)
    # Lưu sigma gốc và sigma hiệu dụng riêng để không che việc chặn sigma.
    params.update({"muSil": mu_sil, "stdSil": std_sil, "muSp": mu_sp,
                   "stdSp": std_sp, "silence_count": len(silence),
                   "speech_count": len(speech), "feature": "ste_norm",
                   "normalization": "per_wav_max", "speech_direction": "high" if mu_sp >= mu_sil else "low",
                   "source": "Assignment TT3: population mean/std and equal Gaussian densities"})
    return params


def predict(features: dict, params: dict) -> list[int]:
    """Áp dụng ngưỡng theo hướng mean speech đã học trước cleanup chung.

    Input: features.ste_norm là chuỗi STE, params có threshold và
    speech_direction. Output: list 0/1 cùng độ dài; không đọc LAB.
    """
    # Thông thường speech có mean cao hơn; hỗ trợ dữ liệu đảo thứ tự mean.
    if params.get("speech_direction", "high") == "low":
        return [1 if value <= params["threshold"] else 0 for value in features["ste_norm"]]
    return [1 if value >= params["threshold"] else 0 for value in features["ste_norm"]]
