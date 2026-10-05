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
    The quadratic uses stable roots to reduce cancellation.
    Input: mu_sil/std_sil và mu_sp/std_sp là mean/std của silence/speech.
    Output: dict threshold và nghiệm/diagnostic; sigma nhỏ bị floor rõ ràng,
    không có nghiệm giữa hai mean dùng midpoint đã công bố.
    """
    floor = 1e-9
    # Chặn sigma rất nhỏ trên thang STE chuẩn hóa để tránh chia cho 0.
    sigma_sil, sigma_sp = max(std_sil, floor), max(std_sp, floor)
    lower, upper = min(mu_sil, mu_sp), max(mu_sil, mu_sp)
    # Nhân phương trình log-density với 2*sigma_sil²*sigma_sp² để tránh
    # hệ số chứa phép chia phương sai nhỏ, rồi giải phương trình bậc hai.
    vs, vp = sigma_sil * sigma_sil, sigma_sp * sigma_sp
    a = vp - vs
    b = 2.0 * (mu_sp * vs - mu_sil * vp)
    c = mu_sil * mu_sil * vp - mu_sp * mu_sp * vs + 2.0 * vs * vp * math.log(sigma_sil / sigma_sp)
    roots = []
    equal_variance = abs(vp - vs) <= 1e-12 * max(vs, vp)
    # Hai phương sai bằng nhau cho nghiệm midpoint; trường hợp khác dùng q
    # và c/q để giảm mất chữ số do trừ hai số gần bằng nhau.
    if equal_variance:
        if mu_sil != mu_sp:
            roots.append((mu_sil + mu_sp) / 2.0)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root_disc = math.sqrt(discriminant)
            q = -0.5 * (b + (root_disc if b >= 0.0 else -root_disc))
            if q != 0.0:
                roots.extend([q / a, c / q])
            else:
                roots.append(-b / (2.0 * a))
    candidates = [value for value in roots if lower <= value <= upper and math.isfinite(value)]
    # Chỉ nhận nghiệm thực trong khoảng giữa hai mean; ghi rõ mọi fallback.
    if candidates:
        # Nếu có hai nghiệm trong miền, chọn nghiệm gần midpoint và lưu cả hai.
        midpoint = (lower + upper) / 2.0
        threshold = min(candidates, key=lambda value: abs(value - midpoint))
        rule = "equal_density_between_means"
    else:
        threshold = (lower + upper) / 2.0
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
