"""Đường dẫn độc lập với thư mục terminal và cấu hình thí nghiệm."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FRAME_MS = 25.0
HOP_MS = 10.0
# round() chọn mẫu gần nhất; tie .5 dùng số chẵn (44100 Hz -> 1102 mẫu).
SAMPLE_ROUNDING = "nearest_integer_samples_python_round_ties_to_even"
MIN_SILENCE_SECONDS = 0.200
# Engineering minimum for isolated speech/noise bursts, configurable globally.
MIN_SPEECH_SECONDS = 0.100
# Prefer W=20 when TRAIN FINAL-region calibration scores tie.
# This is a tie preference only; a better calibration score can select another W.
HISTOGRAM_W_TIE_PREFERENCE = 20.0
BOUNDARY_TOLERANCE_SECONDS = 0.100
NOISE_LEVELS_DB = (20, 10, 0)
RANDOM_SEED = 20261001
ALGORITHM_TITLES = {
    "tt1": "TT1 - Hodgkinson: binary search",
    "tt2": "TT2 - Histogram: energy only",
    "tt3": "TT3 - Gaussian distributions",
    "tt2-context": "Histogram STE - context report variant",
}
FILE_ORDER = ("phone_F2", "phone_M2", "studio_F2", "studio_M2")
