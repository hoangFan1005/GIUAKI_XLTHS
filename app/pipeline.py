"""Học tham số bằng train, dự đoán test, lưu kết quả và điều phối demo."""
import csv
import hashlib
import json
import math
import random
import zlib
from pathlib import Path

import matplotlib.pyplot as plt

from algorithms import tt1_hodgkinson as tt1
from algorithms import tt2_histogram as tt2
from algorithms import tt3_gaussian as tt3
from app.config import (PROJECT_ROOT, TRAIN_DIR, TEST_DIR, OUTPUT_DIR, FILE_ORDER,
                        FRAME_MS, HOP_MS, SAMPLE_ROUNDING,
                        MIN_SILENCE_SECONDS, MIN_SPEECH_SECONDS, BOUNDARY_TOLERANCE_SECONDS,
                        HISTOGRAM_W_TIE_PREFERENCE,
                        NOISE_LEVELS_DB, RANDOM_SEED)
from app.plotting import make_file_figure, show_figures, plot_gaussian_training
from core.io_utils import read_wav, read_lab
from core.features import compute_features
from core.metrics import (frame_labels, frame_metrics, endpoint_metrics,
                          ground_truth_speech_envelope, ground_truth_boundaries,
                          boundary_metrics, ground_truth_regions, region_endpoint_metrics)
from core.endpoints import fit_noise_floor, endpoint_thresholds, hysteresis_regions, regions_to_mask
from app.weight_selection import select_final_weight, sweep_final_weights
from core.postprocess import (fill_short_internal_silences, speech_envelope,
                              boundaries, mask_segments)


def write_csv(path, rows):
    """Lưu CSV UTF-8 BOM để Excel đọc tiếng Việt và cột nhất quán.

    Input: path là Path đích; rows là list dict, thứ tự gặp key đặt thứ tự
    cột. Output: None; tạo thư mục cha và ghi header/dữ liệu vào path.
    """
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if len(fields) != len({key.casefold() for key in fields}):
        raise ValueError('CSV column headers must be unique ignoring case')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    """Lưu cấu hình hoặc diagnostic đã tính dưới dạng JSON UTF-8.

    Input: path là Path đích, value là dữ liệu serialize được. Output:
    None; tạo thư mục cha và ghi JSON thụt lề, giữ nguyên tiếng Việt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_audio_file(path):
    """Đọc một WAV mono và LAB cùng tên để chuẩn bị đánh giá.

    Input: path là tên/đường dẫn WAV. Output: dict name/wav_path, samples,
    sample_rate Hz, duration giây và intervals LAB; thiếu WAV/LAB báo lỗi.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy WAV: {path}")
    lab_path = path.with_suffix(".lab")
    if not lab_path.is_file():
        raise FileNotFoundError(f"Cần LAB cùng tên để đánh giá: {lab_path}")
    samples, sample_rate = read_wav(path)
    duration = len(samples) / sample_rate
    intervals = read_lab(lab_path, duration)
    split = ('train' if path.parent == TRAIN_DIR.resolve() else
             'test' if path.parent == TEST_DIR.resolve() else 'external')
    return dict(name=path.stem, split=split, wav_path=str(path), samples=samples,
                sample_rate=sample_rate, duration=duration, intervals=intervals)


def resolve_input_file(file_argument):
    """Tìm WAV trong data/test, data/train hoặc theo đường dẫn người dùng.

    Input: file_argument là stem, tên .wav hoặc đường dẫn tương đối/tuyệt
    đối. Output: Path tuyệt đối tồn tại; sai đuôi/không tìm thấy báo lỗi.
    """
    requested = Path(file_argument)
    if not requested.suffix:
        requested = requested.with_suffix(".wav")
    if requested.suffix.lower() != ".wav":
        raise ValueError("--file cần tên/đường dẫn WAV, ví dụ phone_F2.wav")
    if requested.is_absolute():
        candidates = [requested]
    elif len(requested.parts) == 1:
        candidates = [TEST_DIR / requested, TRAIN_DIR / requested,
                      Path.cwd() / requested, PROJECT_ROOT / requested]
    else:
        candidates = [Path.cwd() / requested, PROJECT_ROOT / requested]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Không tìm thấy WAV '{file_argument}'. Kiểm tra tên trong data/test, data/train hoặc dùng đường dẫn đầy đủ.")


def load_audio_folder(folder):
    """Đọc bốn cặp WAV/LAB cho batch thực nghiệm.

    Input: folder là Path thư mục chứa đúng bốn WAV. Output: list audio
    records từ load_audio_file, sắp theo tên; sai số WAV báo lỗi.
    """
    paths = sorted(folder.glob("*.wav"))
    if len(paths) != 4:
        raise FileNotFoundError(f"Cần đúng 4 WAV trong {folder}; hiện có {len(paths)}")
    return [load_audio_file(path) for path in paths]


def prepare_records(audio_records, source_histogram=False):
    """Tính đặc trưng 25/10 ms cho mọi bản ghi hiện hành.

    Input: audio_records chứa waveform/Fs/LAB. source_histogram giữ tương
    thích các điểm gọi cũ; mọi thuật toán hiện chỉ cần đặc trưng năng lượng.
    Output: bản sao records có features và nhãn tâm khung, không tính FFT/centroid.
    Metadata timing thực lấy từ compute_features, không ghi đè bằng nominal.
    """
    records = []
    for base in audio_records:
        features = compute_features(base["samples"], base["sample_rate"],
                                    FRAME_MS, HOP_MS)
        record = dict(base, features=features,
                      labels=frame_labels(features["centers"], base["intervals"]))
        records.append(record)
    return records


def algorithm_decision(algorithm: str, features: dict, params: dict) -> tuple[list[int], dict]:
    """Return native unpadded decisions and thresholds without LAB or endpoint policy.

    TT1/high TT3 include equality; low TT3 includes equality in the opposite
    direction. TT2 uses strict raw Energy > threshold (context uses STE).
    Existing TT2 padding diagnostics describe its candidate stage only.
    """
    if algorithm == 'tt1':
        mask, diagnostic = tt1.predict(features, params), dict(threshold=params['threshold'])
    elif algorithm == 'tt3':
        if params.get('speech_direction', 'high') not in ('high', 'low'):
            raise ValueError('Unknown TT3 speech_direction')
        mask, diagnostic = tt3.predict(features, params), dict(threshold=params['threshold'])
    elif algorithm in ('tt2', 'tt2-context'):
        mask, diagnostic = tt2.predict(features, params)
        if diagnostic['variant'] == 'source':
            mask = [int(e > diagnostic['energy_threshold']) for e in features['energy']]
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    source_histogram = algorithm in ('tt2', 'tt2-context') and diagnostic.get('variant') == 'source'
    diagnostic.update(native_threshold=diagnostic['energy_threshold'] if algorithm in ('tt2', 'tt2-context') else params['threshold'],
                      native_threshold_units='sum of squared samples' if source_histogram else 'normalized STE')
    return mask, diagnostic


def detect_regions(algorithm, features, duration, params):
    """Detect FINAL speech from features and a fitted model, without LAB or I/O.

    Input: algorithm, frame features/supports, duration and frozen TRAIN model.
    Output: mask/final_regions/diagnostic and derived predicted_boundaries.
    Core merges native support gaps below 200 ms without a duration filter.
    Enhanced requires TRAIN noise and speech-high Gaussian direction.
    """
    mode = params.get('endpoint_mode', 'enhanced')
    if mode not in ('core', 'enhanced'):
        raise ValueError(f'Unknown endpoint_mode: {mode}')
    noise = params.get('endpoint_noise') if mode == 'enhanced' else None
    if mode == 'enhanced' and noise is None:
        raise ValueError('Fitted model requires endpoint_noise; call fit_training_model on TRAIN before inference')
    # The raw Gaussian classifier supports both directions. FINAL HIGH/LOW
    # requires speech-high, so reject an incompatible imported model early.
    if mode == 'enhanced' and algorithm == 'tt3' and params.get('speech_direction', 'high') != 'high':
        raise ValueError("FINAL endpoint detection requires TT3 speech_direction='high'; "
                         "inverted/unknown directions are not supported by energy hysteresis")
    seed, diagnostic = algorithm_decision(algorithm, features, params)
    source_histogram=algorithm=='tt2' and params.get('variant','source')=='source'
    mask = tt2.pad_speech(seed, diagnostic['padding_frames']) if mode == 'enhanced' and source_histogram else seed

    # Algorithm masks are candidates. Final endpoints require HIGH confirmation,
    # LOW continuation, short-gap merging and duration filtering.
    candidate_regions=[(s,e) for s,e,state in mask_segments(mask,features['starts'],duration,features['ends']) if state]
    if mode == 'core':
        final_regions = []
        for start, end in candidate_regions:
            if final_regions and start - final_regions[-1][1] < MIN_SILENCE_SECONDS - 1e-12:
                final_regions[-1] = (final_regions[-1][0], end)
            else:
                final_regions.append((start, end))
        mask = regions_to_mask(final_regions, features['starts'], features['ends'])
        diagnostic.update(candidate_regions=candidate_regions, final_regions=final_regions,
            low_ste_threshold=None, high_ste_threshold=None, endpoint_noise=None,
            minimum_speech_ms=0., minimum_silence_ms=MIN_SILENCE_SECONDS*1000,
            endpoint_mode=mode, geometry='union of active frame supports',
            final_padding_ms=0., padding_stage='excluded from core FINAL',
            endpoint_policy='native decisions; merge internal support gaps below 200 ms; no duration filter or final padding')
        return dict(mask=mask, final_regions=final_regions, diagnostic=diagnostic,
                    predicted_boundaries=[b for region in final_regions for b in region])
    if algorithm in ('tt1','tt3'):
        base_threshold=params['threshold']
    else:
        peak=max(features['energy'],default=0.)
        base_threshold=diagnostic['energy_threshold']/peak if source_histogram and peak>0 else diagnostic['energy_threshold'] if not source_histogram else 0.
    low,high=endpoint_thresholds(base_threshold,noise,histogram=algorithm in ('tt2','tt2-context'))
    final_regions=hysteresis_regions(features['ste_norm'],features['starts'],features['ends'],duration,
        low,high,MIN_SILENCE_SECONDS,MIN_SPEECH_SECONDS,seed_mask=seed)
    mask=regions_to_mask(final_regions,features['starts'],features['ends'])
    diagnostic.update(candidate_regions=candidate_regions,final_regions=final_regions,
        endpoint_mode=mode,geometry='union of active frame supports',final_padding_ms=0.,
        padding_stage='candidate diagnostics only' if source_histogram else 'none',
        low_ste_threshold=low,high_ste_threshold=high,endpoint_noise=noise,
        minimum_speech_ms=MIN_SPEECH_SECONDS*1000,minimum_silence_ms=MIN_SILENCE_SECONDS*1000,
        endpoint_policy='raw STE hysteresis; candidates/debug excluded; no fixed final padding')
    return dict(mask=mask, final_regions=final_regions, diagnostic=diagnostic,
                predicted_boundaries=[b for region in final_regions for b in region])


def predict_and_score(algorithm, record, params):
    """Invoke LAB-free detection, then score complete regions and frames.

    Input: algorithm, prepared annotated record, fitted TRAIN parameters.
    Output: detection plus reference regions, metrics and scoring status.
    LAB and filename enter only after detect_regions has returned.
    """
    detected = detect_regions(algorithm, record['features'], record['duration'], params)
    mask, final_regions, diagnostic = detected['mask'], detected['final_regions'], detected['diagnostic']
    expected_regions=ground_truth_regions(record['intervals'])
    predicted_boundaries=detected['predicted_boundaries']
    expected_boundaries=[b for region in expected_regions for b in region]
    # Match complete regions in time order. Missing/extra regions make primary
    # MAE undefined; matched-only MAE remains separate to expose the failure.
    region_scores=region_endpoint_metrics(expected_regions,final_regions)
    scores={k:v for k,v in region_scores.items() if k not in ('region_pairs','signed_endpoint_errors_ms')}
    scores.update(endpoint_error_count=len(region_scores['signed_endpoint_errors_ms']),
                  endpoint_squared_error_ms2=sum(v*v for v in region_scores['signed_endpoint_errors_ms']))
    predicted=(final_regions[0][0],final_regions[-1][1]) if final_regions else None
    reference=(expected_regions[0][0],expected_regions[-1][1]) if expected_regions else None
    flags=[]
    if scores['extra_region_count']:flags.append('extra predicted regions')
    if scores['missing_region_count']:flags.append('missing predicted regions')
    if scores['mae_ms'] is not None and scores['mae_ms']>BOUNDARY_TOLERANCE_SECONDS*1000:flags.append('suspiciously high MAE')
    inside=far_inside=0
    for boundary in predicted_boundaries:
        if boundary<0 or boundary>record['duration']+1e-12:flags.append('boundary outside audio range')
        if any(label=='sil' and start+1e-12<boundary<end-1e-12 for start,end,label in record['intervals']):
            inside+=1
            if expected_boundaries and min(abs(boundary-b) for b in expected_boundaries)>FRAME_MS/1000+1e-12:far_inside+=1
    if far_inside:flags.append('boundary inside silence')
    scores.update(status='; '.join(dict.fromkeys(flags)) if flags else 'OK' if expected_regions or final_regions else 'both_no_speech',
                  boundary_inside_silence_count=inside,boundary_inside_silence_far_count=far_inside)
    frames = frame_metrics(record["labels"], mask)
    events = boundary_metrics(expected_boundaries, predicted_boundaries,
                              BOUNDARY_TOLERANCE_SECONDS)
    scores.update({f"frame_{key}": value for key, value in frames.items()})
    scores.update({f"tolerance_boundary_{key}": value for key, value in events.items() if key != "pairs"})
    scores.update(gt_start_s=reference[0] if reference else None,
                  gt_end_s=reference[1] if reference else None,
                  predicted_start_s=predicted[0] if predicted else None,
                  predicted_end_s=predicted[1] if predicted else None)
    # Imported historical JSON remains usable without relabeling its TEST W
    # calibration as TRAIN. Fresh fit_training_model outputs never have this key.
    legacy_test_tuned = algorithm == 'tt2' and 'W_selection_test' in params
    scores.update(metrics_schema_version=2, model_schema_version=params.get('schema_version', 1),
                  parameter_selection_set='test' if legacy_test_tuned else params.get('parameter_selection_set', 'train'),
                  evaluation_protocol='test_tuned_not_independent' if legacy_test_tuned else params.get('evaluation_protocol', 'train_selected_reused_test'),
                  historical_test_exposure=True if legacy_test_tuned else params.get('historical_test_exposure', True))
    return dict(file=record["name"], algorithm=algorithm, mask=mask, metrics=scores,
                final_regions=final_regions, ground_truth_regions=expected_regions,region_pairs=region_scores['region_pairs'],
                predicted_boundaries=predicted_boundaries,
                ground_truth_boundaries=expected_boundaries, diagnostic=diagnostic)


def fit_training_model(algorithm, train_records, *, endpoint_mode='enhanced'):
    """Fit and lock parameters and FINAL W from supplied prepared TRAIN only.

    Input: prepared records with explicit split=train; array records need no
    path and support future folds. Known TEST/external paths are rejected.
    Output: (schema-v3 model, FINAL W sweep rows), empty rows for other methods.
    No files are loaded here, including during the histogram calibration.
    Enhanced TT3 must fit speech-high for FINAL energy hysteresis; core
    retains either native Gaussian decision direction.
    """
    if endpoint_mode not in ('core', 'enhanced'):
        raise ValueError(f'Unknown endpoint_mode: {endpoint_mode}')
    if not train_records:
        raise ValueError('fit_training_model requires TRAIN records')
    for record in train_records:
        if record.get('split') != 'train':
            raise ValueError('Training records require explicit TRAIN provenance (split=train)')
        if 'wav_path' in record and Path(record['wav_path']).resolve().parent != TRAIN_DIR.resolve():
            raise ValueError('Training record path must belong to TRAIN, never TEST/external')
    if algorithm == 'tt1':
        model = tt1.fit(train_records)
    elif algorithm == 'tt3':
        model = tt3.fit(train_records)
        # Do not lock a raw Gaussian model whose inequality contradicts HIGH.
        if endpoint_mode == 'enhanced' and model.get('speech_direction', 'high') != 'high':
            raise ValueError("FINAL endpoint detection requires TT3 speech_direction='high'; "
                             "TRAIN speech mean is below silence mean")
    elif algorithm in ('tt2', 'tt2-context'):
        model = tt2.fit(train_records, variant='context' if algorithm == 'tt2-context' else 'source')
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    model.update(schema_version=3, metrics_schema_version=2, endpoint_mode=endpoint_mode,
                 parameter_selection_set='train', evaluation_protocol='train_selected_reused_test',
                 historical_test_exposure=True, training_files=[record['name'] for record in train_records],
                 frame_ms=FRAME_MS, hop_ms=HOP_MS, sample_rounding=SAMPLE_ROUNDING,
                 minimum_internal_silence_ms=MIN_SILENCE_SECONDS*1000,
                 minimum_speech_ms=0. if endpoint_mode == 'core' else MIN_SPEECH_SECONDS*1000,
                 minimum_silence_ms=MIN_SILENCE_SECONDS*1000,
                 endpoint_noise=fit_noise_floor(train_records) if endpoint_mode == 'enhanced' else None,
                 endpoint_policy='native decisions; merge internal support gaps below 200 ms; no duration filter or final padding' if endpoint_mode == 'core' else 'hysteresis final regions; no fixed final padding',
                 feature_rule='Energy (sum of squared samples)' if algorithm == 'tt2' else 'normalized STE',
                 decision_rule='Energy > threshold' if algorithm == 'tt2' else 'STE <= threshold' if model.get('speech_direction') == 'low' else 'STE >= threshold',
                 padding_stage='excluded from core FINAL' if endpoint_mode == 'core' else 'candidate diagnostics only' if algorithm == 'tt2' else 'none',
                 boundary_convention='union of active frame supports')
    sweep_rows = []
    if algorithm == 'tt2':
        # Preserve the core's candidate-stage proposal under explicit names.
        # Its F1 is not a score for the W selected below by FINAL-region MAE.
        model['candidate_frame_selected_W'] = model['W']
        model['candidate_frame_f1'] = model.pop('train_frame_f1')
        model['candidate_frame_selection_scores'] = [
            dict(W=row['W'], candidate_frame_f1=row['train_frame_f1'],
                 **{key: value for key, value in row.items() if key not in ('W', 'train_frame_f1')})
            for row in model.pop('selection_scores')]
        model['candidate_cleanup_records'] = {
            key: model.pop(key) for key in ('full_pipeline_records', 'predict_only_fallback_records')}
        model['candidate_frame_selection_applicable'] = True
        selection, sweep_rows = sweep_final_weights(train_records, model, predict_and_score,
                                                    tie_preference_W=HISTOGRAM_W_TIE_PREFERENCE)
        model.update(W=selection['selected_W'], finalW=selection['selected_W'],
                     tie_preference_W=HISTOGRAM_W_TIE_PREFERENCE, W_selection_train=selection,
                     parameter_rule=selection['selection_rule'])
    elif algorithm == 'tt2-context':
        # Context W is declared by the report and has no candidate F1 fit.
        model.update(candidate_frame_selected_W=None, candidate_frame_f1=None,
                     candidate_frame_selection_scores=[], candidate_cleanup_records=None,
                     candidate_frame_selection_applicable=False, finalW=model['W'], tie_preference_W=None)
    # Hash the complete locked model before attaching the digest to itself or
    # its manifest. This includes numeric calibration and TRAIN provenance.
    digest = hashlib.sha256(json.dumps(model, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False).encode('utf-8')).hexdigest()
    model['calibration_digest'] = digest
    if algorithm == 'tt2':
        model['W_selection_train']['calibration_digest'] = digest
    return model, sweep_rows


def data_statistics(records, split):
    """Ước lượng SNR môi trường theo công suất speech/silence của LAB.

    Input: records chứa waveform/Fs/LAB, split là tên tập dữ liệu.
    Output: rows thời lượng/biên chuẩn/công suất và snr_proxy_db; SNR là
    proxy speech-plus-noise/silence, không phải SNR sạch đã biết.
    """
    rows = []
    for record in records:
        fs, samples = record["sample_rate"], record["samples"]
        power_sum, count = {0: 0.0, 1: 0.0}, {0: 0, 1: 0}
        for start, end, label in record["intervals"]:
            category = 0 if label == "sil" else 1
            # LAB endpoints đổi về chỉ số mẫu, tránh chọn sai mẫu do round float.
            first = max(0, round(start * fs))
            last = min(len(samples), round(end * fs))
            for index in range(first, last):
                power_sum[category] += samples[index] * samples[index]
            count[category] += last - first
        powers = {key: power_sum[key] / count[key] if count[key] else 0.0 for key in count}
        snr = 10 * math.log10(powers[1] / powers[0]) if powers[0] > 0 and powers[1] > 0 else None
        envelope = ground_truth_speech_envelope(record["intervals"])
        rows.append(dict(split=split, file=record["name"], sample_rate_hz=fs,
                         duration_s=record["duration"], gt_start_s=envelope[0] if envelope else None,
                         gt_end_s=envelope[1] if envelope else None,
                         silence_power=powers[0], speech_plus_noise_power=powers[1],
                         snr_proxy_db=snr))
    return rows


def summarize(metrics):
    """Tổng hợp metric từng thuật toán, phân biệt mean file và pooled RMSE.

    Input: metrics là rows endpoint/frame của từng file. Output: rows mean
    MAE/RMSE, pooled endpoint RMSE, F1 và số file có metric không xác định;
    metric không xác định được loại khỏi mean và được đếm rõ.
    """
    rows = []
    for algorithm in dict.fromkeys(row["algorithm"] for row in metrics):
        selected = [row for row in metrics if row["algorithm"] == algorithm]
        valid = [row for row in selected if row["mae_ms"] is not None]
        squared=sum(row.get('endpoint_squared_error_ms2',(row['rmse_ms']**2)*2) for row in valid)
        error_count=sum(row.get('endpoint_error_count',2) for row in valid)
        ordered=sorted(row['mae_ms'] for row in valid)
        median=(ordered[len(ordered)//2] if len(ordered)%2 else (ordered[len(ordered)//2-1]+ordered[len(ordered)//2])/2) if ordered else None
        count_correct=sum(row.get('ground_truth_region_count')==row.get('predicted_region_count') for row in selected)
        rows.append(dict(algorithm=algorithm, metrics_schema_version=2,
                         files=len(selected), evaluated_files=len(valid),
                         missing_speech_files=len(selected) - len(valid),
                         mean_file_mae_ms=sum(row["mae_ms"] for row in valid) / len(valid) if valid else None,
                         mean_file_rmse_ms=sum(row["rmse_ms"] for row in valid) / len(valid) if valid else None,
                         pooled_endpoint_rmse_ms=math.sqrt(squared/error_count) if error_count else None,
                         median_file_mae_ms=median,min_file_mae_ms=min(ordered) if ordered else None,max_file_mae_ms=max(ordered) if ordered else None,
                         region_count_correct_files=count_correct,region_count_incorrect_files=len(selected)-count_correct,
                         top_mae_files='; '.join(f"{r['file']}:{r['mae_ms']:.2f}ms" for r in sorted(valid,key=lambda r:r['mae_ms'],reverse=True)[:3]),
                         parameter_selection_set=selected[0].get('parameter_selection_set','train'),
                         evaluation_protocol=selected[0].get('evaluation_protocol','train_selected_reused_test'),
                         historical_test_exposure=selected[0].get('historical_test_exposure', True),
                         mean_frame_f1=sum(row["frame_f1"] for row in selected) / len(selected)))
    return rows


def noisy_copy(record, snr_db, seed):
    """Thêm nhiễu trắng Gaussian vào bản sao waveform ở SNR yêu cầu.

    Input: record có samples/Fs/LAB, snr_db là SNR thêm nhiễu, seed tạo
    RNG xác định. Output: audio record mới; LAB speech chỉ định mức sigma
    noise, features phải tính lại bằng prepare_records trước suy luận.
    """
    speech_power, speech_count = 0.0, 0
    fs = record["sample_rate"]
    for start, end, label in record["intervals"]:
        if label == "sil":
            continue
        first, last = round(start * fs), min(len(record["samples"]), round(end * fs))
        for index in range(first, last):
            speech_power += record["samples"][index] ** 2
        speech_count += last - first
    rms = math.sqrt(speech_power / speech_count) if speech_count else 0.0
    noise_sigma = rms / (10 ** (snr_db / 20))
    rng = random.Random(seed)
    samples = [sample + rng.gauss(0, noise_sigma) for sample in record["samples"]]
    return dict(record, samples=samples)


def context_benchmark(records):
    """Đối chiếu lịch sử báo cáo bằng features 20/10 ms tính lại từ waveform.

    Input: records có samples, sample_rate, intervals và name của bốn test.
    Output: rows ghi biên công bố/tính lại, metric và nhãn historical 20/10.
    Features hiện hành 25/10 trong records không bị sửa hoặc dùng đối chiếu.
    """
    expected = {
        "report_fixed_0.0025": [(1.02, 4.08), (.53, 2.52), (.76, 2.36), (.46, 1.93)],
        "report_gaussian_0.002864": [(1.02, 4.08), (.53, 2.52), (.76, 2.36), (.46, 1.93)],
        "report_histogram": [(1.11, 4.01), (.54, 2.51), (.77, 2.21), (.47, 1.89)],
    }
    historical = []
    for record in records:
        features = compute_features(record["samples"], record["sample_rate"], frame_ms=20.0, hop_ms=10.0)
        historical.append(dict(record, features=features))
    lookup = {record["name"]: record for record in historical}
    historical_histogram = tt2.fit([], variant="context")
    historical_histogram.update(frame_ms=20.0, hop_ms=10.0, historical_comparison=True,
                                parameter_rule="historical context report fixed 20/10 ms configuration")
    rows = []
    for method, published in expected.items():
        for name, target in zip(FILE_ORDER, published):
            record = lookup[name]
            if method == "report_histogram":
                mask, _ = tt2.predict(record["features"], historical_histogram)
            else:
                threshold = .0025 if method == "report_fixed_0.0025" else .002864
                mask = [int(value >= threshold) for value in record["features"]["ste_norm"]]
            mask = fill_short_internal_silences(mask, record["features"]["starts"], record["duration"],
                                               MIN_SILENCE_SECONDS, frame_ends=record["features"]["ends"])
            envelope = speech_envelope(mask, record["features"]["starts"], record["duration"], frame_ends=record["features"]["ends"])
            scores = endpoint_metrics(envelope, ground_truth_speech_envelope(record["intervals"]))
            rows.append(dict(method=method, file=name, published_start_s=target[0], published_end_s=target[1],
                             historical_comparison=True, frame_ms=20.0, hop_ms=10.0,
                             actual_frame_ms=record["features"]["frame_ms"],
                             actual_hop_ms=record["features"]["hop_ms"],
                             sample_rounding=SAMPLE_ROUNDING,
                             recomputed_start_s=envelope[0] if envelope else None,
                             recomputed_end_s=envelope[1] if envelope else None,
                             matches_published=bool(envelope and max(abs(envelope[i] - target[i]) for i in (0, 1)) < 1e-9),
                             mae_ms=scores["mae_ms"], rmse_ms=scores["rmse_ms"]))
    return rows


def run_experiment(args):
    """Điều phối học train, dự đoán/đánh giá và xuất artifacts thực nghiệm.

    Input: args từ CLI gồm algorithm, file, compare_context, snr_study,
    no_show, show_seconds và evaluate_all. Output: None; lưu model/JSON/CSV/PNG,
    in metric và mở demo. Mặc định bốn test; --file chọn một WAV;
    --evaluate-all đánh giá toàn bộ train/test WAV và tự headless.
    """
    file_argument = getattr(args, "file", None)
    input_path = resolve_input_file(file_argument) if file_argument else None
    audio_train = load_audio_folder(TRAIN_DIR)
    train_common = prepare_records(audio_train)
    selected = ["tt1", "tt2", "tt3"] if args.algorithm == "all" else [args.algorithm]
    if args.compare_context and "tt2-context" not in selected:
        selected.append("tt2-context")

    # Lock every model, including FINAL W and noise, before TEST WAV/LAB I/O.
    models, sweeps = {}, {}
    for key in selected:
        models[key], sweeps[key] = fit_training_model(key, train_common)

    evaluate_all=getattr(args,'evaluate_all',False)
    audio_test = [load_audio_file(input_path)] if input_path else audio_train+load_audio_folder(TEST_DIR) if evaluate_all else load_audio_folder(TEST_DIR)
    evaluated_names = [record["name"] for record in audio_test]
    display_order = [name for name in FILE_ORDER if name in evaluated_names]
    display_order.extend(name for name in evaluated_names if name not in display_order)
    evaluation_split = 'all_dataset' if evaluate_all else "train" if input_path and input_path.parent == TRAIN_DIR.resolve() else "external" if input_path and input_path.parent != TEST_DIR.resolve() else "test"
    test_common = prepare_records(audio_test)
    records_by_algorithm = {key: test_common for key in selected}
    tables = OUTPUT_DIR / "tables"
    figures_root = OUTPUT_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    run_name = args.algorithm + ("_with_context" if args.compare_context else "")
    if evaluate_all:run_name+='_all_dataset'
    run_folder = tables / run_name
    if input_path:
        # Một lần thử file không ghi đè metric tổng hợp của batch bốn test.
        run_folder = run_folder / "single" / input_path.stem
        print(f"Chạy WAV: {input_path} | tập {evaluation_split}; ngưỡng/noise/W đã chọn trên 4 WAV TRAIN.")

    # Export the already locked TRAIN models and calibration audit.
    for key in selected:
        model = models[key]
        if key=='tt2':
            selection, sweep_rows = model['W_selection_train'], sweeps[key]
            audit=tables/'tt2_w_selection'
            write_csv(audit/'sweep.csv',sweep_rows)
            write_csv(audit/'summary.csv',selection['summaries'])
            write_csv(audit/'best_by_file.csv',[dict(filename=name,**values) for name,values in selection['best_by_file'].items()])
            write_json(audit/'selection.json',selection)
            print(f"TT2 W={model['W']:g}: selected on four TRAIN LABs; TEST scoring reuses historically exposed data; "
                  f"mean final MAE={selection['selected_summary']['mean_MAE_ms']} ms")
        write_json(OUTPUT_DIR / "models" / f"{key}.json", model)

    # Lưu dữ liệu thống kê giúp kiểm tra lại mean/std, normalization và LAB.
    training_frames = []
    for record in train_common:
        for center, value, label in zip(record["features"]["centers"], record["features"]["ste_norm"], record["labels"]):
            training_frames.append(dict(file=record["name"], center_s=center, ste_norm=value,
                                        label="unlabeled" if label is None else "speech" if label else "silence"))
    write_csv(tables / "training_frames.csv", training_frames)
    statistics_path = run_folder / "dataset_statistics.csv" if input_path else tables / "dataset_statistics.csv"
    write_csv(statistics_path, data_statistics(audio_train, "train") + data_statistics(audio_test, evaluation_split))
    if "tt3" in models:
        values_by_class = {0: [], 1: []}
        for record in train_common:
            for value, label in zip(record["features"]["ste_norm"], record["labels"]):
                if label is not None:
                    values_by_class[label].append(value)
        training_figures = figures_root / "training"
        training_figures.mkdir(exist_ok=True)
        plot_gaussian_training(values_by_class, models["tt3"], training_figures / "gaussian_distributions.png")

    metric_rows, threshold_rows, results_by_file = [], [], {}
    demo_figures, noise_rows = [], []
    for key in selected:
        folder = figures_root / key
        folder.mkdir(exist_ok=True)
        for record in records_by_algorithm[key]:
            result = predict_and_score(key, record, models[key])
            metric_rows.append(dict(file=record["name"],split=record.get('split',evaluation_split),algorithm=key, **result["metrics"]))
            threshold_rows.append(dict(file=record["name"], algorithm=key,
                                       **{name: value for name, value in result["diagnostic"].items() if not isinstance(value, (dict, list))}))
            results_by_file.setdefault(record["name"], {})[key] = (record, result)
            write_json(OUTPUT_DIR / "diagnostics" / key / f"{record['name']}.json", result["diagnostic"])
            segments = mask_segments(result["mask"], record["features"]["starts"], record["duration"], frame_ends=record["features"]["ends"])
            write_csv(OUTPUT_DIR / "predictions" / key / f"{record['name']}.csv",
                      [dict(start_s=start, end_s=end, label="speech" if state else "sil") for start, end, state in segments])
            figure = make_file_figure(record["name"], [(record, result)], folder / f"{record['name']}.png")
            # Batch giữ 4 figure, --file chỉ giữ 1; các bản xuất phụ đóng ngay.
            if args.algorithm != "all" and key == args.algorithm and not args.no_show:
                demo_figures.append(figure)
            else:
                plt.close(figure)

            if args.snr_study:
                noise_rows.append(dict(file=record["name"], algorithm=key, added_snr_db="clean", **result["metrics"]))
                for level in NOISE_LEVELS_DB:
                    file_seed = FILE_ORDER.index(record["name"]) if record["name"] in FILE_ORDER else zlib.crc32(record["name"].encode("utf-8"))
                    base_noisy = noisy_copy(record, level, RANDOM_SEED + file_seed * 100 + level)
                    noisy = prepare_records([base_noisy], key == "tt2")[0]
                    noisy_result = predict_and_score(key, noisy, models[key])
                    noise_rows.append(dict(file=record["name"], algorithm=key, added_snr_db=level, **noisy_result["metrics"]))

    # all có một figure so sánh ba hàng/file; subset chỉ vẽ các WAV đã chọn.
    if args.algorithm == "all":
        comparison_folder = figures_root / "comparison"
        comparison_folder.mkdir(exist_ok=True)
        for name in display_order:
            figure = make_file_figure(name, [results_by_file[name][key] for key in ("tt1", "tt2", "tt3")], comparison_folder / f"{name}.png")
            if args.no_show:
                plt.close(figure)
            else:
                demo_figures.append(figure)
    else:
        demo_figures.sort(key=lambda figure: display_order.index(figure._suptitle.get_text()))

    # Lưu từng run độc lập; demo một thuật toán không ghi đè bảng tổng hợp all.
    write_csv(run_folder / "test_metrics.csv", metric_rows)
    write_csv(run_folder / "summary.csv", summarize(metric_rows))
    write_csv(run_folder / "test_thresholds.csv", threshold_rows)
    if args.snr_study:
        write_csv(run_folder / "snr_study.csv", noise_rows)
    if args.compare_context:
        write_csv(tables / "context_benchmark_check.csv", context_benchmark(test_common))
    write_json(run_folder / "run_config.json", dict(algorithm=args.algorithm, selected=selected,
               input_file=str(input_path) if input_path else None, evaluated_files=evaluated_names,
               evaluation_split=evaluation_split,
               schema_version=2, metrics_schema_version=2,
               parameter_selection_set='train', evaluation_protocol='train_selected_reused_test',
               historical_test_exposure=True, training_files=[record['name'] for record in train_common],
               parameter_selection={key:model['evaluation_protocol'] for key,model in models.items()},
               frame_ms=FRAME_MS, hop_ms=HOP_MS, sample_rounding=SAMPLE_ROUNDING,
               minimum_internal_silence_ms=MIN_SILENCE_SECONDS * 1000,minimum_speech_ms=MIN_SPEECH_SECONDS*1000,
               snr_study=args.snr_study, normalization="per_recording_max", lab_center_convention="sample_exact_half_open",
               boundary_convention="final confirmed regions; union of active frame supports", primary_metric="ordered final-region START/END MAE; unmatched regions explicitly undefined",
               noise_seed=RANDOM_SEED))
    print(f"Đã xuất kết quả: {OUTPUT_DIR}")
    for row in summarize(metric_rows):
        mae = f"{row['mean_file_mae_ms']:.2f}" if row["mean_file_mae_ms"] is not None else "N/A"
        rmse = f"{row['mean_file_rmse_ms']:.2f}" if row["mean_file_rmse_ms"] is not None else "N/A"
        print(f"{row['algorithm']}: mean endpoint MAE={mae} ms; "
              f"mean per-file RMSE={rmse} ms; frame F1={row['mean_frame_f1']:.3f}")
        print(f"  median={row['median_file_mae_ms']}; min={row['min_file_mae_ms']}; max={row['max_file_mae_ms']} ms; "
              f"region counts correct={row['region_count_correct_files']}/{row['files']}; wrong={row['region_count_incorrect_files']}")
        print(f"  highest MAE: {row['top_mae_files']}")
    if not args.no_show:
        positions = show_figures(demo_figures, args.show_seconds)
        write_json(run_folder / "demo_layout.json", positions)
        if len(demo_figures) == 4:
            print(f"Demo đã bố trí {len(positions)} cửa sổ tại bốn góc.")
        else:
            print(f"Đã hiển thị {len(demo_figures)} figure cho WAV đã chọn.")
