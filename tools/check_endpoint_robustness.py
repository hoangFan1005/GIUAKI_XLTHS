"""Bounded synthetic characterization and four-arm TRAIN endpoint ablation.

Run: python tools/check_endpoint_robustness.py
Reads only TRAIN WAV/LABs. Writes diagnostic CSV/JSON only beneath
outputs/tables/endpoint_robustness; never changes fitted production artifacts.
Support gaps are estimates: true 200/210ms zero intervals can still be merged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms import tt1_hodgkinson as tt1
from algorithms import tt2_histogram as tt2
from algorithms import tt3_gaussian as tt3
from app.config import TRAIN_DIR, OUTPUT_DIR, MIN_SILENCE_SECONDS, MIN_SPEECH_SECONDS
from app.pipeline import detect_regions, fit_training_model, load_audio_folder, prepare_records
from core.endpoints import endpoint_thresholds, hysteresis_regions, regions_to_mask
from core.features import compute_features
from core.metrics import frame_metrics, ground_truth_regions, region_endpoint_metrics
from core.postprocess import fill_short_internal_silences, mask_segments

ALGORITHMS = ('tt1', 'tt2', 'tt3')
ARMS = ('raw0', 'raw100', 'hysteresis0', 'hysteresis100')
DESTINATION = OUTPUT_DIR / 'tables' / 'endpoint_robustness'


def rectangular_case(fs, phase_samples, gap_ms, burst_ms=300):
    """Return +/-1 samples and exact half-open speech sample intervals."""
    a = round(fs*.100)+phase_samples
    b = a+round(fs*burst_ms/1000)
    c = b+round(fs*gap_ms/1000)
    d = c+round(fs*burst_ms/1000)
    # Squared samples are exactly one inside speech and zero in silence.
    samples = ([0.]*a + [(-1. if i % 2 else 1.) for i in range(b-a)]
               + [0.]*(c-b) + [(-1. if i % 2 else 1.) for i in range(d-c)]
               + [0.]*round(fs*.300))
    return samples, [(a,b), (c,d)]


def _overlap_oracle(sample_regions, frame_count, frame_size, hop_size, fs):
    """Derive normalized STE and LOW supports with integer sample overlap."""
    values, supports = [], []
    for index in range(frame_count):
        start, end = index*hop_size, index*hop_size+frame_size
        count = sum(max(0, min(end,b)-max(start,a)) for a,b in sample_regions)
        values.append(count/frame_size)
        # LOW=.1 is checked in integers, independently of feature extraction.
        if count*10 >= frame_size:
            if supports and start <= supports[-1][1]:
                supports[-1] = (supports[-1][0], end)
            else:
                supports.append((start, end))
    return values, [(s/fs,e/fs) for s,e in supports]


def waveform_characterization():
    """Return all 48 fixed phase/Fs/physical-gap cases with physical GT scores."""
    rows = []
    for fs in (16000,44100):
        hop = round(fs*.010)
        phases = (0,1,hop//4,hop//2,3*hop//4,hop-1)
        for gap_ms in (190,200,210,250):
            for phase in phases:
                samples, speech = rectangular_case(fs, phase, gap_ms)
                features = compute_features(samples, fs)
                n, h = features['frame_size'], features['hop_size']
                expected_values, supports = _overlap_oracle(speech, len(features['ste_norm']), n, h, fs)
                if features['ste_norm'] != expected_values:
                    raise AssertionError('Waveform STE differs from integer sample-overlap oracle')
                duration = len(samples)/fs
                regions = hysteresis_regions(features['ste_norm'], features['starts'], features['ends'],
                                              duration, .1, .5, .200, .100)
                reference = [(a/fs,b/fs) for a,b in speech]
                metrics = region_endpoint_metrics(reference, regions)
                # A count mismatch is explicit; it never receives an OK label.
                status = ('known_limitation' if gap_ms in (200,210)
                          else 'expected_short_gap_merge' if gap_ms==190
                          else 'estimated_gap_preserved')
                rows.append(dict(case='physical_gap', sample_rate_hz=fs, phase_samples=phase,
                    true_gap_ms=(speech[1][0]-speech[0][1])/fs*1000,
                    support_gap_ms=(supports[1][0]-supports[0][1])*1000,
                    frame_size=n, hop_size=h, frame_ms=n/fs*1000, hop_ms=h/fs*1000,
                    low=.1, high=.5, minimum_silence_ms=200., minimum_speech_ms=100.,
                    low_regions=supports, final_regions=regions, ground_truth_regions=reference,
                    **dict(metrics, metric_status=metrics['status'], status=status)))
    return rows


def short_burst_characterization():
    """Show that physical 75ms and estimated 100ms span are distinct concepts."""
    rows = []
    for fs in (16000,44100):
        a, b = round(fs*.100), round(fs*.100)+round(fs*.075)
        samples = [0.]*a+[1.]*(b-a)+[0.]*round(fs*.300)
        f = compute_features(samples, fs)
        regions = hysteresis_regions(f['ste_norm'], f['starts'], f['ends'], len(samples)/fs,
                                     .1, .5, .200, .100)
        metrics = region_endpoint_metrics([(a/fs,b/fs)], regions)
        rows.append(dict(case='physical_short_burst', sample_rate_hz=fs,
            true_speech_ms=75., actual_sample_speech_ms=(b-a)/fs*1000,
            estimated_span_ms=(regions[0][1]-regions[0][0])*1000 if regions else None,
            minimum_speech_ms=100., final_regions=regions,
            **dict(metrics, metric_status=metrics['status'], status='physical_duration_differs_from_span')))
    return rows


def require_train(records):
    """Reject TEST/external provenance before fitting, detection or scoring."""
    if not records:
        raise ValueError('Endpoint ablation requires TRAIN records')
    for record in records:
        if record.get('split') != 'train':
            raise ValueError('Endpoint ablation requires explicit TRAIN provenance')
        path = record.get('wav_path')
        if path is not None and Path(path).resolve().parent != TRAIN_DIR.resolve():
            raise ValueError('Endpoint ablation path must belong to TRAIN')


def _native_decisions(algorithm, features, params):
    """Return unpadded raw/seed masks and native/normalized base thresholds."""
    if algorithm == 'tt1':
        raw = tt1.predict(features, params)
        return raw, params['threshold'], params['threshold'], 'normalized_ste'
    if algorithm == 'tt3':
        raw = tt3.predict(features, params)
        return raw, params['threshold'], params['threshold'], 'normalized_ste'
    # Histogram predict supplies its threshold diagnostic; candidate padding
    # is excluded from every arm and from HIGH confirmation seeds.
    _, diagnostic = tt2.predict(features, params)
    threshold = diagnostic['energy_threshold']
    peak = max(features['energy'], default=0.)
    raw = [int(e>threshold) for e in features['energy']]
    base = threshold/peak if peak>0 else 0.
    return raw, threshold, base, 'mean_square_energy'


def _last_support(mask, ends):
    return next((end for flag,end in zip(reversed(mask),reversed(ends)) if flag), None)


def ablate_training(records, models):
    """Score fixed-model raw/hysteresis x zero/100ms span filter on TRAIN."""
    require_train(records)
    rows = []
    for record in records:
        f, duration = record['features'], record['duration']
        reference = ground_truth_regions(record['intervals'])
        for algorithm in ALGORITHMS:
            params = models[algorithm]
            raw, native, base, units = _native_decisions(algorithm, f, params)
            low, high = endpoint_thresholds(base, params['endpoint_noise'], histogram=algorithm=='tt2')
            low_mask = [int(v>=low) for v in f['ste_norm']]
            high_seed = [int(v>=high and seed) for v,seed in zip(f['ste_norm'],raw)]
            cleaned = fill_short_internal_silences(raw, f['starts'], duration,
                                                  MIN_SILENCE_SECONDS, frame_ends=f['ends'])
            raw_runs = [(s,e) for s,e,k in mask_segments(raw,f['starts'],duration,f['ends']) if k]
            merged = [(s,e) for s,e,k in mask_segments(cleaned,f['starts'],duration,f['ends']) if k]
            low_runs = [(s,e) for s,e,k in mask_segments(low_mask,f['starts'],duration,f['ends']) if k]
            unfiltered = hysteresis_regions(f['ste_norm'],f['starts'],f['ends'],duration,
                                             low,high,MIN_SILENCE_SECONDS,0.,seed_mask=raw)
            for arm in ARMS:
                minimum = MIN_SPEECH_SECONDS if arm.endswith('100') else 0.
                before_filter = merged if arm.startswith('raw') else unfiltered
                if arm.startswith('raw'):
                    regions = [(s,e) for s,e in merged if e-s>=minimum-1e-12]
                else:
                    regions = hysteresis_regions(f['ste_norm'],f['starts'],f['ends'],duration,
                        low,high,MIN_SILENCE_SECONDS,minimum,seed_mask=raw)
                if arm=='hysteresis100':
                    production = detect_regions(algorithm,f,duration,params)['final_regions']
                    if regions != production:
                        raise AssertionError('hysteresis100 differs from production detect_regions')
                scores = region_endpoint_metrics(reference,regions)
                frame_scores = frame_metrics(record['labels'],regions_to_mask(regions,f['starts'],f['ends']))
                rows.append(dict(file=record['name'], split='train', algorithm=algorithm, arm=arm,
                    W=params.get('W'), native_threshold=native, native_threshold_units=units,
                    base_normalized_threshold=base, low_ste_threshold=low, high_ste_threshold=high,
                    seed_count=sum(raw), high_seed_count=sum(high_seed),
                    last_raw_support_end_s=_last_support(raw,f['ends']),
                    last_low_support_end_s=_last_support(low_mask,f['ends']),
                    last_high_seed_support_end_s=_last_support(high_seed,f['ends']),
                    raw_regions=raw_runs, low_regions=low_runs, pre_filter_regions=before_filter,
                    removed_regions=[r for r in before_filter if r[1]-r[0]<minimum-1e-12],
                    final_regions=regions, ground_truth_regions=reference,
                    minimum_speech_ms=minimum*1000, minimum_silence_ms=MIN_SILENCE_SECONDS*1000,
                    **scores, **{f'frame_{key}':value for key,value in frame_scores.items()}))
    return rows


def ablation_summary(rows):
    """Include invalid counts; partial valid means never replace primary means."""
    groups = {}
    for row in rows:
        groups.setdefault((row['algorithm'],row['arm']),[]).append(row)
    summaries = []
    for (algorithm,arm), group in sorted(groups.items()):
        valid = [r for r in group if r['mae_ms'] is not None]
        invalid = len(group)-len(valid)
        mean = sum(r['mae_ms'] for r in valid)/len(valid) if valid else None
        rmse_mean = sum(r['rmse_ms'] for r in valid)/len(valid) if valid else None
        summaries.append(dict(algorithm=algorithm,arm=arm,files=len(group),valid_count=len(valid),
            invalid_count=invalid, mean_mae_ms=mean if not invalid else None,
            mean_file_rmse_ms=rmse_mean if not invalid else None,valid_rows_mean_mae_ms=mean,
            valid_rows_mean_file_rmse_ms=rmse_mean,
            max_valid_mae_ms=max((r['mae_ms'] for r in valid),default=None),
            missing_region_count=sum(r['missing_region_count'] for r in group),
            extra_region_count=sum(r['extra_region_count'] for r in group)))
    return summaries


def _json_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_diagnostics(payload):
    """Write only this tool's diagnostic tables at the fixed project destination."""
    DESTINATION.mkdir(parents=True,exist_ok=True)
    for name, rows in (('waveform_characterization',payload['waveforms']),
                       ('short_burst_characterization',payload['short_bursts']),
                       ('train_ablation',payload['ablation']),
                       ('train_ablation_summary',payload['summary'])):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (DESTINATION/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for row in rows:
                # None is JSON null, preserving undefined scores in CSV too.
                writer.writerow({key:_json_value(value) if value is None or isinstance(value,(list,tuple,dict))
                                 else value for key,value in row.items()})
    (DESTINATION/'manifest.json').write_text(json.dumps(payload['manifest'],ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    """Fit/lock fresh TRAIN models once; diagnostics never choose a new policy."""
    records = prepare_records(load_audio_folder(TRAIN_DIR))
    require_train(records)
    models = {a:fit_training_model(a,records)[0] for a in ALGORITHMS}
    # Input hashing remains inside TRAIN; no TEST file is opened by this tool.
    hashes = {}
    for record in records:
        if 'wav_path' in record:
            wav = Path(record['wav_path'])
            for path in (wav,wav.with_suffix('.lab')):
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(parameter_selection_set='train', historical_test_exposure=True,
        purpose='diagnostic ablation; no policy selection', training_files=[r['name'] for r in records],
        input_sha256=hashes, arms=list(ARMS), minimum_silence_ms=MIN_SILENCE_SECONDS*1000,
        minimum_speech_ms=MIN_SPEECH_SECONDS*1000, boundary_convention='union of active frame supports',
        model_sha256={a:hashlib.sha256(_json_value(m).encode('utf-8')).hexdigest() for a,m in models.items()},
        locked_parameters={a:dict(threshold=m.get('threshold'),W=m.get('W'),endpoint_noise=m['endpoint_noise'])
                           for a,m in models.items()},
        physical_silence_limitation='True200/210ms zero intervals may merge; estimated support gap is shorter.',
        minimum_speech_limitation='100ms filters estimated merged span, not physical speech duration.',
        current_train_scores_are_not_holdout=True)
    payload = dict(waveforms=waveform_characterization(),short_bursts=short_burst_characterization(),
                   ablation=ablate_training(records,models),manifest=manifest)
    payload['summary'] = ablation_summary(payload['ablation'])
    write_diagnostics(payload)
    known = sum(r['status']=='known_limitation' for r in payload['waveforms'])
    print(f"Synthetic: {len(payload['waveforms'])} gap cases; {known} known true200/210ms limitations.")
    print(f"TRAIN: {len(payload['ablation'])} fixed-model rows; W={models['tt2']['W']:g}; "
          f"invalid counts remain explicit. Diagnostics: {DESTINATION}")
    return payload


if __name__ == '__main__':
    main()
