"""Bounded synthetic characterization and four-arm TRAIN endpoint ablation.

Run: python tools/check_endpoint_robustness.py [--timing-research]
Reads only TRAIN WAV/LABs. Writes diagnostic CSV/JSON only beneath
outputs/tables/endpoint_robustness (legacy) or endpoint_modes_oct07/timing_research; never changes fitted production artifacts.
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




# Task 4 diagnostic branches: one factor at a time, never production policy.
from copy import deepcopy
from core.decision_timing import decision_cells
from app.pipeline import algorithm_decision
from app.weight_selection import select_final_weight

TIMING_DESTINATION = OUTPUT_DIR / 'tables' / 'endpoint_modes_oct07' / 'timing_research'
TIMING_TRIALS = [
    dict(trial_id=trial_id,endpoint_mode=mode,geometry=geometry,minimum_speech_ms=minimum,
         low_rule=low_rule,algorithms=ALGORITHMS if low_rule=='native' else ('tt1',))
    for trial_id,mode,geometry,minimum,low_rule in (
        ('core_support_0','core','support',0.,'native'),
        ('core_cells_0','core','cells',0.,'native'),
        ('core_support_100','core','support',100.,'native'),
        ('enhanced_support_100','enhanced','support',100.,'native'),
        ('enhanced_cells_100','enhanced','cells',100.,'native'),
        ('enhanced_support_0','enhanced','support',0.,'native'),
        ('enhanced_tt1_t1_support_100','enhanced','support',100.,'T1'))]


def timing_predictor(algorithm, record, params):
    """Same LAB-free experimental detector for fit W candidates and held-out scores.

    Geometry controls every start/end, merge, duration filter and final mask.
    Labels enter only the scorer after final detection. Native TT2 is unpadded.
    """
    f,duration=record['features'],record['duration']
    geometry=params['geometry']; mode=params['endpoint_mode']
    if geometry not in ('support','cells') or mode not in ('core','enhanced'):
        raise ValueError('Unknown research geometry or endpoint mode')
    starts,ends=decision_cells(f,duration) if geometry=='cells' else (f['starts'],f['ends'])
    seed,diag=algorithm_decision(algorithm,f,params)
    minimum=params['minimum_speech_ms']/1000
    candidate=[(s,e) for s,e,k in mask_segments(seed,starts,duration,ends) if k]
    low=high=None
    if mode=='core':
        merged=[]
        for start,end in candidate:
            if merged and start-merged[-1][1]<.2-1e-12:merged[-1]=(merged[-1][0],end)
            else:merged.append((start,end))
        final=[(s,e) for s,e in merged if e-s>=minimum-1e-12]
    else:
        if algorithm=='tt3' and params.get('speech_direction','high')!='high':
            raise ValueError('Enhanced requires speech-high TT3')
        if algorithm=='tt2':
            peak=max(f['energy'],default=0.)
            base=diag['energy_threshold']/peak if peak else 0.
        else:base=params['threshold']
        if params.get('low_rule')=='T1':
            low=base; high=max(1.5*base,params['endpoint_noise']['noise_upper'])
            if not 0<low<high:raise ValueError('unsupported LOW=T1: invalid hysteresis thresholds')
        else:low,high=endpoint_thresholds(base,params['endpoint_noise'],histogram=algorithm=='tt2')
        final=hysteresis_regions(f['ste_norm'],starts,ends,duration,low,high,.2,minimum,seed_mask=seed)
    # Passing the chosen timing arrays provides whole-cell containment for cells.
    mask=regions_to_mask(final,starts,ends)
    diag.update(endpoint_mode=mode,geometry=geometry,boundary_convention=params.get('boundary_convention','nearest-frame centered decision cells' if geometry=='cells' else 'union of active frame supports'),
        minimum_speech_ms=params['minimum_speech_ms'],minimum_silence_ms=200.,
        padding_stage=params.get('padding_stage','excluded from core FINAL' if mode=='core' else 'none'),low_ste_threshold=low,high_ste_threshold=high,
        candidate_regions=candidate,raw_speech_frames=sum(seed),final_regions=final,
        last_native_active_end_s=_last_support(seed,ends))
    reference=ground_truth_regions(record['intervals'])
    return dict(mask=mask,final_regions=final,diagnostic=diag,metrics=region_endpoint_metrics(reference,final),
                frame_metrics=frame_metrics(record['labels'],mask))


def _branch_params(native,trial):
    params=deepcopy(native)
    # Native calibration digest and support W provenance are preliminary only.
    params.pop('calibration_digest',None)
    params.pop('W_selection_train',None)
    params.update({k:v for k,v in trial.items() if k!='algorithms'})
    params.update(boundary_convention='nearest-frame centered decision cells' if trial['geometry']=='cells'
                  else 'union of active frame supports',minimum_silence_ms=200.,
                  endpoint_policy='diagnostic native/hysteresis branch; merge estimated gaps below 200 ms',
                  padding_stage='excluded from core FINAL' if trial['endpoint_mode']=='core'
                  else 'candidate diagnostics only' if native.get('variant')=='source' else 'none')
    return params


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def timing_training_study(records):
    """Four deterministic TRAIN folds; separate W=1..50 selection per branch."""
    require_train(records)
    records=sorted(records,key=lambda r:r['name'])
    if len(records)!=4 or len({r['name'] for r in records})!=4:
        raise ValueError('Exactly four distinct TRAIN records required')
    heldout_rows=[]; sweep_rows=[]; models=[]
    for fold,heldout in enumerate(records,1):
        fit=[r for r in records if r is not heldout]
        native={(mode,a):fit_training_model(a,fit,endpoint_mode=mode)[0]
                for mode in ('core','enhanced') for a in ALGORITHMS}
        for trial in TIMING_TRIALS:
            for algorithm in trial['algorithms']:
                params=_branch_params(native[trial['endpoint_mode'],algorithm],trial)
                selection=None
                if algorithm=='tt2':
                    branch_rows=[]
                    for w in range(1,51):
                        candidate_params=dict(params,W=float(w),finalW=float(w))
                        for record in fit:
                            result=timing_predictor(algorithm,record,candidate_params)
                            metrics=result['metrics']
                            branch_rows.append(dict(filename=record['name'],W=float(w),
                                endpoint_mode=trial['endpoint_mode'],boundary_convention=params['boundary_convention'],
                                minimum_speech_ms=trial['minimum_speech_ms'],minimum_silence_ms=200.,
                                padding_stage=params['padding_stage'],final_region_mae_ms=metrics['mae_ms'],
                                ground_truth_region_count=metrics['ground_truth_region_count'],
                                predicted_region_count=metrics['predicted_region_count'],status=metrics['status']))
                    selection=select_final_weight(branch_rows)
                    selection.update(trial_id=trial['trial_id'],geometry=trial['geometry'],low_rule=trial['low_rule'],
                                     branch_predictor='tools.check_endpoint_robustness.timing_predictor',
                                     selection_scorer='core.metrics.region_endpoint_metrics',evaluated_files=[r['name'] for r in fit])
                    params.update(W=selection['selected_W'],finalW=selection['selected_W'],
                                  W_selection_train=selection,parameter_rule=selection['selection_rule'])
                    sweep_rows.extend(dict(fold=fold,trial_id=trial['trial_id'],geometry=trial['geometry'],**r) for r in branch_rows)
                params.update(training_files=[r['name'] for r in fit],evaluation_protocol='four_fold_train_holdout',
                              branch_predictor='tools.check_endpoint_robustness.timing_predictor',selection_scorer='core.metrics.region_endpoint_metrics')
                digest=_digest(params)
                models.append(dict(fold=fold,algorithm=algorithm,trial_id=trial['trial_id'],heldout_file=heldout['name'],
                    fit_files=params['training_files'],model_sha256=digest,model=params,
                    preliminary_native_W=native[trial['endpoint_mode'],algorithm].get('W'),
                    preliminary_native_digest=native[trial['endpoint_mode'],algorithm].get('calibration_digest')))
                try:
                    result=timing_predictor(algorithm,heldout,params)
                    scores=result['metrics']
                    row=dict(**scores,final_regions=result['final_regions'],diagnostic=result['diagnostic'],
                             **{f'frame_{k}':v for k,v in result['frame_metrics'].items()})
                    if trial['trial_id'] in ('core_support_0','enhanced_support_100'):
                        production=detect_regions(algorithm,heldout['features'],heldout['duration'],params)
                        if production['final_regions']!=result['final_regions'] or production['mask']!=result['mask']:
                            raise AssertionError('Matching support baseline differs from production')
                except ValueError as error:
                    if 'unsupported LOW=T1' not in str(error):raise
                    row=dict(status=str(error),mae_ms=None,rmse_ms=None,missing_region_count=0,
                             extra_region_count=0,signed_endpoint_errors_ms=[],region_pairs=[],final_regions=None)
                heldout_rows.append(dict(fold=fold,file=heldout['name'],algorithm=algorithm,
                    trial_id=trial['trial_id'],arm=trial['trial_id'],geometry=trial['geometry'],endpoint_mode=trial['endpoint_mode'],
                    low_rule=trial['low_rule'],minimum_speech_ms=trial['minimum_speech_ms'],W=params.get('W'),
                    model_sha256=digest,fit_files=params['training_files'],**row))
        print(f'Timing TRAIN fold {fold}/4 complete',flush=True)
    if len(heldout_rows)!=76 or len(sweep_rows)!=3600:raise AssertionError('Declared evaluation budget changed')
    comparisons=[]
    for row in heldout_rows:
        baseline_id='core_support_0' if row['endpoint_mode']=='core' else 'enhanced_support_100'
        if row['trial_id']==baseline_id:continue
        baseline=next(r for r in heldout_rows if r['fold']==row['fold'] and r['algorithm']==row['algorithm'] and r['trial_id']==baseline_id)
        a,b=row['mae_ms'],baseline['mae_ms']
        comparisons.append(dict(fold=row['fold'],file=row['file'],algorithm=row['algorithm'],trial_id=row['trial_id'],
            baseline_trial_id=baseline_id,baseline_status=baseline['status'],status=row['status'],
            baseline_mae_ms=b,mae_ms=a,delta_mae_ms=a-b if a is not None and b is not None else None,
            baseline_rmse_ms=baseline['rmse_ms'],rmse_ms=row['rmse_ms'],
            delta_rmse_ms=row['rmse_ms']-baseline['rmse_ms'] if row['rmse_ms'] is not None and baseline['rmse_ms'] is not None else None,
            validity_transition=f'{"valid" if b is not None else "undefined"}->{"valid" if a is not None else "undefined"}',
            count_regression=row['missing_region_count']+row['extra_region_count']>baseline['missing_region_count']+baseline['extra_region_count'],
            metric_regression=a is not None and b is not None and a>b+1e-8,
            rmse_regression=row['rmse_ms'] is not None and baseline['rmse_ms'] is not None and row['rmse_ms']>baseline['rmse_ms']+1e-8))
    return dict(heldout=heldout_rows,sweep=sweep_rows,models=models,comparisons=comparisons,summary=ablation_summary(heldout_rows))


def _synthetic_record(samples,fs,speech,name):
    duration=len(samples)/fs
    intervals=[];cursor=0.
    for a,b in speech:
        start,end=a/fs,b/fs
        if start>cursor:intervals.append((cursor,start,'sil'))
        intervals.append((start,end,'v'));cursor=end
    if cursor<duration:intervals.append((cursor,duration,'sil'))
    return prepare_records([dict(name=name,split='train',samples=samples,sample_rate=fs,
                                 duration=duration,intervals=intervals)])[0]


def timing_waveform_study(records):
    """Characterize physical GT separately using locked four-TRAIN models."""
    require_train(records)
    native={(mode,a):fit_training_model(a,records,endpoint_mode=mode)[0]
            for mode in ('core','enhanced') for a in ALGORITHMS}
    rows=[];oracle_cells=[];fixtures=[]
    for fs in (16000,44100):
        hop=round(fs*.01)
        for gap in (190,200,210,250):
            for phase in (0,1,hop//4,hop//2,3*hop//4,hop-1):
                samples,speech=rectangular_case(fs,phase,gap)
                record=_synthetic_record(samples,fs,speech,f'gap_{fs}_{gap}_{phase}')
                metadata=dict(case='physical_gap',sample_rate_hz=fs,phase_samples=phase,
                              true_gap_ms=(speech[1][0]-speech[0][1])/fs*1000)
                fixtures.append((record,metadata))
                f=record['features'];starts,ends=decision_cells(f,record['duration'])
                values,supports=_overlap_oracle(speech,len(starts),f['frame_size'],f['hop_size'],fs)
                # Independent cell LOW gap from literal integer-overlap active indices.
                runs=[]
                for i in range(len(starts)):
                    count=sum(max(0,min(i*f['hop_size']+f['frame_size'],b)-max(i*f['hop_size'],a)) for a,b in speech)
                    if count*10>=f['frame_size']:
                        if runs and i==runs[-1][1]+1:runs[-1]=(runs[-1][0],i)
                        else:runs.append((i,i))
                left=runs[0][1];right=runs[1][0]
                left_edge=((left+.5)*f['hop_size']+f['frame_size']/2)/fs
                right_edge=((right-.5)*f['hop_size']+f['frame_size']/2)/fs
                final=hysteresis_regions(values,starts,ends,record['duration'],.1,.5,.2,.1)
                oracle_cells.append(dict(**metadata,geometry='cells',threshold_source='fixed diagnostic LOW=.1 HIGH=.5',
                    estimated_cell_gap_ms=(right_edge-left_edge)*1000,support_gap_ms=(supports[1][0]-supports[0][1])*1000,
                    low=.1,high=.5,final_regions=final,**region_endpoint_metrics(ground_truth_regions(record['intervals']),final)))
        for burst in (75,95,100,105):
            a=round(fs*.1);b=a+round(fs*burst/1000)
            samples=[0.]*a+[1.]*(b-a)+[0.]*round(fs*.3)
            fixtures.append((_synthetic_record(samples,fs,[(a,b)],f'burst_{fs}_{burst}'),
                dict(case='physical_burst',sample_rate_hz=fs,true_speech_ms=(b-a)/fs*1000)))
        for edge in ('start','end','both','weak_tail'):
            n=round(fs*.6);a=0 if edge in ('start','both') else round(fs*.1)
            b=n if edge in ('end','both','weak_tail') else round(fs*.4)
            samples=[0.]*n
            samples[a:b]=[1.]*(b-a)
            if edge=='weak_tail':samples[round(fs*.4):b]=[.05]*(b-round(fs*.4))
            fixtures.append((_synthetic_record(samples,fs,[(a,b)],f'edge_{fs}_{edge}'),
                dict(case='physical_edge_or_tail',sample_rate_hz=fs,edge=edge,weak_tail_amplitude=.05 if edge=='weak_tail' else None)))
    for record,metadata in fixtures:
        for trial in TIMING_TRIALS:
            for a in trial['algorithms']:
                params=_branch_params(native[trial['endpoint_mode'],a],trial)
                try:
                    result=timing_predictor(a,record,params)
                    extra=dict(final_regions=result['final_regions'],diagnostic=result['diagnostic'],**result['metrics'])
                except ValueError as error:
                    if 'unsupported LOW=T1' not in str(error):raise
                    extra=dict(status=str(error),mae_ms=None,rmse_ms=None,final_regions=None)
                rows.append(dict(**metadata,trial_id=trial['trial_id'],algorithm=a,geometry=trial['geometry'],
                    endpoint_mode=trial['endpoint_mode'],minimum_speech_ms=trial['minimum_speech_ms'],
                    model_sha256=_digest(params),threshold_source='locked four-TRAIN calibration; synthetic characterization only',
                    ground_truth_regions=ground_truth_regions(record['intervals']),**extra))
    return dict(waveforms=rows,oracle_cells=oracle_cells,synthetic_models=[dict(mode=mode,algorithm=a,model=model,
        model_sha256=_digest(model)) for (mode,a),model in native.items()])


def _write_research_table(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (TIMING_DESTINATION/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:_json_value(v) if v is None or isinstance(v,(tuple,list,dict)) else v for k,v in row.items()})


def timing_research_main():
    """Run Task 4 diagnostics explicitly; no TEST reads or automatic selection."""
    TIMING_DESTINATION.mkdir(parents=True,exist_ok=True)
    records=prepare_records(load_audio_folder(TRAIN_DIR));require_train(records)
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for r in records for path in (Path(r['wav_path']),Path(r['wav_path']).with_suffix('.lab'))}
    payload=timing_training_study(records)
    synthetic=timing_waveform_study(records)
    old_oracle=waveform_characterization()
    for name,rows in (('heldout_scores',payload['heldout']),('tt2_branch_fit_sweep',payload['sweep']),
                      ('fold_comparisons',payload['comparisons']),('heldout_summary',payload['summary']),
                      ('locked_model_waveforms',synthetic['waveforms']),('fixed_oracle_support',old_oracle),
                      ('fixed_oracle_cells',synthetic['oracle_cells'])):
        _write_research_table(name,rows)
    manifest=dict(parameter_selection_set='train',historical_test_exposure=True,
        evaluation_protocol='four_fold_train_holdout',input_sha256=hashes,trials=TIMING_TRIALS,
        heldout_rows=len(payload['heldout']),tt2_fit_branch_evaluations=len(payload['sweep']),
        declared_TT2_W_candidates=list(range(1,51)),heldout_used_for_selection=False,
        preliminary_native_support_sweeps_are_not_branch_selection=True,
        branch_predictor='tools.check_endpoint_robustness.timing_predictor',selection_scorer='core.metrics.region_endpoint_metrics',
        models=payload['models'],synthetic_models=synthetic['synthetic_models'],
        limitations=['Cells extend nearest decisions to audio edges and unanalysed tail.',
            'Full analysis windows remain 25 ms (1102 samples at 44.1 kHz).',
            'Cells do not guarantee preservation of physical 200 ms silence.',
            'Physical GT has not been changed; estimated gaps below 200 ms still merge.',
            'Primary full mean remains undefined whenever any held-out score is undefined.',
            'Four TRAIN folds and historically exposed data do not create an independent TEST set.',
            'No default/convention/LOW automatically selected or applied.'])
    (TIMING_DESTINATION/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Research: {len(payload['heldout'])} heldout rows; {len(payload['sweep'])} branch W fit evaluations; {len(synthetic['waveforms'])} locked-model synthetic rows",flush=True)
    return dict(**payload,**synthetic,manifest=manifest)

if __name__ == '__main__':
    timing_research_main() if '--timing-research' in sys.argv else main()
