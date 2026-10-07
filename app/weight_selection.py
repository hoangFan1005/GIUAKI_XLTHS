"""Choose one histogram W using FINAL-region errors on supplied TRAIN records.

TRAIN calibration is reused for scoring historically exposed TEST recordings.
All timing, feature extraction, histogram and hysteresis settings stay fixed.
"""
import math
from app.config import HISTOGRAM_W_TIE_PREFERENCE

W_CANDIDATES=tuple(float(w) for w in range(1,51))
ERROR_EPS_MS=1e-8
POLICY_FIELDS=('endpoint_mode','boundary_convention','minimum_speech_ms',
               'minimum_silence_ms','padding_stage')

def select_final_weight(rows,tie_preference_W=HISTOGRAM_W_TIE_PREFERENCE):
    """Input: one MAE/count row per (W,file). Output: shared W and audit tables.

    Prefer valid region counts, then minimize the worst per-file loss relative
    to that file's own optimum, then mean MAE. Ties use the declared preference;
    if unavailable, choose the smallest W. No candidate/debug boundary is scored.
    """
    if not rows:raise ValueError('W selection needs evaluation rows')
    policy={key:rows[0][key] for key in POLICY_FIELDS if key in rows[0]}
    if any({key:row[key] for key in POLICY_FIELDS if key in row} != policy for row in rows):
        raise ValueError('Each candidate W must use the same endpoint policy')
    grouped={}
    for row in rows:
        w=row['W'];name=row['filename'];mae=row['final_region_mae_ms']
        if not math.isfinite(w) or w<0 or (mae is not None and (not math.isfinite(mae) or mae<0)):
            raise ValueError('Invalid W or endpoint MAE')
        files=grouped.setdefault(w,{})
        if name in files:raise ValueError('Duplicate (W,filename) in selection')
        files[name]=row
    names=sorted(next(iter(grouped.values())))
    if any(set(files)!=set(names) for files in grouped.values()):
        raise ValueError('Each candidate W must be evaluated on the same files')
    def valid(row):
        return row['final_region_mae_ms'] is not None and row['ground_truth_region_count']==row['predicted_region_count']
    best_by_file={}
    for name in names:
        options=[(w,files[name]['final_region_mae_ms']) for w,files in grouped.items() if valid(files[name])]
        minimum=min((mae for _,mae in options),default=None)
        best_by_file[name]=dict(minimum_MAE_ms=minimum,
            optimal_W=sorted(w for w,mae in options if abs(mae-minimum)<=ERROR_EPS_MS))
    summaries=[]
    for w in sorted(grouped):
        valid_rows=[row for row in grouped[w].values() if valid(row)]
        errors=[row['final_region_mae_ms'] for row in valid_rows]
        regrets=[row['final_region_mae_ms']-best_by_file[row['filename']]['minimum_MAE_ms'] for row in valid_rows]
        invalid=len(names)-len(valid_rows)
        mean=sum(errors)/len(errors) if errors else None
        summaries.append(dict(W=w,files=len(names),invalid_files=invalid,
            mean_MAE_ms=mean if not invalid else None,valid_files_mean_MAE_ms=mean,
            max_MAE_ms=max(errors) if errors else None,max_regret_ms=max(regrets) if regrets else None))
    # Compare numeric objectives with a tolerance; avoid arbitrary float tie flips.
    chosen=None
    for candidate in summaries:
        objectives=(candidate['invalid_files'],candidate['max_regret_ms'] if candidate['max_regret_ms'] is not None else math.inf,
                    candidate['valid_files_mean_MAE_ms'] if candidate['valid_files_mean_MAE_ms'] is not None else math.inf)
        if chosen is None:chosen=candidate;best_objectives=objectives;continue
        better=False;tied=True
        for left,right in zip(objectives,best_objectives):
            if left==right or (math.isfinite(left) and math.isfinite(right) and abs(left-right)<=ERROR_EPS_MS):continue
            better=left<right;tied=False;break
        if tied:
            better=(candidate['W']!=tie_preference_W,candidate['W'])<(chosen['W']!=tie_preference_W,chosen['W'])
        if better:chosen=candidate;best_objectives=objectives
    shared=set.intersection(*(set(v['optimal_W']) for v in best_by_file.values()))
    return dict(**policy,selected_W=chosen['W'],tie_preference_W=tie_preference_W,selected_summary=chosen,
        summaries=summaries,best_by_file=best_by_file,shared_optimal_W=sorted(shared),
        candidate_W=sorted(grouped),selection_set='train',evaluation_protocol='train_final_calibration',
        selection_rule='valid regions; minimum worst per-file regret; mean MAE; declared W preference on ties; smallest W otherwise')

def sweep_final_weights(records,model,predictor,candidates=W_CANDIDATES,
                        tie_preference_W=HISTOGRAM_W_TIE_PREFERENCE):
    """Evaluate every W on all calibration files with the real endpoint pipeline.

    Input: annotated TRAIN records, fitted TRAIN/noise model, real predictor.
    Output: selection manifest and rows. Does not mutate records or model.
    """
    if not records:raise ValueError('W sweep needs annotated TRAIN recordings')
    if any(record.get('split')!='train' for record in records):
        raise ValueError('W sweep requires explicit TRAIN provenance (split=train)')
    mode=model.get('endpoint_mode','enhanced')
    if mode not in ('core','enhanced'):raise ValueError('Unknown endpoint_mode')
    policy=dict(endpoint_mode=mode,
        boundary_convention=model.get('boundary_convention','union of active frame supports'),
        minimum_speech_ms=model.get('minimum_speech_ms',0. if mode=='core' else 100.),
        minimum_silence_ms=model.get('minimum_silence_ms',model.get('minimum_internal_silence_ms',200.)),
        padding_stage=model.get('padding_stage','excluded from core FINAL' if mode=='core' else 'candidate diagnostics only'))
    rows=[]
    for w in candidates:
        params=dict(model,W=float(w))
        for record in records:
            result=predictor('tt2',record,params);metrics=result['metrics'];diag=result['diagnostic']
            if diag.get('endpoint_mode','enhanced') != mode:
                raise ValueError('Predictor endpoint_mode disagrees with calibration model')
            if diag.get('geometry', policy['boundary_convention']) != policy['boundary_convention']:
                raise ValueError('Predictor geometry disagrees with calibration boundary_convention')
            for key in ('minimum_speech_ms','minimum_silence_ms','padding_stage'):
                if key in diag and diag[key] != policy[key]:
                    raise ValueError(f'Predictor {key} disagrees with calibration model')
            rows.append(dict(**policy,filename=record['name'],W=float(w),
                ground_truth_region_count=metrics['ground_truth_region_count'],predicted_region_count=metrics['predicted_region_count'],
                final_region_mae_ms=metrics['mae_ms'],status=metrics['status'],
                start_error_ms=metrics['start_error_ms'],end_error_ms=metrics['end_error_ms'],
                energy_threshold=diag['energy_threshold'],
                low_ste_threshold=diag['low_ste_threshold'],high_ste_threshold=diag['high_ste_threshold'],
                raw_speech_frames=diag['raw_speech_frames'],candidate_region_count=len(diag['candidate_regions']),
                final_regions=str(result['final_regions'])))
    result=select_final_weight(rows,tie_preference_W)
    result.update(evaluated_files=[record['name'] for record in records],
                  endpoint_policy=model.get('endpoint_policy','fixed raw STE hysteresis; final START/END only; no padding endpoints'),
                  note='W was calibrated using these TRAIN LABs; TEST has historical exposure and is reused for scoring, so no holdout independence is claimed.')
    return result,rows
