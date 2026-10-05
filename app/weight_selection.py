"""Choose one histogram W using FINAL-region errors on the requested test set.

This is explicitly test-set tuning, not an independent held-out evaluation.
All timing, feature extraction, histogram and hysteresis settings stay fixed.
"""
import math

W_CANDIDATES=tuple(float(w) for w in range(1,51))
ERROR_EPS_MS=1e-8

def select_final_weight(rows,current_weight):
    """Input: one MAE/count row per (W,file). Output: shared W and audit tables.

    Prefer valid region counts, then minimize the worst per-file loss relative
    to that file's own optimum, then mean MAE. Ties retain the existing W;
    if unavailable, choose the smallest W. No candidate/debug boundary is scored.
    """
    if not rows:raise ValueError('W selection needs evaluation rows')
    grouped={}
    for row in rows:
        w=row['W'];name=row['filename'];mae=row['boundary_MAE_ms']
        if not math.isfinite(w) or w<0 or (mae is not None and (not math.isfinite(mae) or mae<0)):
            raise ValueError('Invalid W or endpoint MAE')
        files=grouped.setdefault(w,{})
        if name in files:raise ValueError('Duplicate (W,filename) in selection')
        files[name]=row
    names=sorted(next(iter(grouped.values())))
    if any(set(files)!=set(names) for files in grouped.values()):
        raise ValueError('Each candidate W must be evaluated on the same files')
    def valid(row):
        return row['boundary_MAE_ms'] is not None and row['ground_truth_region_count']==row['predicted_region_count']
    best_by_file={}
    for name in names:
        options=[(w,files[name]['boundary_MAE_ms']) for w,files in grouped.items() if valid(files[name])]
        minimum=min((mae for _,mae in options),default=None)
        best_by_file[name]=dict(minimum_MAE_ms=minimum,
            optimal_W=sorted(w for w,mae in options if abs(mae-minimum)<=ERROR_EPS_MS))
    summaries=[]
    for w in sorted(grouped):
        valid_rows=[row for row in grouped[w].values() if valid(row)]
        errors=[row['boundary_MAE_ms'] for row in valid_rows]
        regrets=[row['boundary_MAE_ms']-best_by_file[row['filename']]['minimum_MAE_ms'] for row in valid_rows]
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
            better=(candidate['W']!=current_weight,candidate['W'])<(chosen['W']!=current_weight,chosen['W'])
        if better:chosen=candidate;best_objectives=objectives
    shared=set.intersection(*(set(v['optimal_W']) for v in best_by_file.values()))
    return dict(selected_W=chosen['W'],previous_W=current_weight,selected_summary=chosen,
        summaries=summaries,best_by_file=best_by_file,shared_optimal_W=sorted(shared),
        candidate_W=sorted(grouped),selection_set='test',evaluation_protocol='test_tuned_not_independent',
        selection_rule='valid regions; minimum worst per-file regret; mean MAE; retain previous W on ties; smallest W otherwise')

def sweep_final_weights(records,model,predictor,candidates=W_CANDIDATES):
    """Evaluate every W on all calibration files with the real endpoint pipeline.

    Input: annotated test records, frozen train/noise model, real predictor.
    Output: selection manifest and rows. Does not mutate records or model.
    """
    if not records:raise ValueError('W sweep needs annotated test recordings')
    rows=[]
    for w in candidates:
        params=dict(model,W=float(w))
        for record in records:
            result=predictor('tt2',record,params);metrics=result['metrics'];diag=result['diagnostic']
            rows.append(dict(filename=record['name'],W=float(w),
                ground_truth_region_count=metrics['ground_truth_region_count'],predicted_region_count=metrics['predicted_region_count'],
                boundary_MAE_ms=metrics['mae_ms'],status=metrics['status'],
                start_error_ms=metrics['start_error_ms'],end_error_ms=metrics['end_error_ms'],
                energy_threshold=diag['energy_threshold'],
                low_ste_threshold=diag['low_ste_threshold'],high_ste_threshold=diag['high_ste_threshold'],
                raw_speech_frames=diag['raw_speech_frames'],candidate_region_count=len(diag['candidate_regions']),
                final_regions=str(result['final_regions'])))
    result=select_final_weight(rows,model['W'])
    result.update(evaluated_files=[record['name'] for record in records],
                  endpoint_policy='fixed raw STE hysteresis; final START/END only; no padding endpoints',
                  note='W was selected using these test LABs at the user request; these errors are not independent generalization estimates.')
    return result,rows
