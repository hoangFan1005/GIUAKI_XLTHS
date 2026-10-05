"""Execute the three standalone notebooks and package only verified notebook source.

This runs notebooks in fresh kernels and retains genuine output in each .ipynb.
The audit is performed outside their submitted code; no saved results are fed
into inference. Existing Python benchmark artifacts are read only for comparison.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
# External auditing uses the current Python selection helper. Direct script
# launch otherwise places only tools/ on sys.path, unlike module launch.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NOTEBOOK_DIR = ROOT / 'notebooks'
TEST_NAMES = ('phone_F2', 'phone_M2', 'studio_F2', 'studio_M2')
TRAIN_NAMES = ('phone_F1', 'phone_M1', 'studio_F1', 'studio_M1')
AUDIT_PREFIX = '__ENDPOINT_NOTEBOOK_AUDIT__'

AUDIT_CELL = r'''
import json
assert_model_locked()
def _compact_result(result):
    return {key: result[key] for key in (
        'file', 'algorithm', 'metrics', 'final_regions', 'ground_truth_regions',
        'predicted_boundaries', 'ground_truth_boundaries', 'diagnostic')}
_audit = {
    'algorithm': ALGORITHM,
    'model': MODEL,
    'model_lock_digest': MODEL_LOCK_DIGEST,
    'tt2_w_sweep_rows': TT2W_SWEEP_ROWS,
    'test_results': [_compact_result(result) for result in TEST_RESULTS],
    'all_results': [_compact_result(result) for result in ALL_RESULTS],
    'test_summary': TEST_SUMMARY,
    'dataset_manifest': DATASET_MANIFEST,
}
print('__ENDPOINT_NOTEBOOK_AUDIT__' + json.dumps(_audit, ensure_ascii=False, allow_nan=False))
'''


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        if len(fields) != len({key.casefold() for key in fields}):
            raise ValueError(f'{path.name}: duplicate casefold CSV headers')
        return list(reader)


def model_digest(model):
    return hashlib.sha256(json.dumps(model, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def audit_metric_keys(scores, label, require_primary=True):
    if len(scores) != len({key.casefold() for key in scores}):
        raise ValueError(f'{label}: duplicate casefold metric keys')
    if (require_primary and 'mae_ms' not in scores) or 'boundary_MAE_ms' in scores:
        raise ValueError(f'{label}: PRIMARY MAE must use mae_ms without a boundary alias')
    if any(key.startswith('boundary_') and not key.startswith('boundary_inside_') for key in scores):
        raise ValueError(f'{label}: tolerance metrics must use tolerance_boundary_ prefix')


def audit_model_protocol(model, locked_digest):
    if 'W_selection_test' in model:
        raise ValueError('New models cannot contain legacy TEST W selection')
    expected = dict(schema_version=2, metrics_schema_version=2,
                    parameter_selection_set='train', evaluation_protocol='train_selected_reused_test',
                    historical_test_exposure=True)
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f'Model {key}: expected schema2 TRAIN protocol ({value})')
    if set(model.get('training_files', [])) != set(TRAIN_NAMES) or len(model.get('training_files', [])) != 4:
        raise ValueError('Model training_files must contain exactly four TRAIN names')
    if model_digest(model) != locked_digest:
        raise ValueError('Frozen TRAIN model lock digest differs after TEST/ALL scoring')


def compare_structure(actual, expected, label):
    """Compare all model/source metadata, tolerating only tiny numeric rounding."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'{label}: metadata keys differ')
        for key, value in expected.items():
            compare_structure(actual[key], value, f'{label}/{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'{label}: list differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_structure(left, right, f'{label}/{index}')
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        close_number(actual, expected, label, 1e-12)
    elif actual != expected:
        raise ValueError(f'{label}: notebook={actual!r}, benchmark={expected!r}')


def audit_train_sweep(audit):
    """Validate captured TRAIN calibration against Python CSV and objective."""
    from app.weight_selection import select_final_weight
    model = audit['model']
    selection = model['W_selection_train']
    rows = audit.get('tt2_w_sweep_rows', [])
    if len(rows) != 200:
        raise ValueError('TT2 TRAIN sweep requires exactly 200 rows')
    if (selection.get('selection_set') != 'train'
            or selection.get('evaluation_protocol') != 'train_final_calibration'
            or set(selection.get('evaluated_files', [])) != set(TRAIN_NAMES)
            or len(selection.get('evaluated_files', [])) != 4):
        raise ValueError('TT2 selection must use exactly four TRAIN files and TRAIN FINAL calibration')
    if selection.get('candidate_W') != list(range(1, 51)) or selection.get('tie_preference_W') != 20:
        raise ValueError('TT2 selection requires integer W1…50 and declared tie preference W20')
    keys = {(row['filename'], row['W']) for row in rows}
    if keys != {(name, float(w)) for name in TRAIN_NAMES for w in range(1, 51)} or len(keys) != 200:
        raise ValueError('TT2 TRAIN sweep rows must be unique and cover all 200 file/W pairs')
    target = ROOT / 'outputs/tables/tt2_w_selection'
    reference = read_csv(target / 'sweep.csv')
    if len(reference) != 200:
        raise ValueError('Python TRAIN sweep benchmark needs 200 rows; regenerate schema2 artifacts')
    reference = {(row['filename'], float(row['W'])): row for row in reference}
    if set(reference) != keys:
        raise ValueError('Python TRAIN sweep file/W manifest differs')
    for row in rows:
        baseline = reference[row['filename'], row['W']]
        if set(row) != set(baseline):
            raise ValueError('TRAIN sweep columns differ; regenerate schema2 artifacts')
        for key, value in row.items():
            expected = baseline[key]
            if key == 'final_regions':
                compare_structure(json.loads(json.dumps(ast.literal_eval(value))),
                                  json.loads(json.dumps(ast.literal_eval(expected))), 'TRAIN sweep/final_regions')
            elif isinstance(value, (int, float)) or value is None:
                close_number(value, expected, f'TRAIN sweep/{row["filename"]}/{row["W"]}/{key}')
            elif value != expected:
                raise ValueError(f'TRAIN sweep/{key} differs')
    recomputed = select_final_weight(rows, tie_preference_W=20)
    for key, value in recomputed.items():
        compare_structure(selection.get(key), value, f'TRAIN selection objective/{key}')
    original = json.loads((target / 'selection.json').read_text(encoding='utf-8'))
    compare_structure(selection, original, 'TRAIN selection manifest')
    for key in ('W', 'finalW'):
        close_number(model[key], selection['selected_W'], f'TRAIN selected model/{key}')
    close_number(model['tie_preference_W'], 20, 'TRAIN model tie preference')


def close_number(actual, expected, label, tolerance=1e-8):
    if expected in (None, ''):
        if actual is not None:
            raise ValueError(f'{label}: expected undefined metric, got {actual}')
    elif actual is None or not math.isclose(float(actual), float(expected),
                                            abs_tol=tolerance, rel_tol=1e-12):
        raise ValueError(f'{label}: notebook={actual}, benchmark={expected}')


def audit_test_summary(algorithm, results, summary_rows):
    """Check the saved TEST aggregate using the four captured result metrics."""
    if not isinstance(summary_rows, list) or len(summary_rows) != 1:
        raise ValueError(f'{algorithm}: need one captured TEST summary row')
    summary = summary_rows[0]
    audit_metric_keys(summary, f'{algorithm}/TEST summary', require_primary=False)
    if (summary.get('metrics_schema_version') != 2
            or summary.get('evaluation_protocol') != 'train_selected_reused_test'
            or summary.get('historical_test_exposure') is not True):
        raise ValueError(f'{algorithm}: TEST summary must disclose schema2 reused TEST protocol')
    if summary.get('algorithm') != algorithm:
        raise ValueError(f'{algorithm}: wrong algorithm in TEST summary')
    scores = [result['metrics'] for result in results]
    valid = [score for score in scores if score['mae_ms'] is not None]
    ordered = sorted(score['mae_ms'] for score in valid)
    count = len(valid)
    correct = sum(score['ground_truth_region_count'] == score['predicted_region_count']
                  for score in scores)
    expected_counts = {
        'files': len(scores), 'evaluated_files': count,
        'missing_speech_files': len(scores) - count,
        'region_count_correct_files': correct,
        'region_count_incorrect_files': len(scores) - correct,
    }
    for key, expected in expected_counts.items():
        if key not in summary or summary[key] != expected:
            raise ValueError(f'{algorithm}/TEST summary/{key}: expected {expected}, '
                             f"got {summary.get(key)}")
    median = (ordered[count // 2] if count % 2 else
              (ordered[count // 2 - 1] + ordered[count // 2]) / 2) if count else None
    squared = sum(score.get('endpoint_squared_error_ms2', score['rmse_ms'] ** 2 * 2)
                  for score in valid)
    error_count = sum(score.get('endpoint_error_count', 2) for score in valid)
    expected_numbers = {
        'mean_file_mae_ms': sum(ordered) / count if count else None,
        'mean_file_rmse_ms': sum(score['rmse_ms'] for score in valid) / count if count else None,
        'pooled_endpoint_rmse_ms': math.sqrt(squared / error_count) if error_count else None,
        'median_file_mae_ms': median,
        'min_file_mae_ms': ordered[0] if count else None,
        'max_file_mae_ms': ordered[-1] if count else None,
        'mean_frame_f1': sum(score['frame_f1'] for score in scores) / len(scores),
    }
    for key, expected in expected_numbers.items():
        if key not in summary:
            raise ValueError(f'{algorithm}/TEST summary: missing {key}')
        close_number(summary[key], expected, f'{algorithm}/TEST summary/{key}')
    highest = sorted((result for result in results if result['metrics']['mae_ms'] is not None),
                     key=lambda result: result['metrics']['mae_ms'], reverse=True)[:3]
    expected_top = '; '.join(f"{result['file']}:{result['metrics']['mae_ms']:.2f}ms" for result in highest)
    if summary.get('top_mae_files') != expected_top:
        raise ValueError(f'{algorithm}: highest-error TEST list differs from actual results')


def audit_results(audit):
    algorithm = audit['algorithm']
    if algorithm not in ('tt1', 'tt2', 'tt3'):
        raise ValueError('Only three primary algorithms are submitted')
    model = audit['model']
    audit_model_protocol(model, audit.get('model_lock_digest'))
    test = audit['test_results']
    all_results = audit['all_results']
    if tuple(result['file'] for result in test) != TEST_NAMES:
        raise ValueError('Notebook must evaluate all four test WAVs in the declared order')
    expected_names = {path.stem for split in ('train', 'test')
                      for path in (ROOT / 'data' / split).glob('*.wav')}
    if len(all_results) != 8 or {result['file'] for result in all_results} != expected_names:
        raise ValueError('All-dataset evaluation must include eight unique WAVs')
    reference = {row['file']: row for row in read_csv(
        ROOT / 'outputs/tables/all_all_dataset/test_metrics.csv') if row['algorithm'] == algorithm}
    for result in all_results + test:
        name = result['file']
        row = reference[name]
        if result['algorithm'] != algorithm:
            raise ValueError('Wrong algorithm in notebook result')
        scores = result['metrics']
        audit_metric_keys(scores, f'{algorithm}/{name}')
        if set(scores) != set(row) - {'file', 'split', 'algorithm'}:
            raise ValueError(f'{algorithm}/{name}: captured metric keys differ from schema2 benchmark')
        for key, value in scores.items():
            if key not in row:
                raise ValueError(f'{algorithm}/{name}: missing schema2 benchmark metric {key}')
            if isinstance(value, bool):
                if str(value).casefold() != row[key].casefold():
                    raise ValueError(f'{algorithm}/{name}/{key} differs')
            elif value is None or isinstance(value, (int, float)):
                close_number(value, row[key], f'{algorithm}/{name}/{key}')
            elif str(value) != row[key]:
                raise ValueError(f'{algorithm}/{name}/{key} differs')
        diagnostic = json.loads((ROOT / f'outputs/diagnostics/{algorithm}/{name}.json').read_text(encoding='utf-8'))
        compare_structure(result['final_regions'], diagnostic['final_regions'],
                          f'{algorithm}/{name}/final regions')
        compare_structure(result['predicted_boundaries'],
                          [edge for region in diagnostic['final_regions'] for edge in region],
                          f'{algorithm}/{name}/predicted boundaries')
        for key in ('low_ste_threshold', 'high_ste_threshold'):
            close_number(result['diagnostic'][key], diagnostic[key], f'{algorithm}/{name}/{key}', 1e-12)
    original = json.loads((ROOT / f'outputs/models/{algorithm}.json').read_text(encoding='utf-8'))
    audit_model_protocol(original, model_digest(original))
    compare_structure(model, original, f'{algorithm}/model/source/framing/noise')
    if algorithm == 'tt2':
        audit_train_sweep(audit)
    elif audit.get('tt2_w_sweep_rows') != []:
        raise ValueError('Non-histogram notebooks cannot contain a W sweep')
    manifest = audit['dataset_manifest']
    if len(manifest) != 8:
        raise ValueError('Need eight input provenance records')
    if {(entry['split'], entry['file']) for entry in manifest} != (
            {('train', name) for name in TRAIN_NAMES} | {('test', name) for name in TEST_NAMES}):
        raise ValueError('Input provenance must contain eight unique correctly split files')
    for entry in manifest:
        for extension, key in (('wav', 'wav_sha256'), ('lab', 'lab_sha256')):
            path = ROOT / 'data' / entry['split'] / f"{entry['file']}.{extension}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry[key]:
                raise ValueError(f'Input fingerprint differs: {path.name}')
    audit_test_summary(algorithm, test, audit.get('test_summary'))
    saved_summary = [row for row in read_csv(ROOT / 'outputs/tables/all/summary.csv')
                     if row['algorithm'] == algorithm]
    if len(saved_summary) != 1:
        raise ValueError('Python TEST benchmark needs one summary per algorithm')
    if set(audit['test_summary'][0]) != set(saved_summary[0]):
        raise ValueError('TEST saved summary metric keys differ')
    for key, value in audit['test_summary'][0].items():
        expected = saved_summary[0][key]
        if isinstance(value, bool):
            if str(value).casefold() != expected.casefold():
                raise ValueError(f'TEST saved summary/{key} differs')
        elif value is None or isinstance(value, (int, float)):
            close_number(value, expected, f'TEST saved summary/{key}')
        elif str(value) != expected:
            raise ValueError(f'TEST saved summary/{key} differs')
    return audit['test_summary'][0]['mean_file_mae_ms']


def audit_notebook(notebook, filename):
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    counts = [cell.execution_count for cell in code_cells]
    if not counts or any(not isinstance(count, int) or count <= 0 for count in counts):
        raise ValueError(f'{filename}: all code cells must have been executed')
    if counts != sorted(set(counts)):
        raise ValueError(f'{filename}: execution counts are not in fresh Run-All order')
    images = 0
    required_plot_cells = {
        ast.dump(ast.parse(f'plot_record(TEST_RECORDS_BY_FILE["{name}"], '
                           f'TEST_RESULTS_BY_FILE["{name}"])'), include_attributes=False): name
        for name in TEST_NAMES
    }
    plotted_tests = set()
    for cell in code_cells:
        # The inline plotting magic is the only IPython-only source syntax.
        source = '\n'.join(line for line in cell.source.splitlines() if not line.lstrip().startswith('%'))
        tree = ast.parse(source)
        plot_file = required_plot_cells.get(ast.dump(tree, include_attributes=False))
        png_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] in ('app', 'core', 'algorithms'):
                raise ValueError(f'{filename}: runtime project import remains')
            if isinstance(node, ast.Import) and any(alias.name.split('.')[0] in ('app', 'core', 'algorithms') for alias in node.names):
                raise ValueError(f'{filename}: runtime project import remains')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('exec', 'eval', '__import__', 'Audio'):
                raise ValueError(f'{filename}: source must be readable and contain no audio payloads')
        if re.search(r'^\s*%run\b', cell.source, re.MULTILINE):
            raise ValueError('Notebook cannot be a wrapper around .py files')
        for output in cell.outputs:
            if output.output_type == 'error':
                raise ValueError(f'{filename}: notebook has saved errors')
            for mime, value in output.get('data', {}).items():
                if mime.startswith('audio/') or mime == 'application/vnd.jupyter.widget-view+json':
                    raise ValueError('Submission contains an audio/widget payload')
                if mime == 'image/png':
                    decoded = base64.b64decode(value)
                    if not decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                        raise ValueError('Invalid embedded result figure')
                    images += 1
                    png_count += 1
        if plot_file is not None:
            if png_count != 1:
                raise ValueError(f'{filename}: {plot_file} test plot cell needs exactly one embedded PNG')
            if plot_file in plotted_tests:
                raise ValueError(f'{filename}: duplicate {plot_file} test plot cell')
            plotted_tests.add(plot_file)
    missing_plots = set(TEST_NAMES) - plotted_tests
    if missing_plots:
        raise ValueError(f'{filename}: missing exact test plot cells: '
                         + ', '.join(sorted(missing_plots)))
    if images != 5:
        raise ValueError(f'{filename}: expected four TEST PNGs and one algorithm illustration')
    audit = notebook.metadata.get('endpoint_execution')
    if not audit:
        raise ValueError(f'{filename}: missing fresh-kernel execution evidence')
    mean = audit_results(audit)
    return images, mean


def execute_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    # Re-execution always starts empty; a failed run does not overwrite the file.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.append(nbformat.v4.new_code_cell(AUDIT_CELL))
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=manager, timeout=600,
                            resources={'metadata': {'path': str(path.parent)}},
                            allow_errors=False, record_timing=False)
    print(f'Executing {path.name} in a fresh kernel...', flush=True)
    try:
        notebook = client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        if client.kc is not None:
            client.kc.stop_channels()
        manager.cleanup_resources()
    output_text = ''.join(output.get('text', '') for output in notebook.cells[-1].outputs
                          if output.output_type == 'stream')
    marker = output_text.find(AUDIT_PREFIX)
    if marker < 0:
        raise ValueError('Execution completed without auditable numerical results')
    evidence = json.loads(output_text[marker + len(AUDIT_PREFIX):].strip())
    notebook.cells.pop()  # Do not submit the external verification cell.
    notebook.metadata['endpoint_execution'] = evidence
    images, mean = audit_notebook(notebook, path.name)
    nbformat.write(notebook, path)
    print(f'{path.name}: 8/8 benchmark matches; {images} embedded plots; test mean MAE {mean:.2f} ms.', flush=True)


def package_notebooks(group_number=None):
    """Use documented STTNhom placeholder until the actual group index is supplied."""
    notebooks = [NOTEBOOK_DIR / f'THUAT_TOAN_{number}.ipynb' for number in range(1, 4)]
    for path in notebooks:
        audit_notebook(nbformat.read(path, as_version=4), path.name)
    group = f'{int(group_number):02d}' if group_number is not None else 'STTNhom'
    prefix = f'{group}-PhanDoanTiengNoiKhoangLang/'
    destination = ROOT / 'submission/JUPYTER_NOTEBOOKS_CODE_ONLY.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in notebooks:
            archive.write(path, prefix + path.name)
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        if len(names) != 3 or any(Path(name).suffix != '.ipynb' for name in names):
            raise ValueError('Notebook source package must contain exactly three .ipynb files')
        if archive.testzip() is not None:
            raise ValueError('Submission ZIP is corrupt')
        for path in notebooks:
            if archive.read(prefix + path.name) != path.read_bytes():
                raise ValueError('Packaged notebook differs from executed notebook')
    print(f'CODE package: {destination.name}; exactly 3 executed notebooks, no WAV/LAB/audio.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Check saved notebook outputs without running them')
    parser.add_argument('--group-number', type=int, help='Optional actual group index for STTnhom-TenDeTai folder naming')
    arguments = parser.parse_args()
    if arguments.group_number is not None and not 1 <= arguments.group_number <= 999:
        parser.error('--group-number must be between 1 and 999')
    for number in range(1, 4):
        path = NOTEBOOK_DIR / f'THUAT_TOAN_{number}.ipynb'
        if arguments.validate_only:
            images, mean = audit_notebook(nbformat.read(path, as_version=4), path.name)
            print(f'{path.name}: saved outputs valid, {images} figures, test mean MAE {mean:.2f} ms.')
        else:
            execute_notebook(path)
    package_notebooks(arguments.group_number)


if __name__ == '__main__':
    main()
