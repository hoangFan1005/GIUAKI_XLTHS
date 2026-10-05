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
NOTEBOOK_DIR = ROOT / 'notebooks'
TEST_NAMES = ('phone_F2', 'phone_M2', 'studio_F2', 'studio_M2')
AUDIT_PREFIX = '__ENDPOINT_NOTEBOOK_AUDIT__'

AUDIT_CELL = r'''
import json
def _compact_result(result):
    return {key: result[key] for key in (
        'file', 'algorithm', 'metrics', 'final_regions', 'ground_truth_regions',
        'predicted_boundaries', 'ground_truth_boundaries', 'diagnostic')}
_audit = {
    'algorithm': ALGORITHM,
    'model': MODEL,
    'test_results': [_compact_result(result) for result in TEST_RESULTS],
    'all_results': [_compact_result(result) for result in ALL_RESULTS],
    'test_summary': TEST_SUMMARY,
    'dataset_manifest': DATASET_MANIFEST,
}
print('__ENDPOINT_NOTEBOOK_AUDIT__' + json.dumps(_audit, ensure_ascii=False, allow_nan=False))
'''


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


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
        for key in ('mae_ms', 'rmse_ms', 'start_error_ms', 'end_error_ms',
                    'predicted_start_s', 'predicted_end_s', 'gt_start_s', 'gt_end_s'):
            close_number(scores[key], row[key], f'{algorithm}/{name}/{key}')
        for key in ('ground_truth_region_count', 'predicted_region_count',
                    'missing_region_count', 'extra_region_count'):
            if scores[key] != int(row[key]):
                raise ValueError(f'{algorithm}/{name}/{key} differs')
        if scores['status'] != row['status']:
            raise ValueError(f'{algorithm}/{name}/status differs')
        diagnostic = json.loads((ROOT / f'outputs/diagnostics/{algorithm}/{name}.json').read_text(encoding='utf-8'))
        if len(result['final_regions']) != len(diagnostic['final_regions']):
            raise ValueError('Final region count differs from benchmark')
        for actual, expected in zip(result['final_regions'], diagnostic['final_regions']):
            for edge, (left, right) in enumerate(zip(actual, expected)):
                close_number(left, right, f'{algorithm}/{name}/region edge {edge}', 1e-12)
        for key in ('low_ste_threshold', 'high_ste_threshold'):
            close_number(result['diagnostic'][key], diagnostic[key], f'{algorithm}/{name}/{key}', 1e-12)
    model = audit['model']
    original = json.loads((ROOT / f'outputs/models/{algorithm}.json').read_text(encoding='utf-8'))
    for key in ('threshold', 'muSil', 'stdSil', 'muSp', 'stdSp', 'W', 'training_selected_W'):
        if key in original:
            close_number(model[key], original[key], f'{algorithm}/model/{key}', 1e-12)
    for key in ('noise_mean', 'noise_std', 'noise_q95', 'noise_upper', 'noise_count'):
        close_number(model['endpoint_noise'][key], original['endpoint_noise'][key],
                     f'{algorithm}/noise/{key}', 1e-12)
    if algorithm == 'tt2':
        selection = model['W_selection_test']
        if selection['candidate_W'] != [float(weight) for weight in range(1, 51)]:
            raise ValueError('TT2 must sweep all integer W values 1–50')
        if set(selection['evaluated_files']) != set(TEST_NAMES):
            raise ValueError('TT2 calibration must use all four test LABs')
        if selection['evaluation_protocol'] != 'test_tuned_not_independent':
            raise ValueError('TT2 must disclose test-set W tuning')
    manifest = audit['dataset_manifest']
    if len(manifest) != 8:
        raise ValueError('Need eight input provenance records')
    for entry in manifest:
        for extension, key in (('wav', 'wav_sha256'), ('lab', 'lab_sha256')):
            path = ROOT / 'data' / entry['split'] / f"{entry['file']}.{extension}"
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry[key]:
                raise ValueError(f'Input fingerprint differs: {path.name}')
    audit_test_summary(algorithm, test, audit.get('test_summary'))
    return sum(result['metrics']['mae_ms'] for result in test) / len(test)


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
        cell_has_png = False
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
                    cell_has_png = True
        if plot_file is not None:
            if not cell_has_png:
                raise ValueError(f'{filename}: {plot_file} test plot cell needs an embedded PNG')
            if plot_file in plotted_tests:
                raise ValueError(f'{filename}: duplicate {plot_file} test plot cell')
            plotted_tests.add(plot_file)
    missing_plots = set(TEST_NAMES) - plotted_tests
    if missing_plots:
        raise ValueError(f'{filename}: missing exact test plot cells: '
                         + ', '.join(sorted(missing_plots)))
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
