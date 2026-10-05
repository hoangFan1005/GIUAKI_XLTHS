"""Notebook generator and external evidence audit contracts; no canonical outputs."""
import ast
import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import wave
from unittest.mock import patch

from tools import build_notebooks as builder
from tools import run_notebooks as runner
from app.weight_selection import select_final_weight
from app import pipeline
import nbformat


def tiny_dataset(root):
    for split, suffix in (("train", "1"), ("test", "2")):
        folder = root / split
        folder.mkdir()
        for prefix in ("phone_F", "phone_M", "studio_F", "studio_M"):
            path = folder / (prefix + suffix + ".wav")
            values = [int((12000 if 200 <= i < 650 else 40) * math.sin(i * .47))
                      for i in range(1000)]
            with wave.open(str(path), "wb") as handle:
                handle.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
                handle.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))
            path.with_suffix(".lab").write_text("0 0.2 sil\n0.2 0.65 v\n0.65 1 sil\n")


class NotebookGenerationTests(unittest.TestCase):
    def generated(self, algorithm):
        try:
            return builder.build_notebook(algorithm)
        except (ValueError, StopIteration) as error:
            self.fail(f"Generator must adapt current public core interfaces: {error}")

    def test_standalone_cells_fit_and_lock_before_test_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / 'data'
            dataset.mkdir()
            tiny_dataset(dataset)
            for algorithm in ("tt1", "tt2", "tt3"):
                with self.subTest(algorithm=algorithm):
                    notebook = self.generated(algorithm)
                    namespace = {}
                    checked = []
                    for cell in notebook["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        source = "".join(cell["source"])
                        tree = ast.parse(source)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ImportFrom):
                                self.assertNotIn((node.module or "").split(".")[0],
                                                 ("app", "core", "algorithms"))
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                                self.assertNotIn(node.func.id, ("exec", "eval", "__import__", "Audio"))
                        if "DATA_ROOT = find_data_root" in source:
                            namespace["DATA_DIR"] = dataset
                            real_loader = namespace["load_dataset_folder"]
                            def guarded_loader(data_root, split):
                                if split == "test":
                                    self.assertIn("MODEL_LOCK_DIGEST", namespace,
                                                  "TEST reads must follow TRAIN calibration and lock")
                                    self.assertEqual(namespace["MODEL_LOCK_DIGEST"],
                                                     namespace["model_digest"](namespace["MODEL"]))
                                    checked.append(split)
                                return real_loader(data_root, split)
                            namespace["load_dataset_folder"] = guarded_loader
                        with contextlib.redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"), \
                                patch("IPython.display.display"):
                            exec(compile(source, "<generated-notebook>", "exec"), namespace)
                    self.assertEqual(checked, ["test"])
                    model = namespace["MODEL"]
                    self.assertEqual(model["schema_version"], 2)
                    self.assertEqual(model["parameter_selection_set"], "train")
                    self.assertEqual(model["evaluation_protocol"], "train_selected_reused_test")
                    self.assertIs(model["historical_test_exposure"], True)
                    self.assertEqual(namespace["MODEL_LOCK_DIGEST"], namespace["model_digest"](model))
                    self.assertEqual(len(namespace["DATASET_MANIFEST"]), 8)
                    for row in namespace["ALL_METRIC_ROWS"]:
                        self.assertEqual(len(row), len({key.casefold() for key in row}))
                        self.assertNotIn("boundary_MAE_ms", row)
                        self.assertIn("tolerance_boundary_mae_ms", row)
                    if algorithm == "tt2":
                        self.assertEqual(len(namespace["TT2W_SWEEP_ROWS"]), 200)
                        self.assertEqual(model["W_selection_train"]["selection_set"], "train")
                        self.assertEqual(model["candidate_cleanup_records"]["full_pipeline_records"], 4)
                        self.assertNotIn("W_selection_test", model)
                    with self.assertRaisesRegex(ValueError, "endpoint_noise"):
                        namespace["detect_regions"](algorithm, {}, 1., {})
                    with self.assertRaisesRegex(ValueError, "only"):
                        namespace["detect_regions"]("another", {}, 1., model)
                    original = json.dumps(model, sort_keys=True)
                    with contextlib.redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show"), \
                            patch("IPython.display.display"):
                        namespace["demo_one_file"](str(dataset / "test/phone_F2.wav"))
                    self.assertEqual(original, json.dumps(model, sort_keys=True))
                    self.audit_with_core_benchmark(namespace, root)

    def audit_with_core_benchmark(self, namespace, root):
        """Use tiny real core runs to catch generated metadata/scorer drift."""
        algorithm = namespace['ALGORITHM']
        with patch.multiple(pipeline, TRAIN_DIR=root / 'data/train', TEST_DIR=root / 'data/test'):
            train = pipeline.prepare_records(pipeline.load_audio_folder(root / 'data/train'))
            test = pipeline.prepare_records(pipeline.load_audio_folder(root / 'data/test'))
            model, rows = pipeline.fit_training_model(algorithm, train)
        results = [pipeline.predict_and_score(algorithm, record, model) for record in train + test]
        metric_rows = [dict(file=result['file'], algorithm=algorithm, **result['metrics']) for result in results]
        pipeline.write_csv(root / 'outputs/tables/all_all_dataset/test_metrics.csv', metric_rows)
        pipeline.write_csv(root / 'outputs/tables/all/summary.csv', pipeline.summarize(metric_rows[4:]))
        pipeline.write_json(root / f'outputs/models/{algorithm}.json', model)
        for result in results:
            pipeline.write_json(root / f'outputs/diagnostics/{algorithm}/{result["file"]}.json', result['diagnostic'])
        if algorithm == 'tt2':
            pipeline.write_csv(root / 'outputs/tables/tt2_w_selection/sweep.csv', rows)
            pipeline.write_json(root / 'outputs/tables/tt2_w_selection/selection.json', model['W_selection_train'])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(runner.AUDIT_CELL, namespace)
        evidence = json.loads(stream.getvalue().split(runner.AUDIT_PREFIX, 1)[1])
        with patch.object(runner, 'ROOT', root):
            self.assertIsInstance(runner.audit_results(evidence), float)
            damaged = copy.deepcopy(evidence)
            damaged['all_results'][0]['metrics']['mae_ms'] += 5
            with self.assertRaisesRegex(ValueError, 'mae_ms'):
                runner.audit_results(damaged)
            damaged = copy.deepcopy(evidence)
            damaged['all_results'][0]['metrics'].pop('frame_f1')
            with self.assertRaisesRegex(ValueError, 'metric|keys'):
                runner.audit_results(damaged)


def executed_plot_notebook():
    # Minimal valid PNG header: the auditor promises signature checks only.
    output = nbformat.v4.new_output('display_data', data={'image/png': 'iVBORw0KGgo='})
    cells = [nbformat.v4.new_code_cell('pass', execution_count=1, outputs=[output])]
    for index, name in enumerate(runner.TEST_NAMES, 2):
        cells.append(nbformat.v4.new_code_cell(
            f'plot_record(TEST_RECORDS_BY_FILE["{name}"], TEST_RESULTS_BY_FILE["{name}"])',
            execution_count=index, outputs=[copy.deepcopy(output)]))
    return nbformat.v4.new_notebook(cells=cells, metadata={'endpoint_execution': {'fixture': True}})


class NotebookAuditTests(unittest.TestCase):
    def test_auditor_finds_current_selection_helper_without_project_on_sys_path(self):
        program = (
            "import runpy, sys\n"
            "runner = runpy.run_path(sys.argv[1])\n"
            "try:\n"
            "    runner['audit_train_sweep']({'model': {'W_selection_train': {}}})\n"
            "except ValueError as error:\n"
            "    print(error)\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            completed = subprocess.run([sys.executable, '-I', '-c', program, str(Path(runner.__file__).resolve())],
                                       cwd=folder, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('200 rows', completed.stdout)

    def test_saved_notebook_requires_exactly_one_png_in_each_test_cell(self):
        notebook = executed_plot_notebook()
        with patch.object(runner, 'audit_results', return_value=10.):
            self.assertEqual(runner.audit_notebook(notebook, 'tiny.ipynb'), (5, 10.))
            duplicate = copy.deepcopy(notebook)
            duplicate.cells[1].outputs.append(copy.deepcopy(duplicate.cells[1].outputs[0]))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                runner.audit_notebook(duplicate, 'duplicate.ipynb')
            missing = copy.deepcopy(notebook)
            missing.cells[0].outputs.clear()
            with self.assertRaisesRegex(ValueError, 'illustration'):
                runner.audit_notebook(missing, 'missing.ipynb')

    def test_train_sweep_audit_requires_200_rows_and_validates_selection_objective(self):
        self.assertTrue(callable(getattr(runner, "audit_train_sweep", None)), "TRAIN sweep audit is required")
        rows = [dict(filename=name, W=float(w), final_region_mae_ms=10.,
                     ground_truth_region_count=1, predicted_region_count=1,
                     final_regions="[(0.2, 0.65)]")
                for w in range(1, 51)
                for name in ("phone_F1", "phone_M1", "studio_F1", "studio_M1")]
        selection = select_final_weight(rows)
        selection.update(evaluated_files=["phone_F1", "phone_M1", "studio_F1", "studio_M1"])
        audit = dict(model=dict(W=20., finalW=20., tie_preference_W=20., W_selection_train=selection),
                     tt2_w_sweep_rows=rows)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "outputs/tables/tt2_w_selection"
            target.mkdir(parents=True)
            with (target / "sweep.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (target / "selection.json").write_text(json.dumps(selection))
            with patch.object(runner, "ROOT", root):
                runner.audit_train_sweep(audit)
                damaged = copy.deepcopy(audit)
                damaged["tt2_w_sweep_rows"][0]["final_region_mae_ms"] = 0.
                with self.assertRaisesRegex(ValueError, "sweep|MAE|mae"):
                    runner.audit_train_sweep(damaged)
                damaged = copy.deepcopy(audit)
                damaged["model"]["W_selection_train"]["selected_W"] = 1.
                with self.assertRaisesRegex(ValueError, "selection|selected|objective"):
                    runner.audit_train_sweep(damaged)
                damaged = copy.deepcopy(audit)
                damaged["tt2_w_sweep_rows"].pop()
                with self.assertRaisesRegex(ValueError, "200|rows"):
                    runner.audit_train_sweep(damaged)

    def test_metric_audit_rejects_casefold_alias_and_uses_primary_mae(self):
        self.assertTrue(callable(getattr(runner, "audit_metric_keys", None)), "Metric-key audit is required")
        bad = {"mae_ms": 140., "MAE_MS": 1., "tolerance_boundary_mae_ms": 1.}
        with self.assertRaisesRegex(ValueError, "case|unique|duplicate"):
            runner.audit_metric_keys(bad, "fixture")
        runner.audit_metric_keys({"mae_ms": 140., "tolerance_boundary_mae_ms": 1.}, "fixture")

    def test_lock_audit_rejects_model_mutation_and_legacy_test_selection(self):
        self.assertTrue(callable(getattr(runner, "audit_model_protocol", None)), "Frozen-model protocol audit is required")
        model = dict(schema_version=2, metrics_schema_version=2, parameter_selection_set="train",
                     evaluation_protocol="train_selected_reused_test", historical_test_exposure=True,
                     training_files=list(runner.TRAIN_NAMES), threshold=.1)
        digest = runner.model_digest(model)
        runner.audit_model_protocol(model, digest)
        changed = dict(model, threshold=.2)
        with self.assertRaisesRegex(ValueError, "digest|lock"):
            runner.audit_model_protocol(changed, digest)
        legacy = dict(model, W_selection_test={})
        with self.assertRaisesRegex(ValueError, "TEST|test|legacy"):
            runner.audit_model_protocol(legacy, runner.model_digest(legacy))


if __name__ == "__main__":
    unittest.main()
