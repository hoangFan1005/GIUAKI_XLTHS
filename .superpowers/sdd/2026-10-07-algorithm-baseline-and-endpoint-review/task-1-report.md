# Task1 implementation report — 07/10/2026

Task1 is complete on `codex/endpoint-modes`. Scope: current-artifact documentation, four assertions in `tests/test_training_protocol.py`, and one execution evidence directory `outputs/tables/endpoint_modes_oct07/`. No detector/model numeric source, canonical models/tables/diagnostics/predictions, notebooks, slides, source/data inputs or ZIP files were regenerated or changed. No push was performed.

## Baseline captured before prose edits

`baseline.json` was captured using `.venv/Scripts/python.exe` from all three saved enhanced models, without fitting. It contains eight records with complete features/labels, 24 detections with full masks, FINAL regions, all numeric metrics/status, threshold diagnostics and explicit T/W/LOW/HIGH, all saved model fields, Python/platform/package versions, Git HEAD, and 148 SHA256 entries. It is one reference evidence file, not a project snapshot.

Read-only inference verified exact equality against all 24 saved diagnostic JSON objects and every metric field in the current 24-row metrics CSV. FINAL regions, MAE and status also match all 24 `hotfix_oct06/regression.csv` acceptance rows. Mean four TEST MAE is TT1 `19.999999999999993`, TT2 `13.749999999999986`, TT3 `12.50000000000001` ms (displayed 20.00/13.75/12.50). All 38 protected Source/data/slides/legacy ZIP hashes recorded in the Oct07 evidence match. After documentation/tests/full-suite execution, all 145 captured non-document artifacts retain their hashes; three owned documents are the intentional differences. Details and baseline digest are in `task1-verification.json`.

Capture command (scratch script retained beside this report, not a production helper):

```powershell
$env:PYTHONUTF8='1'; $env:MPLBACKEND='Agg'
.venv/Scripts/python.exe .superpowers/sdd/2026-10-07-algorithm-baseline-and-endpoint-review/capture-baseline.py
```

The capture writes exclusively using file mode `x` and refuses to overwrite an existing baseline. It must not be rerun to replace this baseline after later implementation tasks.

## Assertions and RED/GREEN evidence

Read `superpowers:test-driven-development/SKILL.md` and `writing-good-tests.md`. The brief explicitly requires README assertions, overriding the general skill advisory that human prose normally earns no tests. Breaks covered: accidentally labeling the TRAIN W sweep as TEST; ambiguous casefold metric headers/schema drift; obsolete metric aliases in current guidance; and treating Oct05 acceptance/Git history as current. Tests consume real existing CSVs and documents, without mocks or regenerated artifacts. The existing notebook protocol tests already check model-lock and TRAIN sweep behavior and were rerun without modification.

RED command:

```powershell
$env:PYTHONUTF8='1'; $env:MPLBACKEND='Agg'
.venv/Scripts/python.exe -m unittest tests.test_training_protocol.CurrentArtifactDocumentationTests -v
```

Result: **4 tests, 2 passed, 2 expected assertion failures**, exit 1, 0.003 s. `test_outputs_readme_labels_current_sweep_as_train_and_schema_two` failed because the linked W-sweep line said `4 test` and had no `TRAIN`. `test_submission_readme_separates_current_acceptance_from_oct05_history` failed because the README had no dated history section and still presented 98/98 as current. These were assertion failures, not collection/runtime errors. Existing artifact header/sweep checks passed immediately as characterization coverage.

GREEN command:

```powershell
$env:PYTHONUTF8='1'; $env:MPLBACKEND='Agg'
.venv/Scripts/python.exe -m unittest tests.test_training_protocol tests.test_notebook_protocol -v
```

Result: **27/27 passed**, exit 0, 4.427 s, including both formerly failing README assertions.

Full-suite command:

```powershell
$env:PYTHONUTF8='1'; $env:MPLBACKEND='Agg'
.venv/Scripts/python.exe -m unittest discover -s tests -v 2>&1 | Out-File -Encoding utf8 outputs/tables/endpoint_modes_oct07/task1-full-suite.txt
```

Result: **137/137 passed**, exit 0, 11.152 s. This includes the accepted existing 133 tests plus four added assertions. Complete per-test output is in `task1-full-suite.txt`; there were no omitted suite failures. One initial sandboxed Python-launch attempt failed with Windows `Access is denied`; the authorized launcher commands then succeeded with `require_escalated`. No alternate interpreter was used for project test/inference execution.

## Documentation changes and self-review

`outputs/README.md` now identifies the current 200-row W sweep as four TRAIN files, distinguishes FINAL W20 from candidate diagnostics, names schema 2 primary `mae_ms` / `rmse_ms` and separate tolerance metrics, retains historical TEST exposure and legacy provenance, and links the accepted 133-test result and locked baseline.

`submission/README.md` now leads with Oct07 acceptance (133 tests, unchanged 24 results, current source/ZIP byte audit), TRAIN/schema 2, historical exposure and delivery limitations. The original Oct05 text and branch/push statements remain under a dated historical section. `reports/KET_QUA_HIEN_TAI.md` adds the baseline reference and explicitly labels its Oct05 section as history. Metric formulas/names and historical numbers were preserved.

Self-review inspected the source/test/doc diff, baseline content and count, full-suite log and preservation hash probe. `git diff --check` passes. Existing line endings were preserved on unchanged prose to keep the diff limited; added/changed lines have no trailing whitespace. Assertions use literal hand-checked TRAIN names and accepted test count. Numerical production/model files do not appear in the change list. Current baseline hashes also cover canonical output artifacts, saved notebooks and the CODE ZIP for later regression gates.

## Remaining concerns

No new Task1 blocker or numeric regression. Existing limitations remain: five late END cases, historical TEST exposure (no new independent holdout), estimated gap support rather than a physical silence guarantee, speech duration measured as support span, FINAL low-direction Gaussian rejection, unverified Python3.10 runtime, placeholders and old slide/Python ZIP versions. No new core model exists at this Task1 gate.

Stage only the three owned documents, `tests/test_training_protocol.py`, this report and the three evidence files (`baseline.json`, `task1-verification.json`, `task1-full-suite.txt`). The scratch capture script is excluded from the commit. Local commit is permitted; no push.
