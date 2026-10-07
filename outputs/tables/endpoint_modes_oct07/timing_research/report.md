# Task 4 timing research report

Status: research and focused checks complete; ready for root's independent review/commit. Full suite, fresh enhanced 24-case regression and global protected-artifact hashes are deferred to coordinated acceptance while Task 5 source edits continue. No fresh TEST regression is claimed by Task 4.

## Scope and execution

Owned implementation: core/decision_timing.py; tests/test_decision_timing.py; tools/check_endpoint_robustness.py; tests/test_endpoint_robustness.py. The opt-in `--timing-research` route writes all generated evidence beneath outputs/tables/endpoint_modes_oct07/timing_research/. The legacy diagnostic command remains usable. No production detector/calibration/default/CLI/notebook/slides/canonical artifact changes were made by Task 4. No TEST WAV/LAB was opened, loaded, or used for tuning.

Requirements and the bounded preflight matrix were read before implementation. TDD evidence: red.txt shows helper missing assertions, branch-red.txt shows research predictor/matrix missing assertions, fold-red.txt shows the real fold metadata integration failure. The helper and shared scorer then passed focused checks. A broad text edit caused a NameError in TT2 sweep metadata; the real TRAIN fold integration test caught it, and it was corrected before evidence generation. focused-green.txt: 21 tests pass, including legacy robustness, literal midpoint coverage, validation, exact estimated 190/200/210ms rules, two Fs, audio edges/short tails, estimated 95/100/105ms span filter, explicit unsupported LOW=T1 zero, and real four-fold budget/provenance/full-undefined checks.

Run command: `.venv/Scripts/python.exe tools/check_endpoint_robustness.py --timing-research` (UTF-8; MPLBACKEND=Agg). Environment's default unified exec helper fails setup refresh; workspace commands were executed with the approved escalation. run.txt records four completed folds and the declared counts.

## Protocol and geometry

Four distinct TRAIN records, deterministic name ordering. Each held-out record is scored with three-file native fits, explicitly fitted per core/enhanced mode. Native TT1/TT3/noise fits are shared within each mode/fold. Each of six TT2 branches separately evaluates W=1..50 on the three fit files: 3600 exact branch evaluations plus 24 held-out TT2 scores. The native support W selection is explicitly preliminary for experiments; each branch replaces it with selection from the shared timing_predictor, region_endpoint_metrics and select_final_weight. The selected W and finalW agree, and model/selection records declare trial_id, geometry, LOW, mode, filters, fit files, predictor, scorer and hashes. No held-out score selects W. Model manifests preserve historical_test_exposure=true.

The six common trials contain three algorithms, plus one enhanced TT1-only LOW=T1 trial: 76 held-out rows. LOW=T1 has no floor; invalid zero thresholds are explicitly unsupported (none occurred in the observed fitted folds).

Decision cells use consecutive actual-center midpoints and outer edges 0/duration. Frame windows remain unchanged: 400/160 samples at 16k; 1102/441 at 44.1k, giving 24.988662ms/10ms. Half-sample midpoint boundaries are retained. The last cell includes the incomplete unanalysed tail. Selected geometry consistently drives candidate timing, START/END, merging, span filtering, final mask whole-cell containment, and last-native-active END. Support baselines are asserted equal to production FINAL regions and masks on held-out TRAIN using identical params (24 baseline scores).

## Primary held-out findings

Every row retains signed per-boundary START/END errors, region_pairs, region counts and status. Primary MAE/RMSE remains undefined for mismatched counts; valid-only averages are separate columns. RMSE summaries explicitly mean per-file RMSE, not pooled-boundary RMSE.

| algorithm / trial | full mean MAE ms | full mean file RMSE ms | invalid folds |
|---|---:|---:|---:|
| TT1 core support 0 | undefined | undefined | 1 |
| TT1 core cells 0 | undefined | undefined | 1 |
| TT1 core support 100 | 134.997 | 148.184 | 0 |
| TT1 enhanced support 100 | 54.997 | 73.454 | 0 |
| TT1 enhanced cells 100 | 51.875 | 70.254 | 0 |
| TT1 enhanced support 0 | 54.997 | 73.454 | 0 |
| TT1 enhanced LOW=T1 support 100 | 106.247 | 142.486 | 0 |
| TT2 core support 0 / 100 | undefined | undefined | 1 each |
| TT2 core cells 0 | undefined | undefined | 1 |
| TT2 enhanced support 100 / 0 | 21.247 | 25.804 | 0 |
| TT2 enhanced cells 100 | 20.000 | 25.502 | 0 |
| TT3 core/enhanced support baselines / alternate span | 20.000 | 24.054 | 0 |
| TT3 core/enhanced cells | 20.000 | 24.011 | 0 |

No count regressions against matching same-mode support baseline occurred. There are ten MAE fold regressions; all 52 alternate fold comparisons, validity transitions, count flags, MAE and RMSE deltas/flags are in fold_comparisons.csv. Undefined core means cannot be replaced by valid subset averages when recommending a convention.

| trial | algorithm | held-out TRAIN file | delta MAE ms |
|---|---|---|---:|
| core_cells_0 | TT3 | phone_F1 | +7.500 |
| enhanced_cells_100 | TT2 | phone_F1 | +7.500 |
| enhanced_cells_100 | TT3 | phone_F1 | +7.500 |
| enhanced_tt1_t1_support_100 | TT1 | phone_M1 | +205.000 |
| core_cells_0 | TT1 | studio_M1 | +2.506 |
| core_cells_0 | TT2 | studio_M1 | +7.494 |
| core_cells_0 | TT3 | studio_M1 | +7.494 |
| enhanced_cells_100 | TT1 | studio_M1 | +2.506 |
| enhanced_cells_100 | TT2 | studio_M1 | +2.506 |
| enhanced_cells_100 | TT3 | studio_M1 | +7.494 |

## Physical characterization

Four-TRAIN locked production models are distinct from the three-fit fold models. Synthetic rows declare this threshold source, actual thresholds and model hashes. The native production W is frozen for this waveform characterization; it is not claimed to optimize each synthetic/cell branch. These are physical characterization scores, separate from estimated duration-rule tests.

The original 48 physical gap waveforms are retained unchanged: 190/200/210/250ms, two sample rates and six phases. 48 additional fixed LOW=.1/HIGH=.5 cell-oracle rows are exported independently from the old support oracle. Add eight physical 75/95/100/105ms burst cases, and eight start/end/both-edge/weak-tail cases. Across 19 declared algorithm/trial combinations this produces 1216 locked-model synthetic rows.

Fixed support oracle: all 24 physical 200/210ms cases still merge and primary MAE is undefined. Fixed cells oracle: 23/24 still merge; only one 210ms case retains both regions. All declared locked-model branches merge all 24 physical 200/210ms cases each. The physical GT was not changed, and the 200ms merge threshold was not reduced. estimated_cell_gap_ms is independently derived from integer-overlap LOW runs and literal sample centers; support_gap_ms retains its original meaning.

Cells therefore do not repair physical silence near 200ms: analysis windows that overlap speech can classify portions of physical silence as speech. A separate estimator refinement based on TRAIN/waveform evidence would be needed for that objective. Audio edge extension and weak tails are retained as characterization outcomes, with signed errors/count mismatches, rather than asserted to match physical GT.

## Recommendation and gates

Keep the production convention and LOW policy pending user review. Cell averages slightly improve enhanced TT1/TT2 here, but several folds worsen, TT3 mean MAE is unchanged, core TT1/TT2 remain undefined, and physical 200/210ms failures persist. LOW=T1 is not supported as a general improvement: TT1 phone_M1 worsens by 205ms and full mean MAE increases from 54.997 to 106.247ms. No automatic timing or LOW selection was applied.

protected_source_hashes.json records ten unchanged source/function hashes against 47cc00b: core features/endpoints/postprocess/metrics, all three algorithm modules, and ASTs of algorithm_decision/detect_regions/fit_training_model. Task 3 enhanced24 evidence remains historical baseline only. Root will record fresh enhanced24 and protected artifact/full suite gates after Task 5 settles.

Evidence inventory: manifest.json (all locked models/provenance/input hashes), heldout_scores.csv, heldout_summary.csv, tt2_branch_fit_sweep.csv, fold_comparisons.csv, locked_model_waveforms.csv, fixed_oracle_support.csv, fixed_oracle_cells.csv, protected_source_hashes.json, run.txt, red/green logs. Scratch scripts remain under the existing SDD folder, outside table evidence.

## Whitespace acceptance follow-up

At root's request, normalized the four owned Python files to UTF-8 without BOM and LF line endings, removed trailing spaces and repeated EOF blank lines. Logic was unchanged. Scoped `git diff --check` is clean (scoped-diffcheck.txt); the same 21 focused tests pass again (focused-green.txt). No staging or commit performed.

## Review P3 provenance follow-up

The independent review passed specification and approved quality with a provenance lookup nit. Addressed by adding manifest `synthetic_branch_configs`, a dictionary keyed by the exact waveform-row `model_sha256`. Its 19 entries contain model_sha256, algorithm, trial_id, source_native_model_sha256 and the actual branch model. The six original `synthetic_models` native calibration entries are retained unchanged, including locked TRAIN W selection provenance.

The manifest was updated by deterministic reconstruction from its already saved six native models and trial definitions. No native refit, full study, synthetic scoring or audio read occurred for the update. All 1216 existing waveform row hashes resolve directly, and entry algorithm/trial matches the row. All seven existing CSV files have identical SHA-256 bytes before/after; provenance-reconstruction-check.json contains both digest inventories. Evidence does not change numerical findings or any production setting.

TDD: provenance-red.txt first fails for the missing reconstruction helper; provenance-green.txt passes the independent literal branch-model hash lookup. All 22 focused tests pass again (provenance-focused-green.txt); scoped diff check clean (provenance-diffcheck.txt). No staging or commit performed. Source fix is scoped to tools/check_endpoint_robustness.py and tests/test_endpoint_robustness.py on top of Task 4 commit 27dc471.
