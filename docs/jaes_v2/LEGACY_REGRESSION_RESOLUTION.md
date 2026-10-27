# Phase 2 inherited regression resolution

Date: 25 October 2026. All three failures were reproduced before editing. Evidence: `outputs/jaes_v2/audit/phase2_before/validation/`. Source backups and original hashes are in `phase2_before/source_backups/` and `snapshot.json`.

| Test | Root cause / classification | Minimal fix | Behavioral effect |
|---|---|---|---|
| `Lava5Tests.test_display_missing_results_is_optional` | PRODUCTION_BUG: optional `spec=None` dereferenced in packaged fallback | Guard `None` in public and packaged benchmark-card functions | Missing specification yields no optional card; valid model paths unchanged |
| `DecisionDisplayTest.test_single_score_chart` | MOCK_CONTRACT_MISMATCH: test patched obsolete `st.pyplot`, production calls `render_interactive_plot` | Patch current renderer; retain exact score position, threshold and axis assertions | No production chart change |
| `StreamlitUITest.test_available_models_do_not_show_missing_artifact_panel` | STALE_TEST_EXPECTATION: test prohibited every expander, including existing measured benchmark | Permit only the measured benchmark label and still reject diagnostic/unavailable panels | No production UI change |

Files changed: `src/lava/benchmark_display.py`, `tests/test_decision_display.py`, `tests/test_streamlit_ui.py`. `app.py`, decisions, score semantics, thresholds and checkpoint loading did not change.

The historical freeze snapshot was **not overwritten**. A separate `outputs/jaes_v2/audit/phase2_baseline_amendment.json` records the original snapshot hash, exact before/after file hashes and reasons. Preflight only permits these three enumerated changes and still verifies every model, manifest, paper and historical benchmark hash. No generic path exemption or automatic acceptance of future source changes was added.

Measured resolution suite: legacy 37/40 → **40/40**; V2 at that point **19/19**. See `outputs/jaes_v2/audit/phase2_regression_fixed/validation.json`. The final expanded V2 results are recorded separately under `phase2_validation_final/`; historical failed reports remain intact.
