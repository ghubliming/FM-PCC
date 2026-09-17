# INBOX — open cross-draft items

Status: ⏳ open · ✅ done (target version) · ❌ declined (reason). Newest first within each target.

## → v2
| status | from | item | note |
| :-- | :-- | :-- | :-- |
| ⏳ | v3.19 · 2026-09-17 | After the v2.17 sync `tools/check.py` warns: a `tabular` row in `04_method.tex` is ≈ 14.9 cm (text width 14.7 cm) — use `tabularx` with an `L` column. Also: DPCC's protocol is **5 seeds × 2 episodes = 10 per geometry** (released `n_trials: 2`), not 50 — check any protocol wording in Ch 1–4 | `v3/tools/check.py` output; `aux_repo/dpcc/config/projection_eval.yaml` |
| ⏳ | v3.20 · 2026-09-17 | Apply the author-approved mechanism names in Ch 1–4 and the abstract: **instantaneous-velocity matching**, **analytic average-velocity matching**, **consistency-interpolated average-velocity matching**; rename the §4 title but keep `sec:method:alphaflow` | [`FROM_v3_20260917_alphaflow_naming.md`](to_v2/FROM_v3_20260917_alphaflow_naming.md) |
| ⏳ | v3.18 · 2026-09-17 | Ch 4 rate table and equation: "33 Hz plan / 100 Hz inner" are simulated-time intervals (lock-step; planning takes 9–4878 ms wall clock) — reword as intervals of simulated time | [`FROM_v3_20260917_uav_rates_simulated_time.md`](to_v2/FROM_v3_20260917_uav_rates_simulated_time.md) |
| ⏳ | v3.18 · 2026-09-17 | Projection cost resolved: the 1.86–3.57× figure was UAV-corridor data misattributed to avoiding; D3IL-avoiding K3 comparison is candidate-matched (4 vs 4). A consistent cost sentence is offered | [`FROM_v3_20260917_projection_cost_resolved.md`](to_v2/FROM_v3_20260917_projection_cost_resolved.md) |
| ✅ v2.17 | v3.16b · 2026-09-16 | Environment names when several appear together: D3IL-avoiding, D3IL-aligning, UAV-corridor, UAV-pillars, UAV-s-curve — apply in Ch 1–4 where environments are listed together; add `UAV` to the acronym list | `Auxiliary/Naming/TRANSLATION_…` § environment names |
| ✅ v2.17 | v3.16 · 2026-09-16 | v3 preamble now sets `secnumdepth` to subsubsection (Ch 6 nests numbered parts); fold into `settings.tex` at merge. Ch 4's `\subsubsection*` are unaffected | `v3/parts/00_preamble_v3.tex`, end |
| ✅ v2.17 | v3.14 · 2026-09-16 | Training is **not** uniform across models (batch, lr, action weight, EMA on avoiding/alignment); check any "architecture-matched / held fixed" wording in Ch 1–4 | [`FROM_v3_20260916_training_not_uniform.md`](to_v2/FROM_v3_20260916_training_not_uniform.md) |
| ✅ v2.17 | v3.11 · 2026-09-15 | `thesis_v2.tex:332` `\hole` — result sentence for the endpoint-projection contribution, supplied | [`FROM_v3_20260915_result_sentences.md`](to_v2/FROM_v3_20260915_result_sentences.md) |
| ✅ v2.17 | v3.10 · 2026-09-15 | Ch 4 §4.5.4: "genuine steps" → "guiding steps"; "regimes" is a banned word | [`FROM_v3_20260915_ch4_naming.md`](to_v2/FROM_v3_20260915_ch4_naming.md) |
| ✅ v2.17 | v3.9 · 2026-09-15 | `04_method.tex` notation table `\multicolumn` ≈ 18.6 cm, overflows; fold `tabularx` into the preamble at merge | `v3/tools/check.py` warning |

## → v3
| status | from | item | note |
| :-- | :-- | :-- | :-- |
| ✅ v3.20 (superseded by author) | v2.20 · 2026-09-17 | Replace the published generative-model brands in Ch 5–6, Ch 8 and the appendix. The author approved a revised mechanism-name set in v3.20: instantaneous-velocity, analytic average-velocity and consistency-interpolated average-velocity matching | [`FROM_v2_20260917_generative_objective_names.md`](to_v3/FROM_v2_20260917_generative_objective_names.md) |
| ✅ v3.18 | v2.17 · 2026-09-16 | The short result sentence offered for contribution 4 ("the faster method wherever it runs") contradicts Ch 6's own `\guard` (1.86–3.57× the cost at equal candidates); v2 used neither cost clause. Also: new bib keys for the visual encoder | [`FROM_v2_20260916_projection_cost_and_encoder.md`](to_v3/FROM_v2_20260916_projection_cost_and_encoder.md) |

## → v4
| status | from | item | note |
| :-- | :-- | :-- | :-- |
| ⏳ | v3.15 · 2026-09-16 | Discussion material found while writing Ch 6 | [`FROM_v3_20260916_discussion_findings.md`](to_v4/FROM_v3_20260916_discussion_findings.md) |
| ⏳ | v3.7 · 2026-09-14 | Removed Ch 7 prose (predates the naming rules) | [`FROM_v3_20260914_discussion_prose.tex`](to_v4/FROM_v3_20260914_discussion_prose.tex) |
