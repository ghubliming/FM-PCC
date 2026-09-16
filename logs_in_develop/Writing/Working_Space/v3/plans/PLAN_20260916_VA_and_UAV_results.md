# PLAN — write the alignment and quadrotor results (v3, Ch. 6.2 and 6.3)

**Owner:** Claude (v3) · **Status:** ✅ executed 2026-09-16 · **Created:** 2026-09-16 (was a Codex handover; the author reassigned it) · **Draft:** `logs_in_develop/Writing/Working_Space/v3/`

Two jobs. Both are **writing from existing data** — no cluster runs, no evaluation, no training. When
data is missing, write a visible placeholder (`\hole{...}` in text, `\todofigure{...}` for a figure)
and **list every placeholder in the final report to the author**.

---

## 0. Read first

| what | where |
| :-- | :-- |
| who owns which chapter — **v3 owns Ch. 5, 6, appendix; never edit Ch. 1–4 or Ch. 7** | `Working_Space/DRAFT_OWNERSHIP.md` |
| thesis names — every dev token has a thesis name | `Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md` |
| style rules, what the author has already rejected | `v3/CHANGELOG.md` (v3.7 → v3.14), `Auxiliary/NOTES_*.md`, `Writing_Hints/` |
| the official data and figure store, and its rules | `Data_Analysis/DA_in_Paper/README.md`, `plotting/NOTEBOOK_20260915_figure_data_sources.md` |
| how the current Ch. 6.2 / 6.3 already read | `v3/chapters/06_results.tex`, `\section{Vision-Conditioned Alignment}` and `\section{Quadrotor Benchmark}` |
| the cross-environment table you must keep consistent | same file, `tab:summary-models`, `tab:summary-projection` |

## 1. Job A — Vision-conditioned alignment (§6.2)

**Goal.** A complete §6.2 showing the ordering of the generative models, then a short discussion of
consistency training.

**Data, in priority order.**
1. `Data_Analysis/analysis_results_checkpoint/15-09/batch_va2_20260915_100754/` — **the latest data;
   use this**. Committed. `per_rollout_detail.csv`, `va2_units_long.csv`, `run_config.csv`.
2. Curated / closing analyses (numbers of record — reproduce, don't just copy):
   - `logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md`
   - `logs_in_develop/Gen14/DA_20260903_Gen14_four_gate_fm_vs_mf_K20_T0.2.md`
   - `logs_in_develop/Gen14/DA_20260907_Gen14_Gate1_AF_vs_MF_K20_flagship_KILL.md`
   - `Data_Analysis/DA_Result_Curated_MD/ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md`
3. Environment and metrics: `Auxiliary/Methodology_Sources/AUX_visual_aligning_env.md`; v3 §5.1.2, §5.4.2, §5.5.2.

**Expected story (the author's):** MeanFlow > flow matching > diffusion, with flow matching ≈ diffusion
possible; consistency training ≈ MeanFlow, not better. **Write what the data shows.** The current
draft already has: MeanFlow > flow matching (0.216 m, p = 0.0039), flow matching ≈ diffusion
(p = 1.0000), MeanFlow > diffusion (0.374 m, p = 0.0020); consistency training ≈ MeanFlow at $\nfe=2$,
behind at $\nfe=20$ (p = 0.0215). If the 15-09 data changes any of these, say so in the chat.

**Known traps — check each one.**
- `run_config.csv` has `split ∈ {train, test}`, and variants tagged `@train_set`. Establish which rows
  are the **held-out** evaluation before computing anything. In a quick look the MeanFlow and
  consistency-training rows did not appear under `split = test` keyed by engine; find out why.
- There is a **`d3il_baseline` engine, seeds 0–4 and 42**, in this batch. Nobody has written about it.
  Find what it is. If it is D3IL's own policy evaluated in our pipeline, it bears directly on
  §5.3.2 (which currently says no baseline exists) — **tell the author before writing it in**.
- Everything for the thesis models is **seed 6 only**. Keep the paired tests over contexts; don't
  present a single-seed result as general. `\provisional{}` where needed.
- Several consistency-training checkpoints exist (constant 0.05, sigmoid, end 0.05, end 0.2; FiLM v1 and
  v2). Use the one the closing DA names; state which in the caption.
- Distance comparisons of generative models are made **without projection**; projection-method
  comparisons **with** it (v3 §5.4.2). Don't mix.

## 2. Job B — Quadrotor (§6.3)

**Goal.** Corridor v2 and pillars fully written with the ordering and a consistency-training
discussion; s-curve as a results listing whose discussion is **about the tracking controller**.

**Data.**
1. Latest: `Data_Analysis/analysis_results_checkpoint/15-09/batch_uav_20260915_100816/`
   (`uav_units_long.csv`, `uav_aggregated_long.csv`, `per_rollout_detail.csv`, `run_config.csv`). All
   seed 6.
2. Corridor v2: `logs_in_develop/Gen15/U16/DA_20260914_corridor_v2_paper_full.md` (the analysis of
   record), `NOTE_20260914_body_margin_vs_dpcc_halfspace.md`, `CHANGELOG_20260913_u16_fix1_fix2_FULL_REVIEW.md`.
3. Pillars: `Gen15/Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md`,
   `DA_20260912_pillars_diffusion_baseline_reference.md`, `DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md`,
   `DA_20260910_T1_af_unet_pillars_K_sweep.md`.
4. S-curve and the controller: `Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md`,
   `Data_Analysis/DA_Result_Curated_MD/Report_20260909_MJPC_vs_PID_s_curve/README.md`,
   `Gen15/U10/CHANGELOG_20260907_mjpc_controller_override.md`, `DA_20260910_T4_s_curve_budget_explore.md`.

**Expected story:** MeanFlow > flow matching > diffusion on corridor and pillars; consistency training
discussed at the end.

**Known traps — check each one.**
- 🔴 **Pillars already contradicts "consistency training ≈ MeanFlow":** the draft has 0.635 (MeanFlow)
  > 0.359 (flow matching) > **0.235 (consistency training)**. Verify against the latest data and write
  the discussion to match what is true, not what was expected.
- 🔴 **Report_20260909 (MJPC vs PID) is partially invalid.** Every endpoint-projection row predates a
  projector fix (`FULL_REVIEW` §3.1). The unprojected and per-step rows stand. The usable finding: the
  unprojected MeanFlow plan reached the goal 0/10 under the PID-style controller and 3/3 under MuJoCo MPC
  on the same initial conditions, and the PID-style controller inverted the vehicle in 10/10. Do not
  cite any s-curve endpoint-projection number from before the fix.
- MuJoCo MPC in this work is a **tracking controller** replacing the geometric tracker, not a planner
  baseline. §5.3.3 already describes MuJoCo MPC's own quadrotor task; keep the two roles distinct.
- Corridor projection configuration (`-bounds_free-pdes-tightened`) differs from pillars / s-curve —
  never pool across scenes (§5.5.3 says so).
- The diffusion baseline is trained with action loss weight 1 and $\nfe=20$ is a **training** budget,
  while flow-based budgets are inference-time. Report it as a reference, as §6.3.2 already does.
- Training is **not** uniform across models on obstacle avoidance and alignment (`tab:train`, v3.14);
  on the quadrotor it is. Don't write "identical training".

## 3. Rules that are not negotiable

**Naming.** No brand names outside Related Work: *α-Flow → consistency training*, *HardFlow → endpoint
projection*, *DPCC projection → per-step projection*. Banned words: arm, engine, funnel, stage, tier,
regime, gate (in the experimental sense), genuine, flagship, Target, Pareto, best/SOTA. "Ahead / behind /
level / not separated" and name the axis. Never define "better" globally.

**Prose.** Short, factual, storytelling rather than justification. No self-talk ("we note that…"). Every
number is followed by a `\dataref{batch; DA §}`. Caveats that must travel with a claim go in `\guard{}`,
statements below the thesis's intended strength in `\provisional{}`.

**Figures.** Produced only in `Data_Analysis/DA_in_Paper/` (builders in `plotting/builders/`, data paths
only in `plotting/sources.py`), then copied with `python3 plotting/export_to_draft.py v3`. Never draw
inside the draft. Any figure copied from elsewhere must first be checked against `/workspaces/aux_repo/`
— a byte-identical match there means it is someone else's figure (see `EXCLUDED`, and the D3IL README
GIF problem in the notebook). Figures need a PNG (`plotting/svg/preview_png.py --scale 3`) or they
render as placeholders on Overleaf.

**Data handling.**
- The 15-09 raw tables are **long** (`metric` / `value` rows); column order differs per batch; the
  avoiding one is gzipped. Read with `csv.DictReader`, never by column index.
- **Pin the seed set across compared models.** Averaging each model over whatever seeds it has once
  produced 0.97 / 0.97 / 0.92 instead of 0.85 / 0.85 / 0.60.
- Two `n_steps` definitions exist (successes-only vs all episodes). Name which one you use.
- `python3.14` has numpy / pandas-free stdlib + PIL + matplotlib; plain `python3` is stdlib only.

**Environment.** Never run training, evaluation or the pipeline locally. Never `git commit`.

## 4. Deliverables

1. `v3/chapters/06_results.tex` — §6.2 and §6.3 rewritten; `tab:summary-models` and
   `tab:summary-projection` in §6.4 updated if any number moved; RQ answers in `08_conclusion.tex`
   checked for consistency.
2. Any new figures in `DA_in_Paper`, exported to v3, with sources added to the notebook.
3. A **concise** changelog `v3/changelogs/v3.15_20260916_va_and_uav_results.md`, linked at the top of
   `v3/CHANGELOG.md`.
4. Verification, and quote the output in the changelog:
   ```bash
   cd logs_in_develop/Writing/Working_Space/v3
   python3 tools/check.py                 # must pass
   python3 bundle/make_bundle.py          # cross-references resolve, N of N figures render
   ```
5. **Final report** to the author: what changed; every placeholder left and why; every place the
   data disagreed with the expected ordering; the `d3il_baseline` finding.

---

## 5. Execution log

Filled in as the plan is executed.

| step | status | note |
| :-- | :-- | :-- |
| A0 establish held-out split and `d3il_baseline` | ✅ | every thesis model evaluated on **train** contexts (disclosed in a `\guard`); `d3il_baseline` = our run of D3IL's image DDPM policy, **test** split, 60 contexts, 1712/3884 untouched — not paired, not written in, raised with the author |
| A1 generative-model ordering, unprojected | ✅ | 15-09 = same 14,102 rollouts as the closure; all numbers reproduced by `DA_in_Paper/analysis/va_results.py`. MeanFlow ahead of both; flow matching ≈ diffusion. Stated on untightened (only geometry with diffusion) |
| A2 consistency training vs MeanFlow | ✅ | level at K2, behind at K20, flat in K; new subsection |
| A3 projection methods | ✅ | K10 latency verified; 🔴 corrected the draft's false claim that diffusion reaches the same freedom from violations (it is 0.10, 130 violating steps, 21/30 unmoved, untightened) |
| B1 corridor v2 | ✅ | all DA_20260914 numbers reproduced (`uav_results.py`, tag u17cv2); endpoint claim moved to K5 — K3 has one guiding step |
| B2 pillars | ✅ | ranking restated on the 11 full-constraint configurations (0.618 / 0.327 / 0.218); 🔴 draft's "0/210 left the constraint set" was physical safety, not constraints, and included ablations — corrected |
| B3 consistency training discussion | ✅ | level on corridor, last on pillars (behind flow matching) |
| B4 s-curve listing + controller discussion | ✅ | listing table; MPC 3/3 vs brake-to-rest 0/10 on the same plan; endpoint-projection rows excluded (pre-fix) |
| C  §6.4 tables, RQs, checks, bundle, changelog | ✅ | 🔴 "MF > FM > diffusion holds on alignment" and "all endpoint results have ≥2 guiding steps" were false — corrected; RQ1/RQ4 updated; `INDEX.md` rows replaced; check.py pass; bundle 18/18 |
