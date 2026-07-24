# npz tools — horizontal (across-method) H-step plan comparison

**Date:** 2026-07-11
**Tool:** `npz_analysis/compare_horizon_plans.py` (NEW standalone script)
**Status:** working; ran on real Gen11 imf-debug npz (`halfspace_both-hard`, 13 variants);
PNG + CSV + stdout table all produced. `analyze_npz.py` **untouched** (safety: no risk to the
existing analyzer).

---

## Why a new script (not an `analyze_npz.py` extension)

`analyze_npz.py` is **vertical**: one row per file, per-file aggregates. The new question is
**horizontal** — *for the SAME trial/seed, at the SAME MPC decision (snapshot), how does each
projection method bend the H-step foresight plan?* That crosses files (one `.npz` per projection
variant) at a single (trial, snapshot) coordinate. Different axis of analysis → separate tool,
per the repo's copy-modify isolation convention, and keeps the working analyzer at zero risk.

## What it does

For a fixed `--trial` + `--snapshot`, pulls each variant's plan snapshot `[batch, H, dim]` from
`sampled_trajectories_all` and emits:
- **`horizon_compare_t<trial>_s<snap>.png`** — overlay: each variant a colour, its candidate fan
  (thin) + candidate-mean (bold), ●=start ■=horizon-end.
- **`horizon_compare_t<trial>_s<snap>.csv`** — tidy long format `variant,trial,snapshot,
  approx_exec_step,candidate,step,x,y` (pivot-ready).
- **stdout table** — per variant: snapshot idx, n_snap, ~exec_step, batch, `path_len`, `max|c|`
  (explosion), plan endpoint, and **`div_ref`** = mean per-waypoint distance of the candidate-mean
  plan to a reference variant's (default `diffuser`) → how far each projection pulls the plan.

Run:
```bash
python npz_analysis/compare_horizon_plans.py <results-dir> --trial 0 --snapshot 0 --env avoiding \
  --variants diffuser,model_free,gradient,dpcc-c,dpcc-r,dpcc-t
```

## Correctness decisions (the non-obvious bits)

- **Snapshot alignment is only valid at snapshot 0.** All variants share the identical start
  state at `s=0` (verified: every variant's trial-0 snap-0 waypoint-0 = `[0.525,-0.28,0.525,-0.28]`),
  so `s=0` is the one apples-to-apples projection comparison. For `s>0` the executed paths diverge —
  episodes even differ in length, so snapshot **counts** differ (diffuser 25 vs dpcc-c 38 vs …),
  meaning the same index is a **different physical state** per method. The script prints a WARNING for
  `s>0 && align=index`, and offers **`--align step`** (interpret `--snapshot` as a target executed
  step, pick the nearest snapshot per variant: snapshot k ≈ executed step `k*(H//2)`, save-every=4).
- **Candidate fan, not "candidate 0".** The npz stores ALL `batch` candidates in original order; the
  **selected/executed** index is NOT recorded (`trajectory_selection` ∈ {random, temporal_consistency,
  minimum_projection_cost}). So the tool plots the whole fan + candidate-mean instead of pretending
  index 0 is the chosen plan.
- **Columns.** Plan `obs_dim=4 = [x_des,y_des,x,y]`; actual position = cols **2,3** (per
  `obs_indices` in `eval_flow_matching_v3_imeanflow.py`, matching the eval's own overlays). Default
  `--env avoiding` → `[2,3]`; `--env uav` → `[3,4]`; `--xy-cols` overrides.
- **Storage-shape robustness.** `plan_snapshots()` normalises BOTH shapes this repo produces: ragged
  `(n_trials,)`-object-of-lists (dpcc-*) AND the homogeneous `(n_trials,n_snap,batch,H,dim)` 5-D array
  numpy builds when every trial has equal snapshot count (imf/diffuser).

## First real result (trial 0, snapshot 0, `halfspace_both-hard`)

`div_ref` vs raw `diffuser`: hard projection (`dpcc-c/r/t`, `post_processing`) all collapse to the
same arc — `path_len≈0.032`, `div_ref≈0.009`, identical endpoint `(0.494,-0.273)`. `gradient` is a
softer bend (`path_len 0.060`, `div_ref 0.010`); `model_free` departs most (`div_ref 0.016`, distinct
endpoint). Raw `diffuser` plan is the longest/curviest (`path_len 0.0997`). Matches expectation:
projection pulls the nominal FM plan onto the constraint-satisfying region.

## Environment note

Verified locally in an **isolated scratchpad venv** (numpy 2.5.1 + matplotlib 3.11.0) — nothing
installed into the container's system env (the "no Python packages" rule stands). On the cluster it
just uses the FMPCC conda env (numpy+matplotlib already present).

## Update — `--draw-constraints` (projection-insight view) + violation flagging

Added the constraint overlay so "plan hugs vs crosses the constraint" is directly visible, plus a
quantified violation column. Reproduces the eval's exact geometry (no import of the eval package):
- **Variant→constraint selection** mirrors `eval_flow_matching_v3_imeanflow.py` (~lines 59-67):
  `both-hard` = `halfspace_constraints[2],[3]` + `obstacle_constraints[5]`;
  `top-left-hard` = `hs[0]`+`ob[3]`; `top-right-hard` = `hs[1]`+`ob[4]`. Read from
  `config/projection_eval.yaml`. `--halfspace-variant` auto-inferred from the results folder name.
- **Halfspace triangle** drawn with the same slope/3rd-vertex logic as
  `utils.constraints_helpers.plot_halfspace_constraints` (avoiding branch); obstacle as a circle.
  Frame locked to config `ax_limits` (so snapshots share one comparable frame).
- **Violation test** ports `formulate_halfspace_constraints`' inequality: `below` feasible ⇔
  `y < m*x + d` (`d = y0 - m*x0`) ⇒ violate when `-m*x + y >= d`; obstacle violate ⇔ inside radius.
- **New `plan_viol` column** = % of ALL candidate waypoints inside a forbidden region; red × marks the
  candidate-**mean** plan's violating waypoints on the PNG (mean-only, so a low `plan_viol` may show
  no ×). **`--show-executed`** overlays each variant's `obs_all` dashed for plan-vs-reality context.
- Lazy `import yaml` under the flag only → base tool stays numpy-only.

### Result (trial 0, `halfspace_both-hard`)
- **snapshot 0**: all variants `plan_viol=0.0%` — feasible start, projection idle, plans identical
  (explains why s0 "looks the same under every projection").
- **snapshot 48** (`--align step`): `diffuser` 6.2%, `gradient` 6.2%, `model_free` 3.1% vs
  `dpcc-c/r/t` **0.0%** — hard projection keeps every waypoint feasible; `model_free` barely projects
  (`div_ref` 0.007, tracks raw `diffuser`) so it still violates. Directly relevant to the
  `INVESTIGATION_geo_free_model_free_worse_than_diffuser` line of work.

## Update — cross-environment safety hardening

Audited env-dependent paths and fixed the one real footgun:
- **`save_every` was hardcoded 4** (avoiding H//2), which is WRONG for UAV (saves a snapshot every
  step → 1), silently corrupting `--align step` off-avoiding. Now **inferred per-variant from data**
  (`(executed_len-1)/(n_snapshots-1)`, floored at 1), env-agnostic and more correct than any constant;
  `--horizon-div` still overrides. Avoiding regression-checked: still resolves to 4, identical results.
- **xy-cols out of range** no longer silently falls back — prints a WARNING telling the user to pass
  `--xy-cols` for that schema.
- **`--draw-constraints` is avoiding-halfspace only by construction** and degrades safely elsewhere
  (variant-infer returns None or config lookup KeyErrors → skip + warn, never draws avoiding geometry
  on a non-avoiding plot). Documented as a hard boundary in USAGE.
- Verdict: safe (no crash / no wrong-geometry) on all envs; fully correct out-of-the-box only for
  avoiding. UAV/visual-aligning need `--xy-cols` and should skip `--draw-constraints`.

## Update — auto-zoom to plans (fix: env frame made plans invisible)

`--draw-constraints` originally locked the frame to the full-env `ax_limits`, which shrank the small
H-step plans to near-invisibility. Now the view **auto-zooms to the plan bounds by default**
(constraint patches AND executed paths are excluded from the bound computation so they can't blow the
view back out to full-env size). The full-environment frame is now **opt-in via `--full-frame`**.
Constraints are still drawn in both modes.

## Update — env background is opt-in (`--show-env`), off by default

Per user preference, `--draw-constraints` no longer draws the avoiding env itself (the blue filled
halfspace funnel + obstacle circle clutter the plot). It now only does the **violation analysis**
(red × on violating mean-plan waypoints + `plan_viol` column) — which is about the plans, not the env.
The filled env is opt-in via **`--show-env`**. Title text adapts (drops the "blue = forbidden region"
line unless `--show-env`).

## Update — `--candidate k` (per-candidate projection comparison) + nested UAV layout

- **Nested layout support**: `find_variant_npz` now also finds `<dir>/<variant>/<variant>.npz` (UAV
  eval layout), not just flat `<dir>/<variant>.npz`. Verified on UAV s_curve (17 variants discovered).
- **`--candidate`** (`mean` default, or int `k`): draw/compare a single candidate per variant instead
  of the mean+fan. Rationale (empirically verified on UAV snap0): the eval seeds RNG per trial, so at
  snapshot 0 candidate `k` is the SAME pre-projection sample across variants (max-diff ~0.01 =
  projection nudge) → `--candidate 0` isolates the projection effect on one shared trajectory. Thin fan
  hidden in this mode; `div_ref`/table/CSV use candidate `k`.
- **Honest limits baked in as warnings**: `--candidate k` only apples-to-apples at snapshot 0 (WARNs
  otherwise); `gradient` desyncs RNG (snap0 candidates differ by ~6.4) so it's not candidate-comparable;
  and at snap0 projection is nearly idle, so the dramatic bends (mid-rollout) remain outside the
  clean-comparison regime — the fundamental separate-rollout limitation, unchanged.

## Not done / future

- Palette wraps past 10 variants (the 13-variant run reused black/green for `diffuser`/`model_free` and
  `dpcc-c`/`post_processing`); use `--variants` to curate, or extend `_PALETTE`.
- No selected-candidate highlighting (index not stored — would require the eval to persist it).
- s>0 comparisons remain confounded (projection effect + diverged rollout states); only snapshot 0 is
  a clean same-input comparison. `--draw-constraints` makes the confound legible but can't remove it.
