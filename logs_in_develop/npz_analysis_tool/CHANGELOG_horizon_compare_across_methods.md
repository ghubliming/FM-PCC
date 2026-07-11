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

## Not done / future

- **`--draw-constraints`**: overlay the halfspace line / obstacle circles behind the plans (needs
  reading the task config the way the eval does) — makes "plan hugs vs crosses the constraint" visible.
- Palette wraps past 10 variants (the 13-variant run reused black/green for `diffuser`/`model_free` and
  `dpcc-c`/`post_processing`); use `--variants` to curate, or extend `_PALETTE`.
- No selected-candidate highlighting (index not stored — would require the eval to persist it).
