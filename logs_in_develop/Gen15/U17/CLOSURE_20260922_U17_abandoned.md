# CLOSURE — Gen15 U17 · ABANDONED 2026-09-22 · total failure

**Decision (author, 2026-09-22): U17 is abandoned. The thesis falls back to `pillars_hg` (Gen15 U7).**
Nothing produced by U17 fills a thesis slot. This file is the closure of record; the changelog and the
DA stay as the history of how it failed.

| | |
| :-- | :-- |
| unit | Gen15 U17 — UAV-pillars with the obstacle keep-out enlarged at test time (`pillars_xl`, R 0.35; `pillars_xxl`, R 0.55) |
| ran | 2026-09-19 → 22, jobs 25923/25940 (verify), 25942–25963 (attempt 1, died of a full disk), 25970–25995 (attempt 2, complete), 25951 (diffusion, walled at 24 h) |
| delivered | 94 cells × 10 flights = 940 flights, seed 6, tag `u7xl` — `Data_Analysis/analysis_results_checkpoint/22-09-UAV-Pillars/batch_uav_20260922_113112` |
| cost | ≈ 5 GPU-days (flow models) + 1 GPU-day (diffusion, incomplete) |
| result | **S&C = 0.00 and collision-free = 0.00 in all 94 cells, unprojected included** |
| status of the code | `pillars_xl` / `pillars_xxl` stay in `config/uav_projection.yaml` behind an ABANDONED banner, never active, so the post-mortem batch remains reproducible. **Do not run them.** |
| fallback | `pillars_hg` (U7), tag `u7hg`, 15-09/19-09 batches; `DA_in_Paper/analysis/pillars_grid.py` and `uav_results.py` repointed back; `v3/withheld/20260918_uav_pillars_section.tex` to be restored by v3 with its caveat |
| post-mortem | [`DA_20260922_pillars_xl_wave.md`](../../../Data_Analysis/DA_in_Paper/analysis/DA_20260922_pillars_xl_wave.md) · [runbook §3c–§3f](../../Writing/Working_Space/data_status/SLURM_RUNBOOK_20260919_pillars_enlarged.md) · [changelog](CHANGELOG_20260918_pillars_enlarged.md) |

## 1 · What U17 set out to do

`pillars_hg` enforces the clearance its own demonstration generator was built to keep
(`trajectories.py:48`, `_Y_L = -1.11 = -(0.6 + 0.12 + 0.31 + 0.08)`), so every demonstrated route
already satisfies it and the projector has nothing to repair; the four models could not be ordered.
U17 enlarged the virtual keep-out radius from 0.12 to 0.35 m at evaluation time — MJCF untouched, no
retraining — so the demonstrated lane sits 0.15 m inside the obstacle and the projector must produce a
detour, as D3IL-avoiding's test-time halfspaces and the corridor slide do.

## 2 · What happened, in the order it was learned

1. **The geometry was wired correctly.** The verify cell flipped the expert-route probe from passing to
   52–54/200 violating samples, the unprojected mf K5 row from S&C 0.90 to 0.00 with 68.7 violating
   steps, and `phys_safe` stayed 1.00 (the enlargement was virtual as designed).
2. **Attempt 1 died of a full filesystem** — eleven of twelve children never started, one crashed
   `Errno 28`. Cleared with `clean_weights` and re-submitted; attempt 2 ran clean.
3. **The first read (strict success) was a floor**: 53 of 54 projected cells at 0.00. The working
   hypothesis was a goal-radius mismatch (0.30 m radius vs a 0.15 m detour plus 0.34–0.45 m tracking
   error). **Refuted by the batch.**
4. **The batch, under the thesis metric, is all zero** — and the per-flight rows show why. The
   corridor between the two pillar rows is physically 0.96 m wide (drone 0.62 m) and the 0.66 m
   keep-out closes it by only 6 cm each side. **Every projector in the study — DPCC per-step and
   HardFlow endpoint — routes the plan into that forbidden centre corridor** instead of 0.15 m outward:
   38 of 40 endpoint flights of MeanFM and FM fly it contact-free, across the finish line, ~1 m from
   the goal point, 0.37–0.44 m inside the keep-out at every column. Under relaxed success they *pass*;
   S&C is zero solely through `collision_free = 0`. Per-step at $\nfe=5$ under the $r$/$c$ rules
   additionally weaves between the lanes and hits pillars on 8–10 flights of 10. HardFlow reports
   `[NLP-FAILURE] non-converged SLSQP` on every variant.
5. **The author's visual check** read the flights as a clean pass with a mild violation. Both
   impressions are explained: the trajectory PNG draws the physical 0.12 m pillars, the scorer and the
   projector enforce 0.66 m. Scoring "only the trajectory" is already what the scorer does (centre
   point against the inflated obstacle, the same set the projector solved); dropping the 0.31 m would
   make both the demonstrated and the centre lane legal and the scene trivially feasible again.
6. **The diffusion baseline's projected block cannot be evaluated at all** inside the cluster's 24 h
   limit: 10.4 s per control step at $\nfe=20$, 18.6 h for one ten-flight variant, the $c$ rules
   projected at 24–26 h. Closed by decision (runbook §3d).

## 3 · Why it is abandoned rather than fixed

The scene is now known to be degenerate in **both** directions: feasible-by-construction at R = 0.12
(nothing to repair), unrepairable-by-a-local-projector at R = 0.35 (the forbidden lane is the one every
projector finds). Making it work needs a third geometry that closes the centre corridor decisively (a
virtual wall along $y=0$ over the pillar span) **and** a re-run of ~5 GPU-days, with no guarantee the
local projectors then find the outer detour either. That is new experimental work, not a fix, and the
draft already rests on UAV-corridor for the quadrotor claim. Two weeks out, the trade is bad.

## 4 · What stays, and what must never happen

- The batch, the DA and this closure stay in the repo as the record. `pillars_grid.py` keeps the
  `pillars_xl` settings as a comment and the `lanes()` classification so the post-mortem reproduces.
- **No `pillars_xl` / `pillars_xxl` number in any thesis slot, ever.** The templates `tab:uav-pillars`,
  `tab:uav-pillars-projection` and `fig_uav_pillars_xl_paths` (v3.56) are to be removed by v3.
- **No further pillars compute**: not the $\nfe=3$ rung, not the diffusion remainder, not `pillars_xxl`.
- `Slurm_Codes/temp_bash/Gen15_U17.sh` (gitignored, on the cluster) must not be re-run.

## 5 · The fallback — `pillars_hg`, with its caveat stated

The thesis returns to the U7 geometry and the numbers already computed for it (15-09 batch via
`pillars_grid.py`; 19-09 batch via `uav_results.py`, `PILLARS_EXCLUDED` flipped back to `False`).
`v3/withheld/20260918_uav_pillars_section.tex` — three tables and the flown-path figure — is restored
by the v3 author. **The caveat that withdrew it in the first place is true and stays in the prose**: on
this scene the unprojected flows are already feasible, so the projected rows measure how much of a
feasible plan each method preserves, not whether a method can repair an infeasible one; the section
already opens by saying exactly that. The scene is reported as what it is — the projection-preservation
case — beside UAV-corridor, which carries the repair case.

## 6 · What was learned, for the next scene design

- Check that the **projector** can reach the feasible set from the demonstrated lane before running a
  wave — U16 did this for the corridor slide by hand; U17 checked only that the demonstrations violate.
- A keep-out that closes a physically open lane by centimetres is an invitation for a local solver to
  take that lane. Close it by more than the solver's tolerance, or with a halfspace.
- Size disk from a **projected** cell, not an unprojected one; and never combine six per-step variants
  of a $\nfe=20$ diffusion model in one 24 h job.
