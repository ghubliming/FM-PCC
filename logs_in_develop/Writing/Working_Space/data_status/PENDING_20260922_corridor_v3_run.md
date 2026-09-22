# PENDING RUN — UAV-corridor v3: the grid, the arms, the Slurm order, and what the thesis reads from it (R33)

> **Superseded on 2026-09-23 by [`PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md`](PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md)** (geometry decided: tilt **and** hump; paper-only variants; tag `p23cv3`; runbook beside it). Kept for history; do not submit from here.

**2026-09-22 · v3.66 · author's methodology (changelog item 5), recorded here as the run specification.**
Ledger entry: `PENDING_20260922_all_lacking_runs.md` §23 / R33. Chapter 5/6 corridor blocks are blank until this lands.

## 0 · Geometry — one decision for the author

| option | what | status |
| :-- | :-- | :-- |
| **A (default assumed here)** | `corridor_v3_hump` — the wide corridor v2 plus the x–z hump of Gen15 U19 (`Gen15/U19/PLAN_20260922_U19_corridor_v3_z_slide.md`), i.e. a constraint on altitude | needs the U19 coding (three small sites, opt-in `plane: xz`) before any job |
| B | `corridor_v2_slide` re-run on the new grid (same lateral slide as the archived results) | no code; submit today |

The grid, arms, seeds and order below are identical under A and B. If the author means B, replace the geo name and the
eval tag and nothing else. **Do not pool with `u17cv2` (the archived v2 corpus) either way.**

## 1 · The grid (one training seed, 6; 12 flights per cell = 4 per route L/C/R)

| model | budgets $\nfe$ | unprojected | per-step (PCC, DPCC projector) | endpoint (HF) |
| :-- | :-- | :-- | :-- | :-- |
| MeanFM (analytic average-velocity) | 1, 2, 3 | ✓ | ✓ at every budget, threshold **0.5** | at **3** only |
| CI-MeanFM, α_end = 0.2 | 1, 2, 3 | ✓ | ✓ at every budget, threshold 0.5 | at 3 only |
| FM (instantaneous-velocity) | 1, 2, 3, 5, 20 | ✓ | ✓ at every budget, threshold 0.5 | at 3, 5, 20 |
| Diffusion (DPCC baseline) | 20 | ✓ | ✓ at 20, threshold 0.5 | — (not defined) |

Why HF only from 3: at threshold 0.5 a budget of 1 or 2 has no guiding step, so endpoint projection would be projection
after sampling (the degenerate case, §6.1.2.7). PCC is always on.

**Arms per cell** (the corridor stack of U16, unchanged: `-bounds_free-pdes-tightened`):

- `diffuser` (unprojected)
- `dpcc-r-bounds_free-pdes-tightened`, `dpcc-c-…`, `dpcc-t-…` — per-step, three rules, four candidates
- `hardflow_sls-bounds_free-pdes-tightened` (single candidate), `hardflow_sls-r/-c/-t-…` (four candidates) — endpoint, K ≥ 3
- ~~one untightened arm per model~~ — **dropped at v3.67** (all-tightened decision; the tightening paragraph is gone)

Cells: MeanFM 3 budgets × (1 + 3) + 1 × 4 HF = 16; CI-MeanFM 16; FM 5 × 4 + 3 × 4 = 32; diffusion 4 = **68 cells**, 816 flights.

## 2 · Slurm order (fast first; K = 20 last so it cannot jam the queue)

| wave | jobs | why here |
| :-- | :-- | :-- |
| 1 | all `diffuser` cells (mf/af K1,2,3; fm K1,2,3,5; diffusion K20 diffuser too — it is cheap unprojected) | the pre-projection table is readable after this wave alone |
| 2 | PCC r/c/t for mf/af/fm at K1, 2, 3 | the bulk of the post-projection table |
| 3 | HF single/r/c/t for mf/af at K3 and fm at K3, 5; PCC r/c/t for fm at K5 | the HF-vs-PCC comparison at the budgets that matter |
| 4 | fm at K20: PCC r/c/t + HF single/r/c/t | slow flow cells |
| 5 | **diffusion K20: PCC r/c/t** — **last**; one variant per job (~1 GPU-day per variant on v2, `c` rules near the 24 h wall) | the baseline; never in front of the fast runs |

Submitter: copy `Slurm_Codes/temp_bash/eval_20260913_u16_corridor_v2_paper.sh` (`plan` / `smoke` / `submit` modes), new
eval tag (`u19cv3` under A), `UAV_MIX_GEO_VARIANTS=<geo>`, `UAV_EVAL_HOURS=24`. Smoke first: mf K3, 3 trials, GIF on.
The DA is `Data_Analysis/DA_UAV_v1` as for U16; check that the new geo tag parses (`discovery.py` l.117).

## 3 · What the thesis reads from it (the structure already in Chapter 6, cells blank)

1. **Before projection** — Table 6.9 (`tab:uav-corridor-raw`): success / collision-free / S&C / violating steps / ms
   per cell. A frontier for this arm **only if not every model crosses the line**; if all pass, the table alone.
2. **After projection** — Table 6.11 (`tab:uav-corridor`): per model and budget, PCC and HF side by side at each one's
   best rule, with Fig 6.9 (frontier, both projectors, circles = per-step, squares = endpoint), Fig 6.10 (paths),
   Fig 6.11 (altitude). Pick the best configuration per model by table + frontier.
3. **Projection methods** — Table 6.16 (`tab:uav-corridor-projection`): r / c / t for both projectors at K ≥ 3.
4. §6.3.5 conclusion sentence and, when unlocked, the §6.4 summary rows.

## 4 · Cost

Roughly the U16 wave (~1 GPU-day of walls, ~10 jobs) plus the K20 flow cells and the HF arms: **~1.5–2 GPU-days**,
of which the diffusion K20 projected arms are about half. Waves 1–3 alone (~6 h) already fill Tables 6.9 and 6.11
for the flow models.

Claude (Fable 5.1, Claude Code) · 2026-09-22 · specification only; nothing submitted.
