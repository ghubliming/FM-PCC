# PENDING — data and analyses the thesis still lacks

**2026-09-16 · companion to** [`NOTE_20260914_official_DA_in_Paper.md`](NOTE_20260914_official_DA_in_Paper.md) · index: [`INDEX.md`](INDEX.md)

Built from every open `\hole`, `\provisional` and `\todofigure` in `v3/chapters/05–09` plus open items found
while writing Ch 6. **Run** = needs the cluster. **DA** = analysis of data that exists. **Author** = a decision.
When an item closes: mark ✅ with the date and the v3 version that absorbed it, or ❌ with the reason.

**Protocol policy (author, 2026-09-17):** on D3IL-avoiding every comparison is made first at DPCC's protocol (5 seeds × 2 episodes per geometry, its released `n_trials: 2`) and then at 5 × 20 as stronger evidence; every model is shown under all three selection rules before the best rule is compared.

**Seed policy (2026-09-16):** D3IL-avoiding is evaluated over five seeds and the thesis reports the
across-seed variation there (`tab:seed-spread`, v3 §5.5.1); D3IL-aligning and the UAV scenes stay at seed 6.
The thesis states this once in §5.5 and does not repeat single-seed caveats in Chapter 6.

## 1. Runs (cluster)

| # | priority | what | why / where it shows | status |
| :-- | :-- | :-- | :-- | :-- |
| R1 | — | **Seeds beyond 6 on the quadrotor**, pillars $\nfe=5$ first (2–3 seeds × MeanFlow, flow matching, consistency training × 11 configurations) | every quadrotor claim is single-seed; §6.3 `\provisional`, §6.4 `\provisional` | ❌ closed 2026-09-16 — **not feasible**: cluster disk (≈7 GB free vs ≈15 GB for 48 trainings plus evaluation plans). Single seed (6) is final; thesis v3.17. [`LOG_20260916_uav_5seed_training.md`](../../../Gen15/Campaign_20260916_uav_5seed_training/LOG_20260916_uav_5seed_training.md) |
| R2 | 🔴 | **Tightened diffusion $\nfe=20$ on alignment** | no constraint comparison against the baseline on alignment; §6.2.4 `\provisional` | ⏳ |
| R3 | 🟠 | Alignment on **held-out** contexts, and 50–60 contexts (D3IL uses 60) | results are on 10 training contexts; §6.2 `\guard`, §5.5.2 `\provisional` | ⏳ |
| R4 | — | Alignment seeds beyond 6 | single-seed; §6.2 `\guard` | ❌ closed 2026-09-16 — author decision after cost assessment (≈82 GPU-h training + a larger evaluation bill); single seed (6) is final; thesis v3.17. [`ASSESS_20260916_va_5seed_training_cost.md`](../../../Gen14/Campaign_20260916_va_5seed_assessment/ASSESS_20260916_va_5seed_training_cost.md) |
| R5 | 🟠 | **Plan-matrix panels**: flow matching $\nfe=1,2$ · diffusion $\nfe=2$ · consistency training $\nfe=1,2$ (seed 6, both-hard, unprojected) | five `\todofigure` in `fig:raw-plans`; commands in [`REQUEST_20260916_cluster_fm_plan_panels.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md) | ⏳ |
| R6 | 🟠 | Consistency training (U-Net, floor 0.2) on seeds 7–10, obstacle avoidance | `tab:state-models` and `fig:avoiding-raw-models` are seed 6 only | ⏳ |
| R7 | 🟡 | Endpoint projection at its budget floor, 5 seeds × 20 episodes | §6.1.4 `\provisional` (currently 4 seeds × 3 × 2) | ⏳ |
| R8 | 🔴 | **D3IL-avoiding at the protocol of DPCC** (5 seeds × 2 episodes) for flow matching K1/K2, MeanFlow U-Net K1/K2 and α-Flow U-Net α_end 0.2 K1/K2 (the last needs training seeds 7–10) | primary comparison of Ch 6 since v3.19: `tab:avoiding-dpcc-protocol` *pending* rows + `\hole`; commands in [`REQUEST_20260917_avoiding_dpcc_protocol.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260917_avoiding_dpcc_protocol.md) | ⏳ |
| R9 | 🟡 | Flow matching re-evaluated at unit initial-noise scale | §5.5.1 `\provisional` | ⏳ |
| R10 | 🟡 | S-curve endpoint-projection rows re-run after the switched-wall fix | excluded from §6.3.4; `Gen15/U16/…FULL_REVIEW` §3.1 | ⏳ |
| R11 | ⚪ optional | Our own sweep of the projection's assumed sampling time (DPCC Table 2) — `dpcc-c-tightened-dt*` rows already exist in the 15-09 avoiding batch, so possibly **DA only** | would replace the citation in §6.1.3 with a measurement | ⏳ |

## 2. Analyses of existing data (no runs)

| # | what | where it shows | status |
| :-- | :-- | :-- | :-- |
| D1 | **Plan smoothness metric** — jerk, path length or curvature over the saved plan `.npz` files (on cluster disk) | §6.1.3 `\hole` | ⏳ |
| D2 | Alignment **orientation error**: the angle columns differ between runs for identical contexts — establish the convention, then report | §6.2.1 `\hole` | ⏳ |
| D3 | Verify the ≈119 ms/step MuJoCo MPC overhead against committed timing (the 15-09 `avg_time_ms` column does not show it) | §6.3.4 | ⏳ |
| D4 | Alignment workspace box per geometry from `config/visual_aligning_eval.yaml` | §5.1.2 `\hole` | ⏳ |
| D5 | Quadrotor: confirm pillars/s-curve use the corridor's workspace box; state tightening 0.025 m | §5.1.3 `\hole` | ⏳ |
| D6 | Accepted demonstrations per quadrotor scene; controller gains per collection run; alignment demonstration count; train/validation split of all three environments | §5.2 `\hole` | ⏳ |
| D7 | Confirm controller and velocity-setpoint policy for pillars and s-curve results | §5.5.3 `\hole` | ⏳ |
| D8 | Total compute (GPU-hours) from job logs | appendix `\hole` | ⏳ |
| D9 | SLURM job IDs and git revision per batch for `tab:corpora` | appendix `\hole` | ⏳ |

## 3. Decisions for the author

| # | what | context | status |
| :-- | :-- | :-- | :-- |
| A1 | Report the **D3IL image policy** run in our pipeline (`d3il_baseline`, test split, 60 contexts: 0.44 m from 0.45 m, box unmoved in 44 % of 3,884 episodes)? | §5.3.2 says no alignment baseline exists | ⏳ |
| A2 | Figs 5.2 / 5.3 use crops of **D3IL's own README GIF** without credit — credit D3IL, or replace with our MuJoCo renders | provenance; `plotting/NOTEBOOK_20260915…` | ⏳ |
| A3 | Keep `tab:target` (published vs reproduced) under v2.16's "describe only our configuration" rule? | §5.3.1; neither of our episode counts equals the published 50 | ⏳ |
| A4 | Training is not uniform across models (batch, lr, action weight, EMA) — accept with the `\guard`, or retrain matched | `tab:train`; `cross_draft/to_v2/FROM_v3_20260916_training_not_uniform.md` | ⏳ |

## 4. Owned elsewhere (not v3 data work)
- Ch 7 Discussion, Ch 8 refinement and future work → v4 (`cross_draft/to_v4/`).
