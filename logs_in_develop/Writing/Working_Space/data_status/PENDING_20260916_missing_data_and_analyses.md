# PENDING — data and analyses the thesis still lacks

**2026-09-16 · companion to** [`NOTE_20260914_official_DA_in_Paper.md`](NOTE_20260914_official_DA_in_Paper.md) · index: [`INDEX.md`](INDEX.md)

Built from every open `\hole`, `\provisional` and `\todofigure` in `v3/chapters/05–09` plus open items found
while writing Ch 6. **Audited cell by cell on 2026-09-18 (v3.34):** see
[`PENDING_20260918_verified_data_audit.md`](PENDING_20260918_verified_data_audit.md) for what was checked and
what turned out to exist. Nothing has landed since the 15-09 checkpoint, so **every ⏳ row below is still
open, none is merely unread**. **Re-traversed against the Results chapter on 2026-09-17 (v3.30):** every marker and every
guard in `06_results.tex` was matched to a row here, and the rows that had none were added (R12–R17).
Locations are given as labels, not section numbers, because the numbers move. **Run** = needs the cluster. **DA** = analysis of data that exists. **Author** = a decision.
When an item closes: mark ✅ with the date and the v3 version that absorbed it, or ❌ with the reason.

**Protocol policy (author, 2026-09-17):** on D3IL-avoiding every comparison is made first at DPCC's protocol (5 seeds × 2 episodes per geometry, its released `n_trials: 2`) and then at 5 × 20 as stronger evidence; every model is shown under all three selection rules before the best rule is compared.

**Seed policy (2026-09-16):** D3IL-avoiding is evaluated over five seeds and the thesis reports the
across-seed variation there (`tab:seed-spread`, v3 §5.5.1); D3IL-aligning and the UAV scenes stay at seed 6.
The thesis states this once in §5.5 and does not repeat single-seed caveats in Chapter 6.

## 1. Runs (cluster)

| # | priority | what | why / where it shows | status |
| :-- | :-- | :-- | :-- | :-- |
| R1 | — | **Seeds beyond 6 on the quadrotor**, pillars $\nfe=5$ first (2–3 seeds × MeanFlow, flow matching, consistency training × 11 configurations) | every quadrotor claim is single-seed; §6.3 `\provisional`, §6.4 `\provisional` | ❌ closed 2026-09-16 — **not feasible**: cluster disk (≈7 GB free vs ≈15 GB for 48 trainings plus evaluation plans). Single seed (6) is final; thesis v3.17. [`LOG_20260916_uav_5seed_training.md`](../../../Gen15/Campaign_20260916_uav_5seed_training/LOG_20260916_uav_5seed_training.md) |
| R2 | 🔴 | **Tightened diffusion $\nfe=20$ on alignment** | no constraint comparison against the baseline on alignment; `sec:res:aligning:projection` `\provisional` | ⏳ |
| R3 | 🟠 | Alignment on **held-out** contexts, and 50–60 contexts (D3IL uses 60) | results are on 10 training contexts; `sec:res:visual` `\guard`, `sec:setup:protocol:aligning` `\provisional` | ⏳ |
| R4 | — | Alignment seeds beyond 6 | single-seed; §6.2 `\guard` | ❌ closed 2026-09-16 — author decision after cost assessment (≈82 GPU-h training + a larger evaluation bill); single seed (6) is final; thesis v3.17. [`ASSESS_20260916_va_5seed_training_cost.md`](../../../Gen14/Campaign_20260916_va_5seed_assessment/ASSESS_20260916_va_5seed_training_cost.md) |
| R5 | 🟠 | **(4 of 5 are a DOWNLOAD — corrected 18-09; the 5th cannot exist)** **Plan-matrix panels**: flow matching $\nfe=1,2$ · diffusion $\nfe=2$ · consistency training $\nfe=1,2$ (seed 6, both-hard, unprojected) | five `\todofigure` in `fig:raw-plans`; commands in [`REQUEST_20260916_cluster_fm_plan_panels.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md) | ✅ closed 2026-09-19 — the four panels were downloaded and are in the store; drafted into `fig:raw-plans` at thesis v3.41. The fifth (diffusion $\nfe=2$) is **impossible**, not pending, and the caption now says so |
| R6 | 🟠 | Consistency training (U-Net, floor 0.2) on seeds 7–10, obstacle avoidance | `tab:state-models` and `fig:avoiding-raw-models` are seed 6 only | ✅ closed 2026-09-19 at DPCC's protocol — seeds 7–10 trained and evaluated, `tab:avoiding-dpcc-protocol` carries all five. The **extended** protocol (20 episodes) is still seed 6 only, which is what `tab:state-models` and `fig:avoiding-raw-models` rest on |
| R7 | 🟡 | Endpoint projection at its budget floor, 5 seeds × 20 episodes | `sec:res:avoiding:projection` `\provisional` (currently 4 seeds × 3 geometries × 2 episodes) | ⏳ |
| R8 | 🟡 | **D3IL-avoiding at the protocol of DPCC** (5 seeds × 2 episodes). **MeanFlow U-Net K1/K2 was NOT missing** — it is in the 15-09 batch under the same `Folder_Name` as the DiT runs, separable only by `Full_Path` (`bbunet`); filled in v3.27b. Still missing: flow matching K1/K2, and α-Flow U-Net α_end 0.2 K1/K2 (the latter needs training seeds 7–10; the 5-seed α-Flow cells there are `bbsit` with `ae0.0`) | primary comparison of Ch 6 since v3.19: `tab:avoiding-dpcc-protocol` *pending* rows + `\hole`; commands in [`REQUEST_20260917_avoiding_dpcc_protocol.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260917_avoiding_dpcc_protocol.md) | ✅ closed 2026-09-19 — batch `…20260919_132703`; twelve new cells drafted at thesis v3.41, the `\hole` removed |
| R9 | 🟡 | Flow matching re-evaluated at unit initial-noise scale | `sec:setup:protocol:avoiding` `\provisional` | ⏳ |
| R10 | 🟡 | S-curve endpoint-projection rows re-run after the switched-wall fix | excluded by the `\guard` in `sec:res:uav:projection`; `Gen15/U16/…FULL_REVIEW` §3.1 | ✅ closed 2026-09-19 — **run, all-zero**. Every post-fix cell is 0.00, and the divergence guard ends 57–92 % of flights because the vehicle inverts, on the unprojected plan too. Drafted as a failure case with its abort counts at thesis v3.41 (`tab:uav-scurve-aborts`) |
| R11 | ⚪ optional | Our own sweep of the projection's assumed sampling time (DPCC Table 2) — `dpcc-c-tightened-dt*` rows already exist in the 15-09 avoiding batch, so possibly **DA only** | would replace the citation in `sec:res:avoiding:smoothness` with a measurement | ⏳ |
| R12 | 🟠 | **Diffusion baseline on UAV-pillars in the six configurations it was never run in**: per-step with random selection, and all four endpoint-projection cells (single candidate, $r$, $c$, $t$). It currently has 5 of the 11 | `tab:uav-pillars` carries \emph{---: not evaluated} for the baseline, so the baseline column is not comparable with the flow-based ones configuration by configuration | ⏸ superseded 2026-09-19 by the pillars redesign (R23); the scene is removed from the draft, so the gap it describes no longer blocks any text |
| R24 | 🟡 | **The last empty panel of `fig:raw-plans`, diffusion $\nfe=2$** — a **run**, not a fetch, and not impossible: two routes exist, one needing a $K{=}2$ training (matches the current $\nfe=1$ panel), one needing two evaluations off the existing $K{=}20$ checkpoint (both diffusion panels then replaced together) | `fig:raw-plans` keeps one `\todofigure`, explained in the caption | ⏳ [`PENDING_20260919_fig63_diffusion_K2_panel.md`](PENDING_20260919_fig63_diffusion_K2_panel.md) |
| R23 | 🔴 | **UAV-pillars re-evaluation on an enlarged constraint set** (`pillars_xl`, obstacle radius 0.35). The scene as evaluated tests the constraint the demonstrations were built to satisfy, so the plan is feasible before projection; §6.3.1.2 is **withheld** from the draft until this lands | [`PENDING_20260918_pillars_geometry_redesign.md`](PENDING_20260918_pillars_geometry_redesign.md) · [`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](SLURM_RUNBOOK_20260919_pillars_enlarged.md) | ⏳ |
| R13 | 🟡 | **MeanFM and FM at $\nfe=1$ on UAV-pillars.** The scene has MeanFM and FM at $\nfe=2,5$ and CI-MeanFM at $\nfe=1,2,5$, so the budget ladder is short at its floor for two of the three | the step-budget paragraph of `sec:res:uav:pillars` compares $\nfe=2$ against $\nfe=5$ only, while CI-MeanFM is quoted from $\nfe=1$ | ⏸ superseded 2026-09-19 by the pillars redesign (R23); the scene is removed from the draft, so the gap it describes no longer blocks any text |
| R14 | 🟡 | **Candidate-matched endpoint-projection ladder on D3IL-avoiding**: endpoint projection was run with one candidate plan against per-step projection with four | `tab:hf-ladder` `\guard` — every row is confounded by the candidate count and the wall-clock columns of that evaluation are unusable | ⏳ |
| R15 | 🟡 | **MuJoCo MPC on UAV-s-curve beyond three flights** (ten, to match the brake-to-rest rows) | `tab:uav-controller` `\guard` (\enquote{Three flights with MuJoCo MPC}); the controller caveat rests on $n=3$ | ⏳ |
| R16 | ⚪ optional | **Budget ladder for FM and CI-MeanFM on D3IL-aligning** ($\nfe=2,10$ for FM; $\nfe=10$ for CI-MeanFM). The task has MeanFM at 2/10/20/100, CI-MeanFM at 2/20, FM at 20/100 only | `sec:res:aligning:budget` states the budget behaviour of MeanFM and CI-MeanFM; an FM ladder would make the budget claim model-independent | ⏳ |
| R20 | 🟠 | **Endpoint projection with random selection and with cumulative projection cost on UAV-corridor**, at every budget, and **all endpoint rows at $\nfe=1$**. The scene has endpoint projection only as a single candidate and under temporal consistency, at $\nfe=3,5$ | `sec:res:uav:projection` compares the methods on two of six endpoint configurations | ✅ closed 2026-09-19 for $\nfe=3,5$ — `hardflow_sls-r` and `-c` exist for all three flow models; drafted as `tab:uav-corridor-projection` at thesis v3.41. The $\nfe=1$ half is ❌ **not runnable**: at one ODE step the only step is the terminal one, so there is no guiding step to place |
| R21 | 🟠 | **Endpoint projection at $\nfe=1,2$ on UAV-pillars** (all four configurations). The scene has them at $\nfe=5$ only | the budget paragraph of `sec:res:uav:pillars` compares per-step rows only below $\nfe=5$ | ⏸ superseded 2026-09-19 by the pillars redesign (R23); the scene is removed from the draft, so the gap it describes no longer blocks any text |
| R22 | 🟠 | **The diffusion baseline under endpoint projection on any quadrotor scene** — it has none — and its per-step random-selection row on UAV-pillars | `tab:uav-pillars` shows the baseline in 5 of 11 configurations; the projection comparison of `sec:res:uav:projection` excludes the baseline entirely | ⏸ superseded 2026-09-19 by the pillars redesign (R23); the scene is removed from the draft, so the gap it describes no longer blocks any text |
| R18 | 🟠 | **The diffusion baseline at the extended protocol below its training budget**: $\nfe=1,2,10$ at 5 seeds × 20 episodes. They exist at DPCC's protocol (5 × 2) and are now in `tab:avoiding-dpcc-protocol`, but the extended table shows the baseline at $\nfe=20$ only | `tab:state-headline` `\hole` (must have); the budget collapse of the baseline is currently evidenced at ten episodes per geometry | ⏳ |
| R19 | 🟡 | **Instantaneous-velocity matching at $\nfe=1,2$ at DPCC's protocol**, and CI-MeanFM at $\nfe=1,2$ there (the latter after R6's trainings) | `tab:avoiding-dpcc-protocol` *pending* rows and its `\hole`; same request as R8 | ✅ closed 2026-09-19 with R8 |
| R17 | ⚪ optional | **Repeat evaluations of the projected alignment configurations** to quantify the ≈0.4 m run-to-run variation rather than cite it | `sec:res:aligning:projection` `\provisional` — it is why the projected median distances are not ranked | ⏳ |

## 2. Analyses of existing data (no runs)

| # | what | where it shows | status |
| :-- | :-- | :-- | :-- |
| D1 | **Plan smoothness metric** — jerk, path length or curvature over the saved plan `.npz` files (on cluster disk) | `sec:res:avoiding:raw` `\hole`, accompanying `fig:raw-plans`; the argument in `sec:res:avoiding:smoothness` rests on violation counts until it exists | ⏳ |
| D2 | Alignment **orientation error**: the angle columns differ between runs for identical contexts — establish the convention, then report | `sec:res:aligning:models` `\hole` | ⏳ |
| D3 | ~~Verify the ≈119 ms/step MuJoCo MPC overhead~~ | ❌ closed 2026-09-17 (v3.28): the claim was dropped from the draft rather than verified — the committed `avg_time_ms` does not support it (89.8 ms MPC against 88.4 ms brake-to-rest, unprojected) | ❌ |
| D4 | Alignment workspace box per geometry from `config/visual_aligning_eval.yaml` | §5.1.2 `\hole` | ✅ v3.25 (filled from `config/visual_aligning_eval.yaml`) |
| D5 | Quadrotor: confirm pillars/s-curve use the corridor's workspace box; state tightening 0.025 m | §5.1.3 `\hole` | ✅ v3.26 — not the same box: shared x∈[-4,4] m and altitude [0.30,1.80] m, lateral y ±2.5 m (pillars) / ±1.8 m (s-curve) / walls (corridor); tightening 0.025 m on top of the 0.31 m inflation (`config/uav_projection.yaml`) |
| D6 | Accepted demonstrations per quadrotor scene; controller gains per collection run; alignment demonstration count; train/validation split of all three environments | §5.2 `\hole` | ✅ v3.26 — 500/500 corridor, 475/500 pillars (18 floor, 7 contact), 500/500 s-curve, gains `pid_default` (collect jobs 21475/21482/21476; training buffers agree); alignment 900 demos, 168,274 windows; 90/10 window split everywhere |
| D7 | Confirm controller and velocity-setpoint policy for pillars and s-curve results | §5.5.3 `\hole` | ✅ v3.26 — `pid_stopgo` (brake-to-rest) on all pillars_hg rows and 880/889 s_curve_hg rows; the 9 others are the MuJoCo MPC comparison |
| D8 | Total compute (GPU-hours) from job logs | appendix `\hole` | ⏳ |
| D9 | SLURM job IDs and git revision per batch for `tab:corpora` | appendix `\hole` | ⏳ |
| D10 | **(needs the cluster)** **Executed trajectories for four result figures** (s-curve under two controllers, pillars per model, corridor through the slide, alignment box paths). The committed batches hold per-rollout scalars only; the positions are on the cluster. Runs and contents listed in [`REQUEST_20260917_trajectory_figures.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260917_trajectory_figures.md) | §6.2, §6.3 `\hole`s + `fig:uav-scurve-paths` `\todofigure` | 🟠 **three of four landed** (2026-09-18) — the quadrotor drop is staged and `fig_uav_scurve_paths`, `fig_uav_pillars_paths` and `fig_uav_corridor_paths` are in the draft (v3.35). **The alignment box paths are still open**: that cell was staged before a naming bug in the fetch script was fixed and is absent from the download; it needs one more run of the stager |

## 3. Decisions for the author

| # | what | context | status |
| :-- | :-- | :-- | :-- |
| A1 | Report the **D3IL image policy** run in our pipeline (`d3il_baseline`, test split, 60 contexts: 0.44 m from 0.45 m, box unmoved in 44 % of 3,884 episodes)? | §5.3.2 says no alignment baseline exists | ⏳ |
| A2 | Figs 5.2 / 5.3 use crops of **D3IL's own README GIF** without credit — credit D3IL, or replace with our MuJoCo renders | provenance; `plotting/NOTEBOOK_20260915…` | ⏳ |
| A3 | Keep `tab:target` (published vs reproduced) under v2.16's "describe only our configuration" rule? | §5.3.1; neither of our episode counts equals the published 50 | ⏳ |
| A4 | Training is not uniform across models (batch, lr, action weight, EMA) — accept with the `\guard`, or retrain matched | `tab:train`; `cross_draft/to_v2/FROM_v3_20260916_training_not_uniform.md` | ⏳ |

## 4. Coverage of the Results chapter (re-traversed 2026-09-17, v3.30)

Every drafting marker and every `\guard` in `chapters/06_results.tex`, and every marker in `05_setup.tex`
and `09_appendix.tex`, mapped to the row that owns it. If a marker is added to the draft, add its row here.

| where in the draft | marker | owned by |
| :-- | :-- | :-- |
| `tab:avoiding-dpcc-protocol` — the four pending rows | ✅ v3.41 filled, `\hole` removed | R8, R19 |
| `tab:avoiding-dpcc-protocol` — no $\nfe=2$ diffusion row, no $\nfe=10$ FM row | `\guard` (acceptable) | covered by `fig:k-ladder` |
| `tab:state-headline` — baseline at $\nfe=20$ only; no CI-MeanFM at the extended protocol | `\hole` (must have) | R18, R6 (extended half only) |
| `tab:state-headline` — no FM $\nfe=10$; MeanFM $\nfe=20$ on two geometries | `\guard` (acceptable) | covered by `fig:k-ladder` |
| `fig:raw-plans` — one empty panel, diffusion $\nfe=2$ | 1 × `\todofigure`, explained in the caption | ✅ R5 closed; the panel itself is **R24** |
| `sec:res:avoiding:raw` — quantify plan smoothness | `\hole` | D1 |
| `sec:res:avoiding:smoothness` — DPCC Table 2 cited, not measured here | (no marker) | R11 (optional) |
| `tab:hf-ladder` — endpoint $B=1$ against per-step $B=4$ | `\guard` | R14 |
| `sec:res:avoiding:projection` — endpoint floor at 4 seeds × 3 × 2 | `\provisional` | R7 |
| `tab:state-models`, `fig:avoiding-raw-models` — CI-MeanFM seed 6 only | `\guard` | R6 |
| `sec:res:visual` — ten training contexts | `\guard` | R3 |
| `sec:res:aligning:models` — orientation error | `\hole` | D2 |
| `sec:res:aligning:budget` — ladder exists for MeanFM (and CI-MeanFM) only | (no marker) | R16 (optional) |
| `sec:res:aligning:projection` — ≈0.4 m run-to-run variation | `\provisional` | R17 (optional) |
| `sec:res:aligning:projection` — no tightened diffusion cell | `\provisional` | R2 |
| `sec:res:aligning:projection` — figure of the ten contexts | `\hole` (still open) | D10 |
| `sec:res:uav:corridor` — figure of the flown paths | ✅ v3.35 `fig:uav-corridor-paths` | D10 |
| `sec:res:uav` — UAV-pillars removed from the draft | `\guard` in the §6.3 opening | R23 (R12, R13, R21, R22 ⏸ with it) |
| `tab:uav-corridor-projection` — no endpoint row at $\nfe=1$ | `\guard` | ✅ R20, not runnable at that budget |
| `tab:uav-scurve-aborts` — the re-run is all-zero | (no marker; abort counts printed) | ✅ R10 |
| `fig:uav-scurve-paths` | ✅ v3.35 drawn | D10 |
| `tab:uav-controller` — three MPC flights | `\guard` | R15 |
| every quadrotor and alignment result — one seed | (stated once in §5.6) | R1 ❌, R4 ❌ (closed, not feasible) |
| `sec:setup:protocol:avoiding` — unit initial-noise scale | `\provisional` | R9 |
| `sec:setup:protocol:aligning` — fifty contexts pending | `\provisional` | R3 |
| `app:repro` — GPU-hours | `\hole` | D8 |
| `tab:corpora` — SLURM job ids and git revisions | `\hole` | D9 |

The four `\hole`s in `07_discussion.tex` and `08_conclusion.tex` are \enquote{written/refined in v4} and are
draft-ownership items, not data gaps (§4 below).

## 5. Owned elsewhere (not v3 data work)
- Ch 7 Discussion, Ch 8 refinement and future work → v4 (`cross_draft/to_v4/`).
