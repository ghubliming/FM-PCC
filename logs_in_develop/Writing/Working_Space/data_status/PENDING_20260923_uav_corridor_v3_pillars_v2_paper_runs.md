# PENDING RUNS 23-09 — UAV-corridor v3 and UAV-pillars v2: the minimal, paper-only runs (R33, R39)

**2026-09-23 · for the run agent · scope: the two quadrotor scenes only.** Everything else stays in
`PENDING_20260922_all_lacking_runs.md`. The corridor items of that ledger (R30, the 22-09 corridor spec) and the
pillars items (R32) are superseded by this file and say so there. Companion runbook:
[`SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md`](SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md).

## 0 · Principles (author, 2026-09-22/23) — read before submitting anything

1. **Paper-only runs.** Only variants that are printed in Chapter 6 are run. No `geo_free`, `model_free`, `bounds_free`
   alone, `gradient`, `post_processing`, `dt*`, `hardflow_new` (the thesis uses `hardflow_sls`), no untightened twin,
   no MuJoCo-MPC controller, no ablation of any kind. Every `UAV_MIX_VARIANTS` list below is exhaustive.
2. **Tightened only.** Every projected variant carries `-tightened` (v3.67 convention, Chapter 5 §5.2.1).
3. **Corridor: one seed (6), twelve flights (4 per route). Pillars v2 (author, 23-09, final): the four Chapter 6 cells
   evaluated live (real eval, no turbo) at the table's protocol, 5 seeds × 2 episodes × 3 geometries, tightened only** (§2).
4. **A special run message on everything**, so the folders say "paper, 23-09": corridor eval tags **`p23cv3t`** /
   **`p23cv3ah`** (`FMPCC_UAV_EVAL_TAG`), pillars-v2 tag **`p23uavpv2live`** (live; must contain `uav`). ~~`p23pv2turbo`~~, ~~`p23pv2live`~~ are not run.
   Never pool with `u17cv2`, `u7hg`, `uavpv2s10*`.
5. **Order = fast first, K = 20 last**, diffusion K = 20 projected very last, so it cannot jam the queue.
6. **Time to the deadline is short.** If a wave has to be cut, cut from the bottom of each table below.

## 1 · R33 — UAV-corridor v3

### 1.1 Geometry — TWO scenes, read separately (author 22/23-09; corrected here after the U19 note of 23-09)

| entry (`config/uav_projection.yaml`) | what | suffix | tag |
| :-- | :-- | :-- | :-- |
| **`corridor_v3_tilt`** — the corridor | the v2 slide **leaned over** by −60° about the launch altitude (`z_lean: {deg: -60, z_ref: 1.11}`): one x–y–z plane that pushes the drone sideways **and down**; walls, caps, box, ceiling 1.80, XML, model, routes, goals byte-identical to v2 | `_cv3t` | **`p23cv3t`** |
| **`corridor_v3_ablation_hump`** — the second constraint | no slide; the x–z roof (0 → 1.10 m at x = 0 → 0 over x ∈ [−1.5, 1.5], two `plane: xz` halfspaces), ceiling 1.80 → 2.80 | `_cv3ah` | **`p23cv3ah`** |

The thesis reads the two as separate constraint geometries, the way D3IL-avoiding reads top-left-hard / top-right-hard /
both-hard: every corridor table has a *tilt* block and a *hump* block, nothing is averaged across them, and nothing is
called "v3" or "ablation" in the text — it is UAV-corridor with two constraints. **There is no combined slide + hump
scene.** (`corridor_v3_ablation_hump_lo`, H = 0.90, is defined and not scheduled.)

**Pilots (C0) are done — 22-09, jobs 26071/26072** (`Gen15/U19/PILOT_20260922_U19_gates_G1-G3.md`): G0 and G1 pass on
both; G3 passes (projected flights descend 0.15–0.29 m under the tilt, climb 0.30–0.40 m over the hump); G2 fails on
every projected arm by 6–11 shallow steps (≤ 2–6 cm) at the geometric switch point with the setpoint clean (plant lag
~0.5 m). **Author's decision: run the waves as they are; no scene change.** The "stop if G2 fails" rule below is
overridden for this reason.

### 1.2 The grid (68 cells per scene → 136 cells, 1 632 flights)

| model | $\nfe$ | unprojected | per-step (DPCC projector, η = 0.5) | endpoint (HF) |
| :-- | :-- | :-- | :-- | :-- |
| MeanFM | 1, 2, 3 | `diffuser` | `dpcc-{r,c,t}-bounds_free-pdes-tightened` at every budget | `hardflow_sls{,-r,-c,-t}-bounds_free-pdes-tightened` at **3** |
| CI-MeanFM α0.2 | 1, 2, 3 | same | same | at **3** |
| FM | 1, 2, 3, 5, 20 | same | same | at **3, 5, 20** |
| Diffusion | 20 | `diffuser` | `dpcc-{r,c,t}-bounds_free-pdes-tightened` | — (not defined) |

Stack as U16: `FMPCC_SAFE_EPS_FRAC=1.0`, `FMPCC_SAFE_EPS_MODE=scaled`, `pid_stopgo`, `mpc4`, `T0.5`, af knobs
`UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest`. HF only where it has a guiding step (K ≥ 3). **Both scenes get the same grid.** Input names for HardFlow are `hardflow_new{,-r,-c,-t}-…` (the eval writes the folders as `hardflow_sls-…`, which is what the DA reads).

### 1.3 Gates (stop at the first failure; the U11–U14 lesson)

| gate | test | pass |
| :-- | :-- | :-- |
| G0 | offline preview: inflated hump + slide vs the v2 flown band (0.86–1.24 m; y band ±0.64) | hump roof at the peak > 1.24 m; escape slot ≥ 0.6 m; lateral gap ≥ 0.31 m stays open |
| G1 | smoke: mf K3, 3 trials (L/C/R), `diffuser` + `dpcc-t-…` + `hardflow_new-t-…`, GIF on — **done 22-09** | log shows the scene, its tag, exactly the requested variants; `diffuser` violates on 3/3 — **pass on both scenes** |
| G2 | same smoke | at least one projected arm collision-free ≥ 2/3 and success ≥ 2/3 — **failed on both by a plant-lag residue; author: run anyway** |
| G3 | paired z of `dpcc-t` vs `diffuser` | > 0.15 m on every projected flight — **pass on both** (descent 0.15–0.29 m, climb 0.30–0.40 m) |

### 1.4 Waves (details and commands in the runbook)

| wave | cells | ~GPU | reads |
| :-- | :-- | --: | :-- |
| C0 | smoke (G1–G3) — **done** | — | — |
| C1 | all `diffuser`: mf/af K1,2,3; fm K1,2,3,5,20; diffusion 20 | 1 h | Table 6.9 complete |
| C2 | PCC r/c/t: mf/af/fm at K1, 2, 3 | 3 h | Table 6.11/6.16 flow rows at K ≤ 3 |
| C3 | HF single/r/c/t: mf/af at K3, fm at K3, 5; PCC r/c/t fm K5 | 3 h | HF-vs-PCC comparison |
| C4 | fm K20: PCC r/c/t + HF single/r/c/t | 4 h | the FM ladder top |
| C5 | **diffusion K20: PCC r/c/t — last, one variant per job** | ~1 GPU-day per variant on v2; `c` near the 24 h wall | the baseline |

Each wave runs on **both scenes** (`p23cv3t` then `p23cv3ah`); driver `Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh` (`plan` / `smoke` / `submit C1..C5|all`). The HF jobs carry `dpcc-t` (the eval's matched-budget guard) and C2 runs `dpcc-t` only at K = 1, 2 — same 68 cells per scene, each once.

### 1.5 What the thesis reads (structure already in Chapter 6, cells blank)

Table 6.9 before projection (+ a frontier only if not every model crosses the line); Table 6.11 after projection with
PCC and HF side by side at each cell's best rule + Fig 6.9 frontier (both projectors) + Fig 6.10 paths + Fig 6.11
altitude; Table 6.16 r/c/t for both projectors at K ≥ 3; §6.3.5 sentence. DA: `Data_Analysis/DA_UAV_v1` (check the
`_cv3` tag parses, `discovery.py` l.117); figure store extracts from the rollouts (`uav_paths.py` groups `corridor`,
`corridor_altitude`).

## 2 · R39 — UAV-pillars v2: the Chapter 6 cells evaluated live with the quadrotor (rev 23-09, author: "real eval, no turbo")

> ✅ **LANDED 2026-09-23** — jobs 26151–26154, 240 flights, complete and clean; analysis of record `DA_in_Paper/analysis/DA_20260923_pillars_v2_live.md`; **read into Chapter 6 at v3.72** (`tab:uav-pillars-raw` air columns, `tab:uav-pillars-geo`, `fig:uav-pillars-paths`). Nothing of R39 is still owed.


### 2.1 What Chapter 6 reads, and therefore what is run

UAV-pillars is the replica of D3IL-avoiding in the air (Ch 5 §5.1): the avoiding obstacle field scaled by 36, the three
test-time geometries mapped into it, the avoiding planner not retrained. The scene tests the **vehicle and its tracking
controller**, not the planner: the configurations that win on the table are flown and their flown outcome is set beside
their table outcome, before and after projection. Agreement or a stated difference are both valid results.

Chapter 6 (v3.69, `sec:res:uav:pillars`) reads exactly:

| Ch 6 item | cells | from R39 |
| :-- | :-- | :-- |
| `tab:uav-pillars-raw`, *air* columns (S&C, violating steps, steps, contact) | MeanFM K1, MeanFM K2, CI-MeanFM α_end 0.2 K1, CI-MeanFM K2 — the *table* columns are the existing avoiding cells (Tables 6.1/6.2, 19-09 corpus) | `diffuser` + `dpcc-t-tightened`, seeds 6–10 × 3 geometries × 2 episodes |
| `fig:uav-pillars-paths` | one configuration picked from the appendix `fig:avoiding-paths` (MeanFM K1 = its first panel, the selected configuration; seed 6, *top-right-hard*, episode 2): the arm's executed path beside the quadrotor's flown path, unprojected grey + projected colour | the same live runs (seed 6 flights + plant sidecar world paths); no extra run |
| conclusion line `sec:res:uav:conclusion` | the four cells | the same |

**Real evaluation only (Mode L).** The planner runs in the loop with the quadrotor as the plant: at every control step it
plans from the vehicle's state, and the vehicle flies the commanded setpoint. The turbo replay (Mode T) is **dropped**: it
re-flies the manipulator's stored setpoints open loop, so it measures the plant on the Panda's plans but the planner and
projector never see the vehicle — it cannot produce a new planning number. No turbo cell is read by the thesis.

### 2.2 Gates — done (22-09)

G0 (scene compiles, frame round-trip) and the live check (job 26077, FM K20, 20 episodes: live within 0.05 of the Panda on
every cell, 0 contacts, 0 divergence; `Gen15/U18/L1_20260922_live_vs_turbo_result.md`) passed. Plant settings as validated
there: scale 36, clock replay 1 Hz, feed-forward off, v_max 1.0 m/s. Nothing to re-gate.

### 2.3 The runs — four GPU jobs, tag `p23uavpv2live`

| # | model | $\nfe$ | source cell of the table (19-09 corpus) | variants | protocol |
| :-- | :-- | --: | :-- | :-- | :-- |
| L1 | MeanFM | 1 | `flow_matching_v3_meanflow/…objmeanflow_bbunet…/H8_K1_Meuler_T0.5_A0.5_B1_…MeanFlowODE` | `diffuser`, `dpcc-t-tightened` | seeds 6–10 × 3 geometries × 2 episodes = 30 per variant |
| L2 | MeanFM | 2 | same train, `H8_K2_…_A0.5_B1_…MeanFlowODE` | same | same |
| L3 | CI-MeanFM α_end 0.2 | 1 | `flow_matching_v3_alphaflow/…bbunet…ae0.2…/H8_K1_…AlphaFlowODE_msgdpccproto` (`AF_EPOCH=latest`) | same | same |
| L4 | CI-MeanFM α_end 0.2 | 2 | same train, `H8_K2_…_msgdpccproto` | same | same |

Four candidate plans (`FMPCC_MPC_BATCH=4`), per-step projection threshold 0.5, tightened margin 0.025 — the table cells'
settings; configs `config/meanflow_projection_eval_u18_live.yaml` and `config/alphaflow_projection_eval_u18_live.yaml`
(the shared eval yamls with only `projection_variants` reduced to the two above). 240 flights in total; each job < 2 h GPU.

🔴 **Tag must contain `uav`** (`uav_avoiding_bridge/factory.py` refuses otherwise, because the live results share the
Panda result layout). The earlier live drivers `live_p2_*.sh` default to `p23pv2live`, which does **not** contain it —
they would stop at the plant switch; they are superseded by this section and are not used.

Submit: `bash Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh submit` (runbook §3).

### 2.4 What the DA produces

- Table: per cell and variant, S&C / violating steps / steps averaged per geometry then over the three (as Table 6.2),
  plus contact count from the plant sidecars; the *table* columns stay the 19-09 cells. Never pooled with `uavpv2s*` /
  `p23pv2turbo` / the Panda runs (the `_msgp23uavpv2live` folders are separate by construction).
- Figure: `fig_uav_pillars_paths` — the picked panel (MeanFM K1, seed 6, *top-right-hard*, episode 2): arm path from the
  existing `diffuser.npz` / `dpcc-t-tightened.npz` (`extract/exec_paths.py`), quadrotor path from the live npz `obs_all`
  of the same seed / geometry / episode (world frame via `uav_avoiding_bridge/frame.py`), unprojected grey, projected colour.

### 2.5 Superseded (kept for the record, not run)

- P1 turbo T1–T5 (tag `p23pv2turbo`, `turbo.sh MODE=paper|papergif`) — dropped 23-09 (turbo cannot give a planning number).
- P2 live L1/L2 of the first draft (MeanFM K1 and diffusion K20, seed 6, tag `p23pv2live`) — replaced by §2.3.

## 4 · UAV-s-curve — moved to its own file (v3.75)

~~R40~~ (struck v3.71) and ~~R42~~ (the v3.71 s-curve rebuild: grid with MeanFM K10 / CI-MeanFM K5, FM K2 selected, projection and MJPC on FM K2) are **superseded** by [`PENDING_20260923_uav_scurve_R44_raw_first.md`](PENDING_20260923_uav_scurve_R44_raw_first.md) (**R44**, author 23-09: grid K ∈ {1, 2, 20}, diffusion K20; raw grid first, projection and controller only after the author reads it). Nothing s-curve is run from this file.

## 5 · Issues noticed while writing the 23-09 runbook (for the run agent)

1. The `--engine` substrings in the pillars-v2 turbo commands are guesses at the `plans/` folder names on the cluster;
   `--dry-run` must list exactly ten cells before `GO=1`.
2. `DA_UAV_v1/discovery.py` has never seen a `_cv3` geometry tag; check l.117 parses it before the C1 batch.
3. The U18 scale changed from 10 (plan) to **36** (fix3, `frame.py`); Chapter 5 now describes the arena at 36
   (21.6 × 25.2 m, keep-out 2.88 m, pillars 0.90/1.08 m). If the pilot ends at another scale, tell v3.
4. ~~joint G0 for slide + roof~~ — moot: the two constraints are separate scenes (U19 note, 23-09); G0 passed on each.
5. ~~coding first~~ — done: `corridor_v3_tilt` (`z_lean`) and `corridor_v3_ablation_hump` (`plane: xz`) are in `config/uav_projection.yaml`; `DA_UAV_v1/discovery.py` parses the leading `corridor` token (checked by the run side).
6. The chapter's figures (`fig_constraints_uav`, `fig_expert_uav`, the pillars renders) still draw `pillars_hg` and the
   corridor without its roof; the figure store needs a pillars-v2 arena panel (from `frame.py`) and a corridor-v3 panel
   after the runs — a DA_in_Paper task, listed here so it is not forgotten.

## 3 · Cost summary

| item | GPU | CPU |
| :-- | --: | --: |
| R33 corridor, C1–C4, **both scenes** | ~22 h | — |
| R33 corridor, C5 diffusion K20, both scenes | ~2 GPU-days (cut to `t` only: ~16 h) | — |
| ~~R39 pillars v2~~ ✅ landed 23-09 (26151–26154) | — | — |
| ~~R40~~ / ~~R42~~ s-curve → **R44**, own file (PENDING_20260923_uav_scurve_R44_raw_first.md) | — | — |

Claude (Fable 5.1, Claude Code) · 2026-09-23 · specification only; nothing submitted, nothing written into the thesis.
