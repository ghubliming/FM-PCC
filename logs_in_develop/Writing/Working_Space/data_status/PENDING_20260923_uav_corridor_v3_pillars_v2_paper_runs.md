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
3. **One seed (6) everywhere.** Corridor: twelve flights (4 per route). **Pillars v2 (author, 23-09): one seed, tightened
   only, turbo (Mode T) only** — seed 6 × 3 geometries × 2 episodes = 6 episodes per cell; T5's source has no seed 6
   (it holds 7–10), so T5 uses seed 7. No live (Mode L) run.
4. **A special run message on everything**, so the folders say "paper, 23-09": corridor eval tags **`p23cv3t`** /
   **`p23cv3ah`** (`FMPCC_UAV_EVAL_TAG`), pillars-v2 tag **`p23pv2turbo`** (Mode T). ~~`p23pv2live`~~ (Mode L) is not run.
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

## 2 · R39 — UAV-pillars v2: the avoiding planner flown by the quadrotor (Gen15 U18)

### 2.1 What is being proved, and therefore what is run

The claim the thesis makes on this scene is small and must be made with a small, defensible set of runs: *the plans the
avoiding planner produces under projection are flyable by the quadrotor in a scene that is the avoiding obstacle field
with pillars, and the closed loop behaves the same when the drone is in it.* Two run modes exist in `uav_avoiding_bridge/`
(Mode T replays the stored avoiding executions through the drone plant on CPU; Mode L runs the planner live with the
drone as the plant). **The thesis text will say only that the avoiding planner was evaluated again in an avoiding-like
quadrotor scene with pillars — the mechanism (replay) is not described there.** This file may say it; Chapter 6 may not.

So: not the full avoiding grid. The **representative set** = the configurations Chapter 6 already selects on
D3IL-avoiding (Table 6.4 and the projector table), each *before* and *after* projection, at the DPCC protocol, plus one
live check.

### 2.2 Gates first (U18 plan §4; unchanged)

| gate | what | pass |
| :-- | :-- | :-- |
| G0 | `scene.py` XML compiles; frame round-trip is the identity | assert |
| G1 | `turbo.sh` pilot (`MODE=pilot`: FM K20 `msg20trials`, seed 6, 3 geometries, `diffuser` + `dpcc-r-tightened`, 20 episodes, `clock` replay) | ≥ 95 % of episodes reproduce the Panda's (success, collision-free); no contact on a Panda-legal episode; gap p95 within the training range. Else `HZ=3`, `EXTRA="--gain pid_high_gain"`, `settle`, then `SCALE` up |
| G3 | the gap check from the pilot log | p95 inside the normaliser's range |

### 2.3 P1 — Mode T, the representative replay (CPU, minutes) · tag `p23pv2turbo`

Filtered with `--engine / --train-glob / --eval-glob / --seeds / --geos / --variants` to exactly these cells, **one seed**
(author, 23-09): **seed 6 × 3 geometries × 2 episodes = 6 episodes per cell** unless stated. Projected variants are
tightened only; `diffuser` is the unprojected reference of the same cell (before / after projection).

| # | model | $\nfe$ | variants (both replayed) | why |
| :-- | :-- | --: | :-- | :-- |
| T1 | MeanFM | 1 | `diffuser`, `dpcc-t-tightened` | the selected configuration of D3IL-avoiding (Table 6.4) |
| T2 | CI-MeanFM α0.2 | 1 | `diffuser`, `dpcc-t-tightened` | the second supported combination |
| T3 | FM | 1 | `diffuser`, `dpcc-t-tightened` | the third flow model, so all four models appear once |
| T4 | Diffusion (DPCC) | 20 | `diffuser`, `dpcc-c-tightened` | the baseline at its own budget and best rule |
| T5 | MeanFM | 3 | `dpcc-t-tightened`, `hardflow_sls-t-tightened` (**seed 7**, 6 episodes — the source has no seed 6) | the matched endpoint-vs-per-step cell of §6.1.2.6 — shows an endpoint-projected plan is flyable too |

Ten cells, 60 episodes, **a few minutes CPU**. `clock` replay only (`settle` only if G1 needed it). `--gif 2` on the GPU
variant of the driver for T1 and T4 (two episodes each) — the only pictures the thesis needs.

### 2.4 P2 — Mode L, the live closed-loop check · **DROPPED (author, 23-09: pillars run in turbo mode only)**

The live gate check of 22-09 (job 26077, `Gen15/U18/L1_20260922_live_vs_turbo_result.md`: live within 0.05 of the Panda on
every cell, 0 contacts) stays the evidence that the replay stands in for the live loop. The table below is kept for the
record only; do not submit it.

| # | model | $\nfe$ | variants | protocol |
| :-- | :-- | --: | :-- | :-- |
| L1 | MeanFM | 1 | `diffuser`, `dpcc-t-tightened` | seed 6, 3 geometries, 2 episodes (12 flights) |
| L2 | Diffusion | 20 | `dpcc-c-tightened` | seed 6, 3 geometries, 2 episodes (6 flights) |

Pass = L1/L2 agree with T1/T4 within the Panda cell's seed spread on (success, collision-free, violating steps). If they
do not, the disagreement is the result to report and no further live cells are run without the author.

### 2.5 What the thesis reads

One table: for T1–T5, the Panda's S&C / violating steps / steps beside the drone's, before and after projection, plus
the plant columns (tracking error, contacts); one world-frame path figure over the physical pillars; one sentence that
the live check agrees (the 22-09 gate check, job 26077 — no new live run). Written into the pillars slot of Chapter 6 after the flawed block is lifted — **without the
replay mechanism**. DA script `DA_in_Paper/analysis/pillars_v2_grid.py` (to write once the data exist; template in
U18 plan §6).

## 4 · R40 — UAV-s-curve: the one datum the controller caveat still lacks (added with v3.68)

The chapter now reports the s-curve as a caveat with its pilot cells only (MeanFM K10, CI-MeanFM K5, FM K20,
diffusion K20, unprojected: 4–10 of 10 flights inverted). The author expected the unprojected plan to pass the goal
line; the corpus says it does not under `pid_stopgo`, while the same MeanFM K10 plans under MuJoCo MPC end 0.30 m from
the goal and cross the line on 2 of 3 flights (Table 6.18). To close the caveat as a *controller* result and not leave
"maybe a bug" open, one cheap unprojected pilot:

| # | cell | controller / velocity setpoint | flights | tag |
| :-- | :-- | :-- | --: | :-- |
| S1 | MeanFM K10 `diffuser` | `pid` ($v_{des} = \Delta p / \Delta t$, the timing-sensitive default) | 10 | `p23sc` |
| S2 | MeanFM K10 `diffuser` | `pid_const_v` (unit direction × 0.4 m/s) | 10 | `p23sc` |
| S3 | the **expert reference** of the same routes tracked by `pid_stopgo` (no network: replay the demonstration setpoints) | brake-to-rest | 10 | `p23sc` |
| S4 (optional) | FM K20 `diffuser` under `pid` and `pid_const_v` | | 20 | `p23sc` |

Reading: if S1/S2 fly the turns, the caveat is the zero velocity setpoint (as the chapter argues); if S3 also inverts,
the tracker cannot fly *any* setpoint sequence through the turns without feed-forward and the demonstrations only
succeeded because they carried it; if S1–S3 all invert, something in the eval loop differs from the collection loop and
that is a bug to find before anything else is claimed. Cost: ~20 min GPU (unprojected only). `UAV_MIX_CONTROLLER` selects
the tracker; no projection variant, no other K.

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
| R39 pillars v2, P1 (turbo, one seed; G1 done 22-09) | — | minutes |
| ~~R39 pillars v2, P2 live~~ — dropped (author, 23-09) | — | — |
| R40 s-curve controller pilot | ~20 min | — |

Claude (Fable 5.1, Claude Code) · 2026-09-23 · specification only; nothing submitted, nothing written into the thesis.
