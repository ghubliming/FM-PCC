# Gen11 Epoch 6 — FM-PCC Training: EXECUTION PLAN

**Date:** 2026-06-21
**Status:** Plan — ready to execute. Builds on [IDEAS.md](IDEAS.md).
**Scope this Epoch:** **build the UAV state-only Flow-Matching policy** by forking FMv3-ODE-selectable and
re-wiring it to the UAV data. DPCC-baseline ablation is an optional tail *after* FM works. **Visual FM
(Gen7) is deferred to the next Epoch.**

---

## What is / isn't the E6 *coding* job (read first)

- **PREP (NOT E6 coding — data side, E4-flavoured):** the pillars recollect, dataset curation, and the
  mini-FM sanity gate (this plan's "Phase 0" and "Phase 1"). These produce/validate the dataset; they are
  prerequisites the build *assumes*, not code I write as the E6 deliverable. They are kept below as a short
  **Prerequisites** section so the dependency is explicit.
- **THE E6 CODING JOB (mine):** fork FMv3-ODE-selectable → a UAV FM, train + eval it closed-loop, write the
  **CHANGELOG** and a **USAGE / how-to-run MD**. Workflow:
  1. **Understand the UAV data** — schema/convention (§ schema box), curated layout (§0b).
  2. **Understand FMPCC / FMv3-ODE-selectable** — model, dataset, sampling, training loop, config wiring.
  3. **Merge → build the UAV FM** — copy/modify per Phase 2, mirroring the source as closely as possible.
  4. **Blocker policy:** if a *fundamental* math / robotics problem surfaces (e.g. action-convention or
     dynamics mismatch the fork can't absorb), **try to solve it first**; if it can't be solved, **ABORT
     and report exactly why we can't move forward** — do not paper over it.

> **Ground rule — minimal new code/files.** Mirror the existing FMPCC codebase 1:1; invent as little as
> possible. Fewer new files = easier debugging. Reuse the legacy config/script naming and structure (see
> §2C); do **not** invent parallel schemes.

---

## 0. How this plan maps onto the IDEAS phases (read first)

The IDEAS.md uses 5 phases (P0 data → P1 mini-FM → P2 state-FM → P3 DPCC → P4 visual). This plan groups
them per the Epoch-6 decision:

| This plan | = IDEAS phase | E6 coding job? |
|---|---|---|
| **Phase 0 — Recollect pillars + curate dataset** | P0 | ⛔ **PREP** (data side / E4) — prerequisite, not E6 code |
| **Phase 1 — Mini-FM gate** | P1 | ⛔ **PREP** — run the *existing* script on curated data |
| **Phase 2 — Build + train UAV state FM, multi-env** (the 2-folder fork) | P2 (+ projector scaffold from P3) | ✅ **THE E6 CODING JOB** — fork + `--scene all/<one>` |
| **Phase 2b — DPCC diffusion baseline (ablation)** | D2 / `diffuser/` | ⏸ optional, **only after** FM succeeds |
| **Phase 3 — Visual FM-PCC** | P4, **based on Gen7** (`fm_encdec_vision`) | ❌ **next Epoch** |

> The DPCC **safety projector** (`sampling/projection.py`) ships with the fork and is copied in Phase 2,
> but its UAV-dynamics constraint wiring is **not activated** this Epoch — state-only FM must close the
> loop first. Full DPCC-safety projection is a follow-on (next Epoch, alongside or after Gen7 visual).

### ⚠ Schema correction (authoritative numbers)
IDEAS.md §2 still says `obs=(T,6)`, `D=9`. **That is stale.** The current, authoritative UAV schema is the
one in `uav_expert_data_collect/mini_fm_sanity.py` (its "U2" update):

```
HORIZON     H  = 8
OBS_DIM        = 9     #  [ p_des(3) | p(3) | v(3) ]
ACTION_DIM     = 3     #  [ Δp_des(3) ]
transition_dim = 12    #  ACTION_DIM + OBS_DIM = 3 + 9
T_FLOW         = 20    #  flow ODE steps (fast)
```
Full on-disk episode pickle (richer than the mini-FM uses — keep all of it, the extra fields feed eval +
the later DPCC stage):
```
ep = { 'episode_id', 'scene', 'homotopy', 'controller', 'dt',
       'obs'      : (T, 9)   [ p_des | p | v ],
       'actions'  : (T-1, 3) Δp_des,            # = np.diff(targets) — POSITION-DELTA convention
       'targets'  : (T, 3)   p_des,
       'q'        : (T, 4)   orientation quat,
       'obstacles': scene geoms,                 # ← consumed by DPCC projector later, not by state FM
       'metadata' : { contact_fraction, total_time, dt_physics, controller_gains, noise_sigma, ... } }
```
**Use 12/9/3/8 everywhere — do NOT copy the D3IL-avoiding `(T,9)` transition shape.**

> **Raw vs curated (important — see §0b):** the FM trainer must **not** read the raw collection tree
> directly. Raw episodes live under `logs/uav_expert_data/<scene>/<homotopy>/<ep>.pkl` *alongside*
> `run_summary.json`, and next to debug siblings (`logs/uav_expert_data_stress/…` from U10, plus E5
> GIF/overview/camera dumps). Phase 0 **curates** the accepted episodes into a separate, versioned,
> training-only dataset folder with a manifest; the loader points only there.

---

# ══ PREREQUISITES (data prep — NOT the E6 coding job) ══

> Phases 0 & 1 below are **data-side prep** (E4-flavoured): recollect pillars, curate the dataset, run the
> existing mini-FM gate. They are listed so the build's dependency on a clean curated dataset is explicit.
> The **E6 coding job starts at Phase 2.** `curate_dataset.py` is a tiny prep helper, not the deliverable.

## 0b. Data layout & naming — raw → curated → training (decouple from debug clutter)

The single most important data decision this Epoch: **the FM pipeline reads a curated dataset folder, not
the E4 raw collection tree.** Three distinct trees, never overloaded:

```
RAW (E4 output, do not point the trainer here):
  logs/uav_expert_data/<scene>/<homotopy>/<ep_id>.pkl   + run_summary.json
  logs/uav_expert_data_stress/<scene>/<case>/<ep>.pkl   ← U10 deliberately-BAD episodes
  (+ E5 GIF / overview-plot / camera dumps near these dirs)

CURATED (NEW — the only thing the trainer reads):
  data/uav_fm/v1/
    ├── manifest.json          # schema, per-scene counts, provenance (which collect run), git rev, date
    ├── empty/<ep_id>.pkl
    ├── corridor/<ep_id>.pkl
    ├── s_curve/<ep_id>.pkl
    └── pillars/<ep_id>.pkl    # the recollected v2 set only

OUTPUTS (NEW — train/eval logs, see Phase 2E):
  logs/fm_uav/<run_id>/<scene-or-all>/{weights,losses,eval,plots,timing}
```

**Why a separate curated folder (the user's point):** pointing the loader at `logs/uav_expert_data/<scene>/`
would (a) ingest `run_summary.json` and any future debug files, (b) sit one glob away from the
`_stress` bad-episode tree, (c) silently include the rejected/old pillars if not careful. Curation copies
**only accepted** episode pkls into a versioned (`v1`, `v2`, …), manifest-tracked tree. Recollects bump the
version; nothing clobbers; the trainer is reproducible from the manifest alone. The raw E4 name is never
imported.

---

## Phase 0 — Recollect pillars + curate the dataset (BLOCKING) — TODO + commands

Goal: (i) replace the bad 274-episode pillars set with a clean 500, then (ii) **curate** all 4 clean scenes
into `data/uav_fm/v1/`. `BLEND_RADIUS` is **already** `0.45` in `trajectories.py:32` (IDEAS step 1 done) —
recollection is run-only. Data state per [U9_EVAL_RESULTS](../Epoch4_expert_data/U9_Smooth_Trajectories/U9_EVAL_RESULTS.md):

| Scene | E4 state | Action |
|---|---|---|
| empty / corridor / s_curve | ✅ 500 ep, 0% reject, stats clean | curate as-is |
| pillars | ❌ 274 ep, 45% reject, homotopy 14:1 (LLL:LRL) | **recollect, then curate** |

> Runs on the **cluster** (MuJoCo). Docker here is code-only.

**TODO checklist:**

- [ ] **0.1** Confirm blend radius + clearance gate:
  ```bash
  python uav_expert_data_collect/verify_blends.py     # pillars ≥0.43 m, s_curve ≥0.31 m → PASS
  ```
- [ ] **0.2** Recollect **pillars only**, fresh seed, clean out-dir (never mix with the old 274):
  ```bash
  python uav_expert_data_collect/collect.py \
      --scene pillars --n-trials 500 --seed 0 \
      --homotopy all --reject-limit 0.30 \
      --out-dir logs/uav_expert_data/pillars_v2
  ```
- [ ] **0.3** If rejection still > 30% or homotopy imbalance > ~2:1: raise the duration floor to
  `(12.0, 16.0)` in `generator.py` and rerun **0.2** once. (Target accel ≈ 4 m/s², see U9 §5.)
- [ ] **0.4** Sanity the recollect: `python uav_expert_data_collect/stats_validator.py` → pillars < 30%
  reject, homotopy ~2:1.
- [ ] **0.5** **NEW — write `uav_expert_data_collect/curate_dataset.py`** (small, no deps): walk the raw
  accepted episodes for the 4 scenes (empty/corridor/s_curve from the U9 run, pillars from `pillars_v2`),
  copy **only** `*.pkl` (skip `run_summary*`, skip `_stress`, skip the old pillars), write a `manifest.json`
  (per-scene counts, source paths, git rev, schema `H=8/D=12`), into `data/uav_fm/v1/<scene>/`.
  ```bash
  python uav_expert_data_collect/curate_dataset.py \
      --scenes empty corridor s_curve pillars \
      --pillars-src logs/uav_expert_data/pillars_v2 \
      --out data/uav_fm/v1
  ```
- [ ] **0.6** Verify the curated set: `manifest.json` shows 4 scenes, ~500 ep each, and **no**
  `run_summary`/stress/old-pillars leaked in.

**Gate:** `data/uav_fm/v1/` exists with 4 scenes < 30% reject, homotopy ~2:1, clean manifest. Until then,
train nothing beyond the empty-scene sanity model.

---

## Phase 1 — Mini-FM sanity gate (CHEAP, DO FIRST) — run the existing script

The gate already exists: `uav_expert_data_collect/mini_fm_sanity.py` (standalone, no pipeline). It validates
the schema/convention before any cluster spend.

- [ ] **1.1** Empty scene first, from the **curated** folder:
  ```bash
  python uav_expert_data_collect/mini_fm_sanity.py \
      --data-dir data/uav_fm/v1/empty --n-episodes 100 --n-steps 1000
  ```
- [ ] **1.2** Repeat on `data/uav_fm/v1/pillars` (the recollected + curated hard scene).

**Pass:** held-out RMS position error < 0.1 m; dataloader yields `(B, H=8, D=12)` cleanly; predicted Δp_des
norm within 2× of GT mean.
**Fail (RMS diverges):** action convention is wrong — re-check E4 Decision 1 (`actions[t] =
targets[t+1] − targets[t]`, position-delta) **before** building Phase 2.

**Gate:** mini-FM passes ⇒ data pipeline trustworthy ⇒ proceed to Phase 2.

---

# ══ E6 CODING JOB STARTS HERE ══

## Phase 2 — Build + train the UAV state-only FM (THE milestone)

Fork the FMv3-ODE-selectable stack into a UAV sibling. **Two new folders**, mirroring the source pair.
Ground rule from the top: **mirror the source 1:1, minimal new files** — every new file must earn its place.

| New folder | Forked from | Role |
|---|---|---|
| `flow_matcher_v3_uav/` | `flow_matcher_v3_ode_selectable/` | model + datasets + sampling + utils package |
| `FM_v3_uav_test/` | `FM_v3_ode_selectable_test/` | train / eval / load-results scripts |

> Naming follows the FMv3 lineage (`_ode_selectable`, `_drifting`, `_imeanflow` → `_uav`); package importable
> as `flow_matcher_v3_uav.utils`, so the scripts mirror the source 1:1.

### 2.0 The headline upgrade vs Legacy FMPCC — multi-env from day one

Legacy FMPCC (`avoiding-d3il`, all the FMv3 siblings) is **single-env**: one `exp` string → one dataset →
one model → flat `logs/<prefix>/` output. The UAV pipeline must instead be **multi-env native** over the 4
meaningful scenes (`empty, corridor, s_curve, pillars`). Concretely the new API must support, from **one**
unified entry:

- **train ALL envs** (one scene-agnostic model on the pooled 4-scene dataset — the IDEAS P2 milestone), and
- **train an INDIVIDUAL env** (scene-specialised model, for ablation / debugging), and
- a **new train/eval output structure** keyed by `<run_id>/<scene-or-all>/…` (not the legacy flat tree).

> Design note: state obs `[p_des|p|v]` does **not** encode which scene/obstacles → an `--scene all` model is
> goal-conditioned but **obstacle-blind** (it learns the expert's smooth-pathing *distribution* across
> scenes; obstacle awareness arrives later via the DPCC projector's `ep['obstacles']`, or via vision in
> Gen7). `--scene <one>` gives a scene-specialised prior. Both are first-class; `all` is the milestone.

### 2A. `flow_matcher_v3_uav/` — copy / modify / rebuild / drop

| Source file(s) | Action | Change for UAV |
|---|---|---|
| `models/diffusion.py`, `unet1d_temporal_cond.py`, `mlp.py`, `helpers.py` | **copy ~as-is** | dim-agnostic — driven by `transition_dim`/`horizon` from config. No structural change. |
| `sampling/policies.py` | **modify** | rollout hooks the **UAV MuJoCo env** (`uav_env_test`/`uav_naive_test`); consume `Δp_des` action → feed PID `p_des`. Scene-parametrised (load the right XML per eval scene). |
| `sampling/projection.py` (DPCC/SLSQP) | **copy, leave inactive** | projector preserved for the safety stage; **not wired to quadrotor constraints this Epoch**. |
| `datasets/sequence.py` | **modify in place** (keep filename — no new file) | add `UAVSequenceDataset(dataset_root, scene=…)` reading the **curated** `data/uav_fm/v1/`; `scene='all'` pools all 4, `scene='pillars'` loads one; lift `load_episodes()`+`episodes_to_chunks()` from `mini_fm_sanity.py`; emit `(H=8, D=12)` chunks `[Δp_des(3) ‖ obs(9)]`; `observation_dim=9`, `action_dim=3`; read `manifest.json` for provenance. |
| `datasets/normalization.py`, `buffer.py` | **copy** | keep `LimitsNormalizer`; **fit normaliser on the pooled selected scenes** so `all` and single-scene runs each get correct stats. |
| `datasets/d4rl.py`, `preprocessing.py` | **drop / stub** | D4RL/D3IL `sequence_dataset()` not used. |
| `datasets/minari-dataset-generation/` | **DROP** | irrelevant scaffolding. |
| `utils/` (`config.py`, `serialization.py`, `training.py`, `arrays.py`, `setup.py`, `logger.py`, `timer.py`, `progress.py`, `plot.py`) | **copy ~as-is** | training loop unchanged; `serialization.py` save-path must include `<scene-or-all>` (see 2E). |
| `__init__.py`, `setup.py` | **copy, rename** | package name `flow_matcher_v3_uav`. |

### 2B. `FM_v3_uav_test/` — copy / modify (the new multi-env CLI)

| Source file | Action | Change for UAV |
|---|---|---|
| `train_flow_matching_v3_ode_selectable.py` → `train_fm_uav.py` | **modify** | set `exp = 'uav'` (→ `config.uav`, block `flow_matching_v3_uav`), exactly like the source's `exp='avoiding-d3il'`; add `--scene {all,empty,corridor,s_curve,pillars}` (default `all`) threaded into the loader; output dir keyed by `<scene>` (2E). |
| `eval_flow_matching_v3_ode_selectable.py` → `eval_fm_uav.py` | **modify** | same `--scene` selector (`all` ⇒ eval each scene in turn + aggregate); closed-loop MuJoCo rollout: FM `Δp_des` chunk → PID → step; metrics = success, contact fraction, tracking error, **live inference timing** (BehaviorLogger). |
| `load_results_flow_matching_v3_ode_selectable.py` → `load_results_fm_uav.py` | **modify** | aggregate across the `<run_id>/<scene>/` output tree (per-scene + combined). |
| `Benchmark_ode_solver_Tests/` | **DROP** | not needed for UAV. |

### 2C. Config — ONE file, mirror `config/avoiding-d3il.py` exactly (no invented scheme)

The unified entry follows the **legacy pattern verbatim**: a single config module `config/uav.py` (mirrors
`config/avoiding-d3il.py`) holding a `base` + experiment **blocks** keyed by name. The train script sets
`exp = 'uav'` → `config = 'config.uav'` and calls `Parser().parse_args(experiment='flow_matching_v3_uav')`,
exactly as the source does with `'avoiding-d3il'` / `'flow_matching_v3_ode_selectable'`. **No `get_config()`
function, no `uav_fm.py`, no per-scene config files** — that was the wrong invention.

`config/uav.py` (one new file), block copied from the source's `flow_matching_v3_ode_selectable` block with
only the dims/loader/prefix changed:
```python
'flow_matching_v3_uav': {
    'model':     'models.Flow_matcher_U_Net_v2',     # unchanged
    'diffusion': 'models.diffusion.FlowMatchingODE', # unchanged
    'horizon': 8,
    'observation_dim': 9,            # [p_des | p | v]   (UAV — re-wired, not D3IL 9)
    'action_dim': 3,                 # Δp_des
    'loader': 'datasets.UAVSequenceDataset',          # the one new dataset class
    'dataset_root': 'data/uav_fm/v1',
    'normalizer': 'LimitsNormalizer',
    'prefix': 'flow_matching_v3_uav/',
    # … all other keys (time_beta_*, ode solver knobs, dim_mults, etc.) copied UNCHANGED from source
},
```
**Scene selection stays minimal:** a `--scene {all,empty,corridor,s_curve,pillars}` CLI arg on the train/
eval scripts (default `all`), threaded into `UAVSequenceDataset(scene=…)`. One config file, one CLI flag —
the "unified entry, similar naming to the old codes" requirement, with zero parallel machinery.

### 2E. New train/eval output structure (keyed by run + scene)

Replace the legacy flat `logs/<prefix>/` with a run/scene tree so `all` and per-scene runs never collide and
aggregation is trivial:
```
logs/fm_uav/<run_id>/
  ├── all/        weights/ losses.pkl  config.pkl  eval/  plots/  timing/        # the pooled model
  ├── empty/      …                                                              # or a single-scene run
  ├── corridor/   …
  ├── s_curve/    …
  ├── pillars/    …
  └── SUMMARY.json   # cross-scene success / contact / tracking / timing table
```
`eval_fm_uav.py --scene all` writes each scene's metrics under its own dir **and** rolls up `SUMMARY.json`.
`run_id` = timestamp+seed (mirror the source's run-id convention).

### 2F. SLURM entry — mirror `train_fmv3_ode_job.sh`, submit via `submit.sh`

All cluster runs go through the repo standard `./Slurm_Codes/submit.sh <script> <args>` (dated unified logs).
**New folder `Slurm_Codes/sbatch/uav_fm/`** (mirrors the `uav_env/`, `uav_expert_data/` convention), with
three scripts forked from `train_fmv3_ode_job.sh` / `eval_fmv3_ode_job.sh` / `fmv3_ode_pipeline.sh`:

| New sbatch | Forked from | Args | Change for UAV |
|---|---|---|---|
| `train_fm_uav.sh` | `train_fmv3_ode_job.sh` | `$1=scene` (def `all`) `$2=seed` | runs `python FM_v3_uav_test/train_fm_uav.py --scene $1 --seed $2`; **swap PYTHONPATH** — drop `GYM_AV`/D3IL, add the UAV MuJoCo env paths (`uav_env_test`/`uav_naive_test`); keep `MUJOCO_GL=egl` headless block. |
| `eval_fm_uav.sh` | `eval_fmv3_ode_job.sh` | `$1=scene` `$2=run_id` | runs `eval_fm_uav.py --scene $1 --run $2`; same env block. |
| `fm_uav_pipeline.sh` | `fmv3_ode_pipeline.sh` | `$1=scene` `$2=seed` | chains train → eval via `--dependency=afterok` (one-shot). |

Keep the source's conda-activate, GPU-info banner, `latest.log` symlink, and `on_exit` trap **verbatim** —
only the `PYTHONPATH` exports and the `python …` line change. Time limit `24:00:00` (carry over).

### 2D. Train + close the loop (the new API in action) — cluster commands

- [ ] **2.1** Build the two folders + `config/uav.py` + sbatch scripts; `python -m py_compile` everything
  (Docker = syntax only; **no torch/GPU/MuJoCo here — all real runs are cluster**).
- [ ] **2.2** Smoke-train one scene (short) to confirm `(B,8,12)` flows + loss decreases:
  ```bash
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh empty 5
  ```
- [ ] **2.3** Full train, **pooled 4-scene** model (the milestone):
  ```bash
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh all 5    # → logs/fm_uav/<run>/all/
  ```
  (Per-scene ablation models: loop `… train_fm_uav.sh empty 5`, `corridor 5`, `s_curve 5`, `pillars 5`.)
- [ ] **2.4** Closed-loop eval, all scenes + rollup (or one-shot pipeline):
  ```bash
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh all <run_id>
  # one submission, train→eval chained:
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh all 5
  ```
  FM-only rollout reaches goal, contact-free on empty/corridor, inference ≥ 33 Hz; compare vs PID expert.

**Gate:** `--scene all` model flies closed-loop to goal contact-free on empty/corridor at ≥ 33 Hz; per-scene
`SUMMARY.json` populated.

---

## Phase 2b — DPCC diffusion baseline (ablation) — OPTIONAL, only after FM works

Per the Epoch-6 decision: *after* the UAV FM runs successfully, optionally build the **original DPCC
(diffusion)** arm on the same UAV data for an apples-to-apples A/B (FM vs diffusion), Gen9-style
common-mode (same data, same infra). Fork from `diffuser/` analogously. **Do not start before Phase 2's
gate passes** — a baseline is only worth building once the primary policy is real.

---

## Phase 3 — Visual FM-PCC — DEFERRED to next Epoch (based on Gen7)

Not this Epoch. When it lands it is built on **Gen7 `fm_encdec_vision`** (visual FM): WS-A camera
collection (RGB↔BGR + FPV-orientation known fixes), dual-camera FiLM-ResNet encoder conditioning the FM,
retrain with visual conditioning. Tracked here only as the forward pointer.

---

## Evaluation — timing-first (applies from Epoch 6)

First Epoch with an actual *policy* to time, so the [REALTIME_RECORDING](../../REALTIME_RECORDING/IDEAS.md)
framework applies:
- **Primary — timing:** `total_ms = fm_ms (+ qp_ms later)` vs the 30 ms / 33 Hz budget. Measured **live**
  via `BehaviorLogger`, never from loaded logs.
- **Secondary — behaviour:** success rate, contact fraction, tracking error (FM vs PID expert; vs DPCC
  baseline if 2b done).
- **Zero-shot scene:** hold out a topology (wall-with-holes / denser pillars) for generalisation.

---

## Risks (Epoch-6 specific)

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| 1 | Train on un-fixed pillars → policy learns to crash at fillets | 🔴 | Phase 0 gate; never mix old 274 ep |
| 2 | Action-convention bug surfaces only after full training | 🔴 | Phase 1 mini-FM gate (~2 h) catches it |
| 3 | Dim mismatch: copy D3IL `(T,9)` instead of UAV `(T,12)` | 🔴 | §0 schema box; new `UAVSequenceDataset`, explicit dims in config |
| 4 | Eval env wiring (Δp_des → PID) wrong → flies nowhere | 🟠 | reuse `uav_env_test`/`uav_naive_test` rollout; validate one chunk before scaling |
| 5 | Forked package import drift (`flow_matcher_v3_uav.*`) | 🟡 | mirror source layout 1:1; py_compile all |
| 6 | Loader ingests debug clutter (`run_summary`, `_stress`, old pillars) | 🔴 | trainer reads **only** curated `data/uav_fm/v1/` (§0b); never the raw E4 tree |
| 7 | `all` vs per-scene normaliser stats mismatch | 🟠 | fit `LimitsNormalizer` on the *selected* scene set; persist stats in `config.pkl` per run |

---

## Deliverables for this Epoch

**E6 CODING deliverables (mine):**

| File / folder | Purpose |
|---|---|
| `flow_matcher_v3_uav/` | forked UAV FM model package (multi-scene `UAVSequenceDataset` in `sequence.py`) |
| `FM_v3_uav_test/` | UAV train / eval / load-results scripts (`exp='uav'`, unified `--scene` flag) |
| `config/uav.py` | **one** config file, mirrors `config/avoiding-d3il.py`; block `flow_matching_v3_uav` (12/9/3/8) |
| `Slurm_Codes/sbatch/uav_fm/` | **SLURM entry** — `train_fm_uav.sh` / `eval_fm_uav.sh` / `fm_uav_pipeline.sh` (via `submit.sh`) |
| `logs/fm_uav/<run_id>/<scene-or-all>/…` | output structure (per-scene + `SUMMARY.json` rollup) |
| `CHANGELOG.md` | per-phase implementation log (written as we build) |
| **`USAGE.md`** | **how-to-run next-step guide** (train all/one scene, eval, read outputs) — explicit deliverable |
| `CLOSURE.md` | final: timing table + success/contact metrics + FM (vs DPCC) verdict |

**PREP deliverables (data side, not the E6 coding focus):**

| File / folder | Purpose |
|---|---|
| `uav_expert_data_collect/curate_dataset.py` | tiny helper — raw E4 → curated `data/uav_fm/v1/` + `manifest.json` |
| `data/uav_fm/v1/` | curated, versioned, training-only 4-scene dataset (the only thing the trainer reads) |
| `EPOCH6_PLAN.md` | this document |

---

## Decisions locked (from IDEAS §3)

- **D1** template = state DPCC / `flow_matcher_v3_ode_selectable` — **chosen** (this fork).
- **D3** action space = `Δp_des` (matches E4 schema, PID consumes `p_des`).
- **D4** DPCC prediction model (when we get there) = differential-flatness analytic (no learning).
- **D2** FM vs DPCC A/B = common-mode, but **DPCC arm only after FM succeeds** (Phase 2b).

*No code written yet — this is the plan. No commit/push (per policy).*
