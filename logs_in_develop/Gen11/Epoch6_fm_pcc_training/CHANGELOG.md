# Gen11 Epoch 6 — FM-PCC Training: CHANGELOG

**Date:** 2026-06-21
**Scope:** the E6 *coding* job — fork FMv3-ODE-selectable → UAV state-only Flow-Matching, per
[EPOCH6_PLAN.md](EPOCH6_PLAN.md) Phase 2. Data prep (Phase 0/1) is not coded here. No model trained
(Docker = no torch/GPU/MuJoCo; all real runs are cluster). Everything below is `py_compile` / `bash -n`
clean. No commit/push.

---

## What was built (the 2-folder fork + config + SLURM + prep helper)

### 1. `flow_matcher_v3_uav/` — model package (copied from `flow_matcher_v3_ode_selectable/`)
- `rsync` copy, **excluding** `datasets/minari-dataset-generation/` and `__pycache__`.
- **Only one file changed:** `datasets/d4rl.py` — rewritten as a UAV-only `sequence_dataset(env,
  preprocess_fn)`:
  - removed the D4RL/Minari/D3IL deps (the source's `avoiding-d3il` branch imported
    `d3il.agents.utils.sim_path` — the UAV fork must not depend on D3IL);
  - added the `uav-<scene>` branch (`uav-all` pools the 4 scenes; `uav-empty` etc. load one), reading the
    **curated** `data/uav_fm/v1/<scene>/*.pkl` (override `UAV_FM_DATA_ROOT`);
  - yields the **identical** episode contract as the source avoiding branch (`observations`/`actions`
    length `T-1`, dummy `rewards`/`terminals`), so the generic `SequenceDataset` is reused **unchanged**.
- **Net new/changed code = 1 file.** `models/`, `sampling/` (incl. the DPCC `projection.py`, copied but
  inactive), `utils/`, `sequence.py`, `normalization.py`, `buffer.py` are byte-identical to the source.
- `import_class` resolves `datasets.*`/`models.*` to `flow_matcher_v3_uav.*` automatically
  (`__name__.split('.')[0]`), so the fork is fully self-contained — no leakage to the old package.

> **Simpler than the plan:** the plan proposed a new `UAVSequenceDataset` class; it proved unnecessary —
> the source already selects the data source by `env` string (the `avoiding-d3il` branch), so a `uav-*`
> branch + the generic `SequenceDataset` is the minimal, faithful path. One fewer new abstraction.

### 2. `FM_v3_uav_test/` — scripts (copied from `FM_v3_ode_selectable_test/`)
- `train_fm_uav.py` (from `train_flow_matching_v3_ode_selectable.py`): `exp='uav'` → `config.uav`, block
  `flow_matching_v3_uav`; `import flow_matcher_v3_uav.utils`; new `--scene {all,empty,corridor,s_curve,
  pillars}` (default `all`) sets `dataset='uav-<scene>'` → selects the data branch AND segregates the
  output path (`logs/uav-<scene>/<exp_name>/<seed>/`). Everything else (seed handling, resume, wandb)
  unchanged.
- `eval_fm_uav.py`: **rewritten from scratch** (the source eval is 700 lines welded to D3IL/minari — wrong
  base, undebuggable). The new eval mirrors the known-good expert loop
  (`uav_expert_data_collect/generator.run_trial`) and swaps the expert trajectory for the FM policy:
  - **receding-horizon (MPC):** each FM step → `obs=[p_des|p|v]` → policy → first `Δp_des` → `p_des +=
    Δp_des` → PID tracks it;
  - **multi-rate:** FM queried at `DATASET_HZ=33`, physics+PID at `model.opt.timestep`, `p_des`
    zero-order-held for `decim=round(1/(dt·33))` physics steps (matches how the data was recorded);
  - metrics: success, contact fraction, FM-target tracking error, **live inference timing** (`fm_ms`
    mean/p95), plus secondary `goal_dist`; writes `<savepath>/eval/results.json` + cross-scene
    `logs/uav-all/SUMMARY.json`.
- `load_results_*` **removed** (avoiding/halfspace-coupled; the new eval self-aggregates) — minimal-files.

### 3. `config/uav.py` — ONE config file, mirrors `config/avoiding-d3il.py`
- Single block `flow_matching_v3_uav`, copied from `flow_matching_v3_ode_selectable` with only:
  `prefix='flow_matching_v3_uav/'`, `max_path_length=600` (UAV episodes ~330-530 steps),
  `include_returns=False` (UAV has no reward), `dataset_root` doc key. Dims (`12/9/3`) are **data-derived**
  at runtime (`transition_dim = observation_dim + action_dim`), not hard-set. **No `get_config()`,
  no per-scene files** — scene is a CLI flag, not config.

### 4. `Slurm_Codes/sbatch/uav_fm/` — SLURM entry (mirror `train_fmv3_ode_job.sh`)
- `train_fm_uav.sh` (`$1=scene $2=seed`), `eval_fm_uav.sh` (`$1=scene $2=seed $3=n_trials`),
  `fm_uav_pipeline.sh` (train→eval via `--dependency=afterok`). Boilerplate (conda, banner, `latest.log`,
  trap) copied verbatim; **only** the `PYTHONPATH` changed — dropped `GYM_AV`/D3IL (the UAV loader has no
  d3il dep), kept `MUJOCO_GL=egl` for the eval rollout. Submit via `./Slurm_Codes/submit.sh`.

### 5. `uav_expert_data_collect/curate_dataset.py` — Phase-0 prep helper
- Copies only accepted episode pkls (skips `run_summary`/`manifest`/`_stress`/old pillars) from the raw
  E4 tree into `data/uav_fm/v1/<scene>/` + a provenance `manifest.json`. (Prep, not the coding deliverable
  — included because the trainer depends on the curated tree existing.)

---

## Verification
- `python -m py_compile` clean on: `config/uav.py`, `curate_dataset.py`, `train_fm_uav.py`,
  `eval_fm_uav.py`, and **all** of `flow_matcher_v3_uav/`.
- `bash -n` clean on all three sbatch scripts.
- No torch/GPU/MuJoCo locally → no training/eval run; correctness of the closed-loop eval is **cluster-
  pending**.

## Open design decision surfaced to the user (NOT silently invented)
**UAV eval "success".** The state-only FM is conditioned on `obs[0]` only, so for the random-goal `empty`
scene it is **not goal-conditioned** — "reach the goal" is ill-defined there. The eval therefore defaults
`success` to the expert's own acceptance gate (**contact-free + airborne**, well-defined for every scene)
and reports `goal_dist` only as a secondary signal. Confirm this is the metric you want, or we make the FM
goal-conditioned (next Epoch / Gen7) before treating goal-reaching as the headline.

## Not done (by design)
- Phase 0/1 (recollect, curate run, mini-FM gate) — data-side prep, run on the cluster.
- DPCC safety projection — `projection.py` copied but inactive (next Epoch).
- Visual FM (Gen7) — next Epoch.
- No commit/push (per policy).
