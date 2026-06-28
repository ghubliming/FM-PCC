# CHANGELOG — U4: opt-in grounding modes for the UAV (re-anchor + cond-on-real-`p`)

**Date**: 2026-06-28
**Plan**: [PLAN_U4_cond_real_p.md](./PLAN_U4_cond_real_p.md)
**Status**: implemented; `py_compile` + YAML + path-rendering checks pass. (Runtime is on the
Slurm cluster — Docker has no Python runtime.)

---

## TL;DR

Added **two opt-in ways to ground the FM command in the *measured* drone position**, so the command
can't run away from the lagging drone (the `corridor_C` crash, see
[../Real_Time_eval_loggging/data_example_anlysis/TRACKING_ERROR_Gen11E7.md](../Real_Time_eval_loggging/data_example_anlysis/TRACKING_ERROR_Gen11E7.md) §8).
**Default = exactly today.** The two modes are **mutually exclusive** (one knob each), selected by
`cond_mode`:

| Mode (`cond_mode`) | Knob | Retrain? | What it does |
|---|---|---|---|
| `'p_des'` (default) + `reanchor_alpha=0.0` | — | — | **exactly today** |
| `'p_des'` + **`reanchor_alpha`∈(0,1]** = **"re-anchor"** | `reanchor_alpha` | **NO** | bleeds the command toward measured `p` each step: `p_des = (1−α)(p_des+act) + α·p` |
| `'real_p'` = **"plan-in-real-position"** | `lead_gain` | **YES** | obs `[p｜v]`, action `Δp`, setpoint `= p + lead_gain·Δp` (rebuilt from measured `p` each step) |

(Names are by function — "re-anchor" / "plan-in-real-position" — not "phase 0/1".)

---

## Files changed

### 🟡 Code

| File | Change |
|---|---|
| `flow_matcher_v3_uav/datasets/d4rl.py` | `sequence_dataset(env, preprocess_fn, **cond_mode='p_des'**)`: in `'real_p'`, build `obs=[p｜v]` (cols 3:9) and `action=Δp=diff(p)` (cols 3:6). Default branch unchanged. |
| `flow_matcher_v3_uav/datasets/sequence.py` | `SequenceDataset.__init__(..., **cond_mode='p_des'**)`; passes it to `sequence_dataset(...)`. |
| `FM_v3_uav_test/train_fm_uav.py` | passes `cond_mode=getattr(args,'cond_mode','p_des')` into the dataset config. **Model auto-sizes** (`transition_dim = obs_dim+act_dim`): 12D for `p_des`, **9D** for `real_p`. No model-construction edit. |
| `FM_v3_uav_test/eval_fm_uav.py` | `load_pcc_config`: defaults `reanchor_alpha=0.0`, `lead_gain=1.0`. `rollout_one(..., cond_mode, reanchor_alpha, lead_gain)`: branches **obs assembly** (`[p｜v]` in real_p), **integration** (`p_des = p + lead_gain·act` in real_p; `(1−α)(p_des+act)+α·p` in p_des), and tags the eval out-dir. `_run_variant`: reads the knobs, tags `out_dir`, passes them through. `setup_dpcc_projector`: `deriv [3,4,5]` comment — same indices bind `p_des` (p_des-mode) or **real `p`** (real_p-mode, feasible because action=`Δp`). |

### 🟢 Config (key + folder path, both train & eval)

| File | Change |
|---|---|
| `config/uav.py` | added `'cond_mode': 'p_des'` to the `flow_matching_v3_uav` block **and** `('cond_mode','cond')` to `args_to_watch` → checkpoint path gains `…_condp_des` / `…_condreal_p`. **Read by BOTH train (dataset) and eval (rollout + which checkpoint to load).** |
| `config/uav_eval.yaml` | added eval-only knobs `reanchor_alpha: 0.0`, `lead_gain: 1.0`. |

> **Why `cond_mode` lives in `config/uav.py` (not duplicated in `uav_eval.yaml`):** it determines
> *both* the dataset *and* the checkpoint, so the eval must read the *same* value used to train (it's
> baked into the savepath). Duplicating it in the eval YAML would risk a mismatch (load a `p_des`
> checkpoint but assemble `real_p` obs → shape/garbage). The eval reads `cond_mode` from the training
> block; only the *per-mode eval knobs* (`reanchor_alpha`, `lead_gain`) live in `uav_eval.yaml`.

### Folder-path encoding (verified by rendering)
- **Checkpoint** (train + eval load): `…/H8_Dmodels.diffusion.FlowMatchingODE_condp_des/…` vs `…_condreal_p/…`.
- **Eval results**: `…/plans/…/<variant>` (default), `…/<variant>_reanchor0.5` (re-anchor sweep), `…/<variant>_lead1.5` (real_p lead sweep) — so sweeps never overwrite.

---

## How to run

### A) "Re-anchor" — NO retrain (test first, on the existing checkpoint)
Keep `cond_mode='p_des'` (existing checkpoint). In `config/uav_eval.yaml` set e.g. `reanchor_alpha: 1.0`
(try `0.0 / 0.5 / 1.0`). Run the normal eval. Results land in `…/<variant>_reanchor1.0`.
→ Tests "does grounding the command stop the `corridor_C` crash?" for **zero training cost**.
⚠ If the pre-U4 checkpoint folder lacks the `_condp_des` fragment, rename it once to add it (a `mv`,
no retrain) so the eval can find it.

### B) "Plan-in-real-position" — retrain (the principled version)
Set `cond_mode: 'real_p'` in `config/uav.py`, **retrain** (`train_fm_uav.py`) → a fresh 9D model under
`…_condreal_p/…`. Then eval (optionally tune `lead_gain` in `uav_eval.yaml` if it under-reaches goals).

### Mutual exclusivity (as designed)
`cond_mode='p_des'` → `reanchor_alpha` active, `lead_gain` ignored. `cond_mode='real_p'` → `lead_gain`
active, `reanchor_alpha` ignored (real_p is already α=1-grounded by construction). You turn on one.

---

## What we deliberately did NOT do (per instruction)
- ❌ **No real drone dynamics model in the projector** (#3). Binding real `p` in `real_p` is feasible
  purely because action=`Δp` makes `p=∫act` a tautology — *not* a plant model.
- ❌ **No termination on tracking error** (#4). Tracking error is **allowed and still logged**
  (`track_err`), so we can measure whether grounding alone reduces drift.

---

## Backward compatibility
- `cond_mode` absent / `'p_des'` **and** `reanchor_alpha=0.0` → **byte-identical to today.**
- `real_p` is a fresh 9D model in an isolated dir → cannot collide with `p_des` checkpoints.
- FM backbone, projector solver, real-time loop: unchanged.
- ⚠ Adding `cond_mode` to `args_to_watch` renames future `p_des` dirs to `…_condp_des`; pre-U4
  checkpoints need a one-time folder rename (no retrain) to be found.

---

## How to revert
1. `git checkout -- flow_matcher_v3_uav/datasets/d4rl.py flow_matcher_v3_uav/datasets/sequence.py FM_v3_uav_test/train_fm_uav.py FM_v3_uav_test/eval_fm_uav.py config/uav.py config/uav_eval.yaml`
2. (Or, to *disable* without reverting: leave `cond_mode='p_des'`, `reanchor_alpha=0.0`, `lead_gain=1.0` — the new paths are dormant. To also restore the *old* checkpoint paths, remove the `('cond_mode','cond')` line from `args_to_watch` in `config/uav.py`.)

---

## Verification status
- [x] `py_compile` passes on all 5 changed `.py` files.
- [x] `config/uav_eval.yaml` parses; knobs present.
- [x] Path rendering: `condp_des` / `condreal_p` checkpoint fragments + `_reanchor{α}` / `_lead{k}` eval tags confirmed.
- [ ] Runtime (Slurm): (B-first per request order is re-anchor test, then real_p retrain) — pending cluster run.

---

## Open items / caveats (from the plan)
- **Under-tracking:** `real_p` (lead 1.0) or high `reanchor_alpha` may be *safe-but-sluggish* (the
  setpoint doesn't lead the drone by the controller lag). Mitigate with `lead_gain>1` / `α<1`. We
  accept some tracking error by instruction.
- **`goal_dim` heuristic** on the 6D `[p｜v]` layout: the constant-column heuristic trims trailing
  (`v`) dims only; the `deriv` touches 0–5 (act, `p`), so it stays safe — same reasoning as the
  existing `p_des` path. Worth a sanity check on first real_p train.
- **Index reuse** (`deriv [3,4,5]` = `p_des` vs real `p`): handled by the dim-layout shift; commented in code.
