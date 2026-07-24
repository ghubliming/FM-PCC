# CHANGELOG — DPCC (Gen0) eval npz: add `sampled_trajectories_all` (MPC plan foresight) for parity with FMv3ODE

**Date:** 2026-07-13 · **Gen:** Gen0 (DPCC baseline, `diffuser/` model ↔ `scripts/` eval)
**Scope:** Issue-2 of the UAV-diffuser debugging session — npz structure parity between DPCC and FMv3ODE.

## Why

The trajectory visualizer (`npz_analysis/npz_traj_visualizer/npz_traj_export.py`) builds its H-step receding-horizon plan fans from the npz key **`sampled_trajectories_all`**. FMv3ODE's eval writes it (tagged `MPC_NPZ_PATCH` in `FM_v3_test/eval_FM_v3.py`); **DPCC's `scripts/eval.py` did not** — so DPCC npz carried only the executed path (`obs_all`) + actions (`act_all`), no plan foresight. That made DPCC unusable as a coupling cross-check (plan-vs-executed) in the visualizer, and left the two pipelines' npz schemas divergent.

Key finding: DPCC **already collected** `sampled_trajectories_all` identically to FMv3ODE — it was simply omitted from the `np.savez(...)` call.

## What changed (1 line)

`scripts/eval.py` — the `np.savez(f'{save_path}/{variant}.npz', ...)` call: added
```python
sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object)   # MPC_NPZ_PATCH
```
No other change. The variable is already:
- initialized: `sampled_trajectories_all = []` (per-variant)
- populated: `sampled_trajectories.append(samples.observations[:, :, :])` every `save_samples_every = args.horizon // 2` steps, then `sampled_trajectories_all.append(sampled_trajectories)` per trial
which is **byte-identical** to FMv3ODE's collection cadence — so the resulting array has the same shape semantics.

## Result — npz schemas now match

| npz key | DPCC before | DPCC after | FMv3ODE |
|---|---|---|---|
| n_success, n_success_and_constraints, n_steps, n_violations, total_violations, avg_time, collision_free_completed, args, obs_all, act_all | ✅ | ✅ | ✅ |
| `sampled_trajectories_all` (H-step plan foresight) | ❌ | ✅ | ✅ |

DPCC `scripts/eval.py` and FMv3ODE `FM_v3_test/eval_FM_v3.py` now emit the **same npz key set**. (UAV's `eval_artifacts.py` already writes `sampled_trajectories_all=plans_all`, so all three share the foresight schema.)

## Verification

- `python3 -m py_compile scripts/eval.py` → OK.
- Variable scope confirmed: init L224 → append L332 → save L392 (all inside the same per-variant block).
- Semantics: `sampled_trajectories_all` = list over trials of lists (every `horizon//2` steps) of `samples.observations[:, :, :]` (batch × horizon × obs_dim) — identical to FMv3ODE.
- **Runtime validation must run on cluster** (no torch/env here): re-run any DPCC `scripts/eval.py` variant and confirm the `.npz` now loads with a `sampled_trajectories_all` array.

## Not changed / notes

- Only `scripts/eval.py` (the Gen0 DPCC entrypoint) touched. FMv3ODE and UAV already had the key.
- No change to collection cadence, plotting, metrics, or any other npz key — this is purely additive.
- The visualizer can now ingest DPCC npz for plan-vs-executed coupling checks (relevant to Issue-1, the UAV foresight decoupling — pending separate investigation).
