# CHANGELOG — Real-Time Recording Cross-Gen Rollout

**Date:** 2026-06-28
**TODO:** [`PATCH_TODO_RealTime_Recording.md`](PATCH_TODO_RealTime_Recording.md) · **Spec:** [`IDEAS.md`](IDEAS.md)
**Grep token (all touched code):** `REAL_TIME_RECORDING_UPDATE`

---

## What this does

Rolls the Gen11 UAV-FM real-time recording framework out to **all 10 in-scope evals**. Each eval
rollout now emits a per-episode `realtime_*.log` (per-step timing + OBS/ACT/PCC lines + a SUMMARY
block answering *"can this close the control loop in budget, and how much is FM vs projection?"*).

All edits are tagged `REAL_TIME_RECORDING_UPDATE` for one-command retrieval:
```bash
grep -rn "REAL_TIME_RECORDING_UPDATE" .
```

---

## New shared module (single source of truth — not 10 copies)

| File | Markers | What |
|---|---|---|
| `realtime_recording/__init__.py` | 1 | package init |
| `realtime_recording/behavior_logger.py` | 1 | **`RTRecorder`** — portable, obs-layout-agnostic generalisation of the Gen11 `FM_v3_uav_test/behavior_logger.BehaviorLogger`. Same grammar + SUMMARY block. Handles **bundled timing** (FM+projection measured together) honestly. Smoke-tested (numpy-only, no torch/mujoco). |

`RTRecorder` API: `step(t, total_ms, obs=, action=, pos=, fm_ms=, proj_ms=, proj_active=, track_err=, step_idx=)`
then `save(path, behaviour={...})`; `summary_dict()` exposes raw stats. `total_ms` is mandatory
(headline); `fm_ms`/`proj_ms` optional (bundled when unset).

---

## Files touched (10 evals)

Each standard eval got **6 markers** (import block, `RT_CONTROL_HZ` const, recorder-create,
timing-tap, per-step record, save). D3IL got **11** (two `agent.predict` branches wrapped).

### DPCC baseline (the reference line) — JOB RT-C
| File | Markers | Integration |
|---|---|---|
| `d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py` | 11 | Wrapped BOTH `agent.predict` branches (vision + state-only) with `_time.time()`. No projector → `proj_active=False`, `proj_ms=0`. Saves `realtime_baseline_ctx<c>_traj<t>.log` per rollout. |

### State-only FM/DPCC (standard loop) — JOB RT-A
| File | Markers | Integration |
|---|---|---|
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | 6 | Tapped existing `start=time.time()` around `policy()`; `total_ms` bundled (proj inside policy). Saves `realtime_<variant>_trial<i>.log`. |
| `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | 6 | same pattern |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | 6 | same pattern |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | 6 | same pattern |

### Visual-aligning (inline `VisualAgentWrapper`, ~2000 lines) — JOB RT-A
| File | Markers | Integration |
|---|---|---|
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | 6 | Recorder lives in the wrapper: created in `reset()`, per-**replan** record at the `curr_rollout_time += ...` point in `predict()`, saved in `update_rollout_info()`. `realtime_<variant>_rollout<ridx>.log`. |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 6 | same (system=`VisualAligning_Diffuser`) |
| `imf_visual_aligning_test/eval_imf_visual_aligning.py` | 6 | same (system=`VisualAligning_iMF`). ⚠ source is incomplete — ported per user request; runs regardless. |

### Visual-avoiding (inline agent, vision encoder) — JOB RT-B
| File | Markers | Integration |
|---|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | 6 | Tapped `start=time.time()` around `agent.predict()`; `total_ms` bundles **encoder + FM + projection**. |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | 6 | same (system=`VisualAvoiding_Diffuser`) |

---

## Design decisions (honest notes)

1. **Bundled timing.** Non-UAV `policies.py` do not expose `projection_ms`, so the standard/visual
   evals run in-loop projection and the recorder logs `total_ms` (the deployability headline). The
   FM-vs-projection split is recovered at the aggregate level by comparing the `diffuser`
   (no-projector) variant's `total_ms` against the `dpcc-*` variants' — exactly IDEAS.md's "vs DPCC
   baseline" comparison. Per-step split would require modifying each model's sampling to record
   `projection_ms` (deferred — invasive).

2. **`RT_CONTROL_HZ = 30`** is a module constant in each eval, an *assumed* deployment loop rate
   (budget = 1000/hz ms). The raw `total_ms` mean/max/p95 are the real product; `budget_ms` /
   `real_time_safe` are advisory and explicitly labelled. Tune per target hardware.

3. **Zero added loop latency.** The recorder only formats numbers the loop already measured; it
   never inserts compute. Writes are gated by each eval's `write_to_file`/`save_path`.

4. **Shared module, not 10 copies.** Repo dirs are otherwise self-contained, but duplicating a
   200-line logger 10× is a maintenance hazard; `realtime_recording/` is importable because every
   eval runs from repo root (they import their package directly).

---

## Verification

- `python3 -m py_compile` passes on the shared module + all 10 edited evals (Python 3.13).
- `RTRecorder` smoke-tested standalone (numpy-only): per-step lines + SUMMARY render correctly,
  bundled-timing path labelled, budget/over-budget math correct.
- **Not run end-to-end** — the Docker dev env has no GPU/torch/mujoco/d3il runtime. Live timing
  numbers require one re-eval per gen on the Slurm cluster (the one remaining USER task).

---

## Excluded (untouched, per user)

EncDec-Vision (`ddpm_encdec_vision_test`, `fm_encdec_vision_test`) and Legacy Gen1–3
(`FM_test`, `FM_v2_test`, `FM_Unet_v2_test`, `FM_hp_tune_test`, `FM_v3_test`) — dead code.
Also untouched: `*_test (legacy_based_on_visual_aligning)/` and `Archived_Codes/`.
