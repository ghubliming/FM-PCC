# PATCH TODO — Real-Time Behavior Recording (Cross-Gen Rollout)

**Date:** 2026-06-28
**Context:** [`IDEAS.md`](IDEAS.md) (framework spec) · reference implementation already shipped in
[`../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/PLAN.md`](../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/PLAN.md)
· pattern mirrors [`../MPC_traj_saved_in_npz/PATCH_TODO_MPC_Plans_in_NPZ.md`](../MPC_traj_saved_in_npz/PATCH_TODO_MPC_Plans_in_NPZ.md)

---

## Executive Summary

Gen11 UAV-FM is **done** — its eval loop records per-step `fm_ms` / `proj_ms` / `total_ms`,
per-step OBS/STATE/CONTACT/DPCC lines, and a SUMMARY timing block (see `eval_fm_uav.py` +
`eval_artifacts.py`). This TODO tracks rolling the **same** real-time recording framework out to
every other finished eval in FM-PCC.

> **The headline metric is timing, and timing can only be measured live** (IDEAS.md §"no
> shortcut"). Every generation must be **re-evaluated once** with the logger attached — old
> `.npz`/`.pkl`/JSON outputs can fill geometric context lines but never the `*_ms` numbers.

> ✅ **ALL IN-SCOPE PORTS COMPLETE (2026-06-28).** See
> [`CHANGELOG_RealTime_Recording_Rollout.md`](CHANGELOG_RealTime_Recording_Rollout.md).
> Grep token across all touched code: `REAL_TIME_RECORDING_UPDATE`.

| Status | Count | Generations |
|---|---|---|
| ✅ Done (pre-existing) | 1 | Gen11 UAV-FM |
| ✅ Done (this rollout) | 10 | D3IL baseline + FMv3-ODE + iMF (state ×2) + Drifting + Visual-Aligning (FM / Diffuser / iMF) + Visual-Avoiding (FM / Diffuser) |
| 🚫 Excluded — dead code | 7 | EncDec-Vision (DDPM / FM); Legacy Gen1–3 (×5) |

Confirmed by scan (2026-06-28): only `FM_v3_uav_test/*` contains any real-time logging
instrumentation today.

**Scope decisions (user, 2026-06-28):**
- **Visual-Aligning iMF (Gen3v4)** — source is *incomplete*, but it runs regardless → **still
  port** (flagged below).
- **EncDec-Vision DDPM / FM** — **dead code, not active → DO NOT TOUCH.** Removed from scope.
- **D3IL DPCC baseline** — **will implement.**
- **Legacy Gen1–3** — **dead code → DO NOT TOUCH.** Removed from scope.

---

## Reference Implementation (the "done" target to copy)

Gen11 UAV-FM, `FM_v3_uav_test/eval_fm_uav.py` + `eval_artifacts.py`:

- Per FM step, wall-clock split: `total_ms` (bundled) → `proj_ms` (projector) → `fm_ms = total − proj`
  (pure inference). See `rollout_one()` L345-352.
- Per-step structured log line via `blog.step(...)` (`eval_artifacts.BehaviorLogger`):
  `t`, `step_idx`, `obs`, `fm_horizon`, `fm_ms`, `proj_ms`, `proj_cost`, `proj_active`,
  `contact`, `track_err`. See L394-399.
- A `text_log` toggle gates the per-step writes to keep loop overhead near zero
  (config `behavior_log`).
- SUMMARY block aggregates `mean/max/p95` + `over_budget` against `budget_ms = 1000/control_hz`.

**Each port below = lift `BehaviorLogger` (or an equivalent) into that eval's loop, wrap the FM
call and the QP/projector call with `perf_counter`, emit the per-step line + SUMMARY, and gate it
with a `behavior_log` / `text_log` flag.**

---

## Architectural split — two eval shapes

The repo has **two** eval-loop shapes; the logger attaches differently to each.

1. **Standard "avoiding/aligning" loop** (most evals): post-FM **QP filter** — FM samples a fan,
   the DPCC projector runs *after* sampling, action extracted, env steps. Timing = `fm_ms`
   (sample) + `qp_ms` (projector) cleanly separable around two calls.

2. **UAV closed-loop** (Gen11, done): the PCC projector runs **inside the FM ODE loop**, not as a
   post-FM filter (noted in the Gen11 PLAN). Hence Gen11 measures `total_ms` then subtracts
   `proj_ms` rather than timing two separate calls. Evals that adopt in-loop projection must copy
   the Gen11 subtraction approach, not the two-call approach.

---

## JOB RT-A — Active Gen, Priority 1 (state-only FM/DPCC evals)

These use the standard loop with `sampled_trajectories` + `np.savez`. Add `BehaviorLogger`,
time the FM sample call and the projector call, emit per-step + SUMMARY.

| # | Generation | File | Loop shape | Status |
|---|---|---|---|---|
| 1 | **FMv3-ODE selectable** | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | standard QP-filter | ✅ DONE |
| 2 | **iMF (state, Gen3v4)** | `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | standard QP-filter | ✅ DONE |
| 3 | **iMF ODE-sel variant** | `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | standard QP-filter | ✅ DONE |
| 4 | **Drifting** | `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | standard QP-filter | ✅ DONE |
| 5 | **Visual-Aligning FM (Gen7)** | `fm_visual_aligning_test/eval_fm_visual_aligning.py` | inline `VisualAgent` | ✅ DONE |
| 6 | **Visual-Aligning Diffuser** | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | inline `VisualAgent` | ✅ DONE |
| 7 | **Visual-Aligning iMF (Gen3v4)** | `imf_visual_aligning_test/eval_imf_visual_aligning.py` | inline `VisualAgent` | ✅ DONE ⚠ incomplete src (ported) |

> Visual-aligning evals (#5–7) carry their own inline `VisualAgent` class (separate from
> `sampling/policies.py`). The timing wrap goes around the agent's sample + projection calls there.
> These are the largest files (~2000 lines) — highest porting effort.
>
> ⚠ **#7 (iMF Gen3v4) source is incomplete** but runs regardless (user, 2026-06-28) → port it
> anyway. Expect missing/partial fields; populate what the run produces and leave the rest blank
> rather than blocking on the incomplete code.

---

## JOB RT-B — Active Gen, Priority 2 (visual avoiding)

| # | Generation | File | Loop shape | Status |
|---|---|---|---|---|
| 8 | **Visual-Avoiding Diffuser** | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | standard QP-filter | ✅ DONE |
| 9 | **Visual-Avoiding FM** | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | standard QP-filter | ✅ DONE |

> These visual evals have an extra encoder stage. Consider a third timing field `enc_ms` (encoder
> latency) in addition to `fm_ms` + `qp_ms` so the per-step budget breakdown is complete — the
> encoder can dominate the loop on real hardware.

---

## JOB RT-C — DPCC baseline (the reference line)

| # | Generation | File | Notes | Status |
|---|---|---|---|---|
| 10 | **D3IL Visual-Aligning baseline** | `d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py` | "original DPCC" — `fm_ms=0`, total = `qp_ms` only | ✅ DONE |

> IDEAS.md §Priority says do the **DPCC baseline first** — simplest (no FM fields), validates the
> logger grammar, and is the reference every FM-PCC `fm_overhead` is measured against. Despite the
> table order, port this one early.

---

## 🚫 Excluded — DO NOT TOUCH (dead code)

Confirmed not active by the user (2026-06-28). Do **not** port the logger to these.

| Generation | File | Reason |
|---|---|---|
| EncDec-Vision DDPM | `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | dead code, not active |
| EncDec-Vision FM | `fm_encdec_vision_test/eval_fm_encdec_vision.py` | dead code, not active |
| Legacy FM (Gen1) | `FM_test/eval_FM.py` | dead code |
| Legacy FM_v2 | `FM_v2_test/eval_FM_v2.py` | dead code |
| Legacy FM_Unet_v2 | `FM_Unet_v2_test/eval_FM_Unet_v2.py` | dead code |
| Legacy FM_hp_tune | `FM_hp_tune_test/eval_FM_hp_tune.py` | dead code |
| Legacy FM_v3 | `FM_v3_test/eval_FM_v3.py` | dead code |

> Also excluded (as before): `*_test (legacy_based_on_visual_aligning)/` duplicate dirs and
> anything under `Archived_Codes/`.

---

## Per-port checklist (how each file WAS ported — see CHANGELOG)

Implementation differs slightly from the original plan: instead of timing two separate calls,
all in-scope evals run **in-loop projection** (projector bundled inside the `policy()`/`agent.predict()`
call), so the recorder taps the **existing** wall-time measurement and records `total_ms` as the
headline (the diffuser/no-projector variant gives pure FM time; `proj_ms` split is bundled). The
shared `realtime_recording.behavior_logger.RTRecorder` replaces a per-dir `BehaviorLogger`.

- [x] Import the shared `RTRecorder` (DRY — one module, not 10 copies).
- [x] Tap the existing FM/agent wall-time → `total_ms` (no new compute inserted).
- [x] In-loop projection bundled into `total_ms` (Gen11 subtraction needs `policies.py` to expose
      `projection_ms`, which non-UAV policies do not — recorded as bundled, honestly labelled).
- [x] Emit per-step line: `t`, `step_idx`, `obs`, `pos`, `action`, `*_ms`, `proj_active`, `track_err`.
- [x] Emit SUMMARY: `mean/max/p95` for each `*_ms`, `over_budget` vs `budget_ms = 1000/RT_CONTROL_HZ`.
- [x] Gate writes via `text_log` (tied to each eval's `write_to_file` / `save_path`).
- [x] Record the **measurement node** in the SUMMARY header (latency is hardware-bound).
- [ ] Re-run each eval once on representative hardware to populate the timing block (USER — runs
      happen on the Slurm cluster; `RT_CONTROL_HZ=30` is an assumed budget, tune per target).

---

## Recommended Port Order

1. **RT-C** D3IL DPCC baseline (#10) — simplest, sets the reference grammar + the comparison line.
2. **RT-A #1–4** state-only FM/DPCC (FMv3-ODE, iMF×2, Drifting) — standard loop, shared structure.
3. **RT-A #5–7** visual-aligning (FM / Diffuser / iMF) — large inline `VisualAgent`, more effort
   (#7 iMF source incomplete — port anyway).
4. **RT-B #8–9** visual-avoiding — add `enc_ms` field.

---

## Status Ledger

| Generation | Real-time recording | Verdict |
|---|---|---|
| **Gen11 UAV-FM** (`FM_v3_uav_test`) | ✅ shipped (`fm_ms`/`proj_ms`/`total_ms` + SUMMARY) | **DONE** (pre-existing) |
| D3IL DPCC baseline | ✅ ported (no projector → proj=0) | **DONE** |
| FMv3-ODE selectable | ✅ ported (bundled total_ms) | **DONE** |
| iMF state (Gen3v4) ×2 | ✅ ported (bundled total_ms) | **DONE** |
| Drifting | ✅ ported (bundled total_ms) | **DONE** |
| Visual-Aligning FM (Gen7) | ✅ ported (per-replan total_ms) | **DONE** |
| Visual-Aligning Diffuser | ✅ ported (per-replan total_ms) | **DONE** |
| Visual-Aligning iMF (Gen3v4) | ✅ ported (per-replan total_ms) ⚠ incomplete src | **DONE** |
| Visual-Avoiding FM / Diffuser | ✅ ported (bundled enc+FM+proj) | **DONE** |
| EncDec-Vision DDPM / FM | 🚫 | **excluded — dead code** |
| Legacy Gen1–3 (×5) | 🚫 | **excluded — dead code** |

**Remaining ❌** are intentional: the single unchecked checklist item is the USER's cluster
re-run to populate live timing numbers (cannot be done in the Docker dev env — no GPU/runtime).
