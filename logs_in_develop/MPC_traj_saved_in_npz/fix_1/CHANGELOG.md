# fix_1 — Save the FULL MPC foresight at EVERY control step in `.npz`

**Date:** 2026-07-15
**Scope:** cross-generation eval scripts
**Related:** [`../PATCH_TODO_MPC_Plans_in_NPZ.md`](../PATCH_TODO_MPC_Plans_in_NPZ.md) (JOB A — *persistence*), [`../CHANGELOG.md`](../CHANGELOG.md)

---

## fix_1.2 — Decouple PLOT cadence from npz cadence (2026-07-15)

**Problem introduced by fix_1:** `save_samples_every = 1` fed **both** the npz *and* the diagnostic
PNG fan-overlay from the same buffer, so the foresight fan was drawn at every control step and the
`{variant}.png` plots (e.g. `.../results/halfspace_both-hard/diffuser.png`) became an unreadable
tangle.

**Fix:** keep npz at full resolution (every step) but restore the **old plot density**. Added a
separate `plot_samples_every = max(1, args.horizon // 2)` and strided only the **plotting** loops
by it. The npz collection gate (`if _ % save_samples_every == 0`, still `1`) is unchanged, so
`sampled_trajectories_all` still holds every step.

Since fix_1 makes the buffer index equal the control-step index, striding the plot by
`args.horizon // 2` reproduces exactly the pre-fix_1 plot cadence.

Applied to the same 12 files:
- **10 files** (`for __ in range(len(...))`): → `for __ in range(0, len(...), plot_samples_every)`
- **2 visual-avoiding files** (`for traj_np in ...`): → `enumerate(...)` + `if _pi % plot_samples_every != 0: continue`
  (`diffuser_visual_avoiding_test`, `fm_visual_avoiding_test`)

Net: **npz = every step (fix_1); PNG fan = every H/2 steps (fix_1.2, = original look).** This
supersedes the "plots get denser" side-effect noted below.

---

## Problem

`PATCH_TODO`/JOB A only fixed **whether** `sampled_trajectories_all` reached `np.savez` at
all. It never touched the **time resolution** of that data. Every state-based avoiding eval
captured the MPC foresight fan only on a subsampled cadence:

```python
save_samples_every = args.horizon // 2
...
if _ % save_samples_every == 0:
    sampled_trajectories.append(samples.observations[:, :, :])
```

So with e.g. `horizon = 32`, the H-step plan (foresight) was persisted only **once every 16
control steps** — roughly 2 snapshots per rollout. Every other step's plan was discarded.
For debugging MPC behaviour (plan drift, re-planning jitter, projection effect per step) this
threw away the majority of the useful data.

### Root cause
The `save_samples_every = args.horizon // 2` gate is inherited **verbatim from DPCC upstream**
(`/workspaces/aux_repo/dpcc/scripts/eval.py:167,255`). There it was only a **plotting aid** —
drawing every step's full H-horizon fan would make the trajectory PNG an unreadable tangle, so
DPCC subsampled it. When JOB A added `sampled_trajectories_all` to `np.savez`, it simply
persisted that existing *plotting* buffer and inherited the subsampling. It was never redesigned
to be complete debug data.

---

## Fix

Set `save_samples_every = 1` so the **full candidate fan** `samples.observations[:, :, :]`
(shape `[batch_size, horizon, obs_dim]`) is appended at **every** control step, and therefore
persisted at full time-resolution in `{variant}.npz` under `sampled_trajectories_all`.

The gate line (`if _ % save_samples_every == 0:`) is left intact — with the divisor now `1` it is
always true, so plotting and npz both receive every step.

### Files changed (12) — exact lines

| # | File | Def line changed |
|---|------|------------------|
| 1 | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | L250 |
| 2 | `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | L287 |
| 3 | `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | L284 |
| 4 | `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | L281 |
| 5 | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | L402 |
| 6 | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | L416 |
| 7 | `FM_v3_test/eval_FM_v3.py` | L129 |
| 8 | `FM_test/eval_FM.py` | L129 |
| 9 | `FM_v2_test/eval_FM_v2.py` | L129 |
| 10 | `FM_Unet_v2_test/eval_FM_Unet_v2.py` | L129 |
| 11 | `FM_hp_tune_test/eval_FM_hp_tune.py` | L129 |
| 12 | `scripts/eval.py` (Gen0 baseline) | L221 |

Each change:
```python
# BEFORE
save_samples_every = args.horizon // 2
# AFTER
save_samples_every = 1  # fix_1: save full-resolution MPC foresight every step (was: args.horizon // 2)
```

All 12 already carry `sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object)`
in their `np.savez` (from JOB A), so the extra data lands in the npz with **no further change** to
the savez call.

---

## Per-module status — problem vs. no problem

### A. HAD THE PROBLEM — fixed by this patch (time-subsampled fan)
State-based evals that stored the full candidate fan but only every `horizon//2` steps. Now
every step. These are the 12 files above.

- `FM_v3_ode_selectable_test`, `FM_v3_imeanflow_test` (both eval scripts), `FM_v3_drifting_test`
- `diffuser_visual_avoiding_test`, `fm_visual_avoiding_test`
- `FM_test`, `FM_v2_test`, `FM_Unet_v2_test`, `FM_hp_tune_test`, `FM_v3_test`
- `scripts/eval.py` (Gen0 baseline)

### B. NO PROBLEM — already per-step, full candidate fan (unchanged)
Visual **aligning** evals use an inline agent that appends the full fan
(`curr_rollout_all_candidates.append(...)`) on **every replan**, with **no `save_samples_every`
gate**, and persist it as `sampled_trajectories_all` (built from `all_candidates`). Time-complete
and candidate-complete already.

- `fm_visual_aligning_test/eval_fm_visual_aligning.py` — saves `cand_all` (`all_candidates`), append L1974/L1976
- `imf_visual_aligning_test/eval_imf_visual_aligning.py` — saves `plans_all`, append L1559/L1561 — *(slated for rewrite; left as-is)*
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — saves `cand_all`, append L1982/L1984

### C. REVERTED — encoder-decoder vision evals archived as dead code
These were briefly patched with a full-fan buffer (`history_all_candidates`), but the whole
encoder-decoder vision line was then declared **dead code** and moved to
`Archived_Codes/encdec_vision(Abandoned_based_on_d3il_ddpm)/` (based on the d3il ddpm baseline).
The fix_1 edits were **reverted** to the original state (they store the selected candidate only) —
no point maintaining dead code.

- `fm_encdec_vision` / `fm_encdec_vision_test` — reverted, archived
- `ddpm_encdec_vision` / `ddpm_encdec_vision_test` — reverted, archived

### D. NO PROBLEM / N/A — UAV
`FM_v3_uav_test/eval_fm_uav.py` runs `batch_size=1` (single candidate) and appends the plan on
**every** step (`plans.append(...)`, L971); persisted via `eval_artifacts.py`
(`sampled_trajectories_all=plans_all`, L120) / `results.json`. No time gate, single candidate ⇒
already complete. Consistent with `PATCH_TODO` (JOB A marked UAV N/A).

### E. NOT TOUCHED — archived / legacy = dead code
`Archived_Codes/*` and any `*(legacy_based_on_*)` / `(Abandoned)` / `(Outdated)` folders are dead
code — never run, never edited. Not part of this fix and not tracked as work.

---

## Side effects & notes

- **npz size / memory:** each rollout now stores `~episode_length` foresight snapshots instead of
  ~2, i.e. roughly `horizon//2 ×` larger `sampled_trajectories_all`. Each snapshot is
  `[batch_size, horizon, obs_dim]`. Expected and intended for full-fidelity debug runs.
- **Plots get denser:** ~~the blue foresight-fan overlay in `{variant}.png` now draws a plan at
  every step~~ → **resolved in fix_1.2** (plot strided to every `H/2` steps; npz stays full).
- **No savez / schema change:** only the sampling cadence changed; downstream consumers
  (`npz_analysis/`, visualizer) read the same `sampled_trajectories_all` key, now denser in time.
- **Not run here** (AI-coding container, no Python env) — **run on cluster** to validate npz
  contents and size. Load with `np.load(..., allow_pickle=True)`; index as
  `sampled_trajectories_all[trial][step_idx] → [batch, horizon, obs_dim]`.

---

## Follow-ups (out of scope for fix_1)
- **`imf_visual_aligning`:** slated for a rewrite, so left untouched (already correct anyway).
- **Optional CLI knob:** expose `--save_plans_every` so runs can trade fidelity for size without a
  code edit. Deferred.
