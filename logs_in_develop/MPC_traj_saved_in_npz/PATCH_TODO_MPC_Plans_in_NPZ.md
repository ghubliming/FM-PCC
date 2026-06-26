# PATCH TODO — Save MPC Open-Loop Plans in `.npz` (Massive Cross-Gen Update)

**Date:** 2026-06-21  
**Context:** [`CHANGELOG.md`](../Gen3v4_imf/U6/Fix-MPC-traj-in-pkl/CHANGELOG.md) · [`MPC_Candidate_Selection_Explained.md`](../npz_analysis_tool/MPC_Candidate_Selection_Explained.md) · [`CAPABILITY_GAP_plan_not_saved.md`](../npz_analysis_tool/CAPABILITY_GAP_plan_not_saved.md)

---

## Executive Summary

Three related issues must be patched across the codebase:

| Job | What | Files | Severity |
|-----|-------|-------|----------|
| **A** | `sampled_trajectories_all` missing from `np.savez` | 10 eval scripts | **High** — plans lost |
| **B** | `desired_next_pos` hardcoded to `observations[0]` | 10 eval scripts | **Medium** — tracking metric wrong for `dpcc-c` |
| **C** | `prev_observations` always copies `observations[0]` | 10 `policies.py` | **Medium** — temporal consistency reference wrong for `dpcc-c` |
| **D** | NPZ analyzer can't consume plan data | 1 tool | **Low** — downstream |
| **E** | Early-gen savez missing `obs_all`/`act_all` entirely | 4 eval scripts | **Low** — legacy |

---

## JOB A — Add `sampled_trajectories_all` to `np.savez`

> The variable is already computed and used for plotting but **never persisted** to the `.npz`.
> This is the primary patch from the CHANGELOG.

### Already Fixed ✅

| File | Line | Status |
|------|------|--------|
| `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | L393 | ✅ Fixed (has `sampled_trajectories_all=...`) |
| `imf_visual_aligning_test/eval_imf_visual_aligning.py` | L2128 | ✅ Fixed (saves `plans_all` as `sampled_trajectories_all`) |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | L2032 | ✅ Fixed |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | L2032 | ✅ Fixed |
| `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | L821 | ✅ Fixed |
| `fm_encdec_vision_test/eval_fm_encdec_vision.py` | L790 | ✅ Fixed |

### NEEDS PATCH ❌ (Active Gen — PRIORITY 1)

These files compute `sampled_trajectories_all` and plot it but do NOT include it in `np.savez`:

#### 1. `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py`
- **savez at:** L376–386
- **`sampled_trajectories_all` computed at:** L248, appended at L337
- **Fix:** Add `sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object)` to the `np.savez` call at L386
```python
# CURRENT (L384-386):
                                 obs_all=np.array(obs_all, dtype=object),
                                 act_all=np.array(act_all, dtype=object))
# PATCHED:
                                 obs_all=np.array(obs_all, dtype=object),
                                 act_all=np.array(act_all, dtype=object),
                                 sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object))
```

#### 2. `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`
- **savez at:** L376–386
- **`sampled_trajectories_all` computed at:** L248, appended at L337
- **Fix:** Same one-line addition after `act_all` at L386

#### 3. `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py`
- **savez at:** L379–389
- **`sampled_trajectories_all` computed at:** L251, appended at L340
- **Fix:** Same one-line addition after `act_all` at L389

#### 4. `FM_v3_uav_test/eval_fm_uav.py`
- **Note:** This file is a completely different architecture (UAV closed-loop). It does NOT use the standard avoiding `np.savez` with `obs_all`/`act_all`. Results are saved as JSON via `results.json`.
- **Status:** **SKIP** — different eval paradigm, no plan fan to persist in the same way.

#### 5. `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py`
- **savez at:** L513–521
- **`sampled_trajectories_all` computed at:** L370, appended at L455
- **Fix:** Add after `act_all` at L521:
```python
                                 act_all=np.array(act_all, dtype=object),
                                 sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object))
```

#### 6. `fm_visual_avoiding_test/eval_fm_visual_avoiding.py`
- **savez at:** L526–534
- **`sampled_trajectories_all` computed at:** L384, appended at L469
- **Fix:** Add after `act_all` at L534:
```python
                                 act_all=np.array(act_all, dtype=object),
                                 sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object))
```

### NEEDS PATCH ❌ (Early Gen — PRIORITY 2 — also missing `obs_all`/`act_all`)

These early-gen files save **only scalar metrics** — no trajectories at all:

#### 7. `FM_test/eval_FM.py`
- **savez at:** L249 — one-liner, no `obs_all`, no `act_all`, no `sampled_trajectories_all`
- **`obs_all` not collected** — need to also add `obs_all=[]` / `act_all=[]` + append in loop
- **Fix:** Must add collection of `obs_all`, `act_all`, AND `sampled_trajectories_all`, then save all three

#### 8. `FM_v2_test/eval_FM_v2.py`
- **savez at:** L249 — same issue as FM_test
- **Fix:** Same as #7

#### 9. `FM_Unet_v2_test/eval_FM_Unet_v2.py`
- **savez at:** L249 — same issue
- **Fix:** Same as #7

#### 10. `FM_hp_tune_test/eval_FM_hp_tune.py`
- **savez at:** L249 — same issue
- **Fix:** Same as #7

#### 11. `FM_v3_test/eval_FM_v3.py`
- **savez at:** L249 — same issue
- **Fix:** Same as #7

---

## JOB B — Fix `desired_next_pos` Tracking Reference

> `desired_next_pos = samples.observations[0, 1, ...]` always references trajectory index **0**, but the executed action may come from a different `which_trajectory` (e.g. for `dpcc-c`).
>
> For `random` selection: `which=0` → consistent ✔  
> For `temporal_consistency`: `observations` is reordered so index 0 IS the selected → consistent ✔  
> For `minimum_projection_cost`: `observations` is NOT reordered → **index 0 ≠ selected** ✘

### Files Affected (all use `samples.observations[0, 1, ...]`)

| # | File | Line | Fix |
|---|------|------|-----|
| 1 | `FM_test/eval_FM.py` | L197 | Use `samples.observations[which_trajectory, 1, ...]` — but `which_trajectory` not returned by policy |
| 2 | `FM_hp_tune_test/eval_FM_hp_tune.py` | L197 | Same |
| 3 | `FM_Unet_v2_test/eval_FM_Unet_v2.py` | L197 | Same |
| 4 | `FM_v2_test/eval_FM_v2.py` | L197 | Same |
| 5 | `FM_v3_test/eval_FM_v3.py` | L197 | Same |
| 6 | `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | L321 | Same |
| 7 | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | L321 | Same |
| 8 | `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | L327 | Same |
| 9 | `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | L324 | Same |
| 10 | `FM_v3_uav_test/eval_fm_uav.py` | L321 | Same (if standard eval loop used) |

> **Implementation note:** To fix this properly, `Policy.__call__` must also **return** `which_trajectory` to the eval script. Currently it only returns `(action, samples)`. Either:
> - Option 1: Return `which_trajectory` as third value (breaking change)
> - Option 2: Attach `which_trajectory` to the `samples` namedtuple/object
> - Option 3: Since `temporal_consistency` reorders observations so [0] is correct, and `random` uses [0], the bug only affects `dpcc-c` — could be deprioritized
>
> **The visual-avoiding evals** (`diffuser_visual_avoiding_test`, `fm_visual_avoiding_test`) use `desired_next_pos = next_pos_des[:2].copy()` instead — **not affected** (they track the actual executed position, not the plan).

---

## JOB C — Fix `prev_observations` in `policies.py`

> In ALL `policies.py` files, line ~70/76:
> ```python
> self.prev_observations = np.repeat(np.expand_dims(observations[0], axis=0), batch_size, axis=0)
> ```
> This always stores trajectory **index 0** regardless of `which_trajectory`.
>
> For `temporal_consistency`: `observations` has been reordered so [0] IS the selected → correct ✔  
> For `minimum_projection_cost`: `observations` NOT reordered → [0] ≠ selected → **wrong reference for next step's temporal match** ✘
>
> The correct fix:
> ```python
> self.prev_observations = np.repeat(np.expand_dims(observations[which_trajectory], axis=0), batch_size, axis=0)
> ```

### Files Affected (ALL `policies.py` in `*/sampling/`)

| # | File | Line | Current | Fix |
|---|------|------|---------|-----|
| 1 | `diffuser/sampling/policies.py` | L76 | `observations[0]` | `observations[which_trajectory]` |
| 2 | `flow_matcher/sampling/policies.py` | L76 | Same | Same |
| 3 | `flow_matcher_unet_v2/sampling/policies.py` | L76 | Same | Same |
| 4 | `flow_matcher_v2/sampling/policies.py` | L76 | Same | Same |
| 5 | `flow_matcher_v3/sampling/policies.py` | L76 | Same | Same |
| 6 | `flow_matcher_v3_drifting/sampling/policies.py` | L70 | Same | Same |
| 7 | `flow_matcher_v3_imeanflow/sampling/policies.py` | L70 | Same | Same |
| 8 | `flow_matcher_v3_ode_selectable/sampling/policies.py` | L70 | Same | Same |
| 9 | `flow_matcher_v3_uav/sampling/policies.py` | L70 | Same | Same |
| 10 | `fm_encdec_vision/sampling/policies.py` | L76 | Same | Same |
| 11 | `ddpm_encdec_vision/sampling/policies.py` | L76 | Same | Same |

> **Note:** The `fm_visual_aligning` and `imf_visual_aligning` eval scripts have their OWN inline `VisualAgent` class with a separate `prev_observations` implementation (L1550: `self.prev_observations = traj_np[which].copy()`). These are already correct — they use `which` not `[0]`.

---

## JOB D — NPZ Analyzer Plan Consumption

> `npz_analysis/analyze_npz.py` already recognizes `sampled_trajectories_all` as a valid key (L51) and has a `--replot` flag for executed trajectories (L316). But it does NOT yet:
> 1. Overlay open-loop plans (blue lines) on replot output
> 2. Compute plan-quality columns (`plan_straightness`, `plan_roughness`, `plan_max_jerk`)

### Needed Extensions

| Feature | Description | Priority |
|---------|-------------|----------|
| `--replot-plans` flag | Overlay `sampled_trajectories_all` blue plan fan on executed path plots | Medium |
| `plan_*` CSV columns | Straightness / roughness / max-jerk computed on the **plans** (not executed path) | Medium |
| Obstacle geometry embedding | Embed obstacle centers/radii from `projection_eval.yaml` into npz for self-contained replot | Low |

---

## JOB E — Early-Gen Missing `obs_all` / `act_all` Entirely

These early-gen eval scripts save **only scalar metrics** in their `np.savez`. They have no `obs_all` or `act_all` keys at all, making post-hoc trajectory analysis impossible:

| # | File | savez Line | Missing |
|---|------|-----------|---------|
| 1 | `FM_test/eval_FM.py` | L249 | `obs_all`, `act_all`, `sampled_trajectories_all` |
| 2 | `FM_v2_test/eval_FM_v2.py` | L249 | Same |
| 3 | `FM_Unet_v2_test/eval_FM_Unet_v2.py` | L249 | Same |
| 4 | `FM_hp_tune_test/eval_FM_hp_tune.py` | L249 | Same |
| 5 | `FM_v3_test/eval_FM_v3.py` | L249 | Same |

These all follow the same pattern — single-line savez with only metric arrays. Each needs:
1. Add `obs_all = []` and `act_all = []` initialization before the trial loop
2. Append `obs_buffer` / `action_buffer` after each trial
3. Extend the `np.savez` call with all three keys

> **Risk assessment:** These are legacy gens (Gen1–Gen3). If they're not being actively re-run, this is low priority. But for consistency across the codebase, it's worth doing.

---

## Summary — Total File Count

| Category | Count | Files |
|----------|-------|-------|
| Eval scripts needing `sampled_trajectories_all` in savez | **5** | Drifting, ODE-sel, IMF-ODE-sel, Visual-avoid-DPCC, Visual-avoid-FM |
| Eval scripts needing `obs_all`/`act_all` + `sampled_trajectories_all` | **5** | FM, FM_v2, FM_Unet_v2, FM_hp_tune, FM_v3 |
| Eval scripts with tracking ref bug (`observations[0]` → `which_trajectory`) | **9** | All state-only avoiding evals (UAV exempt — JSON output) |
| `policies.py` with `prev_observations` bug | **11** | All `*/sampling/policies.py` (UAV: latent, harmless) |
| NPZ analyzer extension | **1** | `npz_analysis/analyze_npz.py` |
| **Total unique files** | **~26** | |

### Recommended Patch Order

1. **JOB A Priority 1** (6 files) — Quick one-liner additions, high value
2. **JOB C** (11 files) — Simple `[0]` → `[which_trajectory]` substitution
3. **JOB A Priority 2** (5 files) — Requires adding collection logic, more invasive
4. **JOB B** (10 files) — Requires changing `Policy.__call__` return signature or attaching metadata
5. **JOB D** (1 file) — Downstream, only useful after JOB A data is available

---

## Gen11 Epoch6 UAV — Per-Job Audit (checked 2026-06-21)

**Files checked:** `FM_v3_uav_test/eval_fm_uav.py` · `flow_matcher_v3_uav/sampling/policies.py`

### Background

Gen11 E6 **rewrote the UAV eval from scratch** (CHANGELOG: *"rewritten from scratch — the source eval
is 700 lines welded to D3IL/minari — wrong base, undebuggable"*). It mirrors
`uav_expert_data_collect/generator.run_trial` and swaps the expert trajectory for the FM policy.
The output paradigm is completely different from every other eval in the repo.

### JOB A — `sampled_trajectories_all` missing from `np.savez` → **NOT APPLICABLE** ✅

`eval_fm_uav.py` has **no `np.savez` call**. Outputs are saved as JSON:
```python
with open(os.path.join(out_dir, 'results.json'), 'w') as f:
    json.dump({'summary': summary, 'rollouts': rollouts}, f, indent=2)
```
There is also no plan fan: the eval always calls `policy({0: obs}, batch_size=1, horizon=horizon)` —
one candidate per step, no `sampled_trajectories` list, no visualization loop. Nothing to persist.

### JOB B — `desired_next_pos` tracking reference → **NOT APPLICABLE** ✅

The UAV eval tracks position error via MuJoCo physics directly:
```python
track_err.append(float(np.linalg.norm(data.qpos[:3] - p_des)))
```
The `samples.observations[0, 1, ...]` pattern from the avoiding evals does not exist here. No bug.

### JOB C — `prev_observations` in `policies.py` → **LATENT BUT HARMLESS** ⚠️

`flow_matcher_v3_uav/sampling/policies.py:70` has the same bug as all other `policies.py`:
```python
self.prev_observations = np.repeat(np.expand_dims(observations[0], axis=0), batch_size, axis=0)
```
However, `eval_fm_uav.py` always uses `batch_size=1`, so `which_trajectory` is always `0` and
`observations[0] == observations[which_trajectory]`. The bug is **inert in practice**.

It would only bite if the UAV eval ever uses `dpcc-c` selection with `batch_size > 1` — the DPCC
safety projector is explicitly not activated this Epoch (per `EPOCH6_PLAN.md`). Safe to patch for
consistency, but zero functional impact today.

### JOB D — NPZ analyzer → **NOT APPLICABLE** ✅

UAV eval writes JSON, not NPZ. The analyzer is irrelevant for Gen11.

### JOB E — Missing `obs_all`/`act_all` in early-gen savez → **NOT APPLICABLE** ✅

`eval_fm_uav.py` is a purpose-built new script with its own output schema — not a legacy eval with
a truncated `np.savez`. The rollout dict already captures all meaningful fields per episode.

### Gen11 U6 Summary

| Job | Status | Reason |
|-----|--------|--------|
| A — save plans in npz | ✅ N/A | JSON output; no plan fan; `batch_size=1` |
| B — `desired_next_pos` tracking ref | ✅ N/A | Tracks MuJoCo `qpos[:3]` directly |
| C — `prev_observations[0]` bug | ⚠️ Latent, harmless | `batch_size=1` always; `dpcc-c` not activated |
| D — analyzer extension | ✅ N/A | JSON, not NPZ |
| E — missing `obs_all`/`act_all` | ✅ N/A | New script, purpose-built JSON schema |
