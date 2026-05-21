# FIX-7 Postmortem & Complete Changelog

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Audit source**: `fix_7/BUG_REPORT_7_Audit.md` (independent auditor research)  
**Prerequisite context**: `KEY_fix_6/BUG_REPORT.md` — the frozen-problem investigation that preceded this fix.  
**Status**: Implemented. Awaiting next Slurm run for validation.

---

## 1. Background — Why KEY_fix_6 Was Insufficient

`KEY_fix_6` applied three fixes (AUDIT-FIX-1/2/3) that broke the cross-variant "frozen" symptom. But JOBs 20634 and the "Failed Again" run (logs: `temp/For_Gen6V4/KEY_OUTPUTS_GEN7_2`) showed a persistent residual failure: **every job that generates expert reference videos for the first time** still produces bp_image std=0.1978 instead of the clean 0.2093, causing a wrong trajectory for all variants in that job.

| Condition | bp_image std | a0 z-component | Final dist ctx0 | Verdict |
|---|---|---|---|---|
| Expert gen skipped (files exist) | **0.2093** | ~-0.24 | 0.218847 m | ✅ GOOD |
| Expert gen ran in same process | **0.1978** | ~-0.70 | 0.312711 m | ❌ BAD |

The bad result is **deterministic**, not random — same bad values appear in every job where expert gen ran.

---

## 2. True Root Cause: `MjRobot.GLOBAL_MJ_ROBOT_COUNTER`

Source: `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py`

```python
class MjRobot(RobotBase, MjIncludeTemplate):
    GLOBAL_MJ_ROBOT_COUNTER = 0        # L24 — process-global class variable

    def __init__(self, scene, ...):
        self._mj_robot_id = MjRobot.GLOBAL_MJ_ROBOT_COUNTER   # L58
        MjRobot.GLOBAL_MJ_ROBOT_COUNTER += 1                  # L59 — monotonically increasing
```

This counter is **never decremented** — not by `env.close()`, not by `gc.collect()`, not by `torch.cuda.empty_cache()`. It persists for the entire Python process lifetime.

### What it controls

The counter is embedded into every robot body name in the compiled MuJoCo XML:

```python
attrib_split.insert(1, "rb{}".format(self._mj_robot_id))   # L282
# counter=0 → body names: "rb0_link_0", "tcp_rb0_tcp", "rgbd_rb0_rgbd", ...
# counter=1 → body names: "rb1_link_0", "tcp_rb1_tcp", "rgbd_rb1_rgbd", ...
```

And into the temporary XML file written per robot instance:

```python
f"./models/mj/robot/panda_tmp_rb{self._mj_robot_id}_{uuid.uuid1()}.xml"   # L331
```

### Contamination chain

```
Job start (counter=0)
  │
  ├─ generate_expert_reference() runs
  │     Robot_Push_Env().__init__() → MjRobot(robot_id=0) → counter = 1
  │     env.close() ← counter stays at 1
  │     gc.collect(), cuda.empty_cache() ← counter STILL 1
  │
  ├─ Variant 0: Robot_Push_Env().__init__() → MjRobot(robot_id=1)
  │     XML body prefix: "rb1_*" instead of "rb0_*"
  │     Inhand camera referenced as "rgbd_rb1_rgbd" — different compiled scene
  │     MuJoCo renders subtly different geometry → bp_image std = 0.1978
  │     FM model receives wrong visual obs → bad trajectory [0.0229, 0.0464, -0.7044]
  │     Final distance = 0.312711 m (BAD)
  │
  ├─ Variant 1: Robot_Push_Env().__init__() → MjRobot(robot_id=2) → same mismatch
  ...
```

When expert gen is **skipped** (files already exist):

```
No Robot_Push_Env() before variant loop
  → counter stays at 0
    → Variant 0 gets robot_id=0 ("rb0") → clean scene → bp_image std = 0.2093
      → good trajectory [-0.0054, 0.0405, -0.2432] → 0.218847 m (GOOD)
```

### Smoking gun from logs

| Job | Expert gen ran? | Temp file created | Robot ID | bp_image std | Verdict |
|---|---|---|---|---|---|
| JOB 20634 | YES | `panda_tmp_rb0_*` (expert), `panda_tmp_rb1_*` (variant) | rb1 | 0.1978 | ❌ BAD |
| JOB 20636 | NO (skipped) | `panda_tmp_rb0_*` (variant only) | rb0 | 0.2093 | ✅ GOOD |

The `rb0` vs `rb1` prefix is the direct proof. No code changed between these jobs.

### Why AUDIT-FIX-1 alone was insufficient

Moving `generate_expert_reference()` before the variant loop (AUDIT-FIX-1) correctly broke inter-variant contamination but did not prevent the expert gen's env creation from incrementing the counter. The first variant's env still got `robot_id=1` instead of `robot_id=0`.

### Seed determinism

Both outcomes are individually deterministic. The apparent "randomness" was purely from hidden file-system state: first invocation at a savepath → expert gen runs → counter contaminated → bad. Subsequent invocations → expert gen skips → clean. Same seed, same model, two repeatable but different outcomes depending on whether expert video files exist.

---

## 3. The Fix (FIX-7)

### Part A — Pre-loop counter reset (before variant loop)

After `generate_expert_reference()` returns and before `for variant in projection_variants:`, reset the counter and clean up stale XML files:

```python
gc.collect()
torch.cuda.empty_cache()

# FIX-7: Reset MuJoCo global robot body counter after expert gen.
try:
    from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot as _MjRobot
    _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
    print('[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)')
except Exception as _e:
    print(f'[ expert ] WARNING: MjRobot counter reset failed: {_e}')
import glob as _glob
_mj_dir = os.path.join(
    os.environ.get('D3IL_DIR', 'd3il/environments/d3il'), 'models/mj/robot')
for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
    try:
        os.remove(_stale)
    except OSError:
        pass
```

**Import path note**: Must use `from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot` — the same path `aligning.py` uses. Python may create separate module objects for different import paths, giving separate class variables. Using the wrong path would reset a shadow counter with no effect.

### Part B — Per-variant counter reset (in `finally` block)

After each variant's `finally:` block (restoring stdout/stderr), also reset the counter so variant N+1 gets `robot_id=0` regardless of what variant N's env creation did:

```python
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_f.close()
    # FIX-7 (per-variant): Reset MuJoCo robot body counter so next
    # variant's Robot_Push_Env gets robot_id=0 (rb0 body prefix),
    # matching the clean-process scene geometry.
    try:
        _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
    except NameError:
        pass
    for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
        try:
            os.remove(_stale)
        except OSError:
            pass
```

### Stale XML cleanup rationale

Each `Robot_Push_Env()` creation writes `panda_tmp_rb{N}_{uuid}.xml` to `d3il/models/mj/robot/`. These are never deleted during normal execution and accumulate across variants. They produce `mju_openResource: could not open resource` warnings in Slurm logs. Deleting them after each cleanup point suppresses these warnings.

---

## 4. Expected Outcome After FIX-7

In the next Slurm job where expert videos are generated for the first time:

- Log line: `[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)`
- All variants: `[ DIAG img ] bp_image std = 0.2093` (not 0.1978)
- All variants produce distinct, good trajectories

**Regression diagnostic**: If bp_image std is still 0.1978 after this fix, contamination goes deeper than the body counter (possibly OpenGL EGL context state). In that case, the subprocess approach (separate OS process for expert gen) would be needed — see `KEY_fix_6/BUG_REPORT.md §7 FIX-4`.

---

## 5. Complete Changelog — All Code Changes (KEY_fix_6 + fix_7)

This section is the authoritative record of every code change in the fix series. Changes applied to both `fm_visual_aligning_test/eval_fm_visual_aligning.py` (**Gen7**) and `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (**Gen6V4**) unless noted.

---

### AUDIT-FIX-1 — Expert gen moved before variant loop (`KEY_fix_6`)

**Files**: `eval_fm_visual_aligning.py`, `eval_visual_aligning_dpcc.py`

**Before**: `generate_expert_reference(save_path, ...)` was called inside `for variant in projection_variants:` at the top of the loop — running once per variant.

**After**: Moved to before the loop. Expert gen runs exactly once, into `_base_results`:

```python
_base_results = (f'{args.savepath}/results_train_set'
                 if args_cli.eval_on_train else f'{args.savepath}/results')
os.makedirs(_base_results, exist_ok=True)
generate_expert_reference(_base_results, n_rollouts=3)
gc.collect()
torch.cuda.empty_cache()

for variant in projection_variants:
    ...
```

**Why**: Expert gen creates a `Robot_Push_Env` each time it runs. Inside the loop, it contaminated every subsequent variant's env via the `GLOBAL_MJ_ROBOT_COUNTER`. Moving it before the loop was a necessary first step (though insufficient alone — see FIX-7).

---

### AUDIT-FIX-2 — Re-enable `constraint_types` (`KEY_fix_6`)

**File**: `config/visual_aligning_eval.yaml`

**Before**:
```yaml
constraint_types: []   # OPTION A: No constraints (kept disabled per user decision — Fix 8)
```

**After**:
```yaml
constraint_types: ['bounds', 'dynamics']   # Re-enabled — AUDIT-FIX-2 (KEY_fix_6/BUG_REPORT.md)
```

**Why**: With empty `constraint_types`, the DPCC projector for `post_processing` and `model_free` is a no-op — the SLSQP solver has zero constraints, returns the input unchanged, all projection costs are zero. This made pp ≡ mf ≡ raw FM, causing the "frozen" appearance for those two variants even in clean jobs. Re-enabling constraints makes the evaluation meaningful.

---

### AUDIT-FIX-3 — Per-variant save_path (`KEY_fix_6`)

**Files**: `eval_fm_visual_aligning.py`, `eval_visual_aligning_dpcc.py`

**Before**:
```python
save_path = f'{args.savepath}/results'
```

**After**:
```python
save_path = f'{args.savepath}/results/{variant}'
```

**Why**: Shared `save_path` caused `results_seed_*.pkl`, `*.npz`, `eval_*.log`, and diagnostic files to be overwritten by each successive variant. The last variant's files would silently clobber all prior results. Per-variant subdirectory prevents this.

---

### Snapshot YAML path fix (`KEY_fix_6`)

**Files**: `fm_visual_aligning/utils/setup.py`, `diffuser_visual_aligning/utils/setup.py`

**Before**:
```python
yaml_path = 'config/projection_eval.yaml'
dest = os.path.join(snapshot_dir, 'projection_eval.yaml')
```

**After**:
```python
yaml_path = 'config/visual_aligning_eval.yaml'
dest = os.path.join(snapshot_dir, 'visual_aligning_eval.yaml')
```

**Why**: The snapshot utility was referencing `projection_eval.yaml`, a filename that doesn't exist in this project. The actual eval config is `config/visual_aligning_eval.yaml`. Without this fix, the snapshot step silently failed to copy the config, so experiment logs had no record of the YAML used.

---

### FIX-7 Part A — Pre-loop counter reset (`fix_7`)

**Files**: `eval_fm_visual_aligning.py`, `eval_visual_aligning_dpcc.py`

Inserted immediately after `gc.collect()` / `cuda.empty_cache()` and before the variant loop. Full code in §3 Part A above.

**What changed**: Added `import gc` at top of file; added the counter reset + stale XML cleanup block before `for variant in projection_variants:`.

---

### FIX-7 Part B — Per-variant counter reset in `finally` block (`fix_7`)

**Files**: `eval_fm_visual_aligning.py`, `eval_visual_aligning_dpcc.py`

**Location**: `finally:` block at end of each variant's `try/except/finally` block  
(`eval_fm_visual_aligning.py:L1071`, `eval_visual_aligning_dpcc.py:L1059`)

**Before**:
```python
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_f.close()
```

**After**:
```python
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_f.close()
    # FIX-7 (per-variant): Reset MuJoCo robot body counter so next
    # variant's Robot_Push_Env gets robot_id=0 (rb0 body prefix),
    # matching the clean-process scene geometry.
    try:
        _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
    except NameError:
        pass
    for _stale in _glob.glob(os.path.join(_mj_dir, 'panda_tmp_rb*.xml')):
        try:
            os.remove(_stale)
        except OSError:
            pass
```

**Why**: Even after the pre-loop reset, each variant's env creation increments the counter (0→1). Without a per-variant reset, variant N+1 always gets `robot_id=1` (rb1 body prefix), contaminating its compiled MuJoCo scene. The `except NameError` guard handles the edge case where the pre-loop import failed.

---

## 6. Files Changed Summary

| File | Changes |
|---|---|
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | AUDIT-FIX-1, AUDIT-FIX-3, FIX-7 Part A, FIX-7 Part B; `import gc` added |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | Same four changes (parallel Gen6V4 file) |
| `config/visual_aligning_eval.yaml` | AUDIT-FIX-2: constraint_types re-enabled |
| `fm_visual_aligning/utils/setup.py` | Snapshot YAML filename corrected |
| `diffuser_visual_aligning/utils/setup.py` | Snapshot YAML filename corrected |
| `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py` | **NOT modified** — root cause lives here (L24, L58-59, L282, L331); fix applied externally via counter reset |

---

## 7. Verification Checklist (Next Slurm Job)

- [ ] Log contains: `[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)`
- [ ] All variants show: `[ DIAG img ] bp_image std = 0.2093`
- [ ] Variant 0 (`diffuser`) a0 z-component ≈ -0.24 (not -0.70)
- [ ] `diffuser` ≠ `post_processing` ≠ `model_free` (distinct results now that constraints are active)
- [ ] Per-variant result dirs created: `results/diffuser/`, `results/post_processing/`, etc.
- [ ] Snapshot contains `visual_aligning_eval.yaml` (not `projection_eval.yaml`)
- [ ] No `panda_tmp_rb1_*.xml` or `panda_tmp_rb2_*.xml` files accumulate in `d3il/models/mj/robot/`
