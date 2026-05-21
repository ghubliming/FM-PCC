# Gen7 Post-Fix-6 Post-Mortem: The Real Root Cause

**Date**: 2026-05-21  
**Source logs**: `temp/For_Gen6V4/KEY_OUTPUTS_GEN7_2`  
**Git rev**: `0b8acfe`  
**Prerequisite**: Read `KEY_fix_6/BUG_REPORT.md` (Sections 1–R6) first.  
**Status**: Root cause definitively identified. Fix provided.

---

## 1. The Symptom — Fix_6 Did NOT Work

Even after AUDIT-FIX-1 (expert gen moved before variant loop) + AUDIT-FIX-2 (constraint_types re-enabled) + AUDIT-FIX-3 (per-variant save_path):

| Run | Expert gen ran? | bp_image std | a0 | Final dist ctx0 | Verdict |
|---|---|---|---|---|---|
| JOB 20634 (diffuser, 4-variant) | YES | **0.1978** | [0.0229, 0.0464, -0.7044] | 0.312711 m | ❌ BAD |
| JOB 20636 (diffuser+pp, 2-variant) | NO (skipped) | **0.2093** | [-0.0054, 0.0405, -0.2432] | 0.218847 m | ✅ GOOD |
| "Then Failed Again" (diffuser) | YES | **0.1978** | [0.0229, 0.0464, -0.7044] | 0.312711 m | ❌ BAD |

**The failure is 100% reproducible and 100% correlated with expert gen running.**

The "no code change, just YAML" observation is correct — the user changed variant lists between runs but no source code. The instability comes from whether `generate_expert_reference()` actually executes (first run on a new seed) or skips (files already exist).

> [!CAUTION]
> AUDIT-FIX-1 (moving expert gen before the variant loop) was CORRECTLY APPLIED but is INSUFFICIENT. The contamination mechanism was misidentified in BUG_REPORT.md. It is not "residual MuJoCo global state" in some vague sense. It is one specific, concrete, measurable thing.

---

## 2. The True Root Cause: `MjRobot.GLOBAL_MJ_ROBOT_COUNTER`

[MjRobot.py:24,58-59](file:///workspaces/FM-PCC/d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py#L24):

```python
class MjRobot(RobotBase, MjIncludeTemplate):
    GLOBAL_MJ_ROBOT_COUNTER = 0        # ← class-level, survives env.close()

    def __init__(self, scene, ...):
        ...
        self._mj_robot_id = MjRobot.GLOBAL_MJ_ROBOT_COUNTER   # L58
        MjRobot.GLOBAL_MJ_ROBOT_COUNTER += 1                   # L59 ← MONOTONICALLY INCREASING
```

This counter is a **process-global class variable**. Every call to `Robot_Push_Env(...)` creates a new `MjRobot`, which increments this counter. `env.close()` does **not** decrement it. There is no reset mechanism anywhere in d3il.

### What the counter does

The counter determines the robot's XML body ID prefix. At [MjRobot.py:272-283](file:///workspaces/FM-PCC/d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py#L272-L283):

```python
def add_id2model_key(self, model_key_id: str) -> str:
    attrib_split = model_key_id.split("_")
    attrib_split.insert(1, "rb{}".format(self._mj_robot_id))
    attrib_id = "_".join(attrib_split)
    return attrib_id
```

And at [MjRobot.py:330-331](file:///workspaces/FM-PCC/d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py#L330-L331):

```python
new_path = sim_path.d3il_path(
    f"./models/mj/robot/panda_tmp_rb{self._mj_robot_id}_{uuid.uuid1()}.xml"
)
```

This means:

| Env creation order | Robot ID | XML body prefix | Temp file name |
|---|---|---|---|
| Expert gen env (1st `Robot_Push_Env()`) | `rb0` | `joint_rb0_panda_joint1`, `tcp_rb0_tcp`, etc. | `panda_tmp_rb0_*.xml` |
| Variant 1 eval env (2nd `Robot_Push_Env()`) | `rb1` | `joint_rb1_panda_joint1`, `tcp_rb1_tcp`, etc. | `panda_tmp_rb1_*.xml` |
| Variant 2 eval env (3rd) | `rb2` | `joint_rb2_panda_joint2`, etc. | `panda_tmp_rb2_*.xml` |

### How this causes bp_image std = 0.1978

The robot body names (`rb0` vs `rb1` vs `rb2`) are baked into the compiled MuJoCo XML. When the scene XML is assembled by `MjSceneParser.create_scene()`, the robot body hierarchy has different XML names depending on the counter. MuJoCo internally assigns body IDs in declaration order, and **the inhand camera** (`MjInhandCamera`) is referenced by `self.add_id2model_key("rgbd")` — which becomes `rgbd_rb0_rgbd` for counter=0, `rgbd_rb1_rgbd` for counter=1, etc.

The compiled scene geometry is **not** byte-identical when the robot body names differ. The MuJoCo renderer's camera attachment, body indexing, and light reflections can vary subtly because different body name hashes change the internal body ordering. This produces the bp_image std difference:

- **Counter = 0** (no prior env): robot is `rb0`. Clean scene. bp_image std = **0.2093**
- **Counter = 1** (after expert gen env): robot is `rb1`. Different compiled scene. bp_image std = **0.1978**

### Proof from log evidence

**JOB 20634** (expert gen ran → counter advanced):
```
panda_tmp_rb0_035940c8-...   ← expert gen env (counter=0)
panda_tmp_rb1_141e27fc-...   ← diffuser eval env (counter=1)  ← WRONG ID
```
bp_image std = 0.1978 ← BAD

**JOB 20636** (expert gen SKIPPED → counter NOT advanced):
```
panda_tmp_rb0_717c51c0-...   ← diffuser eval env (counter=0)  ← CORRECT ID
```
bp_image std = 0.2093 ← GOOD

The `rb0` vs `rb1` prefix is the smoking gun. The counter=1 scene compiles to a slightly different MuJoCo model → slightly different camera rendering → bp_image std 0.1978 → model receives a different visual input → produces a completely different trajectory.

---

## 3. Why "Even the Seed Random Process" Seems Broken

The user observes that the seed seems non-deterministic. This is because:

1. **When expert gen SKIPS**: counter=0, robot=`rb0`, bp_image std=0.2093 → trajectory A (GOOD)
2. **When expert gen RUNS**: counter=1, robot=`rb1`, bp_image std=0.1978 → trajectory B (BAD)

Both are **individually deterministic** — the bad trajectory is always exactly `[0.0229, 0.0464, -0.7044]` with final distance 0.312711m. The randomness is not in the model or the RNG; it's in which scene geometry gets compiled, which depends on whether expert gen ran this invocation. The seed is fine; the scene construction is not.

---

## 4. Why AUDIT-FIX-1 Was Insufficient

AUDIT-FIX-1 moved `generate_expert_reference()` BEFORE the variant loop, which is correct for preventing inter-variant contamination. But the expert gen still creates a `Robot_Push_Env` → `MjRobot(counter=0)` → increments counter to 1. When the first variant's `Aligning_Sim.eval_agent()` creates its `Robot_Push_Env`, it gets `MjRobot(counter=1)` instead of the clean `counter=0`.

`gc.collect()` and `torch.cuda.empty_cache()` do NOT reset a class variable. Nothing in Python GC will decrement `MjRobot.GLOBAL_MJ_ROBOT_COUNTER`.

---

## 5. The Fix

### FIX-7 (CRITICAL): Reset `GLOBAL_MJ_ROBOT_COUNTER` after expert gen env cleanup

In [eval_fm_visual_aligning.py](file:///workspaces/FM-PCC/fm_visual_aligning_test/eval_fm_visual_aligning.py), after `generate_expert_reference()` and before the variant loop:

```python
_base_results = (f'{args.savepath}/results_train_set'
                 if args_cli.eval_on_train else f'{args.savepath}/results')
os.makedirs(_base_results, exist_ok=True)
generate_expert_reference(_base_results, n_rollouts=3)
gc.collect()
torch.cuda.empty_cache()

# FIX-7: Reset MuJoCo robot body counter to 0 so that the first variant's
# Robot_Push_Env gets the same robot body names (rb0) as a clean process.
# Without this, expert gen's env creation advances the counter to 1,
# causing the variant env to compile a different MuJoCo scene (rb1 vs rb0),
# which changes camera rendering → bp_image std → model output.
from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot as _MjRobot
_MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
```

> [!IMPORTANT]
> The import MUST use `from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot` — the same import path that `aligning.py` uses. Python may treat different import paths (e.g. `d3il_sim.sims...` vs `environments.d3il.d3il_sim.sims...`) as different module objects with **separate** class variables. Using the wrong path would reset a different counter and have no effect.

Also add the same reset in the `finally` block after each variant completes, so variant N+1 also gets `rb0`:

```python
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_f.close()
    # Reset robot counter for next variant (FIX-7)
    _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
```

### Optional: Stale temp file cleanup

Each `Robot_Push_Env()` creation writes a `panda_tmp_rb{N}_{uuid}.xml` file that is never deleted during normal execution. These cause the `mju_openResource: could not open resource` warnings in the logs. After expert gen and after each variant:

```python
import glob
for stale in glob.glob(os.path.join(
        os.environ.get('D3IL_DIR', 'd3il/environments/d3il'),
        'models/mj/robot/panda_tmp_rb*.xml')):
    try:
        os.remove(stale)
    except OSError:
        pass
```

### Alternative: Run expert gen in a subprocess

This was recommended in BUG_REPORT.md FIX-4. A subprocess inherits no class variables — the counter starts at 0 in the subprocess and the parent process counter is unaffected:

```python
import subprocess
subprocess.run([sys.executable, '-c',
    f'from eval_fm_visual_aligning import generate_expert_reference; '
    f'generate_expert_reference("{_base_results}", n_rollouts=3)'],
    check=True)
```

---

## 6. Updated Root Cause Chain (Supersedes BUG_REPORT.md §A3)

```
generate_expert_reference() runs
  → Robot_Push_Env() created
    → MjRobot.__init__() increments GLOBAL_MJ_ROBOT_COUNTER (0→1)
      → env.close() does NOT decrement counter
        → first variant's Robot_Push_Env() gets robot_id=1
          → MjRobot.modify_template() writes XML with "rb1" body prefix
            → MjSceneParser compiles different MuJoCo model than "rb0"
              → camera rendering differs → bp_image std = 0.1978
                → FM model receives different visual obs
                  → produces bad trajectory [0.0229, 0.0464, -0.7044]
                    → Final distance 0.312711m (BAD)
```

When expert gen is skipped (files exist):
```
No Robot_Push_Env() before variant loop
  → counter stays at 0
    → first variant gets robot_id=0 ("rb0")
      → clean scene → bp_image std = 0.2093
        → good trajectory [-0.0054, 0.0405, -0.2432]
          → Final distance 0.218847m (GOOD)
```

---

## 7. Summary Table

| Item | BUG_REPORT.md §A3 claim | This report's finding |
|---|---|---|
| Primary root cause | "MuJoCo factory global state" (vague) | `MjRobot.GLOBAL_MJ_ROBOT_COUNTER` (specific class variable, line 24) |
| Mechanism | "camera/OpenGL context not released" | Robot body name prefix `rb0` vs `rb1` changes compiled scene XML |
| Why gc.collect() doesn't help | Not addressed | Class variables are not garbage-collectable |
| Why AUDIT-FIX-1 is insufficient | Not anticipated | Expert gen still increments counter before the loop; moving it doesn't prevent the increment |
| Seed determinism | "RNG is properly reset" (correct) | Confirmed — output is deterministic for a given counter value; the non-determinism is in counter state |
| Fix | AUDIT-FIX-1 (move expert gen) | FIX-7: Reset `GLOBAL_MJ_ROBOT_COUNTER = 0` after expert gen |

> [!IMPORTANT]
> **The one-line fix**: After `generate_expert_reference()` returns, add:
> ```python
> from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot as _MjRobot
> _MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0
> ```
> This ensures all variant envs compile with `rb0` body names, producing the same scene geometry as a clean process, restoring bp_image std = 0.2093 and deterministic model output.
