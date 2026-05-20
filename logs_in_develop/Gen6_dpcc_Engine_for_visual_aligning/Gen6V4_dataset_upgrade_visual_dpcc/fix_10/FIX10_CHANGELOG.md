# Fix 10 Changelog — max_episode_length Wire

**Date:** 2026-05-20  
**Branch:** update_into_FM  
**Scope:** Eval-only wiring — no model weights, no training code changed

---

## Root Cause

`max_episode_length: 1000` in `config/aligning-d3il-visual.py` was a dead field.
All evals silently ran under a 400-step cap — the hardcoded `max_steps_per_episode=400`
default in D3IL's `Robot_Push_Env`. The config key was never forwarded to the environment.

Sorting and stacking simulations both wire this correctly via `self.max_episode_length`.
Aligning was the only task that missed it.

See `FIX10_REPORT.md` for full root-cause analysis and cross-codebase verification.

---

## Changes

### 1. `d3il/simulation/aligning_sim.py`

Added `max_episode_length` parameter to `Aligning_Sim.__init__()` and forwarded it to
`Robot_Push_Env`.

```python
# Before
class Aligning_Sim(BaseSim):
    def __init__(
            self,
            seed: int, device: str, render: bool,
            n_cores: int = 1, n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False, eval_on_train: bool = False,
    ):
        ...
        self.eval_on_train = eval_on_train

# In eval_agent():
env = Robot_Push_Env(render=self.render, if_vision=self.if_vision)
```

```python
# After
class Aligning_Sim(BaseSim):
    def __init__(
            self,
            seed: int, device: str, render: bool,
            n_cores: int = 1, n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False, eval_on_train: bool = False,
            max_episode_length: int = 400,          # NEW
    ):
        ...
        self.eval_on_train = eval_on_train
        self.max_episode_length = max_episode_length  # NEW

# In eval_agent():
env = Robot_Push_Env(render=self.render, if_vision=self.if_vision,
                     max_steps_per_episode=self.max_episode_length)  # CHANGED
```

---

### 2. `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Added `max_episode_length` argument to the `Aligning_Sim(...)` constructor call.

```python
# Before
sim = Aligning_Sim(
    seed=seed, device=args.device,
    render=False, n_cores=1,
    n_contexts=n_contexts,
    n_trajectories_per_context=n_trajectories,
    if_vision=getattr(args, 'if_vision', True),
    eval_on_train=args_cli.eval_on_train,
)
```

```python
# After
sim = Aligning_Sim(
    seed=seed, device=args.device,
    render=False, n_cores=1,
    n_contexts=n_contexts,
    n_trajectories_per_context=n_trajectories,
    if_vision=getattr(args, 'if_vision', True),
    eval_on_train=args_cli.eval_on_train,
    max_episode_length=getattr(args, 'max_episode_length', 400),  # NEW
)
```

---

### 3. `config/aligning-d3il-visual.py` — comment fix

Fixed wrong comment that implied `max_episode_length` already controlled rollout steps
(it did not, until this fix).

```python
# Before (WRONG — described desired behavior as if it already existed)
# NOTE: max_episode_length controls rollout steps; max_path_length is only a loadpath key.

# After (CORRECT)
# NOTE: max_episode_length is forwarded to Robot_Push_Env(max_steps_per_episode=...).
#       max_path_length is a loadpath key only (checkpoint directory name fragment).
```

---

### 4. `config/aligning-d3il-visual.py` — `plan_visual_aligning_dpcc` block cleanup

Removed 4 dead parameters, corrected `max_episode_length` value, and wrapped long path
strings for readability.

**Removed dead params:**

| Param | Reason |
|---|---|
| `policy: 'sampling.Policy'` | Eval uses `VisualDPCCAgent` directly — never instantiated |
| `test_ret: 0` | Only read when `returns_condition: True`; dead here |
| `value_loadpath` | No value function exists in the DPCC pipeline |
| `dynamic_loss: False` | Training-only flag — meaningless in eval block |

**`max_episode_length` corrected:**

```python
# Before — aspirational value, was a dead field before Fix 10 wired it
'max_episode_length': 1000,

# After — D3IL proven baseline; comment documents when to raise it
# D3IL Robot_Push_Env default is 400 (hardcoded, proven stable for the aligning task).
# Fix 10 wired this field so it now actually reaches the env. Start at 400 (proven baseline).
# Increase only after confirming the model benefits from a longer rollout budget.
'max_episode_length': 400,
```

**Path strings wrapped** (same runtime value, readable in editor):

```python
# Before — single long line
'prefix': 'f:plans/visual_aligning_dpcc/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_V{if_vision}_steps{max_path_length}/',
'diffusion_loadpath': 'f:visual_aligning_dpcc/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_V{if_vision}_steps{max_path_length}',

# After — implicit string concatenation, three lines each
'prefix': (
    'f:plans/visual_aligning_dpcc/'
    'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
    '_aw{action_weight}_V{if_vision}_steps{max_path_length}/'
),
'diffusion_loadpath': (
    'f:visual_aligning_dpcc/'
    'H{horizon}_K{n_diffusion_steps}_D{diffusion}'
    '_aw{action_weight}_V{if_vision}_steps{max_path_length}'
),
```

---

## Impact

| Item | Before Fix 10 | After Fix 10 |
|------|--------------|--------------|
| Effective episode cap | 400 steps (D3IL hardcoded) | 400 steps (from config, now wired) |
| Config `max_episode_length` | Dead field (value was 1000, ignored) | Live — forwarded to env; value corrected to 400 |
| Prior eval results comparability | 400-step budget | Comparable (same effective cap, now intentional) |
| Dead config noise in plan block | 4 dead params present | Removed |

No retraining required. The model checkpoint is step-local (H=8 window).
`max_steps_per_episode` only affects the `while not done` eval budget.

All eval runs from fix_9 and earlier used a 400-step cap regardless of config.
Post-fix_10 runs use 400 steps from config explicitly — same budget, now correct and intentional.
To test a longer budget, change `max_episode_length` in the config and note the change in eval logs.

---

## Verification

Run on cluster with the existing `state_best.pt` checkpoint. No code changes beyond
the four file edits listed above. The fix is a pure eval-side wire — model inference,
diffusion pipeline, SLSQP projector, and 9D trajectory format are all unchanged.
