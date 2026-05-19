# Legacy Revert Log

**Date:** 2026-05-19

## FIX_7.5 — Restore CPU Affinity Bypass for Visual Mode (2026-05-19)

### Reason
FIX_7.3 restored `assign_process_to_cpu` unconditionally to match original D3IL.
However, original D3IL has no visual mode — the pinning was designed for lightweight
state-only rollouts. With `if_vision=True` and K=100, forcing MuJoCo OpenGL rendering,
ResNet inference, CUDA polling, and SciPy SLSQP projection onto a single CPU core ({0})
causes thread starvation and a permanent hang at Context 0 Rollout 0 (observed: 15+ min
with no progress). This was previously diagnosed and documented in
`logs_in_develop/Gen7_FMPCC_Viusal_Aligning/d3il_modification/CPU_AFFINITY_BYPASS.md`.

### File changed
**File:** d3il/simulation/aligning_sim.py

**Before (FIX_7.3 unconditional pinning):**
```python
print(os.getpid(), cpu_set)
assign_process_to_cpu(os.getpid(), cpu_set)
```

**After (FIX_7.5 — gates pinning on non-visual mode):**
```python
print(os.getpid(), cpu_set)
if not self.if_vision:
    assign_process_to_cpu(os.getpid(), cpu_set)
else:
    print(f"Process {os.getpid()} unpinned — visual eval requires all CPU threads (OpenMP/CUDA/SLSQP).")
```

### Impact
- State-only eval (`if_vision=False`): pinning unchanged — exact D3IL parity preserved.
- Visual eval (`if_vision=True`): process runs unpinned, allowing all 64 cores to serve
  PyTorch workers, OpenMP threads, OpenGL drivers, and SLSQP optimizer without starvation.
- Resolves the 15+ min hang at Context 0 Rollout 0 introduced by FIX_7.3.
- This is an intentional FM-PCC extension; original D3IL never runs visual rollouts.

### Also fixed in this patch (same file)

**Bug A — Video/GIF never saved:**
`eval_agent` called `agent.reset()` at rollout start (which clears `video_frames`) but never
called `agent.update_rollout_info()` after rollout end — the only path to `_save_diagnostics()`.
Every rollout's frames were silently discarded by the next `reset()`.

```python
# Added after info tensors are recorded:
if hasattr(agent, 'update_rollout_info'):
    agent.update_rollout_info({**info, 'context': context})
```

**Bug B — Return value unpack crash:**
`test_agent` returned 2 values `(success_rate, mode_encoding)` but eval script unpacked 4.
Would crash immediately after all rollouts finished.

```python
# Before:
return success_rate, mode_encoding#, mean_distance
# After:
return success_rate, mode_encoding, successes, mean_distance
```

---

## FIX_7.1 — Revert Fix 38 (max_episode_length plumbing)

### What was reverted
Fix 38 introduced `max_episode_length` plumbing into `Aligning_Sim` and forwarded it to
`Robot_Push_Env(max_steps_per_episode=...)`, plus passed the value from the eval script.
This was reverted because it caused server problems in the reported commit.

### Files changed

#### 1) Aligning_Sim constructor and env creation
**File:** d3il/simulation/aligning_sim.py

**Removed from __init__ signature and state:**
```python
            if_vision: bool = False,
            eval_on_train: bool = False,
            max_episode_length: int = 400
```

**Removed stored field:**
```python
        self.max_episode_length = max_episode_length
```

**Restored env construction without max_steps_per_episode:**
```python
        env = Robot_Push_Env(render=self.render, if_vision=self.if_vision)
```

#### 2) Eval script no longer passes max_episode_length
**File:** ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py

**Removed extra arg in Aligning_Sim call:**
```python
                sim = Aligning_Sim(seed=seed, device=args.device, render=False, n_cores=1,
                                  n_contexts=n_contexts, n_trajectories_per_context=n_trajectories, if_vision=sim_vision,
                                  eval_on_train=args_cli.eval_on_train)
```

### Impact
- `Aligning_Sim` no longer tracks `max_episode_length`.
- `Robot_Push_Env` runs with its internal default episode length.
- Eval no longer tries to override episode length from config.

---

## FIX_7.4 — Restore eval_on_train to Aligning_Sim (2026-05-19)

### Reason
FIX_7.3 removed `eval_on_train` from `Aligning_Sim.__init__` to restore D3IL parity.
However `eval_visual_aligning_dpcc.py` still passes `eval_on_train=args_cli.eval_on_train`,
causing a `TypeError` crash at eval launch. The feature is required for evaluating on
training contexts (in-distribution check during development).

### Files changed

#### 1) Restore eval_on_train parameter and context selection
**File:** d3il/simulation/aligning_sim.py

**Restored __init__ signature:**
```python
        n_contexts: int = 30,
        n_trajectories_per_context: int = 1,
        if_vision: bool = False,
        eval_on_train: bool = False       # ← restored
```

**Restored stored field:**
```python
        self.eval_on_train = eval_on_train
```

**Restored context selection in eval_agent (replaces hardcoded test_contexts):**
```python
# Before (FIX_7.3 hardcoded):
obs = env.reset(random=False, context=test_contexts[context])

# After (FIX_7.4 restored):
ctx_pool = train_contexts if self.eval_on_train else test_contexts
obs = env.reset(random=False, context=ctx_pool[context])
```

#### 2) Restore eval_on_train arg in visual-aligning eval script
**File:** diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py

**Restored in Aligning_Sim call (also removed max_episode_length which is not in constructor):**
```python
# Before (broken after FIX_7.3):
sim = Aligning_Sim(
    seed=seed, device=args.device,
    render=False, n_cores=1,
    n_contexts=n_contexts,
    n_trajectories_per_context=n_trajectories,
    if_vision=getattr(args, 'if_vision', True),
    eval_on_train=args_cli.eval_on_train,
    max_episode_length=getattr(args, 'max_episode_length', 400),  # ← not in constructor
)

# After (fixed):
sim = Aligning_Sim(
    seed=seed, device=args.device,
    render=False, n_cores=1,
    n_contexts=n_contexts,
    n_trajectories_per_context=n_trajectories,
    if_vision=getattr(args, 'if_vision', True),
    eval_on_train=args_cli.eval_on_train,
)
```

### Impact
- `TypeError: Aligning_Sim.__init__() got an unexpected keyword argument 'eval_on_train'` crash resolved.
- `--eval_on_train` flag works as expected: True → train contexts, False → test contexts (default).
- `max_episode_length` kwarg removed from visual eval call (was never in the reverted constructor).

---

## FIX_7.2 — Revert D3IL Aligning_Img_Dataset RGB Conversion

### What was reverted
Commit `6f42a73427cb0197377e3ed5556f9a5cfb5d6f5e` introduced BGR→RGB conversion when loading
images in the D3IL Aligning_Img_Dataset. This change was reverted in FM-PCC to restore
byte-for-byte parity with the original D3IL repository and avoid distribution shift for
existing checkpoints.

### Source of truth (original D3IL)
- File: d3il/environments/dataset/aligning_dataset.py
- Behavior: `cv2.imread(...).astype(np.float32)` with no `cv2.cvtColor` conversion.

### FM-PCC revert actions

#### File changed
**File:** d3il/environments/dataset/aligning_dataset.py

**Reverted code (bp camera loop):**
```python
            for img in bp_imgs:
                image = cv2.imread(img).astype(np.float32)
                image = image.transpose((2, 0, 1)) / 255.
```

**Reverted code (inhand camera loop):**
```python
            for img in inhand_imgs:
                image = cv2.imread(img).astype(np.float32)
                image = image.transpose((2, 0, 1)) / 255.
```

### Reason
- Maintain strict parity with original D3IL image preprocessing.
- Avoid unintended color-space distribution shifts for legacy checkpoints.

### Notes
If RGB conversion is desired for new training, reintroduce it explicitly and retrain.

---

## FIX_7.3 — Revert Material D3IL Behavior Drift (Aligning Sim + Camera + Rod Collision)

### Commit reference
Commit ID not provided in repo history; revert applied to restore parity with /workspaces/d3il.

### What was reverted
Three material behavior changes in the vendored D3IL code were reverted to match the original
D3IL repo:

1) Aligning simulation behavior changes (eval_on_train, CPU pinning, return signature)
2) Aligning environment camera instantiation (named camera key)
3) Rod tip collision activation in the panda model

### Files changed

#### 1) Aligning_Sim behavior parity
**File:** d3il/simulation/aligning_sim.py

**Restored __init__ signature (removed eval_on_train):**
```python
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False
```

**Restored CPU pinning for all modes:**
```python
        assign_process_to_cpu(os.getpid(), cpu_set)
```

**Restored test-context reset path:**
```python
                obs = env.reset(random=False, context=test_contexts[context])
```

**Restored return signature:**
```python
        return success_rate, mode_encoding#, mean_distance
```

#### 2) Aligning environment camera parity
**File:** d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py

**Restored BPCageCam constructor signature and name:**
```python
    def __init__(self, width: int = 96, height: int = 96, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],
            *args,
            **kwargs,
        )
```

**Restored camera instantiation:**
```python
        self.bp_cam = BPCageCam()
```

#### 3) Panda rod collision parity
**File:** d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml

**Restored rod tip non-colliding contact flags:**
```xml
<geom type="sphere" size="0.01" pos="0 0 0.225" rgba="1 0 0 1" contype="0" conaffinity="0" name="rod:tip"/>
```

### Reason
These were confirmed material behavior changes relative to the original D3IL repo and
could alter evaluation dynamics, camera wiring, or physics contacts. Reverted to ensure
parity and avoid silent drift in results.
