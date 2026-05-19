# Legacy Revert Log

**Date:** 2026-05-19

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
