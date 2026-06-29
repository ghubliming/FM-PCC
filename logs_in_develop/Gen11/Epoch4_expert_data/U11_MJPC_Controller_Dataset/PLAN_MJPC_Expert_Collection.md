# U11 — MJPC Controller Expert Dataset Collection

**Status:** PLANNED (not started)
**Motivation:** E8 eval may reveal obs distribution mismatch — the training `p(t)` reflects PID dynamics; MJPC eval produces different `p(t)`. This collection provides a matched training set.
**Prerequisite reading:**
- [`DESIGN_dataset_pid_vs_mjpc.md`](../../../Gen11/Epoch8_UAV_Mjpc_thrust_control/DESIGN_dataset_pid_vs_mjpc.md) — full analysis of why PID data may or may not suffice
- [`E8 PLAN`](../../../Gen11/Epoch8_UAV_Mjpc_thrust_control/PLAN_MJPC_Thrust_Control.md) — E8 architecture
- [`mjpc_tracker.py`](../../../../FM_v3_uav_test/mjpc_tracker.py) — MJPC API wrapper already built for E8

**Decision rule:** Run E8 eval with existing PID dataset first. If success rate is clearly limited by `p` distribution mismatch (FM generates geometrically wrong paths despite MJPC tracking well), proceed with this collection. Otherwise, PID dataset is sufficient.

---

## 1 What Changes vs PID Collection

The E4 pipeline has two layers. Only **Layer 2** changes:

```
Layer 1 — Geo trajectory (NO CHANGE)
  traj_fn(t) → p_des(t), v_des(t), a_des(t), yaw_des(t)
  Same scenes, same homotopy classes, same waypoint geometry, same randomisation.

Layer 2 — Physics execution (CHANGE: CascadedPID → MJPCTracker)
  PID:  pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des) → u[4]
  MJPC: tracker.compute(p, q, v, om, p_des) → u[4]
  Same mujoco.mj_step(model, data).
  Same recording: {'p': p, 'v': v, 'p_des': p_des, 'q': q}
```

`dataset_writer.py`, `trajectories.py`, `curate_dataset.py`, pkl schema — **all unchanged**.
The only output difference: `p(t)` and `v(t)` inside `obs` reflect MJPC tracking instead of PID tracking.

---

## 2 Code Changes Required

### 2.1 `uav_expert_data_collect/generator.py`

**Add MJPC constructor** alongside `_make_pid`:

```python
def _make_mjpc_tracker(model, task_id='Quadrotor',
                        n_trajectories=32, horizon=0.3, planner_steps=10):
    from FM_v3_uav_test.mjpc_tracker import MJPCTracker
    return MJPCTracker(model, task_id=task_id,
                       n_trajectories=n_trajectories,
                       horizon=horizon, planner_steps=planner_steps)
```

**Add `controller_type` param to `run_trial`:**

```python
def run_trial(scene, homotopy, gain_variant, seed, duration=None,
              controller_type='pid', mjpc_kwargs=None):
    ...
    if controller_type == 'mjpc':
        tracker = _make_mjpc_tracker(model, **(mjpc_kwargs or {}))
    else:
        tracker = _make_pid(model, gain_variant)
    ...
    for k in range(n_step):
        ...
        u = tracker.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)
        # MJPCTracker.compute() accepts v_des/a_des/yaw_des for API parity (ignores them)
        ...
    if controller_type == 'mjpc' and hasattr(tracker, 'close'):
        tracker.close()   # release gRPC agent server
```

**Update return dict:**
```python
    return {
        ...
        'controller': controller_type,    # was: gain_variant (still store gain_variant in metadata)
        ...
    }
```

### 2.2 `uav_expert_data_collect/collect.py`

Add CLI flags:
```python
p.add_argument('--controller',       default='pid', choices=['pid', 'mjpc'])
p.add_argument('--mjpc-task-id',     default='Quadrotor')
p.add_argument('--mjpc-trajectories',type=int, default=32)
p.add_argument('--mjpc-horizon',     type=float, default=0.3)
p.add_argument('--mjpc-planner-steps',type=int, default=10)
```

Pass through to `run_trial`:
```python
mjpc_kwargs = {
    'task_id':       args.mjpc_task_id,
    'n_trajectories': args.mjpc_trajectories,
    'horizon':       args.mjpc_horizon,
    'planner_steps': args.mjpc_planner_steps,
} if args.controller == 'mjpc' else None

rollout = run_trial(
    scene=args.scene,
    homotopy=homotopy,
    gain_variant=args.gain_variant,
    seed=trial_seed,
    controller_type=args.controller,
    mjpc_kwargs=mjpc_kwargs,
)
```

Default output dir should include controller tag:
```python
out_root = args.out_dir or os.path.join(
    _REPO, 'logs', 'uav_expert_data_mjpc', args.scene)
```

### 2.3 `uav_expert_data_collect/dataset_writer.py`

**No schema changes.** The pkl `controller` field already stores a string (was `'pid_default'` etc.); it will now store `'mjpc'`. `obs`, `actions`, `targets`, `q` — identical shape and meaning.

Optional: add `mjpc_planner_ms` to `metadata` dict (read from `tracker.last_plan_ms`) for timing diagnostics.

### 2.4 `uav_expert_data_collect/curate_dataset.py`

Add `--raw-root` pointing at `logs/uav_expert_data_mjpc` and `--out` pointing at `data/uav_fm/v2` (separate versioned tree so PID `v1` dataset is never overwritten):

```bash
python uav_expert_data_collect/curate_dataset.py \
    --raw-root logs/uav_expert_data_mjpc \
    --scenes empty corridor s_curve pillars \
    --out data/uav_fm/v2
```

Then update `config/uav.py` `dataset_root` to `data/uav_fm/v2` for the MJPC-trained variant.

---

## 3 Run Commands (when ready)

```bash
# Per scene — run in parallel on cluster (4 jobs)
python uav_expert_data_collect/collect.py \
    --scene empty    --n-trials 200 --controller mjpc \
    --mjpc-task-id Quadrotor --mjpc-trajectories 32 --mjpc-planner-steps 10

python uav_expert_data_collect/collect.py \
    --scene corridor --n-trials 300 --controller mjpc ...

python uav_expert_data_collect/collect.py \
    --scene s_curve  --n-trials 300 --controller mjpc ...

python uav_expert_data_collect/collect.py \
    --scene pillars  --n-trials 400 --controller mjpc ...

# Then curate into v2
python uav_expert_data_collect/curate_dataset.py \
    --raw-root logs/uav_expert_data_mjpc \
    --out data/uav_fm/v2
```

---

## 4 Open Questions (resolve before running)

### 4.1 MJPC task_id — most critical blocker

The `MJPCTracker` sets the goal via `agent.set_state(mocap_pos=p_des)`. Whether the stock `'Quadrotor'` task respects this depends on whether its cost function uses `mocap_pos` as a position target. The stock task has auto-advancing gate waypoints (TransitionLocked) which may override our goal.

**Options:**
- A. Verify the stock `'Quadrotor'` task uses `mocap_pos` as position residual target (check `quadrotor/task.xml` residuals). If yes, use as-is.
- B. Write a minimal `position_tracking` task XML (no gate waypoints, pure `||q_pos - mocap_pos||²` cost). This is the safest path.

Resolve this with a 5-minute standalone test on the cluster: give MJPC a fixed `p_des` and see if it reaches it.

### 4.2 Collection speed

MJPC is ~50× slower per step than PID (sampling rollouts vs analytic cascade). At 100 Hz physics × 10 planner steps × 32 trajectories, each collection step may take 30–100 ms wall time. A 10 s episode = 1000 steps → 30–100 s per episode vs <1 s with PID.

**Mitigation:**
- Reduce physics to 33 Hz during collection (`stride=1` instead of `stride=3`). Dataset is at 33 Hz anyway; running physics at 33 Hz for collection is fine for MJPC (it runs at eval freq too).
- Reduce `n_trials` vs PID collection if speed is bottleneck (MJPC data may need fewer episodes since tracking is tighter).
- Run scenes in parallel on cluster (4 nodes × 1 scene each).

### 4.3 Rejection threshold adjustment

The PID rejection criteria (`SCENE_MAX_CONTACT_FRACTION`, `Z_FLOOR_MARGIN`) were tuned for PID tracking behavior. MJPC may have different transient characteristics at trajectory start (cold-start planner). Consider:
- Warm-start MJPC for 10–20 planner steps before the episode starts (hover at `init_pos`)
- Check if the existing floor margin `Z_FLOOR_MARGIN=0.5 m` is sufficient for MJPC startup

### 4.4 `gain_variant` parameter

The `gain_variant` field in the pkl currently stores PID gain info (`'pid_default'` etc.). For MJPC collection, store a string encoding the MJPC config, e.g. `'mjpc_N32_H03_S10'` (N=trajectories, H=horizon, S=steps). This preserves the discriminator function without changing the schema.

### 4.5 Dataset version and config key

Train the MJPC-dataset model under a new `dataset_root = 'data/uav_fm/v2'`. Add a `dataset_version` key to `config/uav.py` so the exp_name path encodes which dataset was used:

```python
# e.g. _uav_exp_name appends _dv2 when dataset_version != 'v1'
```

This keeps the three checkpoint families discriminated:
- `H8_D…` — E7 PID 12D (v1 data)
- `H8_D…_cmpos_only_ctrlmjpc` — E8 MJPC 9D, PID-collected data (v1)
- `H8_D…_cmpos_only_ctrlmjpc_dv2` — E8 MJPC 9D, MJPC-collected data (v2)

---

## 5 Files to Touch

| File | Change | Scope |
|---|---|---|
| `uav_expert_data_collect/generator.py` | Add `_make_mjpc_tracker`, `controller_type` param to `run_trial`, `tracker.close()` | ~30 lines |
| `uav_expert_data_collect/collect.py` | Add `--controller` + `--mjpc-*` CLI flags, plumb to `run_trial` | ~20 lines |
| `uav_expert_data_collect/dataset_writer.py` | Optional: add `mjpc_planner_ms` to metadata | ~3 lines |
| `config/uav.py` | Add `dataset_version` discriminator to `_uav_exp_name` | ~5 lines |
| `uav_expert_data_collect/curate_dataset.py` | No change — just call with `--raw-root` + `--out` pointing at v2 | CLI only |

---

## 6 Decision: When to Do This

**Do this ONLY IF** E8 eval (on PID-collected data) shows clear evidence of obs mismatch:
- MJPC tracks `p_des` accurately (tracking error small, no crashes)
- BUT FM generates geometrically bad paths (collision-prone, wrong homotopy class)
- i.e. the tracker is fine but the planner misbehaves — suggesting the `p` conditioning shifted

**Skip this if** E8 eval success rate is reasonable — the PID `p` distribution is close enough.

**Start with:** E8 eval → examine failure modes → decide.
