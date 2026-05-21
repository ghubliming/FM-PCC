# Fix 1 — Dataset Buffer Overflow (2026-05-19)

## Crash

```
ValueError: could not broadcast input array from shape (330,20) into shape (256,20)
  File "d3il/environments/dataset/aligning_dataset.py", line 83
    zero_obs[0, :valid_len, :] = input_state[:-1]
```

Triggered at first execution of `train_visual_aligning_dpcc.py` on the cluster.

---

## Root Cause

`Aligning_Dataset.__init__()` allocates a fixed `zero_obs = np.zeros((1, max_len_data, obs_dim))` buffer with `max_len_data=256` (hardcoded default). At least one episode in `train_files.pkl` contains 331 state frames → `valid_len=330 > 256` → the slice `zero_obs[0, :330, :]` is shape `(256, 20)` but `input_state[:-1]` is `(330, 20)` → crash.

We cannot modify `d3il/` (copy-modify-only rule), so the fix is to bypass `Aligning_Dataset` entirely.

---

## Fix

**File**: `diffuser_visual_aligning/datasets/sequence.py`

**Before**: `ParityAligningDataset.__init__()` called:
```python
base = Aligning_Dataset(
    data_directory=dataset_path,
    obs_dim=20, action_dim=3, window_size=1, device='cpu',
)
# then extracted base.observations, base.actions, base.masks
```

**After**: Reads state pickles directly — no fixed buffer, no truncation:
```python
for file in tqdm(state_files[:n_eps], desc='Loading states'):
    with open(os.path.join(rp_data_dir, file), 'rb') as f:
        env_state = pickle.load(f)
    robot_des_pos = env_state['robot']['des_c_pos']   # (T+1, 3)
    robot_c_pos   = env_state['robot']['c_pos']       # (T+1, 3)
    T = len(robot_des_pos) - 1
    obs_6d  = np.concatenate([robot_des_pos[:T], robot_c_pos[:T]], axis=-1)  # (T, 6)
    actions = robot_des_pos[1:] - robot_des_pos[:-1]                          # (T, 3)
    all_obs_6d.append(obs_6d)
    all_actions.append(actions)
```

`_obs_6d` and `_actions` are now **lists of variable-length arrays** (not a padded 3D tensor).

### Cascading change in `__getitem__`

Old indexing (3D array):
```python
obs_raw = self._obs_6d[ep, start:end]
act_raw = self._actions[ep, start:end]
```

New indexing (list of arrays):
```python
obs_raw = self._obs_6d[ep][start:end]
act_raw = self._actions[ep][start:end]
```

### Cascading change in `_make_indices`

Old: `valid_len = int(self._masks[ep].sum())`
New: `T = len(self._obs_6d[ep])`  — masks field eliminated entirely

---

## Semantic equivalence

`Aligning_Dataset` computes:
- `obs = np.concatenate((robot_des_pos, robot_c_pos, push_box_pos, ...), axis=-1)` → 20D but stores first 6 as `[des_c_pos | c_pos]`
- `vel_state = robot_des_pos[1:] - robot_des_pos[:-1]`

Our direct load replicates this exactly for the 6D subset we need:
- `obs_6d = [robot_des_pos[:T] | robot_c_pos[:T]]`
- `actions = robot_des_pos[1:] - robot_des_pos[:-1]`

No change to trajectory semantics, normalizer math, or DPCC principle.

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning/datasets/sequence.py` | Bypass `Aligning_Dataset`; load pickles directly; list-of-arrays storage; add `import pickle` |
