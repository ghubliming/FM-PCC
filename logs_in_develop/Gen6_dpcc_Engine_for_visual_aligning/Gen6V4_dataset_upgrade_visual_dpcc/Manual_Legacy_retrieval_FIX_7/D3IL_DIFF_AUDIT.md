# D3IL Diff Audit (FM-PCC vs Original)

**Date:** 2026-05-19
**Comparison:** /workspaces/FM-PCC/d3il vs /workspaces/d3il

## Summary
The following files differ between FM-PCC’s vendored D3IL folder and the original D3IL repo.
Cache folders and git metadata are listed separately.

## Code/Asset Differences
- d3il/agents/models/bet/libraries/mingpt/trainer.py
- d3il/environments/d3il/d3il_sim/sims/mj_beta/MjLoadable.py
- d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py
- d3il/environments/d3il/d3il_sim/utils/sim_path.py
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_7.3)
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_7.3)
- d3il/scripts/avoiding_vision/ (folder exists only in FM-PCC)
- d3il/simulation/aligning_sim.py (reverted FIX_7.3)
- d3il/.gitignore.bkp (only in FM-PCC)
- d3il/VENDORED_FROM.md (only in FM-PCC)

## Non-code / cache differences (safe to ignore)
- __pycache__/ in multiple subfolders
- /workspaces/d3il/.git and /workspaces/d3il/.gitignore

## Potentially Dangerous Differences (high-risk areas)
- d3il/simulation/aligning_sim.py (reverted FIX_7.3)
  - Simulation control path; mismatches can change rollout logic, seeding, or env construction.
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_7.3)
  - Environment dynamics and observation/action contracts; drift here changes training/eval behavior.
- d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py
  - MuJoCo robot model/controller changes can invalidate behavior or break physics assumptions.
- d3il/environments/d3il/d3il_sim/sims/mj_beta/MjLoadable.py
  - Low-level sim loading; mismatches can break assets or runtime initialization.
- d3il/environments/d3il/d3il_sim/utils/sim_path.py
  - Path resolution changes can cause dataset/model lookup failures or silent wrong data.
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_7.3)
  - Model asset change can alter collision/geometry behavior.

## Material Behavior Changes (confirmed by diff)
- d3il/simulation/aligning_sim.py (reverted FIX_7.3)
  - Adds `eval_on_train`, changes CPU pinning behavior for vision, alters return values, and adds rollout info hooks.
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_7.3)
  - `BPCageCam` now takes a named model key; camera instantiation behavior changes.
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_7.3)
  - `rod:tip` contact flags changed from `contype=0/conaffinity=0` to `1/1`, enabling collisions.

## Next step (if you want a line-by-line audit)
Run a targeted diff for any file above. Example:
```
diff -u /workspaces/d3il/simulation/aligning_sim.py /workspaces/FM-PCC/d3il/simulation/aligning_sim.py
```

---

## 📊 Current Status & Audit Matrix

Below is the definitive status mapping for every file identified in the diff audit between FM-PCC's vendored D3IL directory and the original D3IL repository.

| File / Directory | Current Status | Revert Reason / Technical Difference (vs. Original D3IL) |
| :--- | :--- | :--- |
| `d3il/simulation/aligning_sim.py` | **Reverted (FIX_7.3)** | Restored strict CPU process affinity pinning, original 30 test-context seeding limits, and reverted the return signature to omit the experimental `mean_distance` tuple value, ensuring absolute evaluation interface parity. |
| `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py` | **Reverted (FIX_7.3)** | Restored the original `BPCageCam` camera constructor instantiation signature, removing custom string keys which caused rendering pipeline initialization crashes. |
| `d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml` | **Reverted (FIX_7.3)** | Restored the contact parameters of `rod:tip` sphere geometry (`contype="0" conaffinity="0"`), making it non-colliding to align with original D3IL physics bounds and prevent solver warnings. |
| `d3il/agents/models/bet/libraries/mingpt/trainer.py` | **Still Active (Optimized)** | **Logging Optimization:** Limits the terminal progress bar refresh interval (`mininterval=1e10` and only updates `pbar` every 100 iterations or at epoch end) to prevent Slurm stdout buffer overflow crashes. |
| `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjLoadable.py` | **Still Active (Robust)** | **Cleanup Handler:** Employs an `atexit` clean-up callback to reliably remove intermediate loaded XML files upon script termination, preventing file system locks. |
| `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py` | **Still Active (Robust)** | **File Sync Guarantee:** Adds an explicit directory generation check and forces `f.flush()` + `os.fsync()` during intermediate robot description writes to prevent race conditions during MuJoCo loading. |
| `d3il/environments/d3il/d3il_sim/utils/sim_path.py` | **Still Active (Flexible)** | **Flexible Paths:** Enhances directory resolution to support an external environment variable override (`os.getenv("D3IL_DIR")`), enabling modular cluster path configuration. |
| `d3il/scripts/avoiding_vision/` | **Still Active (Unique)** | **Baseline Utilities:** An added directory containing local cluster launch scripts for avoiding-vision baselines, completely unique to FM-PCC. |
| `d3il/.gitignore.bkp` | **Still Active (Unique)** | Backup configuration for internal Git ignore paths. |
| `d3il/VENDORED_FROM.md` | **Still Active (Unique)** | Documentation tracking the exact timestamp and commit ID of the parent D3IL codebase from which the folder was originally vendored. |
