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
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_9)
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_9)
- d3il/scripts/avoiding_vision/ (folder exists only in FM-PCC)
- d3il/simulation/aligning_sim.py (reverted FIX_9)
- d3il/.gitignore.bkp (only in FM-PCC)
- d3il/VENDORED_FROM.md (only in FM-PCC)

## Non-code / cache differences (safe to ignore)
- __pycache__/ in multiple subfolders
- /workspaces/d3il/.git and /workspaces/d3il/.gitignore

## Potentially Dangerous Differences (high-risk areas)
- d3il/simulation/aligning_sim.py (reverted FIX_9)
  - Simulation control path; mismatches can change rollout logic, seeding, or env construction.
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_9)
  - Environment dynamics and observation/action contracts; drift here changes training/eval behavior.
- d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py
  - MuJoCo robot model/controller changes can invalidate behavior or break physics assumptions.
- d3il/environments/d3il/d3il_sim/sims/mj_beta/MjLoadable.py
  - Low-level sim loading; mismatches can break assets or runtime initialization.
- d3il/environments/d3il/d3il_sim/utils/sim_path.py
  - Path resolution changes can cause dataset/model lookup failures or silent wrong data.
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_9)
  - Model asset change can alter collision/geometry behavior.

## Material Behavior Changes (confirmed by diff)
- d3il/simulation/aligning_sim.py (reverted FIX_9)
  - Adds `eval_on_train`, changes CPU pinning behavior for vision, alters return values, and adds rollout info hooks.
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_9)
  - `BPCageCam` now takes a named model key; camera instantiation behavior changes.
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_9)
  - `rod:tip` contact flags changed from `contype=0/conaffinity=0` to `1/1`, enabling collisions.

## Next step (if you want a line-by-line audit)
Run a targeted diff for any file above. Example:
```
diff -u /workspaces/d3il/simulation/aligning_sim.py /workspaces/FM-PCC/d3il/simulation/aligning_sim.py
```
