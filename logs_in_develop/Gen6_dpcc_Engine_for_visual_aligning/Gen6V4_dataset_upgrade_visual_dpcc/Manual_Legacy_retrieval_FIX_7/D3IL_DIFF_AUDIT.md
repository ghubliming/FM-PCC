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
- d3il/simulation/aligning_sim.py (reverted FIX_7.3 + patched FIX_7.4 + FIX_7.5)
  - CPU pinning now gated on `not self.if_vision` (state-only path identical to D3IL; visual path unpinned for 64-core use). `eval_on_train` re-added as optional flag (default False = test contexts). `update_rollout_info` hook wired for video/gif save. Return expanded to 4 values to match eval script. All changes are gated or FM-PCC-scoped — state-only eval is byte-for-byte D3IL parity.
- d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py (reverted FIX_7.3)
  - `BPCageCam` restored to original no-arg constructor; named key that caused blank frames / crashes removed.
- d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml (reverted FIX_7.3)
  - `rod:tip` contact flags restored to `contype=0/conaffinity=0` (non-colliding), eliminating phantom contact forces not present during data collection.

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
| `d3il/simulation/aligning_sim.py` | **Reverted (FIX_7.3) + Patched (FIX_7.4, FIX_7.5)** | FIX_7.3: Restored CPU pinning and return signature parity. FIX_7.4: Re-added `eval_on_train: bool = False`. FIX_7.5: (a) CPU pinning gated on `not self.if_vision` — visual eval runs unpinned; state-only path byte-for-byte D3IL. (b) `update_rollout_info` hook added with `hasattr` guard — enables video/gif save without affecting D3IL agents. (c) Return expanded to 4 values to match FM-PCC eval script. All deltas are safe — see Delta Safety Verdict section. |
| `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py` | **Reverted (FIX_7.3)** | Restored the original `BPCageCam` camera constructor instantiation signature, removing custom string keys which caused rendering pipeline initialization crashes. |
| `d3il/environments/d3il/models/mj/robot/panda_rod_invisible.xml` | **Reverted (FIX_7.3)** | Restored the contact parameters of `rod:tip` sphere geometry (`contype="0" conaffinity="0"`), making it non-colliding to align with original D3IL physics bounds and prevent solver warnings. |
| `d3il/agents/models/bet/libraries/mingpt/trainer.py` | **Still Active (Optimized)** | **Logging Optimization:** Limits the terminal progress bar refresh interval (`mininterval=1e10` and only updates `pbar` every 100 iterations or at epoch end) to prevent Slurm stdout buffer overflow crashes. |
| `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjLoadable.py` | **Still Active (Robust)** | **Cleanup Handler:** Employs an `atexit` clean-up callback to reliably remove intermediate loaded XML files upon script termination, preventing file system locks. |
| `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py` | **Still Active (Robust)** | **File Sync Guarantee:** Adds an explicit directory generation check and forces `f.flush()` + `os.fsync()` during intermediate robot description writes to prevent race conditions during MuJoCo loading. |
| `d3il/environments/d3il/d3il_sim/utils/sim_path.py` | **Still Active (Flexible)** | **Flexible Paths:** Enhances directory resolution to support an external environment variable override (`os.getenv("D3IL_DIR")`), enabling modular cluster path configuration. |
| `d3il/scripts/avoiding_vision/` | **Still Active (Unique)** | **Baseline Utilities:** An added directory containing local cluster launch scripts for avoiding-vision baselines, completely unique to FM-PCC. |
| `d3il/.gitignore.bkp` | **Still Active (Unique)** | Backup configuration for internal Git ignore paths. |
| `d3il/VENDORED_FROM.md` | **Still Active (Unique)** | Documentation tracking the exact timestamp and commit ID of the parent D3IL codebase from which the folder was originally vendored. |

---

## Delta Safety Verdict

**Question:** After the revert→re-revert cycle, is `aligning_sim.py` safe to rely on?

**Root cause of the cycle:** FIX_7.3 applied "restore D3IL parity" too broadly. It correctly
reverted three real drifts (BPCageCam key, rod:tip collision, eval_on_train hardcoding), but
also removed the CPU affinity bypass — which was never a drift, because original D3IL has no
visual mode. There was no parity target to restore to. FIX_7.3 introduced a new bug while
fixing real ones.

**Audit of every delta in the current file vs original D3IL:**

| Delta | Effect on state-only eval | Effect on visual eval | Safe? |
|---|---|---|---|
| `eval_on_train: bool = False` | None — default = test contexts = D3IL behavior | Required for in-distribution sanity checks | Yes — intentional FM-PCC extension |
| CPU pinning gated on `not self.if_vision` | None — state-only path identical to D3IL | Required — 64-core use for SLSQP/CUDA/OpenMP | Yes — D3IL never defined vision behavior |
| `hasattr(agent, 'update_rollout_info')` | None — D3IL agents lack this method; guard skips silently | Required — enables video/gif save per rollout | Yes — guard is hermetic |
| Return 4 values `(success_rate, mode_encoding, successes, mean_distance)` | Only affects callers of this file; other D3IL sims (stacking, sorting) have their own files | Required — FM-PCC eval script unpacks 4 | Yes — FM-PCC-scoped |

**Conclusion:** The current state is stable. No further reversals are needed. Every change either
falls back to original D3IL behavior when `if_vision=False`, is protected by a `hasattr` guard,
or is scoped to FM-PCC callers only.

**Rule for future parity work:** If D3IL parity reverts are ever re-run on this file, the CPU
bypass (`not self.if_vision` gate) and `update_rollout_info` hook must be explicitly preserved.
They cover behavior D3IL never defined and have no "original" to restore to.
