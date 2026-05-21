# Gen7 FIX-7 / FIX-7.2 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Source MD Reference**: [fix_7/POSTMORTEM&change_log.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/POSTMORTEM%26change_log.md) · [fix_7/7.2/CHANGELOG_FIX7.2.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_7/7.2/CHANGELOG_FIX7.2.md)  
**Scope**: Both FIX-7 and FIX-7.2 applied to `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`.

## FIX-7 — `GLOBAL_MJ_ROBOT_COUNTER` reset
Reset `MjRobot.GLOBAL_MJ_ROBOT_COUNTER = 0` after expert gen and in per-variant `finally` block, plus stale `panda_tmp_rb*.xml` cleanup. Ensures all variant envs compile with `rb0` body names.

## FIX-7.2 — Render singleton cache reset
Clear `__RENDER_CTX_MAP` in `mj_render_singleton.py` via `reset_singleton()` after expert gen and after each variant. This was the **true root cause** of `bp_image std = 0.1978`: expert gen populated the cache with `RenderContextOffscreen(expert_model, expert_data)`; the variant's cameras hit the cached entry and rendered the expert gen's robot pose instead of the variant's. After this fix, all variant envs create fresh render contexts → `bp_image std = 0.2093` → correct trajectory.
