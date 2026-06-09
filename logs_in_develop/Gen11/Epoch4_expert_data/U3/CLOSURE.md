# Gen11 Epoch 4 — U3 Closure

**Closed:** 2026-06-09  
**Final job:** 21356  
**Status:** ✅ Complete

---

## What U3 set out to do

Opened by `Epoch5_visual_and_validation/U2/INVESTIGATE_pillar_passthrough.md`:  
full geometry audit of all scenes + re-collection with corrected trajectories + `q=(T,4)`
quaternion field populated across all episodes.

---

## Final dataset

| Scene | Saved | Rejected | Rejection % | Notes |
|---|---|---|---|---|
| empty | 500 | 0 | 0% | ✅ |
| corridor | 500 | 0 | 0% | ✅ |
| s_curve | 356 | 144 | 28.8% | ✅ usable — structural (geometry too tight for some random curves) |
| pillars | 473 | 27 | 5.4% | ✅ PID tracking noise only, zero geometry collisions |
| **Total** | **1829** | **171** | **8.6%** | |

Output: `logs/uav_expert_data/{scene}/`

---

## Fixes applied

### Fix_1 — Pillar trajectory geometry (`trajectories.py`)

**Problem:** 5-waypoint `pillar_path` placed waypoints AT pillar x-positions. Rotor 3
(`+0.14, +0.18` offset) reached into the pillar cylinder on approach → −10.3 cm clearance
→ 100% rejection (job 21342).

**Fix:** 8-waypoint design with transitions in mid-spans (`x = −2.5, −1.5, −0.5, +0.5,
+1.5, +2.5`). Analytical min clearance = +8.0 cm for all 4 homotopies.  
See `Fix_1/ANALYSIS.md` for full geometry derivation.

### Fix_1 — `collect.sh` GL backend (`MUJOCO_GL=disabled`)

**Problem:** Group C EGL cleanup (SLURM GPU IT fix) removed `MUJOCO_GL=egl` from
`collect.sh`. MuJoCo 2.3.7 fell back to osmesa (not installed) → `AttributeError` at
import, 0 episodes (job 21354).

`MUJOCO_GL=egl` was also wrong: `mujoco/egl/__init__.py:65` calls `eglInitialize()` at
module import time → GPU 0 opened even without rendering → IT violation.

**Fix:** `MUJOCO_GL="disabled"` — skips the entire `gl_context.py` backend block
(`gl_context.py:25`). Physics APIs work with no GL backend; `collect.py` never creates
a `Renderer`. Zero GPU footprint confirmed by job 21356 debug check:
`[ GPU-LEAK CHECK ] DRI fds: NONE — clean`.  
See `Fix_1/COLLECT_SH_GL_FIX.md` for full source-verified analysis.

---

## Jobs

| Job | Date | Result | Notes |
|---|---|---|---|
| 21342 | 2026-06-08 | ❌ 0/500 pillars | 100% rejection — geometry bug |
| 21354 | 2026-06-09 | ❌ crash | osmesa crash — MUJOCO_GL removed |
| 21356 | 2026-06-09 | ✅ 473/500 pillars | Clean run; GPU-leak verified |

---

## Artefacts

| File | Purpose |
|---|---|
| `Fix_1/ANALYSIS.md` | Pillar geometry root cause + 8-waypoint design + all fixes |
| `Fix_1/COLLECT_SH_GL_FIX.md` | MuJoCo GL backend analysis (source-verified) |
| `uav_expert_data_collect/trajectories.py` | `pillar_path` rewritten (8-waypoint) |
| `Slurm_Codes/sbatch/uav_expert_data/collect.sh` | `MUJOCO_GL=disabled`; debug check line (comment out) |
| `logs_in_develop/SLURM_GPU_IT_WARNING/CHANGELOG.md` | Group C entry corrected |

---

## Unblocks

- **E5 U3** (`Epoch5_visual_and_validation/U3/PLAN.md`) — physics replay GIFs — depends on these clean pillar pickles with `q` field
- **E6 training** — full 1829-episode dataset available
