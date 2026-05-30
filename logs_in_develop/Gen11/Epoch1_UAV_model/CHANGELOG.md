# Gen11 Epoch 1 — UAV Model Migration: Changelog

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Scope**: Model assets only — Skydio X2 quadrotor XML, mesh, texture, gates, LICENSE.
**Plan**: [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) §11
**Sources**: `mujoco_menagerie/skydio_x2/` (upstream, GitHub) + `mujoco_mpc/mjpc/tasks/quadrotor/`
**Method**: CLI only (`git clone`, `cp`, `patch`) — zero LLM-generated content.

---

## Files Created

| File | Source | Why |
|---|---|---|
| `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor.xml` | Verbatim `cp` from `mujoco_menagerie/skydio_x2/x2.xml` (sparse-cloned to `/tmp/`) | Base X2 model XML. Identical to upstream — `diff -q` empty. |
| `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml` | `patch -o … quadrotor.xml < mujoco_mpc/.../quadrotor.xml.patch --binary` | MJPC's drone-task-ready variant: adds `quat="0 0 0 1"` on the x2 body, removes the imu-site sensor block and the hover keyframe (MJPC supplies those at task level). |
| `d3il/environments/d3il/models/mj/robot/quadrotor/gates.xml` | Verbatim `cp` from `mujoco_mpc/mjpc/tasks/quadrotor/gates.xml` | 8 racing-gate static bodies. Kept as-is for any future racing/obstacle work. |
| `d3il/environments/d3il/models/mj/robot/quadrotor/assets/X2_lowpoly.obj` | Verbatim `cp` from `mujoco_menagerie/skydio_x2/assets/` | Low-poly mesh referenced by both XML variants. |
| `d3il/environments/d3il/models/mj/robot/quadrotor/assets/X2_lowpoly_texture_SpinningProps_1024.png` | Verbatim `cp` from Menagerie assets | Texture for the mesh. |
| `d3il/environments/d3il/models/mj/robot/quadrotor/LICENSE-skydio_x2.txt` | Verbatim `cp` from `mujoco_menagerie/skydio_x2/LICENSE` | Upstream license preserved — required by Menagerie redistribution terms. |

## Files Modified

**None.** No existing file in the FM-PCC repo was modified. The migration is purely additive under a new directory.

## Files Deleted

**None.**

---

## Verification

| Check | Result |
|---|---|
| `quadrotor.xml` byte-identical to upstream `skydio_x2/x2.xml` | ✅ `diff -q` returns empty |
| `quadrotor_modified.xml` differs from `quadrotor.xml` by exactly the MJPC patch | ✅ Quat init added on line 34; sensor + keyframe blocks removed at lines 61-70 |
| All 5 deliverable files present | ✅ `ls` confirmed |
| MuJoCo smoke-load (`mj_step` for 100 ticks) | ⏭️ Deferred to cluster — local Docker has no Python runtime per project convention |

---

## What Was Explicitly NOT Done

- No env class (`MjQuadrotor`, `gym_quadrotor_env/`, …).
- No Python port of `quadrotor.cc` residual/transition logic.
- No `config/`, `Slurm_Codes/`, training script, or eval script touched.
- No D3IL existing env (avoiding/aligning/etc.) modified.
- No LLM-generated XML — every byte is `cp`/`patch` output from upstream.

Those belong to future epochs (per `MIGRATION_PLAN.md` §4–§10).

---

## How to Reverse This Epoch

```
rm -rf d3il/environments/d3il/models/mj/robot/quadrotor/
rm -rf /tmp/mujoco_menagerie
```

Repository state is then identical to pre-Epoch-1.
