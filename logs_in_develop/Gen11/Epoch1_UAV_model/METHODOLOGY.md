# Gen11 Epoch 1 — UAV Model Migration: Methodology

**Date**: 2026-06-06  
**Status**: ✅ Complete  
**Maximum fix index**: none (pure file migration, no code bugs)  
**CHANGELOG**: [`CHANGELOG.md`](CHANGELOG.md)

---

## What this epoch does

Brings the Skydio X2 quadrotor physics model into the FM-PCC repo so all subsequent
epochs have a stable, flyable MuJoCo asset to build on.  No Python code, no env class,
no controller — only model files.

---

## Why these two XML variants exist

The X2 ships from two upstream repos with different purposes:

| File | Origin | Purpose |
|---|---|---|
| `quadrotor.xml` | `mujoco_menagerie/skydio_x2/x2.xml` — verbatim copy | The base physics model: X2 body, 4 rotors, mesh references, actuators |
| `quadrotor_modified.xml` | `quadrotor.xml` + MJPC's `quadrotor.xml.patch` | MJPC-task-ready variant: adds explicit quaternion initialisation (`quat="0 0 0 1"`), removes MJPC-only sensor block and hover keyframe |

**Real-world meaning of the patch**:
- `quat="0 0 0 1"` sets the drone's initial orientation to "level, nose forward".  Without
  it the drone spawns in whatever default MuJoCo assigns, which may not be upright.
- Removing the MJPC sensor block and keyframe strips MJPC planner internals that FM-PCC
  has no use for.  Keeping dead tags would confuse future readers.

`quadrotor_modified.xml` is the file all subsequent epochs include.

---

## Why "copy, don't generate"

Every byte in these files is `cp` or `patch` output from a known upstream source (`mujoco_menagerie`, `mujoco_mpc`).  This is explicit policy: LLM-generated XML risks subtle physics errors (wrong inertia tensors, wrong actuator gear ratios, wrong mesh scales) that would corrupt every downstream training run silently.  The upstream Menagerie X2 is already validated by Google DeepMind.

---

## What was NOT done

No Python env class, no gym wrapper, no config, no SLURM script, no existing file modified.
The full epoch is reversible with `rm -rf d3il/environments/.../quadrotor/`.

---

## Cross-references

| Document | Content |
|---|---|
| [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) | Full plan: source inventory, target tree, per-file disposition |
| [`../Epoch2_UAV_mujoco_run/METHODOLOGY.md`](../Epoch2_UAV_mujoco_run/METHODOLOGY.md) | First use of this model in a physics sim loop |
