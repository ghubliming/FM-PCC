# Problem — visual-aligning silently dropped the velocity `bounds` when it added a geo box

**TL;DR:** DPCC-avoiding used the `bounds` family for **one** job: a *velocity/action* limit
(keep sampled actions inside the dataset's normalized range). Visual-aligning **repurposed**
the same `bounds` key for a **different** job: a *Cartesian workspace box* on position. These
two jobs are **orthogonal** — but they were treated as mutually exclusive ("replaced"), so
visual-aligning quietly **lost the action/velocity guard**. Nothing now bounds the action
dims except the dynamics Euler link. This is latent risk (out-of-distribution action
magnitudes → poor tracking / instability), sharpest on the lagging UAV plant.

Full binding analysis this builds on:
`../../Gen11/Epoch9_PCC_Constraints/Plan/STUDY_DPCC_constraint_dim_binding.md`.

---

## What DPCC-avoiding did (two roles, one used)

From the study MD — avoiding `bounds` bind to the **action** (velocity), not position:

```yaml
# config/projection_eval.yaml
bounds: {   # need to be within the limits of the dataset due to the normalization
  'avoiding-d3il': [
    {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
    {'type': 'upper', 'dimensions': ['vx', 'vy'], 'values': [0.01, 0.01]},
    ... ] }
```
> **bounds → in avoiding, on `['vx','vy']` = the action (velocity), not position (dims 0,1).**
> It is a guard that keeps sampled actions inside the trained/normalized range, not a spatial
> safety box.

So avoiding had a **velocity/action magnitude bound**, and *no* Cartesian workspace box
(obstacles + halfspace did the spatial work).

## What visual-aligning did (swapped role, didn't keep both)

`config/visual_aligning_eval.yaml` explicitly retires the velocity bounds and reuses the key
for a position box:
```yaml
#     {'type': 'lower', 'dimensions': ['vx', 'vy'], 'values': [-0.01, 0]},
# } # NOT NEEDED: Replaced by 'workspace_bounds' below, which enforces safe physical Cartesian ranges.
...
    workspace_bounds:
      lb: [0.30, -0.35, 0.05]     # Cartesian box on ACTUAL position p (dims 6,7,8)
      ub: [0.70,  0.35, 0.40]
```
From the faithfulness table in the study MD:
> **bounds** | avoiding: **action** velocity limit (dims 0,1) | visual-aligning:
> **`workspace_bounds`** Cartesian box on actual `p` (6,7,8) | ⚠️ **repurposed**

---

## The actual problem

The word "Replaced" is the bug in reasoning. The two uses are **not substitutes**:

| Role | Binds to | Purpose | avoiding | visual-aligning |
|---|---|---|---|---|
| velocity/action bound | action (0,1[,2]) | keep sampled Δ/action inside trained-normalized range | ✅ present | ❌ **lost** |
| workspace box | actual `p` (6,7,8) | physical Cartesian safety envelope | ❌ absent | ✅ added |

After the swap, **no constraint touches the action dims for magnitude** anymore. The action
is now shaped only *indirectly*, through the `deriv` dynamics rows (which enforce Euler
*consistency* `p[t+1]=p[t]+dt·a`, **not** a magnitude cap) and through geometry on `p`. So a
projected plan can command an out-of-distribution large step `a` as long as the resulting
`p` still lands inside the obstacle/halfspace/box set — exactly the case the avoiding
velocity bound existed to prevent ("within the limits of the dataset due to the
normalization").

**Why it usually hasn't bitten yet:** on a perfect-tracking arm/point, the FM prior already
emits in-range actions and geometry-on-`p` keeps steps small, so the missing action bound is
rarely the binding constraint. **Why it matters for E9/UAV:** the drone's `p` lags `p_des`,
tracking is imperfect, and an unbounded commanded Δ`p_des` is precisely what destabilizes the
second-order plant (cf. the E6 multi-mode explosion). The guard that avoiding had for free is
missing right where it would help most.

---

## Fix direction (orthogonal, both should coexist)

The engine already supports lb/ub on **any** dims (`formulate_bounds_constraints` +
`SafetyConstraints`), so this is a config/wiring gap, not a solver gap. Keep **both**:
1. **`workspace_bounds`** — Cartesian box on actual `p` (the added, wanted behavior). Keep.
2. **Re-add a velocity/action bound** — lb/ub on the action dims (0,1[,2]), sized from the
   dataset's normalized action range, restoring avoiding's dataset-normalization guard. On
   the UAV this is a Δ`p_des` (per-step step-size) cap.

Represent them as two entries (or two `bounds`-family sub-lists on different dims) rather
than one, so the naming stops implying they are alternatives. Recommended for E9: carry an
action-magnitude bound alongside the per-scene `workspace_bounds` in every constrained scene.

*(Note: this is a problem statement + direction, not an applied patch. Sizing the action
bound needs the dataset's normalized action range / measured step-size distribution.)*

---

>[!warning] notice
>For both Gen7 and Gen6v4