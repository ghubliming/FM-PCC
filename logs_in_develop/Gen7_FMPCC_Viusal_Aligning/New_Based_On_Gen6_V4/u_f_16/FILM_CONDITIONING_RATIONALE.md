# FiLM Conditioning in Visual DPCC — Why It Is Not an Arbitrary Add-On

**Date**: 2026-05-28  
**Context**: Design note arising from the UF-17 non-visual planning discussion.  
**TL;DR**: FiLM is `apply_conditioning` done at every UNet layer instead of only at the
trajectory boundary. Removing it and pinning image features into the trajectory is
technically possible but architecturally wrong for image inputs.

---

## 1. Original DPCC Conditioning Recap

Original DPCC (`/workspaces/dpcc`) has **no conditioning vector to the UNet**. The
`cond` parameter in `UNet1DTemporalCondModel.forward(x, cond, time, ...)` is accepted
but never used in the body — no FiLM, no cross-attention, no global context injection.

Conditioning is entirely via `apply_conditioning`:

```python
x[:, 0, action_dim:] = current_obs   # pin obs dims of trajectory step 0
```

This pins the full state (robot pos + goal pos) into the trajectory at `t=0`. The UNet
generates future steps consistent with that frozen start via temporal convolutions.
Works cleanly because the obs is low-dimensional and semantically homogeneous with the
trajectory.

---

## 2. Visual DPCC Conditioning

Visual DPCC adds two things over original DPCC:

1. **`apply_conditioning`** — still pins obs dims at step 0, but only 6D robot kinematics
   `[des_c_pos(3) | c_pos(3)]`. Goal/box/target info is NOT in the trajectory.
2. **FiLM** — ResNet encodes `(bp_img, inhand_img)` → 128D; injected into every
   ResNet block of the UNet at every denoising step.

The trajectory stays 9D: `[act(3) | des_c_pos(3) | c_pos(3)]`. Box and target position
are never placed inside the trajectory — they arrive as context through FiLM.

---

## 3. Why Not "Strict DPCC" — Put Images into the Trajectory?

The natural question: can we remove FiLM and instead pin image features into the
trajectory at step 0, exactly as original DPCC does with state?

```
# Hypothetical strict-DPCC approach
trajectory = [act(3) | des_c_pos(3) | c_pos(3) | img_feat(128)]  →  134D
apply_conditioning: x[:, 0, 3:] = [des_c_pos | c_pos | img_feat]
```

Technically executable but the following problems make it unsound:

### 3.1 Images are qualitatively different from state obs

Original DPCC's step-0 pinning works for state because:
- The obs is **semantically homogeneous** with the rest of the trajectory
  (positions, velocities — same space as future trajectory steps)
- The obs is **low-dimensional** (4–20D) — temporal convolutions easily learn to
  spread it across the H=8 horizon

Image features are neither. A 128D encoder output at step 0 represents a holistic
visual scene embedding. Horizon steps 1-7 would carry **noisy image dims** during
denoising — the model would have to learn to both generate the future visual features
(meaningless for a push task) and use the frozen step-0 image correctly.

### 3.2 Image context needs to influence every denoising step at every UNet depth

FiLM injects the 128D visual context at **every ResNet block** at **every spatial
resolution** at **every denoising/ODE step**. The visual context is continuously
available to the model at the deepest computation layers.

Step-0 pinning provides the context only at the **trajectory boundary**, once per
denoising step, in the **input channel** only. The UNet must propagate 128 channels of
visual context across the temporal dimension via convolutions — far weaker than FiLM's
direct injection at every layer.

### 3.3 Trajectory and projector grow needlessly

- Trajectory: 9D → 134D (3+3+3+128). UNet input channels triple.
- Constraint projector: still only cares about dims 6-8 (c_pos). The 128D image dims
  are dead weight in the 134D projection space.
- Training cost increases significantly with no accuracy benefit.

---

## 4. FiLM as the Correct Generalization of `apply_conditioning`

`apply_conditioning` and FiLM are the same concept at different granularities:

| | `apply_conditioning` | FiLM |
|---|---|---|
| When | Once per denoising step, at trajectory boundary | Every ResNet block, every resolution level, every denoising step |
| Where | Input space (trajectory dims) | Latent feature space (after conv) |
| Operation | Hard pin: `x = val` (overwrite) | Soft scale+shift: `x = γ·x + β` |
| Context | Current obs (state dims) | Encoded visual context |
| Suited for | Low-dim state (trajectory-homogeneous) | High-dim sensor data (image, audio) |

FiLM is not an add-on — it is the principled extension of `apply_conditioning` to the
case where the conditioning information is high-dimensional and needs to be available at
every computation depth.

> **Principle**: In original DPCC, the goal is already "inside" the trajectory because
> obs and actions live in the same Cartesian space. In visual DPCC, goal context (where
> is the box, where is the target) lives in pixel space and must be projected into a
> latent representation before it can influence the planning trajectory. FiLM provides
> the injection point for that representation at the right abstraction level.

---

## 5. Implication for Non-Visual Mode

In the non-visual variant (UF-17), images are replaced by the full 20D state:

```
Visual:      ResNet(bp_img, inhand_img)  →  128D FiLM
Non-visual:  MLP(state_20D)              →  128D FiLM  (same injection mechanism)
```

The FiLM injection architecture is retained. Only the source of the 128D context vector
changes. This preserves:
- 9D trajectory (DPCC principle, projector unchanged)
- `apply_conditioning` pinning 6D robot obs at step 0
- FiLM injecting goal/scene context at every UNet layer

The non-visual mode is a clean ablation: one module swapped (ResNet → state MLP),
everything else identical.
