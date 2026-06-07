# UAV FM Obs Format — Deviation from DPCC Avoiding

**Date**: 2026-06-06  
**Status**: Open concern — not resolved before Epoch 4 data collection  
**Related**: [`METHODOLOGY.md`](METHODOLOGY.md) §7, [`Materials/AUDIT.md`](Materials/AUDIT.md) R4

---

## The question

The D3IL avoiding DPCC uses a specific observation structure.  The UAV Epoch 4 dataset
uses a different one.  Was this deviation deliberate, physically justified, or an
oversight?  What are the downstream consequences?

---

## Side-by-side: what DPCC avoiding uses vs. what we built

| | **D3IL avoiding (DPCC)** | **UAV Epoch 4** |
|---|---|---|
| Task space | 2D horizontal plane (arm tip) | 3D space (drone COM) |
| "Where I'm commanded to go" | `des_xy(2)` ✅ in obs | ❌ `p_des(3)` in `targets` only — **not in obs** |
| "Where I am now" | `c_xy(2)` ✅ in obs | `p(3)` ✅ in obs |
| "How fast I'm moving" | ❌ not in obs | `v(3)` ✅ in obs |
| **Obs vector** | **4D** `[des_xy(2), c_xy(2)]` | **6D** `[p(3), v(3)]` |
| **Action** | `Δdes_xy(2)` — delta of desired pos | `Δp_des(3)` — delta of desired pos |
| **FM trajectory tensor** | **6D** `[Δdes_xy(2) ‖ des_xy(2), c_xy(2)]` | **9D** `[Δp_des(3) ‖ p(3), v(3)]` |

Two concrete differences:
1. **`p_des` is MISSING from the UAV obs.** Avoiding always had `des_xy` — the current commanded/desired position — as a first-class observation component.
2. **`v` was ADDED to the UAV obs.** Avoiding has no velocity in its observation.

---

## Deviation 1 — `v` added: justified by physics

**Why `v` is unavoidable for a drone but not an arm:**

The robot arm is a heavily-damped 1st-order system at the dataset frequency.  Its velocity
at any step is approximately determined by `des_c_pos − c_pos` (the position error drives
the IK controller which drives velocity).  Two arm states with the same `(des_xy, c_xy)`
will follow essentially the same future — position alone is nearly Markovian.

The drone is a **2nd-order system**: `F = ma`.  Two drones at the same position `p` can
have completely different velocities — one approaching fast, one hovering, one moving
sideways.  Each will follow a completely different future trajectory.  Without `v`, the FM
model cannot distinguish these cases.  Position alone is **not Markovian** for a UAV.

Including `v` in the UAV obs is therefore unavoidable and correct.  This is not a
deviation from the DPCC principle — it is an adaptation to the physics of the plant.

---

## Deviation 2 — `p_des` missing: a real concern

**What `des_xy` does in the DPCC avoiding FM:**

In the avoiding pipeline, `des_xy` is the *moving commanded setpoint* — the position the
arm's IK controller is currently trying to reach.  It changes every timestep as the
planner issues new commands.  Including `des_xy` in the obs means the FM is conditioned
on **where the controller currently intends to go** — it is an implicit goal signal.

At inference time, the DPCC planner chooses a new `des_xy` at each step.  The FM
generates trajectories conditioned on `{0: [des_xy(0), c_xy(0)]}`.  Because `des_xy`
encodes the planner's intent, the FM can generate directionally appropriate trajectories.

**What the UAV FM sees instead:**

The UAV FM is conditioned on `{0: [p(0), v(0)]}` — current position and velocity only.
There is no signal telling the FM where the drone is supposed to go.

**The consequence at inference time:**

During training, the dataset contains episodes from all homotopy classes:
- Corridor: L (y≈−0.18), C (y≈0.0), R (y≈+0.18)
- Pillars: (L,L,L), (L,R,L), (R,L,R), (R,R,R)

All episodes in a scene start at approximately the same entry position.  The FM condition
`[p(0), v(0)]` is nearly identical across all homotopy classes at t=0 (same entry
position, similar initial velocity).  At inference the FM sees an ambiguous condition that
matches training data from all homotopy classes simultaneously — it will sample from a
**mixture** of homotopies.

In the avoiding case this was prevented by `des_xy`: the planner issues a specific desired
direction at t=0, so the FM generates a trajectory aligned with that intent.  The UAV FM
has no equivalent signal.

**This is not the same as "FM doesn't work without goal conditioning."**  The FM can still
generate trajectories — they will be blended averages of the training data's homotopy
classes.  For a corridor, that blend is "fly down the middle" (average of L/C/R), which
happens to be safe.  For a pillars scene, the blend of (L,L,L) and (R,R,R) is "fly
through y=0" — which also happens to work (the centre passage is valid).  But this is
coincidental geometry, not principled goal conditioning.

---

## Was this discussed in the logs?

**Partially.**  AUDIT.md R4 (line 119) identified the related but narrower gap:

> *"Epoch 2's closure locked 9D `[p, v, a]` for the controller. But the FM dataset uses
> 9D `[act(3) ‖ p(3), v(3)]` — note that `a` (acceleration) is in the controller format
> but not in the FM format. The controller needs acceleration feed-forward, but the FM
> policy will only output position targets."*

R4 flagged the **acceleration** mismatch.  It did not explicitly flag the **missing
`p_des` / goal conditioning** gap — that is a new finding from this analysis.

AUDIT R8 specified the schema with `obs = [p(3), v(3)]` without questioning whether
`p_des` should be included.

EPOCH4_EXECUTION_PLAN.md Decision 3 locked the schema before this concern was raised.

**Nothing in the logs explicitly discussed the goal-conditioning consequence of omitting
`p_des` from the UAV observation.**

---

## Risk assessment

| Risk | Severity | When it matters |
|---|---|---|
| FM samples a mixture of homotopies at inference | 🟡 Medium | Multi-homotopy scenes (corridor L/C/R) — sampled trajectory may not commit to one side |
| FM has no goal signal → cannot be directed to a specific destination | 🟠 High | If the task requires navigating to a specific target (not just "get through the scene") |
| Acceleration feed-forward gap (AUDIT R4) | 🟡 Medium | FM outputs Δp_des; controller derives a_des by finite difference → noisy feed-forward |

**Likely to pass the mini-FM sanity gate (WS-C)** on the empty scene regardless — empty
scene has only one homotopy, no goal ambiguity.  The problem surfaces in multi-homotopy
obstacle scenes.

---

## If we fix it — what needs to be redone across Epochs 1–5?

### What the fix looks like (Option A)

New obs: `[p_des(3), p(3), v(3)]` = **9D**.  FM tensor: **12D** = `[Δp_des(3) ‖ p_des(3), p(3), v(3)]`.

The key code change is one line in `dataset_writer.py:rollout_to_episode()`:

```python
# Current (line 52-53):
obs = np.array([np.concatenate([s['p'], s['v']]) for s in steps], dtype=np.float32)

# Fixed:
obs = np.array([np.concatenate([s['p_des'], s['p'], s['v']]) for s in steps], dtype=np.float32)
```

`s['p_des']` is the **unnoisy** trajectory-function output — the exact position the PID
was commanded to track.  It is already present in every step dict (recorded at
`generator.py:228` as `steps.append({'p': p, 'v': v, 'p_des': np.asarray(p_des)})`).
The noise offset is applied to `targets` afterward — it does NOT corrupt `s['p_des']`.

### Epoch-by-epoch impact

| Epoch | What it produced | Affected by obs change? | Action needed |
|---|---|---|---|
| **1** — UAV model | XML + mesh files | ❌ No | Nothing |
| **2** — Controller validation | Validation of 9D PID setpoint format | ❌ No | Nothing — the 9D/6D controller setpoint is a separate concept from FM obs |
| **3** — Scene environments | 4 scene XMLs, controller in scenes | ❌ No | Nothing |
| **4** — Expert data (1769 pickles) | `obs=(T,6)`, `targets=(T,3)` per episode | ✅ Yes — `obs` field changes from 6D to 9D | **See below** |
| **5 WS-A** — Camera images | `bp-cam` + `track-cam` PNGs per step | ❌ No | Nothing — images rendered from `qpos[:3]` (position), not obs format |
| **5 WS-B** — GIFs | Per-episode GIFs | ❌ No | Nothing — GIF rendering uses `obs[:, :3]` (position), unchanged |
| **5 WS-C** — Mini-FM gate | Not yet run | ✅ Yes — runs against the pickles | Run once with 12D format after pickle update |

**Epochs 1–3 and Epoch 5 WS-A/B require zero changes.** The fix is entirely contained
in Epoch 4 pickles and downstream training (Epoch 6, not yet started).

---

### Epoch 4 — two paths to fix, no physics re-run required

The critical finding: `s['p_des']` (unnoisy) is already recorded in the physics rollout
step dicts.  The 1769 episode pickles were written by `dataset_writer.py` which reads
from those step dicts.  The unnoisy p_des **is not lost** — it is in the rollout but was
never written into the pickle's `obs` field.

**Path 1 — Post-processing existing pickles (fastest, slight impurity)**

Write a one-time script that rewrites each pickle:
```python
# For each existing episode pickle:
new_obs = np.concatenate([episode['targets'], episode['obs']], axis=-1)  # (T, 9)
episode['obs'] = new_obs
pickle.dump(episode, f)
```

This uses `targets` (noisy p_des = unnoisy p_des + 2cm constant offset per episode).
The constant offset means `targets[t] ≠ actual p_des the PID tracked` by ~2 cm — a small
but real impurity.

- **No SLURM re-submission needed**
- **No physics re-simulation** — pure pickle rewrite
- **Impurity**: `targets` is noisy; the PID tracked the unnoisy version.  The 2 cm offset
  is the same order as position tracking error (~3 cm RMS), so it is swamped by the obs noise.

**Path 2 — Re-collect from scratch (clean, costs ~2 SLURM jobs)**

Modify `dataset_writer.py` line 52 as shown above, then re-submit Epoch 4 collection:

```bash
sbatch --array=0-3 Slurm_Codes/sbatch/uav_expert_data/collect.sh all_scenes 500
```

This produces 1769 clean new episodes with exact unnoisy `p_des` in obs.  All existing
episode pickles are discarded.  Camera images (WS-A) would need to be re-collected
against the new pickles (though the GIFs and state positions are identical, so GIFs are
also equivalent — only the pickle metadata changes).

- **Clean**: obs contains exact `p_des` the controller tracked
- **Cost**: ~same as original Epoch 4 (~2 hours wall time across 4 parallel SLURM jobs)
- **WS-A images**: need re-collect if the pipeline keys on episode_id/timestamp (which
  it does via `skip-existing` on the folder name — same episode_id → skipped safely)

**Recommended path**: **Path 1** (post-process) if Epoch 6 eval shows homotopy ambiguity
and speed matters; **Path 2** (re-collect) if starting fresh is acceptable — it is clean
and takes only a few hours.

---

## Potential fixes — decision tree

```
Start: is the goal-conditioning gap causing problems in Epoch 6 eval?
│
├── No (FM commits to a single homotopy without p_des)
│   └── Option D: Accept current format. Document the lucky geometry. ✅ Done.
│
└── Yes (blended / unsafe trajectories, homotopy ambiguity)
    │
    ├── Option A1: Post-process pickles (Path 1 above)
    │   └── +Fast  −Slightly noisy p_des in obs
    │
    ├── Option A2: Re-collect Epoch 4 (Path 2 above)
    │   └── +Clean  −2 SLURM jobs
    │
    └── Option B: Class-conditional FM (homotopy label as discrete input)
        └── +No re-collection  −Architecture change; doesn't solve general goal problem
```

**Recommended evaluation order**: D → A1 → A2.  Try Epoch 6 first (D).  If it fails on
homotopy, apply post-process fix (A1, ~1 hour).  If cleanliness is required, re-collect (A2).

---

## The "9D incl a" naming collision — which 9D are we actually using?

The logs contain two completely different things both called "9D."  They share a number but
have nothing else in common.

| Label | Where | What it means | Includes acceleration? |
|---|---|---|---|
| **Controller 9D** | Epoch 2 closure, AUDIT S6 | PID setpoint `[p_des(3), v_des(3), a_des(3)]` — the 9 numbers fed to the cascaded PID each timestep | **✅ Yes** — `a_des` is feed-forward; Task C 9D gives 0.029 m RMS vs 0.214 m RMS for 6D (7.4× improvement) |
| **FM tensor 9D** | Epoch 4 schema, `dataset_writer.py` | Packed flow-matching trajectory `[Δp_des(3) ‖ p(3), v(3)]` — the 9D column per timestep in the training tensor | **❌ No** — obs is `[p, v]`, action is `Δp_des`; acceleration is absent |

AUDIT R4 made this explicit (line 121–135):

> *"Epoch 2's closure locked 9D `[p, v, a]` for the controller. But RESEARCH §1 proposes 9D
> `[act(3) ‖ p(3), v(3)]` for the FM dataset — `a` is in the controller format but NOT in the
> FM format."*

**What "9D incl a" in past decisions referred to:**  
The Epoch 2 lock ("9D locked") was about the **controller** gaining the acceleration
feed-forward.  The FM dataset format was decided later (EPOCH4_EXECUTION_PLAN Decision 3)
and chose `[Δp_des ‖ p, v]` — without acceleration.  So when a log says "we chose 9D
including a," it means the PID gained `a_des` as input.  The FM tensor 9D does **not**
include acceleration.

**The inference-time consequence:**

At inference the FM model outputs a sequence of `Δp_des` vectors.  The controller converts
them to positions `p_des[t] = p_des[0] + Σ Δp_des`.  To obtain `a_des` for the PID
feed-forward it must **double-difference**:
```
v_des[t] ≈ (p_des[t] - p_des[t-1]) / dt
a_des[t] ≈ (v_des[t] - v_des[t-1]) / dt
```
Finite-differencing amplifies noise by `1/dt²`.  At 33 Hz and FM output noise of ~1 cm
(typical), the inferred acceleration noise is ~`0.01 / (0.03)² ≈ 11 m/s²` — comparable
to 1g.  AUDIT R4 warned explicitly:

> *"If you proceed with 9D and the controller double-differences to get acceleration, the
> same limit-cycle instability from Epoch 2 §3 may reappear during low-speed FM-planned
> segments."*

**The 12D alternative (AUDIT R8 priority action #8):**

If the FM outputs both `Δp_des(3)` and `Δv_des(3)` as the action (total 6D action), the
obs can remain `[p(3), v(3)]` (6D), giving a **12D tensor** `[act(6)=[Δp_des, Δv_des] ‖ obs(6)=[p, v]]`.  The controller receives explicit velocity feed-forward without needing to
differentiate, eliminating the noise amplification.

| Format | FM tensor | Controller gets | Risk |
|---|---|---|---|
| **Current: 9D** | `[Δp_des(3) ‖ p(3), v(3)]` | Must double-diff for `a_des` | Noisy `a_des` → possible limit-cycle |
| **12D** | `[Δp_des(3), Δv_des(3) ‖ p(3), v(3)]` | Explicit `v_des` → single diff for `a_des` | Halved noise; still one diff away |
| **Full 12D with a** | `[Δp_des(3) ‖ p_des(3), p(3), v(3)]` | Explicit `p_des` goal signal → FM can recover `v_des` | Goal conditioned; richest obs |

**Current state**: Epoch 4 dataset is 9D (no acceleration).  This was a deliberate choice
at Decision 3 to keep the format simple for the first training run.  The instability risk
is real but only manifests at **sustained low speed** — which rarely occurs in the Epoch 4
training trajectories (traverse_line and weave have non-zero velocity throughout).

---

## Can the Gen9 FiLM visual avoiding pipeline be reused for UAV?

**Short answer**: the FiLM *architecture* is reusable; the observation and trajectory
dimensions must change.

### What Gen9 visual avoiding uses

Gen9 `fm_visual_avoiding` trains a `VisualUNet` (`fm_visual_avoiding/models/visual_unet.py`)
that combines:

1. **Image encoder** — `MultiImageObsEncoder` with a single ResNet, `agentview_image` only
   (96×96 RGB) → 64D image latent.  Single camera because only the bp-cam sees the obstacle
   field; wrist cam adds nothing for avoiding.

2. **FiLM conditioning** — the 64D image latent is projected to `dim`-dimensional scale +
   shift vectors via `nn.Linear(64, dim)` inside each `ResidualTemporalBlock` of the UNet.
   Applied as `h = h * scale + shift` channel-wise.  This lets the image modulate the
   trajectory denoising at every temporal resolution.

3. **1D temporal UNet** — `UNet1DTemporalCondModel` takes a **6D trajectory tensor**
   `[Δdes_xy(2) ‖ des_xy(2), c_xy(2)]` (2D action + 4D obs, horizon H=8) and denoises it.
   `transition_dim = config.action_dim + obs_dim` — set by config, not hardcoded.

4. **State obs** — 4D `[des_xy(2), c_xy(2)]`: goal direction + current arm position.

### What the UAV pipeline would need to change

| Component | Gen9 avoiding | UAV adaptation | Effort |
|---|---|---|---|
| Camera | `agentview` (overhead arena view) | `bp-cam` (overhead drone view) — already collected in E5 WS-A | ✅ Same concept, already done |
| Image encoder | ResNet → 64D | Same ResNet → 64D | ✅ Zero change |
| FiLM mechanism | `Linear(64, dim)` scale/shift in UNet blocks | Identical | ✅ Zero change |
| Obs dim | 4D `[des_xy, c_xy]` | 6D `[p, v]` or 9D `[p_des, p, v]` | Config change: `obs_dim = 6` or `9` |
| Action dim | 2D `Δdes_xy` | 3D `Δp_des` | Config change: `action_dim = 3` |
| Trajectory tensor | 6D | 9D (or 12D) | Config change: `transition_dim = 9` |
| Scene geometry | 2D arm in obstacle arena | 3D drone in corridor/pillars | Training data change only |

The `VisualUNet` constructor already reads `obs_dim` from config (`line 76: obs_dim = getattr(config, 'obs_dim', 4)`) and computes `transition_dim = config.action_dim + obs_dim`.  The 6D default is just the avoiding default — widening to 9D is a single config edit.

### What does NOT transfer

- **DPCC avoiding-specific geometry**: The Gen9 FiLM model was trained on a 2D arm-workspace
  obstacle field.  Its weights encode arm-task priors — not UAV dynamics or 3D flight.
  Training from scratch on Epoch 4 UAV data is required; there is no useful weight transfer.
- **`des_xy` goal conditioning**: Gen9 has explicit goal direction in obs.  UAV current format
  does not (Deviation 2 above).  The FiLM architecture can *accommodate* goal conditioning
  once `p_des` is added to obs — but the gap persists regardless of architecture.

### Reusability verdict

| Aspect | Reusable? | Notes |
|---|---|---|
| FiLM image conditioning mechanism | ✅ Yes | `Linear(64, dim)` scale/shift — architecture-level |
| ResNet image encoder (random init) | ✅ Yes | Same topology, retrain on UAV views |
| UNet1D temporal denoiser | ✅ Yes | Widen `transition_dim` in config |
| Gen9 trained weights | ❌ No | Different task, 2D vs 3D, different obs format |
| Goal conditioning pattern | ⚠️ Partial | Architecture supports it; obs format needs `p_des` added |

**Conclusion**: drop the Gen9 model weights; keep the Gen9 `VisualUNet` + `UNet1DTemporalCondModel` architecture; update `action_dim=3`, `obs_dim=6` (or 9), point at UAV bp-cam images.  The FiLM visual avoiding is a viable UAV architecture once obs dimensions are adapted.

---

## Summary

| Deviation | Justified? | Consequence | Discussed in logs? |
|---|---|---|---|
| `v` added to obs | ✅ Yes — 2nd-order UAV dynamics require it | None — physically necessary | Implicit in 9D format choice |
| `p_des` absent from obs | ⚠️ Not explicitly justified | Goal conditioning gap; homotopy ambiguity at inference | **Partially** — AUDIT R4 caught acceleration gap, not the goal-signal gap specifically |

**If we fix it**: Epochs 1–3 untouched.  Epoch 5 WS-A/B untouched (images and GIFs
depend on position only).  Epoch 4 pickles need a one-time rewrite or re-collection —
no physics re-simulation required.  Epoch 6 FM tensor widens from 9D → 12D (config
change only).  Total cost: low.

---

## Cross-references

| Document | Content |
|---|---|
| [`Materials/AUDIT.md`](Materials/AUDIT.md) R4 | Acceleration feed-forward gap (controller vs. FM format) |
| [`Materials/AUDIT.md`](Materials/AUDIT.md) R3 | Mono-modality / homotopy coverage concern |
| [`METHODOLOGY.md`](METHODOLOGY.md) §7, §8 | Action convention and schema decisions |
| [`../Epoch2_UAV_mujoco_run/METHODOLOGY.md`](../Epoch2_UAV_mujoco_run/METHODOLOGY.md) | 9D controller setpoint vs. 9D FM tensor naming collision |
| [`../Epoch5_visual_and_validation/METHODOLOGY.md`](../Epoch5_visual_and_validation/METHODOLOGY.md) | WS-A/B render from position only — unaffected by obs change |
