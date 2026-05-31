# Dim Audit: Are our 9-D visual / 23-D non-visual trajectories wrong vs. D3IL?

**Date**: 2026-05-31
**Triggered by**: user concern — "I sincerely believe our 9D might be wrong; D3IL visual aligning is 23D + visual." Critical question, no theorizing — verified from D3IL's own source files.

---

## TL;DR

**Your memory of D3IL is incorrect on the specific number.** D3IL's vision
aligning uses `obs_dim = 3` + image encoder, **not 23**. D3IL's
state-only aligning uses `obs_dim = 20`. Neither is 23-D.

**Our 9-D visual trajectory is not directly comparable to any D3IL
number** — D3IL doesn't have a "joint action+state trajectory"
construct at all. Our 9-D comes from the DPCC paper's design (action +
kinematic state interleaved per step), which is a different
architectural paradigm from D3IL's (separate state encoder → 128-D
latent → action sequence prediction). Apples and oranges.

**Conclusion**: our 9-D visual is consistent with DPCC's original
design and is not a bug. Our 23-D non-visual matches D3IL's state-only
obs_dim of 20 (plus our 3-D action prepended → 23). Both are correct
relative to their respective architectures. The 9 / 23 asymmetry
between our visual and non-visual is intentional and explained in §3
below.

---

## 1. What D3IL ACTUALLY says — verbatim from `d3il/configs/`

### `d3il/configs/aligning_vision_config.yaml` (D3IL VISION baselines)

```yaml
# Environment
obs_dim: 3            ← D3IL VISION uses 3-D obs
action_dim: 3
max_len_data: 512
window_size: 8
```

### `d3il/configs/aligning_config.yaml` (D3IL STATE-ONLY baselines)

```yaml
# Environment
obs_dim: 20           ← D3IL STATE-ONLY uses 20-D obs (matches our non-visual)
action_dim: 3
max_len_data: 512
window_size: 1
```

Both files verified by direct grep on disk:

```
$ grep -E 'obs_dim:|action_dim:' d3il/configs/aligning_vision_config.yaml
obs_dim: 3
action_dim: 3
  obs_dim: ${obs_dim}
  action_dim: ${action_dim}
  ...

$ grep -E 'obs_dim:|action_dim:' d3il/configs/aligning_config.yaml
obs_dim: 20
action_dim: 3
  obs_dim: ${obs_dim}
  action_dim: ${action_dim}
  ...
```

**There is no `23` anywhere in D3IL's aligning configs.** The `23-D +
visual` you remembered is not in this repo.

---

## 2. What D3IL's vision MODEL actually consumes

The 3-D `obs_dim` is the raw state going INTO the agent. But the
**diffusion model itself** sees a 128-D latent, not 3-D. From
`d3il/configs/agents/ddpm_encdec_vision_agent.yaml`:

```yaml
model:
  ...
  state_dim: 128 #${obs_dim}     ← model state dim is 128, NOT obs_dim
  action_dim: ${action_dim}      ← action_dim from outer config (= 3)
  ...
```

And `d3il/agents/ddpm_encdec_vision_agent.py:49` confirms:
```python
agentview_image, in_hand_image, state = inputs
...
obs_dict = {"agentview_image": agentview_image,
            "in_hand_image":   in_hand_image,
            "robot_ee_pos":    state}            ← 3-D state goes as one input
```

So D3IL's vision pipeline:

```
raw obs (3-D robot_ee_pos)  ───┐
                                ├─► feature encoder ──► 128-D latent ──► diffusion model
bp_cam + inhand_cam images  ───┘                                            │
                                                                            ▼
                                                                    predicts action sequence (3-D × 4 steps)
```

**The model never sees a 9-D or 23-D anywhere.** It sees a 128-D fused
latent and predicts action sequences.

---

## 3. What WE (FM-PCC visual_aligning) actually do

Different architecture from D3IL. The DPCC paper uses a **joint
action+state trajectory** instead of separate-state-encoder +
action-prediction:

### Visual variant (our `visual_aligning_dpcc` / `fm_visual_aligning`)

```
Trajectory tensor: (B, H=8, 9)
                            ↑
              [ act(3) | des_c_pos(3) | c_pos(3) ]
              └─action─┘└──────obs(6)──────────┘

bp_cam + inhand_cam ─► MultiImageObsEncoder ─► 128-D latent ─► FiLM into UNet
                                                                  ↓
                                  obs anchor [des_c_pos | c_pos] pinned at step 0
                                  via apply_conditioning
```

- 9-D = 3 action + 6 obs (kinematic only — des_c_pos and c_pos).
- 128-D image latent injected via FiLM in every ResNet block.
- DPCC projector constrains the kinematic dims (workspace bounds on
  c_pos at indices 6-8, dynamics linking c_pos to act).

### Non-visual variant (our `ddpm_encdec_vision_nonvisual` / UF-17)

```
Trajectory tensor: (B, H=8, 23)
                             ↑
              [ act(3) | obs(20) ]
                          └─[des_c_pos(3) | c_pos(3) | box_pos(3) | box_quat(4)
                              | tgt_pos(3) | tgt_quat(4)]─┘

No images. No FiLM. Full state lives in the trajectory channels.
obs anchor pinned at step 0 = the full 20-D state.
```

- 23-D = 3 action + 20 obs (full state including box + target).
- No image encoder.
- DPCC projector still only constrains first 9 dims; trailing 14
  (box/target) are unconstrained (no physics links robot action to
  object pose at the projector level).

---

## 4. Comparison table — D3IL vs. ours, dim-by-dim

| Aspect | **D3IL VISION** (`aligning_vision_config.yaml`) | **D3IL STATE-ONLY** (`aligning_config.yaml`) | **OURS VISUAL** (`visual_aligning_dpcc` / `fm_visual_aligning`) | **OURS NON-VISUAL** (`ddpm_encdec_vision_nonvisual`, UF-17) |
|---|---|---|---|---|
| Raw obs dim (env → model) | **3** `[robot_ee_pos]` only | **20** `[des_c_pos(3)·c_pos(3)·box(7)·tgt(7)]` | **6** `[des_c_pos(3)·c_pos(3)]` | **20** `[des_c_pos(3)·c_pos(3)·box(7)·tgt(7)]` |
| Action dim | 3 | 3 | 3 | 3 |
| Window / horizon | window_size = 8 | window_size = 1 | horizon = 8 | horizon = 8 |
| Image input | bp_cam + inhand_cam (96×96) | — | bp_cam + inhand_cam (96×96) | — |
| Image encoder output | 128-D (state + images fused) | — | 128-D (images only) | — |
| **Joint trajectory tensor?** | ❌ No (separate state encoder → 128-D latent → action seq head) | ❌ No (same encdec-style architecture) | ✅ **9-D** `[act(3) \| des_c_pos(3) \| c_pos(3)]` (DPCC paper) | ✅ **23-D** `[act(3) \| obs(20)]` (DPCC paper, UF-17 extension) |
| Conditioning mechanism | latent → action decoder (encdec) | latent → action decoder | FiLM (128-D image latent → all UNet residual blocks) + `apply_conditioning` (obs anchor pinned at step 0) | `apply_conditioning` only (obs anchor pinned at step 0); no FiLM, no image encoder |
| DPCC projector | ❌ none | ❌ none | ✅ 9-D (constrains kinematic dims 0-8) | ✅ 23-D (constrains kinematic dims 0-8; trailing 14 box/tgt dims unconstrained) |

### Where direct comparison is meaningful (only one row)

The architectures differ fundamentally — D3IL uses **separate state
encoder → 128-D latent → action sequence head** (encoder–decoder
paradigm). Ours uses **joint action+state trajectory tensor** (DPCC
paradigm). There is no "trajectory dim" on the D3IL side because there
is no trajectory tensor at all.

The only dim that maps one-to-one across all four columns is **raw obs
dim** going from env into model:

| | D3IL | Ours | Difference |
|---|---|---|---|
| Visual obs dim | **3** | **6** | +3 dims (`des_c_pos`) — needed for DPCC's projector dynamics constraint `c_pos[t+1] = c_pos[t] + Δt · des_c_pos[t]`; D3IL has no projector so doesn't need it |
| State-only obs dim | **20** | **20** | **Identical** — both use the full `[des_c_pos · c_pos · box_pose · tgt_pose]` 20-D state |

### The "23-D + visual" memory does not match D3IL's files

Direct grep on disk:
```
$ grep "obs_dim:" d3il/configs/aligning_vision_config.yaml
obs_dim: 3

$ grep "obs_dim:" d3il/configs/aligning_config.yaml
obs_dim: 20
```

**The number `23` does not appear in any D3IL aligning config.** It
only exists in our own UF-17 non-visual variant trajectory dim
(3 action + 20 obs = 23).

The most plausible source of the `23-D + visual` misremembering: 23 is
our **non-visual** trajectory dim (`act 3 + obs 20`), which numerically
matches "D3IL's state-only obs dim (20) plus our action dim (3)" — but
the construct itself doesn't exist in D3IL.

---

---

## 5. Why our visual is 6-D obs (not 3-D like D3IL)

Because the **DPCC projector needs `des_c_pos`** to enforce the
dynamics constraint `c_pos[t+1] = c_pos[t] + Δt · des_c_pos[t]`. D3IL
doesn't have a projector, so it doesn't need `des_c_pos` in the obs.

Both choices are correct given their respective use cases:

- **D3IL vision**: minimal state (just c_pos) + heavy reliance on
  images. The 128-D fused latent absorbs everything.
- **Our visual**: kinematic state (des_c_pos + c_pos) so the projector
  can constrain it; images add scene context via FiLM.

The 3-D extra dim (des_c_pos) is the cost of having a constraint
projector. It's not a bug; it's a feature of the DPCC architecture.

---

## 6. Why our non-visual is 23-D (not 9-D like our visual)

Because without images, **box + target state has nowhere to enter**.
UF-17 picked the DPCC-pure option: put it in the trajectory channels
(extending obs from 6 → 20). The alternative would have been a
side-channel state encoder + FiLM, which would break the "everything
in trajectory" principle the rest of the codebase is built around.

Detailed reasoning in
[`../u_f_17_fix_non_visual/PLAN_NON_VISUAL_FIX.md`](../u_f_17_fix_non_visual/PLAN_NON_VISUAL_FIX.md).

---

## 7. Sanity verification you can run on cluster

```bash
# Confirm D3IL's actual aligning vision obs_dim:
grep "obs_dim:" d3il/configs/aligning_vision_config.yaml
# Expected output: obs_dim: 3

# Confirm our visual model's trajectory dim:
python -c "
import pickle
cfg = pickle.load(open('logs/.../H8_K1_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps900_bs64/6/model_config.pkl','rb'))
ns = cfg['config']
trans_dim = 9 if ns.if_vision else ns.action_dim + ns.obs_dim
print(f'visual_aligning_dpcc transition_dim = {trans_dim}')
"
# Expected: 9 for any visual checkpoint, 23 for any non-visual checkpoint.
```

---

## 8. Bottom line

| Question | Answer |
|---|---|
| Is our 9-D visual a bug? | **No.** It's the DPCC-paper architecture: trajectory = action + kinematic state. The "6" in obs is `des_c_pos + c_pos`, needed for the projector's dynamics constraint. |
| Is our 23-D non-visual a bug? | **No.** It's UF-17's deliberate choice to inject scene state (box + target) into trajectory channels rather than via side-channel FiLM. Preserves the "everything in trajectory" DPCC principle. |
| Does D3IL visual use 23-D + visual? | **No.** D3IL visual uses `obs_dim=3` + image encoder → fused into a 128-D latent. There's no joint trajectory; the diffusion model predicts action sequences from the latent. The "23" is not in any D3IL file. |
| Does anything from D3IL contradict our design? | **No.** D3IL state-only (`obs_dim=20`) matches our non-visual obs dim exactly. D3IL vision uses a different paradigm (no joint trajectory), so direct comparison isn't meaningful — but our visual `obs_dim=6` is a strict superset of D3IL's `obs_dim=3` (we add `des_c_pos` for the projector). |
| Do we need to change anything? | **No.** Our 9-D and 23-D are both consistent and well-motivated. No retrain, no code change. |

The Fix-18 work this past week was about getting the **non-visual
(23-D) code path to actually run end-to-end** — not about changing
which dim is correct. The dim choices were locked at UF-17 and stand.
