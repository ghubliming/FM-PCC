# READ ME — Trajectory Dim Invariants for Gen6V4 / Gen7 DPCC Stack

**Why this file exists**: a multi-hour debugging session on 2026-05-31
was caused by confusion over which trajectory dim a given checkpoint
uses. This file lives at the Gen6V4 root so the rules are visible
before any future change is attempted.

---

## The two dim regimes

| Mode | Trajectory dim | Composition | Where it's set |
|---|---|---|---|
| **Visual** | **9-D** | `[act(3) ‖ des_c_pos(3) ‖ c_pos(3)]` | `VisualUNet.TRANSITION_DIM = 9` (hardcoded in the visual branch of `models/visual_unet.py`) |
| **Non-visual** | **23-D** | `[act(3) ‖ obs(20)]` where obs = `[des_c_pos(3) ‖ c_pos(3) ‖ box_pos(3) ‖ box_quat(4) ‖ tgt_pos(3) ‖ tgt_quat(4)]` | Computed: `transition_dim = action_dim + obs_dim = 3 + 20 = 23`. The `obs_dim = 20` part is enforced by Fix-18.1's train-script override (`args.obs_dim = 20` for non-visual) |

**Do not break these.** They are load-bearing for:
- The U-Net first-conv input channel count.
- The state-dict tensor shapes saved during training.
- The DPCC projector's bound matrix construction.
- The `apply_conditioning` obs anchor pinning.

---

## Authoritative way to identify a checkpoint's dim

**Use the state-dict tensor shape. NOT the saved `model_config.pkl`.**

```bash
python -c "
import torch, glob, sys
ckpt = sorted(glob.glob('<your_checkpoint_dir>/state_*.pt'))[-1]
sd = torch.load(ckpt, map_location='cpu')['model']
k = next(k for k in sd if 'downs.0.0.blocks.0.block.0.weight' in k)
n = sd[k].shape[1]
print(f'first-conv input channels: {n}  → {n}-D  '
      f'({\"visual\" if n == 9 else \"non-visual\" if n == 23 else \"UNKNOWN\"})')
"
```

`model_config.pkl` was historically unreliable due to the
STALE_CONFIG bug (`utils.Config.save()` skipped overwriting if file
existed). Fix at `b125365` makes future saves always overwrite, but
**any checkpoint trained before that commit may have a stale config
that contradicts the weights**.

If `model_config.pkl` says `obs_dim=6` but state_dict says shape
`[32, 23, 5]`, the weights are 23-D and the config is lying. Trust
the weights.

---

## Why the asymmetry exists

The 9-D / 23-D split is NOT arbitrary. It reflects the architectural
decision about where scene information (box + target pose) enters the
model:

- **Visual**: scene info enters via images → ResNet image encoder → 128-D
  latent → FiLM into UNet. Trajectory only needs kinematic state.
- **Non-visual**: no images, so scene info has to enter via trajectory
  channels (DPCC-pure principle: "everything in trajectory"). UF-17
  picked this over the alternative of side-channel state encoding.

Reverting non-visual to 9-D is **not a valid simplification** — it
would either drop scene info entirely (model can't solve aligning) or
require a brand-new side-channel state encoder that doesn't currently
exist in the codebase.

---

## Comparison with D3IL (for context)

D3IL doesn't have a "joint action+state trajectory" construct at all —
it uses separate state encoder → 128-D latent → action sequence head.
So D3IL has no number directly comparable to our 9 or 23. The only
apples-to-apples comparison is **raw obs dim**:

| | D3IL VISION | D3IL STATE-ONLY | Ours VISUAL | Ours NON-VISUAL |
|---|---|---|---|---|
| Raw obs dim | 3 | 20 | 6 | 20 |

D3IL vision uses obs_dim=3 (just c_pos). We use 6 (`des_c_pos + c_pos`)
because the DPCC projector's dynamics constraint
`c_pos[t+1] = c_pos[t] + Δt · des_c_pos[t]` needs both. D3IL has no
projector so doesn't need `des_c_pos`.

Full audit: `../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/DIM_AUDIT_VS_D3IL.md`.

---

## Fix-18 history (in order)

| Fix | Commit | Touches | What it solved |
|---|---|---|---|
| 18.1 | `606ad1e` | train scripts | Non-visual training crashed at first conv; now overrides `args.obs_dim = 20` so model builds at 23-D |
| 18.2 | `606ad1e` | eval scripts | Eval `_traj_dim` now derived from saved normalizer dims (defensive vs. CLI/checkpoint disagreement) |
| STALE_CONFIG | `b125365` | utils/config.py | `model_config.pkl` was being skipped on re-save → eval loaded wrong-shape model |
| 18.3 | `761b2ef` | eval scripts | UF-13 record-mode flip now gated on normalizer dim; prevents `(1,6) vs (20,)` crash for genuine 23-D |
| 18.4 | `20a1895` | eval scripts | First-replan diagnostic in `predict()` now branch-aware; prevents `UnboundLocalError` on non-visual path |
| 18.5 | `a361854` | eval scripts | `setup_dpcc_projector` slices normalizer to `trajectory_dim - action_dim`; prevents `(23,) vs (9,)` crash in projector for non-visual variants beyond `diffuser` |
| 18.6 | (Fix-18.6 commit) | aligning_sim.py + eval scripts | Added `Policy.record_sim_frame(env)` env-render hook so genuine 23-D non-visual eval produces GIFs |

Visual path is unchanged at every fix — verified by code inspection.
All fixes are gated to non-visual branches OR resolve to no-op on
visual via dim-derived conditions.

---

## Common pitfalls (avoid these)

1. **Trusting `model_config.pkl` over `state_*.pt` for checkpoint identity.**
   STALE_CONFIG makes the config unreliable for any pre-Fix-18 checkpoint.

2. **Assuming `_VFalse_` in a checkpoint path means non-visual.**
   The flag was cosmetic before Fix-18.1 enforced 23-D training. A
   `_VFalse_` checkpoint trained pre-Fix-18.1 could be 9-D visual under
   the hood (image encoder present, flag misleading). Always verify via
   state_dict shape.

3. **CLI-overriding `if_vision`.** Don't. Use the variant block
   (`visual_aligning_dpcc` for visual, `ddpm_encdec_vision_nonvisual`
   for non-visual). CLI overrides bypass Fix-18.1's enforcement of
   `obs_dim`.

4. **Forgetting that visual DPCC and non-visual DPCC are TWO different
   model architectures**, not just one model with a flag. Visual has an
   image encoder + FiLM; non-visual doesn't.

---

## When to update this file

- A new Fix-19+ that changes how dims are derived.
- A new mode beyond visual/non-visual (e.g., a hypothetical
  vision-state-fused mode with side-channel encoder).
- Architectural refactor of `VisualUNet` that changes the
  `TRANSITION_DIM = 9` hardcoded constant.

If you change anything that affects either the 9 or the 23 in the
table above, **update this file in the same commit**.
