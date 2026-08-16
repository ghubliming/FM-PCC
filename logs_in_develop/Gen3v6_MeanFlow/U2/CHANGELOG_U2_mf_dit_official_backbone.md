# Gen3v6 U2 — add the **official-MeanFlow DiT** as a third backbone (`imf_backbone='mf_dit'`)

## Motivation

Gen3v6 runs the **MeanFlow objective**, but its two existing backbones are neither from the
MeanFlow paper:

| `imf_backbone` | class | conditioning | positions | MLP | u/v heads | origin |
|---|---|---|---|---|---|---|
| `unet` | `Flow_matcher_U_Net_v2` | additive `time_mlp(t)+h_mlp(h)` | — (conv) | — | dual-head opt. | DPCC U-Net |
| `dit` | `MFDiTTrajectory` | **in-context prefix tokens** | **RoPE** | **SwiGLU** (8/3) | shared trunk → head blocks | **iMF** (`imeanflow`) |
| **`mf_dit`** (NEW) | **`MFDiTOfficialTrajectory`** | **adaLN-zero** (`t+r+w`) | **abs. sin-cos** | **GELU-tanh** (4.0) | twin FinalLayers on one trunk | **MeanFlow** (`aux_repo/MeanFlow`) |

So the previous "faithful" DiT arm was actually the **iMF** architecture running the MeanFlow
objective — an architecture/objective mismatch. U2 adds the network from the MeanFlow reference
repo itself (`/workspaces/aux_repo/MeanFlow/models/dit.py`, class `MFDiT`) so the architecture
finally matches the objective, and so the project can run a clean **three-way backbone A/B**
(DPCC-UNet vs iMF-DiT vs MeanFlow-DiT) under one fixed objective.

## What was added

**New file `flow_matcher_v3_meanflow/models/mf_dit_official_trajectory.py`** — a 100%-faithful
port of `MFDiT`, class `MFDiTOfficialTrajectory`. Every **learned** component is verbatim from
the reference:

- `TimestepEmbedder` (freq=256, `scale=1000`, cos-then-sin order) for **t, r, w**; conditioning
  `c = t_emb + r_emb + w_emb`.
- **adaLN-zero `DiTBlock`** (6·d modulation, zeroed at init ⇒ identity start).
- QK-RMSNorm softmax `Attention`; GELU-tanh `Mlp` at `mlp_ratio=4.0`.
- Learned **absolute** pos-embed, sin-cos initialised (`requires_grad=True`).
- Twin `final_layer_u` / `final_layer_v` reading the **same** trunk output (MFDiT has **no**
  shared-trunk/aux-head split — so `dit_aux_head_depth` is N/A for this arm).
- The exact `initialize_weights` scheme (xavier trunk, 0.02 timestep-MLPs, zeroed adaLN, zeroed
  output layers).

### The only deviations — all forced by 1-D trajectory data, none of them change the math

| reference (`MFDiT`, images) | port (`mf_dit`, trajectories) | why |
|---|---|---|
| `PatchEmbed` (Conv2d) | `TrajPatchEmbedder` (Linear over `patch_size` steps) | data is `[B,H,D]`, not `[N,C,H,W]` |
| `get_2d_sincos_pos_embed` | `get_1d_sincos_pos_embed` | the sequence is 1-D |
| `unpatchify` (image) | `[B,H,D]` reshape | ″ |
| timm `Mlp` / `PatchEmbed` | inline equivalents | avoid a timm dependency on the cluster |
| `nn.RMSNorm` | local `RMSNorm` (identical math) | avoid a hard dep on a recent torch — same choice the iMF port made |
| class-label `y_embedder` | **off** (`num_classes=None`) | trajectory conditioning is the pinned obs applied to x externally (unconditional MFDiT path) |

### JVP-safety (no new risk)

The MeanFlow objective differentiates the backbone with `torch.func.jvp`. This is **safe by
construction**: the MeanFlow repo itself differentiates *this exact `MFDiT`* with
`torch.autograd.functional.jvp` (`meanflow.py:175`, `use_flash_attention=False`). There is **no
RoPE complex-bitcast hazard** here (this DiT has no RoPE — it uses a learned absolute pos-embed),
so unlike the iMF port nothing had to be rewritten. We keep only the native (non-flash) softmax
attention path, matching the reference's jvp settings.

### Contract & time mapping (objective/JVP/sampler untouched)

`MFDiTOfficialTrajectory.forward` satisfies the same `velocity_net` contract as the other two
backbones, so **nothing in the objective, JVP, or sampler changed**. The objective queries the
net at anchor `time = r` with interval `h = t − r`, and JVP-differentiates w.r.t. `(x, time, h)`
with tangents `(v_inst, +1, −1)` (`mf_diffusion.py:454`). MFDiT wants two boundary times; we feed:

```
r_abs = time         (anchor)     → JVP tangent +1
t_abs = time + h     (endpoint)   → JVP tangent (+1) + (−1) = 0
```

so the perturbation lands on the anchor time — the mirror image of MeanFlow's own `(t:+1, r:0)`
under the noise-at-1 ↔ data-at-1 convention flip already documented at `mf_diffusion.py:404`.
CFG is off (Gen3v6): `omega/t_min/t_max` are ignored and `w` falls back to ones (a constant
w-embed, guidance off), exactly as the reference does when `w is None`.

## Files changed

- **NEW** `flow_matcher_v3_meanflow/models/mf_dit_official_trajectory.py` — the port.
- `flow_matcher_v3_meanflow/models/__init__.py` — export `MFDiTOfficialTrajectory`.
- `flow_matcher_v3_meanflow/models/mf_trajectory_model.py`:
  - import the new class;
  - new `elif imf_backbone == 'mf_dit':` branch (reuses `dit_hidden_size/dit_depth/dit_num_heads/
    dit_patch_size`; `dit_aux_head_depth` / `dit_condition_on_t` are iMF-only and inapplicable);
  - `forward` dual-head routing now includes `'mf_dit'`;
  - `ValueError` message updated to list `'mf_dit'`.

**Unchanged (deliberately):** the objective (`mf_diffusion.py`), engine (`mf_engine.py` already
forwards `imf_backbone` + all `dit_*`), trainer, sampler, config **default** (`imf_backbone:'dit'`
— the live baseline is not disturbed), the sbatch scripts, and the `diffuser/` shim (it
re-imports the real package, which now transitively imports the new backbone — no new file).

## How to run the `mf_dit` arm

The backbone is selected the same way the UNet A/B was run — `add_extras()` is disabled, so there
is **no CLI flag**; you flip the config **value** in `config/avoiding-d3il.py`:

1. **Train block** `flow_matching_v3_meanflow` (~line 642): `'imf_backbone': 'mf_dit'`.
2. **Plan block** `plan_fm_v3_meanflow` (~line 1270): `'imf_backbone': 'mf_dit'` (must match, or
   the eval state_dict load fails).

The `('imf_backbone','bb')` `args_to_watch` token isolates checkpoints automatically
(`..._bbmf_dit_...`), so the new arm cannot collide with the `dit`/`unet` checkpoints, and the
eval `diffusion_loadpath` (which also carries `bb{imf_backbone}`) matches by construction. Then
submit the **existing** sbatch unchanged:

```bash
./submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh      # trains seed 6, mf_dit arm
# then eval / load_results as usual (same folder token)
```

(Revert both config values to `'dit'` afterwards if you want the default baseline back, exactly
as was done for the `unet` A/B.)

## Validation status

- **Local:** `python3 -m py_compile` passes on the new file and all edited files. No
  torch/MuJoCo in this container ⇒ **no numerical run here; must be validated on the cluster.**
- **First cluster run — what to check:** that `mf_dit` trains (unlike `unet`, which never left
  init — see `fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md`): `train/loss` should fall below
  the ≈2.0 adaptive ceiling, `per_dim_rms_u` should drop, and the h-stratified `h_mse_b0..b3`
  curves should shrink (esp. `h_mse_b0`, the FM anchor). Then compare goal-reach / violations at
  K=2 against the `dit` arm to see whether the paper-faithful architecture matches, beats, or
  trails the iMF DiT under the MeanFlow objective.

## Scientific note

This does not change the Gen3v6 hypothesis (analytic-v JVP tangent). It removes an architectural
confound: with `mf_dit`, the MeanFlow objective now runs on the MeanFlow paper's own network, and
the DPCC-UNet / iMF-DiT / MeanFlow-DiT three-way is a clean single-variable backbone A/B under a
fixed objective.
