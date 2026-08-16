# Gen3v7 U2 — add **α-Flow's own backbone, the SiT**, as a third backbone (`imf_backbone='sit'`)

Mirrors the *principle* of **Gen3v6 U2** — but not its network. Gen3v6 U2 mirrored the
**MeanFlow repo's own** `MFDiT` because Gen3v6 runs the MeanFlow objective. The correct analogue
for Gen3v7 is to mirror **α-Flow's own repo** (`/workspaces/aux_repo/alphaflow`), whose network
is the **`SiT`** (`src/training/dit.py`). This U2 ports that SiT, so the α-Flow objective finally
runs on α-Flow's own architecture and Gen3v7 gets a clean three-way backbone A/B
(DPCC-UNet / iMF-DiT / **α-Flow-SiT**).

> **Correcting a first wrong attempt (recorded for honesty):** the initial pass at this U2 ported
> MeanFlow's `MFDiT` (the `mf_dit` token). That was the zero-thought move — mirroring the wrong
> paper's network. It was fully reverted; this U2 mirrors α-Flow's SiT instead.

## Why the SiT ≠ MeanFlow's MFDiT ≠ iMF's DiT

All three are the DiT/SiT transformer family, but every **learned** detail differs — so which one
you use is a real architectural variable, not cosmetic:

| `imf_backbone` | class | norm | QK-norm | positions | time embed | heads | origin |
|---|---|---|---|---|---|---|---|
| `unet` | `Flow_matcher_U_Net_v2` | — | — | — (conv) | `time_mlp(t)+h_mlp(h)` | dual opt. | DPCC U-Net |
| `dit` | `AFDiTTrajectory` | RMSNorm | on (RoPE) | **RoPE** | (rope) | shared trunk→heads | **iMF** (`imeanflow`) |
| **`sit`** (NEW) | **`AFSiTTrajectory`** | **LayerNorm** (affine-off, fp32) | **OFF** | **frozen sin-cos** | freq=256, **no scale** | single (u) + aux v | **α-Flow** (`aux_repo/alphaflow`) |

Key α-Flow-specific facts the port preserves (and that MeanFlow's MFDiT does NOT share):
`torch.nn.LayerNorm(elementwise_affine=False)` run in **fp32** inside `modulate`
(`x*(scale+1)+shift`); softmax `Attention` with **`qk_norm=False`** (α-Flow passes this
explicitly at `dit.py:121`); the **gate applied outside** the modulated sublayer
(`x + gate·f(modulate(...))`); a **frozen** sin-cos pos-embed (`requires_grad=False`); a
`TimestepEmbedder` with **no `scale=1000`** multiplier (MeanFlow has one); and the two time
embedders `noise_labels` (t) + `noise_labels_next` (r) whose sum is the adaLN conditioning
(`dit.py:199`, comment: "modified for MeanFlow with r and t").

## What was added

**New file `flow_matcher_v3_alphaflow/models/af_sit_trajectory.py`** — class `AFSiTTrajectory`,
a faithful port of α-Flow's `SiT`/`SiTBlock`/`FinalLayer`/`TimestepEmbedder`/`modulate`. Every
learned component is verbatim (adaLN-zero 6·d block; LayerNorm-affine-off fp32 modulate;
qk_norm=False softmax attention; GELU-tanh `Mlp` at `mlp_ratio=4.0`; the two freq-256/no-scale
time embedders; α-Flow's `initialize_weights` — xavier trunk, 0.02 time-MLPs, zeroed adaLN,
zeroed output linear).

### Deviation family (A) — forced by 1-D trajectory data, no math change

(the same class Gen3v6's port carried for images→trajectories)

| α-Flow `SiT` (images) | port (`sit`, trajectories) | why |
|---|---|---|
| `PatchEmbed` (Conv2d) | `TrajPatchEmbedder` (Linear over `patch_size` steps) | data is `[B,H,D]`, not `[N,C,H,W]` |
| 2-D sin-cos `pos_embed` (frozen) | 1-D sin-cos `pos_embed` (**still frozen**) | the sequence is 1-D |
| `unpatchify` (image) | `[B,H,D]` reshape | ″ |
| timm `Attention`/`Mlp` | inline equivalents (qk_norm=False, GELU-tanh) | no timm dependency on the cluster |
| class-label `y_embedder` | **off** — α-Flow's unconditional path | trajectory conditioning is the pinned obs applied to x externally ⇒ `c = t_emb + r_emb` |

### Deviation family (B) — forced by the FM-PCC lineage, NOT by α-Flow

α-Flow's SiT is **single-head** (predicts u; its instantaneous v is analytic). The FM-PCC
Gen3v7 lineage trains a v-head as an auxiliary loss against the analytic `v_inst`
(`af_diffusion.py:735`, `err_v=(v_pred−v_inst.detach())²`) and its wrapper expects `(u, v)` —
exactly as the `unet` (dual_head) and `dit` arms already do. So `AFSiTTrajectory` adds a
**second `final_layer_v`** reading the same trunk (twin-head SiT). This changes nothing α-Flow
deploys: **v is dropped at inference**, the deployed field is u, so the α-Flow-faithful u-path is
untouched; the v-head only feeds the lineage's shared-trunk aux regulariser. A strictly-single-head
α-Flow run can drop `final_layer_v` and route v via the wrapper's legacy aux MLP — noted as a
possible later refinement.

### JVP-safety & time mapping (objective/JVP/sampler untouched)

α-Flow's α=0 branch differentiates the net with `torch.func.jvp` (`af_diffusion.py:575`, tangents
`(v_inst, +1, −1)` on `(x, time, h)`). Safe by construction: LayerNorm + softmax (no QK-norm) +
GELU are all forward-AD friendly, and there is **no RoPE complex-bitcast hazard** (SiT uses a
plain frozen sin-cos pos-embed). SiT wants two boundary times; we feed:

```
r_abs = time         (anchor)     → noise_labels_next (r)   JVP tangent +1
t_abs = time + h     (endpoint)   → noise_labels      (t)   JVP tangent (+1)+(−1) = 0
```

so the perturbation lands on the anchor — the mirror image of α-Flow's own noise-at-1
`(t, r=t_next)` under the noise-at-1 ↔ data-at-1 convention flip. α-Flow's
`noise_labels ≥ noise_labels_next` invariant holds since `t_abs = r_abs + h ≥ r_abs` for `h ≥ 0`.
These are the **same** tangents the iMF `dit` arm uses, so the mapping is valid unchanged.

## Files changed

- **NEW** `flow_matcher_v3_alphaflow/models/af_sit_trajectory.py` — the α-Flow SiT port.
- `flow_matcher_v3_alphaflow/models/__init__.py` — export `AFSiTTrajectory`.
- `flow_matcher_v3_alphaflow/models/af_trajectory_model.py`:
  - import the new class;
  - new `elif imf_backbone == 'sit':` branch (reuses `dit_hidden_size/dit_depth/dit_num_heads/
    dit_patch_size`; `dit_aux_head_depth` / `dit_condition_on_t` are iMF-only and inapplicable);
  - `forward` dual-head routing now `self.imf_backbone in ('dit','sit')`;
  - `ValueError` message updated to list `'sit'`.

**Unchanged (deliberately):** the objective (`af_diffusion.py`), engine (`af_engine.py` already
forwards `imf_backbone` + all `dit_*` at `:70-88`), trainer, sampler, gates, config **defaults**
(`imf_backbone:'dit'` in the train/plan blocks — the live baseline is not disturbed), the sbatch
scripts, and the `diffuser/flow_matcher_v3_alphaflow/` shim (it re-imports the real package via
`af_engine`, which now transitively imports the new backbone — **no new shim file**).

## How to run the `sit` arm

Backbone selection is by **config value**, not CLI (`Parser.add_extras` is disabled here). Flip
both blocks in `config/avoiding-d3il.py` so train and eval agree (the state_dict depends on the
architecture):

1. **Train block** `flow_matching_v3_alphaflow` (~line 642): `'imf_backbone': 'sit'`.
2. **Plan block** `plan_fm_v3_alphaflow` (~line 758): `'imf_backbone': 'sit'` (must match).

The `('imf_backbone','bb')` `args_to_watch_fmv3_af_train` token isolates checkpoints
automatically (`..._bbsit_...`), and the eval `diffusion_loadpath` (`config:1383`) carries the
same `bb{imf_backbone}` token, so the new arm cannot collide with the `dit`/`unet` checkpoints and
eval matches by construction. Then submit the **existing** sbatch unchanged:

```bash
./submit.sh Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh   # trains seed 6, sit arm
# then eval / load_results as usual (same folder token; AF_FLOW_STEPS grid unchanged)
```

(Revert both config values to `'dit'` afterwards to restore the default baseline.)

## Validation status

- **Local:** `python3 -m py_compile` passes on the new file and both edited files. No
  torch/MuJoCo in this container ⇒ **no numerical run here; validate on the cluster.**
- **First cluster run — what to check:**
  1. `sit` **trains** (unlike `unet`, which the ablation showed never leaves per-dim RMS ≈ 1.0):
     `train/loss` below the adaptive ceiling, `per_dim_rms_u` falling, and the h-stratified
     `h_mse_b0..b3` curves shrinking (esp. `h_mse_b0`, the FM anchor).
  2. The **α-schedule alive** summary and `alpha`/`discrete_frac`/`clamp_frac` curves behave as
     in the `dit` run (U2 does not touch the objective — sanity check the new backbone doesn't
     destabilise the annealing).
  3. **Matched-K** planning A/B vs the `dit` arm at the same NFE grid — does α-Flow's own SiT
     match, beat, or trail the iMF DiT under the α-Flow objective?

## Scientific note

This removes the architecture/objective confound the RIGHT way: with `sit`, the α-Flow objective
runs on **α-Flow's own network** (LayerNorm, qk_norm-off, frozen sin-cos, t+r conditioning), and
the DPCC-UNet / iMF-DiT / α-Flow-SiT three-way is a clean single-variable backbone A/B under a
fixed objective. The only concession to the FM-PCC lineage is the twin v-head aux output, which is
inert at inference. (Contrast: the reverted first attempt would have run α-Flow's objective on
MeanFlow's network — a cross-paper mismatch.)
