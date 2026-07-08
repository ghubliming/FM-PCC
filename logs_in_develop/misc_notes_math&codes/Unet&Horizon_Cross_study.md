# TemporalUnet Architecture & Horizon Adaptability: A Deep-Dive

> [!NOTE]
> **Revision note (2026-07-08):** This doc was re-verified line-by-line against the actual code. The original version made a critical mistake: it compared the **class defaults** (`dim=128`) of the DPCC/FM-PCC U-Nets against HardFlow's **runtime-overridden** config (`dim=32`). In reality, **every actual training config runs `dim=32`**, so the old "DPCC bottleneck (B, 1024, 1)" and "HardFlow has far fewer parameters" claims were wrong. It also mislabeled FM-PCC's `flow_matching_v3_ode_selectable` as "SafeFlow MPC". All fixed below.

## 0. Naming clarification

- **Original Diffuser** — `/workspaces/aux_repo/diffuser`, class `TemporalUnet`.
- **HardFlow** — `/workspaces/aux_repo/HardFlow`, class `TemporalUnet` (its own copy).
- **DPCC baseline** — upstream `/workspaces/aux_repo/dpcc` and the `diffusion` config block in this repo; class `UNet1DTemporalCondModel` (`diffuser/models/unet1d_temporal_cond.py`).
- **FM-PCC FM-v3** — the `flow_matching_v3_ode_selectable` config block in this repo; class `Flow_matcher_U_Net_v2` (`flow_matcher_v3_ode_selectable/models/unet1d_temporal_cond.py`). This is **FM-PCC's own model**, *not* the separate `SafeFlowMPC` upstream repo — it only borrows SafeFlow-*style* Beta(α,β) time sampling (`time_beta_alpha_v3` / `time_beta_beta_v3`).

## 1. The Models & Where Their Horizons are Set

Horizon values confirmed from the actual configs and scripts:

**HardFlow — H=16** (defined in the training shell script):
```bash
# /workspaces/aux_repo/HardFlow/run_scripts/train.sh  (L13-15)
horizon_list=(
    16
)
```

**DPCC / FM-PCC baseline — H=8** (defined in config dict):
```python
# /workspaces/FM-PCC/config/avoiding-d3il.py  (L81-85)
'diffusion': {
    'model': 'models.UNet1DTemporalCondModel',
    ...
    'horizon': 8,
```
(Upstream `/workspaces/aux_repo/dpcc/config/avoiding-d3il.py` uses the same H=8.)

**FM-PCC FM-v3 (`flow_matching_v3_ode_selectable`) — H=8**:
```python
# /workspaces/FM-PCC/config/avoiding-d3il.py  (L339-343)
'flow_matching_v3_ode_selectable': {
    'model': 'models.Flow_matcher_U_Net_v2',
    ...
    'horizon': 8,
```

**Original Diffuser — H=32** (defined in base config):
```python
# /workspaces/aux_repo/diffuser/config/locomotion.py  (L21-25)
base = {
    'diffusion': {
        'model': 'models.TemporalUnet',
        'horizon': 32,
```

> [!IMPORTANT]
> Original Diffuser uses $H=32$, HardFlow uses $H=16$, DPCC and FM-PCC use $H=8$.
> The UNet class is **not the same class** across projects — but it shares the same structural skeleton. See Section 2.

---

## 2. Are the U-Nets Actually the Same?

**No — they are different classes.** But the key subtlety is the difference between **class defaults** and **values actually used at runtime**. Every project overrides the defaults from its config, and in practice **everyone runs `dim=32`**:

| Property | Original Diffuser (`TemporalUnet`) | HardFlow [`TemporalUnet`](file:///workspaces/aux_repo/HardFlow/hardflow/models_flow/unet.py#L249) | DPCC [`UNet1DTemporalCondModel`](file:///workspaces/FM-PCC/diffuser/models/unet1d_temporal_cond.py#L84) | FM-PCC FM-v3 [`Flow_matcher_U_Net_v2`](file:///workspaces/FM-PCC/flow_matcher_v3_ode_selectable/models/unet1d_temporal_cond.py#L87) |
|---|---|---|---|---|
| Class default `dim` | `32` | `32` | `128` | `128` |
| **`dim` actually used** | **32** (default) | **32** (eval/train args) | **32** (`config: 'dim': 32`) | **32** (`config: 'dim': 32`) |
| Class default `dim_mults` | `(1, 2, 4, 8)` | `(1, 2, 4, 8)` | `(1, 2, 4, 8)` | `(1, 2, 4, 8)` |
| **`dim_mults` actually used** | `(1, 2, 4, 8)` (locomotion) | **`(1, 4, 8)`** (eval/train args) | `(1, 2, 4, 8)` | `(1, 2, 4, 8)` |
| Attention | arg, default `False` | arg, default `False` | *(no attention arg)* | *(no attention arg)* |
| Conditioning | Inpainting (cond unused in forward) | Inpainting (cond unused in forward) | Inpainting; optional `cond_mlp` FiLM path only when `use_cond_projection=True` (visual pipelines) | Inpainting; optional returns-FiLM (`returns_mlp`) — **off** here (`returns_condition: False`) |
| Residual conv | ✅ | ✅ | ✅ | ✅ |
| Skip connections | `torch.cat` on `h` stack | `torch.cat` on `h` stack | `torch.cat` on `h` stack | `torch.cat` on `h` stack |

Two important corrections vs. the old version of this doc:

1. **HardFlow's class default is `dim_mults=(1, 2, 4, 8)` — 4 levels, same as everyone else.** The 3-level `(1, 4, 8)` only appears because `eval.py` (and train) instantiate with explicit non-default args:

```python
# /workspaces/aux_repo/HardFlow/run/eval.py  (L531-538)
flow_model = TemporalUnet(
    horizon=cfg.horizon,
    transition_dim=cfg.state_dim + cfg.action_dim,   # = 4+2 = 6
    cond_dim=cfg.state_dim,
    dim=32,
    dim_mults=(1, 4, 8),        # <-- 3 levels, overriding the 4-level class default
    attention=False,
).to(cfg.device)
```

Notice: `dim_mults=(1, 4, 8)` — only **3 elements**, meaning **only 2 actual downsampling steps**.

2. **DPCC's/FM-v3's class default `dim=128` is never used.** Both this repo's config and upstream DPCC pass `'dim': 32` (`config/avoiding-d3il.py` in both repos), and `scripts/train.py` forwards it: `model_config = utils.Config(..., dim=args.dim, dim_mults=args.dim_mults, ...)`. So the real difference between HardFlow and DPCC/FM-v3 is **not width** — it's `dim_mults` shape (`(1,4,8)` vs `(1,2,4,8)`) and the conditioning machinery.

---

## 3. Why Convolutions Are Horizon-Agnostic

The reason all these U-Nets can handle different `horizon` values without code changes is that they are **fully convolutional** along the time axis.

In all models, the first thing the forward pass does is rearrange the tensor so that the horizon dimension becomes the **Conv1d spatial dimension**:

```python
# /workspaces/aux_repo/HardFlow/hardflow/models_flow/unet.py  (L358)
x = einops.rearrange(x, "b h t -> b t h")
# Result: (batch, transition_dim=channels, horizon=spatial_length)
```
```python
# /workspaces/FM-PCC/diffuser/models/unet1d_temporal_cond.py  (L202)
x = einops.rearrange(x, 'b h t -> b t h')
# Identical rearrange.
```

After this rearrange, every `Conv1d` layer inside `Conv1dBlock` (used in `ResidualTemporalBlock`) operates on the last dimension. A `Conv1d` with kernel `k` and appropriate padding does not impose a size constraint on its input length — the kernel slides across however many timesteps there are.

```python
# /workspaces/aux_repo/HardFlow/hardflow/models_flow/unet.py  (L45-57)
self.block = nn.Sequential(
    nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
    ...
)
```

The `padding=kernel_size // 2` ensures the spatial length is **preserved** through each residual block.

---

## 4. The Divisibility Constraint: Where It Actually Comes From

The only hard constraint on horizon length comes from `Downsample1d`, which halves the spatial dimension:
```python
# /workspaces/aux_repo/HardFlow/hardflow/models_flow/unet.py  (L27-33)
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)  # stride=2 → halves length
```

And `Upsample1d`, which doubles it:
```python
# (L36-42)
class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)  # stride=2 → doubles length
```

The skip connections `torch.cat((x, h.pop()), dim=1)` in the decoder require that the upsampled tensor's spatial length **exactly matches** the skip tensor from the encoder. If they don't match, PyTorch raises a shape error at runtime.

The number of downsampling steps = `len(dim_mults) - 1` (the last level has no `Downsample1d`). So:
- **Original Diffuser** (`dim_mults=(1,2,4,8)` → **3 downs**): requires $H$ divisible by $2^3 = 8$
- **DPCC / FM-PCC FM-v3** (`dim_mults=(1,2,4,8)` → **3 downs**): requires $H$ divisible by $2^3 = 8$
- **HardFlow as run** (`dim_mults=(1,4,8)` → **2 downs**): requires $H$ divisible by $2^2 = 4$

> [!NOTE]
> HardFlow (as configured) has a **softer** constraint than DPCC or the original Diffuser: any $H$ divisible by 4 works (4, 8, 12, 16, 20...). DPCC requires $H$ divisible by 8. HardFlow's $H=16$ and Diffuser's $H=32$ satisfy both constraints.

---

## 5. Full Tensor Shape Trace

With the **actual runtime values** (`dim=32` everywhere):

**DPCC / FM-PCC FM-v3: `(B, H=8, C=6)`, `dim=32`, `dim_mults=(1,2,4,8)`** → channels `[6, 32, 64, 128, 256]`

| Stage | Layer | Channels | Spatial (Horizon) |
|---|---|---|---|
| Input | — | 6 | 8 |
| down[0] | ResBlock ×2 | 6→32 | 8 |
| down[0] | `Downsample1d` | 32 | 8→**4** |
| down[1] | ResBlock ×2 | 32→64 | 4 |
| down[1] | `Downsample1d` | 64 | 4→**2** |
| down[2] | ResBlock ×2 | 64→128 | 2 |
| down[2] | `Downsample1d` | 128 | 2→**1** |
| down[3] *(last)* | ResBlock ×2, **no downsample** | 128→256 | 1 |
| mid | ResBlock ×2 | 256→256 | 1 |
| up[0] | concat skip, ResBlock ×2, `Upsample1d` | 512→128→128 | **1→2** |
| up[1] | concat skip, ResBlock ×2, `Upsample1d` | 256→64→64 | **2→4** |
| up[2] | concat skip, ResBlock ×2, `Upsample1d` | 128→32→32 | **4→8** |
| final_conv | Conv1dBlock + Conv1d | 32→6 | 8 |
| Output rearrange | — | 6 | 8 |

---

**HardFlow: `(B, H=16, C=6)`, `dim=32`, `dim_mults=(1,4,8)`** → channels `[6, 32, 128, 256]`

| Stage | Layer | Channels | Spatial (Horizon) |
|---|---|---|---|
| Input | — | 6 | 16 |
| down[0] | ResBlock ×2 | 6→32 | 16 |
| down[0] | `Downsample1d` | 32 | 16→**8** |
| down[1] | ResBlock ×2 | 32→128 | 8 |
| down[1] | `Downsample1d` | 128 | 8→**4** |
| down[2] *(last)* | ResBlock ×2, **no downsample** | 128→256 | 4 |
| mid | ResBlock ×2 | 256→256 | 4 |
| up[0] | concat skip, ResBlock ×2, `Upsample1d` | 512→128→128 | **4→8** |
| up[1] | concat skip, ResBlock ×2, `Upsample1d` | 256→32→32 | **8→16** |
| final_conv | Conv1dBlock + Conv1d | 32→6 | 16 |
| Output rearrange | — | 6 | 16 |

> [!NOTE]
> Both bottlenecks reach the **same channel width (256)** — the interesting difference is spatial: DPCC's H=8 with 3 downsamples **collapses the bottleneck to a single timestep** `(B, 256, 1)`, while HardFlow's H=16 with 2 downsamples keeps 4 timesteps `(B, 256, 4)`. Parameter counts are in the same ballpark (DPCC has one extra level but with a gentler channel ramp 32→64→128→256 vs HardFlow's 32→128→256) — the old claim that HardFlow is "substantially smaller" was based on the wrong `dim=128` assumption and is **not true** for the real configs.

---

## 6. Why You Can't Transfer Weights Between Them

Even at the same `dim=32`, HardFlow and DPCC/FM-v3 weights are **incompatible**:

1. **Number of levels differs**: HardFlow (as run) has 3 levels / 2 downsamples, DPCC has 4 levels / 3 downsamples — the `self.downs` / `self.ups` `ModuleList` structures have different lengths.
2. **Channel progressions differ**: HardFlow `32→128→256` vs DPCC `32→64→128→256`. E.g. HardFlow's down[1] ResBlock maps 32→128, DPCC's maps 32→64.
3. **Decoder concat widths differ accordingly** (e.g. first decoder ResBlock input is `256*2=512` in both, but the next levels diverge: HardFlow 256 vs DPCC 256/128).
4. **Extra conditioning modules**: `UNet1DTemporalCondModel` / `Flow_matcher_U_Net_v2` carry `cond_mlp` / `returns_mlp` machinery and a diffusers `ModelMixin/ConfigMixin` base that plain `TemporalUnet` lacks — state-dict keys don't line up even where shapes agree.

---

## 7. The `WrappedFlowUnet` and CasADi Flattening

One place where the horizon size has a **concrete, non-trivial implication** is `WrappedFlowUnet` used for the CasADi NLP formulation:

```python
# /workspaces/aux_repo/HardFlow/hardflow/models_flow/unet.py  (L392-395)
def add_info(self, horizon, transition_dim):
    self.horizon = horizon
    self.transition_dim = transition_dim
    self.vector_dim = horizon * transition_dim + 1  # +1 for time scalar
```

For HardFlow: `vector_dim = 16 * 6 + 1 = 97`
For an H=8 equivalent: `vector_dim = 8 * 6 + 1 = 49`

This flattened vector is what CasADi's NLP solver sees as its decision variable. In `create_constrained_casadi_function` in `eval.py`:
```python
# /workspaces/aux_repo/HardFlow/run/eval.py  (L312-314)
transition_dim = action_dim + state_dim   # = 6
N_full = horizon * transition_dim         # H=16: N_full=96, H=8: N_full=48
dof = N_full - state_dim                  # H=16: dof=92, H=8: dof=44
```

A larger horizon = a **larger NLP** with more degrees of freedom → the optimization problem is harder and more expensive to solve per planning step.

---

## Summary Comparison Table

All values are **as actually configured/run**, not class defaults:

| Property | Original Diffuser (`TemporalUnet`) | DPCC (`UNet1DTemporalCondModel`) | FM-PCC FM-v3 (`Flow_matcher_U_Net_v2`) | HardFlow (`TemporalUnet`) |
|---|---|---|---|---|
| **Horizon** | **32** | 8 | 8 | **16** |
| **dim** | 32 | 32 | 32 | 32 |
| **dim_mults** | (1, 2, 4, 8) | (1, 2, 4, 8) | (1, 2, 4, 8) | **(1, 4, 8)** |
| **# Downsamples** | 3 | 3 | 3 | **2** |
| **Divisibility req.** | $H \bmod 8 = 0$ | $H \bmod 8 = 0$ | $H \bmod 8 = 0$ | $H \bmod 4 = 0$ |
| **Bottleneck shape** | `(B, 256, 4)` | `(B, 256, 1)` | `(B, 256, 1)` | `(B, 256, 4)` |
| **CasADi dof** | N/A | N/A | N/A | 92 (vs 44 for H=8) |
| **Conditioning** | Inpainting only | Inpainting (+ optional `cond_mlp` FiLM for visual pipelines, off for state-based) | Inpainting (`returns_condition: False`; Beta-time refers to *time sampling*, not conditioning) | Inpainting only |
| **Weight transfer** | ❌ incompatible | ❌ incompatible | ❌ incompatible | ❌ incompatible |
