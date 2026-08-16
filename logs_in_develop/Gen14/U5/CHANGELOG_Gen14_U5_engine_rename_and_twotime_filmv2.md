# CHANGELOG — Gen14 U5: `ddpm` → `diffusion`, and True FiLM on the two-time arms

**Date:** 2026-08-02 · **Follows:** [`../U4/CHANGELOG_Gen14_U4_wandb_metric_parity.md`](../U4/CHANGELOG_Gen14_U4_wandb_metric_parity.md)

> **Numbering.** Filed as **U5**, not U4. U4 is already taken by today's earlier
> W&B-metric-parity pass, which is a different concern and still uncommitted. Both
> land in the same working tree; the changelogs are kept separate so a bisect can tell
> "the logger changed" from "the model can change shape now".

Three requests, one pass:

1. `ddpm` is the wrong name for that arm — this repo says *diffusion*.
2. Do Gen6V4/Gen7 still support `film_mode: 'v2'`? (They do, and nothing here changed that.)
3. Get FiLM v2 onto `mf`/`af` too, **or report that the math forbids it.** It does not
   forbid it. It is implemented.

---

## 1. Why the two-time arms could not run v2 — and why that reason is now gone

The pre-U5 guard was real, not defensive padding:

```python
# visual_unet_twotime.py (before U5)
if film_mode not in ('v1', None):
    raise ValueError("... unet1d_temporal_film.py has no h_mlp, so the "
                     "MeanFlow/alpha-Flow h-conditioning would be silently dropped.")
```

Gen7's v2 backbone (`UNet1DTemporalFiLMModel`) is a **one-time** network. It has
`time_mlp` and no `h_mlp`. Routing `mf`/`af` through it would have produced a model
that trains happily, logs a falling `raw_mse_u`, and is **not conditioned on the
interval h at all** — the MeanFlow identity's entire premise, silently absent. Raising
was correct.

But the obstacle was a *missing file*, not a theorem. v2 changes **where the visual
latent enters**; `h` enters on the **time** side. The two are orthogonal. U5 writes the
file that carries both.

### The JVP question — the one that actually needed checking

MeanFlow's target comes from a forward-mode derivative (`mf_diffusion.py:454`):

```python
u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
```

Tangents ride on **z, r, h only**. `cond` is closed over by `_u_of` as a captured
constant — the pre-encoded `visual_latent` (PLAN §6.1). So under v2's

```
out = (1 + γ(cond)) · ( Conv(x) + time_mlp(t) ) + β(cond)
```

γ and β have identically-zero tangent, and

```
d/dr [ (1+γ)·f + β ]  =  (1+γ) · df/dr
```

— a per-channel **rescaling** of the same directional derivative v1 computes. No new
term enters the MeanFlow identity. Forward-mode AD only has to push a dual number
through `mul` and `add`, both of which have forward-AD rules; nothing here is in the
same class as the torchvision-ResNet-under-JVP problem that motivated the short-circuit.
γ/β are zero-initialised, so step 0 is numerically identical to no-FiLM.

**Verdict: implementable, and implemented.** Gated anyway (G7.4) — "should be fine" is
precisely what G2 exists to disprove.

### The variant that is NOT forbidden but must not be done here

The tempting next step is to condition the gate on the interval too:
`(γ, β) = film_proj([c ‖ MLP_h(h)])`. This is not a mathematical error — the JVP would
correctly include the extra `∂γ/∂h · f` term, and forward-AD handles it. It is wrong for
two other reasons:

1. **It changes the model class.** `u` is then a function whose h-dependence is entangled
   with the vision gate, and the MeanFlow identity `u ← v + h·du/dr` is being applied to a
   different `u` than Gen3v6's. The generation's whole premise is that `mf`/`af` are
   Gen3v6/Gen3v7 with vision bolted on; this would make them something else.
2. **It costs JVP compute.** `film_proj` moves from "evaluated once on a constant" to
   "inside the differentiated path", which is exactly what PLAN §6.1's pre-encode exists
   to avoid.

The warning is written into the new file's header where the next agent will hit it.

---

## 2. The math: what is Gen6V4/Gen7's, what is Gen3v6/v7's, what is new

This section exists because "we ported FiLM v2" is ambiguous about *how much* changed.
The short answer: **the conditioning operator is Gen7's, unmodified; only the argument it
is handed differs.** Below is the derivation.

### 2.1 First, the prior question: WHICH BACKBONE runs? (U-Net — not the DiT/SiT)

Everything below is about FiLM, which is a *conditioning* choice. It only makes sense
once the *backbone* question is settled, and that answer surprises people:

**Gen14's `mf`/`af` do NOT use Gen3v6/Gen3v7's transformer backbones. They use the same
1-D temporal U-Net as `diffusion` and `fm`.**

Gen3v6/Gen3v7 are **state-only** generations, and their own defaults are transformers:

| | `imf_backbone` default | what it is |
|---|---|---|
| Gen3v6 `mf` | `'mf_dit'` (`config/avoiding-d3il.py:644`) | official MeanFlow DiT, adaLN-zero |
| Gen3v7 `af` | `'sit'` (`config/avoiding-d3il.py:761`) | α-Flow's own SiT |
| **Gen14 `mf`/`af`** | **`'unet'`** (never set ⇒ default, `mf_trajectory_model.py:45`) | the DPCC 1-D temporal U-Net |

You can see it in Gen3v6's eval candidates — e.g. `CAND_102` is
`…MeanFlowODE_aw10_objmeanflow_bbmf_dit_tslogit_normal_dp0.5`. That `bbmf_dit` fragment
is the backbone, and Gen14's paths have no `bb` fragment at all.

**And in visual mode it is not selectable — it raises:**

```python
# mf_trajectory_model.py:76-81   (af_trajectory_model.py:76-81 is identical)
if imf_backbone not in ('unet',):
    raise ValueError("if_vision=True requires imf_backbone='unet' ... The DiT/SiT "
                     "backbones have no visual conditioning path and would train image-blind.")
```

So `if_vision=True` ⇒ U-Net, full stop. What each arm actually builds:

| arm | backbone | vision | construction chain |
|---|---|---|---|
| `diffusion` | 1-D temporal U-Net | dual ResNet-18 | `VisualUNet` → `UNet1DTemporalCondModel` |
| `fm` | 1-D temporal U-Net | dual ResNet-18 | `VisualUNet` → `UNet1DTemporalCondModel` |
| `mf` | 1-D temporal U-Net **+ `h_mlp`** | dual ResNet-18 | `MeanFlowEngine` → `MFTrajectoryModel` → `VisualUNetTwoTime` → `Flow_matcher_U_Net_v2` |
| `af` | 1-D temporal U-Net **+ `h_mlp`** | dual ResNet-18 | `AlphaFlowEngine` → `AFTrajectoryModel` → `VisualUNetTwoTime` → `Flow_matcher_U_Net_v2` |

**All four arms share one backbone family — by design.** That is the point of the
generation (PLAN §3.1, "locked backbone"): the four-way comparison isolates the
**objective** (DDPM vs FM vs MeanFlow vs α-Flow), not the architecture. FiLM v1/v2 is a
second axis on top of that, and it applies to all four arms equally.

#### What job 24124 was, in full

```
H8_D…VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal / seed 6
```

| | |
|---|---|
| objective | MeanFlow (Gen3v6), `t_schedule=logit_normal`, `p_mean=-0.4`, `dp=0.5` |
| backbone | two-time 1-D temporal U-Net, `dim=128`, `dim_mults=(1,2,4,8)`, `dual_head=True` |
| vision | `MultiImageObsEncoder` — two ResNet-18s (agentview + wrist) → 128-D latent |
| conditioning | **FiLM v1** (additive bias via time-embedding concat) |

So it is **"U-Net + FiLM v1"**, *not* "MF-DiT + FiLM v1". There is no DiT anywhere in
Gen14.

#### The cross-generation confound this creates

Gen3v6's published `mf` numbers come from an **MF-DiT**; Gen14's `mf` comes from a
**U-Net**. Same objective, different architecture. A Gen14-mf-vs-Gen3v6-mf comparison is
therefore **not** a clean read of "does MeanFlow transfer to vision" — it is confounded
with a backbone swap. *Within* Gen14 the comparison is clean, because all four arms share
the backbone; *across* generations it is not. Worth stating in any write-up that puts the
two side by side.

#### The Gen8 precedent — every visual generation in this repo is a U-Net

`imf_visual_aligning/` (Gen8, June 2026) asked this exact question first and answered it
the same way. `imf_trajectory_model.py:39-44`:

```python
if if_vision:
    from .visual_unet import VisualUNet
    self.velocity_net = VisualUNet(vis_config)
```

and Gen8's own `unet1d_temporal_cond.py:86-93` is **the same graft Gen14 later repeated**
— Gen7's cond model plus Gen3v4's `h_mlp` — injected the same way
(`t = t + self.h_mlp(h_val)`, line 249). So the substitution in §2.4 is not a Gen14
invention; it is the repo's standing convention, arrived at independently twice.

Two details worth carrying forward:

- **Gen8 has no `film_mode` key at all** (zero hits in the whole folder). It predates the
  FiLM_V2 upgrade (2026-06-27), so it is permanently v1 — with no way to reach v2 short of
  the same port U5 just did for `mf`/`af`.
- **Gen8's folder contains no DiT module.** ⚠️ `MASTER_TEST_HISTORY.md:31` states Gen8
  "Supports Official DiT backbone" — that is **not true of the visual folder**.
  `ls imf_visual_aligning/models/` has no DiT file, and `imf_trajectory_model.py` offers
  exactly two branches (`VisualUNet` / `Flow_matcher_U_Net_v2`). Read that claim as
  describing the state-only iMF lineage. Noted here, **not** edited there — the master
  index is documented as not 100 % accurate and is not this changelog's to correct.

The scoreboard across the whole repo:

| generation | visual? | backbone | conditioning |
|---|---|---|---|
| Gen6V4 `diffuser_visual_aligning` | yes | 1-D temporal U-Net | FiLM v1 (v2 selectable) |
| Gen7 `fm_visual_aligning` | yes | 1-D temporal U-Net | FiLM v1 (v2 selectable) |
| Gen8 `imf_visual_aligning` | yes | 1-D temporal U-Net **+ `h_mlp`** | FiLM v1 (**no knob**) |
| Gen14, all four arms | yes | 1-D temporal U-Net (+ `h_mlp` on mf/af) | FiLM v1; v2 selectable (U5) |
| Gen3v6 / Gen3v7 | **no** — state-only | MF-DiT / SiT | inpainting; no cond vector |

**No transformer backbone has ever been run with vision anywhere in this repo.** That one
sentence is why a visual DiT is a new capability rather than a config recovery — see §8.

#### A latent trap, noted not fixed

`imf_backbone` is **not** in `args_to_watch_mix_visual_train`, so the backbone does not
appear in Gen14 path names. Harmless today — visual mode admits exactly one value — but
if a vision-capable DiT ever lands, **add the key to the watch list first**, or U-Net and
DiT checkpoints will collide in the same directory.

### 2.2 There are exactly two modes. There is no v3.

`film_mode ∈ {'v1', 'v2'}` — repo-wide, in every generation. Nothing else has ever
existed, and U5 adds no third mode. The key is optional; absent means `'v1'`.

### 2.3 The residual block, in both modes

Let `x ∈ R^{B×C_in×T}` be the block input, `e` the conditioning embedding, `c` the
projected visual latent, `Conv₁ / Conv₂` the two `Conv1dBlock`s and `Res` the 1×1
skip projection. Both blocks broadcast their conditioning over the time axis `T`.

**v1** — `ResidualTemporalBlock` (Gen7 `unet1d_temporal_cond.py`, and Gen3v6's identical copy):

```
y₁ = Conv₁(x) + W_t · Mish(e)
y  = Conv₂(y₁) + Res(x)
```

**v2** — `FiLMResidualTemporalBlock` (Gen7 `unet1d_temporal_film.py`):

```
y₁     = Conv₁(x) + W_t · Mish(e)
(γ, β) = W_f · Mish(c)                     W_f zero-initialised (weight AND bias)
y₂     = y₁ ⊙ (1 + γ) + β                  per-channel, broadcast over T
y  = Conv₂(y₂) + Res(x)
```

So **v2 = v1 with one affine modulation inserted between the two convolutions.** That is
the entire architectural difference, and it is Perez et al. (2018) FiLM. Note the
placement: *between* `Conv₁` and `Conv₂`, not at the block output.

### 2.4 What differs across the four (generation × mode) combinations

Only `e` and `c` change. `τ` is the flow/diffusion time, `h` the two-time interval,
`v` the 128-D visual latent, `d = dim = 128`.

| generation × mode | `e` (time slot) | `c` (visual slot) |
|---|---|---|
| Gen6V4 / Gen7 · **v1** | `[ MLP_τ(τ) ‖ MLP_c(v) ] ∈ R^{2d}` | — (no separate path) |
| Gen6V4 / Gen7 · **v2** | `MLP_τ(τ) ∈ R^d` | `MLP_c(v) ∈ R^d` |
| Gen14 mf/af · **v1** (pre-U5, and still the default) | `[ MLP_τ(τ) + MLP_h(h) ‖ MLP_c(v) ] ∈ R^{2d}` | — |
| Gen14 mf/af · **v2** (U5, new) | `MLP_τ(τ) + MLP_h(h) ∈ R^d` | `MLP_c(v) ∈ R^d` |

Read down the `c` column: **the mode decides how `v` enters.** Read across a row: **`h`
only ever enters the time slot, additively, and never touches `c`.** That is the formal
statement of "orthogonal", and it is why the port is a port and not a redesign.

Equivalently, as a substitution:

```
Gen14_v2(z, v, τ, h)  =  Gen7_v2(z, v, τ) │ MLP_τ(τ) ← MLP_τ(τ) + MLP_h(h)     (+ Gen3v6 dual head)
Gen14_v1(z, v, τ, h)  =  Gen7_v1(z, v, τ) │ MLP_τ(τ) ← MLP_τ(τ) + MLP_h(h)     (+ Gen3v6 dual head)
```

The same one-line substitution in both modes. Gen14's v1 was already built this way —
U5 did not invent the pattern, it applied the existing pattern to the other mode.

### 2.5 One incidental simplification v2 brings

v1's `unet1d_twotime_cond.py` carries a 🔴 ORDER MATTERS comment: `h_mlp` and the
interval-CFG terms emit `[B, d]` and must be added **before** the cond concat widens `e`
to `[B, 2d]`, or the first block gets a shape error. In v2 there is no concat — `e` stays
`[B, d]` all the way to the blocks — so that ordering constraint **disappears**. One less
way to break the file.

### 2.6 Was v1 working for mf/af before this change?

**On `mf`: yes, demonstrably.** Job 24124 is `…_bs64_filmv1_Emf_tslogit_normal`, seed 6,
1e5 steps, completed. `val/raw_mse_u` went 91.70 → 7.29 (12.6×) and `val/h_mse_b3`
94.27 → 0.98 (96×). The v1 visual conditioning demonstrably learns. **U5 is an added
option, not a repair** — nothing about v1 was broken and v1 remains the default on all
four arms.

**On `af`: unknown — `af` has never been run in Gen14, in any film mode.** So has
`diffusion`, and so has `fm`. Job 24124 is the only Gen14 training run that exists.

**And no Gen14 run has ever used v2, on any arm.** The v2 path is supported and gated,
never executed. G7 will be its first contact with a GPU.

### 2.7 Cost of the switch, exactly

`dim=128`, `dim_mults=(1,2,4,8)`, `transition_dim=9` ⇒ 16 residual blocks,
`Σ out_channels = 7680`. Per block the time path is `Linear(embed_dim → C)` and the FiLM
head is `Linear(d → 2C)`:

| path | v1 | v2 |
|---|---:|---:|
| per-block `time_mlp.in_features` | `2d` = **256** | `d` = **128** |
| time path, all blocks | 1,973,760 | 990,720 |
| `film_proj` heads, all blocks | — | 1,981,440 |
| **conditioning total** | **1,973,760** | **2,972,160** |
| `cond_mlp` (v → R^d) | 33,024 | 33,024 (identical) |

**v2 costs ≈ +1.0 M parameters** (`+998,400`) in the backbone. `cond_mlp` is untouched:
the projection of the visual latent is the same network in both modes; only its
destination changes.

This also means the two modes' state_dicts are structurally incompatible in *two*
independent ways — `time_mlp` weights change shape, and `film_proj` has no v1
counterpart. `film_mode` being a path key is what keeps that from becoming a silent
mis-load.

### 2.8 A behavioural difference worth knowing before reading v1-vs-v2 curves

`W_f` is **zero-initialised**, and in v2 `v` reaches the network through `W_f` and
nowhere else. Therefore at step 0, `γ = β = 0`, `y₂ = y₁`, and **a v2 model is exactly
blind to vision at initialisation**; it must learn to open the gate. A v1 model is not —
`W_t` is randomly initialised over the concatenated `[t ‖ c]`, so v1 responds to the
image (randomly) from step 0.

This is Gen7's design choice, inherited verbatim, not a Gen14 decision. Two consequences:

- Early-epoch curves of v1 and v2 are **not** comparable step-for-step. A v2 run's first
  epochs are effectively a time-only model warming up.
- The zero-init is also what makes v2 safe under the JVP at step 0 (§1): the derivative
  identity holds trivially when the gate is closed.

### 2.9 So — is the math still Gen6V4/Gen7's?

Precisely:

| component | provenance | modified? |
|---|---|---|
| FiLM block `y₁ ⊙ (1+γ) + β` | Gen7 `unet1d_temporal_film.py` | **no — same class object, imported** |
| visual projection `MLP_c` | Gen7 (grafted verbatim into Gen3v6 body) | no |
| time embedding `MLP_τ` | Gen3v6 == Gen7 | no |
| `+ MLP_h(h)` into the time slot | Gen3v6 | no — same as Gen14 v1 already does |
| interval-CFG terms | Gen3v6/v7 (off by default) | no |
| `dual_head` / shared-trunk `v` head | Gen3v6 | no |
| U-Net skeleton (downs/mid/ups, `Conv1dBlock`, sampling) | Gen3v6 body | no |

**Nothing in the list is newly-authored math.** U5's contribution is a *wiring* file: it
selects Gen7's conditioning operator instead of Gen7's other conditioning operator, and
hands it the Gen3v6 time embedding. Every equation above already existed and had already
been run — just never in this combination.

The claim that is genuinely new, and therefore the one G7 tests, is the **composition**:
that Gen7's multiplicative gate and Gen3v6's forward-mode `h`-derivative are compatible.
§1 argues they are (γ has zero tangent ⇒ `d/dr[(1+γ)f + β] = (1+γ)·df/dr`). G7 checks it
on hardware rather than trusting the argument.

## 3. Changes

### (a) NEW — `mix_visual_aligning/models/unet1d_twotime_film.py`

`Flow_matcher_U_Net_v2_FiLM`: the v2 conditioning route **plus** the two-time surface.

| | v1 two-time | v2 two-time (new) |
|---|---|---|
| visual latent | concat into `t`, then per-block additive bias | per-block `γ` scale + `β` shift |
| `embed_dim` | `dim + cond_embed_dim` | `dim` (time-only) |
| `h_mlp` | ✅ additive into `t` | ✅ additive into `t` — **unchanged** |
| interval-CFG | ✅ additive into `t` | ✅ additive into `t` — unchanged |
| `dual_head` / `return_v` | ✅ shared trunk | ✅ shared trunk |

The FiLM block itself is **imported, not reimplemented**:

```python
from .unet1d_temporal_film import FiLMResidualTemporalBlock
```

That file is a **G0 verbatim copy of Gen7**. So "is mf's v2 the same FiLM as fm's v2?"
has a one-line answer — it is the same class object. Only the surrounding U-Net differs.
This also means a future Gen7 change to the block propagates to both arms by
construction instead of by somebody remembering.

One incidental improvement over the v1 file: v1's `forward` does `h = []` for the skip
stack, **shadowing its own `h` parameter** after use. It works, but it is a trap. The new
file uses `h_stack`.

**§3.1 is not weakened.** The new module is reached only via `visual_unet_twotime.py`,
i.e. only by `mf`/`af`. It *imports* a verbatim file; nothing new enters the
`diffusion`/`fm` import closure. G0 re-run: **PASS, 23/23 verbatim files still match.**

### (b) `visual_unet_twotime.py` — construction-time branch

The `raise` becomes a dispatch mirroring `visual_unet.py:91-97` exactly. Unknown modes
still raise. Absence of the key still means `'v1'`, so **every existing config and
checkpoint is untouched** — including job 24124's.

### (c) `ddpm` → `diffusion`

The name was inconsistent with everything around it: the source folder is
`diffuser_visual_aligning/`, the class is `VisualGaussianDiffusion`, the parent config
block is `visual_aligning_dpcc`, and the Data_Analysis variant names are
`diffuser` / `dpcc-*`. `ddpm` appeared nowhere else in the repo's vocabulary — except
`ddpm_encdec_vision`, which is **a different generation and was left alone** (the rename
was tokenised with that prefix explicitly protected).

Renamed: registry key, `base['mix_visual_aligning_diffusion']` and its plan sibling,
`ENGINE == 'diffusion'` branches in train and eval, G1's expectation table, the three
sbatch `case` arms, and the prose. The `label` keeps DDPM as *description* —
`'Diffusion / DDPM (Gen6V4)'` — because the algorithm genuinely is DDPM; it is the
identity key that was wrong.

**`ddpm` still works as a deprecated input alias** (`ENGINE_ALIASES`), normalised by
`canonical_engine()` at each entry point and by a `case` arm in each sbatch wrapper, both
printing a notice. An alias must **never** reach `exp_name` — that would fork a second
checkpoint tree for one arm — so normalisation happens before anything is named.
`assert_engine_matches` now compares *canonical* keys, so a hypothetical pre-U5
checkpoint recording `engine='ddpm'` still loads.

**Timing is why this is free.** `engine` is a path identity key (`E{engine}`), so the
rename moves the checkpoint tree — `mix_visual_aligning_ddpm/…_Eddpm` →
`mix_visual_aligning_diffusion/…_Ediffusion`. **The `diffusion` arm has never been run.**
Only `mf` seed 6 (job 24124) exists, and its path is byte-identical before and after
(verified by re-deriving `exp_name` from the config). A month from now this rename costs
a checkpoint migration.

### (d) NEW GATE — G7 (`--gate g7`, needs GPU)

Builds **all four** arms at `film_mode='v2'` and asserts:

1. construction succeeds on every arm;
2. `FiLMResidualTemporalBlock`s exist **and** `use_film` is live on all of them — catching
   the failure where v2 is accepted and silently ignored;
3. block `time_mlp.in_features == dim`, not `2*dim`. That width *is* the v1/v2
   difference, so it is the cheapest possible proof the latent left the time path;
   and on the two-time arms, that an `h_mlp` still exists — **the pre-U5 objection,
   turned into an assertion**;
4. one `mf` loss step at v2 is finite: forward-mode AD survives the multiplicative gate.

## 4. Verification (local container — no torch, no GPU)

| check | result |
|---|---|
| `ast.parse` on all 7 edited/new Python files | pass |
| `bash -n` on all 4 sbatch scripts | pass |
| **G0 copy fidelity, re-run** | **PASS — 23/23** |
| registry: `canonical_engine('ddpm'/'DDPM')` → `diffusion` | pass, with notice |
| registry: `resolve('nope')` raises; `assert_engine_matches('mf','diffusion')` raises | pass |
| registry: `assert_engine_matches('diffusion','ddpm')` accepted | pass (back-compat) |
| config: all 4 train + 4 plan blocks resolve; no stale `_ddpm` key | pass |
| config: `exp_name` ↔ `diffusion_loadpath` reproduce key-for-key | pass |
| config: mf path **unchanged** vs job 24124's actual savepath | pass |
| AST signature parity, `Flow_matcher_U_Net_v2` vs `…_v2_FiLM` (`__init__`/`forward`/`get_pred`) | **identical, incl. defaults** |
| new file imports Gen7's FiLM block rather than redefining it | pass |

**Not run locally:** anything needing torch. G1–G7 are cluster jobs.

## 5. Blast radius

**Zero on any existing run.** `film_mode` defaults to `'v1'` and every config still sets
`'v1'`; the `mf`/`fm`/`af` checkpoint trees keep their exact names. The only path that
moves belongs to an arm with no checkpoints.

The v2 path is **opt-in and untested on hardware** — it has never executed a single
tensor op. G7 is the first thing that will.

⚠️ v1 and v2 state_dicts are **not** interchangeable (`embed_dim` differs, `film_proj` is
new). `film_mode` is a path key, so they land in parallel directories automatically — but
a v2 config aimed at a v1 checkpoint still dies in `load_state_dict`.

## 6. Commands

```bash
# G0 + G1 first — the rename touched the dispatch table and the ledger
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh

# then G7 specifically (GPU): the first execution of the v2 path, ever
python -m mix_visual_aligning_test.gates_mix_visual --gate g7
```

To actually train a v2 arm, add one key to the arm's override dict in
`config/aligning-d3il-visual.py` — `'film_mode': 'v2'` — and the parallel tree, the
`diffusion_loadpath` and the eval identity all follow automatically.

🔴 **But not on `mix_visual_aligning_fm` as it stands.** That block carries
`config/aligning-d3il-visual.py:977-978`: *"Must remain a pure re-pointing of
fm_visual_aligning. Do not tune anything here: gate G1 compares this arm against Gen7 and
expects bit-identical training."* A `film_mode` override there will trip G1. A v2 sweep
wants a **fifth block** (e.g. `mix_visual_aligning_fm_filmv2`) rather than mutating the
G1-compared reference arm. Not written — it needs a decision on whether v2 is a fifth arm
or a per-arm flag, and that is a research call.

## 7. Still open

- **G7 has never run.** Everything above is static verification.
- The **fifth-block-vs-flag** decision for running v2 without breaking G1 (§6).
- Unchanged from U4: job 24124's W&B backfill (run `8eb9bo8t`); Gen3v7's `--auto-resume`
  CLI mirror; the Gen7 K=1 projector defect (blocks only a low-NFE sweep); the
  `VisualAgentWrapper` candidate-selection audit against `ecbae16f` / `a6a7a8ad`.
- From the 24124 curve analysis, unrelated to this pass but the strongest open lead:
  **`gradient_clip: 1.0` vs a median pre-clip grad norm of 67** — 100 % of steps clipped,
  ~67× scale-down. Worth resolving before spending GPU on any new arm, v1 or v2.
- **A vision-capable DiT/SiT** — scoped below as a U6 candidate. Not started.

---

## 8. U6 CANDIDATE (scoped, not started) — a vision-capable DiT / SiT

Raised while answering "can `mf`/`af` run the old ML backbone?". Filed here so the
analysis is not lost; **nothing in this section has been implemented.**

### 8.1 Current answer: no, not with vision

`imf_backbone` accepts `'unet' | 'dit' | 'mf_dit'` (`mf`) and `'unet' | 'dit' | 'sit'`
(`af`), and all of those branches exist and work — **but only at `if_vision=False`**,
which is Gen3v6/Gen3v7 re-run inside Gen14's frame. Those generations already do that,
with their own tuned configs.

**This is not a conservative guard — there is no input to attach the image to.** All
three transformer backbones declare `cond` and then never use it:

```python
def forward(self, x, cond, time, returns=None, ...)   # mf_dit, mf_dit_official, af_sit
```

`cond` is signature parity so `MFTrajectoryModel` can call every backbone identically.
Their real conditioning is adaLN-zero from `(t, r, ω)` embeddings, or in-context prefix
tokens. The state-only task never needed a conditioning vector at all — DPCC conditions
by **inpainting** (`apply_conditioning` pins the obs dims), which is why the gap went
unnoticed. Hand a 128-D latent to `MFDiTOfficialTrajectory` today and it is silently
discarded: precisely the "trains image-blind" failure the `ValueError` names.

### 8.2 What it would take

1. **Injection point.** For `mf_dit` / `sit` the natural one is an extra adaLN source:
   `c = t_emb + r_emb (+ w_emb) + MLP_v(v)`. adaLN already sums embeddings, so this is one
   more addend. Zero-init the `v` branch to inherit FiLM v2's identity-at-init property
   (§2.8). `MFDiTTrajectory` is token-based, so an in-context prefix token is more
   idiomatic there.
2. **JVP safety: already argued.** Same structure as §1 — `v_emb` is a captured constant
   with zero tangent, and adaLN modulation is affine in the activations it scales, so
   `du/dr` picks up the scale factor and nothing else. No new term in the identity.
3. **🔴 The real cost — G0.** `mf_dit_trajectory.py`, `mf_dit_official_trajectory.py`,
   `af_dit_trajectory.py` and `af_sit_trajectory.py` are **all in the G0 VERBATIM ledger**.
   Editing any of them breaks copy fidelity. It must follow the copy-modify pattern: a new
   `visual_mf_dit_trajectory.py` (etc.), exactly the way `visual_unet_twotime.py` was made
   from `visual_unet.py`, and exactly the way U5 made `unet1d_twotime_film.py`. One new
   file per backbone.
4. **Path keys first.** Add `imf_backbone` to `args_to_watch_mix_visual_train` **before**
   the first run, or U-Net and DiT checkpoints collide in one directory (§2.1).
5. Relax the `if_vision` guard in `mf_trajectory_model.py:76` / `af_trajectory_model.py:76`
   to admit the new visual-DiT values only, and extend G7 (or add G8) to cover them.

### 8.3 The research trade-off — this is the part that needs a decision

**For:** it removes a real cross-generation confound. Gen3v6's headline `mf` numbers come
from an MF-DiT (`bbmf_dit`, e.g. `CAND_102`); Gen14's `mf` comes from a U-Net. As things
stand, "does MeanFlow transfer to vision?" cannot be answered cleanly, because the answer
is entangled with a backbone swap.

**Against:** it breaks Gen14's locked-backbone premise. Right now all four arms share one
architecture, so the four-way comparison isolates the **objective**. Give a DiT to `mf`/`af`
only and `mf`-vs-`fm` becomes objective *and* architecture — you would have to give
`diffusion`/`fm` a DiT too to keep it controlled, which is most of Gen10's stated scope
(`MASTER_TEST_HISTORY.md:33`).

**Assessment:** genuinely interesting, but Gen15-shaped rather than a U5 addendum — and
note it would be the **first** visual transformer anywhere in this repo (§2.1 scoreboard),
so there is no prior art to copy from internally.

**Cheaper first move, if the confound is the actual worry:** run Gen3v6 *state-only* at
`imf_backbone='unet'` and A/B it against its existing `mf_dit` result. That isolates the
backbone effect on the state task for a fraction of the GPU cost and tells you whether the
confound is even large enough to be worth Gen15. Both configs already exist
(`config/avoiding-d3il.py:644` — the comment there literally says *"use 'dit'/'unet' for
A/B"*), so it is a config flip and one training job, with **no new code at all**.
