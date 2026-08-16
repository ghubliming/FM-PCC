# REPORT — Fix_8: the UNet-width defect (`dim=freq_dim`) across Gen3v4 / Gen3v6 / Gen3v7

**Date:** 2026-08-05 · **Type:** cross-generation defect audit
**Status:** confirmed → **FIXED 2026-08-05**, see [`CHANGELOG_Fix_8_unet_width.md`](./CHANGELOG_Fix_8_unet_width.md)
**Retrieval flag:** every touched line carries `FIX_8_UNET_WIDTH` — `grep -rn FIX_8_UNET_WIDTH`
**Scope:** every code path that builds `Flow_matcher_U_Net_v2` from a MeanFlow-family trajectory model.

> **Short answer to "is Gen3v4 / v5 / v6 all wrong?"**
> **Gen3v4 — every pre-U6 run, but the generation is abandoned** (the UNet was the *only*
> backbone for a month; see §3.1 for why this is a smaller deal than it first looks).
> **Gen3v6 — one run** (the deliberate `bbunet` A/B).
> **Gen3v7 — latent only** (no UNet run exists).
> **Gen3v5 does not exist** — MASTER_TEST_HISTORY line 23 lists it as *BNS Solver, Pending / Pending*.
> Nothing else in the repo is affected; the visual generations (Gen8, Gen14) are clean, and so is
> every FMv3ODE-lineage folder.

---

## 1. The defect

Three trajectory-model wrappers build the DPCC UNet and pass the **frequency-embedding**
hyperparameter into the **channel-width** slot:

| file | line |
|---|---|
| `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` | **83** `dim=freq_dim,` |
| `flow_matcher_v3_meanflow/models/mf_trajectory_model.py` | **96** `dim=freq_dim,` |
| `flow_matcher_v3_alphaflow/models/af_trajectory_model.py` | **99** `dim=freq_dim,` |

`Flow_matcher_U_Net_v2.__init__` uses that one argument for **two unrelated roles**
(`models/unet1d_temporal_cond.py`):

```python
:106   dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]   # ← CHANNEL WIDTH
:110   self.time_dim = dim                                          # ← TIME-EMBED WIDTH
:113   self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim*4), ...)
```

The configs set `'freq_dim': 256` (`config/avoiding-d3il.py:497`, `:611`, `:711`) while every
other UNet in the repo is built at `'dim': 32`. So the MeanFlow-family UNet is **8× wider per
level** — channels `256/512/1024/2048` instead of `32/64/128/256`.

### 1.1 Measured cost

Parameter counts derived analytically from the layer definitions (`Conv1dBlock` = `Conv1d(k=5)`
+ `GroupNorm(8)`, `Downsample1d` = `Conv1d(k=3,s=2)`, `Upsample1d` = `ConvTranspose1d(k=4,s=2)`,
`ResidualTemporalBlock` = 2×`Conv1dBlock` + `Linear(embed→out)` + 1×1 residual conv), with
`transition_dim=6`, `dim_mults=(1,2,4,8)`, `horizon=8`:

| `dim` | `dual_head` | channels | parameters |
|---:|---|---|---:|
| 32 | False | 32 / 64 / 128 / 256 | **3,962,854** |
| 32 | True | 32 / 64 / 128 / 256 | **3,968,268** |
| 128 | True | 128 / 256 / 512 / 1024 | 63,292,428 |
| **256** | **True** | **256 / 512 / 1024 / 2048** | **253,036,556** |

**63.8×.** Width enters quadratically in every conv weight, so an 8× width is a ~64× parameter
count. Against the DiT/SiT arms (`dit_hidden_size: 256`, `dit_depth: 8` ⇒ ≈10 M, as reported in
`Gen3v6_MeanFlow/U2/`; not re-derived here) the UNet arm is roughly **25×** the capacity.

For scale: the D3IL `avoiding` dataset is **96 demonstrations**.

### 1.2 `freq_dim` is not what its name says

The report that raised this described `freq_dim` as *"the time-embedding width the transformer
engines need"*. **That is wrong, and it matters for the fix.** Every occurrence of the symbol,
all three generations:

```
mf_engine.py:28,52,73          signature / docstring / passthrough
mf_trajectory_model.py:31      signature
                       :56     self.freq_dim = freq_dim     ← stored, never read
                       :96     dim=freq_dim                 ← the ONLY consumer, anywhere
```

The DiT / SiT / mf_dit branches take `hidden_size=dit_hidden_size`. **On every non-UNet backbone
`freq_dim` is dead.** Its sole effect in the entire codebase is setting the UNet's width.

Its neighbours in the same config block are worse: `mlp_dim` and `time_dim` are accepted by both
the engine and the trajectory model and are **never used at all**; `depth` and `num_heads` are
stored but shadowed by `dit_depth` / `dit_num_heads`. The whole
`## architecture sizing (UNet arm)` block is inert apart from `freq_dim`.

### 1.3 There is no way to set the width from a config today

Neither `MeanFlowEngine` / `AlphaFlowEngine` / `iMeanFlowEngine` nor their trajectory models
accept a `dim` (or `unet_dim`) keyword, and none takes `**kwargs`. The UNet width is reachable
**only** through `freq_dim`.

## 2. Origin

| commit | date | event |
|---|---|---|
| `d0eff110` | 2026-05-13 | Gen3v4 iMF fix3 — first `freq_dim: 256` in the config. The wrapper still called `Flow_matcher_U_Net_v2(input_dim=…, freq_dim=…, depth=…, mlp_dim=…, time_dim=…)` — **kwargs the class does not accept**, so this could not have run. |
| `1a3fcb58` | 2026-05-13 | *"Update iMFTrajectoryModel to align with Flow_matcher_U_Net_v2 API"* — the API mismatch is repaired by deleting `input_dim/output_dim/freq_dim/mlp_dim/time_dim` and writing `dim=freq_dim`. **The defect is born here.** |
| `87b11a70` | 2026-06-17 | Gen3v4 **U6** adds the config-switchable DiT and makes `'dit'` the default. |
| `832ca559` | 2026-07-22 | Gen3v6 forks from Gen3v4 — inherits the line verbatim; ships with `'dit'`, later `'mf_dit'`. |
| `4342ed4f` | 2026-07-23 | Gen3v7 forks — inherits the line verbatim; ships with `'dit'`, later `'sit'`. |

This is the sibling copy-modify pattern working exactly as designed and propagating a defect
exactly as designed: one bad line in Gen3v4, three copies, zero divergence.

The root cause is a **name collision at a port boundary**. The caller had a variable called
`freq_dim`; the callee had a parameter called `dim`; the two were joined because they looked
alike, not because they mean the same thing. Nothing downstream could catch it — the model
builds, trains, and reports sane-looking losses at any width.

## 3. Per-generation impact

### 3.1 Gen3v4 (iMeanFlow) — **widest exposure, smallest practical cost**

Between `1a3fcb58` (2026-05-13) and `87b11a70` (U6, 2026-06-17) `imf_trajectory_model.py` had
**no backbone selector at all** — the UNet was unconditional. `freq_dim` was `256` from the very
first commit and never changed.

**Therefore every Gen3v4 training run before U6 was a 253 M-parameter UNet on 96 demonstrations.**
That covers the U1–U5 era in `logs_in_develop/Gen3v4_imf/`. Any conclusion from that window about
"the iMF objective", convergence, overfitting, loss shape or sample quality is confounded by a
63× capacity error and should not be cited as an objective result without re-checking the
backbone.

Post-U6 Gen3v4 runs default to `'dit'` (`config/avoiding-d3il.py:547`, plan `:1209`) and are
unaffected.

**Why this is nonetheless the mildest case in practice** (user assessment, 2026-08-05, and it is
the right read): the iMF UNet is believed not to work at *any* width, Gen3v4 is marked
**abandoned** in MASTER_TEST_HISTORY, and nothing downstream depends on those curves. So the
defect changes the *explanation* of the pre-U6 failures, not the *verdict*.

The one thing it does change is evidential independence. Gen3v4's UNet failure and Gen3v6 fix_1's
UNet failure were two separate-looking data points against the UNet; they are in fact **one
untested condition observed twice**. Neither is evidence about a baseline-width UNet.

**No Gen3v4 re-run is planned or recommended.** The fix is applied there anyway, because leaving
one sibling wrong is precisely how this propagated in the first place.

### 3.2 Gen3v6 (MeanFlow) — **one run: the `bbunet` A/B**

Defaults have been `'dit'` then `'mf_dit'` (`:644`) since birth, matched in the plan block
(`:1280`). The UNet branch fired exactly once, deliberately:

> `logs_in_develop/Gen3v6_MeanFlow/fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md`
> train `23813` + eval `23814`, `imf_backbone='unet'`, seed 6, 100 k steps.

That document's central claim is:

> *"Swapping **only the backbone** (DiT → UNet), holding the MeanFlow objective, data, schedule,
> seed, and K=2 eval fixed, breaks training completely."*
> *"This is a clean, controlled result — the two runs differ in exactly one flag."*

**It is not a controlled result.** The two runs differ in one *flag* but two *variables*:
architecture **and** ~25× capacity. Its own evidence is consistent with the capacity error rather
than with the architecture:

| observed in the doc | reading under "UNet can't fit the objective" | reading under "253 M net, 96 demos" |
|---|---|---|
| loss pinned at the adaptive ceiling for 100 k steps | plausible | plausible |
| **best checkpoint at step 3000**, never beaten | plausible | plausible |
| `train/raw_mse_u` 64 → **69.6** (rises), spikes to 5038 | plausible | expected — ill-conditioned at width |
| `grad_norm` 8.9 → **1.5** (decays) | "gives up" | expected between spikes |
| no train/test gap reported | argues *against* pure overfit | consistent with never-fit |

The honest conclusion is narrower and it is still a real finding: **the MeanFlow JVP objective is
unstable on a 253 M UNet.** Whether it is unstable on the *baseline-width* 4 M UNet — the
architecture control this experiment was built to be — **has never been tested.** The §0
verdict ("the analytic-v MeanFlow JVP objective requires the DiT backbone") is not supported and
should be downgraded to "was never run at the baseline's width".

Every other Gen3v6 result — the `mf_dit` headline runs, fix_4, fix_5, the K=2 tables, the whole
DPCC/HardFlow arm comparison — is **untouched**: those never construct a UNet.

### 3.3 Gen3v7 (α-Flow) — **latent, no run affected**

Defaults `'dit'` then `'sit'` (`:761`, plan `:1381`). A repo-wide search finds no α-Flow UNet run;
the only mention of `unet` in `logs_in_develop/Gen3v7_AlphaFlow/` is a checkpoint-collision note
in `U2/CHANGELOG_U2_sit_alphaflow_backbone.md:112`. The defective line exists and would fire the
moment anyone runs the AF backbone A/B.

### 3.4 Not affected — verified, with the reason

| generation / folder | how the UNet width is set | verdict |
|---|---|---|
| Gen0 DPCC (`scripts/`) | `UNet1DTemporalCondModel`, `'dim': 32` (`:128`) | ✅ different class, config-driven |
| FMv3ODE (`flow_matcher_v3/`, `_ode_selectable/`, `_hardflow/`, `_drifting/`, `_uav/`, `flow_matcher_v2/`, `flow_matcher_unet_v2/`) | `Flow_matcher_U_Net_v2`, `'dim': 32` (`:386` etc.) direct from config | ✅ no `freq_dim` in the path |
| **Gen14 Mix-ML visual** (`mix_visual_aligning/`) | `if_vision=True` ⇒ `VisualUNetTwoTime`, which reads `dim=getattr(config,'dim',128)` (`visual_unet_twotime.py:136`) and the config says `'dim': 32` (`aligning-d3il-visual.py:448`) | ✅ **clean — the running pipeline is safe** |
| Gen8 iMF visual (`imf_visual_aligning/`) | `if_vision=True` ⇒ `VisualUNet(vis_config)` | ✅ clean |

**Two dormant copies** carry the same line on branches that are currently unreachable:

- `mix_visual_aligning/models/{mf,af}_trajectory_model.py:117/119` — the `elif imf_backbone == 'unet'`
  branch, reachable only at `if_vision=False`. Gen14 always runs `if_vision=True`, and `:76`
  raises if a non-`unet` backbone is combined with vision, so the branch is dead today.
- `imf_visual_aligning/models/imf_trajectory_model.py:46` — the non-visual fallback, same story.

They are not bugs *now*; they are the same bug waiting for a state-only visual ablation.
**Both were fixed anyway** (2026-08-05) — same line, same flag, zero behavioural change to any
current config, because the branch they sit on is unreachable at `if_vision=True`.

## 4. Consequences worth stating plainly

1. **A "backbone A/B" that was never a backbone A/B.** Gen3v6 fix_1 and (implicitly) the whole
   pre-U6 Gen3v4 era compared architectures while varying capacity 25–64×. The UNet was set up to
   fail and did.
2. **The architecture control for the MeanFlow family does not currently exist.** There is no run
   anywhere that pairs the MF/AF objective with the *baseline-width* DPCC UNet. That is the run
   that would tell you whether MF/AF depend on transformers or merely on not being 253 M.
3. **A whole config block is decorative.** `freq_dim` / `depth` / `num_heads` / `mlp_dim` /
   `time_dim` read as an architecture-sizing block for the UNet arm; four of the five keys do
   nothing, and the fifth does something other than its name.
4. **It is silent.** Wrong width produces no error, no warning, and no log line. Nothing in the
   train logs states the parameter count, so the only way to notice was to read the constructor.

## 5. The fix — **applied 2026-08-05**

File-by-file record: [`CHANGELOG_Fix_8_unet_width.md`](./CHANGELOG_Fix_8_unet_width.md).
Everything carries the retrieval flag `FIX_8_UNET_WIDTH`.

### 5.1 The core: one config value, three blocks

Because `freq_dim`'s only consumer *is* the UNet width (§1.2), the live generations need no code
change at all:

```python
# config/avoiding-d3il.py — flow_matching_v3_imeanflow (:497),
#                           flow_matching_v3_meanflow  (:611),
#                           flow_matching_v3_alphaflow (:711)
'freq_dim': 32,   # UNet CHANNEL WIDTH — its only consumer is dim=freq_dim in
                  # *_trajectory_model.py. DiT/SiT/mf_dit use dit_hidden_size and
                  # ignore this key entirely. 32 = the DPCC/FMv3ODE baseline width.
```

The earlier report proposed adding a new `unet_dim` kwarg threaded engine → wrapper. That is
larger and introduces a second knob shadowing an already-dead one. Preferable instead: **rename**
`freq_dim` → `unet_dim` in a later pass so the key stops lying, and delete `mlp_dim` / `time_dim`
/ `depth` / `num_heads` from the MF/AF blocks. Rename and re-run should not be the same commit.

**One code change was needed after all**, for a reason the config edit cannot reach: the two
visual generations never pass `freq_dim` at all, so their dormant UNet branch fell through to the
signature default of 256. That default is now `32` in all six engines and all six trajectory
models. For the three state-only generations this is inert — their configs always pass the key
explicitly — so it is a pure backstop.

### 5.2 🔴 Path hazard — and why the watch-list token was **rejected**

`freq_dim` is **not** in `args_to_watch_fmv3_mf_train` (`:85-94`), `_af_train` (`:101-112`), or
`_imf_train` (`:70-80`). The path carries `bb{imf_backbone}`, which separates `unet` from `dit`,
**but not the width**. A re-run at `freq_dim: 32` therefore writes into the *same directory* as
the 253 M run.

An earlier draft of this report proposed adding `('freq_dim', 'fd')` to all three watch lists.
**That was checked and abandoned — it would have been worse than the problem it solves.**
`watch()` builds `exp_name` from the whole list, so a new key changes the folder name for
**every** run in all three generations, not only the UNet ones. Every existing `bbmf_dit` /
`bbsit` / `bbdit` checkpoint — including all of Gen3v6's headline results — would become
unreachable at its recorded path, and the matching `diffusion_loadpath` templates would stop
resolving. The guard would have invalidated the results it was meant to protect.

**What the exposure actually is**, traced through the loader
(`FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:113-196`):

- `model = model_config()` — the backbone is rebuilt **purely from the pickled
  `model_config.pkl`** written at train time. The CONFIG-OVERRIDES-PKL reconciliation at `:147`
  operates on `diffusion_config` only and never touches `model_config`.
- `utils/config.py:36-38` writes that pkl **only if it does not already exist**.

So existing 256-width checkpoints still evaluate correctly (their pkl says 256), and new
32-width ones will too. The single bad path is **re-training into a pre-existing directory**: the
stale pkl would still claim 256 while the new checkpoint is 32-wide, and eval would then die in
`load_state_dict` on a shape mismatch. Loud and recoverable, not silent.

**Operational rule instead of a code change:** delete or rename the existing `_bbunet_` trees
before re-running that arm. They are the only affected directories, and they are known-bad.

### 5.3 What was actually applied

1. `'freq_dim': 32` in the three state-only config blocks.
2. Default `freq_dim: int = 256` → `32` in all six engines and all six trajectory models — this
   is what closes the two dormant visual branches, whose configs never pass the key at all.
3. The `dim=freq_dim` line annotated at all six call sites.
4. The §5.4 build-time guard in all six trajectory models.
5. **No** `args_to_watch` change, for the reason in §5.2.

### 5.4 Cheap guard so this cannot recur silently

Print the constructed parameter count at build time in all three wrappers:

```python
n = sum(p.numel() for p in self.velocity_net.parameters())
print(f'[ {type(self).__name__} ] backbone={imf_backbone} params={n/1e6:.1f}M')
```

A 253 M line in a train log next to a 10 M baseline is noticed on the first run. Nothing in the
current logs would have revealed this. Applied in all **six** trajectory models (the state-only
variants also print the backbone name; the visual ones print `vision=`).

## 6. What to re-run, and what not to

| | action |
|---|---|
| Gen3v6 `bbunet` A/B (23813/23814) | **re-run at `freq_dim: 32`** — it is the only way the fix_1 verdict becomes an architecture statement |
| Gen3v6 `mf_dit` headline, fix_4, fix_5, K=2 tables | **nothing** — no UNet involved |
| Gen3v7 | **nothing to re-run**; apply the fix before the AF backbone A/B |
| Gen3v4 pre-U6 (U1–U5) | **do not re-run** — generation is abandoned. Annotate the affected docs instead |
| Gen3v4 post-U6 `dit` | **nothing** |
| Gen14, Gen8, FMv3ODE, DPCC | **nothing** — verified clean (§3.4) |

## 7. Documents that need a correction note

Not edited by this report — listing them so the correction is deliberate:

- `Gen3v6_MeanFlow/fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md` — §0 and §1. §1's "it is a
  true A/B (one variable)" is the specific claim that fails.
- Any Gen3v4 U1–U5 document treating pre-U6 training curves as a property of the iMF objective.
- `MASTER_TEST_HISTORY.md` — Gen3v4 row (line 22) attributes the failure to the objective. Per
  standing convention this file is **not** edited here; flagging only.

## 8. Verification method

Everything above was checked against the working tree at `205c494a`, not inferred:

- constructor argument and its two uses — read directly (`unet1d_temporal_cond.py:106,110,113`);
- `freq_dim`'s complete usage set — exhaustive `grep` over all three model packages, three hits
  per package (signature, unread store, `dim=`);
- absence of a width kwarg — read the full `__init__` signatures of the engines and trajectory
  models; no `dim`, no `**kwargs`;
- parameter counts — recomputed analytically in pure Python from the layer definitions
  (`scratchpad/unet_params.py`), independently reproducing the 4.0 M / 253 M / 63.8× figures;
- affected generations — repo-wide `grep` for `Flow_matcher_U_Net_v2(` call sites (not class
  definitions), then each call site's width source traced to its config key;
- history — `git log -S` on `imf_backbone` and `'freq_dim': ` in `config/avoiding-d3il.py`, and
  `git show` of the pre/post `1a3fcb58` wrapper.

**Not verified:** the ≈10 M DiT figure (cited from `Gen3v6_MeanFlow/U2/`) and every runtime
claim — no Python environment exists in this container. Parameter counts are analytic, not
`sum(p.numel())`. Confirming them on the cluster is a one-line check and worth doing before the
config change is committed.
