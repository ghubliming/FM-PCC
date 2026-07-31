# CHANGELOG — Gen14 coding 1: Visual-Mix-ML frame + four ML engines

**Date:** 2026-07-31 · **Plan:** [`PLAN_Gen14_visual_mix_ml.md`](./PLAN_Gen14_visual_mix_ml.md)
**Status:** code complete, **UNVERIFIED ON HARDWARE** — nothing here has run. Gate G0 is the
only gate executed (it needs no torch); G1–G6 must run on the cluster.

**What was built:** one visual-aligning frame in which the ML engine is a config switch —
`ddpm` (Gen6V4) · `fm` (Gen7) · `mf` (Gen3v6 MeanFlow) · `af` (Gen3v7 α-Flow) — on a locked
`VisualUNet` backbone, so the four-way comparison is architecture-controlled.

**No iMF arm.** Gen8 `imf_visual_aligning/` was treated as dead code: nothing was copied
from it, and nothing imports it.

---

## 1. Impact on existing code: NONE

```
$ git status --porcelain
 M config/aligning-d3il-visual.py      <- +267 lines, 0 deletions (append-only)
?? mix_visual_aligning/
?? mix_visual_aligning_test/
?? Slurm_Codes/sbatch/mix_visual_aligning/
?? logs_in_develop/Gen14/
```

`git diff --numstat` on the one modified file: **`267  0`** — purely additive. Every existing
generation (`fm_visual_aligning`, `diffuser_visual_aligning`, `flow_matcher_v3_meanflow`,
`flow_matcher_v3_alphaflow`, `imf_visual_aligning`) is **byte-untouched**, and their existing
checkpoints and config blocks resolve exactly as before. Gen14 writes to its own
`mix_visual_aligning_<engine>/` prefix.

### The structural guarantee (PLAN §3.1)

> The `ddpm` and `fm` arms import **only verbatim copies**. Every newly-authored line lives
> in a module that only `mf` and `af` import.

This is enforced by the file layout, not by a test, and gate G1 asserts no `ddpm`/`fm`
registry entry reaches a `twotime` module. Consequence: Gen14's reproduction of Gen6V4 and
Gen7 is a property of the copy, not something a numerical test has to establish.

---

## 2. Files created

### `mix_visual_aligning/` — the package

| Kind | File | Provenance |
|---|---|---|
| **V/S** | `datasets/*`, `utils/{arrays,config,constraints_helpers,logger,plot,progress,serialization,setup,training}.py`, `sampling/projection.py`, `models/{helpers,unet1d_temporal_cond,unet1d_temporal_film,visual_unet}.py` | Gen7 `fm_visual_aligning/`, package-name `sed` only |
| **S** | `models/fm_diffusion.py` ← `diffusion.py`, `models/visual_fm_diffusion.py` ← `visual_gaussian_diffusion.py` | Gen7, **renamed** to free the DDPM names |
| **S** | `models/diffusion.py` (`GaussianDiffusion`), `models/visual_gaussian_diffusion.py` | Gen6V4 `diffuser_visual_aligning/` |
| **S** | `models/{mf_diffusion,mf_dit_trajectory,mf_dit_official_trajectory,mlp}.py` | Gen3v6 |
| **S** | `models/{af_diffusion,af_dit_trajectory,af_sit_trajectory}.py` | Gen3v7 |
| **S** | `utils/training_twotime.py` | Gen3v7 `utils/training.py`, **verbatim** |
| **G** | `models/unet1d_twotime_cond.py` | Gen3v6 backbone **+** Gen7 `cond_mlp` graft |
| **G** | `models/{mf,af}_trajectory_model.py`, `models/{mf,af}_engine.py` | Gen3v6/v7 **+** `if_vision`/`vis_config` |
| **N** | `models/engine_registry.py`, `models/visual_unet_twotime.py`, `models/visual_mf_diffusion.py`, `models/visual_af_diffusion.py`, `models/__init__.py` | new |

### `mix_visual_aligning_test/`
- `train_mix_visual_aligning.py` — Gen7's train script + 3 blocks marked `Gen14`
- `eval_mix_visual_aligning.py` — Gen7's eval script (2 838 lines, carries C4/C5/C6/D1/U19) + 2 blocks marked `Gen14`
- `gates_mix_visual.py` — **new**, the G0–G6 battery
- `plot_yaml_constraints.py`, `README_plot_constraints.md`, `constraint_plots/` — Gen7 verbatim

### `Slurm_Codes/sbatch/mix_visual_aligning/`
`train_mix_visual_aligning.sh` · `eval_mix_visual_aligning.sh` · `gates_mix_visual.sh` ·
`mix_visual_aligning_pipeline.sh` — all copied from the Gen7 trio; only the job name and the
final `python` invocation differ. Conda env, `PYTHONPATH`, EGL/GPU isolation guard and W&B
login are inherited **unchanged**.

---

## 3. The four substantive pieces of engineering

### 3.1 The backbone graft — `unet1d_twotime_cond.py`

Verified by inspection that no single existing backbone could serve the two-time visual arms:

| | `h_mlp` (two-time) | `cond_mlp` (visual FiLM) |
|---|---|---|
| Gen7 `UNet1DTemporalCondModel` | ✗ | ✓ |
| Gen3v6/v7 `Flow_matcher_U_Net_v2` | ✓ | ✗ |

So the new file is Gen3v6's backbone copied verbatim with **two additive hunks pasted from
Gen7**: the `cond_mlp` construction (+ `embed_dim = dim + cond_embed_dim`) and the forward
pooling/projection/concat. Verified additive: `diff` against the Gen3v6 source removes
exactly **3 lines**, all part of the intended `embed_dim` replacement.

🔴 **Order dependency, documented in-file:** `h_mlp` and the interval-CFG embeddings are
*added* to `t` (both `[B, dim]`), so they must run while `t` is still `[B, dim]` — before the
cond *concat* widens it to `[B, dim + cond_embed_dim]`. Reordering silently produces a shape
error at the first `ResidualTemporalBlock`. With `use_cond_projection=False` the arithmetic
collapses to Gen3v6's exactly (`dim`, or `2*dim` with returns-conditioning).

Gen7's own `unet1d_temporal_cond.py` was **not edited** — that is what keeps `ddpm`/`fm` clean.

### 3.2 🔴 The pre-encoded visual latent (PLAN §6.1) — the reason `visual_unet_twotime.py` exists

`MeanFlowODE._p_losses_meanflow` differentiates the network with a forward-mode JVP whose
closure captures `cond`. If `cond` carried raw images, **both ResNet-18 encoders would run
inside the JVP** — for a derivative that is identically zero, and through ops whose
forward-mode AD may not even be implemented.

`VisualMeanFlow.loss()` / `VisualAlphaFlow.loss()` therefore call `encode_visual()` **once**
and pass `cond['visual_latent']` (a `(B,128)` tensor). `VisualUNetTwoTime.resolve_visual_cond`
short-circuits on that key. Inside the JVP the latent is a captured constant, so its tangent
is zero **by construction**.

- The pre-encode is **not** wrapped in `no_grad`: the encoder trains end-to-end in Gen6V4/Gen7.
  Capturing the tensor is what zeroes the tangent; freezing would change what is learned.
  `mf_freeze_vision_encoder` exists as an explicit ablation, default **OFF**.
- It matters more for `af`: `compute_u_target` evaluates the net a *second* time per step.
- Also applied at inference, where it is numerically identical (GroupNorm, no dropout,
  images constant across the ODE loop) and turns K ResNet passes per replan into 1.

### 3.3 🔴 `dual_head=True` — caught during implementation

Both Gen3v6 and Gen3v7 ship `dual_head: True` (`config/avoiding-d3il.py:635`, `:754` —
*"the v head carries a FULL loss, not a stabiliser"*, FIX-4). The library default is `False`,
which routes `v` to an orphan MLP on raw `x` that shares nothing with the backbone and
**silently guts half the objective**. Had the default been taken, the mf/af arms would have
trained and produced plausible-looking numbers that were not MeanFlow/α-Flow.

Wired explicitly in the config blocks *and* in the train script, with the rationale inline.
`interval_cfg=False` likewise (both gens run without CFG; on the UNet arm the flag changes
the `state_dict`).

### 3.4 🔴 The α-schedule budget binding (PLAN §6.2a)

`af_diffusion` asserts `af_n_train_steps == af_alpha_end_step`. They coincide today only by
accident (both 1e5). Now both derive from a single name `_MIX_N_TRAIN_STEPS` in the config,
and the train script re-derives `af_alpha_end_step` from `args.n_train_steps` and warns if it
had to override. Raising the budget can no longer produce an anneal that finishes early and
silently trains the back half as pure MeanFlow.

**Also caught:** α-Flow uses *different names and different values* for the same ideas as
MeanFlow — `af_ratio_fm` vs `meanflow_data_proportion`, and `af_adp_eps=1e-3` vs
`mf_adp_eps=0.01`. `af_diffusion.py:97` explicitly forbids harmonising the eps. The train
script keeps the two blocks separate with that warning inline.

---

## 4. Config: `config/aligning-d3il-visual.py` (+267, −0)

Eight blocks (4 train + 4 plan) plus helpers. The `ddpm` arm inherits from
`visual_aligning_dpcc` (**not** the FM block) so it is genuinely the Gen6V4 baseline —
`action_weight=10`, live `n_diffusion_steps=100`.

**The path-drift trap is now unrepresentable.** Every `diffusion_loadpath` and plan `prefix`
is *derived* from `args_to_watch_mix_visual_train` by `_mix_loadpath()`, and every training
identity value is *mirrored* into the plan block. The historical failure — a plan block whose
loadpath does not reproduce `args_to_watch` key-for-key, resolving to a non-existent directory
minutes into a GPU allocation — cannot occur because there is only one list. Verified:

```
train exp_name : mix_visual_aligning_mf/H8_D…VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal
plan  loadpath : f:mix_visual_aligning_mf/H{horizon}_D{diffusion}_a{…}_b{…}_aw{…}_V{…}_steps{…}_bs{train_batch_size}_film{…}_E{engine}_ts{t_schedule}
```

The `ddpm` plan block **drops** every continuous-time key inherited from the FM template
(`flow_steps_v3`, `ode_solver_*`, the Beta params) so they reach neither the folder name nor
the constructor.

---

## 5. Verification actually performed

| Check | Result |
|---|---|
| `py_compile` on every new/changed `.py` | **PASS** (one pre-existing `SyntaxWarning` in Gen7's `projection.py`, carried over verbatim) |
| **Gate G0** — copy fidelity, 23 verbatim files vs sources | **PASS** |
| AST import-graph check across the package | **PASS** — all relative imports resolve to defined names |
| All 12 `engine_registry` dotted targets resolve | **PASS** |
| Config parses; all 8 blocks build; paths inspected | **PASS** (with stubbed `yaml`/`diffuser.utils`) |
| `bash -n` on all 4 sbatch scripts | **PASS** |
| `git diff --numstat` on the one touched file | **267 / 0** |

**Not verified — needs the cluster:** every forward/backward pass, the JVP through the visual
backbone (G2), the h=0 identity (G3), the α schedule at runtime (G4/G5), the projector at K=1
(G6), and any training or eval. **No GPU work has been done.**

---

## 6. Honest deviation from the plan's line budget

The plan budgeted **~390 newly-authored lines**; the actual figure is higher:

| | total lines | code lines (comments/docstrings excluded) |
|---|---|---|
| 5 new package files | 700 | 280 |
| `gates_mix_visual.py` | 264 | 264 |
| 5 graft files (added lines) | 138 | ~60 |
| train script delta | 177 | ~90 |
| eval script delta | 53 | ~20 |
| **total** | **~1 330** | **~715** |

Where it went, honestly:
- **The gates battery (264 lines) was under-estimated at 120.** It is test infrastructure, not
  on any production path.
- **The train-script delta is 177, not ~40.** The four arms genuinely need different engine
  kwargs (§3.3, §3.4), and folding those into the registry would have put `mf`-only knobs on a
  path the `fm` arm reaches — breaking §3.1. Splitting them out was the correct trade.
- Roughly **46 % of the total is comments**, in line with the house style of the files copied.

The *principle* held where it matters: **23 files are byte-verbatim**, the two reference arms
touch no new code, and no existing generation was modified. But the budget number in the plan
was too optimistic and should be read as ~700 code lines, not 390.

---

## 7. Known issues and open items

1. 🔴 **Gen7/Gen6V4 upstream bug, deliberately NOT patched here.** Gen3v6's sampler guards the
   projector with `... or (loop_idx == flow_steps - 1)` (`mf_diffusion.py:283`); Gen7's FM loop
   has only the threshold term (`fm_diffusion.py:178`). **At `flow_steps_v3=1` with a low
   threshold the DPCC projection can be skipped entirely**, and the run reports "FM is unsafe"
   when nothing was ever projected. Gate G6 detects it and its `fm` leg is *expected* to
   report the missing fallback. Fixing it inside Gen14 would breach the verbatim rule — **it
   belongs upstream in Gen7 as its own hotfix**, after which Gen14 re-copies.
2. ⚠️ **Cross-arm split confound.** `mf`/`af` use the two-time trainer's seeded split
   (`split_seed=42`); `ddpm`/`fm` use Gen7's unseeded one. Compare arms on **unguided task
   success**, never on `test_loss`. (PLAN §4 — accepted and documented rather than fixed,
   because fixing it means editing Gen7's verbatim trainer.)
3. ⚠️ **`film_mode='v2'` is rejected on `mf`/`af`** with an explicit error:
   `unet1d_temporal_film.py` has no `h_mlp`, so v2 would silently drop the h-conditioning.
   v1 only for the two-time arms; `fm` can still run v2.
4. **Not done, deferred:** the HardFlow sampler on the visual normalizer (needs a linear-dynamics
   refit — its own generation), the `mf_dit`/`af_sit` backbones (no visual conditioning path),
   and the `VisualAgentWrapper` candidate-selection audit against `ecbae16f` / `a6a7a8ad`.
5. **`MASTER_TEST_HISTORY.md` not edited** — the draft row sits in the plan, unapplied.

---

## 8. SLURM commands

All commands assume repo root, submitted through the standard wrapper.

### Quick test first (do this before anything else)

```bash
# Gates only — minutes, 1 GPU. Runs G0-G6. Non-zero exit if any gate fails.
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh

# The no-GPU subset (G0, G1, G4, G6) if you just want the fidelity + wiring check:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh static
```

G0 also runs locally in this container (no torch needed):

```bash
python3 mix_visual_aligning_test/gates_mix_visual.py --gate g0
```

### Single-arm train / eval

```bash
# train:  <engine> [seed]   engine = ddpm | fm | mf | af   (defaults: fm, 6)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh mf 6

# eval:   <engine> [seed] [record_mode]   (record_mode default: all)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6
```

⚠️ `--engine` at eval **must** match the arm the checkpoint was trained with; the script
asserts on it rather than dying later inside `load_state_dict`.

### Full run — gates → train → eval, chained per arm

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh mf 6
```

The four arms are independent and can run concurrently (separate checkpoint trees):

```bash
for E in ddpm fm mf af; do
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh $E 6
done
```

### Recommended first sequence

1. `gates_mix_visual.sh` — must be green, especially **G2** (JVP survives the visual path).
   If G2 fails with `NotImplementedError: jvp for aten::…`, stop: the pre-encode is not
   reaching the backbone and nothing downstream is meaningful.
2. `train_mix_visual_aligning.sh fm 6` — the reference arm. Compare its `losses.pkl` against
   the existing Gen7 run at the same seed; they should agree.
3. Only then the `mf` and `af` arms.

**Reading results:** track `raw_mse_u`, never `diffusion_loss` — the adaptive loss is pinned
at its ceiling by construction. For `af` also watch `alpha` (must move 1 → 0) and `clamp_frac`
(rising ⇒ the bootstrap is diverging before the anneal reaches it). Per the Gen13 verdict,
**rank arms by unguided task success only**.

### Output locations

```
logs/aligning-d3il-visual/mix_visual_aligning_<engine>/<exp>/<seed>/          # checkpoints
logs/aligning-d3il-visual/plans/mix_visual_aligning_<engine>/<exp>/results/<seed>/   # eval
```
