# INVESTIGATION 2026-09-03 — porting the AF-**U-Net** result to UAV (Gen15) and Visual-Aligning (Gen14)

**Type:** investigation / readiness audit. **No code, config or sbatch file was changed by this document.**
**Trigger:** AF-UNet on `avoiding-d3il` now beats MF-UNet at K=1 with α actually enabled
([`DA/DA_20260903_AF_UNet_alphaflow_ENABLED_seed6_diffuser.md`](DA/DA_20260903_AF_UNet_alphaflow_ENABLED_seed6_diffuser.md)).
Question asked: **can that be extended to the UAV and to Visual-Aligning, and do those two carry the
same defect that was just fixed on `avoiding`?**
**Method:** git history + config/code read only. Nothing was executed (no Python here — cluster only).

---

## 0. TL;DR

| | **UAV — Gen15 `mix_uav`** | **Visual-Aligning — Gen14 `mix_visual_aligning`** |
|---|---|---|
| **AF backbone today** | **`sit`, hardcoded** (`config/uav_mix.py:432`, plan mirror `:465`) — your assumption is **correct**, it is a pure SiT run | **`unet` (visual U-Net) by default** (`_ml_bone('af')` → `'unet'`, `config/aligning-d3il-visual.py:1257-1277`) — your assumption is **correct** |
| **Param-matched to the 4.0 M U-Net?** | ❌ No — `dit_hidden_size=256, dit_depth=8` ≈ **9.4 M** by the config's own `18·depth·d²` rule. The config itself calls this arm *"the deferred appendix arm … never the architecture-matched claim"* (`uav_mix.py:427-431`) | ✅ Yes — `VisualUNetTwoTime`, `dim=32`, FiLM `v1`, ~4.0 M; and the optional DiT/SiT bone is deliberately sized to **160/8 ≈ 3.9 M** to stay matched |
| **Same α-never-on defect?** | 🔴 **Yes, and in its worst form** — `af_alpha_end: 0.0` is hardcoded (`:441`, `:464`) and **`config/uav_mix.py` reads no environment variable at all**. There is no knob. Every UAV `af` checkpoint ever trained is a **MeanFlow** model in an α-Flow folder | 🟡 **The knob already exists** (Gen14 U10, commit `d3ac1c3f`): `MIX_AF_ALPHA_{SCHED,INIT,END,CLAMP,GAMMA}`. The **default is still `0.0`** (`:1615`), so every V_A `af` run that did not set it is also MeanFlow |
| **Already tested with α on?** | ❌ Never | ⚠️ **Yes — and it failed** (`DA_20260831_…U10_alpha_const…` Part 1). **But it tested a different recipe** than the one that just won (§2.3) |
| **Path-collision safe if we turn it on?** | ✅ Yes — `('af_alpha_end','ae')` and `('imf_backbone','bb')` are **unconditional** `exp_name_tokens` (`mix_uav/models/engine_registry.py:316-317`) | ✅ Yes — U10's conditional `af_alpha` tag (`_AFend0p2`) moves checkpoint + `plans/` + `diffusion_loadpath` together |
| **Blocking gap** | needs (a) an `af_alpha_end` knob, (b) an `imf_backbone` knob to get AF+U-Net at all, (c) an epoch override | needs an **epoch override** — `diffusion_epoch: 'best'` is hardcoded and `best` is structurally α-biased (§A.4) |

> **Bottom line.** Neither generation can reproduce the winning `avoiding` recipe today.
> V_A is **one missing knob** away (epoch). UAV is **three** away, and its AF arm is on the wrong
> backbone for the architecture-matched claim in the first place.

---

# §A — What the fix actually was (Gen3v7 trace)

## A.1 The defect

`af_alpha_end = 0.0` anneals α to **exactly** zero. `_get_ratio`'s symmetric clamp
(`af_diffusion.py:472-476`) snaps anything below `af_alpha_clamp=0.005` to `0.0`, and
`af_diffusion.py:552` routes `alpha <= 0.0` into a branch whose own comment reads
*"Gen3v6's `_p_losses_meanflow` body, **UNMODIFIED**"*.

At γ=25 over 100 k steps the α=0 tail starts at step **71 173**, and the last checkpoint that is
ever written is step **80 000** (§A.4). So the final ~9 k gradient steps before the deployed weights
were **pure MeanFlow**:

> **Every α-Flow checkpoint this project had ever evaluated was a MeanFlow model.**
> Machine-readable proof: `train/discrete_frac = 0.0` and the savepath token `_ae0.0_`.

## A.2 The nuance that must survive into any port

This is **not a porting bug.** Verified against upstream in the 09-01 DA §3.1:
`aux_repo/alphaflow` ships `end_value: 0` in every recipe, and α=0 routes to its own JVP
mean-velocity target. **α-Flow is a curriculum whose destination *is* MeanFlow.** Its claim is that
the route `FM → bootstrapped → MeanFlow` reaches a better model than training MeanFlow directly.

So `AF_ALPHA_END > 0` is **a deliberate deviation from the paper**, not a correction of ours. It is
sanctioned only because upstream ships `discrete_training: true`
(`aux_repo/alphaflow/src/training/loss.py:421-426`), whose purpose is exactly to floor α instead of
snapping it to zero. Any UAV/V_A write-up must carry the same framing or it misstates the method.

**Why it was worth doing anyway:** on our 4.0 M U-Net the curriculum's only distinctive signal is
the bootstrap probe `dt = α·h ≈ 0.0013–0.013`, while the U-Net's time code is
`SinusoidalPosEmb(dim)` with `dim = freq_dim = 32` → ~4 resolving frequencies on [0,1] vs the SiT's
~32. The middle of the curriculum teaches the U-Net almost nothing while still moving its weights.
`α_end = 0.2` makes the probe **4×** larger (0.05 instead of 0.0126) — the 0.05-vs-0.2 pair was run
precisely to measure that. 🔒 The U-Net itself was **not** touched; param count stayed 4.0 M.

## A.3 The fix, mechanically (commit `beb7f26c`, 2026-09-01)

| file | change |
|---|---|
| `config/avoiding-d3il.py:114-129` | `_af_alpha_end = float(os.environ.get('AF_ALPHA_END', 0.0))` — default keeps every existing path byte-identical |
| `config/avoiding-d3il.py:952` | training block: `'af_alpha_end': _af_alpha_end` |
| `config/avoiding-d3il.py:1599` | **plan block mirror** — must match or eval finds no weights |
| `config/avoiding-d3il.py:249` | `('af_alpha_end', 'ae')` is an **unconditional** `args_to_watch` token ⇒ each value gets its own `_ae<val>` tree ⇒ no `--auto-resume` collision |
| `Slurm_Codes/sbatch/AlphaFlow/{train,eval}_alphaflow.sh` | echo `AF_ALPHA_END` so the log is self-describing |

Supporting commit `e9440fff` added `AF_SEEDS`; the same DA wired `AF_NTRIALS`. Both exist because
`config/alphaflow_projection_eval.yaml` is read at **job runtime**, so a submit-time `sed` would be
read hours later by the dependent eval — two arms could not otherwise be queued together.

## A.4 Two supporting faults, both of which apply verbatim to Gen14/Gen15

1. **`latest` is step 80 000, never 100 000.** `save_freq = n_train_steps // 5` = 20 000 and the
   loop ends at 99 999, so `step % 20000 == 0` never fires at 100 k. We always deploy 20 % short.
   → **Gen15 has the identical line**: `mix_uav/utils/training_twotime.py:81` + `:203`, loop
   `for step in range(n_train_steps)` (`:171`).
   → **Gen14 mitigates it**: `mix_visual_aligning/utils/training.py:96` accepts `save_freq`
   (CLI `--save-every`, env `MIX_SAVE_EVERY`), defaulting to the same `n//5`.
2. **`best` is structurally the *wrong* checkpoint for an AF run.** `state_best.pt` is selected on
   `test_loss`, which for AF ≈ `0.75 + 0.25·α` — i.e. it **prefers a mid-homotopy α ≈ 0.01–0.02
   model**, one caught *inside* the curriculum rather than after it. The winning `avoiding` runs
   used `AF_EPOCH=latest`. 🔴 **Neither Gen14 nor Gen15 has any epoch override** —
   `'diffusion_epoch': 'best'` is hardcoded at `config/uav_mix.py:174` and
   `config/aligning-d3il-visual.py:628,679,741,829`, and neither eval script reads an env var for it.

## A.5 The verification checklist (reuse unchanged in both ports)

| signal | ❌ α off | ✅ α on |
|---|---|---|
| train banner `AF_ALPHA_END=` | `0.0` | `0.05` / `0.2` |
| `val/alpha`, final epoch | `0.0006` | ≈ `0.05` / `0.2` |
| **`train/discrete_frac`, final epochs** | **`0.0`** | **0.25 – 0.41** |
| savepath token | `_ae0.0_` | `_ae0.2_` |

`discrete_frac` is the batch fraction taking the bootstrapped no-grad branch. **If it is 0.0 at the
end of training, the run is MeanFlow regardless of what the folder is called.** This is the single
check that would have caught the defect years earlier, and it is the one to demand from any
UAV/V_A α-run before reading a single task number.

## A.6 What the fix bought on `avoiding` (the thing we are trying to extend)

- **Raw `diffuser` arm:** AF `α→0.2` **Pareto-dominates** MF-UNet at K=1, 5, 10, with **identical
  `avg_time`** — the extra `no_grad` forward is a *training-time* cost only.
- **Projected:** at K=1, `dpcc-t-tightened` posts S&C 1.00 / 57.20 steps / 1.07 s/ep on
  `top-left-hard`, **beating the pinned DPCC K20 target 33.3× cheaper**.
- ⛔ **Two caveats that also gate any port:** n=20 on a **single seed**, and `n_steps` has **two
  incompatible definitions** across the toolchain.

---

# §1 — UAV / Gen15 (`mix_uav` ↔ `mix_uav_test`, `config/uav_mix.py`)

## 1.1 Backbone: yes, it is a pure SiT run — and that is a claim problem, not just a fact

```python
# config/uav_mix.py:426-432  (arm 'mix_uav_af')
# 🔴 OVERRIDES _TWO_TIME_BACKBONE's 'unet' for THIS ARM ONLY — `mf` keeps 'unet' …
# 'sit' is alpha-Flow's OWN backbone (af_sit_trajectory.py, Gen3v7 U2); it sizes from
# dit_hidden_size/dit_depth, NOT from freq_dim, so this arm is NOT parameter-matched to the
# 4.0 M U-Net rows — it is the deferred appendix arm (PLAN §6), never the architecture-matched claim.
'imf_backbone': 'sit',
```

- `fm` and `mf` sit on `unet` (`freq_dim=32`, ~4.0 M — post-Fix_8, so **not** the 253 M defect).
- `af` sits on SiT at `dit_hidden_size=256, dit_depth=8` → **≈ 9.4 M**, ~2.4× the U-Net.
- The plan block `:465` mirrors `'imf_backbone': 'sit'` (it must, or eval rebuilds a different savepath).

**Consequence:** the current UAV AF-vs-MF row is *objective × backbone × params* confounded. It can
never be the headline. **Porting the `avoiding` result to UAV means running AF on the U-Net bone**,
which is exactly the arm that does not exist yet.

✅ The machinery for it is already present: `mix_uav/models/engine_registry.py` declares
`backbones=('unet','dit','sit')` for `af`, and `mix_uav/models/unet1d_twotime_cond.py` +
`af_trajectory_model.py` are in the tree. It is a config change, not a port.

## 1.2 The α defect: present, and with no escape hatch

```python
# config/uav_mix.py:441  (train block)      # :464 (plan block)
'af_alpha_end': 0.0,         # alpha at the end  (0.0 ⇒ end as MeanFlow)
```

`grep -n "os.environ" config/uav_mix.py` → **no matches**. The whole file is env-free, so unlike
`avoiding` and unlike Gen14 there is **no knob at any level**: not env, not `--engine`-adjacent CLI
(`mix_uav_test/train_mix_uav.py` exposes only `--engine/--scene/--seed(s)/--resume*/--wandb*`), not
sbatch (`train_mix_uav.sh` forwards `$ENGINE $SCENE $SEEDS` only).

`mix_uav/models/af_diffusion.py:568` is the same `if alpha <= 0.0:` → *"Gen3v6's
`_p_losses_meanflow` body, UNMODIFIED"* routing. So:

> **Jobs 25135–25138 (`af` on `pillars`) and 24xxx (`af` on `corridor`) trained and deployed
> MeanFlow-on-SiT.** The UAV "α-Flow" rows published to date compare *MeanFlow-on-SiT (9.4 M)*
> against *MeanFlow-on-U-Net (4.0 M)* — an architecture ablation wearing an objective's name.
> This is the strongest single finding in this document and it invalidates the *label*, not the
> numbers, of every UAV `af` row.

## 1.3 What is already safe

`mix_uav/models/engine_registry.py:316-317`:

```python
exp_name_tokens=(('af_alpha_init','as'), ('af_alpha_end','ae'), ('imf_backbone','bb')),
```

`_uav_mix_exp_name` (`config/uav_mix.py:91-108`) renders these **unconditionally** (`if hasattr`).
So both `af_alpha_end` and `imf_backbone` already land in the checkpoint/plan folder name — a new
value trains into a new tree and **cannot overwrite or auto-resume into** an existing run. The
`avoiding` fix relied on the identical guarantee. **No path-safety work is required for Gen15.**

## 1.4 The three gaps, in order of size

| # | gap | where | size |
|---|---|---|---|
| 1 | no `af_alpha_end` override | `config/uav_mix.py:441` + `:464` | ~15 lines, mirroring `avoiding-d3il.py:114-129`; must touch **both** blocks |
| 2 | no `imf_backbone` override for the `af` arm | `config/uav_mix.py:432` + `:465` | ~10 lines; the registry already validates the value against `backbones=(…)` |
| 3 | no epoch override; `latest` is step 80 k | `config/uav_mix.py:174`; `mix_uav/utils/training_twotime.py:81` | ~10 lines. Without it the eval loads the **α-biased `best`** checkpoint and the experiment is unreadable (§A.4) |

Gap 3 is the one that quietly ruins the run: with `af_alpha_end=0.2` the deployed weights *must* be
the post-curriculum ones, and `best` will not give them.

## 1.5 🔴 Scene confound — do not run this on `pillars` yet

[`../Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af.md`](../Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af.md):

- **0 success+constraint rollouts out of 1707**, every engine, every K, every projector; only
  2/1707 collision-free. **No ranking is possible from that batch.**
- **No `diffusion` (DPCC/GaussianDiffusion) arm exists for `pillars` at all** → the pinned DA target
  (per `da-target-is-best-baseline-variant`) is **missing** for this scene.
- Divergence aborts fire on 14–78 % of trials, 83 % of them on the uncalibrated `inverted` trigger.

Fix_16 (`d2102257`, [`../Gen15/fix_16/CHANGELOG_fix16_degenerate_action_channel.md`](../Gen15/fix_16/CHANGELOG_fix16_degenerate_action_channel.md))
addresses a *separate* Gen15 defect — `SafeLimitsNormalizer(eps=1)` turning the constant `Δz`
channel into the loudest output in the action space, giving it a ±1 m ceiling against ~0.044 m on
the real channels. It is checkpoint-compatible and needs no retrain, and its A/B is in
[`../Gen15/DA/DA_20260903_fix16_AB_mf_pillars.md`](../Gen15/DA/DA_20260903_fix16_AB_mf_pillars.md).
It is **not** the α defect and fixing one does not touch the other — but it does mean **`pillars`
numbers taken before fix_16 are not a baseline** for an AF-UNet arm.

**Recommendation: land the AF-UNet UAV experiment on `corridor` first**
([`../Gen15/DA/DA_20260824_af_sit_K_sweep_corridor.md`](../Gen15/DA/DA_20260824_af_sit_K_sweep_corridor.md) gives an existing AF-SiT ladder to compare against, and
[`../Gen15/DA/DA_20260820_fm_K_sweep_corridor.md`](../Gen15/DA/DA_20260820_fm_K_sweep_corridor.md) an FM one), then `s_curve`, and only revisit `pillars` once
fix_16's A/B is accepted.

## 1.6 Sketch of the experiment (for when the knobs exist — **not run, not submitted**)

```bash
# arm A — AF on the U-Net bone, α floored (the avoiding recipe, transplanted)
UAV_AF_BONE=unet UAV_AF_ALPHA_END=0.2 UAV_EPOCH=latest \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_pipeline.sh af corridor 6
# arm B — the α=0.05 probe, to separate "α on" from "probe big enough"
UAV_AF_BONE=unet UAV_AF_ALPHA_END=0.05 UAV_EPOCH=latest  … (same)
# control — AF U-Net with α→0, i.e. today's objective on the matched bone.
#           REQUIRED: without it the comparison confounds α with the backbone change.
UAV_AF_BONE=unet UAV_AF_ALPHA_END=0.0  UAV_EPOCH=latest  … (same)
```

⚠️ Env names above are **proposals**, not existing variables. Gen15's house style is bare config
literals; whether to follow `avoiding`'s `AF_*` prefix or introduce `UAV_*` is a decision for the
user, and it fixes the path tokens forever, so it should be made once.

---

# §2 — Visual Aligning / Gen14 (`mix_visual_aligning` ↔ `mix_visual_aligning_test`, `config/aligning-d3il-visual.py`)

## 2.1 Backbone: yes, it is the visual U-Net

`_ml_bone(engine)` (`config/aligning-d3il-visual.py:1257-1277`) resolves
`MIX_BONE_AF` → `MIX_BONE` → **`'unet'`**, and `_MIX_ML_BONES['af'] = ('unet','sit','dit')`.
On the `unet` bone `_mix_bone_keys` deliberately emits **no** `ml_bone` token, so the path stays
byte-identical to every pre-U8 run — which is why the DA paths read `filmv1` with no `_B` fragment.
Confirmed empirically in `DA_20260831_…U10…` §1.1: *"ML bone `unet` — `VisualUNetTwoTime`, FiLM
`v1`, `dim=32`, `dim_mults=(1,2,4,8)`"*.

So **V_A's AF arm is already on the architecture-matched ~4.0 M visual U-Net** — the same class of
model that just won on `avoiding`. If you want the SiT comparison it is `MIX_BONE_AF=sit`, and the
transformer bone is sized `dit_hidden_size=160, dit_depth=8` ≈ **3.9 M** *specifically* to stay
param-matched (the config cites Fix_8's public retraction as the reason). **Gen14 got this right;
Gen15 did not.**

`VisualAlphaFlow` subclasses `AlphaFlowODE` (`mix_visual_aligning/models/visual_af_diffusion.py:25`),
so the `alpha <= 0.0` → MeanFlow routing at `mix_visual_aligning/models/af_diffusion.py:552` is
inherited unchanged. Same engine, same defect surface.

## 2.2 The α defect: the knob exists, the default does not use it

Gen14 U10 (`d3ac1c3f`, [`../Gen14/U10/CHANGELOG_Gen14_U10_af_alpha_schedule_knob.md`](../Gen14/U10/CHANGELOG_Gen14_U10_af_alpha_schedule_knob.md))
shipped five env knobs **before** `avoiding` did, and for the same reason — U5 had measured the
snap's cost directly:

| step | α | val `raw_mse_u` |
|---:|---:|---:|
| 70 000 | 0.0067 | **2.657** ← best the anneal ever reached |
| **72 000** | **0.0** | **8.504** — a **3.2×** jump that never recovers |

`MIX_AF_ALPHA_{SCHED,INIT,END,CLAMP,GAMMA}` (`:1462-1470`) with validation
(`[0,1]` range checks, a hard refusal for bare `SCHED=constant`, and a refusal when a held α sits
below the clamp) and a **conditional** path tag `af_alpha = 'AF' + …` that joins
`args_to_watch_mix_visual_train`, so checkpoint tree, `plans/` tree and `diffusion_loadpath` move
together. Defaults emit nothing ⇒ every pre-U10 path, `cand6` included, is untouched.

**But `af_alpha_end` is still `0.0` at `:1615`.** Any V_A `af` run that did not set the env — which
is every run except U10's cand5 — **deployed MeanFlow**, exactly as on `avoiding`.

*Cosmetic nit, worth a one-word fix if the file is ever touched:* the emitted fragment reads
`_afschconstant_AFAFconst0p05` — the watch label `AF` is prepended to a value that already starts
with `AF`. Harmless, ugly, and now frozen into cand5's path.

## 2.3 🔴 The important finding: V_A already ran an α-enable experiment, and it is **not** the recipe that won

[`../Gen14/DA_20260831_Gen14_U10_alpha_const_and_U11_K100_projection_budget.md`](../Gen14/DA_20260831_Gen14_U10_alpha_const_and_U11_K100_projection_budget.md)
Part 1, jobs 25239–25242, 320 exactly-paired rollouts, one-knob design:

> **Verdict: the U10 repair did exactly what it promised to the loss and nothing good to the robot.**
> `raw_mse_u` 8.504 → **2.626** (3.2×). Distance did not move (`p = 0.67`). Constraint satisfaction
> **fell 0.444 → 0.284**, `p = 7.0e-6`. `avg_time_ms` rose 43.4 → 53.5.

That is a real, well-controlled negative result — and it is **not evidence against the `avoiding`
recipe**, because the two configurations differ in almost every way that matters:

| | **Gen14 U10 cand5 — failed** | **Gen3v7 09-03 — won** |
|---|---|---|
| `af_alpha_scheduler` | **`constant`** | `sigmoid` |
| `af_alpha_init` | **0.05** | **1.0** |
| `af_alpha_end` | 0.05 | **0.2** (and a 0.05 arm) |
| α(t) | **flat 0.05 for all 100 k steps** | 1.0 → floor 0.2 |
| pure-FM head | **none** | ~47.8 % of steps at α = 1 |
| genuine bootstrap phase | the whole run, at a *fixed* near-MF α | ~29.4 % of steps, sweeping the homotopy |
| curriculum | **destroyed** | intact, just floored instead of snapped |
| probe `dt = α·E[h]` | 0.0126 | **0.0501** (4×) |
| deployed checkpoint | **`best`** (α-biased) | **`latest`** |
| K reported | 2 | 1 / 2 / 5 / 10 / 20 |

**What U10 settled:** *the α→0 snap is not why α-Flow underperforms on visual-aligning* — removing
the snap left task numbers flat and made constraints worse. That hypothesis is dead.
**What U10 did not test:** the floored **anneal**. Its own §1.7 proposes exactly the right shape —

```bash
MIX_AF_ALPHA_END=0.02 MIX_AF_ALPHA_CLAMP=1e-4 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 6
```

— and it **was never run** (no `_AFend` tag appears anywhere in `logs_in_develop/` or
`Data_Analysis/`). Note also that **0.02 is 10× below the 0.2 that won**, and by the resolution
argument (§A.2) 0.02 is comfortably inside the regime the U-Net's 4-frequency time code cannot see.
If it is run, run **0.2** alongside it.

⚠️ One inherited caution: cand5's constant-α arm *also* cost ~23 % `avg_time_ms`. On `avoiding` the
floored-anneal arm cost **zero** time (identical `avg_time` at every K). If V_A repeats the
slowdown, that would be a genuinely new signal and would need explaining before any claim.

## 2.4 The one blocking gap

**Epoch selection.** `'diffusion_epoch': 'best'` is hardcoded at `:628, :679, :741, :829`, and
`mix_visual_aligning_test/eval_mix_visual_aligning.py:2932` passes `args.diffusion_epoch` straight
through — no env var, no CLI flag. Per §A.4, `best` on an AF run selects a **mid-curriculum**
checkpoint, which is precisely the model the floored anneal is trying to move past. **A floored-α
V_A run evaluated at `best` would not test the hypothesis.**

Partial mitigation already present: `MIX_SAVE_EVERY` / `--save-every`
(`mix_visual_aligning/utils/training.py:96`) can checkpoint more often, so a true final-step
checkpoint is reachable. That fixes the *availability* of `latest`, not the *selection* of `best`.

## 2.5 Readiness summary

| item | status |
|---|---|
| architecture-matched U-Net bone | ✅ already the default |
| α floor knob | ✅ `MIX_AF_ALPHA_END`, validated, path-tagged |
| clamp knob (so a small floor is not snapped away) | ✅ `MIX_AF_ALPHA_CLAMP` |
| path/auto-resume safety | ✅ U10's `af_alpha` tag |
| **deploy the post-curriculum checkpoint** | ❌ **missing — the blocker** |
| a pinned DPCC target for the scene | ✅ Gen6V4 diffusion arm exists |
| prior α evidence | ⚠️ one negative result, on a different recipe (§2.3) |

---

# §3 — Cross-cutting risks

1. **Single seed.** The `avoiding` win is seed 6, n=20, one seed. Do not spend cluster hours porting
   a result that has not survived a second seed on its home task. **This is the highest-value next
   job and it is on `avoiding`, not on UAV or V_A.**
2. **`n_steps` has two incompatible definitions** across the toolchain (successes-only vs
   all-episode). On `avoiding` the K=1 headline flips between "Pareto-dominates" and "trade-off"
   depending on which is used. Both ports inherit the ambiguity; per `pareto-definition-of-good` the
   word "beats" cannot be used until it is pinned.
3. **`best` vs `latest` is not a detail.** It changed AF-UNet on `avoiding` from *never reaching the
   goal* to *always reaching it* (09-01 DA §4). Any port that cannot select `latest` is not a test.
4. **The framing must stay honest.** `α_end > 0` deviates from the α-Flow paper (§A.2). In a thesis
   this is "α-Flow with upstream's `discrete_training` floor", not "α-Flow".
5. **`discrete_frac` in every run log.** Non-negotiable gate before reading any task number.

---

# §4 — Recommended order

| # | action | cost | why here |
|---|---|---|---|
| **0** | **Second/third seed for AF `α→0.2` on `avoiding`** | ~4.2 h train + ~15 min eval per seed | The claim being ported is n=1-seed. Everything below is wasted if it does not replicate. |
| **1** | **V_A: add the epoch override**, then run `MIX_AF_ALPHA_END=0.2` (and `0.05`) at `latest` | ~10 lines + ~4.6 h/arm | Cheapest real port: bone is already matched, knob already exists, one gap. Also finally answers U10 §1.7. |
| **2** | **Gen15: add `af_alpha_end` + `imf_backbone` + epoch knobs to `config/uav_mix.py`** | ~35 lines across train+plan blocks | Unblocks *any* AF-UNet UAV run; also makes the existing SiT rows relabelable. |
| **3** | **UAV AF-UNet on `corridor`**, three arms (`ae=0.2 / 0.05 / 0.0`) at `latest` | 3 × train + eval | `ae=0.0` on the U-Net bone is the mandatory control — otherwise α and backbone move together. |
| **4** | Revisit `pillars` only after fix_16's A/B is accepted **and** a `diffusion` target arm exists for it | — | §1.5 |

**Relabelling that costs nothing:** every existing UAV `af` row should be annotated
*"α→0 ⇒ MeanFlow objective on a 9.4 M SiT"* wherever it is cited. Same for pre-U10 V_A `af` rows
(*"MeanFlow objective on the 4.0 M visual U-Net"*). This is a documentation change, and it is the
single highest-value action in this list per unit of effort.

---

# §5 — Open questions for the user

1. **Env-name convention for Gen15.** Follow `avoiding`'s `AF_ALPHA_END` / `AF_BONE`, or introduce
   `UAV_*`? This freezes path tokens permanently — decide once, before any run.
2. **Epoch override scope.** Add it to Gen14 and Gen15 together (consistent), or Gen14 only (the
   immediate blocker)?
3. **UAV scene for the first AF-UNet run** — `corridor` (proposed), or `s_curve`?
4. **Does the UAV AF arm keep its SiT row at all?** Keeping both `bb=sit` and `bb=unet` doubles the
   cluster cost but is the only way to say anything about backbone × objective on the UAV.
5. **Seeds before ports?** §4 step 0 assumes yes. If the thesis timeline says otherwise, say so and
   the order changes.
