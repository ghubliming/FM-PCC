# Gen14 U10 — the α-Flow schedule becomes an overridable, **path-bearing** knob

**Date** 2026-08-29 · **Arm** `engine=af` only · **Task** `aligning-d3il-visual`
**Status** code + sbatch only. **Nothing has been trained or evaluated** (no torch in the AI
container). Every claim below about behaviour is a claim about *source*, not a measured result.

---

## Why

`af_alpha_end: 0.0` + `af_alpha_scheduler: 'sigmoid'` + `af_alpha_clamp: 0.005` force α to
**exactly 0** from ≈71.2 % of the budget onward, so the af arm trains the **MeanFlow** target for
its last ~28 820 of 100 000 steps. Gen14 U5 §3 measured what that costs on this task:

| step | α | `discrete_frac` | test raw MSE (u) |
|---:|---:|---:|---:|
| 70 000 | 0.006693 | 0.547 | **2.657** ← af's best |
| 71 000 | 0.005220 | 0.484 | 2.911 |
| **72 000** | **0.0** | **0.00** | **8.504** |

A 2.9× jump in one logging interval, never recovering, landing on `mf`'s own 7–10 plateau.
Sources: [`../U5/DA_20260804_mf_af_visual_aligning_first_run.md`](../U5/DA_20260804_mf_af_visual_aligning_first_run.md) §3 ·
[`ANALYSIS_20260829_alphaflow_vs_meanflow…`](../../../Data_Analysis/DA_Result_Curated_MD/ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md) §5.

Before U10 there was **no way to change this without editing the config**, and — the actual
blocker — **no way to change it safely**.

## 🔴 The bug this closes (path collision)

`args_to_watch_mix_visual_train` watched `af_alpha_scheduler` (`afsch`) and **nothing else** from
the α block. So:

```
af_alpha_end: 0.0   -> mix_visual_aligning_af/H8_…_filmv1_Eaf_tslogit_normal_afschsigmoid/6
af_alpha_end: 0.02  -> mix_visual_aligning_af/H8_…_filmv1_Eaf_tslogit_normal_afschsigmoid/6   ← SAME
```

A re-tuned run would have written into `cand6`'s directory — or, with `MIX_AUTO_RESUME=1`, resumed
*from* it. Same class as the `ml_bone` trap U8 fixed and the `train_budget` trap `_budget_tag`
fixes.

## What changed

### 1. `config/aligning-d3il-visual.py`

- **`_AF_ALPHA_DEFAULTS`**, `_AF_ALPHA_SCHEDULERS`, `_af_num()`, `_af_frag()`, **`_mix_af_alpha_keys()`**
  — new, placed just above the mix training blocks.
- **`args_to_watch_mix_visual_train`** gains `('af_alpha', 'AF')`, immediately after `afsch`.
- The `af` training block gains `**_mix_af_alpha_keys()` as its **last** entry (it must override the
  `af_alpha_*` literals above it).

Env knobs, all af-only, all optional:

| env | key | shipped default |
|---|---|---|
| `MIX_AF_ALPHA_SCHED` | `af_alpha_scheduler` | `sigmoid` |
| `MIX_AF_ALPHA_INIT` | `af_alpha_init` | `1.0` (= pure FM) |
| `MIX_AF_ALPHA_END` | `af_alpha_end` | `0.0` (= MeanFlow) |
| `MIX_AF_ALPHA_CLAMP` | `af_alpha_clamp` | `0.005` |
| `MIX_AF_ALPHA_GAMMA` | `af_alpha_gamma` | `25.0` |

**Absent at the defaults.** Every key is emitted only when it differs from the shipped value, and
`watch()` skips undefined keys (`diffuser/utils/setup.py:21-28`), so the `af_alpha` tag does not
exist unless something moved. Same trick as `_budget_tag()` and `_mix_u9_keys()`.
**→ every path that exists today, `cand6` included, is character-for-character unchanged.**

Tag rendering (verified by direct exec of the helper; `.`→`p`, `-`→`m`):

| env | emitted keys | tag |
|---|---|---|
| *(none)* | `{}` | *(absent)* |
| `SCHED=constant INIT=0.05 END=0.05` | scheduler, init, end | `AFconst0p05` |
| `END=0.02` | end | `AFend0p02` |
| `CLAMP=1e-4` | clamp | `AFclamp0p0001` |
| `END=0.01 CLAMP=1e-4` | end, clamp | `AFend0p01-clamp0p0001` |
| `SCHED=sigmoid END=0.0` (explicit defaults) | `{}` | *(absent)* |

Because the tag joins the **training** watch list, it propagates automatically to all three places
that must agree, from one list: the checkpoint tree, `plans/…` (via `_mix_plan_block`'s `_ckpt_id`)
and the eval's `diffusion_loadpath` (via `_mix_loadpath`).

**Guards (all raise at config-import, before any GPU is allocated):**

- unknown scheduler → refused, with the valid set from `af_diffusion._get_ratio`.
- `init`/`end` outside `[0, 1]` → refused.
- `clamp` outside `[0, 0.5)` → refused.
- non-float → refused.
- 🔴 **bare `SCHED=constant`** → refused. `_get_ratio('constant')` returns `af_alpha_init`, whose
  shipped value is `1.0`, so it would have trained **pure Flow Matching for every step**.
- 🔴 **constant α below the clamp** → refused. The clamp fires on *every* scheduler
  (`af_diffusion.py:472-475`), so e.g. α = 0.002 with the default clamp snaps to 0 and the arm
  trains MeanFlow from step 0 **while the folder still reads `AFconst0p002`** — precisely the lie
  the tag exists to prevent.

### 2. `Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh`

New block before the multi-seed warning: documents the five knobs, echoes them, errors if any is
set on a non-`af` engine, and — when they are unset on an `af` run — prints a ⚠ naming the U5
finding, so the default's cost is visible in every batch log rather than buried in a DA.

### 3. `…/mix_visual_aligning_pipeline.sh`

Appends each set knob to `EXPORT_OPTS`, so it reaches **both** child stages. Same contract, and the
same reason, as the existing `MIX_TRAIN_STEPS` export: an eval that does not see the env resolves
`diffusion_loadpath` to the default tree and either dies on a missing checkpoint or silently scores
the wrong model under the right name. Errors if set with a non-`af` engine.

### 4. `…/eval_mix_visual_aligning.sh`

Echoes the resolved schedule on `af` runs and warns when it is absent, since a standalone eval does
not inherit the pipeline's exports.

## Not changed

- `mf`, `fm`, `diffusion` arms — untouched, no path change, no behaviour change.
- `af_diffusion.py` / `visual_af_diffusion.py` — **no math touched**. `_get_ratio` already supported
  every scheduler exposed here; U10 only makes them reachable and nameable.
- `af_alpha_end_step` stays bound to `_MIX_N_TRAIN_STEPS` and is still asserted by the train script.
- Default behaviour. A run with no `MIX_AF_ALPHA_*` is byte-identical to a pre-U10 run.

## Verification done here (container-side only)

- `ast.parse` on the config: OK.
- `bash -n` on all three sbatch scripts: OK.
- `_mix_af_alpha_keys()` exec'd standalone over 6 valid and 6 invalid inputs — table above; the
  empty-at-default case and all six guards confirmed.

## 🔴 Still to verify on the cluster

1. **Config imports under the new env** — the AI container cannot import `diffuser`.
   Cheapest check: `gates_mix_visual.sh` on the `af` arm with the knobs set.
2. **The emitted path string.** Confirm the train log's savepath carries `_AFconst0p05` and that it
   is a NEW directory, before letting the job run to the wall.
3. **Eval finds it.** Confirm `diffusion_loadpath` in the eval log points at the same `_AF…` tree.
