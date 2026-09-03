# CHANGELOG — Gen14 U12: `MIX_EPOCH`, the checkpoint selector

**Date:** 2026-09-03
**Scope:** `config/` + `mix_visual_aligning/` + `mix_visual_aligning_test/` + `Slurm_Codes/sbatch/mix_visual_aligning/` + `diffuser/utils/provenance.py`
**Retrain required:** ❌ **No** for the selector itself — it picks among checkpoints that already exist.
✅ Yes for the α-floor experiment it unblocks (that is U10's knob, not this one).
**Status:** patched locally, **not committed**, **not yet run on the cluster.**
**Motivation:** [`../../Gen3v7_AlphaFlow/INVESTIGATION_20260903_af_unet_port_to_UAV_Gen15_and_VisualAligning_Gen14.md`](../../Gen3v7_AlphaFlow/INVESTIGATION_20260903_af_unet_port_to_UAV_Gen15_and_VisualAligning_Gen14.md) §2.4 — the one blocking gap between Gen14 and the AF-UNet result that just landed on `avoiding`.

---

## 1. The problem

### 1.1 What was wrong

`'diffusion_epoch': 'best'` is inherited by all four mix plan blocks from `plan_fm_visual_aligning`
via `_mix_plan_common`, and **nothing could move it** — no env var, no CLI flag, no sbatch argument.
Every Gen14 rollout ever produced came from `state_best.pt`.

`state_best.pt` is written whenever `test_loss` hits a new low (`utils/training*.py`, `save_best()`).

### 1.2 Why that is the wrong checkpoint on the `af` arm

α-Flow's test loss carries an **α-weighted term**, so its minimum sits **mid-homotopy**:
`best` selects a model caught *inside* the curriculum, never the one the schedule ends on.
Gen3v7 measured the size of that effect on `avoiding-d3il` — the *same training run* went from

> **0/2 → 2/2 goals at K=1, purely by evaluating `latest` instead of `best`**

([`DA_20260901_AF_UNet_alpha_clamp_T1_negative.md`](../../Gen3v7_AlphaFlow/DA/DA_20260901_AF_UNet_alpha_clamp_T1_negative.md) §4.1/§4.2; the same DA §3.1 explains why `best ≈ argmin(0.75 + 0.25·α)` prefers mid-curriculum).

### 1.3 Why this blocked the experiment Gen14 actually wants to run

U10 shipped `MIX_AF_ALPHA_END`, which floors α so the bootstrapped target trains the final weights.
**Pairing it with `best` runs the experiment and then throws away its result**: you floor α for
100 000 steps and then deploy a checkpoint selected by a criterion that prefers small α. U10's own
§1.7 next step (`MIX_AF_ALPHA_END=0.02`) could not have been read correctly without this fix.

### 1.4 Two supporting defects found in the same path

| # | defect | consequence |
|---|---|---|
| a | **`latest` never means the end of training.** The periodic save fires on `self.step % save_freq == 0` inside `train_epoch`, and `self.step` only reaches `n_train_steps - 1`. At the default `save_freq = n_train_steps // 5` the newest numeric checkpoint is **step 80 000 of 100 000**. No `MIX_SAVE_EVERY` cadence fixes it — the last multiple of *any* frequency is `< n_train_steps`. | every `latest` rollout deploys a model 20 % short of the schedule |
| b | **`latest` on a tree with no numeric checkpoint walked into `state_-1.pt`.** `utils.get_latest_epoch` returns `-1` when nothing matches, and the old code passed it straight to `trainer.load`. | `FileNotFoundError` from inside `torch.load`, naming a file nobody asked for. Gen3v7 job 25253 died exactly this way and the cause took a manual `ls` on the cluster to find |

---

## 2. What changed

### 2.1 `config/aligning-d3il-visual.py` — the knob and its path key

| change | why |
|---|---|
| **`_mix_epoch_keys(raw=None)`** — new resolver. Reads `MIX_EPOCH` (blank == unset, via `_env_or_none`). Returns `{}` for `'best'`, else `{'diffusion_epoch': v, 'diffusion_epoch_tag': v}`. Accepts `best` \| `latest` \| a non-negative integer step; **raises on anything else**. | One validator for both entry points, so the CLI form and the env form can never disagree about what is legal or how the tag is spelled. Rejecting here turns `MIX_EPOCH=lastest` into a config-time error instead of a `state_lastest.pt` `FileNotFoundError` minutes into a GPU allocation. |
| **`('diffusion_epoch_tag', 'EP')` appended LAST to `args_to_watch_mix_visual_plan`** | 🔴 **The whole point.** Without it a `latest` pass writes into the SAME results directory as the `best` pass of the SAME weights — and on the `af` arm those are two genuinely different models. Emitted only when non-default, and `watch()` skips keys a block does not define, so **every results path that exists today is byte-identical**. Appended last so a non-default name reads as the shipped one plus a suffix (the trick `train_budget` already uses). |
| **`blk.update(_mix_epoch_keys())` at the end of `_mix_plan_block`** | Applies to all four arms from one line. Deliberately **after** `diffusion_loadpath`: these are EVAL keys and nothing here may touch the checkpoint identity the two mirror loops just built. |

### 2.2 `mix_visual_aligning_test/eval_mix_visual_aligning.py`

| change | why |
|---|---|
| **`--epoch SEL`** CLI flag, env fallback `MIX_EPOCH`, precedence **CLI > env > config default** | Matches U6 (`--flow-steps`) and U11 (`--proj-threshold` / `MIX_PROJ_T`). |
| Plan-block mutation placed **before any `Parser().parse_args()`** | Same timing rule as U6/U11: `exp_name` is resolved *inside* `parse_args`, so setting it later would give a run whose loader honoured the new checkpoint while the folder still claimed the old one. Validation is delegated to the config module's `_mix_epoch_keys`. |
| The mutation **pops `diffusion_epoch_tag` before writing** | An explicit `--epoch best` must REMOVE a tag that `MIX_EPOCH` may have set at config-import time, or the folder would carry `_EPlatest` for a `best` rollout. |
| **Fail-fast in `load_diffusion_with_override` when `latest` resolves to `-1`** | Defect 1.4(b). The message lists what `state_*.pt` files *are* present and names both ways out (`--epoch best`, or retrain with `MIX_SAVE_EVERY`). |
| **`[ eval loading ] checkpoint = state_<sel>.pt (trained to step N)`** | `trainer.step` is read back out of the checkpoint file itself, so this is the file's own claim about where in training it came from — not the config's. |
| **α breadcrumb, `af` arm only** — prints `alpha(step N)` from the TRAIN-time `diffusion_config.pkl`, with an explicit verdict line | 🔴 **The one check that separates α-Flow from MeanFlow at deployment.** `af_diffusion.py:552` routes `alpha <= 0` into Gen3v6's MeanFlow JVP body *unmodified*, so a checkpoint from the α=0 tail **is** a MeanFlow model however the folder is named. This puts §A.5's verification into the *eval* log, which is usually a different job hours after the training log. Uses the `_get_ratio` staticmethod, which exists for exactly this "schedule questions without a training loop" purpose. |
| **⚠ warning when `ENGINE == 'af'` and the selector is `best`** | Names the failure before it happens rather than after. |
| **Provenance:** `diffusion_epoch`, `diffusion_epoch_source`, `checkpoint_epoch_resolved`, `checkpoint_step` added to the eval `run_config` record | `latest` is a *request*; the resolved step is the *answer*, and only the answer identifies the weights. Two evals of the "same" checkpoint months apart can resolve `latest` differently. Same reasoning U11 used for `t_override_source`. |

### 2.3 `mix_visual_aligning/utils/training.py` + `training_twotime.py` — save the endpoint

A final `self.save(self.step)` at the end of `train()`, fixing defect 1.4(a). Fires **only on a
completed run** (the early `return` at the top of `train()` means there are steps left and that
run's periodic saves still stand). Costs **one extra `state_<n_train_steps>.pt` per completed run**;
overwriting is safe because `save()` is atomic (Fix_10) and would write identical weights.

> 🔴 **G0 impact: none.** Both files are `GRAFTED_DIFF` entries with `removed=3`. The insertion is
> purely additive, so the rewritten-source-line count is **unchanged at 3** — verified locally with
> G0's own `difflib` accounting:
> `training.py +62 −3` · `training_twotime.py +132 −3` · `projection.py +18 −0`. **All three match.**

### 2.4 `Slurm_Codes/sbatch/mix_visual_aligning/`

| file | change |
|---|---|
| `eval_mix_visual_aligning.sh` | Reads `MIX_EPOCH`, validates it with the **same** `best\|latest\|digits` rule as the Python side (verified case-for-case), echoes the resolved selector and the `_EP` fragment, and passes `--epoch`. Prints the af-arm `best` warning when the selector is left at the default. |
| `mix_visual_aligning_pipeline.sh` | Exports `MIX_EPOCH` to the child stages **explicitly**, not via `--export=ALL` — same doctrine as `MIX_TRAIN_STEPS`: it is a results-path key, and a stage that does not see it writes into a differently-named directory than the submitter is watching. Warns when an `af` pipeline sets `MIX_AF_ALPHA_END` without `MIX_EPOCH`. |
| `gates_mix_visual.sh` | Documents G-B12 and the `sbatch gates_mix_visual.sh gb12` form. |

### 2.5 `diffuser/utils/provenance.py`

`MIX_EPOCH` added to `TRACKED_ENV`. A plain `best` run carries no `_EP` fragment and is therefore
indistinguishable from a pre-U12 one by name alone; this is where that gets recorded.

### 2.6 `mix_visual_aligning_test/gates_mix_visual.py` — **G-B12**

New static gate (no GPU; runs under `--gate all` and `--gate static`). It asserts, on **all four arms**:

1. `diffusion_epoch_tag` is registered **last** on the plan watch list, and **absent** from the training watch list;
2. at the default: no tag emitted, `diffusion_epoch == 'best'`;
3. under `MIX_EPOCH=latest`: both keys set;
4. 🔴 **`prefix` and `diffusion_loadpath` are byte-identical** between the two — the selector must never touch the checkpoint identity;
5. the results `exp_name` is **strictly extended** by `_EPlatest` and nothing else (`custom_msg` forced to `''`, because `watch_plan` appends its `_msg` suffix *after* the watch fragments);
6. malformed selectors (`lastest`, `LATEST`, `-1`, `8e4`, `state_80000.pt`) are rejected, and an explicit `best` emits nothing.

---

## 3. Path-safety proof (run locally, no cluster)

The config module was exec'd three times under different `MIX_EPOCH` values with stub `yaml` /
`diffuser.utils`, and every assertion above was checked directly:

```
plan list last key : diffusion_epoch_tag
epoch keys in train: []
  ok  diffusion   ckpt frozen | results + '_EPlatest'
  ok  fm          ckpt frozen | results + '_EPlatest'
  ok  mf          ckpt frozen | results + '_EPlatest'
  ok  af          ckpt frozen | results + '_EPlatest'
  with custom_msg : ..._Eaf_EPlatest_msgafon02_s6     <- tag lands BEFORE the msg suffix
ALL LOCAL ASSERTIONS PASSED
```

`MIX_EPOCH=100000` behaves identically with `_EP100000`. **Composition with U10 is clean and
orthogonal** — the two knobs move different halves of the path:

| env | checkpoint path | results path |
|---|---|---|
| *(none)* | `…_afschsigmoid/6/` | `…_Eaf` |
| `MIX_EPOCH=latest` | `…_afschsigmoid/6/` **(unchanged)** | `…_Eaf_EPlatest` |
| `MIX_AF_ALPHA_END=0.2` | `…_afschsigmoid_AFAFend0p2/6/` | `…_Eaf` |
| **both** | `…_afschsigmoid_AFAFend0p2/6/` | `…_Eaf_EPlatest` |

*(The `AFAF` doubling is pre-existing U10 cosmetics — the watch label `AF` prepended to a tag that
already starts with `AF`. Harmless, already frozen into cand5's path, not touched here.)*

**Nothing that exists today moves.** With `MIX_EPOCH` unset, every checkpoint directory, every
`plans/` directory and every `diffusion_loadpath` is byte-identical to its pre-U12 self.

## 4. What was NOT changed

- No training-side key, no architecture, no parameter count, no projector, no constraint YAML.
- `mix_visual_aligning/utils/serialization.py` is a **G0 VERBATIM** file and was **not touched** —
  the `latest → -1` fail-fast lives in the eval script instead, which is not in the ledger.
- The `af` arm's α schedule is untouched. U12 decides **which checkpoint is deployed**, U10 decides
  **what α does during training**. They are independent and the experiment needs both.
- Gen15 (UAV) is untouched. Its three gaps are listed in the investigation §1.4 and are separate work.

---

## 5. Verification — what to submit on the cluster

### Stage 0 — gates (~seconds, no real GPU work). **Do this first.**

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh gb12
# and the full static set, which also re-checks G0's additive-graft accounting after the
# two trainer edits (§2.3):
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh static
```

**Pass condition:** `G-B12: PASS` and `G0: PASS`.
🔴 If G0 reports `graft is no longer additive — 4 source lines removed/rewritten, expected 3`, the
trainer edit was not purely insertive. Do **not** bump the number in `GRAFTED_DIFF`; re-diff first.

### Stage 1 — the free A/B: `latest` vs `best` on a checkpoint that already exists (**eval only, no training**)

This is the cheapest possible test of the whole fix, and it is a real experiment: it measures
Gen14's own version of the Gen3v7 `best`-vs-`latest` effect on the **shipped α→0 af run (cand7)**,
whose `best` numbers are already published in
[`../DA_20260831_Gen14_U10_alpha_const_and_U11_K100_projection_budget.md`](../DA_20260831_Gen14_U10_alpha_const_and_U11_K100_projection_budget.md).

```bash
MIX_EPOCH=latest FMPCC_RUN_MSG=ep_ab_s6 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh af 6 all 2
```

*(positional args: `<engine> <seeds> <record-mode> <K>`)*

**Reads the same weights as every existing af run** —
`…_filmv1_Eaf_tslogit_normal_afschsigmoid/6/` — and writes to a **new** directory:

```
plans/mix_visual_aligning_af/<same ckpt id>/H8_K2_Meuler_T0.5_D…VisualAlphaFlow_VTrue_mpc4_filmv1_Eaf_EPlatest_msgep_ab_s6/
```

**What to check in the log, in order:**

| line | expect |
|---|---|
| `[ eval ] checkpoint selector = 'latest' (source: env MIX_EPOCH) -> results dir carries _EPlatest` | the knob arrived |
| `[ eval loading ] checkpoint = state_80000.pt (trained to step 80000)` | **80000, not 100000** — cand7 predates §2.3, so its newest periodic save is still 80 k |
| `[ eval loading ] alpha(step 80000) = 0.0000 … 🔴 alpha = 0 -> … MeanFlow target` | ✅ **expected and correct** for the shipped α→0 schedule. This line is the fix working, not failing |

⚠ **If instead you get** `ERROR: --epoch latest found NO numeric state_<step>.pt`, the cand7 tree
holds only `state_best.pt` — the same condition that killed Gen3v7 job 25253. That is a real finding
about the tree, not a bug in this change; report the `present:` list from the message.

### Stage 2 — the experiment U12 exists to unblock: α floored **and** the endpoint deployed

Two arms, each `gates → train → eval` chained on `afterok`. ~4.6 h training each (U10 §1.2 timing).

```bash
# arm C — the recipe that won on avoiding-d3il (sigmoid 1.0 -> floor 0.2, endpoint deployed)
MIX_AF_ALPHA_END=0.2 MIX_EPOCH=latest FMPCC_RUN_MSG=afon02_s6 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 6 2

# arm D — the minimal "alpha is on" probe; separates "alpha on" from "probe big enough"
MIX_AF_ALPHA_END=0.05 MIX_EPOCH=latest FMPCC_RUN_MSG=afon005_s6 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 6 2
```

Expected trees (verified by simulation, §3):

| arm | checkpoint | results |
|---|---|---|
| C | `…_afschsigmoid_AFAFend0p2/6/` | `…_Eaf_EPlatest_msgafon02_s6` |
| D | `…_afschsigmoid_AFAFend0p05/6/` | `…_Eaf_EPlatest_msgafon005_s6` |

**Train-log gates before any task number is read** (α-on checklist, investigation §A.5):

| signal | ❌ α off | ✅ α on |
|---|---|---|
| `[ pipeline ] alpha schedule: MIX_AF_ALPHA_END=0.2` | absent | present |
| `val/alpha`, final epoch | `0.0` | ≈ `0.2` |
| **`train/discrete_frac`, final epochs** | **`0.0`** | **> 0** (tracks `af_ratio_fm = 0.5`) |
| savepath | `…_afschsigmoid/` | `…_AFAFend0p2/` |

**Eval-log gate:**
`[ eval loading ] checkpoint = state_100000.pt (trained to step 100000)` — §2.3's final save,
then `alpha(step 100000) = 0.2000 … alpha-Flow objective ACTIVE at this checkpoint.`
🔴 **If it says `state_80000.pt`, the trainer edit did not reach the cluster** (stale rev) — the
run is still usable (α is at its floor by 80 k) but say so in the DA.

### Stage 3 — the comparison to make

Against **MF-UNet on the same bone** and the **pinned Gen6V4 DPCC target**, on the U10 metric set
(distance, `collision_free_completed`, `avg_time_ms`), paired context-by-context as U10 did.

🔴 **Do not compare arm C against U10's cand5 (`AFconst0p05`) and call it a replication.** They are
different experiments: cand5 held α **flat at 0.05 from step 0** (no FM head, no curriculum) and was
read at `best`; arm C is the full **sigmoid anneal floored at 0.2**, read at `latest`. The correct
baselines are cand7 (`best`, published) and the Stage-1 `_EPlatest` re-read of cand7.

---

## 6. Risks and open items

1. **Disk.** §2.3 adds one checkpoint per completed run. `/data` was at 100 % during the Gen3v7 runs
   (27 G free of 7.0 T). Check free space before Stage 2.
2. **Stage 1 may abort immediately** if the cand7 tree has no numeric checkpoint (see the ⚠ above).
   That is information, and it is cheap to get.
3. **Single seed.** Stage 2 is seed 6, matching U10 and the Gen3v7 win. The Gen3v7 result it is
   chasing is **itself n = 1 seed** — investigation §4 step 0 recommends replicating that on
   `avoiding` before spending Gen14 GPU hours here. Stage 1 is unaffected: it is eval-only.
4. **`best` remains the default** everywhere, deliberately: U12 adds a capability, it does not change
   any existing run's behaviour. The `af` arm now *warns* rather than silently doing the wrong thing.
5. **Not run on the cluster yet.** Everything above is local static verification (syntax, G0 diff
   accounting, config path simulation, shell `-n` and case-rule equivalence). Nothing has touched a GPU.
