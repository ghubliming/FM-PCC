# GUIDE — HardFlow-style planning (H16, replan-8) on MeanFlow-UNet, avoiding-d3il

**Date:** 2026-08-16 · **Type:** setup + submit guide (nothing applied, nothing submitted)
**Goal:** reproduce HardFlow's *planning structure* — plan a 16-step horizon, execute the first 8,
replan — on our own MeanFlow field, instead of our current H8 / execute-1 / replan-every-step.
**Backbone decision (locked):** `imf_backbone: 'unet'`, `freq_dim: 32` (3.97 M, DPCC-parity size).
**Companions:** `../../HF_iMF/Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md` (where the
8+8 number comes from) · `../../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md` (why K ≥ 5 for arm C)
· `../DA/DA_20260815_ntrials20_stability_MF_UNet.md` (the cost figures used in §7)

---

## 0. Status board — read this first

**🟢 UPDATED 2026-08-16 — the code landed (U10).** Every phase is now **env vars only**; no file
edits are needed to run anything. What was implemented and why:
`CHANGELOG_Gen3v6_U10_H16_replan8.md` (same folder).

| Phase | What | Needs | Runnable? |
|---|---|---|---|
| **0** | ~~Config edits~~ → now the `MF_HORIZON` / `MF_BACKBONE` env vars | — | ✅ **done in code** |
| **1** | Train MeanFlow-UNet at **H16** | — | ✅ submit now |
| **2** | Eval **H16 / replan-1** (the horizon-only rung) | Phase 1 checkpoint | ✅ after Phase 1 |
| **3** | Eval **H16 / replan-8** (the actual goal) | Phase 1 checkpoint | ✅ **unblocked** — `MF_REPLAN_STEPS=8` |

**Nothing has been run and nothing committed.** Defaults are unchanged: with no new env var set,
every pre-U10 command behaves byte-identically (changelog §1).

### 0.1 The complete knob set

| var | default | meaning |
|---|---|---|
| `MF_HORIZON` | `8` | planning horizon. **Training property** — must equal what the checkpoint was trained at; the eval aborts on a mismatch (gate G1). |
| `MF_BACKBONE` | `mf_dit` | `unet` \| `dit` \| `mf_dit`. Must equal the trained backbone. **This study: `unet`.** |
| `MF_REPLAN_STEPS` | `1` | actions executed per plan. `1` = replan every env step (every result to date). `8` = HardFlow's cadence. Must be `< horizon` (gate G2). |
| `HFFM_FLOW_STEPS` | plan block | matched K for **every** arm; one K per job |
| `MF_FLOW_STEPS` | `1 2 5 10 20` | multi-K grid in one job (ignored when `HFFM_FLOW_STEPS` is set) |
| `HFFM_ACT_THRESHOLD` | `0.5` | arm-C activation A |
| `HFFM_BATCH` | `1` | arm-C candidate fan |
| `TRAIN_SEEDS` | `6` | training seeds (train job only) |

All of them reach the job through `submit.sh`'s `--export=ALL` — no plumbing, no file edits.

### 0.2 TL;DR — the whole study is three commands

**There is no config file to edit at any point.** Everything is set on the submit line.

```bash
# ① train the H16 checkpoint (the only training run this study needs)
MF_HORIZON=16 MF_BACKBONE=unet TRAIN_SEEDS="6" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh

# ② H16, replan every step  — the horizon rung
MF_HORIZON=16 MF_BACKBONE=unet MF_FLOW_STEPS="1 2" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh

# ③ H16, execute 8 per plan — HardFlow's cadence, the goal
MF_HORIZON=16 MF_BACKBONE=unet MF_REPLAN_STEPS=8 HFFM_FLOW_STEPS=5 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

The only file you may want to touch is `config/meanflow_projection_eval.yaml`, and only to widen the
run's **scope** — `seeds:` and `n_trials:` (§4). That is not part of the feature.

---

## 1. What is actually changing, and why each piece is legal

| | current | target | training-time property? |
|---|---|---|---|
| horizon | 8 | **16** | ✅ **yes** — dataset windows (`flow_matcher_v3_meanflow/datasets/sequence.py:71-83`) and loss weights (`models/helpers.py:295-314`) are both sized by H ⇒ **retrain required** |
| executed actions / plan | 1 | **8** | ❌ no — `grep -rn "replan" flow_matcher_v3_meanflow/ scripts/train.py` returned **zero hits** before U10; cadence lives only in the eval rollout loop. `loss_discount: 1.0` (`config/avoiding-d3il.py:235`) ⇒ flat per-step weights, so training does not privilege step 0 |

So the horizon needs a new checkpoint; the cadence does not. That asymmetry is why both H16 rungs
share **one** training run, and why the cadence could be added as a pure eval-time knob.

### 1.1 The three-rung ladder — do not skip the middle

H16/8+8 changes **two** variables at once against today's numbers. The middle rung separates them:

| rung | how to get it | isolates |
|---|---|---|
| H8 / replan-1 | already have it (DA 2026-08-11, 2026-08-15) | — |
| **H16 / replan-1** | `MF_HORIZON=16 MF_BACKBONE=unet` (needs the retrain) | the **horizon** effect |
| **H16 / replan-8** | the same + `MF_REPLAN_STEPS=8` | the **cadence** effect |

Without the middle rung, any H16/8+8 result is uninterpretable.

---

## 2. Phase 0 — ✅ already in code, nothing to edit

`horizon` and `imf_backbone` are now read once from the environment
(`config/avoiding-d3il.py:59-60`) and used by **both** the training block (`:739`, `:783`) and the
plan block (`:1401`, `:1443`). The two are joined by `diffusion_loadpath`, which reproduces
`args_to_watch_fmv3_mf_train` token-for-token (`:139-149`, `:1465`); half-applying the change used to
resolve to a directory that does not exist (trap #6 in the config header) and is now unreachable.

```
MF_HORIZON=16 MF_BACKBONE=unet <any train or eval command>
```

Defaults are `8` / `'mf_dit'`, so every existing command is untouched.

> ⚠️ **CLI flags cannot do this.** `utils.Parser.add_extras` is commented out
> (`diffuser/utils/setup.py:77`), so `--horizon 16` is silently ignored. Use the env var.

### What you do NOT need to change

- `freq_dim: 32` — already correct (FIX_8_UNET_WIDTH; do **not** raise it, 256 ⇒ 253 M params).
- `dim_mults: (1,2,4,8)` — 3 downsamples ⇒ H must be divisible by 8. **16 is fine** (16→8→4→2).
- `max_path_length: 150`, `use_padding: True` — see §6.1, no change needed.
- Anything in `config/meanflow_projection_eval.yaml` for Phase 1.

---

## 3. Phase 1 — train MeanFlow-UNet at H16

```bash
MF_HORIZON=16 MF_BACKBONE=unet TRAIN_SEEDS="6" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh
```

Full 5-seed version (only after seed 6 has produced a sane loss curve):

```bash
MF_HORIZON=16 MF_BACKBONE=unet TRAIN_SEEDS="6 7 8 9 10" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh
```

`TRAIN_SEEDS` and `AUTO_RESUME` are already env-overridable (`train_meanflow.sh:92-100`); extra
flags pass through to `FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py`.

**Confirm the env vars survived** — the log prints `[ train ] MF_HORIZON=16  MF_BACKBONE=unet` near
the top. If either says `(default)`, the job is training an H8 `mf_dit` model instead; kill it. (The
eval would catch it later via G1, but only after the GPU time is spent.)

### 3.0 Chaining the eval onto a training job already submitted

`Slurm_Codes/submit_after.sh <JOB_ID> <script>` submits with `--dependency=afterok` and the same
`--export=ALL`, so env vars behave exactly as with `submit.sh`. Training does not have to be
finished — or even started — to queue its eval:

```bash
# rung 2 — H16, replan every step (the horizon rung)
MF_HORIZON=16 MF_BACKBONE=unet MF_FLOW_STEPS="1 2 5" \
  ./Slurm_Codes/submit_after.sh <TRAIN_JOB_ID> Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh

# rung 3 — H16, execute 8 per plan (auto-tags its results path _msgr8)
MF_HORIZON=16 MF_BACKBONE=unet MF_REPLAN_STEPS=8 MF_FLOW_STEPS="1 2 5" \
  ./Slurm_Codes/submit_after.sh <TRAIN_JOB_ID> Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

Both hang off the same training job and start together when it finishes — fine at smoke scale
(`seeds: [6]`, `n_trials: 2` — the shipped yaml), **not** fine at 5 seeds × 20 trials until §7's
wall-time question is settled. `afterok` cancels the evals automatically if training fails.

### 3.1 Will the folder name be correct? — yes, and here it is in full

The training folder is built by `watch(args_to_watch_fmv3_mf_train)`
(`config/avoiding-d3il.py:139-149`), and `savepath = logbase / dataset / exp_name / seed`
(`diffuser/utils/setup.py:174-176`). With `MF_HORIZON=16 MF_BACKBONE=unet TRAIN_SEEDS="6"`:

```
logs/avoiding-d3il/flow_matching_v3_meanflow/H16_Dflow_matcher_v3_meanflow.models.MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/6/
```

The existing H8 UNet run, for comparison — **identical except the `H` token**:

```
logs/avoiding-d3il/flow_matching_v3_meanflow/H8_Dflow_matcher_v3_meanflow.models.MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/6/
```

**Why the eval will find it.** The plan block reaches the checkpoint through
`diffusion_loadpath` (`:1465-1466`), an f-string that must reproduce the train watch-list
token-for-token. Both sides now read the same `_mf_horizon` / `_mf_backbone`, and the two
constructions were checked to produce byte-identical strings at H8 and at H16:

| token | source | value here |
|---|---|---|
| *(prefix)* | `'flow_matching_v3_meanflow/'` | note **matching**, while the module is `flow_matcher_…` — this asymmetry is original |
| `H` | `horizon` ← `MF_HORIZON` | `H16` |
| `D` | `diffusion` | `Dflow_matcher_v3_meanflow.models.MeanFlowODE` (full dotted path, not just the class) |
| `aw` | `action_weight` | `aw10` |
| `obj` | `mf_objective` | `objmeanflow` |
| `bb` | `imf_backbone` ← `MF_BACKBONE` | `bbunet` |
| `ts` | `t_schedule` | `tslogit_normal` |
| `dp` | `meanflow_data_proportion` | `dp0.5` |

Two mechanical details worth knowing, because they look like bugs and are not: `watch()` joins with
`_` and then collapses `'/_' → '/'`, which is why the prefix's trailing slash absorbs the separator;
and the `D` token carries the **whole import path**, so the folder name legitimately contains dots.

**Verification before moving on**
1. The folder above exists and holds `state_*.pt`.
2. The H8 UNet folder is untouched — the `H` token guarantees no collision.
3. Only the `H` token differs between the two paths. If anything else moved, a knob drifted; stop
   and fix before evaluating, because the eval resolves the checkpoint by this exact string.
4. Loss curve comparable to the H8 UNet run at the same step count. A markedly worse curve is the
   first place §6.1's longer padded tails would show up.

---

## 4. Phase 2 — eval H16 / replan-1 (zero code, the horizon rung)

This is today's eval, unmodified, pointed at the H16 checkpoint. It runs **arms A/B/C in one
process** (unguided / DPCC post-hoc / HardFlow in-loop) at matched K, sharing seeds and env resets.

**Where the results land.** The plan block nests under a checkpoint-identifying prefix and then a
per-run folder from `watch_plan(args_to_watch_fmv3_hf_plan)` (`config/avoiding-d3il.py:123-131`),
i.e. `H{H}_K{K}_M{solver}_T{dpcc_threshold}_A{hf_threshold}_B{hf_batch}_D{diffusion}` plus the
`_msg…` suffix when one is set. At K=1 that is:

```
logs/avoiding-d3il/plans/flow_matching_v3_meanflow/H16_D…_bbunet_tslogit_normal_dp0.5/H16_K1_Meuler_T0.5_A0.5_B1_D…MeanFlowODE/
```

So the horizon appears **twice** — once identifying the checkpoint, once in the run folder — and
both flip to 16 together. Phase 3 adds `_msgr8` to the tail (§6.2); Phase 2 has no suffix.

**Smoke test first — one seed, 2 trials, two cheap K:**

```bash
MF_HORIZON=16 MF_BACKBONE=unet MF_FLOW_STEPS="1 2" \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

**Full run** — edit `config/meanflow_projection_eval.yaml` first: `seeds: [6,7,8,9,10]` (`:19`),
`n_trials: 20` (`:33`). Then one job per K, so a wall-clock overrun kills only that K:

```bash
for K in 1 2 5; do
  MF_HORIZON=16 MF_BACKBONE=unet HFFM_FLOW_STEPS=$K \
  HFFM_ACT_THRESHOLD=0.5 HFFM_BATCH=1 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
done
```

Knobs, all env, all already wired (`eval_meanflow_hardflow.sh:81-108`):

| var | meaning | use here |
|---|---|---|
| `HFFM_FLOW_STEPS` | pins a single matched K for **every** arm | one K per job |
| `MF_FLOW_STEPS` | multi-K grid in one job (unset ⇒ `1 2 5 10 20`) | smoke test only |
| `HFFM_ACT_THRESHOLD` | arm-C activation A | `0.5` (parity with the H8 ladder) |
| `HFFM_BATCH` | arm-C candidate fan | `1` (faithful) |
| `MF_REPLAN_STEPS` | actions executed per plan | **unset (=1)** in this phase — that is the point of the rung |

**On arm C and K:** per `DEGENERACY_HardFlow_at_low_K.md` §0.1, at `A=0.5` the K=1 and K=2 rows run
**zero** genuine HardFlow steps — they are `Π_S(Euler sample)`, i.e. sample-then-project. Keep them
(they are the cheap ladder and match the existing H8 rows), but the HardFlow-vs-DPCC question only
has content at **K ≥ 5**.

---

## 5. Phase 3 — H16 / replan-8 · ✅ unblocked

`MF_REPLAN_STEPS=8` makes the eval plan once, execute the first 8 actions, then replan — HardFlow's
own cadence (`HardFlow/run/eval.py:390-397`). Same job script, same arms, one extra env var:

```bash
for K in 1 2 5; do
  MF_HORIZON=16 MF_BACKBONE=unet MF_REPLAN_STEPS=8 HFFM_FLOW_STEPS=$K \
  HFFM_ACT_THRESHOLD=0.5 HFFM_BATCH=1 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
done
```

**You do not need `FMPCC_RUN_MSG` here** — a non-default cadence auto-tags its own results path with
`_msgr8`, because the cadence is not one of the folder-name tokens and an r1 and an r8 run at the
same K would otherwise overwrite each other (§6.2). Set `FMPCC_RUN_MSG` yourself only if you want a
different label; an explicit value always wins.

### 5.1 What the cadence changes, mechanically

| | replan-1 (default) | replan-8 |
|---|---|---|
| `policy()` calls per episode | ~60 | ~8 |
| NLP solves per episode (arm C) | ~60 × `n_active` | ~8 × `n_active` |
| which action is executed | always the plan's step 0 | steps 0…7 of the same plan |
| observation feeding the plan | always current | up to 7 steps stale |
| `avg_time` | per-step planning cost | the **same total, amortised** over 8× more env steps |

### 5.2 Two derived quantities that were fixed with it

- **`-t` candidate selection.** `temporal_consistency` ranks candidates against the previously
  executed plan shifted by one step. Under replan-8 the previous plan advanced by **8**, so the
  comparison was misaligned by 7 and `-t` silently stopped meaning what it says. The shift now
  follows the cadence in both policy classes.
- **`pos_tracking_error`.** Its reference used to be step 1 of a plan made this step. Under replan-8
  it now walks along the cached plan instead of freezing on the reference of a plan made up to 7
  steps ago.

Both are no-ops at the default cadence.

---

## 6. Traps, verified against the code

### 6.1 Training data at H16 — no window loss, but longer held-still tails
`max_path_length: 150` (`config/avoiding-d3il.py:783`) and `use_padding: True` (`:787`) while the
longest avoiding demo is **106** steps. So `make_indices` gives `max_start = min(path_length-1, 150-H)`
= `path_length-1` at **both** H8 and H16 — **identical window count, no demos dropped**
(`sequence.py:71-83`). What does change: padding repeats the **last** observation/action, not zeros
(`sequence.py:38-42`), so at H16 up to 15 trailing steps of a window can be "frozen at the last
state" versus 7 at H8. Not corrupt, but a larger share of the training signal is stationary tail.
This is the one place to look if the H16 field (arm A) comes out worse than H8.

### 6.2 Results-path collisions — handled automatically
**Horizon is already a path token (`H16_`), but the replan cadence is not**, so an H16/1 and an
H16/8 run at the same K/A/T would write to the same directory. The eval therefore auto-sets
`FMPCC_RUN_MSG=r<N>` whenever `replan_steps != 1` (`config/avoiding-d3il.py:210` consumes it,
the same slot the 2026-08-15 DA used for its 20-trial rows). An explicit `FMPCC_RUN_MSG` always
wins. Promoting the cadence to a real token in `args_to_watch_fmv3_hf_plan` was deliberately
**not** done — it would rename every historic H8 path.

### 6.3 An H8 checkpoint at H16 fails **loudly on `mf_dit`, silently on `unet`**
- `mf_dit` → `MFDiTOfficialTrajectory` has a **learned** `pos_embed = nn.Parameter(1, num_patches, D)`
  (`models/mf_dit_official_trajectory.py:294`), which **is** in the state_dict ⇒ shape mismatch ⇒ crash.
- `dit` → RoPE tables are `register_buffer(..., persistent=False)` (`models/mf_dit_trajectory.py:301-303`)
  ⇒ **not** in the state_dict ⇒ loads fine and silently extrapolates to unseen positions.
- **`unet` (our choice)** → `ResidualTemporalBlock` accepts `horizon` and **never uses it**
  (`models/unet1d_temporal_cond.py:55-70`); the weights are Conv1d + Linear, all length-agnostic
  ⇒ **loads fine, runs fine, produces a scientifically invalid number with no error.**

Because we picked `unet`, the model itself gives **no protection**. Gate **G1** now supplies it: the
eval compares the checkpoint's trained horizon against `args.horizon` and **aborts** (`SystemExit`),
rather than relying on the CONFIG-OVERRIDES-PKL reconciler, which only prints a `[WARNING]` for
architecture keys and keeps running with `args.horizon` still driving the Projector and the policy
call. Gate **G2** likewise aborts when `MF_REPLAN_STEPS >= horizon`.

Sanity-check both once, cheaply, before trusting a batch:
```bash
MF_HORIZON=16 MF_BACKBONE=unet ./Slurm_Codes/submit.sh …/eval_meanflow_hardflow.sh   # on the H8 ckpt -> must ABORT
MF_HORIZON=16 MF_BACKBONE=unet MF_REPLAN_STEPS=16 …                                   # must ABORT
```

---

## 7. Cost and `--time`

Measured H8 MeanFlow-UNet eval, 5 seeds × `n_trials=20`, one job per K
(`../DA/DA_20260815_ntrials20_stability_MF_UNet.md` §header):

| K | job | wall |
|---|---|---|
| 1 | 24559 | 5 h 49 m |
| 2 | 24560 | 6 h 56 m |
| 5 | 24561 | 15 h 40 m |
| 10 | 24562 | 🔴 hit the 24 h wall, killed mid-sweep |
| 20 | 24563 | 🔴 hit the 24 h wall, unusable |

Scaling to this study:
- **H16 / replan-1** — the NLP dof roughly doubles (H×transition_dim), and the network sees 2× the
  sequence. Expect ≥ 2× the H8 figures ⇒ **K=5 alone would exceed 24 h at n=20**. Start at
  `n_trials: 2, seeds: [6]`, measure, then scale. K ≥ 10 at 20 trials is not viable without
  splitting by halfspace.
- **H16 / replan-8** — ~8× fewer plans per episode. This should *more* than pay back the larger NLP,
  and is itself an interesting result: the 2.8–3.3× arm-C wall-time penalty that sank HardFlow in
  the 2026-08-02 DA may largely disappear under HardFlow's own planning cadence.

The eval sbatch is `--time=24:00:00` (the cluster cap). Split by K — one job per K, as above.

---

## 8. What this can and cannot claim

**Can:** arms A/B/C are compared inside a single process at matched K, shared seeds and shared env
resets. "Does HardFlow-style in-loop constraint beat DPCC post-hoc **under HardFlow's own planning
structure**" is fully answerable with one retrain.

**Cannot:** beat *the* baseline. Diffusion-DPCC K20/aw10 — the pinned paper baseline — exists only
at **H8 / replan-1**. Putting our H16/8+8 MeanFlow next to it compares two things that differ in
horizon, cadence *and* generative engine. Doing that properly needs a **second retrain**: the
diffusion baseline at H16, evaluated at replan-8. Not in scope here; flag it before any H16 number
goes into a comparison table.

**Also not comparable:** the HardFlow-native H16 pipeline
(`Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh ML_TYPE=mf`, checkpoint `mf/H16_ml_mf_100k`).
That is our MeanFlow *objective* inside HardFlow's harness — different network
(`TemporalImfUnet`), `max_episode_length=100`, no DPCC arms, HardFlow's own metrics. It is a useful
cross-check, not a row for our tables.

---

## 9. Order of operations

0. **Regression check first — this is an EVAL, not a train, and it edits nothing.**
   U10 touched shared code (`policies.py`, `hardflow_projection.py`, the eval loop), so before
   trusting any new number, confirm the **old** path still behaves. Runs on the H8 checkpoint you
   already have:

   ```bash
   MF_BACKBONE=unet HFFM_FLOW_STEPS=1 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   ```

   No `MF_HORIZON` (default 8 = the existing checkpoint), no `MF_REPLAN_STEPS` (default 1 = the old
   controller). `MF_BACKBONE=unet` is required because the config default is `mf_dit` — it is
   exactly the value that used to be hand-edited into the file.

   **Pass condition:** the results path matches the historic shape
   `…_bbunet_tslogit_normal_dp0.5/H8_K1_Meuler_T0.5_A0.5_B1_…MeanFlowODE` with **no `_msg` token**,
   and the seed-6 metrics match the recorded seed-6 row. Skippable, but then a surprising H16
   result has two candidate causes instead of one.
1. **Phase 1** — `MF_HORIZON=16 MF_BACKBONE=unet TRAIN_SEEDS="6"` train. Check the loss curve and
   that the folder differs from the H8 one in the `H` token only.
2. **Phase 2** — smoke-test H16/replan-1 (`n_trials: 2, seeds: [6]`), measure wall time, then scale.
   **This is a publishable rung on its own** (the horizon ablation).
3. **Gates** — one deliberate G1 abort and one G2 abort (§6.3), so you know they fire.
4. **Phase 3** — `MF_REPLAN_STEPS=8`. Confirm `nlp_solves` drops to ≈ 1/8 of the replan-1 run at the
   same K; that is the direct observable that the cadence is real.
5. Decide separately whether the H16 diffusion baseline retrain (§8) is in scope.

⚠️ **Reading the numbers:** under replan-8, `avg_time` is the same total planning cost amortised over
8× more env steps. It is *not* the same quantity as the replan-1 rows — say so in any table that puts
them side by side.
5. Decide separately whether the H16 diffusion baseline retrain (§8) is in scope.
