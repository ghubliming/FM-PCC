# CHANGELOG — Gen15 (UAV Mix-ML) · coding pass 1

**Date:** 2026-08-10 · **Status:** code complete, **NOTHING RUN** (this container has no Python
packages — every check below is syntax-level; all execution is a cluster job on i6-gpu-1).
**Plan:** [`PLAN_Gen15_uav_mix_ml.md`](./PLAN_Gen15_uav_mix_ml.md)

Gen11's UAV pipeline now takes the ML objective as a config switch:
`--engine fm | mf | af` → Flow Matching (Gen11) · MeanFlow (Gen3v6) · α-Flow (Gen3v7).

---

## 1. What was built

| path | kind | notes |
|---|---|---|
| `mix_uav/` | 40 `.py` | copy of `flow_matcher_v3_uav/` @ HEAD + the two-time engines |
| `mix_uav_test/` | 6 `.py` | Gen11's test folder + engine dispatch + gates |
| `config/uav_mix.py` | **new, 354 ln** | Gen15's own config module (§3) |
| `Slurm_Codes/sbatch/uav_mix/` | 9 `.sh` + README | Gen11's `uav_fm/` sextet + gates + K-sweep |

**Newly-authored code** (the budget the plan is accountable to, §4.2 target < 450):

| file | lines | what |
|---|---|---|
| `mix_uav/models/engine_registry.py` | 329 | the dispatch table + kwarg builders |
| `mix_uav/models/__init__.py` | 36 | exports, incl. the aliased two-time U-Net |
| `mix_uav/utils/__init__.py` | 18 | `TrainerTwoTime` alias (never `import *`) |
| **total new logic** | **383** | under the 450 budget |
| `mix_uav_test/gates_mix_uav.py` | 498 | test code, outside the model budget |
| `config/uav_mix.py` | 354 | declarative config, outside the budget (plan §4.2) |

**Diffs against the Gen11 originals** (line-ending–normalised, so these are real code deltas):

| file | +/− | content |
|---|---|---|
| `mix_uav_test/eval_mix_uav.py` | +109 / −22 | engine dispatch, K plumbing, engine token in the eval tag, docstring |
| `mix_uav_test/train_mix_uav.py` | +/− 134 | `--engine`, registry-routed Config blocks, extra-metric W&B curves |
| `mix_uav/sampling/policies.py` | +/− 21 | the `fix_5` graft (§4) + an engine-agnostic comment |
| everything else in `mix_uav/` | 0 | verbatim, modulo one path comment and LF normalisation |

---

## 2. Isolation — verified, not assumed

```
git status --short flow_matcher_v3_uav/ FM_v3_uav_test/ config/uav.py \
                   config/uav_projection.yaml Slurm_Codes/sbatch/uav_fm/ \
                   flow_matcher_v3_meanflow/ flow_matcher_v3_alphaflow/
→ (empty)
```

- **`config/uav.py` is untouched** — Gen15 has its own module; Gen11's `_uav_exp_name`,
  `_COND_MODE_DIM` and `MAX_PATH_LENGTH_PER_SCENE` are **copied, not imported** (documented in
  the new file's docstring so nobody "fixes" it back into an import).
- **`logbase = 'logs/UAV_MIX'`** — a different root from Gen11's `logs/UAV_FM`, so isolation
  holds at the top of the path, not only at `prefix`.
- **One deliberate share: `config/uav_projection.yaml`**, read-only. Identical constraints are
  what make Gen11 and Gen15 comparable; each run still snapshots the yaml it actually read.

⚠️ **One scare during the build, fully reverted.** A `git mv` while creating `uav_mix/`
removed `fm_uav_pipeline.sh` and `fm_uav_all_pipeline.sh` from Gen11's sbatch folder. Caught by
the isolation check, restored with `git checkout HEAD -- Slurm_Codes/sbatch/uav_fm/`, and
re-verified byte-identical to HEAD. Gate **G2(c)** now asserts `config/uav.py` is unmodified on
every gate run, precisely because this class of accident is silent.

> `logs_in_develop/MASTER_TEST_HISTORY.md` shows as modified in `git status`. That is
> pre-existing work (the Aug 10 DAv3/DA_VA_v2 entries), **not** written by this pass. The Gen15
> row already existed in the master index and was not edited.

---

## 3. `config/uav_mix.py` — the new config module

Six blocks — `mix_uav_{fm,mf,af}` (train) and `plan_mix_uav_{fm,mf,af}` (eval) — composed from
three shared dicts so the task can never drift between arms:

- `_UAV_TASK` — horizon 8, `cond_mode='pos_only'` (obs 6-D, action 3-D, **transition 9-D**),
  `SafeLimitsNormalizer`, `n_train_steps=100000`, batch 8, lr 1e-4. Identical on all three arms,
  so the comparison is compute-matched.
- `_UAV_PLAN` — eval scalars (`flow_steps_v3`, `mpc_batch_size`, threshold, `control_hz`, MJX knobs).
- `_TWO_TIME_BACKBONE` — the mf/af backbone + trainer knobs, `imf_backbone='unet'`,
  **`freq_dim=32`** (FIX_8_UNET_WIDTH: on the unet backbone this IS the channel width — 32 ⇒
  3.97 M params, 256 ⇒ 253 M), `gradient_clip=1.0`, `dual_head=True`, `interval_cfg=False`.

`_uav_mix_exp_name` is Gen15's own path discriminator: Gen11's shape plus registry-driven engine
tokens (`_dp{v}_bb{v}` for mf, `_as{v}_ae{v}_bb{v}` for af, **empty for fm**). Without them, two
`mf` runs differing only in `meanflow_data_proportion` would share a checkpoint directory.

**α-Flow trap closed:** `af_alpha_end_step = 100000` is pinned equal to `n_train_steps`.
`AlphaFlowODE.__init__` hard-asserts this; upstream's larger value would hold α≈1 for the whole
run — i.e. train plain flow matching under an α-Flow folder name.
`af_adp_eps = 1e-3` is deliberately ≠ MeanFlow's `0.01` and is commented as such.

---

## 4. Deviations from the plan (all deliberate)

**D1 — The DiT/SiT backbones WERE copied.** The plan said not to. Two reasons to change:
`MFTrajectoryModel`/`AFTrajectoryModel` import them unconditionally, so omitting them breaks the
module; and copying them is zero authored lines and makes the backbone switch real. The result
is that `imf_backbone` is a genuine ML-backbone selector —
`mf: unet | dit | mf_dit`, `af: unet | dit | sit`, `fm: unet` only (the FM lineage never grew a
transformer backbone). **Config stays locked to `unet` for the headline comparison** (§6 of the
plan: with the backbone fixed, the arms differ only in objective and sampler). `check_backbone()`
rejects an invalid engine/backbone pair at startup instead of hours in.

**D2 — Trainer graft G2 cost zero lines, not ~40.** The plan assumed Gen3v7's trainer needed
Gen11's `current_test_loss` caching and the `(step + 1) % log_freq` off-by-one grafted in.
Gen3v7's copy **already has both**, independently derived — verified at
`training_twotime.py:137,233-234,262-265,269`. It also already carries `EXTRA_METRIC_KEYS`,
`split_seed=42`, wired `gradient_clip`, and the `set_train_step` hook. So `training_twotime.py`
is a pure verbatim copy. Two trainers still live side by side, unmerged.

**D3 — `fix_5` grafted into all three arms** (plan §5 G1 option (a), as recommended). Gen11's
`policies.py` lacked Gen3v6's `executed_idx` correction: under `temporal_consistency` selection
the code reordered `observations` but then indexed it with `which_trajectory`, so
`prev_observations` recorded a *different* candidate than the one executed.
⚠️ **This changes `fm`-arm behaviour vs Gen11 on the `dpcc-t*` variants.** It is applied
identically to all three arms so the comparison stays internally consistent, and **parity gate
G1 is asserted on `diffuser` + `dpcc-c` only.** Gen11 is not patched — syncing it is the user's
call.

**D4 — All of `mix_uav/` normalised to LF.** Gen11's `utils/training.py` and Gen3v6's files are
CRLF; without this, every future `diff` inside `mix_uav/` reads as a whole-file rewrite. Content
is unchanged; `training.py` is no longer *byte*-identical to Gen11's, only content-identical.

---

## 5. 🔴 Finding: Gen11's `flow_steps_v3` (K) was inert in BOTH directions

Discovered while wiring the K sweep. In Gen11, `config/uav.py`'s plan block sets
`'flow_steps_v3': 20`, and **neither half of that reached anything**:

1. **It never reached the sampler.** `build_experiment` parses the *training* block and passes
   those args as `override_args` to `load_diffusion`. The training block has no `flow_steps_v3`
   key, so the reconciliation loop's `hasattr(override_args, k)` check skips it and the pickled
   **training** value survives. Training passed `getattr(args, 'flow_steps_v3', 10)` → **10**.
2. **It never reached the folder name.** `_uav_eval_tag` reads `flow_steps_v3` off the
   YAML-derived `cfg` dict; `_load_base_cfg` never puts it there and `config/uav_projection.yaml`
   does not define it, so `config.get('flow_steps_v3', 20)` always returned the **default 20**.

Net effect: **Gen11 evals sampled at K=10 inside folders labelled `K20`**, and changing the plan
block's K changed neither. Two runs at different intended K would also have overwritten each
other.

**Gen15 closes both paths** (Gen11 is NOT patched — that is a separate decision for the user):

- `build_experiment(..., flow_steps=K)` calls `engine_registry.apply_nfe(diffusion, K)`, setting
  `flow_steps_v3` and `ode_inference_steps_v3` on the loaded object, and prints the resolved K.
- `_load_base_cfg` injects `cfg['flow_steps_v3']` from the plan block (or `--flow-steps`), so
  `K{n}` in the path is truthful.
- The two-time arms additionally get an explicit `num_steps=K` through `Policy.sample_kwargs`;
  `FlowMatchingODE.p_sample_loop` has no such parameter and would `TypeError`, so the registry
  returns `{}` for `fm`. This is the only engine-dependent line in the rollout path.
- `eval_scene` now resolves `base_cfg` **before** building the model (a safe reorder —
  `_load_base_cfg` needs only scene and seed).

**Consequence for the DA: do not carry any Gen11 `K20`-labelled number into a Gen15 comparison
without re-running it.** Those runs were K=10.

---

## 6. Engine dispatch — how it hangs together

```
config/uav_mix.py            engine key per block
        │
        ▼
engine_registry.ENGINES[e] = { model, diffusion, trainer,          ← class paths (package-relative)
                               model_kwargs, diffusion_kwargs,     ← builders
                               trainer_kwargs,
                               two_time, supports_num_steps,
                               backbones, exp_name_tokens }
        │
        ├── train_mix_uav.py  → three utils.Config blocks, no if-chain
        ├── eval_mix_uav.py   → block name, K pinning, sample kwargs, eval tag
        └── config/uav_mix.py → exp_name tokens (so config and table can't disagree)
```

`model_config.pkl` describes **the U-Net on `fm`** and **the engine on `mf`/`af`** (Gen8's L3
lesson). Anything reading it must go through the registry. `utils/serialization.py:load_diffusion`
already handles both shapes unchanged — it builds `model_config()` then `diffusion_config(model)`.

Two `Trainer` classes coexist: `utils.Trainer` (Gen11, `fm`) and `utils.TrainerTwoTime`
(Gen3v7, `mf`/`af`). `utils/__init__.py` imports the second **aliased** — a star-import would
silently shadow the first. Same trap in `models/__init__.py`: both U-Net files define
`Flow_matcher_U_Net_v2`, so the two-time twin is exported as
`Flow_matcher_U_Net_v2_TwoTime` and is never selected by name from config.

**Known, accepted confound:** `split_seed=42` on the two-time trainer vs Gen11's unseeded split
means `mf`/`af` train on a different train/test split than `fm`. Compare arms on closed-loop task
metrics (split-independent: eval uses MuJoCo scenes and fixed episode seeds), **never** on
`test_loss`.

---

## 7. Gates (`mix_uav_test/gates_mix_uav.py`)

Cheap, dataset-free, MuJoCo-free — a wiring bug should cost 30 seconds, not a day.

| gate | asserts |
|---|---|
| **G0** | every arm builds model + diffusion, and its trainer kwargs match the trainer signature |
| **G1** | `fm` parity vs Gen11 — structural half (config pkls identical); needs both savepaths, else SKIP |
| **G2** | (a) 19 knob combinations → 19 distinct `exp_name`s; (b) `logbase == logs/UAV_MIX`; (c) `git diff --quiet config/uav.py` |
| **G3** | velocity-net parameter counts match across arms (the FIX_8_UNET_WIDTH lesson, executable) |
| **G4** | every two-time `(t, h)` sampling query satisfies `t,h ∈ [0,1]`, `t + h ≤ 1` |
| **G5** | with `goal_dim = 0`, the projector receives the full **9-D** trajectory on every arm |
| **G6** | per-plan wall clock per arm per K — **measured and printed, never pass/fail** (whether an arm meets the 33 Hz deadline IS the experiment) |

G3 tolerates a small positive delta vs `fm` (the two-time U-Net adds `h_mlp` and a v-head) and
fails only above 25 % — a `freq_dim` width defect would be ~60×.

---

## 8. How to test-run on the cluster

Everything below is `bash Slurm_Codes/submit.sh <script> [args]` from repo root. **`engine` is
always the first argument.** Full matrix in
[`Slurm_Codes/sbatch/uav_mix/README.md`](../../../Slurm_Codes/sbatch/uav_mix/README.md).

### Step 0 — gates (do this first, ~minutes)

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/gates_mix_uav.sh
```

Expect `G0 PASS · G1 SKIP · G2 PASS · G3 PASS · G4 PASS · G5 PASS · G6 PASS`, with G3 printing
three near-equal parameter counts and G6 printing a ms-per-plan table against the 30.3 ms budget.
Any FAIL: stop, do not train. (Subset: `... gates_mix_uav.sh cuda "G3 G4"`.)

### Step 1 — smoke train, one arm, one scene, one seed

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_mix_uav.sh mf corridor "6"
```

Watch for, early in the log:
```
[ train ] Gen15 UAV Mix-ML — engine: mf  (MeanFlow (Gen3v6, arXiv 2505.13447))
[ train ] engine=mf  backbone=unet  params=X,XXX,XXX (3.97 M)
[ train ] savepath: logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_D..._9D_dp0.5_bbunet/6
```
Three things to confirm: **params ≈ 4 M** (not 253 M), the savepath is under **`logs/UAV_MIX`**,
and the folder carries the `dp`/`bb` tokens. On the `af` arm also confirm `alpha` appears in the
logged metrics and **moves** — an α stuck at 1.0 means it trained plain flow matching.

### Step 2 — `fm` parity vs Gen11 (gate G1)

```bash
python mix_uav_test/gates_mix_uav.py --gates G1 \
  --gen11-savepath logs/UAV_FM/uav-corridor/flow_matching_v3_uav/<exp>/6 \
  --gen15-savepath logs/UAV_MIX/uav-corridor/mix_uav_fm/<exp>/6
```
Structural half only. The behavioural half is a rollout comparison — run the same scene/seed/
`--projection diffuser` under both `FM_v3_uav_test/eval_fm_uav.py` and
`mix_uav_test/eval_mix_uav.py --engine fm`, and compare success/steps/goal_dist.
⚠️ **Do not use a `dpcc-t*` variant for this** — `fix_5` (§4 D3) intentionally changes it.
⚠️ **Match K explicitly** — Gen11 really ran K=10 (§5), so pass `--flow-steps 10` to Gen15.

### Step 3 — full train, one arm at a time

```bash
for e in fm mf af; do
  bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/train_all_scenes.sh \
      $e "empty corridor s_curve pillars" "6 7 8 9 10"
done
```
One job per scene, seeds loop inside; `--time` scales with seed count (24 h cap).

### Step 4 — the K sweep (the actual experiment)

```bash
for e in fm mf af; do
  bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh \
      $e corridor "6 7 8 9 10" "1 2 4 10 20"
done
```
🔴 **Matched budget or nothing** — same K list for every arm. Each K writes its own
`E{engine}_K{k}_mpc{B}_{controller}_T{thresh}/` folder, so nothing overwrites anything.

### Step 5 — aggregate, per arm (never pooled)

```bash
for e in fm mf af; do
  bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/aggregate_summaries.sh \
      $e "empty corridor s_curve pillars" dpcc-c
done
```
→ `logs/UAV_MIX/uav_mix_<engine>_ALL_SCENES_SUMMARY.json`.

### One-shot alternative (single arm, end to end)

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/uav_mix_all_pipeline.sh \
    mf "empty corridor s_curve pillars" "6 7 8 9 10" 20 dpcc-c
```

---

## 9. What is NOT verified

- **Nothing has been executed.** Only `py_compile` (all files pass) and `bash -n` (all 9 scripts
  pass). No import test, no forward pass, no shape check — this container has no torch.
  Gate G0 is the first real check and it runs on the cluster.
- **G1 behavioural parity** is unproven until a Gen11 and a Gen15 `fm` run are compared.
- **Backbone param equality (G3)** is asserted by a gate but not yet observed.
- **`mf`/`af` on UAV data have never trained.** Both were developed on state-only avoiding-d3il
  with a DiT/SiT backbone; this is their first run on a 9-D UAV transition with a U-Net backbone
  and `SafeLimitsNormalizer`. Expect the first smoke train to surface something.
- **The `af` arm's α schedule** is pinned to `n_train_steps=100000`. If the training budget is
  ever changed, `af_alpha_end_step` must change with it or `AlphaFlowODE` will assert.

---

## 10. Still open (decisions, not code)

1. **Sync the K fix to Gen11?** (§5). Gen11's numbers are labelled with a K it did not run.
2. **Sync `fix_5` to Gen11's `policies.py`?** (§4 D3).
3. **No DDPM/DPCC baseline exists for UAV**, so Gen15's claim is capped at *"vs Gen11 naive
   FM + DPCC"* — never *"beats DPCC"*. Adding a `ddpm` arm is one more registry row plus a
   training sweep (`diffuser/models/diffusion.py` has the state-only `GaussianDiffusion` on the
   same temporal-UNet family).
4. **iMF stays out** (plan §1.4). The registry has room for a 4th key.
5. **DiT/SiT arms** are buildable today (§4 D1) but off by config — a deferred appendix, to be
   run only after the UNet-locked comparison lands.
