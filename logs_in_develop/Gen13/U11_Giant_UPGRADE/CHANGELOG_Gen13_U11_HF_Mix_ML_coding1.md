# Gen13 U11 — CHANGELOG (coding 1): HF_Mix_ML assembled (iMF + MeanFlow + α-Flow)

**Scope:** additive-only. The `imf/` package, `run/train_imf.py`, `run/eval_imf.py`,
and every existing sbatch/run_script are **untouched**. Implements
`PLAN_Gen13_U11_HF_Mix_ML_imf_mf_alphaflow.md`.

**Status:** code complete, syntax-checked locally (`py_compile` + `bash -n`). Not yet
run — training/eval is a cluster job. Gates G0–G5 (PLAN §10) to be verified on i6-gpu-1.

---

## What this delivers

One `--ml_type` flag selects the training objective inside HardFlow:

```
ml_type = imf | mf | af
   imf → ImfMatcher  (FROZEN Gen13 iMF — predicted-v_c JVP tangent)
   mf  → MfMatcher   (Gen3v6 MeanFlow — analytic-v JVP tangent)
   af  → AfMatcher   (Gen3v7 α-Flow — bootstrapped α:1→0 anneal)
```

All three share ONE dual-head backbone (`TemporalImfUnet`) and ONE u-only sampler /
NLP policy (`ImfFlowPolicy`), both reused unchanged from `imf/`. They differ **only**
in the training-time u-target. Each family has its own separated, adjustable knob block.

## Files added (11)

**Package** `HardFlow/hardflow/models_flow/ml/`
- `__init__.py` — exports `MlTrainingConfig`, `MfMatcher`, `AfMatcher`, `build_matcher`; re-exports frozen `ImfMatcher`.
- `mf_matcher.py` — MeanFlow objective, ported from `mf_diffusion.py::_p_losses_meanflow`.
- `af_matcher.py` — α-Flow objective, ported from `af_diffusion.py` (`_p_losses_alphaflow` + `compute_u_target` + `_get_ratio`).
- `ml_config.py` — `MlTrainingConfig(ImfTrainingConfig)`: `ml_type` + `mf_*` + `af_*` blocks.
- `matcher_factory.py` — `build_matcher(cfg, model)` dispatch.
- `README_PROVENANCE.md` — port map, deviations, open gates.

**Run entry** `HardFlow/run/`
- `train_ml.py` — copy-modify of `train_imf.py`; matcher via factory; AF α-step wiring; family-aware metrics.

**Bash / sbatch**
- `HardFlow/run_scripts/train_ml.sh` — `ML_TYPE` + per-family knob forwarding; derives `H16_ml_<ml_type>_<steps>k`.
- `Slurm_Codes/sbatch/hardflow/train_ml_hardflow.sh` — shared gates → `train_ml.sh`.
- `Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh` — chained train→eval orchestrator.

## The three objectives — one-line differences (all else shared)

```
z = tau·x1 + (1−tau)·x0 ,   v = x1 − x0 ,   dual head (u,v) ,   adaptive loss + aux v-head

iMF (frozen):  u-target JVP z-tangent = PREDICTED  v_c            [imf_matcher.py]
MF  (new):     u-target JVP z-tangent = ANALYTIC   v = x1 − x0    [mf_matcher.py]
AF  (new):     u-target = α·v + (1−α)·u_next ,  α:1→0             [af_matcher.py]
               (α=0 ⇒ exactly MF's JVP branch; α=1 ⇒ pure FM)
```

Both HardFlow and Gen3v6/v7 are DATA-AT-1 (τ=0 noise, τ=1 data) ⇒ **no sign flip**;
the port is a dialect swap (`_predict_uv`/`apply_conditioning`/`q_sample` → HardFlow
idiom, CFG dropped), not a re-derivation. Red-bannered in both matchers: the MF/AF
z-tangent is the ANALYTIC velocity — replacing it with a predicted `v_c` collapses them
into iMF and voids the A/B.

## Two deliberate deviations from the PLAN

1. **Eval reuses the frozen iMF path — no `eval_ml.py`.** Evaluation is
   objective-agnostic: MF/AF checkpoints load into the SAME `TemporalImfUnet` and run
   through `run/eval_imf.py` + `run_scripts/eval_{original,hardflow_new}_imf.sh`
   unchanged. The pipeline forwards `IMF_EXP_NAME=<the ML run>`, and those scripts
   auto-tag outputs `_from_<exp_name>`, so MF/AF eval dirs never collide with iMF or the
   FM baseline. Less code, and a stronger expression of the Gen13 closure finding.
2. **One `MlTrainingConfig` (namespaced blocks), not separate `mf_config.py`/`af_config.py`.**
   Subclassing `ImfTrainingConfig` guarantees `ml_type="imf"` inherits every iMF default
   byte-identically (gate G0), and the `imf_*`/`mf_*`/`af_*` namespaces keep the blocks
   separated and independently adjustable via one CLI.

## Safety / non-damage

- `imf/`, `train_imf.py`, `eval_imf.py`, and all existing sbatch are byte-unchanged.
- New ML runs use an `ml_` exp-name prefix (`H16_ml_mf_100k`, …) → **cannot** collide
  with the frozen `H16_imf_100k/_300k/_lrfix` checkpoints.
- `train_ml.sh` and the pipeline refuse to clobber a finished run (`FORCE_OVERWRITE=1`
  to override), same guard as the iMF path.
- AF enforces `af_alpha_end_step == n_train_steps` (α anneal spans the real budget);
  `train_ml.sh` sets it automatically from `N_TRAIN_STEPS`.
- `train_ml_hardflow.sh` runs `run/imf_gates.py` first (validates the shared frozen
  TemporalImfUnet + convention + sampler that all three ride on).

## How to run (your planned one-seed MF + AF vs HF baseline)

```bash
# MeanFlow — train + eval (K=1,2, n=200), one seed
ML_TYPE=mf N_TRAIN_STEPS=100000 IMF_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh

# α-Flow — train + eval
ML_TYPE=af N_TRAIN_STEPS=100000 IMF_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
```

Each trains `H16_ml_{mf,af}_100k` then evaluates it (unguided `original_imf` +
projected `hardflow_new_imf`). Results land in
`logs/hardflow/avoiding-v0/eval/*_from_H16_ml_{mf,af}_100k*/trajectories.csv`.

**Baselines to compare against** (already available, no retrain):
- **FM** — `eval_imf.py` with `backbone=fm` (HardFlow's original TemporalUnet+FlowPolicy).
- **iMF** — the frozen Gen13 `H16_imf_*` runs (or `ML_TYPE=imf` through this same pipeline).

Read `raw_mse_u` (not the adaptive `loss`) for convergence; for AF also watch `alpha`
climbing 1→0 and `discrete_frac` — an α that never moved is a silent-failure run.

## Verify on cluster (gates)

G0 iMF byte-identity (`ml_type=imf` loss curve == `train_imf.py`) · G1/G2 MF/AF train
smoke (`raw_mse_u` drops, no NaN, grad_norm sane) · G3 α endpoints · G5 eval parity
(an iMF checkpoint via `eval_imf.py` reproduces the frozen numbers). Everything is
**AI-coded here / run on cluster** — no local pipeline execution.
