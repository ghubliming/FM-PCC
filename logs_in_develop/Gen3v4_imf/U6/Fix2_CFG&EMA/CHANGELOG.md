# Fix2 — CFG randomization + EMA-at-eval switch (Gen3v4 iMF)

**Date:** 2026-06-20

## Background

A third-party review of the Gen3v4 iMF code raised two claims:
1. Training uses a **fixed-constant** CFG (`meanflow_cfg_omega/t_min/t_max` set once from config),
   not the official iMF's per-sample randomized CFG.
2. Eval samples from **live/raw** model weights, never the EMA weights, even though `Trainer`
   tracks an EMA model throughout training.

Both were verified directly against the official repo (`/workspaces/imeanflow`) before patching —
see prior chat analysis. Findings:

- **CFG claim: confirmed real bug.** Official `imf.py` (`sample_cfg_scale`/`sample_cfg_interval`,
  ~L140-178) draws a **new** `ω ~ power-law(0, s_max]` and `(t_min, t_max) ~ U(0,0.5)×U(0.5,1)` for
  **every training sample, every step**, unconditionally in `forward()`. Our code instead filled
  `omega/t_min/t_max` once from fixed config constants — equivalent to training at a single,
  unvarying CFG operating point instead of the distribution the official model is actually built
  for. Verdict: worth fixing outright (user instruction: "cfg just fix it").

- **EMA claim: real, but not a regression.** The DPCC baseline this repo was forked from
  *also* evals raw weights — `diffuser/utils/serialization.py:75` (and every per-variant copy)
  has historically dropped the `ema` field from `DiffusionExperiment` and returns
  `trainer.model.model` (live weights), confirmed via `git log` back to the original "Add DPCC
  Code" commit. This matches finding **B6** in
  `logs_in_develop/Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/U4/DPCC_DIVERGENCE_AND_COMPARABILITY.md`:
  the *published* DPCC paper's own convention is raw weights at eval — not a bug introduced here.
  The official **iMF** paper's convention differs: `imeanflow/utils/sample_util.py:11,16` defaults
  `ema=True` at sampling. Verdict: legitimate divergence from iMF's convention, but changing it
  unconditionally would silently alter the DPCC-comparable baseline behavior. User instruction:
  add a **config switch**, default to legacy (raw), with a comment noting iMF's convention uses EMA.

## What changed

### 1. CFG — per-sample randomization (no switch, fixed outright)

**File:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py`

- Added `meanflow_cfg_beta: float = 1.0` constructor param (power-law shape for ω sampling).
- Added two helper methods, direct ports of the official iMF functions:
  - `_sample_cfg_scale(shape, device, dtype)` — `ω ~ power-law(0, s_max]`, `s_max =
    meanflow_cfg_omega`. Closed-form quantile-function sampling (matches
    `imeanflow/imf.py:140-159`); `beta=1.0` reduces to log-uniform.
  - `_sample_cfg_interval(shape, device, dtype)` — `t_min ~ U(0,0.5)`, `t_max ~ U(0.5,1)`,
    independently per sample (matches `imeanflow/imf.py:163-178`).
- Inside `_p_losses_meanflow_jvp`, replaced the old constant-fill block with calls to these two
  helpers whenever `meanflow_cfg_omega > 0`. FM-anchor samples (`r==t`, no CFG conditioning
  semantics) get the full `[0,1]` interval — i.e. no CFG restriction — matching the official
  `fm_mask` behavior ("for flow matching samples, there is no CFG interval").
- `meanflow_cfg_omega` is now semantically the **training-time sampling ceiling** `s_max`, not a
  fixed operating point. The eval-time fixed-operating-point use of
  `omega/t_min/t_max` (guided sampling, `p_sample_loop`) is **unchanged** — at inference there is
  still one chosen `(ω, t_min, t_max)`, since you must pick *something* to actually sample with.

**File:** `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` — forwards
`meanflow_cfg_beta=getattr(args, 'meanflow_cfg_beta', 1.0)` into `diffusion_config`.

**File:** `config/avoiding-d3il.py` (train block, `flow_matching_v3_imeanflow`) — added
`'meanflow_cfg_beta': 1.0`; updated comments on `meanflow_cfg_omega/t_min/t_max` to reflect the
new semantics (omega = sampling ceiling; t_min/t_max now unused at train time when interval_cfg
sampling is active, kept only as fallback values).

### 2. EMA — config switch, default = legacy (off)

**File:** `flow_matcher_v3_imeanflow/utils/serialization.py`
- `DiffusionExperiment` namedtuple re-gained an `ema` field:
  `'dataset model diffusion ema trainer epoch losses'` (was missing `ema` entirely).
- `load_diffusion()` now also loads and returns `trainer.ema_model`.

**File:** `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`
- Local `load_diffusion_with_override()` override updated to match the new 7-field tuple.
- New toggle: `use_ema = bool(getattr(args, 'eval_use_ema', False))`;
  `fm_model = fm_experiment.ema if use_ema else fm_experiment.diffusion`. Logs which weight
  source was used at eval start.

**File:** `config/avoiding-d3il.py` (eval/plan block, `plan_fm_v3_imeanflow`) — added
`'eval_use_ema': False` with a comment explaining the default (DPCC-legacy / raw weights) and
noting official iMF defaults to `ema=True`.

## Verification

`python3 -m py_compile` on all five touched files (`imf_diffusion.py`,
`train_flow_matching_v3_imeanflow.py`, `eval_flow_matching_v3_imeanflow.py`,
`serialization.py`, `config/avoiding-d3il.py`) → all OK. Also checked the sibling
`flow_matcher_v3_ode_selectable` package's own `DiffusionExperiment` (a separate, untouched
6-field namedtuple in a different module) and its eval script still compiles — confirmed no
collateral breakage from the field-count change.

No local torch/GPU runtime in this Docker env — no actual training/eval run was executed. Real
verification (loss curves, eval rollouts, raw-vs-EMA A/B) must happen on the Slurm cluster.

## Not done / follow-ups

- No retrain executed — these are code-only changes pending a cluster run.
- No raw-vs-EMA A/B comparison run yet (recommended before flipping `eval_use_ema` to `True` for
  any reported numbers).
- `meanflow_cfg_t_min`/`meanflow_cfg_t_max` config values are now dead at train time (only used as
  fallback if `meanflow_cfg_omega <= 0`); left in config for backward compatibility with the
  fallback path, not removed.
- No commit/push (per policy).
