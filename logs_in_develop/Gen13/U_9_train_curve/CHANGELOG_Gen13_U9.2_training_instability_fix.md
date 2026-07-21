# Gen13 U9.2 — root cause of the bad iMF field: effective LR ~14–27× too hot

**Date:** 2026-07-21 · **Question answered:** *"how to fix?"* (the "raw unguided traj is pure BS" problem)
**Track back:** `grep -rn "U9.2" HardFlow/run/train_imf.py HardFlow/hardflow/models_flow/imf/imf_config.py`
**Evidence base:** `INSIGHTS_Gen13_U9_300k_train_and_eval.md` §9–§10.

---

## 1. The diagnosis — a hyper-parameter porting error, not an iMF failure

**The adaptive loss makes the effective learning rate depend on the number of dimensions, and we ported the official iMF's hyper-parameters from a thousands-of-dims image task to a 96-dim trajectory task without rescaling.**

| | reduction | effective gradient scale |
|---|---|---|
| HardFlow FM (reference) | `mse_loss(...)` = **mean** over 96 dims | `1/96 = 0.0104` |
| **our iMF** | **sum** over dims, then `adp(L) = L/sg(L+eps)` | **`1/(err+eps) = 1/14.15 = 0.0707`** |

⇒ **6.8× larger gradient per head**, and with **two heads sharing the backbone, ~14×**.
Then LR: we used **2e-4** (copied from HardFlow's `train.sh`); the official iMF uses **1e-4** → another **2×**.

**Net: effective LR ~14–27× hotter than either reference.** And there was **no gradient clipping anywhere**.

### 1.1 The positive-feedback loop (this is the smoking gun)

The adaptive denominator is the *current error*, so as training improves, the gradient **grows**:

| `raw_mse_u` | grad scale `1/(err+eps)` |
|---|---|
| 19.1 (0–25k) | 0.0523 |
| 16.2 (25–50k) | 0.0617 |
| 15.5 (50–100k) | 0.0645 |
| 14.1 (100–300k) | **0.0709** |

**Prediction: instability should escalate over training.** Observed max spike in the 300k run:

| window | max `raw_mse_u` |
|---|---|
| 0–100k | 331 |
| 100–200k | **1,117** |
| 200–300k | **7,548** |

A 500×-median outlier, growing 23× across the run. **The mechanism predicts exactly what the curve shows.** This also explains the plateau: too-hot updates bounce around a wide basin instead of settling — which is why 3× more compute moved the median only 5.6%.

### 1.2 Why this also explains the weak **v-head**

The v-head is plain flow matching, yet sits at 16% of the normalized range (§9.1). It shares the backbone, so it absorbs the same over-scaled updates. **A pipeline-level LR problem damages both heads — which is precisely what we see, and what a "the 2-time object is hard" explanation cannot account for.**

## 2. The fix

| Change | File | Detail |
|---|---|---|
| **Gradient clipping** | `run/train_imf.py` | `clip_grad_norm_(..., cfg.grad_clip)` — **was absent entirely**; standard practice for JVP-based losses |
| **`grad_clip` knob** | `imf/imf_config.py` | default **1.0** (0 disables) |
| **`grad_norm` logged** | `run/train_imf.py` | added to `metric_keys` → CSV/W&B/console, so instability is now *visible* instead of inferred |
| **LR is a knob** | `run_scripts/train_imf.sh` | `IMF_LR` (default 2e-4 **kept**, so existing runs stay reproducible) and `IMF_GRAD_CLIP` |

**LR default deliberately unchanged.** The evidence says 2e-4 is wrong for this configuration, but silently changing it would make the 100k/300k runs non-reproducible. The corrected value is passed explicitly in the command below.

## 3. ⭐ THE FIX RUN

```bash
cd /u/home/llim/FMPCC/FM-PCC && git pull

IMF_LR=2e-5 IMF_GRAD_CLIP=1.0 N_TRAIN_STEPS=100000 \
  IMF_EXP_NAME=H16_imf_lrfix_100k \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh
```
~4 h. `2e-5` ≈ 2e-4 ÷ 10, the like-for-like value from §1 (target: match the reference's effective step). New `IMF_EXP_NAME` ⇒ existing checkpoints untouched.

**Then evaluate it** (remember `IMF_CP` = steps/25000 = **4**):
```bash
IMF_EXP_NAME=H16_imf_lrfix_100k IMF_CP=4 IMF_K=2 RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_imf_hardflow.sh
```

## 4. What success looks like

| signal | current (300k, LR 2e-4) | fixed run should show |
|---|---|---|
| `grad_norm` | unmeasured | stable, clipping engaging rarely after warmup |
| max `raw_mse_u` spike | 7,548 | **no escalation** |
| median `raw_mse_u` @100k | 15.5 | **< 10** if the diagnosis is right |
| `raw_mse_v` (plain FM head) | 10.2 | should approach FM-like quality |
| unguided success | **0%** | **> 0%** — the single clearest signal |

**If unguided success stays at 0% and the median does not move, the LR hypothesis is wrong** and the remaining suspects are the data ceiling (fall back to Test A/B in `INSIGHTS…` §10.1) or a deeper pipeline bug.

## 5. Standing caveat

Every Gen13 conclusion — including the fix_7.3 refutation — was measured on a model trained with this over-hot configuration. **If the fixed run produces a substantially better field, those conclusions must be re-derived.** The comparison would then be against **FM@K=2 (100% safe, 0.1894 s/plan)**, which remains the configuration to beat.

## 6. Verification (container)

`py_compile` clean on `train_imf.py` and `imf_config.py`; `bash -n` clean on `train_imf.sh`; the gradient-scale arithmetic in §1 is reproduced from the measured `raw_mse_u` values. Frozen HardFlow files untouched. Not executed here (no torch) — the fix is exercised by the run above.
