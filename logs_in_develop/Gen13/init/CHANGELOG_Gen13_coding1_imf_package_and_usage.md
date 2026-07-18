# Gen13 coding-1 — iMF backbone package built (additive) + USAGE GUIDE

**Date:** 2026-07-18 · **Spec:** `../init/PLAN_Gen13_iMF_backbone_in_HardFlow.md` (all phases 0–4 coded) · **Rule check:** `git status HardFlow/` shows **ZERO modified files** — every pre-existing FM path is byte-identical; iMF is selected only via new entry points.

---

## PART A — Changelog

### New files (16 — nothing else touched)

**iMF package `HardFlow/hardflow/models_flow/imf/` (7):**

| File | What |
|---|---|
| `convention.py` | THE one place owning time-convention logic. HF-native formulation (τ=0 noise → τ=1 data), documented mapping to official iMF (τ=1−t, u_HF=−u_iMF), derivation of the HF-form MeanFlow identity `u = v − h·D_tot` with JVP tangents **(v_c, +1, −1)**, and the (t,r)→(τ,h) logit-normal sampling map (`sample_tau_h`). |
| `temporal_imf_unet.py` | `TemporalImfUnet` — TemporalUnet architecture (blocks **imported** from `unet.py`, not copied) + second sinusoidal embedding for interval width `h` (summed with τ-embedding) + dual-head final conv → `(u, v)`. Signature `(x, tau, h) → (u, v)`. |
| `imf_matcher.py` | `ImfMatcher` — official iMF objective ported from aux JAX `main` `imf.py forward()`: predicted-v JVP tangent (`torch.func.jvp`, detached extra forward for `v_c`), compound `V = u + h·sg(D_tot)`, adaptive `L/sg((L+eps)^p)`, u+v head losses. CFG fully dropped (D3). Reports `raw_mse_u/raw_mse_v/a0_mse` — **judge convergence on these, never on the flat adaptive `loss`**. |
| `imf_sampler.py` | `imf_sample` — K-step exact-jump composition (`τ_i=i/K`, `x += (1/K)·u(x,τ_i,1/K)`), ported from aux `origin/torch` `sample_one_step`, conditioning masked like `ConditionedODESolver`. NFE = K. |
| `imf_flow_policy.py` | `ImfFlowPolicy(FlowPolicy)` — the **seam swap**. `original_imf` (unguided K-step) and `hardflow_new_imf` (prox-NLP loop with (1) ref step = exact u-jump, (2) terminal prediction = exact u-endpoint `x̂1 = z+(1−τ)·u(z,τ,1−τ)`; NLP/pull-back/timing untouched). Overrides `warmstart` (u-jumps) and `x1_estimate` (u-endpoints); `hardflow_formulate` reuses the base CasADi build via a guidance-name shim (no base edit). Full NFE accounting (`nfe_warmstart/sampling/diag/total`). FM guidance methods deliberately unreachable (raise). |
| `imf_config.py` | `ImfTrainingConfig`/`ImfEvaluationConfig` — dataclass children of the FM configs; Gen13 defaults (H16, 100k/25k, `data_proportion=0.25`, `p_std=1.4`, K=`ode_t_steps`=2, `flow_cp=4`). |
| `README_PROVENANCE.md` | port provenance table (what came from which aux branch, what changed). |

**Entry points `HardFlow/run/` (3):** `train_imf.py` (sibling of train.py: ImfMatcher, tensorboard **optional** try-import + always-on `metrics.csv`, cosine LR over FULL budget, final checkpoint cp=4), `eval_imf.py` (sibling of eval.py: **reuses `ProxyValueModel` + `run_env` via import** — eval.py is `__main__`-guarded; identical dataset/dynamics/env/CSV flow + `nfe_*` CSV columns; no l4casadi), `imf_gates.py` (G0 shape gate + **G1 1D-GMM end-to-end gate**: h→0 limit u≈v, 1-NFE lands on ±2 modes = the sign gate, K1≈K2 W1, jump-composition W1; exit-code gated).

**Run scripts `HardFlow/run_scripts/` (3):** `train_imf.sh`, `eval_original_imf.sh`, `eval_hardflow_new_imf.sh` — same paper params as the FM siblings; `IMF_K`/`IMF_CP` env knobs; exp names `H16_imf_{original,hardflow_new}_K<k>`.

**SLURM `Slurm_Codes/sbatch/hardflow/` (2):** `train_imf_hardflow.sh` (runs gates first, **aborts if they fail**, then trains), `eval_imf_hardflow.sh` (dynamics guard → loops `IMF_METHODS × IMF_KS`, defaults `original hardflow_new` × `1 2` = the E1–E4 matrix in one job). Both source the existing `_hardflow_common.sh` bridge unchanged.

### Design decisions realized (from plan §3)
D1 additive subpackage ✓ · D2 temporal-UNet two-time backbone ✓ · D3 CFG dropped ✓ · D4 official objective ✓ · D5 interval-sampling defaults 0.25/1.4 ✓ · D6 100k budget ✓ · D7 K==ode_t_steps, K∈{1,2} defaults ✓ · D8 Level-1 seam only (MF-Newton not built — optional Phase 5) · D9 tensorboard optional ✓ · D10 `H16_imf_100k` checkpoint tree ✓.

### Verified in container
All 10 `.py` pass `py_compile`; all 5 `.sh` pass `bash -n` + are executable; **`git status HardFlow/` = additions only** (rule §0 satisfied). NOT run here (no torch in container): gates, training, eval — all cluster-side, and the train sbatch runs the gates automatically before training.

### Known limitations / notes for the record
- `nfe_*` CSV columns record the **last planning call** of each episode (representative — every plan call in an episode has identical NFE structure).
- The pull-back gain stays HardFlow's `τ` (Level 1 per plan D8); THEORY's Newton gain is the optional Phase-5 upgrade.
- Aux-repo iMF is vendored via port; no aux imports, no JAX anywhere.

---

## PART B — Usage guide (concise)

**0. Sync:** commit/push these files, `git pull` on cluster. Env: the existing `hardflow_clone` — **no new packages needed** (tensorboard optional).

**1. Gates (automatic, or manual first):**
```bash
cd /u/home/llim/FMPCC/FM-PCC/HardFlow && conda activate hardflow_clone
export PYTHONPATH="$PWD:$PWD/../Slurm_Codes/sbatch/hardflow/shims"
python run/imf_gates.py        # CPU, ~2 min. Must print ALL GATES PASSED.
```

**2. Train the iMF backbone (~hours; gates run first automatically):**
```bash
cd /u/home/llim/FMPCC/FM-PCC
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh
```
→ `logs/hardflow/avoiding-v0/flow/H16_imf_100k/model_ema_{0..4}.pth` + `metrics.csv`.
**Read the curves:** `raw_mse_u` should drop ≥3× and plateau; ignore the flat adaptive `loss`; spikes are normal, divergence is not (G2 gate, plan §4).

**3. Evaluate — the E1–E4 showdown matrix in one job:**
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_imf_hardflow.sh
# knobs: IMF_METHODS="original hardflow_new"  IMF_KS="1 2"  IMF_CP=4
```
→ CSVs in `logs/hardflow/avoiding-v0/eval/H16_imf_{original,hardflow_new}_K{1,2}/trajectories.csv` (with `nfe_*` columns).

**4. Verdict (plan §5):** compare `H16_imf_hardflow_new_K2` (E3) against frozen B2 (FM hardflow_new: 100%/100%, 0.847 s/step, K=10). **iMF superior iff** safety 100% + 0 violations AND lower NFE/compute AND success/steps not degraded. Unguided E1/E2 vs B1 (4%/4%) is secondary/reported-only.

**FM baselines still run exactly as before** — `run/eval.py`, `eval_hardflow.sh` etc. are untouched; nothing about Gen13 needs to be "switched off".
