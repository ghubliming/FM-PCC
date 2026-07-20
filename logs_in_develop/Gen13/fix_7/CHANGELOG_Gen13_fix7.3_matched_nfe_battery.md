# Gen13 fix-7.3 — matched-budget test battery (T1–T4) to falsify the Gen13 conclusion

**Date:** 2026-07-20 · **Extends:** fix_7 / fix_7.2 (same folder). **Test design:** `RESULTS_Gen13_fix7_smoothness_2x2.md` §9.
**Track back to the code:** `grep -rn "fix_7.3" HardFlow/run_scripts/eval_smoothness_diag.sh`

---

## 1. Why

Every Gen13 comparison so far ran **iMF at K=5 against FM at K=10**. Since `K == ode_t_steps` sets **both** the NFE **and** the number of NLP projections, backbone was confounded with budget — so the efficiency claim in `../u_5/RESULTS_Gen13_u5_paired_n200.md` is **not causally attributed** (see `RESULTS_…_2x2.md` §8.4).

Combined with the visual observation that iMF's raw plans are far off the trajectory manifold while FM's look on track, the honest move is a battery designed so a **negative result is clearly visible**.

## 2. What blocked the control (the root cause)

`run_scripts/eval_smoothness_diag.sh` had FM's budget **hardcoded**:
```bash
flow_exp_name="H16_1e6steps"; flow_cp="20"; k_steps=10   # FM's native setting
```
There was literally no way to run FM at K=5. That single hardcoded value is why the decisive control was never executed.

## 3. Code changes

| File | Type | Change |
|---|---|---|
| `HardFlow/run_scripts/eval_smoothness_diag.sh` | ✏️ | `k_steps="${FM_K:-10}"` — FM budget now env-overridable. **Default 10 ⇒ every earlier invocation is byte-identical.** |
| `HardFlow/run/analyze_x1_accuracy.py` | 🆕 | **T4** — per-ODE-step terminal-prediction error `‖x̂1(k) − x_final‖` vs τ, from the npz chains. Pure post-processing (no GPU/sim/model). |
| `Slurm_Codes/sbatch/hardflow/eval_matched_nfe_hardflow.sh` | 🆕 | runs T1+T2+T3 (both backbones × K∈{1,2,5,10} × {guided,unguided}, n=20), prints a matched-budget summary, then runs T4 |

`run/eval.py`, `flow_policy.py`, `run_scripts/eval_hardflow_new.sh` — **untouched** (verified).

## 4. The tests

| # | What | Falsifies the conclusion if… |
|---|---|---|
| **T1** | FM at K=1,2,5,10 (guided) | FM@K=5 ≈ 100% safe at ≈0.48 s/plan ⇒ **iMF contributes nothing** |
| **T2** | iMF at K=1,2,5,10 (guided) | iMF@K=10 still violates ⇒ field has a hard ceiling |
| **T3** | both unguided, K sweep | iMF-unguided ~0% at every K ⇒ field genuinely poor, result rests entirely on the NLP |
| **T4** | x̂1 error vs τ | iMF's τ=0 error ≈ FM's ⇒ **the exact endpoint map is not delivering**, mechanism refuted |

**The decisive cell is FM@K=5** — the one never run. Reading across backbones at *equal K* equalises NFE and projection count simultaneously.

## 5. Verification (container)

- `bash -n` / `py_compile` clean; frozen files untouched.
- **T4 validated against a known answer** before use: synthetic chains with `0.06·(1−τ)²` error injected for FM and a flat `0.008` for iMF → tool recovered **decay 1.0× (iMF, flat)** vs a large decay (FM). So a null result on real data would be a finding, not a tooling artifact.
- T4 on the **existing** `temp/fix_7` dumps correctly reports *"no chain_full — re-run with IMF_PLOT_FAN=1"* (those predate u_8.2). The new run writes them.

## 6. ⭐ COMMAND

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_matched_nfe_hardflow.sh
```
~60–90 min (4h requested per the 2× rule). Knobs: `N=20` · `KS="1 2 5 10"` · `MODES="guided"` (halves runtime).

**Outputs:** `logs/hardflow/avoiding-v0/eval/diag_smooth_{imf,fm}_{guided,unguided}_K{1,2,5,10}_n20/trajectories.csv` (+ fans/npz), the summary table in the job log, and `x1_accuracy_matched.png`.

## 7. Scope

**n=20/cell** separates large effects (0% vs ~100%) cheaply; it will **not** resolve the ~1.5-pt residual gap — that stays the u_5 n=200 job's role. If T1 shows FM@K=5 competitive, the follow-up is a **paired n=200 at matched K**.

Diagnostic only: no method changed, and no previously reported number is affected. Interpretation was **pre-committed** in `RESULTS_…_2x2.md` §9.6 so a negative outcome is reported as such.
