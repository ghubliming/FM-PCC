# Gen13 fix-7 — MPC planned-trajectory SMOOTHNESS diagnostic (small-n, FM vs iMF)

**Date:** 2026-07-20
**Goal:** measure MPC trajectory smoothness and compare methods, **without** a full 200-episode run.
**Answer to "does this need code?":** partly — iMF-only would have needed none, but **comparing against FM did**, because `run/eval.py` discards the planned trajectory and is protected by the no-edit rule. Hence fix_7.

---

## 1. Why n=5 is enough here (unlike the safety run)

The n=200 run was needed because **violations are a rare-event rate** — you cannot estimate a ~1.5% rate from 5 samples.

**Roughness is not a rate — it is a per-plan mean.** Each episode does ~6–7 replans, so **n=5 gives ~30–35 planned horizons per cell**, each contributing a roughness value. That is ample for comparing means that differ by orders of magnitude. Runtime ~1–3 min per cell, ~10 min for all four.

## 2. What is measured

`plan_roughness` = **mean squared second difference of the planned x-y path**:
```
mean_t || p[t+1] - 2·p[t] + p[t-1] ||²        (metres², lower = smoother)
```
Zero for a straight/constant-velocity plan; grows with jitter. Computed on the **final planned horizon at every replan**, averaged per episode, written to `trajectories.csv` and printed per episode as `rough=`.

**Sanity-tested** (not just compiled): straight `3.7e-33` < mild jitter `1.6e-03` < zig-zag `4.0e-02` — ordering correct.

## 3. The 2×2 matrix and the hypothesis under test

| | unguided (no NLP) | guided (NLP) |
|---|---|---|
| **iMF** | raw average-velocity field | after projection |
| **FM** | raw instantaneous-velocity field | after projection |

Tests `../../HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md`: HardFlow enforces dynamic feasibility (`A·s+B·a+c = s'`) as a **hard NLP equality**, so smoothness should be **manufactured by the projection**, not inherited from the generative field.

**Prediction:** unguided iMF roughest (coarse field, 0.37/dim vs FM's finer field) → **both guided cells collapse to similar, much lower roughness**. If that holds, it explains why iMF solves ~0–2% unguided but 98.5% guided, and confirms that field coarseness is masked by the projection.

## 4. Code changes (all Gen13-owned; `run/eval.py` untouched)

| File | Change |
|---|---|
| `run/eval_imf.py` | `_traj_smoothness()`; per-replan capture; `plan_roughness` in CSV + episode line; **`backbone` switch** so this entry can also drive HardFlow's original `TemporalUnet`+`FlowPolicy`; `hasattr` guards for iMF-only NFE/NLP instrumentation |
| `imf/imf_config.py` | `backbone: str = "imf"` |
| `run_scripts/eval_smoothness_diag.sh` | 🆕 one matrix cell (`BACKBONE`/`GUIDANCE`/`N`) |
| `Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_hardflow.sh` | 🆕 loops all 4 cells + prints a roughness summary table |

**Track back to the code:** every change is tagged in-place with the comment marker `fix_7` —
```bash
grep -rn "fix_7" HardFlow/run/eval_imf.py HardFlow/hardflow/models_flow/imf/
```

**Key design choice:** rather than adding fan/roughness to the FM path (impossible — no-edit rule), the **FM backbone is routed through `eval_imf.py`**. Both methods therefore share one identical instrumented code path, making the comparison apples-to-apples by construction rather than by careful matching.

**Defaults unchanged:** `backbone="imf"`, `imf_plot_fan=False`. Every previous invocation behaves identically; only the new diagnostic scripts opt in.

## 4b. Note on HardFlow's unused Fig. 11 style

Checked whether the paper's Fig. 11 ("Visualization of the generation process … one representative planning instance") ships reusable figure code. **It does not** — but an unused `style="predicted"` exists in `rendering.py`. It was evaluated and **deliberately not adopted** (built for a single planning instance; its per-point markers clutter when plans are overlaid as a fan). The fan keeps its own clean grey styling.

Full finding recorded as a memo: `../../HF_iMF/Research/MEMO_hardflow_fig11_predicted_style.md`.

## 5. ⭐ COMMAND TO RUN

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_hardflow.sh
```
~10 min. Knobs: `N=10` (more episodes), `CELLS="imf:guided fm:guided"` (subset).

**Outputs** — `logs/hardflow/avoiding-v0/eval/diag_smooth_{imf,fm}_{unguided,guided}_K*_n5/`:
- `trajectories.csv` with the `plan_roughness` column
- `{0..4}_fan.png` — **foresight fans for both methods** (grey planned horizons, orange terminal predictions x̂1, black executed path)
- `{0..4}_real.png` — executed trajectory

The job prints a summary table at the end, so the headline comparison is visible without opening anything.

## 6. What to look for

1. **guided ≪ unguided** in both rows ⇒ the NLP manufactures smoothness (hypothesis confirmed).
2. **iMF-guided ≈ FM-guided** ⇒ the projection erases the field-quality difference — explains Gen13's 98.5% vs 100% being so close despite a much coarser field.
3. **iMF-unguided ≫ FM-unguided** ⇒ quantifies the coarse-field penalty that the n=200 analysis blamed for the residual 1.5-pt safety gap.
4. **The fans** are the visual counterpart — and for iMF the orange x̂1 curves are the *exact* endpoint map vs FM's Euler shot, i.e. the seam swap made visible.

Send me the CSVs / summary table and I'll write the analysis.

## 7. Note

This is diagnostic instrumentation only — it does not change any method or any previously reported number. The u_5 n=200 safety results stand unchanged.
