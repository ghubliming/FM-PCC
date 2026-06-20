# U2 — D3IL Visual-Aligning Baseline: paper-faithful eval upgrade

**Date:** 2026-06-20
**Goal:** make `d3il_visual_aligning_baseline_test/` actually reproduce the D3IL paper's reported metrics
(DDPM-ACT image aligning = **success 0.278 / entropy 0.139**), by closing the gaps identified in
[D3IL_Metrics_SuccessRate_Entropy_Explained.md §4G](../../D3IL_Code_Guide/D3IL_Metrics_SuccessRate_Entropy_Explained.md).
**Status:** code complete, `py_compile` / `bash -n` / YAML clean, **entropy unit-tested** against the
native d3il formula. Cluster run pending (no local GPU/MuJoCo).

---

## Why
The baseline trains the correct paper agent (`ddpm_encdec_vision` = DDPM-ACT image) but its eval was a
**smoke-test**: it computed `success_rate` + `mode_0_rate` only — **no behavior entropy**, the paper's
diversity metric — and defaulted to a tiny `3 × 1` rollout scale where entropy is undefined. You could not
compare to `0.139` at all. (See §4G "3 gaps": **G1** eval scale, **G2** missing entropy, **G3** model
selection.)

---

## Files changed

| File | Change |
|---|---|
| `d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py` | **G2** add `compute_behavior_entropy()` (paper Eq. 2 / `aligning_sim.py:178-194`); add `entropy` + `score=0.5·(SR+H)` to results, print, and cross-seed aggregate. **G1** add `--n-trajectories` and `--paper` CLI; record `n_contexts`/`n_trajectories_per_context` in output; warn when trajs/ctx is too small for entropy. |
| `d3il_visual_aligning_baseline_test/d3il_eval_config.yaml` | Document smoke vs **paper-faithful** scale (60×18); annotate the two scale keys. |
| `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh` | New `$4="paper"` arg → passes `--paper` (60 ctx × 18 traj) for faithful entropy. |

> **G3 (model selection)** is **documented, not auto-changed** — training still uses val-loss
> checkpointing; the paper selects best-task-performance every 1/10 training. Noted as a follow-up so we
> don't silently diverge (see "Not done").

---

## G2 — the entropy implementation (the core fix)

`compute_behavior_entropy(records, n_contexts, n_trajs, n_modes=2)` — a faithful port of d3il
`simulation/aligning_sim.py:178-194` (= paper Eq. 2):
```
for each context c:
    among SUCCESSFUL rollouts, count mode-0 vs mode-1  →  divide by n_trajs   = p̃(m|c)
p(m|c) = p̃(m|c) / (Σ_m p̃ + 1e-12)                      # row-normalize
entropy_c = −Σ_m p(m|c)·log(p(m|c)) / log(n_modes)       # base-|B| ⇒ ∈[0,1]
entropy   = mean_c(entropy_c)                            # MC over the S0 contexts
```
- **success-conditioned** (only `success==True` rollouts count) — matches `successes[c,:]==1`.
- **base-`|B|` normalized** (`/log(n_modes)`) ⇒ range exactly **[0,1]**; `|B|=2` for aligning.
- divides by **`n_trajs`** (not #successes), exactly like the native code.

**Unit test (logic verified locally, no GPU needed):**
| Case | Input | Expected | Got |
|---|---|---|---|
| even 50/50 (1 ctx) | mode 0 & 1, both succeed | 1.0 | **1.0** |
| collapse (1 ctx) | both mode 0 | 0.0 | **0.0** |
| zero successes | none succeed | 0.0 | **0.0** |
| mixed (2 ctx: 50/50 + collapse) | — | 0.5 | **0.5** |
| cross-check vs native d3il torch formula | case "mixed" | match | **match (0.5)** |

---

## G1 — paper-faithful scale, made easy

Entropy needs many rollouts per context (paper: **60 contexts × 18 = 1080**). Three ways now:
```bash
# 1) one-flag CLI preset (no file edit)
python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py --paper

# 2) explicit override
python ... eval_d3il_visual_aligning.py --n-contexts 60 --n-trajectories 18

# 3) cluster sbatch ($4 = "paper")
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh ddpm_encdec_vision 42 none paper
```
The smoke default (`3 × 1`) is unchanged for fast checks; eval now **warns** when `n_trajs < 8` that
entropy isn't trustworthy, and records the scale in every `results_seed_*.json`.

---

## Output (now paper-comparable)

`results_seed_{s}.json` and `aggregate_results.json` now include **`entropy`** and **`score`** alongside
`success_rate`. Per-seed print:
```
[ Seed 42 Summary ]  (60 ctx × 18 traj = 1080 rollouts)
  success_rate:       0.xxx
  entropy:            0.xxx          ← NEW (paper metric)
  score (0.5·SR+H):   0.xxx          ← NEW
  ...
```

---

## How to replicate `0.278 / 0.139`
1. Train DDPM-ACT image agent (6 seeds, realistic epochs — **not** the vision config's debug `epoch: 4`).
2. Eval with `--paper` (60×18).
3. Compare printed `success_rate` / `entropy` to `0.278 ± 0.071` / `0.139 ± 0.054`; acceptance = within
   ~1 std **and** the state-vs-image gap reproduced (state ≈ 0.85/0.75). Read with
   `npz_analysis/analyze_npz.py` or directly from the JSON.

---

## Not done (follow-ups)
- **G3 best-task-performance model selection** during training (paper §4.2: eval every 1/10, keep best) —
  still val-loss checkpointing. Approximation for now; flagged for a U3 if exact replication drifts.
- Native-sim delegation (route eval through `Aligning_Sim.test_agent`) was considered but **not** taken —
  the existing in-script rollout loop already yields per-rollout `mode`/`success`/`context`, so adding the
  entropy formula on top is lower-risk than swapping the whole eval path. Both give the same metric.
- No commit/push (per policy).
