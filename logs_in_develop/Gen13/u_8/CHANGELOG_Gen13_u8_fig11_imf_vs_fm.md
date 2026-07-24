# Gen13 upgrade-8 (u_8) — paper-Fig.11-style comparison figure: iMF vs original FM

**Date:** 2026-07-20
**Question answered:** can HardFlow's unused Fig. 11 styling be used to generate an **iMF vs original FM** comparison? **Yes — and it turns out to be its correct use.**
**Related:** `../fix_7/` (smoothness diagnostic, which this reuses), `../../HF_iMF/Research/MEMO_hardflow_fig11_predicted_style.md` (the finding, now updated).

---

## 1. The insight that made this work

fix_7 rejected HardFlow's `style="predicted"` because its per-point markers clutter when 5–7 planned horizons are overlaid as a fan. That objection **does not apply here**:

| Figure | Plans per panel | `"predicted"` verdict |
|---|---|---|
| Foresight fan (fix_7) | 5–7 overlaid | ❌ cluttered → kept grey lines |
| **Fig.11 comparison (u_8)** | **ONE** | ✅ **clean — its native use case** |

The paper's caption is literally *"we show **one representative planning instance** during execution."* The style was never wrong — it was being applied to the wrong figure. **Both now coexist.**

## 2. What the figure shows

A two-panel side-by-side on identical axes/obstacles:

| Panel | Backbone | Terminal prediction x̂1 |
|---|---|---|
| left | **iMF** — average velocity `u` | `z + (1−τ)·u(z,τ,1−τ)` — the **exact** endpoint map |
| right | **FM** — instantaneous velocity `v` | `z + (1−τ)·v(z,τ)` — the **Euler shot** |

Each panel: one planning instance in upstream magenta `"predicted"` style, the executed rollout in `"actual"` black, and an orange dashed x̂1 overlay (Gen13-specific, no paper counterpart).

**This is the Gen13 seam swap made visible, side by side** — the single change that produced the 1.95× NFE reduction.

## 3. Why n=3 is enough

This is a **qualitative figure, not a statistic.** The paper shows *one* instance; n=3 merely provides a couple of episodes to pick a clean representative from. (Contrast: the n=200 safety run measured a rare-event *rate*, which genuinely needed the samples; fix_7's roughness needed ~30 plans as a per-plan mean.)

## 4. Code changes (all Gen13-owned; `run/eval.py` untouched)

| File | Change |
|---|---|
| `run/eval_imf.py` | when fan capture is on, also dump **`{run_id}_fan.npz`** (raw planned horizons, x̂1, rollout, action_dim, backbone, guidance) |
| `run/make_fig11_comparison.py` | 🆕 assembles the two-panel figure — **pure post-processing** |
| `Slurm_Codes/sbatch/hardflow/fig11_compare_hardflow.sh` | 🆕 runs both guided backbones at n=3 with capture on, then builds the figure |

**Track back to the code:** the dump site is tagged in-place with the comment marker `u_8` —
```bash
grep -rn "u_8" HardFlow/run/eval_imf.py
```

**Design:** the `.npz` dump means the comparison needs **no GPU, no simulator, no model reload** — it reads what the eval already computed. It also means fix_7's diagnostic run (if executed with capture on) already produces the data.

**Defaults unchanged:** dumping only happens when `imf_plot_fan` is enabled, which is still **off by default**. No previously-run configuration behaves differently.

**Verified (not just compiled):** rendered end-to-end from synthetic `.npz` data and visually inspected — both panels correct, environment/obstacles/legend/titles right, and confirmed the `"predicted"` markers read cleanly with a single plan per panel.

## 5. ⭐ COMMAND TO RUN

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/fig11_compare_hardflow.sh
```
~2–4 min.

**Knobs:** `N=5` (more episodes to choose from) · `RUN_ID=2` (which episode) · `PLAN_IDX=0` (which replan instant; default = middle).

**Output:**
```
logs/hardflow/avoiding-v0/eval/fig11_imf_vs_fm_run0.png     ← the comparison figure
logs/hardflow/avoiding-v0/eval/diag_smooth_{imf,fm}_guided_*_n3/{0..2}_fan.{png,npz}
```

**Re-plot a different instance without re-running the sim** (post-processing only):
```bash
python run/make_fig11_comparison.py \
  --imf_dir logs/avoiding-v0/eval/diag_smooth_imf_guided_K5_n3 \
  --fm_dir  logs/avoiding-v0/eval/diag_smooth_fm_guided_K10_n3 \
  --run_id 1 --plan_idx 0 --out logs/avoiding-v0/eval/fig11_run1_plan0.png
```

## 6. What to look for

1. **The x̂1 curves (orange).** iMF's is the exact endpoint map, FM's a first-order extrapolation — if the theory holds, iMF's should sit closer to the eventual plan endpoint, especially at early τ where the Euler shot is worst (`THEORY_DeepMix_HF_iMF.md` Thm 1).
2. **Plan vs executed agreement** — how well the magenta plan predicts the black rollout, per method.
3. **Obstacle clearance** of the planned instance — qualitative context for the 98.5% vs 100% safety gap.

Note this figure is *illustrative, not evidential*: one instance cannot support a claim. The quantitative claims live in `../u_5/RESULTS_Gen13_u5_paired_n200.md` (safety/efficiency) and fix_7 (roughness).

## 7. Note

Visualization only — no method, no metric, and no previously reported number changes.
