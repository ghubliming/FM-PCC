# Gen13 fix-7.2 — capture the RAW (pre-projection) plan: DPCC's `diffuser` equivalent

**Date:** 2026-07-20 · **Extends:** `CHANGELOG_Gen13_fix7_smoothness_diagnostic.md` (same folder).
**Track back to the code:** `grep -rn "fix_7.2" HardFlow/run/eval_imf.py HardFlow/hardflow/models_flow/imf/imf_flow_policy.py`

---

## 1. The two observations that prompted this — both correct

| Observation | Verdict |
|---|---|
| *"I only see <10 trajectories in the fan"* | ✅ **Correct and expected.** `replan_steps=8` over ~52 steps ⇒ **~6–7 plans per episode**. Each is one H=**16**-step line. Not a bug — see `../../HF_iMF/Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md` §3. |
| *"the lines are only after the projection, right?"* | ✅ **Correct.** fix_7 stored only `x_chain[-1]` — the post-NLP plan. The raw generative output was never surfaced. |

## 2. Where the raw plan comes from (it already existed)

`hardflow_new_forward` starts with `self.warmstart(conditions)` (`flow_policy.py:1302`), a **pure generative rollout with no NLP** — HardFlow's exact analogue of DPCC's un-projected **`diffuser`** variant. It is computed on every plan and then never exposed (`best_final_dof` at `:1303` is assigned but unused).

## 3. How it is captured without touching frozen code

The base forward calls `self.warmstart(...)`, so **overriding only `warmstart` to stash its result** yields the raw plan for *both* backbones — no need to copy the 150-line forward method.

- **`WarmstartCaptureMixin`** (`imf_flow_policy.py`) — `_stash_warmstart()` + `raw_plan()`, which rebuilds and un-normalizes the raw trajectory using **exactly** the same assemble/reshape/`unnormalize_chain` path the projected chain uses (`flow_policy.py:1375–1392`), so raw and projected are directly comparable.
- **`InstrumentedFlowPolicy(WarmstartCaptureMixin, FlowPolicy)`** — HardFlow's original policy **plus** the stash, nothing else. Used when `backbone="fm"`, so the diagnostic works identically for FM and iMF.
- `ImfFlowPolicy` gains the same stash in its `warmstart` override.

`run/eval.py` and `flow_policy.py` remain **untouched** (verified).

## 4. What is new in the outputs

| Output | Addition |
|---|---|
| `trajectories.csv` | **`plan_roughness_raw`** alongside `plan_roughness` |
| episode line | `rough=<projected> raw=<raw>` |
| `{run_id}_fan.png` | **blue** = RAW pre-projection plans, **dark grey** = projected plans, orange = x̂1, black = executed |
| `{run_id}_fan.npz` | new `raw` array (for post-processing / u_8) |
| job summary | per cell: `projected=… raw=… ratio=…x` |

**The `ratio` is the headline number:** raw-roughness ÷ projected-roughness = *how much smoothing the NLP performs*. This is the paired, within-plan version of the before/after-projection measurement requested in `DISCUSSION_foresight_fan_and_smoothness_paradigms.md` §Part 6 item 3 — far stronger than comparing separate guided/unguided runs, because both numbers come from **the same planning call on the same sample**.

## 5. Verified (container)

- `py_compile` clean; `InstrumentedFlowPolicy.raw_plan` present.
- Rendered a raw-vs-projected fan from synthetic data — blue raw lines under dark-grey projected lines, legend correct.
- Metric ordering confirmed on synthetic input: `raw = 1.72e-03` ≫ `projected = 4.6e-33`.
- `git status`: frozen files untouched.

## 6. ⭐ COMMAND (unchanged — same job now emits raw too)

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_hardflow.sh
```
~10 min, n=5, the 2×2 matrix. Knobs: `N=10`, `CELLS="imf:guided fm:guided"`.

## 7. What to look for

1. **`ratio` on the guided cells** — if the NLP is manufacturing smoothness, raw ≫ projected. This directly tests the central claim of the smoothness discussion.
2. **iMF raw vs FM raw** — quantifies the coarse-field penalty (iMF ≈0.37/dim vs Gen3v4's 0.25/dim) *before* the projection hides it.
3. **iMF projected vs FM projected** — if these converge despite different raws, the projection is erasing the field-quality gap, explaining 98.5% vs 100%.
4. **The blue-vs-grey gap in the fans** — the visual form of the same thing; large gaps mean the NLP is doing heavy lifting.

⚠️ On the **unguided** cells (`original`/`original_imf`) there is no warmstart and no projection, so `plan_roughness_raw` is `nan` and the summary shows `raw=n/a`. That is expected — for those cells `plan_roughness` *is* the raw number.
