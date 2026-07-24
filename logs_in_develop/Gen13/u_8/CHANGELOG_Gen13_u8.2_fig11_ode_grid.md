# Gen13 u_8.2 — the REAL Fig. 11: ODE-step grid (correcting u_8)

**Date:** 2026-07-20 · **Corrects:** `CHANGELOG_Gen13_u8_fig11_imf_vs_fm.md` (same folder).
**Track back to the code:** `grep -rn "u_8.2" HardFlow/run/eval_imf.py`

---

## 1. u_8 reproduced the wrong figure — my error

Given the appendix description, Fig. 11 is a **2×6 grid tracking ONE planning instance across ODE steps 0→10**:
- **top row** — intermediate samples `x_τ`: chaotic near τ=0, contracting to a clean path at τ=1
- **bottom row** — terminal predictions `x̂1` decoded at each step

u_8 instead built a **2-panel side-by-side of final plans** (iMF vs FM). I built it from the caption alone, without the appendix detail, and the axis of variation was wrong: **method**, where it should have been **ODE step**.

| | u_8 (wrong) | Fig. 11 (actual) |
|---|---|---|
| Layout | 2 panels | **2×6 grid** |
| Varies along | method | **ODE step** |
| Shows | final plan only | **the trajectory forming** |

## 2. It was a data-discarding bug, not a capability limit

Both rows were **already computed and returned** by the policy — I was throwing them away:

```python
planned_fan.append(np.asarray(x_chain)[0, -1, :, :])   # kept 1 of 11 ODE states
x1_fan.append(np.asarray(x1_est)[0, -1, :, :])         # discarded the rest
```

`x_chain` is `(1, oc_N_steps+1, H, T)` = **11 states** (FM, `ode_t_steps=10`) / **6** (iMF, K=5), and `x1_estimate()` loops over *all* of them. So the entire generation process was one slice away. **No new computation, no retraining, no re-running the model.**

## 3. What changed

| File | Change |
|---|---|
| `run/eval_imf.py` | keep the **full** chains: `chain_full` / `x1_full` buffers; both added to `{run_id}_fan.npz` |
| `run/make_fig11_ode_grid.py` | 🆕 renders the 2×N grid; `--both` stacks two runs into **4×N** (iMF `x_τ` / iMF `x̂1` / FM `x_τ` / FM `x̂1`) |
| `Slurm_Codes/sbatch/hardflow/fig11_compare_hardflow.sh` | now emits the grid too |

**Reused existing code as requested** — `AvoidingTrajectoryPlotter._configure_axis(compact=True)`, `add_environment_elements()`, and upstream `style="predicted"`. Note the style is *correct* here: one plan per cell is exactly what it was designed for (see `../../HF_iMF/Research/MEMO_hardflow_fig11_predicted_style.md`).

**u_8's side-by-side is kept** — still a useful method comparison, just not a Fig. 11 reproduction.

## 4. Deliberate simplifications (similar, not identical)

- The paper draws a dashed **ground-truth "future" reference**; HardFlow's eval carries no such reference (it likely comes from the demo set). The **executed rollout** is drawn as grey context instead.
- ODE steps are **subsampled** to `--n_cols` (default 6). iMF at K=5 has only 6 states so all are shown; FM's 11 become 0,2,4,6,8,10 — matching the paper's spacing.

## 5. Why the bottom row is the Gen13 payoff

The paper notes FM's `x̂1` shows *"positional shifting or deformation"* at early ODE steps — **that is exactly the Euler-shot error** `x̂1 = z + (1−τ)·v`, worst when `1−τ` is large. iMF's `x̂1 = z + (1−τ)·u` is the **exact endpoint map**, so it should be well-localised from **step 0**.

The 4×N grid puts those two bottom rows directly above each other: **if the seam swap works, it is visible here** — and this is the one place the mechanism shows up qualitatively rather than as an NFE count.

## 6. Verified (container)

`py_compile` clean; rendered the 4×N grid end-to-end from synthetic chains and inspected it — convergence reads correctly (chaotic step 0 → clean final), row labels/titles/obstacles correct, iMF's 6 states and FM's 11 subsampled independently. Frozen files (`run/eval.py`, `flow_policy.py`) untouched.

## 7. ⭐ COMMAND (unchanged — same job now emits both figures)

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/fig11_compare_hardflow.sh
```

**Outputs:**
```
logs/hardflow/avoiding-v0/eval/fig11_ode_grid_run0.png   ← u_8.2, the real Fig.11 structure
logs/hardflow/avoiding-v0/eval/fig11_imf_vs_fm_run0.png  ← u_8, side-by-side final plans
```
Knobs: `N_COLS=8` (more ODE columns) · `PLAN_IDX=0` (which replan) · `RUN_ID` · `N`.

**Re-plot without re-running** (post-processing only):
```bash
python run/make_fig11_ode_grid.py \
  --dir logs/avoiding-v0/eval/diag_smooth_imf_guided_K5_n3 \
  --dir2 logs/avoiding-v0/eval/diag_smooth_fm_guided_K10_n3 \
  --both --run_id 1 --n_cols 6 --out logs/avoiding-v0/eval/grid_run1.png
```

⚠️ Requires npz written **after** u_8.2 (`chain_full` key). Older dumps raise a clear error telling you to re-run.
