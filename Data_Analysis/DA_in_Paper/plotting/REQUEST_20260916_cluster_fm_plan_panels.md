# REQUEST — cluster runs for the missing panels of the plan matrix (`fig:raw-plans`)

**2026-09-16 · needed for** `fig:raw-plans` (v3 §6.1.3), now a 2×4 matrix: MeanFlow, flow matching,
diffusion, consistency training × $\nfe\in\{1,2\}$. Existing: MeanFlow K1, K2; diffusion K1.
**Missing (5):** flow matching K1, K2 · diffusion K2 · consistency training K1, K2.

The flow-matching run is spelled out below. The other two follow the same pattern with their own
eval jobs; keep seed 6, both-hard, variant `diffuser`, default `n_trials: 2`:
- consistency training: `eval_flow_matching_v3_alphaflow.py` with `AF_BONE=unet AF_ALPHA_END=0.2`
  (the `…_ae0.2_…` checkpoint of `tab:state-models`), budgets 1 and 2;
- diffusion K2: the diffusion K1 panel came from a model *trained* at K=1
  (`H8_K1_T0.5_Dmodels.GaussianDiffusion`). Use the K=2-trained checkpoint if one exists; if not, that
  panel stays a placeholder rather than mixing training budgets across the row.

## CORRECTION 2026-09-18 (v3.39): four of the five are a DOWNLOAD, not a run

The claim below -- "there is no flow-matching plan file anywhere on disk, so it has to be re-run" --
confused *not on the AI container* with *never run*. `analysis_results_checkpoint/15-09/
batch_avoiding_combined_20260915_100757/candidates_detailed.csv` lists the evaluations themselves:

| panel | run folder on the cluster | seeds |
| :-- | :-- | :-- |
| FM $K{=}1$ | `logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials` | 6--10 |
| FM $K{=}2$ | same model folder, `H8_K2_Meuler_T0.5_…_msg20trials` | 6--10 |
| CI-MeanFM $K{=}1$ | `…flow_matcher_v3_alphaflow.models.AlphaFlowODE_msgafon02_s6`, K1 | 6 |
| CI-MeanFM $K{=}2$ | same, K2 | 6 |

So these four panels are a **fetch** of `…/6/both-hard/diffuser.png` from those folders (the dashboard the
eval writes as it runs), at the same crop as the existing three. Only re-run them if the fetch shows the
dashboard was not saved for that job.

**The fifth panel cannot be fetched or run as specified.** The diffusion baseline has no $K{=}2$
checkpoint --- every `GaussianDiffusion` folder in the batch is `K20` --- because its step count is fixed
when the noise schedule is discretised at training time. Either train a $K{=}2$ diffusion model or leave
that cell empty; the thesis leaves it empty.

## Why the flow-matching panels were thought to need the cluster

Checked exhaustively on the AI container: there is **no** flow-matching plan file (`*.npz`) or
diagnostic dashboard at K1/K2 anywhere on disk, and no avoiding checkpoint. The panel draws the
model's own sampled plans, so it cannot be derived from aggregate CSV rows, and evaluation is never
run locally. The MeanFlow and diffusion panels exist only because their dashboards were saved by an
earlier cluster evaluation.

## The run

```bash
FMV3_FLOW_STEPS="1 2" FMPCC_RUN_MSG=planpanel \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_fmv3_ode_job.sh
```

- `eval_fmv3_ode_job.sh` loops `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py
  --flow-steps $K` over the grid; `FMV3_FLOW_STEPS` narrows it to the two budgets needed.
- `FMPCC_RUN_MSG=planpanel` keeps the output out of the `msg20trials` result folders.
- Keep `config/projection_eval.yaml` at its default **`n_trials: 2`**: the dashboard is then
  3000×1000 px, two episodes, which is the layout the crop box below is calibrated for.
- This evaluates all seeds, geometries and projection variants in the config — as the earlier runs
  did. Only **seed 6, both-hard, variant `diffuser`** is used.
- The eval draws the plan fan exactly as the existing panels do (`plot_samples_every = horizon // 2`,
  `min(batch_size, 4)` candidates, `ax[i, 5]`), so the three panels are comparable.
- ⚠️ Use the same checkpoint as `tab:state-models` / `fig:avoiding-raw-models`
  (`…FlowMatchingODE_a1.5_b1.0_aw10`), the config default.

## What to bring back

```bash
find logs/avoiding-d3il -path '*FlowMatchingODE*planpanel*' -path '*both-hard*' -name diffuser.png
```

Two files (K1, K2), seed 6. Put them at
`Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/fig8e_plans_fm_K1_seed6.png` and
`fig8f_plans_fm_K2_seed6.png` — the names that report's §8 already reserves.

## Then, locally (no cluster)

1. In `plotting/sources.py`, move `fig_raw_plans_fm_K1` / `_K2` from `PLANNED` to `VENDORED` (source =
   the path above) and add both to `VENDORED_CROP` with box `(2312, 92, 2715, 492)`; open one first and
   confirm the top-row, last-column panel sits in that box.
2. `python3.14 plotting/prep/crop_vendored.py && python3 plotting/make_figs.py && python3 plotting/export_to_draft.py v3`
3. Add the two panels to `fig:raw-plans` in `v3/chapters/06_results.tex`, and drop the caption sentence
   saying flow-matching panels have not been rendered.
