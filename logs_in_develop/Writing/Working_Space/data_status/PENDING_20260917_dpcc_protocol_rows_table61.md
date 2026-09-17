# PENDING — 2026-09-17 · the two missing rows of Table 6.1 (D3IL-avoiding at DPCC's protocol)

**Scope: one table.** This is a focused companion to
[`PENDING_20260916_missing_data_and_analyses.md`](PENDING_20260916_missing_data_and_analyses.md), which stays
the general ledger and is **not** superseded by this file. It exists because `tab:avoiding-dpcc-protocol`
carries the primary comparison of Chapter 6 and is the only table in the thesis still incomplete.

## Why this table matters

Chapter 6 makes every comparison first at the protocol of \ac{DPCC} — five training seeds × two episodes per
geometry, ten episodes per geometry, the released `n_trials: 2` — and repeats it at twenty episodes per seed
as stronger evidence. The first of those two is the one the baseline's published numbers are directly
comparable with, so a missing row there is a missing leg of the main claim.

## State on 2026-09-17 (after v3.28)

| model | budget | 5 seeds × 2 episodes, temporal U-Net | in the table |
| :-- | :-- | :-- | :-- |
| Diffusion (DPCC baseline) | K=20 | ✅ `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` | yes |
| MeanFM | K=1, K=2 | ✅ **recovered 2026-09-17** — see the trap below | yes |
| FM | K=20 | ✅ `H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE` | yes |
| **FM** | **K=1, K=2** | ❌ **missing** — only the 20-episode (`_msg20trials`) runs exist | *pending* |
| **CI-MeanFM** (α_end 0.2) | **K=1, K=2** | ❌ **missing** — U-Net checkpoint exists for **seed 6 only** | *pending* |

### The trap that hid the MeanFM rows

`Folder_Name` in the batch export does **not** carry the backbone. The 2-episode MeanFlow cells exist under
both `…_bbmf_dit_…` and `…_bbunet_…` model folders with the *same* `Folder_Name`, so selecting on the folder
name alone silently mixed the two, and the architecture-matched U-Net rows were reported as missing. They are
separable only by `Full_Path`. `analysis/avoiding_rules_by_protocol.py` now selects on
(`Folder_Name`, backbone substring of `Full_Path`) and documents this at the top.

**Check any other cell selected by `Folder_Name` alone before declaring it missing.**

### Why the 5-seed α-Flow cells there do not count

`H8_K{1,2}_Meuler_T0.5_A0.5_B1_…AlphaFlowODE` over five seeds is `bbsit` with `ae0.0`: the consistency ratio
annealed to zero, which is the analytic average-velocity target. It is not the consistency-interpolated model
of the thesis (α_end = 0.2) and is not its backbone.

## What is needed

1. **FM at K=1 and K=2, 5 seeds × 2 episodes.** Evaluation only, cheap; the checkpoints exist.
2. **CI-MeanFM (U-Net, α_end 0.2) at K=1 and K=2.** Needs four more trainings (seeds 7–10) before the
   evaluation; ≈430 MiB each, and cluster disk was ≈7 GB free on 2026-09-16.

Commands for both are in
[`DA_in_Paper/plotting/REQUEST_20260917_avoiding_dpcc_protocol.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260917_avoiding_dpcc_protocol.md).

## Where it shows in the draft

- `v3/chapters/06_results.tex`: the two *pending* rows of `tab:avoiding-dpcc-protocol` and the `\hole` under it.
- `v3/chapters/05_setup.tex`: the `\provisional` in §5.6.1 naming which models the protocol currently covers.
- Ledger row **R8** in `PENDING_20260916_…` (🔴 → 🟡 in v3.28).

## When it lands

Re-run `python3 Data_Analysis/DA_in_Paper/analysis/avoiding_rules_by_protocol.py` (add the new folders to its
`DPCC protocol` block), fill the two rows, delete the `\hole` and the `\provisional`, and close R8.
