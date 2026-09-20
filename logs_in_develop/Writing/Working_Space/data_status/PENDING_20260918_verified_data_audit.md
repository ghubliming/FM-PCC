# PENDING — 2026-09-18 · what is really missing, after a cell-by-cell audit

**Supersedes nothing.** [`PENDING_20260916_…`](PENDING_20260916_missing_data_and_analyses.md) stays the
general ledger (rows R1–R19, D1–D10) and [`PENDING_20260917_…`](PENDING_20260917_dpcc_protocol_rows_table61.md)
stays the Table 6.1 file. This one records **what was checked, what turned out to exist, and what is
confirmed absent**, so the next pass does not re-derive it.

**Submission wave for everything below:** [`SLURM_RUNBOOK_20260918_pending_runs.md`](SLURM_RUNBOOK_20260918_pending_runs.md)
(driver `Slurm_Codes/temp_bash/pipeline_20260918_pending_all.sh`). It also records the two items that turned out
**not to be runs at all**: the diffusion baseline can never take endpoint projection (no velocity field, R22),
and no endpoint row exists at $\nfe=1$ at any activation threshold (R20/R21 floors).

## How this was checked

| source | what was done |
| :-- | :-- |
| `Data_Analysis/analysis_results_checkpoint/15-09/` | all three batches enumerated cell by cell: avoiding (model × $\nfe$ × backbone × protocol), alignment (model × $\nfe$ × geometry × projection variant), UAV (scene × model × $\nfe$ × projection variant) |
| `Data_Analysis/DA_Result_Curated_MD/` | read for runs the checkpoint might not carry; it agrees with the audit and adds one bug (below) |
| `temp/` tree and `Slurm_Codes/logs/` | newest batch is `temp/1409`, newest log directory `2026-09-15` |

**Nothing has landed since the 15-09 checkpoint.** Every ⏳ row of the two earlier PENDING files is
therefore still open — none is "ready but unread". Treat them as *submitted-or-not-yet-run*, not as
*waiting to be analysed*.

## Found to exist after all (do not re-request)

| what | where it was hiding |
| :-- | :-- |
| MeanFM (U-Net) at $\nfe=1,2$, 5 seeds × 2 episodes | same `Folder_Name` as the DiT runs; separable only by `Full_Path` (`bbunet`). Filled into `tab:avoiding-dpcc-protocol` in v3.28 |
| FM at $\nfe=5,20$ and MeanFM at $\nfe=5,10,20$, 5 seeds × 20 episodes | present all along; added to `tab:state-headline` in v3.32 |
| CI-MeanFM (U-Net, $\alpha_{\mathrm{end}}=0.2$) at $\nfe=1,2,5,10,20$, seed 6, 2 episodes | the `temp/0309` corpus, not the main batch; now the single-seed series in `fig:k-ladder` (v3.32) |
| Diffusion at $\nfe=1,5,10$, 5 seeds × 2 episodes | present; added to `tab:avoiding-dpcc-protocol` in v3.32 |

## Confirmed missing

### A · D3IL-avoiding
| # | missing | ledger |
| :-- | :-- | :-- |
| A1 | **FM at $\nfe=1,2$ at DPCC's protocol** (5 × 2). It exists at $\nfe=5,10,20$ only | R8, R19 |
| A2 | **CI-MeanFM (U-Net) on seeds 7–10** — every CI-MeanFM number in the thesis is seed 6 | R6 |
| A3 | **Diffusion at $\nfe=1,2,10$ at the extended protocol** (5 × 20); they exist at 5 × 2 only | R18 |
| A4 | Diffusion at $\nfe=2$ at either protocol (absent everywhere) | R18 |

### B · D3IL-aligning
| # | missing | ledger |
| :-- | :-- | :-- |
| B1 | **The diffusion baseline has no tightened geometry at all** — only `combined_5`, 6 projection variants, no endpoint projection. Confirmed in the batch and in `NOTEBOOK_20260829` H10 | R2 |
| B2 | FM at $\nfe=2,10$ and CI-MeanFM at $\nfe=10$ (no budget ladder for either) | R16 |
| B3 | ~~Held-out contexts, and 50–60 contexts~~ ❌ retired 2026-09-20 (author's call) | R3 |

### C · The quadrotor, and this is the largest gap
The endpoint-projection (`hardflow_sls`) matrix is far from complete, and the baseline is missing
per-step cells as well.

| scene | what exists | what is missing |
| :-- | :-- | :-- |
| UAV-corridor | per-step $r$/$c$/$t$ at $\nfe=1,3,5$ for all three flow models and the baseline; endpoint only as **single candidate** and **$t$**, and only at $\nfe=3,5$ | endpoint **$r$ and $c$ at every budget**, and **all endpoint rows at $\nfe=1$**; the baseline has **no endpoint row at all** |
| UAV-pillars | the full 11 configurations at $\nfe=5$ for MeanFM, FM and CI-MeanFM | **all four endpoint configurations at $\nfe=1,2$**; the baseline has **no endpoint rows** and **no per-step $r$** (5 of 11 configurations only) |
| UAV-s-curve | per-step $r$/$c$/$t$ plain and tightened; endpoint at af $\nfe=5$, fm $\nfe=20$, mf $\nfe=10$ | those endpoint rows **predate the switched-wall fix and are excluded** (R10), so the scene has no usable endpoint data; the baseline again has no $r$ row and no endpoint rows |

New ledger rows for this section: **R12** (baseline's pillars configurations), **R20** (endpoint $r$/$c$ on
UAV-corridor and endpoint at $\nfe=1$), **R21** (endpoint at $\nfe=1,2$ on UAV-pillars), **R22** (the
baseline under endpoint projection on any scene).

### D · Closed, not missing
Multi-seed UAV (R1) and multi-seed alignment (R4) are ❌ closed by author decision after the cost
assessments; single seed 6 is final for both.

## One bug to carry, not a gap

`DA_20260819_ntrials20_…` §"AlphaFlow K=2 collapses under the `c` projector": CI-MeanFM at $\nfe=2$ with
`dpcc-c` scores 0.12/0.16 with `n_steps` at the episode cap, while the same checkpoint scores 0.91–1.00
under $r$/$t$. It is projector-specific divergence, to be investigated rather than reported. The thesis
does not quote that cell; `tab:state-headline`'s MeanFM $\nfe=2$ `c` row shows the analogous 98.0-step
stall and is reported as a selection-rule effect.
