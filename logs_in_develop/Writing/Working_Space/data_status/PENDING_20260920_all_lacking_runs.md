# PENDING — 2026-09-20 · audit: what is still lacking that is **not already recorded**

Written after v3.42. The question asked was "is any run lacking?". The answer, after checking every
open item against the existing ledgers and runbooks, is:

> **Almost nothing is unrecorded.** One run found at v3.42 has no R-number. Everything else already has
> one, and in three cases an entrypoint, a tag and identity checks as well — those rows are waiting on a
> **decision**, not on a discovery.

This file therefore records only the new item and the standing blockers. It is **not** a second copy of
the gap list. The lists of record stay:

* [`PENDING_20260916_missing_data_and_analyses.md`](PENDING_20260916_missing_data_and_analyses.md) — the
  general ledger, R1–R24
* [`SLURM_RUNBOOK_20260918_pending_runs.md`](SLURM_RUNBOOK_20260918_pending_runs.md) — groups A–I, with
  submission record and identity checks
* [`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](SLURM_RUNBOOK_20260919_pillars_enlarged.md) — `pillars_xl`
* [`PENDING_20260919_avoiding_tables_locked.md`](PENDING_20260919_avoiding_tables_locked.md) — the two
  locked D3IL-avoiding tables

---

## 1 · 🆕 The one run with no R-number

### CI-MeanFM at the **extended** protocol on D3IL-avoiding, $\nfe=1$ and $\nfe=2$

| | |
| :-- | :-- |
| **What exists** | five seeds at **DPCC's** protocol (5 × 2 episodes), closed by R6/R19 at v3.41 and now in `tab:avoiding-dpcc-protocol`; at the **extended** protocol (5 × 20) the model has **training seed 6 only** |
| **What is missing** | the same checkpoints evaluated at 20 episodes per seed |
| **Cost** | two evaluations, **no training** — the checkpoints are the ones R6 produced |
| **Why it is new** | R6 and R19 both stop at DPCC's protocol; R18 covers the extended protocol for the **diffusion baseline** only. No existing R-number covers CI-MeanFM there. It became visible at v3.42, when `tab:state-headline` was cut to the same model × budget set as `tab:avoiding-dpcc-protocol` and CI-MeanFM had to be marked pending in it. |
| **What breaks without it** | `tab:state-headline` carries three models where `tab:avoiding-dpcc-protocol` carries four; the `\hole` under it says so. No claim depends on it. |

Suggested ledger entry: **R25**, 🟡, alongside R18 — same entrypoint (`eval_dpcc_job.sh`), same
`_msg20trials` suffix, same identity check (`n_trials` 20 in the config echo, seeds 6–10).
I have not edited `PENDING_20260916…` to add it; say the word and it goes in.

---

## 2 · ⛔ Recorded, prepared, and blocked on a decision — not on a run

These three are groups **G**, **H** and **I** of
[`SLURM_RUNBOOK_20260918_pending_runs.md`](SLURM_RUNBOOK_20260918_pending_runs.md). They have an
entrypoint, a driver and identity checks; their job-ID cells read `XXX` because they were never
submitted.

| group | ledger | run | blocked on |
| :-- | :-- | :-- | :-- |
| **G** | R2 🔴 | D3IL-aligning, diffusion baseline re-run so `combined_5-tightened` is generated | **`n_contexts`**: `config/visual_aligning_eval.yaml` has said `3` since 2026-08-04 while the draft reports **ten** contexts for every alignment row. A run today is not comparable with the published rows until that is resolved. The driver skips G/H without `ALIGN_OK=1`. |
| **H** | R16 ⚪ | D3IL-aligning budget ladder: FM $\nfe=2,10$; CI-MeanFM $\nfe=10$ | same |
| **I** | R18 🟠 | D3IL-avoiding, diffusion baseline at the extended protocol, $\nfe=1,10$ | nothing — ready to submit |

**G is the only lacking run in the thesis that a reader would notice as an absence in an argument.**
Without it §6.2.3 makes no constraint comparison between the baseline and the flow-based models on
D3IL-aligning, and the draft carries a `\provisional` saying so. Resolving `n_contexts` unblocks G and H
together.

---

## 3 · 🟡 Recorded, and orphaned by a draft decision

**R23 / `pillars_xl`** — config and driver implemented (Gen15 U17), verify cell **run** (`u7xlchk`:
unprojected MeanFM $K{=}5$ scores S&C 0.00 with 51–53 violating steps against 0.90 on the old geometry).
The 10-driver matrix in
[`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](SLURM_RUNBOOK_20260919_pillars_enlarged.md) has not been
submitted.

**The scene was deleted from v3 at v3.41.** §6.3.1.2 is gone, `tab:uav-pillars*` with it, and no number
from the scene is used anywhere in the draft. This is a **decision before it is a run**: the matrix is
worth the cluster time only if UAV-pillars is to return as a third quadrotor environment. R1, R12, R13,
R21 and R22 are all pillars rows and are orphaned with it.

---

## 4 · ❌ Decided against at v3.42 — recorded here so they are not re-proposed

| run | why not |
| :-- | :-- |
| D3IL-aligning at $\nfe=1$ | does not exist for any model or projector, and no R-number asks for it. It is the wrong end of this task: §6.2.2 shows the median final distance *falls* with the budget ($0.2354 \to 0.1194 \to 0.0741$\,m at $\nfe=2,10,20$), so a one-step budget would be evaluated only to be reported as a failure. |
| UAV-s-curve, per-step projection under $r$ and $c$ | 1361–3610\,ms per control step against a 30\,ms simulated-time planning interval, and **every cell of the re-run is already 0.00**. The justification is written into §6.3.1.2 beside `tab:uav-scurve-projection`. |
| D3IL-avoiding, diffusion at $\nfe=5$ | no checkpoint was ever trained there and nothing in the draft needs the point. The row that claimed it was a **flow** model misread — [`PENDING_20260919_avoiding_tables_locked.md`](PENDING_20260919_avoiding_tables_locked.md) §2. |
| D3IL-avoiding, FM at $\nfe=5,10$ at DPCC's protocol | $\nfe=5$ is seed 6 only, $\nfe=10$ does not exist; both are above FM's operating point and `fig:k-ladder` carries its budget behaviour. Dropped under the v3.42 table lock. |
| D3IL-avoiding, diffusion $\nfe=2$ at seeds 7–10 | already recorded twice — the "Blocked before submission" table of the 18-09 runbook, and `PENDING_20260919_avoiding_tables_locked.md` §3. Four trainings; ⚠️ 43 GB free on `/u/home` at 2026-09-19. Not needed by any claim. |

---

## 5 · Not runs

No cluster time; blocked on somebody's time only. All already carried by an R-number or a `\hole`.

* Aligning box **paths** per context and projector — cluster **fetch**,
  `DA_in_Paper/plotting/REQUEST_20260917_trajectory_figures.md`. `fig:aligning-contexts` (new at v3.42)
  draws the ten start and target poses; the missing half is the path between them.
* Plan smoothness — jerk, path length or curvature over plans already on disk.
* Box final-orientation error — the angle **convention** has to be checked before a column can be printed.
* Demonstration-path figures: D3IL-avoiding end-effector paths, and the UAV per-episode altitude draw.
* Appendix bookkeeping: SLURM job identifiers and git revision per batch; total compute spent.

---

## 6 · One-line answer

**One unrecorded run: CI-MeanFM at the extended protocol (§1).** One recorded run matters and is blocked
on the `n_contexts` decision, not on compute (§2, group G). One large recorded matrix is orphaned with a
scene the draft no longer contains (§3). Everything else is already in a ledger, already decided
against, or not a run.
