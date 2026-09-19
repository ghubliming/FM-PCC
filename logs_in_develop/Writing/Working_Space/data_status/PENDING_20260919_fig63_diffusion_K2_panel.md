# PENDING — the last empty panel of `fig:raw-plans` needs a RUN, and there are two ways to run it

**2026-09-19 · opened by the v3 author's question: "is Fig 6.3 lacking data or just not fetched?"**
Answer: **neither a fetch nor impossible — it needs a new run.** Ledger row **R24**.
Related and now closed: **R5** (the four panels that were a download; they landed and are drafted at v3.41).

---

## 1. Where the figure stands

`fig:raw-plans` is a 4 × 2 matrix — four generative models × budgets $\nfe = 1, 2$ — of the plan fan an
episode produces with projection switched off. **Seven of eight panels exist and are in the draft.**

| model | $\nfe=1$ | $\nfe=2$ |
| :-- | :-- | :-- |
| MeanFM | ✅ `fig_raw_plans_meanflow_K1` | ✅ `fig_raw_plans_meanflow_K2` |
| FM | ✅ `fig_raw_plans_fm_K1` | ✅ `fig_raw_plans_fm_K2` |
| CI-MeanFM | ✅ `fig_raw_plans_af_K1` | ✅ `fig_raw_plans_af_K2` |
| **Diffusion** | ✅ `fig_raw_plans_diffusion_K1` | 🔴 **empty — this note** |

Nothing else in the figure is open.

## 2. Why it is not a download

There is no diffusion $\nfe=2$ run anywhere in the corpus of record. Checked directly against
`batch_avoiding_combined_20260919_132703/candidates_detailed.csv`: every `GaussianDiffusion` cell in it is
at $K \in \{1, 5, 10, 20\}$, and **none at 2**. So there is no artefact to fetch.

## 3. Why "impossible" was too strong — the two routes differ

The earlier note (`sources.PLANNED`, and the v3.41 caption) said the cell *cannot exist* because a
diffusion model's step count is fixed at training. **That is true of the route the existing panel came
from, and not true of the codebase as a whole.** Two plan routes run the diffusion engine:

| route | model folder in the corpus | what $K$ is | evidence |
| :-- | :-- | :-- | :-- |
| `plans/diffusion/…` — the DPCC baseline route, **which drew the existing $\nfe=1$ panel** | `H8_K1_…`, `H8_K10_…`, `H8_K20_Dmodels.GaussianDiffusion_aw10` — **one folder per $K$** | **a training decision.** `config/avoiding-d3il.py:1074` sets `diffusion_loadpath: 'f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}'` — the checkpoint path *contains* $K$, so planning at $K{=}2$ looks for a checkpoint trained at 2 | config line 1074; folder listing of the batch |
| `plans/flow_matching_v3_ode_selectable/…` with the `GaussianDiffusion` engine | a single `H8_Dmodels.diffusion.GaussianDiffusion_a1.5_b1.0_aw1`, **sampled at $K = 1, 5, 10, 20$** | **a sampling choice.** `config/avoiding-d3il.py:1268`: `flow_steps_v3` is "K in the results exp_name (**not the loadpath**)" | config line 1268; four $K$ under one model folder in the batch |

So the cell is reachable, and the question is which of the two it should be drawn from — because that
choice changes what the bottom row of the figure *means*.

## 4. The two options

### Option A — train a $K{=}2$ diffusion checkpoint (matches the existing panel)

* **Run:** one training of `diffusion/H8_K2_D…_aw10`, then one plan-dumping evaluation, seed 6,
  `both-hard`, unprojected (`diffuser` variant), same protocol as the $\nfe=1$ panel.
* **Gives:** a bottom row that reads *the diffusion baseline at its own budget of one, and of two*, each
  model trained and run at that budget — exactly what the seven other panels mean for their models.
* **Costs:** one training plus one evaluation.

### Option B — sample the existing $K{=}20$ checkpoint at 2, and redraw $\nfe=1$ the same way

* **Run:** two plan-dumping evaluations, no training, through the `ode_selectable` route at $K = 1$ and
  $K = 2$ off `H8_Dmodels.diffusion.GaussianDiffusion_a1.5_b1.0_aw1`.
* **Gives:** a bottom row that reads *one diffusion model read at two budgets* — the same reading as the
  flow rows, and the more direct illustration of the asymmetry the chapter argues.
* **Costs:** two evaluations, no training.
* 🔴 **Both diffusion panels must be replaced together.** Mixing a trained-at-1 model with a
  sampled-at-2 model in one row would compare two different things across two cells of the same row.

**Recommendation: Option B**, unless the author wants the bottom row to stay the *baseline as DPCC
deploys it*. B is cheaper, needs no GPU training, and makes the row say the thing §6.1.1.4 argues.

## 5. What the draft does in the meantime

v3.41 prints the cell empty and the caption says why: the step count of a diffusion model is fixed when
its noise schedule is discretised at training, so a panel at that budget would need a checkpoint trained
there. **That caption is accurate for the route the existing panel came from and needs no change if
Option A is taken.** If **Option B** is taken, the caption must be rewritten — the cell stops being a
statement about training and becomes an ordinary filled panel.

## 6. One thing this uncovered, already fixed in the draft

The same question applies to `fig:k-ladder` and it had been described the wrong way round. Its diffusion
series is `AVOIDING_T1_DIFFUSION_FOLDERS` = `H8_K1_…`, `H8_K10_…`, `H8_K20_…` — **three separately trained
checkpoints**, one per budget. The draft said the curve showed "a diffusion model evaluated away from the
discretisation used in training", which is a different experiment from the one that produced it. Corrected
at v3.41: the curve is three models each trained and run at its own budget, and the claim it supports —
that for diffusion the budget is a training decision a later choice cannot undo, while for a flow model it
is an inference-time dial — is unchanged and, if anything, cleaner.

**For the DA to confirm:** no other figure or table reads a diffusion budget as an inference-time choice.
`tab:avoiding-dpcc-protocol`'s diffusion rows at $\nfe = 1, 5, 10$ should be checked against the same two
routes and their protocol footnote made explicit.

---

# SLURM RUNBOOK — 2026-09-19 · R24, the `fig:raw-plans` diffusion row

**Option B was taken** (see §4): one job, two evaluations, no training. The row is redrawn as *one
diffusion model read at two budgets*, which is what the other three rows of the figure mean — so
**both** diffusion panels are replaced, and the §5 caption must be rewritten.

Driver: `Slurm_Codes/temp_bash/pipeline_20260919_fig63_diffusion_K2.sh` (gitignored throwaway, the
09-17/09-18 convention). It generates `_fig63_panel_eval.py` and `_fig63_panel_job.sh` beside itself.

## 1 · The command

```bash
bash Slurm_Codes/temp_bash/pipeline_20260919_fig63_diffusion_K2.sh            # PLAN — prints, submits nothing
bash Slurm_Codes/temp_bash/pipeline_20260919_fig63_diffusion_K2.sh submit     # SUBMIT
```

Knobs, all optional: `KS="1 2"` · `SEED=6` · `PANEL_AW=auto|10|1` · `RUN_MSG=planpanel63`.

## 2 · Run map

| | |
| :-- | :-- |
| jobs | **1** (`fig63_panel`), two evaluations inside it, 12 h wall limit, 1 GPU |
| route | `plans/flow_matching_v3_ode_selectable/` — K is a *sampling* choice here (`avoiding-d3il.py:1268`) |
| engine | `models.diffusion.GaussianDiffusion`, patched into `plan_fm_v3_ode_selectable` **in memory** |
| checkpoint | `flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.GaussianDiffusion_a1.5_b1.0_aw{10\|1}`, auto-picked, `aw10` preferred |
| budgets | K = 1 **and** 2 — the row is replaced as a pair |
| protocol | seed 6, `both-hard`, variant `diffuser` (unprojected), `n_trials: 2` |
| tag | `_msgplanpanel63` — writes into its own results folder, clobbers no published cell |

## 3 · Why no tracked file is edited

`plan_fm_v3_ode_selectable` hard-codes `diffusion='models.diffusion.FlowMatchingODE'` and has no env
override for it. Rather than `sed` a tracked config in the working tree — the trap
[`SLURM_RUNBOOK_20260918`](SLURM_RUNBOOK_20260918_pending_runs.md) documents for `n_trials` — the driver
writes a wrapper that patches the config **module dict** and then runs the stock eval through `runpy`.
That is the same data path the eval already uses for `--flow-steps`
(`eval_flow_matching_v3_ode_selectable.py:57-72`), so nothing is patched that the eval does not patch itself.

⚠️ `_fig63_panel_eval.py` is read **at job start**. Do not delete `Slurm_Codes/temp_bash/_fig63_*`
while the job is queued or running.

## 4 · Guards the driver enforces before it submits

| guard | why |
| :-- | :-- |
| `n_trials: 2` in `config/projection_eval.yaml` | 2 episodes ⇒ the 3000×1000 dashboard the crop box `(2312, 92, 2715, 492)` in `plotting/sources.py` is calibrated for. At 20 the panel would be cropped wrong. |
| `'diffuser'` present in `projection_variants` | that variant **is** the panel — the plan fan with projection switched off |
| checkpoint exists (in-job) | if neither `_aw10` nor `_aw1` is on disk, Option B is not runnable and the job exits 2 rather than half-running |

Not narrowed, deliberately: the five seeds are cut to seed 6 by `--seed`, but the projection-variant list
and the three halfspace variants are read from a hard-coded yaml path, so trimming them would mean editing
a tracked file. The extra cells are sunk cost and land as ordinary rows in the next DA batch.

## 5 · Submission record

| job | what | id | submitted | ended |
| :-- | :-- | :-- | :-- | :-- |
| `_fig63_panel_job` | GaussianDiffusion via ode_selectable, K=1 and K=2, seed 6 | **25964** | 2026-09-19 | _(running)_ |

Submitted from the cluster copy `Slurm_Codes/temp_bash/19-09-K2.sh`. Log:
`Slurm_Codes/logs/2026-09-19/*_fig63_panel_job_25964.log`. The `libtinfo.so.6` line in the
submission output is the login node's conda bash, not this job.

**First thing to read in the log** — the checkpoint the job auto-picked:
`[ fig63 ] PANEL_AW=auto -> aw10` (good: the baseline's action weight) or `-> aw1`
(runs, but the caption must say `aw1`). An exit code 2 means neither checkpoint is on disk and
Option B is not runnable at all — fall back to Option A in §4 of the runbook.

## 6 · What to bring back, and what to do with it

```bash
find logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable \
     -path '*GaussianDiffusion*_msgplanpanel63*' -path '*both-hard*' -name diffuser.png
```

Two files. Land them as
`Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/fig8g_plans_diffusion_K1_seed6.png` and
`fig8h_plans_diffusion_K2_seed6.png` — committed sources, like the other seven panels, not a gitignored drop.

Then, locally and with no cluster:

1. `plotting/sources.py`: move `fig_raw_plans_diffusion_K2` out of `PLANNED` into `VENDORED`, **and repoint
   `fig_raw_plans_diffusion_K1`** at `fig8g…` — the old K=1 panel (`Report_20260819_MF_UNet/fig6b…`,
   trained at K=1 via `plans/diffusion/`) leaves the figure with it. Add both to `VENDORED_CROP` at
   `(2312, 92, 2715, 492)`; open one and confirm the panel sits in the box before trusting it.
2. `python3.14 plotting/prep/crop_vendored.py && python3 plotting/make_figs.py && python3 plotting/export_to_draft.py v3`
3. → v3 (do not edit the draft from here): the caption must stop saying the cell cannot exist. The bottom
   row now reads *one diffusion model sampled at two budgets*, the same reading as the flow rows. If the
   auto-picked checkpoint was `aw1`, the caption must say so — it is not the baseline's `aw10`.
   `fig:k-ladder` is **unaffected**: its diffusion series is three separately trained checkpoints, already
   corrected at v3.41 (§6 above).

## 7 · Completion record — ❌ 25964 FAILED, and Option B is dead

**Job 25964, 2026-09-19 16:18:33 → 16:18:38 UTC (4 seconds), exit on an unpickle error at K=1.**
The checkpoint was found (`PANEL_AW=auto -> aw1`) and the loader reached it, then:

```
AttributeError: Can't get attribute 'GaussianDiffusion' on
  <module 'flow_matcher_v3_ode_selectable.models.diffusion'>
```

That is not a wiring fault in the driver. It is the checkpoint telling us what it is.

### 🔴 The route table in §3 is wrong, and this is the correction

`flow_matching_v3_ode_selectable`'s `GaussianDiffusion` **is not the diffusion engine.** It is the
flow-matching ODE class under its old name. Commit `cac7cc6a` (*"KEY API refactor: rename diffusion
classes to FlowMatching", 25 May 2026*) renamed it, and the diff is one line:

```diff
-class GaussianDiffusion(nn.Module):
+class FlowMatchingODE(nn.Module):
```

The pre-rename class already had `flow_steps_v3`, the ODE solver backend, and `dt = 1/flow_steps` —
it was always the flow model. So the folders
`plans/flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.GaussianDiffusion_a1.5_b1.0_aw*/` are
**pre-26-May flow-matching runs carrying a stale class name in their path**, and the K=1,5,10,20 cells
under them are a flow model sampled at four budgets — which is exactly what one expects, and no
evidence at all that the diffusion engine can be sampled off one checkpoint.

**Consequently there is no second route.** The two-route table of §3 collapses to one: the diffusion
engine runs through `plans/diffusion/` only, where K is in `diffusion_loadpath`
(`config/avoiding-d3il.py:1074`) and is therefore fixed at training. **The original claim — the one the
v3.41 caption already makes — was right.** §3's "why *impossible* was too strong" is withdrawn.

### What this leaves

| | |
| :-- | :-- |
| Option B | ❌ **not possible.** Its premise was a folder name from before a rename |
| Option A | the only route: train a $K{=}2$ DDPM checkpoint, then one plan-dumping eval |
| the existing $\nfe=1$ panel | ✅ **stays.** Nothing replaces it; the row keeps its current meaning |
| the v3.41 caption | ✅ **stays as written** — no draft change is needed for this |
| `fig:k-ladder` | unaffected; its §6 correction stands on its own |

Option A is spelled out in the driver header: a tracked config edit (`n_diffusion_steps` 20 → 2 in the
`diffusion` training block), a training run, the edit reverted, then
`FMPCC_RUN_MSG=planpanel63 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_dpcc_job.sh --seed 6`.
**Not submitted** — it needs an author decision: a GPU training and a tracked edit, for one panel of one
figure whose caption already explains the gap honestly.

### §84's open question, answered

*"For the DA to confirm: no other figure or table reads a diffusion budget as an inference-time choice."*
**Confirmed clean.** Every diffusion cell in `analysis/` keys on `Dmodels.GaussianDiffusion` — the
`plans/diffusion/` route, one folder per trained K. The misnamed flow folders read
`Dmodels.diffusion.GaussianDiffusion` (note the extra `.diffusion.`) under the
`flow_matching_v3_ode_selectable` prefix, and no DA cell selects them. `tab:avoiding-dpcc-protocol`'s
diffusion rows at $\nfe=1,5,10$ are three separately trained checkpoints, like `fig:k-ladder`'s.

⚠️ **The stale name is a live trap for the next reader**, in the batch CSVs and in the ledger alike:
`Dmodels.diffusion.GaussianDiffusion` under a flow prefix is a **flow** row. The `PLANNED` entry for
`fig_raw_plans_diffusion_K2` in `plotting/sources.py` currently repeats the two-route story and should be
cut back to the one-route fact.

### Cleanup

`Slurm_Codes/temp_bash/_fig63_panel_eval.py`, `_fig63_panel_job.sh` and the driver can be deleted —
they have nothing left to run.

## 8 · Original completion-record slot (superseded by §7)

_(fill in when the job ends: per-K exit status, whether both dashboards were written, and whether the
K=1 panel visibly differs from the retired trained-at-1 one — if it does, that difference is itself the
point §6.1.1.4 argues and is worth a sentence.)_

---

# RUNBOOK · Option A — train the $K{=}2$ checkpoint, then plot it

Written 2026-09-19 after Option B died. Driver:
`Slurm_Codes/temp_bash/pipeline_20260919_fig63_K2_optionA.sh` (gitignored throwaway; it generates
`_fig63a_train.py`, `_fig63a_eval.py` and the two sbatch files beside itself).

```bash
bash Slurm_Codes/temp_bash/pipeline_20260919_fig63_K2_optionA.sh            # PLAN
bash Slurm_Codes/temp_bash/pipeline_20260919_fig63_K2_optionA.sh submit     # SUBMIT both, chained
```

Knobs: `SEED=6` · `K=2` · `TRAIN_HOURS=24` · `RUN_MSG=planpanel63` · `SKIP_TRAIN=1` (eval only, if the
checkpoint is already there).

## Two jobs, chained on `afterok`

| # | job | what | writes |
| --: | :-- | :-- | :-- |
| 1 | `fig63a_trainK2` | `scripts/train.py --seed 6` with `n_diffusion_steps = 2` | `logs/avoiding-d3il/diffusion/H8_K2_Dmodels.GaussianDiffusion_aw10/6/` |
| 2 | `fig63a_evalK2` | `scripts/eval.py --seed 6` with `n_diffusion_steps = 2` | `…/plans/diffusion/H8_K2_…_msgplanpanel63/6/results/halfspace_both-hard/diffuser.png` |

They meet because the plan block's loadpath is
`f:diffusion/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}` (`avoiding-d3il.py:1074`)
and the train block's `exp_name` watches the same four keys (`args_to_watch_dpcc_train`, `:146`).
Setting $K$ on both sides is the whole of the change.

## No tracked file is edited

`n_diffusion_steps` has no env override in either block. Editing `config/avoiding-d3il.py` would mean a
20 → 2 → 20 round trip held open **for the hours the training runs**, in a working tree other queued jobs
read at start — the `n_trials` trap of [`SLURM_RUNBOOK_20260918`](SLURM_RUNBOOK_20260918_pending_runs.md),
but with a much longer window. Each job instead patches the config module dict in memory and runs the
stock script through `runpy`. ⚠️ `_fig63a_*.py` are read at job start — keep them until both jobs end.

## Cost, and the one thing to check first

Training is `n_train_steps` 1e5, batch 8, grad-accum 2, one A5000 — a few hours; the 24 h limit is the
usual 2× margin. **$K{=}2$ is not cheaper than $K{=}20$**: the budget sets the noise schedule, not the
network or the number of gradient steps. `df -h ~/FMPCC` before submitting — one more checkpoint folder,
and disk has been tight all week.

## What the panel will mean, and what stays true

A diffusion model **trained at $K{=}2$ and run at $K{=}2$** — the same reading as the existing $\nfe=1$
panel and as every other cell of the figure. So:

* the existing $\nfe=1$ panel **stays**; the row is not replaced as a pair (that was Option B's problem);
* the v3.41 caption's *claim* survives — the budget was still fixed at training time, which is why this
  cell needed a training run at all. Only the sentence saying the panel is therefore absent comes out;
* `fig:k-ladder` is untouched.

## Submission record

| job | id | submitted | ended |
| :-- | :-- | :-- | :-- |
| `fig63a_trainK2` | **25965** | 2026-09-19 16:24 | ✅ **19:05 UTC** (2 h 41 m) |
| `fig63a_evalK2` (afterok:25965) | **25966** | 2026-09-19 16:24 | ✅ **19:22 UTC** (17 m) |

Submitted from the cluster copy `Slurm_Codes/temp_bash/19-09-K2.sh`. Logs:
`Slurm_Codes/logs/2026-09-19/*_fig63a_train_job_25965.log` and `*_fig63a_eval_job_25966.log`.
25966 sits in `DependencyNotSatisfied` until 25965 exits 0 — if the training fails, the eval is
**cancelled by SLURM, not run against a missing checkpoint**. First line worth reading in 25965:
`[ fig63a ] TRAIN  n_diffusion_steps -> 2`, then the `df -h logs` line beneath it.

## After it lands

```bash
find logs/avoiding-d3il/plans/diffusion -path '*K2*_msgplanpanel63*' -path '*both-hard*' -name diffuser.png
```

→ `Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/fig8h_plans_diffusion_K2_seed6.png`, then
move `fig_raw_plans_diffusion_K2` from `PLANNED` to `VENDORED` in `plotting/sources.py` with crop box
`(2312, 92, 2715, 492)`, rebuild, export, and tell v3 the caption sentence can go.

---

# ✅ DONE — 2026-09-19, the panel exists

Both jobs clean. `state_best.pt` written to
`logs/avoiding-d3il/diffusion/H8_K2_Dmodels.GaussianDiffusion_aw10/6/` (train loss 0.00649, test
0.00952), and the eval wrote the panel at

```
logs/avoiding-d3il/plans/diffusion/H8_K2_Dmodels.GaussianDiffusion_aw10/
  H8_K2_T0.5_Dmodels.GaussianDiffusion_msgplanpanel63/6/results/halfspace_both-hard/diffuser.png
```

**Fetch:** `Slurm_Codes/temp_bash/fetch_20260919_fig63_diffusion_K2.sh` (plan / stage / stage tar).
It checks the panel is 3000×1000 before staging — at any other size the crop box is wrong.

⚠️ Disk was **43 G free (100 % used)** on `/data` when the training started. It fit, but the next
training job may not.

## The by-product, and it is worth more than the panel — ledger A4 / R18

25966 ran the **full 13-variant × 3-geometry sweep at seed 6, 2 episodes**, so *"diffusion at $K=2$ is
absent everywhere, at either protocol"* is no longer true. Aggregated across the three geometries, seed 6
(S&C = goal and constraints):

| variant | S&C | steps | ms/step | per geometry (TR / TL / both) |
| :-- | --: | --: | --: | :-- |
| `dpcc-r` | 0.33 | 77.8 | 224 | 1.0 / 0.0 / 0.0 |
| **`dpcc-r-tightened`** | **1.00** | 64.3 | 225 | 1.0 / 1.0 / 1.0 |
| `dpcc-c` | 0.67 | 68.8 | 222 | 1.0 / 0.5 / 0.5 |
| **`dpcc-c-tightened`** | **1.00** | 61.3 | 226 | 1.0 / 1.0 / 1.0 |
| `dpcc-t` | 0.33 | 77.0 | 198 | 1.0 / 0.0 / 0.0 |
| `dpcc-t-tightened` | 0.83 | 62.7 | 276 | 1.0 / 0.5 / 1.0 |
| `diffuser` (unprojected) | 0.00 | 59.0 | 18 | 0.0 / 0.0 / 0.0 |
| `post_processing` | 0.50 | 79.8 | 36 | 1.0 / 0.5 / 0.0 |
| `post_processing-tightened` | 1.00 | 63.7 | 26 | 1.0 / 1.0 / 1.0 |
| `gradient` (± tightened) | 0.00 | 85.7 | 21 | 0.0 / 0.0 / 0.0 |
| `model_free` (± tightened) | 0.00 | 59.5 | 36 | 0.0 / 0.0 / 0.0 |

Two episodes per geometry, so each cell is a count out of 2 and 0.5 means one episode. Aggregated by
geometry first, as elsewhere.

🔴 **Seed 6 only** — every other diffusion row in `tab:avoiding-dpcc-protocol` is five seeds, so this
cannot be dropped into the table as a peer. Either run seeds 7–10 (four more trainings, and disk says no
for now) or report it separately and say what it is.

Two things it shows, both consistent with the rest of the analysis. The **tightening margin again carries
the projector**: untightened `dpcc-r`/`-t` fall to 0.33 and `dpcc-c` to 0.67, while their tightened twins
hold 1.00 — the same effect the corridor guard cell showed on 09-19, now at a second scene and a second
engine. And the baseline at $K=2$ still costs **~225 ms/step** against flow matching's 17.3 ms at
$\nfe=1$: dropping the budget from 20 to 2 does not buy back the gap, because the projector, not the
denoiser, is where the baseline's time goes. `dpcc-t-tightened` is the one tightened cell below 1.00
(0.83 — a single episode of the top-left geometry).

## What is still true, and what changes in the draft

* the existing $\nfe=1$ panel **stays** — both cells are now trained-at-their-own-budget;
* the v3.41 caption's **claim** stays: the budget was fixed at training, which is precisely why this
  panel needed 2 h 41 m of GPU. Only the sentence saying the panel is therefore absent comes out;
* `fig:k-ladder` unaffected; §7's Option-B correction stands as written.
