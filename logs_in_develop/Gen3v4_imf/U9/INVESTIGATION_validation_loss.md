# U9 Investigation: Where is the Gen3v4 iMF validation loss (in W&B)?

**Date:** July 7, 2026
**Scope:** `flow_matcher_v3_imeanflow/` + `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` + `Slurm_Codes/sbatch/iMF/train_imf.sh`, cross-checked against the three candidate upstream styles: DPCC (`/workspaces/aux_repo/dpcc`), D3IL (`/workspaces/aux_repo/d3il`), and the reference iMF repo (`/workspaces/aux_repo/imeanflow`).
**Status:** code-read investigation only — nothing verified at runtime (no Python in this container). Cluster checklist at the bottom.

---

## TL;DR

1. **We DO have a validation loss.** The DPCC-inherited Trainer does a 90/10 held-out split and computes a test loss every 1000 steps. It is saved to `losses.pkl` and uploaded to W&B under the name **`test/loss`** (not `val/loss`).
2. **But it only reaches W&B after a seed finishes training** — logging is post-hoc reconstruction from `losses.pkl`, not live. A job killed by the 24 h SLURM limit leaves an **empty W&B run** (no train loss either).
3. **The W&B run config also hides it:** the iMF config block never sets `train_test_split`, so `wandb.init(config=vars(args))` shows no split key at all — the run *looks* like it has no validation, even when it does. It only works via a silent `getattr(..., 0.9)` fallback in the train script.
4. Under the active `meanflow_jvp` objective, the current test loss is a **self-referential, adaptively re-weighted consistency metric**, not a clean data-fit measure — it is the weakest of the three candidate styles for judging generalization.
5. **Recommendation: keep the DPCC-style split (already wired), but log a model-independent companion metric on the held-out set, and add live W&B logging.** D3IL style adds one good idea (validate with EMA weights); imeanflow style adds another (periodic held-out *sampling* check, the trajectory analog of their FID). Details in §5.

---

## 1. What exists today (mechanism trace, file:line)

The whole chain is DPCC's, inherited verbatim (diff against
`/workspaces/aux_repo/dpcc/diffuser/utils/training.py:38,68-85,144-155` — identical logic):

| Step | Where | What happens |
|---|---|---|
| Split | `flow_matcher_v3_imeanflow/utils/training.py:68-83` | if `train_test_split < 1`: `torch.utils.data.random_split` into train/test dataloaders; `best_test_loss` tracking initialized |
| Compute | `utils/training.py:146-155` | every `log_freq` (=1000, `:48` default, not overridden) steps: `self.test()` |
| `test()` | `utils/training.py:203-222` | `model.eval()`, `torch.no_grad()`, averages `self.model.loss(*batch)` over **100 batches** from the held-out loader; returns `(test_loss, test_a0_loss)` |
| Best ckpt | `utils/training.py:152-154` | `test_loss < best_test_loss` → `save_best()` → `state_best.pt` |
| Persist | `utils/training.py:259-265` (`save_losses`) | keys `training_losses` / `test_losses` / `training_a0_losses` / `test_a0_losses` → `losses.pkl` (with resume-merge logic) |
| W&B | `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py:44-60` (`log_wandb_from_losses`) | **after** `trainer.train()` returns (`:248`): replays `losses.pkl` → `train/loss` + `test/loss` per step |
| Enable | `Slurm_Codes/sbatch/iMF/train_imf.sh:81-84` | `--use-wandb --wandb-project FMPCC-iMF` — W&B **is** enabled on the cluster path |

Key config fact: `config/avoiding-d3il.py` block `'flow_matching_v3_imeanflow'` (lines 449–551) does
**not** contain `train_test_split` — the neighboring `flow_matching_v3_ode_selectable` block does
(line 444: `0.9`). The train script covers this with a fallback:

```python
# train_flow_matching_v3_imeanflow.py (trainer_config)
train_test_split=getattr(args, 'train_test_split', 0.9),
```

So validation **runs** (split = 0.9), but the value exists nowhere in `args` → nowhere in the W&B
run config → invisible to anyone auditing the run.

## 2. So why does W&B show no validation loss? (ranked causes)

1. **Post-hoc-only logging.** `log_wandb_from_losses` is called once, after `trainer.train()`
   completes (`train_flow_matching_v3_imeanflow.py:245-250`). While a seed is training, its W&B run
   has config + nothing else. If the job dies (crash, or the sbatch's `--time=24:00:00` limit,
   `train_imf.sh:9`) the run stays **permanently empty** — no `train/loss` *and* no `test/loss`.
   With 5 seeds trained serially in one job (`--seeds 6 7 8 9 10`, `train_imf.sh:82`), a timeout
   mid-seed-3 means seeds 3–5 all show empty runs.
2. **Naming.** When it does arrive, it's `test/loss` / `a0` companions — anyone scanning for
   "val" finds nothing. The v3ode sibling script additionally sets
   `run.summary['final_test_loss']` (`train_flow_matching_v3_ode_selectable.py:59-60`); the iMF
   script does **not**, so the run summary/table view shows no test metric either.
3. **Invisible config** (§1): no `train_test_split` in the W&B config for iMF runs → a reasonable
   reader concludes validation was never configured.
4. **No live console signal either.** The tqdm postfix does carry `loss_test`
   (`utils/training.py:177-181`), but tqdm is muted for SLURM (`mininterval=1e10`,
   `utils/training.py:119`), so in `.log` files only the one-time
   `Initial test loss: ...` line at step 0 (`utils/training.py:157`) appears.

**Conclusion: the answer to "do we have it or not" is "yes, in `losses.pkl` and `state_best.pt`
selection, always; in W&B, only for seeds whose training ran to completion — and even then named
`test/loss` with no config evidence that a split exists."**

To confirm which of these applies to the runs you looked at: check whether the run in question has
`train/loss` curves. If yes → look for `test/loss` (it should be right there, same steps). If the
run is empty → cause 1 (seed never finished; check the job's `.log` and `losses.pkl` on the
cluster, which will still contain `test_losses`).

## 3. What *is* a validation loss in iMF, mathematically?

This depends on the objective (`imf_objective`, active = `'meanflow_jvp'`,
`config/avoiding-d3il.py:480`):

- **`fm_equivalent` path** (`imf_diffusion.py:420-468`): target is `(x_t − x_r)/h`, which for the
  linear interpolant is the constant `x_start − noise` — a **model-independent** regression
  target. Held-out loss = honest generalization measure of velocity-field fit, up to Monte-Carlo
  variance from the random draws of `(t, r, noise)` per test batch (100 batches × bs 8 averages
  most of it out).
- **`meanflow_jvp` path** (`imf_diffusion.py:497-609`): target is
  `u_tgt = v_inst + h·(du/dr)` with the JVP taken **through the current model** and stop-gradiented
  (`:578-580`). Two consequences for a "validation loss":
  1. **Self-referential**: the target moves with the model. A falling held-out value can mean
     "better data fit" *or* "the model became more self-consistent" — these are not the same
     thing, and the metric cannot distinguish them.
  2. **Adaptively re-weighted**: `w = 1/(per_sample + c)^p` with `p=0.5` (`:587-588`) compresses
     the loss scale (≈ square-root of the raw MSE) **per sample, per batch**. Values are not
     comparable across runs, objectives, or even training stages in the way a plain MSE is.
  So under the active objective, today's `test/loss` is a *consistency monitor*, useful for
  detecting divergence/overfit trends within one run, weak as a model-selection or cross-run
  comparison signal. Note `state_best.pt` is currently selected by exactly this metric
  (`utils/training.py:152-154`).
- One more wrinkle: `test()` runs under `torch.no_grad()` while the JVP path calls
  `torch.func.jvp` (`imf_diffusion.py:525-527,571`). Per PyTorch's documented semantics,
  func-transforms still compute when the `no_grad` is *outside* the transform, so this should
  work — but it has never been exercised knowingly and belongs on the cluster checklist (§6).
  If it *did* throw, training would crash at step 0, which would itself explain empty W&B runs —
  the `.log` files settle this instantly.

## 4. The three candidate styles, compared honestly

| | DPCC style (current) | D3IL style | iMF-repo style |
|---|---|---|---|
| What | Held-out same-objective loss, 100 random batches every 1000 steps (`utils/training.py:203`) | Full pass over held-out set per epoch, **with EMA weights**; best ckpt by test MSE (`d3il/agents/ddpm_agent.py:92-114`, `evaluate()` at `:177` swaps in EMA params) | **No val loss at all.** Periodic *sampling* + FID every `fid_per_epoch` (`imeanflow/train.py:229-232`), vis samples every `sample_per_epoch` (`:192-198`) |
| Measures | Objective generalization (but see §3 caveats under `meanflow_jvp`) | Generalization of the *deployed* weights (EMA = what eval uses) | Actual generation quality — the thing you ultimately care about |
| Cost | Cheap-ish (JVP × 100 batches / 1000 steps) | Moderate (full val pass/epoch) | Expensive (sampling loop) |
| Fits FM-PCC infra | Already wired end-to-end | Small trainer change | Needs a sampling-eval hook in the trainer |
| Cross-gen comparability | High — every other gen (Gen0–Gen3v3) logs the same thing | New metric, no history | New metric, no history |

Two style-specific notes:
- **D3IL's EMA point half-applies here**: Gen3v4 eval currently runs with raw weights
  (`'eval_use_ema': False` — "dpcc-legacy", `config/avoiding-d3il.py:874`), and `test()` also uses
  raw `self.model` — so today they're *consistent*. If eval ever flips to EMA, validation should
  flip with it, or the best-checkpoint selection validates weights nobody deploys.
- **The imeanflow analog of FID** for trajectories is not FID — it's held-out *sampling* metrics:
  run the actual sampler (1-NFE and/or the K-NFE used at eval) from held-out conditions and score
  a0-action MSE / trajectory MSE / constraint-violation rate against the held-out ground truth.
  That is the only metric family that directly tests what iMF is *for* (few-step generation), and
  it is also the check that would have caught things like the U8/audit-flagged concerns
  empirically.

## 5. Recommendation

**Hybrid, in order of effort — keep DPCC as the spine, borrow one idea from each of the others:**

1. **(trivial, do first) Make the existing validation visible.**
   - Add `'train_test_split': 0.9` explicitly to the `flow_matching_v3_imeanflow` block in
     `config/avoiding-d3il.py` (kills the silent fallback *and* puts it in the W&B run config).
   - Add `run.summary['final_test_loss']` to the iMF script (parity with the v3ode sibling).
2. **(small) Log live to W&B, not only post-hoc.** Either `run.log` inside the training loop, or
   call the `losses.pkl` replay every epoch. This is what actually fixes "nothing in W&B" for
   timed-out jobs — same 24 h-blindness problem the Gen11 Fix11 breadcrumbs just solved for eval.
3. **(small, high value) Add a model-independent companion val metric** on the same held-out
   batches, logged alongside the raw objective loss:
   - `val/raw_mse` — `per_sample.mean()` *before* adaptive weighting (one line; the quantity
     already exists at `imf_diffusion.py:587`), and/or
   - `val/fm_equiv` — the finite-difference-target loss evaluated in the same `test()` pass.
   These are comparable across runs and objectives; keep `test/loss` too for continuity.
   Consider selecting `state_best.pt` by the raw/fm-equiv metric instead of the adaptive one.
4. **(fix while touching this) Seed the split.** `random_split` at `utils/training.py:76` uses no
   generator → a resume re-splits differently → the old test set leaks into training and
   `best_test_loss` compares across different test sets. `generator=torch.Generator().manual_seed(seed)`.
   (Same latent issue exists in DPCC upstream and every gen that inherited this Trainer — worth a
   sweep later.)
5. **(optional, U9+ follow-up) iMF-style sampling validation**: every N epochs, sample from M
   held-out conditions at 1-NFE and eval-NFE, log a0/trajectory MSE. This is the trajectory-domain
   FID and the strongest early-warning signal for the JVP/consistency concerns from the
   Condition_Analysis audit.

Not recommended: switching wholesale to D3IL style (loses cross-gen comparability for no gain) or
to pure imeanflow style (no val loss at all — strictly less information).

## 6. Cluster verification checklist (nothing above is runtime-verified)

- [ ] Open an existing Gen3v4 run dir on the cluster: `python -c "import pickle; d=pickle.load(open('.../losses.pkl','rb')); print(d.keys(), len(d['test_losses']))"` — confirms whether `test_losses` is populated for past runs.
- [ ] Check whether the W&B runs that "have no val loss" also lack `train/loss` (→ cause 1, seed never completed) or have it (→ then `test/loss` should exist; if it truly doesn't, report back — that would contradict this code-read).
- [ ] Smoke-test `torch.func.jvp` inside `torch.no_grad()` with the `meanflow_jvp` objective (one forward of `test()`); also just check step-0 of any past `.log` for the `Initial test loss:` line — its presence proves the JVP-under-no_grad path works.
- [ ] After applying §5.1–5.3: one short training run, confirm `test/loss`, `val/raw_mse`, and `train_test_split` all appear in W&B live.

## 7. Addendum (July 8): what `test/loss` actually is & how to compare against DPCC W&B numbers

Answers to the follow-up "what is really the old test loss / is it the same as dpcc/imeanflow / how do I compare?", after the U9 fixes landed (see `CHANGELOG_U9_validation_loss_fix.md`).

### 7.1 The old `test/loss` — identity (U9 did NOT change it)

`test/loss` = **the current generation's own training objective, evaluated on the 10% held-out split** (100 batches every 1000 steps). The *mechanism* is DPCC's verbatim (§1); the *quantity* is whatever `model.loss()` computes in that generation:

| Runs (W&B) | `train/loss` & `test/loss` are… | Prediction target | Scale character |
|---|---|---|---|
| DPCC baseline / Gen0 (`scripts/train.py`) | weighted L2 on **ε-prediction** (`predict_epsilon=True`, `action_weight=10`) | true noise ε ~ N(0,1) per element; model-independent | plain MSE, starts ≈1 |
| Gen1–3v3 FM | weighted MSE on **velocity** | `x_start − noise`; model-independent | plain MSE, data-scale |
| Gen3v4 iMF, `meanflow_jvp` (active) | **adaptively reweighted** MeanFlow consistency loss: `mean(w·per_sample)`, `w = 1/(per_sample+c)^0.5` (`imf_diffusion.py:587-591`) | `v + h·du/dr` via JVP **through the current model** (self-referential, §3) | compressed ≈ √MSE |
| Gen3v4 iMF, `fm_equivalent` | weighted MSE on finite-difference target (no adaptive reweighting) | `(x_t−x_r)/h` = `x_start−noise`; model-independent | plain MSE |

So: **same name and pipeline as DPCC, different mathematical quantity.** Note DPCC upstream itself never logs to W&B at all (its `logs` dict only feeds the tqdm postfix, `dpcc/diffuser/utils/training.py:169-181`) — the `train/loss`/`test/loss` W&B keys are FM-PCC's own replay convention (`scripts/train.py:135-165` and siblings), which is *why* the names look identical across generations while the quantities are not.

Vs. the **imeanflow reference repo**: same adaptive trick (`loss / sg((loss+eps)^p)`, `imeanflow/imf.py:380-393`) — so the *form* of our `meanflow_jvp` loss is faithful — but their per-sample reduction is a **sum over pixels** (ours: `mean` over horizon×dims, with `loss_weights`), it's images not trajectories, and they log **no validation loss at all** (§4: train loss + FID only). Their absolute numbers are meaningless for us too.

### 7.2 Consequence: the three W&B curves and what each is for

- `train/loss` / `test/loss` — the adaptive-weighted objective on train / held-out batches. Same quantity as each other → the **train–test gap and its trend** are meaningful (overfit/divergence monitor within a run). Still selects `state_best.pt`.
- `val/raw_mse` (new in U9) — same held-out batches, `per_sample.mean()` **before** the `1/(mse+c)^p` reweighting. Removes the scale compression → comparable across iMF runs, seeds, and hyperparameter variants. **Honest caveat:** under `meanflow_jvp` it does *not* remove self-referentiality — the JVP target still moves with the model (except the r==t anchor samples, whose target is the model-independent `v_inst`). It is a scale-stable consistency metric, not a pure data-fit metric; the fully model-independent option (`val/fm_equiv` on the same batches, §5.3) was not implemented.

### 7.3 How to compare against DPCC W&B numbers — the rules

1. **Never compare absolute loss values across engines.** DPCC ε-MSE vs FM velocity-MSE vs iMF adaptive-consistency are different targets, different scales, one of them √-compressed. Overlaying `test/loss` panels from the DPCC project and `FMPCC-iMF` and reading off "lower = better" is invalid. (Same for `a0_loss`: ε-space vs u-space.)
2. **Within the iMF family** (seeds, U-variants, hyperparams): compare `val/raw_mse` first (scale-stable), `test/loss` second (continuity + best-ckpt semantics). `run.summary['final_val_raw_mse']` / `['final_test_loss']` support table views.
3. **Across engines, from training curves you may only compare shapes**: convergence speed (steps to plateau, relative decrease from step 0), train–test gap trend, divergence events. If overlay is needed, normalize (e.g. loss ÷ its step-0/step-1k value) or use log-scale and read slopes.
4. **The only apples-to-apples DPCC-vs-iMF numbers are downstream eval metrics** — success rate / trajectory & a0 MSE / constraint-violation from the actual eval rollout pipeline. That, not training loss, is the cross-engine scoreboard; the deferred sampling-validation (§5.5, 1-NFE held-out trajectory MSE) would bring a proxy of it into training-time W&B.

### 7.4 Metric inventory: do we log more than dpcc / imeanflow now? (post-U9 state)

Short answer: **more than both upstreams on the loss side; still less than imeanflow on the generation-quality side.**

| Training-time metric | DPCC upstream | Gen0 (FM-PCC dpcc baseline) | imeanflow repo | Gen3v4 iMF post-U9 |
|---|---|---|---|---|
| train objective loss | pkl + tqdm only (**no W&B at all**) | pkl + W&B `train/loss` (post-hoc replay) | logged (adaptive `loss`) | pkl + W&B `train/loss`, **live per epoch** |
| held-out objective loss | pkl + tqdm | pkl + W&B `test/loss` (post-hoc) | **none** | pkl + W&B `test/loss`, live |
| raw (un-reweighted) companion | n/a (loss already raw MSE) | n/a | `loss_u`/`loss_v` raw components (train only) | **`val/raw_mse` on held-out** — new in U9, exists in neither upstream |
| a0 / first-action loss | pkl + tqdm | pkl + W&B `train/a0_loss`, `test/a0_loss` | n/a | pkl + tqdm only — **not replayed to W&B** (Gen0's replay does; ours doesn't) |
| aux v-head loss | n/a | n/a | `loss_v` logged | in `info['aux_loss']` only — not persisted, not in W&B |
| run summaries | none | `final_train_loss`, `final_test_loss` | none | `final_test_loss`, `final_val_raw_mse` (no `final_train_loss`) |
| sampling-quality check | none | none | **FID + sample grids** every N epochs | **none** — the §5.5 gap, still open |

So vs **DPCC**: strictly more — same split/test machinery plus `val/raw_mse`, live (per-epoch) W&B instead of end-of-seed replay, and summary keys. Vs **imeanflow**: more on validation (they have zero held-out loss) but less on generation quality (no sampling/FID analog yet — §5.5).

Two small W&B parity gaps left on our side, both replay-only (data already in `losses.pkl`/`info`): the a0 curves (Gen0 logs them, iMF replay doesn't) and the aux v-head loss (imeanflow logs its `loss_v`, we drop `info['aux_loss']`). Trivial adds if wanted.

> **Update (July 8, same day):** all loss-side gaps in this table are now closed — a0 curves, train-side raw MSE (`loss_u` analog), and aux loss (`loss_v` analog) on both splits are tracked, persisted, and uploaded to W&B. See `CHANGELOG_U9_metric_parity_wandb.md`. The only remaining row is the sampling-quality check (§5.5), still deferred.

## Files read for this investigation

- `flow_matcher_v3_imeanflow/utils/training.py` (Trainer: split/test/save/log paths)
- `flow_matcher_v3_imeanflow/models/imf_diffusion.py` (both loss paths, adaptive weighting, JVP call)
- `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` (Parser, trainer_config, W&B replay)
- `FM_v3_imeanflow_test/train_flow_matching_v3_ode_selectable.py` (sibling comparison)
- `config/avoiding-d3il.py` (iMF block 449–551; neighbor blocks for `train_test_split`)
- `Slurm_Codes/sbatch/iMF/train_imf.sh` (actual cluster entry point, W&B flags, time limit)
- `/workspaces/aux_repo/dpcc/diffuser/utils/training.py` (origin of the mechanism)
- `/workspaces/aux_repo/d3il/agents/base_agent.py`, `agents/ddpm_agent.py` (D3IL style)
- `/workspaces/aux_repo/imeanflow/train.py`, `main.py` (iMF-repo style: sampling + FID, no val loss)
