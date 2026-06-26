# MEMO — Gen9 E2: How to Revert to DPCC-Identical Behaviour (if insisted)

**Date:** 2026-06-13
**Author:** Claude Opus 4.8
**Status:** Plan only — no code changed by this memo. Execute on the cluster side.
**Companion to:** `U4/DPCC_DIVERGENCE_AND_COMPARABILITY.md`, `U4/CHANGELOG.md`

---

## Recommendation up front (corrected 2026-06-13)

**Do not revert for "paper purity" — that was the wrong frame.** The B6/B7/B8/B9 changes are
**correctness improvements** that make train+eval *better*, not worse: B7 removes train/test
leakage, B8 stops discarding the final 20% of training, B9 stops training in eval mode, and B6
evaluates the standard EMA weights. Keep them.

**But fixing B6 introduced a real, self-inflicted train/eval inconsistency that DOES need
repair** — see the next section. This has nothing to do with the DPCC paper; it is about our
own pipeline being internally consistent so that "train then eval" actually selects and deploys
the same network.

The hard-revert mechanics that follow are retained only as an **appendix** for the narrow case
where a reviewer explicitly demands the visual baseline mirror published DPCC byte-for-byte.
That is not the recommended action.

---

## ★ KEY DIVERGENCE THAT NEEDS REPAIR — EMA-eval vs raw-selection mismatch

**This is the one thing to actually fix.** It is a correctness bug in *our* train+eval loop,
introduced when U4 fixed B6 (EMA eval) without updating checkpoint selection to match.

### The broken chain (verified in code)

1. `test()` evaluates **`self.model`** — the **raw** online weights
   (`training.py:208`, `loss, infos = self.model.loss(*batch)`).
2. `state_best.pt` is saved whenever that **raw** test loss improves
   (`training.py:153-155`, `if test_loss < self.best_test_loss: self.save_best()`).
3. The visual config evaluates **`diffusion_epoch: 'best'`**
   (`config/avoiding-d3il-visual.py:202, 248`).
4. But the eval loads **`trainer.ema_model`** — the **EMA** weights
   (`eval_visual_avoiding_dpcc.py:179-180`, B6 fix).

**Result:** the checkpoint chosen as "best" is the step where the *raw* model tracked the
validation set best, but at deploy time we run the *EMA* model from that step. The step that is
optimal for raw weights is **not** necessarily optimal for EMA weights. We are selecting on one
network and evaluating another. This silently degrades the eval — possibly a lot, depending on
how far raw and EMA diverge over training — and it makes the success-rate numbers a function of
an accidental selection mismatch rather than of model quality.

This is the divergence that must be repaired for train+eval to be **correct and consistent**.
(Note: the upstream DPCC code is self-consistent here only by coincidence — it selects on raw
*and* evals raw. We fixed the eval half (B6) but left the selection half on raw. Half a fix is
worse than either whole.)

### Repair — Option 1 (recommended, correctness-forward, needs one retrain)

Make checkpoint selection evaluate the **same weights eval uses (EMA)**. In
`*/utils/training.py`, change `test()` to score the EMA model:

```python
def test(self, n_test=100):
    eval_model = self.ema_model          # match eval-time weights: select 'best' on EMA
    eval_model.eval()
    test_loss = 0
    test_a0_loss = 0
    with torch.no_grad():
        for step in range(n_test):
            batch = next(self.test_dataloader)
            batch = batch_to_device(batch, device=self.device)
            loss, infos = eval_model.loss(*batch)
            loss /= self.gradient_accumulate_every
            test_loss += loss.item()
            test_a0_loss += infos['a0_loss'].item() if 'a0_loss' in infos else 0
        test_loss /= n_test
        test_a0_loss /= n_test
    return test_loss, test_a0_loss      # self.model never touched → B9 leak gone for free
```

Why this is the right fix:
- `state_best` is now chosen on EMA validation loss = exactly the network eval deploys. Selection
  and evaluation are the same model. **Self-consistent.**
- `self.model` is never switched to `.eval()`, so the **B9 eval-mode leak disappears entirely**
  — the `self.model.train()` band-aid (training.py:223) becomes unnecessary.
- Apply to **both** `diffuser_visual_avoiding/utils/training.py` and
  `fm_visual_avoiding/utils/training.py` so FM and DPCC stay common-mode.
- Takes effect on the **next retrain** (it changes which step becomes `state_best`).

### Repair — Option 2 (immediate stopgap, no retrain)

If there is no retrain budget right now, restore consistency on the **existing** checkpoints by
evaluating the weights that `state_best` was actually selected on — i.e. **raw**. That means
reverting *only* B6 at eval (return `trainer.model`, see appendix B6). Existing `state_best`
checkpoints were selected on raw loss, so raw eval is self-consistent with them today.

- Pro: zero retrain; numbers become trustworthy immediately.
- Con: gives up EMA's usual quality gain. This is a stopgap, not the destination.

**Do not stay in the current half-fixed state** (raw selection + EMA eval) — that is the only
genuinely wrong configuration of the three.

### Decision

| Situation | Do this |
|---|---|
| Retrain budget available | **Option 1** — EMA selection + EMA eval. Correct and best-quality. |
| Need valid numbers from existing checkpoints now | **Option 2** — raw eval (revert B6 only) for consistency, schedule Option 1 retrain. |
| Status quo (raw select + EMA eval) | **Not acceptable** — repair via Option 1 or 2. |

---

## Appendix — DPCC byte-for-byte revert (only if a reviewer demands it)

Everything below is the *paper-mirror* path. It is **not** the recommended action; the
recommended action is the EMA/selection repair above. Only **four** fixes diverge from DPCC:
B6, B7, B8, B9. The other U4 fixes (B1, B2, B3, B5, B10) are visual-only and must **stay** —
reverting them would re-break the visual pipeline without buying any DPCC comparability.

> **If you do revert: flag-gate, do not delete.** Add one `dpcc_faithful` switch that defaults
> to DPCC behaviour. You keep both behaviours, lose no work, and the revert is auditable.

---

## STEP 0 — The one decision that determines whether you need a retrain

**Has any training run happened since the U4 fix pass (2026-06-12)?**

| Answer | What it means | Work required |
|---|---|---|
| **No retrain since U4** | Existing checkpoints were trained with DPCC-identical training code (B7/B8/B9 only affect *training*). | **Eval-side only.** Revert B6, re-eval. Done. No retrain. |
| **Yes, retrained after U4** | Checkpoints carry the episode-split / final-save behaviour. | Revert B6+B7+B8+B9 **and retrain** to reproduce DPCC training. |

B6 is **always** eval-side and immediate. B7/B8/B9 are **training-side** — reverting their code
does nothing to an already-trained checkpoint; it only changes the *next* training run.

Check the checkpoint timestamps / run logs before deciding. If unsure, assume "Yes, retrained."

---

## The four reverts (exact, both `diffuser_visual_avoiding` and `fm_visual_avoiding`)

Every change below exists in **both** forks. Apply to both `diffuser_visual_avoiding*` and
`fm_visual_avoiding*` so FM and DPCC stay common-mode (or you re-introduce a new asymmetry).

### B6 — EMA → raw weights  *(eval-side, immediate, no retrain)*

**File:** `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` (≈line 179) and
`fm_visual_avoiding_test/eval_fm_visual_avoiding.py` (mirror).

Current (diverged — EMA):
```python
return utils.DiffusionExperiment(dataset, trainer.ema_model.model,
                                 trainer.ema_model, trainer, epoch, losses)
```
DPCC-faithful (raw weights, matches `serialization.py:75`):
```python
return utils.DiffusionExperiment(dataset, trainer.model.model,
                                 trainer.model, trainer, epoch, losses)
```
*Flag-gated version:*
```python
_m = trainer.ema_model if getattr(args, 'use_ema_eval', False) else trainer.model
return utils.DiffusionExperiment(dataset, _m.model, _m, trainer, epoch, losses)
```
→ default `use_ema_eval=False` reproduces DPCC; set `True` for the EMA variant.

### B7 — episode split → window-level `random_split`  *(training-side, retrain to matter)*

**File:** `diffuser_visual_avoiding/utils/training.py` (≈lines 73-80) and FM mirror.

Current (diverged — episode split when the method exists):
```python
else:
    if hasattr(self.dataset, 'episode_split'):
        train_idx, test_idx = self.dataset.episode_split(train_test_split)
        train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        test_dataset  = torch.utils.data.Subset(self.dataset, test_idx)
    else:
        n_train = int(train_test_split * len(self.dataset))
        n_test  = len(self.dataset) - n_train
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_test])
```
DPCC-faithful (window-level split — delete the `episode_split` branch):
```python
else:
    n_train = int(train_test_split * len(self.dataset))
    n_test  = len(self.dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_test])
```
*Flag-gated version:* keep the branch but guard it —
`if getattr(self, 'dpcc_faithful', True) is False and hasattr(self.dataset, 'episode_split'):`
→ defaults to DPCC window-split; the `episode_split` method can stay defined (harmless).

### B8 — remove terminal save  *(training-side, retrain to matter; mostly inert)*

**File:** `diffuser_visual_avoiding/utils/training.py` (≈line 203) and FM mirror.

Delete:
```python
        self.save(self.step)  # B8: persist final weights (last periodic save is at step 80000)
```
DPCC `train()` ends after the epoch loop with no final save. (Inert if you eval `'best'`, which
is the default — reverting only matters if you eval `'latest'`.)

### B9 — remove train-mode restore  *(training-side, retrain to matter; zero practical effect)*

**File:** `diffuser_visual_avoiding/utils/training.py` (≈line 223) and FM mirror.

Delete:
```python
        self.model.train()  # B9: restore train mode after validation pass
```
DPCC `test()` returns while leaving the model in `.eval()`. (No effect on produced weights given
GroupNorm + manual CFG mask — reverting is bit-for-bit fidelity, not a numbers change.)

---

## DO NOT revert these (keep them — they are not DPCC divergences)

| Fix | Why it must stay |
|---|---|
| **B1** RGB/BGR | DPCC has no images; reverting re-breaks visual conditioning. |
| **B2** render 96×96 | Same — visual-only correctness. |
| **B3** dead-code removal + `mpc_batch_size` wiring | Removed dead code (no behaviour change); `mpc_batch_size` is common-mode. |
| **B5** | No code change was made. |
| **B10** pkl banner | Warning only, no behaviour change. |

Reverting any of these buys **zero** DPCC comparability and costs visual correctness.

---

## Execution checklist

1. **Decide STEP 0** (retrain-since-U4? → eval-only vs eval+retrain).
2. Choose **flag-gate** (preferred) or **hard-delete** (fallback). Apply B6 to **both** eval
   scripts; apply B7/B8/B9 to **both** `*/utils/training.py`.
3. **Eval-only path (no retrain since U4):**
   - Revert B6 in both eval scripts.
   - Re-run the two evals (`eval_visual_avoiding_dpcc.sh`, `eval_fm_visual_avoiding.sh`).
   - Existing checkpoints now report on raw weights = DPCC-faithful.
4. **Full path (retrained after U4):**
   - Revert B6 + B7 + B8 + B9 in both forks.
   - Retrain both visual-DPCC and visual-FM from scratch (episode-split → window-split changes
     the data split; must retrain).
   - Re-eval.
5. **Verify parity** against `/workspaces/dpcc`:
   - `diff <(grep -n "ema_model\|trainer.model" eval_...py)` → returns raw-weight form.
   - `grep -n "episode_split\|self.save(self.step)\|self.model.train()" */utils/training.py`
     under the reverted-behaviour expectation (none active, or all flag-guarded to DPCC default).
6. **Keep FM and DPCC symmetric** — whatever you revert, revert in *both* forks, or you trade a
   DPCC-paper gap for a fresh FM-vs-DPCC gap (worse).

---

## Bottom line

1. **Keep the U4 fixes.** B6/B7/B8/B9 make train+eval *more* correct, not less. The paper-mirror
   revert (appendix) is a niche, reviewer-driven action — not the goal.
2. **Repair the one real bug:** EMA-eval is currently paired with raw-model `state_best`
   selection. Fix it with **Option 1** (select on EMA, retrain) — which also eliminates the B9
   eval-mode leak for free — or hold with **Option 2** (raw eval, no retrain) until you can
   retrain. The current half-fixed state (raw select + EMA eval) is the only configuration that
   is actually wrong; do not ship numbers from it.
3. The state `diffuser/` baseline stays untouched and paper-faithful for any DPCC head-to-head;
   Gen9 visual numbers should be labelled "EMA-eval / episode-split," never presented as a
   reproduction of the state-only DPCC paper table.

---

## Caveats

- Line numbers are as of `2026-06-13`, branch `update_into_FM` — re-locate if files moved.
- Nothing executed (Docker = AI-coding only); the "raw underperforms EMA" expectation is the
  standard diffusion prior, not measured here.
- Confirm whether a post-U4 retrain occurred before assuming the eval-only path.
