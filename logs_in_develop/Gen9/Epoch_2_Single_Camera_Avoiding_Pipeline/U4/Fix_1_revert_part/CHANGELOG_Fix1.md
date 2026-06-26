# Fix 1 — EMA Selection / Deployment Consistency

**Date:** 2026-06-13
**Branch:** `update_into_FM`
**Author:** Claude Sonnet 4.6
**Parent audit:** `U4/CHANGELOG.md` (B1–B10 fix pass)
**Companion:** `U4/MEMO_DPCC_REVERT_PLAN.md` (Option 1 recommended fix)

---

## What was wrong (the U4 B6 half-fix)

U4 fixed B6 by switching eval to EMA weights (`trainer.ema_model`) in both eval scripts.
That was correct. But `test()` in both `*/utils/training.py` was **not updated at the same
time**, so it continued scoring `self.model` (raw weights). The broken chain:

```
training loop:
    test()  →  scores  self.model (raw)           ← still old
    state_best saved when raw loss improves        ← chosen on raw criterion

eval:
    load state_best  →  deploy  trainer.ema_model  ← EMA weights (B6 fix)
```

Selecting "best" on one network and deploying another. The step that is optimal for raw
weights is **not** necessarily optimal for EMA weights. This is a self-inflicted
correctness bug — the two halves of the train+eval pipeline disagreed.

---

## What DPCC originally did (the "purity" baseline)

```python
# DPCC /workspaces/dpcc — serialization.py:75 (eval) + training.py test()
# Both use self.model (raw). Self-consistent, just not EMA-optimal.

def test(self, n_test=100):
    self.model.eval()                       # DPCC: raw model scored
    ...
    loss, infos = self.model.loss(*batch)   # raw
    ...
    # DPCC: no self.model.train() restore — model left in eval mode (B9 was our add-on)
```

DPCC was self-consistent because BOTH selection and deployment used raw weights.
The "revert to DPCC purity" path would require reverting B6 in the eval scripts as well
(return `trainer.model` instead of `trainer.ema_model`). That is not done here because it
gives up the EMA quality improvement without fixing the real consistency problem.

---

## Fix applied (Option 1 from MEMO)

Score `self.ema_model` in `test()` so `state_best` is selected on the network that eval
will actually deploy. Both selection and deployment now use EMA — fully self-consistent,
and EMA quality is preserved.

**Bonus: eliminates B9 for free.** `self.model` is never switched to `.eval()` in the new
path, so the `self.model.train()` restore (B9 band-aid) is no longer needed and has been
removed.

### Files changed

| File | Lines changed |
|------|--------------|
| `diffuser_visual_avoiding/utils/training.py` | `test()` method — score `ema_model`; remove B9 restore |
| `fm_visual_avoiding/utils/training.py` | Same (mirror) |

### Code diff summary (applies identically to both files)

```python
# BEFORE (U4 state, scoring raw model):
def test(self, n_test=100):
    self.model.eval()
    ...
    loss, infos = self.model.loss(*batch)   # raw weights → state_best on raw criterion
    ...
    self.model.train()  # B9 band-aid
    return test_loss, test_a0_loss

# AFTER (Fix1, scoring EMA model):
def test(self, n_test=100):
    # <inline comments explaining DPCC origin and why Fix1 is correct — see source>
    eval_model = self.ema_model   # must mirror eval's trainer.ema_model
    eval_model.eval()
    ...
    loss, infos = eval_model.loss(*batch)  # EMA weights → state_best on EMA criterion
    ...
    # self.model.train() removed — self.model never touched, no restore needed
    return test_loss, test_a0_loss
```

---

## Why Option 1 over reverting to DPCC (Option 2)

| | DPCC-purity revert (Option 2) | Fix1 — EMA selection (Option 1) |
|---|---|---|
| **State_best selected by** | raw loss | EMA loss |
| **Eval deploys** | raw weights | EMA weights |
| **Self-consistent?** | Yes | Yes |
| **EMA quality retained?** | No (gives up EMA) | Yes |
| **Retrain needed?** | No (existing checkpoints selected on raw — valid immediately) | Yes (existing `state_best` was selected on raw; only new training runs benefit) |
| **Changes to eval scripts?** | Must revert B6 (return `trainer.model`) | No change needed |
| **Recommended?** | Stopgap only | Permanent fix |

Option 2 is valid as an **immediate stopgap** (consistent with existing checkpoints) but
should be followed by a Fix1 retrain. The current state before this fix (raw select + EMA
eval) is the **only wrong configuration** — never ship numbers from it.

---

## WARNING — Which fixes are material vs cosmetic

Not all B-series fixes move numbers equally. Before reverting or reporting results, know
which levers actually matter:

| Fix | Real numbers impact? | When active | Direction if reverted |
|-----|---------------------|-------------|----------------------|
| **B6** (EMA eval) | **BIG — active now** | Existing checkpoints | Numbers drop: raw weights are noisier than EMA |
| **B7** (episode split) | **Significant** | Next retrain only | Leaky window split → optimistic `state_best` selection (near-duplicate train/test windows) |
| **Fix1** (EMA selection) | **Significant** | Next retrain only | `state_best` chosen at wrong step (raw-optimal ≠ EMA-optimal) |
| **B8** (terminal save) | **Zero** with `diffusion_epoch:'best'` | Next retrain, but irrelevant | Only matters if someone evals `'latest'` |
| **B9** (eval-mode restore) | **Zero** | Removed by Fix1 | GroupNorm + manual CFG mask: no layer responds to `.eval()` mode |

### What this means in practice

- **B6 is the sharpest single lever.** EMA smooths gradient noise accumulated over 100k
  steps. The gap between raw and EMA at deploy time is typically the largest single-source
  quality difference. If a revert drops success rate noticeably, B6 is the likely cause.

- **Fix1 + B7 together determine which checkpoint wins.** B7 makes the test-loss signal
  honest (no near-duplicate leakage). Fix1 makes the selection criterion match the deployed
  network. Both must be active for `state_best` to be meaningful. They only take effect on
  the next retrain — existing checkpoints carry neither guarantee.

- **B8 and B9 are bookkeeping.** Reverting or keeping them changes zero numbers in the
  standard eval workflow. Do not spend retrain budget on these.

- **Do not report numbers from the pre-Fix1 state** (U4 B6 active, Fix1 not yet applied):
  that is the only configuration where selection and deployment are genuinely mismatched.
  Numbers from that state are neither DPCC-faithful nor Fix1-clean.

---

## What still needs to happen

1. **Retrain** both `diffuser_visual_avoiding` and `fm_visual_avoiding` with this fix in
   place. The training code now scores EMA in `test()`, so the new run will generate a
   `state_best` that reflects EMA loss → correct for eval.

2. **Re-eval** after the retrain completes. Existing `state_best` checkpoints (trained
   before this fix) should be treated as "raw-selected, EMA-evaluated" and labelled
   accordingly — they are neither DPCC-faithful nor Fix1-clean.

3. **FM fork stays common-mode with DPCC fork** — both files are patched identically.
   If one fork ever diverges from the other on this point, the FM-vs-DPCC comparison
   becomes unfair.

---

## Inline comments in source

The `test()` function in both changed files contains inline comments that explain:
- What DPCC originally did and the "purity revert" path
- What U4 B6 broke (the half-fix mismatch)
- Why Fix1 (scoring EMA) is the correct resolution
- What to change if a reviewer demands DPCC-identical behaviour

Read the source comments for the canonical decision rationale co-located with the code.

---

## Caveats

- Fix1 only affects the **next retrain**. Any checkpoint trained before 2026-06-13 was
  selected on raw loss; re-train to get a clean Fix1 checkpoint.
- Eval scripts (`eval_visual_avoiding_dpcc.py`, `eval_fm_visual_avoiding.py`) are
  **unchanged** — they already return `trainer.ema_model` (B6 fix, correct).
- State baseline (`diffuser/`) is untouched; no change needed there (never used EMA eval).
