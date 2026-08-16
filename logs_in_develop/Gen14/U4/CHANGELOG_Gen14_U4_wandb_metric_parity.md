# CHANGELOG — Gen14 U4: W&B metric parity + per-epoch flush

**Date:** 2026-08-02 · **Follows:** [`../fix_3/CHANGELOG_Gen14_fix_3_multiseed.md`](../fix_3/CHANGELOG_Gen14_fix_3_multiseed.md)
**Trigger:** first completed Gen14 run (job 24124, `mf`, seed 6, 1e5 steps) reached W&B with
almost none of its metrics — notably **not** `raw_mse_u`, the one curve the MeanFlow arm must
be read on.

---

## 1. The claim, checked

Confirmed, and worse than stated. Measured on the run's own `losses.pkl`:

| | count |
|---|---|
| metric series the trainer wrote to `losses.pkl` | **34** (30 non-empty) |
| series reaching W&B | **4** |

The four were `train/loss`, `test/loss`, `train/a0_loss`, `test/a0_loss`. Everything else was
computed, reduced over the test set, pickled to disk — and dropped.

Among the dropped: `raw_mse_u`, `raw_mse_v`, `per_dim_rms_u`, `h_mse_b0..b3`, `h_mean`,
`fm_frac`, `lr_history`, `grad_norm_history`, plus their `val/` counterparts.

**Why this mattered on job 24124 specifically.** The four keys that *did* arrive are exactly the
four that are least informative on `mf`. `test/loss` moved 1.00 → 0.926 across the whole run,
because the adaptive weight pins it near its ceiling by construction (COMPARE §7.1 — never read
`train/loss` as convergence on an adaptive-weight objective). The curve that actually carries the
signal, `val/raw_mse_u`, ended at 7.29 with `train/raw_mse_u` at 8.84 — and was invisible.

Note the console log had them all along (`raw_mse_u=8.19` etc. in the tqdm postfix at
`00_53_50_train_mix_visual_aligning_24124.log:230`). The data was never lost; only the W&B view
was blind.

## 2. Root cause — an inherited pre-U9 logger

`log_wandb_curves_from_losses` (`train_mix_visual_aligning.py:42`) hardcoded four keys by name.
It is Gen7's version, carried over verbatim when Gen14 was copy-modified from
`fm_visual_aligning_test/train_fm_visual_aligning.py`.

That is the whole story: **Gen7's script predates the Gen3v6 U9 metric-parity pass.** Gen14 then
adopted Gen3v7's `utils/training_twotime.py` as the `mf`/`af` trainer — which emits the full
`EXTRA_METRIC_KEYS` family (`training_twotime.py:21`) — without adopting the matching reader.
Producer upgraded, consumer did not.

The sibling that already had it right:

| file | line | map size |
|---|---|---|
| `FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py` | 56 | 27 keys ✅ |
| `FM_v3_alphaflow_test/train_flow_matching_v3_alphaflow.py` | 84 | 32 keys ✅ (adds α telemetry) |
| `mix_visual_aligning_test/train_mix_visual_aligning.py` | 42 | **4 keys** ❌ |

U4 is sibling-parity restoration, not new design.

## 3. Second defect found while fixing it — log-at-the-end

`log_wandb_curves_from_losses` was called **once, after `trainer.train()` returned** (`:426`).
Gen3v6 had already moved to a per-epoch flush at U9 (`trainer.train(on_epoch_end=_flush_wandb)`);
Gen14 inherited the pre-U9 call site too.

Consequence: **a run killed by the wall wrote nothing to W&B at all**, despite a complete and
valid `losses.pkl` on disk. Job 24124 took 9 h 24 m for one seed at 2.96 it/s — close enough to
the 24 h wall that a slower arm, a busier node, or a resume could cross it. This is the same
class of trap as fix_3b's resume gap, and Gen14 is more exposed than Gen3v6 for the same reason:
two ResNet-18 encoders training alongside the U-Net.

## 4. Changes

One file: `mix_visual_aligning_test/train_mix_visual_aligning.py` (+115 / −19). No trainer, no
model, no config, no sbatch.

### (a) `WANDB_COMPANION_KEYS` — module-level 30-key map

Ported from the Gen3v7 script (the superset — it carries the α-schedule keys the `af` arm needs
on top of Gen3v6's set). Lifted to module scope so the gates and any future analysis script can
import the canonical mapping rather than re-deriving it.

Missing pkl keys are skipped silently, so **one map serves all four arms**: `ddpm` has no
h-buckets, `mf` has no `alpha`, and neither errors. Verified on the real `mf` pickle — the five
α keys are present-but-empty and correctly produce no series.

### (b) `log_wandb_curves_from_losses(..., after_step=-1) -> last_step`

Incremental replay: skips steps `<= after_step`, returns the highest step logged. Same contract
as the Gen3v6/v7 siblings, so the three scripts now behave identically.

Summary keys extended to match: `final_val_raw_mse`, and `first_train_h_mse_b*` /
`final_train_h_mse_b*` for all four buckets — the kill-criterion inputs. If `b3` (h∈[0.6,1]) sits
flat at its step-0 value while `b0` (h=0) dropped ~10×, the field is untrained exactly where
low-NFE sampling lives.

### (c) Per-epoch flush, guarded by signature

```python
if run is not None and 'on_epoch_end' in inspect.signature(trainer.train).parameters:
    trainer.train(on_epoch_end=_flush_wandb)
else:
    trainer.train()
```

**The guard is load-bearing, not defensive padding.** Gen14 dispatches two different trainers
(`engine_registry.get_trainer_cls`):

| arm | trainer | `train()` signature |
|---|---|---|
| `ddpm`, `fm` | `mix_visual_aligning/utils/training.py:185` | `def train(self)` — **no hook** |
| `mf`, `af` | `mix_visual_aligning/utils/training_twotime.py:275` | `def train(self, on_epoch_end=None)` |

An unconditional `trainer.train(on_epoch_end=...)` would `TypeError` the `ddpm` and `fm` arms at
step 0 — and neither may be edited to add the parameter, because **both are G0-locked verbatim
copies** (§3.1 structural rule). The `else` branch keeps them on end-of-run logging, which is
their current behaviour exactly.

The final catch-up call after `train()` is retained. It picks up the last partial epoch, and
degrades to a full replay when the cursor is still −1 (i.e. on the hookless arms).

## 5. Verification (local, container)

- `ast.parse` — pass.
- **Replayed the real `temp/2026-08-02/losses.pkl` (job 24124) through the new function** with a
  stub run object: **100 rows, 26 distinct metrics** (was 4), `last_step=99000`, and 11 summary
  keys. The 26 is correct rather than 31 — the five α series are legitimately empty on an `mf`
  run.
- Incremental path: `after_step=50000` → 49 rows, cursor 99000. Cursor arithmetic checks out.
- Signature guard checked against both trainer classes by `grep`, per the table in §4c.
- Line endings: LF preserved.
- **Not run locally:** anything needing the FMPCC env. Cluster job required.

## 6. Blast radius

Zero on training dynamics. U4 touches only what is read out of `losses.pkl` and when it is
shipped — no loss, no optimizer, no checkpoint, no seed, no sampler. The `.pkl` and `.pt` files
are byte-identical to what the old code would have produced.

**Job 24124 is not lost.** Its `losses.pkl` holds every one of the 30 series. To backfill its W&B
run, replay the pickle against the existing run id (`8eb9bo8t`) rather than retraining.

## 7. Commands

```bash
# G0 must still pass — U4 touches no copied file, but confirm before the next chain
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/gates_mix_visual.sh

# next arm, with full telemetry from epoch 1
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 6
```

What to watch in W&B, in priority order: **`val/raw_mse_u`** (the convergence signal),
`train/h_mse_b3` vs `train/h_mse_b0` (low-NFE viability), `train/grad_norm` (is `gradient_clip=1.0`
biting?), `train/lr` (schedule sane / resumed correctly). Ignore `train/loss` and `test/loss` on
`mf`/`af` — pinned by construction.

## 8. Still open

- **Backfill of job 24124's W&B run** — not done; needs a small replay script pointed at run
  `8eb9bo8t`. Offered, not written.
- Unchanged from fix_3: Gen3v7's `--auto-resume` CLI mirror is deferred; the Gen7 K=1 projector
  defect (`fm` arm, G6 `KNOWN UPSTREAM DEFECT` banner) is untouched and blocks only a low-NFE
  sweep, not the default `flow_steps_v3: 100` eval.
- `VisualAgentWrapper` candidate-selection audit against `ecbae16f` / `a6a7a8ad`, outstanding
  since the init changelog.
