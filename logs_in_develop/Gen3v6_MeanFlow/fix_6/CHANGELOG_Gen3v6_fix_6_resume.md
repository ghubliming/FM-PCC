# CHANGELOG — Gen3v6 fix_6: restore auto-resume (and make "resume" mean resume)

**Date:** 2026-08-01 · **Follows:** [`../fix_5/CHANGELOG_Gen3v6_fix_5_temporal_consistency_reference.md`](../fix_5/CHANGELOG_Gen3v6_fix_5_temporal_consistency_reference.md)
**Trigger:** cluster run **24069** (`13_21_35_train_meanflow_24069.log`, node i6-gpu-1, git `cb859e3`)
**Scope:** 6 files — Gen3v6 (primary) + Gen3v7 mirror + both train sbatch scripts. **No model/objective/config code touched.**

---

## 1. What actually happened in job 24069

The forensics matter, because the resume point depends on them and two of the four facts
that were floated in chat were wrong.

| | |
|---|---|
| Seeds requested | `[7, 8, 9, 10]` (`cli --seeds` — the sbatch's hardcoded `--seeds 6` was bypassed) |
| Seed 7 | ✅ complete (log:321) |
| Seed 8 | ✅ complete (log:608) |
| **Seed 9** | ❌ **died at step 80000**, ~6 h in |
| Seed 10 | ❌ never started |
| Killed by | `RuntimeError: File .../9/state_80000.pt cannot be opened.` → `OSError: [Errno 28] **No space left on device**` |
| Failing frame | `training.py:190 self.save(label)` → `:350 torch.save(...)` |

**Corrections to the initial read of this log:**

1. It was **seed 9**, not seed 7. Seed 7 finished ten hours earlier. The traceback path
   (`.../9/state_80000.pt`) is the only place the seed appears.
2. **`state_79000.pt` does not exist and never could.** `save_freq = n_train_steps // 5`
   (`training.py:75`) = **20000** for this config, so the periodic checkpoints are
   `state_0 / 20000 / 40000 / 60000 / 80000`. The newest periodic file for seed 9 is
   **`state_60000.pt`**. Resuming from a step number read off the epoch counter would have
   thrown a `FileNotFoundError`.
3. `state_best.pt` **is** the better resume point here: it is written at every log step that
   improves val loss (`training.py:222`), val loss was still improving at epoch 79
   (`loss_test=0.856`, the run's best), and it carries the identical full payload. It should
   be at step ~79000 — **19000 steps ahead of the newest periodic file**.
4. The root cause is a full disk, not the training code. Nothing here fixes that; §5 adds a
   pre-flight so it is at least visible at job start.

**Gen3v7 / α-Flow (job 24070) was unaffected** — all four seeds completed (log:1277). Its
copy of the fix is preventative.

---

## 2. Why the feature was missing

Gen1/Gen2/Gen3v1 all have it: `FM_test/train_FM.py` and `FM_v3_test/train_FM_v3.py:98-100,
150-160, 284-293` carry `--auto-resume / --resume-step / --resume-seed` plus
`find_latest_checkpoint_step()`. Gen3v4-iMF rewrote the launcher and dropped the block;
Gen3v6 was copy-modified from Gen3v4 and inherited the gap, Gen3v7 from Gen3v6.

**The Trainer never lost the capability.** `Trainer.train()` (`training.py:259-273`) is
written entirely against `self.step` — it computes `remaining_steps` and the starting epoch
from it, and returns immediately if the budget is already met — and `Trainer.load()`
(`:430`) restores `self.step` plus the full loss history. The single missing piece was the
CLI wiring, so **nothing ever called `load()`**, and `self.step = 0` at `:144` stood.

### The second, quieter defect

Restoring `self.step` alone is not a resume. `_checkpoint_payload()` stored `step`, `model`,
`ema` and the loss curves — **not the optimizer**. So a "resumed" run would have:

- restarted **Adam** with zeroed first/second moments, and
- rebuilt the **cosine LR schedule from scratch**, i.e. at step 80000 the LR would jump from
  the `4.87e-5` the run had annealed to (log:791) **back into the 3e-4 warmup** — a 6× jump
  at 80 % of the budget.

`training.py:127-129` already admitted this in a comment ("it is rebuilt from scratch on
resume"). It is fixed here rather than documented again.

---

## 3. What changed

### `flow_matcher_v3_meanflow/utils/training.py` (+ Gen3v7 mirror)

- `_checkpoint_payload()` now also stores `optimizer`, `lr_scheduler`, `best_test_loss`.
- New `_restore_optimizer_state(data)`, called from `load()` after `self.step` is set:
  - checkpoint written by fix_6 or later → **exact restore** of Adam + schedule + best-val;
  - **pre-fix_6 checkpoint** (no such keys) → Adam moments are unrecoverable, but the LR
    schedule is a pure function of the step count, so it is **replayed forward to
    `self.step`** and a `⚠️` is printed. This is what makes seed 9's existing
    `state_best.pt` usable *today* without a warm-restart LR spike.

### `FM_v3_meanflow_test/train_flow_matching_v3_meanflow.py` (+ Gen3v7 mirror)

New flags: `--auto-resume`, `--resume-step`, `--resume-seed`, `--force-restart`.

| Helper | Job |
|---|---|
| `find_latest_checkpoint(savepath)` | returns `(label, step)` across **both** checkpoint families, reading `state_best.pt`'s stored `step` so it is not silently outranked by an older periodic file (see §1.3) |
| `last_logged_step(savepath)` | max step in `losses.pkl` |
| `training_already_complete(savepath, trainer)` | see below |
| `resolve_resume(...)` | manual `--resume-step` wins for its seed and **raises** if the file is absent; otherwise auto |

**Why `training_already_complete` reads `losses.pkl` and not the checkpoints:** with
`save_freq = n_train_steps // 5`, the newest periodic file of a *finished* seed is
`state_80000.pt` — byte-for-byte indistinguishable in name from a seed that *died* at 80000.
Job 24069 produced both at once (seeds 7/8 finished, seed 9 died). Without this check,
`--auto-resume --seeds 7 8 9 10` would silently re-train the last 20 % of the two good seeds
with cold Adam moments and overwrite their `losses.pkl`. `losses.pkl` settles it: a completed
seed logs its last point at `n_train_steps - log_freq` = 99000.
*Caveat:* a seed killed inside its final log period reads as complete — at most `log_freq`
steps, and strictly better than the alternative.

Wiring sits **before** `wandb.init()`, so a skipped seed does not create an empty W&B run.
On resume, `log_wandb_from_losses` replays the restored history from step 0, so the new run's
curves are complete rather than starting mid-air.

### `Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh`, `Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh`

- `--seeds 6` was hardcoded → `TRAIN_SEEDS` env var (default `6`, unchanged behaviour).
- `AUTO_RESUME` env var, **default on**: a no-op on a fresh savepath, and on a restart it
  continues each seed and skips finished ones.
- `"$@"` is now forwarded to python, so `--resume-step` / `--resume-seed` reach the script
  through `submit.sh` (which already forwards its own trailing args, `submit.sh:11-13,43`).
- **Disk pre-flight**: prints `df -h` for the repo and `$HOME` and warns under 10 G. It does
  not abort — it just makes the thing that killed 24069 visible in the first 20 lines of the
  log instead of the last.

---

## 4. Recovering job 24069

```bash
# on i6-gpu-1, in ~/FMPCC/FM-PCC
git pull

# 0) FREE SPACE FIRST — this is what killed the job
df -h ~ .
conda clean --all -y && pip cache purge
rm -rf ~/.local/share/wandb/artifacts/staging/*

# 1) confirm what seed 9 actually has (expect state_0/20000/40000/60000 + state_best.pt)
ls -la logs/avoiding-d3il/flow_matching_v3_meanflow/H8_Dflow_matcher_v3_meanflow.models.MeanFlowODE_aw10_objmeanflow_bbmf_dit_tslogit_normal_dp0.5/9/

# 2) resume: seed 9 continues from its newest checkpoint, seed 10 starts fresh,
#    seeds 7 and 8 are detected as complete and skipped
TRAIN_SEEDS="7 8 9 10" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh
```

Expected header for each seed:

```
[ train ] Seed 7 already reached 100000 steps — skipping
[ train ] Seed 8 already reached 100000 steps — skipping
[ train ] Seed 9
[ train ] Resuming seed 9 from state_best.pt
[ utils/training ] ⚠️  Pre-fix_6 checkpoint: no optimizer state stored. ...
[ utils/training ] LR schedule fast-forwarded to step 79000 (lr=4.87e-05)
[ train ] Restored step 79000; 21000 of 100000 steps remain
[ train ] Seed 10: no checkpoint found, starting from step 0
```

If `state_best.pt` turns out to be missing or corrupt (it was being written around the time
the disk filled), force the periodic one:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/train_meanflow.sh \
    --resume-step 60000 --resume-seed 9
```

Runtime: seed 9 ≈ 1.6 h (21000 steps at the log's ~3.75 it/s) + seed 10 ≈ 7.5 h. The sbatch
`--time` is already 24 h and covers it.

**Read `train/lr` in W&B on the resumed run before trusting it.** A resume that worked shows
LR continuing its cosine descent from ~4.9e-5; a spike back toward 3e-4 means
`_restore_optimizer_state` did not run.

---

## 5. Verification

| Check | Result |
|---|---|
| `py_compile` — both train scripts, both trainers | PASS |
| `bash -n` — both sbatch scripts | PASS |
| CRLF line endings preserved (3 of the 4 py files are CRLF) | PASS — `git diff --stat` shows only the intended lines |
| `submit.sh` forwards trailing args to the job script | confirmed, `submit.sh:11-13,43` |
| `Trainer.train()` is resume-aware without modification | confirmed, `training.py:259-273` |
| `save_freq` arithmetic vs. the log's checkpoint set | confirmed, `100000 // 5 = 20000` |
| Files touched outside Gen3v6/Gen3v7 | **none** |

**Not verified:** nothing was executed — this container has no torch. The resume path itself
(`trainer.load()` on a real checkpoint, the LR fast-forward, the completeness check against a
real `losses.pkl`) **must be exercised on the cluster**, and §4's expected header is the
thing to check it against.

---

## 6. Still open — NOT done here

1. 🔶 **Gen3v4-iMF (`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`) still has no
   resume**, and its trainer has no `_checkpoint_payload()` at all, so the port is more
   invasive than a mirror. Gen3v4 is not currently training, so it was left alone rather than
   edited speculatively. `FM_v3_imeanflow_test/train_flow_matching_v3_ode_selectable.py` and
   `FM_v3_drifting_test/` **do** have the CLI half already (but not the optimizer half).
2. **No final checkpoint at `n_train_steps`.** Because `save_freq = n_train_steps // 5` and
   the loop ends at step 99999, a completed seed's newest file is `state_80000.pt` — the
   final 20000 steps of weights exist only in `state_best.pt`. Worth deciding on separately;
   adding a sixth checkpoint per seed on a disk that just filled up is not obviously right.
3. **The disk itself.** The pre-flight warns; it does not clean up. `~/.local/share/wandb`
   and the conda pkg cache are the two large, safely-deletable trees.
4. Everything carried over from fix_5 §7 (sibling `policies.py` copies, `cand_prox` ranking).
