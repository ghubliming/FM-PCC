# Gen14 Fix_10 — resume that actually resumes, a real save cadence, and a budget that shows in the path

**Date:** 2026-08-22 · **Mirrored into:** Gen16 (`mix_visual_avoiding*`) — see §5
**Triggered by:** job 24838 (Gen14 aligning). Gen16 is a sibling-sync only.
**Status:** code complete, unverified on hardware (container is AI-coding only)

---

## 1. What triggered this

Two dead training jobs on i6-gpu-1, with two different faults:

| job | arm | what happened |
|---|---|---|
| 24857 | Gen16 avoiding `mf` | `FileNotFoundError` on the dataset list, 4 s in. Already fixed by `db4cb99f`; the job ran at `afc525f`, one commit earlier. Not this changelog. |
| 24838 | Gen14 aligning | disk filled (`OSError: [Errno 28]`), then the 24 h wall killed it at **step 83999 of 1e5**. |

24838 is the one that mattered. It had a `state_80000.pt` on disk and **no way to use it**, because:

1. `--auto-resume` exists in the train script but no sbatch ever passed it.
2. Even if it had, `find_latest_checkpoint_step` would have returned **0**.
3. And the newest periodic save was 4000 steps behind the kill, because `save_freq` is
   hard-wired to `n_train_steps // 5` — five saves for an entire run.

Separately, the Gen16 avoiding tree at
`.../mix_visual_avoiding_mf/H8_D..._Emf_tslogit_normal/6/` contained **only `state_0.pt` and
`state_best.pt`** — a run that died before step 20000, i.e. before its first real save. That
directory is exactly the case bug #2 turns into a silent disaster.

---

## 2. Root causes

### 2.1 `state_0.pt` is not progress

`training_twotime.py:221` saves on `self.step % self.save_freq == 0`. That predicate is true
on the **first iteration**, so every run that has ever started owns a `state_0.pt`.

`find_latest_checkpoint_step` globbed `state_*.pt`, discarded `state_best.pt` via the
`int('best')` → `ValueError` path, and returned `max([0]) == 0`. The caller then did:

```python
if resume_step is not None:      # 0 is not None -> True
    trainer.load(resume_step)    # loads a randomly-initialised network
```

and printed `Resuming seed 6 from step 0`. **A fresh run wearing a resume's clothes** — and
one that suppresses the `Resume checkpoint not found` warning that should have fired, because
`state_0.pt` genuinely exists.

### 2.2 Five checkpoints per run

`save_freq = n_train_steps // 5` = 20000. A visual run trains a ResNet-18 alongside the U-Net
at ~5:50 per 1000 steps, so 20000 steps is ~2 h of work in flight at all times. Against a 24 h
wall that is a guaranteed multi-hour loss on any kill.

### 2.3 A shortened run would have been indistinguishable from a full one

`n_train_steps` was the literal `int(1e5)` and appeared in **no path key**. Training at 50k
would have written into the *same* directory as a 100k run — same `exp_name`, same
`diffusion_loadpath` — silently overwriting it, or worse, being evaluated later as if it were
the full-budget model.

---

## 3. What changed

### 3.1 `save_freq` is a real parameter — 4 trainers

`mix_visual_{avoiding,aligning}/utils/{training,training_twotime}.py`

```python
        log_freq=1000,
        save_freq=None,          # NEW
        ...
        self.save_freq = int(save_freq) if save_freq else max(1, int(n_train_steps) // 5)
```

Default behaviour is bit-for-bit unchanged. `int()` also fixes a latent wart: `n_train_steps`
arrives as the float `1e5`, so `save_freq` used to be `20000.0`.

### 3.2 Step 0 no longer counts as a resume point — 2 train scripts

```python
    steps = [s for s in steps if s > 0]
    return max(steps) if steps else None
```

A run killed before its first periodic save now correctly returns `None` and **starts over
openly** instead of pretending to resume.

### 3.3 `--save-every` CLI — 2 train scripts

New argument, forwarded as `save_freq=cli_args.save_every` into `_trainer_kwargs`.

### 3.3b Resume from `state_best.pt` — 2 train scripts

The savepath for 24838 ended up holding **`state_best.pt` and nothing else**: ENOSPC and the
24 h kill between them left no numbered checkpoint, and the periodic saves that did exist are
gone. `find_latest_checkpoint_step` parses an int out of the filename, so it cannot see
`state_best.pt` and returned `None` — ~84k steps of GPU time looked unrecoverable.

It was not. `save_best()` writes `self._checkpoint_payload()` — the **same full payload** as a
periodic save (`training_twotime.py:385-391`): `step`, `model`, `ema`, `optimizer`,
`lr_scheduler`, `best_test_loss`, loss histories. It is a legitimate resume point.

Three changes make it reachable:

1. `--resume-step` now takes a step number **or** the literal `best`, via `_resume_target()`.
   `trainer.load()` already interpolates `f'state_{epoch}.pt'`, so `'best'` resolves to
   `state_best.pt` with no change to the trainer at all.
2. `--auto-resume` falls back to `state_best.pt` when no numbered checkpoint survives.
   Numbered checkpoints still win when present — they are the *later* training state;
   `state_best` is only the last val improvement.
3. `MIX_RESUME_FROM=best|<step>` on the sbatch, for the explicit case.

🔴 **The resumed step is read from the payload, never assumed.** `state_best.pt` is rewritten
on every val improvement, so its step is whenever the last improvement landed — for 24838 that
is somewhere at or below 83999 and is *not* necessarily 80000 or 84000. The train script now
prints `resumed at step N of 100000 (M remaining)` immediately after loading.

A truncated checkpoint (written while the disk was filling) now raises a `RuntimeError` naming
the file, instead of `torch.load` blowing up anonymously — silently restarting from scratch
would burn the whole wall on a run the operator believes is a resume.

### 3.3c Checkpoint writes are atomic — 4 trainers

`torch.save` opens the **destination** path and writes in place. On ENOSPC that leaves a
truncated archive exactly where a known-good checkpoint used to be: the write target and the
only copy are the same file. This is the mechanism behind 24838's loss — not just "the disk
filled", but "the disk filled *during a checkpoint write*".

All eight call sites (`save()` and `save_best()` in both trainers, both siblings) now go
through `_atomic_torch_save()`: write to `<path>.tmp.<pid>`, then `os.replace()`. On POSIX the
replace is atomic, so a full disk fails the **temp** write and leaves the previous checkpoint
intact and resumable. The temp file is removed on any exception, including KeyboardInterrupt
and SIGTERM-driven unwinds, so a failed save cannot itself consume the space that is already
short.

Cost: one checkpoint of transient space during each save.

### 3.4 The budget is now a path key — 2 configs

`config/avoiding-d3il-visual-mix.py`, `config/aligning-d3il-visual.py`

```python
_MIX_FULL_N_TRAIN_STEPS = int(1e5)
_MIX_N_TRAIN_STEPS = int(float(os.environ.get('MIX_TRAIN_STEPS', _MIX_FULL_N_TRAIN_STEPS)))

def _budget_tag():
    if _MIX_N_TRAIN_STEPS >= _MIX_FULL_N_TRAIN_STEPS:
        return None
    pct = 100.0 * _MIX_N_TRAIN_STEPS / _MIX_FULL_N_TRAIN_STEPS
    return f'{int(round(pct))}pct' if abs(pct - round(pct)) < 1e-9 else f'{_MIX_N_TRAIN_STEPS}steps'
```

plus `('train_budget', 'TB')` **appended last** to `args_to_watch_mix_visual_train`, and in the
train-block finaliser:

```python
    if _MIX_BUDGET_TAG is not None:
        blk['train_budget'] = _MIX_BUDGET_TAG
```

🔴 **The tag is set only when the budget is cut.** `watch()` skips keys the block does not
define, so at the full budget the key is absent and every existing path is byte-identical:

```
100000  ->  ..._filmv1_Emf_tslogit_normal
 50000  ->  ..._filmv1_Emf_tslogit_normal_TB50pct
```

Because `_mix_plan_block` mirrors every training identity key and `_mix_loadpath` derives
`diffusion_loadpath` from the **same watch list**, the eval picks the tag up for free — *if it
sees the same env var*. Hence §3.5.

An odd budget that is not a whole percent falls back to `_TB33333steps` rather than rounding
into a collision with a neighbouring run.

### 3.5 Three env knobs on the sbatch entrypoints — 2 train scripts + 2 pipelines

| env | effect | reaches |
|---|---|---|
| `MIX_AUTO_RESUME=1` | `--auto-resume` | train |
| `MIX_SAVE_EVERY=<n>` | `--save-every n` | train |
| `MIX_TRAIN_STEPS=<n>` | budget **and path tag** | train **and eval** |

All three use `${VAR:+...}`, so unset means the command line is character-for-character what
it was before.

The pipelines append `MIX_TRAIN_STEPS` to `EXPORT_OPTS` explicitly rather than trusting
`--export=ALL`. Same reasoning the existing `film_mode` comment gives: a silently-dropped path
key means the eval builds `diffusion_loadpath` for the full-budget directory and dies on a
missing checkpoint *after* the GPU is allocated.

---

## 4. Verification (offline — no torch/numpy in this container)

- `ast.parse` clean on all 6 Python files; `bash -n` clean on all 4 shell scripts.
- `watch()` + `_budget_tag` reimplemented in stdlib and run against the real `mf` block:
  full budget produces a name with **no `TB` fragment** (asserted), 50k appends `_TB50pct`.
- `_budget_tag`: `1e5 -> None`, `5e4 -> 50pct`, `2e4 -> 20pct`, `33333 -> 33333steps`.
- `_resume_target` exercised via `exec` of the real source: `'best'`/`'BEST'` -> `'best'`,
  `'80000'` -> `80000`, `'lastest'` -> `ArgumentTypeError`.
- resume resolution: `['state_best.pt']` -> `'best'` (24838's actual state) ·
  `['state_0','state_best']` -> `'best'` · `['state_0','state_20000','state_80000','state_best']`
  -> `80000` (numbered wins) · `[]` -> `None`.
- `find_latest_checkpoint_step` on the three real cases:
  `['state_0','state_best'] -> None` · `['state_0','state_20000','state_80000','state_best'] -> 80000`
  · 5k cadence killed at 42k `-> 40000`.
- **A0 unaffected**: both trainer files are already declared in the gate's `EDITED` ledger
  (`gates_mix_visual_avoiding.py:77,79`) for the pre-existing episode-split divergence, so
  they were never required to be byte-identical to Gen14's.

Nothing here has executed a tensor op. Run on cluster.

---

## 4.5 G0 ledger — caught by the gate, job 24864

The first cluster run of this fix **failed G0**:

```
! mix_visual_aligning/utils/training.py: DIFFERS from fm_visual_aligning/utils/training.py
! mix_visual_aligning/utils/training_twotime.py: DIFFERS from flow_matcher_v3_alphaflow/utils/training.py
```

Gen16's A0 ledger already lists both trainers under `EDITED`, and that was checked before
editing. **Gen14's G0 is a different gate with a different ledger** — it holds those two files
as `VERBATIM` against their *upstreams* (Gen7 and Gen3v7), not against Gen16. That was not
checked, and the gate caught it. Working as designed.

Resolution: both files move from `VERBATIM` to `GRAFTED_DIFF`, which keeps a **real** check
rather than dropping them out of coverage — the upstreams are actively edited and the training
loop is precisely what G0 must keep watching. The graft is additive with exactly **3** rewritten
source lines each, enumerated in the ledger entry:

| rewritten line | becomes |
|---|---|
| `self.save_freq = n_train_steps // 5` (×1) | honours the `save_freq` argument |
| `torch.save(<payload>, savepath)` (×2) | `_atomic_torch_save(...)` |

Everything else is insertion-only (`save_freq=None` kwarg, the helper). A 4th removal means
something that is not Fix_10 changed, and the gate will say so.

Re-running G0's exact logic offline: VERBATIM 17/17 clean, all 6 `GRAFTED_DIFF` entries at
their declared counts (`training.py` +39/−3, `training_twotime.py` +39/−3) → **G0 PASS**.

Everything else in 24864 was already green, including the whole U8 bone battery
(G-B1/B2/B3/B4/B5/B6/B7) and G1–G7 — so the DiT bones themselves are verified on hardware.

## 5. Sibling sync

Applied identically to Gen16 avoiding: `mix_visual_avoiding/utils/training{,_twotime}.py`,
`mix_visual_avoiding_test/train_mix_visual_avoiding.py`,
`Slurm_Codes/sbatch/mix_visual_avoiding/{train_mix_visual_avoiding,mix_visual_avoiding_pipeline}.sh`,
`config/avoiding-d3il-visual-mix.py`. Partial application would have been worse than none —
`MIX_TRAIN_STEPS` set on an unpatched config is a silent no-op.

Not applied to the UAV (`mix_uav*`) or state-only siblings: different config module, no
visual budget problem, and they were not part of this failure.

---

## 6. How to use it

```bash
# 50 % budget, checkpoint every 5k steps, full pipeline (gates -> train -> eval), seed 6
MIX_TRAIN_STEPS=50000 MIX_SAVE_EVERY=5000 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_avoiding/mix_visual_avoiding_pipeline.sh mf 6

# picking job 24838 back up from state_best.pt (Gen14 aligning, full 1e5 budget)
MIX_RESUME_FROM=best MIX_SAVE_EVERY=5000 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh <engine> 6

# equivalent here, since no numbered checkpoint survives for it to prefer
MIX_AUTO_RESUME=1 MIX_SAVE_EVERY=5000 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh <engine> 6
```

🔴 Do NOT pass `MIX_TRAIN_STEPS` when resuming an existing full-budget run: it appends
`_TB<pct>pct` to the savepath, and the resume would then look in a directory that does not
exist. The budget knob is for FRESH runs.

🔴 **Any manual eval of a reduced-budget run must carry the same `MIX_TRAIN_STEPS`**, or it
resolves the full-budget path. The pipeline handles this; a standalone
`eval_mix_visual_avoiding.sh` / `eval_k_sweep.sh` invocation does not unless you set it.

---

## 7. Still open

1. **Disk.** Fix_10 does not create free space. `df -h`, `du -sh wandb logs` before resubmitting
   — 24838's ENOSPC will recur otherwise, and a shorter save cadence writes *more* files.
2. **The 16 unaccounted hours.** 24838 did 84 epochs × 5:50 ≈ 8 h 10 m of measured stepping
   inside a 24 h wall. Either dataset load is enormous, per-epoch `test()` is heavy, or the
   full disk was stalling writes for most of the run. Worth timing on the next run before
   trusting any wall-clock budget.
3. **tqdm in batch logs**, against the standing rule. Contributes to log bloat and therefore
   to #1.
4. **A7 has never passed on hardware** — its constructor-bug fix (`afc525fb`) is unverified.
   The pipeline's gates stage is the first thing that will exercise it.
5. Consider making `save_freq` default to `min(n_train_steps // 5, 5000)` rather than relying
   on the operator to pass `MIX_SAVE_EVERY` every time.
