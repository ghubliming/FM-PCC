# CHANGELOG — Gen14 fix_3: multi-seed default + per-seed job fan-out

**Date:** 2026-08-01 · **Follows:** [`../fix_2/CHANGELOG_Gen14_fix2.md`](../fix_2/CHANGELOG_Gen14_fix2.md)
**Trigger:** the Gen14 train + eval defaults were **seed 6 only** — fine for a smoke run,
useless for a result table.

---

## 1. What was wrong

| stage | where | old default |
|---|---|---|
| train | `train_mix_visual_aligning.sh` | `SEED="${2:-6}"` → `--seeds "$SEED"`, one seed |
| eval  | `eval_mix_visual_aligning.sh` | `$2` blank → fell through to the yaml |
| yaml  | `config/visual_aligning_eval.yaml:21` | `seeds: [6]` (`[6,7,8,9,10]` commented out) |
| pipeline | `mix_visual_aligning_pipeline.sh` | `SEED="${2:-6}"`, one seed to both stages |

The python defaults never applied: `train_mix_visual_aligning.py:32` carries
`DEFAULT_SEEDS = [5,6,7,8,9]`, but the sbatch always passed `--seeds`, so it was dead.

## 2. Two constraints that shaped the fix

**(a) The eval yaml is shared.** `config/visual_aligning_eval.yaml` is read by the Gen6V4
and Gen7 evals as well as Gen14 — its `exps:` list names all three. Uncommenting `seeds:
[6,7,8,9,10]` there would make *those* generations start looking for seed 7–10 checkpoints
they may not have. **The yaml is therefore left untouched at `seeds: [6]`;** Gen14 carries its
seed list on the command line instead. Everything else Gen14 evaluates — variants,
constraints, `n_contexts`, `enlarge_constraints` — still comes from the shared yaml, which is
the point: same benchmark for every arm, only the seed set is Gen14's own.

**(b) Visual training cannot serialise 5 seeds.** One seed is 1e5 steps with a ResNet-18
pair training alongside the U-Net — already a large fraction of the 24 h wall. Five
sequential seeds in one job would be killed mid-run. So the multi-seed default is delivered
as **one job per seed**, not as a longer job.

## 3. Changes

### `mix_visual_aligning_test/eval_mix_visual_aligning.py` (+11 / −1)

New `--seeds` (plural, `nargs='+'`), and the resolution made explicit with a printed source:

```python
if args_cli.seed:      seeds, _seed_src = [args_cli.seed], 'cli --seed'
elif args_cli.seeds:   seeds, _seed_src = [int(s) for s in args_cli.seeds], 'cli --seeds'
else:                  seeds, _seed_src = config['seeds'], 'config/visual_aligning_eval.yaml'
print(f'[ eval ] seeds: {seeds}  (source: {_seed_src})')
```

Precedence `--seed` > `--seeds` > yaml. Singular `--seed` keeps its old meaning exactly, so
every existing invocation is unaffected. This is the only python change in fix_3.

### `Slurm_Codes/sbatch/mix_visual_aligning/train_mix_visual_aligning.sh`

`SEEDS="${2:-${MIX_SEEDS:-6 7 8 9 10}}"`, passed **unquoted** to `--seeds` so it word-splits.
`$2` still accepts a single seed for fan-out. Prints a wall-time warning when handed more
than one seed, pointing at the pipeline.

### `Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh`

Same `SEEDS` resolution. Now always passes `--seeds $SEEDS` rather than sometimes falling
through to the yaml — the seed set is explicit in the log either way.

### `Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh`

Fan-out. One **shared** gates job, then one train→eval chain per seed:

```
gates ──┬─> train(seed 6) ──> eval(seed 6)
        ├─> train(seed 7) ──> eval(seed 7)
        └─> ...
```

The gates are submitted **once**, not per seed: G0–G6 check copy fidelity, the JVP, the α
schedule and the projector, none of which touch a seed, so five copies would be pure waste.
Each eval depends only on its own train, so one seed dying does not block the others.

**Not changed:** `config/visual_aligning_eval.yaml`, `config/aligning-d3il-visual.py`, and
every Gen6V4 / Gen7 / Gen3v6 / Gen3v7 file. fix_3 touches four Gen14-only files.

## 4. Race check

Parallel per-seed eval jobs write to `logs/aligning-d3il-visual/plans/
mix_visual_aligning_<engine>/<exp>/results/<seed>/` — the seed is in the path
(`eval_mix_visual_aligning.py:2335`), and there is no cross-seed aggregation step in the
script. Concurrent seeds cannot collide. Cross-seed aggregation stays a downstream
`Data_Analysis/` job, as before.

## 5. Verification (local)

- `bash -n` on all four sbatch scripts — pass.
- `ast.parse` on the eval script — pass.
- Word-splitting simulated: `for SEED in $SEEDS` over `"6 7 8 9 10"` yields exactly 5
  iterations.
- **Not run locally:** anything requiring the FMPCC env. Cluster job required.

## 6. Commands

```bash
# full multi-seed run for one arm (5 train jobs + 5 eval jobs + 1 gates job)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh fm

# single-seed smoke run first (recommended — validates the training plumbing cheaply)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh fm 6

# a subset, quoted
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af "6 7 8"
```

## 6b. fix_3b — upstream re-sync of the two-time trainer (job 24111 G0 FAIL)

The first pipeline submission at `ff3f3c2` (job 24110 → gates 24111) came back **G0 FAIL,
all six other gates PASS**. Cause was NOT fix_3:

```
! mix_visual_aligning/utils/training_twotime.py: DIFFERS from
  flow_matcher_v3_alphaflow/utils/training.py
```

`training_twotime.py` was copied at `b5846ee5` (Gen14 init). Its source then moved twice,
**after** the copy was cut:

| commit | what it added to `flow_matcher_v3_alphaflow/utils/training.py` |
|---|---|
| `d97eb92c` | Gen3v6/7 Fix6 — `optimizer`/`lr_scheduler` in `_checkpoint_payload()`, `_restore_optimizer_state()` called from `load()` |
| `15b82d6b` | Fix6.2 — recover the `best_test_loss` watermark from restored history |

The diff was **55 lines, all deletions on the Gen14 side, zero additions** — Gen14's copy had
no divergence of its own, it was simply the pre-fix_6 version. So this was not a defect in
Gen14 and not a regression; it is the sibling-sync pattern CLAUDE.md warns about, and G0
caught it exactly as designed. The gate went red because the *upstream* moved.

**Action:** verbatim re-copy, same CRLF→LF treatment as the original.

```bash
tr -d '\r' < flow_matcher_v3_alphaflow/utils/training.py > mix_visual_aligning/utils/training_twotime.py
```

G0 re-run locally: **PASS, 23/23, exit 0.** `ast.parse` clean; `numpy as np` (needed by the
fix_6.2 `np.inf` check) already imported at line 4; `from .arrays import batch_to_device`
resolves against Gen14's own `utils/arrays.py`.

**Why it was worth doing rather than bypassing the gate.** The old copy trained correctly —
this bought no bug fix. It bought *resume*. Without fix_6 a resumed run restarts Adam with
zeroed moments and rewinds the cosine LR (measured 4.87e-5 → 3e-4 at step 80000), and
without fix_6.2 the first post-resume test overwrites `state_best.pt` — the file eval
loads — with a possibly worse model. Gen3v6 hit this for real (job 24069, seed 9, full
disk). Visual aligning at 1e5 steps with two ResNet-18 encoders is *more* exposed to the
24 h wall than Gen3v6's state-only runs, so the trap was more likely here, not less.

**Still not mirrored:** Gen3v7's `--auto-resume` CLI (`find_latest_checkpoint()`,
`training_already_complete()`, `resolve_resume()`) is not in
`mix_visual_aligning_test/train_mix_visual_aligning.py`. Not needed for the above — Gen14's
train script already has `--resume-step` and calls `trainer.load()`, so
`_restore_optimizer_state()` fires automatically. The CLI would only remove the need to pass
a step number by hand on a requeue. Deferred.

## 7. Note recorded while answering: FiLM default

Not a change — logged because it was asked and is easy to get wrong later.

**All four Gen14 arms default to `film_mode: 'v1'`.** No Gen14 block sets it; `ddpm` inherits
from `visual_aligning_dpcc` (`config/aligning-d3il-visual.py:377`), `fm`/`mf`/`af` from
`fm_visual_aligning` (`:461`). It is a path key, so checkpoints land under `..._filmv1/`.

`mf` and `af` **cannot** run `v2`: `visual_unet_twotime.py:106` raises, because the true-FiLM
backbone has no `h_mlp` and so cannot carry the two-time signal. v1 is not merely their
default, it is their only option — which is also what keeps the four-arm comparison
architecturally controlled.
