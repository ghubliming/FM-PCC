# Checkpoint-epoch audit — `latest` → `best` repo-wide

**Date:** 2026-08-20
**Trigger:** all 5 `mf` jobs of the Gen15 corridor K sweep crashed at launch (jobs 24708–24712)
**Severity:** 🟠 **every UAV evaluation ever run — Gen11 and Gen15 — used a checkpoint from 80 % of training.** Consistent across arms, so relative comparisons survive; absolute numbers are understated and UAV-vs-D3IL comparisons are invalid.
**Status:** code fixed repo-wide (40 files). Evals NOT yet re-run.
**Related:** [`../Gen15/DA/DA_20260820_fm_K_sweep_corridor.md`](../Gen15/DA/DA_20260820_fm_K_sweep_corridor.md)

---

## 1. What happened

The K sweep submitted 10 jobs on 2026-08-19: `fm` and `mf`, corridor, K ∈ {1,2,5,10,20}. All five `mf` jobs died within ~20 s:

```
FileNotFoundError: [Errno 2] No such file or directory:
'logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/6/state_-1.pt'
```

Tracing `state_-1.pt` back:

1. `--epoch` defaulted to `'latest'` (`eval_mix_uav.py:173`), and no sbatch script overrode it.
2. `'latest'` routed to `get_latest_epoch()` (`serialization.py:99`).
3. That function globs `state_*` and does `int(name.replace('state_','').replace('.pt',''))`. On **`state_best.pt` it raises `ValueError`, catches it, and scores that file as epoch `-1`.**
4. The mf directory contains *only* `state_best.pt`, so `latest_epoch` never rose above its `-1` initialiser.
5. `trainer.load(-1)` → `state_-1.pt` → crash, three frames from the real cause.

The mf checkpoint directory, confirmed on the cluster:

```
-rw-r--r-- 63954430 Aug 15 07:02 state_best.pt      ← the only checkpoint
```

## 2. The bigger finding: `latest` was never the final model

While diagnosing, the `fm` directory told a different and more serious story:

```
-rw-r--r-- 31815726 Aug 17 23:00 state_0.pt
-rw-r--r-- 31819526 Aug 17 23:33 state_20000.pt
-rw-r--r-- 31821126 Aug 18 00:06 state_40000.pt
-rw-r--r-- 31822726 Aug 18 00:38 state_60000.pt
-rw-r--r-- 31824390 Aug 18 01:11 state_80000.pt    ← what `latest` resolves to
-rw-r--r-- 31825312 Aug 18 01:33 state_best.pt     ← saved 22 min LATER
-rw-r--r--    29135 Aug 18 01:41 losses.json       ← training ended here
```

- `n_train_steps = 100000` (`config/uav.py:161`, `config/uav_mix.py:155`).
- `save_freq = n_train_steps // 5` = **20000** (`training_twotime.py:81`).
- The loop runs `self.step` over `0 … 99999`, so `step % 20000 == 0` fires at 0/20000/40000/60000/80000 and **step 100000 is never reached and never saved.**

Therefore `get_latest_epoch()` returns **80000, always**. Confirmed in the job log:

```
[ utils/training ] Restored loss history from checkpoint at step 80000
```

**"latest" does not mean "the final model". It means "the last of the five periodic saves", which is 20 000 steps — 20 % of training — before the end.** The final fifth of training survives in exactly one file: `state_best.pt`, which is both *later* than `state_80000.pt` and, by construction, the lowest validation loss.

### 2.1 Who was affected

| Location | Was | Lineage |
|---|---|---|
| `mix_uav_test/eval_mix_uav.py:173` | `--epoch` default `'latest'` | 🔴 Gen15 |
| `config/uav_mix.py:174` | `'diffusion_epoch': 'latest'` | 🔴 Gen15 |
| `FM_v3_uav_test/eval_fm_uav.py:129` | `--epoch` default `'latest'` | 🔴 Gen11 |
| `config/uav.py:203` | `'diffusion_epoch': 'latest'` | 🔴 Gen11 |
| `traj_gen_script_for_v4.py` ×3 | hardcoded `epoch="latest"` | 🟡 benchmark helpers |
| `config/avoiding-d3il.py` (12 blocks) | `'best'` | ✅ |
| `config/aligning-d3il-visual.py` (4) | `'best'` | ✅ |
| `config/avoiding-d3il-visual.py` (2) | `'best'` | ✅ |

**The entire D3IL half of the repo was already correct.** Only the UAV lineage used `latest`, and Gen15 inherited it from Gen11 by copy-modify. The two UAV eval scripts were line-for-line identical at both the argparse default and the `int(epoch)` cast.

### 2.2 Is the existing data wrong?

**Biased, not corrupted — and the bias is uniform, which is what saves it.**

- ✅ **Within-UAV comparisons hold.** `fm` Gen11, `fm` Gen15 and `mf` Gen15 all resolved to step 80000 (the mf K=10 run's `run_config` records `epoch=latest`, and its numbered checkpoints still existed at eval time). The K sweep, the fm-vs-mf tables and the Gen11 baselines are internally apples-to-apples. **Nothing published so far needs retracting on these grounds.**
- 🔴 **Absolute numbers are understated.** Every UAV success rate came from an 80 %-trained model.
- 🔴 **UAV-vs-D3IL comparison is invalid** — it silently pits 80 %-trained UAV models against best-checkpoint D3IL models.
- 🔴 **The forward risk was the real hazard.** `mf` now has no numbered checkpoint, so it *must* load `best`. Had we fixed only `mf`, we would have compared a best-checkpoint `mf` against an 80 %-trained `fm` — turning a harmless uniform bias into a fake result. That is why the fix is repo-wide rather than mf-only.

### 2.3 🔴 OPEN — unrelated, possibly larger: the two arms may differ in size

`_checkpoint_payload()` always stores `model + ema + optimizer(2 Adam moments)` = 4 float32 copies = **16 bytes per parameter**. That constant is calibrated by the mf eval log, which prints `params=4.0M`:

| arm | `state_best.pt` bytes | ÷ 16 | implied params |
|---|---:|---:|---|
| mf | 63,954,430 | 3,997,152 | **4.0 M** ✓ matches the logged value |
| fm | 31,825,312 | 1,989,082 | **~2.0 M** |

If this holds, **the `fm` arm has roughly half the parameters of the `mf` arm**, and every `fm`-vs-`mf` number in Gen15 is confounded on capacity — against the arm currently winning. This is NOT addressed by this changelog. Confirm with:

```bash
python -c "
import torch
for a,p in [('fm','logs/UAV_MIX/uav-corridor/mix_uav_fm/H8_Dmodels.diffusion.FlowMatchingODE_9D/6'),
            ('mf','logs/UAV_MIX/uav-corridor/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/6')]:
    d=torch.load(f'{p}/state_best.pt',map_location='cpu')
    print(a,'step',d['step'],'params',sum(v.numel() for v in d['model'].values()))"
```

---

## 3. What was changed

40 files, +136 / −41. Three tiers:

### 3.1 Behavioural — the actual fix (6 edits)

| File | Change |
|---|---|
| `config/uav.py:203` | `'diffusion_epoch': 'latest'` → `'best'` |
| `config/uav_mix.py:174` | `'diffusion_epoch': 'latest'` → `'best'` |
| `mix_uav_test/eval_mix_uav.py:173` | `--epoch` default `'latest'` → `'best'`, help text rewritten to spell out that `latest` = step 80000 of 100000 |
| `mix_uav_test/eval_mix_uav.py:193` | `ep = epoch if epoch == 'latest' else int(epoch)` → `epoch in ('latest', 'best')` |
| `FM_v3_uav_test/eval_fm_uav.py:129` | same default change |
| `FM_v3_uav_test/eval_fm_uav.py:153` | same cast guard |

Without the cast guard, `--epoch best` would have died on `int('best')` — the UAV frame previously had **no way to load `state_best.pt` at all**.

### 3.2 Diagnostic — fail loudly instead of three frames later

`mix_uav/utils/serialization.py` and `flow_matcher_v3_uav/utils/serialization.py`, at the `epoch == 'latest'` branch:

```python
if epoch == 'latest':
    epoch = get_latest_epoch(loadpath)
    if epoch < 0:
        _dir = os.path.join(*loadpath)
        raise FileNotFoundError(
            f'no numbered state_<N>.pt checkpoint in {_dir} '
            f'(found: {sorted(glob.glob1(_dir, "state_*"))}). '
            f'Pass epoch="best" to load state_best.pt.')
```

### 3.3 Defence in depth — unreachable defaults closed

`epoch='latest'` → `epoch='best'` in **19 `load_diffusion` / `load_diffusion_with_override` signatures** across all live generations, plus the 3 hardcoded `epoch="latest"` in `traj_gen_script_for_v4.py`.

Verified **no behavioural effect**: every live call site passes `epoch=` explicitly (`epoch=args.diffusion_epoch`, which now resolves to `'best'` in all 20 config blocks). These were dead fallbacks; they are closed so a future caller that omits the argument cannot silently reintroduce the bug.

`Archived_Codes/` and the `*(legacy/Abandoned/Outdated)` folders were **deliberately skipped** (11 files) — dead code.

### 3.4 Verification run

- `grep` for `epoch='latest'` / `epoch="latest"` / `default='latest'` / `'diffusion_epoch': 'latest'` across live code → **zero hits**.
- All 20 `diffusion_epoch` entries in `config/` now read `best`.
- All 40 modified files pass `ast.parse`.
- Not executed locally (no Python env in the container) — **needs a smoke run on the cluster.**

---

## 4. Next: re-run the mf eval

The mf arm can now load `state_best.pt`. No extra flag is needed — `best` is the default:

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf corridor "6" "1 2 5 10 20" 10 fm_only none
```

The first job should print `Restored loss history from checkpoint at step <N>` with N > 80000 instead of crashing. **If it crashes again, stop and read the new error** — the guard in §3.2 now names the directory and lists what it actually found.

### Then, for a matched comparison

The `fm` sweep on disk ran on step 80000. Once `mf` lands on `best`, the two arms no longer share a checkpoint rule, so `fm` has to be re-run too:

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm corridor "6" "1 2 5 10 20" 10 fm_only none
```

Consider narrowing both to **K ∈ {1,2,3,5}**: K=10 and K=20 are already known-saturated for `fm` (identical S&C, tracking error and steps-to-goal at both), the decision lives at the low end, and K=20 hit the 24 h wall clock cap last time.

### Carry-over cautions

- The existing `mf` K=10 result came from a numbered checkpoint that **no longer exists** — it is not reproducible. Discard it rather than reusing it as the K=10 point.
- The `mf` corridor data also predates Gen15 Fix_1 (`0da86dc6`), so its `proj_ms` split is invalid (totals are fine). Re-running fixes this too.
- §2.3 (parameter-count asymmetry) is **unresolved** and gates any fm-vs-mf claim.
