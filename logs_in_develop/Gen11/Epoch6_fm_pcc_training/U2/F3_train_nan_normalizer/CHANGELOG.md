# Gen11 E6 F3 — UAV-FM train NaN (constant-feature normalizer 0/0)

Triaged from the 3 SLURM outputs in `temp/Gen11_E6_F3_FIX`.

## The 3 SLURM outputs — verdict

| Job | What | Verdict |
|-----|------|---------|
| 21895 | eval `s_curve` seed 6 | OLD argparse crash (`unrecognized arguments: --scene …`). **Already fixed** by the eval `sys.argv` strip (prior fix). No new action. |
| 21882 | train `empty` seed 6 | Trained cleanly to step 1e5, `loss≈0.0019`, no NaN. **Healthy** — confirms the pipeline is fine when no feature column is constant. |
| 21898 | train `pillars` seed 5 | **NaN from epoch 0** for every loss. This is the bug fixed here. |

## Root cause of the NaN

Job 21898 printed the smoking gun *before* training even started:

```
flow_matcher_v3_uav/datasets/normalization.py:159: RuntimeWarning: invalid value encountered in divide
  x = (x - self.mins) / (self.maxs - self.mins)
```

`config/uav.py` used `'normalizer': 'LimitsNormalizer'`, which normalizes each
feature as `(x - min) / (max - min)`. The `pillars` data has at least one
**constant feature column** (zero range, e.g. a fixed altitude / always-zero
velocity component), so for that column `max == min` → `0/0 = NaN`. The NaN
enters the normalized observations/actions and poisons the whole network →
all losses NaN from epoch 0.

`empty` (job 21882) has spread in every dimension, so no constant column, so no
NaN — which is exactly why it was "sometimes good, sometimes NaN" depending on
the scene.

## Fix

Two changes:

1. **`config/uav.py`**: `'normalizer': 'LimitsNormalizer'` →
   `'SafeLimitsNormalizer'`. The repo already ships `SafeLimitsNormalizer`
   (`flow_matcher_v3_uav/datasets/normalization.py`), written specifically for
   constant-dimension data. It widens a constant dim so it maps to the midpoint
   (0) instead of dividing by zero. When **no** dim is constant it is identical
   to `LimitsNormalizer`, so `empty/corridor/s_curve` are unaffected — this is a
   strict, safe superset.

2. **`flow_matcher_v3_uav/datasets/normalization.py`** — fixed a latent bug in
   `SafeLimitsNormalizer`: on finding a constant dim `i` it did
   `self.mins -= eps` / `self.maxs += eps` (the **whole** mins/maxs arrays),
   which corrupts the scale of *every other* dimension and compounds once per
   constant dim. Changed to index the offending dim only:
   ```python
   self.mins[i] -= eps
   self.maxs[i] += eps
   ```

## Verification (local numpy repro of the exact math)

Synthetic 12-D data with one constant column (mimicking `pillars`):

```
OLD LimitsNormalizer      -> any NaN? True   | NaN cols: [8]   (same RuntimeWarning)
NEW SafeLimitsNormalizer  -> any NaN? False
  constant col 8 normalized value: 0.0          (maps to midpoint, correct)
  non-constant col 0 range:        [-1.0, 1.0]  (other dims unchanged)
```

Both edited files compile (`py_compile` OK).

## Consistency / scope notes

- Eval uses the same normalizer (rebuilt from data via the same config), so
  train and eval stay consistent under `SafeLimitsNormalizer`.
- Already-trained models (e.g. `empty` from job 21882) were fine and are not
  invalidated; they just used the equivalent plain-Limits path. Scenes that
  previously NaN'd (`pillars`, possibly others with a constant column) must be
  **re-trained** to benefit.
- Working-tree only — sync to the cluster before re-submitting `pillars`.
