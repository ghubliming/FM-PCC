# Fix 2 — Progress Bar Exhaustive / Console Pollution (2026-05-19)

## Problem

When running training on a non-interactive or non-live console (such as a cluster, Slurm log, or redirected file), the `tqdm` progress bars update extremely frequently (defaulting to every 0.1 seconds). Because a non-TTY environment outputs a new line for each progress bar update, this fills the log files with thousands of exhaustive lines (specifically during the "Loading images" phase and at the end of each epoch), resulting in massive console pollution.

---

## Root Cause

**Dataset Loading**: `diffuser_visual_aligning/datasets/sequence.py` utilized `tqdm` for showing progress when loading states and images. Since loading images takes around 0.1–0.2 seconds per iteration, and no `mininterval` was set, it printed a new line for nearly every step in non-TTY logs.

---

## Fix

We resolved this by setting a standard, larger `mininterval` parameter in the dataset loading tqdm loops to throttle progress bar prints.

### Progress Bar Throttle
In `diffuser_visual_aligning/datasets/sequence.py`, we set `mininterval=10.0` directly inside the tqdm progress bar initialization. This reduces print frequency to at most once every 10 seconds, which cleanly avoids console logging pollution in non-interactive/non-live consoles.

```python
for file in tqdm(state_files[:n_eps], desc='Loading states', mininterval=10.0):
    ...

for file in tqdm(state_files[:n_eps], desc='Loading images', mininterval=10.0):
    ...
```

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning/datasets/sequence.py` | Set progress bar `mininterval=10.0` to throttle output in non-live consoles during loading stages. |
