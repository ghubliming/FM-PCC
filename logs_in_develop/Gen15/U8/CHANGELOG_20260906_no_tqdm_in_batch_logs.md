# Gen15 U8 — kill tqdm progress bars in batch logs

**Filed under Gen15/U8** (next epoch after U7; the fix also touches Gen14 V_A and avoiding —
say the word if it belongs elsewhere).
**Found in:** af train log 25440, surfaced in
[`DA_20260906_U7_honest_geometry_first_results.md`](../DA/DA_20260906_U7_honest_geometry_first_results.md) §8.6.

## 0. TL;DR

Training job logs carried a rendered tqdm bar **twice per epoch**. Fixed in the 3 active `mix_*`
generations by disabling the bar when stderr is not a TTY and printing the same metrics as one
plain greppable line instead. 17 further inherited copies left untouched — listed in §4.

## 1. Why `mininterval=1e10` did not already do this

```python
progress_bar = tqdm(total=n_train_steps, mininterval=1e10)   # the old line
```

`mininterval` throttles only the **intermediate** redraws. tqdm still renders:

1. once at **construction** — `0%|          | 0/1000 [00:00<?, ?it/s]`, and
2. once at **close** — `100%|██████████| 1000/1000 [01:24<00:00, 11.79it/s, a0_loss=…]`.

Nothing in these trainers ever called `close()`, so (2) happened via `__del__`. `train_epoch` is
called once per epoch, so a 100-epoch run wrote ~200 carriage-return bar frames into the sbatch
log. That bloats the file and breaks `grep` on the training curve, because the metrics only ever
reached stdout as a tqdm **postfix**, glued onto a bar frame.

## 2. The fix

```python
_TQDM_OFF = os.environ.get('FMPCC_TQDM', '') != '1' and not sys.stderr.isatty()
```

* `tqdm(..., disable=_TQDM_OFF)` — when disabled tqdm writes nothing at construction, on
  `update`, on `set_postfix` or on close, so all four render paths are closed at once. No
  `close()` call is needed and none was added.
* At the existing `log_freq` gate, when the bar is off, emit the same `logs` dict as one line:

  ```
  [ train ] epoch 3 step 4000/100000  loss=0.95215  alpha=0.2  discrete_frac=0.5  lr=0.0001  step=3999
  ```

* Interactive runs are unchanged (TTY ⇒ bar as before). `FMPCC_TQDM=1` forces the bar back on.

Cadence is identical to the old postfix updates, so log volume drops by roughly the bar-frame
overhead while the curve stays fully readable — and now greppable, which it never was.

## 3. Files changed (6)

| file | generation |
|---|---|
| `mix_uav/utils/training.py` | Gen15 UAV — `fm`, `diffusion` |
| `mix_uav/utils/training_twotime.py` | Gen15 UAV — `mf`, `af` ← **where it was observed** |
| `mix_visual_aligning/utils/training.py` | Gen14 V_A |
| `mix_visual_aligning/utils/training_twotime.py` | Gen14 V_A |
| `mix_visual_avoiding/utils/training.py` | avoiding |
| `mix_visual_avoiding/utils/training_twotime.py` | avoiding |

Gen14 was included because jobs 25416/25417 are in the queue now and would otherwise keep
producing the same logs.

## 4. Not changed — 17 inherited copies

The identical line exists in 17 further trainers, all descended from the DPCC upstream by
copy-modify: `diffuser/`, `flow_matcher{,_v2,_v3,_unet_v2}/`, `flow_matcher_v3_{uav,meanflow,
drifting,hardflow,ode_selectable,imeanflow,alphaflow}/`, `fm_visual_{aligning,avoiding}/`,
`diffuser_visual_{aligning,avoiding}/`, `imf_visual_aligning/`, plus
`d3il/agents/models/bet/libraries/mingpt/trainer.py` (vendored — do not edit).

Left alone deliberately: those generations are not currently producing batch logs, and the repo
convention is per-generation isolation rather than a shared refactor. Mirror on demand — the
patch is three anchors and applies unchanged.

## 5. Verification

* All 6 files pass `python3 -m py_compile`.
* `sys.stderr.isatty()` is `False` under a pipe and under sbatch ⇒ `_TQDM_OFF = True` ⇒ no bar.
* 🔴 **Not yet run on the cluster.** The next `train_mix_uav.sh` job is the real check: the log
  should contain **zero** `it/s` occurrences and one `[ train ] epoch … step …` line per
  `log_freq`.

  ```bash
  grep -c "it/s" <new train log>          # expect 0
  grep -c "^\[ train \] epoch" <new log>  # expect n_train_steps / log_freq
  ```
