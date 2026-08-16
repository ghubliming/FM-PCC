# U_3 — a NaN is not a zero: both viewer charts now say "n/a"

**Date:** 2026-08-07
**Changed:** `Data_Analysis/Visualizer/index.html` (DAv3, marked **U12** in-file) and
`Data_Analysis/Visualizer_VA_v2/index.html` (regenerated from it)
**New:** `Data_Analysis/Visualizer_VA_v2/test_nan_not_zero.py`

> ⚠ **This edits DAv3's page.** Every prior epoch held the "don't touch the old viewers"
> line; this one crosses it on explicit instruction, because the bug is *in* DAv3's plotting
> code and VA v2 inherits that code verbatim. Fixing it only in the derivative would have
> left the avoiding-side tool lying. The two in-file epoch markers differ on purpose:
> **U12** continues DAv3's own U7…U11 sequence, **U_3** is this log folder's.

---

## 1. The bug

`pandas.DataFrame.plot(kind='bar')` runs **`fillna(0)`** on the frame before handing it to
matplotlib. So a candidate/variant combination that was never run reached the canvas as a
**real 0-height bar**, drawn at exactly the same place as a method that genuinely scored
`0.000`. The chart could not distinguish *"this method failed every rollout"* from *"this
was never evaluated"*.

Measured on a real batch before the fix (out1, env `both-hard`, 3 candidates × 6 variants):

```
bars=18   n/a marks=0   real-zero-height bars=16      <- 16 of 18 cells had NO data
```

The existing U8 comment (`if h != h: # NaN -> absent combo, no label`) shows the intent was
already there — it just never fired, because by the time the code reads `rect.get_height()`
pandas has already turned every NaN into `0.0`.

A second, quieter half: `cols = [v for v in checked_vars if v in pivot_mean.columns]` dropped
a selected variant that produced no rows **at all**, so it vanished from the plot *and* from
the legend — indistinguishable from "not ticked".

## 2. The fix

Three changes inside `trigger_plot`, in both pages:

1. **Reindex instead of filter.** `pivot_mean.reindex(columns=checked_vars)` keeps a
   no-data variant as an all-NaN column, so it still gets an x-slot and a legend entry. In
   candidate mode the index is reindexed to `checked_cands` too, so a candidate that never
   ran in the shown environment keeps its group on the axis instead of disappearing.
   *(Environment mode is left alone — there the x-axis is whatever environments exist, and
   there is no user-supplied list to reindex against.)*
2. **Read the NaN mask from the frame, not the rects.** `pivot_mean` still holds the truth
   after pandas has zero-filled the drawing. One `BarContainer` per column, one rect per
   index row, both in frame order, so `pivot_mean.iat[ri, ci]` pairs back exactly. If that
   pairing ever stops holding (a pandas/matplotlib change), the code falls back to the rect
   height and marks nothing rather than mislabelling a real value.
3. **Mark it.** A missing cell gets `rect.set_height(nan)` — so nothing is drawn — plus a
   red rotated **`n/a`** at the baseline, and the axes gain a red footnote:
   *"16 of 18 bars have NO DATA (marked n/a) - an absent bar is not a zero"*. The footnote
   appears only when something is actually missing.

`set_height(nan)` rather than `set_visible(False)` on purpose: an invisible patch makes
matplotlib draw an **empty swatch** for that variant in the legend, which silently destroys
the colour key. A NaN height simply isn't rendered and the legend survives intact.

### What it looks like now

| case | before | after |
|---|---|---|
| never run | bar of height 0, no label | no bar, red `n/a`, counted in the footnote |
| genuinely 0.000 | bar of height 0, label `0` | unchanged — bar of height 0, label `0` |
| variant with no rows at all | column dropped, gone from the legend | all-`n/a` column, legend entry kept |
| candidate absent from this env | x-group dropped | all-`n/a` x-group |

## 3. Validation

`test_nan_not_zero.py` execs **both** pages' Python blocks against stubbed
`document`/`window`, feeds each the CSV shape it actually reads (DAv3 →
`candidates_multidimensional_aggregated.csv`, VA v2 → the native long CSVs), calls the real
`trigger_plot`, and inspects the resulting matplotlib figure against ground truth
recomputed from the frame. 5 checks × 2 pages, on two real batches:

```
out1  (16 of 18 cells missing)   DAv3 + VAv2   ALL CHECKS PASSED
out5  (nothing missing, 1-2 genuine zeros)     ALL CHECKS PASSED
```

* every selected cell gets an x-slot — nothing silently dropped
* the number of `n/a` marks equals the number of NaN/absent cells, exactly
* those cells draw **no** bar (pandas' `fillna(0)` undone)
* a genuine `0.000` keeps its bar and its `0` label and is **not** marked
* the footnote appears if and only if something is missing

Rendered PNGs were also inspected directly: `n/a` cells blank with red marks, the legend
keeping all six colour swatches, and on the second batch two variants at a true `0` still
showing their `0` labels with no `n/a` anywhere.

`Visualizer_VA_v2/test_page_offline.py` still passes on both batches (47 checks) — all 33
build anchors held, so the U3 matrices, run-coverage table and INIT XY row are unaffected.

## 4. Regenerating

`Visualizer_VA_v2/index.html` is generated. The fix was made in DAv3's file and picked up by:

```bash
python Data_Analysis/Visualizer_VA_v2/build_from_dav3.py
python Data_Analysis/Visualizer_VA_v2/test_nan_not_zero.py <batch_va2_dir>
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py <batch_va2_dir>
```

DAv3's file is CRLF; the patch preserved that, and the builder normalises on read.

## 5. Known limit — the zeros this cannot catch

This makes the viewer honest about NaN. It cannot rescue a value that was **already zeroed
upstream**: the eval writes defaults for constraint fields it never computed (e.g.
`_cm('exec_constraint_sat_rate', 1.0)`, several `0.0` defaults), and `DA_Visual_Aligning`'s
older loader used `_arr('mean_distance', 0.0)`. Those arrive as genuine numbers and no
viewer can tell them from measurements. Auditing those defaults is a separate job — see
`../U_2/METRICS_REFERENCE.md` §7.
