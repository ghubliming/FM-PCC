# DA_Code v3 — U8: exact value labels above every bar (HTML Visualizer)

**Scope:** `Data_Analysis/Visualizer/index.html` only. No Python DA / no cluster re-run.
**Depends on:** U7 (hardflow variants + the two HTML render fixes §9/§10).

## What changed

Every bar in the dynamic plot now prints its **exact numeric value directly above the
column**. Previously values had to be read off the y-axis by eye, which is unusable for
the close comparisons this tool exists for (e.g. hardflow_new 0.497 vs dpcc-c-tightened
0.637 s).

### Edit (`trigger_plot`, right after the `pivot_mean.plot(kind='bar', ...)` call)

```python
from matplotlib.container import BarContainer
for container in ax.containers:
    if not isinstance(container, BarContainer):
        continue
    labels = [f"{v:.3g}" if v == v else "" for v in container.datavalues]
    ax.bar_label(container, labels=labels, padding=3, fontsize=7, rotation=90)
ax.margins(y=0.15)
```

## Design notes

- **`ax.bar_label` per container** — the grouped bar plot creates one `BarContainer` per
  variant (series). Labelling each container places the number above every individual bar,
  aligned to its column.
- **Skip the ErrorbarContainer** — when `yerr=pivot_std` is passed, matplotlib also appends
  an `ErrorbarContainer` to `ax.containers`; `bar_label` would raise on it. The
  `isinstance(..., BarContainer)` guard labels only the real bars. Labels use `padding=3`
  so they sit just above the error-bar cap when std is shown.
- **NaN → blank** — absent variant/candidate combos (a variant a candidate doesn't have,
  or a hardflow-only metric on a DPCC bar) have `value == NaN`; `v == v` is `False` for NaN,
  so those bars get an empty label instead of a literal `"nan"`. (Ties into U7 §9/§10:
  missing facets now degrade quietly everywhere.)
- **`:.3g`** — 3 significant figures: `1.0→"1"`, `0.497→"0.497"`, `0.0→"0"`,
  `0.00123→"0.00123"`. Compact and precise across the metric ranges (percentages, ms,
  NLP counts).
- **`rotation=90`** — vertical labels avoid overlap in dense grouped bars (many
  variants × candidates).
- **`ax.margins(y=0.15)`** — adds top headroom so rotated labels stay on-canvas instead of
  clipping at the axes edge.

## Verify

- Reload `index.html`, re-SYNC a batch, pick any metric/variants/candidates → each bar
  shows its value on top.
- Static: the label block compiles; formatting checked (`1.234567→"1.23"`, `nan→""`,
  `0.0→"0"`).

No cluster re-run — reload the page and re-SYNC.

---

## Fix (same update) — `bar_label` IndexError on non-existent variant bars

**Symptom:** selecting hardflow variants a candidate doesn't have threw a full-screen
traceback:
```
File ".../matplotlib/axes/_axes.py", line 2641, in bar_label
    endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
```

**Cause:** `ax.bar_label` auto-couples to the `ErrorbarContainer` that `yerr=pivot_std`
adds. When a selected variant/candidate combo doesn't exist its bar height is NaN and its
per-bar error entry is degenerate (1-D), so `bar_label`'s internal `err[:, 1]` indexing
crashes. Blanking the label text (`labels=[... if v==v else ""]`) was not enough — the
crash happens while bar_label computes the label *position* from the error array, before
the text matters.

**Fix:** dropped `ax.bar_label` entirely; annotate bar tops manually with `ax.annotate`,
which reads only `rect.get_height()` and never touches the error data. NaN bars are skipped
(`if h != h: continue`), so absent combos silently get no label:
```python
from matplotlib.container import BarContainer
for container in ax.containers:
    if not isinstance(container, BarContainer):
        continue
    for rect in container:
        h = rect.get_height()
        if h != h:  # NaN -> absent combo, no label
            continue
        ax.annotate(f"{h:.3g}", xy=(rect.get_x() + rect.get_width() / 2.0, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=90)
ax.margins(y=0.15)
```
Same visual result (value above each bar), but robust to NaN/std combinations — no more
crash when selecting variants a candidate lacks. Reload `index.html` and re-SYNC.
