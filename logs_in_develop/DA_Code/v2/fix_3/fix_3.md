# DA Tool v2 Fix 3: Accept .log and nested variant file names

This document records the change to the batch data loader that allows the DA v2 tool to find evaluation artifacts named as `.log` files (and other nested names) instead of strictly `.npz` files.

## Failure Summary

The batch loader previously expected files at a precise path and filename (`{halfspace}/{variant}.npz`). Real results sometimes use different naming (for example `eval_dpcc-c-tightened-dt0p5.log`) and can be stored anywhere under the `results` folder tree.

## Fix Applied

- `Data_Analysis/DA_Code/data_loader.py` was updated so that for each `variant` the loader looks for any filename under the `halfspace` folder that contains the `variant` substring.
- It now supports both `.npz` (parsed into numeric metrics) and `.log` (stored as `raw_log` text) result files.

This makes discovery robust to nested result layouts and to file naming variations.

## Result

After this change the example file:

```
.../6/results/halfspace_both-hard/eval_dpcc-c-tightened-dt0p5.log
```

will be found when scanning for the `dpcc-c-tightened-dt0p5` variant, and its raw contents will be available in the batch loading output.

## Notes

- The loader still assigns candidate letters (A, B, C...) to top-level matched candidate folders found by the recursive discovery; inner nesting does not change assignment.
- Future enhancement: add optional parsing of common log formats to extract summary metrics from `.log` files.