# DA Tool v2 Fix 2: Recursive Candidate Discovery for Nested Result Trees

This document records the fix applied after the DA v2 batch analysis job reported that no candidates were found, even though valid evaluation logs existed deep inside the results tree.

## Failure Summary

The batch run exited with:

```text
[2026-05-13 10:40:33] DA_Batch_v2 - INFO - [PHASE 1/5] AUTO-DISCOVERING CANDIDATES
[2026-05-13 10:40:33] DA_Batch_v2 - INFO - ----------------------------------------------------------------------
No candidates found in logs/avoiding-d3il/plans
[2026-05-13 10:40:33] DA_Batch_v2 - ERROR - No candidates found. Exiting.
```

The user confirmed that valid files do exist, for example:

```text
FMPCC/FM-PCC/logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/
  H8_Dmodels.diffusion.GaussianDiffusion_a1.5_b1.0_aw10/
    H8_K10_Meuler_Dmodels.diffusion.GaussianDiffusion/
      6/results/halfspace_both-hard/eval_dpcc-c-tightened-dt0p5.log
```

## Root Cause

The original discovery logic in `Data_Analysis/DA_Code/multi_candidate_discovery.py` only scanned the immediate children of the parent directory and required a candidate folder to contain seed directories directly beneath it.

That worked for shallow layouts, but it failed for the actual DA result structure, where candidates are nested several levels below the parent path.

In other words, the code was looking for:

```text
parent_path/<candidate>/<seed>/results/...
```

but the real structure is more like:

```text
parent_path/<method>/<configuration>/<candidate>/<seed>/results/...
```

## Fix Applied

The DA batch CLI was switched to use the recursive discovery helper.

### Code Delta

File: `Data_Analysis/DA_Code/main_da_batch.py`

```diff
-from multi_candidate_discovery import discover_candidates, filter_candidates, assign_custom_names, get_candidate_summary
+from multi_candidate_discovery import discover_candidates_recursive, filter_candidates, assign_custom_names, get_candidate_summary
...
-        candidates = discover_candidates(args.parent_path)
+        candidates = discover_candidates_recursive(args.parent_path, max_depth=10)
```

## Result

The batch analysis now searches through nested result trees and can discover candidates that are not direct children of the parent directory.

This is the correct fix to retain because the data layout is genuinely nested, not flat.

## Notes

- The `libtinfo.so.6` shell warning is unrelated.
- The batch loader itself was not the problem; it was never reached because discovery returned no candidates.