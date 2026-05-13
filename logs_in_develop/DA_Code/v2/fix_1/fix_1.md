# DA Tool v2 Fix 1: Batch Output Directory Return Mismatch

This document records the fix applied to the DA v2 batch analysis pipeline after the multi-candidate SLURM job failed during output directory setup.

## Failure Summary

The batch analysis job aborted with:

```text
Traceback (most recent call last):
  File "/data/home/llim/FMPCC/FM-PCC/Data_Analysis/DA_Code/main_da_batch.py", line 271, in <module>
    sys.exit(main())
  File "/data/home/llim/FMPCC/FM-PCC/Data_Analysis/DA_Code/main_da_batch.py", line 118, in main
    output_dir, output_timestamp = create_output_directory(output_base, 'FM_V3_BATCH')
ValueError: too many values to unpack (expected 2)
```

## Root Cause

`main_da_batch.py` expected `create_output_directory(...)` to return two values:

```python
output_dir, output_timestamp = create_output_directory(...)
```

However, the shared helper in `Data_Analysis/DA_Code/utils.py` only returned a single value:

```python
return output_dir
```

That meant the batch CLI and the helper were out of sync.

## Fix Applied

The helper was extended with an optional timestamp return mode so it remains backward compatible with the single-candidate CLI while supporting the batch CLI.

### 1. Utility helper update

File: `Data_Analysis/DA_Code/utils.py`

```diff
-def create_output_directory(base_path, prefix='FM_V3_ODE_Analysis'):
+def create_output_directory(base_path, prefix='FM_V3_ODE_Analysis', return_timestamp=False):
...
-    return output_dir
+    if return_timestamp:
+        return output_dir, timestamp
+
+    return output_dir
```

### 2. Batch CLI update

File: `Data_Analysis/DA_Code/main_da_batch.py`

```diff
-    output_dir, output_timestamp = create_output_directory(output_base, 'FM_V3_BATCH')
+    output_dir, output_timestamp = create_output_directory(
+        output_base,
+        'FM_V3_BATCH',
+        return_timestamp=True,
+    )
```

## Result

The DA v2 batch analysis job can now create its timestamped output folder correctly and proceed to discovery, loading, aggregation, visualization, and reporting.

## Compatibility Note

This fix keeps the helper backward compatible:

- `main_da.py` still receives only the output directory
- `main_da_batch.py` explicitly opts in to the `(output_dir, timestamp)` return value

That avoids breaking the existing single-candidate DA flow while restoring the batch workflow.