# Gen7 KEY_fix_6 Applied to Gen6V4

**Date**: 2026-05-21  
**Branch**: update_into_FM  
**Source fix**: `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/KEY_fix_6/CHANGELOG.md`  
**Scope**: The same three fixes from Gen7's KEY_fix_6 audit applied identically to the Gen6V4 eval script (`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`). The Gen7 and Gen6V4 eval scripts share the same variant-loop architecture and are affected by the same bugs.

| File | Fix |
|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | AUDIT-FIX-1: expert gen before loop; AUDIT-FIX-3: per-variant save_path |
| `config/visual_aligning_eval.yaml` | AUDIT-FIX-2: re-enable `constraint_types` (shared config — same file as Gen7) |

---

## Why Gen6V4 Needed the Same Fix

`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` was the origin file from which Gen7's `fm_visual_aligning_test/eval_fm_visual_aligning.py` was copy-modified. Both contain the identical variant loop structure:

```python
# Old (both files):
for variant in projection_variants:
    save_path = f'{args.savepath}/results'
    generate_expert_reference(save_path, n_rollouts=3)  # BUG: inside loop
```

The same MuJoCo global state contamination (AUDIT-FIX-1) and shared output paths (AUDIT-FIX-3) are present in Gen6V4. Any Gen6V4 multi-variant Slurm job that generates expert videos for the first time would exhibit the same "frozen problem".

---

## Changes Applied

### AUDIT-FIX-1 — Expert Video Generation Before Loop + GC Cleanup

**File**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Moved `generate_expert_reference()` to before the `for variant` loop. Added `gc.collect()` and `torch.cuda.empty_cache()` after it. Added `import gc` to file imports.

```python
# Added before the variant loop:
_base_results = (f'{args.savepath}/results_train_set'
                 if args_cli.eval_on_train else f'{args.savepath}/results')
os.makedirs(_base_results, exist_ok=True)
generate_expert_reference(_base_results, n_rollouts=3)
gc.collect()
torch.cuda.empty_cache()
```

### AUDIT-FIX-3 — Per-Variant save_path

**File**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

```python
# Before:
save_path = f'{args.savepath}/results'

# After:
save_path = f'{args.savepath}/results/{variant}'
```

### AUDIT-FIX-2 — constraint_types Re-enabled

**File**: `config/visual_aligning_eval.yaml` (shared between Gen6V4 and Gen7 eval).

```yaml
# Before:
constraint_types: []

# After:
constraint_types: ['bounds', 'dynamics']
```

---

## New Output Structure (Gen6V4)

```
logs/aligning-d3il-visual/visual_aligning_dpcc/<exp>/results/<seed>/
├── expert_references/expert_rollout_<r>.{mp4,gif}
└── <variant>/
    ├── <variant>.npz
    ├── <variant>.png
    ├── results_seed_<s>.pkl
    ├── eval_<variant>.log
    ├── diag_first_replan.txt
    ├── diagnostics/<variant>/rollout_<r>.*
    └── realtime_diagnostics/<variant>/rollout_<r>.*
```

---

## Full Root Cause Reference

See `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/KEY_fix_6/BUG_REPORT.md` for the complete forensic analysis, numerical evidence (4 Slurm jobs), auditor corrections, and developer response.

Summary of root causes fixed here:
1. **MuJoCo scene contamination** (AUDIT-FIX-1): Expert gen's `Robot_Push_Env` lifecycle leaves global factory state that contaminates the first variant's scene → bp_image std wrong → bad FM trajectory for all variants.
2. **Empty constraint list** (AUDIT-FIX-2): With `constraint_types: []`, pp and mf are no-ops over raw FM — all variants compute identical results even in a clean run. Evaluation was measuring nothing.
3. **Shared output paths** (AUDIT-FIX-3): All variants wrote to the same `results/` directory — later variants silently overwrote earlier variants' artifacts.
