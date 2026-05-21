# KEY_fix_6 — Gen7 Multi-Variant State Contamination Fix

**Date**: 2026-05-21  
**Branch**: update_into_FM  
**Audit source**: `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/KEY_fix_6/BUG_REPORT.md`  
**Scope**: Fix the "frozen problem" — all DPCC variants producing identical results when `diffuser` runs first in a multi-variant eval job.

| File | Fix |
|---|---|
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | AUDIT-FIX-1: expert gen before loop; AUDIT-FIX-3: per-variant save_path |
| `config/visual_aligning_eval.yaml` | AUDIT-FIX-2: re-enable `constraint_types` |

---

## AUDIT-FIX-1 — Expert Video Generation Moved Before Variant Loop

**Problem**: `generate_expert_reference()` was called inside the `for variant in projection_variants` loop, once per variant. In the first Slurm job (JOB 20627), expert videos did not yet exist, so the function actually ran — creating and destroying a `Robot_Push_Env` (MuJoCo). The `env.close()` does not fully release MuJoCo global factory state (body counter, OpenGL/camera context, `panda_tmp_rb*.xml` naming). The first variant's subsequent `Robot_Push_Env.__init__()` inherited this residual state, producing a different scene composition → bp_image std=0.1978 instead of the clean 0.2093. All subsequent variants in the same process shared the same contaminated env path.

The RNG reset at `aligning_sim.py:62-64` does NOT protect against this because scene construction happens at L58 (env creation), before the RNG reset at L62.

**Fix**: Moved `generate_expert_reference()` to BEFORE the `for variant` loop, using a computed `_base_results` path. Added `gc.collect()` and `torch.cuda.empty_cache()` immediately after to force Python/GPU memory cleanup before any variant env is created. The expert videos (in `results/expert_references/`) are still written to the same base results directory and will be skipped on re-runs (files already exist guard unchanged).

```python
# Before (inside loop — WRONG):
for variant in projection_variants:
    save_path = f'{args.savepath}/results'
    generate_expert_reference(save_path, n_rollouts=3)  # ran every iteration

# After (before loop — CORRECT):
_base_results = f'{args.savepath}/results'
os.makedirs(_base_results, exist_ok=True)
generate_expert_reference(_base_results, n_rollouts=3)   # once, isolated
gc.collect()
torch.cuda.empty_cache()

for variant in projection_variants:
    save_path = f'{args.savepath}/results/{variant}'     # per-variant (FIX-3)
```

---

## AUDIT-FIX-2 — Re-enable `constraint_types` in YAML

**Problem**: `constraint_types: []` made `setup_dpcc_projector()` return a projector with an empty constraint list. The SLSQP solver with no constraints is a no-op — it returns the input unchanged. As a result, `post_processing` and `model_free` were structurally identical to raw FM (`diffuser`), making all three variants compute the same 6 trajectories (same RNG seed, batch_size=6) and select index 0 (all projection costs are zero). The evaluation was measuring nothing beyond raw FM diffusion.

**Fix**: Changed `constraint_types: []` to `constraint_types: ['bounds', 'dynamics']` in `config/visual_aligning_eval.yaml`. This enables:
- `bounds`: workspace bounds enforcement on c_pos dims [6,7,8] — active for both `post_processing` and `model_free`
- `dynamics`: derivative continuity constraints — active for `post_processing` only (the `'model_free' not in variant` guard at `eval_fm_visual_aligning.py:104` excludes it from model_free)

After this fix, `post_processing` ≠ `model_free` ≠ raw FM. Expect `diffuser ≠ pp ≠ mf` in final distances. If pp ≈ mf on short horizons (H=8), that is acceptable — dynamics constraints have minimal impact at short horizon.

```yaml
# Before:
constraint_types: []

# After:
constraint_types: ['bounds', 'dynamics']
```

---

## AUDIT-FIX-3 — Per-Variant Output Paths

**Problem**: All variants wrote their `.npz`, `.pkl`, `.log`, and `diag_first_replan.txt` files to the same `results/` directory. On a re-run (or in a multi-variant job), later variants silently overwrote earlier variants' files — no cross-variant comparison was possible from saved artifacts.

**Fix**: `save_path` inside the variant loop now includes the variant name:

```python
# Before:
save_path = f'{args.savepath}/results'         # shared by all variants

# After:
save_path = f'{args.savepath}/results/{variant}'  # unique per variant
```

All downstream paths (`{save_path}/{variant}.npz`, `eval_{variant}.log`, `results_seed_{seed}.pkl`, `diag_first_replan.txt`) are now variant-isolated. Expert reference videos remain at the shared `_base_results/expert_references/` path (generated before the loop).

**New output structure**:
```
results/<seed>/
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

## Not Fixed

**FIX-1 (original, per-variant RNG reset)**: Already implemented in `d3il/simulation/aligning_sim.py:62-64`. No additional code needed.

**FIX-2 (original, deep-copy image buffer)**: Not needed — each variant creates a new `VisualAgentWrapper` and `Aligning_Sim` with fresh state. No buffer is shared in memory between variants.

**FIX-5 (original, DPCC guard investigation)**: Not a bug. The `'diffuser' not in variant` guard at `eval_fm_visual_aligning.py:867` is intentional — `diffuser` runs raw FM without DPCC projection by design.

**FIX-6 (original, result-hash assertion)**: Deferred. With `constraint_types: []` (now fixed), pp and mf would legitimately produce identical results, causing false positives. Only useful once constraints are verified to produce diverse outputs.
