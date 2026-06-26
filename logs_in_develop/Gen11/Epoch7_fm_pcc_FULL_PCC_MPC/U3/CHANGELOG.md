# U3 — Restore full DPCC variant suite (gradient, post_processing, model_free, tightened)

**Date:** 2026-06-25

## What changed

### `FM_v3_uav_test/eval_fm_uav.py` — `setup_dpcc_projector`

Three variant flags were missing from our projector build, all ported verbatim from
`fm_visual_aligning_test/eval_fm_visual_aligning.py:148–162`:

| flag | before | after |
|------|--------|-------|
| `gradient` | always `False` | `'gradient' in variant` → `gradient=True`, `gradient_weights=[1,0.5,2]` |
| `post_processing` | always `threshold=0.5` | `threshold=0.0` (projects at ALL FM ODE steps, not just last 50%) |
| `tightened` | `enlarge=0.0` hardcoded | reads `enlarge_constraints` from config, applies to spatial constraints when `'tightened' in variant` |

`model_free` was already handled (line 147 skips dynamics when `'model_free' in variant`);
no code change needed for it.

### `config/uav_eval.yaml` — `projection_variants`

Restored full DPCC paper Table 1 variant suite:

```yaml
# before (4 variants):
projection_variants: ['diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t']

# after (13 variants):
projection_variants: [
  'diffuser',
  'gradient', 'gradient-tightened',
  'post_processing', 'post_processing-tightened',
  'model_free', 'model_free-tightened',
  'dpcc-r', 'dpcc-r-tightened',
  'dpcc-c', 'dpcc-c-tightened',
  'dpcc-t', 'dpcc-t-tightened',
]
```

Also added `enlarge_constraints: 0.025` (matches DPCC paper value).

## What each new variant will produce this epoch

| variant | spatial constraints | dynamics | expected result |
|---------|--------------------|---------|--------------—|
| `gradient` | — | ✓ | dynamics enforced via gradient (vs SLSQP) |
| `gradient-tightened` | empty | ✓ | same as gradient (tightening no-op without spatial) |
| `post_processing` | — | ✓ | dynamics enforced at ALL FM steps (vs last 50%) |
| `post_processing-tightened` | empty | ✓ | same as post_processing |
| `model_free` | empty | ✗ | **no-op = diffuser** (spatial constraints not yet designed) |
| `model_free-tightened` | empty | ✗ | **no-op = diffuser** |
| `dpcc-r/c/t-tightened` | empty | ✓ | same as dpcc-r/c/t (tightening no-op without spatial) |

`model_free` and `*-tightened` are scaffolded and wired — they will automatically
produce real results once per-scene spatial constraints (obstacle/halfspace/bounds)
are defined in a future epoch. No code change needed at that point, only yaml.

## DPCC paper Table 1 — dt ablation variants (NOT included)

`dpcc-c-tightened-dt0p25/dt0p5/dt2p0/dt4p0` (Table 2) require per-variant dt override
in `setup_dpcc_projector`. Deferred: not in Table 1, not needed for core comparison.

## Reference

- DPCC variant logic: `/workspaces/dpcc/scripts/eval.py:126–162`
- FMv3ODE port reference: `fm_visual_aligning_test/eval_fm_visual_aligning.py:148–162`
- Spatial constraint gap documented in: `../U1&2/WHY_PCC_WORKS_PILLARS.md` (FAQ section)
