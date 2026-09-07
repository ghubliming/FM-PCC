# Fix_11 — `encode_visual_cond` assumed a two-time engine, making arm C unreachable for `fm`

**Date:** 2026-09-01
**File touched:** `mix_visual_aligning/sampling/hardflow_projection.py` (one function, `encode_visual_cond`)
**Trigger:** job **25248** (flagship fm K=20 T=0.2) died at item 17/38 — the first `hardflow_new-r` cell.
**Analysis:** `logs_in_develop/Gen14/DA_20260901_Gen14_flagship_K20_T0.2_dpcc_vs_hardflow.md` Part 4.

## Symptom

```
File "mix_visual_aligning_test/eval_mix_visual_aligning.py", line 2437, in predict
    _hf_cond = encode_visual_cond(self.model, cond)
File "mix_visual_aligning/sampling/hardflow_projection.py", line 938, in encode_visual_cond
    'visual_latent': model._encode_once(bp_imgs, inhand_imgs)}
AttributeError: 'VisualFlowMatching' object has no attribute '_encode_once'
```

## Cause

`_encode_once` is a **two-time-engine method**, defined only in `visual_mf_diffusion.py:43` and
`visual_af_diffusion.py:42`. Its own docstring states the reason it exists: the returned latent is
*"a captured CONSTANT inside `_p_losses_meanflow`'s JVP closure, so its forward-mode tangent is zero
by construction"* — a MeanFlow/α-Flow training requirement, not a conditioning convention.

`VisualFlowMatching` (Gen7, single-time) has no such method and never needed one. Its `forward()`
(`visual_fm_diffusion.py:93-101`) passes the raw dual-cam window through as
`'visual': (bp_imgs, inhand_imgs, obs_seq)` and lets the backbone encode.

`encode_visual_cond` called `_encode_once` unconditionally while its docstring claimed it
*"Mirrors `VisualMeanFlow.forward` / `VisualAlphaFlow.forward` / `VisualFlowMatching.forward`
exactly"*. **The third clause was false**, and had been since arm C was added. It never surfaced
because every prior arm-C run on this task was `mf` or `af`; the flagship was the first time arm C
was pointed at `fm`.

## Change

One `hasattr` branch. **No math touched** — this is a conditioning-dict shape fix only.

```python
    if 0 in cond and isinstance(cond[0], tuple):
        bp_imgs, inhand_imgs, obs_seq = cond[0]
        if not hasattr(model, '_encode_once'):
            # Single-time engine: hand back exactly what its own forward() builds.
            return {0: obs_seq[:, -1],
                    'visual': (bp_imgs, inhand_imgs, obs_seq)}
        return {0: obs_seq[:, -1],
                'visual_latent': model._encode_once(bp_imgs, inhand_imgs)}
```

The docstring was rewritten to state the two-family split explicitly and to record the bug, so the
next reader cannot re-derive the same false assumption.

## Why this is safe

- **Shape parity is exact.** The returned dict is byte-for-byte what `VisualFlowMatching.forward`
  builds at `visual_fm_diffusion.py:98-101` — same `0: obs_seq[:, -1]` anchor, same `'visual'` tuple.
- **The sampler's own guard already admits it.** `_VISUAL_COND_KEYS = frozenset({'visual_latent',
  'visual'})` (line 884), and the conditioning check at line 1112 tests membership in that set, so
  the `'visual'` form passes without further change.
- **Two-time engines are untouched.** `mf` and `af` have `_encode_once`, take the original branch,
  and are bit-identical to before. Job 25247 (mf, 38/38 complete) is unaffected and nothing already
  in the corpus is invalidated.
- **No gate risk.** `hardflow_projection.py` appears in **none** of `GRAFTED_DIFF`, `GRAFTED`, or the
  COPIED ledger in `mix_visual_aligning_test/gates_mix_visual.py` — G0 does not watch this file, so
  the edit cannot fail the gate job and cannot kill a pipeline through `afterok`. (This was checked
  deliberately: an unregistered graft failing G0 is exactly what killed the α-Flow pipeline 25190
  when gates 25206 failed.)
- Syntax verified by `ast.parse`. **Not executed** — no Python env in this container; needs a
  cluster run to confirm.

## Behavioural difference worth knowing

For `mf`/`af` the images are encoded **once** per replan, before the ODE loop. For `fm` the
`'visual'` tuple is handed to the backbone, which encodes on each `_predict_velocity` call — i.e.
**K times per replan instead of once**. That is what the non-HardFlow `fm` path already does
(arms A and B go through `VisualFlowMatching.forward` → `p_sample_loop` identically), so arm C is
now consistent with arms A and B for this engine rather than faster than them — which is the
correct choice for a matched comparison. It does mean `fm` arm C will not be cheaper than `fm`
arm B on encoder time, and may partly explain the unexplained 1.55× `fm`-vs-`mf` unguided sampler
gap noted in the DA §3.5. **Run on cluster to confirm.**

## Validation required (cluster)

1. Resubmit the fm flagship; confirm items 17-19 and 36-38 produce `hardflow_sls-{r,c,t}` artifacts.
2. Confirm `[ hardflow ] engine=fm ... act_thr=0.2 visual=True` appears with no `[BLOCKED]` line
   (`hardflow_step_budget(20, 0.2)` → `n_genuine=3`, tier `HF_OK`).
3. Confirm `mf` is unchanged — a rerun of any `mf` arm-C cell should reproduce 25247 exactly.
