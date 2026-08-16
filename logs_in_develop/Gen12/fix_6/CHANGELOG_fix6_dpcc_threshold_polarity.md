# CHANGELOG — Gen12 fix_6: flip the activation threshold to DPCC polarity

**Date:** 2026-07-26 · **Type:** fix (semantics) · **Status:** code complete, verified static
**Sibling:** Gen13 got the identical flip — see
[`../../Gen13/fix_11/CHANGELOG_fix11_dpcc_threshold_polarity.md`](../../Gen13/fix_11/CHANGELOG_fix11_dpcc_threshold_polarity.md)
**Nothing committed. Nothing run.**

> ⚠️ **Numbering:** the user said "fix5", but Gen12 `fix_5` already exists (the FMv3ODE path
> layout). To avoid overwriting it this is filed as **fix_6**. Rename if you intended otherwise.

---

## 0. The bug: our threshold was INVERTED vs DPCC

The U4 activation threshold was defined as *"solve the NLP when `τ ≥ threshold`"* → **higher
threshold = LESS projection**. DPCC's `diffusion_timestep_threshold` is the opposite: it projects
when `loop_idx ≥ (1 − T)·K` → **higher T = MORE projection**. So arm B (DPCC) and arm C (our
HardFlow) sat in the same eval with the word "threshold" meaning **opposite** things — the same
number gave opposite behaviour (except at the symmetric point 0.5):

| behaviour | DPCC `T` | old ours | fix_6 ours |
|---|---|---|---|
| project **all** steps (full) | 1.0 | 0.0 | **1.0** |
| project **last half** | 0.5 | 0.5 | **0.5** |
| project **terminal only** | 0.0 | 1.0 | **0.0** |

The relation was `old_ours = 1 − DPCC_T`. fix_6 makes our `threshold` **identical to DPCC's**.

## 1. The fix — exact DPCC gate

The activation gate is now DPCC's own formula, `k ≥ (1 − threshold)·K`, plus the mandatory
final-step guard:

```python
# flow_matcher_v3_hardflow/sampling/hardflow_projection.py
active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
```

Verified to reproduce DPCC's projected-step set exactly (K=10):
`thr 1.0 → [0..9]` (all), `0.5 → [5..9]` (last half), `0.0 → [9]` (terminal-only via the guard;
DPCC-`T=0` is truly none, but our safety guarantee requires the terminal solve).

`threshold` now means, verbatim, DPCC's `diffusion_timestep_threshold`: **the fraction of the late
trajectory that is projected; higher = more projection.**

## 2. Files changed

| file | change |
|---|---|
| `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | gate flipped to `k >= (1−thr)·K`; `resolve_activation_threshold` alias **`all` → 1.0** (was 0.0), `late` → 0.5; docstring/comments |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | fallback default `activation` `0.0 → 1.0` (keep "every step" as default); comment |
| `FM_v3_hardflow_test/load_results_FM_v3_hardflow.py` | matching fallback `0.0 → 1.0` (path tag stays consistent with eval) |
| `FM_v3_hardflow_test/gates_hardflow.py` | G4 expected-count uses `k >= (1−thr)·K`; monotonicity flipped to **non-decreasing** (higher thr ⇒ more solves) |
| `config/hardflow_projection_eval.yaml` | `activation_threshold: 0.0 → 1.0` (default = every step, unchanged behaviour); docs rewritten to DPCC polarity |

## 3. Default behaviour is preserved

Old default was `0.0` = every step (full projection). New default `1.0` = every step. So a run with
no threshold set behaves **identically** before and after fix_6. Only the *meaning of intermediate
values* changed (to match DPCC).

## 4. Impact on already-collected results

The Gen12 U4 K=20 runs (jobs 23829–31) and their `RESULTS_Gen12_U4_threshold_K20.md` used the OLD
(pre-flip) labels. Mapping to the new (DPCC) labels:

| old label | means | new (DPCC) label |
|---|---|---|
| `thres0.0` | full-step | **`thres1.0`** |
| `thres0.5` | last half | `thres0.5` (unchanged) |
| `thres1.0` | terminal-only | **`thres0.0`** |

The **0.5 free-lunch finding is unaffected** (fixed point). The results MD has a fix_6 note appended
so the old folder labels aren't misread. Result folders on disk keep their old names; a re-run under
fix_6 will produce DPCC-labelled folders.

## 5. Verification (static)
- All touched Python compiles; the two sbatch/run scripts pass `bash -n` (Gen13 side).
- Exact DPCC step-set parity confirmed for K=10 at thr ∈ {1.0, 0.5, 0.0}.
- `all`→1.0 / `late`→0.5 alias resolution confirmed.
- Default (threshold 1.0) reproduces the pre-fix "every step" behaviour.

## 6. Note
The safety-guarantee final-step guard means DPCC-`threshold 0.0` is **terminal-only**, not
truly-no-projection — a deliberate, necessary divergence (HardFlow needs the terminal solve for
`h(x_N) ≤ 0`; DPCC has no such guarantee to protect).
