# CHANGELOG — Gen13 fix_11: flip the U10 activation threshold to DPCC polarity

**Date:** 2026-07-26 · **Type:** fix (semantics) · **Status:** code complete, verified static
**Codebase:** vendored `HardFlow/`. Fixes the threshold introduced in
[`../U_10/PLAN_Gen13_U10_HF_activation_threshold.md`](../U_10/PLAN_Gen13_U10_HF_activation_threshold.md).
**Sibling:** identical flip in Gen12
[`../../Gen12/fix_6/CHANGELOG_fix6_dpcc_threshold_polarity.md`](../../Gen12/fix_6/CHANGELOG_fix6_dpcc_threshold_polarity.md)
**Nothing committed. Nothing run.**

---

## 0. The bug

U10's `hardflow_activation_threshold` was defined as *"solve NLP when `t_{k+1} ≥ threshold`"* →
**higher threshold = LESS projection**, the OPPOSITE of DPCC's `diffusion_timestep_threshold`
(`project when loop_idx ≥ (1 − T)·K` → higher T = MORE projection). Inverted (except at 0.5).

## 1. The fix — exact DPCC gate

`HardFlow/hardflow/models_flow/flow_policy.py` (`hardflow_new_forward`):

```python
control_flag = (k >= (1.0 - activation_threshold) * self.oc_N_steps) or (k == self.oc_N_steps - 1)
```

`threshold` now == DPCC's `diffusion_timestep_threshold` (fraction of the late trajectory projected,
higher = more). Verified step-set parity at K=10: `1.0 → all`, `0.5 → last half`, `0.0 →
terminal-only` (via the mandatory final-step guard).

## 2. Files changed (vendored `HardFlow/` + one sbatch)

| file | change |
|---|---|
| `HardFlow/hardflow/models_flow/flow_policy.py` | gate flipped to `k >= (1−thr)·N`; `all` fallback → **1.0** (was 0.0), `late` → 0.5; comments |
| `HardFlow/hardflow/config/flow_matching.py` | `hardflow_activation_threshold` doc rewritten to DPCC polarity (`all → 1.0`) |
| `HardFlow/run_scripts/eval_hardflow_new.sh` | header doc: 1.0=full / 0.5=half / 0.0=terminal-only |
| `Slurm_Codes/sbatch/hardflow/eval_threshold_sweep_hardflow.sh` | default grid `0.0 0.5 1.0` → **`1.0 0.5 0.0`** (full → half → terminal); DPCC-polarity docs |

Default is preserved: `hardflow_activation_threshold = -1.0` (disabled) → falls back to
`hardflow_activation="all"` → now **1.0** = every step, i.e. unchanged behaviour.

## 3. Impact on already-collected results

The U10 sweep (job 23832) and `RESULTS_Gen13_U10_threshold_sweep.md` used the OLD labels:

| old label | means | new (DPCC) label |
|---|---|---|
| `thres0.0` | full-step (10/10 NLP) | **`thres1.0`** |
| `thres0.5` | last half (6/10) | `thres0.5` (unchanged) |
| `thres1.0` | terminal-only (1/10) | **`thres0.0`** |

The result folders on disk keep their old names. **Findings are unchanged** — the free-lunch is at
0.5 (fixed point) and terminal-only degrades — only the 0.0/1.0 axis labels swap. A fix note is
appended to the results MD; a re-run under this fix will produce DPCC-labelled folders.

## 4. Verification (static)
- `flow_policy.py` and `flow_matching.py` compile; both shell scripts pass `bash -n`.
- Exact DPCC step-set parity confirmed (K=10, thr ∈ {1.0, 0.5, 0.0}).
- Disabled default (-1.0 → `all` → 1.0) reproduces the pre-fix every-step behaviour.

## 5. Note
As in Gen12, the final-step guard makes DPCC-`threshold 0.0` **terminal-only**, not
truly-none — necessary for HardFlow's terminal safety guarantee.
