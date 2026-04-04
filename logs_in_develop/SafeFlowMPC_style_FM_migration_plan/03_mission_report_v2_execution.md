# Mission Report: SafeFlowMPC-Style FM Migration (v2 Only)

## Mission Status
Completed.

This report documents the executed work for the approved mission:
- copy v2 architecture/test folders,
- implement SafeFlowMPC-style time sampling in copied v2 only,
- add exactly 2 new v2 parameters in `config/avoiding-d3il.py`,
- keep all other original branches unchanged.

---

## Scope Compliance Check

### In-scope targets (executed)
1. New copied architecture folder: `flow_matcher_v2`
2. New copied test folder: `FM_v2_test`
3. Copied v2 diffusion training-time `t` sampling switched to SafeFlowMPC style.
4. `config/avoiding-d3il.py` updated with new `flow_matching_v2` and `plan_fm_v2` entries.
5. Exactly two new v2 time-sampling parameters added.

### Out-of-scope protections (respected)
1. No edits made to original `flow_matcher_unet_v2`.
2. No edits made to original `FM_Unet_v2_test`.
3. No DPCC/projector logic changes.
4. No broad sampler framework or extra mode system added.

---

## Folder Operations Performed

1. Copied folder:
- from `flow_matcher_unet_v2`
- to `flow_matcher_v2`

2. Copied folder:
- from `FM_Unet_v2_test`
- to `FM_v2_test`

Result:
- New isolated v2 implementation path created.

---

## Code Changes Performed

## A) SafeFlowMPC-style Beta time sampling (copied v2 architecture only)
Edited file:
- `flow_matcher_v2/models/diffusion.py`

Changes:
1. Added two constructor parameters:
- `time_beta_alpha_v2`
- `time_beta_beta_v2`

2. Stored those parameters as instance fields.

3. Updated `_time_from_timestep` to support both:
- float `t` already in `[0,1]` (for Beta sampling),
- integer timestep index fallback (existing behavior path compatibility for non-training parts).

4. Replaced loss-time sampling from:
- uniform discrete `torch.randint(...)`

To:
- `Beta(alpha, beta)` sample,
- fixed flip `t = 1.0 - t`.

This implements the SafeFlowMPC-style time sampling behavior in the copied v2 FM training path.

---

## B) Copied v2 train/eval scripts rewired to v2 naming

Edited files:
1. `FM_v2_test/train_FM_Unet_v2.py`
2. `FM_v2_test/eval_FM_Unet_v2.py`
3. `FM_v2_test/load_results_FM_Unet_v2.py`

Changes:
1. Imports changed from `flow_matcher_unet_v2...` to `flow_matcher_v2...`.
2. Training experiment key changed:
- `flow_matching_unet_v2` -> `flow_matching_v2`
3. Planning experiment key changed:
- `plan_fm_unet_v2` -> `plan_fm_v2`

Additional train wiring:
- Passed `time_beta_alpha_v2` and `time_beta_beta_v2` from config args into copied diffusion config constructor.

---

## C) Config updates (required location)
Edited file:
- `config/avoiding-d3il.py`

Added new train config block:
- `flow_matching_v2`

Added new plan config block:
- `plan_fm_v2`

Added exactly two new v2 parameters (as requested):
1. `time_beta_alpha_v2`: default `1.5`
2. `time_beta_beta_v2`: default `1.0`

Sampling flip behavior:
- fixed in code (`t = 1 - t`),
- no extra flip parameter added.

No extra time-sampling config parameters were introduced.

---

## Validation and Sanity Results

Checks performed:
1. Folder existence checks for `flow_matcher_v2` and `FM_v2_test`.
2. Static error checks on all edited files.

Result:
- No errors found in:
  - `flow_matcher_v2/models/diffusion.py`
  - `FM_v2_test/train_FM_Unet_v2.py`
  - `FM_v2_test/eval_FM_Unet_v2.py`
  - `FM_v2_test/load_results_FM_Unet_v2.py`
  - `config/avoiding-d3il.py`

---

## Final Edited/Created Paths

Created by copy:
1. `flow_matcher_v2/`
2. `FM_v2_test/`

Edited:
1. `flow_matcher_v2/models/diffusion.py`
2. `FM_v2_test/train_FM_Unet_v2.py`
3. `FM_v2_test/eval_FM_Unet_v2.py`
4. `FM_v2_test/load_results_FM_Unet_v2.py`
5. `config/avoiding-d3il.py`

Created report:
1. `logs_in_develop/SafeFlowMPC_style_FM_migration_plan/03_mission_report_v2_execution.md`

---

## Notes for Run Usage

Use copied v2 scripts:
- train via `FM_v2_test/train_FM_Unet_v2.py`
- eval via `FM_v2_test/eval_FM_Unet_v2.py`

These now target:
- train experiment key: `flow_matching_v2`
- plan experiment key: `plan_fm_v2`

And train-time Beta parameters come from:
- `time_beta_alpha_v2`
- `time_beta_beta_v2`

in `config/avoiding-d3il.py`.

---

## Mission Conclusion
Mission executed successfully under the approved constraints.

---

## Code-Math Meaning (What Changed, From What, and What It Means)

### 1) Training-time time sampling
From (before in v2 source):
- sampled timestep index uniformly: `k ~ Uniform{0, ..., K-1}`
- then converted to continuous time: $t = \frac{k}{K-1}$

To (after in copied v2):
- sampled continuous time using Beta: $u \sim \mathrm{Beta}(\alpha, \beta)$
- fixed transform: $t = 1 - u$

Meaning:
- Before: near-uniform coverage of the full time interval.
- After: non-uniform emphasis controlled by $(\alpha,\beta)$.
- With configured defaults $(\alpha,\beta)=(1.5,1.0)$ and $t=1-u$, the effective $t$ distribution is biased toward smaller $t$.
- This shifts training mass toward earlier/noisier interpolation regions, matching SafeFlowMPC-style intent.

### 2) Interpolation/path equation remains the same
Unchanged FM path equation in training:
$$
x_t = (1-t)\,x_{\text{base}} + t\,x_{\text{start}}
$$

Meaning:
- We changed *how often each time region is sampled*, not the FM path definition itself.
- Model objective structure is preserved; only the sampling measure over $t$ changed.

### 3) Loss target structure remains the same
Unchanged velocity target form:
$$
v_{\text{target}} = x_{\text{start}} - x_{\text{base}}
$$

Meaning:
- The supervised signal formula is unchanged.
- Practical effect comes from which $t$ values dominate expectation during training.

### 4) Statistical objective interpretation
Before (uniform-like):
$$
\mathcal{L}_{\text{old}} = \mathbb{E}_{t\sim p_{\text{uniform}}}\,[\ell(t)]
$$

After (Beta-shaped):
$$
\mathcal{L}_{\text{new}} = \mathbb{E}_{u\sim \mathrm{Beta}(\alpha,\beta)}\,[\ell(1-u)]
$$

Meaning:
- We effectively reweighted the training integral over time.
- Same model family and loss form, different weighting over regions of the trajectory-noise bridge.

### 5) Why two new parameters are sufficient
Added only:
- `time_beta_alpha_v2`
- `time_beta_beta_v2`

Meaning:
- These two parameters fully define the Beta family shape.
- The flip $t=1-u$ is fixed in code, so no additional sampling-mode parameter is required.
- This satisfies strict scope while still giving control over time-density emphasis.

### 6) Experiment-key rewiring meaning
From:
- `flow_matching_unet_v2`, `plan_fm_unet_v2`

To:
- `flow_matching_v2`, `plan_fm_v2`

Meaning:
- New runs are isolated to copied v2 path/config.
- Original branches remain behaviorally untouched, enabling clean A/B comparison.
