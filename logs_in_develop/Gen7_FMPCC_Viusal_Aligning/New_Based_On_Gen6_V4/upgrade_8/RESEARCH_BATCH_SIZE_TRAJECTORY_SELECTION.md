# Research Report: Batch Size & Trajectory Selection Design
## Origin of `batch_size = 6` and `trajectory_selection` in Gen6V4 / Gen7

**Author:** AI Audit  
**Date:** 2026-05-21  
**Scope:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` · `fm_visual_aligning_test/eval_fm_visual_aligning.py`  
**References:** `/workspaces/dpcc` (L4DC 2025) · `/workspaces/diffuser` (Janner et al.) · `FM_v3_ode_selectable_test/`

---

## 1. The Question

In both visual eval scripts, every non-`diffuser` variant runs with a hardcoded `batch_size = 6`:

```python
# eval_visual_aligning_dpcc.py  &  eval_fm_visual_aligning.py  (commit 7b14333)
batch_size = getattr(args, 'batch_size', 1)
if 'diffuser' not in variant:
    batch_size = 6
```

There is no comment, no config key, no citation. The questions:
1. Where did `6` come from?
2. What does the DPCC paper repo use, and why?
3. What does the original Diffuser repo use?
4. What does our FMv3ODE pipeline use?
5. Are the semantics of `trajectory_selection='random'` the same across repos?
6. Verdict for Gen7 / Gen6V4.

---

## 2. Repo-by-Repo Investigation

### 2.1 Original Diffuser (`/workspaces/diffuser`)

**Config:** `config/locomotion.py` → `plan.batch_size: 32`

**Policy call:**
```python
# scripts/plan_guided.py
action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
```

**Selection logic** (`diffuser/sampling/policies.py`):
```python
## extract first action
action = actions[0, 0]   # always index 0
```

**Why large batch?**  
The original Diffuser uses **classifier-guided diffusion** (value function). The batch is a guidance beam — all `batch_size` samples are simultaneously guided by the value gradient at every reverse diffusion step. The batch is NOT for post-hoc selection; it is the guidance mechanism itself. Picking `[0]` at the end is arbitrary since all trajectories converge toward high-value regions via the shared guidance signal.

**Summary:** `batch=32`, always picks index 0. Batch size is a guidance-beam parameter, not a diversity parameter.

---

### 2.2 DPCC (`/workspaces/dpcc`, Römer et al. L4DC 2025)

**Config:** `config/avoiding-d3il.py` → `plan.batch_size: 4`

**Policy call:**
```python
# scripts/eval.py
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, ...)
```
`args.batch_size = 4` for **every** variant including `diffuser`.

**Variant naming + selection logic** (`diffuser/sampling/policies.py`):
```python
# dpcc-r  →  trajectory_selection = 'random'
# dpcc-c  →  trajectory_selection = 'minimum_projection_cost'
# dpcc-t  →  trajectory_selection = 'temporal_consistency'

if trajectory_selection == 'temporal_consistency' and prev_obs is not None:
    which_trajectory = argsort(temporal_diff)[0]          # closest to previous step
elif trajectory_selection == 'minimum_projection_cost':
    which_trajectory = argmin(sum(projection_costs))      # lowest SLSQP cost
else:                                                     # 'random' — MISNAMED
    which_trajectory = 0                                  # always index 0
```

**Key finding:** In DPCC, `'random'` does NOT mean random. It means **always index 0** (deterministic, identical to just taking the first sample). The variant label `dpcc-r` is shorthand for "random starting point from diffusion" — meaning batch members are IID samples, and we just take the first one. It is a deterministic policy.

**Why batch=4 in DPCC?**  
No explicit comment in code. From the paper context: DPCC demonstrates that its projection improves constraint satisfaction vs. the plain `diffuser` baseline. Using batch=4 (consistent across all variants) ensures fair comparison — all methods get the same number of function evaluations. The size 4 appears to be a practical choice balancing GPU memory and diversity.

**Summary:** `batch=4` from config (configurable), all variants including `diffuser`. `'random'` = always index 0.

---

### 2.3 FM-PCC FMv3ODE (`FM_v3_ode_selectable_test/`, our repo)

**Config:** `config/avoiding-d3il.py` → `plan.batch_size: 4` / `plan_fm.batch_size: 4`

**Policy call:**
```python
# eval_flow_matching_v3_ode_selectable.py
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, ...)
```

Uses the **same `Policy` class** inherited from DPCC. Same behavior: `batch=4`, `'random'` = index 0.

**Summary:** Identical to DPCC in batch design. `batch=4`, deterministic selection for `'random'`.

---

### 2.4 FM-PCC Visual Aligning: Gen6V4 + Gen7 (our visual eval)

**Where `6` was introduced:** Commit `7b14333` — "feat(KEY UPDATE Gen6v4; DEBUG unfinshed): pivot from ddpmact to 9D Visual-DPCC safety engine". No comment or justification.

**How it is applied:**
```python
# Both eval_visual_aligning_dpcc.py and eval_fm_visual_aligning.py
batch_size = getattr(args, 'batch_size', 1)
if 'diffuser' not in variant:
    batch_size = 6
```

| Variant | `batch_size` | `trajectory_selection` |
|:--|:--|:--|
| `diffuser` | `1` (from args default) | `'random'` → always index 0 (batch=1, moot) |
| `gradient`, `gradient-tightened` | `6` | `'random'` → **genuinely random** |
| `post_processing`, `post_processing-tightened` | `6` | `'minimum_projection_cost'` (Fix 9.4) |
| `model_free`, `model_free-tightened` | `6` | `'minimum_projection_cost'` (Fix 9.4) |

**Critical difference from DPCC/FMv3ODE:** Our `VisualAgentWrapper` implements `'random'` as a **true random draw**:
```python
# Our VisualAgentWrapper — eval_visual_aligning_dpcc.py / eval_fm_visual_aligning.py
elif self.trajectory_selection == 'random':
    which = np.random.randint(self.batch_size)   # ← genuinely random
```
vs. DPCC/FMv3ODE Policy:
```python
else:  # 'random' in DPCC
    which_trajectory = 0                          # ← always first, deterministic
```

---

## 3. Cross-Repo Comparison Table

| Aspect | `diffuser` (Janner) | DPCC (Römer) | FMv3ODE (FM-PCC) | Gen6V4/Gen7 Visual (FM-PCC) |
|:--|:--|:--|:--|:--|
| `diffuser` variant batch | `32` | `4` | `4` | **`1`** |
| Projected variant batch | `32` (no projection) | `4` | `4` | **`6`** |
| Batch source | Config | Config | Config | **Hardcoded** |
| `'random'` semantics | Always index `0` | Always index `0` | Always index `0` | **True `np.random`** |
| Batch purpose | Guidance beam | Diversity pool | Diversity pool | Diversity pool |

---

## 4. Analysis

### 4.1 Why `6` is black magic

The number 6 does not appear in the DPCC paper, the DPCC code, or the original Diffuser code. It was written in the first Gen6V4 visual eval commit (`7b14333`) without justification. There is no mathematical or empirical reason to prefer 6 over 4 (DPCC default) or any other value.

### 4.2 Does batch size matter per variant?

**`diffuser` variant (batch=1 in ours):**  
Correct. The `diffuser` variant runs no projection and has no selection logic. A single sample is sufficient. DPCC's batch=4 here is wasteful (4× the denoising compute for no benefit).

**`gradient`/`gradient-tightened` (batch=6, random selection):**  
The gradient projection computes `∇_{trajectory} projection_cost` and updates the trajectory in-place during denoising. It does not benefit from multiple candidates — the gradient is applied per sample independently. Using batch=6 and then picking one at random is equivalent to batch=1 and running once: all 6 samples are IID from the same conditional distribution, and random selection picks one at random, yielding an arbitrary IID draw. The only effect is 6× the compute.  
Additionally: our `'random'` is truly random (adds evaluation stochasticity), while DPCC always picks index 0 (deterministic). Under DPCC's semantics, batch=1 and batch=4 with `random` are equivalent.

**`post_processing`/`model_free` (batch=6, `minimum_projection_cost`):**  
Here batch size genuinely matters. The post-processing projection runs after the full denoising chain; `model_free` adds no model-prior cost. With `minimum_projection_cost`, a larger batch increases the probability of finding at least one feasible trajectory under tight constraints. 6 is reasonable but arbitrary. DPCC uses 4.

**`post_processing-tightened`/`model_free-tightened` (batch=6, `minimum_projection_cost`):**  
Same reasoning as above. The "tightened" variants shrink the feasible region further, so more candidates arguably help more. 6 vs 4 is still arbitrary.

### 4.3 The true random vs. always-first divergence

This is a subtle but real behavioral difference for the `gradient` variants.

- **DPCC semantics (`which=0`):** Deterministic. Given the same seed, the same trajectory is always selected. Reproducibility is guaranteed across runs.
- **Our semantics (`np.random.randint`):** Stochastic. The selected trajectory changes between runs unless the eval `np.random` seed is fixed. This means the reported success rate includes an additional source of variance — the trajectory-selection lottery — on top of the diffusion sampling variance.

For fair ablation comparisons, the `gradient` variant in our eval is less reproducible than in DPCC.

---

## 5. Verdict for Gen6V4 / Gen7

| Variant | Current `batch_size` | Recommended | Rationale |
|:--|:--|:--|:--|
| `diffuser` | `1` | **Keep `1`** | Correct and efficient. DPCC's `4` here is wasteful. |
| `gradient`, `gradient-tightened` | `6`, random | **Consider `1` or `4`** | Random selection from IID batch adds variance without benefit. If keeping batch>1, change `'random'` to always-first (index 0) to match DPCC determinism. |
| `post_processing`, `model_free` (+tightened) | `6`, min-cost | **Keep `6` or move to config** | min-cost selection benefits from diversity. 6 is fine but should be a config key, not a magic number. DPCC uses 4. |

**Recommended action (research direction):**

1. **Make `batch_size` a YAML config key** (e.g., `eval_batch_size: 6`) so it can be ablated without code changes.
2. **For `gradient`/`gradient-tightened`**: either drop to batch=1 (sufficient, saves 6× GPU) or keep batch>1 but change to `which=0` (deterministic, matches DPCC) instead of true random.
3. **For `post_processing`/`model_free`**: ablate batch size (4 vs 6 vs 8) as part of Gen7/Gen6V4 evaluation sweep to find the optimal trade-off between compute and feasibility rate.
4. **The `'random'` label is misleading in the DPCC lineage.** Rename our variant's behavior or standardize to DPCC semantics (index 0) for reproducibility.

---

## 6. Open Question (Next Research Direction)

The `'minimum_projection_cost'` trajectory selection (S3 in the POSTMORTEM table) was introduced in Fix 9.4 specifically for `post_processing`/`model_free` variants. The DPCC paper uses this as `dpcc-c` — one of three formal trajectory selection methods (`r=random/first`, `c=cost`, `t=temporal_consistency`).

The open question: **for the visual aligning task with flow matching**, which selection method yields the highest success rate?

- `first` (index 0): DPCC default for most variants; deterministic, cheapest.
- `minimum_projection_cost`: needs batch>1, picks the trajectory that was cheapest to project; makes most sense for `post_processing`/`model_free`.
- `temporal_consistency`: selects trajectory most similar to the previous step's trajectory; most relevant for receding-horizon replanning.

Our Gen7/Gen6V4 visual eval currently uses:
- `gradient` variants → `'random'` (true random, effectively first since IID)
- `post_processing`/`model_free` → `'minimum_projection_cost'` (Fix 9.4)
- No variant uses `'temporal_consistency'`

A systematic ablation of `{selection_method} × {batch_size}` across all variants would determine whether S3 (`minimum_projection_cost` for `post_processing`/`model_free`) actually improves success over the simpler `first` baseline, and whether adding `temporal_consistency` for `gradient` is worth the overhead.
