# Gen12 U5 — full DPCC-parity variant scheme for arm C (`hardflow_new-{r,c,t}[-tightened]`)

**Goal:** give arm C (HardFlow in-loop) the *exact same* variant matrix DPCC has, so every
comparison against arm B (`dpcc-c-tightened`) is matched on both axes it varies:
**candidate-selection rule** (`-r`/`-c`/`-t`) and **constraint geometry** (`''` vs `-tightened`).

**Type:** direct coding (no plan MD, per request). No new sampler logic — the machinery
already existed (U4.2 selection + the generic `tightened` dispatch); U5 fixes the *naming/parsing*
so the two axes compose, and registers the full set.

---

## 1. What "tightened" actually is (confirmed from the code)

`-tightened` is a **pure eval-time enlargement of the geometric constraints** by
`enlarge_constraints` (0.025 for avoiding). No model / no training involvement:

| constraint | non-tightened (`constraint_list`) | tightened (`constraint_list_tightened`) | file |
|---|---|---|---|
| halfspace | offset margin `0` | offset margin `enlarge_constraints` | `eval_FM_v3_hardflow.py:223-224` |
| obstacles | `radius` | `radius + enlarge_constraints` | `:232-233` |
| bounds | lb/ub | **same lb/ub (not enlarged)** | `:228-229` |
| dynamics | identical | identical | `:238-240` |

So it is exactly the halfspace + obstacle geometry, matching the intuition. DPCC uses the
same margin on its `-tightened` arms — `dpcc-c-tightened` vs `hardflow_new-c-tightened` is
therefore matched on the feasible set (the untightened `hardflow_new` is *not* — it enforces
exact constraints with zero margin).

## 2. DPCC parity (confirmed from `aux_repo/dpcc`)

- Variant set (`dpcc/scripts/load_results.py:79`): `['dpcc-r','dpcc-t','dpcc-c']` ×
  `{'', '-tightened'}` = **6**.
- Selection parse (`dpcc/scripts/eval.py:155-157`) uses **substring** matching, so selection
  composes with tightening independently:
  ```python
  trajectory_selection = 'random'
  if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
  if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
  ```
- Selection **semantics** (`dpcc/diffuser/sampling/policies.py:65-73`) — verified identical to
  ours in `flow_matcher_v3_hardflow/sampling/hardflow_projection.py:579` (`HardFlowPolicy._select`):
  - `temporal_consistency`: `argsort(‖obs[:-1] − prev_obs[1:]‖)[0]`
  - `minimum_projection_cost`: `argmin(Σ projection_costs)` — ours ranks the prox cost
    `Σ‖x̂₁_proj − x̂₁_ref‖²` (`candidate_costs`), the flow-matching analog.
  - **`batch_size == 1 → index 0`**, so `-r`/`-c`/`-t` are byte-identical at mpc==1 and only
    diverge once the candidate fan is on (mpc>1). This is why the trio is redundant at mpc==1.

## 3. The bug U5 fixes

Arm C parsed the selection suffix with `endswith`:
```python
if variant.endswith('-t'): ...     # OLD
elif variant.endswith('-c'): ...
```
This **does not compose with `-tightened`**: `hardflow_new-c-tightened` ends in `d`, so it
silently fell back to **random** instead of minimum-projection-cost — the tightened arms
would have run the wrong selection. It also collides on any bare `hardflow_new-tightened`
(`'hardflow_new-t'` is a prefix of `-tightened`).

**Fix** (`eval_FM_v3_hardflow.py`): strip the geometry marker first, then read the selection
suffix — collision-free and composes exactly like DPCC's substring parse:
```python
_sel_base = variant.replace('-tightened', '')
hf_selection = 'random'
if _sel_base.endswith('-t'): hf_selection = 'temporal_consistency'
elif _sel_base.endswith('-c'): hf_selection = 'minimum_projection_cost'
```
The `tightened` geometry is still chosen by the pre-existing generic dispatch
(`elif not 'model_free' in variant and 'tightened' in variant → constraint_list_tightened`,
`:269`), which already applies to hardflow variants. **No sampler / policy code changed.**

Verified parse for all 8 names (static sim, no numpy):

| variant | hardflow | geometry | selection |
|---|---|---|---|
| `hardflow_new` | ✓ | exact | random |
| `hardflow_new-r` | ✓ | exact | random |
| `hardflow_new-c` | ✓ | exact | min-proj-cost |
| `hardflow_new-t` | ✓ | exact | temporal |
| `hardflow_new-r-tightened` | ✓ | tightened | random |
| `hardflow_new-c-tightened` | ✓ | tightened | min-proj-cost |
| `hardflow_new-t-tightened` | ✓ | tightened | temporal |
| `dpcc-c-tightened` | – | tightened | min-proj-cost |

## 4. Files changed

1. **`FM_v3_hardflow_test/eval_FM_v3_hardflow.py`** — selection parser now strips `-tightened`
   before reading `-r/-c/-t` (§3).
2. **`config/hardflow_projection_eval.yaml`** — `projection_variants` documents the full
   parity set. **Default run** = the two non-redundant-at-mpc==1 arms
   (`hardflow_new`, `hardflow_new-c-tightened`) alongside `diffuser` + `dpcc-c-tightened`;
   the remaining `-r/-c/-t[-tightened]` are commented, to be uncommented for an mpc>1 sweep
   (they are identical to these at mpc==1, so running them there only burns NLP time).
3. **`Data_Analysis/DA_Code_v3/config.py`** — `HARDFLOW_VARIANTS` now lists all 7
   (`hardflow_new` + `-{r,c,t}` × `{'', -tightened}`) so the loader/visualizer pick up
   whichever exist; `MAJOR_VARIANTS` gains `hardflow_new-c-tightened` as the headline
   matched-margin arm next to `dpcc-c-tightened`.

## 5. How to run

Default (matched-margin, mpc==1) — the headline B-vs-C' comparison:
```bash
HFFM_ACT_THRESHOLD=0.5 HFFM_FLOW_STEPS="20" \
  sbatch Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
```
Full DPCC-parity selection sweep — set `batch_size: N` (N>1) in the `hardflow:` block (or
`HFFM_BATCH=N`) and uncomment the full parity set in `projection_variants`. Only then do
`-r`/`-c`/`-t` produce different results.

**Note:** the eval-name folder is `K{K}_thres{thr}_mpc{batch}_n{n}`, so an mpc>1 sweep writes
a *different* folder (`mpc4…`) than the current `mpc1` data — no clobbering.
