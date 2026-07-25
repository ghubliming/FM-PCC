# CHANGELOG — Gen12 U4 + U4.2 coding pass

**Date:** 2026-07-25 · **Type:** implementation · **Status:** code complete, **NOTHING RUN** (no deps here)
**Implements:** [`PLAN_Gen12_U4_late_activation_threshold.md`](PLAN_Gen12_U4_late_activation_threshold.md)
and [`PLAN_Gen12_U4.2_mpc_candidate_selection.md`](PLAN_Gen12_U4.2_mpc_candidate_selection.md)
**Nothing committed.** Static checks only (compile + pure-logic simulation); real runs on the cluster.

---

## 0. TL;DR

Two upgrades to arm C (`hardflow_new`), both eval-only, both additive:

- **U4 — late-activation threshold.** The per-step NLP is now solved only when the flow time
  `τ_next ≥ threshold`, **with the final step always solved** (the terminal solve is what the safety
  guarantee rides on — HardFlow paper Prop. safety_guarantee). `threshold=0.0` = every step (old
  behaviour), `0.5` = last half (DPCC-parity), `1.0` = terminal-only.
- **U4.2 — MPC candidate fan + DPCC-style selection.** `batch_size>1` fans independent NLP-steered
  candidates (the loop already existed); selection is now `random` / `temporal_consistency` /
  `minimum_projection_cost`, chosen by the variant suffix `hardflow_new-{r,t,c}`, mirroring DPCC.

Both are off by default (`threshold 0.0`, `batch 1`) so the existing arm-C behaviour is unchanged
until you turn them on.

## 1. Files changed

| file | change |
|---|---|
| `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | U4: `resolve_activation_threshold()`; `HardFlowSampler` takes `activation_threshold` (replaces binary `activation`); threshold gate + final-step guard. U4.2: per-candidate cost accumulation (`candidate_costs`, `_control`); `HardFlowPolicy` takes `trajectory_selection` + `candidate_cost`; `_select()` implements r/t/c. |
| `flow_matcher_v3_hardflow/sampling/__init__.py` | export `resolve_activation_threshold` |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | resolve threshold/batch/cost from config (+ env overrides); map `hardflow_new-{r,t,c}` → selection; pass new args; provenance tag `thres{t}_mpc{b}`; report + npz metadata |
| `FM_v3_hardflow_test/load_results_FM_v3_hardflow.py` | build the matching `thres{t}_mpc{b}` run tag; plot name follows |
| `FM_v3_hardflow_test/gates_hardflow.py` | G3 updated to new kwarg; **G4** (U4 threshold invariant) and **G5** (U4.2 fan+selection) added |
| `config/hardflow_projection_eval.yaml` | `hardflow.activation_threshold` (0.0), `candidate_cost` (prox); reworded `batch_size`/selection docs |

## 2. U4 — the threshold gate (the math-critical part)

The activation decision (`HardFlowSampler.sample`):

```python
active = (tau_next >= self.activation_threshold) or (k == K - 1)
```

- **`or (k == K-1)` is the one invariant** (PLAN U4 §5.2). HardFlow's feasibility guarantee comes
  from the final step alone (`α₁=1, β₁=0 ⇒ x_N = x̂_N*`); the terminal NLP must always run. The guard
  also covers the float edge case where `τ_next` at the last step isn't exactly 1.0.
- Aliases preserved: `resolve_activation_threshold('all')→0.0`, `('late')→0.5`. Config accepts either
  `activation_threshold: <float>` or the legacy `activation: all|late`.

Verified by pure-logic simulation (final step always active; solve count monotone ↓ in threshold):

```
K=10 thr=0.0 -> 10 solves   thr=0.5 -> 6 (last half)   thr=0.9 -> 2   thr=1.0 -> 1 (terminal only)
K= 2 thr=0.5 -> 2           thr=1.0 -> 1
every case includes step K-1  ✓
```

## 3. U4.2 — candidate fan + selection

- **Fan already existed** (`for b in range(batch_size)`), so `batch_size>1` just works. What was
  added:
  - **per-candidate cost** in `sample()`: `candidate_costs = Σ_k ‖x1_proj − x1_ref‖²` (prox / total
    NLP intervention — the ranking key) and `candidate_costs_control = Σ_k ‖u_k‖` (descriptor).
  - **`HardFlowPolicy._select()`**, mirroring DPCC's `Policy`:
    - `random` → index 0;
    - `temporal_consistency` → `argsort ‖obs[:,:-1] − prev_obs[:,1:]‖` (DPCC's block, verbatim);
    - `minimum_projection_cost` → `argmin(candidate_costs)` (least NLP intervention = most faithful to
      the field; the principled analog of DPCC-c, per the minimal-intervention principle).
  - `prev_observations` updated with the **selected** candidate each replan (as DPCC does), so
    temporal consistency works across the MPC loop.
- **Variant suffix → rule** (eval): `hardflow_new`/`-r` → random, `-t` → temporal, `-c` → min-cost.

## 4. Provenance (per the user's tip)

Results now encode the arm-C config in the folder name so a threshold/mpc sweep never collides:

```
.../plans/flow_matching_v3_hardflow/H8_K10_D…FlowMatchingODE/6/results/
    halfspace_both-hard/K10_n2_thres0.5_mpc4/hardflow_new-c.npz
```

`load_results` builds the identical `K{K}_n{n}_thres{t}_mpc{b}` tag (and the summary PNG name), and
the `all_seeds/` plot path matches. Env overrides `HFFM_ACT_THRESHOLD` / `HFFM_BATCH` feed both the
policy and the tag, so a sweep is collision-free without editing config.

## 5. Verification (static — nothing executed)

- All five touched Python files compile (`py_compile`).
- `resolve_activation_threshold`: aliases map correctly; out-of-range / bad strings raise.
- Threshold gate: final step always active for every (K, threshold); solve count monotone.
- YAML: no tabs; both `activation_threshold` and legacy `activation` present (former wins).
- No stale `activation=` kwarg or `self.activation` references remain.
- **New gates G4/G5 written but NOT run** (need torch + casadi on the cluster). G4 asserts the
  final-step feasibility invariant + monotone solve count across (K, thr); G5 asserts the fan
  produces per-candidate costs, all terminals feasible, and `-c/-r` select correctly.

## 6. How to run (cluster)

Defaults unchanged (single-run, threshold 0, batch 1). To exercise U4/U4.2:

```bash
# U4: DPCC-parity late activation (edit yaml activation_threshold: 0.5, or:)
HFFM_ACT_THRESHOLD=0.5 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh

# U4.2: 4-candidate fan + min-cost selection (add hardflow_new-c to projection_variants, batch 4)
HFFM_BATCH=4 ./Slurm_Codes/submit.sh …/eval_fmv3_hardflow_job.sh

# combined (the headline cell): late-activated, cost-selected, 4-candidate fan
HFFM_ACT_THRESHOLD=0.5 HFFM_BATCH=4 ./Slurm_Codes/submit.sh …/eval_fmv3_hardflow_job.sh
```

Run the gates first (adds G4/G5): `…/gates_hardflow_fmv3.sh`.

## 7. Notes / traps carried forward

- **G4 is the safety check** — if it ever fails (a threshold skipping the terminal solve), the
  feasibility guarantee is void. Do not run eval past a G4 failure.
- **Selection needs diversity:** all candidates share `s0`, differing only by initial noise; at very
  low K the chains may be near-identical, so selection has little to choose from (note when reading
  K=2). Gate G5 uses K=5, batch 4.
- **Cost key:** `candidate_cost: prox` is the default ranking key; `control` (effort) is also
  recorded. Pick one and keep it fixed within a comparison.
- Still smoke-scale (`n_trials=2`, seed 6) — raise for a real n≥100 (fix_3 §7). U4/U4.2 do not change
  that.
- Arms A/B are threshold/mpc-invariant, so their numbers repeat across `thres*_mpc*` tags — expected,
  not a bug. For a pure U4/U4.2 sweep, set `projection_variants` to arm C only to avoid re-running A/B.
