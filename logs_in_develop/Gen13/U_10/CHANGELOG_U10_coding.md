# CHANGELOG — Gen13 U10 coding pass

**Date:** 2026-07-25 · **Type:** implementation · **Status:** code complete, **NOTHING RUN** (no deps here)
**Implements:** [`PLAN_Gen13_U10_HF_activation_threshold.md`](PLAN_Gen13_U10_HF_activation_threshold.md)
**Codebase:** the **vendored HardFlow repo** `HardFlow/` (the paper's own code), eval-only.
**Nothing committed.** Static checks only (compile + syntax + pure-logic); real runs on the cluster.

---

## 0. TL;DR

Adds a **continuous late-activation threshold** to the ORIGINAL HardFlow (FM backbone), so the
per-step NLP is solved only when the flow time `t_{k+1} ≥ threshold`, with the **final step always
solved** (terminal safety guarantee, paper Prop.). Lets us compare original HF **threshold ON vs the
full-step baseline** on HardFlow's own env (avoiding-v0, H16, `hardflow_new`) — a clean HF-vs-HF
ablation.

- Original HF FM path was already intact/runnable (iMF was additive) — **nothing to restore**.
- Default behaviour is **unchanged**: threshold defaults to `-1.0` (disabled → falls back to the
  binary `hardflow_activation="all"` = every step). You opt in with `HF_ACT_THRESHOLD`.

## 1. Files changed (all in vendored `HardFlow/`, plus one sbatch)

| file | change |
|---|---|
| `HardFlow/hardflow/config/flow_matching.py` | new field `hardflow_activation_threshold: float = -1.0` (tyro auto-exposes it as `--hardflow_activation_threshold`) |
| `HardFlow/hardflow/models_flow/flow_policy.py` | `hardflow_new_forward`: replaced the binary `all/late` gate with a threshold gate + **final-step guard** |
| `HardFlow/run_scripts/eval_hardflow_new.sh` | `HF_ACT_THRESHOLD` env (default `-1.0` = disabled); encode `_thres<t>` in `exp_name`; pass `--hardflow_activation_threshold` |
| `Slurm_Codes/sbatch/hardflow/eval_threshold_sweep_hardflow.sh` | **new** — sweeps the threshold on the FM checkpoint |

`run/eval.py` needed **no edit** — it builds the config via `tyro.cli(...)`, so the new dataclass
field is a CLI flag automatically.

## 2. The gate (the math-critical part)

`hardflow_new_forward` now resolves a threshold once, then per step:

```python
control_flag = ((t_k + dt) >= activation_threshold) or (k == self.oc_N_steps - 1)
```

- **`or (k == N-1)` is the invariant** — HardFlow's feasibility guarantee (`h(x_N) ≤ 0`) comes from
  the final step alone (`α₁=1, β₁=0 ⇒ x_N = x̂_N*`); the terminal NLP must always run. The guard also
  covers the float case where `t_{k+1}` at the last step isn't exactly 1.0.
- **Back-compat:** `hardflow_activation_threshold < 0` (default `-1.0`) is DISABLED → falls back to
  the binary `hardflow_activation` (`all→0.0`, `late→0.5`). So an existing run with no threshold set
  behaves exactly as before.

Verified (pure-logic, K=10 = the run's `ode_t_steps`): final step always active; solve count monotone
↓ in the threshold — `thr 0.0 → 10 solves`, `0.5 → 6 (last half)`, `1.0 → 1 (terminal-only)`.

## 3. Provenance

Each threshold writes its own eval dir via `exp_name`:

```
logs/avoiding-v0/eval/H16_1e6steps_hardflow_new_10steps_thres0.5/trajectories.csv
```

- disabled (`-1.0`) → the original name `..._hardflow_new_10steps` (baseline, unchanged);
- `0.0/0.5/1.0` → `..._thres0.0 / _thres0.5 / _thres1.0`.

So a sweep never collides, and the full-step baseline (`thres0.0`) sits beside the late variants.

## 4. What is DELIBERATELY different from Gen12 U4

Same idea, different codebase/setup (per the plan's §1 table):
- edits the **vendored HardFlow** (`HardFlow/…`), not `flow_matcher_v3_hardflow/`;
- **original TemporalUnet FM**, H16, avoiding-v0, HardFlow's `avoiding_geometry` + l4casadi-free
  `hardflow_new` + fitted `linear_model.npz` — none of Gen12's DPCC/D3IL machinery;
- the comparison is **HF threshold-on vs HF full-step** (same FM model), not HF-vs-DPCC.

Note on the bridge rule: `_hardflow_common.sh` says "run HardFlow unmodified". U10 does make a
**source edit** to the vendored HardFlow — but additively and behind a disabled-by-default flag,
exactly as the Gen13 iMF work added code. With the threshold unset, HardFlow runs byte-identically to
before.

## 5. Verification (static — nothing executed)

- `flow_matching.py` and `flow_policy.py` compile (`py_compile`).
- `eval_hardflow_new.sh` and the sweep sbatch pass `bash -n`.
- Threshold gate logic: final step always active; monotone solve count.
- `exp_name` encoding: disabled → baseline name; thresholds → `_thres<t>` (awk float check, no `bc`).
- **Not run:** needs the `hardflow_clone` conda env (gym 0.20, tyro, casadi) + the FM checkpoint on
  the cluster. No G-gate script was added for Gen13; the shared threshold math is already covered by
  Gen12's G4, and the run itself (thres 0.0 vs 0.5) is the on-hardware check.

## 6. How to run (cluster)

Pull first (cluster runs committed code), then:

```bash
# U10 threshold sweep on original HF (FM): full-step vs late vs terminal-only
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_threshold_sweep_hardflow.sh
# custom grid:
HF_THRES_GRID="0.0 0.5 0.75 1.0" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_threshold_sweep_hardflow.sh
```

Requires `logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth` (run `train_hardflow.sh` first if
absent). Results: `logs/avoiding-v0/eval/H16_1e6steps_hardflow_new_10steps_thres<t>/`.

**Read:** compare Safety Rate (should stay 1.00 at every threshold — terminal guarantee), Total Steps,
Computation Time, and NLP-solve count (drops with threshold). If safety holds even at `thres1.0`
(terminal-only), HardFlow's per-step NLP is largely unnecessary on H16 avoiding-v0 — a strong
efficiency result.

## 7. Notes / traps
- **Final-step guard is the safety check** — never remove it.
- The l4casadi `hardflow_forward` (guidance_method `hardflow`) still uses its own binary
  `projection_option`; U10 scopes to `hardflow_new` (the paper's canonical black-box algorithm),
  matching Gen12. Extend to the l4casadi path later if needed.
- Console: the run inherits Gen13's `solver_print_level=5` in `eval_hardflow_new.sh`; fewer solves
  means a shorter log, but consider `--solver_print_level 0` for the big sweep (Gen13 hit spam thrice).
- avoiding-v0 geometry + fitted `linear_model.npz` are HardFlow's own — do not swap in Gen12's.
