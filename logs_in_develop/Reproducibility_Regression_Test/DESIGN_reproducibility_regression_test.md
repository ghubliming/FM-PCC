# Reproducibility Regression Test — Design Document

> **Status**: DRAFT — awaiting review  
> **Date**: 2026-08-28  
> **Location**: `logs_in_develop/Reproducibility_Regression_Test/`

---

## TL;DR — The Core Flow (2 Steps)

The whole idea is intentionally simple:

```
Step 1:  Give it old data     →  point to an existing .npz from a known-good eval run
Step 2:  Run quick check      →  re-run ONLY 3 trials of the same setup, diff against old data
                               →  PASS (identical) = no problem, code change is safe
                               →  FAIL (mismatch)  = regression, something broke
                               →  Also checks avg_time per trial is within ±range of golden
```

**Fastest possible usage** — no recording step needed if you already have `.npz` results:

```bash
# Point at old data, run 3 trials, diff. Done in ~30 seconds.
python repro_test/verify.py \
  --golden logs/avoiding-d3il/plan_fm_v3/6/results/halfspace_top-right-hard/dpcc-c.npz \
  --eval-script FM_v3_test/eval_FM_v3.py \
  --seed 6 --variant dpcc-c --halfspace top-right-hard --n-trials 3
```

If same → ✅ zero problem. If different → 🔴 investigate.

---

## 1 — Problem Statement

After many iterations (Gen0 → Gen16, upcoming GEN_X rebuild), any code refactor, dependency upgrade, or folder consolidation can **silently break the eval pipeline** even though training weights are unchanged. A full evaluation sweep (all seeds × all variants × all envs) is expensive and slow. We need a **fast, sampled regression gate** that answers:

> "Does the current eval code, given the **exact same** checkpoint + seed + initial noise + env state, produce **bit-identical** (or ε-identical) trajectories and metrics as the recorded golden reference, and is per-step compute time still within the expected performance range?"

### What this is NOT

- **Not a re-training check** — training is assumed fine. Only the inference / eval path is verified.
- **Not a full benchmark** — we sample a tiny subset (≤ 3 trials × 1 seed × 1 variant per env) as a fast smoke test.
- **Not a statistical significance test** — we compare deterministic outputs under locked randomness.
- **Not a precise latency benchmark** — the `avg_time` check only catches **large** performance regressions (e.g., 2× slowdown), not micro-optimizations.

---

## 2 — Key Insight: What Makes Eval Deterministic?

Looking at the existing eval scripts (e.g., [`scripts/eval.py`](file:///workspaces/FM-PCC/scripts/eval.py), [`FM_v3_test/eval_FM_v3.py`](file:///workspaces/FM-PCC/FM_v3_test/eval_FM_v3.py)), determinism is controlled by:

| Factor | How it's set | Where |
|---|---|---|
| **Training seed** | `args.seed` | `Parser().parse_args(seed=seed)` |
| **Per-trial torch seed** | `torch.manual_seed(i)` | eval loop, line ~298 |
| **Environment seed** | `env_seeds[i]` or `i` | eval loop, line ~299 |
| **Checkpoint epoch** | `args.diffusion_epoch` | YAML / CLI |
| **ODE solver** | `flow_steps`, solver type | YAML / code |
| **Projection variant** | `variant` string | YAML |
| **Constraint config** | halfspace, obstacle, bounds | YAML |
| **Normalizer** | `dataset.normalizer` | loaded with checkpoint |

If **all of the above are frozen**, the eval output is deterministic (modulo CUDA non-determinism, which we handle with tolerances).

---

## 3 — Architecture: The "Golden Snapshot" System

The system has two modes — **give-it-old-data** (primary, zero setup) and **managed snapshots** (for batch/CI):

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  MODE A: "Just give me old data" (ad-hoc, fastest)              │
  │  ────────────────────────────────────────────────────────────    │
  │  INPUT:  path to an existing .npz from a known-good run         │
  │          + the eval setup (script, seed, variant, halfspace)    │
  │                                                                  │
  │  WHAT:   re-run 3 trials of the SAME setup on current code      │
  │          diff the fresh .npz against the old .npz               │
  │                                                                  │
  │  OUTPUT: PASS / FAIL per trial                                  │
  │          (if PASS → code change is safe, zero problem)          │
  │                                                                  │
  │  MODE B: Managed golden snapshots (batch/CI, more formal)       │
  │  ────────────────────────────────────────────────────────────    │
  │  Step 1: RECORD — freeze .npz + metadata from known-good code  │
  │  Step 2: VERIFY — re-run all cells, diff against frozen golden  │
  │  Step 3: UPDATE — re-record when intentional changes are made   │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **Mode A is the 80% case.** You already have `.npz` files from past eval runs in `logs/`. Just point the verify script at one and run. No recording step, no manifest, no setup. Mode B is for when you want to formalize multiple cells into a repeatable batch.

---

## 4 — Golden Snapshot Specification

### 4.1 — What to Sample (the "Regression Cell")

Each **regression cell** is identified by the tuple:

```
(env_key, ml_model, projector_variant, seed, halfspace_variant)
```

We do NOT test every cell in the 4-axis matrix. We pick a **representative subset** that covers the critical code paths:

| # | `env_key` | `ml_model` | `projector_variant` | `seed` | `halfspace` | Why |
|---|---|---|---|---|---|---|
| 1 | `avoiding-d3il` | DPCC baseline (Gen0) | `dpcc-c` | `6` | `top-right-hard` | Baseline: catches DPCC projection regressions |
| 2 | `avoiding-d3il` | FMv3ODE (Gen3) | `dpcc-c` | `6` | `top-right-hard` | Core FM path |
| 3 | `avoiding-d3il` | FMv3ODE (Gen3) | `diffuser` (no projection) | `6` | `top-right-hard` | Unconstrained path — catches model-only regressions |
| 4 | `avoiding-d3il` | MeanFlow (Gen3v6) | `dpcc-c` | `6` | `both-hard` | MF engine + hardest constraint |
| 5 | `avoiding-d3il` | α-Flow (Gen3v7) | `dpcc-c` | `6` | `both-hard` | AF engine |
| 6 | `visual_aligning` | FM (Gen7) | `dpcc` | `6` | n/a | Visual backbone + FM + DPCC |
| 7 | `visual_avoiding` | FM (Gen9) | `dpcc` | `6` | n/a | Visual avoiding |
| 8 | `uav_*` | FM (Gen11) | `dpcc` | `6` | n/a | UAV closed-loop |

> [!TIP]
> Start with cells **1–3** (state-only avoiding, cheapest to run). Add visual/UAV cells only after the state-only path is validated. Visual cells require D3IL rendering and take ~10x longer.

### 4.2 — How Many Trials per Cell

- **N_SAMPLE = 3** trials per cell (trial indices 0, 1, 2)
- This is enough to catch most code-path regressions while keeping the total runtime under 2–5 minutes for state-only cells

### 4.3 — Golden Snapshot File Format

Each cell produces one directory under `golden_snapshots/`:

```
golden_snapshots/
├── manifest.json                     ← master index of all cells + metadata
├── avoiding__dpcc_baseline__dpcc-c__s6__top-right-hard/
│   ├── golden.npz                    ← the reference outputs
│   ├── cell_meta.json                ← full config that produced this
│   └── rng_state.pt                  ← torch + numpy RNG state at trial start
├── avoiding__fmv3ode__dpcc-c__s6__top-right-hard/
│   ├── golden.npz
│   ├── cell_meta.json
│   └── rng_state.pt
└── ...
```

#### `cell_meta.json` Schema

```json
{
  "cell_id": "avoiding__fmv3ode__dpcc-c__s6__top-right-hard",
  "recorded_at": "2026-08-28T14:00:00Z",
  "git_commit": "abc1234",
  "git_branch": "main",
  "python_version": "3.10.12",
  "torch_version": "2.1.0+cu118",
  "numpy_version": "1.26.4",

  "env_key": "avoiding-d3il",
  "ml_model": "fmv3ode",
  "eval_script": "FM_v3_test/eval_FM_v3.py",
  "config_yaml": "config/projection_eval.yaml",
  "projector_variant": "dpcc-c",
  "seed": 6,
  "halfspace_variant": "top-right-hard",
  "n_trials": 3,
  "trial_indices": [0, 1, 2],
  "diffusion_epoch": "best",
  "checkpoint_path": "logs/avoiding-d3il/plan_fm_v3/6/diffusion/...",

  "tolerance": {
    "obs_atol": 1e-5,
    "obs_rtol": 1e-4,
    "act_atol": 1e-5,
    "act_rtol": 1e-4,
    "metric_exact": ["n_success", "n_success_and_constraints",
                     "collision_free_completed"],
    "time_rel_tol": 0.50,
    "time_hard_fail": false
  }
}
```

#### `golden.npz` Contents

Mirrors the existing eval `.npz` but only for the sampled trials:

| Key | Shape | Dtype | Comparison |
|---|---|---|---|
| `obs_all` | `(N_SAMPLE, T_max, obs_dim)` | `float64` | ε-match (`atol=1e-5, rtol=1e-4`) |
| `act_all` | `(N_SAMPLE, T_max, act_dim)` | `float64` | ε-match |
| `n_success` | `(N_SAMPLE,)` | `float64` | exact (cast to int) |
| `n_success_and_constraints` | `(N_SAMPLE,)` | `float64` | exact |
| `n_violations` | `(N_SAMPLE,)` | `float64` | exact |
| `total_violations` | `(N_SAMPLE,)` | `float64` | ε-match (`atol=1e-4`) |
| `collision_free_completed` | `(N_SAMPLE,)` | `float64` | exact |
| `n_steps` | `(N_SAMPLE,)` | `float64` | exact |
| `avg_time` | `(N_SAMPLE,)` | `float64` | **range** (within ±`time_rel_tol` of golden, per trial) |
| `sampled_traj_first` | `(N_SAMPLE, B, H, dim)` | `float64` | ε-match (first MPC plan) |

---

## 5 — Tolerance Strategy

### 5.1 — Why Not Exact Match?

CUDA floating-point operations are **not bitwise deterministic** across:
- Different GPU architectures (A100 vs V100 vs T4)
- Different CUDA versions
- Different cuDNN autotuning states

### 5.2 — Tolerance Tiers

| Tier | What | Tolerance | Rationale |
|---|---|---|---|
| **Exact** | Integer metrics (`n_success`, `collision_free_completed`, `n_violations` count) | `==` | These are decision outcomes — if they flip, something is broken |
| **Tight ε** | Trajectories (`obs_all`, `act_all`) | `atol=1e-5, rtol=1e-4` | Catches numerical drift but allows GPU non-determinism |
| **Loose ε** | Aggregate floats (`total_violations`) | `atol=1e-3` | `total_violations` accumulates rounding |
| **Range** | `avg_time` (per-trial) | `rel_tol=0.50` (default) | Pass if `avg_time[i]` is within `golden ± 50%`. Catches big perf regressions without being noise-sensitive (see §5.4) |

### 5.3 — CUDA Determinism Settings

The verify script should set these before loading any model:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

> [!WARNING]
> Some operations (e.g., `torch.scatter_add_`) have no deterministic CUDA kernel. The `warn_only=True` flag allows these to proceed with a warning rather than crashing.

### 5.4 — AvgTime Sample Test (Performance Range Check)

#### Key Finding: `avg_time` Is Already Per-Trial

Looking at the eval scripts (e.g., [`eval_FM_v3.py:L139-209`](file:///workspaces/FM-PCC/FM_v3_test/eval_FM_v3.py#L139-L209), [`eval.py:L288-383`](file:///workspaces/FM-PCC/scripts/eval.py#L288-L383)), `avg_time` is stored as:

```python
avg_time = np.zeros(n_trials)           # one entry per trial
for i in range(n_trials):
    for _ in range(max_steps):
        start = time.time()
        action, samples = policy(...)   # MPC inference
        avg_time[i] += time.time() - start
    avg_time[i] /= n_steps[i]           # → avg compute time per step for trial i
```

So the `.npz` already contains `avg_time` as a **`(n_trials,)` array** — each element is the average per-step inference time for that trial. This means:

- ✅ **We can compare individual trials** (e.g., 2–3 sampled trials) — no need to run many seeds and average
- ✅ **No full evaluation needed** — the same 3 trials used for the correctness check also provide timing data
- ❌ **Cannot require exact match** — wall-clock time varies with system load, GPU thermal state, etc.

#### Strategy: Range-Based Pass/Fail

```
  For each sampled trial i:
    golden_t = golden_avg_time[i]     (from the .npz recorded on known-good code)
    fresh_t  = fresh_avg_time[i]      (from the re-run on current code)

    PASS if:   golden_t × (1 - rel_tol) ≤ fresh_t ≤ golden_t × (1 + rel_tol)
    WARN if:   golden_t × (1 + rel_tol) < fresh_t ≤ golden_t × (1 + 2×rel_tol)
    FAIL if:   fresh_t > golden_t × (1 + 2×rel_tol)

  Default: rel_tol = 0.50 (±50%).
  Being faster than golden is always PASS (performance improved).
```

> [!NOTE]
> The default `rel_tol = 0.50` is intentionally generous. The goal is to catch **major** performance regressions (e.g., accidentally switching from 10 ODE steps to 100, or loading a model twice) — not micro-level timing jitter. You can tighten to `0.20` if running on a dedicated GPU with no contention.

#### AvgTime Verdict: Soft-Fail vs Hard-Fail

By default, the avg_time check is a **soft-fail** — it reports WARN/FAIL in the output but does **not** affect the exit code. The rationale:

1. Timing is inherently noisy — a cold GPU or background process can cause a 30% swing
2. The correctness check (obs/act/metrics) is the hard gate
3. A WARN on timing prompts investigation; a FAIL on timing logs the regression but doesn't block merge

To make it a hard gate (e.g., in CI with dedicated hardware):

```bash
python repro_test/verify.py --all --ci --time-hard-fail
```

#### Minimum Trials Needed

Since `avg_time` is per-trial, we need **at least 2 trials** to have any statistical power. The default N_SAMPLE=3 is sufficient:

| N_SAMPLE | Timing confidence | Notes |
|---|---|---|
| 1 | Very low | Single-point comparison, too noisy |
| 2 | Low-medium | Can detect 2× regressions |
| **3** (default) | **Medium** | **Recommended minimum — catches >50% regressions reliably** |
| 5+ | High | Only needed for tight `rel_tol` (<0.20) |

---

## 6 — CLI Interface (Proposed)

### 6.1 — Quick Check with Old Data (Primary Usage — Mode A)

The most common case: **you have an old `.npz`, you just want to check if current code still produces the same output.**

```bash
# Point at old data, run 3 trials under the same setup, diff. ~30s.
python repro_test/verify.py \
  --golden logs/avoiding-d3il/plan_fm_v3/6/results/halfspace_top-right-hard/dpcc-c.npz \
  --eval-script FM_v3_test/eval_FM_v3.py \
  --seed 6 --variant dpcc-c --halfspace top-right-hard --n-trials 3

# Or give a directory containing multiple old .npz files:
python repro_test/verify.py \
  --golden-dir logs/avoiding-d3il/plan_fm_v3/6/results/halfspace_top-right-hard/ \
  --eval-script FM_v3_test/eval_FM_v3.py \
  --seed 6 --halfspace top-right-hard --n-trials 3
# (runs verify for every .npz in the directory — each variant is one check)
```

Each `.npz` already contains `obs_all`, `act_all`, `n_success`, etc. — the verify script loads the old data, re-runs the same eval setup for 3 trials, and diffs. **No recording step needed.**

### 6.2 — Managed Batch Verify (Mode B — Formal / CI)

```bash
# Verify all cells defined in cells_config.yaml (~5 min for state-only)
python repro_test/verify.py --all

# Verify a specific cell
python repro_test/verify.py --cell avoiding__fmv3ode__dpcc-c__s6__top-right-hard

# Verbose mode: print per-trial diffs
python repro_test/verify.py --all --verbose

# CI mode: exit code 0 on pass, 1 on any failure
python repro_test/verify.py --all --ci
```

### 6.3 — Recording Golden Snapshots (Mode B Setup)

```bash
# Freeze from existing .npz (most common — no re-run, just copies + adds metadata)
python repro_test/record_golden.py \
  --from-npz logs/avoiding-d3il/plan_fm_v3/6/results/dpcc-c.npz \
  --cell-id avoiding__fmv3ode__dpcc-c__s6__top-right-hard

# Record all cells from scratch (runs eval)
python repro_test/record_golden.py --all

# After intentional code changes, re-record and annotate
python repro_test/record_golden.py --all --reason "Gen3v6 MeanFlow eval refactor"
```

---

## 7 — Verification Report Format

The verify script outputs a human-readable report:

```
═══════════════════════════════════════════════════════════════
  REPRODUCIBILITY REGRESSION TEST — 2026-08-28 14:42:00
  Git: abc1234 (main)
  Golden recorded at: def5678 (2026-08-20)
═══════════════════════════════════════════════════════════════

  [PASS] avoiding__dpcc_baseline__dpcc-c__s6__top-right-hard
         3/3 trials match (max obs delta = 2.1e-7)
         ⏱  avg_time: [0.031, 0.029, 0.030] vs golden [0.028, 0.028, 0.029] → PASS (within ±50%)

  [PASS] avoiding__fmv3ode__dpcc-c__s6__top-right-hard
         3/3 trials match (max obs delta = 1.8e-7)
         ⏱  avg_time: [0.042, 0.043, 0.041] vs golden [0.040, 0.039, 0.041] → PASS

  [FAIL] avoiding__fmv3ode__diffuser__s6__top-right-hard
         Trial 1: n_success MISMATCH (golden=1, got=0)
         Trial 1: obs_all max delta = 0.0342 (> atol=1e-5)
         Trial 2: OK
         Trial 3: obs_all max delta = 0.0087 (> atol=1e-5)
         ⏱  avg_time: [0.038, 0.040, 0.039] vs golden [0.037, 0.038, 0.037] → PASS

  [WARN] avoiding__meanflow__dpcc-c__s6__bh  (timing)
         3/3 trials match (max obs delta = 3.4e-7)
         ⏱  avg_time: [0.091, 0.088, 0.090] vs golden [0.055, 0.054, 0.056] → WARN (+62%)

  [SKIP] visual_aligning__fm__dpcc__s6  (no golden snapshot found)

═══════════════════════════════════════════════════════════════
  RESULT: 2 PASS / 1 FAIL / 1 WARN(time) / 1 SKIP
  Exit code: 1
═══════════════════════════════════════════════════════════════
```

---

## 8 — Integration with GEN_X Rebuild

This system is specifically designed to survive the GEN_X unified rebuild:

1. **Golden snapshots are checkpoint-agnostic** — they record raw numpy outputs, not model internals
2. **Cell metadata includes the eval script path** — when GEN_X replaces `FM_v3_test/eval_FM_v3.py` with `scripts/eval.py --model fm`, the manifest is updated but the golden `.npz` stays the same
3. **The verify script is self-contained** — it only needs `numpy` and `torch` to compare outputs; it does NOT import any `fmpcc` / `diffuser` / `flow_matcher_*` code
4. **Migration path**:
   - Before rebuild: record golden snapshots on every active Gen with the current eval scripts
   - After rebuild: re-run verify against the same checkpoints using the new unified `eval.py`
   - Any mismatch = the rebuild changed eval behavior (a bug unless intentional)

---

## 9 — Directory Layout

```
logs_in_develop/Reproducibility_Regression_Test/
├── DESIGN_reproducibility_regression_test.md    ← this document
├── golden_snapshots/                            ← gitignored (large), kept locally
│   ├── manifest.json
│   └── <cell_id>/
│       ├── golden.npz
│       ├── cell_meta.json
│       └── rng_state.pt
├── repro_test/                                  ← the scripts
│   ├── record_golden.py                         ← Phase 1: record
│   ├── verify.py                                ← Phase 2: verify
│   ├── cells_config.yaml                        ← defines which cells to test
│   └── compare.py                               ← comparison utilities
└── reports/                                     ← verification run reports
    └── verify_<timestamp>.txt
```

---

## 10 — `cells_config.yaml` (Starter Template)

```yaml
# Reproducibility Regression Test — Cell Definitions
# Each cell = one (env, model, variant, seed, halfspace) tuple to test

defaults:
  n_trials: 3
  trial_indices: [0, 1, 2]
  tolerance:
    obs_atol: 1.0e-5
    obs_rtol: 1.0e-4
    act_atol: 1.0e-5
    act_rtol: 1.0e-4
    metric_exact:
      - n_success
      - n_success_and_constraints
      - collision_free_completed
      - n_violations
      - n_steps
    metric_loose_atol: 1.0e-3    # for total_violations
    time_rel_tol: 0.50           # ±50% range for avg_time per trial
    time_hard_fail: false        # if true, timing FAIL affects exit code
    skip_keys:
      - args

cells:
  # -- State-Only Avoiding (cheapest, ~30s per cell) --
  - cell_id: avoiding__dpcc_baseline__dpcc-c__s6__trh
    env_key: avoiding-d3il
    ml_model: dpcc_baseline
    eval_script: scripts/eval.py
    config_yaml: config/projection_eval.yaml
    projector_variant: dpcc-c
    seed: 6
    halfspace_variant: top-right-hard
    checkpoint_path: logs/avoiding-d3il/diffusion/H32_T20/dropout0.25/100

  - cell_id: avoiding__fmv3ode__dpcc-c__s6__trh
    env_key: avoiding-d3il
    ml_model: fmv3ode
    eval_script: FM_v3_test/eval_FM_v3.py
    config_yaml: config/projection_eval.yaml
    projector_variant: dpcc-c
    seed: 6
    halfspace_variant: top-right-hard
    checkpoint_path: logs/avoiding-d3il/plan_fm_v3/6/diffusion/...

  - cell_id: avoiding__fmv3ode__diffuser__s6__trh
    env_key: avoiding-d3il
    ml_model: fmv3ode
    eval_script: FM_v3_test/eval_FM_v3.py
    config_yaml: config/projection_eval.yaml
    projector_variant: diffuser
    seed: 6
    halfspace_variant: top-right-hard

  - cell_id: avoiding__meanflow__dpcc-c__s6__bh
    env_key: avoiding-d3il
    ml_model: meanflow
    eval_script: FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py
    config_yaml: config/meanflow_projection_eval.yaml
    projector_variant: dpcc-c
    seed: 6
    halfspace_variant: both-hard

  - cell_id: avoiding__alphaflow__dpcc-c__s6__bh
    env_key: avoiding-d3il
    ml_model: alphaflow
    eval_script: FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py
    config_yaml: config/alphaflow_projection_eval.yaml
    projector_variant: dpcc-c
    seed: 6
    halfspace_variant: both-hard

  # -- Visual Aligning (heavier, ~2 min per cell) --
  - cell_id: visual_aligning__fm__dpcc__s6
    env_key: visual_aligning
    ml_model: fm_visual
    eval_script: fm_visual_aligning_test/eval_fm_visual_aligning.py
    config_yaml: config/visual_aligning_eval.yaml
    projector_variant: dpcc
    seed: 6

  # -- Visual Avoiding --
  - cell_id: visual_avoiding__fm__dpcc__s6
    env_key: visual_avoiding
    ml_model: fm_visual_avoiding
    eval_script: fm_visual_avoiding_test/eval_fm_visual_avoiding.py
    config_yaml: config/visual_avoiding_eval.yaml
    projector_variant: dpcc
    seed: 6

  # -- UAV (Gen11) --
  - cell_id: uav__fm__dpcc__s6
    env_key: uav
    ml_model: fm_uav
    eval_script: FM_v3_uav_test/eval_FM_v3_uav.py
    config_yaml: config/uav_projection.yaml
    projector_variant: dpcc
    seed: 6
```

---

## 11 — Workflow: How to Use This in Practice

### 11.1 — First-Time Setup (Once, on Known-Good Code)

```bash
# 1. Make sure you're on a commit with known-good eval results
git checkout <known-good-commit>

# 2. Record golden snapshots for the cheap cells first
python repro_test/record_golden.py \
  --cells avoiding__dpcc_baseline__dpcc-c__s6__trh \
         avoiding__fmv3ode__dpcc-c__s6__trh \
         avoiding__fmv3ode__diffuser__s6__trh

# 3. Verify they pass immediately (sanity check)
python repro_test/verify.py --all
# Expected: all PASS
```

### 11.2 — After a Code Change

```bash
# Quick smoke test (~2 min)
python repro_test/verify.py --all --ci
# Exit code 0 -> safe to merge
# Exit code 1 -> regression detected, investigate
```

### 11.3 — After an Intentional Eval Change

```bash
# 1. Run verify to confirm what changed
python repro_test/verify.py --all --verbose

# 2. Review the diff report — make sure only EXPECTED cells fail

# 3. Re-record the affected cells
python repro_test/record_golden.py \
  --cells <affected_cell_ids> \
  --reason "Refactored projection threshold logic (Gen0fix3)"

# 4. Commit new golden snapshots
git add golden_snapshots/
git commit -m "golden update: refactored projection threshold (Gen0fix3)"
```

### 11.4 — Before GEN_X Rebuild

```bash
# Record ALL cells on the old code
python repro_test/record_golden.py --all

# After the rebuild, verify with new code paths
# (update eval_script paths in cells_config.yaml to point to unified eval.py)
python repro_test/verify.py --all --verbose
```

---

## 12 — "From-NPZ" Bootstrap: No Re-Run Needed

> [!IMPORTANT]
> You likely already have valid `.npz` results from past eval runs sitting in `logs/`. The `--from-npz` mode can extract golden snapshots from these **without re-running any eval**, as long as you know which commit produced them.

```bash
# Bootstrap from existing results
python repro_test/record_golden.py \
  --from-npz logs/avoiding-d3il/plan_fm_v3/6/results/halfspace_top-right-hard/dpcc-c.npz \
  --cell-id avoiding__fmv3ode__dpcc-c__s6__trh \
  --git-commit $(git rev-parse HEAD) \
  --n-trials 3
```

This reads the first 3 trials from the existing `.npz` and writes them as the golden reference. The only caveat is that you must verify the commit that produced the `.npz` is actually correct.

---

## 13 — Future: CI / Pre-Commit Hook Integration

Once the golden snapshots are stable, this can become a git pre-push hook or CI step:

```yaml
# .github/workflows/repro-test.yml (future)
name: Reproducibility Gate
on: [push, pull_request]
jobs:
  repro-check:
    runs-on: self-hosted  # needs GPU + checkpoints
    steps:
      - uses: actions/checkout@v4
      - run: python repro_test/verify.py --all --ci
```

> [!NOTE]
> This requires the CI runner to have access to model checkpoints. For cloud CI, consider storing a minimal "test checkpoint" (e.g., a small model trained for 5 epochs) as a CI artifact.

---

## 14 — Comparison: `verify.py` Core Logic (Pseudocode)

```python
def verify_cell(cell_config, golden_dir):
    """Compare fresh eval output against golden snapshot."""
    golden = np.load(golden_dir / "golden.npz", allow_pickle=True)
    meta = json.load(open(golden_dir / "cell_meta.json"))
    tol = meta["tolerance"]

    # Re-run eval (or load from fresh .npz)
    fresh = run_eval_for_cell(cell_config)

    results = []
    time_results = []

    # 1. Exact-match metrics
    for key in tol["metric_exact"]:
        g, f = golden[key], fresh[key]
        match = np.array_equal(g.astype(int), f.astype(int))
        results.append(CompareResult(key, "exact", match,
                                     detail=f"golden={g}, got={f}" if not match else ""))

    # 2. Tight epsilon-match for trajectories
    for key in ["obs_all", "act_all"]:
        for trial_idx in range(meta["n_trials"]):
            g = golden[key][trial_idx]
            f = fresh[key][trial_idx]
            max_delta = np.max(np.abs(g - f))
            match = np.allclose(g, f, atol=tol[f"{key.split('_')[0]}_atol"],
                                      rtol=tol[f"{key.split('_')[0]}_rtol"])
            results.append(CompareResult(
                f"{key}[{trial_idx}]", "epsilon", match,
                detail=f"max_delta={max_delta:.2e}"))

    # 3. AvgTime range check (per-trial)
    # avg_time is (n_trials,) — avg compute time per step for each trial
    rel_tol = tol.get("time_rel_tol", 0.50)
    hard_fail = tol.get("time_hard_fail", False)
    if "avg_time" in golden and "avg_time" in fresh:
        g_times = golden["avg_time"]
        f_times = fresh["avg_time"]
        for trial_idx in range(min(len(g_times), len(f_times))):
            gt, ft = g_times[trial_idx], f_times[trial_idx]
            if gt <= 0:  # skip if golden has no valid time
                continue
            ratio = ft / gt
            in_range = (1 - rel_tol) <= ratio <= (1 + rel_tol)
            warn_zone = ratio <= (1 + 2 * rel_tol)
            if in_range:
                verdict = "PASS"
            elif warn_zone:
                verdict = "WARN"
            else:
                verdict = "FAIL"
            time_results.append(CompareResult(
                f"avg_time[{trial_idx}]", "range", in_range,
                detail=f"golden={gt:.4f}, got={ft:.4f}, ratio={ratio:.2f}, verdict={verdict}"))

    # 4. Skip keys
    # args -- not compared

    correctness_passed = all(r.match for r in results)
    time_passed = all(r.match for r in time_results)
    # timing only affects exit code if time_hard_fail is set
    passed = correctness_passed and (time_passed if hard_fail else True)
    return CellVerdict(cell_id=cell_config["cell_id"],
                       passed=passed, results=results,
                       time_results=time_results)
```

---

## 15 — Open Questions

> [!IMPORTANT]
> **Q1**: Should we store golden snapshots in git (LFS) or keep them local-only?
> - **Pro git**: reproducible across machines, survives laptop wipes
> - **Con git**: `.npz` files can be 1-10 MB each; 8 cells x 10 MB = 80 MB in LFS
> - **Recommendation**: Use git LFS for the state-only cells (~small), keep visual/UAV cells local-only with a download script

> [!IMPORTANT]
> **Q2**: Should we also snapshot intermediate ODE solver states (per-step `x_t`) for deeper debugging?
> - **Pro**: pinpoints exactly where divergence starts (step 3 vs step 15 of the ODE)
> - **Con**: much larger files, and current eval scripts don't expose this
> - **Recommendation**: Defer to Phase 2; the per-trial obs/act trajectories already pinpoint "which trial diverged" -- ODE internals are a debugging aid, not a gate

> [!IMPORTANT]
> **Q3**: How to handle the `--from-npz` bootstrap when existing `.npz` files have variable-length object arrays?
> - Current eval saves `obs_all=np.array(obs_all, dtype=object)` -- trials can have different lengths
> - **Plan**: pad to max length with NaN, compare only non-NaN positions

> [!IMPORTANT]
> **Q4**: What `time_rel_tol` to use for the avg_time range check?
> - `0.50` (±50%) is safe for shared/cloud GPUs — catches only gross regressions (e.g., 10→100 ODE steps)
> - `0.20` (±20%) is tighter — usable on dedicated hardware with stable clocks
> - **Recommendation**: Ship with `0.50` default, let users override in `cells_config.yaml`
> - Should the initial golden `avg_time` be recorded as a single run or the median of 3 runs? Single is simpler and consistent with the correctness snapshot; median is more robust to outliers. Current plan: **use the same single run** that records the golden `.npz` (already has `avg_time`).

---

## 16 — Summary

| Property | Value |
|---|---|
| **What** | Fast regression gate for eval-code reproducibility |
| **When to run** | After any code change, before merge, before/after GEN_X rebuild |
| **Runtime** | ~2 min (state-only) to ~10 min (all cells) |
| **False positive risk** | Low -- CUDA non-determinism handled by epsilon tolerances |
| **Maintenance** | Re-record golden snapshots only on intentional eval changes |
| **Prerequisite** | Trained checkpoints accessible on the machine |
