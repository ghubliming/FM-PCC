# Gen0 Fix2 — restore `diffusion_timestep_threshold` **and** the `post_processing` baseline

**Date:** 2026-08-04
**Dev flag:** `[Gen0fix2]`
**Status:** applied, **not committed**, **not yet validated on cluster**
**Upstream `aux_repo/dpcc` NOT touched** (read-only reference).

Two defects, one origin: upstream DPCC commit `7f09d3a` "Removed unused configs" (2 Dec 2024)
deleted the lines that wired the YAML threshold *and* the lines that defined `post_processing`.
FM-PCC copied the post-cleanup file verbatim and inherited both. This fix restores both, across
every live generation, and follows the paper's own definition:

> "Post-processing methods impose constraints on the generated samples by **modifying them after
> the last denoising step**, usually by solving an optimization problem."
> — DPCC paper, related work

| defect | symptom | fix |
|---|---|---|
| **A. threshold orphaned** | YAML value never reached `Projector`; every run at θ = 0.5 regardless; savepath tagged with a number the sampler never saw | forward `diffusion_timestep_threshold=` at the call site |
| **B. `post_processing` undefined** | no branch → arm inherits the normal schedule → byte-identical duplicate of `dpcc-r` | `threshold = 0.0 if 'post_processing' in variant else …` |
| **C. gate cannot express B** | the bare-float gate returns **zero** projections at θ = 0, so B would have turned `post_processing` into `diffuser` | add the terminal guard (match the form the FMv3 line already uses) |

---

## 1. Evidence

### 1a. Defect A — three thresholds, one set of trajectories

| job | YAML `T` | savepath tag | intended `n_active` (K=20) |
|---|---|---|---|
| 24215 | 0.1 | `H8_K20_T0.1_Dmodels.GaussianDiffusion` | 3 |
| 24226 | 0.05 | `H8_K20_T0.05_Dmodels.GaussianDiffusion` | 2 |
| 24254 | **1** | `H8_K20_T1_Dmodels.GaussianDiffusion` | **20** |

- **39/39 arm × env cells identical** on every metric
- **39/39 `sha256(obs_all.npy)` identical** — same bytes
- 39-minute wall in all three

A 20× span in projection budget cannot leave a 200-step stochastic MPC rollout bit-identical.

Timing corroborates independently. With `a = 8.88 ms/net call` from this run's own `diffuser`
arm and `b ≈ 8 ms/solve`:

| schedule | predicted `s/step` | measured `dpcc-t-tightened` |
|---|---|---|
| n = 20 (θ=1.0) | 0.82 | — |
| **n = 11 (θ=0.5, the default)** | **0.530** | — |
| n = 3 (θ=0.1) | 0.20 – 0.27 | — |
| n = 2 (θ=0.05) | 0.19 – 0.24 | — |
| | | **0.532 / 0.538 / 0.539** |

All three land on the θ=0.5 prediction. And the `projector = None` arm drifted by the same
+1 – 5% as the DPCC arms between jobs: everything moved with the machine, nothing with the
setting.

**Positive control** — the same YAML edit through a script that *does* forward the value
(`FM_v3_ode_selectable_test/…:242`): `dpcc-t-tightened` 0.449 → **0.189** → **0.180** while
`diffuser` stays flat at 0.176 → 0.174 → 0.170. That is what a live threshold looks like.

### 1b. Defect B — `post_processing` is `dpcc-r`

`sha256(obs_all.npy)`, Gen0 baseline run:

| env | `dpcc-r` | `post_processing` | |
|---|---|---|---|
| top-right-hard | `26896f11bc2afddb` | `26896f11bc2afddb` | identical |
| top-left-hard | `87c4847e1734ed86` | `87c4847e1734ed86` | identical |
| both-hard | `75ecaf727e9c4dc9` | `75ecaf727e9c4dc9` | identical |

Same for the `-tightened` pair, and for all four `temp/0408/FMv3ODE` runs.

**Why this matters beyond a duplicate column.** The paper's related work sets post-processing up
as the *contrast case* — the thing that ignores the data likelihood and therefore drifts
off-distribution — against which in-loop projection is argued. With the branch missing, the
"post-processing" baseline **is** in-loop projection. The row meant to represent the alternative
was running the method itself.

### 1c. Defect C — the gate that cannot fire at θ = 0

Three arithmetics for one config key were live in this repo:

| form | expression | `n_active` |
|---|---|---|
| **C** DPCC | `t <= T·K`, `t` counting **down** from `K−1` | `min(floor(T·K) + 1, K)` |
| **A** int + terminal guard | `idx = int((1−T)·K)`; `(loop_idx >= idx) or (loop_idx == K−1)` | `max(K − int((1−T)·K), 1)` |
| **B** float, bare | `loop_idx >= (1−T)·K` | `K − ceil((1−T)·K)` — **can be 0** |

| K | T | C | A | B |
|---|---|---|---|---|
| 20 | **0.0** | 1 | 1 | **0** |
| 20 | 0.05 | 2 | 1 | 1 |
| 20 | 0.5 | 11 | 10 | 10 |
| 10 | 0.25 | 3 | 3 | **2** |
| 10 | 0.05 | 1 | 1 | **0** |

Only **A** guarantees at least one projection. Under **B**, `threshold = 0` means *no projection
at all* — so defect B's fix would have silently converted `post_processing` into `diffuser` on
every form-B path. Hence C is a prerequisite for B, not an optional extra.

---

## 2. Files changed — 17

### 2a. Threshold wiring (defect A) — 2 files

| file | generation | change |
|---|---|---|
| `scripts/eval.py` | Gen0 DPCC state baseline | `diffusion_timestep_threshold=threshold` at the `Projector` call; startup echo of the resolved YAML value |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Gen9 visual avoiding, DPCC arm (`working on`) | same; reads `config/visual_avoiding_eval.yaml`; startup echo |

The Gen9 case was found by the audit, not by the original report, and was worse in one respect:
`config/avoiding-d3il-visual.py:242` tags the savepath `..._T{diffusion_timestep_threshold}_...`,
so that path has been emitting `T`-labelled folders that all ran θ = 0.5.

Note on `_SAMPLING_OVERRIDE_KEYS` (`:191` of that file): it lists
`'diffusion_timestep_threshold'`, but that reconciles the eval config against the checkpoint's
stored `diffusion_config.pkl` — a **different consumer**. No model class anywhere in this repo
stores the threshold; every gate reads `projector.diffusion_timestep_threshold`. Its presence
there is exactly why the omission was easy to miss.

### 2b. `post_processing` restored (defect B) — 10 eval scripts

Restored line, matching upstream `7f09d3a^:scripts/eval.py:130`:

```python
threshold = 0.0 if 'post_processing' in variant else <yaml threshold>
```

then passed as `diffusion_timestep_threshold=threshold`.

| file | YAML | lists `post_processing`? |
|---|---|---|
| `scripts/eval.py` | `projection_eval.yaml` | ✅ |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | `visual_avoiding_eval.yaml` | ✅ |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | `visual_avoiding_eval.yaml` | ✅ |
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | `projection_eval.yaml` | ✅ |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | `projection_eval.yaml` | ✅ |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | `projection_eval.yaml` | ✅ |
| `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | `projection_eval.yaml` | ✅ |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | `meanflow_projection_eval.yaml` | not currently — added for uniformity |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | `alphaflow_projection_eval.yaml` | not currently — added for uniformity |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | `hardflow_projection_eval.yaml` | not currently — added for uniformity (overrides `dpcc_threshold`) |

The last three are inert until the variant is added to their YAML, and cost nothing.

**Already correct, unchanged:** `diffuser_visual_aligning_test:269`,
`fm_visual_aligning_test:276`, `imf_visual_aligning_test:157`,
`mix_visual_aligning_test:297`, `FM_v3_uav_test:752`. Those five independently re-derived
upstream's deleted line — `threshold = 0.0 if 'post_processing' in variant else …` — rather
than copying it.

### 2c. Gate terminal guard (defect C) — 5 model packages

| file | was | now |
|---|---|---|
| `flow_matcher_v3/models/diffusion.py` | form B | form A |
| `flow_matcher_v3_hardflow/models/diffusion.py` | form B | form A |
| `fm_visual_aligning/models/diffusion.py` | form B | form A |
| `fm_visual_avoiding/models/diffusion.py` | form B | form A |
| `mix_visual_aligning/models/fm_diffusion.py` | form B | form A |

```python
# now, identical to the form ode_selectable/meanflow/imeanflow/alphaflow/drifting/uav already use
if projector is not None:
    snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
    near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
else:
    near_end = False
```

> **No past result moves.** A and B agree whenever `(1−T)·K` is an integer, which covers every
> threshold used to date: K=20 with T ∈ {0.5, 0.1, 0.05} → 10, 18, 19; K=100 T=0.5 → 50;
> K=2 T=0.5 → 1. They differ only at **T = 0** (the new `post_processing` case) and on
> non-integer `(1−T)·K` — e.g. K=10, T=0.25: 2 → 3.

This also removes a Gen14 comparability defect. None of the four `Visual*` classes overrides the
sampling loop, so each inherited its base's gate: `diffusion`→C, `mf`/`af`→A, `fm`→**B**. In an
experiment whose entire design is "only the generative engine differs", the `fm` arm was running
a different projection schedule. Now three of four agree; the `diffusion` arm still carries
form C's `+1` (see §4).

`python3 -m py_compile` passes on all 17 files.

---

## 3. Not changed, deliberately

### 3a. `aux_repo/dpcc`

Read-only reference. Untouched. Its state is analysed in
`UPSTREAM_DPCC_same_bug_analysis.md`; the same three lines would fix it there and would be a
reasonable patch to send upstream.

### 3b. Form C's `+1`

DPCC's gate yields `floor(T·K) + 1` — one more projected step than the FM line at the same T
whenever `T·K` is an integer (11 vs 10 at θ=0.5, K=20). **Left as is:** it is DPCC's published
behaviour and changing it would break comparison against the paper.

> **Standing rule: match runs on `n_active`, never on `T`.** A "matched threshold"
> DPCC-vs-FM comparison at K=20, T=0.1 is really **3 solves vs 2**.

### 3c. Five superseded generations

`FM_test`, `FM_v2_test`, `FM_v3_test`, `FM_Unet_v2_test`, `FM_hp_tune_test` (Gen1 / Gen2 /
Gen2-UNetv2 / Gen3-U1 / Gen3-U2 / Gen3-U3, early-to-mid April 2026) still omit the threshold at
their `Projector` call, and `flow_matcher/`, `flow_matcher_unet_v2/`, `flow_matcher_v2/` still
carry form-B gates. Frozen on purpose:

1. The copy-modify convention keeps older generations intact for rollback and A/B.
2. `config/projection_eval.yaml` is **shared** and currently reads `diffusion_timestep_threshold: 1`.
   Wiring them would change what a rollback run produces relative to its own historical numbers
   — a silent behaviour change to archived baselines, for no benefit.

One line each if reactivated.

### 3d. `diffuser_visual_aligning_test/test_projector_b1.py`

Unit test with a hand-built projector; never reads a YAML. Not applicable.

---

## 4. State after this fix

| path | threshold honoured | `post_processing` = 1 final projection |
|---|---|---|
| **Gen0 DPCC state baseline** | ✅ fixed | ✅ fixed |
| **Gen9 visual avoiding — DPCC** | ✅ fixed | ✅ fixed |
| **Gen9 visual avoiding — FM** | ✅ was ok | ✅ fixed (script + gate) |
| **Gen12 HardFlow** | ✅ `[Gen12fix8]` | ✅ fixed (script + gate) |
| **FMv3 ODE / MF / iMF / AF / Drifting** | ✅ was ok | ✅ fixed |
| **Gen14 Visual-Mix** | ✅ was ok | ✅ `diffusion`/`mf`/`af` were ok; **`fm` fixed via gate** |
| Visual aligning — DPCC / FM / iMF | ✅ was ok | ✅ DPCC+iMF were ok; **FM fixed via gate** |
| FMv3 UAV | ✅ was ok | ✅ was ok |
| upstream DPCC HEAD | ❌ | ❌ |
| FM-PCC frozen generations (§3c) | ❌ frozen | ❌ frozen |

---

## 5. Blast radius — what has to be re-run

- **Any DPCC-baseline result from `scripts/eval.py` with a YAML threshold ≠ 0.5** ran at 0.5 and
  is mislabeled. That includes `temp/0408/dpcc/` (24215, 24226) and
  `temp/0408/H8_K20_T1_…` (24254) — **three copies of θ=0.5, not a sweep.** The measurements are
  valid; only the label is wrong. Written up as Part III of
  `logs_in_develop/Gen12/DA/DA_20260803_HardFlow_activation_threshold_0p1.md`.
- **Any `T ≠ 0.5` visual-avoiding DPCC result** — same, via §2a.
- **Every `post_processing` / `post_processing-tightened` column in the avoiding family** is a
  duplicate `dpcc-r` column and must be regenerated before it can be cited as a baseline.
  Affects `Data_Analysis/` exports and the DA-v3 Visualizer matrices.
- **`post_processing` on the FM visual-aligning arm and Gen14's `fm` arm** ran zero projections,
  i.e. was identical to `diffuser`. Also needs regenerating.
- **Unaffected:** everything run at θ = 0.5 through an already-wired path — correct and
  correctly labeled. Parts I and II of the Gen12 DA MD stand: their thresholds all give integer
  `(1−T)·K`, so §2c does not move them.

---

## 6. Validation required (run on cluster)

1. **Gen0, θ = 0.05.** Expect the new `[ eval ] diffusion_timestep_threshold (from YAML) = 0.05`
   line and `dpcc-t-tightened` at **~0.19–0.24 s/step** instead of 0.53.
   *If it still reports ~0.53, defect A's fix is wrong — revert.*
2. **Gen0, θ = 1.** Opposite direction: expect **~0.82 s/step**. Confirm 1 and 2 are no longer
   byte-identical.
3. **`post_processing` ≠ `dpcc-r`.** In the same run, `sha256(obs_all.npy)` must now **differ**
   between the two arms, and `post_processing` should cost ≈ `diffuser` + one solve per step.
   *This is the single cheapest check that defects B and C are both fixed.*
4. **`post_processing` ≠ `diffuser`** on the FM visual-aligning arm and Gen14's `fm` engine —
   the two paths that previously ran zero projections.
5. **Gen9 visual avoiding, DPCC arm**, one job at `T ≠ 0.5` in
   `config/visual_avoiding_eval.yaml`: expect the echo line and a cost change in the `dpcc-*`
   arms with `diffuser` unmoved.
6. **Regression guard:** one run at θ = 0.5, K = 20 on any FMv3 path — results must be
   **unchanged** from before this fix, since `(1−0.5)·20 = 10` is an integer (§2c).

---

## 7. See also

- `UPSTREAM_DPCC_same_bug_analysis.md` — provenance: which upstream commit removed what, whether
  the published DPCC results are affected (they are not), and the per-file audit.
- `GATE_ARITHMETIC_audit.md` — the three gate forms, the divergence table, and how `T = 0` maps
  to "one projection after the last denoising step" in each.
- `logs_in_develop/Gen12/DA/DA_20260803_HardFlow_activation_threshold_0p1.md` §21 — discovery
  write-up and the data.
- `logs_in_develop/Gen12/fix_8/` — the sibling gate-parity fix on the Gen12 HardFlow NLP path.
