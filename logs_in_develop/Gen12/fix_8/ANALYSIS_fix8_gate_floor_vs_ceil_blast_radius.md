# fix_8 — the activation gate uses **ceil** where DPCC and upstream use **floor**

> **Status: ✅ FIXED — see [`CHANGELOG_fix8_gate_rounding_and_dpcc_threshold.md`](./CHANGELOG_fix8_gate_rounding_and_dpcc_threshold.md).**
> This document is the *diagnosis*; the changelog is the *fix*. Code tagged `[Gen12fix8]`.
>
> **Numbering:** this is **fix_8**. `fix_7/` is the (already-landed, unrelated) DPCC-parity batched
> compute work; `fix_6/` is the threshold-polarity change that introduced the gate analysed here.
> Both are untouched.

## TL;DR

fix_6 made HardFlow's activation gate match DPCC's **polarity** (higher θ = more projection). It did
**not** match DPCC's **rounding**. DPCC truncates the boundary with `int()` (floor); HardFlow compares
against the raw float (effectively ceil). Whenever `(1−θ)·K` is not an integer, **HardFlow does
exactly one FEWER projection step than DPCC** — and than upstream HardFlow, which also floors.

**Severity: LOW / latent. It has never fired.** Every Gen12 run on disk sits on an integer boundary.
It is a budget-fairness hazard for *future* runs, not a defect in any existing result.

---

## 1. The three gates

```python
# DPCC (what actually runs in Gen12: the checkpoint's own FlowMatchingODE)
#   flow_matcher_v3_ode_selectable/models/diffusion.py:207-208
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)   # FLOOR
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)

# HardFlow, Gen12 (and its v6/v7 copies)
#   flow_matcher_v3_hardflow/sampling/hardflow_projection.py:454
active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)                            # CEIL

# Upstream HardFlow  aux_repo/HardFlow/hardflow/models_flow/flow_policy.py:864-873
if self.cfg.projection_option == "late":
    if k < self.oc_N_steps // 2:            # FLOOR (integer division); no continuous θ
        projection_flag = False
```

```
n_active(DPCC)     = max( K − floor((1−θ)K), 1 )
n_active(HardFlow) = max( K − ceil ((1−θ)K), 1 )      ← one fewer when (1−θ)K ∉ ℤ
```

Upstream has **no explicit** `or (k == K−1)` guard, but `k = N−1 ≥ N//2` always holds, so its final
step is projected regardless — the guard exists in effect. Gen12 made it explicit (fix_6/U4) and
generalised upstream's hardcoded "half" into a continuous θ. **All three always project ≥ 1 step;
none can ever fall back to unguided.**

### Gen12 HF is the odd one out (θ=0.5)

| K | upstream `late` | DPCC | **Gen12 HF** |
|---|---|---|---|
| 3 | 2 | 2 | **1** ❌ |
| 5 | 3 | 3 | **2** ❌ |
| 7 | 4 | 4 | **3** ❌ |
| 2, 4, 6, 10, 20 | = | = | = ✅ |

---

## 2. Exactly when it fires (the "explosion circle")

**The rule is NOT "odd K".** The rule is:

> **Affected ⟺ `(1−θ)·K` is not an integer *in float arithmetic*.**

"Odd K" is only the special case at θ=0.5. Measured over K=1…20:

| θ | affected K | shape |
|---|---|---|
| **1.00** | *(none)* | always safe — `(1−θ)K = 0` |
| **0.00** | *(none)* | always safe — `(1−θ)K = K` |
| **0.50** | 3, 5, 7, 9, 11, 13, 15, 17, 19 | **odd only** ✔ (the intuition, valid here) |
| 0.35 | 3–19, **including even** | broad |
| 0.25 | 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19 | **includes even** |
| 0.20 | 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19 | **includes even** |
| 0.10 | 11–19 | **includes even** |

Across the full K=1…20 × θ=0.05…1.00 grid, **283 of 400 combinations differ** — always by exactly 1,
always DPCC ≥ HF.

⚠️ **Float trap:** `(1−0.35)·20` evaluates to `12.999999999999998`, not `13`. Check in float, never on
paper:

```python
x = (1.0 - theta) * K
assert x == int(x), f"(1-θ)K = {x!r} not integral — gates differ by one step"
```

### Is K=1 safe?

**Yes — K=1 is safe at every θ.** Verified over θ = 0.00…1.00 in 0.01 steps: **0 of 101 affected.**
Reason: `(1−θ)·1 ∈ [0,1]`, so DPCC's floor gives `n=1` and HF's ceil gives `n=0`, which the
`or (k == K−1)` guard immediately clamps back to **1**. Both arms do exactly one projection, always.
θ is completely inert at K=1.

---

## 3. Blast radius — which generations carry the ceil gate

Grepped repo-wide for both gate styles (excluding `Archived_Codes/`).

### ❌ AFFECTED — 3 files, all the same line, all HardFlow-engine copies

| gen | file | line |
|---|---|---|
| **Gen12** | `flow_matcher_v3_hardflow/sampling/hardflow_projection.py` | 454 |
| **Gen3v6 (MeanFlow)** | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` | 502 |
| **Gen3v7 (α-Flow)** | `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` | 541 |

Confirmed byte-identical gate line and comment in all three — v6/v7 are copy-modify descendants of
Gen12's sampler, so they inherited the deviation verbatim. (Both also have HF wired into their test
folders: `gates_hardflow_meanflow.py`, `gates_hardflow_alphaflow.py`.)

### ✅ NOT AFFECTED — no `hardflow_projection.py`, only the floor gate

`flow_matcher_v3_ode_selectable` (Gen3v2/FMv3ODE), `flow_matcher_v3_imeanflow` (**Gen3v4**),
`flow_matcher_v3_drifting`, `flow_matcher_v3_uav`, `imf_visual_aligning` (Gen8),
**`mix_visual_aligning` (Gen14)**.

> **Gen14 is NOT affected.** Gen14 (`mix_visual_aligning/`, the Visual-Mix-ML four-engine framework)
> has **no HardFlow sampler at all** — only the DPCC-style floor gate in `mf_diffusion.py:284` /
> `af_diffusion.py:342`. Same for **Gen3v4 (iMeanFlow)**. The deviation cannot reach them.

So the "explosion circle" is **exactly Gen12 + Gen3v6 + Gen3v7**, and only via their HardFlow arms.
DPCC/diffuser arms in every generation are on the floor gate and mutually consistent.

---

## 4. Has any real run been damaged? **No.**

Every Gen12 results directory on disk, checked against both gates:

| run dir | `(1−θ)K` | DPCC | HF | verdict |
|---|---|---|---|---|
| `K20_thres0.5_mpc4_n2` (fix_7 headline) | 10.0 | 10 | 10 | ✅ identical |
| `K20_thres0_mpc4_n2` | 20.0 | 1 | 1 | ✅ identical |
| `K20_thres0.5_mpc1_n2` | 10.0 | 10 | 10 | ✅ identical |
| `K20_thres0_mpc1_n2` | 20.0 | 1 | 1 | ✅ identical |
| `K10_thres0_mpc1_n2` | 10.0 | 1 | 1 | ✅ identical |
| `K5_thres0_mpc1_n2` | 5.0 | 1 | 1 | ✅ identical |
| `K2_thres0_mpc1_n2` | 2.0 | 1 | 1 | ✅ identical |

Note `K5` — the one **odd** K ever run — was at **θ=0**, where `(1−0)·5 = 5.0` is integral, so it is
clean too. The two thresholds ever used (0.0 and 0.5) are exactly the two that are safest.

**Nothing to re-run. No published Gen12/v6/v7 number is invalidated**, including every fix_7 timing
conclusion (K=20, θ=0.5 → both arms 10 steps).

---

## 5. Severity

| dimension | impact |
|---|---|
| Projection / NLP math | ✅ none — solver and constraints untouched |
| **Safety** | ✅ **none** — the final-step guard fires in every arm, at every K and θ |
| Existing Gen12 / v6 / v7 results | ✅ none — all on integer boundaries |
| fix_7 timing conclusions | ✅ none |
| **Future runs off an integer boundary** | ⚠️ HF gets **one fewer** solve step than DPCC → silent budget advantage to DPCC in an "equal-cost" comparison |

**Not a severe bug.** It is a *latent parity deviation*: worst case is "HardFlow does one less
projection than intended", never "no projection" and never "wrong projection". It matters only for
the equal-cost premise of the Test_NFE plan — which is why that plan now mandates integer
`(1−θ)·K` for every arm (all of them satisfy it: 18.0, 10.0, 1.0, 0.0).

---

## 6. The fix (✅ APPLIED — tagged `[Gen12fix8]`)

One line, three files, restoring exact **three-way** parity (Gen12 HF ≡ DPCC ≡ upstream HardFlow):

```python
# hardflow_projection.py  (hardflow:454, meanflow:502, alphaflow:541)
- active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
+ active = (k >= int((1.0 - self.activation_threshold) * K)) or (k == K - 1)
```

**Behaviour change:** none at integer boundaries (so **every existing run reproduces bit-identically**,
including all K=20 θ=0.5 and θ=0 data); at non-integer boundaries HF gains the one step it was
missing, matching both references.

**Applied** in all three files under the `[Gen12fix8]` tag, together with the separate
DPCC-threshold restoration. Post-fix parity verified: **0/400 mismatches vs DPCC** across the
K=1..20 x θ=.05..1 grid (was 283/400), and all five (K, θ) combinations with data on disk are
**unchanged**. Full detail in the changelog.

**Open question for the user:** upstream HardFlow only ever offered `all` / `late`, so there is no
upstream ground truth for arbitrary θ. Matching DPCC's floor is the defensible choice *for this
project* (it makes the A/B fair), which is the same argument fix_6 used for polarity.

---

### Verification commands used

```python
def dpcc(K, th): s = int((1.0-th)*K); return len([i for i in range(K) if i >= s or i == K-1])
def hf(K, th):                        return len([k for k in range(K) if k >= (1.0-th)*K or k == K-1])
# K=1: 0/101 θ values differ.  Grid K=1..20 × θ=.05..1.0: 283/400 differ, always by 1, always DPCC ≥ HF.
```

### Code references
- DPCC floor gate: `flow_matcher_v3_ode_selectable/models/diffusion.py:207-208`.
- HF ceil gate: `flow_matcher_v3_hardflow/sampling/hardflow_projection.py:454`;
  `flow_matcher_v3_meanflow/…:502`; `flow_matcher_v3_alphaflow/…:541`.
- Upstream: `aux_repo/HardFlow/hardflow/models_flow/flow_policy.py:864-873`; default
  `projection_option = "all"` at `hardflow/config/flow_matching.py:78`.
- Which class Gen12 loads: `FM_v3_hardflow_test/eval_FM_v3_hardflow.py:148, :171`.
- Test plan that surfaced this: `logs_in_develop/Gen12/Test_NFE/PLAN_hardflow_vs_dpcc_equal_cost_test.md` §1.
- The fix: `logs_in_develop/Gen12/fix_8/CHANGELOG_fix8_gate_rounding_and_dpcc_threshold.md`.
