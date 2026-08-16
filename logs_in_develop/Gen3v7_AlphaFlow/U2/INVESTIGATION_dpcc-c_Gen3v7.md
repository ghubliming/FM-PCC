# Investigation — Gen3v7 (α-Flow) `dpcc-c` defects

**Date:** 2026-07-30 · **Scope:** the `dpcc-c` / `dpcc-c-tightened` arms of **both** Gen3v7 backbones,
across the full K-sweep, from the raw `.npz` in `temp/`:

| arm | backbone | export | jobs | snapshot |
|---|---|---|---|---|
| **U1** | `bbdit` (iMF-DiT) | `temp/Gen3V7/H8_…_bbdit_…` | train 23759 / eval 23786 | 2026-07-24 |
| **U2** | `bbsit` (α-Flow's own SiT) | `temp/2807/2807/H8_…_bbsit_…` | train 23929 / eval 23930 | 2026-07-27 |

Both: seed 6, 2 trials, 3 halfspace scenarios, K ∈ {1, 2, 5, 10}, mpc batch 4, H=8, threshold 0.5.
**288 rollouts / 27 400 decoded candidate plans** in total.

Companion to Gen3v6's
[`INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../../Gen3v6_MeanFlow/U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md)
(referred to below as **v6-INV**). Read that first — this document assumes its vocabulary
(*collapsed candidate*, *horizon span*, *selection rule*).

---

## Verdict

Gen3v7 has **two distinct `dpcc-c` defects**, and the user's reading is right on both counts —
Gen3v7's headline failure is *not* Gen3v6's freeze, and it *is* confined to plain `dpcc-c`:

| | **Defect A — boundary hugging** | **Defect B — start-pose freeze** |
|---|---|---|
| symptom | robot travels normally, then rides the constraint surface and crosses it; commanded setpoint runs away from the robot by up to **0.55** | robot never leaves the start cell; 197–199 of 200 actions are a literal zero |
| affects | **plain `dpcc-c` only** — `dpcc-c-tightened` is clean in **24/24** cells | `dpcc-c` **and** `dpcc-c-tightened`, identically |
| backbones | **both** `bbdit` and `bbsit` | **`bbsit` only** (`bbdit` has 0.00–0.06 % collapse at every K) |
| K | all K; worst at K=1/2, fades by K=10 | **K=2 only** |
| cause | `-c`'s cost is **identically zero on the whole feasible set** (`projection.py:145`) ⇒ argmin is indifferent to clearance and drifts onto the boundary | a degenerate "stay put" mode **localised to a basin of radius ≈0.01 around the start pose**, present only at the `(r,t)` coordinates K=2 visits |
| same as Gen3v6? | **new** — Gen3v6 never showed this | **yes**, same signature, ~⅓ the rate (17 % vs 28 %) |

**The single most important new result** is in §3.3: v6-INV concluded the collapse is *"i.i.d. in the
noise draw, not state-driven"*. Gen3v7 supplies the control v6-INV lacked — non-frozen rollouts from
the same checkpoint, K, and seed — and that conclusion is **wrong**. The collapse is sharply
state-conditioned: **17 % inside 0.01 of the start pose, and exactly 0.00 % in 15 964 candidates
beyond it.** The freeze is a self-reinforcing trap, not a fixed-probability lottery — which makes it
cheap to break.

---

## 0. Method and self-validation

Everything below is decoded from `obs_all` / `sampled_trajectories_all` / `act_all` in the `.npz`
files already produced on the cluster; nothing was executed locally. Two independent validations
were run before any number here was trusted:

1. **Constraint geometry reproduced exactly.** The active constraint set per scenario is *one*
   halfspace plus *one* obstacle, selected by index in `eval_flow_matching_v3_alphaflow.py:88-96`
   (`obstacle_constraints[3/4/5]` for TL/TR/BH). Re-implementing the eval's own violation loop
   (`:363-382`) against `obs_all` reproduces the stored `n_violations` and `total_violations`
   **exactly in 191 of 192 runs** (the one miss is a 3e-3 rounding on a final step). An earlier
   attempt that used *all six* obstacle entries disagreed wildly — that was my error, not an eval
   bug; the per-scenario indexing is correct.
2. **Decode-free cross-checks.** All success rates come from the plain-text `eval_*.log` summaries
   and all frozen-action counts from the plain-text `realtime_*_trial*.log` files, both read with
   `grep`. The `.npz` decode is used only for the candidate-fan and clearance statistics, which have
   no plain-text equivalent.

**Definitions used throughout.**
- **span** = `‖plan[-1] − plan[0]‖` over the 8-step planned horizon of one raw candidate (v6-INV's metric).
- **collapsed** = span < 1e-3.
- **clearance** = signed metric distance from the executed position to the nearest *active untightened*
  constraint (> 0 feasible). The tightening margin is 0.025, so the band `|clearance| < 0.025` is
  exactly the region where tightening changes the projection cost and plain tightening does not.
- **runaway gap** = `max_t ‖(x_des, y_des) − (x, y)‖`, i.e. how far the commanded setpoint
  (`obs` cols 0–1) diverges from the actual end-effector (`obs` cols 2–3). Healthy baseline ≈ 0.03–0.09.

---

## 1. The scoreboard

SC = success rate on **goal *and* constraints**, 2 trials per cell.

**`bbdit` (U1)**

| variant | K1:TL | K1:TR | K1:BH | K2:TL | K2:TR | K2:BH | K5:TL | K5:TR | K5:BH | K10:TL | K10:TR | K10:BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dpcc-r            | 0.5 | 1.0 | 0.5 | 1.0 | 0.5 | 1.0 | 1.0 | 0.5 | 0.5 | 1.0 | 0.5 | 0.5 |
| dpcc-r-tightened  | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.5 | 0.0 | 0.5 |
| dpcc-t            | 0.0 | 1.0 | 0.5 | 0.0 | 0.5 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.0 | 0.5 |
| dpcc-t-tightened  | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **dpcc-c**        | **0.5** | **0.5** | 1.0 | **0.5** | **0.5** | 1.0 | **0.5** | 1.0 | 1.0 | **0.5** | **0.5** | 1.0 |
| dpcc-c-tightened  | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 |

**`bbsit` (U2)**

| variant | K1:TL | K1:TR | K1:BH | K2:TL | K2:TR | K2:BH | K5:TL | K5:TR | K5:BH | K10:TL | K10:TR | K10:BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dpcc-r            | 1.0 | 0.5 | 0.0 | 0.0 | 0.5 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| dpcc-r-tightened  | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| dpcc-t            | 1.0 | 1.0 | 1.0 | 0.0 | 0.5 | 0.5 | 0.5 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 |
| dpcc-t-tightened  | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 |
| **dpcc-c**        | 1.0 | 1.0 | **0.5** | **0.0** | **0.0** | **0.0** | 1.0 | **0.5** | **0.5** | 1.0 | 1.0 | 1.0 |
| **dpcc-c-tightened** | 1.0 | 1.0 | 1.0 | **0.0** | **0.0** | **0.0** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Mean SC over all 24 cells, and excluding the 3 frozen `bbsit` K=2 cells:

| variant | all | excl. frozen |
|---|---|---|
| dpcc-r | 0.521 | 0.524 |
| dpcc-r-tightened | 0.875 | 0.857 |
| dpcc-t | 0.604 | 0.643 |
| dpcc-t-tightened | 0.958 | **0.976** |
| dpcc-c | 0.667 | 0.762 |
| dpcc-c-tightened | 0.854 | **0.976** |

Two things to read off:
- Once the K=2 freeze is set aside, **`dpcc-c-tightened` is the joint-best arm in Gen3v7** (0.976,
  tied with `dpcc-t-tightened`) — the `-c` *concept* is not broken.
- Plain `dpcc-c` loses 0.21 SC relative to its own tightened twin. That gap is what §2 explains.

---

## 2. Defect A — `dpcc-c` hugs the constraint boundary and falls off it

### 2.1 The measurement

Executed-path clearance, pooled over both trials of each cell. The `%band` column is the fraction of
executed steps inside `|clearance| < 0.025`; `%c<0` is the fraction strictly infeasible.

| bb | K | scen | variant | min clearance | %band | %c<0 | n_viol |
|---|---|---|---|---|---|---|---|
| bbdit | 1 | TL | dpcc-c            | **−0.0508** | 37.4 % | 12.6 % | 23 |
| bbdit | 1 | TL | dpcc-c-tightened  | +0.0154 | 5.3 % | 0.0 % | 0 |
| bbdit | 1 | TR | dpcc-c            | −0.0064 | 19.2 % | 1.8 % | 3 |
| bbdit | 1 | TR | dpcc-c-tightened  | +0.0113 | 8.4 % | 0.0 % | 0 |
| bbdit | 2 | TL | dpcc-c            | **−0.0468** | 39.9 % | 14.5 % | 24 |
| bbdit | 2 | TL | dpcc-c-tightened  | +0.0184 | 9.4 % | 0.0 % | 0 |
| bbdit | 2 | TR | dpcc-c            | −0.0033 | 14.4 % | 1.4 % | 2 |
| bbdit | 5 | TL | dpcc-c            | −0.0098 | 23.8 % | 4.9 % | 7 |
| bbdit | 10 | TL | dpcc-c           | −0.0010 | 17.8 % | 0.6 % | 1 |
| bbsit | 1 | BH | dpcc-c            | −0.0033 | 58.9 % | 1.5 % | 2 |
| bbsit | 5 | BH | dpcc-c            | −0.0204 | 29.7 % | 2.1 % | 2 |
| bbsit | 5 | TL | dpcc-c            | +0.0001 | 19.3 % | 0.0 % | 0 |
| *(all 24 `-c-tightened` cells)* | | | | **+0.011 … +0.030** | 0.0–12.5 % | **0.0 %** | **0** |

The pattern is uniform, not anecdotal:

- In **21 of 24** paired cells, plain `dpcc-c`'s minimum clearance is **lower** than
  `dpcc-c-tightened`'s in the same cell.
- Plain `dpcc-c` spends **17.6 %** of executed steps inside the tightening band; the tightened twin
  spends **4.9 %**.
- **Every** infeasible step in the whole `-c` family belongs to plain `dpcc-c`.
  `dpcc-c-tightened` records **0 violations in all 24 cells** — 64 for plain `dpcc-c`.

### 2.2 The runaway — what "explosion in the MPC foresight" is, numerically

The clearest single case is **`bbdit`, K=2, top-left-hard, trial 0** (24 violations, total 0.946):

```
step   :   0     20     30     40     45     50     60     70     80     90    100
gap    : 0.000  0.044  0.011  0.033  0.037  0.052  0.162  0.280  0.398  0.512  0.464
clear  : +.111  +.089  +.012  +.020  +.007  −.027  −.033  −.037  −.044  −.056  −.033
```

The commanded setpoint and the actual robot **decouple**: the episode ends with
`des = (0.217, 0.640)` while `act = (0.509, 0.354)`. `y_des` runs to **0.640**, far past the green
goal line at y ≈ 0.35 and clean out of the reachable workspace, while the actual `y` saturates at
0.354. This is visible directly in `dpcc-c.png` — the `y_des` panel climbs to 0.65 while the `y`
panel flattens at 0.35, and the executed path (black) threads the gap hard against the halfspace
edge instead of swinging wide.

Note the ordering: **clearance goes negative at ≈ step 45, and only then does the gap open.** The
runaway is a *consequence* of the plan being driven into the infeasible region — the planner keeps
commanding into a wall the robot cannot pass, and the setpoint integrates away. It is not an
independent numerical blow-up in the network.

Across all 288 rollouts the runaway gap is a strong predictor of violations:

| | runs | fraction with ≥1 violation | mean n_viol | mean total violation |
|---|---|---|---|---|
| gap > 0.15 | 26 | **61.5 %** | **6.88** | 0.208 |
| gap ≤ 0.15 | 262 | 13.7 % | 0.52 | 0.006 |

`corr(gap, n_viol) = 0.50`, `corr(gap, total_viol) = 0.45`.

**Caveat, stated plainly:** the runaway itself is *not* `-c`-specific. The two largest gaps in the
whole sweep are `bbdit / K5 / BH / dpcc-r` (**1.319**) and `bbdit / K10 / BH / dpcc-r-tightened`
(**1.202**), and `bbsit`'s `dpcc-r` shows 0.59–0.66 gaps at TL for every K. Reference runaway is a
**general Gen3v7 property** of this checkpoint family under MPC. What is `-c`-specific is that `-c`
*systematically steers into the conditions that trigger it* (§2.3).

### 2.3 Mechanism — `-c`'s cost is degenerate on the feasible interior

`-c` = `trajectory_selection = 'minimum_projection_cost'` (`policies.py:63-67`,
`which_trajectory = np.argmin(costs_total)`). The cost it minimises is, after substituting
`r = −Q·traj` (`projection.py:88`) into `projection.py:145`, exactly

```
projection_cost_i  =  0.5 · (sol_i − traj_i)ᵀ Q (sol_i − traj_i)
```

i.e. the squared Q-distance the projector had to move the candidate. The formula is correct and
identical to Gen12's — v6-INV §4b already audited it and so did I. **But it is identically zero on
the entire feasible set.** A plan clearing the obstacle by 1e-6 and a plan clearing it by 0.1 both
score exactly 0. `-c`'s objective is therefore *flat* over the interior and only starts to
discriminate *outside* it, where it prefers whichever infeasible plan is **closest to the boundary**.

The consequence, in one line: **`-c` cannot express a preference for clearance, and its only active
gradient points toward the constraint surface.** That is precisely the measured behaviour — 17.6 %
band occupancy vs 4.9 %, in 21 of 24 cells.

The other two rules do not have this property:
- `-r` (`policies.py:68-69`) takes candidate 0 — **indifferent** to clearance, not attracted to it.
- `-t` (`policies.py:59-62`) takes the temporally most consistent candidate — inertia, again no
  boundary attraction.

Both are worse than `-c` on average SC (0.52 / 0.60 vs 0.67) but neither shows `-c`'s systematic
clearance deficit.

### 2.4 Why the tightened twin is immune — the answer to "only `dpcc-c`, not the tightened, which is weird"

It is not weird; it is the direct fix for §2.3. Tightening moves the constraint surface 0.025 into
the feasible region (`enlarge_constraints: 0.025`, applied at
`eval_flow_matching_v3_alphaflow.py:244,253`). The flat-zero region of `-c`'s cost moves with it, so
a candidate that grazes the *true* boundary now sits *outside* the tightened set and pays a strictly
positive cost. The argmin is pushed off the surface and onto a plan with ≥ 0.025 of real clearance.

That is a **complete** account of the 24/24 result: `-c` is the selection rule most dependent on the
tightening margin precisely because its cost carries no clearance information of its own.

Honest limit on the claim: tightening helps *every* family here (plain-worse-than-tightened in 20/24
cells for `-r`, 22/24 for `-t`, 21/24 for `-c`), so tightening is not a `-c`-only benefit. What is
`-c`-only is the **perfect** outcome — 0 violations in 24/24 cells, which neither `-r-tightened`
(19 violations) nor plain anything achieves.

---

## 3. Defect B — the K=2 start-pose freeze (`bbsit` only)

### 3.1 It is the Gen3v6 defect, reproduced

`bbsit` at K=2 reproduces v6-INV's failure exactly, including every fingerprint:

- `dpcc-c` and `dpcc-c-tightened` both score **0.0 on all 3 scenarios**; every other K scores 1.0.
- **197/200 and 199/200** executed actions are a literal `ACT (±0.000,±0.000)` — identical counts in
  all three scenarios.
- `obs_all` is **byte-identical** across all 3 scenarios *and* across `-c`/`-c-tightened`
  (`np.array_equal` → True, max diff 0.00e+00). A plan that never approaches a constraint cannot
  tell the scenarios apart, and tightening a margin it never reaches changes nothing.
- The candidate-span histogram is **bimodal with an empty gap**, exactly v6-INV §3b's signature
  (both-hard, trial 0, N=800):

  ```
  [0.000,0.001): 138 (17.2%)   <-- machine-precision zero
  [0.001,0.005):   4 ( 0.5%)   <-- near-empty gap
  [0.005,0.010):  17 ( 2.1%)
  [0.010,0.020):  35 ( 4.4%)
  [0.020,0.030):  78 ( 9.8%)
  [0.030,0.040): 124 (15.5%)
  [0.040,0.050): 127 (15.9%)
  [0.050,0.060): 112 (14.0%)
  [0.060,0.080): 139 (17.4%)
  [0.080,1.000):  26 ( 3.2%)
  ```

  Per-replan counts of collapsed-of-4: `[92, 80, 26, 2, 0]`; implied per-draw `p = 0.176` vs measured
  marginal 0.172.
- The collapsed plan is **coherent, not noise** — all 8 waypoints agree to ~1e-4 and sit on the
  robot's current pose (replan 4, both-hard, trial 0):

  ```
  COLLAPSED cand0 (span 0.00027)      HEALTHY cand1 (span 0.0790)
    [ 0.52726 -0.27798]                 [ 0.52726 -0.27798]
    [ 0.52724 -0.27797]                 [ 0.52560 -0.27319]
    [ 0.52713 -0.27794]                 [ 0.51997 -0.26559]
    [ 0.52708 -0.27791]                 [ 0.51317 -0.25634]
    [ 0.52703 -0.27779]                 [ 0.50580 -0.24613]
    [ 0.52705 -0.27779]                 [ 0.49815 -0.23539]
    [ 0.52707 -0.27778]                 [ 0.49006 -0.22445]
    [ 0.52708 -0.27778]                 [ 0.48206 -0.21323]
  robot actual pose at replan 4: (0.52712, -0.27875)
  ```

The sampler's `(r, t)` coordinates are identical to Gen3v6's: `dt = 1/K = 0.5`,
`tau = loop_idx/K` (`af_diffusion.py:295`), `h = dt` (`:268`) ⇒ K=2 queries **`(r=0, t=0.5)` and
`(r=0.5, t=1.0)`** — the two interior/midpoint coordinates, twice, for the whole trajectory.
Projection still fires only on the last step (`snapping_start_idx = int((1−0.5)·2) = 1`,
`af_diffusion.py:342`).

### 3.2 It is a switch at K=2, and `bbdit` does not have it

Pooled over all 6 DPCC variants × 3 scenarios × 2 trials:

| bb | K | h/step | candidates | collapsed (<1e-3) | median span |
|---|---|---|---|---|---|
| bbdit | 1  | 1.00 | 10 960 | **0.00 %** | 0.0837 |
| bbdit | 2  | 0.50 | 10 116 | **0.06 %** | 0.0843 |
| bbdit | 5  | 0.20 | 10 972 | **0.04 %** | 0.0847 |
| bbdit | 10 | 0.10 | 10 668 | **0.00 %** | 0.0844 |
| bbsit | 1  | 1.00 | 10 828 | 0.06 % | 0.0840 |
| bbsit | 2  | 0.50 | 16 560 | **9.96 %** | **0.0604** |
| bbsit | 5  | 0.20 | 9 916 | 0.11 % | 0.0849 |
| bbsit | 10 | 0.10 | 9 960 | 0.21 % | 0.0849 |

So the freeze is a property of the **`sit` checkpoint's field at h = 0.5**, not of α-Flow, not of the
eval code, and not of the `-c` rule — `bbdit` runs the identical eval, selection rule and K grid with
0.00–0.06 % collapse. It is also **milder than Gen3v6's**: 17.0 % vs 28.1 % on the matched
`dpcc-c`/both-hard cell.

### 3.3 🔴 New result — the collapse is state-conditioned, not i.i.d.

v6-INV §8b concluded the collapse is *"a fixed-size basin in the noise → trajectory map, hit at
random — **not** a property of where the robot happens to be."* That conclusion was drawn **entirely
from inside the frozen `-c` rollout**, where the robot's state is constant by construction — so a
binomial fit was guaranteed to succeed and could not have detected state dependence.

Gen3v7 provides the missing control: at `bbsit` K=2, eleven *non-frozen* variants run the same
checkpoint, same K, same seed, and do leave the start pose. Stratifying every replan by distance
travelled from the start (all 13 variants × 3 scenarios × 2 trials, 6 850 replans / 27 400 candidates):

| ‖x − x₀‖ | replans | candidates | collapse rate | median span |
|---|---|---|---|---|
| [0.000, 0.001) | 288 | 1 152 | **11.98 %** | 0.0379 |
| [0.001, 0.002) | 837 | 3 348 | **17.03 %** | 0.0380 |
| [0.002, 0.005) | 1 593 | 6 372 | **14.12 %** | 0.0407 |
| [0.005, 0.010) | 141 | 564 | **10.64 %** | 0.0637 |
| [0.010, 0.050) | 243 | 972 | **0.00 %** | 0.0949 |
| [0.050, ∞) | 3 748 | 14 992 | **0.00 %** | 0.0889 |

**Zero collapses in 15 964 candidates once the robot is more than 0.01 from its start pose.** Not
"rarer" — absent. And the *healthy* candidates are attenuated in the same basin too (median span
0.038 inside vs 0.089 outside), so the whole field is damped there, not just bimodal.

Restricting to only the non-frozen rollouts (so the frozen `-c` data cannot bias the near-zero bin)
gives the same edge: **2.38 %** collapse at d < 0.005, **0.00 %** in 16 288 candidates at d ≥ 0.005.

**This makes the freeze a trap, not a lottery.** `-c` selects a collapsed plan on the first replan →
the robot does not move → the next replan is again inside the basin → 17 % per draw, so
`1 − 0.83⁴ ≈ 53 %` chance at least one of 4 candidates is collapsed, and `-c` takes it every time it
is offered. `-r` and `-t` have no preference for it, escape the 0.01 basin within a few steps, and
then the mode is simply not available to them any more. Per-variant at `bbsit` K=2 / both-hard:

| variant | collapse | median span | episode length | SC |
|---|---|---|---|---|
| dpcc-c | **17.0 %** | 0.0396 | 200 | 0.0 |
| dpcc-c-tightened | **17.0 %** | 0.0396 | 200 | 0.0 |
| dpcc-r | 0.6 % | 0.0843 | 60 | 1.0 |
| dpcc-t | 0.0 % | 0.0860 | 76 | 0.5 |
| model_free | 0.0 % | 0.0936 | 58 | 1.0 |
| post_processing | 0.6 % | 0.0843 | 60 | 1.0 |

### 3.4 Training-side corroboration — Gen3v7 supplies the mechanism v6-INV left open

v6-INV's §3c hypothesis ("bigger `h` is worse") was falsified by its own K-sweep, and §8c closed with
*"the remaining open question is **why** `(r=0.5)` hosts a dead mode when `(r=0, t=1)` does not"*.
Gen3v7 logs the h-stratified validation error that answers it. Buckets
(`af_diffusion.py:796-800`): `b0: h==0`, `b1: (0,0.3)`, `b2: [0.3,0.6)`, `b3: ≥0.6`.

| bucket | h range | K that lands there | **sit** val, mean of last 10 | sit val max | mf_dit val, last 10 |
|---|---|---|---|---|---|
| b0 | 0 | (the `h=0` anchor) | **2.78** | 59.7 | 2.79 |
| b1 | (0, 0.3) | **K=5 (0.2), K=10 (0.1)** | **22.0** | 69.1 | 40.5 |
| b2 | [0.3, 0.6) | **K=2 (0.5)** | **116** | 390 | 44.8 |
| b3 | ≥ 0.6 | **K=1 (1.0)** | **73** | 374 | 136 |

**`b2` is `sit`'s worst sustained bucket — 42× `b0` and 5× `b1` — and it is not the largest-`h`
bucket.** `b3` (which is where K=1's `h = 1.0` lives) is *better* than `b2`. That non-monotonicity in
`h` is exactly the behavioural pattern v6-INV measured but could not explain: K=1 clean, K=2 broken,
K=5/10 clean. The predicted badness ordering from the validation curves —
**b2 (116) > b3 (73) > b1 (22) > b0 (2.8)**, i.e. **K=2 > K=1 > K=5/K=10** — matches the observed
collapse ordering (9.96 % > 0.06 % ≈ 0.11 % ≈ 0.21 %) in which K is worst, though the effect is far
more than proportional at K=2.

Read together with §3.3: the α-Flow field has a **mid-interval hole** (`h ≈ 0.5`) that is deepest in
the neighbourhood of the data distribution's start pose, and K=2 is the only step count that puts
100 % of its queries into it.

---

## 4. What is *not* wrong (audited, so it stops being re-litigated)

- **The `-c` cost formula.** Algebraically exactly `0.5·(sol−traj)ᵀQ(sol−traj)`; no sign error, no
  missing term, no batch cross-talk. Its degeneracy on the interior (§2.3) is a **specification gap**,
  not an implementation bug.
- **The tightening branch.** `eval_flow_matching_v3_alphaflow.py:293-304` genuinely hands
  `constraint_list_tightened` to `*-tightened` variants; verified by reading the branch.
- **The violation counter.** Reproduced exactly in 191/192 runs (§0). It checks the *untightened*
  set against the *actual* position only — `x_des`/`y_des` carry no constraint row
  (`constraints_helpers.py:12-19` writes only the `x`/`y` indices), so the runaway setpoint of §2.2
  is **not** double-counted as a violation.
- **Per-scenario constraint selection.** One halfspace + one obstacle per scenario, by index. Paths
  that pass through a *plotted* red circle which is not the active obstacle for that scenario are
  correctly not counted.
- **The eval driver, the projector, and the K plumbing.** `bbdit` and `bbsit` share all of it and
  differ completely in outcome; `bbdit` shows no freeze at any K.

---

## 5. Differences from Gen3v6, summarised

| | Gen3v6 (`mf_dit`) | Gen3v7 (`bbsit`) | Gen3v7 (`bbdit`) |
|---|---|---|---|
| K=2 collapse rate | 28.1 % | 17.0 % | 0.06 % |
| collapse at other K | exactly 0 (0/1828) | 0.06–0.21 % | 0.00–0.06 % |
| K=2 median span | 0.0316 | 0.0396 | 0.0843 (normal) |
| freeze affects `-c-tightened` too | yes | yes | n/a |
| boundary-hugging Defect A | not reported | yes (K=1/5) | **yes, dominant** |
| mechanism established | "i.i.d. in the noise" (§8b) | **state-conditioned, basin r ≈ 0.01** | — |
| `h`-bucket evidence | not logged | **b2 = worst bucket** | — |

Gen3v6's `-c` degenerates by preferring **motionless** plans; Gen3v7-`bbdit`'s `-c` degenerates by
preferring **boundary-grazing** plans. Both are the same specification gap — *least correction ≠ best
plan* — expressed through different fields. `bbsit` has both at once, at different K.

---

## 6. Recommended follow-ups

1. **Give `-c` a clearance term (the actual fix for Defect A, cheap).** Either (a) always evaluate
   `projection_costs` against the *tightened* set even for the untightened variant, or (b) add a
   clearance bonus so the cost stops being flat on the interior. (a) is a one-line change and is
   already validated by the 24/24 result. **This is not a code change I have made — proposal only.**
2. **Add a minimum-progress guard to `-c` (the fix for Defect B, also cheap).** Exclude any candidate
   with horizon span below a floor (e.g. 0.01) before the argmin. §3.3 shows the trap is escapable —
   the mode does not exist 0.01 away from the start — so a single successful step breaks it
   permanently. This also protects Gen3v6.
3. **Probe the field directly on the cluster (highest scientific value).** Sweep a grid of `(r, t)`
   pairs × start-state neighbourhoods, push N Gaussian draws through one `u(x, r, t)` call each, and
   map the near-zero fraction. §3.3 predicts a specific shape: a hole at `h ≈ 0.5` that is deep near
   the start pose and absent 0.01 away. That turns a trajectory-level inference into a measured
   property of the weights.
4. **Check `af_ratio_fm = 0.5`'s effect on interior coverage.** Half of every α-Flow batch is forced
   to `r == t` (`h = 0`, `af_diffusion.py:694`), so the `h > 0` interior gets half the sample budget
   MeanFlow gives it — yet `bbsit`'s collapse rate is *lower* than `mf_dit`'s. Worth resolving: is the
   `b2` hole caused by thin coverage, by the α→0 endpoint, or by neither?
5. **Cross-check against the `ae` (α-end) cap already proposed in
   [`RESULTS_Gen3v7_U2_sit_first_run_2807.md`](RESULTS_Gen3v7_U2_sit_first_run_2807.md) §5.** That run
   found `h_mse_b3` spiking to 374 at the *final* training step (α→0). `b2 = 116` sustained is the
   same family of symptom one bucket down. Capping `ae` at 0.05–0.1 is the single change that plausibly
   addresses both.
6. **Seeds.** Everything here is seed 6, 2 trials. The candidate-level statistics are large-sample
   (27 400 candidates) and the 0.00 %/17 % split is not marginal, but the SC tables are 2 episodes per
   cell and **individual SC deltas of 0.5 are one episode**. Seeds 7–10 at K=2 and K=5 would settle
   whether Defect A's cell-level pattern is stable.
7. **Practical guidance meanwhile:** do not run `bbsit` at K=2; prefer `dpcc-c-tightened` or
   `dpcc-t-tightened` (both 0.976 mean SC excluding K=2) over any untightened arm.

---

## How this was checked

No Python packages are available in this container for the pipeline; a throwaway venv
(`numpy` only, scratchpad, not part of the repo) was used purely to decode `.npz` files already
produced on the cluster. **No training or eval was run locally.** Every headline number has either a
decode-free cross-check (success rates from `eval_*.log`, frozen-action counts from
`realtime_*_trial*.log`, both via `grep`) or the geometry self-validation of §0 (191/192 exact
reproduction of the stored violation counters).

Sources: `temp/Gen3V7/` (bbdit, jobs 23759/23786) and `temp/2807/2807/` (bbsit, jobs 23929/23930,
plus `sit_losses.pkl` / `mf_dit_losses.pkl` for §3.4).
