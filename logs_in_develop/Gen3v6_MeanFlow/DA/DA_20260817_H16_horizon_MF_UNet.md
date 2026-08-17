# DA 2026-08-17 — How does H16 behave? (MeanFlow-UNet, avoiding-d3il, replan-1)

**Batch:** `temp/1708/batch_avoiding_combined_20260817_092728/`
**New runs:** MeanFlow-UNet trained at **horizon 16** — train **24633** (seed 6, 100k steps, 7 h 32 m)
· eval **24634** (K = 1, 2, 5 in one job, 1 h 33 m total).
All `MF_HORIZON=16 MF_BACKBONE=unet`, `hf_batch=1 · A=0.5 · dpcc_threshold=0.5`,
**seed 6 only, `n_trials = 2`** ⇒ 3 halfspaces × 2 = **6 episodes per cell**.
**Candidates:** C121 (H16 K1) · C122 (H16 K2) · C123 (H16 K5).
**H8 comparators:** C135/C139/C144 (`bbunet`, 5 seeds, n=20 → 300 ep/cell, from
`DA_20260815_ntrials20_stability_MF_UNet.md`) and C134/C138/C143 (same, n=2). The two H8 batches
agree on timing to ~2 %, so the H8 side of every ratio below is solid.

> **🔴 SCOPE — this is the horizon rung ONLY.** The log line reads
> `[ h8+8 ] MF_HORIZON=16 MF_BACKBONE=unet MF_REPLAN_STEPS=1 (default)`. **No 8+8 / replan-8 data
> exists yet** — no `_msgr8` directory was produced. Everything here is H16 with per-step
> replanning, i.e. rung 2 of the three-rung ladder in
> `../H8+8_U10/GUIDE_H16_replan8_MF_UNet.md` §1.1.

---

## 0. Bottom line

**H16 is a projector-cost story, not a network-cost story, and it inverts the arm-B-vs-arm-C cost
ranking that the whole HardFlow-is-too-expensive conclusion rested on.**

1. Doubling the horizon costs the **generative field essentially nothing** (unprojected `diffuser`:
   **1.01–1.06×**) and costs the **DPCC projector 3.4–7.3×**. HardFlow's IPOPT arm pays only
   **1.42–1.49×**.
2. Consequently **HardFlow is now cheaper than DPCC at every K** — 1.26–1.29× at K1/K2 and
   **6.4× at K5** — where at H8 it was 2.4× *more* expensive at K1. The 2026-08-02 DA's
   "HardFlow costs 2.8–3.3× DPCC, recommend dropping it" is an **H8-specific artifact**.
3. Every mechanism prediction from `HF_Study/DEGENERACY_HardFlow_at_low_K.md` reproduces **exactly**
   at H16: NLP solves/plan = 1.02 / 1.02 / 3.05 and NFE/plan = 2.03 / 3.05 / 8.13 for K = 1/2/5.
   The degeneracy is a function of (K, A) only — the horizon does not rescue it.
4. **The safety numbers prove nothing.** Every `-tightened` arm scores S&C 1.000, but n = 6 gives a
   95 % CI of **[0.541, 1.000]** — statistically indistinguishable from H8's 0.943 at n = 300.
   Do not report "H16 improves safety".

---

## 1. The headline — cost scaling with horizon

`avg_time` is seconds of planning per **env step** (total plan time ÷ steps taken).
H8 column = C135/C139/C144 (n=20, 5 seeds).

| variant | K1 H8 → H16 | K2 H8 → H16 | K5 H8 → H16 |
|---|---|---|---|
| `diffuser` (no projection) | 0.0096 → 0.0102 **1.06×** | 0.0187 → 0.0194 **1.04×** | 0.0462 → 0.0467 **1.01×** |
| `dpcc-c-tightened` | 0.0180 → 0.0812 **4.52×** | 0.0268 → 0.0921 **3.44×** | 0.2245 → **1.2731** **5.67×** |
| `dpcc-r-tightened` | 0.0184 → 0.0756 **4.11×** | 0.0273 → 0.0989 **3.62×** | 0.2247 → **1.6425** **7.31×** |
| `dpcc-t-tightened` | 0.0181 → 0.0900 **4.98×** | 0.0271 → 0.1126 **4.15×** | 0.2250 → **1.5411** **6.85×** |
| `hardflow_new-c-tightened` | 0.0423 → 0.0629 **1.49×** | 0.0506 → 0.0733 **1.45×** | 0.1408 → 0.1999 **1.42×** |

**Read the first row first.** The network sees a 2× longer sequence and costs ~1.03× more — a 1-D
temporal U-Net over twice the length is nearly free. So none of the blow-up below is generative
cost; **all** of it is constraint solving.

### 1.1 The order flip

Ratio `dpcc-c-tightened / hardflow_new-c-tightened` — **> 1 means HardFlow is cheaper**:

| K | H8 | H16 |
|---|---|---|
| 1 | 0.42× (HF 2.4× *more* expensive) | **1.29×** (HF cheaper) |
| 2 | 0.53× (HF 1.9× *more* expensive) | **1.26×** (HF cheaper) |
| 5 | 1.59× | **6.37×** |

At K5/H16 the DPCC arms cost **1.27–1.64 s per env step** — a ~60-step episode spends over a minute
of wall time purely in SLSQP. HardFlow at the same K and horizon costs 0.20 s.

**Why (mechanistic, not yet measured):** two effects that both scale with horizon and are *not*
matched between the arms. (i) The DPCC arms run `batch_size = 4` candidates, HardFlow `B = 1` — a
4× handicap that exists at H8 too, so it cannot explain the *change*. (ii) Scipy **SLSQP** on a
dense QP whose dof doubles (H·transition_dim) scales roughly cubically ⇒ ~8×, while **IPOPT** with
limited-memory BFGS and `solve_limited` scales far better. Effect (ii) is the one that grows with
H, and it matches the observed 3.4–7.3× vs 1.4×. **A per-solve timing breakdown would settle it and
does not exist yet** (§7).

---

## 2. Safety — real numbers, but not evidence

Mean over 3 halfspaces; each halfspace is 2 episodes, so every cell is a multiple of 1/6.

| variant | H16 K1 | H16 K2 | H16 K5 | H8 K1 (n=300) | H8 K2 | H8 K5 |
|---|---|---|---|---|---|---|
| `diffuser` | 0.167 | 0.000 | 0.000 | 0.120 | 0.120 | 0.107 |
| `dpcc-r-tightened` | **1.000** | **1.000** | **1.000** | 0.930 | 0.933 | 0.927 |
| `dpcc-c-tightened` | **1.000** | **1.000** | **1.000** | 0.943 | 0.927 | 0.867 |
| `dpcc-t-tightened` | **1.000** | 0.833 | 0.833 | 0.993 | 0.993 | 0.980 |
| `hardflow_new-*-tightened` | **1.000** | **1.000** | **1.000** | 0.950 | 0.900 | 0.897 |
| untightened (`dpcc-c`, `hardflow_new-r`, …) | 0.167–0.667 | 0.333–0.833 | 0.500–0.667 | — | — | — |

🔴 **Do not read the 1.000s as an improvement.** Clopper–Pearson 95 % CI for **6/6 is
[0.541, 1.000]**. It contains H8's 0.943 comfortably. With `n_trials = 2` and one seed, this design
cannot detect anything smaller than a ~40-point swing. The honest statement is
**"H16 did not break safety"** — which is worth knowing, and is all that is established.

The one row that moves in the *other* direction is `dpcc-t-tightened` (0.833 at K2/K5 vs 0.993 at
H8) — 5/6 episodes. Temporal-consistency selection compares the new plan against the previous plan
shifted by one step; at H16 that overlap window is a smaller *fraction* of the plan, which is a
plausible mechanism. At n=6 it is equally plausibly one unlucky episode. **Flagged, not concluded.**

**Constraint violations** are unambiguous, though: `diffuser` accumulates **13.2 / 18.5 / 20.2**
violations per episode at K1/K2/K5 while every `-tightened` arm sits at exactly **0.00**. The H16
field on its own is *not* safer — projection is doing all the work, exactly as at H8.

---

## 3. Mechanism validation — the degeneracy math reproduces exactly at H16

From `HF_Study/DEGENERACY_HardFlow_at_low_K.md` §4.1, `n_active = K − floor((1−A)·K)` with A = 0.5,
and §6's NFE accounting `K + n_active`. Measured here (`nlp_solves_total`, `nfe_total` ÷ plans):

| K | predicted NLP/plan | **measured** | predicted NFE/plan | **measured** | genuine HardFlow steps |
|---|---|---|---|---|---|
| 1 | 1 | **1.02** | 2 | **2.03** | **0** |
| 2 | 1 | **1.02** | 3 | **3.05** | **0** |
| 5 | 3 | **3.05** | 8 | **8.13** | 2 |

Two consequences:

1. **The low-K degeneracy is horizon-independent.** It is a function of (K, A) only. The K1 and K2
   HardFlow rows above are *still* terminal-only projection — `Π_S(Euler sample)` — and must not be
   read as HardFlow results, exactly as at H8.
2. **The wasted terminal network call is confirmed at H16** (2.03 NFE/plan at K=1, where the second
   evaluation is multiplied by `(1−τ⁺) = 0`). §10 of the degeneracy study proposed skipping it;
   this run re-measures the waste rather than removing it.

This is also the strongest evidence that the H16 pipeline is *wired correctly*: three independent
counters land on their predicted values to within 2 %.

---

## 4. Episode length

| variant | H16 K1 | H16 K2 | H16 K5 | H8 K1 | H8 K2 | H8 K5 |
|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | 58.83 | 62.67 | 59.00 | 72.01 | **98.00** | 63.96 |
| `dpcc-t-tightened` | 61.00 | 63.00 | 71.83 | 60.99 | 60.41 | 60.84 |
| `hardflow_new-c-tightened` | 60.50 | 63.83 | 61.17 | 63.40 | 67.06 | 67.31 |

Most cells are flat (58–72 steps) — **a 2× longer plan does not make the controller wander**, which
was not guaranteed. The exception is worth noting: H8's `dpcc-c-tightened` at K2 took **98 steps**
(confirmed independently by both H8 batches, 97.20 at n=2 and 98.00 at n=20 — so the H8 value is
real, not noise), while H16 takes **62.67**. A 36 % reduction on a known H8 pathology. Small-n on
the H16 side, but the H8 side is solid and the gap is large; worth re-testing at scale.

---

## 5. IPOPT failures — fewer at H16, per episode

`nlp_failures_total`, normalised per episode (H16: ÷2 trials; H8: ÷20 trials):

| K | H16 `-tightened` | H8 `-tightened` (n=20) |
|---|---|---|
| 1 | **0.00** | 0.36 |
| 2 | 1.00 | 0.59 |
| 5 | 1.83 | 2.79 |

Untightened HardFlow variants recorded **zero** failures at every K — the failures are entirely on
the tightened (harder) feasible set, which is the expected direction. Since a failed solve makes
HardFlow keep the last IPOPT iterate — *not guaranteed feasible* (`hardflow_projection.py:339-346`)
— nonzero counts are a standing caveat. Here they produced no violations (§2, all tightened arms at
0.00), but that is 6 episodes' worth of reassurance, not a guarantee.

---

## 6. Cost of running this study

| stage | job | wall |
|---|---|---|
| train H16 seed 6, 100k steps | 24633 | **7 h 32 m** |
| eval K=1 (6 episodes × 13 variants) | 24634 | 10 m 22 s |
| eval K=2 | 24634 | 11 m 48 s |
| eval K=5 | 24634 | **70 m 43 s** |

🔴 **Scaling warning for the 5-seed / n=20 version.** 300 episodes is 50× this workload, so K5 alone
extrapolates to ≈ **59 h** — versus 15 h 40 m for the same cell at H8 (2026-08-15 DA). That is
**~3.8× the H8 cost and 2.5× over the 24 h Slurm cap.** A full-scale H16 K5 run is not submittable
as one job; split by halfspace or by seed. K1/K2 extrapolate to ~9 h each and are fine.

---

## 7. Threats to validity

1. **n = 6 per cell, one seed.** Everything in §2 and §4 is underpowered. §1 (timing) and §3
   (counters) are not — they average over ~360 per-step measurements with 3–7× effects.
2. **Arm B runs 4 candidates, arm C runs 1.** A 4× handicap on the DPCC side in every timing number.
   It is present at H8 too, so the *ratio change* in §1.1 stands, but the absolute "HardFlow is
   6.4× cheaper" is inflated by it. **The clean version of this experiment sets `HFFM_BATCH=4`** —
   never run at H16.
3. **No H16 baseline.** Diffusion-DPCC K20/aw10 exists only at H8/replan-1, so nothing here can be
   compared to *the* baseline. Unchanged from the guide's §8.
4. **Solver identity is inferred, not measured.** §1's SLSQP-vs-IPOPT explanation is consistent with
   the numbers but untested; a per-solve timing histogram would confirm or kill it.
5. **No `run_provenance.json`.** The U10.1 provenance writer post-dates this run
   (`../../CLI_Override_Snapshot/`), so config confirmation here comes from the job log's
   `[ h8+8 ]` / `[ train ]` echo lines only. Future runs will carry the JSON.
6. **`dpcc-t-tightened` at 0.833** (§2) is either a real horizon interaction with the shift-by-1
   temporal-consistency window or one unlucky episode. Unresolved.

---

## 8. What this predicts for the 8+8 run

Not yet measured — stated so the next DA can check it:

- Replan-8 makes ~1/8 as many plans per episode, so the DPCC arms' K5 catastrophe (1.27–1.64 s per
  env step) should fall to **~0.16–0.21 s** amortised. The H16 cost objection largely evaporates.
- HardFlow's advantage should therefore *narrow* under replan-8, not widen — the opposite of §1.1.
  If it does not, the difference is per-solve cost rather than solve count.
- `nlp_solves` per env step must drop by ~8× at fixed K. That is the arithmetic check that the
  cadence is actually engaged (guide §9 step 4).
- The unprojected `diffuser` arm should degrade the most: it has no feedback correction at all, so
  7 open-loop steps are pure extrapolation.

## 9. Recommended next runs, in order

1. **`HFFM_BATCH=4` at H16, K5** — one job. Closes threat #2 and decides whether §1.1 is a real
   algorithmic advantage or a candidate-count artifact. Highest information per GPU-hour here.
2. **Replan-8** (`MF_REPLAN_STEPS=8`) at K1/K2/K5, same seed/trials — completes the ladder and tests
   §8 directly.
3. **Scale up whichever of those survives** to 5 seeds × n=20, K1/K2 only, splitting K5 by halfspace
   (§6).
4. Not recommended yet: an H16 diffusion baseline retrain. Two cheap runs above can still kill the
   H16 direction; spend the second training budget only after they do not.
