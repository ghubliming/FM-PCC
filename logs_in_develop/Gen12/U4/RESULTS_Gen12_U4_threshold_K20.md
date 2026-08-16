# Gen12 U4/U4.2 — first threshold + candidate-fan results (K=20)

**Date:** 2026-07-26 · **Type:** results / insight · **Status:** preliminary (K=20 only, n=2, seed 6)
**Runs:** jobs 23829 (git 18fa5c2), 23830 + 23831 (git b9bc2ec), node i6-gpu-1
**Logs+data:** `temp/Gen12/2607/` (3 halfspace variants × n_trials=2, all K=20)
**Model:** FMv3ODE checkpoint `…FlowMatchingODE_a1.5_b1.0_aw10`, seed 6

> Three arm-C configs were run against the fixed A/B baselines: **batch1/thres0** (23830, faithful
> baseline), **batch1/thres0.5** (23831, U4), **batch4/thres0** (23829, U4.2 fan). All at K=20 — the
> **saturated regime**, where safety/success can't discriminate, so this run measures **compute**.

---

## 1. Everyone is safe at K=20 — so read the compute

| arm | config | goal | constraints | violations | notes |
|---|---|---|---|---|---|
| A `diffuser` | — | 1.00 | **0.00** | 33 / 10 / 12.5 | reaches goal, never safe |
| B `dpcc-c-tightened` | batch4 | 1.00 | **1.00** | 0 | incumbent |
| C `hardflow_new` | b1/thres0 | 1.00 | **1.00** | 0 | faithful baseline |
| C `hardflow_new` | b1/thres**0.5** | 1.00 | **1.00** | 0 | **U4** |
| C `hardflow_new` | b4/thres0 | 1.00 | **1.00** | 0 | **U4.2 fan** |

`NLP failures = 0` in every arm-C config. Safety/success identical across B and all three C configs —
so the story is entirely in cost.

## 2. Compute (avg over the 3 halfspace variants)

| arm / config | time/step (s) | NLP solves (Σ 3 hs) |
|---|---|---|
| B `dpcc-c-tightened` | **0.477** | 0 (SLSQP projection) |
| C  b1 / thres0 (faithful) | 0.754 | 8 400 |
| C  b1 / **thres0.5 (U4)** | **0.488** | **4 532** |
| C  b4 / thres0 (U4.2 fan) | 3.008 | 33 600 |

## 3. Findings

### 3.1 U4 works — ~35% faster, ~46% fewer NLP solves, ZERO safety cost ✅
Arm C at batch1, threshold `0` → `0.5`:
- **NLP solves 8 400 → 4 532 (−46%)**, **time 0.754 → 0.488 s/step (−35%)**,
- **safety/success unchanged** (100% / 100%, 0 violations).

This is exactly the HardFlow paper's efficiency claim ("skip early steps … good balance"), now
**confirmed on FMPCC's own model**: the early-step NLP solves were pure overhead here — dropping them
costs nothing and saves a third of the wall time. The U4 threshold + final-step guard behaves as
designed on hardware.

### 3.2 U4 brings arm C to cost-parity with DPCC — softening the fix_3 negative ⭐
fix_3 concluded arm C "ties B on safety but loses on cost (~1.6× slower)". With U4, arm C batch1
drops from 0.754 → **0.488 s/step**, essentially **tied with arm B (0.477)** — both 100% safe at
K=20. Per-variant it's mixed (B faster on top-right 0.316 vs 0.492; tie on top-left; **C faster on
both-hard 0.497 vs 0.640**), because DPCC's projection cost swings with constraint tightness while
U4-gated arm C is more uniform (~0.47–0.50). Net: **U4 removes arm C's cost disadvantage** — at K=20
it's now a genuine tie on *both* safety and cost, not a dominated method.

### 3.3 U4.2 candidate fan (batch4) is pure waste at K=20 ❌ (as expected)
batch1 → batch4 (thres0): **4× the NLP solves and 4× the time** (0.754 → 3.008 s/step) for **zero**
gain — arm C is already 100%/100% at batch1, so there is nothing for the fan to improve. This is the
saturated regime; the fan can only matter at **low K**, where fix_3 showed batch1 arm C *failing*.
That cell is **not tested here** (all runs K=20). So U4.2's value is still open — this run only shows
it's not worth paying for when batch1 already saturates.

## 4. The headline picture (K=20)

- **arm A** unguided: fast (0.17 s) but never safe — useless alone.
- **arm B** DPCC: 100% safe, 0.477 s/step.
- **arm C + U4 (thres0.5)**: 100% safe, 0.488 s/step — **matches B on safety and cost**.
- arm C full-step (thres0): 100% safe but 0.754 s/step — the cost U4 removes.
- arm C batch4: 3.0 s/step — don't, at high K.

U4 is the clear win of this run. U4.2 is confirmed expensive-with-no-upside *in the saturated regime*.

## 5. Caveats (binding)

1. **K=20 only — the saturated regime.** Every guided arm is already perfect, so this run **cannot**
   test the thing that matters most: does U4's threshold *fix the low-K failure* fix_3 found (arm C
   collapsing at K=2/5)? **Untested.** The decisive experiment is U4 at **low K**.
2. **n = 6** (2 trials × 3 halfspace × 1 seed). A "1.00" is 6/6 — huge CIs. Not statistically
   separable; treat time numbers as indicative.
3. **K was 20, not the configured 10** — the recurring cluster plan-block / `--flow-steps` issue.
   Doesn't change the U4/U4.2 *relative* conclusions (all arms shared K=20).
4. **Old path layout.** These ran before fix_5, so results are under
   `results/halfspace_*/K20_n2_thres*_mpc*/` (run_tag subdir), not the new
   `<train>/<eval-name>/…` tree. `load_results` (post-fix_5) won't find them — read the npz directly
   or re-run.

## 6. Next (in priority order)

1. **U4 at LOW K** (K ∈ {2,5,10}, thres ∈ {0, 0.5}) — the real test: does late activation *rescue*
   the low-K failure fix_3 documented, not just save compute? This is the experiment that decides
   whether U4 is merely an efficiency knob or an actual quality fix.
2. **U4.2 fan at low K** — where batch1 arm C failed; does batch4 + `-c` selection recover it?
   (Pair with U4: `thres0.5 + mpc4 + hardflow_new-c`.)
3. **n ≥ 100** + more seeds once the above shows a signal worth powering.
4. Re-run under fix_5 so results land in the FMv3ODE-style `<train>/<eval-name>/` tree.

---

## ⚠️ POST-HOC NOTE (fix_6, 2026-07-26): threshold labels were pre-flip

These runs used the OLD (inverted) threshold polarity. fix_6 flipped it to DPCC polarity
(higher = more projection). Re-map the labels in this MD:

| this MD says | means | now written as (DPCC) |
|---|---|---|
| `thres0` | full-step (every-step NLP) | `thres1.0` |
| `thres0.5` | last half | `thres0.5` (unchanged) |

The **0.5 free-lunch finding is unaffected** (0.5 is the fixed point). Only the `thres0` baseline
label becomes `thres1.0` under the corrected convention. See
[`../fix_6/CHANGELOG_fix6_dpcc_threshold_polarity.md`](../fix_6/CHANGELOG_fix6_dpcc_threshold_polarity.md).
