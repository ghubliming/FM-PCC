# DA — HardFlow vs DPCC across K on `avoiding-d3il`: does it pay once it actually runs?

**Date:** 2026-08-24 · **Type:** re-analysis of existing runs (no new cluster time)
**Primary data:** `temp/1108/Revised_2/batch_avoiding_combined_20260811_221322/candidates_multidimensional_raw.csv`
— candidates **C115 (K=1), C117 (K=2), C119 (K=5), C114 (K=10)**, the post-Fix_9 `A0.5_B1` MeanFlow
UNet@32 ladder. Same checkpoint at every K; folder names differ only in `K`.
**Sample:** 5 seeds × 3 scenarios = **n = 15 per cell**. S&C resolution 0.067.
**Settings:** H8, `A = 0.5`, `T = 0.5`, replan 1.
**Secondary (§5):** UAV corridor, `temp/2108/batch_uav_20260821_105229/uav_units_long.csv`.
**Regime labels** from `hardflow_regime(K, 0.5)` — see
[`CHANGELOG_…_K1_guard.md`](./CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md) §B.1.

> **Why this file was rewritten.** The first version led with UAV corridor data because that sweep
> is candidate-fan-matched. `avoiding-d3il` is the benchmark, so it leads now — with its fan
> mismatch stated up front (§1) rather than avoided.

---

## 0. The answer

**On `avoiding-d3il`, no. HardFlow does not get better when it starts running.**

Counting how many of the three DPCC selection rules HardFlow's single arm beats on S&C:

| K | regime | n_genuine | untightened | tightened |
|---:|---|---:|---|---|
| 1 | `DEGENERATE` | **0** | beats 2 of 3 | **beats 3 of 3** |
| 2 | `DEGENERATE` | **0** | beats 2 of 3 | beats 0 of 3 |
| 5 | `OK` | 2 | beats 2 of 3 | beats 2 of 3, ties 1 |
| 10 | `OK` | 4 | **beats 3 of 3** | beats 0 of 3 |

There is no threshold effect. The only clean sweep untightened is **K=10**; the only clean sweep
tightened is **K=1 — where HardFlow executes no HardFlow math at all.**

This contradicts the UAV corridor result (§5), where the win switches on cleanly at K=5. The two
datasets differ in the one thing that matters for a cost comparison — candidate fan — so §1 first.

---

## 1. 🔴 Read this before any number below: the fan is mismatched

On this ladder **HardFlow ran at candidate fan 1; DPCC ran at fan 4** (`hf_batch_size = 1.0`,
`batch_size = 4.0` in the CSV). Two consequences, in opposite directions:

**(a) HardFlow has ONE arm, not three.** At B=1 all selection rules collapse to slot 0 — verified
exactly:

| K | `hardflow_new-r` | `-c` | `-t` | steps |
|---:|---:|---:|---:|---|
| 1 | 0.533 | 0.533 | 0.533 | 59.27 / 59.27 / 59.27 |
| 2 | 0.567 | 0.567 | 0.567 | 59.37 / 59.37 / 59.37 |
| 5 | 0.533 | 0.533 | 0.533 | 67.63 / 67.63 / 67.63 |
| 10 | 0.633 | 0.633 | 0.633 | 63.53 / 63.53 / 63.53 |

Bit-identical at every K. So DPCC gets **3 rules × 4 candidates**; HardFlow gets **1 × 1**. Every
HardFlow S&C win below is therefore *conservative*; every loss is confounded.

**(b) The time column is unusable.** Both arms loop serially over candidates around their CPU solve,
so DPCC at fan 4 pays ~4 projections per plan to HardFlow's 1. That is why the time ratio **flips
sign with K** in §2 — HardFlow looks 2.2× *more* expensive at K=1 and 0.65× *cheaper* at K=10. Neither
is a real cost measurement. This is warning **W3** in the register
(`0f1aa7fc`, 2026-08-20 fixed it; this ladder predates the fix).

The only fan-matched HardFlow-vs-DPCC measurement that exists on `avoiding` is the K=2 batch-parity
run (§4), and it is inside the degenerate regime.

---

## 2. The full table — S&C, steps, time

MeanFlow UNet@32, `avoiding-d3il`, n=15 per cell. HardFlow rows are the single collapsed arm.
`time` = HardFlow s/step ÷ DPCC s/step, **fan-confounded per §1(b)**.

### K = 1 — `DEGENERATE` (n_active 1, n_genuine 0)

| rule | DPCC S&C | HF S&C | ΔS&C | DPCC steps | HF steps | Δsteps | time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-r` | 0.333 | **0.533** | **+0.200** | 63.47 | **59.27** | **−4.20** | 2.11× |
| `-c` | **0.600** | 0.533 | −0.067 | 73.97 | **59.27** | **−14.70** | 2.30× |
| `-t` | 0.500 | **0.533** | +0.033 | 65.47 | **59.27** | **−6.20** | 2.22× |
| `-r-tight` | 0.900 | **1.000** | **+0.100** | 64.27 | **63.77** | −0.50 | 2.18× |
| `-c-tight` | 0.933 | **1.000** | +0.067 | 72.37 | **63.77** | **−8.60** | 2.27× |
| `-t-tight` | 0.967 | **1.000** | +0.033 | **58.57** | 63.77 | +5.20 | 2.26× |

### K = 2 — `DEGENERATE` (n_active 1, n_genuine 0)

| rule | DPCC S&C | HF S&C | ΔS&C | DPCC steps | HF steps | Δsteps | time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-r` | 0.467 | **0.567** | **+0.100** | 63.03 | **59.37** | −3.67 | 1.78× |
| `-c` | **0.800** | 0.567 | **−0.233** | 97.30 | **59.37** | **−37.93** | 1.85× |
| `-t` | 0.467 | **0.567** | **+0.100** | **58.77** | 59.37 | +0.60 | 1.79× |
| `-r-tight` | **0.967** | 0.933 | −0.033 | 65.07 | 64.80 | −0.27 | 1.78× |
| `-c-tight` | 0.933 | 0.933 | 0.000 | 97.20 | **64.80** | **−32.40** | 1.81× |
| `-t-tight` | **0.967** | 0.933 | −0.033 | **59.43** | 64.80 | +5.37 | 1.77× |

### K = 5 — `OK` (n_active 3, n_genuine 2)

| rule | DPCC S&C | HF S&C | ΔS&C | DPCC steps | HF steps | Δsteps | time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-r` | 0.433 | **0.533** | **+0.100** | 69.13 | **67.63** | −1.50 | 0.66× |
| `-c` | **0.633** | 0.533 | **−0.100** | **64.33** | 67.63 | +3.30 | 0.73× |
| `-t` | 0.500 | **0.533** | +0.033 | **60.37** | 67.63 | +7.27 | 0.73× |
| `-r-tight` | 0.867 | **0.933** | +0.067 | 69.17 | **66.57** | −2.60 | 0.63× |
| `-c-tight` | 0.833 | **0.933** | **+0.100** | **61.53** | 66.57 | +5.03 | 0.65× |
| `-t-tight` | 0.933 | 0.933 | 0.000 | **60.60** | 66.57 | +5.97 | 0.61× |

### K = 10 — `OK` (n_active 5, n_genuine 4)

| rule | DPCC S&C | HF S&C | ΔS&C | DPCC steps | HF steps | Δsteps | time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-r` | 0.467 | **0.633** | **+0.167** | 72.90 | **63.53** | **−9.37** | 0.66× |
| `-c` | 0.533 | **0.633** | **+0.100** | **60.17** | 63.53 | +3.37 | 0.77× |
| `-t` | 0.400 | **0.633** | **+0.233** | **61.37** | 63.53 | +2.17 | 0.76× |
| `-r-tight` | 0.833 | 0.833 | 0.000 | 73.93 | **66.37** | **−7.57** | 0.63× |
| `-c-tight` | 0.833 | 0.833 | 0.000 | **59.40** | 66.37 | +6.97 | 0.66× |
| `-t-tight` | **0.933** | 0.833 | **−0.100** | **63.63** | 66.37 | +2.73 | 0.61× |

### What the table says

- **No K threshold.** HardFlow beats 2 of 3 rules untightened at K=1, 2 **and** 5 alike. Whatever
  separates those cells, it is not whether HardFlow ran.
- **The one clean untightened sweep is K=10** (+0.167 / +0.100 / +0.233, beats all three rules).
  That is the only cell in the ladder consistent with "more genuine steps helps".
- **The one clean tightened sweep is K=1** (+0.100 / +0.067 / +0.033, beats all three) — the fully
  degenerate configuration. It carries the curated snapshot's architecture-matched headline
  (`K1 hardflow-tightened`, S&C 1.000), and it runs **no HardFlow math**.
- **Tightened S&C falls with K on the HardFlow arm**: 1.000 → 0.933 → 0.933 → 0.833. The DPCC arm
  falls too (0.967 → 0.967 → 0.933 → 0.933), so this is largely a checkpoint property, not a
  projector property — but HardFlow falls *further*.
- **Steps are where HardFlow does best, and it is systematic**: it beats DPCC `-c` on steps by
  −14.70 (K=1), −37.93 (K=2), −32.40 (K=2 tight). DPCC `-c` at fan 4 selects long, stalled plans
  (97.30 and 97.20 steps at K=2) — the known `-c` ranking pathology. HardFlow at fan 1 cannot make
  that mistake because it has nothing to choose from.

---

## 3. Degeneracy confirmed on this exact ladder

`n_active` predicted from `hardflow_regime(K, 0.5)` vs measured, from the same CSV:

| K | predicted `n_active` | measured NFE/plan | `K + n_active` | measured solves/plan (÷ 2.03) |
|---:|---:|---:|---:|---:|
| 1 | **1** | 2.00 | 2 | **1.00** |
| 2 | **1** | 3.00 | 3 | **1.00** |
| 5 | **3** | 8.00 | 8 | **3.00** |
| 10 | **5** | 15.00 | 15 | **5.00** |

Exact at every K. (The raw per-plan figures are 4.07 / 6.10 / 16.24 / 30.47 NFE and 2.03 / 2.03 /
6.09 / 10.16 solves; both carry the same constant 2.03 plans-per-recorded-step factor, which cancels.)

**So K=1 and K=2 on this ladder ran exactly one NLP solve per plan — the terminal one.** Zero
HardFlow-specific arithmetic. DEGENERACY §9.1's prediction, confirmed on the benchmark corpus.

---

## 4. The one fan-matched `avoiding` measurement — and it is K=2

`HF_Batch_Parity/DA_20260824_mpc1_parity_MF_vs_FM.md`, both arms at fan 1, K=2, `A = 0.5`:

- **On `fm`, the two arms produced bit-identical rollouts.** Step delta exactly **0.00**
  (71.0 / 61.5 / 62.5 in both arms), zero violations on both sides, at **2.45× the cost.**
- Measured NFE: `diffuser` 2.03/plan, `hardflow_new-c` **3.05** → `3 = 2 + n_active` → `n_active = 1`.
- On `mf` they differ (HF loses TR 0.50 vs 1.00) — that is the D3 confound (§6), not the projector.

**That is what a degenerate cell is: the same algorithm, at 2.45× the price.** It also fixes the
honest cost ratio at K=2 — which means §2's `1.78–1.85×` at K=2 is roughly right, and §2's
`0.61–0.77×` at K≥5 is the fan artefact, not a HardFlow saving.

---

## 5. UAV corridor — the fan-matched dataset, which disagrees

`temp/2108/batch_uav_20260821_105229`, corridor, seed 6, n=10, **fan B=4 on both arms**, `A = 0.5`.
Here HardFlow keeps all three selectors, so rules pair up properly.

**K=5, `fm` (no D3 confound):**

| rule | DPCC S&C | HF S&C | DPCC steps | HF steps | DPCC ms/step | HF ms/step |
|---|---:|---:|---:|---:|---:|---:|
| `-r` | 0.90 | **1.00** | **267.6** | 275.5 | **164.7** | 474.7 |
| `-c` | 0.90 | **1.00** | **270.3** | 275.8 | **167.0** | 492.9 |
| `-t` | 0.90 | **1.00** | **267.3** | 273.6 | **163.8** | 477.5 |

**K=5, `mf`:**

| rule | DPCC S&C | HF S&C | DPCC steps | HF steps | DPCC ms/step | HF ms/step |
|---|---:|---:|---:|---:|---:|---:|
| `-r` | 0.50 | 0.50 | 301.0 | **288.9** | **238.3** | 851.0 |
| `-c` | 0.60 | **0.80** | 288.8 | **273.4** | **224.7** | 480.5 |
| `-t` | 0.50 | **0.80** | 313.1 | **272.9** | **269.7** | 501.1 |

Across K on UAV, rule-matched and excluding saturated/floored pairs: **K=1–2 → 2 wins, 4 losses
(Δ −6 episodes); K=5 → 5 wins, 0 losses (Δ +8).** The switch lands on the degeneracy boundary.

**At matched fan, HardFlow is 1.86–3.57× DPCC's cost — never cheaper.** That is the honest cost
picture, and it is the opposite of what §2's confounded time column suggests.

---

## 6. Confound carried on `mf` / `af` at low K

Arm C queries the instantaneous field (`h = 0`); arms A/B use the trained interval field (`h = dt`).
At K=1 arm C is projecting a first-order Euler extrapolation across `Δt = 1` instead of the trained
MeanFlow jump — a **different and worse base trajectory**. The two converge as K rises.

The entire `avoiding` ladder in §2 is MeanFlow, so **every row there carries this**. It also means
the K=1/K=2 → K=10 movement on that ladder has two explanations pulling together (D3 fading, genuine
steps appearing) and this data cannot separate them. `fm` is the only clean engine, and the only
`fm` HardFlow rows that exist are UAV (§5) and the K=2 batch-parity cell (§4).

---

## 7. Verdict

**On the benchmark (`avoiding-d3il`, 5 seeds, n=15):** HardFlow shows **no K threshold**. It beats
2 of 3 DPCC rules untightened at K=1, 2 and 5 alike; sweeps all three only at K=10 untightened; and
its one clean tightened sweep is at K=1, where it runs no HardFlow math. Its consistent, real
advantage is **steps**, and the mechanism is unflattering — DPCC `-c` at fan 4 picks stalled 97-step
plans, and HardFlow at fan 1 simply cannot.

**On UAV corridor (fan-matched, n=10, 1 seed):** the win does switch on at K=5, cleanly, 5–0.

**Cost:** at matched fan HardFlow is **1.86–3.57×** DPCC. There is no cell, on either task, where
HardFlow is both better and cheaper. Under the Pareto rule nothing here is a win — it is a
trade-off.

**What is actually established:** K=1 and K=2 execute no HardFlow math (§3, §4 — proven, not
inferred). Everything about whether HardFlow *helps* remains open, because the two datasets that
could answer it disagree and each is confounded in a different way.

---

## 8. What would settle it — one run

**`avoiding-d3il`, K ∈ {2, 5, 10}, `fm` engine, `FMPCC_MPC_BATCH=1` on both arms, untightened,
5 seeds × 20 trials.**

That single sweep removes every confound at once: `fm` kills D3 (§6), fan 1 on both arms kills W3
(§1) and makes the time column real, `avoiding` is the benchmark, and K spans the boundary.
Untightened because §2 shows the tightened arm has no headroom to win in.

**Secondary, already in the data — an unexplained solver-failure spike at exactly K=5:**

| K | mean `nlp_failures` | mean solves | failure rate | worst cell |
|---:|---:|---:|---:|---:|
| 1 | 0.8 | 121 | 0.66 % | 12 |
| 2 | 0.8 | 121 | 0.66 % | 6 |
| **5** | **6.9** | 412 | **1.67 %** | **66** |
| 10 | 1.3 | 645 | 0.21 % | 10 |

A failed solve returns IPOPT's last iterate, which is **not guaranteed feasible** — so for those
plans the terminal-solve safety guarantee does not hold. K=5 is 2.5–8× the failure rate of every
other K on the ladder, and it is exactly the K where HardFlow underperforms untightened (§2). This
costs no cluster time to investigate and has never been looked at. The new `[hardflow][NLP-FAILURE]`
banner will surface it live on the next run.

---

## 9. Status

Re-analysis only. No new runs; no source DA edited. Every figure is computed from the two batch CSVs
named in the header; regime labels from `hardflow_regime(K, 0.5)`.
