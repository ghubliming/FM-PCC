# DA — Full MPC-fan parity on `avoiding-d3il`: does candidate selection buy anything, and MF vs FM at K=2

**Date:** 2026-08-24
**Data root:** `temp/2308/` (eval jobs 24991, 24992) + batch DA `temp/2308/batch_avoiding_combined_20260824_091518/`
**Git rev of the runs:** `a649a70`
**Task:** `avoiding-d3il` · **Seed:** 6 (single seed) · **n_trials:** 2 per variant per scenario · **Status:** both eval jobs completed cleanly
**Code enabling this run:** `logs_in_develop/HF_Batch_Parity/CHANGELOG_20260823_mpc_fan_arms_ab.md` (`FMPCC_MPC_BATCH`)

---

## 0. TL;DR

Four questions were asked of this run. Answers first:

| # | Question | Answer |
|---|---|---|
| **Q1** | Is the MPC candidate fan dragging performance? | **Partly. It is pure waste for `-r`, actively harmful for `-c-tightened`, and load-bearing only for untightened `-c`.** |
| **Q2** | Does `mpc=1` speed things up, and by 4×? | **Yes, 1.24–1.36× end-to-end. Not 4× — the projection scales ~3.3× and the generator not at all.** |
| **Q3** | DPCC vs HardFlow once *both* run at fan 1? | **DPCC wins the comparison we report (tightened): equal-or-better S&C at 2.4× lower cost. HardFlow's one win is *untightened*, where its in-loop solve beats post-hoc DPCC 0.500 vs 0.333 — still at 2.3× the cost.** |
| **Q4** | MF K=2 vs FM K=2? | **No separation on the DPCC arm (both 3/3 scenarios). FM leads only on the HardFlow arm, by one episode.** |

**What it shows.** With the fan matched at 1 in every arm, DPCC-`c-tightened` reaches
**S&C = 1.00 in all three scenarios on both models** at **0.020–0.021 s/step**, while
HardFlow-`c-tightened` costs **0.047–0.051 s/step** for the same or worse S&C. The
2026-08-20 "HardFlow is ~25% cheaper" artefact is now closed from both directions.
Dropping the DPCC fan 4 → 1 costs **zero** S&C on `-r`/`-r-tightened`/`-t-tightened`, *gains*
S&C 0.833 → 1.000 **and 33.5 steps** on `-c-tightened`, and loses 1.00 → 0.50 S&C on untightened `-c`.

**What it does not show.** 6 episodes per variant (2 trials × 3 scenarios, one seed). Per-scenario
S&C resolution is **0.50**; pooled resolution is **0.167**. No per-rollout export exists for the
avoiding batch, so **no significance test is computable** — every S&C difference below is a
1–2 episode difference. Q1's fan effect is corroborated by timing and is consistent in direction
across six variants; Q4's is not corroborated by anything and should not be quoted.

**Action arising.** Drop `-r`/`-r-tightened` to fan 1 now (bit-identical output, 1.29× cheaper);
hold `-c-tightened` until the baseline is re-run at fan 1 and the 5-seed × 20-trial repeat lands.
Full reasoning and the fair-comparison objection in **§8**.

---

## 1. What ran

| Job | What | Result |
|---|---|---|
| 24991 | `eval_meanflow_hardflow`, K=2, 13 variants × 3 scenarios | completed, 22:23→22:30 UTC |
| 24992 | `eval_fmv3_hardflow_job`, K=2, 8 variants × 3 scenarios | completed, 22:29→22:34 UTC |

| Candidate | Model | params | arms A/B fan | arm C fan |
|---|---|---:|---:|---:|
| **C147** `H8_K2_Meuler_T0.5_A0.5_B1_D…MeanFlowODE_msgmpc1` | MeanFlow `bbunet` | 4.0 M | **1** | 1 |
| **C64** `K2_thres0.5_mpc1_n2_msgmpc1` | FMv3ODE `a1.5_b1.0_aw10` | 4.0 M | **1** | 1 |
| C145 `H8_K2_Meuler_T0.5_A0.5_B1_D…MeanFlowODE` (2026-08-11) | MeanFlow `bbunet` | 4.0 M | 4 | 1 |

**Eval protocol — identical across C145/C147/C64.** H8, K=2, `dpcc_threshold = 0.5`,
`hf_act_threshold = 0.5`, `replan_steps = 1`, seed 6, `n_trials = 2`, scenarios
`top-right-hard` (TR) / `top-left-hard` (TL) / `both-hard` (BH), enlarge margin 0.025 for
`-tightened`. **C145 vs C147 differ in `batch_size` and nothing else** — proven in §6.2.

### 1.1 Step-comparison rule (applies to every table below)

**`n_steps` is only compared between cells where BOTH sides scored S&C = 1.00.** Episodes terminate
early on collision, so a failing variant records *fewer* steps — the metric is anti-correlated with
success and comparing it at unequal S&C flatters the worse variant. Across all 102 (candidate,
variant, scenario) cells in this run:

| S&C | cells | `n_steps` min / median / max |
|---:|---:|---|
| 0.00 | 22 | 36.5 / **52.0** / 75.5 |
| 0.50 | 25 | 50.0 / **63.5** / 94.0 |
| 1.00 | 55 | 57.0 / **62.5** / 99.0 |

Every step figure below is therefore annotated with how many of the 3 scenarios were fully
successful on both sides (`valid n/3`); where that is 0/3, steps are reported as **n/c**
(not comparable) and no step claim is made.

**Coverage.** C147: 39/39 runs. C64: 24/24 runs. C64 carries only `dpcc-c-tightened` for arm B —
that is `config/hardflow_projection_eval.yaml` by design (arm B there is a port-correctness
safeguard, not the full DPCC matrix), so the untightened DPCC rows exist for MF only.

---

## 2. Q1 — Is the MPC candidate fan dragging performance?

**Answer: it depends entirely on the variant, and for the variant we report as "DPCC" the fan is
actively harmful.** MF, C145 (fan 4) → C147 (fan 1), seed 6.

### 2.1 Pooled over the three scenarios

Steps are over fully-successful cells only, per §1.1.

| arm B variant | S&C @4 | S&C @1 | valid | steps @4 | steps @1 | Δsteps | s/step @4 | s/step @1 | verdict at fan 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dpcc-r` | 0.333 | 0.333 | 0/3 | n/c | n/c | — | 0.027 | 0.021 | **free win** — rollouts *bit-identical*, 1.29× cheaper |
| `dpcc-c` | **0.667** | 0.333 | 0/3 | n/c | n/c | — | 0.026 | 0.021 | **loss** (−0.333 S&C) |
| `dpcc-t` | **0.500** | 0.333 | 0/3 | n/c | n/c | — | 0.027 | 0.021 | **loss** (−0.167 S&C) |
| `dpcc-r-tightened` | 1.000 | 1.000 | 3/3 | 63.17 | 63.00 | −0.17 | 0.027 | 0.021 | **free win** (steps unchanged) |
| `dpcc-c-tightened` | 0.833 | **1.000** | 2/3 | 94.00 | **60.50** | **−33.50** | 0.027 | 0.021 | **strict win** (Pareto) |
| `dpcc-t-tightened` | 1.000 | 1.000 | 3/3 | 58.67 | 63.00 | **+4.33** | 0.027 | 0.021 | **trade-off** (non-dominated) |

`dpcc-c`/`dpcc-t`/`dpcc-r` have **no** fully-successful scenario at either fan, so their step
columns carry no information — the fan-4 `dpcc-c` figure of 85.00 vs fan-1's 58.67 that an earlier
cut of this analysis reported as an improvement is an artefact of §1.1.

### 2.2 Per scenario — S&C and steps

S&C first (`fan4 → fan1`); steps quoted **only** where both sides are 1.00, greyed as n/c otherwise.

| variant | TR | TL | BH |
|---|---|---|---|
| `dpcc-r` | 0.00 → 0.00 · n/c | 0.50 → 0.50 · n/c | 0.50 → 0.50 · n/c |
| `dpcc-c` | 0.00 → 0.00 · n/c | **1.00 → 0.50** · n/c | **1.00 → 0.50** · n/c |
| `dpcc-t` | **0.50 → 0.00** · n/c | 0.50 → 0.50 · n/c | 0.50 → 0.50 · n/c |
| `dpcc-r-tightened` | 1.00 → 1.00 · 68.0 → 68.0 | 1.00 → 1.00 · 62.5 → 62.0 | 1.00 → 1.00 · 59.0 → 59.0 |
| `dpcc-c-tightened` | **0.50 → 1.00** · n/c | 1.00 → 1.00 · **99.0 → 62.0** | 1.00 → 1.00 · **89.0 → 59.0** |
| `dpcc-t-tightened` | 1.00 → 1.00 · **57.0 → 68.0** | 1.00 → 1.00 · 60.5 → 62.0 | 1.00 → 1.00 · 58.5 → 59.0 |

Reading it:
- **`dpcc-r`: every cell bit-identical** (§6.1) — the fan changed nothing at all, so no step question arises.
- **`dpcc-c` loses S&C in TL *and* BH** (1.00 → 0.50 both). Two scenarios, same direction.
- **`dpcc-c-tightened` is the clean win:** TR goes 0.50 → 1.00, and in the two cells that were
  already fully successful it drops **37.0** and **30.0** steps. Its TR step figure is not quotable
  (fan 4 only half-succeeded there).
- **`dpcc-t-tightened` pays for its 1.29× in steps** — worse in all 3 fully-valid cells, worst
  TR +11.0. This is the one variant where the fan was buying something measurable and legitimate.

### 2.3 Mechanism

**`dpcc-r` is unchanged in all six cells.** `policies.py` sets `which_trajectory = 0` for `-r`
regardless of fan, so `-r` always used candidate 0 — the other 3 SLSQP solves per step were computed
and discarded. For this variant the fan of 4 was **pure waste**: a free 1.29× is available with
literally zero behavioural change.

**`-c` is where the fan lives, and its sign flips with the margin.** At fan 1 all of
`dpcc-{r,c,t}` reproduce `dpcc-r` @ fan 4 exactly (0.333 / 58.67, all three scenarios), so every
difference between the fan-4 rows is the selection rule alone:
- **untightened:** ranking 4 candidates by projection cost recovers a feasible plan that candidate 0
  misses — 1.00 vs 0.50 in TL **and** BH. Real, and consistent across two scenarios.
- **tightened:** the same rule picks the *lazier* trajectory — on the two cells that are fully
  successful at both fans, **99.0 → 62.0** and **89.0 → 59.0** steps — and it loses TR outright
  (0.50 vs 1.00). The candidate needing least projection is not the one that reaches the goal
  soonest, and once the margin already guarantees feasibility that is all the ranking selects for.

**`-t-tightened` is non-dominated:** S&C ties at 1.00 in all three scenarios, time drops 1.3×, steps
rise in all three (worst TR 57.0 → 68.0). `temporal_consistency` was buying shorter paths with the
fan. Calling fan 1 "better" here would be wrong.

---

## 3. Q2 — Does `mpc=1` speed up, and by how much?

**Answer: yes, 1.24–1.36× end to end. It is not 4×, because only the CPU projection scales.**

### 3.1 End-to-end `avg_time` ratio, fan 4 ÷ fan 1

| variant | TR | TL | BH |
|---|---:|---:|---:|
| `dpcc-r` | 1.26× | 1.30× | 1.29× |
| `dpcc-c` | 1.26× | 1.27× | 1.28× |
| `dpcc-t` | 1.24× | 1.28× | 1.31× |
| `dpcc-r-tightened` | 1.29× | 1.33× | 1.28× |
| `dpcc-c-tightened` | 1.36× | 1.29× | 1.27× |
| `dpcc-t-tightened` | 1.27× | 1.33× | 1.30× |

Consistent across all 18 cells; no variant escapes the band.

### 3.2 Why not 4× — the generator does not scale, the projector does

`diffuser` is network-only. Its cost is **flat in the fan**:

| scenario | `diffuser` @ fan 4 | @ fan 1 | ratio |
|---|---:|---:|---:|
| TR | 0.02055 | 0.02070 | 0.99 |
| TL | 0.01883 | 0.01839 | 1.02 |
| BH | 0.01886 | 0.01848 | 1.02 |

**Generating 4 candidates costs the same as generating 1** — that fan is one batched GPU forward.
Projection-only cost (`arm − diffuser`, ms/step) is what moves:

| variant | proj @4 | proj @1 | ratio |
|---|---:|---:|---:|
| `dpcc-r` TL / BH | 8.40 / 7.91 | 2.57 / 2.22 | 3.27× / 3.56× |
| `dpcc-c-tightened` TL / BH | 8.51 / 7.66 | 2.84 / 2.37 | 2.99× / 3.24× |
| `dpcc-t-tightened` TL / BH | 8.97 / 8.09 | 2.56 / 2.29 | 3.50× / 3.53× |

**~3.0–3.6×, sub-linear.** The serial per-candidate SLSQP loop recovers ~3.3 of the ideal 4×; the
missing ~0.7 is fixed per-plan overhead the fan does not multiply. Net: `mpc` is a **CPU-side knob**.
An 18% end-to-end effect, not a 4× one, because at K=2 the 18.5 ms generator dominates the 2–8 ms
projector.

⚠️ **TR is unusable for the delta method** — its `diffuser` reference is inflated (0.0207 vs 0.0184
in TL/BH; it is the first eval of the job and pays warm-up), which drives proj@1 to 0.13 ms and to
−0.08 ms for the tightened arms. Only TL/BH are quoted.

---

## 4. Q3 — DPCC vs HardFlow at true parity (fan = 1 in every arm)

**Answer: on the tightened arm DPCC dominates — equal-or-better S&C, comparable steps, 2.4×
cheaper. HardFlow's only win is on the untightened arm (§4.2), and it is one episode.**

### 4.1 Tightened — the arm we report

Both arms at fan 1, K=2, `act_threshold = 0.5`. S&C / steps / s-per-step:

| model | variant | TR | TL | BH | pooled S&C | pooled s/step |
|---|---|---|---|---|---:|---:|
| MF | `dpcc-c-tightened` | **1.00** / 68.0 / 0.021 | 1.00 / 62.0 / 0.021 | 1.00 / 59.0 / 0.021 | **1.000** | **0.021** |
| MF | `hardflow_new-c-tightened` | 0.50 / 68.0 / 0.051 | 1.00 / 64.0 / 0.050 | 1.00 / 64.0 / 0.047 | 0.833 | 0.049 |
| FM | `dpcc-c-tightened` | 1.00 / 71.0 / 0.019 | 1.00 / 61.5 / 0.020 | 1.00 / 62.5 / 0.020 | **1.000** | **0.020** |
| FM | `hardflow_new-c-tightened` | 1.00 / 71.0 / 0.049 | 1.00 / 61.5 / 0.049 | 1.00 / 62.5 / 0.050 | **1.000** | 0.049 |

- **FM: the two arms are behaviourally identical.** All 3 cells are fully successful on both sides
  (valid 3/3) and the step delta is **exactly 0.00** — 71.0/61.5/62.5 in both arms — with zero
  violations throughout. HardFlow costs **2.45×** more for the identical rollout.
- **MF: HardFlow is worse on both axes.** It loses TR outright (0.50 vs 1.00), and on the 2 valid
  cells it takes **+3.50 steps** (60.50 vs 64.00) — at 2.4× the cost.
- **Cost decomposition at fan 1:** DPCC projection ≈ **2.4 ms/step** (1 SLSQP), HardFlow ≈
  **30 ms/step** (~1 IPOPT solve/step), on top of an 18.5 ms generator. Arm C also burns **3 NFE/plan
  vs 2** — FM `diffuser` 248 NFE / (61.0 × 2) = 2.03; FM `hardflow_new-c` 337 / (55.17 × 2) = 3.05.

### 4.2 Untightened — where HardFlow does earn something

Drop the 0.025 margin and the in-loop solve is doing work post-hoc projection cannot. MF only —
C64 ships no untightened `dpcc-*` row. S&C / steps / violations (total magnitude):

| model | variant | TR | TL | BH | pooled S&C | s/step |
|---|---|---|---|---|---:|---:|
| MF | `dpcc-c` (post-hoc) | 0.00 / 52.0 / 2 (0.05) | 0.50 / 63.5 / 2 (0.02) | 0.50 / 60.5 / 0 (0.00) | 0.333 | **0.021** |
| MF | `hardflow_new-c` (in-loop) | 0.00 / 45.0 / 3 (0.08) | 0.50 / 63.5 / 2 (0.01) | **1.00** / 61.5 / 0 (0.00) | **0.500** | 0.049 |
| FM | `hardflow_new-c` (in-loop) | 0.00 / 36.5 / 1 (0.00) | 1.00 / 60.0 / 0 (0.00) | 1.00 / 69.0 / 0 (0.00) | 0.667 | 0.048 |

**This is the one result in the run that favours HardFlow, and it is thin:** the entire pooled
0.500-vs-0.333 gap is **one episode in BH**; TR and TL tie exactly. HardFlow also carries *more*
violations than DPCC in TR (3 vs 2, magnitude 0.08 vs 0.05). And it still costs **2.3×**.
**No step claim is available here at all** — 0/3 cells are fully successful on both sides.

So the fair reading of Q3 is not "HardFlow loses everywhere." It is: **once the feasible set is
tightened, in-loop constrained sampling has nothing left to add** — DPCC's post-hoc projection
already reaches S&C 1.00 in every scenario on both models — **so HardFlow's 2.4× is spent on a
guarantee that is already met.** Its mechanism only shows up in the regime where post-hoc projection
is allowed to fail, and one episode is not enough to size that advantage.

⚠️ **Open item, flagged not resolved.** At matched fan and ~1 solve/step in both arms, HardFlow's
projection is **~12–15× DPCC's per step**. `DA_20260820_HF_lower_avgtime_batchsize_confound.md`
estimated IPOPT at only **1.8–2.2× per solve** vs scipy SLSQP. Either per-call CasADi/IPOPT setup
dominates at K=2, or the solves/step accounting differs between arms. No per-solve cost claim should
be repeated until this is profiled.

---

## 5. Q4 — MF K=2 vs FM K=2 on `avoiding-d3il`, both at fan 1

**Answer: no separation on the DPCC arm. FM leads only on the HardFlow arm, by a single episode.**

S&C per scenario, then steps over fully-successful cells only (§1.1).

| variant | model | TR | TL | BH | pooled S&C | valid | steps (valid) | s/step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `diffuser` | MF | 0.00 | 0.00 | 0.00 | 0.000 | 0/3 | n/c | 0.019 |
| `diffuser` | FM | 0.00 | 0.00 | 0.50 | 0.167 | 0/3 | n/c | 0.018 |
| `dpcc-c-tightened` | MF | 1.00 | 1.00 | 1.00 | **1.000** | 3/3 | **63.00** | 0.021 |
| `dpcc-c-tightened` | FM | 1.00 | 1.00 | 1.00 | **1.000** | 3/3 | 65.00 | 0.020 |
| `hardflow_new-c` | MF | 0.00 | 0.50 | 1.00 | 0.500 | 1/3 | **61.50** | 0.049 |
| `hardflow_new-c` | FM | 0.00 | 1.00 | 1.00 | **0.667** | 1/3 | 69.00 | 0.048 |
| `hardflow_new-c-tightened` | MF | 0.50 | 1.00 | 1.00 | 0.833 | 2/3 | 64.00 | 0.049 |
| `hardflow_new-c-tightened` | FM | 1.00 | 1.00 | 1.00 | **1.000** | 2/3 | **62.00** | 0.049 |

- **On `dpcc-c-tightened` — the matched, reported arm — the two tie at 3/3 scenarios**, and with
  all 3 cells valid MF is **2.0 steps shorter** (63.00 vs 65.00; per-cell 68.0/62.0/59.0 vs
  71.0/61.5/62.5) at +0.001 s/step. Non-dominated, no winner.
- **FM's S&C lead is confined to arm C** and is one episode each time:
  `hardflow_new-c-tightened` 1.000 vs 0.833 (TR), `hardflow_new-c` 0.667 vs 0.500 (TL).
- **Steps cut the other way on `hardflow_new-c`:** in the single cell where both fully succeed (BH),
  FM takes **+7.50 steps** (69.0 vs 61.5). One cell — not a finding, but it blocks any claim that
  FM dominates arm C.
- Unguided floors are both unsafe (S&C 0.000 / 0.167; 12–32 violations per scenario) and neither
  has a fully-successful cell, so all safety here comes from the projector, not the field.

**This does not support "FM beats MF at K=2."** It supports "at K=2 with a projector, both models are
saturated on the DPCC arm and 6 episodes cannot separate them."

---

## 6. Validity checks

### 6.1 Selection collapse at fan 1 — ✅

`policies.py` and `hardflow_projection.py::_select` both return candidate 0 when there is one
candidate. Bit-for-bit check on the raw npz values:

- **Every behavioural metric is bit-identical** across `-r`/`-c`/`-t` — `n_success`,
  `n_success_and_constraints`, `n_steps`, `n_violations`, `total_violations`,
  `collision_free_completed`, `nfe`, `nlp_solves`, `nlp_failures` — in all four families
  (`dpcc-*`, `dpcc-*-tightened`, `hardflow_new-*`, `hardflow_new-*-tightened`), both jobs.
- **Only `avg_time` / `avg_time_std` differ** — 12 diffs per family, all wall-clock jitter on
  identical work (MF DPCC 0.020828 / 0.020839 / 0.021009; FM HF 0.050251 / 0.047763 / 0.048075).

The collapse holds **independently in each arm**; it does not mean DPCC equals HardFlow (§4).
**Consequence:** the trio is redundant compute at `mpc=1`. A repeat should run one representative
per family and spend the saved time on seeds and trials.

### 6.2 The `FMPCC_MPC_BATCH` patch is isolated — ✅

C147 (arms A/B = 1) vs C145 (arms A/B = 4), arm C at B1 in both:

| variant | C145 | C147 |
|---|---|---|
| `hardflow_new-c` | 0.500 / 56.67 / 0.049 | 0.500 / 56.67 / 0.049 |
| `hardflow_new-c-tightened` | 0.833 / 65.33 / 0.049 | 0.833 / 65.33 / 0.049 |
| `diffuser` | 0.000 / 63.00 / 0.019 | 0.000 / 63.00 / 0.019 |

Identical per scenario as well. Every §2 difference is attributable to the DPCC fan alone.

### 6.3 Knob reached the cluster — ✅

```
[ hardflow ] HFFM_BATCH=1 (arm C)  FMPCC_MPC_BATCH=1 (arms A/B)  HFFM_ACT_THRESHOLD=0.5  HFFM_FLOW_STEPS=2
[ eval ] replan_steps=1  |  mpc fan: arms A/B=1, arm C=1
[ config/avoiding-d3il ] custom_msg="mpc1" -> results dirs end in "_msgmpc1"
```

---

## 7. What this run cannot say

1. **Steps are only comparable on fully-successful cells (§1.1), which leaves very few.** Across
   the three headline contrasts the usable cell counts are 0–3 of 3; `dpcc-c`, `dpcc-t`, `dpcc-r`
   and both `diffuser` rows have **no** usable cell. Any step conclusion here rests on 1–3 cells
   of 2 episodes each.
2. **No significance is computable.** 6 episodes/variant, one seed. Per-scenario S&C granularity is
   0.50; pooled 0.167. The avoiding batch exports no per-rollout detail (unlike the UAV batch's
   `per_rollout_detail.csv`), so no paired/McNemar test is possible. Q4's separations are 1 episode.
3. **No fan-4 control for FM at K2/thres0.5.** Gen12 has only `K20_thres0.5_mpc4`, so §2 is
   **MF-only**; FM's numbers cannot be attributed to the fan change.
4. **No diffusion-DPCC at fan 1.** The DA target `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`
   (C16) exists at fan 4 only. Whether the *baseline* also gains 1.29× at fan 1 — preserving the gap
   rather than closing it — is unanswered, and it is load-bearing for any paper claim from §2.
5. **Untightened DPCC exists for MF only** (§1 coverage), so §2's untightened finding is unreplicated.
6. **Aggregator key mismatch.** MF writes `nfe_total` / `nlp_solves_total` / `hf_batch_size` /
   `hf_act_threshold` / `is_hardflow`; Gen12 writes `nfe` / `nlp_solves` / `batch_size` /
   `activation_threshold` / `flow_steps` / `dpcc_threshold`. So
   `candidates_multidimensional_aggregated.csv` shows **`nan` in the compute columns for Gen12
   candidates**. §4/§5 NFE and NLP figures were read under the Gen12 names by hand; the loader needs
   an alias map.

---

## 8. Recommendation — should `mpc` be dropped from 4 to 1?

**Yes for the tightened variants, no as a global default yet, and one variant can change today at
zero risk.** This is not a "better results" change: it is **same-or-better results at 1.29× less
compute**, plus a real step win on `-c-tightened`, against a documented loss on the untightened arm.

### 8.1 What the data licenses

| variant | fan 4 → 1 | drop to 1? |
|---|---|---|
| `dpcc-r`, `dpcc-r-tightened` | rollouts **bit-identical**, 1.29× cheaper | **Yes, unconditionally** — dead compute, not a trade-off |
| `dpcc-c-tightened` | S&C 0.833 → 1.000, **−33.5 steps** (2/3 valid), 1.29× cheaper | Yes on this evidence, **pending §8.2** |
| `dpcc-t-tightened` | S&C ties 1.000, **+4.33 steps** (3/3 valid), 1.29× cheaper | Only as a deliberate steps-for-time trade |
| `dpcc-c` (untightened) | S&C **0.667 → 0.333** | **No** |
| `dpcc-t` (untightened) | S&C **0.500 → 0.333** | **No** |

`-r` is the free case for a concrete reason: `policies.py` sets `which_trajectory = 0` for `-r`
regardless of fan, so the other 3 SLSQP solves per step were computed and discarded. §6.1 confirms
the rollouts are bit-identical across the fan change.

The rule the data fits: **once the tightening margin already guarantees feasibility, candidate
ranking has nothing left to select for and simply picks the laziest trajectory** (`-c`: 99.0 → 62.0
and 89.0 → 59.0 steps). Without the margin, ranking is doing real work and dropping it costs S&C.

### 8.2 Three reasons not to flip it globally yet

1. **The baseline has not been tested at fan 1.** diffusion-DPCC `K20/aw10` *is* the benchmark, and
   the candidate fan is part of DPCC's published method. Three outcomes need different handling:
   - baseline also improves at fan 1 → change both, gap preserved, clean;
   - baseline degrades → dropping the fan on the baseline **weakens the thing we claim to beat**,
     which is a fair-comparison objection a reviewer will raise;
   - we drop ours and not the baseline's → mismatched configs, not a comparison.

   This is why it is run #1 in §9 — a **prerequisite**, not an optimisation.
2. **Choosing fan 1 because it scored better here is selecting on the eval.** The `-c-tightened`
   S&C gain is **one episode at seed 6**. The step gain is sturdier (continuous, and large) but rests
   on 2 fully-successful cells. The decision belongs to the 5-seed × 20-trial run, not to this one.
3. **K=2 only.** Every headline DPCC number is K=20, where the fan's effect is untested and the
   18.5 ms generator no longer dominates the 2–8 ms projector — the 1.29× will not carry over
   unchanged.

### 8.3 Decision

- **Now:** set `-r` / `-r-tightened` to fan 1. Provably identical output, 1.29× cheaper.
- **Hold** `-c-tightened` until run #1 (baseline at fan 1) and run #3 (5 seeds × 20 trials) return.
- **If both hold:** make fan 1 the default for the tightened variants, apply it to **baseline and
  method alike**, and state the change explicitly — including the untightened loss — rather than
  quietly shipping the cheaper config.

### 8.4 Which projector should be reported — DPCC @ mpc1 or HardFlow @ mpc1?

**Report DPCC @ mpc1.** On this data HardFlow costs **2.4×** and returns identical (FM: 3/3 valid
cells, step delta exactly 0.00, zero violations both) or worse (MF: loses TR 0.50 vs 1.00, +3.50
steps) behaviour. Paying 2.4× for a bit-identical rollout is not defensible in a results table.

**But that verdict is narrower than it looks, and it is not a refutation of HardFlow.** In-loop
constrained sampling is theoretically the stronger construction — it keeps the sample *on* the
constraint manifold during generation, where post-hoc projection can push a trajectory off the
learned data manifold to satisfy the constraint. That advantage can only appear **where post-hoc
projection fails**, and in this configuration it never does: `dpcc-c-tightened` scores S&C **1.000
in every scenario on both models**. There is no headroom for a better projector to occupy.

The one signal pointing HardFlow's way is exactly where DPCC starts to break — the **untightened**
arm (§4.2), where DPCC falls to 0.333 and HardFlow holds 0.500. One episode, but it is the right
direction and the right regime.

**This run also handed HardFlow the setting where it has no structural advantage.** Per the
benchmark hierarchy, HardFlow's claim is that it beats the DPCC projector *via a lower projection
threshold* — constraint satisfaction for **less** projection. Here `hf_act_threshold` was pinned to
0.5 to match DPCC's `dpcc_threshold`, so both arms solve **~1 NLP per plan**
(HF: 125 solves / (61.5 steps × 2 trials) = 1.02; DPCC: `snapping_start_idx = 1` at K=2 ⇒ one SLSQP),
and HardFlow's entire 2.4× is per-solve cost with nothing bought back.

⚠️ **At K=2 the threshold knob has almost no range.** The gate is `k >= int((1 − thr) · K)` and the
terminal step is always solved, so at K=2 both `thr = 0.5` and `thr = 0.0` reduce to a single solve.
**HardFlow's actual claim is untestable at K=2** — it needs a K where the threshold can vary the
solve count.

**So:** report DPCC @ mpc1 today; give HardFlow its real test at **K=10–20 with an
`hf_act_threshold` sweep**, and/or on the untightened / harder-constraint setting where post-hoc
projection is allowed to fail. If HardFlow holds S&C at a threshold where DPCC cannot, that is both
a quality win *and* a cost win, and it is the only form in which the HardFlow claim can be made.

---

## 9. Next runs, in priority order

| # | Run | Closes | Why |
|---|---|---|---|
| 1 | diffusion-DPCC `K20/aw10` at `FMPCC_MPC_BATCH=1` | gap 4 | decides whether §2 helps or hurts the paper story |
| 2 | FM Gen12 twin at `FMPCC_MPC_BATCH=4 HFFM_BATCH=4 HFFM_FLOW_STEPS=2 HFFM_ACT_THRESHOLD=0.5` | gap 3 | gives FM its own fan contrast |
| 3 | re-run at 5 seeds × 20 trials, one variant per family | gaps 1+2 | the only way any step or S&C claim becomes quotable |
| 4 | **HardFlow `hf_act_threshold` sweep at K=10/20**, untightened + tightened | §8.4 | the only design in which HardFlow's claim is testable — at K=2 the gate has no range |
| 5 | profile one IPOPT vs one SLSQP solve at K=2 | §4 open item | reconciles ~12–15×/step with the 1.8–2.2×/solve estimate |
| 6 | alias map in the batch loader for Gen12 compute keys | gap 6 | cheap |

---

## 10. Reproduce

```bash
FMPCC_MPC_BATCH=1 HFFM_BATCH=1 HFFM_FLOW_STEPS=2 HFFM_ACT_THRESHOLD=0.5 DPCC_THRESHOLD=0.5 \
MF_BACKBONE=unet MF_HORIZON=8 \
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh

FMPCC_MPC_BATCH=1 HFFM_BATCH=1 HFFM_FLOW_STEPS=2 HFFM_ACT_THRESHOLD=0.5 DPCC_THRESHOLD=0.5 \
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
```

### 10.1 Paths on the cluster (i6-gpu-1)

`REPO_ROOT = $HOME/FMPCC/FM-PCC`, which resolves as **`/u/home/llim/FMPCC/FM-PCC`**
(the same tree also appears as `/data/home/llim/FMPCC/FM-PCC` — same location, two mount prefixes;
the eval logs print `/u/...` and the batch-DA `Full_Path` column records `/data/...`).

**Slurm job logs** — `submit.sh:22-27` writes `Slurm_Codes/logs/<YYYY-MM-DD>/<HH_MM_SS>_%x_%j.log`,
with stdout and stderr merged into the one file. The date/time are **submit** time (local), not job
start, so they sit ~30 min before the `JOB START` stamp inside the file:

```
$REPO_ROOT/Slurm_Codes/logs/2026-08-23/23_52_20_eval_meanflow_hardflow_24991.log
$REPO_ROOT/Slurm_Codes/logs/2026-08-23/23_52_28_eval_fmv3_hardflow_job_24992.log
$REPO_ROOT/Slurm_Codes/logs/latest.log        # symlink to the most recent, set by the sbatch script
```

Locally these are the copies in `temp/2308/2026-08-23/` used for this DA.

**Result artefacts** (`.npz`, `run_provenance.json`, plots) — auto-tagged `_msgmpc1`, so no historic
folder was overwritten:

```
$REPO_ROOT/logs/avoiding-d3il/plans/flow_matching_v3_meanflow/
  H8_D…MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/
  H8_K2_Meuler_T0.5_A0.5_B1_D…MeanFlowODE_msgmpc1/6/

$REPO_ROOT/logs/avoiding-d3il/plans/flow_matching_v3_hardflow/
  H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/
  K2_thres0.5_mpc1_n2_msgmpc1/6/
```

**Checkpoints read** (not written by this run):

```
$REPO_ROOT/logs/avoiding-d3il/flow_matching_v3_meanflow/
  H8_D…MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/6      # MF, EMA, step 99000
$REPO_ROOT/logs/avoiding-d3il/flow_matching_v3_ode_selectable/
  H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/6                # FM, step 98000
```

`logs/` is gitignored — it exists only on the cluster and on whatever machine pulled it down via
`Slurm_Codes/download_remote_logs/export_to_laptop.sh` (which tars `logs/` from `$REPO_ROOT` and
rsyncs it out).

⚠️ The MF eval has **no skip guard** — re-submitting rewrites `<variant>.npz` in place. Gen12 skips
finished npz unless `FORCE_OVERWRITE=1`.

All tables read from `temp/2308/batch_avoiding_combined_20260824_091518/candidates_multidimensional_raw.csv`,
candidates 147 / 64 / 145, `seed == 6`.
