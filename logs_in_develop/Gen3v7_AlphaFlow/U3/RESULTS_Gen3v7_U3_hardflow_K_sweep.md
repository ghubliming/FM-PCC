# Gen3v7 U3 — first HardFlow-on-α-Flow matched-K sweep: results & analysis

**Runs**: jobs **24044 / 24045 / 24046 / 24047 / 24048** (K = 1 / 2 / 5 / 10 / 20), node i6-gpu-1, 2026-07-30 23:06 UTC.
**Git**: `cb859e3` — *"Gen3v7 U3feat: port HardFlow arm to AlphaFlow with full evaluation infrastructure and gates"*.
**Config**: `config/alphaflow_projection_eval.yaml`, seed **6**, `n_trials: 2`, 3 halfspace scenarios, `HFFM_BATCH=4`, `HFFM_ACT_THRESHOLD=0.5`, `candidate_cost: prox`.
**Checkpoint**: `bbsit` (α-Flow's own SiT), `ai1.0_ae0.0_ag25.0_rf0.5`, **step 68000**, EMA weights.
**Gates**: `GATES: ALL PASS (4/4)` in all five logs — H0/H1/H3/H4 executed before every eval.
**Raw data**: `temp/3107/`. **Pre-port control**: `temp/2807/` (U2, DPCC-only, same checkpoint).
**Cross-generation baseline**: Gen3v6 post-fix_4 sweep in `temp/2026-07-30/II/`, analysed in
[`../../Gen3v6_MeanFlow/fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md`](../../Gen3v6_MeanFlow/fix_4/RESULTS_Gen3v6_fix_4_post_fix_K_sweep.md) (**v6-RES** below).

Companion docs: [`CHANGELOG_Gen3v7_U3_hardflow_arm.md`](CHANGELOG_Gen3v7_U3_hardflow_arm.md) (what was built),
[`../U2/INVESTIGATION_dpcc-c_Gen3v7.md`](../U2/INVESTIGATION_dpcc-c_Gen3v7.md) (**U2-INV** — the DPCC `-c` defects),
[`../init/PORT_GUIDE_hardflow_into_Gen3v7.md`](../init/PORT_GUIDE_hardflow_into_Gen3v7.md) (the port spec this run validates).

---

> ## ⚠️ SUPERSEDED IN PART — the `-t` rows need re-eval (added 2026-07-31, fix_4)
>
> This sweep ran at `cb859e3`, which carries the `prev_observations` double-permutation bug fixed in
> [`../fix_4/CHANGELOG_Gen3v7_fix_4_temporal_consistency_reference.md`](../fix_4/CHANGELOG_Gen3v7_fix_4_temporal_consistency_reference.md)
> (mirror of Gen3v6 fix_5, `ecbae16f`). At `HFFM_BATCH=4` about **75% of replans stored a plan the agent
> never executed** as the temporal-consistency reference.
>
> **Affected: `dpcc-t` and `dpcc-t-tightened` only** — rows in §3, §3.1, §4, §7, and both DPCC subtotals.
> Arm C is unaffected (`HardFlowPolicy._select` never reorders `observations`), as are `-r*`, `-c*`, and
> `diffuser`.
>
> **Do not cite the §1/§3 headline "HardFlow edges ahead, 29.0 vs 28.5" until re-run.** The bug handicaps
> *only* the losing arm, by up to 1.5 points against a 0.5-point gap — a bias with a known sign, not noise.
> §2's isolation control, the whole `-c` analysis (§4/§5/§5(b′)/§6), §7's real-time finding, and §8 are
> unaffected and stand as written. Full accounting of what breaks if this is not re-run: fix_4 §5.1.
>
> Re-run cost: ~3 h 25 m for all five K.

---

## 1. Verdict

1. **The port is correct and provably isolated.** Across the **84** DPCC/`diffuser` cells shared with the
   pre-port U2 run on the same checkpoint, **zero** behavioural metrics changed. Only wall-clock timings
   moved (≤ 0.014 s). The HardFlow arm was added without touching the shared path. §2.
2. **The h=0 query works.** `hardflow_new-t-tightened` and `hardflow_new-r-tightened` both score
   **3.0 / 3.0 / 2.5 / 3.0 / 3.0** goal-and-constraints across K = 1/2/5/10/20 — matching or beating their
   DPCC partners, with **zero** constraint violations at every K. Querying α-Flow's interval-average head
   at `h=0` as if it were an instantaneous velocity field is empirically sound on this checkpoint. §3.
3. **No K-degradation.** The pre-fix_4 pathology that made Gen3v6's arm C collapse as K grew does not
   appear here — as expected, since Gen3v7 shipped with `init_noise_scale=1.0` from day one and gate H3
   asserts it numerically. §3, §5.
4. **v6-RES §5 replicated exactly, on a different architecture.** `hardflow_new-c` freezes **71.8–78.8%**
   of control steps at K ≥ 5 (Gen3v6: 71.4–79.2%) while `dpcc-c` sits at 1.3–3.6%, with the same
   zero-NLP-failure signature. The `cand_prox` ranking defect is **architecture-independent** — it
   reproduces across MeanFlow/MFDiT and α-Flow/SiT. This was the prediction in v6-RES §9.4 and it is now
   confirmed. **New**: the frozen-action count is *bit-identical* between `-c` and `-c-tightened` at every K,
   proving constraint tightening cannot repair this arm. §5(b′).
5. **The K=2 `-c` collapse is total on this checkpoint.** `dpcc-c`, `dpcc-c-tightened`, `hardflow_new-c`
   and `hardflow_new-c-tightened` all read **99.0% frozen, 0.0 success, 0.0 g&c** — all four engines/variants
   agree to within 0.0 pp. §6.
6. **New operational result: on α-Flow, DPCC is real-time at K ≤ 2 and HardFlow never is.** `dpcc-r/t-tightened`
   overrun the 33.3 ms budget on 0.0–0.8% of steps at K ∈ {1, 2}; every HardFlow arm overruns on **100%** of
   steps at **every** K. §7.

**Headline for the paper**: on α-Flow, in-loop constrained sampling (`hardflow_new-t-tightened`) **ties**
post-hoc projection (`dpcc-t-tightened`) on goal-and-constraints at every K, and edges ahead on the
tightened r+t subtotal (29.0 vs 28.5 out of 30 pooled over K) — but at 4.6–5.7× the per-step cost at low K
and with no real-time operating point at all. The advantage is not in the score; it is that the score does
not depend on tightening the constraint set by hand.

---

## 2. The control — the port did not leak

`temp/2807/` holds the U2 DPCC-only evaluation of the *same* `bbsit` checkpoint at K ∈ {1, 2, 5, 10}, run
before `hardflow_projection.py` existed in `flow_matcher_v3_alphaflow/`. Comparing every shared cell
(4 K × 3 halfspaces × 7 DPCC/`diffuser` variants) on `sr`, `cs`, `gc`, `steps`, `nviol`, `tviol`:

```
DPCC + diffuser cells compared:                   84
behavioural-metric mismatches (excluding ctime):   0
ctime cells differing:                            42   (max delta 0.014 s — scheduler noise)
```

This is the arm-B control the PORT_GUIDE asked for, and it came for free with the sweep. The shared code
path is bit-reproducible across the port commit, so **every arm-C number below is attributable to the
HardFlow arm and not to run-to-run variation**. It also confirms the eval driver's rewiring
(`_default_cfg` repoint, `flow_steps` plumbing, the `batch_size` bug fix) changed nothing for arms A/B.

*(Same conclusion v6-RES §2 reached for fix_4, by the same method. Two for two — a DPCC-only control run
remains cheap insurance but is not required when the commit provably touches only the HardFlow files.)*

---

## 3. Goal-and-constraints, all 13 variants

Sum over the three halfspace scenarios; max 3.0. Two trials per cell, so each halfspace contributes
{0, 0.5, 1.0} and differences below 0.5 per halfspace are not interpretable.

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` (no projection) | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 |
| `dpcc-r` | 1.50 | 1.50 | 1.00 | 0.00 | 0.50 |
| `dpcc-r-tightened` | **3.00** | **3.00** | **3.00** | **3.00** | **3.00** |
| `dpcc-c` | 2.50 | **0.00** | 2.00 | 3.00 | 2.00 |
| `dpcc-c-tightened` | 3.00 | **0.00** | 3.00 | 3.00 | 2.50 |
| `dpcc-t` ⚠️ | 3.00 | 1.00 | 2.50 | 2.50 | 2.00 |
| `dpcc-t-tightened` ⚠️ | 3.00 | 2.50 | 3.00 | 2.50 | 2.50 |
| `hardflow_new-r` | 1.00 | 1.50 | 1.00 | 1.50 | 1.50 |
| `hardflow_new-r-tightened` | **3.00** | **3.00** | 2.50 | **3.00** | **3.00** |
| `hardflow_new-c` | 2.50 | **0.00** | 1.00 | 1.00 | 1.00 |
| `hardflow_new-c-tightened` | 3.00 | **0.00** | 1.50 | 1.50 | 1.50 |
| `hardflow_new-t` | 1.50 | 2.00 | 1.00 | 1.50 | 2.50 |
| `hardflow_new-t-tightened` | **3.00** | **3.00** | 2.50 | **3.00** | **3.00** |

Arm subtotals:

| subtotal | K=1 | K=2 | K=5 | K=10 | K=20 | Σ |
|---|---|---|---|---|---|---|
| DPCC, all 6 (max 18) ⚠️ | 16.0 | 8.0 | 14.5 | 14.0 | 12.5 | 65.0 |
| HardFlow, all 6 (max 18) | 14.0 | 9.5 | 9.5 | 11.5 | 12.5 | 57.0 |
| DPCC, tightened **r + t** (max 6) ⚠️ | 6.0 | 5.5 | 6.0 | 5.5 | 5.5 | **28.5** |
| HardFlow, tightened **r + t** (max 6) | 6.0 | **6.0** | 5.0 | **6.0** | **6.0** | **29.0** |

⚠️ = contains `dpcc-t*`, affected by the fix_4 bug (see the banner above). The DPCC rows are
**understated by up to 1.5**; the HardFlow rows are correct. The 0.5-point DPCC-vs-HardFlow gap below
is therefore not defensible until re-run.

Read the two rows that matter (the bottom pair). Once `-c` — a known-broken selection rule on *both*
engines — is excluded, the two arms are statistically indistinguishable, with HardFlow nominally ahead by
0.5 out of 30. The 8-point gap in the all-6 rows is **entirely** the `-c` variants: HardFlow's `-c` pair
contributes 2.5+3.0 at K=1 and then 1.0–1.5 each at K ≥ 5, against DPCC's 2.0–3.0.

Constraint quality for the tightened arms is perfect on both engines: `dpcc-{r,c,t}-tightened` and
`hardflow_new-{r,c,t}-tightened` all record **0.00 constraint violations and 0.000 total violation at every
K**, against `diffuser` at 23.5–55.5 violations / 0.73–5.33 total. Both projection engines do their job; the
differences above are all in *goal* reaching, not in feasibility.

### 3.1 Cross-generation comparison

Same table, `-t-tightened` only (the headline arm), against Gen3v6 post-fix_4:

| arm | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| Gen3v7 `dpcc-t-tightened` ⚠️ | 3.00 | 2.50 | 3.00 | 2.50 | 2.50 |
| Gen3v7 `hardflow_new-t-tightened` | 3.00 | **3.00** | 2.50 | **3.00** | **3.00** |
| Gen3v6 `dpcc-t-tightened` | 3.00 | 2.50 | 3.00 | 3.00 | 3.00 |
| Gen3v6 `hardflow_new-t-tightened` | 2.50 | 3.00 | 3.00 | 3.00 | 3.00 |

⚠️ **Mixed provenance after fix_4.** Both `dpcc-t-tightened` rows are bug-affected today, so they are at
least comparable *to each other*. Once Gen3v6 is re-run at `ecbae16f` and Gen3v7 is not, this table
silently compares fixed against buggy — and would suggest α-Flow's DPCC baseline is worse than
MeanFlow's when the gap may be entirely the fix. Re-run both, or neither.

All four rows live in the 2.5–3.0 band. At n=2 trials per halfspace this is a ceiling effect, not a ranking
— the honest statement is **`-t-tightened` is saturated on this task for both generations and both engines**.
Any claim of a Gen3v7-over-Gen3v6 improvement needs seeds 7–10 (see §9).

The one non-saturated, genuinely different row is `hardflow_new-r-tightened`: Gen3v7 scores 3.0/3.0/2.5/3.0/3.0
where Gen3v6 scores 3.0/2.0/2.5/2.5/2.5. Slot-0 selection (no cost ranking at all) is more reliable on the
α-Flow field than on the MeanFlow field. That is consistent with — but not proof of — H4's premise that
α-Flow *trains* the `h=0` anchor (`af_ratio_fm=0.5`, `af_diffusion.py:694`) rather than merely inheriting the
identity.

---

## 4. Freeze rate

Fraction of emitted control steps that are exactly `(±0.000, ±0.000)`, pooled over 3 halfspaces × 2 trials,
counted with the signed-zero-aware pattern `^ACT\s+\(-?0\.000,-?0\.000\)`.

**Gen3v7 (α-Flow / SiT, step 68000)**

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` | 2.3 % | 6.6 % | 6.5 % | 6.2 % | 7.0 % |
| `dpcc-r-tightened` | 2.7 % | 6.1 % | 2.0 % | 1.9 % | 3.3 % |
| `dpcc-c` | 5.5 % | **99.0 %** | 1.7 % | 1.9 % | 3.6 % |
| `dpcc-c-tightened` | 5.9 % | **99.0 %** | 1.3 % | 1.3 % | 1.6 % |
| `dpcc-t-tightened` | 0.8 % | 2.2 % | 1.8 % | 1.0 % | 1.0 % |
| `hardflow_new-r-tightened` | 2.9 % | 6.1 % | 6.3 % | 6.2 % | 6.2 % |
| `hardflow_new-c` | 5.5 % | **99.0 %** | **71.8 %** | **76.1 %** | **76.1 %** |
| `hardflow_new-c-tightened` | 5.8 % | **99.0 %** | **74.5 %** | **78.8 %** | **78.8 %** |
| `hardflow_new-t-tightened` | 0.8 % | 2.3 % | 2.4 % | 2.3 % | 2.4 % |

**Gen3v6 post-fix_4 (MeanFlow / MFDiT, step 97000)**, same measurement, for reference:

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` | 2.2 % | 6.4 % | 6.2 % | 6.3 % | 6.2 % |
| `dpcc-c` | 1.2 % | 99.5 % | 4.3 % | 4.4 % | 1.3 % |
| `hardflow_new-c` | 1.2 % | 99.8 % | 78.7 % | 79.2 % | 71.4 % |
| `hardflow_new-t-tightened` | 0.6 % | 2.3 % | 2.3 % | 2.3 % | 2.2 % |

Three numbers survive the change of architecture, checkpoint, training objective and backbone almost
unchanged:

- the **unprojected base rate** of frozen candidates (`diffuser`): 6.2–7.0 % vs 6.2–6.4 % for every K ≥ 2;
- the **`-c` amplification** under HardFlow: ~72–79 % on both;
- the **healthy floor** (`-t-tightened`): 2.3–2.4 % on both.

The base rate being a shared property of two independently-trained fields is worth stating plainly: **~6 % of
decoded plans are degenerate "stay put" plans, and that is a property of the D3IL avoiding task's velocity
field as learned by every flow model tried so far, not of any one checkpoint.**

---

## 5. `hardflow_new-c` — v6-RES §5 reproduces verbatim

Every diagnostic v6-RES used to pin the mechanism gives the same answer here.

**(a) It is selection, not generation.** `hardflow_new-r-tightened` freezes at **6.2 %** from the *same* fan
of 4 candidates that `-c` ranks. Same sampler, same NLP, same draws; only the selection rule differs, and it
costs 6 % → 76 %.

**(b) The solver has a perfect record.** NLP failure rate, pooled over 3 halfspaces × 2 trials:

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `hardflow_new-r` | 0.99 % | 2.33 % | 0.00 % | 0.00 % | 0.99 % |
| `hardflow_new-r-tightened` | 0.22 % | 0.00 % | 0.52 % | 0.52 % | 0.52 % |
| `hardflow_new-t-tightened` | 0.00 % | 0.26 % | 0.26 % | 0.26 % | 0.27 % |
| `hardflow_new-c` | **0.00 %** | **0.00 %** | 0.12 % | **0.00 %** | **0.00 %** |
| `hardflow_new-c-tightened` | **0.00 %** | **0.00 %** | **0.00 %** | **0.00 %** | **0.00 %** |

`-c-tightened` records **zero failures in 38 360 solves at K=20** while the arms that work tolerate up to
2.33 % without any loss of task performance. `-c` is not failing to optimise; it has converged onto the
easiest possible problem — a motionless plan is trivially feasible, so IPOPT returns it untouched and
`cand_prox` scores it **exactly 0**, the global minimum of a sum of squares.

**(b′) Tightening changes nothing about *which* candidate is chosen.** The absolute count of frozen emitted
actions is **bit-identical** between `-c` and `-c-tightened` at every K — 27, 1188, 600, 753, 756 for
K = 1/2/5/10/20 — even though the two arms run different constraint sets and therefore different episode
lengths (487 vs 467, 836 vs 805, 989 vs 956, 994 vs 959 total steps; the percentages in §4 differ only
through the denominator). This is a direct confirmation of the mechanism: a motionless plan is feasible
under *both* the nominal and the enlarged constraint set, so the NLP leaves it untouched in both,
`cand_prox = 0` in both, and `argmin` selects the same frozen candidate in both. Constraint tightening —
which repairs `dpcc-c` at K ≥ 5 and repairs every other HardFlow arm — is structurally incapable of
repairing `-c`.

**(c) The accounting is exact.** Per-replan solve and NFE counts, measured as
`Σ solves / Σ emitted ACT steps`:

| K | active steps `k ≥ (1−0.5)·K` | solves/replan (predicted `4 × n_active`) | measured | NFE/replan (predicted `4 × (K + n_active)`) | measured |
|---|---|---|---|---|---|
| 1 | {0} | 4 | **4.0** | 8 | **8.0** |
| 2 | {1} | 4 | **4.0** | 12 | **12.0** |
| 5 | {3,4} | 8 | **8.0** | 28 | **28.0** |
| 10 | {5…9} | 20 | **20.0** | 60 | **60.0** |
| 20 | {10…19} | 40 | **40.0** | 120 | **120.0** |

Exact to the unit for all six HardFlow variants at all five K. The activation gate, the candidate fan and
the NFE counter added in the U3 port are all correct.

**(d) The freeze is a leading block, not an absorbing tail.** Per-step traces at `both-hard`
(`F` = frozen action, first 110 steps):

```
K=5   hardflow_new-c            trial0  n=73   FF.......................................................................
K=5   hardflow_new-c            trial1  n=200  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF.FFFFFFFFFFFFFFFFFF…
K=5   dpcc-c                    trial0  n=91   ...........................................................................
K=5   hardflow_new-r-tightened  trial0  n=63   FFFFFFFF.......................................................
K=20  hardflow_new-c            trial0  n=200  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF…
K=20  hardflow_new-c            trial1  n=119  FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF.......................
```

The stall is at the start pose; escape happens or it doesn't. This matches the state-localisation result in
U2-INV §3.3 (17 % collapse within 0.01 of the start pose, **0.00 % in 15 964 candidates beyond it**) — a
frozen action leaves the observation unchanged, so the next replan re-samples from the same high-probability
basin. Note the `-r-tightened` trace also opens with an 8-step F block and then leaves cleanly: the basin is
real for every arm; only `-c` cannot climb out of it.

The cost shows up as episode length. Mean `Avg number of steps`:

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `hardflow_new-t-tightened` | 64.8 | 63.3 | 63.5 | 63.5 | 61.8 |
| `hardflow_new-c` | 80.2 | — | 77.7 | **128.7** | **130.3** |
| `hardflow_new-c-tightened` | 76.8 | — | 67.3 | **117.7** | **118.7** |

**Bearing on the fix.** Gen3v7's `hardflow_projection.py` is a verbatim port and carries the *identical*
unweighted `cand_prox` accumulation at `:515-516`. v6-RES §5.7 proposed Fix A (multiply by `tau_next**2`,
matching the NLP's own τ² objective weight and the τ-weighted pull-back at `:518`) and Fix B
(`candidate_cost: control`, YAML-only). Neither has been run. **Whichever lands must be mirrored into both
generations in the same commit** — this is precisely the sibling-sync failure class that produced the fix_4
σ bug.

---

## 6. K=2: four variants, one number

| K=2 | frozen % | success | g&c | avg steps |
|---|---|---|---|---|
| `dpcc-c` | 99.0 % | 0.00 | 0.00 | 0.00 |
| `dpcc-c-tightened` | 99.0 % | 0.00 | 0.00 | 0.00 |
| `hardflow_new-c` | 99.0 % | 0.00 | 0.00 | 0.00 |
| `hardflow_new-c-tightened` | 99.0 % | 0.00 | 0.00 | 0.00 |

All four agree to the decimal, across two structurally different projection engines. Combined with §2's
isolation control, this closes the question v6-RES §6 opened: **the K=2 "stay put" mode is a property of the
generative field at the `(r, t)` coordinates K=2 visits, not of either projection engine.** Neither DPCC nor
HardFlow creates it; both select it when told to minimise projection cost.

Why K=2 specifically: at `dt = 1/K` the sampler queries `(r=0, t=0.5)` and `(r=0.5, t=1.0)`, i.e. `h = 0.5`,
which lands in validation h-bucket **b2** — and b2 is the *worst* bucket for this checkpoint
(U2-INV §3.4: last-10 val `h_mse` means b0=2.78, b1=22.0, **b2=116**, b3=73). The bucket that K=2 lives in is
not the largest-h bucket, which is why the defect is non-monotone in K.

Note the `-t-tightened` arms are fine at K=2 (2.2–2.3 % frozen, 2.5–3.0 g&c). **The K=2 field defect is only
fatal in combination with min-projection-cost selection.**

---

## 7. Compute and real-time

Mean `Average computation time per step` over the 3 halfspaces (seconds):

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` | 0.006 | 0.012 | 0.030 | 0.058 | 0.115 |
| `dpcc-t-tightened` | 0.015 | 0.021 | 0.188 | 0.364 | 0.968 |
| `hardflow_new-t-tightened` | 0.085 | 0.097 | 0.202 | 0.471 | 0.950 |
| **HF / DPCC ratio** | **5.67×** | **4.62×** | **1.07×** | **1.29×** | **0.98×** |

Same shape as Gen3v6 (5.27× / 4.13× / 1.03× / 1.44× / 1.14×): HardFlow's overhead is a fixed per-replan cost
that dominates when the ODE is cheap and amortises away as K grows. At K=20 HardFlow is marginally *cheaper*
than DPCC here.

The α-Flow SiT backbone is materially faster than Gen3v6's MFDiT — `diffuser` 0.115 s vs 0.164 s at K=20,
`dpcc-t-tightened` 0.015 s vs 0.024 s at K=1 — and that changes the operating picture. Fraction of steps
exceeding the 33.3 ms real-time budget:

| variant | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| `diffuser` | 0.3 % | 0.3 % | 1.4 % | 100 % | 100 % |
| `dpcc-r-tightened` | **0.0 %** | **0.0 %** | 100 % | 100 % | 100 % |
| `dpcc-t-tightened` | **0.3 %** | **0.8 %** | 100 % | 100 % | 100 % |
| `hardflow_new-r-tightened` | 100 % | 100 % | 100 % | 100 % | 100 % |
| `hardflow_new-t-tightened` | 100 % | 100 % | 100 % | 100 % | 100 % |

**Gen3v7 has a real-time DPCC configuration and no real-time HardFlow configuration.** `dpcc-r-tightened`
at K=2 is 3.0/3.0 g&c, zero violations, and 0.0 % budget overruns — that is a genuine deployable operating
point, and it is new relative to Gen3v6 (whose `dpcc-t-tightened` overran on 7.0 % of steps even at K=1).
The HardFlow arm's floor is ~85 ms/step because it always solves the terminal NLP, so it is 2.6× over budget
even at K=1.

This should be reported alongside the g&c tie in §3: **HardFlow matches DPCC's quality and forfeits the only
real-time operating point on the board.**

---

## 8. Reconciliation with U2-INV

U2-INV identified two `dpcc-c` defects. This sweep, being `bbsit`-only, speaks to both:

- **Defect B (K=2 start-pose freeze)** — fully confirmed, §6. U2-INV called it `bbsit`-only; nothing here
  contradicts that, and the 99.0 % emitted-action freeze matches its "197–199 of 200 actions are a literal
  zero".
- **Defect A (boundary hugging on plain `dpcc-c`)** — **it is a `bbdit` phenomenon, not a `bbsit` one.**
  On this checkpoint, plain `dpcc-c` records `nviol` sums of 1.0 / 0.0 / 1.0 / 0.0 / 3.0 across
  K = 1/2/5/10/20, making it the *least*-violating untightened arm — better than `dpcc-r` (19.0 / 23.5 / 9.0
  / 12.0 / 35.5) and `dpcc-t` (0.0 / 7.5 / 2.5 / 2.5 / 11.0). U2-INV's headline "64 violations for plain `-c`"
  was pooled over **both** backbones × 24 cells; ~60 of those 64 are `bbdit`'s.

  This does not falsify U2-INV's mechanism — the `-c` cost really is identically zero on the feasible
  interior (`projection.py:145`), so argmin really is indifferent to clearance on both backbones. It bounds
  the *consequence*: the indifference only becomes a violation when the field's plans already ride close to
  the boundary, which `bbdit`'s do and `bbsit`'s do not. U2-INV §2 should be read with that scope.

---

## 9. Caveats

- **Seed 6 only, `n_trials: 2`.** Every cell is 2 trials, so per-halfspace g&c is quantised to {0, 0.5, 1.0}
  and the 3-halfspace sums move in 0.5 steps. **Differences of 0.5 in §3 are noise.** The 2.5 entries at K=5
  for both tightened HardFlow arms are a single failed trial in `top-right-hard` and should not be read as a
  K=5 dip.
- **One backbone.** `bbdit` was not run in this sweep. Given §8, the `bbdit` arm-C behaviour is unknown and
  could differ — Defect A's presence there is a live reason to expect it might.
- **`hardflow_new-c` at K=10 and K=20 look identical in §4** (76.1 % / 76.1 %). The raw counts are 753/989
  and 756/994 — close but distinct, and the underlying `steps` and `ctime` differ (128.7 vs 130.3;
  0.476 vs 0.931 s). Not a plumbing artefact; the K plumbing is independently verified exact by §5(c).
- **`candidate_cost: control` remains untested** in both generations.
- The cross-generation comparisons in §3.1 and §4 are across different checkpoints, training objectives and
  step counts (68000 vs 97000). They establish *replication of qualitative signatures*, not a ranking.

---

## 10. Recommended next steps

1. **Fix the `-c` ranking key, in both generations at once.** v6-RES §5.7 Fix B (`candidate_cost: control`,
   YAML-only, zero code risk) then Fix A (the τ² weight at `hardflow_projection.py:515-516`) if B is
   insufficient. Run at K ∈ {5, 20}. Prediction to check: Fix A pulls `-c`'s freeze rate from ~76 % toward
   the ~22 % that i.i.d. selection from a 6 %-contaminated fan of 4 predicts; Fix B pulls it toward the `-r`
   band (~6 %). Mirror into `flow_matcher_v3_meanflow/` and `flow_matcher_v3_alphaflow/` in the same commit.
2. **Seeds 7–10** at K ∈ {2, 5, 20} for the `-t-tightened` and `-r-tightened` headline, to turn the §3 tie
   into a number with an error bar. This is the single highest-value run outstanding — §3's central claim is
   currently resting on 6 trials per cell.
3. **Run `bbdit` through the same sweep.** It is the other Gen3v7 backbone, it carries Defect A, and §8 makes
   its arm-C behaviour genuinely unpredictable.
4. **Add the gate call to `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh`.** The AlphaFlow script runs
   its gates and they passed 4/4 here; the MeanFlow script still does not, so Gen3v6's H3 assertion has never
   executed (v6-RES §8). One line, `set -e` already active.
5. **Do not report `hardflow_new-c` / `-c-tightened` anywhere.** They are a known selector pathology on both
   engines. Report `-t-tightened` (best), with `-r-tightened` as the no-selection control.
