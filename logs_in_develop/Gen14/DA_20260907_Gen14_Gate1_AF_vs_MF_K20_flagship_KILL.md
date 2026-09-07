# DA 2026-09-07 — Gen14 Gate 1: α-Flow vs MeanFlow at the K=20 flagship, Visual Aligning

**Verdict: ⛔ KILL. α-Flow does not beat MeanFlow on `aligning-d3il-visual`, and the arm Gate 0
favoured (`α_end = 0.05`) is *significantly worse* — paired p = 0.0215, winning 1 context out of 10.
The pre-registered kill threshold is met by both arms. The attack plan is closed.**

This is the read-out of the two jobs specified in
`PLAN_20260904_Gen14_AF_attack_plan_beat_MF_on_aligning.md` §3. No further AF submits on Visual
Aligning are warranted under the frozen-parameter constraint.

---

## 0. Conventions

- **Stage-1 metric** (the funnel's lead metric): `context_final_xy_dist / context_init_xy_dist`,
  reported as **× start**. Lower is better; `1.000×` means the box never moved.
- **Arm**: `diffuser` — the *unguided* sampler output, no projector. Stage 1 is deliberately read
  here so the generative field is measured on its own, before any constraint machinery can mask it.
- All rows: `split = train`, `geo = combined_5`, `seed = 6`, `n = 10` contexts, paired by
  fingerprint `(box_init_xy, target_xy, box_angle)`.
- Significance: exact two-sided **sign test**, pure stdlib, on the 10 matched pairs.
- "Good" is Pareto-dominance only. Nothing below is called "best".

---

## 1. Provenance and validity gates — all green

Both jobs ran to completion and both loaded a checkpoint on which α is genuinely alive.

| gate | job 25416 (`α_end=0.05`) | job 25417 (`α_end=0.2`) |
|---|---|---|
| exit | `Job completed successfully.` | `Job completed successfully.` |
| git rev | `8648c41` | `8648c41` |
| checkpoint | `state_100000.pt` (trained to step 100000) | `state_100000.pt` |
| α at that step | **`0.0500` — ACTIVE** | **`0.2000` — ACTIVE** |
| NFE | `flow_steps_v3 = 20` | `flow_steps_v3 = 20` |
| projection threshold | `T = 0.2` (4 projector calls/replan) | `T = 0.2` |
| backbone / FiLM | `unet` / `v1` (from train-time `model_config.pkl`) | `unet` / `v1` |
| selector | `latest` → `_EPlatest` | `latest` → `_EPlatest` |

```
[ eval loading ] checkpoint = state_100000.pt  (trained to step 100000)
[ eval loading ] alpha(step 100000) = 0.0500  [schedule sigmoid 1.0 -> 0.05 over 100000 steps, clamp 0.005]
[ eval loading ]   alpha-Flow objective ACTIVE at this checkpoint.
```

**This is not another α-never-on result.** The U12 mechanism did its job; the objective under test
really is α-Flow. The negative below is a property of the method, not of the plumbing.

---

## 2. Stage 1 — the board

`diffuser` arm, K = 20, T = 0.2 unless stated. Ratio = final/init distance, **× start**.

| arm | K | mean × | median × | 0-viol | steps | ms | abort |
|---|---|---|---|---|---|---|---|
| **`mf` — THE TARGET** | 20 | **0.210** | **0.165** | 0.100 | 400.0 | 190.5 | **0.000** |
| `af` α_end = 0.2 | 20 | 0.709 | 0.890 | 0.100 | 321.8 | 189.1 | 0.200 |
| `af` @latest (no msg tag) | 20 | 0.717 | 0.931 | 0.100 | — | 191.3 | 0.200 |
| `fm` | 20 | 0.740 | 1.000 | 0.100 | — | 295.8 | — |
| `diffusion` (DPCC target) | 20 | 0.853 | 0.957 | 0.300 | — | 298.3 | — |
| **`af` α_end = 0.05** | 20 | **0.969** | **0.969** | 0.400 | 372.9 | 187.3 | 0.200 |

**Compute is matched.** 187–191 ms for every K=20 flow arm — α-Flow is not buying its result with a
time budget, and it is not paying one either. There is no Pareto trade-off to argue: MeanFlow is
better on the lead metric at equal cost.

Two secondary negatives for α-Flow, both visible above:

- **Divergence.** 2 of 10 rollouts abort on *both* α-Flow arms. MeanFlow aborts 0 of 10.
- **`α_end=0.05`'s 0-viol advantage is the old artefact.** 0.400 vs MeanFlow's 0.100 looks like a
  win until you read the distance column: at 0.969× the box is essentially untouched. It satisfies
  constraints by not acting. This is the same failure the U12 DA identified, unchanged.

---

## 3. Paired tests against `mf`, same 10 contexts

| arm vs `mf` K20 | mean × (arm / mf) | contexts won | **p (distance)** | 0-viol arm/mf | p (0-viol) |
|---|---|---|---|---|---|
| **`af` α_end = 0.05** | 0.969 / 0.210 | **1 / 10** | **0.0215 — significantly WORSE** | 0.400 / 0.100 | 0.375 — tie |
| `af` α_end = 0.2 | 0.709 / 0.210 | 3 / 10 | 0.3438 — tie by sign test | 0.100 / 0.100 | 1.000 — tie |
| `af` @latest (no msg) | 0.717 / 0.210 | 3 / 10 | 0.3438 — tie | 0.100 / 0.100 | 1.000 — tie |
| `fm` | 0.740 / 0.210 | 1 / 10 | **0.0215 — significantly WORSE** | 0.100 / 0.100 | 1.000 — tie |

`α_end=0.2` and the untagged arm escape significance only because n = 10 and the sign test throws
away magnitude. **The effect size is 3.4×.** Calling that a tie would be a misreading: it is a large
deficit that this sample size cannot certify, not an absence of one. Nothing here is a win, and
nothing here is close enough that more seeds would plausibly flip it.

### 3.1 Against the pre-registered thresholds

The plan fixed these before the run:

| outcome | condition | met? |
|---|---|---|
| 🏆 WIN | ratio ≤ 0.267 mean **and** 0-viol ≥ 0.150 | ❌ neither arm |
| 🟡 BUFFER | ratio tied **and** 0-viol > 0.150 | ❌ `α_end=0.05` has the 0-viol but not the tie |
| ⛔ **KILL** | ratio worse, **or** > 0.517× | ✅ **0.969× and 0.709× — both arms, decisively** |

---

## 4. Mechanism — why K=20 was the wrong bet

Gate 0 measured everything at K = 2, where the two engines look comparable. The flagship is K = 20.
Tracking the same checkpoints across that gap explains the whole result:

| arm | K = 2 mean × | K = 20 mean × | response to 10× NFE |
|---|---|---|---|
| `mf` | 0.976 | **0.210** | **4.6× better** |
| `af` α_end = 0.05 | 0.744 | 0.969 | **1.3× worse** |
| `af` α_end = 0.2 | 0.787 | 0.709 | 1.1× better — effectively flat |

**MeanFlow's velocity field sharpens with integration steps; α-Flow's does not.** At K = 2 α-Flow is
ahead of MeanFlow on the mean. At K = 20 MeanFlow has moved 4.6× and α-Flow has stood still, so the
ordering inverts and the gap is large.

This is exactly the counter-signal recorded in the plan (§2.3) before the run — af K2→K100
0.370 → 0.689 while mf went 0.517 → 0.277. It was flagged as *the strongest argument the run would
fail*, and it is what happened. The bootstrapped target `u_tgt = (dt·v + (h−dt)·u_next)/h` with
`dt = α·h` trains the network to be accurate over *finite* jumps; that is precisely the regime a
low-K sampler exercises and a high-K sampler does not. The flagship's K = 20 is the worst place to
meet MeanFlow, and it is also the only place a matched-setup Pareto claim was available.

**The two constraints — "match MF exactly" and "beat MF" — are in direct tension for this method.**
Matching MF's setup means adopting K = 20, and K = 20 is where α-Flow's advantage does not exist.

---

## 5. Stage 2 — not run, and it would not have helped

Per funnel discipline an arm leaves the moment it fails a stage, so Stage 2 is not adjudicated.
For the record, the data that exists says it is undecidable anyway: **`n_success_and_constraints`
is 0.000 for all three arms across all 19 projection variants**, except a single rollout
(0.05 = 1/20) for `α_end=0.05` on `dpcc-r`, `gradient`, and `model_free-bounds_free`. Nothing can
be ranked on S&C in Visual Aligning at this seed.

**Honest setup caveat:** the two Gate-1 jobs ran with `arm C (HardFlow) OFF` — the log line is
`[ eval ] arm C (HardFlow) OFF — set HFFM_VARIANTS to enable` — so the `hardflow_sls-*` variants are
`NaN` for both α-Flow arms while MeanFlow's flagship has them. That is a genuine mismatch against
the "match MF exactly" requirement **at Stage 2**. It does not touch this verdict: Stage 1 reads the
`diffuser` arm, which runs no projector at all, so HardFlow's presence or absence cannot affect a
single number in §2 or §3. Had Stage 1 passed, this would have needed a re-run with
`HFFM_VARIANTS` set before any projected comparison.

---

## 6. The kill statement

> **α-Flow does not beat MeanFlow on Visual Aligning under the frozen-parameter constraint, and the
> Gen3v7 `avoiding` win does not transfer to this task.**
>
> Tested at MeanFlow's own flagship operating point (K = 20, T = 0.2, `unet`, FiLM v1, 26.4 M
> parameters, seed 6, matched contexts, matched wall-clock), on the unguided generative field, with
> α verified active at the deployed checkpoint. `α_end = 0.05` — the arm the Gate-0 paired analysis
> selected — lost 9 of 10 contexts, p = 0.0215. `α_end = 0.2` — the arm that won on `avoiding` —
> lost 7 of 10 with a 3.4× deficit in mean. Both exceed the pre-registered kill threshold. Two of
> ten rollouts diverge on each α-Flow arm; none diverge on MeanFlow.
>
> The cause is identified and is structural, not a tuning miss: α-Flow's bootstrapped target
> optimises accuracy over finite jumps, so its field is flat in NFE (K2 → K20: 0.744 → 0.969 and
> 0.787 → 0.709) while MeanFlow's sharpens 4.6× over the same range. Matching MeanFlow's setup
> requires K = 20; K = 20 is where α-Flow has no advantage to offer. The requirement to match and
> the requirement to win are not simultaneously satisfiable for this method on this task.

### 6.1 What the thesis can still say

- **`avoiding-d3il` stands.** The Gen3v7 α-Flow-U-Net win at K = 1 is unaffected by this and remains
  the architecture-matched result. This DA sharpens it: the win is now understood as a **low-NFE**
  win, which is a more precise and more defensible claim than a general one.
- **Visual Aligning reports `mf > af`, `mf > fm`** on the generative field at the flagship, with
  `af ≈ fm` (0.709–0.969 vs 0.740). The `af > mf > fm` ladder does **not** hold here and should not
  be claimed.
- The honest cross-task statement is: **α-Flow's benefit is NFE-dependent — real at K = 1–2,
  absent at K = 20.** That is a finding, and it is testable.

### 6.2 What is now closed

`α_end` is exhausted: 0.0 (α dead), 0.05, 0.2, and a constant-0.05 variant have all been run at the
flagship and at K = 2. `af_alpha_clamp` was already negative on `avoiding`
(`DA_20260901_AF_UNet_alpha_clamp_T1_negative.md`). `af_ratio_fm` is ruled out by the root-cause
study §9.4. `action_weight` is cosmetic on both engines (FIX-3 deliberately does not apply it to the
loss). The two mechanism fixes that the root-cause study identified — the `E_τ(r) + E_h(h)`
cancellation and `freq_dim = 32` — were **withdrawn under the frozen-parameter constraint** and are
the only untried levers left. Reopening either means changing the network, which forfeits the
architecture-matched claim that makes the comparison worth making.

**No further AF submits on Visual Aligning.**

---

## 7. Limits of this result

- **n = 10 contexts, one seed (6), `split = train`.** Enough to kill (the losing arm lost 9/10), not
  enough to certify the size of the `α_end=0.2` gap.
- `S&C = 0` across the board means this task cannot rank anything on the thesis's headline
  success-and-constraints axis at this seed; Stage 1 is the only stage with signal.
- HardFlow was off in both α-Flow jobs (§5) — irrelevant to Stage 1, disqualifying for Stage 2.
- The `diffuser` rows for `combined_5` and `combined_5-tightened` are separate eval passes of a
  geometry-independent sampler; only `combined_5` is used above, consistently for every arm.

---

## Provenance

- Jobs **25416** (`afon005_s6`) and **25417** (`afon02_s6`), submitted per plan §3; both
  `Job completed successfully.`, git rev `8648c41`, node i6-gpu-1.
- Logs: `temp/0609/I/2026-09-05/00_12_59_eval_mix_visual_aligning_25416.log`,
  `…/00_13_13_eval_mix_visual_aligning_25417.log`.
- Data: `temp/0609/II/batch_va2_20260907_141036/per_rollout_detail.csv`, candidates 8
  (`…AFAFend0p05/…_EPlatest_msgafon005_s6`), 11 (`…AFAFend0p2/…_EPlatest_msgafon02_s6`),
  24 (`…VisualMeanFlow_VTrue_mpc4_filmv1_Emf`, K20 T0.2), plus 9/12/25 for the K = 2 row.
- Supersedes the recommendation in `DA_20260904_…_U12_alpha_floor_and_latest_checkpoint.md` §6.4
  and closes `PLAN_20260904_Gen14_AF_attack_plan_beat_MF_on_aligning.md`.
