# DATA STATUS — what v3 may write, entry by entry

**Created:** 2026-09-10 · **Type:** data-readiness ledger for the draft · **Scope:** the three major
experimental entries — **`avoiding-d3il`**, **`aligning-d3il-visual` (V_A)**, **`uav`** — as they
stand for the **v2 → v3** pass (v3 = the pass that puts experimental results into Chapters 5–7).

**Governed by:** [`../TARGET_20260905_thesis_claim_ladder.md`](../TARGET_20260905_thesis_claim_ladder.md)
(goals) and [`../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md`](../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md)
(**fired — the active engine plan**).
**Current draft:** [`../v2/thesis_v2.tex`](../v2/thesis_v2.tex) · [`../v2/README.md`](../v2/README.md) · [`../v2/CHANGELOG.md`](../v2/CHANGELOG.md)
**Evidence of record:** [`Data_Analysis/DA_Result_Curated_MD/`](../../../../Data_Analysis/DA_Result_Curated_MD/),
indexed by [`NOTEBOOK_20260829_key_headlines.md`](../../../../Data_Analysis/DA_Result_Curated_MD/NOTEBOOK_20260829_key_headlines.md)
**Naming:** [`../../Auxiliary/NOTES_method_naming.md`](../../Auxiliary/NOTES_method_naming.md) — mechanism names, not code tags (§9.11)

> 🔴 **What this file is.** A *status* file: for each entry, what the data supports **today**, at what
> strength, and what blocks the rest. Every number here is a **pointer into a DA of record** (§10),
> never new analysis. Never cite this file for a number — cite what it points at.
>
> 🔴 **What this file is not.** Not a target (that is `TARGET_…`), not a plan (that is
> `FALLBACK_…`), not a DA. It records *readiness*, not findings.

---

## 0. The answer, in one table

Organised by the **three topics** of [`TARGET §0.5`](../TARGET_20260905_thesis_claim_ladder.md), not
by environment — that is the 2026-09-10 framing.

| topic | leg | engine claim | projector claim | methodology | **verdict for v3** |
|---|---|---|---|---|---|
| **1 Avoiding** — the foundation | 2-D planar state | 🟢 ready | 🟡 side topic, underpowered | 🟢 inherited | ✅ **WRITE NOW — the paper's foundation** (one cheap run owed: Tier 1) |
| **2 Aligning** — harder task | **2a · 3-D state** | ⬜ **no data, no config** | ⬜ | ⬜ | 🔴 **BUILD + TRAIN + EVAL** — the largest structural gap |
| | **2b · 3-D visual** | 🟢 ready | 🟢 ready — **select K = 10** | 🟡 one section owed | ✅ **WRITE NOW**, one table held |
| **3 UAV** — other embodiment | underactuated, unstable | 🔴 one seed, one scene | 🟡 strong but one seed | 🟢 ready, **and it is the strength** | ⚠️ **WRITE THE CHAPTER, NOT THE RANKING**; headline scene not finished |

> ### Two legs are ready to carry claims. One is ready to carry a chapter. One does not exist.
> **`avoiding` (1) and `aligning-visual` (2b)** can be written today, at the fallback ladder.
> **UAV (3)** carries the embodiment argument, the regime taxonomy and the projector safety/cost
> result — but its engine ranking is one seed on one scene, and the scene the author wants to lead
> with (`corridor`) is **constraint-trivial** until `corridor_ball` binds.
> **The 3-D state aligning leg (2a) has no config and no checkpoints** — without it, *harder control*
> and *harder perception* were added in one step, and the state→visual transfer cannot be claimed.

**"Which only reach the fallback?"** — **all of them, on Goal A.** The four-rung TARGET ladder is
dead in every environment measured; the shipping claim is `mf ≫ {af, fm, diffusion}`. The one
fragment of the original target still standing is `fm ≻ diffusion` on **`avoiding` only**, as a
**cost** claim — on V_A the same pairing is a **tie**, and on UAV it is untestable. Full ledger in §5.

**"`af` at least no worse?"** — true only where the task has no power to separate (low-K `avoiding`,
`corridor`, V_A K=2). Where it does separate, `af` is worse (V_A K=20) or **last, behind naive `fm`**
(`uav-pillars`). The containment argument that predicts otherwise is answered in
[`FALLBACK_… §10.2`](../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md).

---

## 1. How to read the grades

| grade | meaning | what v3 may do with it |
|---|---|---|
| 🟢 **ready** | paired/exact test or an exact structural argument; caveats known and statable | write it as a claim, with its caveat sentence |
| 🟡 **partial** | direction is clear, power or coverage is not | write it as *directional*, never as a headline; or hold the table |
| 🔴 **blocked** | one seed / one scene / a coverage hole that changes what may be said | write the mechanism and the measurement, **not** the ranking |
| ⛔ **closed** | tested and refuted; belongs in `sec:disc:negative` | write it as a negative result with its mechanism |
| ⬜ **no data** | never measured | do not write; remove from the matrix or declare it out of scope |

**"Ready" always means ready *at a stated strength*.** Nothing in this project is multi-seed except
`avoiding`; single-seed is the norm and must be said out loud rather than buried in a caption.

🏷️ **Vocabulary.** This file deliberately keeps the **artefact** names (`hardflow_sls-*`, `dpcc-c`,
`mf`/`af`/`fm`) so a row can be traced to a CSV and a DA. The **thesis** names are different and are
decided in [`NOTES_method_naming.md`](../../Auxiliary/NOTES_method_naming.md) — translate at the
thesis boundary, never in the evidence (§9.11).

---

## 2. Entry A — topic 1, **`avoiding-d3il`** (2-D planar state, manipulator) · ✅ **the foundation of the paper**

**Why it matters:** DPCC's own benchmark, the pinned Target's home turf, and the **only entry with
multi-seed data**. **Author's decision, 2026-09-10: this entry carries the paper.** Everything else
is transfer (modality, embodiment) or a side topic.

**Pinned Target** (never renegotiated): diffusion DPCC `K=20`, `aw=10`, `T=0.5`, `GaussianDiffusion`,
at its best projection variant `dpcc-c-tightened`.

---

### 2.0 How `avoiding` is presented — two protocol tiers

| tier | protocol | episodes/cell | role in the paper |
|---|---|---|---|
| **Tier 1 — theirs** | **5 seeds × 2 trials** (DPCC's published evaluation) | 10 | *"we evaluated the baseline the way its authors did"* — answered before it is asked |
| **Tier 2 — ours** | **5 seeds × 20 trials** | 100 | every claim in the paper rests here |

🔑 **The tier shift is itself a result, not a formality.** We have already measured what it does
([`DA_20260819_n20_vs_n2`](../../../../Data_Analysis/DA_Result_Curated_MD/DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md)):

- **the 1.00 ceiling is an artefact of 10 episodes** — `dpcc-c-tightened` @ top-right reads **1.00 at
  n=2 and 0.95 at n=20**; `dpcc-t-tightened` 1.00 → 0.92. *Any n=2 "1.00" means "≥ 0.90 at ±0.10
  resolution".*
- **untightened mid-range cells move by up to 0.35, in both directions** (`dpcc-t` @ top-left
  0.80 → 0.54). The sampling-lucky ones got worse; the weak ones got slightly better.
- **tightened variants are stable to ~0.08**, and the bottom of the table (unguided, `gradient*`,
  `model_free*`) does not move at all — 0.00–0.15 at both counts, and unrankable within itself.

⇒ Tier 1 does not just show politeness to the baseline; it shows that **the published protocol's
resolution (±0.10) is coarser than most of the differences the field reports on this benchmark.**
That belongs in `sec:setup:protocol` as a measured methodological point.

#### 🔴 Data status of the two tiers — Tier 1 needs a run

| | status |
|---|---|
| **Tier 2** | 🟢 **complete** — `batch_avoiding_combined_20260825_143212`, 5 seeds (6–10) × 20 trials, all engines |
| **Tier 1, the baseline** | 🟡 exists as the n=2 reference cells (job 24639 / `batch_avoiding_combined_20260818_152911`) |
| **Tier 1, the flow arms** | 🟡 exists only in **pre-2026-08-19** batches — older eval binary, mixed configs |
| **Tier 1 by subsampling Tier 2** | 🔴 **not possible.** The avoiding corpus stores per-`(seed, variant, geometry, metric)` aggregates with `_std`; **the trial dimension is already collapsed** and there is no per-episode row (unlike UAV's `per_rollout_detail.csv`) |
| ⚠️ known defect in the old n=2 data | the n=2 `post_processing` rows are **corrupt** — numerically identical to `dpcc-r` on 12–16 of 14–18 metrics (the branch resolved to the `dpcc-r` projector). Never quote them |

> **⇒ A clean Tier 1 needs a fresh eval at `n_trials = 2`, all engines, one binary.** It is the
> cheapest run on the whole list — 10 episodes/cell against Tier 2's 100, i.e. ≈ 1/10 the wall clock
> of a tier we have already paid for. See §8 item 3.

---

### 2.1 The headline, as it will be written

> ### At K = 1–2, {MeanFlow, bootstrapped-target} ≻ naive flow matching at equal S&C; naive flow matching matches or beats the diffusion baseline **on cost**; and **MeanFlow Pareto-dominates the pinned Target.**
> The last clause is the load-bearing one. **It is enough on its own**, and it is the sentence the
> rest of the paper is built on.

Three guards this phrasing needs — each is a place where a looser wording would be wrong:

| the phrase | what the data says | the guard |
|---|---|---|
| **"MeanFlow *and* α-Flow"** | on `avoiding` the two are **not separable**: nominal 57.20 vs 60.75 steps, **Fisher p = 0.231**, and a **wash** (8 vs 9 Pareto cells) once averaged over all six projection rules | ✅ write them as a **pair at low K**. 🔴 never as a ladder — *"α-Flow > MeanFlow"* is inside the noise, and V_A + `uav-pillars` both put α-Flow **below** MeanFlow. The pair phrasing is the only one that survives all three environments |
| **"`fm` ≈ or ≻ diffusion"** | **basis-dependent, and both bases are real** — see 2.1a | ✅ say **"matches or beats the baseline's success at 21–27× less compute"**. 🔴 never a bare *"`fm` > `diffusion`"*: on V_A that ordering is **refuted**, so an unqualified claim here reads as a contradiction |
| **"Pareto-dominant"** | quote it from the **300-episode-a-side** comparison, not from the K=1 cell that reads 0.97 vs 1.00 | ✅ see A1 — at 0.993 vs 0.983 the S&C axis is **equal-or-better**, which is what dominance requires |

#### 2.1a The two bases for `fm` vs the baseline — state which one you are on

| basis | protocol | reading |
|---|---|---|
| **powered, 5 seeds × 20** (TL + TR, `dpcc-c-tightened` both sides) | FM K=2: S&C **1.00 / 1.00** vs DPCC **1.00 / 0.95**; **65.5 / 71.6** vs **70.0 / 77.6** steps; **1.8 / 1.9** vs **39.1 / 40.2** s/ep | **Pareto-dominates — 21×** |
| **seed 6, cheapest S&C = 1.00 cell**, `top-left-hard`, K=1, `dpcc-t-tightened` | fm **65.65** steps / 1.31 s-ep vs DPCC **61.00** / 35.66 | DPCC is **fewer steps**; fm is **27× cheaper** ⇒ a *cost* win, not a dominance |

**Both are true and they are different cells.** The paper's `fm` claim is therefore a **cost claim**,
and it must be labelled as one — which is also what keeps it compatible with V_A's refutation of the
*quality* version of the same rung (§5).

---

### 2.2 Claim ledger

| # | claim | grade | evidence of record | what must travel with it |
|---|---|---|---|---|
| **A1** ⭐ | **MeanFlow-U-Net Pareto-dominates the Target** — `mf_unet` K1 `dpcc-t-tightened`: **S&C 0.993 · 61.0 steps · 18.1 ms/step** vs Target **0.983 · 69.0 · 564 ms** ⇒ **better S&C, fewer steps, 31× cheaper**; K2 the same at 21× | 🟢 **ready — the paper's foundation** | [`HF_Batch_Parity/DA_20260827`](../../../HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md) §10.1 — **5 seeds × 20 trials = 300 episodes on both sides**, all three scenarios complete, fan 4 | this is the **strong** form: equal-or-better on *every* axis. Architecture-matched (U-Net 4.0 M vs the U-Net baseline) |
| A2 | **The low-K pair** — at the cheapest S&C = 1.00 cell, `{mf, af}` ≻ `fm` on both cost axes; `af` 57.20 · `mf` 60.75 · `fm` 65.65 steps | 🟢 **ready as a pair** | [`Report_20260903_AF_UNet`](../../../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) §1 | seed 6, n = 20. **`af` vs `mf` is p = 0.231** and a wash over all six rules — never split the pair (§2.1) |
| A3 | **`fm` matches or beats the baseline at 21–27× less compute** | 🟡 **partial** | [`DA_20260819_ntrials20`](../../../../Data_Analysis/DA_Result_Curated_MD/DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md) §1 | complete on TL/TR; **`both-hard` K=20 is 2 seeds + a partial third** (24 h wall) — never rank on it. State the basis (§2.1a) |
| A4 | **K is inference-time for the flow family, training-time for diffusion** — DPCC K=1 → S&C 0.67, so the baseline cannot follow down the K-ladder | 🟢 **ready as mechanism** | H1 | the low-K DPCC cells are `n_trials = 2` — the *mechanism* sentence, not a powered comparison. **Tier 1 (§2.0) repairs exactly this** |
| A5 | **`af_unet` does not improve on `mf`** — nominal only, p = 0.231, a wash over all rules | ⛔ **closed → negative result** | `Report_20260903_AF_UNet` §1, §10 | seed 6; **two α floors were run and the better is quoted** — a selection effect, state it where the number appears |
| A6 | **Goal C, strong form:** at fan `B = 1` the three selection rules are **bit-identical** in all 15 blocks ⇒ running three is exactly 3× wasted projection compute | 🟢 **ready — exact, not statistical** | H5 F3 / H7 §7.1 | needs no further experiment |
| A7 | **Goal C, weak form:** the cumulative-cost rule never earns its cost; temporal-consistency is the right default *here* | 🟡 **directional** | H7 §7.3–7.5 | the default is **task-dependent** — temporal-consistency on `avoiding`, random on V_A (§3) |
| A8 | **The fan is free on the generator and linear on the projector** ⇒ parallelise, don't shrink `B` (`projection.py:132` runs SLSQP serially; `parallelize` is assigned and never read) | 🟢 **ready — a code fact** | H5 F1, H7 §7.4 | an engineering recommendation. **The fan question does not decide the paper claim** — A1 already wins at fan 4 |
| — | `aligning-d3il` **3-D state** | ⬜ **no data at all** | evidence board, gap 4 | see §6 |

---

### 2.3 Side topic — endpoint projection (arm C) on `avoiding`

**Author's framing, 2026-09-10: this is a side topic, not the headline.** It asks a narrower
question — *at which budgets does our endpoint-projection arm beat the DPCC projector?* — and it is
the least powered result in the entry.

🏷️ **Naming:** *"HF-SLSQP"* is a code tag and is retired from thesis prose. The thesis name for arm C
is **in-ODE endpoint projection** (vs arm B's **per-step iterate projection**); the solver is a
backend row in `app:repro`, never part of the name. Rules, options and the code-verified mechanism:
[`NOTES_method_naming.md`](../../Auxiliary/NOTES_method_naming.md).

| # | claim | grade | evidence | what must travel with it |
|---|---|---|---|---|
| A9 | **At K = 3 the endpoint-projection arm Pareto-dominates the Target** — MeanFlow-U-Net, tightened, temporal-consistency: **S&C 1.000 · 58.88 steps · 0.0745 s** ⇒ 16 % fewer steps, **7.4× cheaper**, architecture-matched | 🟡 **underpowered** | H9 · [`DA_20260906_hf_minK`](../../../../Data_Analysis/DA_Result_Curated_MD/Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md), job 25444 | **24 rollouts** (4 seeds × 3 geometries × `n_trials = 2`), **seed 6 absent**, Target pools 5 seeds. The **time** differences survive the small n; the **safety** differences (1.000 vs 0.917 = 2 rollouts) do not |
| A10 | **It does *more* NLP solves and costs *less*** — arm C flat ≈ 28 ms/solve, arm B rises with K; **crossover at K = 3**, which is also the lowest citable budget | 🟢 **ready** | H9 | the mechanism follows from the name: the endpoint is near-feasible at every τ, so SLSQP converges in few iterations. Currently **inferred from timing** — `solve_ms` and iteration counts are not exported for arm B |
| A11 | **The budget floor is a `(K, A)` pair, not a K** — `n_genuine = max(K − int((1−A)·K), 1) − 1`; **K = 1 is structurally impossible at every A** | 🟢 **ready — exact** | [`Proposal_20260905`](../../../../Data_Analysis/DA_Result_Curated_MD/Proposal_20260905_HF_minK_mf_af_unet/README.md) | this is **our** contribution, not the source paper's — say so (naming note §2 fact 4) |

**Known limits that qualify every `avoiding` table.** Suspected projector degeneracy at K ≤ 2 under
`T = 0.5` (s/step jumps ~14× between K=2 and K=5 — the projector may execute fewer than one projected
step at K ≤ 2; it affects both sides equally, so comparisons stand, but it changes what "33× cheaper"
measures) · the AF report's comparators are three weeks older on earlier eval code · no smoothness
metric exists.

## 3. Entry B — the **aligning** topic (harder task) · ⚠️ **one leg ready, one leg missing**

**Why it matters:** this is where difficulty is added to the foundation — and per the 2026-09-10
framing it is **two legs, not one**:

| leg | environment | what it adds | data |
|---|---|---|---|
| **2a** | `aligning-d3il` — **3-D state** | **harder control**: 3-D actions (`action_dim = 3`), box pose + target pose, no halfspace walls | ⬜ **none** — §3.0 |
| **2b** | `aligning-d3il-visual` (V_A) | **harder perception**: pixels instead of state, on the same 3-D control problem | 🟢 one corpus, seed 6 — §3.2 |

> 🔑 **Running only 2b confounds two increments in one step.** Every conclusion drawn from V_A today
> is "harder control **and** harder perception, jointly". The state leg is what separates them, and
> it is the difference between *"the ladder survives a harder task"* and *"the ladder survives a
> different task"*.

---

### 3.0 🔴 Leg 2a — the 3-D state case has no data, and no config

Checked on disk, 2026-09-10:

| | status |
|---|---|
| the environment | 🟢 **exists** — vendored: `d3il/environments/d3il/envs/gym_aligning_env`, `d3il/simulation/aligning_sim.py`, `d3il/configs/aligning_config.yaml` (`obs_dim 20`, `action_dim 3`, `n_contexts 60`) |
| an FM-PCC config | 🔴 **does not exist** — `config/` holds `aligning-d3il-visual.py` only; there is no state sibling |
| trained state checkpoints | 🔴 none for any engine |
| eval corpus | 🔴 none |

⇒ **This is a build + train + eval item, not an eval item.** Scope: one config (the visual one is the
template — the constraint set, the Euler dynamics rows and the projector are shared), then four
engines trained, then one eval sweep. It is the **largest single gap between the current data and the
thesis's own claim structure**, and TARGET §8 now carries the kill criterion for it not landing.

🟢 **One cheap consequence if it does run:** the bootstrapped-target arm rides along at no extra
design cost, giving the α-Flow closure a **fifth** environment
([`FALLBACK_…` §10.3](../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md)).

---

### 3.1 The protocol — V_A has no `n_trials`, and that is the credibility mechanism

🔴 **Correction worth carrying into the text.** V_A is **not** "random init × 10/20 trials". The
eval enumerates a **fixed context set** and resets to each context explicitly
(`eval_mix_visual_aligning.py:1240` — `env.reset(random=False, context=context)`); a context is a
**box initial pose** (`box_init_xy`, `box_init_angle_deg`, `init_xy_dist`). The knobs are
`n_contexts` × `n_trajectories_per_context` (`config/visual_aligning_eval.yaml:44–45`), and
`config/aligning-d3il-visual.py:166` says so in as many words: *"visual-aligning has no `n_trials` —
it rolls out `n_contexts` contexts; more trials here means raising `n_contexts`."*

**This is better than random init, and it is why the statistics work:** every arm sees the *same*
contexts, so the closure can run **exact paired sign tests and paired sign-flip permutation tests**
over 10 matched pairs. Random per-rollout init would make the comparison unpaired and cost most of
that power. Write it as a design decision, not as an apology for one seed.

| | value |
|---|---|
| corpus of record | seed 6, **10 contexts**, 1 trajectory each |
| repo default today | `n_contexts: 3`, `n_trajectories_per_context: 1`, `seeds: [6]` (7–10 commented out) |
| **D3IL's own aligning protocol** | **`n_contexts: 60`** (`d3il/configs/aligning_config.yaml:75`) |

> ### ⭐ The cheap power upgrade the author proposed — and the right knob for it
> **1 seed × 50 contexts**, i.e. `n_contexts: 50` (or 25 × 2), **not** `n_trials`.
> It multiplies the paired sample by **5×** without retraining anything, moves us from 10 pairs to
> 50 pairs — where a sign test can resolve what 10 pairs cannot (at n = 10 the minimum two-sided
> sign-test p is 0.002 and a 7/3 split is p = 0.34) — and lands **close to D3IL's own 60**, so the
> protocol stops being ours and starts being the benchmark's. **This is the highest
> power-per-GPU-hour item in the entry.** See §8.

---

### 3.2 Leg 2b — claim ledger (V_A)

**Regime:** all-fail — S&C is floored, so ranking uses the **three-stage funnel**
(distance → constraints → time), Stage 1 on the **unguided** arm only. Initial distance **0.4530 m**.

| # | claim | grade | evidence of record | what must travel with it |
|---|---|---|---|---|
| B1 | **`mf` ≻ the diffusion-DPCC baseline** — unguided median **0.0741 vs 0.4140 m**; paired **−0.3744 m, 0/10, p = 0.0020**, at **0.64×** the per-step cost | 🟢 **ready, decisive** | [`Gen14 CLOSURE_20260907`](../../../Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md) R.1/§1 · H10 §10.2 | a **plan-quality** ranking on **untightened** geometry — the only surface where all five arms coexist (see B6) |
| B2 | **`mf` ≻ naive `fm`** — paired **+0.216 m, 9/0, p = 0.0039**; `fm` frozen 5/10 vs `mf` 1/10 | 🟢 **ready** | closure R.3 · H10 §10.3 | — |
| B3 | **`fm` ≈ `diffusion`** | 🟢 **ready as a tie** | closure R.3 | **0.0055 m inside a ~0.4 m band**, p = 1.0000. 🔴 The author's *"(or it is > here)"* does **not** hold on V_A — this leg gives a **tie**, and the `fm` ≻ baseline claim lives on `avoiding`, as a **cost** claim (§2.1a). Stating it as a tie here is what keeps the two legs consistent |
| B4 | **The flat-in-K result** — `mf` 0.2354 → **0.0741** (K 2→20) while `af` α=0.2 goes 0.3738 → 0.3918 and α=0.05 0.3978 → 0.4488 | 🟢 **ready — the centrepiece figure** | closure §5.1 / §M.4 · `FALLBACK_…` §2.3 | the mechanism (`(1−α)·ε_next + O(α²h²)` ⇒ K-independent field bias) is derived; write it in `sec:bg:fewstep` |
| B5 | **`af` ≤ `mf` — tie at K=2, worse at K=20, better nowhere** | ⛔ **closed → negative result** | [`DA_20260907 Gate1 KILL`](../../../Gen14/DA_20260907_Gen14_Gate1_AF_vs_MF_K20_flagship_KILL.md) · closure R.4 | 🔴 the author's *"`af` at least same as `mf`"* holds **at K = 2 only**, and even there only in paired distance — on contexts-ever-solved `mf` takes 7/10, `af` 2/10. **Name the α floor**: `p = 0.0215`, 1/10 is the **α = 0.05** arm; α = 0.2 is a 3.4× mean deficit the sign test does **not** reject |
| B6 | **The constraint comparison against the baseline** | 🟡 **incomplete — the one held table** | closure §8.1 | **`diffusion` has no tightened K=20 cell.** Publish the plan-quality ranking and the flagship cell; **hold** the constraint comparison against the Target |
| B7 | **The flagship cell** — `mf` · U-Net 4.0 M · K=20 · T=0.2 · endpoint projection (`-r`), tightened: **zero-violation 1.00, 0.00 violations, 275.3 ms/step**, still moving the box (2/10 untouched); the baseline reaches comparable safety only at **2157.9 ms/step (7.8×)** | 🟢 **ready** | H10 §10.5 | `dpcc-c-dt4p0` also reaches 1.000 — **by freezing 10/10**, which is disqualified. Say so in the same breath |
| B8 | ~~"endpoint projection is safer than DPCC at K=20"~~ | 🔴 **do not quote** | H10 §10.4 | 10/10 vs 9/10 is **one discordant rollout**, p = 1.0000 |
| B9 | Three findings that only exist since this drop: the endpoint arm's benefit **scales with plan constraint error and is free on a high-violation field** (−124.70 violations on `af`, p = 0.0078, zero distance cost) · the **τ = 0.850 NLP failure is engine-independent** (24/24 arm-C items, three objectives) · a measured **~0.4 m reproducibility floor** on projected arms | 🟢 **ready** | closure R.5 / §8 | the 0.4 m floor **qualifies every MIN in every V_A table** |

### 3.3 ⭐ "Show HF > PCC strongly here" — select **one** cell, and it is K = 10

The author's ask: the projector claim should be shown strongly on this topic. It can be — but only
at one operating point, and the corpus is unambiguous about which:

| | K = 10 (T = 0.4) | K = 20 (T = 0.2) |
|---|---|---|
| latency vs DPCC | 🟢 **−324.96 ms/step, 0/9, p = 0.0039** (`-r`); −320.08, 0/9 (`-c`); −211.36, 1/8, p = 0.0391 (`-t`) | ⚪ **parity** — all p = 0.18–1.00 |
| zero-violation vs DPCC | n.s. | 🟢 10/10 vs 9/10 — **one rollout**, p = 1.0000 |
| distance cost | none | none |

> ### The one citable V_A projector claim
> **At K = 10, in-ODE endpoint projection delivers DPCC-level safety and distance at ~325 ms/step
> less — a 0/9 sweep at p = 0.0039.**
> 🔴 Do **not** pair it with a K=20 safety claim (B8). What is separately true at K=20 is that the
> endpoint arms are the **only** arms in the corpus reaching zero-violation 1.000 with 0.00
> violations *while still moving the box* — a property of the configuration, not a margin.

**And the honest limit:** this is a **latency** win at equal safety, on one seed, at one K. It is
strong *because* it is a clean 0/9 sweep, not because it is large.

### 3.4 ⚠️ "The tests may be implausible" — what a retrain would and would not fix

The concern is legitimate and it has a specific shape: **V_A sits in the all-fail regime** — no arm
satisfies the constraint set, S&C is floored, and ranking needs the funnel.

| suspect | is it real? | does a **retrain** fix it? |
|---|---|---|
| the **constraint set is over-strict** — nothing passes | 🟢 real, measured (S&C floored on every arm) | 🔴 **no.** This is a property of the geometry, not the weights. A retrain cannot make an infeasible constraint set feasible |
| ~**0.4 m reproducibility floor** on projected arms | 🟢 real, measured | 🔴 no — it is run-to-run variance; the fix is **repeat a cell 3×** (§8) |
| **τ = 0.850 NLP non-convergence**, 24/24 arm-C items, three engines | 🟢 real, engine-**independent** | 🔴 no — engine-side is already ruled out; suspect is constraint-Jacobian conditioning |
| **10 contexts is thin** | 🟢 real | 🔴 no — the fix is `n_contexts: 50` (§3.1), which needs **no retraining at all** |
| the **checkpoints** themselves (training length, α floors, conditioning) | 🟡 open | 🟢 **yes — this is the only thing a retrain addresses** |

> ### Recommendation, in order
> 1. **`n_contexts: 50`** — 5× the paired sample, no retraining, cheapest thing on the list.
> 2. **Tightened `diffusion` K=20** — unblocks the one held table (B6).
> 3. **Repeat one projected `mf` cell 3×** — pins the 0.4 m floor.
> 4. **Retrain** — only after 1–3, and only with a stated hypothesis about *what* was wrong with the
>    checkpoint. A retrain that changes the constraint set at the same time answers nothing, because
>    the two effects are then confounded.
>
> 🔴 **If a retrain does not move the scene out of the all-fail regime, that is itself the finding** —
> the over-strictness is a property of the constraint set, and the funnel is the correct treatment,
> not a workaround. TARGET §8 now carries this as a kill criterion.

**Known limits that qualify every V_A table.** Single seed (**seed 6**), 10 contexts ·
`geo_free`/`bounds_free`/`model_free` are constraint ablations, **never results** · MIN is n = 1, a
capability ceiling · `mean_dist_per_rollout` is **not** a distance · the success-criterion caveat is
still owed in `sec:setup:tasks`.

## 4. Entry C — the **UAV** topic (other embodiment) · ⚠️ **chapter ready, ranking not**

### 4.0 The headline, and the control-theory vocabulary for it

The author asked what the contrast with the manipulator is actually called. The precise terms:

| | manipulator (`avoiding`, `aligning`) | quadrotor (`uav-*`) |
|---|---|---|
| actuation | **fully actuated** — one actuator per DOF | **underactuated** — 4 rotor inputs for 6 pose DOF; thrust acts along body-*z* only, so it **must tilt to translate** |
| equilibrium | **statically / quasi-statically stable** — holds its configuration under gravity compensation, no control input needed to stay put | **open-loop unstable** — hover is an unstable equilibrium requiring **continuous closed-loop stabilisation** |
| what a plan must satisfy | **kinematic feasibility** — a reachable sequence of setpoints (IK → joint PD) | **dynamic feasibility** — second-order, thrust- and attitude-limited |
| control structure | one loop, planner-rate | **cascaded** (outer position → attitude → moment → rotor allocation), **multi-rate**: 33 Hz plan / 100 Hz physics, `n_dec = 3` |

> ### The UAV headline sentence
> **The transfer being claimed is from a fully-actuated, statically stable, kinematically-controlled
> plant to an underactuated, open-loop-unstable, second-order one** — and it is why this embodiment
> needs a cascaded geometric controller (Lee et al. on SE(3)) and a multi-rate loop at all. That is
> the contribution of the topic, independently of which engine wins.

### 4.1 Scene roles — per the 2026-09-10 framing

| scene | intended role | regime | status against that role |
|---|---|---|---|
| **`corridor`** | ⭐ **the headline scene** | all-pass | 🔴 **not finished** — §4.2 |
| **`pillars`** K=5 | the ranking scene | discriminating | 🟡 **almost ready** — needs seeds (§8 item 1) |
| **`s_curve`** | ⭐ **extreme case: what the controller cannot do** | all-fail | 🟢 **ready in that role** — §4.4 |
| `corridor_ball` (U11) | the fix for `corridor` | intended: discriminating | ⏳ in flight, gated (§4.5) |

### 4.2 🔴 Why `corridor` is not yet a headline

`corridor` is the scene the author wants to lead with, and it cannot lead yet — for a measured
reason, not a missing-run reason:

- **The constraint half of S&C carries no information there.** `n_violations` = 0.00 and
  `collision_free_completed` = 1.000 on **every** arm, **projected and unprojected alike**. The same
  columns report 197.70 violations on `pillars`, so the metric is sensitive — `corridor` really is
  constraint-trivial. ⇒ **S&C ≡ success**, and the scene cannot exercise the constrained-control
  problem the thesis is about.
- **The engine ordering there is statistically empty** — af 270.9 / mf 271.7 / fm 273.3 steps, spread
  **2.4 steps against σ ≈ 4.5–5.0**, paired split 5/10–5/10.
- **The projector is provably solving a problem that does not exist** — 155–202 ms/step to remove
  violations that were never there. *(This is a genuine finding and it belongs in
  `sec:disc:threats`; it is just not an engine result.)*
- ⚠️ **Coverage gap:** `af`'s K=1 arm would dominate everything, but **`mf`/`fm` have no `u7hg` K=1
  corridor arm**. Never quote it.

> **⇒ To make `corridor` the headline, the scene must be made to bind.** That is exactly what
> `corridor_ball` (U11) is for (§4.5). Until its gate passes, the UAV headline is `pillars`.

### 4.3 Claim ledger

| # | claim | grade | evidence of record | what must travel with it |
|---|---|---|---|---|
| C1 | **`mf > fm > af` on S&C** — mean over **17 matched variants**, `pillars` K=5: **mf 0.635 · fm 0.359 · af 0.235**; ceilings 1.000 / 0.900 / 0.700; cells ≥ 0.8: 7/17, 2/17, **0/17** | 🔴 **blocked on seeds** | [`CLOSURE_20260910`](../../../Gen15/Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md) §2 · [`T3`](../../../Gen15/Campaign_20260907_five_missions/DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md) | **seed 6, n = 10, one scene.** 0.1 in mean S&C **is one rollout** |
| C2 | 🔴 **The general principle does not hold here** — the author's *"`af` at least no worse, could be better"* is **contradicted** on the one UAV scene with dynamic range: `af` finishes **behind naive `fm`** and is the only engine with **zero** cells above S&C 0.8 | 🔴 **blocked on seeds**, ⛔ **safe as corroboration** | closure §2, §4 | this is the fourth environment to refuse the bootstrapped target. The containment argument that predicts otherwise is answered in [`FALLBACK_…` §10.2](../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md) |
| C3 | **`corridor`'s ladder is statistically empty** — the **same failure mode as `avoiding`**, reproduced independently | 🟢 **ready as a non-result** | closure §1 | two environments, same nominal order, same absence of power ⇒ a consistent *non-result* |
| C4 | **On `corridor` the constraint half of S&C carries no information** | 🟢 **ready — verified** | closure §1.1 | a methodological finding (§4.2) |
| C5 | **Endpoint projection is 8.2–21.3× cheaper to project than full-geometry DPCC** (5.6–12.4× end-to-end) at S&C parity (tally **8 · 9 · 1**) | 🟡 **strong, one seed** | T3 §3–§4 | ⚠️ against `dpcc-*-geo_free` the two cost the same (≈ 33.7 vs 34.8 ms) and the endpoint arm's dearer generator makes it **~27 % more expensive end-to-end in all 9 pairs** |
| C6 | **Every unsafe rollout in the study is a DPCC row** — endpoint projection **0 unsafe in 210**, DPCC **43 in 270**; **7 of 8 failing cells are the random rule**, simultaneously the most expensive projection row on every engine | 🟢 **ready — new, and nowhere else measured** | T3 §6 · closure §5 | one scene, one seed. UAV's strongest independent contribution; it feeds **Goal C** (§5) |
| C7 | **The single S&C = 1.000 cell** in the whole `u7hg` `pillars` family is `mf` · **`dpcc-t-geo_free`** · 411.1 steps · 78.74 ms · 0 violations | 🟡 **ready with a red label** | T3 §5, §7 | 🔴 `geo_free` removes the geometric rows from the projector's NLP. Violations are still *measured* against true geometry, so the pass is real — but the cell demonstrates **the generator, not the constrained projector** |
| C8 | **Projection is not universally worth it** — unprojected is 0.000 for `af` (essential), 0.900 for `mf` (marginal), **0.900 for `fm` and its best cell** (projection *hurts*) | 🟢 **ready** | T3 §7 | an engine-level question, not a global one |
| C9 | **Anything about the diffusion baseline on UAV** | ⬜ **no data** | closure §7.2 · [`MASTER campaign`](../../../Gen15/Campaign_20260907_five_missions/MASTER_20260908_five_missions_campaign.md) §0 | **no `u7hg` diffusion arm on any UAV scene** ⇒ `fm ≈/≻ diffusion` is **untestable on UAV**. That half of the general principle rests on `avoiding` |

### 4.4 ⭐ `s_curve` as the extreme case — a role it can actually fill

Reframed per the author: `s_curve` is **not** a ranking scene, it is the demonstration of **where the
control stack runs out**. In that role it is 🟢 ready, and the evidence is already analysed:

- **S&C ceiling 0.100 across 40 `u7hg` cells**; no arm clears funnel Stage 2. As a *ranking* scene it
  is ⛔ retired — and that retirement is the point being made.
- **The tracker, not the planner, is the binding element.** T4's *"`fm > mf` on `s_curve`"* was
  **tracker-confounded and is retracted** (closure §3.1) — which is precisely the evidence that the
  scene measures the controller.
- **The controller study exists:** [`T5 MJPC vs PID`](../../../Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md) ·
  [`Report_20260909_MJPC_vs_PID_s_curve`](../../../../Data_Analysis/DA_Result_Curated_MD/Report_20260909_MJPC_vs_PID_s_curve/README.md).
- **Budget was tested and does not rescue it** — K > 5 is non-monotone (closure §6).

> **Write it in `sec:res:uav` as the aggressive-reference limit case and in `sec:disc:limitations`,
> never in an engine table.** It is the scene that shows the *embodiment* claim has an edge — which
> is a stronger and more honest use than a failed ranking.

### 4.5 In flight as of 2026-09-10 — **not yet landed**

Verified against the newest drop (`temp/0909/batch_uav_20260910_092309`, 96 candidates): **no
`corridor_ball` scene, no `u7hg` `diffusion` arm.**

| jobs | what | effect on v3 |
|---|---|---|
| **25599–25604** | `corridor_ball` (U11): a virtual centre ball forcing a vertical detour; tier A (K=2, endpoint arm degenerate) and tier B (K=5, `n_genuine = 2`) | ⭐ **this is what could make `corridor` the headline.** 🔴 **Gate first:** the unprojected row must report `n_violations > 0`. If it is still 0.00 the ball is not binding and the scene is geometry work, not analysis |
| **25611–25614** | `pillars` diffusion (train + eval), `s_curve` diffusion, `s_curve` af K=5 | **§7.2 resolved, not upgraded** — the diffusion arm is a **reference row for scale, not a proof**: single seed, budgets deliberately unmatched (diffusion K=20 vs the flow family at K=5). *"`mf` beats the DPCC baseline"* stays an **`avoiding`** claim |
| **— (not submitted)** | **seeds on `pillars` K=5** | 🔴 **the one required run.** Nothing in flight addresses it |

### 4.6 Does the TARGET's UAV kill criterion fire?

> **Partially — and the honest reading is: no, but the ranking half is one seed deep.**
> Under honest geometry `pillars` K=5 **does** discriminate, so UAV is not demoted to
> feasibility-only. `s_curve` is infeasible and is retained **as the controller-limit case** (§4.4).
> `corridor` is feasible but **constraint-trivial** (§4.2). The chapter carries: the embodiment
> argument (🟢) + the regime taxonomy (🟢) + the projector cost/safety result (🟡/🟢) + **one**
> ranking scene whose every sentence says *seed 6, n = 10*.

**Binding, unchanged:** `budget_ms` and the 33 Hz figure are a data-rate and cluster-latency
artefact, never a pass/fail criterion.

## 5. Target vs. fallback — which entries land where

🎯 = at the TARGET's own statement · 🪂 = only at the FALLBACK statement · ⬜ = untestable here

| goal | `avoiding` (1) | *aligning 3-D state* (2a) | `V_A` (2b) | `uav` (3) | net |
|---|---|---|---|---|---|
| **A — engine ladder** `af ≥ mf > fm > diffusion` | 🪂 top rung nominal-only (p = 0.231) | ⬜ | 🪂 top rung **killed**; bottom rung **refuted** | 🪂 top rung **refuted**; bottom rung ⬜ | 🪂 **fallback everywhere** |
| A′ — fallback ladder `mf ≫ {af, fm, diffusion}` | 🎯 holds | ⬜ | 🎯 holds, decisively | 🎯 holds on `pillars`, one seed | ✅ **the shipping claim** |
| A.bottom — `fm ≻ diffusion` | 🟡 **survives here only**, as a *cost* claim (21×) | ⬜ | ⚪ **a tie** — 0.0055 m in a 0.4 m band | ⬜ no diffusion arm | 🟡 **one environment, cost only.** The author's *"or > here"* does not hold on V_A |
| **B — projector ladder** | 🎯 at K=3, 🟡 underpowered (24 rollouts) — **a side topic here** (§2.3) | ⬜ | 🎯 **at K=10, p = 0.0039 — select this one** (§3.3) | 🎯 on cost + safety, 🟡 one seed, narrowed by `geo_free` | ✅ **target-level, three environments** |
| B — the *regime split* (low `n_genuine` → iterate projection, high → endpoint projection) | 🟡 the `n_genuine` boundary is exact; the *split* is shown at two points, not mapped | ⬜ | 🟡 K=10 win, K=20 parity — consistent with the split | 🟡 K=5 parity on S&C, win on cost | 🟡 **argued, not mapped** |
| **C — remove selection machinery** | 🎯 strong form **exact** at B=1 | ⬜ | 🟡 the **random** rule is the flagship one here | 🎯 **new:** the random rule is the only unsafe configuration (7 of 8 failing cells) *and* the most expensive | ✅ **strong form proven; the default is task-dependent** |

### The three sentences this table licenses

1. **Goal A ships on the fallback.** `mf ≫ {af, fm, diffusion}` — one engine separates, three
   cluster — demonstrated across **three environments spanning all three regimes**. The α-Flow rung
   is a **negative result with a derived mechanism** (`FALLBACK_…` §2), not a gap.
2. **Goal B holds at target level in all three entries**, at three different strengths, and the
   thesis should state the *citable* form per environment (K=3 `avoiding` · K=10 V_A · cost+safety
   UAV) rather than one pooled claim.
3. **Goal C's strong form is proven exactly and UAV upgrades the weak form.** `-r` moves from
   "documented null control" to "the only configuration that put the aircraft in an unsafe state,
   at the highest projection cost" — an argument for disabling it in the *deliverable* that the
   `avoiding`/V_A analyses could not make. It stays in the *evidence* as the control that makes the
   rule comparison interpretable.

---

## 6. The benchmark matrix as it actually stands

TARGET §4 now orders the matrix by **what each environment adds**. Against that:

| topic | environment | adds | status |
|---|---|---|---|
| 1 | `avoiding-d3il` | — (2-D planar state) | 🟢 5 seeds × 20 trials; **Tier 1 (5 × 2) owed** |
| **2a** | `aligning-d3il` **3-D state** | **harder control** | 🔴 **env exists (vendored d3il), config does not.** Build + train + eval |
| 2b | `aligning-d3il-visual` | **harder perception** | 🟢 seed 6, 10 contexts; `n_contexts: 50` owed |
| 3 | `uav-corridor` | underactuated, unstable plant | 🟡 all-pass **and constraint-trivial** — `corridor_ball` in flight |
| 3 | `uav-pillars` | a discriminating constraint set | 🟡 seed 6, n = 10 — seeds owed |
| 3 | `uav-s_curve` | an unholdable reference | 🟢 **as the controller-limit case**; ⛔ as a ranking scene |

🔑 **The matrix no longer has an orphan row.** In the previous revision `aligning-d3il` 3-D state was
a row with no data and no owner, and the recommendation was to drop it. Under the 2026-09-10 framing
it is **leg 2a of topic 2** and it is *required* — it is what separates the two difficulty increments.
TARGET §8 carries the kill criterion for it not landing: ship topic 2 on the visual leg alone and
**say plainly that both increments were added at once**.

**Also unchanged and binding:** `Gen16` visual-avoiding and `Gen9` remain *undecided* in
[`NOTES_open_questions.md`](../../Auxiliary/NOTES_open_questions.md) §2 — no results have landed, so
v3 proceeds as if they are out.

---

## 7. v3 writability map — section by section

Chapters 5–8 of [`../v2/thesis_v2.tex`](../v2/thesis_v2.tex) are bone (headers only), except the
compute-environment subsection in `sec:setup:protocol` and `tab:compute` in `app:repro`.

| section | can v3 write it? | source | note |
|---|---|---|---|
| `sec:setup:tasks` | ✅ yes | AUX methodology sources + UAV closure §0 | includes the **regime taxonomy** (all-pass / discriminating / all-fail) — write it here, it is used by three chapters |
| `sec:setup:baselines` | ✅ yes | TARGET §6.1 | pin the Target once, with its numbers (§2 above) |
| `sec:setup:protocol` | ⚠️ **Tier 1 owed** | §2.0 | the **two protocol tiers** and the measured n=2 → n=20 shift (±0.10 resolution, 1.00 ceilings, 0.35 swings). Needs the fresh `n_trials = 2` run |
| `sec:setup:metrics` | ✅ yes | TARGET §6.2, open question 6 | Pareto definition + the **funnel** + the 0.4 m V_A reproducibility floor |
| `sec:res:state` | ✅ **yes — this is the foundation section** | §2 A1–A8 | lead with **A1** (Pareto dominance, 300 episodes a side); the low-K pair `{mf, af}` stays a pair; `fm` is a **cost** claim (§2.1a) |
| `sec:res:fewstep` | ✅ **yes — lead with it** | §3 B4 + `FALLBACK_…` §2.3 | the **K-ladder / flat-in-K table** is the figure the engine chapter turns on |
| `sec:res:constraints` | ✅ yes | §3 B3, §4 C5–C6, §2.3 A9–A11 | state the citable **endpoint-projection** claim **per environment**; never pool. On `avoiding` it is a **side topic** (§2.3) |
| `sec:res:constraints:degenerate` | ✅ yes | `n_genuine` boundary, exact | `n_genuine = max(K − int((1−A)·K), 1) − 1`; K = 1 is structurally impossible for arm C at every A |
| `sec:res:visual` | ⚠️ **mostly** | §3.2 | write B1, B2, B4, B7 · state `fm ≈ diffusion` as a **tie** (B3) · **select K = 10** for the projector claim (§3.3) · **hold the constraint-vs-baseline table** until tightened `diffusion` K=20 (B6) |
| *aligning 3-D state* | 🔴 **no** | §3.0 | no config, no checkpoints, no corpus. Either run 4 lands, or `sec:setup:tasks` declares the leg out of scope |
| `sec:res:uav` | ⚠️ **chapter yes, ranking guarded** | §4 | lead with the **embodiment argument** (§4.0: underactuated, open-loop unstable, dynamic feasibility) · `pillars` is the ranking scene until `corridor_ball` binds (§4.2) · `s_curve` is the **controller-limit case**, never an engine table (§4.4) · every ordering sentence carries *seed 6, n = 10* |
| `sec:res:ablations` | ✅ yes | §2 A7–A9 | Goal C: exact at B=1, directional on the default rule, plus the parallelisation recommendation |
| `sec:disc:negative` | ✅ **yes — two entries owed** | `FALLBACK_…` §5 · B8 · C9 | α-Flow (with the mechanism) · the withdrawn `fm > diffusion` · `s_curve`'s retirement · iMF is *out of scope* per open question 2 |
| `sec:disc:interpretation` | ✅ yes | TARGET §2 tension + C8 | where Goal A (push K down) and Goal B (needs genuine steps) meet: moderate K, low threshold |
| `sec:disc:threats` | ✅ yes | C4 + `FALLBACK_…` §6 | **task saturation is a threat to validity of single-benchmark engine comparisons in general** — found in our own results. Add the `corridor` case: a projector spending 155–202 ms/step on violations that never existed |
| `sec:disc:limitations` | ✅ yes | §4.4, §3.4 | `s_curve` as the controller's edge · the V_A all-fail regime and what a retrain can and cannot fix |
| `sec:bg:fewstep` | 🔄 **reverse the deferral** | `FALLBACK_…` §6.1 | v2 says α-Flow's mathematics is "deliberately deferred". Under the fallback **the mathematics is the contribution** — write eq. (1)–(4) here or in `app:derivations` |
| Abstract · `sec:intro:contributions` | ⏳ last | Goal A′ | the `\hole{}` headline numbers come from §5, not from the old ladder |

---

## 8. Required tests, ordered by value per GPU-hour

Re-ordered 2026-09-10 against the topic framing. **Cheap-and-unblocking first, retrain last.**

| # | run | cost | unblocks | if it never lands |
|---|---|---|---|---|
| **1** | ⭐ **V_A at `n_contexts: 50`** (1 seed, no retrain; or 25 × 2 trajectories) | 💰 low — eval only | 5× the paired sample on the leg that carries the engine claim; brings the protocol near D3IL's own 60 | every V_A claim stays at **10 pairs**, where a 7/3 split is p = 0.34 |
| **2** | **Seeds on `uav-pillars` K=5** — 2–3 seeds × 3 engines × 17 variants | 💰💰 medium | every UAV ordering sentence in `sec:res:uav` | UAV ranking stays 🔴 one-seed; *"worst where the measurement is sharpest"* is withdrawn |
| **3** | ⭐ **Tier 1 on `avoiding`** — all engines at `n_trials = 2`, seeds 6–10, one binary | 💰 low — ≈ 1/10 of the n=20 tier | the two-tier protocol story in `sec:setup:protocol`; A4's low-K baseline cells. **Cannot be subsampled from Tier 2** (§2.0) | the baseline is compared only at *our* protocol, and the n=2 → n=20 point rests on baseline-only data |
| **4** | 🔴 **Leg 2a — `aligning-d3il` 3-D state**: write `config/aligning-d3il.py`, train 4 engines, one eval sweep | 💰💰💰 high — build + train + eval | **topic 2's first leg.** Separates *harder control* from *harder perception*; makes the state→visual transfer claimable | topic 2 ships visual-only and the thesis states that both increments were added at once (TARGET §8) |
| **5** | **Tightened `diffusion` K=20 on V_A** | 💰 low — eval only | the held constraint table (B6) | the V_A ranking stays **plan-quality-only** |
| **6** | **`corridor_ball` gate** (25599–25604, in flight) | ⏳ submitted | ⭐ could make `corridor` the UAV **headline** scene | `corridor` stays constraint-trivial; `pillars` carries UAV alone |
| **7** | **Power A9** — endpoint projection on `avoiding`, `A=1.0`, K ∈ {2,3,5}, seeds 6–10, `n_trials = 20` | 💰💰 medium | promotes the `avoiding` projector side topic from directional to a Pareto claim | it stays 🟡 at 24 rollouts |
| **8** | **Repeat one projected `mf` V_A cell 3×** | 💰 low | pins the ~0.4 m reproducibility floor qualifying every MIN | the floor stays a single-DA estimate |
| **9** | `pillars`/`s_curve` diffusion reference rows (25611–25614, in flight) | ⏳ submitted | matrix completeness only — **a reference row, not a proof** | a reviewer asks why an engine is missing |
| **10** | ⚠️ **V_A retrain + re-eval** | 💰💰💰 high | only the checkpoint-quality hypothesis (§3.4) | **run 1, 5 and 8 first.** A retrain that also changes the constraint set answers nothing — the effects confound |
| **11** | Export `solve_ms` + SLSQP iteration counts for arm B | 💰 code | turns A10's *inferred* cost mechanism into a measured one | A10 stays an inference from timing |
| **12** | τ = 0.850 — constraint-Jacobian conditioning | 💰 code | the one open bug; engine-side already ruled out | a known, documented solver failure |

**Explicitly not needed** (closed): α-Flow compute for its own sake — one pre-registered re-entry
only (α ≥ 0.4 on V_A K=20); any `s_curve` ranking compute; UAV K > 5; a `corridor` K-sweep;
re-running the endpoint arm at `pillars` K=5.

🟢 **One free rider:** if run 4 happens, the bootstrapped-target arm rides along at no extra design
cost and the α-Flow closure gains a **fifth** environment
([`FALLBACK_… §10.3`](../fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md)).

---

## 9. Rules binding on every v3 table

Carried from TARGET §6 and `FALLBACK_…` §7, plus three the UAV drop adds.

1. **Target pinned once** — diffusion DPCC K=20 / aw=10 / `GaussianDiffusion`, at its best
   projection variant. Never renegotiated per table.
2. **Every table carries backbone + parameter count.** Architecture-matched (U-Net 4.0 M) rows lead;
   SiT/DiT rows are secondary and labelled confounded.
3. **No aggregation across projectors.** Unit = `(engine × projector × geometry × split)`.
   Model-vs-model uses each model's own best projector, **always named**.
4. **Degenerate HardFlow rows tagged, never averaged in.**
5. **UAV `budget_ms` / 33 Hz is never a pass/fail criterion.**
6. **Never write "α-Flow fails"** — write *"does not improve on MeanFlow"*, and **name the α floor**
   (0.2 vs 0.05 are different arms with different verdicts).
7. **Tag the nominal orderings.** `avoiding` (p = 0.231) and `corridor` (2.4 steps vs σ ≈ 4.5) are
   inside noise — never "wins", not even "slight wins".
8. 🆕 **Label `geo_free` wherever it appears.** It removes the geometric rows from the projector's
   NLP; violations are still measured against true geometry, so a `geo_free` pass is real — but it
   demonstrates the generator, not the constrained projector (C7).
9. 🆕 **State the seed depth** on every UAV sentence: *seed 6, n = 10*.
10. 🆕 **Never quote `corridor` K = 1** — only `af` has that arm.
11. 🆕 **Mechanism names, not code tags.** *"HF-SLSQP"*, `hardflow_sls-*`, `dpcc-c-tightened`, bare
    *"diffuser"* are artefact tokens and never appear in prose. Arm C is **in-ODE endpoint
    projection**, arm B **per-step iterate projection**, and the solver is a backend row in
    `app:repro` — see [`NOTES_method_naming.md`](../../Auxiliary/NOTES_method_naming.md), whose §3.1
    decision is still the author's to confirm.

---

## 10. Corpora of record — what the numbers come from

| entry | corpus (on disk) | scale | protocol |
|---|---|---|---|
| `avoiding` — engine ladder | `temp/2508/batch_avoiding_combined_20260825_143212` | 240 512 raw rows | 5 seeds (6–10) × 20 trials = **100 episodes/cell** |
| `avoiding` — **Tier 1 (n=2) baseline reference** | `temp/1808/batch_avoiding_combined_20260818_152911` (job 24639) | — | 5 seeds × 2 trials = **10 episodes/cell**; ⚠️ `post_processing` rows corrupt; flow arms only in pre-08-19 batches on an older binary |
| `avoiding` — α-Flow U-Net | `temp/0309/batch_avoiding_combined_20260903_133730` | 266 567 raw rows | **seed 6**, `n_trials = 20` |
| `avoiding` — HardFlow min-K (job 25444) | `temp/0609/I/batch_avoiding_combined_20260906_125724` | 277 925 raw rows | 4 seeds (7–10) × 3 geometries × `n_trials = 2` = **24 rollouts/row** |
| `V_A` | `temp/0609/II/batch_va2_20260907_141036` | 673 config rows / 14 102 rollout rows / 33 832 raw | **seed 6**, 10 paired contexts, train split |
| `uav` (campaign) | `temp/0909/batch_uav_20260910_092309` (+ `…_20260908_153947`, `…_20260909_205118`) | 96 candidates / 48 486 raw rows / 1 299 units, 0 failed | **seed 6**, n = 10 (C94: n = 3 by design) |

**Compute environment of record:**
[`AUX_compute_environment.md`](../../Auxiliary/Methodology_Sources/AUX_compute_environment.md) —
2 × AMD EPYC 7282, 8 × RTX A5000 24 GB, one GPU per job, SLURM 21.08.5; node shared (CPU/memory/IO
contention only), and evaluation is plausibly **CPU-limited** (MuJoCo rollout + per-step SLSQP).

---

## 11. Pending one-line edits elsewhere in `Writing/` — not made silently

Left open by `FALLBACK_…` §9 and still true today:

| file | what is stale | fix |
|---|---|---|
| [`../v2/README.md`](../v2/README.md) | `sec:method:engine` is described as *"better nowhere **across two environments**"* | **four** environments and three regimes |
| [`../../Auxiliary/NOTES_open_questions.md`](../../Auxiliary/NOTES_open_questions.md) §2 | the scope list still carries Gen3v7 α-Flow as an "in" generation with no demotion note | add: in, **as a negative result with a derived mechanism** |
| [`../v2/thesis_v2.tex`](../v2/thesis_v2.tex) `sec:bg:fewstep` | *"No mathematics is given for it in this draft — deliberately deferred."* | reverse the deferral (§7) |

---

## 12. Maintenance

- **One file per status date.** Add a new `DATASTATUS_<YYYYMMDD>_*.md` in this folder when the
  picture changes; do not rewrite history in place. The newest file wins.
- **A grade may only move on a DA**, never on a run landing. Data on disk is not a result.
- **This folder never holds findings.** Findings live in
  [`Data_Analysis/DA_Result_Curated_MD/`](../../../../Data_Analysis/DA_Result_Curated_MD/) and in the
  per-generation logs; this folder holds *readiness*.
- `logs_in_develop/MASTER_TEST_HISTORY.md` is **not** updated from here.
