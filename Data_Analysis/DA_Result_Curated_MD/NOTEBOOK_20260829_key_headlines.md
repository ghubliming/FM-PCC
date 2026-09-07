# FM-PCC — Key Headlines Notebook

**Last updated:** 2026-09-07 · **Author:** curated from the reports in this directory  
**Status:** 🟢 Visual aligning **closed** · 🟢 Avoiding **strongest result to date** · 🔴 UAV **unrankable**

## How to read this file

Headlines are **append-only and numbered by the date they were written**, so a later headline can
overturn an earlier one. Before quoting any number, check the 🔴/⚠️ banner on its headline.

| block | written | trust |
|---|---|---|
| **[Headlines 1–5](#headline-1--avoiding-state-based-meanflow-unet-is-pareto-dominant-)** | 2026-08-29 | ⚠️ H2 and H3 are **superseded** by H10; H1 and H5 stand |
| **[Headlines 6–8](#addendum-2026-09-05--evidence-against-the-thesis-target)** | 2026-09-05 | ✅ current; H6 is *strengthened* by H9 |
| **[Headlines 9–10](#addendum-2026-09-07--two-results-that-close-two-questions)** | 2026-09-07 | ✅ **current — start here** |

**The two-sentence state of the project.** On `avoiding-d3il`, MeanFlow-UNet with the HardFlow-SLSQP
sampler at K = 3 **Pareto-dominates the pinned DPCC baseline** — equal safety, 16 % fewer steps,
7.4× cheaper, architecture-matched (H9). On `aligning-d3il-visual`, MeanFlow is the flagship and the
engine comparison is **closed**: α-Flow is never better, naive FM is excluded, and only MeanFlow
separates from the diffusion baseline (H10). UAV remains the one environment that cannot yet rank
anything.

---

## Document index — this directory

🟢 current · 🗄️ outdated (renamed `outdated_*`, banner at the top of each says what replaced it)

| doc | covers | state |
|---|---|---|
| **this file** | the headline index | 🟢 |
| [`Proposal_20260905_HF_minK_mf_af_unet/`](Proposal_20260905_HF_minK_mf_af_unet/README.md) → [`DA_20260906`](Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md) | HardFlow's K floor + the result at it | 🟢 **newest result** |
| [`Report_20260903_AF_UNet/`](Report_20260903_AF_UNet/README.md) | α-Flow with α actually on, avoiding | 🟢 |
| [`Report_20260819_MF_UNet/`](Report_20260819_MF_UNet/README.md) | MF-UNet Pareto dominance, avoiding | 🟢 |
| [`DA_20260819_ntrials20_…`](DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md) | n=20 cross-family, avoiding | 🟢 |
| [`DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2`](DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md) | why n=2 overstates — pins the Target | 🟢 methodology |
| [`DA_20260827_mpc_candidate_fan_avoiding`](DA_20260827_mpc_candidate_fan_avoiding.md) | the B=4 → B=1 fan study | 🟢 |
| [`SNAPSHOT_20260826_visual_avoiding_env_status`](SNAPSHOT_20260826_visual_avoiding_env_status.md) | `avoiding-d3il-visual` — the only doc on it | 🟢 |
| [`ANALYSIS_20260829_alphaflow_vs_meanflow…`](ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md) | α-Flow ≡ MeanFlow at α=0, from source | ⚠️ mechanism 🟢, open question **answered** |
| [`outdated_RESPONSE_20260826_did_HardFlow_ever_beat_DPCC`](outdated_RESPONSE_20260826_did_HardFlow_ever_beat_DPCC.md) | "No" — measured on IPOPT at an unmatched threshold | 🗄️ **reversed** |
| [`outdated_AUDIT_20260827_hardflow_paper_timing…`](outdated_AUDIT_20260827_hardflow_paper_timing_and_baselines.md) | "HF loses 1.4–14× to our SLSQP baseline" | 🗄️ **reversed** |
| [`outdated_Report_20260829_VA_funnel/`](outdated_Report_20260829_VA_funnel/README.md) | the funnel that crowned `mf` K=2 | 🗄️ rankings stale, **method still standard** |
| [`outdated_SNAPSHOT_20260823_visual_aligning_env_status`](outdated_SNAPSHOT_20260823_visual_aligning_env_status.md) | V_A whole-env, pre-closure | 🗄️ |
| [`outdated_SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline`](outdated_SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md) | avoiding whole-env at the n=2 tier | 🗄️ |
| [`outdated_SNAPSHOT_20260825_uav_mix_env_status_PILOT`](outdated_SNAPSHOT_20260825_uav_mix_env_status_PILOT.md) | UAV whole-env, pre-Fix_16 | 🗄️ **no replacement exists** |
| [`outdated_DA_20260826_K_sampler_steps_visual_aligning`](outdated_DA_20260826_K_sampler_steps_visual_aligning.md) | the V_A K ladder, pre-α-live | 🗄️ |

> External anchors this notebook leans on:
> [`Gen14/CLOSURE_20260907`](../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md) (V_A closed) ·
> [`Writing/Working_Space/TARGET_20260905_thesis_claim_ladder.md`](../../logs_in_develop/Writing/Working_Space/TARGET_20260905_thesis_claim_ladder.md) (the goals this is scored against).

---

## Timeline

| date | milestone |
|---|---|
| 2026-08-04 | First MeanFlow vs α-Flow training run on visual-aligning; α-cliff measured (2.5× MSE jump) |
| 2026-08-13 | Avoiding-d3il snapshot: DPCC baseline vs FM-PCC first comparison |
| 2026-08-18 | MF-UNet avoiding batch collected (5 seeds × 2 trials, K-ladder 1–20) |
| 2026-08-19 | **MF-UNet Report** — Pareto dominance established (30× `avg_time`, S&C tied) |
| 2026-08-19 | n=20 cross-family DA: DPCC vs FM vs MeanFlow vs AlphaFlow on avoiding |
| 2026-08-23 | Visual-aligning batch: 4 engines × multiple projectors, seed 6, 30 contexts |
| 2026-08-25 | UAV corridor PILOT snapshot: first whole-env pass, 13 candidates, 3 engines |
| 2026-08-25 | FM K=20 (NFE-matched control) added to avoiding DA — 21× decomposed |
| 2026-08-26 | K-sampler-steps sweep on visual-aligning |
| 2026-08-29 | **VA funnel report** — 4-engine funnel, mf/af win Stage 1–3 |
| 2026-08-27 | MPC candidate-fan study: B=4 → 1, fan scales projector only, sign flips by model |
| 2026-08-29 | α-Flow vs MeanFlow analysis — confirmed same engine, different curriculum |
| 2026-08-30 | **IPOPT → SLSQP adopted**; at matched threshold HF-SLSQP `-t` Pareto-dominates DPCC on avoiding K20 |
| 2026-09-01 | **VA flagship** K20/T0.2 — first non-degenerate HF cell on visual aligning; HF ≥ DPCC on constraints 3/3 |
| 2026-09-03 | **Fix_16 A/B on `pillars`** — divergence abort 100 % → 0 %; scene still unrankable (S&C 0/2876) |
| 2026-09-04 | Honest-geometry audit — UAV scene slack measured below policy tracking error |
| 2026-09-05 | **Thesis target written**; evidence board + Headlines 6–8 |
| 2026-09-06 | **HardFlow's floor run** (job 25444) — MF-UNet, `A=1.0`, K ∈ {2,3,5}; first ✅-tier arm-C row to clear the Target |
| 2026-09-07 | **Gen14 V_A closed** — `mf` flagship, `af` closed, `af`-SiT abandoned, `fm` excluded |

---

## Headline 1 — Avoiding (State-Based): MeanFlow-UNet is Pareto-Dominant ✅

> **Source:** [Report_20260819_MF_UNet](Report_20260819_MF_UNet/README.md) + [DA_20260819_ntrials20](DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md)

### The claim

MeanFlow-UNet using the **JVP training target** on the **same temporal U-Net** backbone achieves **Pareto dominance** over the DPCC diffusion baseline on the avoiding-d3il task:

| | S&C | `avg_time` | `n_steps` |
|---|---|---|---|
| DPCC K20 (baseline) | 1.00 | 0.553 s | 70.1 |
| **MF-UNet K1** | **0.97** | **0.018 s** | **58.6** |
| **ratio** | tied | **30×** | **12 fewer** |

### Why it works

- MeanFlow trains the network to predict the **average velocity over an interval** → the whole ODE traversable in **one evaluation** (K=1).
- K is **inference-only** for MeanFlow but a **training parameter** for DPCC diffusion — so the baseline cannot follow down the K-ladder (DPCC K1 → S&C 0.67).
- Architecture is **matched**: same UNet, same dim/mults, same action weight, same constraint projector. Only the generative model and K differ.

### Also: Naive FM beats DPCC too

The 21× speedup decomposes into: **≈1.4×** from FM's model advantage at equal NFE + **≈15×** from FM's ability to run at K=2 where diffusion cannot. FM K=2 Pareto-dominates DPCC K20 on both complete halfspaces (S&C ≥ 1.00, 21× cheaper per episode).

> **Status: ✅ Established.** Robust across trial counts (n=2 and n=20 agree within 8% on cost axes). One known limit: MF-UNet `both-hard` under cost-based selection degrades (0.85 at n=20), mitigated by `dpcc-t-tightened` selection.

---

## Headline 2 — α-Flow vs MeanFlow: Same Engine, Different Curriculum ⚠️

> 🗄️ **SUPERSEDED on the verdict — see [Headline 10](#headline-10--visual-aligning-is-closed-meanflow-is-the-flagship-).**
> The open question this headline ends on — *"turn α on and re-measure"* — **was run**
> (`AFAFend0p2` / `AFAFend0p05`). With α provably live and architecture matched, α-Flow ties
> MeanFlow at K=2, loses at K=20, and is **better nowhere**; `af`-U-Net is **closed** and every
> `af`-SiT row is **abandoned** (9.4 M vs 4.0 M *and* `ae0.0` ⇒ MeanFlow mislabelled).
> **The mechanism below still stands** — it is what let the SiT rows be identified as MeanFlow.

> **Source:** [ANALYSIS_20260829_alphaflow_vs_meanflow](ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md)

### The finding

α-Flow is **not a different method** from MeanFlow on the visual-aligning task — it is a **training curriculum** for MeanFlow. The α-Flow paper's own title says it: *"Understanding and Improving MeanFlow Models"*.

| phase | optimizer steps | α | what trains | share |
|---|---|---|---|---|
| Phase 1 | 0 → 28 820 | **1.0** | pure Flow Matching | 28.8% |
| Phase 2 | 28 830 → 71 170 | 1.0 → 0.0 | bootstrapped target | 42.4% |
| Phase 3 | **71 180 → 100 000** | **0.0** | **MeanFlow (identical)** | **28.8%** |

The evaluated `af` checkpoint was **last optimised against MeanFlow's exact JVP loss for 28 820 steps**. Same bone (visual UNet), same sampler (byte-identical), same trainer, same budget.

### ⚠️ The α → 0 snap costs 2.5× on this task

At step 70k, α ≈ 0.007: test MSE(u) = **2.657**. At step 72k, α = 0.0: test MSE(u) = **8.504** — a 2.9× step-change, never recovered. MeanFlow's own plateau is 7–10. **The V_A results do not measure α-Flow at its own operating point.**

### Does α-Flow with the SiT achieve slightly better results?

**On visual-aligning: no.** Both arms use the same visual UNet (4.04 M). The SiT backbone exists in code (`af_sit_trajectory.py`, 10.00 M params) but was **deliberately not used** — to keep the comparison architecture-controlled. The two are **not statistically separable** (sign p = 0.136, Wilcoxon p = 0.393).

**On UAV: possibly, but confounded.** α-Flow uses the SiT backbone (10.00 M, 2.53× the U-Net) and achieves strong results (21 W / 1 L vs FM at K1–K5). But this is **architecture-confounded** — the win could be the bigger network, not the training target. No `af` @ UNet exists for UAV.

> **Status: ⚠️ Partially established.** The decisive experiment — a **constant-α** run without the α→0 snap — has **never been run**. Do not report mf and af as two independent wins.

---

## Headline 3 — Visual Aligning: MeanFlow K2 is the Sole Survivor ✅

> 🗄️ **SUPERSEDED — see [Headline 10](#headline-10--visual-aligning-is-closed-meanflow-is-the-flagship-).**
> The engine survives; **the operating point does not.** The flagship is now **`mf` K=20, T=0.2,
> arm C `hardflow_sls-r`**, not `mf` K=2 + `dpcc-t`. Two things changed after this was written:
> applying Stage 2 in the strict order put **K=20 ahead of K=10 and K=2**, and the K=20/T=0.2 cell —
> the first non-degenerate arm-C cell on this task — did not exist yet. The entrant list here also
> has **no `af`** and no arm C at all.
> **The three-stage method is not superseded**; it is still the house standard.

> **Source:** [outdated_Report_20260829_VA_funnel](outdated_Report_20260829_VA_funnel/README.md) (rankings stale)

The report now runs a strict **three-stage funnel** — an arm leaves the moment it fails a stage. No `af` in the entrants (see Headline 2 — it is the same engine as `mf`). Entrants: **MF K2/K100 · FM K20/K100 · Diffusion K20/K100** on the matched UNet FiLM v1 bone.

### The funnel

| entrant | **Stage 1** · unguided `diffuser` | **Stage 2** · projected | **Stage 3** · cost | |
|---|---|---|---|---|
| **MeanFlow K100** | **0.28×** ✅ | ❌ truncated 11/30 — needs **50 h** | — | 🔴 24h wall |
| **Diffusion K100** ⚠️ | **0.41×** ✅ | ❌ truncated 19/30 — needs **28 h** | — | 🔴 24h wall |
| **MeanFlow K2** | **0.60×** ✅ | ✅ `dpcc-t` tightened · **`0-viol` 1.00** | **42 ms** | **🏆 sole survivor** |
| FlowMatching K100 | 0.95× ❌ | — | — | never engages |
| Diffusion K20 | 0.96× ❌ | — | — | never engages |
| FlowMatching K20 | 0.98× ❌ | — | — | never engages |
| *d3il baseline* | *1.000×* ❌ | — | — | *never engages* |

### 🏆 One arm crosses all three stages: **MeanFlow at K = 2**

**Stage 1 (unguided, no projection):** Three arms move the box — MF K100 (0.28×), Diffusion K100 (0.41×), MF K2 (0.60×). Everything else is at 0.95–1.00× of the starting gap — no-ops. Gate set at 0.80×; nothing sits near the line.

**Stage 2 (projected, constraints):** The two K=100 arms that beat MF K2 on distance **were eliminated by cost, not quality**. At `T = 0.5`, K=100 multiplies the SLSQP projection solves 50× — their cells need 28–50 h against a 24 h Slurm wall. **The configurations that get closest to the goal are precisely the ones that cannot be evaluated.** MeanFlow K2 is the only arm scorable at Stage 2: tightened `dpcc-t` → **`0-viol` = 1.00**, 10/30 within 15 cm, at zero distance cost.

**Stage 3 (cost):** The survivor is also the **cheapest thing on the board**, by 8–357×:

| | clean & <15 cm | `0-viol` | `ms` | vs survivor |
|---|---|---|---|---|
| **🏆 MeanFlow K2 · tightened · `dpcc-t`** | **10/30** | **1.00** | **42** | — |
| FM K20 · `post_processing` | 1/30 | 0.13 | 323 | 8× |
| Diffusion K20 · `dpcc-r` | 2/30 | 0.10 | 2 158 | 51× |
| *MF K100 · `dpcc-r`* (unscorable) | — | — | 14 988 | 357× |

### Statistical confirmation against Stage-1 eliminations

The eliminated K=20 arms have complete projected cells, confirming the elimination (exact McNemar):

| comparison | A-only | B-only | **p** |
|---|---|---|---|
| **MF K2 (tightened) vs FM K20 `post_processing`** | **9** | 0 | **0.004** |
| **MF K2 (tightened) vs Diffusion K20 `dpcc-r`** | **10** | 2 | **0.039** |

On distance: −0.154 m vs FM (Wilcoxon **0.004**), −0.132 m vs Diffusion (sign **0.029**, Wilcoxon **0.012**).

### 🔴 What this does NOT license: sim2real

The report explicitly disclaims any transfer claim:
- The task is **not solved** in simulation — success 0–2/30 everywhere, surviving arm still leaves 49% of the gap
- **Rotation is uncontrolled** — median final/initial ≈ 1.00× on every arm
- **No held-out split** — all entrants are train-split
- **`0-viol` = 1.00 needs tightened geometry** — uncontested, not won
- 42 ms is cluster GPU + CPU SLSQP, not an embedded target

### Caveats

- ⚠️ **Single seed (6)**, one checkpoint per engine, 30 paired contexts
- ⚠️ **Train split only** — no generalisation demonstrated
- ⚠️ **Two best Stage-1 arms unscorable** — MF K100 is closer unguided (0.28×) than the survivor manages *with* projection (0.49×). The result could move if they were scored.
- ⚠️ **Geometry not matched** at the top of Stage 2 — FM/Diffusion have no tightened cells
- ⚠️ Diffusion K100/K20 pair is **checkpoint-confounded**; MF and FM pairs are clean inference-only contrasts

> **Status: ✅ Established as "sole survivor" on train pool.** The funnel structure is robust — but the two arms that might have challenged it were killed by the 24h wall, not by the data.

---

## Headline 4 — UAV Task: It Works, But Needs Refinement 🟡

> **Source:** [outdated_SNAPSHOT_20260825_uav_mix_env_status_PILOT](outdated_SNAPSHOT_20260825_uav_mix_env_status_PILOT.md)

### What is built

| component | detail | status |
|---|---|---|
| **Environment** | MuJoCo Skydio X2 quadrotor, 3 scenes (corridor, pillars, s_curve) | ✅ |
| **Low-level controller** | **CascadedPID** — Lee/Mellinger SO(3) cascaded PID (`flight_controller.py`): position PD → attitude SO(3) PD → motor allocation. Fully custom, zero hardcoded geometry. | ✅ |
| **MPC tracker** | **MJPC Predictive Sampling** — DeepMind's `predictive_sampling.py`, a pure Python/JAX GPU-vectorized sampling-based MPC planner (`mjpc_tracker.py`). Replaced the original gRPC C++ `agent_server` binary. | ✅ |
| **Constraint projection** | Same DPCC projector as the robot tasks, adapted for 9-D UAV state | ✅ |
| **Engines** | `fm` (3.96 M UNet), `mf` (3.97 M UNet), `af` (10.00 M SiT ⚠️) | ✅ |

vs the old robot IK pipeline: the UAV uses **thrust-level control** through the CascadedPID, not inverse kinematics. The MJPC Predictive Sampling replaces the role of a trajectory optimizer.

### Corridor: solved

Six cells reach **10/10 constraint-clean**:
- `fm` K10/K20 (all projectors) — **cheapest clean: 112 ms**
- `af` K5 (all DPCC projectors) — **cheapest 8/10: 34 ms**
- `mf` K10 (HardFlow arms)

### Low-K story: MeanFlow wins at K ≤ 2, loses at K ≥ 5

| K | `mf` vs `fm` (matched UNet) | `af` vs `fm` ⚠️ (SiT, 2.53× params) |
|---|---|---|
| 1 | **W7 L0** T4 | **W8 L0** T2 |
| 2 | **W6 L0** T5 | **W9 L0** T1 |
| 5 | W0 **L10** T1 | W4 L1 T5 |
| 10 | W0 **L8** T3 | — |
| 20 | W0 **L7** T0 | — |

**The general model works** — projection to avoid constraints is effective. But in such a dynamic case, the UAV **loses control extremely easily**:

- 🔴 **`mf`'s unprojected field diverges** — leaves the arena in 7–10/10 rollouts at K ≥ 5. Its wins sit entirely on the selector and the projection, not the generator.
- 🔴 **The constraint benchmark is not a constraint benchmark yet** — `geo_free` and `bounds_free` cost nothing at working operating points. Only the dynamics class is load-bearing.

### Scene analysis: why **pillars** 🎯

| scene | verdict |
|---|---|
| **corridor** | ❌ **Too simple.** Obstacles not binding. Currently a trackability task wearing a constraint task's clothes. |
| **s_curve** | ❌ **Too hard.** 90° + 90° turns make the UAV lose control super easily. **0/3 CF on every engine, every projector** at n=3. Sequential sharp turns are beyond what the current CascadedPID + generative pipeline can track. |
| **pillars** 🎯 | ✅ **Right middle ground.** Obstacles actually in the flight path, gentle enough to be solvable, hard enough that geometry matters. Will use this scene to prove the **MF > FM > DPCC** hierarchy on the UAV platform. |

> **Status: 🟡 Under construction.** Corridor solved, env pipeline validated, both controllers work. But: one seed, one usable scene, no diffusion baseline, constraint geometry not binding. **Pillars is the next proving ground.**

---

## Headline 5 — MPC Candidate Fan B=4 May Not Be Necessary; B=1 Is Statistically Good ⚠️

> **Source:** [DA_20260827_mpc_candidate_fan_avoiding](DA_20260827_mpc_candidate_fan_avoiding.md) · Full analysis: [`DA_20260827_mpc1_full_seeds_state_avoiding.md`](../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md)

### The finding

The MPC candidate fan (`FMPCC_MPC_BATCH`, default B=4) draws B trajectories per replan and executes one via a selection rule. The fan parity study (5 seeds × 3 scenarios × 2 trials = **30 episodes per arm**, paired at 15 blocks) shows that **dropping B from 4 → 1 is statistically harmless — or beneficial — for the flow-based engines**, and that the fan's entire cost sits in serial CPU projection solves, not the generator.

### Three mechanistic findings

| # | Finding | Strength |
|---|---|---|
| **F1** | Fan scales **only the projector** — 3.2–4.4× on projection, generator flat ≤ 2%. End-to-end: DPCC 1.86–1.89×, AlphaFlow 1.39–1.51×, **MeanFlow 1.29–1.33×**. | tightly measured |
| **F2** | Safety effect **changes sign by model**. DPCC untightened loses (20/30 → 7/30, *p*=0.016). **AlphaFlow gains** — `-c-tightened` 6/30 → **30/30** (*p*=0.0005, survives Bonferroni) with 113 fewer steps. | resolved, 4 arms |
| **F3** | At B=1 the `-r`/`-c`/`-t` selection rules are **bit-identical** in all 15 blocks, every generation. Running all three = **3× wasted projection compute**. | exact |

> *"The fan is free on the generator and linear on the projector. Every millisecond the fan costs is a serial CPU solve — the generator batch is not the problem and is not the thing to change."*
> — [DA_20260827_mpc_candidate_fan_avoiding.md](DA_20260827_mpc_candidate_fan_avoiding.md), §2.1

### For the flagship `mf_unet`: B=1 is flat-or-better at the tightened operating point

The flagship already wins at fan 4 on 300 episodes a side (§10.1 of the full analysis):

| | S&C | steps | ms/step | vs DPCC target |
|---|---:|---:|---:|---:|
| DA target — DPCC K20 `dpcc-c-tightened` | 0.983 | 69.0 | 564 | — |
| **`mf_unet` K1** `dpcc-t-tightened` fan 4 | **0.993** | **61.0** | **18.1** | **31×** |
| **`mf_unet` K2** `dpcc-t-tightened` fan 4 | **0.993** | **60.4** | **27.1** | **21×** |

At K2 fan 1 (seed 6, 6 episodes): all three tightened arms hit **1.000 S&C** at 63.0 steps, 20.9 ms — flat-or-better. At K1 fan 1: **zero runs exist yet**.

> *"The flagship already wins at fan 4 on 300 episodes a side. The fan question does not decide the paper claim. It is margin on a claim already won."*
> — [DA_20260827_mpc1_full_seeds_state_avoiding.md](../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md), §10.1

### The better option: parallelise, don't shrink B

The four SLSQP solves per replan are independent (same constraints, different `x0`) but run **serially in a Python `for` loop** on one CPU core. Parallelising would give **B=4 the B=1 latency with no safety exposure**:

| `mf_unet` | total today (B=4 serial) | **B=4 parallelised** | B=1 today |
|---|---:|---:|---:|
| **K1** | 18.1 ms | **≈ 11.7 ms** | ~11.7 ms |
| K2 | 27.1 ms | **≈ 20.8 ms** | 20.9 ms (measured) |

> *"A parallelised B=4 costs what B=1 costs. The second strictly dominates — it also removes the -c stall exposure rather than trading against it."*
> — [DA_20260827_mpc1_full_seeds_state_avoiding.md](../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md), §10.5

### Caveats

- ⚠️ **Fan-1 evidence is 50× thinner** — 300 episodes at fan 4 vs 6 (K2) or 0 (K1) at fan 1
- ⚠️ **DPCC tightened is unresolved, not null** — −2/30, *p*=0.50, CI [−5, 0]; design MDE ≈ 8/30 at 80% power
- ⚠️ **2 of 4 generations excluded** — MeanFlow and FMv3ODE seed sets split across incompatible configs
- ⚠️ **Single task** — `avoiding-d3il` only; F2's sign may not transfer

> **Status: ⚠️ Directional.** For the flagship, B=1 is likely harmless at the tightened operating point, but the fan-1 evidence is too thin (6 episodes at K2, 0 at K1) to bank. The robust recommendation is: **parallelise the projector** (code change, not an experiment) to get B=1 speed at B=4 safety. Run `mf_unet` K1 fan 1 at 20 trials only if the parallelisation is not done.

---

## Summary Scorecard

*Live board — rewritten 2026-09-07. `H<n>` names the headline carrying the claim.*

| task | claim | evidence | blocking item |
|---|---|---|---|
| **Avoiding** (state) | **MF-UNet + HardFlow-SLSQP ≻ Target: 16 % fewer steps, 7.4× cheaper, equal S&C, K=3** | 🟢 **Pareto, architecture-matched** (H9) | ⚠️ 24 rollouts, no seed 6 → **run queue #1** |
| **Avoiding** (state) | HardFlow does *more* NLP solves and costs **0.50–0.58× DPCC** at K ≥ 3 | 🟢 systematic, survives the small n (H9) | export `solve_ms` to measure, not infer |
| **Avoiding** (state) | MF-UNet ≫ DPCC (30×), unguided | ✅ Strong (5 seeds, n=20, Pareto) (H1) | DPCC K ∈ {1,2,5} at n=20 |
| **Avoiding** (state) | FM ≫ DPCC (21×) | ✅ Strong (5 seeds, n=20) (H1) | — |
| **Avoiding** (state) | HF-SLSQP `-t` ≻ DPCC at K20, matched threshold | 🟢 Pareto, n = 6 seed 6 (H6a) | seeds 7–10 |
| **Avoiding** (state) | α-Flow-UNet ≱ MeanFlow with α live | 🔴 **negative, clean** (`Report_20260903_AF_UNet`) | seed 6 only |
| **Avoiding** (state) | MPC fan B=4 not needed; B=1 statistically good | ⚠️ Directional (30 ep. fan 4, 6 ep. fan 1) (H5) | K1 fan 1 @ 20 trials; parallelise projector |
| **Visual Aligning** | **`mf` is the flagship — K=20, T=0.2, `hardflow_sls-r`** | 🟢 **closed** (H10) | seed 6 only |
| **Visual Aligning** | `mf` ≻ `diffusion` baseline | 🟢 −0.3744 m, **0/10, p = 0.0020** (H10) | — |
| **Visual Aligning** | `mf` ≻ naive `fm` | 🟢 +0.216 m, **9/0, p = 0.0039** (H10) | — |
| **Visual Aligning** | ~~`fm` > `diffusion`~~ | 🔴 **fails** — 0.0055 m inside a 0.4 m band (H10) | *do not quote; not a required rung* |
| **Visual Aligning** | α-Flow ≥ MeanFlow | ⛔ **closed** — ties at K=2, worse at K=20, better nowhere (H10) | *parked at user request* |
| **Visual Aligning** | HF ≻ DPCC **on latency at K=10** | 🟢 −325 ms/step, **0/9, p = 0.0039** (H10) | — |
| **Visual Aligning** | ~~HF safer than DPCC at K=20~~ | 🔴 **one discordant rollout, p = 1.0000** (H10) | *do not quote* |
| **Visual Aligning** | constraint comparison vs the baseline | 🟡 **incomplete** — `diffusion` has no tightened cell (H10) | **tightened `diffusion` K=20** |
| **All tasks** | `-r`/`-c`/`-t` do not earn their compute | 🟢 exact at B=1; `-c` known-bad at B=4 (H7, H9) | default rule is **task-dependent** — `-t` on avoiding, `-r` on V_A |
| **UAV corridor** | MF > FM at K ≤ 2 | 🟡 Directional (1 seed, n=10) | Multi-seed, diffusion baseline |
| **UAV pillars** | Fix_16 confirmed; scene still unrankable | 🟢 A/B clean · 🔴 S&C 0/2876 (H8) | `*_hg` re-run, all 3 engines |

---

## Next Key Headlines

> 📍 **Moved.** The prioritised queue is now the single
> [**Run queue**](#run-queue) at the end of this file, kept beside the live evidence board so the
> two cannot drift apart. Four items that stood here on 2026-09-05 have since been **answered or
> closed** — the K=100 V_A projection, the constant-α training run, α-Flow at MF's flagship, and the
> `af_unet` re-entry — and are listed there as dropped.



---

# Addendum 2026-09-05 — evidence against the thesis target

> **Written against** [`logs_in_develop/Writing/Working_Space/TARGET_20260905_thesis_claim_ladder.md`](../../logs_in_develop/Writing/Working_Space/TARGET_20260905_thesis_claim_ladder.md).
> That file holds **goals only**; this section holds the evidence for and against them.
> Nothing below was independently computed — every number is quoted from a curated DA, cited inline.

---

## Headline 6 — HardFlow vs DPCC: the threshold was the confound, not the solver ✅⚠️

> **Sources:** [`Gen14/DA_20260901_…flagship_K20_T0.2_dpcc_vs_hardflow`](../../logs_in_develop/Gen14/DA_20260901_Gen14_flagship_K20_T0.2_dpcc_vs_hardflow.md) §2.1 ·
> [`aggregated_hf_nlp_backend/DA_20260830_ipopt_vs_slsqp_fmv3ode_K10_K20`](../../logs_in_develop/aggregated_hf_nlp_backend/DA_20260830_ipopt_vs_slsqp_fmv3ode_K10_K20.md) §5.4

### The correction

The standing answer in this directory — [`outdated_RESPONSE_20260826_did_HardFlow_ever_beat_DPCC`](outdated_RESPONSE_20260826_did_HardFlow_ever_beat_DPCC.md) §Q3 ("**No.** DPCC wins every axis at every K") and [`outdated_SNAPSHOT_20260823_visual_aligning_env_status`](outdated_SNAPSHOT_20260823_visual_aligning_env_status.md) §5 ("**No**") — **is superseded on both tasks.** Those runs carried two confounds that have since been removed:

| confound | old setting | fixed in | effect |
|---|---|---|---|
| **NLP backend** | IPOPT (`hardflow_new-*`) | job 25222 → SLSQP | HF/DPCC cost **4.07–4.75× → 1.05–1.21×** |
| **Activation threshold** | HF `A=1.0` vs DPCC `thr=0.5` — HF did ~2× the projection work | job 25237 → `A=0.5` | comparison becomes like-for-like |
| **Projection threshold (VA)** | only ever `K=2` at `thr=0.5` → `n_active=1`, **`n_genuine=0`** | job 25247 → `K=20, T=0.2` (`n_genuine=3`) | the old VA rows were **degenerate — not HardFlow at all**, but sample-then-project; 25247 is the first genuine HF cell on this task |

Gen14 U7 had already flagged the third one: *"the comparison the benchmark hierarchy actually asks for (a lower projection threshold) has never been run."* Running it flipped the direction. **Present this as a threshold being located, not a result being reversed.**

### 6a — Avoiding (state): HardFlow-SLSQP Pareto-dominates at K=20 ✅

Job 25237, `A=0.5` matched, K20, seed 6, n_trials 2, 3 geometries. `hf_n_genuine = 9` → ✅ genuine.

| arm | S&C | total_viol | steps | s/step |
|---|---:|---:|---:|---:|
| DPCC `dpcc-c-tightened` | 100 % | 0.00000 | 62.2 | 0.475 |
| **HF-SLSQP `-t-tightened`** | **100 %** | **0.00000** | **61.0** | **0.343 (0.72×)** |
| HF-SLSQP `-r-tightened` | 100 % | 0.00000 | 68.2 | 0.338 |
| HF-SLSQP `-c-tightened` | 100 % | 0.00000 | 103.0 | 0.334 |

**Fewer steps *and* 0.72× wall-clock at equal S&C and equal violations — strict Pareto dominance, measured not extrapolated.** At K=10 every tightened arm is 0.82–0.84× DPCC's time but uses more steps: trade-off, not a win.

⚠️ n = 6 per cell (seed 6 only) · ⚠️ SLSQP `-t` logged **15 non-convergences** at `A=0.5` (both backends degrade at the lower threshold) · ⚠️ this is the **`-t` arm only**; `-r` and `-c` are non-dominated.

### 6b — Visual aligning: direction right, power missing ⚠️

Job 25247, `mf`, K=20, T=0.2, `combined_5-tightened`, fan 4 **both arms**, `n_genuine = 3` → ✅ `HF_OK`. Both arms fire at the same four ODE steps, same solver dof, same feasible set — **they differ only in *when* the constraint is applied.**

| rule | arm | dist (m) | 0-viol | viol | ms/step |
|---|---|---:|---:|---:|---:|
| `-r` | DPCC | 0.2001 | 0.900 | 2.70 | 265.9 |
| `-r` | **HF-SLSQP** | 0.2108 | **1.000** | **0.00** | 275.3 |
| `-c` | DPCC | 0.3059 | 0.900 | 4.30 | 325.3 |
| `-c` | **HF-SLSQP** | 0.2863 | **1.000** | **0.00** | **282.3** |
| `-t` | DPCC | 0.2398 | 0.800 | 1.50 | 323.2 |
| `-t` | **HF-SLSQP** | 0.2555 | **0.900** | **0.60** | **301.5** |

HF ≥ DPCC on constraints **3/3**, strictly better **2/3**, cheaper **2/3**; distance a wash.

🔴 **Nothing is significant, and it structurally cannot be at this n.** Pooled n = 30 paired rollouts: 0-viol 0.967 vs 0.867 (McNemar 3/0, *p* = 0.250); violations 0.200 vs 2.833 (sign 4/0/26, *p* = 0.125); 286.4 vs 304.8 ms (*p* = 0.248). **26 of 30 pairs are tied at zero violations, so *p* = 0.125 is the floor a 4/0 split can reach** — a perfect result at this n could not clear 0.05. **Power problem, not a null result.**

> The honest sentence until more seeds land: *"matched on distance, directionally ahead on constraints and cost, n too small."*

#### 🔴 Matched-pair dominance ≠ best-vs-best dominance

Read rule-by-rule, **exactly one pair is a clean Pareto win: `-c`**, where HF is better on all four axes (0.2863 vs 0.3059 m, 1.000 vs 0.900, 0.00 vs 4.30, 282.3 vs 325.3 ms). `-r` and `-t` are trade-offs — HF ahead on constraints, behind on distance.

**But `-c` is DPCC's worst arm.** Under this directory's own reporting rule (*model-vs-model uses each side's own best projector, always named*), the comparison is HF `-c` vs **DPCC `-r`** (0.2001 m, 0.900, 265.9 ms) — where HF is better on constraints and **worse on distance and cost**. So:

> **Non-dominated, not dominant.** Do not quote the `-c` pair as "HardFlow Pareto-dominates on visual aligning" without naming that it is the rule-matched comparison against DPCC's weakest selection rule.

**The mechanism, and it is consistent across two independent batches.** `-c` picks the candidate needing the *least* projection correction — the worst possible rule for safety — and HF's in-loop enforcement repairs exactly that deficit. The 08-23 DA found the same asymmetry: its single result surviving multiple-comparison correction was also `-c` on tightened `mf` (−0.067 m, dz = −0.69, *p* = 1e-4). **HardFlow's advantage is largest where the DPCC selection rule is weakest.**

⚠️ **Tension with Goal C (Headline 7): deleting `-c` also deletes HardFlow's best showing.** Decide that deliberately. The defensible reading is that both point the same way — `-c` is a bad rule, and needing an in-loop NLP to rescue it is an argument against the rule, not for the NLP.

**One unambiguous HF win exists:** latency on `mf` at K=10 — 206.3 vs 491.8 ms, 26/1, ***p* < 0.001**. It is a cost claim, not a safety one.

**Blocking item: seeds 7–10 for arm C on `mf` at K=10 and K=20.** Already the standing top priority in the Gen14 DA; it is the single run between this and a defensible RQ3 answer.

---

## Headline 7 — The `-r`/`-c`/`-t` selection rules do not earn their compute ⚠️

> **Sources:** [`DA_20260827_mpc_candidate_fan_avoiding`](DA_20260827_mpc_candidate_fan_avoiding.md) F3 ·
> [`HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding`](../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md) §10 ·
> plus the two projector runs in Headline 6.
> **Target:** Goal C — remove the candidate-selection machinery from the final deliverable.

### 7.1 At fan 1 the three rules are the same computation — exact, not statistical

F3: at `B=1` the `-r`/`-c`/`-t` arms are **bit-identical** — same S&C *and* same steps, **all 15 blocks, every generation**. There is only one candidate; every selection rule returns slot 0.

> **Running all three at `B=1` is exactly 3× wasted projection compute, with a provable zero information gain.** This half of Goal C needs no further experiment — it is closed.

### 7.2 🔴 A consequence nobody has drawn: the AlphaFlow "`-c` win" is a fan effect, not a rule effect

F2 reports AlphaFlow `-c-tightened` going **6/30 → 30/30** (*p* = 0.0005, survives Bonferroni) with 113 fewer steps when the fan drops 4 → 1. But by 7.1, **at `B=1` the label `-c` denotes no selection at all.** The improvement therefore cannot be attributed to the min-cost rule; it is *"AlphaFlow does better when the fan stops discarding its trajectories."* Re-read as: **`-c` at `B=4` was destroying 24/30 AlphaFlow episodes that survive without it.** That is evidence *for* deleting `-c`, not for keeping it.

### 7.3 At fan 4 the rules diverge widely — and `-c` is the consistent loser

Same run, same engine, same everything except the rule:

| task / arm | `-r` | `-c` | `-t` |
|---|---:|---:|---:|
| avoiding, HF-SLSQP K20 tightened — **steps** | 68.2 | **103.0** 🔴 | **61.0** ✅ |
| VA flagship, DPCC K20 T0.2 — **dist (m)** | **0.2001** ✅ | **0.3059** 🔴 | 0.2398 |
| VA flagship, DPCC K20 T0.2 — **ms/step** | **265.9** ✅ | 325.3 | 323.2 |

`-c` is worst on steps by **69 %** on avoiding and worst on distance **and** cost on visual aligning. `-t` wins where the flagship actually operates: `mf_unet` K1/K2 both ship `dpcc-t-tightened` (0.993 S&C, 61.0/60.4 steps, 18.1/27.1 ms), and the only HF Pareto win in the project is `-t`.

### 7.4 The cost is real and structural

The projector is **serial** — `diffuser/sampling/projection.py:132` runs one SLSQP per candidate in a Python `for` loop; `parallelize` (`projection.py:9,20`) is **assigned and never read**. Cost is `S + B·P`, linear in `B`, with all of it on one CPU core. The fan is free on the generator (batched GPU forward, invariant to within 2 %).

### 7.5 Verdict and recommendation

| claim | status |
|---|---|
| At `B=1`, running three rules is pure waste | ✅ **proven exactly** — closed |
| `-c` never earns its cost | ⚠️ **strongly directional** — worst on 3/3 axes across two tasks and two projector arms, plus 7.2; no cell found where it leads |
| `-t` is the right single default | ⚠️ directional — wins the flagship and the HF Pareto cell |
| `-r` is redundant with `-t` | ⬜ open — `-r` leads on VA distance and cost |

> ### Recommendation for the final deliverable
> **Ship one rule (`-t`) as the default. Disable `-c`. Keep `-r` as the documented null control.**
> Handle the fan by **parallelising the projector** — a code change that gives `B=4` safety at `B=1` latency (`mf_unet` K1: 18.1 ms → ≈11.7 ms) — **not** by shrinking `B`.
>
> ⚠️ **`-r` stays in the thesis as an ablation even if it leaves the shipped default.** Deleting the random control from the *deliverable* is an engineering decision; deleting it from the *evidence* would remove the baseline that makes 7.3 interpretable.

**To close 7.3 and 7.4 into a hard claim:** the rules need a paired A/B at `B=4` across seeds 7–10 on both tasks. Cheap — eval-only, no retraining — and it rides along with the Headline 6 blocking run.

---

## Headline 8 — UAV `pillars`: the engine was fixed, the scene still is not 🟢🔴

> **Sources:** [`Gen15/DA/DA_20260903_fix16_AB_mf_pillars`](../../logs_in_develop/Gen15/DA/DA_20260903_fix16_AB_mf_pillars.md) ·
> [`Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af`](../../logs_in_develop/Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af.md) ·
> [`Gen15/U7/CHANGELOG_20260904_honest_geometry…`](../../logs_in_develop/Gen15/U7/CHANGELOG_20260904_honest_geometry_and_slack_gate.md)

**Two results that must not be conflated.** "Pillars is good now" is true of the *policy* and false of the *benchmark*.

### 8a — 🟢 Fix_16 passed a falsifiable prediction, cleanly

Jobs 25316–25321, one git rev, `mf`, seed 6, K ∈ {1,2,5}, `SAFE_EPS_MODE=scaled` vs `legacy`:

| | pre-fix (`legacy`) | post-fix (`scaled`) |
|---|---|---|
| unguided (`diffuser`) divergence abort, every K | **100 %** | **0 %** |
| goal distance, K1 / K2 / K5 | 6.50 / 6.46 / 6.49 m | **0.62 / 0.66 / 0.36 m** |
| K5 `dpcc-r` abort | 100 % | 20 % |
| K1 `dpcc-t` abort | 80 % | 0 % |

This is the strongest methodological result in the UAV line: a mechanism was proposed in a study, converted into a falsifiable prediction, and confirmed — with **`legacy` reproducing the pre-fix numbers bit-for-bit** (clean single-variable A/B) and the already-healthy `*-geo_free` arms **not moving** (internal control). ⚠️ Cost: projection time rises **1.2–1.6×**.

### 8b — 🔴 It changes nothing about the ladder

**Success + constraints = 0 / 2876 rollouts** — every engine, every K, both arms; collision-free completion 4 / 2876. Unchanged from the 08-30 batch (0 / 1707, 2 collision-free). **No engine ranking can be drawn from this scene**, and there is **no `diffusion` arm for `pillars` at all**, so the pinned DA target does not even exist here.

The cause is geometric, not generative, and is now measured: the tightest `pillars` planning channel is **6 cm** of slack against a **34 cm** median tracking error; `corridor` has **0.000 m**. **S&C was bounded near zero before any engine ran.** Compounding it: the projector **never plans in z** — every wall and pillar binds `(x,y)` only, so each obstacle is an infinite cylinder it cannot route over or under.

⚠️ **`fm` and `af` were never re-run with Fix_16.** Every cross-engine number in that batch is fixed-`mf` vs unfixed `fm`/`af` — **not an engine comparison.** Do not quote it as one.

### 8c — What this means for the thesis

Fix_16 and the honest-geometry work belong in **methodology and threats-to-validity**, where they are a genuine strength (a measured, self-diagnosed benchmark defect). Until the `*_hg` scenes are re-run with all three engines fixed, **UAV cannot carry Goal A**, and the target's UAV kill criterion is live.

---

# Addendum 2026-09-07 — two results that close two questions

> Written from
> [`Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md`](Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md)
> and
> [`Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md`](../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md).
> Nothing below was independently computed; every number is quoted, cited inline.

---

## Headline 9 — HardFlow at its own floor Pareto-dominates the Target ✅⚠️

> **Source:** [`DA_20260906_hf_minK_mfunet_A1_K2_K3_K5`](Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md) ·
> job `25444`, `avoiding-d3il`, MeanFlow U-Net **4.0 M**, `A = 1.0`, K ∈ {2, 3, 5}, B4 parity, SLSQP

### The claim

**This is the strongest `avoiding-d3il` result in the corpus.** Against the pinned Target —
DPCC K20 / aw10 / T0.5, `dpcc-c-tightened`, S&C 1.000 / 70.13 steps / 0.5534 s:

| row | backbone | params | K | tier | S&C | steps | t (s) |
|---|---|---|---|---|---|---|---|
| **Target** `dpcc-c-tightened` | U-Net | (DPCC) | 20 | — | 1.000 | 70.13 | 0.5534 |
| **MF-UNet `hardflow_sls-t-tightened`** | U-Net | **4.0 M** | **3** | ✅ | **1.000** | **58.88** | **0.0745** |
| MF-UNet `hardflow_sls-t-tightened` | U-Net | 4.0 M | 5 | ✅ | 1.000 | 59.00 | 0.1304 |
| MF-UNet `hardflow_sls-t-tightened` | U-Net | 4.0 M | 2 | ⚠️ THIN | 1.000 | 58.62 | 0.0468 |
| MF-UNet `dpcc-t-tightened` (arm B, same job) | U-Net | 4.0 M | 2 | — | 0.958 | 59.62 | 0.0270 |

**Pareto-dominant, not a trade-off:** S&C equal, **16 % fewer steps**, **7.4× lower `avg_time`**,
both axes improving and neither degrading. **Architecture-matched** — 4.0 M U-Net against the U-Net
baseline — so this is the strong form of the claim, not a backbone confound. It is the **first time
in this corpus a `hardflow` row on a ✅ genuine tier has cleared the Target**, and the arm-B row at
the same K does *not* clear it (0.958), so the guidance arm is load-bearing here.

### 🔑 The mechanism: HardFlow does *more* NLP solves and costs *less*

The striking result is cost, not safety. Same job, same checkpoint, same solver, same fan:

| K | arm B solves/plan | arm B t | arm C solves/plan | arm C t | ratio |
|---|---|---|---|---|---|
| 2 | 1 | 0.0270 | 2 | 0.0468 | 1.73× *slower* |
| 3 | 2 | 0.1478 | 3 | **0.0745** | **0.50×** |
| 5 | 3 | 0.2268 | 5 | **0.1304** | **0.58×** |

Arm C's cost is **flat per solve** (~28 ms, fit `t ≈ 0.0279·K − 0.009`); arm B's *rises* with K —
its second and third solves cost 80–120 ms each against a first solve of ≤27 ms. The explanation is
HardFlow's stated mechanism showing up as a cost effect: **DPCC projects the off-manifold flow
iterate, HardFlow projects the predicted clean endpoint `x1_ref`**, which is near-feasible at every
τ, so SLSQP converges in a few iterations instead of many.

**The crossover is K = 3** — which is also the floor for a citable HardFlow row. *The K at which
HardFlow becomes attributable and the K at which it starts paying for itself are the same K.*

### The floor, confirmed at run time

`n_genuine = max(K − int((1−A)·K), 1) − 1`. The eval printed `[hardflow][THIN] K=2 A=1.0:
n_active=2, n_genuine=1` and ran `n_genuine = 2 / 4` at K = 3 / 5; `nlp_solves_total` scaled
954 : 1437 : 2400 (1 : 1.51 : 2.52 vs the predicted 1 : 1.5 : 2.5). **K = 1 is structurally
impossible for arm C at every `A`** — the only step is terminal, so there is no successor to react
to the correction. No knob fixes it.

### Two secondary findings from the same job

- **Tightening is a bigger lever than the choice of arm.** `dpcc-t` 0.458 → `-tightened` 0.958;
  `hardflow_sls-t` 0.583 → `-tightened` 1.000. **No untightened row clears 0.75 anywhere in the
  run.** Any arm comparison on untightened geometry is measuring the margin, not the arm.
- **Unguided is unusable at low K here.** `diffuser` S&C = 0.042 at K=2 and **0.000** at K=3 and 5,
  at 1.000 success and 15–17 executed violations — the model reaches the goal and walks through the
  halfspace every time. On this task projection is not a refinement, it is the whole constraint story.

### ⚠️ Caveats — read before citing

1. **24 rollouts per row** (4 seeds × 3 geometries × `n_trials = 2`). The S&C gap carrying the
   headline (1.000 vs 0.917) is **2 rollouts** and is not significant. The **time** differences are
   systematic and survive; the **safety** differences do not.
2. **Not seed-matched** — seed 6 was never run (yaml said 7–10), the Target pools 5 seeds.
3. `-c` at B=4 is the **known-bad** arm (49 % timeouts, `logs_in_develop/HF_Batch_Parity/`); its
   114.6-step blow-up at K=5 is that pathology, not a HardFlow property. Reported, never cited.
4. **No α-Flow arm and no `A = 0.5` reference at matched K** — so how much of arm C's advantage is
   the *activation schedule* rather than HardFlow itself is still open.
5. The run tag says `s6` and the run is not seed 6. Folder names are misleading; no number affected.

> **Status: 🟢 the result to power next.** One run — 5 seeds, `n_trials = 20`, corrected tag —
> stands between this and a citable headline claim.

---

## Headline 10 — Visual aligning is closed; MeanFlow is the flagship ✅

> **Source:** [`Gen14/CLOSURE_20260907`](../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md) ·
> corpus `batch_va2_20260907_141036` — 673 config rows, 14 102 rollout rows, seed 6, 10 paired contexts

**Supersedes [Headline 3](#headline-3--visual-aligning-meanflow-k2-is-the-sole-survivor-) (operating point) and
[Headline 2](#headline-2--α-flow-vs-meanflow-same-engine-different-curriculum-) (verdict).**

### 10.1 The verdict table

| engine | status | operating point | basis |
|---|---|---|---|
| **`mf`** | ✅ **flagship** | K=20, T=0.2, arm C `hardflow_sls-r` | only engine separating from `diffusion` (0/10, p = 0.0020); only zero-violation 1.000 **unfrozen** |
| `af`-U-Net | ⛔ **closed** | — | ties `mf` at K=2, loses at K=20, flat in K, no arm reaching zv 1.000 |
| `af`-SiT | ⛔ **abandoned** | — | 9.4 M vs 4.0 M **and** `ae0.0` ⇒ MeanFlow mislabelled |
| `fm` | ⛔ **excluded** | — | four gates, three 🔴 |
| `diffusion` | 🟡 **partial** | K=20 / K=100, untightened only | **tightened K=20 still pending**; K=100 competitive but 8.0× the cost |

### 10.2 🔴 The ladder was wrong — the true shape is `mf` ≫ {`af`, `fm`, `diffusion`}

Unguided plan, untightened `combined_5`, initial distance **0.4530 m**:

| rank | engine | K | median | MIN | untouched | ms/step | vs `diffusion` K=20 |
|---|---|---|---|---|---|---|---|
| 1 | **`mf`** | 20 | **0.0741** | 0.0278 | **0/10** | 190.5 | **−0.3744 · 0/10 · p = 0.0020** |
| 2 | `af`/α=0.2 | 20 | 0.3918 | 0.0110 | 2/10 | 191.3 | −0.1389 · 5/5 · p = 1.0000 |
| 3 | `fm` | 20 | 0.4085 | 0.0394 | 4/10 | 295.8 | −0.1411 · 4/5 · p = 1.0000 |
| 4 | `diffusion` | 20 | 0.4140 | 0.0280 | 10/30 | 298.3 | — |
| 5 | *d3il baseline* | — | *0.4152* | *0.0091* | *1712/3884 (44 %)* | *22.7* | — |

**Paired on the same contexts, `af`, `fm` and `diffusion` are mutually indistinguishable** — every
pairwise p is 0.51–1.00. They are one cluster at 0.39–0.42 m, i.e. *"barely moved the box"* from a
0.4530 m start, sitting alongside the d3il baseline. **Only MeanFlow separates, categorically.**

> 🔴 **Do not quote `fm > diffusion`.** It is a **0.0055 m gap inside a ~0.4 m band**. The thesis
> ladder never required it — it requires `mf` ≻ baseline and `mf` ≻ naive FM, and both hold at
> p ≤ 0.0039. Finding this before it went into the thesis is a save, not a loss.

### 10.3 The three required rungs all hold

| # | required claim | status | evidence |
|---|---|---|---|
| 1 | MeanFlow ≻ the diffusion-DPCC baseline *(THE baseline)* | ✅ **decisive** | −0.3744 m, **0/10, p = 0.0020**, at 0.64× the per-step cost |
| 2 | MeanFlow ≻ naive Flow Matching | ✅ **holds** | +0.216 m, **9/0, p = 0.0039**; `fm` frozen 5/10 vs `mf` 1/10 |
| 3 | HardFlow ≻ the DPCC projector | ✅ **holds — on latency at K=10** | −324.96 ms/step, **0/9, p = 0.0039** (`-r`) |
| — | *α-Flow ≻ MeanFlow* | ❌ fails | **never a required rung** — an optional extra arm |

### 10.4 ⚠️ How to state the HardFlow claim on V_A — and how not to

| | K = 10 (T=0.4) | K = 20 (T=0.2) |
|---|---|---|
| latency vs DPCC | 🟢 **−324.96 ms/step, 0/9, p = 0.0039** | ⚪ parity (all p = 0.18–1.00) |
| zero-violation vs DPCC | n.s. | 🟢 10/10 vs 9/10 — **one discordant rollout, p = 1.0000** |
| distance cost | none | none |

> ✅ **Citable:** *at K=10, HardFlow-SLSQP delivers DPCC-level safety and distance at ~325 ms/step
> less — a 0/9 sweep at p = 0.0039.*
>
> 🔴 **Not citable:** *"HardFlow is safer than DPCC at K=20."* That edge is one rollout. What **is**
> true at K=20 is that `hardflow_sls-r`/`-c` are **the only arms in the entire V_A corpus reaching
> zero-violation 1.000 with 0.00 violations while still moving the box** — `dpcc-c-dt4p0` also
> reaches 1.000, but by freezing 10/10, which is disqualified.

### 10.5 The flagship cell

`mf` · `VisualMeanFlow` two-time · `unet` **4.0 M** · FiLM v1 · seed 6 · `mpc4` · H8 · K=20 · T=0.2
(4 projector calls/replan) · arm C on SLSQP, tightened `combined_5-tightened`:

| variant | MIN | median | untouched | **zero-viol** | violations | ms/step |
|---|---|---|---|---|---|---|
| `diffuser` (unguided) | 0.0278 | **0.0902** | 1/10 | 0.20 | 31.60 | 172.5 |
| **`hardflow_sls-r`** ⭐ | 0.0220 | 0.1967 | 2/10 | **1.00** | **0.00** | 275.3 |
| **`hardflow_sls-c`** ⭐ | 0.0241 | 0.3227 | 4/10 | **1.00** | **0.00** | 282.3 |
| `dpcc-r` | 0.0329 | 0.1786 | 2/10 | 0.90 | 2.70 | 265.9 |
| `dpcc-c` | 0.0415 | 0.3328 | 4/10 | 0.90 | 4.30 | 325.3 |

The baseline's projector reaches comparable safety only at **2157.9 ms/step** (`diffusion` +
`dpcc-r`) — **7.8× `hardflow_sls-r`**.

### 10.6 What was gained by closing it

Three findings that did not exist before this drop:

1. **HardFlow's benefit scales with plan constraint error and is free on a high-violation field** —
   −124.70 violations on `af` at p = 0.0078, at **zero distance cost**.
2. **The τ = 0.850 NLP failure is engine-independent** — same non-converged SLSQP solve at call #2,
   24/24 arm-C items, across three objectives. Engine-side is ruled out; constraint-Jacobian
   conditioning is the remaining suspect.
3. **A measured ~0.4 m run-to-run reproducibility floor on projected arms**, which now qualifies
   **every MIN in every V_A DA**. MIN gaps under ~0.1 m on projected arms are noise.

> **A negative result on a well-posed question is a thesis contribution.** "α-Flow, α provably live
> at two settings, architecture-matched on the same 4.0 M U-Net, does not improve on MeanFlow and
> does not scale with NFE — because its target bootstraps the network's own output rather than an
> analytic derivative" is a defensible chapter section.

### ⚠️ Caveats

- **Seed 6 only, 10 contexts.** The whole V_A corpus is single-seed.
- **`diffusion` has no tightened cell**, so the §10.2 ranking is a **plan-quality** ranking; the
  constraint comparison against the baseline is incomplete until tightened `diffusion` K=20 lands.
- The `af` K=20 loss is **weak evidence** — the sign test does not reject (perm p = 0.0469 only).
  α-Flow is closed for being *never better across two environments and flat in K*, not for being
  decisively beaten at one point.

---

## Live evidence board vs. the target (2026-09-07)

🟢 supported · 🟡 partial / underpowered · 🔴 contradicted · ⛔ closed · ⬜ not measured

### Goal A — engine ladder `af_unet ≥ mf > fm > diffusion`

| environment | `mf > fm` | `fm > diffusion` | `af_unet ≥ mf` |
|---|---|---|---|
| `avoiding-d3il` | 🟢 Pareto-dominant, 30× | 🟡 21×, needs NFE-matched restatement | 🔴 **α-live, never better** (H9 source job, `Report_20260903_AF_UNet`) |
| `aligning-d3il-visual` | 🟢 **9/0, p = 0.0039** | 🔴 **fails — 0.0055 m inside a 0.4 m band** (H10) | ⛔ **closed: ties at K=2, worse at K=20** (H10) |
| `aligning-d3il` (3-D state) | ⬜ | ⬜ | ⬜ — **no data at all** |
| `uav-corridor` | 🔴 regime-split: W13 at K1–K2, L25 at K5–K20 | 🟡 | ⬜ `af_unet` landed with Gen15 U6, never ranked |
| `uav-pillars` | ⬜ **unrankable** — S&C 0/2876, no `diffusion` arm (H8) | ⬜ | ⬜ |

🔑 **The top rung of Goal A is now decided negatively, on two environments.** The ladder that
survives is **`mf` ≫ {`af`, `fm`, `diffusion`}** — one winner, one cluster. Rewrite the target's
Goal A to that shape rather than carrying a four-rung ladder the data does not support.

### Goal B — projector ladder

| environment | status |
|---|---|
| `avoiding-d3il`, `A = 1.0`, K ∈ {3, 5}, SLSQP, B4 | 🟢 **HF `-t` Pareto-dominates the Target** — 16 % fewer steps, 7.4× cheaper (H9); ⚠️ 24 rollouts, no seed 6 |
| `avoiding-d3il`, `A = 0.5`, K=20 | 🟢 HF `-t` Pareto-dominates DPCC (n = 6, seed 6) |
| `aligning-d3il-visual`, K=10 / T=0.4 | 🟢 **−325 ms/step, 0/9, p = 0.0039** at equal safety and distance (H10) |
| `aligning-d3il-visual`, K=20 / T=0.2 | 🟡 only arms reaching zv 1.000 unfrozen — but the *margin* over DPCC is one rollout |
| `uav-corridor` | 🟡 win switches on at K ≥ 5; 1 seed, pilot, pre-honest-geometry |

### Goal C — selection machinery

| claim | status |
|---|---|
| `B=1` → three rules identical → 3× waste | 🟢 **exact** |
| `-c` is deletable | 🟢 **strengthened** — known-bad at B=4 (49 % timeouts); 114.6-step blow-up at K=5 (H9) |
| a single default rule | ⚠️ **task-dependent, do not over-generalise** — `-t` is the rule that works at B=4 on `avoiding`; `-r` is the flagship on V_A |

### The four real gaps

1. **H9 is underpowered** — 24 rollouts, no seed 6. One run fixes it and it is the strongest result
   in the corpus. **Highest value per GPU-hour of anything on this list.**
2. **Tightened `diffusion` K=20 on V_A** — the funnel's binding gap. Until it lands there is no
   complete constraint comparison against the DPCC baseline, and the H10 ranking stays
   plan-quality-only.
3. **UAV cannot rank anything** — S&C 0/2876 on `pillars`, no `diffusion` arm, `*_hg` scenes never
   re-run with all three engines. The target's UAV kill criterion is live.
4. **`aligning-d3il` 3-D state has no data** — the modality-transfer argument claims state→visual on
   *aligning*, but aligning was only ever measured visually.

### Run queue

| # | run | closes |
|---|---|---|
| 1 | **Power H9** — MF-UNet `A=1.0`, K ∈ {2,3,5}, seeds 6–10, `n_trials = 20`, corrected tag | gap 1 — the flagship paper number |
| 2 | **Tightened `diffusion` K=20**, V_A | gap 2 |
| 3 | The `A = 0.5` reference ladder at matched K (proposal §4.3) | separates the activation schedule from HardFlow itself |
| 4 | UAV `*_hg` K-sweep, **all three engines with Fix_16**, ≥ 3 seeds | gap 3 — makes UAV rankable at all |
| 5 | Repeat one projected `mf` V_A cell 3× | pins the ~0.4 m reproducibility floor qualifying every MIN |
| 6 | Export `solve_ms` + SLSQP iteration counts per arm; give `dpcc-*` the NLP counters arm C has | turns H9's *inferred* mechanism into a measured one |
| 7 | `aligning-d3il` 3-D state, engine ladder | gap 4 |
| 8 | τ = 0.850 — check constraint-Jacobian conditioning | the one open bug (engine-side already ruled out) |

**Dropped from the queue** (previously listed, now answered or closed): project the K=100 V_A arms ·
constant-α training run · α-Flow at MF's flagship · `af_unet` re-entry at `α_end ∈ {0.4, 0.6}`.

---

# Folded-in user notes

*Kept as an audit trail of what was raised and where it landed.*

**1. "The pillars results are good"** →
[`Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af.md`](../../logs_in_develop/Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af.md)

> ✅ **Folded in → [Headline 8](#headline-8--uav-pillars-the-engine-was-fixed-the-scene-still-is-not-).**
> Split in two, because the halves point opposite ways: **8a** the Fix_16 A/B is a clean
> methodological win (100 % → 0 % divergence abort, 6.50 m → 0.62 m); **8b** the scene is still
> unrankable (S&C 0/2876, no `diffusion` arm).

**2. "HF beat DPCC in d3il avoiding"** →
[`aggregated_hf_nlp_backend/DA_20260830_ipopt_vs_slsqp_fmv3ode_K10_K20.md`](../../logs_in_develop/aggregated_hf_nlp_backend/DA_20260830_ipopt_vs_slsqp_fmv3ode_K10_K20.md)

> ✅ **Confirmed → [Headline 6a](#6a--avoiding-state-hardflow-slsqp-pareto-dominates-at-k20-),
> then extended by [Headline 9](#headline-9--hardflow-at-its-own-floor-pareto-dominates-the-target-).**
> You were right, and it superseded this directory's standing "no" — which is why
> `RESPONSE_20260826` and `AUDIT_20260827` are now `outdated_*`.

**3. "GOOD RESULT — maybe consider using this as a flagship for the paper!!!"** →
[`Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md`](Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md)

> ✅ **Folded in → [Headline 9](#headline-9--hardflow-at-its-own-floor-pareto-dominates-the-target-).**
> **Agreed on the flagship, with one condition.** It is the only row in the corpus that
> Pareto-dominates the Target on an architecture-matched backbone at a ✅ genuine tier — that is
> exactly what a flagship needs. The condition is **power**: 24 rollouts and no seed 6. The *time*
> axis (7.4×) is systematic and will survive; the *safety* axis (1.000 vs 0.917 = 2 rollouts) will
> not. Run queue #1 before it goes in a paper.

**4. "Stop exploring `af_unet` in V_A"** →
[`Gen14/CLOSURE_20260907`](../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md)

> ✅ **Applied → [Headline 10](#headline-10--visual-aligning-is-closed-meanflow-is-the-flagship-).**
> `af`-U-Net is **closed** and `af`-SiT **abandoned**. The CLOSURE §7 leaves one optional re-entry —
> `α_end ∈ {0.4, 0.6}` at K=2, ~24 ms/step — and per this instruction it is **parked, not queued**.
> It is off the run queue above.

---

*All numbers sourced from the curated reports in this directory and the cited `logs_in_develop/`
DAs. No number was independently computed for this notebook. See the individual reports for
methodology, statistics, and reproduction instructions.*
