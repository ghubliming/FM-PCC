# FM-PCC — Key Headlines Notebook

**Last updated:** 2026-09-05 · **Author:** auto-generated from curated reports  
**Status:** 🟡 Two tasks established, one under construction, several open experiments

> ⚠️ **Headlines 1–5 are the 2026-08-29 snapshot. Read the
> [Addendum 2026-09-05](#addendum-2026-09-05--evidence-against-the-thesis-target) before quoting
> anything about HardFlow or the `-r`/`-c`/`-t` selection rules** — Headline 6 supersedes this
> directory's earlier "HardFlow never beat DPCC" answer, and Headline 7 refines Headline 5.

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
| 2026-09-05 | **Thesis target written**; evidence board + Headlines 6–8 (this addendum) |

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

> **Source:** [Report_20260829_VA_funnel](Report_20260829_VA_funnel/README.md) (updated)

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

> **Source:** [SNAPSHOT_20260825_uav_mix_env_status_PILOT](SNAPSHOT_20260825_uav_mix_env_status_PILOT.md)

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

| task | claim | evidence | blocking item |
|---|---|---|---|
| **Avoiding** (state) | MF-UNet ≫ DPCC (30×) | ✅ Strong (5 seeds, n=20, Pareto) | DPCC K∈{1,2,5} at n=20 |
| **Avoiding** (state) | FM ≫ DPCC (21×) | ✅ Strong (5 seeds, n=20) | — |
| **Visual Aligning** | MF K2 sole survivor of 3-stage funnel | ✅ Significant (McNemar p ≤ 0.039) | K=100 arms unscorable (24h wall) |
| **Visual Aligning** | MF ≈ AF (same engine) | ⚠️ Not separable (p = 0.39) | Constant-α training run |
| **UAV corridor** | MF > FM at K ≤ 2 | 🟡 Directional (1 seed, n=10) | Multi-seed, diffusion baseline |
| **UAV pillars** | MF > FM > DPCC? | 🔴 Not yet tested | Expand pillars to full sweep |
| **Avoiding** (state) | MPC fan B=4 not needed; B=1 statistically good | ⚠️ Directional (30 ep. fan 4, 6 ep. fan 1) | K1 fan 1 @ 20 trials; parallelise projector |
| **Avoiding** (state) | HF-SLSQP `-t` ≻ DPCC at K20, matched threshold | 🟢 Pareto, but n=6 seed 6 (H6a) | seeds 7–10 |
| **Visual Aligning** | HF-SLSQP ≥ DPCC on constraints 3/3 | ⚠️ *p* floor 0.125 at n=30 (H6b) | seeds 7–10, arm C |
| **All tasks** | `-r`/`-c`/`-t` do not earn their compute | 🟢 exact at B=1; ⚠️ directional at B=4 (H7) | paired B=4 A/B, seeds 7–10 |
| **UAV pillars** | Fix_16 confirmed; scene still unrankable | 🟢 A/B clean · 🔴 S&C 0/2876 (H8) | `*_hg` re-run, all 3 engines |

---

## Next Key Headlines (prioritised)

### Near-term (next batch)

1. **🔑🔑 Project the K=100 arms on visual-aligning** — the only thing that could change the result. At `T=0.5`, K=100 needs 50 SLSQP solves/replan → 28–50h. Must raise the Slurm wall, split one variant per job, or cut `diffusion_timestep_threshold`. MF K100 (0.28×) is closer unguided than the survivor is projected.
2. **🔑🔑 Test-split eval for all VA entrants** — the biggest hole in the funnel. No generalisation demonstrated.
3. **🔑🔑 Constant-α training run on visual-aligning** — tests whether the α→0 snap is an artefact or the JVP target is genuinely worse on vision-conditioned trajectories.
4. **🔑🔑 UAV pillars full sweep** — K ∈ {1, 2, 5, 10, 20} × {fm, mf} × 5 seeds on `pillars`, to prove MF > FM > DPCC where constraints bind.

### Medium-term

4. **🔑 DPCC K ∈ {1, 2, 5} at n=20 on avoiding** — pins diffusion's floor.
5. **🔑 UAV diffusion baseline** — without it no hierarchy claim on UAV.
6. **`af` @ UNet on UAV** — isolate backbone from objective.
7. **Resolve the two FM checkpoints on VA** — Gen7 `cand4` works, Gen14 `cand11` is a no-op. Why?

---

*All numbers sourced from the curated reports in this directory. No number was independently computed for this notebook. See individual reports for methodology, statistics, and reproduction instructions.*


---

# by User notes to update
1. 
the Pillars results is good
/workspaces/FM-PCC/logs_in_develop/Gen15/DA/DA_20260830_pillars_K_sweep_fm_mf_af.md

> ✅ **Folded in → [Headline 8](#headline-8--uav-pillars-the-engine-was-fixed-the-scene-still-is-not-).**
> Split in two, because the two halves point opposite ways: **8a** the Fix_16 A/B is a clean
> methodological win (100 % → 0 % divergence abort, 6.50 m → 0.62 m); **8b** the scene is still
> unrankable (S&C 0/2876, no `diffusion` arm). The 08-30 K-sweep you linked is the *negative* half;
> the positive half is the newer [`DA_20260903_fix16_AB_mf_pillars`](../../logs_in_develop/Gen15/DA/DA_20260903_fix16_AB_mf_pillars.md).


2.
HF beat DPCC in the d3il avoiding. 
/workspaces/FM-PCC/logs_in_develop/aggregated_hf_nlp_backend/DA_20260830_ipopt_vs_slsqp_fmv3ode_K10_K20.md

> ✅ **Confirmed and folded in → [Headline 6a](#headline-6--hardflow-vs-dpcc-the-threshold-was-the-confound-not-the-solver-).**
> You were right and it **supersedes** this directory's standing answer
> ([`RESPONSE_20260826_did_HardFlow_ever_beat_DPCC`](RESPONSE_20260826_did_HardFlow_ever_beat_DPCC.md) §Q3 "No" and
> [`SNAPSHOT_20260823_visual_aligning_env_status`](SNAPSHOT_20260823_visual_aligning_env_status.md) §5 "No").
> The old "no" was measured with **IPOPT at an unmatched activation threshold** (HF `A=1.0` vs DPCC `0.5`,
> i.e. HF doing ~2× the projection work). With SLSQP + `A=0.5`, `hardflow_sls-t-tightened` at K=20
> **Pareto-dominates** DPCC: 61.0 vs 62.2 steps and 0.343 vs 0.475 s/step, both at 100 % S&C and 0.00000
> violations. ⚠️ n = 6, seed 6 only, and only the `-t` arm — `-r`/`-c` are non-dominated.

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

The standing answer in this directory — [`RESPONSE_20260826_did_HardFlow_ever_beat_DPCC`](RESPONSE_20260826_did_HardFlow_ever_beat_DPCC.md) §Q3 ("**No.** DPCC wins every axis at every K") and [`SNAPSHOT_20260823_visual_aligning_env_status`](SNAPSHOT_20260823_visual_aligning_env_status.md) §5 ("**No**") — **is superseded on both tasks.** Those runs carried two confounds that have since been removed:

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

## Evidence board vs. the target (2026-09-05)

🟢 supported · 🟡 partial / underpowered · 🔴 contradicted · ⬜ not measured

### Goal A — engine ladder `af_unet ≥ mf > fm > diffusion`

| environment | `mf > fm` | `fm > diffusion` | `af_unet ≥ mf` |
|---|---|---|---|
| `avoiding-d3il` | 🟢 Pareto-dominant, 30× | 🟡 21×, needs NFE-matched restatement | ⬜ AF ran (24515), never ranked |
| `aligning-d3il` (3-D state) | ⬜ | ⬜ | ⬜ — **no data at all** |
| `aligning-d3il-visual` | 🟢 K2 beats K20 on every axis | 🟢 | 🔴 **AF has never been run at MF's flagship K=20** |
| `uav-corridor` | 🔴 **regime-split**: W13 at K1–K2, L25 at K5–K20 | 🟡 | ⬜ AF was SiT 10.0 M (unmatched); `af_unet` only landed with Gen15 U6 (25434/25439) |
| `uav-pillars` | ⬜ **unrankable** — S&C 0/2876, no `diffusion` arm exists (Headline 8) | ⬜ | ⬜ |

### Goal B — projector ladder

| environment | status |
|---|---|
| `avoiding-d3il`, matched threshold + SLSQP | 🟢 **HF `-t` Pareto-dominates at K=20** (n = 6, seed 6) |
| `aligning-d3il-visual`, K20/T0.2 | 🟡 3/3 constraints, 2/3 strictly better, **but *p* floor is 0.125** |
| `uav-corridor` | 🟡 win switches on at K≥5; 1 seed, pilot, pre-honest-geometry |

### Goal C — selection machinery

| claim | status |
|---|---|
| `B=1` → three rules identical → 3× waste | 🟢 **exact** |
| `-c` is deletable | 🟡 strongly directional |
| `-t` as sole default | 🟡 directional |

### The three real gaps

1. **`af_unet ≥ mf` is unproven everywhere** — the top rung of the headline ladder. Fix: AF at MF's exact flagship (unet, FiLM v1, K=20, T=0.2, seed 6, 26.4 M), **`diffuser` arm first** — an arm that loses unprojected is never ranked projected.
2. **Every projector result is underpowered** — Goal B and Goal C both bottleneck on the same run: seeds 7–10, arm C, `mf`, K=10 and K=20.
3. **`aligning-d3il` 3-D state has no data** — the modality-transfer argument claims state→visual on *aligning*, but aligning was only ever measured visually.

### Run queue implied by the target

| # | run | closes |
|---|---|---|
| 1 | AF U-Net at MF's flagship, VA, `diffuser` arm | gap 1, Stage 1 |
| 2 | if #1 wins: same config through DPCC + HF arms | gap 1, Stage 2 |
| 3 | **Seeds 7–10, arm C, `mf`, K=10 + K=20, VA** | gap 2 — Goal B *and* Goal C |
| 4 | `af_unet` UAV (25434/25439) ranked vs `mf`/`fm` on `pillars_hg` | Goal A row 4 |
| 5 | UAV `*_hg` K-sweep, **all three engines re-run with Fix_16**, ≥3 seeds | UAV S&C → paper grade; makes UAV rankable at all |
| 6 | `aligning-d3il` 3-D state, four-engine ladder | gap 3 |



# DA — HardFlow's minimum K, run: MeanFlow-UNet at `A = 1.0`, K ∈ {2, 3, 5} on `avoiding-d3il`

...

GOOD REUSLT

MAYBE CONSIDER USING THIS AS A FLAGSHIP FOR THE PAPER!!!
