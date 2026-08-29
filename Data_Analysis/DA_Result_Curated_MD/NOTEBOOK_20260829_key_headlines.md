# FM-PCC — Key Headlines Notebook

**Last updated:** 2026-08-29 · **Author:** auto-generated from curated reports  
**Status:** 🟡 Two tasks established, one under construction, several open experiments

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
| 2026-08-29 | α-Flow vs MeanFlow analysis — confirmed same engine, different curriculum |

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

## Summary Scorecard

| task | claim | evidence | blocking item |
|---|---|---|---|
| **Avoiding** (state) | MF-UNet ≫ DPCC (30×) | ✅ Strong (5 seeds, n=20, Pareto) | DPCC K∈{1,2,5} at n=20 |
| **Avoiding** (state) | FM ≫ DPCC (21×) | ✅ Strong (5 seeds, n=20) | — |
| **Visual Aligning** | MF K2 sole survivor of 3-stage funnel | ✅ Significant (McNemar p ≤ 0.039) | K=100 arms unscorable (24h wall) |
| **Visual Aligning** | MF ≈ AF (same engine) | ⚠️ Not separable (p = 0.39) | Constant-α training run |
| **UAV corridor** | MF > FM at K ≤ 2 | 🟡 Directional (1 seed, n=10) | Multi-seed, diffusion baseline |
| **UAV pillars** | MF > FM > DPCC? | 🔴 Not yet tested | Expand pillars to full sweep |

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
