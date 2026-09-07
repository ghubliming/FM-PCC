# Did HardFlow ever say *why* it uses H8+8 (H=16, execute 8)?

**Date:** 2026-08-21 · **Type:** paper + upstream-code forensics (no code touched, nothing run)
**Question:** *"HardFlow builds on the DPCC setup but deliberately switches from DPCC's H8 / execute-1
to H16 / execute-8. Did the paper say why?"*

**Short answer: no.** The paper states `H=16, T=8` exactly **once**, as a bare fact in the appendix
experiment details, and never argues for it, never ablates it, and never mentions that DPCC — the
paper it says it follows — uses `H=8, T=1`. The choice is not defended anywhere; the evidence says it
was **inherited from the codebase HardFlow forked for the flow-matching model** (Feng et al.'s
flow-guidance / Diffuser lineage), not derived from the D3IL task.

**Sources:** `aux_repo/HardFlow_Paper_Files/arXiv-2511.08425v3/main.tex` (arXiv 2511.08425v3, TPAMI
2026) · `aux_repo/HardFlow` (branch `d3il`) · `aux_repo/flow_guidance` (= `feng2025on`) ·
`aux_repo/dpcc`
**Companions:** `../../HF_iMF/Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md` (the counted
planning structure) · `GUIDE_H16_replan8_MF_UNet.md` · `../DA/DA_20260818_H16_replan8_MF_UNet.md`

---

## 1. Everything the paper says about the planning horizon and the replan cadence

`replan*` occurs **4 times** in a 1549-line source file. That is the whole corpus:

| # | main.tex | Where | What it says |
|---|---|---|---|
| 1 | `:728` | Sec. VII-A, task setup | *"…execute the actions \{a_i\}_{i=0}^{T-1} sequentially in the environment, where **T ≤ H denotes the replanning horizon**."* — a **definition**, no value, no reason. Same paragraph opens with *"This task follows the setup of [romer2025diffusion]"* (= DPCC). |
| 2 | `:737` | Sec. VII-A, metrics | *"Computation Time: the average time taken to sample a trajectory … **at each replanning step**."* — the metric is **per replan**, not per env step. |
| 3 | `:1242` | App. "Experiment Details → Robotic Manipulation" | *"The trajectory horizon is **H=16** and the replanning horizon is **T=8**."* — one sentence, stated, not justified. |
| 4 | `:1276` | Fig. 11 caption | *"Since the policy replans in a receding-horizon manner, we show one representative planning instance."* |

**Not present anywhere:** any sentence explaining the value 16 or the value 8; any comparison to
DPCC's H8/replan-1; any sensitivity study on H or T. The paper's Sensitivity Analysis appendix covers
only the **regularization coefficient** (`:1382`), the **control activation schedule** (`:1414`), and
**stress testing** (`:1444`). Horizon and cadence are simply not among the knobs it examines.

### 1.1 The trap: the paper *does* justify a "one-step horizon" — a different horizon

App. "One-Step Decomposition" (`:1214`) opens with

> *"The one-step receding-horizon formulation is a **deliberate design choice**…"*

This is easy to misread as a defence of the planning cadence. It is not. That paragraph is about the
**MPC decomposition in flow time** — treating each ODE/sampling step `x_{i+1} = x_i + v(x_i)Δt + u_iΔt`
as a one-step subproblem instead of optimizing the whole `N=10`-step sampling chain jointly. Its
argument is about the neural-dynamics feasible set and the reverse reparameterization, and it never
touches the environment-time horizon `H` or the executed-action count `T`.

So HardFlow has **two** "horizons", and only the flow-time one is argued for:

| horizon | symbol | value | argued in the paper? |
|---|---|---|---|
| flow-time MPC horizon (per sampling step) | 1 step of `N=10` | 1 | ✅ yes, at length (`:1214`, `:1223`) |
| trajectory / planning horizon | `H` | 16 | ❌ stated only (`:1242`) |
| executed actions per plan | `T` | 8 | ❌ stated only (`:1242`) |

---

## 2. Where H16/T8 actually comes from — the code says "inherited"

### 2.1 The repo's own defaults are DPCC's numbers; the run scripts override them

| | value | file:line |
|---|---|---|
| `horizon` **default** | **8** | `hardflow/config/flow_matching.py:12` (train), `:47` (eval) |
| `replan_steps` **default** | **1** | `hardflow/config/flow_matching.py:44` |
| `horizon` **used for every paper run** | **16** | `run_scripts/{train,eval_original,eval_hardflow,eval_hardflow_new,eval_projection*,eval_gradient_guidance,eval_oc_flow}.sh` — `horizon=16` |
| `replan_steps` **used for every paper run** | **8** | same scripts — `replan_steps=8` |
| cadence enforcement | `assert replan_steps < horizon` | `run/eval.py:380-382` |
| rollout: replan when the buffer runs out, else execute the next planned action | | `run/eval.py:391, 396` |

The dataclass defaults `horizon=8, replan_steps=1` are exactly DPCC's configuration
(`aux_repo/dpcc/config/avoiding-d3il.py:22, 83`, and DPCC calls the policy inside the per-step loop —
`scripts/eval.py:231` — i.e. replan every env step). **The DPCC-shaped defaults are still sitting in
the config, and every shipped run script overrides them.** That is the strongest single piece of
evidence that H16/T8 was an override applied on top of a DPCC-derived skeleton, not a value the task
demanded.

### 2.2 The override matches the *other* upstream — the flow-matching codebase

HardFlow's D3IL training pipeline is a black-reformatted fork of Feng et al.'s flow-guidance code
(`aux_repo/flow_guidance/offline_rl/gflower/`, cited as `feng2025on`, itself a Diffuser fork). The
paper says so directly: *"trained on the D3IL dataset … with **training details as in [feng2025on]**"*
(`:1242`). The fingerprints:

| fingerprint | HardFlow | flow_guidance |
|---|---|---|
| dataset class | `hardflow/datasets/sequence.py` | `offline_rl/gflower/datasets/sequence.py` — same code, reformatted |
| experiment-name pattern | `exp_name="H${horizon}_1e6steps"` (`run_scripts/train.sh:27`) | `exp_name="$flow_prefix"H"$horizon"_1e6steps` (`run_scripts/train.sh:26`) |
| train length | `--n_train_steps 1000001` | `--n_train_steps 1e6`-style, same convention |
| horizon in that convention | `16` | `20` (D4RL locomotion) |

And the same "inherit the upstream's horizon" habit repeats across HardFlow's other tasks: maze
navigation uses `H=384` (`:1246`), which is the Diffuser/SafeDiffuser `maze2d-large` convention, with
*"All other implementation details are identical to those in the robotic manipulation task."*

**Reading:** per task, HardFlow adopts the generative-model recipe (architecture, horizon, training
length) of whichever repo supplied that task's pretrained-model pipeline, and takes only the *task
semantics* — env, obstacle constraints, fitted linear dynamics constraints — from DPCC
(`:1242`: *"Obstacle-avoidance constraints … follow [romer2025diffusion]"*). H=16 arrives with the
model recipe; T=8 = H/2 is the round number that follows once you decide not to replan every step.

---

## 3. Is the switch a problem? Two separate answers

**Internally: no.** Every row of the D3IL table (`:748-770`) runs at the same H16/T8, same `N=10`
sampling steps, same solver, same 50 trials. `Original`, `Projection-All`, `Projection-Late`,
`Projection-Relaxed`, `Gradient Guidance`, `OC-Flow` are all re-implementations inside HardFlow's own
harness. **DPCC is not a row in that table** — `Projection-All` is DPCC-style per-step projection
re-implemented on HardFlow's flow model, not DPCC's published numbers. So the comparison HardFlow
actually makes is fair.

**Across papers: yes, and it is unstated.** Anyone reading *"this task follows the setup of
[romer2025diffusion]"* and then putting HardFlow's `52.5 steps / 1.00 safety / 0.190 s` next to DPCC's
published table is comparing runs that differ in at least five ways:

| | DPCC | HardFlow | source |
|---|---|---|---|
| planning horizon `H` | 8 | **16** | `dpcc/config/avoiding-d3il.py:22` vs `main.tex:1242` |
| executed actions per plan `T` | 1 | **8** | `dpcc/scripts/eval.py:231` vs `run_scripts/*.sh` |
| candidate fan (`batch_size`) | 4 (+ value selection) | **1**, hard-asserted | `dpcc/…:69` vs `flow_policy.py:798,959,1148,1293,1522` |
| `max_episode_length` | 200 | **100** | `dpcc/…:68` vs `flow_matching.py:53` |
| obstacles | original pillars | **+ novel purple test-time regions** | `main.tex:728` |

Note also which way the metric definitions cut: **Computation Time is defined per replanning step**
(`:737`), so `T=8` does *not* inflate that number — but it does cut per-episode planning compute by
~8×, and that per-episode figure is never reported. Our own count on the real n=200 runs:
**~7 plans per episode, 35 NLP solves per episode** at `T=8`
(`../../HF_iMF/Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md` §3). At `T=1` the same
episode would need ~50 plans / ~250 NLP solves.

---

## 4. The unstated reasons that are most likely true (our inference, not the paper's claim)

Flagged explicitly as **inference** — none of this appears in the text:

1. **Affordability of the NLP.** HardFlow solves an IPOPT problem at each of the *second half* of the
   `N=10` sampling steps, every replan. `T=8` is what makes ~35 solves/episode instead of ~250. A
   training-free inference-time method that replans every step would look ~8× slower with no change
   to the algorithm. This is the reason with the most force, and it is the one the paper is quietest
   about.
2. **The terminal cost needs lookahead.** The task cost is the squared distance from `s_{H-1}` to the
   target (`:1242`), and the whole algorithm optimizes the *predicted terminal sample*. A longer `H`
   puts the goal-reaching cost further ahead, which plausibly drives the headline "fewest steps"
   result (52.5). At H=8 the same cost sees half as far.
3. **Mechanical constraint.** `assert replan_steps < horizon` (`run/eval.py:380`) means `T=8` forces
   `H ≥ 9`; 16 is the natural power-of-two, and the temporal U-Net's `dim_mults=(1,4,8)`
   (`run/train.py:35`) wants `H` divisible by 4. Neither forces 16 over 8 — `H=8, T=4` would satisfy
   both — so this constrains but does not explain.

---

## 5. What this means for us

- **Do not cite HardFlow as having a rationale for H8+8.** If we adopt the structure, the
  justification has to be ours. Saying "HardFlow showed H16/T8 is better" would be unsupported —
  they never ran the comparison.
- **The three-rung ladder in `GUIDE_H16_replan8_MF_UNet.md` §1.1 is doing work the paper skipped.**
  H8/r1 → H16/r1 → H16/r8 is, as far as we can tell, the first place the horizon effect and the
  cadence effect are separated on this task at all.
- **Our DA finding stands on its own:** replan cadence and constraint tightening are coupled knobs;
  H16/8+8 is affordable *because* the projection carries a margin
  (`../DA/DA_20260818_H16_replan8_MF_UNet.md` §"transferable result"). The paper offers nothing that
  contradicts or supports this — it is silent.
- **Buffering assumption is inherited silently.** "Execute 8" presumes the actions can be buffered
  and played open-loop for 8 env steps. HardFlow never states this; if we report H16/8+8 numbers we
  should state it (DA_20260818 §"deployment assumption").

---

## 6. Is it cheating? — our judgement (inference, not the paper's claim)

Everything in this section is **our reading**, flagged as such. The paper makes no claim here either
way, and we have not run the counterfactual.

### 6.1 The verdict: not cheating; under-reported

**Cheating would require handicapped baselines. There are none.** Every row of the D3IL table runs at
the same H16/T8, the same `N=10`, the same solver, the same 50 trials, and DPCC's published numbers
are not a row at all (§3). The comparison HardFlow actually makes is internally clean.

**What is genuinely criticisable is provenance reporting.** Writing *"this task follows the setup of
[romer2025diffusion]"* (`main.tex:728`) while silently changing `H` (8→16), `T` (1→8), the candidate
fan (4→1), `max_episode_length` (200→100) and the obstacle set is an omission that happens to be
convenient. It invites a cross-paper comparison the numbers do not support.

**But the code says inheritance, not intent.** The DPCC-shaped defaults `horizon=8, replan_steps=1`
are still sitting unused in the dataclass (§2.1), and H=16 arrives with the flow-guidance fork (§2.2).
Forensics on the published repo: **only `initial commit` + readme commits** across the four task
branches, `horizon_list=(16)` with a single entry, no H8 checkpoints, no sweep artifacts. There is
**no evidence of an H8/T1 run that was performed and buried** — and equally none that it was ever
tried. Occam favours "took the upstream recipe and never revisited it".

### 6.2 Does their *math* force T=8? No — and this is the cleanest finding here

`T` appears in **zero equations**. Not in Problems 1–6, not in the MPC-suboptimality theorem, not in
the fixed-point-error theorem, not in Algorithm 1. The entire optimal-control formulation lives in
**flow time**: `u_i` perturbs the sampling ODE across the `N=10` steps, and the trajectory
`x ∈ ℝ^{6H}` is merely the *sample* being drawn. `H` enters only as sample dimensionality. `T` is an
environment-loop wrapper (`run/eval.py:391`), outside the algorithm entirely.

The safety proposition (`h(x̂_N) ≤ 0`, exact at `t_N` because `M_{t_N}^θ(x_N)=x_N`) is a **per-sample**
property. It holds identically at `T=1`. Nothing in the theory prefers 8 — the only hard requirement
is the mechanical `assert replan_steps < horizon` (`run/eval.py:380`).

**The direction of the pressure is the opposite of the intuitive one, and this point favours them:**
`T=8` commits to 8 open-loop steps under a least-squares-fitted *linear* dynamics model
`s_{i+1}=As_i+Ba_i+c`, so model error compounds across the executed prefix. `T=8` makes the
**execution** problem *harder*, not easier — and they still report 1.00 safety. What `T=8` makes
easier is only **compute**: ~35 NLP solves/episode instead of ~250 (§3).

### 6.3 Would HardFlow fail at H8/T1? Split by claim — they are not equally fragile

| Headline claim | Fragility at H8/T1 | Reasoning |
|---|---|---|
| Safety Rate **1.00** | **low** | Per-sample terminal guarantee, independent of `T`; replanning every step is *closer* to closed-loop and gives more corrective opportunities. Expect it to hold or improve. |
| **Fewest steps, 52.5** | **high** | The cost is `‖s_{H-1} − target‖²`. At H=16 the optimizer pulls a point **16 steps ahead** toward the goal; H=8 halves that lookahead. The "fewest steps" win is plausibly in part a *horizon* artifact rather than an algorithm artifact. |
| **"Mild overhead", 0.190 s** | **high** | The metric is defined *per replanning step* (`:737`), so `T` is invisible in it — but per-episode planning compute rises ~8× at `T=1`. The practicality framing depends on the cadence, and the per-episode figure is never reported. |

So the suspicion "they would fail at H8/1" is defensible for **steps** and for the **efficiency
story**, and not defensible for the **safety** claim — which is the one the paper actually leads with.

### 6.4 It is cheap to settle

Their config already defaults to `horizon=8, replan_steps=1`, so the counterfactual is a retrain plus
one eval — no code change. Anyone (us included) can run it. Until someone does, both
"H16/T8 was necessary" and "H16/T8 was flattering" remain unevidenced.

---

## 7. How to re-verify in 30 seconds

```bash
P=/workspaces/aux_repo/HardFlow_Paper_Files/arXiv-2511.08425v3/main.tex
grep -n "replanning horizon\|H=16\|T=8" $P            # → 728, 1242 only
grep -oi "replan[a-z]*" $P | sort | uniq -c            # → 4 occurrences total
grep -rn "horizon\|replan_steps" /workspaces/aux_repo/HardFlow/hardflow/config/flow_matching.py
grep -rn "^horizon=\|^replan_steps=" /workspaces/aux_repo/HardFlow/run_scripts/*.sh
grep -n "horizon" /workspaces/aux_repo/dpcc/config/avoiding-d3il.py
```
