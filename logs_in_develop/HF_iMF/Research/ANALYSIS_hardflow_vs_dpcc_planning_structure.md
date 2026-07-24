# Why HardFlow's avoiding output looks nothing like DPCC's — the planning structure, counted

**Date:** 2026-07-20
**Question:** *"what is the total steps and mpc×4 fan in HardFlow? In DPCC I felt there is ~60 steps and each step has mpc×4 × H8 receding horizon, so tons of MPC candidate trajectories — how is it in HF?"*
**Short answer:** your DPCC recollection is correct (`batch_size: 4`), and **HardFlow deliberately has no candidate fan at all — `batch_size` is hard-asserted to 1.** DPCC *selects* among stochastic candidates; HardFlow *projects* a single trajectory onto the feasible set. That single architectural choice is why the outputs look incomparable.
**Companions:** `DISCUSSION_foresight_fan_and_smoothness_paradigms.md` (why HardFlow never plots or discusses smoothness), `MEMO_hardflow_fig11_predicted_style.md`.

---

## 1. The core difference — selection vs. projection

| | **DPCC** | **HardFlow** |
|---|---|---|
| plan-time `batch_size` | **4** — `aux_repo/dpcc/config/avoiding-d3il.py:69` | **1** — default `hardflow/config/flow_matching.py:62`, and **hard-asserted** at `flow_policy.py:798, 959, 1148, 1293, 1522` (`"batch_size must be 1 for optimal control"`) |
| how the plan is chosen | sample 4 candidates → **pick best by value** | sample **one** → **hard-project** it with IPOPT |
| `max_episode_length` | 200 (`…dpcc/config/avoiding-d3il.py:68`) | 100 (`flow_matching.py:53`) |
| horizon H | 8 | **16** |
| candidate fan to plot | **yes, 4 per replan** | **none** |

**The one number that explains everything:** DPCC's "mpc×4" is `batch_size: 4`. HardFlow removes it — a batched sample would be meaningless because the NLP is solved for one trajectory, and the guarantee comes from *constraining* it rather than from *choosing* among draws.

> HardFlow's only vestige of candidate selection is the **warm-start**: `warmstart()` (`flow_policy.py:753`) samples `cfg.warmstart_batch` rollouts and takes `argmax` of the value (`:777`). But `warmstart_batch = 1` by default (`flow_matching.py:82`, and in every run script), so **argmax over 1 item = no selection**. The machinery exists but is switched off.

## 2. The three different "many trajectories" — the actual source of confusion

Three distinct objects get called "the fan". Conflating them is why HardFlow's plots look wrong:

| # | Object | What it is | DPCC | HardFlow |
|---|---|---|---|---|
| **1** | **Candidate fan** | alternative plans *at one replan*, choose one | **4** | **0** (B=1) |
| **2** | **Generation chain** | ODE/NLP intermediates *within one plan* — the trajectory forming | n_diffusion_steps | **11** (FM, `ode_t_steps=10`) / **6** (iMF, K+1 at K=5) |
| **3** | **Replan sequence** | successive final plans *across the episode* | ~episode/replan | **~6–7** |

- **#2 is what the paper's Fig. 11 visualizes** ("the generation process … one representative planning instance"). Built in `hardflow_new_forward` as `X_optimized` (`flow_policy.py:892`) → reshaped to `optimized_x_chain` (`:905`), returned as `x_chain`.
- **#3 is what our foresight-fan diagnostic plots** (`run/eval_imf.py`, `_save_foresight_fan`) — it stores `x_chain[-1]`, i.e. only the *final* state of #2, once per replan.
- **#1 simply does not exist in HardFlow.**

## 3. The counts, from the real n=200 run

Derived from `temp/U5_debug/{imf,HF_new}_trajectories.csv` — the `nlp_solves` column gives an exact, independent read of the replan count (iMF does K NLP solves per plan):

| | iMF (K=5) | FM (`ode_t_steps=10`) |
|---|---|---|
| mean steps / episode | **52.1** | **50.6** |
| `nlp_solves` / episode | **35** | (not instrumented) |
| **replans / episode** | **35 ÷ 5 = 7.0** | 50.6 ÷ 8 = **6.3** |
| chain states per replan (#2) | K+1 = **6** | **11** |
| **total H16 plans generated / episode** | 7 × 6 = **42** | 6.3 × 11 = **70** |
| **final plans / episode** (#3) | **7** | **6** |

Replan cadence is `replan_steps = 8` — `run_env` re-plans when `action_index >= cfg.replan_steps` (`run/eval.py:391`) and executes the first 8 actions of the H=16 plan (`:396`).

**Answering the original framing directly:** HardFlow is **~50 steps × (1 candidate) × H16**, replanning every 8 steps ⇒ **~7 plans per episode**, not DPCC's *~60 steps × 4 candidates × H8*.

## 4. Why this matters for the smoothness investigation

1. **There is no candidate spread to measure.** Any DPCC-style "how wide is the fan" metric is undefined for HardFlow — a real structural difference, not missing instrumentation.
2. **`plan_roughness` (fix_7) measures object #3** — the ~7 final plans per episode, ≈35 per 5-episode cell. That is the correct object for "is the MPC trajectory smooth", and it is why n=5 suffices.
3. **A genuine asymmetry worth noting:** iMF generates **42** intermediate plans per episode vs FM's **70**, because K=5 < 10 ODE steps. Fewer refinement stages is exactly the efficiency win (1.95× fewer NFE) — and is a plausible contributor to the residual coarseness behind the 98.5% vs 100% safety gap (`../../Gen13/u_5/RESULTS_Gen13_u5_paired_n200.md` §6).
4. **Object #2 is currently uncaptured.** fix_7/u_8 store only `x_chain[-1]`. Capturing all 6/11 chain states would show the plan *forming* under successive projections — the truest Fig. 11 reproduction, and the view that would reveal whether iMF's plan is rough **from birth** (field problem) or **gets roughened / stays rough through projection** (correction problem). This is the natural next diagnostic; not yet built.

## 5. Reference index (file:line)

| Claim | Where |
|---|---|
| HardFlow forces single-trajectory planning | `hardflow/models_flow/flow_policy.py:798, 959, 1148, 1293, 1522` |
| default `batch_size = 1`, `max_episode_length = 100`, `warmstart_batch = 1` | `hardflow/config/flow_matching.py:62, 53, 82` |
| warm-start sampling + argmax-over-batch | `flow_policy.py:753, 756, 777` |
| generation chain assembled / returned | `flow_policy.py:892, 905` |
| replan cadence (`replan_steps=8`) | `run/eval.py:391, 396` |
| eval run parameters (H16, `ode_t_steps=10`, `replan_steps=8`, `warmstart_batch=1`) | `run_scripts/eval_hardflow_new.sh:14,16,20,22` |
| **DPCC candidate batch = 4**, `max_episode_length = 200` | `aux_repo/dpcc/config/avoiding-d3il.py:69, 68` |
| our roughness metric + fan capture (#3) | `run/eval_imf.py` — `grep -n "fix_7" run/eval_imf.py` |
