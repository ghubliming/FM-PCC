# UAV-FM `diffuser` (pure-ML) baseline — quality is good on unimodal scenes; multi-mode coherence is Epoch-7 PCC's job

> **Headline (code-grounded, see Conclusion):** Epoch 6 deliberately evaluates the **pure-ML
> `diffuser` baseline** — `eval_fm_uav.py` runs `batch_size=1` with **no candidate selection,
> no projector** *by design*. (The model/policy is a faithful FMv3ODE fork and still *contains*
> `temporal_consistency`/projection; we simply don't use it yet — that is the **Epoch-7 PCC**
> work.) Result: the pure FM produces clean, well-tracked trajectories on **single-mode**
> scenes, and on **multi-mode** scenes a single un-selected sample oscillates between modes →
> a 2nd-order drone can't absorb it → explosion. That multi-mode gap is **expected for a
> diffuser baseline and is exactly what PCC (E7) exists to fix** — not an E6 bug, and NOT
> "missing goal conditioning" (the obs already has `p_des`, like avoiding's `des_xy`).

**Date:** 2026-06-25
**Method:** all-4-scene eval npz analyzed with the U3-extended `npz_analysis/analyze_npz.py`
(`--xy-cols 3 4`; UAV obs = `[p_des(0:3) | p(3:6) | v(6:9)]`).
**Data:** `temp/Gen11_E6_debugs/runs/` — 3 new runs (empty, corridor, s_curve, seed 6,
20 trials) + the legacy pillars run (seed 6, 4 trials).
**Supersedes the open question in** `../U2/Debug_results/NEXT_STEPS_trajectory_diagnosis.md`
(that doc's "discriminating empty A/B" — now answered here; that MD left unchanged).

## The decisive result

| #homotopy modes | scene | success | executed explosion (`exec_maxabs`) | airborne `min_z` |
|----------------:|-------|--------:|-----------------------------------:|-----------------:|
| 1 | **empty**   | **1.00** | none (1.8)        |  0.378 |
| 1 | **s_curve** | **0.95** | mostly fine (116, one drifting trial) | 0.192 |
| 3 | **corridor** | **0.00** | explodes (256)   | −0.021 |
| 4 | **pillars**  | **0.00** | explodes (263)   |  0.086 |

**Perfect correlation: single-homotopy scenes fly (95–100%); multi-homotopy scenes
collapse to 0% success and explode.**

> ⚠️ **CAVEAT — `success` ≠ goal-reaching.** `success` is defined as *contact-free AND
> airborne* (`contact_frac ≤ limit AND min_z > 0.2`), **not** reaching the target. By the
> real-task metric, **`goal_reached_rate = 0.00` for ALL FOUR scenes**, including the
> "100%-success" empty:
>
> | scene | success_rate | goal_reached_rate | goal_dist_mean |
> |---|---:|---:|---:|
> | empty   | 1.00 | **0.00** | 1.51 m |
> | s_curve | 0.95 | **0.00** | 5.74 m |
> | corridor | 0.00 | **0.00** | 18.70 m |
> | pillars  | 0.00 | **0.00** | 6.45 m |
>
> So read the table above as a **stability/coherence** distinction, not task completion:
> single-homotopy = *flies coherently but never reaches the goal*; multi-homotopy =
> *diverges/crashes*. The robust signal is the explosion (`exec_maxabs`, `track_err`),
> **not** `success_rate`. By the goal metric the whole epoch is 0% — which only reinforces
> the conclusion: with **no goal signal** the FM cannot aim at a target, so it can neither
> disambiguate homotopies nor reach a goal. Goal conditioning (E7) is required for both.
> (`success` is the eval's deliberate contact-free/airborne proxy — flagged "please
> confirm" in `eval_fm_uav.py`'s docstring; see `EVAL_METRICS_REFERENCE.md`.)

## Interpretation (see Conclusion for the code-grounded mechanism)

In a scene with **one** expert mode the policy produces a coherent command and flies. In a
scene with **multiple** modes (corridor L/C/R, pillars LLL/LRL/RLR/RRR) the eval samples
**one** trajectory per MPC step with **no temporal-consistency selection** (`batch_size=1`),
so it can commit to "go-left" at one step and "go-right" at the next → the command
**oscillates between modes** → the `p_des` integrator drifts → runaway → crash. A damped arm
absorbs such oscillation; a 2nd-order drone does not. (The avoiding eval avoids this by
sampling a batch and selecting one mode with `temporal_consistency` — which the from-scratch
UAV eval skipped. See Conclusion.) The scene-count correlation also matches the *prediction*
of `logs_in_develop/Gen11/Epoch2_UAV_mujoco_run/DPCC_OBS_DEVIATION.md` ("passes on empty …
surfaces in multi-homotopy obstacle scenes"), though the precise cause is the dropped
selection, not the obs schema.

## What this rules OUT

- **Model capacity / training** — NOT it. `empty` 100%, `s_curve` 95%. The network learned fine.
- **Normalizer / constant-Δz / `SafeLimitsNormalizer eps`** — NOT the root. `empty` uses
  the identical normalizer and does not explode. The dead-Δz/eps issue only influences
  *which axis* blows up once ambiguity has triggered the drift, not *whether* it does.
- **"Can't take off" / undertrained** — NOT it. Flies cleanly when unambiguous.

## Evidence from the new plan-fan metrics (why we trust the above)

`plan_maxabs` (the FM's predicted foresight, all-axis max) stays **~2–3 m even on the
failing scenes**, while `exec_maxabs` → 256. So the FM's *plan* is bounded/sane; only the
*executed* command path explodes — the divergence is created by the incoherent mixture
driving the open-loop integrator, not by the FM's learned dynamics. (The tool exposing
`plan_maxabs` vs `exec_maxabs` is what made this measurable.)

## Conclusion → the eval dropped candidate-selection (code-grounded; supersedes "goal conditioning")

### Correction to an earlier draft
An earlier version of this doc blamed "missing goal conditioning." **That was wrong.** The
UAV obs already carries `p_des` — the same self-accumulated setpoint role that `des_xy`
plays in D3IL avoiding; avoiding does **not** carry a final goal either. So the difference is
NOT a missing input signal.

### What the code actually shows
The model/policy is a faithful FMv3ODE fork — `flow_matcher_v3_uav/sampling/policies.py`
still implements `trajectory_selection` (`temporal_consistency`, `minimum_projection_cost`,
`which_trajectory`). But `eval_fm_uav.py` was **rewritten from scratch** (its header says so)
and **does not use any of it**:

| | FMv3ODE avoiding eval (works) | UAV `eval_fm_uav.py` (explodes) |
|---|---|---|
| candidates / MPC step | `batch_size=args.batch_size` (the run was **mpc4** → 4) | **`batch_size=1`** (`:193`) |
| candidate selection | `temporal_consistency` / `minimum_projection_cost` (commit to one mode) | **none** — takes the single sample |
| projector | yes (DPCC variants) | none |

So the UAV eval runs avoiding's **weakest** mode (≈ `diffuser`: one candidate, no commitment).

### Mechanism
With `batch_size=1` and no temporal selection, each MPC step **independently** samples one
trajectory. In a multi-mode scene that sample can be "go-left" at step *t* and "go-right" at
step *t+1* → the command **oscillates between modes step-to-step**. A heavily-damped arm
absorbs that; a **2nd-order drone** does not — the oscillation destabilises → runaway. Single-
mode scenes (empty, s_curve) have nothing to oscillate between → coherent flight. This is the
homotopy correlation, explained by the eval config + the plant — not by a missing goal.

### Is our `diffuser` faithful to legacy FMv3ODE? — YES (checked 2026-06-25)
- `flow_matcher_v3_uav/sampling/policies.py` is **byte-identical** to the ODE-selectable
  policy (same sampling, obs `_format_conditions`/normalization, action extraction).
- Legacy `diffuser` = `projector=None` + `trajectory_selection='random'` (candidate 0)
  (`eval_flow_matching_v3_ode_selectable.py:240,244`). Our UAV eval matches: no projector,
  default random → candidate 0.
- `test_ret=0` is **correct** for the UAV model (`returns_condition:False`,
  `include_returns:False` → returns ignored), not a discrepancy.
- Open-loop desired-pos accumulation matches (`next_pos_des = action + obs[:2]` ≡ `p_des += action`).
- **Only deviation: `batch_size` (legacy run = 4, UAV = 1).** Immaterial for the diffuser
  *metric* — the executed action is always candidate 0, and candidate 0 of a batch of 4 is
  one i.i.d. sample, distributionally identical to a batch of 1. `batch>1` only buys the
  candidate **fan** (visualization + `plan_cand_spread`); the trajectory quality numbers are
  unaffected.

→ The pure-ML quality measurement is **valid and faithful**.

### What this means (E6 scope) and what is E7
This IS the intended pure-ML `diffuser` baseline for Epoch 6. The multi-mode explosion is the
**expected limitation of a pure generative baseline**: with one un-selected sample per step
it cannot commit to a single mode, so on multi-mode scenes the command oscillates and a
2nd-order drone destabilises. **Resolving that is exactly what Epoch-7 PCC adds — the
`temporal_consistency`/`minimum_projection_cost` candidate selection (needs `batch>1`) and the
constraint projector — which the policy already supports but E6 deliberately does not enable.**
So this finding is not an E6 bug to fix; it is the **baseline result that motivates PCC**.

Pure-ML takeaways, ruled in / out:
- ✅ Single-mode trajectory quality is good (smooth, well-tracked: `track_err` ~mm).
- ✅ Multi-mode coherence requires PCC selection (E7) — pure FM alone can't self-commit.
- ❌ NOT capacity / training / normalizer / obs-schema / "missing goal conditioning"
  (the obs already carries `p_des`, the avoiding `des_xy` analog).
- Note: even single-mode scenes don't *reach the goal* (`goal_reached=0`) — a separate
  matter (no goal signal); see the success-metric fix in `Fix2_metrics/`.

## Where to see the DA numbers yourself (inspect the CSVs)

Run the analyzer; it writes machine-readable CSVs **and** prints the headline table:
```bash
python npz_analysis/analyze_npz.py temp/Gen11_E6_debugs/runs --xy-cols 3 4
#   add  --out <dir>          to choose where the CSVs go
#   add  --replot-plans       to also dump per-trial plan-fan PNGs
```

**Where the numbers land** — by default a `_npz_analysis/` folder is created *inside the
path you pointed at*:
```
temp/Gen11_E6_debugs/runs/_npz_analysis/
    files_summary_<timestamp>.csv   # ONE row per npz (per scene) — the per-scene headline numbers
    per_trial_<timestamp>.csv       # ONE row per (scene, trial) — the 20/4 trials broken out
```

**Which column is which** (the numbers quoted above come straight from these):

| You want… | CSV | Column | Notes |
|---|---|---|---|
| which scene a row is | both | `file` | path contains the `uav-<scene>` segment |
| **success rate** | files_summary | `n_success__mean` | 0–1 (empty=1.0, pillars=0.0) |
| **executed explosion** | files_summary | `traj_max_abs__mean` | the 256/263 blow-ups; per-trial = `traj_max_abs` |
| **plan (foresight) magnitude** | files_summary | `plan_max_abs__mean` | stays ~2–3 m even when failing |
| plan-vs-executed gap | files_summary | `plan_exec_div__mean` | per-trial = `plan_exec_div` |
| executed path smoothness | files_summary | `traj_straightness__mean` | 1=straight, →0 chaotic |
| per-trial everything | per_trial | `trial`, `n_success`, `traj_max_abs`, `plan_max_abs`, … | one row per rollout |

The same headline (`succ_rate | exec_maxabs | plan_maxabs | plan_exdiv | …`) also prints to
stdout when you run the command — no need to open the CSV for a quick look.

> The per-scene success/explosion table at the top of this doc = `n_success__mean` +
> `traj_max_abs__mean` from `files_summary`, with the scene read off the `file` path. To
> sort the CSV by scene quickly: `column -s, -t < files_summary_*.csv | less -S`.
