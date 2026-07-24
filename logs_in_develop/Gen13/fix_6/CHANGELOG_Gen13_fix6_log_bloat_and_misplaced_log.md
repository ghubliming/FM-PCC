# Gen13 fix-6 — CasADi log bloat (91%) + verbose log written into the results tree

**Date:** 2026-07-20
**Trigger:** user inspection of the u_5 paired n=200 run (job **23602**) — "why the hell is `u5_armB_fm_verbose_23602.log` in my logs folder?" and the job log being 3.2 MB.
**Nature:** observability only. **No numerical result is affected**; the u_5 results in `../u_5/RESULTS_Gen13_u5_paired_n200.md` stand unchanged.

---

## Problem A — 91% of the job log was CasADi timing output

**Evidence (job 23602):** main job log = **53,915 lines / 3.2 MB**. Composition:

| Content | Lines | Share |
|---|---|---|
| **CasADi timing tables** (`solver : t_proc …`, `nlp_f | nlp_g | nlp_jac_g …`) | **49,209** | **91%** |
| `Norm of Control Inputs` + its blank lines | ~4,200 | 8% |
| episode lines (wanted) | 200 | 0.4% |
| IPOPT iteration tables | **0** | — (fix_4 worked) |

**Root cause:** fix_4 set `ipopt.print_level = 0`, which correctly silenced IPOPT. But **CasADi prints its own ~7-line timing table after every solve**, governed by a *separate* option — `print_time` (default `True`). At 7,030 NLP solves that is 49k lines. I silenced the wrong knob's neighbour and never checked the residual.

**Fix** (`imf/imf_flow_policy.py`, `hardflow_formulate` override): after `super()` builds the CasADi problem, re-declare the solver with `print_time` disabled — additive, so the pre-existing `flow_policy.py` stays untouched:
```python
self.oc_cs_opti.solver("ipopt", {
    "ipopt.print_level": print_level,
    "ipopt.hessian_approximation": "limited-memory",
    "print_time": False,          # <-- the 91% fix
})
```

**Also fixed:** the per-plan `Norm of Control Inputs` print (1,406 lines + 2,812 blanks) is now gated behind a new default-off flag `imf_verbose_control` (`imf_config.py`).

**Expected effect:** next paired run's job log ≈ **53,900 → ~1,500 lines**.

## Problem B — verbose FM log written INTO the results tree

**Evidence:** `logs/hardflow/avoiding-v0/eval/u5_armB_fm_verbose_23602.log` (8.5 MB) sitting among the eval results.

**Root cause:** my own u_5 sbatch line
`FM_LOG="logs/avoiding-v0/eval/u5_armB_fm_verbose_${SLURM_JOB_ID}.log"` — I redirected arm B's output (a sensible idea, since pre-existing `run/eval.py` still has the noisy tqdm `run_env`) but chose the *results* directory as its home. Result dirs must contain **only** results (`trajectories.csv`, `*_real.png`).

**Fix** (`eval_paired_n200_hardflow.sh`): the log now goes where every other job log lives, with the standard naming convention —
```
$REPO/Slurm_Codes/logs/<SUBMIT_DATE>/<SUBMIT_TIME>_u5_armB_fm_verbose_<jobid>.log
```

**User action:** the stray `logs/hardflow/avoiding-v0/eval/u5_armB_fm_verbose_23602.log` can be deleted; nothing reads it.

## Not a bug (asked about)

`Slurm_Codes/logs/2026-07-20/00_33_01_eval_paired_n200_hardflow_23602.log` is the **normal `submit.sh` job log** — correct location and naming (`<HH_MM_SS>_<jobname>_<jobid>.log` under a date folder). The date/time differing from the in-log `JOB START` is just local-time vs UTC. Working as intended.

---

## Files changed (all Gen13-owned; frozen HardFlow files untouched)

| File | Change |
|---|---|
| `HardFlow/hardflow/models_flow/imf/imf_flow_policy.py` | `print_time: False` in the solver override; gated the `Norm of Control Inputs` print |
| `HardFlow/hardflow/models_flow/imf/imf_config.py` | new `imf_verbose_control: bool = False` |
| `Slurm_Codes/sbatch/hardflow/eval_paired_n200_hardflow.sh` | FM verbose log moved out of the results tree into `Slurm_Codes/logs/<date>/` |

**Track back to the code:** every change is tagged in-place with the comment marker `u_5 fix` (written during the u_5 debugging session, before this changelog was labelled fix_6) —
```bash
grep -rn "u_5 fix" HardFlow/hardflow/models_flow/imf/ Slurm_Codes/sbatch/hardflow/
```

**Verified:** `py_compile` + `bash -n` clean; `ImfEvaluationConfig()` confirms `imf_plot_fan=False` **and** `imf_verbose_control=False`; `git status` confirms `run/eval.py` and `run_scripts/eval_hardflow_new.sh` remain untouched.

## Lesson recorded

Silencing a library's verbosity means checking **what remains in the log afterwards**, not just that the intended knob was set — fix_4 declared victory on IPOPT while CasADi's separate `print_time` kept emitting 91% of the output. Consistent with the standing SLURM memory rule (no live/per-iteration output in batch logs); the rule now has a second failure mode on record: *a second library printing through a different option*.

## Scope note

Arm B (FM, via pre-existing `run/eval.py`) still produces verbose output — it is protected by the no-edit rule. That is why it is redirected to a side file rather than silenced. Its CSV and results are unaffected.
