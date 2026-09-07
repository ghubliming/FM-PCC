# Gen15 U10 — `UAV_MIX_CONTROLLER`, and the Gen15 mjpc env-detection bug

## 0. TL;DR

🔴 **Gen15 could not run `controller='mjpc'` at all.** The eval sbatch picked its conda env by
reading a **Gen11** config block, so a Gen15 mjpc run silently landed in the wrong environment
and would die on `import mujoco.mjx`. Fixed, plus a per-job `UAV_MIX_CONTROLLER` override so the
tracker is no longer a shared-config edit.

## 1. The bug

`Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh:118` (pre-fix):

```bash
DETECTED_CONTROLLER=$(awk "/'plan_flow_matching_v3_uav': \{/,0" "$REPO/config/uav.py" ...)
```

`plan_flow_matching_v3_uav` is a **Gen11** block in **`config/uav.py`**. Gen15's controller lives
in **`config/uav_mix.py`**. The env switch exists because `mjpc` needs `mujoco>=3` / `mjx`, which
conflicts with the `mujoco==2.3.7` pin the rest of the repo requires — so `mjpc` must run in the
isolated `FMPCC_mjx` clone env.

Consequence: set Gen15 to `mjpc` and the script still selects plain `FMPCC` → the job dies inside
the tracker import, hours after submission. Inherited from the Gen11 sbatch by copy-modify and
never repointed at the new config file.

It never fired in practice only because every Gen15 run so far used `pid_stopgo`, and both files
happen to say `pid_stopgo` — see §4.

## 2. Second problem: `controller` was not settable per job

It is read from a **shared base dict** (`config/uav_mix.py:271, 308`), inherited by every
`plan_mix_uav_*` block. Editing it to try `mjpc` would switch *every* Gen15 run — including
anything already queued, whose env is captured at submit time but whose **config is read at run
time**. That is the U6 failure mode exactly.

## 3. The change

| file | change |
|---|---|
| `mix_uav_test/eval_mix_uav.py` | `UAV_MIX_CONTROLLER` override after the `cfg['controller']` read; validates against `pid`/`pid_stopgo`/`pid_const_v`/`mjpc`, `exit 2` otherwise; echoes the config→env transition |
| `Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh` | env-detection honours `UAV_MIX_CONTROLLER` first, else reads **`config/uav_mix.py`** |
| `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` | exports `UAV_MIX_CONTROLLER` + echo |

`controller` is a **results-path key** (`Emf_K3_mpc4_<controller>_T0.5`, `_uav_eval_tag()`), so a
controller A/B writes two separate folders and cannot collide.

## 4. 🔴 Safety check — the seven in-flight jobs (25486–25492)

These were submitted **before** this change, so `UAV_MIX_CONTROLLER` is absent from their
captured environment, but they read the repo at **run** time. Verified the patch is a no-op for
them:

| | detected controller | conda env |
|---|---|---|
| before patch (`config/uav.py`, Gen11 block) | `pid_stopgo` | `FMPCC` |
| after patch (`config/uav_mix.py`, Gen15) | `pid_stopgo` | **`FMPCC`** |

Same value, same env. Both files independently say `pid_stopgo`, which is why the latent bug was
never triggered — and why repointing the grep is safe mid-flight.

Python side, simulated: unset → `pid_stopgo` (unchanged); `mjpc` → `mjpc`; `bogus` → `exit 2`.

## 5. Verification

* `python3 -m py_compile mix_uav_test/eval_mix_uav.py` — OK
* `bash -n` on both sbatch scripts — OK
* detection + override simulated on all paths (§4)
* 🔴 **Not run on the cluster.** Two unknowns remain, both cluster-side:
  1. does the `FMPCC_mjx` env exist? — `conda env list | grep mjx`
  2. does `MJPCTracker` actually run on this scene? It has never been exercised on Gen15 UAV.

  First-run checks: `[ env-select ] ... -> conda env 'FMPCC_mjx'`, then
  `[ U10 ] controller: 'pid_stopgo' -> 'mjpc' (env override)`, then a results path containing
  `_mpc4_mjpc_T0.5`.
