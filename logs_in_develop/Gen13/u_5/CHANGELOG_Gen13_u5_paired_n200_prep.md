# Gen13 upgrade-5 (u_5) — preparation for the decisive paired n=200 safety run

**Date:** 2026-07-19
**Implements:** `../fix_3/INSIGHTS_Gen13_first_run.md` §15 (the "next step" plan).
**Goal:** make the decisive experiment runnable — iMF K5 **and** FM B2, both at **n=200**, same seed, one job — without touching any frozen artifact or any pre-existing HardFlow file.
**Status:** ✅ code ready. Nothing has been run (cluster-side).

---

## 1. Why this run is the decisive one

At n=50, iMF K5 (1 violation) vs FM B2 (0 violations) gives Fisher **p = 1.000** — literally the least significant result possible. Neither method's true violation rate is pinned down (95% CI upper bounds: FM ≤7.1%, iMF ≤10.6%). The Bayesian posterior leans 75% toward iMF being slightly worse, but with a credible interval of **[−4.3, +9.0] pts** — uninformative.

**n=200 raises power from 64% → 98%** for detecting a true 2% violation rate. It converts Gen13's central open question from "undetermined" into a defensible yes/no.

**Both arms must be re-run.** The frozen FM baseline is only 0/50; comparing a fresh n=200 iMF run against it would be unfair and underpowered. Identical n on both arms is the only way to get two comparable intervals.

---

## 2. The two blockers (identified in §15.3) and how they were solved

### Blocker 1 — `random_repeat=50` hardcoded in both eval scripts

| Arm | File | Editable? | Solution |
|---|---|---|---|
| A (iMF) | `run_scripts/eval_hardflow_new_imf.sh` | ✅ Gen13-owned | made env-overridable: `random_repeat="${RANDOM_REPEAT:-50}"` |
| B (FM) | `run_scripts/eval_hardflow_new.sh` | ❌ **pre-existing, no-edit rule** | **NOT edited.** Created an additive Gen13-owned sibling: `run_scripts/eval_hardflow_new_paired.sh` |

The FM sibling is an **exact copy** of the frozen script — every numerical parameter byte-identical (horizon 16, cp 20, `ode_t_steps=10`, seed 0, replan 8, `constraint=novel`, all cost/value scales, `--dynamics_constraint`) — with only three deliberate differences:
1. `random_repeat` → `${RANDOM_REPEAT:-200}`
2. `exp_name` → gains `_n<N>` suffix
3. `solver_print_level` 5 → `${SOLVER_PRINT_LEVEL:-0}` — **cosmetic only** (IPOPT's ~45 lines/solve would be ~4× worse at n=200; does not affect the solution or any CSV value)

So arm B is a faithful re-run of the frozen baseline, just at larger n.

### Blocker 2 — exp_name collision would overwrite frozen results

Added a **guarded suffix** to both iMF scripts:
```bash
[ "$random_repeat" != "50" ] && exp_name="${exp_name}_n${random_repeat}"
```
Verified behaviour:

| RANDOM_REPEAT | exp_name produced |
|---|---|
| *(unset → 50)* | `H16_imf_hardflow_new_K5` ← **unchanged**, all K1/K2/K4/K5 artifacts safe |
| 200 | `H16_imf_hardflow_new_K5_n200` |
| 200 (FM arm) | `H16_1e6steps_hardflow_new_10steps_n200` |

Every earlier invocation stays byte-identical; the n=200 arms land in fresh directories.

---

## 3. Part B — MPC foresight fan (implemented, **DEFAULT OFF**)

Added alongside the blockers, per the §15.5 optional diagnostic. HardFlow computes the planned horizon and terminal prediction at every replan but **discards both** (`run/eval.py:393` uses `_, _`), so only the executed rollout was ever rendered — DPCC/FMPCC by contrast plots the fan explicitly.

**What it draws** (`{run_id}_fan.png`, alongside the usual `{run_id}_real.png`):
- **grey** — planned H-step horizons, one per replan instant (the fan)
- **orange dashed** — the terminal prediction `x̂1` at each replan; for iMF this is the **exact endpoint map** `z + (1−τ)·u` vs FM's Euler shot — i.e. the *visual counterpart of the Gen13 seam swap*
- **black** — the executed trajectory

**Index detail (easy to get wrong):** the executed rollout is observations (`state_dim=4`, x/y at 2,3) while planned/predicted trajectories are full transitions (x/y at `action_dim+2, action_dim+3` = 4,5). Both are handled explicitly.

### ⚠️ Verified OFF at all three layers
Because this run decides Gen13's central claim, "off by default" was **verified, not assumed**:

| Layer | Check | Result |
|---|---|---|
| Config | `ImfEvaluationConfig().imf_plot_fan` | **`False`** ✅ |
| Run script | simulated the exact arm-A invocation (`RANDOM_REPEAT=200`, `IMF_PLOT_FAN` unset) | `fan_flag=[<EMPTY>]` ✅ |
| Sbatch | `grep -c IMF_PLOT_FAN eval_paired_n200_hardflow.sh` | **0 occurrences** ✅ |

With the flag off, the capture branch is skipped entirely — no `x_chain` retention, no plotting, no extra cost — so **the n=200 paired run behaves byte-identically to the verified pre-u_5 code.**

**Functionally tested** (not just syntax): rendered a fan figure from synthetic data in the minimal CPU venv and inspected the image — planned fan, terminal predictions, executed path, obstacles, target region and legend all draw correctly.

**To use it later** (a separate, small diagnostic run — never on the decisive one):
```bash
IMF_PLOT_FAN=1 RANDOM_REPEAT=5 IMF_K=5 bash run_scripts/eval_hardflow_new_imf.sh
# -> logs/.../H16_imf_hardflow_new_K5_n5/{0..4}_fan.png
```

---

## 4. Files changed

| File | Type | Change |
|---|---|---|
| `HardFlow/run_scripts/eval_hardflow_new_paired.sh` | 🆕 new | FM arm B — additive clone with n/exp_name/print-level parametrized |
| `Slurm_Codes/sbatch/hardflow/eval_paired_n200_hardflow.sh` | 🆕 new | the paired-run job (both arms, one submission) |
| `HardFlow/run_scripts/eval_hardflow_new_imf.sh` | ✏️ | `RANDOM_REPEAT` knob, guarded exp_name suffix, `IMF_PLOT_FAN` knob (off) |
| `HardFlow/run_scripts/eval_original_imf.sh` | ✏️ | same three |
| `HardFlow/run/eval_imf.py` | ✏️ | `_save_foresight_fan()`; capture `x_chain`/`x1_est` **only when enabled** |
| `HardFlow/hardflow/models_flow/imf/imf_config.py` | ✏️ | `imf_plot_fan: bool = False`, `imf_fan_every: int = 1` |

**No-edit rule verified:** `git status` confirms `run_scripts/eval_hardflow_new.sh` and `run/eval.py` are **untouched**. Only Gen13-owned files were modified.

---

## 5. What the new sbatch does

`eval_paired_n200_hardflow.sh` — 1 GPU, 32G, **`--time=02:00:00`**:

1. **Guards** — fits dynamics if absent; **aborts** if either checkpoint is missing (iMF cp 4, FM cp 20) *before* burning any GPU time.
2. **Arm A** — iMF `hardflow_new_imf`, K=5, n=200.
3. **Arm B** — FM `hardflow_new`, n=200. ⚠️ `run/eval.py` is pre-existing and still uses the noisy tqdm `run_env` (fix_4 only cleaned the iMF path), which at n=200 would flood the job log — so **arm B's verbose output is redirected to a side file**, with only the tail (or 40 lines on failure) echoed. The CSV is unaffected.
4. **Summary** — prints a per-arm tally (n, safe%, violations, nlp_failures) at the end so the headline numbers are visible without opening the CSVs.

**Walltime rationale:** measured n=50 runtimes were iMF K5 = 3m44s, FM ≈1.74× slower per plan → ~15 min + ~26 min ≈ **45 min expected** at n=200. Requested **2h** per the standing 2×-safety-margin rule (24h partition cap).

---

## 6. ⭐ THE COMMAND TO RUN

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_paired_n200_hardflow.sh
```

That's it — one job, both arms, ~45 min expected.

**Optional knobs** (defaults are what §15 specifies, so you shouldn't need them):
```bash
N=400   ./Slurm_Codes/submit.sh ...   # even more power
ARMS="A" ./Slurm_Codes/submit.sh ...   # iMF arm only
ARMS="B" ./Slurm_Codes/submit.sh ...   # FM arm only
IMF_K=4  ./Slurm_Codes/submit.sh ...   # different K
```

### Outputs to collect afterwards
```
logs/hardflow/avoiding-v0/eval/H16_imf_hardflow_new_K5_n200/trajectories.csv        ← arm A
logs/hardflow/avoiding-v0/eval/H16_1e6steps_hardflow_new_10steps_n200/trajectories.csv  ← arm B
```
Send me both and I'll run the paired statistics (Fisher + Bayesian posteriors + CIs) and write the final verdict into `INSIGHTS_Gen13_first_run.md`.

---

## 7. Free regression check — please glance at this

`env.set_seed(cfg.seed)` is called **once** before the episode loop and episodes run sequentially, so **the first 50 episodes of each 200-episode run should reproduce the frozen 50-episode results exactly** (iMF K5: 1 violation; FM: 0 violations, in the same run_ids).

If they don't match, something non-deterministic changed between runs and the new numbers should not be trusted until that's explained. This costs nothing to check and would catch a silent config drift.

---

## 8. Interpretation guide (decision tree, from §15.2)

| Outcome at n=200 | Meaning | Next action |
|---|---|---|
| iMF ≈ FM (CIs overlap, non-inferiority met) | **Pareto win** — equal safety at ~half the cost | Declare the Gen13 result and write it up |
| iMF clearly worse (p<0.05) | Real mechanistic gap (coarse-field floor) | Go to the §15.4 levers — **not** more K |
| **Both** nonzero (e.g. 4 vs 3) | Neither method is a closed-loop hard guarantee | Reframe the plan §5 criterion as a *rate with CI*, not "0 violations" |
| iMF better than FM | Unexpected — suspect a confound | Re-check seeds/config before believing it |

Reminder of what is **already settled** and cannot change with this run: **efficiency**. iMF K5 uses 21 vs 41 NFE and 0.487 vs 0.847 s/plan with **zero distribution overlap** (FM's fastest episode is slower than iMF's slowest). This run is purely about safety.
