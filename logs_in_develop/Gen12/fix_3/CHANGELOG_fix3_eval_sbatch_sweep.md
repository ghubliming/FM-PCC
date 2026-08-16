# CHANGELOG — Gen12 fix_3: eval sbatch — single by default, sweep on demand

**Date:** 2026-07-24 · **Type:** ergonomics (sbatch) · **Status:** code complete, syntax + logic verified
**Follows:** [`../fix_2/CHANGELOG_fix2_gate_g2_tau_invariance.md`](../fix_2/CHANGELOG_fix2_gate_g2_tau_invariance.md),
[`../fix_2/RESULTS_Gen12_first_eval_K20.md`](../fix_2/RESULTS_Gen12_first_eval_K20.md)
**Nothing committed. Nothing run** (this container has no deps; the sbatch runs on the cluster).

---

## 0. TL;DR

The eval + aggregate sbatch scripts now **default to a single run** (K from the plan block) and
**support a K-sweep** via one env var. No env var to type for the common case; the sweep is opt-in
for the real matched-K experiment (PLAN §5).

## 1. Why

The first eval (fix_2 RESULTS) ran at a single K=20 and only exposed the cost gap — the
interesting **low-K regime** (K ∈ {2,5,10}) was untested. A sweep is needed for the real result.
But fix_1 §8 had simplified the eval sbatch to a *single* `python …` call (no K loop), so a sweep
required hand-typing `--flow-steps` in a login-shell loop (no GPU/EGL) — not a real job.

The user's requirement: **default stays one run; the sbatch should also support a sweep.** So the
loop is back, but gated behind the env var rather than being the default.

## 2. What changed

| file | before (fix_1 §8) | after (fix_3) |
|---|---|---|
| `Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh` | single `python eval…` (K from plan block) | `if HFFM_FLOW_STEPS set → loop --flow-steps per K; else single run` |
| `Slurm_Codes/sbatch/hardflow_fmv3/load_results_hardflow_fmv3.sh` | single aggregate | same gate — default single bucket, sweep reports each |

Logic (both scripts):

```bash
if [ -n "${HFFM_FLOW_STEPS:-}" ]; then
    for K in $HFFM_FLOW_STEPS; do python … --flow-steps "$K"; done   # sweep
else
    python …                                                          # single, K from plan block
fi
```

Nothing else touched. No Python changed — the eval already accepts `--flow-steps` (CLI overrides
the plan block); this just decides when the sbatch passes it.

## 3. Behaviour

| invocation | result |
|---|---|
| `./submit.sh …/eval_fmv3_hardflow_job.sh` | **single run**, K = plan block `flow_steps` (10). No `--flow-steps`, no env var. |
| `HFFM_FLOW_STEPS="10" ./submit.sh …` | single K=10 (explicit) |
| `HFFM_FLOW_STEPS="2 5 10" ./submit.sh …` | sweep; each K → its own `K<K>_n<n>` results dir (matched budget per K, PLAN §5) |
| `FORCE_OVERWRITE=1 …` | re-run a finished bucket (PLAN §3.6) — unchanged |

`load_results` mirrors the same env var, so a swept eval is aggregated with
`HFFM_FLOW_STEPS="2 5 10" ./submit.sh …/load_results_hardflow_fmv3.sh` and reports one table per K.

Because each K writes a distinct dir, a sweep also **doubles as the K-override confirmation**: the
appearance of `K2/K5/K10` dirs (not `K20`) proves the plan-block K / `--flow-steps` override is
live on hardware — the thing fix_2's debug left verified only by local simulation.

## 4. Verification (static)

- Both scripts pass `bash -n`.
- Branch logic simulated: unset → single (no `--flow-steps`); `"10"` → one `--flow-steps 10`;
  `"2 5 10"` → three runs. Correct.
- `set -e` safe: `${HFFM_FLOW_STEPS:-}` handles the unset case.

## 5. How to run next (cluster)

First discard the K=20 config hand-edit (fix_2 debug), then:

```bash
cd ~/FMPCC/FM-PCC
git checkout -- config/avoiding-d3il.py     # committed config already points at the aw10 ckpt, K=10
git pull                                     # get the sweep-capable sbatch

# single run (default, K=10):
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
# or the sweep:
HFFM_FLOW_STEPS="2 5 10" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/eval_fmv3_hardflow_job.sh
```

Still smoke-scale (`n_trials=2`, seed 6). For a real result raise `n_trials` (≥ ~34 on seed 6 → n≈100
across the 3 halfspace variants) and, once more seeds are trained, add them in the hardflow yaml.
