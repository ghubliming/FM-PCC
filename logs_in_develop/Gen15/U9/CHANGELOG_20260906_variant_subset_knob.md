# Gen15 U9 — `UAV_MIX_VARIANTS`: per-job variant subset

**Why now:** [`DA_20260906_U7_honest_geometry_first_results.md`](../DA/DA_20260906_U7_honest_geometry_first_results.md)
§12.4 identified the missing knob as the blocker for the `pillars` K=5 HardFlow test.

## 0. TL;DR

A K-sweep job runs `len(projection_variants) × n_trials` rollouts. At **K ≥ 3** the HardFlow arm
re-enables and the set goes **10 → 17**. Both `pillars` K=5 jobs in the Fix_16 A/B died at the
24 h wall at 17 variants, and the only way to trim was editing the shared yaml — which silently
changes every other job. `UAV_MIX_VARIANTS` makes it a per-job, read-only choice.

**Effect on the runs this unblocks: 17 → 8 variants, 53 % less work, comparison intact.**

## 1. Usage

```bash
UAV_MIX_VARIANTS='diffuser,dpcc-r,dpcc-c,dpcc-t,hardflow_new,hardflow_new-r,hardflow_new-c,hardflow_new-t'
```

Empty / unset = run everything, i.e. current behaviour. Nothing changes for existing jobs.

## 2. Where it sits

`mix_uav_test/eval_mix_uav.py`, immediately after the HardFlow concat (~line 622) — the
read-only counterpart to the existing `UAV_MIX_HF_OFF` a few lines above.

It filters the **already-assembled** list (yaml `projection_variants` + `config/uav_mix.py`
`hardflow_variants`), which is deliberate:

* it can select across both families in one string;
* it can never invent a variant the eval does not implement;
* it sees the post-guard list, so a name blocked by the degeneracy guard is reported as such
  rather than silently ignored.

## 3. Two guards

| condition | behaviour |
|---|---|
| a name is not in the assembled list | `ERROR`, prints the available set, `exit 2` — plus a hint naming `UAV_MIX_HF_OFF` / low K if the unknown names are `hardflow_*` and the HF arm is off |
| the subset keeps `hardflow_*` but **no** `dpcc-*` | `ERROR`, `exit 2` |

🔴 The second guard is the important one. HardFlow-vs-DPCC needs **both arms at the same K** —
comparing a HardFlow row at K=5 against a `dpcc` row from a K=2 job breaks the matched-budget
rule the sbatch headers already enforce for K. "Only the HF solver" is the natural thing to ask
for and the one thing that silently produces an uncitable result, so it is refused rather than
warned about. `UAV_MIX_HF_OFF=1` remains the way to run the DPCC side alone.

Both exit **before** any model loads, so a bad string costs seconds, not hours.

## 4. What the trim drops, and why it is safe

| dropped | n | evidence |
|---|---|---|
| `dpcc-{r,c,t}-tightened` | 3 | within noise of their base siblings on `pillars` (0.40 vs 0.40, n=10) — DA §3 |
| `dpcc-{r,c,t}-geo_free` | 3 | now the **worst** group under honest geometry (0.20–0.30 vs 0.40–0.70) — DA §3 |
| `hardflow_new-{r,c,t}-geo_free` | 3 | same ablation, same reason |

Kept: `diffuser` (unprojected reference), `dpcc-{r,c,t}` (PCC arm), `hardflow_new` +
`hardflow_new-{r,c,t}` (HF arm) — **matched `r`/`c`/`t` selection rules on both sides**, which is
what the HF-vs-PCC question needs.

## 5. Files changed

| file | change |
|---|---|
| `mix_uav_test/eval_mix_uav.py` | the filter + two guards (~40 lines incl. comments) |
| `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` | `export UAV_MIX_VARIANTS` + echo |
| `Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh` | `export UAV_MIX_VARIANTS` + echo |

`sbatch` defaults to `--export=ALL` so the var would ride along regardless; naming it explicitly
follows the U6 doctrine — the job log alone must say what ran.

## 6. Verification

* `python3 -m py_compile mix_uav_test/eval_mix_uav.py` — OK
* `bash -n` on both sbatch scripts — OK
* Filter semantics simulated on 5 paths:

  | input | result |
  |---|---|
  | the 8-variant trim at K≥3 (17 available) | ✅ `8/17 kept, 9 dropped` |
  | same string at K=2 (10 available, HF off) | ✅ `exit 2`, unknown `hardflow_*`, + HF-off hint |
  | HardFlow-only | ✅ `exit 2`, refused |
  | typo `dpcc-rr` | ✅ `exit 2`, unknown |
  | DPCC-only (`diffuser,dpcc-r,dpcc-c,dpcc-t`) | ✅ `4/17 kept` |

* 🔴 **Not yet run on the cluster.** First real check is the log line
  `[ eval ] UAV_MIX_VARIANTS -> running 8/17 variants: [...]` followed by
  `[ eval ]   dropped 9: [...]`.
