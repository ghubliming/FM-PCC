# Does the low-K HardFlow warning actually reach the reader, and does the data confirm the degeneracy?

**Date:** 2026-08-30 · **Type:** audit + data check (no run, no code changed, no result edited)
**Scope:** Gen15 UAV (`mix_uav` / `mix_uav_test`), warning plumbing audited across all five HF ports.
**Trigger:** "why in low K does the HF still run? is there any warning, and where? and does the data
degenerate — does it yield similar results to DPCC?"
**Evidence:** `temp/3008/` — the K=1 pillars/mf run
`Emf_K1_mpc4_pid_stopgo_T0.5/6/`, the sweep aggregate `batch_uav_20260830_110536/`,
and the cluster stdout logs `2026-08-27/*.log`.
**Companions:** [`REGISTER_20260824_degenerate_HF_rows_and_warnings.md`](./REGISTER_20260824_degenerate_HF_rows_and_warnings.md)
(which rows are affected) · [`CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md`](./CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md)
(the code that emits the flag) · [`../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md`](../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md)
(the derivation).

---

## 0. Answers, up front

1. **It runs at K=1 by design, not by bug.** The terminal ODE step is force-activated, so there is
   always ≥1 NLP solve. At K=1 that solve is the *only* solve, hence `n_genuine = 0`.
2. **Yes, there are warnings — three of them, all present and correct.** Cluster stdout, the
   per-variant `results.json`, and the DA loader log. Exact locations in §2.
3. **But two places where a reader would plausibly look have no warning at all:** the per-variant
   `eval_<variant>.log`, and every wide DA table used for ranking. §3.
4. **The data confirms degeneracy.** At K=1/2 HardFlow is a coin flip against DPCC (wins 1 of 7
   corridor S&C comparisons, mean Δ **−0.057**); at K≥5 the sign flips and stays flipped
   (mean Δ **+0.10 / +0.125**, violations **−14.5 / −15.9**). §4.
5. **Recommendation: gate low-K HF off. There is no sufficient reason to keep running it.** The
   ablation it was supposed to provide does not survive inspection — 25 of its 32 cells are
   `0.00 → 0.00` floor effects, and the clean version of that ablation is a *different run*
   (A=0.0 at matched K) that does not exist in the corpus. §6.

---

## 1. Why the sampler still runs at K = 1

The activation gate is

```
active(k)  ⇔  k >= int((1 - A) * K)   OR   k == K - 1
                                      ^^^^^^^^^^^^^^^^
                              the terminal step is ALWAYS active
```

That `or` clause is deliberate — the terminal solve is what discharges the paper's safety
proposition `h(x_N) ≤ 0`. Consequence:

```
n_active  = max(K - int((1 - A) * K), 1)      # >= 1, always
n_genuine = n_active - 1                       # the non-terminal (= real HardFlow) steps
```

At `k = K-1` the flow time is `tau_next == 1.0` exactly, which independently kills all three
HardFlow ingredients — endpoint lookahead `(1 - tau_next) = 0`, damped pull-back `tau_next = 1`
(a full snap), and feedback (no step `k+1` exists). So at K=1 the sampler executes
`Π_S(Euler sample)`: **sample-then-project — DPCC's algorithm modulo solver and variable scope.**

Code: `mix_uav/sampling/hardflow_projection.py` → `hardflow_step_budget()` / `hardflow_regime()`.
Mirrored verbatim in `flow_matcher_v3_meanflow`, `flow_matcher_v3_alphaflow`,
`flow_matcher_v3_hardflow`, `mix_visual_aligning`.

Three regimes, not two:

| tier | `n_genuine` | meaning | at A = 0.5 |
|---|---|---|---|
| ❌ `DEGENERATE` | 0 | no HardFlow arithmetic runs at all | K = 1, 2 |
| ⚠️ `THIN` | 1 | one nudge, carrying that K's largest lookahead — inside seed noise | K = 3, 4 |
| ✅ `OK` | ≥ 2 | attributable | K ≥ 5 |

---

## 2. Warning coverage — the three layers that DO fire

### ① Cluster stdout (the loud one)

Printed once per `(K, A)` per policy instance, from
`mix_uav/sampling/hardflow_projection.py:831-850`.

```bash
grep '\[hardflow\]\[DEGENERATE\]' <slurm log>
```

Confirmed firing — `temp/3008/2026-08-27/18_34_44_eval_mix_uav_25131.log:264`:

```
[hardflow][DEGENERATE] K=1 A=0.5: n_active=1, n_genuine=0 — every NLP solve is the terminal
tau=1 solve, so this arm runs Pi_S(Euler sample): sample-then-project, == DPCC modulo
solver/variable-scope, NOT HardFlow. The result is still SAFE and still worth having as a
one-shot-projection comparison — just do NOT label it a HardFlow result.
[hardflow][DEGENERATE] first non-degenerate: K>=3 at A=0.5 or K>=2 at A=1.0; for an
attributable effect use n_genuine>=2 — K>=5 at A=0.5 ...
```

It fires separately for each variant instance (`hardflow_new`, `-r`, `-c`, `-t`), because each
builds its own sampler — verified at lines 264, 288, 308 of that log.

### ② Per-variant `results.json` (machine-readable)

Written at `mix_uav_test/eval_mix_uav.py:1894-1912`:

```
<variant>/results.json → summary.hardflow.{ n_active, n_genuine, is_degenerate }
```

Confirmed in the run under investigation
(`Emf_K1_mpc4_pid_stopgo_T0.5/6/pillars_.../hardflow_sls/results.json`):

```json
"hardflow": { "is_hardflow": true, "n_active": 1, "n_genuine": 0, "is_degenerate": true,
              "activation_threshold": 0.5, "nlp_backend": "slsqp", ... }
```

### ③ DA loader log

`Data_Analysis/DA_UAV_v1/data_loader.py:337-356` reads the field, and **derives** it for the
pre-2026-08-24 corpus that predates it (`n_active = max(K - int((1-A)*K), 1)`). Emits:

```bash
grep DEGENERATE batch_*/logs/loading.log
```
```
[WARN] cand42/seed6/pillars_bounds+dynamics+geo_bounds+obstacles/hardflow_sls:
HardFlow arm is DEGENERATE (K=1, A=0.5, n_genuine=0) — this row is sample-then-project,
not HardFlow. Do not label it a HardFlow result. See logs_in_develop/aggregated_hardflow_lowK/
```

Confirmed in `batch_uav_20260830_110536/logs/loading.log` — 20+ rows flagged across cand29/30/33/
35/38/40/42.

**Summary table — where to look manually:**

| # | Layer | Present? | Path | Grep |
|---|---|---|---|---|
| ① | cluster stdout | ✅ | `Slurm_Codes/logs/.../<job>.log` | `\[hardflow\]\[DEGENERATE\]` |
| ② | per-variant artifact | ✅ | `<variant>/results.json` | `is_degenerate` |
| ③ | DA loader | ✅ | `batch_*/logs/loading.log` | `DEGENERATE` |
| ④ | per-variant eval log | ❌ | `<variant>/eval_<variant>.log` | — |
| ⑤ | file-tree sentinel | ❌ | `<variant>/HF_DEGENERATE.txt` | — |
| ⑥ | wide DA / ranking tables | ❌ | `uav_k_sweep.csv`, `candidates_*.csv` | — |
| ⑦ | `<variant>.npz` | ❌ | — | — |

---

## 3. The two real gaps

### Gap A — `eval_<variant>.log` has no banner

`mix_uav_test/eval_artifacts.py:741-770` writes loud `!!!!!!` banners for **two** failure modes:

* `PROJECTION CIRCUIT-BREAKER TRIPPED` (Fix_15.3)
* `DIVERGENCE ABORT` (Div_Abort)

…and **nothing** for HF degeneracy. The K=1 log under investigation demonstrates it exactly — it
opens with a divergence banner and never mentions HardFlow:

```
======================================================================
UAV FM eval  |  scene=pillars  seed=6  variant=hardflow_sls
======================================================================
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !!! DIVERGENCE ABORT on 1/10 trials: [1]
  ...
```

Same for the file tree: `DIVERGENCE_ABORT.txt` and `PROJECTION_CB_TRIPPED.txt` sentinels exist,
`HF_DEGENERATE.txt` does not. `write_eval_log` is also the reason `.npz` carries nothing —
`eval_artifacts.py` never sees the `hardflow` block.

### Gap B — the flag dies before the ranking tables

`hf_degenerate` / `n_genuine` survive into the **long** table only:

| CSV in `batch_uav_20260830_110536/` | carries `hf_degenerate`? |
|---|---|
| `uav_units_long.csv` | ✅ (as `metric` rows) |
| `uav_k_sweep.csv` | ❌ |
| `uav_aggregated_long.csv` | ❌ |
| `candidates_ranking.csv` | ❌ |
| `candidates_per_variant.csv` | ❌ |
| `candidates_detailed.csv` / `_multidimensional_*` | ❌ |
| `data_quality.csv` | ❌ |

So the exact artifacts a DA reads to build a Pareto/ranking claim are the ones with no degeneracy
column. This is the mechanical root of the failure this register was created for: a K=1 row can be
promoted to "HardFlow's best result" without any table on the path saying otherwise.

---

## 4. Does the data actually degenerate? — yes

### 4.1 The run in question: K=1, pillars, mf, seed 6

`logs/UAV_MIX/uav-pillars/plans/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/Emf_K1_mpc4_pid_stopgo_T0.5/6/`
· n = 10 trials · A = 0.5 · backend = slsqp · **all HF rows `n_genuine = 0` ❌**

| variant | succ | **S&C** | safe | coll-free | n_viol | total_viol | goal_dist | steps | proj_ms | aborts |
|---|---|---|---|---|---|---|---|---|---|---|
| diffuser | 0.00 | 0.00 | 0.00 | 0.00 | 28.5 | 16.27 | 6.503 | 634.0 | 0.0 | 10 |
| dpcc-r | 0.30 | **0.00** | 0.30 | 0.00 | 66.2 | 20.44 | 1.437 | 574.9 | 75.3 | 5 |
| dpcc-c | 0.10 | **0.00** | 0.10 | 0.00 | 50.3 | 7.01 | 0.926 | 585.1 | 75.9 | 3 |
| dpcc-t | 0.00 | **0.00** | 0.00 | 0.00 | 67.9 | 18.60 | 2.463 | 614.4 | 99.4 | 8 |
| ❌ hardflow_sls | 0.10 | **0.00** | 0.10 | 0.00 | 42.6 | 3.35 | 0.591 | 546.3 | 16.9 | 1 |
| ❌ hardflow_sls-r | 0.20 | **0.00** | 0.20 | 0.00 | 50.1 | 4.09 | 0.292 | 521.2 | 93.5 | 0 |
| ❌ hardflow_sls-c | 0.00 | **0.00** | 0.00 | 0.00 | 66.3 | 15.39 | 1.654 | 592.0 | 92.4 | 7 |
| ❌ hardflow_sls-t | 0.20 | **0.00** | 0.30 | 0.00 | 57.3 | 9.97 | 2.191 | 595.9 | 87.3 | 6 |

**`success_and_constraints = 0.00` and `collision_free_rate = 0.00` for every single arm**, DPCC and
HF alike. The success-rate spread (0.00–0.30) is entirely inside seed noise at n = 10
(s.e. ≈ 0.15), and its sign is random across the three suffixes: HF−DPCC = −0.10 (`-r`),
−0.10 (`-c`), +0.20 (`-t`). **Pillars at K=1 carries no signal for anyone** — it cannot separate
HF from DPCC because it cannot separate anything from anything.

### 4.2 The sweep: where the separation actually appears

Paired HF − DPCC deltas over matched `(scene, engine, suffix)` cells, `batch_uav_20260830_110536`,
`mask=all`, geo_free arms excluded. Corridor is the discriminating scene (pillars is floored at
S&C = 0 for every arm and every K, see 4.1).

| K | regime | mean Δ S&C | HF better (S&C) | mean Δ n_violations | cells |
|---|---|---|---|---|---|
| 1 | ❌ degenerate | **−0.057** | **1 / 7** | +3.2 | 7 |
| 2 | ❌ degenerate | **−0.057** | **1 / 7** | +13.2 | 7 |
| 5 | ✅ genuine (`n_gen`=2) | **+0.100** | 4 / 7 | **−14.5** | 7 |
| 10 | ✅ genuine (`n_gen`=4) | **+0.125** | 2 / 4 | **−15.9** | 4 |

The sign flip lands exactly on the `n_genuine ≥ 2` boundary, on both axes independently. Spelled
out on corridor/mf, the clearest cell:

| K | S&C dpcc → HF | n_violations dpcc → HF |
|---|---|---|
| 1 ❌ | 0.80 → 0.50 (`-c`) | 56.8 → 111 |
| 2 ❌ | 0.80 → 0.50 (`-c`) | 71.2 → 114 |
| 5 ✅ | 0.60 → **0.80** | 41.3 → **0.9** |
| 10 ✅ | 0.60 → **1.00** | 40.6 → **0.0** |

At K ≥ 5 HardFlow drives corridor violations to essentially zero while DPCC plateaus at 20–40.
At K ≤ 2 it does not, and is marginally worse. **This is the degeneracy, measured.**

> ⚠️ Timing must not be read off this table. The corridor rows are `hardflow_new` (IPOPT) while the
> pillars K=1 rows are `hardflow_sls` (SLSQP) — a 4.3× per-solve difference. Any `avg_time_ms`
> comparison across those cells is a backend comparison wearing a K label.

### 4.3 Side finding — SLSQP non-convergence on the degenerate arm

From the same run's `summary.hardflow`:

| variant | nlp_solves | nlp_failures | rate | nfe/plan |
|---|---|---|---|---|
| hardflow_sls | 5 345 | 408 | **7.6 %** | 0.98 |
| hardflow_sls-r | 20 848 | 2 037 | **9.8 %** | 4.00 |
| hardflow_sls-c | 19 916 | 3 043 | **15.3 %** | 3.36 |
| hardflow_sls-t | 16 780 | 2 419 | **14.4 %** | 2.82 |
| hardflow_sls-*-geo_free | 23 720 – 25 360 | 0 | **0.0 %** | 4.00 |

Two things follow. **(a)** On a degenerate arm *every* solve is the terminal solve, so each failure
is a plan whose safety guarantee does not hold — the run keeps scipy's last iterate, which may be
infeasible. **(b)** The `geo_free` arms fail 0 % of the time, which localises the difficulty to the
obstacle constraints, not to the NLP size. `-c` selects candidates by projection cost, so at 15.3 %
failure it is partly ranking candidates on costs computed from infeasible iterates.

This is orthogonal to degeneracy but shares its blast radius, and is worth its own look at K ≥ 5.

---

## 5. Verdict

* The warning exists, is correct, and fires — at three layers.
* It is invisible at the two layers a human actually browses (`eval_*.log`, the ranking CSVs).
* The register's rule is upheld on Gen15 UAV data: **K ≤ 2 HF rows carry no HardFlow signal and
  must not carry a HardFlow claim; K ≥ 5 rows separate from DPCC on both S&C and violations.**
* But the K ≤ 2 rows are **not** a usable equivalence result either — 25 of their 32 cells are
  `0.00 → 0.00` floor effects (§6a), and in the 7 live cells HF is *worse* in 5. They neither
  support a HardFlow claim nor establish that HF's projector matches DPCC's.
* Pillars K=1 specifically is a *doubly* dead cell — degenerate arm on a floored task.
* **Net: the low-K HF rows have no consumer.** Nothing in the corpus cites them, and the one
  ablation they were retained for needs a run that does not exist (`A=0.0` at matched K, §6).

---

## 6. Recommendation — make it loud, and stop generating it

### What low-K HardFlow was supposed to be for — and why that fails

The stated justification was a **confound control**: "HardFlow beats DPCC at K=10" admits the
alternative explanation that HF's NLP/solver is simply a better projector than DPCC's, independent
of K. Degenerate rows run *DPCC's algorithm* (sample-then-project) with *HF's projector*, so if the
projector were the cause they would already win at K≤2. They do not — therefore (the argument goes)
the K≥5 flip is in-loop guidance.

**The argument is sound; the instrument is not.** Three problems, in increasing severity:

**(a) 78 % of the control's cells measure nothing.** Of 32 matched degenerate cells, 25 read
"identical" only because *both arms are 0.00* — pillars is floored for every arm at every K, and
corridor/`fm` is floored at K=1/2. A control that returns `0.00 → 0.00` in 25 of 32 cells has not
controlled for anything.

**(b) Where there IS signal, it is not a null.** The 7 live cells (corridor `af`/`mf`):

| S&C at degenerate K | HF worse | tie | HF better |
|---|---|---|---|
| count | **5** | 0 | 2 |

e.g. corridor/`mf`/`-c` K=1: DPCC 0.80 → HF 0.50, violations 56.8 → 111.1. Violations swing the
other way elsewhere (pillars/`fm`/`-r` K=2: 242.2 → 50.4). No consistent direction, n=10, single
seed. This is noise, with a weak hint that HF's projection machinery is *worse* than DPCC's — not
the clean equivalence §4.2 originally read into it.

**(c) K=1/2 is the wrong operating point for the question.** It varies the projector *and* K
simultaneously. At K=1 the underlying sample is a single Euler step, so the projector comparison is
run where nothing can succeed — which is the direct cause of (a).

**The clean instrument is `A = 0.0` at matched K.** The gate gives
`n_active = max(K - floor((1-0)*K), 1) = 1` — terminal-only at *any* K. So `A=0.0, K=10` isolates
the projector at unchanged K and unchanged sample quality. That is the run that would settle it.

**It does not exist, and currently cannot be launched on the UAV path.** `activation_threshold` has
exactly one value across all 105 UAV rows in `batch_uav_20260830_110536`: **0.5**.
`config/uav_mix.py:247` hardcodes it, and `HFFM_ACT_THRESHOLD` — wired into
`Slurm_Codes/sbatch/MeanFlow/`, `AlphaFlow/` and `mix_visual_aligning/` — is **not** wired into
`Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh`, which only mentions it in a comment at line 34.

> 🔴 **Correction (second revision of this file).** Two successive justifications for keeping low-K
> HF have now failed. The first draft claimed the rows underpinned the SLSQP-vs-IPOPT comparison in
> `logs_in_develop/aggregated_hf_nlp_backend/` — **wrong**: that DA is job 25222 at K in {10,20},
> A=1.0, `hf_degenerate = 0` on every row, and its prior (job 25121) is a per-solve microbenchmark
> needing no eval rows. The second draft claimed the rows were a one-time confound control — **not
> supported by its own data**, per (a)-(c) above. §4.2's "null" is predominantly floor effects and
> must not be cited as an equivalence result.

### Recommendation

**Gate low-K HardFlow off.** No claim in the corpus depends on it, and the one job it was kept for
is better done by a run that does not yet exist. If the projector ablation is wanted, it is a
separate item: wire `HFFM_ACT_THRESHOLD` into the UAV sbatch and run `A=0.0, K=10` against
`A=0.5, K=10` and the DPCC arm — three rows, one seed sweep, at a K where the task is not floored.

> ✅ **All four were implemented on 2026-08-30**, after the user approved the change. See
> [`CHANGELOG_20260830_hardflow_degeneracy_guard.md`](./CHANGELOG_20260830_hardflow_degeneracy_guard.md)
> for what landed where, and for the parts that still need a cluster run. The table below
> is the spec they were built to.

Four changes:

| # | Change | File | Why |
|---|---|---|---|
| R1 | `!!!!` banner in `eval_<variant>.log` + `HF_DEGENERATE.txt` sentinel in the variant dir | `mix_uav_test/eval_artifacts.py::write_eval_log` (mirror the Fix_15.3 / Div_Abort blocks) | closes Gap A — degeneracy becomes visible from the file tree alone |
| R2 | carry `hf_degenerate` / `n_genuine` into `uav_k_sweep.csv` and `candidates_*.csv` | `Data_Analysis/DA_UAV_v1/` aggregation stage | closes Gap B — a ranking can no longer silently promote a degenerate row |
| **R3** | `FMPCC_HF_MIN_K` guard **on by default** (3; 5 for attributable claims) skipping HF variants below it in K-sweeps, `FMPCC_HF_ALLOW_DEGENERATE=1` to override | `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` + siblings | no claim depends on these rows; every low-K HF cell is cluster time that cannot be cited |
| R4 *(optional)* | wire `HFFM_ACT_THRESHOLD` into the UAV sbatch, as MeanFlow/AlphaFlow/visual_aligning already do | `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh`, `config/uav_mix.py:247` | prerequisite for the `A=0.0, K=10` projector ablation — the run that actually answers what low-K HF was kept for |

R1/R2 stop the degenerate rows already on disk from being *misread*; R3 stops new ones being
*generated*. R4 is only worth doing if the projector-vs-guidance attribution is something a paper
claim needs to rest on — decide that before spending on it.

---

## 7. Provenance

* **No code was changed and no result was edited.** Read-only analysis of `temp/3008/`.
* Not verified on the cluster — every number above is read from committed artifacts
  (`results.json`, the DA CSVs, cluster stdout), not recomputed. Nothing here needs a re-run:
  the gate is pure arithmetic.
* Deltas in §4.2 are unpaired-by-seed cell means (n = 10 trials per cell, 1 seed), so they carry
  seed-level confounding. The sign-flip pattern is consistent across two metrics and two engines,
  which is what the claim rests on — not the magnitude of any single delta.
* Reproduce §4.1 / §4.2:
  ```bash
  # 4.1 — per-variant table
  cd temp/3008/Emf_K1_mpc4_pid_stopgo_T0.5/6/pillars_bounds+dynamics+geo_bounds+obstacles
  for d in */; do python3 -c "import json;s=json.load(open('$d/results.json'))['summary'];\
    print('$d', s['success']['strict_and_constraints_rate'], s['constraint']['collision_free_rate'],\
          (s.get('hardflow') or {}).get('n_genuine'))"; done

  # 4.2 — paired deltas by K
  cd temp/3008/batch_uav_20260830_110536   # then group uav_k_sweep.csv on (scene,engine,K,variant)
  ```
