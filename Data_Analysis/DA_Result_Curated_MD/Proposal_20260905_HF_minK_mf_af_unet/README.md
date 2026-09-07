# Proposal — the minimum K at which HardFlow-SLSQP actually runs, and the MF/AF-UNet K-ladder that tests it

**Date:** 2026-09-05 · **Status:** proposal, nothing submitted · **Task:** `avoiding-d3il` (state, H8)
**Gens:** Gen3v6 (MeanFlow, `flow_matcher_v3_meanflow`) · Gen3v7 (α-Flow, `flow_matcher_v3_alphaflow`) · arm C = the HardFlow sampler ported into both
**Answers:** "what is the min K to let HF-SLSQP start to *really* run?" and "has this already been run in `temp/0309/batch_avoiding_combined_20260903_133730`?"

---

## 0. Bottom line

1. **The SLSQP swap has nothing to do with the floor.** `slsqp` vs `ipopt` is only the NLP
   backend (`FMPCC_HF_NLP_BACKEND`, default `slsqp` since the 3.88× swap). Whether HardFlow's
   *algorithm* runs is decided by the activation gate `k >= int((1-A)*K) or k == K-1`, which is
   solver-agnostic. So "min K for HF-SLSQP" == "min K for HardFlow", full stop.

2. **The floor is a (K, A) pair, not a K.** With `n_genuine = max(K − int((1−A)·K), 1) − 1`:

   | activation `A` | first K with *any* HardFlow step | first K with a *citable* ladder (`n_genuine ≥ 2`) |
   |---|---|---|
   | 0.0 | never | never |
   | **0.5** *(shipped default)* | **K = 3** (thin, lookahead 0.333) | **K = 5** |
   | 0.75 | K = 2 (thin, lookahead 0.500) | K = 3 |
   | **1.0** | **K = 2** (thin, lookahead 0.500) | **K = 3** (lookahead 0.667, 0.333) |

   **K = 1 is impossible at every A** — the only step is the terminal step, so lookahead is 0,
   the pull-back gain is 1, and no successor step ever sees the correction. There is no knob
   that fixes this.

3. **Therefore the honest answer to the question as asked:** the absolute floor is
   **K = 2 with `HFFM_ACT_THRESHOLD=1.0`**, and the floor at which a result may be *cited as a
   HardFlow finding* is **K = 3 @ A = 1.0** or **K = 5 @ A = 0.5**.

4. **`temp/0309/batch_avoiding_combined_20260903_133730` does not answer this.** It has 195
   candidate rows, 21 of them HardFlow — and **all 21 are `flow_matching_v3_hardflow`
   (Gen12, the naive-FM FMv3ODE engine)**. There is **not one `mf`/`af` HardFlow row** in that
   batch. The MF/AF × arm-C cell on `avoiding-d3il` is empty. This test is new.

---

## 1. Why K=1/K=2 cannot be a HardFlow result (the derivation, one screen)

Source of truth: `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:584-646`
(`hardflow_step_budget`, `hardflow_regime`), matching
`logs_in_develop/HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md` §0.1 and
`logs_in_develop/aggregated_hardflow_lowK/REGISTER_20260824_degenerate_HF_rows_and_warnings.md`.

A step does real HardFlow work only if it is **active** *and* **not the last step**:

```
n_active  = max(K − int((1 − A)·K), 1)      # `or k == K-1` forces the terminal solve
n_genuine = n_active − 1
```

At the terminal step `τ⁺ = 1` exactly, and all three HardFlow mechanisms vanish at once:
the lookahead `(1 − τ⁺)·f(X_ref, τ⁺)` is 0, the pull-back gain `τ⁺` is a full snap rather than
a damped nudge, and there is no step `k+1` for the network to react on. What executes is
`Π_S(Euler sample)` — sample-then-project, i.e. DPCC's algorithm with a different solver.

`n_genuine == 0` rows are **not** "a weak HardFlow"; they are **not HardFlow at all**. The eval
already enforces this: `FMPCC_HF_ALLOW_DEGENERATE` is unset by default, so a degenerate arm C is
**dropped** and an `HF_DEGENERATE_SKIPPED.txt` sentinel is written instead of burning GPU
(`Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh:113-122`).

**Consequence for the requested experiment:** a "K1/K2 mf/af_unet HardFlow" run at the shipped
`A = 0.5` produces **zero HF rows** — the guard skips both. The experiment only exists if `A` is
raised. That is the design below.

### 1.1 Full regime table for the grid proposed here

| A | K | n_active | n_genuine | first lookahead | tier |
|---|---|---|---|---|---|
| 0.5 | 1 | 1 | 0 | — | ❌ DEGENERATE (skipped by the guard) |
| 0.5 | 2 | 1 | 0 | — | ❌ DEGENERATE (skipped by the guard) |
| 0.5 | 3 | 2 | 1 | 0.333 | ⚠️ THIN |
| 0.5 | 5 | 3 | 2 | 0.400 | ✅ OK |
| 1.0 | 1 | 1 | 0 | — | ❌ DEGENERATE (skipped) |
| 1.0 | 2 | 2 | 1 | 0.500 | ⚠️ THIN — **the floor** |
| 1.0 | 3 | 3 | 2 | 0.667 | ✅ OK |
| 1.0 | 5 | 5 | 4 | 0.800 | ✅ OK |

Arms A (`diffuser`) and B (`dpcc-*`) are unaffected by `A` — they run at every K including K=1,
so the **K=1 column still exists for the PCC comparison**; it just has no arm C in it.

---

## 2. What this test is for

Two questions, and they must not be conflated:

- **Q1 (the PCC question the user asked).** At the low budgets where MF/AF-UNet actually live
  (K = 1, 2), how do the *guidance arms* rank on `avoiding-d3il` for the **UNet** backbone —
  unguided vs DPCC projector vs HardFlow? This is the architecture-matched comparison; the UNet
  row is the one that carries the claim, DiT/SiT/mf_dit rows are confounded on parameter count.
- **Q2 (the HardFlow floor).** Does HardFlow's in-loop guidance buy anything over the DPCC
  projector *once it genuinely runs*, and how far down in K does that survive?

Q1 is answered at K ∈ {1, 2} with arms A/B only (arm C is structurally absent at K=1, and at K=2
only under `A = 1.0`). Q2 needs the `A = 1.0` ladder K ∈ {2, 3, 5}, where the genuine-step count
climbs 1 → 2 → 4.

### 2.1 Who must beat whom (benchmark hierarchy)

- Target = the best projection variant of the pinned **DPCC K20 / aw10 / GaussianDiffusion**
  baseline. Beating it on any axis with success + constraints held is a win.
- MF and AF must also beat **naive FM** (Gen7/FMv3ODE) at matched K, matched backbone.
- **HardFlow's own bar is the DPCC projector**, and it must clear it on ✅ rows only.
- "Good" = **Pareto-dominant**: at equal success + constraint satisfaction, fewer steps **and**
  lower avg time. Otherwise say *trade-off* / *non-dominated* — never "best".

---

## 3. Design

| axis | setting | why |
|---|---|---|
| task | `avoiding-d3il` (state, H8) | matches the whole existing corpus; the batch in `temp/0309` is this task |
| backbone | **`unet`** for both engines (`MF_BACKBONE=unet`, `AF_BONE=unet`) | architecture-matched to the DPCC UNet baseline; the defaults are `mf_dit` / `sit` and would silently confound |
| engines | MeanFlow (Gen3v6) and α-Flow (Gen3v7) | the two live low-NFE arms |
| arms | A `diffuser` · B `dpcc-{r,c,t}[-tightened]` · C `hardflow_new[-c-tightened]` | one run, shared K/seeds/env resets by construction |
| K | **{1, 2}** at A=0.5 (arms A/B; the PCC question) and **{2, 3, 5}** at A=1.0 (arm C alive) | §1.1 |
| activation A | `0.5` (reference) and `1.0` (floor probe) | `A` is a folder-name token, so the two do **not** collide |
| fan | `HFFM_BATCH=4` **and** `FMPCC_MPC_BATCH=4` | B4_PARITY — an unequal fan is a 4× compute discount for arm C that reads as a HardFlow speedup and voids every timing comparison |
| solver | `slsqp` (default) | the shipped backend; `_slsqp` is appended to the artifact name so the IPOPT corpus is not overwritten |
| seeds | 6 first (smoke), then 7–10 | cheap fail-fast before spending the ladder |
| trials | yaml default 2 for the smoke, 20 for the powered run | ntrials-2 rows are for wiring checks only, never for a claim |

**Degeneracy guard stays ON.** Do **not** set `FMPCC_HF_ALLOW_DEGENERATE=1`. If a cell is
degenerate we want the sentinel file, not a row that will later be mis-cited. The one legitimate
use of that flag is the projector-only control (`A = 0.0`, `K ≥ 5`), which is not part of this test.

---

## 4. Commands

All run from the repo root on **i6-gpu-1**. Submit with `./Slurm_Codes/submit.sh`.

### 4.1 Step 0 — the free sanity check (no GPU, runs anywhere)

Confirms the floor arithmetic against the shipped code before spending a job:

```bash
python - <<'EOF'
from flow_matcher_v3_meanflow.sampling.hardflow_projection import hardflow_regime
for A in (0.5, 1.0):
    for K in (1, 2, 3, 5, 10, 20):
        print(A, K, hardflow_regime(K, A))
EOF
```

### 4.2 MeanFlow-UNet — smoke, seed 6, the A=1.0 floor ladder

`eval_meanflow_hardflow.sh` already loops the K grid internally (`MF_FLOW_STEPS`), so this is
**one job** for all three budgets.

```bash
MF_BACKBONE=unet MF_HORIZON=8 \
HFFM_ACT_THRESHOLD=1.0 MF_FLOW_STEPS="2 3 5" \
HFFM_BATCH=4 FMPCC_MPC_BATCH=4 \
FMPCC_HF_NLP_BACKEND=slsqp \
FMPCC_RUN_MSG=hfmink_A1_mfunet_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

### 4.3 MeanFlow-UNet — the A=0.5 reference incl. the K=1/K=2 PCC cells

Arm C is skipped at K=1/2 here **by design** (sentinel written); the value of this job is the
matched arms A/B rows at the budgets MF-UNet actually operates at, plus the K=5 ✅ HF row that
lets A=0.5 and A=1.0 be compared at equal K.

```bash
MF_BACKBONE=unet MF_HORIZON=8 \
HFFM_ACT_THRESHOLD=0.5 MF_FLOW_STEPS="1 2 3 5" \
HFFM_BATCH=4 FMPCC_MPC_BATCH=4 \
FMPCC_HF_NLP_BACKEND=slsqp \
FMPCC_RUN_MSG=hfmink_A05_mfunet_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
```

### 4.4 α-Flow-UNet — same two ladders

⚠️ **`Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh` has no K loop** (unlike its
MeanFlow sibling — the comment in the MF script claiming Gen3v7 "has had the grid loop from day
one" is stale). It is a single `python …eval_flow_matching_v3_alphaflow.py` call driven by
`HFFM_FLOW_STEPS`. So α-Flow needs **one submit per K**:

```bash
for K in 2 3 5; do
  AF_BONE=unet AF_EPOCH=latest AF_ALPHA_END=0.2 \
  HFFM_ACT_THRESHOLD=1.0 HFFM_FLOW_STEPS=$K \
  HFFM_BATCH=4 FMPCC_MPC_BATCH=4 \
  FMPCC_HF_NLP_BACKEND=slsqp \
  FMPCC_RUN_MSG=hfmink_A1_afunet_s6_K$K \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh
done

for K in 1 2 3 5; do
  AF_BONE=unet AF_EPOCH=latest AF_ALPHA_END=0.2 \
  HFFM_ACT_THRESHOLD=0.5 HFFM_FLOW_STEPS=$K \
  HFFM_BATCH=4 FMPCC_MPC_BATCH=4 \
  FMPCC_HF_NLP_BACKEND=slsqp \
  FMPCC_RUN_MSG=hfmink_A05_afunet_s6_K$K \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh
done
```

*Optional one-line fix (needs a go-ahead — it is a code edit):* port the `FLOW_STEPS_GRID` loop
from `eval_meanflow_hardflow.sh:150-172` into the α-Flow script so both siblings take
`AF_FLOW_STEPS="2 3 5"` in a single job. Recommended before the powered run, not required for
the smoke.

### 4.5 Powered run (only after the smoke rows look sane)

Re-issue §4.2–4.4 with `AF_SEEDS="6 7 8 9 10"` / the MeanFlow seed equivalent and
`n_trials: 20` (yaml, or `AF_NTRIALS=20`). Budget ~5× the smoke wall time; `--time` should be
set to 2× the expected duration, capped at 24 h.

---

## 5. Pre-flight checklist

- [ ] `AF_BONE=unet` / `MF_BACKBONE=unet` — **the eval aborts on a horizon mismatch but the
      backbone silently changes the checkpoint path** (`bb{imf_backbone}` prefix token). Confirm a
      UNet checkpoint exists for both engines before submitting.
- [ ] `HFFM_BATCH == FMPCC_MPC_BATCH == 4` in every command (B4_PARITY).
- [ ] `FMPCC_HF_ALLOW_DEGENERATE` unset, `FMPCC_HF_MIN_GENUINE` unset.
- [ ] Disk: `/data` was at 100 % (27 G free of 7.0 T) before the 2026-09-03 AF-UNet runs. Check
      before submitting.
- [ ] Folder-name tokens carry `K`, `T`, `A`, `B`, `bb`, so the A=0.5 and A=1.0 ladders write to
      distinct directories and cannot overwrite each other. Verify on the first job's log line.

## 6. Reporting rules for the resulting DA

- Tag **every** `hardflow_new-*` row in the table itself — ❌ degenerate / ⚠️ thin / ✅ genuine —
  never in a footnote. Compute best-of / win-count / Pareto claims over ✅ rows **only**.
- Carry **backbone + parameter count** in every table. Lead with the `unet` row; any SiT/DiT/mf_dit
  win is secondary and must be labelled confounded.
- A K=1 arm-C row must never appear. If one does, the guard was bypassed — discard the run.
- If the ladder's only genuine HF rows turn out to be `n_genuine == 1`, state that the test
  carries **no** HardFlow signal and escalate to K ≥ 5 rather than reporting a thin row.
- `n_steps` basis: carry both bases explicitly labelled (open item #5 from
  `Report_20260903_AF_UNet/README.md` §11).

## 7. What this cannot answer

- **Timing across backends.** SLSQP-vs-IPOPT is a *solver* finding; it says nothing about
  HardFlow-vs-DPCC as algorithms. Keep the two questions apart.
- **Smoothness.** No jerk/curvature metric exists in this pipeline yet (same open item as the
  AF-UNet report §8) — any smoothness statement stays qualitative.
- **The visual tasks.** This is state-based `avoiding-d3il` only. The visual mix generations
  (`mix_visual_avoiding`, `mix_visual_aligning`) carry their own arm-C ports and their own
  `A`/`K` grids.

## 8. Sources read

- `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:25-46, 542-646` (gate, `hardflow_step_budget`, `hardflow_regime`, `resolve_nlp_backend`, `NLP_BACKENDS`)
- `flow_matcher_v3_alphaflow/sampling/hardflow_projection.py` (sibling port)
- `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh` (K loop, guard knobs, B4_PARITY note)
- `Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow_hardflow.sh` (no K loop — §4.4)
- `config/avoiding-d3il.py:73-105, 195-204, 1473-1493` (`MF_BACKBONE`, `AF_BONE`, `args_to_watch_fmv3_hf_plan`, `HFFM_FLOW_STEPS`)
- `config/meanflow_projection_eval.yaml`, `config/alphaflow_projection_eval.yaml`
- `logs_in_develop/HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md` §0–0.3
- `logs_in_develop/aggregated_hardflow_lowK/REGISTER_20260824_degenerate_HF_rows_and_warnings.md`
- `temp/0309/batch_avoiding_combined_20260903_133730/candidates_detailed.csv` (195 rows, 21 HF, all Gen12)
- `Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md` §11
