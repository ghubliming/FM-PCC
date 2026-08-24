# DA — Gen15 UAV Mix-ML: alpha-Flow @ SiT, corridor K-sweep (K=1/2/5)

**Date:** 2026-08-24
**Data root:** `temp/2308/` (train job 24924; eval jobs 24933/24934/24935) + batch DA `temp/2308/batch_uav_20260824_091511/`
**Git rev of the runs:** `7cc86c7` — *"(Gen15 af_sit) config: switch the af arm to alpha-Flow's own SiT backbone"*
**Scene:** `corridor` · **Seed:** 6 (single seed) · **n_trials:** 10 per variant · **Status:** all three eval jobs completed cleanly

---

## 0. TL;DR

The SiT run finished and the data is complete (23 variants × 3 K values × 10 rollouts, 0 failed units).

**What it shows.** `af @ sit` is the first UAV Mix-ML arm whose *generator* produces a usable
corridor trajectory at very low NFE. At **K=2** its unguided rollout already lands **0.45 m** from
goal; naive FM at the same K is **24.7 m** off and MeanFlow-UNet is **31.2 m** off. Consequently
`af@sit` reaches **8–9 / 10 success+constraints at K=2** where `fm@unet` scores **0/10 on every
projector**. At **K=5** it saturates the task — **10/10 on all five projectors** — matching what
`fm@unet` only achieves at K=5–20 and what `mf@unet` never achieves at any K.

**What it does not show.** The SiT arm carries **10,003,654 parameters (10.00 M)** against
**3,955,177 (3.96 M)** for `fm@unet` and **3,969,222 (3.97 M)** for `mf@unet` — a **2.53×**
parameter advantage, because SiT sizes from `dit_hidden_size = 256`, not from `freq_dim`.

**This is by design and is already documented in the config**, not an oversight. `config/uav_mix.py:401-405`
states it in the arm block itself:

> *"'sit' is alpha-Flow's OWN backbone (af_sit_trajectory.py, Gen3v7 U2); it sizes from
> dit_hidden_size/dit_depth, NOT from freq_dim, so this arm is **NOT parameter-matched to the 4.0 M
> U-Net rows** — it is the **deferred appendix arm (PLAN §6), never the architecture-matched claim**."*

So the run did exactly what it was set up to do, and the result should be read exactly as the plan
intended: **an appendix result**. Engine (alpha-Flow) and capacity (SiT @ 10 M) move together, so
neither can be credited on its own. Same framing as the D3IL side in
`Gen3v7_AlphaFlow/DA/DA_20260817_AF_SiT_ntrials20_K1_K2.md`. **There is no `af @ unet` corridor run,
so the backbone still cannot be isolated.**

Verdict: a genuinely interesting low-NFE appendix result, **not the architecture-matched claim** —
and it was never meant to be.

---

## 1. What ran

| Job | What | Result |
|---|---|---|
| 24924 | `train_mix_uav --engine af --scene corridor --seed 6` | completed, `final_test_loss 0.93525` |
| 24933 | eval K=1, 23 variants × 10 trials | completed |
| 24934 | eval K=2, 23 variants × 10 trials | completed |
| 24935 | eval K=5, 23 variants × 10 trials | completed |

Model dir: `logs/UAV_MIX/uav-corridor/mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0_bbsit/6`
Batch DA candidates: **C29 (K=1), C30 (K=2), C31 (K=5)**.

### Training protocol — identical across the three arms

| | `af @ sit` (C29–31) | `fm @ unet` (C32–36) | `mf @ unet` (C37–41) |
|---|---|---|---|
| dataset | 500 × 360 × 6 | 500 × 360 × 6 | 500 × 360 × 6 |
| `n_train_steps` | 100 000 | 100 000 | 100 000 |
| `train_batch_size` / `grad_accum` | 8 / 2 | 8 / 2 | 8 / 2 |
| `train_lr` / `ema_decay` | 1e-4 / 0.995 | 1e-4 / 0.995 | 1e-4 / 0.995 |
| horizon | H=8 | H=8 | H=8 |
| FM-anchor ratio | `af_ratio_fm = 0.5` | — | `meanflow_data_proportion = 0.5` |
| **backbone** | **SiT** `hidden=256 depth=8 heads=4 patch=1` | UNet `dim=32` | UNet `dim=32` |
| **params** | **10,003,654 (10.00 M)** | 3,955,177 (3.96 M) | 3,969,222 (3.97 M) |

> Note on `dp0.5` in the mf folder name: `meanflow_data_proportion` is the **fraction of the batch
> forced to r==t (FM anchors)** (`mix_uav/models/mf_diffusion.py:66`), *not* a data subset. All three
> arms see the full 500 trajectories. It is the direct analogue of af's `af_ratio_fm = 0.5`, so it is
> a **matched** hyperparameter, not a confound.

### Eval protocol — identical

Seed 6, 10 trials, `mpc_batch = 4`, `pid_stopgo`, threshold `T=0.5`, `max_episode_length = 396`,
`n_fm_steps = n_proj_steps = 396` (projection runs every control step), real-time budget
`30.3 ms`. Rollouts are **paired by index**: the homotopy sequence is `L C R L C R L C R L` for
`rollout_idx 0…9` in *every* candidate, so per-rollout contrasts are legitimate paired comparisons.

**HardFlow batch parity holds here.** The eval log prints `(B=4, proj=off, sel=…)` for every
`hardflow_new-*` variant and `mpc_batch = 4` for the DPCC variants — so unlike the visual-aligning
case, HF and DPCC are compared at the **same candidate fan** on UAV. No B1-vs-B4 correction needed.

---

## 2. Headline table — success+constraints (out of 10), matched projector

| Arm | K | NFE | `dpcc-c` | `dpcc-r` | `dpcc-t` | `hf-c` | `hf-t` | gen ms | total ms (dpcc-c) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **af @ sit** | 1 | 1.9 | 4 | 2 | 4 | 2 | 4 | 6.4 | 44.0 |
| **af @ sit** | 2 | 3.3 | **8** | 4 | **7** | **7** | **9** | 12.1 | **33.6** |
| **af @ sit** | 5 | 8.5 | **10** | **10** | **10** | **10** | **10** | 30.1 | 169.2 |
| fm @ unet | 1 | 1.9 | 0 | 0 | 0 | 0 | 0 | 9.1 | 107.2 |
| fm @ unet | 2 | 3.3 | 0 | 0 | 0 | 0 | 0 | 17.3 | 69.1 |
| fm @ unet | 5 | 8.5 | 9 | 9 | 9 | 10 | 10 | 43.7 | 167.0 |
| fm @ unet | 10 | 16.5 | 10 | 10 | 10 | 10 | 10 | 85.0 | 241.9 |
| fm @ unet | 20 | 29.1 | 10 | 10 | 10 | 10 | — | 170.7 | 415.5 |
| mf @ unet | 1 | 1.9 | 8 | 0 | 3 | 5 | 4 | 9.5 | 58.9 |
| mf @ unet | 2 | 3.3 | 8 | 1 | 2 | 5 | 2 | 18.1 | 64.8 |
| mf @ unet | 5 | 8.5 | 6 | 5 | 5 | 8 | 8 | 45.0 | 224.7 |
| mf @ unet | 10 | 16.5 | 6 | 8 | 8 | 10 | 9 | 90.1 | 339.4 |
| mf @ unet | 20 | 20.0 | 9 | 7 | 8 | — | — | 179.0 | 452.8 |

`hf-c` / `hf-t` = `hardflow_new-c` / `hardflow_new-t`. `hardflow_new-r` is omitted from the table —
see §6. Missing cells were not run.

**Reading it.**
- `af@sit` is the **only arm that is non-trivial at K≤2 with a well-conditioned trajectory**
  (see §4 — mf's K=1/K=2 numbers come from the projector rescuing a divergent generator).
- `af@sit` at **K=5 (NFE 8.5, 30 ms generation)** equals `fm@unet` at **K=10–20 (NFE 16.5–29.1,
  85–171 ms generation)** on every projector.
- `mf@unet` never reaches 10/10 on any DPCC projector at any K.

---

## 3. Paired significance (exact McNemar, n=10, rollout-matched)

Binary outcome = `n_success_and_constraints`. Two-sided exact binomial on discordant pairs.

| Contrast | projector | S&C | discordant | p |
|---|---|---|---:|---:|
| **af@sit K2 vs fm@unet K2** | `hf-t` | 9 vs 0 | 9 : 0 | **0.004** |
| | `dpcc-c` | 8 vs 0 | 8 : 0 | **0.008** |
| | `dpcc-t` | 7 vs 0 | 7 : 0 | **0.016** |
| | `hf-c` | 7 vs 0 | 7 : 0 | **0.016** |
| | `dpcc-r` | 4 vs 0 | 4 : 0 | 0.125 |
| **af@sit K2 vs mf@unet K2** | `hf-t` | 9 vs 2 | 7 : 0 | **0.016** |
| | `dpcc-t` | 7 vs 2 | 5 : 0 | 0.062 |
| | `dpcc-c` | 8 vs 8 | 2 : 2 | 1.000 |
| **af@sit K5 vs mf@unet K5** | `dpcc-r` | 10 vs 5 | 5 : 0 | 0.062 |
| | `dpcc-t` | 10 vs 5 | 5 : 0 | 0.062 |
| | `dpcc-c` | 10 vs 6 | 4 : 0 | 0.125 |
| **af@sit K5 vs fm@unet K5** | `dpcc-c/r/t` | 10 vs 9 | 1 : 0 | 1.000 |
| | `hf-c`, `hf-t` | 10 vs 10 | 0 : 0 | 1.000 |
| **af@sit K5 vs fm@unet K10/K20** | all | 10 vs 10 | 0 : 0 | 1.000 |
| **af@sit K1 vs mf@unet K1** | `dpcc-c` | 4 vs 8 | 1 : 5 | 0.219 |

**Honest reading of the p-values.** With n=10 paired rollouts, exact McNemar can never go below
p=0.002 (all 10 discordant), so this is a low-power design. The K=2 collapse of `fm@unet`
(0/10 on *all five* projectors while af scores 4–9) is the only effect that clears a
Bonferroni correction over the 5 projectors within that contrast (α/5 = 0.010: `hf-t` p=0.004 passes,
`dpcc-c` p=0.008 passes). Across the **full family of 50 contrasts** run here (α/50 = 0.001)
**nothing survives**. Treat everything below the K=2-vs-fm result as descriptive.

---

## 4. The mechanism: this is a *generator* result, not a projector result

Unguided rollout (`variant = diffuser`, no projection at all), goal distance in metres:

| Arm | K | mean | median | min | max | succ/10 | phys_safe/10 | track_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **af @ sit** | 1 | 9.43 | 4.42 | 2.46 | 24.23 | 0 | 1 | 4.51 |
| **af @ sit** | 2 | **0.45** | 0.36 | 0.28 | **1.24** | 3 | 9 | **0.47** |
| **af @ sit** | 5 | 0.67 | 0.30 | 0.29 | 2.74 | 6 | 9 | 0.75 |
| fm @ unet | 1 | 60.66 | 44.79 | 13.56 | 146.28 | 0 | 0 | 174.58 |
| fm @ unet | 2 | 24.68 | 7.74 | 0.27 | 83.72 | 3 | 5 | 48.43 |
| fm @ unet | 5 | **0.36** | 0.29 | 0.27 | 1.05 | 9 | 9 | 1.17 |
| fm @ unet | 10 | 0.80 | 0.60 | 0.26 | 2.44 | 4 | 7 | 16.83 |
| fm @ unet | 20 | 1.27 | 1.24 | 0.29 | 3.01 | 4 | 4 | 28.93 |
| mf @ unet | 1 | 31.51 | 31.28 | 5.81 | 66.14 | 0 | 0 | 114.24 |
| mf @ unet | 2 | 31.22 | 32.73 | 6.36 | 59.04 | 0 | 0 | 109.45 |
| mf @ unet | 5 | 31.74 | 30.34 | 18.63 | 50.83 | 0 | 0 | 99.54 |
| mf @ unet | 10 | 29.83 | 29.80 | 19.67 | 48.65 | 0 | 0 | 98.99 |
| mf @ unet | 20 | 30.11 | 30.69 | 12.62 | 44.37 | 0 | 0 | 94.29 |

Three separate findings live in this table:

1. **`af@sit` converges at K=2 and stays converged.** Worst-case unguided goal distance at K=2 is
   1.24 m over 10 rollouts. This is why the projector has so little work to do at K=2 — and why
   `af@sit K2 + dpcc-c` costs only **21.5 ms of projection** against fm@K2's 51.8 ms.

2. **`fm@unet` is non-monotone in K.** Unguided quality is best at K=5 (0.36 m) and *degrades*
   at K=10 (0.80 m) and K=20 (1.27 m, track_err 28.9). More Euler steps make the naive-FM
   trajectory worse. Worth a separate look — it may be an ODE-solver or normalisation issue rather
   than anything about the model.

3. **`mf@unet`'s generator never converges on corridor — at any K.** ~30 m unguided goal distance
   and track_err ~100 across the whole sweep. Every success `mf@unet` records comes from the
   projector dragging a divergent reference back onto the constraint set. That reframes the earlier
   corridor mf numbers: they are a **projector benchmark**, not a policy benchmark.

---

## 5. Cost, and the real-time budget

Generation time per control step, matched K:

| K | `af @ sit` (10.0 M) | `fm @ unet` (3.96 M) | `mf @ unet` (3.97 M) |
|---:|---:|---:|---:|
| 1 | **6.4 ms** | 9.1 ms | 9.5 ms |
| 2 | **12.1 ms** | 17.3 ms | 18.1 ms |
| 5 | **30.1 ms** | 43.7 ms | 45.0 ms |

**The 10.0 M SiT is faster per NFE than the 3.96 M UNet** — about 1.45× at every K, despite 2.53×
the parameters. This is consistent with the structural point in
`Gen14/U8/ARCH_20260823_dit_vs_visual_unet.md` §5: at `H=8` the 1-D UNet's three stride-2 levels
collapse the temporal axis to length 1, so ~80% of its parameters run as `kernel_size=5` convolutions
over a single timestep — poor arithmetic intensity — while the SiT runs 8 tokens through batched
attention at full width.

**Real-time.** Budget is `30.3 ms`. Over-budget fraction (dpcc-c):

| Arm | K | total ms | p95 ms | over-budget frac | S&C |
|---|---:|---:|---:|---:|---:|
| **af @ sit** | **2** | **33.6** | 56.7 | **0.17** | **8/10** |
| af @ sit | 1 | 44.0 | 91.8 | 0.28 | 4/10 |
| af @ sit | 5 | 169.2 | 223.4 | 1.00 | 10/10 |
| fm @ unet | 2 | 69.1 | 129.1 | 1.00 | 0/10 |
| fm @ unet | 5 | 167.0 | 230.4 | 1.00 | 9/10 |
| fm @ unet | 10 | 241.9 | 304.9 | 1.00 | 10/10 |
| mf @ unet | 2 | 64.8 | 108.1 | 1.00 | 8/10 |

`af@sit K=2 + dpcc-c` is the **only configuration in the whole corridor sweep that is close to the
control budget while still scoring well**: 33.6 ms mean, only 17% of steps over budget, 8/10 S&C.
Every other arm that scores ≥8/10 is 100% over budget at 2–13× the budget. For a UAV that is the
number that matters, and it is the strongest practical result in this run.

---

## 6. Data-quality notes

- **`hardflow_new-r` is pathological at low K.** At K=1 it costs `proj_ms = 1296.1` (p95 = 30 655 ms)
  with `track_err = 29.1` and 2/10 S&C; at K=2, `proj_ms = 551.5`, track_err 24.0. Random selection
  over the B=4 fan picks a bad candidate and the solver then fights it. At K=5 it is fine (10/10,
  330 ms) because every candidate is already good. Do not read `hf-r` at K≤2 as a HardFlow result.
- **One suspect row:** `fm@unet K20 / hardflow_new-t` reports `over_budget_frac = 0.00` at
  `total_ms = 1402` and `steps_to_goal = 0`. Internally inconsistent; excluded from §5. Everything
  else in `data_quality.csv` is clean (623 units loaded, 0 failed, 21 without timing).
- The batch's cross-candidate **ranking table averages over all 23 variants**, including degenerate
  ablations (`model_free`, `geo_free`, `*-tightened`). It puts `af@sit K5` at 51.7% and `fm@unet K10`
  at 51.7% at NFE 8.5 vs 16.5 — a fair like-for-like efficiency read — but the absolute percentages
  are diluted by the ablations and should not be quoted as success rates.
- **Single seed (6), 10 trials.** No seed-level variance is available for any arm on corridor.

---

## 7. Verdict against the benchmark hierarchy

| Claim | Supported? |
|---|---|
| `af@sit` beats naive FM at low NFE (K≤2) on corridor | **Yes**, and it is the largest and only near-significant effect here (p=0.004–0.016 on 4/5 projectors) |
| `af@sit` beats naive FM at saturation (K≥5) | **No** — tied at 10/10. It gets there at lower K and lower generation cost, but the ceiling is the same |
| `af@sit` Pareto-dominates `fm@unet K10` | **Yes on the numbers**: equal 10/10 S&C on all projectors, 169 ms vs 242 ms total, 30 ms vs 85 ms generation, equal steps-to-goal (264 vs 270). Confounded by params |
| `af@sit` beats `mf@unet` | **Yes descriptively at every K**, but `mf@unet`'s generator is broken on corridor (§4), so this is a weak comparison |
| **SiT (backbone) beats UNet (backbone)** | **No — and this arm was never set up to claim it.** 10.00 M vs 3.96 M, 2.53×; `config/uav_mix.py:401-405` labels it the deferred appendix arm. Engine and capacity move together |
| HardFlow beats the DPCC projector via a lower threshold | **Not tested in this run.** All variants ran at `T=0.5`. HF matches or slightly beats DPCC at equal threshold and equal B=4, at 2–4× the projection cost |

Under the Pareto rule this is a **favourable trade-off, not a clean win**: `af@sit K5` dominates
`fm@unet K10`, but the comparison changes two things at once. That is the expected status of an
appendix arm — the headline row for Gen15 corridor remains `fm@unet` vs `mf@unet` at matched 4.0 M.

---

## 8. What to run next

1. **[train] `af @ unet` on corridor, seed 6.** The missing control, and the single highest-value run
   in Gen15 right now — it promotes this appendix result into an architecture-matched one.
   `imf_backbone` is a plain forwarded string (`mix_uav/models/af_engine.py:46,88`) and the UNet path
   prints **4.0 M at `freq_dim = 32`**, so it is one config value in the `mix_uav_af` block (and the
   matching `plan_mix_uav_af` block — they must agree or eval rebuilds a different savepath). Note the
   `FIX_8_BACKBONE_DEFAULT` comment warns that the UNet arm is the one confounded by the `freq_dim`
   width defect — that defect is `freq_dim = 256 ⇒ 253 M`; at the 32 used by `fm`/`mf` here it is the
   correct 4.0 M and is directly comparable.
2. **[train] `af @ sit` with `dit_hidden_size = 160`** — the other way to close the same gap.
   Transformer parameters scale ~ hidden², so `256 · sqrt(3.96/10.00) ≈ 162`, i.e. **160 is very likely
   the parameter-matched UAV width** — the same value Gen14 U8 settled on for the visual side. Cheaper
   to reason about than (1) but it changes alpha-Flow's own backbone away from upstream's sizing.
3. **[eval] `af @ sit` at K=10 and K=20.** The sweep stops at K=5 where af has already saturated, so we
   cannot say whether it holds the ceiling or degrades the way `fm@unet` does (§4.2):

   ```bash
   ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af corridor "6" "10 20"
   ```
4. **[eval] seeds 7–10** on `af@sit` at K=2 and K=5 — n=10 single-seed is the binding limitation on
   every p-value in §3 (requires training those seeds first).
5. **[investigate] `mf@unet` corridor generator divergence** (§4.3). ~30 m unguided at every K means
   the earlier corridor mf conclusions need re-reading as projector results.
6. **[investigate] `fm@unet` non-monotonicity in K** (§4.2) — unguided quality peaks at K=5 and
   degrades through K=20.

---

## 9. One-line summary

`af @ sit` finished cleanly on corridor and is the first UAV arm to generate a usable trajectory at
K=2 (0.45 m unguided vs fm's 24.7 m), giving 8–9/10 success+constraints where naive FM gives 0/10, and
saturating at 10/10 by K=5 at half the NFE and a third of the generation cost of `fm@unet K10` — but
at **10.00 M parameters against 3.96 M** — the deferred appendix arm the config says it is — so it
stays an alpha-Flow-*plus*-capacity result until `af @ unet` is trained.
