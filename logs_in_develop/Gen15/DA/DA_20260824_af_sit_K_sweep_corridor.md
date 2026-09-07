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

On the **constraint** side (§2c/§2d): at **K=5 (NFE 8.5)** `af@sit` flies **zero violations,
CF 10/10** on 9 of 22 variants — including all three DPCC selectors and, notably, `geo_free`,
where the projector enforces **no geometry at all** (63.7 ms, `Σ viol = 0.000`). `fm@unet` needs
**K=10 (NFE 16.5)** for a clean DPCC row and **K=20 (NFE 29.1)** for a clean `geo_free` row —
**1.94×** and **3.42×** the NFE. `mf@unet` never produces a clean DPCC or `geo_free` row at any K.

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

## 2b. Best-horse comparison — best variant per family, and where the drone ends up

Selection rule, applied per arm × K × family: **S&C desc → `goal_dist` asc → `avg_time_ms` asc**.
`goal_radius = 0.30 m` (`eval_mix_uav.py:121`) and the rollout **early-stops** the step it gets
inside that ball, so `goal_dist ≈ 0.28–0.30` is the converged floor, not a measurement —
the informative columns are **worst** (`max` over the 10 rollouts) and §2b.3's far-miss list.
`crossed` = `goal_crossed_line` (half-plane past the finish **or** inside 0.30 m; `goal_reached ⇒ crossed`).

### 2b.1 Best variant in each family

| arm | K | family | best variant | S&C | succ | goal | crossed | gdist mean | worst | steps | terr | ms | p95 | over |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| af@sit | 1 | DPCC | `dpcc-c` | 4 | 7 | 7 | 8/10 | 0.94 | 3.27 | 266 | 1.17 | 44.0 | 91.8 | 0.28 |
| af@sit | 1 | HardFlow | `hardflow_new-t` | 4 | 7 | 8 | 8/10 | 0.69 | 2.49 | 277 | 4.59 | 173.9 | 308.6 | 1.00 |
| af@sit | 1 | other | `geo_free` | 4 | 5 | 6 | 8/10 | 2.37 | 8.93 | 279 | 2.37 | 40.9 | 73.3 | 0.33 |
| af@sit | 1 | unguided | `diffuser` | 0 | 0 | 0 | 1/10 | 9.43 | 24.23 | — | 4.51 | 6.4 | 6.4 | 0.00 |
| fm@unet | 1 | DPCC | `dpcc-r-tightened` | 0 | 1 | 3 | 4/10 | 4.98 | 32.52 | 270 | 40.05 | 148.7 | 213.6 | 0.71 |
| fm@unet | 1 | HardFlow | `hardflow_new-c` | 0 | 0 | 0 | 2/10 | 1.74 | 3.11 | — | 15.01 | 612.3 | 1742.1 | 1.00 |
| fm@unet | 1 | other | `bounds_free` | 0 | 0 | 0 | 1/10 | 2.11 | 2.86 | — | 46.94 | 89.1 | 213.2 | 0.69 |
| fm@unet | 1 | unguided | `diffuser` | 0 | 0 | 0 | 5/10 | 60.66 | 146.28 | — | 174.58 | 9.1 | 9.2 | 0.00 |
| mf@unet | 1 | DPCC | `dpcc-c` | 8 | 8 | 8 | 8/10 | 1.00 | 4.86 | 275 | 13.50 | 58.9 | 103.7 | 0.64 |
| mf@unet | 1 | HardFlow | `hardflow_new-c` | 5 | 6 | 6 | 7/10 | 2.63 | 9.66 | 275 | 9.32 | 1015.7 | 4311.6 | 1.00 |
| mf@unet | 1 | other | `geo_free` | 1 | 3 | 3 | 5/10 | 1.96 | 5.00 | 277 | 26.78 | 43.2 | 79.1 | 0.33 |
| mf@unet | 1 | unguided | `diffuser` | 0 | 0 | 0 | 2/10 | 31.51 | 66.14 | — | 114.24 | 9.5 | 9.4 | 0.00 |
| af@sit | 2 | DPCC | `dpcc-c` | 8 | 10 | 10 | 10/10 | 0.29 | 0.30 | 276 | 0.56 | 33.6 | 56.7 | 0.17 |
| af@sit | 2 | HardFlow | `hardflow_new-t` | 9 | 9 | 9 | 9/10 | 0.76 | 4.99 | 271 | 3.73 | 163.9 | 266.6 | 1.00 |
| af@sit | 2 | other | `bounds_free` | 7 | 7 | 7 | 7/10 | 1.36 | 4.57 | 275 | 1.64 | 60.9 | 100.1 | 0.38 |
| af@sit | 2 | unguided | `diffuser` | 0 | 3 | 3 | 10/10 | 0.45 | 1.24 | 312 | 0.47 | 12.3 | 12.5 | 0.00 |
| fm@unet | 2 | DPCC | `dpcc-t-tightened` | 0 | 5 | 5 | 6/10 | 2.86 | 11.45 | 265 | 35.33 | 147.1 | 171.8 | 1.00 |
| fm@unet | 2 | HardFlow | `hardflow_new` | 1 | 7 | 8 | 10/10 | 1.13 | 6.31 | 275 | 2.00 | 513.8 | 2204.1 | 1.00 |
| fm@unet | 2 | other | `post_processing` | 1 | 5 | 5 | 7/10 | 7.08 | 23.56 | 274 | 12.01 | 78.9 | 167.6 | 1.00 |
| fm@unet | 2 | unguided | `diffuser` | 0 | 3 | 3 | 9/10 | 24.68 | 83.72 | 260 | 48.43 | 17.6 | 18.1 | 0.00 |
| mf@unet | 2 | DPCC | `dpcc-c` | 8 | 8 | 8 | 9/10 | 6.99 | 44.91 | 273 | 19.20 | 64.8 | 108.1 | 1.00 |
| mf@unet | 2 | HardFlow | `hardflow_new-c` | 5 | 5 | 5 | 5/10 | 1.71 | 6.03 | 269 | 16.00 | 459.6 | 828.3 | 1.00 |
| mf@unet | 2 | other | `geo_free` | 2 | 3 | 3 | 4/10 | 3.01 | 11.21 | 269 | 25.32 | 52.5 | 96.4 | 0.39 |
| mf@unet | 2 | unguided | `diffuser` | 0 | 0 | 0 | 2/10 | 31.22 | 59.04 | — | 109.45 | 18.2 | 18.6 | 0.00 |
| af@sit | 5 | DPCC | `dpcc-c` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 264 | 0.55 | 169.2 | 223.4 | 1.00 |
| af@sit | 5 | HardFlow | `hardflow_new-t` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 268 | 0.56 | 383.4 | 599.7 | 1.00 |
| af@sit | 5 | other | `geo_free` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 265 | 0.55 | 63.7 | 64.0 | 1.00 |
| af@sit | 5 | unguided | `diffuser` | 3 | 6 | 6 | 9/10 | 0.67 | 2.74 | 273 | 0.75 | 30.1 | 30.2 | 0.03 |
| fm@unet | 5 | DPCC | `dpcc-c` | 9 | 10 | 10 | 10/10 | 0.29 | 0.30 | 270 | 0.52 | 167.0 | 230.4 | 1.00 |
| fm@unet | 5 | HardFlow | `hardflow_new` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 276 | 0.51 | 474.7 | 671.7 | 1.00 |
| fm@unet | 5 | other | `bounds_free` | 9 | 10 | 10 | 10/10 | 0.29 | 0.30 | 268 | 0.52 | 145.5 | 196.4 | 1.00 |
| fm@unet | 5 | unguided | `diffuser` | 0 | 9 | 9 | 10/10 | 0.36 | 1.05 | 317 | 1.17 | 44.1 | 44.0 | 1.00 |
| mf@unet | 5 | DPCC | `dpcc-c` | 6 | 8 | 9 | 9/10 | 0.85 | 5.87 | 277 | 7.55 | 224.7 | 278.0 | 1.00 |
| mf@unet | 5 | HardFlow | `hardflow_new-c` | 8 | 10 | 10 | 10/10 | 0.29 | 0.30 | 273 | 0.53 | 480.5 | 664.2 | 1.00 |
| mf@unet | 5 | other | `bounds_free` | 7 | 8 | 8 | 9/10 | 6.21 | 54.64 | 279 | 16.10 | 215.4 | 272.5 | 1.00 |
| mf@unet | 5 | unguided | `diffuser` | 0 | 0 | 0 | 3/10 | 31.74 | 50.83 | — | 99.54 | 44.6 | 44.6 | 1.00 |
| fm@unet | 10 | DPCC | `dpcc-c` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 270 | 0.51 | 241.9 | 304.9 | 1.00 |
| fm@unet | 10 | HardFlow | `hardflow_new-t` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 275 | 0.50 | 725.4 | 1056.1 | 1.00 |
| fm@unet | 10 | other | `post_processing` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 271 | 0.52 | 111.9 | 142.3 | 1.00 |
| fm@unet | 10 | unguided | `diffuser` | 0 | 4 | 4 | 7/10 | 0.80 | 2.44 | 314 | 16.83 | 85.7 | 86.0 | 1.00 |
| mf@unet | 10 | DPCC | `dpcc-r` | 8 | 9 | 9 | 10/10 | 0.48 | 2.14 | 272 | 2.24 | 297.7 | 387.9 | 1.00 |
| mf@unet | 10 | HardFlow | `hardflow_new-c` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 274 | 0.52 | 772.3 | 1064.9 | 1.00 |
| mf@unet | 10 | other | `geo_free-bounds_free` | 5 | 9 | 9 | 9/10 | 0.49 | 2.26 | 276 | 4.31 | 152.2 | 171.9 | 1.00 |
| mf@unet | 10 | unguided | `diffuser` | 0 | 0 | 0 | 5/10 | 29.83 | 48.65 | — | 98.99 | 89.4 | 91.8 | 1.00 |
| fm@unet | 20 | DPCC | `dpcc-t-tightened` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 265 | 0.54 | 417.4 | 479.0 | 1.00 |
| fm@unet | 20 | HardFlow | `hardflow_new-c` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 276 | 0.50 | 1376.6 | 2026.7 | 1.00 |
| fm@unet | 20 | other | `bounds_free-tightened` | 10 | 10 | 10 | 10/10 | 0.29 | 0.30 | 264 | 0.54 | 385.2 | 435.7 | 1.00 |
| fm@unet | 20 | unguided | `diffuser` | 4 | 4 | 4 | 4/10 | 1.27 | 3.01 | 280 | 28.93 | 171.9 | 173.3 | 1.00 |
| mf@unet | 20 | DPCC | `dpcc-c` | 9 | 10 | 10 | 10/10 | 0.29 | 0.30 | 275 | 0.53 | 452.8 | 509.2 | 1.00 |
| mf@unet | 20 | other | `geo_free-bounds_free` | 8 | 9 | 9 | 9/10 | 0.54 | 2.79 | 274 | 5.01 | 303.1 | 335.6 | 1.00 |
| mf@unet | 20 | unguided | `diffuser` | 0 | 0 | 0 | 3/10 | 30.11 | 44.37 | — | 94.29 | 181.9 | 187.4 | 1.00 |

### 2b.2 Overall champion per arm × K (all 23 guided variants, `diffuser` excluded)

| K | arm | champion | family | S&C | goal | crossed | gdist mean | worst | steps | ms |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | af@sit | `hardflow_new-t` | HardFlow | 4 | 8 | 8/10 | 0.69 | 2.49 | 277 | 173.9 |
| 1 | fm@unet | `hardflow_new-c` | HardFlow | 0 | 0 | 2/10 | 1.74 | 3.11 | — | 612.3 |
| 1 | mf@unet | `dpcc-c` | DPCC | 8 | 8 | 8/10 | 1.00 | 4.86 | 275 | 58.9 |
| 2 | af@sit | `hardflow_new-t` | HardFlow | 9 | 9 | 9/10 | 0.76 | 4.99 | 271 | 163.9 |
| 2 | fm@unet | `hardflow_new` | HardFlow | 1 | 8 | 10/10 | 1.13 | 6.31 | 275 | 513.8 |
| 2 | mf@unet | `dpcc-c` | DPCC | 8 | 8 | 9/10 | 6.99 | 44.91 | 273 | 64.8 |
| 5 | af@sit | `geo_free` | other | 10 | 10 | 10/10 | 0.29 | 0.30 | 265 | 63.7 |
| 5 | fm@unet | `hardflow_new` | HardFlow | 10 | 10 | 10/10 | 0.29 | 0.30 | 276 | 474.7 |
| 5 | mf@unet | `hardflow_new-c` | HardFlow | 8 | 10 | 10/10 | 0.29 | 0.30 | 273 | 480.5 |
| 10 | fm@unet | `dpcc-c` | DPCC | 10 | 10 | 10/10 | 0.29 | 0.30 | 270 | 241.9 |
| 10 | mf@unet | `hardflow_new-c` | HardFlow | 10 | 10 | 10/10 | 0.29 | 0.30 | 274 | 772.3 |
| 20 | fm@unet | `dpcc-t-tightened` | DPCC | 10 | 10 | 10/10 | 0.29 | 0.30 | 265 | 417.4 |
| 20 | mf@unet | `dpcc-c` | DPCC | 9 | 10 | 10/10 | 0.29 | 0.30 | 275 | 452.8 |

### 2b.3 Far-miss detail — final distance (m) of every rollout that did NOT reach

| arm | K | family | variant | reached | crossed | non-reaching rollouts' final `goal_dist` (m) |
|---|---:|---|---|---:|---:|---|
| af@sit | 1 | DPCC | `dpcc-c` | 7/10 | 8/10 | 1.95, 2.14, 3.27 |
| af@sit | 1 | HardFlow | `hardflow_new-t` | 8/10 | 8/10 | 2.14, 2.49 |
| fm@unet | 1 | DPCC | `dpcc-r-tightened` | 3/10 | 4/10 | 1.42, 1.70, 2.05, 3.00, 3.40, 4.85, 32.52 |
| fm@unet | 1 | HardFlow | `hardflow_new-c` | 0/10 | 2/10 | 1.22, 1.25, 1.43, 1.45, 1.45, 1.66, 1.74, 1.83, 2.30, 3.11 |
| mf@unet | 1 | DPCC | `dpcc-c` | 8/10 | 8/10 | 2.82, 4.86 |
| mf@unet | 1 | HardFlow | `hardflow_new-c` | 6/10 | 7/10 | 2.02, 3.43, 9.48, 9.66 |
| af@sit | 2 | DPCC | `dpcc-c` | 10/10 | 10/10 | — (10/10 reached) |
| af@sit | 2 | HardFlow | `hardflow_new-t` | 9/10 | 9/10 | 4.99 |
| fm@unet | 2 | DPCC | `dpcc-t-tightened` | 5/10 | 6/10 | 1.65, 3.93, 4.66, 5.51, 11.45 |
| fm@unet | 2 | HardFlow | `hardflow_new` | 8/10 | 10/10 | 2.66, 6.31 |
| mf@unet | 2 | DPCC | `dpcc-c` | 8/10 | 9/10 | 22.66, 44.91 |
| mf@unet | 2 | HardFlow | `hardflow_new-c` | 5/10 | 5/10 | 2.27, 2.40, 2.45, 2.53, 6.03 |
| af@sit | 5 | DPCC | `dpcc-c` | 10/10 | 10/10 | — (10/10 reached) |
| af@sit | 5 | HardFlow | `hardflow_new-t` | 10/10 | 10/10 | — (10/10 reached) |
| fm@unet | 5 | DPCC | `dpcc-c` | 10/10 | 10/10 | — (10/10 reached) |
| fm@unet | 5 | HardFlow | `hardflow_new` | 10/10 | 10/10 | — (10/10 reached) |
| mf@unet | 5 | DPCC | `dpcc-c` | 9/10 | 9/10 | 5.87 |
| mf@unet | 5 | HardFlow | `hardflow_new-c` | 10/10 | 10/10 | — (10/10 reached) |
| fm@unet | 10 | DPCC | `dpcc-c` | 10/10 | 10/10 | — (10/10 reached) |
| fm@unet | 10 | HardFlow | `hardflow_new-t` | 10/10 | 10/10 | — (10/10 reached) |
| mf@unet | 10 | DPCC | `dpcc-r` | 9/10 | 10/10 | 2.14 |
| mf@unet | 10 | HardFlow | `hardflow_new-c` | 10/10 | 10/10 | — (10/10 reached) |
| fm@unet | 20 | DPCC | `dpcc-t-tightened` | 10/10 | 10/10 | — (10/10 reached) |
| fm@unet | 20 | HardFlow | `hardflow_new-c` | 10/10 | 10/10 | — (10/10 reached) |
| mf@unet | 20 | DPCC | `dpcc-c` | 10/10 | 10/10 | — (10/10 reached) |

### 2b.4 Notes on the tables

- `af @ sit` has no K=10 / K=20 rows: not run (§8 item 3).
- `mf @ unet` K=20 (C39) carries 20 variants, **no HardFlow rows** — that eval predates the arm.
- Variant naming: C29–C31 (`af`, 2026-08-23/24) use the post-parity `hardflow_new-r/-c/-t`;
  C32–C41 (`fm`/`mf`) still carry bare `hardflow_new`, which at `B=4` is byte-identical to `-r`
  (`config/uav_mix.py:203` B4_PARITY note).
- Every `hardflow_new-*` champion row runs at `over_budget_frac = 1.00`.
- **K=1 and K=2 HardFlow rows are degenerate** at the shipped `activation_threshold = 0.5`:
  they run `Pi_S(Euler sample)` = sample-then-project, i.e. DPCC's algorithm with IPOPT
  instead of SLSQP (`eval_k_sweep.sh:31-42`, `HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md`).
  That covers the `af@sit` K=1 and K=2 champions in §2b.2 and the `fm`/`mf` K≤2 HardFlow rows
  in §2b.1/§2b.3 — matched-NFE and quotable as one-shot projection, not as HardFlow.
  Non-degenerate from K≥3 at `A=0.5`.
- At K=5, three arms all hit 10/10 goal: `af@sit` `geo_free` **63.7 ms**, `fm@unet`
  `hardflow_new` **474.7 ms**, `mf@unet` `hardflow_new-c` **480.5 ms** (S&C 10 / 10 / 8).
- No champion row is inside the 30.3 ms budget. Closest: `af@sit` K=2 `dpcc-c` —
  33.6 ms mean, 56.7 ms p95, `over = 0.17`, S&C 8, 10/10 goal.
- `mf @ unet` K=2 `dpcc-c` reaches 8/10 with far-misses at **22.66 m and 44.91 m**; its
  `diffuser` row is 31.22 m mean (§4).
- `fm @ unet` K=1 `hardflow_new-c` is 0/10 goal with all ten rollouts stalled at 1.22–3.11 m.

---

## 2c. Constraint violations

Definitions (`eval_mix_uav.py:442-449`, called at `:1375` with the **full** `config`):
- `CF` = `collision_free_completed` — rollouts out of 10 with **zero** violated steps. This is
  the constraint half of S&C.
- `n viol` = mean number of executed steps per rollout violating ≥1 active spatial family
  (`geo_bounds` / `halfspace` / `obstacles`), out of ~265–280 flown steps.
- `Σ viol` = mean summed penetration depth per rollout, in metres.
- `safe` = `phys_safe`; `contact` = `phys_contact_frac`; `min z` = minimum altitude (m).
- 🔴 The metric checks the **flown** path against **RAW geometry ⊕ r_drone** and is
  **variant-independent** — the ablation variants (`geo_free`, `bounds_free`, `model_free`)
  are scored against the same full constraint set as `dpcc-c`. A `geo_free` row with zero
  violations means the *generator* stayed clean with the projector enforcing no geometry.

### 2c.1 Violations on the champion rows

| arm | K | NFE | family | variant | S&C | CF | n viol | Σ viol (m) | safe | contact | min z |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| af@sit | 1 | 1.9 | DPCC | `dpcc-c` | 4 | 4/10 | 61.2 | 55.063 | 7/10 | 0.0738 | 0.713 |
| af@sit | 1 | 1.9 | HardFlow | `hardflow_new-t` | 4 | 4/10 | 51.4 | 20.012 | 7/10 | 0.1009 | 0.707 |
| af@sit | 1 | 1.9 | unguided | `diffuser` | 0 | 0/10 | 353.8 | 1745.727 | 1/10 | 0.2199 | 0.168 |
| fm@unet | 1 | 1.9 | DPCC | `dpcc-r-tightened` | 0 | 0/10 | 180.4 | 269.363 | 1/10 | 0.3158 | 0.372 |
| fm@unet | 1 | 1.9 | HardFlow | `hardflow_new-c` | 0 | 0/10 | 187.9 | 117.485 | 0/10 | 0.3429 | 0.141 |
| fm@unet | 1 | 1.9 | unguided | `diffuser` | 0 | 0/10 | 367.7 | 15929.191 | 0/10 | 0.0000 | -275.354 |
| mf@unet | 1 | 1.9 | DPCC | `dpcc-c` | 8 | 8/10 | 56.8 | 20.468 | 8/10 | 0.1293 | 0.927 |
| mf@unet | 1 | 1.9 | HardFlow | `hardflow_new-c` | 5 | 5/10 | 111.1 | 233.593 | 6/10 | 0.1291 | 0.666 |
| mf@unet | 1 | 1.9 | unguided | `diffuser` | 0 | 0/10 | 379.0 | 9273.166 | 0/10 | 0.0004 | 0.082 |
| af@sit | 2 | 3.3 | DPCC | `dpcc-c` | 8 | 8/10 | 6.6 | 0.261 | 10/10 | 0.0003 | 0.957 |
| af@sit | 2 | 3.3 | HardFlow | `hardflow_new-t` | 9 | 9/10 | 27.1 | 9.501 | 9/10 | 0.0507 | 0.836 |
| af@sit | 2 | 3.3 | unguided | `diffuser` | 0 | 0/10 | 160.5 | 47.936 | 9/10 | 0.0007 | 0.905 |
| fm@unet | 2 | 3.3 | DPCC | `dpcc-t-tightened` | 0 | 0/10 | 143.7 | 165.683 | 5/10 | 0.1528 | 0.581 |
| fm@unet | 2 | 3.3 | HardFlow | `hardflow_new` | 1 | 1/10 | 43.8 | 75.123 | 7/10 | 0.0110 | 0.707 |
| fm@unet | 2 | 3.3 | unguided | `diffuser` | 0 | 0/10 | 267.2 | 4644.115 | 5/10 | 0.1006 | 0.290 |
| mf@unet | 2 | 3.3 | DPCC | `dpcc-c` | 8 | 8/10 | 71.2 | 1899.535 | 8/10 | 0.0000 | 0.841 |
| mf@unet | 2 | 3.3 | HardFlow | `hardflow_new-c` | 5 | 5/10 | 113.5 | 71.870 | 5/10 | 0.1676 | 0.603 |
| mf@unet | 2 | 3.3 | unguided | `diffuser` | 0 | 0/10 | 374.4 | 9603.924 | 0/10 | 0.0000 | 0.082 |
| af@sit | 5 | 8.5 | DPCC | `dpcc-c` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.021 |
| af@sit | 5 | 8.5 | HardFlow | `hardflow_new-t` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 0.997 |
| af@sit | 5 | 8.5 | unguided | `diffuser` | 3 | 3/10 | 67.3 | 24.994 | 9/10 | 0.0036 | 0.793 |
| fm@unet | 5 | 8.5 | DPCC | `dpcc-c` | 9 | 9/10 | 0.9 | 0.001 | 10/10 | 0.0000 | 1.053 |
| fm@unet | 5 | 8.5 | HardFlow | `hardflow_new` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.061 |
| fm@unet | 5 | 8.5 | unguided | `diffuser` | 0 | 0/10 | 124.6 | 31.448 | 9/10 | 0.0015 | 0.355 |
| mf@unet | 5 | 8.5 | DPCC | `dpcc-c` | 6 | 6/10 | 41.3 | 33.587 | 8/10 | 0.0053 | 0.939 |
| mf@unet | 5 | 8.5 | HardFlow | `hardflow_new-c` | 8 | 8/10 | 0.9 | 0.001 | 10/10 | 0.0000 | 1.051 |
| mf@unet | 5 | 8.5 | unguided | `diffuser` | 0 | 0/10 | 367.6 | 9553.215 | 0/10 | 0.0000 | 0.074 |
| fm@unet | 10 | 16.5 | DPCC | `dpcc-c` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.058 |
| fm@unet | 10 | 16.5 | HardFlow | `hardflow_new-t` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.065 |
| fm@unet | 10 | 16.5 | unguided | `diffuser` | 0 | 0/10 | 127.4 | 38.133 | 7/10 | 0.1274 | 0.418 |
| mf@unet | 10 | 16.5 | DPCC | `dpcc-r` | 8 | 8/10 | 16.7 | 16.197 | 9/10 | 0.0063 | 0.947 |
| mf@unet | 10 | 16.5 | HardFlow | `hardflow_new-c` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.056 |
| mf@unet | 10 | 16.5 | unguided | `diffuser` | 0 | 0/10 | 368.4 | 9151.881 | 0/10 | 0.0000 | 0.084 |
| fm@unet | 20 | 29.1 | DPCC | `dpcc-t-tightened` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.063 |
| fm@unet | 20 | 29.1 | HardFlow | `hardflow_new-c` | 10 | 10/10 | 0.0 | 0.000 | 10/10 | 0.0000 | 1.063 |
| fm@unet | 20 | 29.1 | unguided | `diffuser` | 4 | 4/10 | 131.0 | 56.607 | 4/10 | 0.1789 | 0.572 |
| mf@unet | 20 | 29.1 | DPCC | `dpcc-c` | 9 | 9/10 | 0.4 | 0.001 | 10/10 | 0.0000 | 1.062 |
| mf@unet | 20 | 29.1 | unguided | `diffuser` | 0 | 0/10 | 366.1 | 8929.045 | 0/10 | 0.0002 | 0.056 |

Three reads from that table:

- **`fm @ unet` K=1 `diffuser`: `min z = -275.354 m`, `Σ viol = 15 929.191 m`.** A divergence,
  not a low flight. Its `n viol` (367.7) and `Σ viol` are not comparable to any other row.
  The divergence-abort guard that would truncate it landed in `1288118a` (2026-08-23 20:09),
  **after** the git rev all three arms ran at (`7cc86c7`, 2026-08-23 11:00) — so no arm in this
  DA has it, and the asymmetry is zero.
- **`mf @ unet` `diffuser` sits at `min z ≈ 0.05–0.08 m` at every K** (0.082 / 0.082 / 0.074 /
  0.084 / 0.056 for K=1/2/5/10/20) with `Σ viol ≈ 8 900–9 600 m` and `safe 0/10`. The MeanFlow
  generator is on the ground at every budget — the constraint-side form of §4.
- **`phys_safe` and `CF` decouple.** `af@sit` K=2 `dpcc-c` is `safe 10/10` with `CF 8/10`:
  the drone never crashed, it clipped geometry for 6.6 steps totalling 0.261 m.


### 2c.2 Zero-violation ledger — clean rows (`n viol = 0` **and** `CF = 10/10`)

| arm | K | NFE | clean variants | of |
|---|---:|---:|---:|---:|
| af@sit | 1 | 1.9 | **0** | 22 |
| af@sit | 2 | 3.3 | **0** | 22 |
| af@sit | 5 | 8.5 | **9** | 22 |
| fm@unet | 1 | 1.9 | **0** | 22 |
| fm@unet | 2 | 3.3 | **0** | 22 |
| fm@unet | 5 | 8.5 | **3** | 22 |
| fm@unet | 10 | 16.5 | **8** | 22 |
| fm@unet | 20 | 29.1 | **14** | 22 |
| mf@unet | 1 | 1.9 | **0** | 22 |
| mf@unet | 2 | 3.3 | **0** | 22 |
| mf@unet | 5 | 8.5 | **0** | 22 |
| mf@unet | 10 | 16.5 | **2** | 22 |
| mf@unet | 20 | 29.1 | **0** | 19 |

Every clean row is also 10/10 goal and 10/10 `phys_safe`. Full list:

| arm | K | clean variant | min z | ms |
|---|---:|---|---:|---:|
| af@sit | 5 | `bounds_free` | 1.019 | 146.2 |
| af@sit | 5 | `dpcc-c` | 1.021 | 169.2 |
| af@sit | 5 | `dpcc-r` | 1.010 | 169.5 |
| af@sit | 5 | `dpcc-t` | 1.017 | 169.3 |
| af@sit | 5 | `geo_free` | 1.018 | 63.7 |
| af@sit | 5 | `geo_free-bounds_free` | 1.004 | 58.1 |
| af@sit | 5 | `hardflow_new-c` | 1.008 | 380.9 |
| af@sit | 5 | `hardflow_new-r` | 0.986 | 382.3 |
| af@sit | 5 | `hardflow_new-t` | 0.997 | 383.4 |
| fm@unet | 5 | `hardflow_new` | 1.061 | 474.7 |
| fm@unet | 5 | `hardflow_new-c` | 1.045 | 492.9 |
| fm@unet | 5 | `hardflow_new-t` | 1.063 | 477.5 |
| fm@unet | 10 | `bounds_free` | 1.063 | 217.3 |
| fm@unet | 10 | `dpcc-c` | 1.058 | 241.9 |
| fm@unet | 10 | `dpcc-r` | 1.066 | 241.8 |
| fm@unet | 10 | `dpcc-t` | 1.064 | 240.8 |
| fm@unet | 10 | `hardflow_new` | 1.065 | 719.8 |
| fm@unet | 10 | `hardflow_new-c` | 1.057 | 756.4 |
| fm@unet | 10 | `hardflow_new-t` | 1.065 | 725.4 |
| fm@unet | 10 | `post_processing` | 1.040 | 111.9 |
| fm@unet | 20 | `bounds_free` | 1.063 | 382.7 |
| fm@unet | 20 | `bounds_free-tightened` | 1.061 | 385.2 |
| fm@unet | 20 | `dpcc-c` | 1.059 | 415.5 |
| fm@unet | 20 | `dpcc-c-tightened` | 1.058 | 417.1 |
| fm@unet | 20 | `dpcc-r` | 1.065 | 414.4 |
| fm@unet | 20 | `dpcc-r-tightened` | 1.063 | 416.8 |
| fm@unet | 20 | `dpcc-t` | 1.065 | 415.5 |
| fm@unet | 20 | `dpcc-t-tightened` | 1.063 | 417.4 |
| fm@unet | 20 | `geo_free` | 1.066 | 283.0 |
| fm@unet | 20 | `geo_free-bounds_free` | 1.061 | 263.4 |
| fm@unet | 20 | `hardflow_new` | 1.065 | 1359.9 |
| fm@unet | 20 | `hardflow_new-c` | 1.063 | 1376.6 |
| fm@unet | 20 | `hardflow_new-t` | 1.095 | 1402.2 |
| fm@unet | 20 | `post_processing` | 1.038 | 196.0 |
| mf@unet | 10 | `hardflow_new` | 1.057 | 779.8 |
| mf@unet | 10 | `hardflow_new-c` | 1.056 | 772.3 |

### 2c.3 `geo_free` — the projector-off geometry test

`geo_free` removes `geo_bounds` + `halfspace` + `obstacles` from the **projector**
(`eval_mix_uav.py:869/927/937`); the violation metric still scores the full geometry.
So this column is the generator's own constraint compliance, with dynamics + bounds
projection only.

| arm | K | NFE | S&C | goal | CF | n viol | Σ viol (m) | gdist | ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| af@sit | 1 | 1.9 | 4 | 6 | 4/10 | 121.5 | 237.905 | 2.37 | 40.9 |
| af@sit | 2 | 3.3 | 3 | 3 | 3/10 | 139.0 | 98.768 | 1.65 | 48.4 |
| af@sit | 5 | 8.5 | 10 | 10 | 10/10 | 0.0 | 0.000 | 0.29 | 63.7 |
| fm@unet | 1 | 1.9 | 0 | 0 | 0/10 | 340.8 | 111.429 | 3.56 | 27.9 |
| fm@unet | 2 | 3.3 | 0 | 0 | 0/10 | 290.7 | 387.209 | 6.13 | 35.6 |
| fm@unet | 5 | 8.5 | 2 | 10 | 2/10 | 53.4 | 2.198 | 0.29 | 77.1 |
| fm@unet | 10 | 16.5 | 4 | 8 | 4/10 | 40.9 | 449.871 | 6.78 | 164.1 |
| fm@unet | 20 | 29.1 | 10 | 10 | 10/10 | 0.0 | 0.000 | 0.29 | 283.0 |
| mf@unet | 1 | 1.9 | 1 | 3 | 1/10 | 190.2 | 97.652 | 1.96 | 43.2 |
| mf@unet | 2 | 3.3 | 2 | 3 | 2/10 | 194.8 | 191.728 | 3.01 | 52.5 |
| mf@unet | 5 | 8.5 | 3 | 6 | 3/10 | 126.2 | 37.937 | 1.26 | 98.1 |
| mf@unet | 10 | 16.5 | 5 | 9 | 5/10 | 43.7 | 14.318 | 0.54 | 167.6 |
| mf@unet | 20 | 29.1 | 6 | 9 | 6/10 | 36.4 | 12.529 | 0.48 | 328.2 |

---

## 2d. Insight — did `af @ sit` win?

### 2d.1 The clean-flight budget: cheapest K at which each arm flies with ZERO violations

| bar | `af @ sit` | `fm @ unet` | `mf @ unet` | af NFE advantage |
|---|---|---|---|---:|
| any zero-violation row | **K=5** (NFE 8.5) | K=5, HardFlow only (NFE 8.5) | K=10, HardFlow only (NFE 16.5) | tie vs fm |
| zero-violation **DPCC** row | **K=5** (NFE 8.5) | K=10 (NFE 16.5) | **never** — best `dpcc-c` K=20: CF 9/10, 0.4 viol | **1.94×** vs fm |
| zero-violation **`geo_free`** row (generator alone) | **K=5** (NFE 8.5) | K=20 (NFE 29.1) | **never** — best `geo_free` K=20: CF 6/10, 36.4 viol | **3.42×** vs fm |

Every row in that table is also 10/10 goal, 10/10 `phys_safe`, `gdist ≤ 0.30 m`.

### 2d.2 How the arms fail differently at equal S&C

`af @ sit` and `mf @ unet` both score **S&C 8/10** on `dpcc-c` at K=2. The violations are not
the same shape:

| arm | K=2 `dpcc-c` | CF | n viol (steps) | Σ viol (m) | worst gdist (m) | min z (m) | track err |
|---|---:|---:|---:|---:|---:|---:|---:|
| `af @ sit` | S&C 8 | 8/10 | **6.6** | **0.261** | **0.30** | 0.957 | 0.56 |
| `mf @ unet` | S&C 8 | 8/10 | 71.2 | **1899.535** | 44.91 | 0.841 | 19.20 |

10.8× the violating steps, 7 280× the penetration depth, 150× the worst final distance,
34× the tracking error — at the same headline score.

### 2d.3 Cost of the cheapest clean flight (budget = 30.3 ms)

| arm | cheapest zero-violation row | K | ms | × budget |
|---|---|---:|---:|---:|
| `af @ sit` | `geo_free-bounds_free` | 5 | **58.1** | 1.9× |
| `af @ sit` (full-geometry projector) | `dpcc-c` | 5 | 169.2 | 5.6× |
| `fm @ unet` | `post_processing` | 10 | 111.9 | 3.7× |
| `mf @ unet` | `hardflow_new-c` | 10 | 772.3 | 25.5× |

### 2d.4 Where `af @ sit` did NOT win

- **K=1.** `mf @ unet` `dpcc-c` = S&C 8, CF 8/10, 56.8 viol, 58.9 ms. `af @ sit` `dpcc-c` =
  S&C 4, CF 4/10, 61.2 viol, 44.0 ms. `mf` takes K=1 outright.
- **Breadth of the clean band is untested at the top.** `fm @ unet` has **14/22** clean variants
  at K=20 and 8/22 at K=10; `af @ sit` has 9/22 at K=5 and **no K=10 / K=20 rows at all**.
  Whether af's clean band widens the way fm's does is unmeasured (§8 item 3).
- **Parameters.** 10.00 M (SiT) vs 3.96 M (U-Net) — the deferred-appendix confound of §0/§7.
  None of §2d is an architecture-matched claim.
- **Single seed (6), 10 rollouts.** No seed-level variance; §3's contrasts do not survive a
  family-wide Bonferroni.

### 2d.5 What survives

At **matched NFE 8.5 (K=5)**, `af @ sit` is the only arm in the sweep that flies the corridor
with zero constraint violations under a projector that enforces **no geometry at all**
(`geo_free`, 63.7 ms, S&C 10, CF 10/10, `Σ viol = 0.000`). `fm @ unet` needs NFE 29.1 for the
same row; `mf @ unet` never reaches it. That is a generator claim, consistent with §4 —
and it is the constraint-side counterpart of §4's goal-distance evidence.

---

## 2e. Pareto read on (S&C, steps, avg time)

Objective vector, no wall-clock target assumed: **maximise `n_success_and_constraints`,
minimise `steps_to_goal`, minimise `avg_time_ms`**. `diffuser` excluded (no projector);
rows that never reached goal carry `steps = —` and lose on that axis.

**283 guided rows → 19 non-dominated.**

Axis strength across the S&C ≥ 8 rows: **steps spans 263.7–276.9 (1.05×)**, **time spans
33.6–1402.2 ms (41.7×)**. Steps barely separates anything; the ranking is effectively
(S&C, time) with steps as the tiebreak.

### 2e.1 The S&C = 10 tier — 36 rows, 4 non-dominated

| arm | K | NFE | variant | steps | ms | verdict |
|---|---:|---:|---|---:|---:|---|
| af @ sit | 5 | 8.5 | `geo_free-bounds_free` | 265.5 | 58.1 | **PARETO-OPTIMAL** |
| af @ sit | 5 | 8.5 | `geo_free` | 264.7 | 63.7 | **PARETO-OPTIMAL** |
| fm @ unet | 10 | 16.5 | `post_processing` | 270.7 | 111.9 | dominated by `af @ sit K5 geo_free-bounds_free` |
| af @ sit | 5 | 8.5 | `bounds_free` | 264.5 | 146.2 | **PARETO-OPTIMAL** |
| af @ sit | 5 | 8.5 | `dpcc-c` | 263.7 | 169.2 | **PARETO-OPTIMAL** |
| af @ sit | 5 | 8.5 | `dpcc-t` | 264.5 | 169.3 | dominated by `af @ sit K5 bounds_free` |
| af @ sit | 5 | 8.5 | `dpcc-r` | 266.5 | 169.5 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `post_processing` | 271.2 | 196.0 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `bounds_free` | 268.2 | 217.3 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `dpcc-t` | 268.2 | 240.8 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `dpcc-r` | 265.9 | 241.8 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `dpcc-c` | 270.5 | 241.9 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `geo_free-bounds_free` | 268.8 | 263.4 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `geo_free` | 267.0 | 283.0 | dominated by `af @ sit K5 geo_free-bounds_free` |
| af @ sit | 5 | 8.5 | `hardflow_new-c` | 266.0 | 380.9 | dominated by `af @ sit K5 geo_free-bounds_free` |
| af @ sit | 5 | 8.5 | `hardflow_new-r` | 270.7 | 382.3 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `bounds_free` | 267.7 | 382.7 | dominated by `af @ sit K5 geo_free-bounds_free` |
| af @ sit | 5 | 8.5 | `hardflow_new-t` | 268.1 | 383.4 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `bounds_free-tightened` | 264.1 | 385.2 | dominated by `af @ sit K5 dpcc-c` |
| fm @ unet | 20 | 29.1 | `dpcc-r` | 267.2 | 414.4 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `dpcc-c` | 271.7 | 415.5 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `dpcc-t` | 269.3 | 415.5 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `dpcc-r-tightened` | 264.8 | 416.8 | dominated by `af @ sit K5 geo_free` |
| fm @ unet | 20 | 29.1 | `dpcc-c-tightened` | 266.8 | 417.1 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `dpcc-t-tightened` | 264.8 | 417.4 | dominated by `af @ sit K5 geo_free` |
| fm @ unet | 5 | 8.5 | `hardflow_new` | 275.5 | 474.7 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 5 | 8.5 | `hardflow_new-t` | 273.6 | 477.5 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 5 | 8.5 | `hardflow_new-c` | 275.8 | 492.9 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `hardflow_new` | 273.4 | 719.8 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `hardflow_new-t` | 275.2 | 725.4 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 10 | 16.5 | `hardflow_new-c` | 276.9 | 756.4 | dominated by `af @ sit K5 geo_free-bounds_free` |
| mf @ unet | 10 | 16.5 | `hardflow_new-c` | 274.2 | 772.3 | dominated by `af @ sit K5 geo_free-bounds_free` |
| mf @ unet | 10 | 16.5 | `hardflow_new` | 271.2 | 779.8 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `hardflow_new` | 273.5 | 1359.9 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `hardflow_new-c` | 276.3 | 1376.6 | dominated by `af @ sit K5 geo_free-bounds_free` |
| fm @ unet | 20 | 29.1 | `hardflow_new-t` | — | 1402.2 | dominated by `af @ sit K5 geo_free-bounds_free` |

Arm split of the 36: `af @ sit` 9, `fm @ unet` 25, `mf @ unet` 2. **All 4 Pareto-optimal
rows are `af @ sit` at K=5**, and all 4 carry `n viol = 0`.

### 2e.2 Full non-dominated set (all S&C levels)

| S&C | arm | K | variant | steps | ms | CF | n viol |
|---:|---|---:|---|---:|---:|---:|---:|
| 10 | af @ sit | 5 | `geo_free-bounds_free` | 265.5 | 58.1 | 10/10 | 0.0 |
| 10 | af @ sit | 5 | `geo_free` | 264.7 | 63.7 | 10/10 | 0.0 |
| 10 | af @ sit | 5 | `bounds_free` | 264.5 | 146.2 | 10/10 | 0.0 |
| 10 | af @ sit | 5 | `dpcc-c` | 263.7 | 169.2 | 10/10 | 0.0 |
| 8 | af @ sit | 2 | `dpcc-c` | 276.0 | 33.6 | 8/10 | 6.6 |
| 4 | af @ sit | 1 | `dpcc-c` | 266.3 | 44.0 | 4/10 | 61.2 |
| 4 | mf @ unet | 20 | `dpcc-c-tightened` | 261.8 | 953.6 | 4/10 | 148.2 |
| 3 | af @ sit | 2 | `dpcc-c-tightened` | 262.7 | 121.3 | 3/10 | 226.0 |
| 3 | fm @ unet | 5 | `dpcc-r-tightened` | 261.9 | 280.5 | 3/10 | 43.9 |
| 3 | fm @ unet | 5 | `dpcc-t-tightened` | 259.6 | 280.7 | 3/10 | 65.5 |
| 2 | mf @ unet | 10 | `dpcc-t-tightened` | 258.5 | 554.1 | 2/10 | 188.3 |
| 1 | mf @ unet | 5 | `post_processing-tightened` | 251.0 | 160.9 | 1/10 | 294.5 |
| 0 | af @ sit | 1 | `gradient-tightened` | — | 8.3 | 0/10 | 363.0 |
| 0 | af @ sit | 2 | `gradient-tightened` | 324.5 | 14.0 | 0/10 | 332.3 |
| 0 | af @ sit | 2 | `geo_free-model_free` | 300.0 | 17.7 | 0/10 | 141.7 |
| 0 | af @ sit | 2 | `model_free-bounds_free` | 285.3 | 25.2 | 0/10 | 130.1 |
| 0 | af @ sit | 2 | `model_free` | 281.7 | 27.9 | 0/10 | 120.3 |
| 0 | fm @ unet | 2 | `dpcc-c-tightened` | 258.0 | 100.2 | 0/10 | 106.8 |
| 0 | mf @ unet | 1 | `post_processing-tightened` | 247.0 | 119.8 | 0/10 | 293.7 |

### 2e.3 Projector-matched head-to-head

`geo_free-bounds_free` enforces dynamics only, so the 58.1 ms row is not projector-matched to a
`dpcc-c` row. Holding the projector fixed at full DPCC `dpcc-c`:

| arm | K | NFE | S&C | CF | n viol | steps | ms | vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`af @ sit`** | **5** | **8.5** | **10** | **10/10** | **0.0** | **263.7** | **169.2** | **dominates on all three axes** |
| `fm @ unet` (baseline) | 10 | 16.5 | 10 | 10/10 | 0.0 | 270.5 | 241.9 | — |
| `fm @ unet` | 20 | 29.1 | 10 | 10/10 | 0.0 | 271.7 | 415.5 | dominated |
| `mf @ unet` | 20 | 29.1 | 9 | 9/10 | 0.4 | 275.0 | 452.8 | dominated |

Same projector, same threshold, same `mpc_batch = 4`, same controller: `af @ sit` K=5 is
**1.43× faster, 6.8 steps shorter, at equal 10/10 S&C and equal zero violations** — a genuine
Pareto domination of the matched-projector baseline, at **1.94× fewer NFE**.

### 2e.4 Tiers

| tier | rows | read |
|---|---|---|
| **Optimal** | `af @ sit` K=5 × {`geo_free-bounds_free` 58.1 ms, `geo_free` 63.7, `bounds_free` 146.2, `dpcc-c` 169.2} | S&C 10, CF 10/10, `n viol = 0`, steps 263.7–265.5. Nothing in the sweep dominates any of them |
| **Good** | `af @ sit` K=2 `dpcc-c` — S&C 8, 276.0 steps, **33.6 ms** | the cheapest non-dominated row with S&C ≥ 8, at NFE 3.3. Costs 2/10 S&C and 6.6 violating steps (0.261 m) to halve the time |
| **Acceptable, dominated** | `fm @ unet` K=10 `post_processing` (111.9 ms) and K=10 `dpcc-c` (241.9 ms) | S&C 10 with zero violations, but every one of the 25 `fm` S&C-10 rows is dominated — 23 of them by the single row `af @ sit` K=5 `geo_free-bounds_free` |
| **Least acceptable** | `af @ sit` K=1 `dpcc-c` — S&C 4, 266.3 steps, 44.0 ms | non-dominated only because nothing cheaper scores ≥ 4. 61.2 violating steps, `gdist` 0.94 m |
| **Not acceptable** | everything at S&C ≤ 3, all HardFlow rows, all of `mf @ unet` | HF's cheapest S&C-10 row is 380.9 ms — 6.6× the matched `af @ sit` K=5 `dpcc-c`, for identical scores. `mf @ unet` places 2 rows in the S&C-10 tier, both dominated |

**Caveat carried from §0/§7:** every row above is single-seed (6), 10 rollouts, and `af @ sit`
is 10.00 M parameters against 3.96 M. The domination in §2e.3 is real on these numbers and
still not architecture-matched.
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
| `af@sit` flies constraint-clean at lower NFE than `fm@unet` | **Yes** (§2d.1). Zero-violation DPCC row at NFE 8.5 vs fm's 16.5 (**1.94×**); zero-violation `geo_free` row at NFE 8.5 vs fm's 29.1 (**3.42×**). `mf@unet` reaches neither at any K. Same param confound |
| `af@sit` has the widest constraint-clean band | **No.** 9/22 clean variants at K=5; `fm@unet` reaches 14/22 at K=20. af has no K=10/20 rows, so the top of its band is unmeasured (§2c.2) |
| `af@sit` Pareto-dominates the matched-projector baseline on (S&C, steps, time) | **Yes** (§2e.3). `af@sit K5 dpcc-c` = S&C 10, 0 viol, 263.7 steps, 169.2 ms vs `fm@unet K10 dpcc-c` = S&C 10, 0 viol, 270.5 steps, 241.9 ms — better on all three axes at 1.94× fewer NFE. All 4 Pareto-optimal S&C-10 rows in the sweep are `af@sit K=5`; all 25 `fm@unet` S&C-10 rows are dominated. Same param confound |

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
saturating at 10/10 by K=5 at half the NFE and a third of the generation cost of `fm@unet K10` —
and flying **zero constraint violations at NFE 8.5** where `fm@unet` needs 16.5 (DPCC) or 29.1
(`geo_free`, projector geometry off) and `mf@unet` never gets there — but
at **10.00 M parameters against 3.96 M** — the deferred appendix arm the config says it is — so it
stays an alpha-Flow-*plus*-capacity result until `af @ unet` is trained.
