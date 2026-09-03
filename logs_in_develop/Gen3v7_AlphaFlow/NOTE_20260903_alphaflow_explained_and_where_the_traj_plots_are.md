# NOTE — What α-Flow is, what "AF **enabled**" means, and where the trajectory plots live

**Date:** 2026-09-03 · **Gen:** 3v7 (AlphaFlow)
**Tree capture:** `temp/0309/alphaflow_plans_tree.txt` (2026-09-03 22:30, 30 173 files, 2.6 GiB)
**Analysis:** [`DA 2026-09-03 — With α actually ON, does α-Flow beat MeanFlow on the U-Net?`](DA/DA_20260903_AF_UNet_alphaflow_ENABLED_seed6_diffuser.md)
**Paper report:** [`Report_20260903_AF_UNet`](../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md)

---

## 1. What α-Flow actually is

Three ways to train the same 4.0 M U-Net to produce a plan:

| engine | what the net learns | sampling cost |
|---|---|---|
| **naive FM** | instantaneous velocity `v(x, t)` | K Euler steps — K is a real cost |
| **MeanFlow** | average velocity `u(x, r, t) = 1/(t−r) ∫ᵣᵗ v dτ`, via an **analytic JVP** target | K = 1 is the design point |
| **α-Flow** | the same `u`, but via a **bootstrapped** target | K = 1 is the design point |

α-Flow's target (`flow_matcher_v3_alphaflow/models/af_diffusion.py`):

```
dt      = α · h
z_shift = x_r + dt · v                                   # take a short probe step
u_next  = net(z_shift, r + dt, h − dt)      # no_grad     # ask the net what happens after it
u_tgt   = ( dt · v  +  (h − dt) · u_next ) / h            # stitch the two together
```

The point of α: it is a **homotopy dial** between the two other engines.

- **α = 1** → `u_tgt = v`. The model is **plain Flow Matching**, bitwise.
- **0 < α < 1** → the genuine α-Flow bootstrap: a short exact step plus the network's own
  (detached) estimate of the rest.
- **α = 0** → `af_diffusion.py:552` takes the `alpha <= 0.0` branch, which is Gen3v6's MeanFlow JVP
  body **unmodified**. The model **is MeanFlow**, byte-for-byte.

Training anneals α from 1 down to a floor with a sigmoid (γ = 25) over 100 k steps — a curriculum
that walks from FM toward MeanFlow. **`af_alpha_end` is where the walk stops.**

---

## 2. What "AF **enabled**" means — and why it matters

### The problem

`af_alpha_end` defaulted to **0.0**. So every α-Flow training run in this project annealed α all the
way to zero, and from ~step 62 000 onward **took the MeanFlow branch for the rest of training**. The
saved checkpoint (step 80 000) had α = 0.0006.

> **Every α-Flow checkpoint deployed before 2026-09-01 was a MeanFlow model.**
> α-Flow's own objective had never trained a single deployed weight. "AF vs MF" was
> "MF-with-a-curriculum vs MF".

This is not a bug in the port — it is upstream's design (`aux_repo/alphaflow` ships
`end_value: 0` everywhere; the curriculum is *supposed* to land on MeanFlow). It just means the
bootstrapped objective was never what we were measuring.

### The fix

`AF_ALPHA_END` (env var → `config/avoiding-d3il.py:129`, applied at train `:952` and plan `:1599`)
holds α at a **floor** instead of letting it collapse. Default stays `0.0`, so every historic path
is byte-identical. Upstream's equivalent switch is `discrete_training: true`.

`af_alpha_end` is an **unconditional** savepath token (`'ae'`), so each value gets its own tree and
`--auto-resume` cannot collide across floors.

### How to tell, for any run, whether AF was on

| signal | AF **OFF** (is MeanFlow) | AF **ON** |
|---|---|---|
| `[ train ] AF_ALPHA_END=` in the train log | `0.0` | `0.05`, `0.2`, … |
| `alpha` in the final epoch line | ≈ 0.0006 | = the floor |
| **`train/discrete_frac`** ← *the decisive one* | **`0.0`** | **≈ 0.25 – 0.5** |
| checkpoint-folder token | `_ae0.0_` | `_ae0.05_`, `_ae0.2_` |

`discrete_frac` is the fraction of the batch that took the bootstrapped no-grad branch. **`0.0`
means not one sample used α-Flow's target.** It tracks `rf0.5` (the FM/u split) when AF is on.

---

## 3. The five α-Flow checkpoint trees on the cluster

Root: `/u/home/llim/FMPCC/FM-PCC/logs/avoiding-d3il/plans/flow_matching_v3_alphaflow/`
(line numbers = `temp/0309/alphaflow_plans_tree.txt`)

| # | tree | backbone | AF? | seeds | note |
|---:|---|---|---|---|---|
| 1 | `(Unsafe)…bbsit…_ae0.0_…` | SiT | ❌ off | 7 | quarantined, `(Unsafe)` prefix — do not use |
| 2 | `…bbsit…_ae0.0_…` (L2607) | SiT | ❌ off | 6–10 | the AF-SiT `msg20trials` runs — **actually MeanFlow-on-SiT** |
| 3 | `…bbunet…_ae0.0_…` (L8563) | U-Net | ❌ off | 7 | the `msgac05_latest` run — **actually MeanFlow-on-U-Net** |
| 4 | **`…bbunet…_ae0.2_…`** (L10390) | **U-Net** | ✅ **ON** | **6** | **arm B — the headline run**, 405.5 MiB |
| 5 | **`…bbunet…_ae0.05_…`** (L11442) | **U-Net** | ✅ **ON** | **6** | **arm A** — low floor, 407.3 MiB |

Full names of 4 and 5:

```
H8_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.2_ag25.0_rf0.5
H8_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.05_ag25.0_rf0.5
```

Token glossary: `aw10` action weight · `bbunet` backbone · `tslogit_normal` time sampler ·
`ai1.0` α **initial** · **`ae…` α end — the floor, i.e. the AF-on switch** · `ag25.0` anneal γ ·
`rf0.5` FM/u split.

⚠️ **Trees 2 and 3 are labelled "AlphaFlow" but are MeanFlow.** Anything published from them is a
MeanFlow result under an α-Flow name. Only 4 and 5 are α-Flow.

---

## 4. Where the trajectory plots are

Each AF-ON tree holds five K-runs; each K-run holds seed `6`; each seed holds three halfspace
environments; each environment holds **one PNG + one PDF + one NPZ per projection arm**.

```
<checkpoint tree>/
  H8_K{1,2,5,10,20}_Meuler_T0.5_A0.5_B4_D…AlphaFlowODE_msgafon02_s6/     ← α→0.2
  H8_K{1,2,5,10,20}_Meuler_T0.5_A0.5_B4_D…AlphaFlowODE_msgafon005_s6/    ← α→0.05
      └── 6/                                   ← the seed
          ├── run_provenance.json              ← what actually ran (read this first)
          ├── config_snapshot_avoiding-d3il/   ← avoiding-d3il.py + alphaflow_projection_eval.yaml
          └── results/
              ├── halfspace_top-left-hard/
              ├── halfspace_top-right-hard/    ← the discriminating environment
              └── halfspace_both-hard/
                  ├── diffuser.png / .pdf / .npz          ← ⭐ RAW network output, no projection
                  ├── dpcc-{r,c,t}.png / -tightened.*     ← after the DPCC projector
                  ├── hardflow_sls-*.png                  ← arm C (see the warning below)
                  ├── all_slsqp.png                       ← every arm on one sheet
                  ├── eval_<arm>.log                      ← the per-arm summary numbers
                  ├── realtime_<arm>_trial<0..19>.log     ← per-episode traces, 20 per arm
                  └── HF_DEGENERATE_SKIPPED.txt           ← present at K ≤ 2 (see below)
```

**For plan quality, use `diffuser.png`.** It is the unprojected rollout — the network's own plan,
before the MPC repairs anything. That is the only panel that shows what α-Flow changed.
`.npz` holds the same trajectories numerically, for computing jerk / path length / curvature.

### Exact paths for the eight K1/K2 panels

Prefix everything with `/u/home/llim/FMPCC/FM-PCC/logs/avoiding-d3il/plans/`, and suffix with
`/6/results/halfspace_<env>/diffuser.png`.

| panel | engine | K | path fragment |
|---|---|---:|---|
| 8a | **α-Flow `α→0.2`** | 1 | `flow_matching_v3_alphaflow/…_ae0.2_ag25.0_rf0.5/H8_K1_Meuler_T0.5_A0.5_B4_D…AlphaFlowODE_msgafon02_s6` |
| 8b | **α-Flow `α→0.2`** | 2 | same tree, `H8_K2_…_msgafon02_s6` |
| 8c | MeanFlow-UNet | 1 | `flow_matching_v3_meanflow/H8_D…MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/H8_K1_Meuler_T0.5_A0.5_B1_D…MeanFlowODE_msg20trials` |
| 8d | MeanFlow-UNet | 2 | same tree, `H8_K2_…_msg20trials` |
| 8e | naive FM | 1 | `flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials` |
| 8f | naive FM | 2 | same tree, `H8_K2_…_msg20trials` |
| 8g | DPCC diffusion | 1 | ⚠️ **no K=1 `msg20trials` run exists** — nearest is `diffusion/H8_K1_Dmodels.GaussianDiffusion_aw10/H8_K1_T0.5_Dmodels.GaussianDiffusion` (n_trials = 2) |
| 8h | DPCC diffusion | 2 | ⚠️ **does not exist.** The baseline's K is a *training* parameter; its only 20-trial run is `diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10/H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials` at **K = 20** |

Also add `α→0.05` if you want the dose–response: same as 8a/8b with `_ae0.05_` and `_msgafon005_s6`.

These are the eight placeholders in
[`Report_20260903_AF_UNet/README.md` §8](../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md).

### Pulling them down

```bash
# on a machine that can reach the cluster
R=/u/home/llim/FMPCC/FM-PCC/logs/avoiding-d3il/plans
A=$R/flow_matching_v3_alphaflow/H8_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.2_ag25.0_rf0.5

for K in 1 2; do
  scp i6-gpu-1:"$A/H8_K${K}_Meuler_T0.5_A0.5_B4_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msgafon02_s6/6/results/halfspace_both-hard/diffuser.png" \
      ./fig8_af_K${K}.png
done
```

**Do not `scp -r` a whole tree** — arms A and B are ~406 MiB each, and 53 % of that is PNG
(`temp/0309/alphaflow_plans_tree.txt`, by-extension table). Pull single files.

---

## 5. Two traps in these folders

1. **`hardflow_sls-*` is not comparable to MeanFlow's `hardflow_new-*`.** The AF runs used a
   4-candidate HardFlow fan (`hf_batch=4`, the `B4` folder token); the MeanFlow `msg20trials` run
   used 1 (`B1`). AF picks best-of-4. **Every AF-vs-MF HardFlow number is void.**
   The `B` token is `hf_batch_size` (`config/avoiding-d3il.py:202`) — it is *not* the MPC fan that
   `diffuser` and `dpcc-*` use, which is `batch_size`, not a folder token
   (`config/avoiding-d3il.py:67`) and equal to 4 on every run compared here. So `diffuser` and
   `dpcc-*` **are** fair; only arm C is not.
2. **`HF_DEGENERATE_SKIPPED.txt` at K ≤ 2.** At `A = 0.5` with K ≤ 2 there is no HardFlow math to
   run, so the eval skips the arm and drops this marker. Older runs (MeanFlow, 2026-08-13) predate
   the guard and still emit degenerate K1/K2 HardFlow rows — **do not use them.**

---

## 6. One-line summary

> `_ae0.0_` in the folder name means the run is **MeanFlow wearing an α-Flow label**.
> `_ae0.2_` / `_ae0.05_` (seed 6, tags `msgafon02_s6` / `msgafon005_s6`) are the **only** runs in
> this repo where α-Flow's bootstrapped objective actually trained the deployed weights.
> For plan quality, open `…/6/results/halfspace_<env>/diffuser.png`.
