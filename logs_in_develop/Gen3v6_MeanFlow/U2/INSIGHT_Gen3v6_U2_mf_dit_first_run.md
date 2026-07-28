# Gen3v6 U2 — first run of the official-MeanFlow DiT backbone (`imf_backbone='mf_dit'`)

**Run:** train job 23926 / eval job 23927 (`temp/2807/2807`), git `b82e290`, node i6-gpu-1.
**Config:** `H8`, `aw10`, `obj=meanflow`, **`bb=mf_dit`**, `dp0.5`, dual_head, `dit_hidden=256`,
`dit_depth=8`, 100k steps, EMA 0.995, `gradient_clip=1.0`, adaptive-L2 (p=1.0, eps=0.01).
**Eval:** K=2, Euler, threshold 0.5, seed 6, **2 trials/cell** (directional only).

This is the first cluster run of the network added in U2 — the faithful port of the **MeanFlow
paper's own DiT** (`aux_repo/MeanFlow/models/dit.py`, adaLN-zero), as opposed to the `dit` arm
which is the **iMF** transformer (RoPE + in-context tokens + SwiGLU).

---

## Headline

**The MeanFlow-paper architecture trains, and DPCC projects its output to 100%-safe K=2 control.**
This is the decisive contrast with U1's UNet arm, which never left its initialisation:

| metric (train, EMA-free) | UNet (U1, `bbunet`) | iMF-DiT (`bbdit`) | **MF-DiT (U2, `bbmf_dit`)** |
|---|---|---|---|
| `train/loss` (≈2.0 adaptive ceiling per head → ~1.0/head) | **0.9998 (frozen)** | broke ceiling | **0.9998 → 0.740** ✅ |
| `per_dim_rms_u` | 1.20 (stuck) | 0.187 | **0.195** ✅ |
| `raw_mse_u` | 64 → 70 (worse) | ↓ | **58.9 → 1.83** (32×) ✅ |
| `h_mse_b0` (FM anchor) | 66.9 → 46 | 58.6 → 1.60 | **58.6 → 2.50** ✅ |
| best checkpoint | **step 3000 (never improved)** | — | improved to the end ✅ |

So MF-DiT lands **essentially on top of the iMF-DiT** (`per_dim_rms_u` 0.195 vs 0.187), and both
are worlds apart from the frozen UNet. **The adaLN-zero conditioning learns the two-time field
just as well as iMF's token/RoPE conditioning does.**

---

## Training analysis

- **Loss descends monotonically** (0.9998 → 0.981 → 0.958 → 0.935 → 0.908 → 0.879 → 0.816 → 0.762
  → 0.740 across the 100k steps). It breaks below the 1.0/head adaptive ceiling — real learning,
  not the saturated plateau the UNet sat on. (As always: `train/loss` is adaptive-weighted and
  must NOT be read as an absolute convergence number — the raw MSEs below are the honest signal.)
- **`raw_mse_u` 58.9 → 1.83** (min 1.135), **`raw_mse_v` 58.9 → 1.77** — both u and v heads learn;
  the dual head is doing real work.
- **`a0_loss` 1.70 → 0.022** — the first-action prediction (what the controller actually consumes)
  becomes accurate.
- **`per_dim_rms_u` 1.108 → 0.195** — the deployed field's per-dim error drops ~5.7×.

### Two honest caveats visible in the curves

1. **Grad norm GROWS to ~527 (max 636)** from 3.6 at init — `grad_norm_history` logs the
   **pre-clip** norm, so with `gradient_clip=1.0` the clip is **biting hard** by end of training:
   updates are direction-only, magnitude-capped. Training stays stable and the loss keeps falling,
   but the large, rising raw gradients (iMF-DiT only reached ~28) suggest the adaLN-zero net drives
   a stiffer optimisation against the MeanFlow target. Worth watching if longer runs destabilise —
   a grad-clip sweep or LR check is the natural follow-up.
2. **Train/val gap (overfitting on the 96-trajectory dataset).** `raw_mse_u` train 1.83 vs
   **val 25.3**; `per_dim_rms_u` train 0.195 vs val 0.447. The large-h validation buckets are very
   noisy (`h_mse_b1/b2/b3` val = 39 / 40 / 20, with single-batch maxima in the 10³–10⁴ range) —
   the same sparse-bucket noise pattern flagged for the iMF-DiT's `b3` outlier, not divergence
   (train buckets are all ≤2.5). The control eval sidesteps this by using EMA weights + projection.

---

## Control eval (K=2, 2 trials — directional)

Full table across the three halfspace scenarios (`succ` / `con` / `g&c` = goal-and-constraints):

| scenario | variant | succ | con | g&c | total viol |
|---|---|---|---|---|---|
| top-right-hard | diffuser (raw) | 0.5 | 0.0 | **0.0** | 1.906 |
| | dpcc-r | 1.0 | 1.0 | **1.0** | **0.000** |
| | dpcc-r-tightened | 0.5 | 0.5 | 0.5 | 0.000 |
| | dpcc-t / -tightened | 0.5 | 0.0/0.5 | 0.0/0.5 | 0.006 / 0.000 |
| | **dpcc-c / -tightened** | **0.0** | 0.0 | **0.0** | 0.000 |
| top-left-hard | diffuser (raw) | 1.0 | 0.0 | **0.0** | 6.729 |
| | dpcc-r-tightened | 1.0 | 1.0 | **1.0** | **0.000** |
| | dpcc-t-tightened | 1.0 | 1.0 | **1.0** | **0.000** |
| | **dpcc-c / -tightened** | **0.0** | 0.0 | **0.0** | 0.000 |
| both-hard | diffuser (raw) | 1.0 | 0.5 | 0.5 | 2.254 |
| | dpcc-r / -tightened | 1.0 | 1.0 | **1.0** | **0.000** |
| | dpcc-t / -tightened | 1.0 | 1.0 | **1.0** | **0.000** |
| | **dpcc-c / -tightened** | **0.0** | 0.0 | **0.0** | 0.000 |

**Reading:**

- **Projection works on MF-DiT generation.** Raw `diffuser` rides 11–24 violations per plan
  (1.9–6.7 total); every DPCC-projected variant that reaches the goal does so with **0 violations**
  (tightened variants are clean everywhere they run). The safety half of the dual-path pipeline is
  intact on the new backbone — MF-DiT gives DPCC a projectable reference.
- **`dpcc-r` / `dpcc-t` carry the goal; the tightened forms are 100%-safe** on 2/3 scenarios each,
  and both are `g&c=1.0` / 0-violation on `both-hard`.
- **⚠️ `dpcc-c` collapses to `succ=0.0` (goal never reached) in all three scenarios** — a notable
  contrast with the iMF-DiT arm, where `dpcc-c-tightened` was the star (`g1/b1/v0` on all three).
  With 2 trials this means the cost-selected candidate (min control cost of K=2) was a "lazy"
  near-stationary trajectory that satisfies constraints trivially but never reaches the goal. This
  is a **selection-strategy interaction, not a training failure** (the net clearly learned — see
  above), but it is the single most important thing to confirm with more trials/seeds before
  concluding anything about MF-DiT vs iMF-DiT on control.

---

## What this establishes (and what it doesn't)

**Establishes** — the U2 hypothesis holds: the **DiT *class* is the load-bearing ingredient**, not
iMF's specific RoPE/token machinery. Swapping in the MeanFlow paper's plain adaLN-zero DiT
reproduces the iMF-DiT's training quality (`per_dim_rms_u` 0.195 ≈ 0.187) and yields DPCC-projectable,
100%-safe K=2 control. The MeanFlow objective now runs on the MeanFlow paper's own network — the
architecture/objective mismatch that motivated U2 is closed, and the three-way backbone A/B
(UNet ≪ iMF-DiT ≈ MF-DiT) is meaningful.

**Does NOT establish** — any ranking of MF-DiT vs iMF-DiT on control. This is **1 seed, 2 trials**;
the training curves are trial-count-independent and unambiguous, but the eval is directional. The
`dpcc-c` collapse in particular could be a 2-trial artifact or a real selection interaction.

## Recommended follow-ups

1. **More trials/seeds** on the MF-DiT eval (esp. to resolve the `dpcc-c` goal collapse — is it real
   or a 2-trial fluke?).
2. **Grad-norm watch:** the pre-clip norm rose to ~530 with clip=1.0; a longer run or a small LR /
   clip sweep would confirm stability isn't marginal.
3. **Matched three-way A/B** (UNet / iMF-DiT / MF-DiT) at the same K, seeds, trials, so
   MeanFlow-paper-DiT vs iMF-DiT becomes a clean single-variable architecture comparison under the
   fixed MeanFlow objective.

*(Scope note: an AlphaFlow/SiT run — `bbsit`, Gen3v7 — shares this `temp/2807/2807` batch but is a
different generation and is analysed separately; this MD covers only the Gen3v6 MF-DiT arm.)*
