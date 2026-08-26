# K (sampler steps) sweep — visual-aligning, distance axis

**Date:** 2026-08-26 · **Task:** aligning-d3il-visual · **Batch:** `temp/2608/batch_va2_20260826_142750`

📄 **Full analysis:**
[`logs_in_develop/Gen14/DA_20260826_Gen14_K_sampler_steps_MF_AF_FM_diffusion.md`](../../logs_in_develop/Gen14/DA_20260826_Gen14_K_sampler_steps_MF_AF_FM_diffusion.md)

High-K vs low-K on four engines, `diffuser` variant, n=30, seed 6, train split, `unet` bone throughout.
`K` = sampler/integrator steps (NFE), **not** the MPC fan (`mpc4`, fixed at 4). Metric is raw
`context_final_xy_dist`, reported as fraction of the starting gap left.

| engine | high K | low K | note |
|---|---|---|---|
| MeanFlow | 0.28× (K100) | 0.60× (K2) | K100 better, trend only (p=0.069) |
| AlphaFlow | 0.69× (K100) | 0.29× (K2) | **reversed** — low K better (p=0.008) |
| FlowMatching | 0.95× (K100) | 0.98× (K20) | flat; arm doesn't move the box |
| Diffusion | 0.41× (K100) | 0.96× (K20) | ⚠️ checkpoints differ, confounded |

No consistent direction — `K` must be set per engine. Cost tracks NFE (5–32×).

**Side observation worth following up:** the d3il vision baseline (`d3il_baseline_ddpm_encdec_vision`, test
split, n=1080/2804) sits at 1.000× / 0.999× — the box ends where it started, with 56–70 % of rollouts never
moving it by 5 mm. Not directly comparable to the rows above (train vs test split, `geo=none` vs
`combined_5`), so it is flagged, not claimed.

⚠️ Binary success is 0–2/30 everywhere and does not rank anything — the gate needs `pos ≤ 1.8 cm` **and**
`rot ≤ 8.64°`, and rotation is uncontrolled on every arm. Also: `mean_dist_per_rollout` is not a distance
(it is `0.5*(pos + rot/π)`) — don't use it for distance claims.
