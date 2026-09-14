# UAV `pillars` — DPCC-diffusion baseline (K=20) reference row vs mf / fm / af

**Full DA:** [`logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260912_pillars_diffusion_baseline_reference.md`](../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260912_pillars_diffusion_baseline_reference.md) (rev 2, 2026-09-13)
**Data:** `temp/1209/batch_uav_20260912_201035` · candidates C59 (diffusion K20), C78/C66/C53 (mf/fm/af K5), C74/C64/C52 (K2), C50 (af K1) · seed 6, n=10, `pillars_hg`

**Description:** DPCC-diffusion (U-Net 3.96 M, `action_weight=1`) on honest-geometry `pillars`, next to the
architecture-matched flow engines. Diffusion crosses the finish line on 50/50 rollouts. It never comes within 0.30 m of
the goal point and breaches the pillar clearance on 101–156 steps per flight, so S&C is 0.00 strict / ≤ 0.30 crossed-line.
mf/fm/af at K=5 reach 0.70–1.00 / 0.90–1.00. It is a reference row (one seed, unmatched K), not a matched-budget claim.
