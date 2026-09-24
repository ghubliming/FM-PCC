#!/usr/bin/env python3.14
"""D3IL-aligning: which contexts end IN POSITION, and the final box-to-target distance of each, for the supported
combination (MeanFM, nfe 20, endpoint projection, random selection, tightened) and its unprojected plan (v3.95,
author: "10/10 contexts free of violations, box moved in 8/10 -- not just moved! show which is in position and the
dist_xy").

In position = final xy distance <= 0.018 m, the position half of D3IL-aligning's own success test
(d3il/.../gym_aligning/envs/aligning.py:198, pos_min_dist = 0.018; the rotation half, rot_min_dist = 0.048, is not
used: the recorded box angle's convention is inconsistent across runs, Ch 5 sec:setup:metrics:aligning).

    python3.14 Data_Analysis/DA_in_Paper/analysis/aligning_in_position.py
"""
import csv
import os
import statistics as st

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/15-09/batch_va2_20260915_100754/per_rollout_detail.csv')
TOL = 0.018
rows = list(csv.DictReader(open(CSV)))
for variant in ('hardflow_sls-r', 'diffuser'):
    sel = [r for r in rows if 'K20_Meuler_T0.2' in r['FolderName'] and 'visual_mf_diffusion' in r['FolderName']
           and r['variant'] == variant and r['geo'] == 'combined_5-tightened']
    assert len(sel) == 10, (variant, len(sel))
    print(f'MeanFM K20 {variant}:')
    for r in sorted(sel, key=lambda r: float(r['context_final_xy_dist'])):
        d, d0 = float(r['context_final_xy_dist']), float(r['context_init_xy_dist'])
        moved = not (float(r['frozen']) or abs(d - d0) < 1e-6)
        print(f'  rollout {r["rollout_idx"]:>2s}: final {d * 100:5.1f} cm of {d0 * 100:4.1f} ({(d0 - d) / d0 * 100:5.1f} % '
              f'closed)  in position {d <= TOL}  moved {moved}  D3IL success {r["n_success"]}  violations {r["n_violations"]}')
    ds = [float(r['context_final_xy_dist']) for r in sel]
    print(f'  median {st.median(ds):.3f} m, mean {st.mean(ds):.3f} +- {st.stdev(ds):.3f}, in position '
          f'{sum(x <= TOL for x in ds)}/10')
