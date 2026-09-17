#!/usr/bin/env python3
"""D3IL-avoiding: every selection rule for every model, at DPCC's protocol and at the extended one.

    python3 Data_Analysis/DA_in_Paper/analysis/avoiding_rules_by_protocol.py

Source: analysis_results_checkpoint/15-09/batch_avoiding_combined_20260915_100757 (long format, gzipped).
Protocols
  DPCC   5 training seeds x 2 episodes per geometry = 10 episodes (DPCC's released config: n_trials 2)
  ext.   5 training seeds x 20 episodes per geometry = 100 episodes
Aggregation: mean per geometry over the seeds present, then mean over the geometries (the diffusion baseline
has both-hard at 20 episodes for seed 6 only). Tightened constraints, 4 candidate plans.
Only the architecture-matched temporal U-Net checkpoints are listed; the 2-episode MeanFlow and alpha-Flow
runs over five seeds used DiT/SiT backbones and are not comparable -- they are reported as missing.
"""
import csv
import gzip
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/15-09/batch_avoiding_combined_20260915_100757/'
                         'candidates_multidimensional_raw.csv.gz')
RULES = ('dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened')
GEOS = ('top-left-hard', 'top-right-hard', 'both-hard')
CELLS = {
  'DPCC protocol (5 seeds x 2 episodes)': [
    ('Diffusion (DPCC)', 20, 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5'),
    ('Flow matching', 20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE'),
  ],
  'extended (5 seeds x 20 episodes)': [
    ('Diffusion (DPCC)', 20, 'H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials'),
    ('Flow matching', 1, 'H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials'),
    ('Flow matching', 2, 'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials'),
    ('Flow matching', 20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials'),
    ('MeanFlow', 1, 'H8_K1_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials'),
    ('MeanFlow', 2, 'H8_K2_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials'),
  ],
}


def main():
    want = {f for cells in CELLS.values() for _, _, f in cells}
    d = defaultdict(dict)
    with gzip.open(CSV, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['Folder_Name'] in want and r['variant'] in RULES:
                try:
                    d[(r['Folder_Name'], r['variant'], r['halfspace_variant'], r['seed'])][r['metric']] = float(r['value'])
                except ValueError:
                    pass
    for proto, cells in CELLS.items():
        print(f'== {proto} ==')
        for name, K, fol in cells:
            for rule in RULES:
                per_geo = []
                seeds_seen = set()
                for g in GEOS:
                    rows = [m for (f, v, gg, s), m in d.items() if f == fol and v == rule and gg == g]
                    seeds_seen |= {s for (f, v, gg, s) in d if f == fol and v == rule and gg == g}
                    if rows:
                        per_geo.append({k: st.mean(m[k] for m in rows) for k in ('n_success_and_constraints', 'n_steps', 'avg_time')})
                if not per_geo:
                    print(f'  {name:17s} K={K:<2d} {rule:17s} (no rows)')
                    continue
                sc = st.mean(x['n_success_and_constraints'] for x in per_geo)
                stp = st.mean(x['n_steps'] for x in per_geo)
                ms = 1000 * st.mean(x['avg_time'] for x in per_geo)
                print(f'  {name:17s} K={K:<2d} {rule:17s} S&C={sc:.3f} steps={stp:5.1f} ms/step={ms:7.1f} '
                      f'geos={len(per_geo)} seeds={len(seeds_seen)}')
        print()


if __name__ == '__main__':
    main()
