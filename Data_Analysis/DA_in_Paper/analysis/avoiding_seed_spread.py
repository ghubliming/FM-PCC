#!/usr/bin/env python3
"""D3IL-avoiding: variation across the five training seeds -- the numbers of `tab:seed-spread` (v3 §5.5.1).

    python3 Data_Analysis/DA_in_Paper/analysis/avoiding_seed_spread.py

Source: analysis_results_checkpoint/15-09/batch_avoiding_combined_20260915_100757 (long format, gzipped).
Each seed is averaged over top-left-hard and top-right-hard: the diffusion baseline covers both-hard with
seed 6 only, so the two single-halfspace geometries are the ones every seed of every model has.
Mean and sample standard deviation are then taken over seeds 6-10. Stdlib only.
"""
import csv
import gzip
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/15-09/batch_avoiding_combined_20260915_100757/'
                         'candidates_multidimensional_raw.csv.gz')
CELLS = {   # thesis name, K -> (evaluation folder, selection rule as in tab:state-headline)
    ('MeanFlow', 1): ('H8_K1_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials', 'dpcc-t-tightened'),
    ('Flow matching', 2): ('H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', 'dpcc-c-tightened'),
    ('Diffusion (DPCC)', 20): ('H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials', 'dpcc-c-tightened'),
}
GEOS = ('top-left-hard', 'top-right-hard')
SEEDS = ('6', '7', '8', '9', '10')


def main():
    folders = {f for f, _ in CELLS.values()}
    d = defaultdict(dict)
    with gzip.open(CSV, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['Folder_Name'] in folders:
                try:
                    d[(r['Folder_Name'], r['variant'], r['seed'], r['halfspace_variant'])][r['metric']] = float(r['value'])
                except ValueError:
                    pass
    per = {}
    for (name, K), (fol, var) in CELLS.items():
        for s in SEEDS:
            v = [d[(fol, var, s, g)] for g in GEOS]
            per[(name, s)] = (st.mean(x['n_success_and_constraints'] for x in v),
                              st.mean(x['n_steps'] for x in v),
                              1000 * st.mean(x['avg_time'] for x in v))
    print('per seed (S&C / control steps / ms per step), mean over top-left-hard and top-right-hard')
    for s in SEEDS:
        print(f'  seed {s:>2}  ' + '   '.join(f'{n} {per[(n, s)][0]:.3f}/{per[(n, s)][1]:.1f}/{per[(n, s)][2]:.1f}'
                                        for n, _ in CELLS))
    print('\nacross seeds 6-10: mean +- sd')
    for name, K in CELLS:
        cols = [[per[(name, s)][i] for s in SEEDS] for i in range(3)]
        print(f'  {name:18s} K={K:<2d} S&C {st.mean(cols[0]):.3f} +- {st.stdev(cols[0]):.3f}   '
              f'steps {st.mean(cols[1]):.1f} +- {st.stdev(cols[1]):.1f}   ms {st.mean(cols[2]):.1f} +- {st.stdev(cols[2]):.1f}')
    ratios = [per[('Diffusion (DPCC)', s)][2] / per[('MeanFlow', s)][2] for s in SEEDS]
    print(f'\n  time ratio diffusion / MeanFlow per seed: {[round(x, 1) for x in ratios]}')


if __name__ == '__main__':
    main()
