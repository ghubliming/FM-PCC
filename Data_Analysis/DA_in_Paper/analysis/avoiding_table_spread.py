#!/usr/bin/env python3
"""D3IL-avoiding: every published cell as mean +/- spread, for the +/- columns of Chapter 6.

    python3.14 Data_Analysis/DA_in_Paper/analysis/avoiding_table_spread.py

Companion of avoiding_rules_by_protocol.py: same corpus, same cells, same aggregation -- this one
additionally reports the variation the point estimate hides.

WHICH SPREAD, AND WHY
---------------------
A cell of `tab:avoiding-dpcc-protocol` / `tab:state-headline` is five training seeds x three constraint
geometries. Two different spreads could be quoted and they answer different questions:

  * across EPISODES within a cell  -- how noisy one evaluation is.  Already in the CSV as `<metric>_std`.
  * across TRAINING SEEDS          -- how much the result depends on which model was trained.

The thesis reports the second, because every claim in Chapter 6 is a claim about a MODEL and not about
one training run, and because it is the larger and therefore the honest one. The procedure is the one
`avoiding_seed_spread.py` already uses for `tab:seed-spread`: average each seed over the geometries it
covers, then take the mean and the SAMPLE standard deviation over the seeds.

The point estimates printed here reproduce avoiding_rules_by_protocol.py exactly wherever every geometry
carries every seed; where a geometry is short of seeds the two differ slightly and the mean printed here
is the seed-mean, which is the one the +/- belongs to. Cells with fewer than two seeds get no +/-.
Stdlib only.
"""
import csv
import gzip
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/'
                         'batch_avoiding_combined_20260919_132703/candidates_multidimensional_raw.csv.gz')
RULES = ('dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened')
GEOS = ('top-left-hard', 'top-right-hard', 'both-hard')
MF = 'Dflow_matcher_v3_meanflow.models.MeanFlowODE'
AF = 'Dflow_matcher_v3_alphaflow.models.AlphaFlowODE'
M = ('n_success_and_constraints', 'n_steps', 'avg_time')

CELLS = {
  'DPCC protocol (5 seeds x 2 episodes)': [
    ('MeanFM', 1, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}', 'bbunet'),
    ('MeanFM', 2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}', 'bbunet'),
    ('CI-MeanFM', 1, f'H8_K1_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', 'bbunet+ae0.2'),
    ('CI-MeanFM', 2, f'H8_K2_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', 'bbunet+ae0.2'),
    ('FM', 1, 'H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msgdpccproto', None),
    ('FM', 2, 'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msgdpccproto', None),
    ('FM', 20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE', None),
    ('Diffusion', 1, 'H8_K1_T0.5_Dmodels.GaussianDiffusion', None),
    ('Diffusion', 10, 'H8_K10_Dmodels.GaussianDiffusion_aw10_thres0.5', None),
    ('Diffusion', 20, 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5', None),
  ],
  'extended (5 seeds x 20 episodes)': [
    ('MeanFM', 1, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', 'bbunet'),
    ('MeanFM', 2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', 'bbunet'),
    ('FM', 1, 'H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('FM', 2, 'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('FM', 20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('Diffusion', 20, 'H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials', None),
  ],
}


def backbone(path):
    bb = 'bbunet' if 'bbunet' in path else 'bbmf_dit' if 'bbmf_dit' in path else ''
    return 'bbunet+ae0.2' if bb == 'bbunet' and '_ae0.2' in path else bb


def load():
    want = {f for cells in CELLS.values() for _, _, f, _ in cells}
    d = defaultdict(dict)
    with gzip.open(CSV, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['Folder_Name'] in want and r['variant'] in RULES:
                try:
                    d[(r['Folder_Name'], backbone(r['Full_Path']), r['variant'],
                       r['halfspace_variant'], r['seed'])][r['metric']] = float(r['value'])
                except ValueError:
                    pass
    return d


def main():
    d = load()
    for proto, cells in CELLS.items():
        print(f'== {proto} ==')
        for name, K, fol, bb_want in cells:
            for rule in RULES:
                per_seed = defaultdict(dict)      # seed -> geometry -> metrics
                for (f, bb, var, geo, seed), m in d.items():
                    if f == fol and var == rule and geo in GEOS and (bb_want is None or bb == bb_want):
                        per_seed[seed][geo] = m
                if not per_seed:
                    continue
                out = []
                for k in M:
                    vals = []
                    for seed, geos in per_seed.items():
                        g = [gm[k] for gm in geos.values() if k in gm]
                        if g:
                            vals.append(st.mean(g))
                    if not vals:
                        out.append('   n/a  ')
                        continue
                    mu = st.mean(vals)
                    sd = st.stdev(vals) if len(vals) > 1 else None
                    scale = 1000.0 if k == 'avg_time' else 1.0
                    dp = 3 if k == 'n_success_and_constraints' else 1
                    s = f'{mu * scale:.{dp}f}'
                    s += f' +/- {sd * scale:.{dp}f}' if sd is not None else ' (1 seed)'
                    out.append(s)
                ngeo = sorted({g for geos in per_seed.values() for g in geos})
                print(f'  {name:10s} K={K:<3d} {rule[5]}  S&C={out[0]:20s} steps={out[1]:16s} '
                      f'ms={out[2]:18s} seeds={len(per_seed)} geos={len(ngeo)}')
        print()


if __name__ == '__main__':
    main()
