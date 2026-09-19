#!/usr/bin/env python3
"""D3IL-avoiding: every selection rule for every model, at DPCC's protocol and at the extended one.

    python3 Data_Analysis/DA_in_Paper/analysis/avoiding_rules_by_protocol.py

Source: analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703
(long format, gzipped). It supersedes the 15-09 batch and adds the 2026-09-17 wave: flow matching and
consistency-interpolated MeanFlow (U-Net, floor 0.2) at K=1 and K=2 at DPCC's protocol, tagged `_msgdpccproto`,
over seeds 6-10 -- the four rows `tab:avoiding-dpcc-protocol` carried as *not yet evaluated*.
Protocols
  DPCC   5 training seeds x 2 episodes per geometry = 10 episodes (DPCC's released config: n_trials 2)
  ext.   5 training seeds x 20 episodes per geometry = 100 episodes
Aggregation: mean per geometry over the seeds present, then mean over the geometries (the diffusion baseline
has both-hard at 20 episodes for seed 6 only). Tightened constraints, 4 candidate plans.
Backbone (2026-09-17 correction): Folder_Name does NOT carry the backbone, and the 2-episode MeanFlow cells
exist under BOTH `bbmf_dit` and `bbunet` model folders with the same Folder_Name. Selecting on Folder_Name
alone mixed the two and hid the architecture-matched U-Net rows, which were previously reported as missing.
Cells are now selected by (Folder_Name, a substring of Full_Path naming the backbone).
The 5-seed alpha-Flow cells at K=1/K=2 are `bbsit` with `ae0.0` -- the consistency ratio annealed to zero,
which is MeanFlow's target -- so they are not the consistency-interpolated model of the thesis and stay
excluded; that model exists on the U-Net for seed 6 only.
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
CELLS = {
  'DPCC protocol (5 seeds x 2 episodes)': [
    ('Diffusion (DPCC)', 20, 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5', None),
    ('Flow matching', 20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE', None),
    ('MeanFlow', 1, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}', 'bbunet'),
    ('MeanFlow', 2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}', 'bbunet'),
    ('MeanFlow [DiT]', 1, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}', 'bbmf_dit'),
    ('MeanFlow [DiT]', 2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}', 'bbmf_dit'),
    # 2026-09-17 wave (jobs 25878, 25879+25880). The consistency-interpolated cells are
    # keyed on 'ae0.2' as well as the backbone: the 5-seed alpha-Flow rows of the old batch
    # are bbsit with ae0.0 (MeanFlow's target) and must not be pooled with them.
    ('Flow matching', 1, 'H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msgdpccproto', None),
    ('Flow matching', 2, 'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msgdpccproto', None),
    ('CI-MeanFlow', 1, f'H8_K1_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', 'bbunet+ae0.2'),
    ('CI-MeanFlow', 2, f'H8_K2_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', 'bbunet+ae0.2'),
  ],
  'extended (5 seeds x 20 episodes)': [
    ('Diffusion (DPCC)', 20, 'H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials', None),
    ('Flow matching', 1, 'H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('Flow matching', 2, 'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('Flow matching', 20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('MeanFlow', 1, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', 'bbunet'),
    ('MeanFlow', 2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', 'bbunet'),
  ],
}


def main():
    want = {f for cells in CELLS.values() for _, _, f, _ in cells}
    d = defaultdict(dict)
    with gzip.open(CSV, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['Folder_Name'] in want and r['variant'] in RULES:
                bb = ('bbunet' if 'bbunet' in r['Full_Path'] else
                      'bbmf_dit' if 'bbmf_dit' in r['Full_Path'] else '')
                if bb == 'bbunet' and '_ae0.2' in r['Full_Path']:
                    bb = 'bbunet+ae0.2'
                try:
                    d[(r['Folder_Name'], bb, r['variant'], r['halfspace_variant'], r['seed'])][r['metric']] = float(r['value'])
                except ValueError:
                    pass
    for proto, cells in CELLS.items():
        print(f'== {proto} ==')
        for name, K, fol, bb_want in cells:
            for rule in RULES:
                per_geo = []
                seeds_seen = set()
                for g in GEOS:
                    def sel(f, b, v, gg, rule=rule, g=g):
                        return (f == fol and v == rule and gg == g
                                and (bb_want is None or b == bb_want))
                    rows = [m for (f, b, v, gg, s), m in d.items() if sel(f, b, v, gg)]
                    seeds_seen |= {s for (f, b, v, gg, s) in d if sel(f, b, v, gg)}
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
