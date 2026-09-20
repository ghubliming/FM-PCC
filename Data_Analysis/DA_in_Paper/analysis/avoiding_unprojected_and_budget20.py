#!/usr/bin/env python3
"""D3IL-avoiding: (A) the unprojected mirror of `tab:avoiding-dpcc-protocol`, and
(B) the flagship step-budget check, twenty network evaluations against two.

    python3 Data_Analysis/DA_in_Paper/analysis/avoiding_unprojected_and_budget20.py

Written for v3.52 (author, 2026-09-20). Two questions the draft could not answer before:

A.  `tab:avoiding-raw-models` was one training seed, one geometry and a mixture of budgets, so it
    could not be read beside the projected table. The same corpus carries the **unprojected arm**
    (`variant == 'diffuser'`) of exactly the cells `tab:avoiding-dpcc-protocol` reports, at five
    training seeds and three geometries -- so the unprojected table can be an exact mirror of the
    projected one, with the projector as the only difference between them.

B.  Filling every pending cell of the extended table needs evaluations (and four diffusion training
    runs) the author has ruled out as too expensive. The question those cells were there to answer --
    does a larger step budget buy anything once the projector is in the loop? -- is answered by the
    cells that already exist at twenty episodes: FM and MeanFM at five seeds, CI-MeanFM at seed 6,
    each at K=2 and at K=20, against the diffusion baseline at its own K=20.

Source: the same corpus of record as `avoiding_rules_by_protocol.py`,
analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703.
Aggregation is that script's: mean per geometry over the seeds present, then mean over the
geometries; the spread is the sample standard deviation over the per-seed geometry-means.

Naming trap (see MASTER naming note): `Dmodels.diffusion.GaussianDiffusion` under
`flow_matching_v3_*` is the FLOW model under its pre-26-May class name. The diffusion baseline is
the one whose Full_Path contains `/plans/diffusion/`, and every diffusion cell below is asserted
against that.
"""
import csv
import gzip
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/'
                         'batch_avoiding_combined_20260919_132703/candidates_multidimensional_raw.csv.gz')
GEOS = ('top-left-hard', 'top-right-hard', 'both-hard')
RULES = ('dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened')
MF = 'Dflow_matcher_v3_meanflow.models.MeanFlowODE'
AF = 'Dflow_matcher_v3_alphaflow.models.AlphaFlowODE'
METRICS = ('n_success', 'n_success_and_constraints', 'n_violations', 'n_steps', 'avg_time')

# (label, nfe, Folder_Name, backbone key or None)
UNPROJECTED = [                                     # mirrors tab:avoiding-dpcc-protocol row for row
    ('MeanFM',    1, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}', 'bbunet'),
    ('MeanFM',    2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}', 'bbunet'),
    ('CI-MeanFM', 1, f'H8_K1_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', 'bbunet+ae0.2'),
    ('CI-MeanFM', 2, f'H8_K2_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', 'bbunet+ae0.2'),
    ('FM',        1, 'H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msgdpccproto', None),
    ('FM',        2, 'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msgdpccproto', None),
    ('Diffusion', 1, 'H8_K1_T0.5_Dmodels.GaussianDiffusion', None),
    ('Diffusion', 20, 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5', None),
]

BUDGET20 = [                                        # extended protocol, 20 episodes per seed
    ('FM',        2,  'H8_K2_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('FM',        20, 'H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials', None),
    ('MeanFM',    2,  f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', 'bbunet'),
    ('MeanFM',    20, f'H8_K20_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', 'bbunet'),
    ('CI-MeanFM', 2,  f'H8_K2_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', 'bbunet+ae0.2'),
    ('CI-MeanFM', 20, f'H8_K20_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', 'bbunet+ae0.2'),
    ('Diffusion', 20, 'H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials', None),
]


def backbone(full_path):
    bb = ('bbunet' if 'bbunet' in full_path else
          'bbmf_dit' if 'bbmf_dit' in full_path else
          'bbsit' if 'bbsit' in full_path else '')
    for tag in ('ae0.2', 'ae0.05', 'ae0.0'):
        if '_' + tag in full_path:
            return bb + '+' + tag
    return bb


def load(folders):
    """-> {(folder, backbone, variant, geometry, seed): {metric: value}}, and the paths seen."""
    d = defaultdict(dict)
    paths = defaultdict(set)
    with gzip.open(CSV, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            if r['Folder_Name'] not in folders or r['metric'] not in METRICS:
                continue
            try:
                value = float(r['value'])
            except ValueError:
                continue
            d[(r['Folder_Name'], backbone(r['Full_Path']), r['variant'],
               r['halfspace_variant'], r['seed'])][r['metric']] = value
            paths[r['Folder_Name']].add(r['Full_Path'])
    return d, paths


def rows_of(d, folder, bb_want, variant, keep=None):
    """{(geometry, seed): metrics} for one cell; `keep` restricts to a set of (geometry, seed)."""
    out = {}
    for (f, b, v, g, s), m in d.items():
        if f == folder and v == variant and (bb_want is None or b == bb_want):
            if all(k in m for k in METRICS) and (keep is None or (g, s) in keep):
                out[(g, s)] = m
    return out


def agg(cells):
    """The chapter's convention, both halves of it:

    point estimate -- mean per geometry over the seeds present, then mean over the geometries
                      (avoiding_rules_by_protocol.py; what every published cell prints).
    spread         -- average each seed over the geometries it covers, then the sample standard
                      deviation over the seeds (avoiding_table_spread.py).

    They agree exactly wherever every geometry carries every seed and differ only on ragged cells,
    which is what the dagger of tab:state-headline records.
    """
    if not cells:
        return None
    geos = sorted({g for g, _ in cells})
    seeds = sorted({s for _, s in cells}, key=int)
    out = {'geometries': geos, 'seeds': seeds,
           'coverage': {g: sorted((s for gg, s in cells if gg == g), key=int) for g in geos}}
    for k in METRICS:
        per_geo = [st.mean(m[k] for (g, s), m in cells.items() if g == gg) for gg in geos]
        by_seed = [st.mean(m[k] for (g, s), m in cells.items() if s == ss) for ss in seeds]
        out[k] = (st.mean(per_geo), st.stdev(by_seed) if len(by_seed) > 1 else 0.0)
    return out


def line(name, K, rule, a):
    return (f'{name:<11}{K:>3}  {rule:<3}'
            + ''.join(f'{a[k][0]:>9.3f}+-{a[k][1]:<5.3f}' for k in
                      ('n_success', 'n_success_and_constraints'))
            + ''.join(f'{a[k][0]:>9.1f}+-{a[k][1]:<5.1f}' for k in ('n_violations', 'n_steps'))
            + f"{a['avg_time'][0]*1000:>9.1f}+-{a['avg_time'][1]*1000:<6.1f}"
            + f"  {len(a['seeds'])}x{len(a['geometries'])}")


HEAD = (f"{'model':<11}{'K':>3}  {'rul':<3}{'success':>16}{'S&C':>16}"
        f"{'viol.steps':>16}{'steps':>16}{'ms/step':>17}  cover")


def block_a(d):
    print('\n===== A. UNPROJECTED, DPCC protocol (5 seeds x 2 episodes) '
          '-- row-for-row mirror of tab:avoiding-dpcc-protocol =====')
    print(HEAD); print('-' * len(HEAD))
    for name, K, folder, bb in UNPROJECTED:
        a = agg(rows_of(d, folder, bb, 'diffuser'))
        print(line(name, K, '--', a) if a else f'{name:<11}{K:>3}   -- no data --')


def block_b(d):
    """K=20 against K=2 on the cells both budgets actually cover."""
    print('\n===== B. PROJECTED, extended protocol (20 episodes per seed) -- K=20 against K=2, '
          'MATCHED cells =====')
    print(HEAD); print('-' * len(HEAD))
    by_model = defaultdict(dict)
    for name, K, folder, bb in BUDGET20:
        by_model[name][K] = (folder, bb)
    for name, budgets in by_model.items():
        if not {2, 20} <= set(budgets):
            f2, bb2 = budgets[20]
            for rule in RULES:
                a = agg(rows_of(d, f2, bb2, rule))
                if a:
                    print(line(name, 20, rule.split('-')[1], a))
            print()
            continue
        for rule in RULES:
            f2, bb2 = budgets[2]
            f20, bb20 = budgets[20]
            c2 = rows_of(d, f2, bb2, rule)
            c20 = rows_of(d, f20, bb20, rule)
            keep = set(c2) & set(c20)
            if not keep:
                continue
            a2, a20 = agg(rows_of(d, f2, bb2, rule, keep)), agg(rows_of(d, f20, bb20, rule, keep))
            print(line(name, 2, rule.split('-')[1], a2))
            print(line(name, 20, rule.split('-')[1], a20))
            print(f'{"":<11}{"":>3}  {"":<3}{"ratio ms/step: %.1fx" % (a20["avg_time"][0] / a2["avg_time"][0]):>32}'
                  f"   matched on {sorted(set(g for g, _ in keep))}")
        print()


def coverage(d):
    print('\n===== C. WHAT EACH K=20 EXTENDED CAMPAIGN ACTUALLY COVERS =====')
    for name, K, folder, bb in BUDGET20:
        if K != 20:
            continue
        print(f'  {name} K=20  ({folder})')
        for variant in ('diffuser',) + RULES:
            r = rows_of(d, folder, bb, variant)
            if not r:
                print(f'      {variant:<20} -- nothing')
                continue
            cov = defaultdict(list)
            for g, s in r:
                cov[g].append(int(s))
            print(f'      {variant:<20} ' + '  '.join(
                f'{g}:{sorted(v)}' for g, v in sorted(cov.items())))
        print()


if __name__ == '__main__':
    print(__doc__)
    folders = {f for _, _, f, _ in UNPROJECTED} | {f for _, _, f, _ in BUDGET20}
    d, paths = load(folders)
    for f in folders:                                    # the naming trap, asserted
        if 'GaussianDiffusion' in f and 'Meuler' not in f:
            assert all('/plans/diffusion/' in p for p in paths[f]), f
        if 'Meuler' in f:
            assert all('/plans/diffusion/' not in p for p in paths[f]), f
    block_a(d)
    block_b(d)
    coverage(d)
