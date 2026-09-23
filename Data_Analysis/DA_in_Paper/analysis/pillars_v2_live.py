#!/usr/bin/env python3.14
"""UAV-pillars (pillars_v2, Gen15 U18): the four Chapter-6 avoiding cells evaluated LIVE with the quadrotor as the plant.

    python3.14 Data_Analysis/DA_in_Paper/analysis/pillars_v2_live.py

Corpus of record : analysis_results_checkpoint/23-09-UAV-Pillars-v2/live_p23uavpv2live  (jobs 26151-26154, tag
                   p23uavpv2live; 5 seeds x 3 geometries x 2 episodes per cell; MeanFM K1/K2 and CI-MeanFM K1/K2, U-Net;
                   variants diffuser + dpcc-t-tightened). npz = the avoiding eval's own output with the drone as env.
Table reference  : analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703
                   (the same four cells on the table, DPCC protocol) - the numbers of tab:avoiding-dpcc-protocol.
Sidecars         : the plant's per-episode records (tracking error, setpoint gap, contact steps, world path).

Aggregation = the table's: mean per geometry over seeds, then mean over the three geometries.
"S&C" = success AND constraint-free (the eval's n_success_and_constraints). "Declared constraint-free" is the eval's
collision_free_completed restricted to flights that ended successfully; the eval sets it to 0 for ANY failed flight,
so the physical-contact failures are separated here: a flight is a CONTACT if the drone's last position lies within
the radial rotor reach (0.36 m = 0.0100 avoiding units) of a pillar surface (the plant ends the flight on contact;
no flight hit the 200-step cap and none diverged - checked below).
"""
import csv
import glob
import gzip
import json
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

REPO = '/workspaces/FM-PCC'
sys.path.insert(0, REPO)
from uav_avoiding_bridge import frame as F                    # noqa: E402

LIVE = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/23-09-UAV-Pillars-v2/live_p23uavpv2live')
TABLE = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/'
                     'batch_avoiding_combined_20260919_132703/candidates_multidimensional_raw.csv.gz')
SCALE = 36.0
REACH_A = F.DRONE_REACH_M / SCALE
GEOS = ('top-right-hard', 'top-left-hard', 'both-hard')
CELLS = [('MeanFM', 'mf', 1), ('MeanFM', 'mf', 2), ('CI-MeanFM', 'af', 1), ('CI-MeanFM', 'af', 2)]
TABLE_FOLDERS = {('mf', 1): ('H8_K1_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE', 'bbunet'),
                 ('mf', 2): ('H8_K2_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE', 'bbunet'),
                 ('af', 1): ('H8_K1_Meuler_T0.5_A0.5_B4_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msgdpccproto', 'ae0.2'),
                 ('af', 2): ('H8_K2_Meuler_T0.5_A0.5_B4_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msgdpccproto', 'ae0.2')}
VARIANTS = ('diffuser', 'dpcc-t-tightened')
PILLARS = [(n, np.array([x, y]), r) for n, x, y, r in F.OBSTACLES_A]


def load_live():
    eps = []
    for f in sorted(glob.glob(os.path.join(LIVE, 'logs', '**', 'results', 'halfspace_*', '*.npz'), recursive=True)):
        eng = 'mf' if 'meanflow' in f else 'af'
        K = int(f.split('/H8_K')[-1].split('_')[0])
        seed = int(f.split('/results/')[0].split('/')[-1])
        geo = f.split('halfspace_')[1].split('/')[0]
        var = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        for i, o in enumerate(d['obs_all']):
            o = np.asarray(o, float)
            sp, dr = o[:, :2], o[:, 2:4]
            end = min(((np.linalg.norm(dr[-1] - c) - r), n) for n, c, r in PILLARS)
            eps.append(dict(eng=eng, K=K, seed=seed, geo=geo, var=var, ep=i,
                            succ=float(d['n_success'][i]), sc=float(d['n_success_and_constraints'][i]),
                            cf=float(d['collision_free_completed'][i]), viol=float(d['n_violations'][i]),
                            steps=float(d['n_steps'][i]), ms=1e3 * float(d['avg_time'][i]),
                            end_dist=end[0], end_pillar=end[1],
                            sp_clear=min((np.linalg.norm(sp - c, axis=1) - r).min() for _, c, r in PILLARS),
                            gap95=float(np.percentile(np.linalg.norm(sp - dr, axis=1), 95))))
    for e in eps:
        e['contact'] = int(e['succ'] == 0 and e['end_dist'] <= REACH_A + 0.002)
        e['cap'] = int(e['succ'] == 0 and e['steps'] >= 199)
    return eps


def load_table():
    D = defaultdict(lambda: defaultdict(dict))
    with gzip.open(TABLE, 'rt') as fh:
        for r in csv.DictReader(fh):
            for key, (fn, sub) in TABLE_FOLDERS.items():
                if r['Folder_Name'] == fn and sub in r['Full_Path'] and 'uav' not in r['Full_Path'] and r['variant'] in VARIANTS:
                    D[(key, r['variant'], r['halfspace_variant'])][r['seed']][r['metric']] = float(r['value'])
    out = {}
    for (key, var, geo), seeds in D.items():
        out[(key, var, geo)] = {m: st.mean(s[m] for s in seeds.values() if m in s)
                                for m in ('n_success', 'n_success_and_constraints', 'collision_free_completed',
                                          'n_violations', 'n_steps', 'avg_time')}
    return out


def load_sidecars():
    rec = []
    for f in sorted(glob.glob(os.path.join(LIVE, 'sidecars_p23uavpv2live', '*', 'uav_plant_records*.json'))):
        d = json.load(open(f))
        eng_k = f.split('/')[-2]
        for e in d['episodes']:
            s = e['summary']
            rec.append(dict(eng_k=eng_k, ended=s['ended'], te=s['track_err_mean_m'], te95=s['track_err_p95_m'],
                            gap=s['gap_a_p95'], contact_steps=s['contact_steps'], div=s['diverged'], v=s['speed_max_ms']))
    return rec


def geo_mean(eps, key):
    per = {}
    for g in GEOS:
        v = [e[key] for e in eps if e['geo'] == g]
        if v:
            per[g] = st.mean(v)
    return (st.mean(per.values()) if per else float('nan')), per


def main():
    eps, tab, side = load_live(), load_table(), load_sidecars()
    print(f'live episodes {len(eps)}  (expected 240)   sidecar episodes {len(side)}')
    print(f'ended on the 200-step cap: {sum(e["cap"] for e in eps)}   diverged (sidecar): {sum(1 for r in side if r["div"])}   '
          f'sidecar ended={dict((k, sum(1 for r in side if r["ended"] == k)) for k in set(r["ended"] for r in side))}')

    print('\n== TABLE 1 · the four cells, table (Panda) vs air (quadrotor), DPCC protocol; geometry mean of seed means')
    print(f"{'model':9s} {'nfe':>3s} {'variant':17s} | {'S&C tbl':>7s} {'S&C air':>7s} | {'succ air':>8s} {'contact':>7s} {'viol tbl':>8s} {'viol air':>8s} | {'steps tbl':>9s} {'steps air':>9s} | {'ms tbl':>6s} {'ms air':>6s}")
    for name, eng, K in CELLS:
        for var in VARIANTS:
            E = [e for e in eps if e['eng'] == eng and e['K'] == K and e['var'] == var]
            T = {g: tab.get(((eng, K), var, g)) for g in GEOS}
            tm = lambda m: st.mean(T[g][m] for g in GEOS if T[g])
            sc_air, _ = geo_mean(E, 'sc'); su_air, _ = geo_mean(E, 'succ'); ct, _ = geo_mean(E, 'contact')
            vi_air, _ = geo_mean(E, 'viol'); ms_air, _ = geo_mean(E, 'ms')
            steps_air = st.mean(e['steps'] for e in E if e['succ'] > 0)
            print(f"{name:9s} {K:3d} {var:17s} | {tm('n_success_and_constraints'):7.2f} {sc_air:7.2f} | {su_air:8.2f} {ct:7.2f} "
                  f"{tm('n_violations'):8.1f} {vi_air:8.1f} | {tm('n_steps'):9.1f} {steps_air:9.1f} | {1e3 * tm('avg_time'):6.1f} {ms_air:6.1f}")

    print('\n== TABLE 2 · per geometry, projected (dpcc-t-tightened): S&C table / air, and contact rate in the air')
    for name, eng, K in CELLS:
        E = [e for e in eps if e['eng'] == eng and e['K'] == K and e['var'] == 'dpcc-t-tightened']
        cells = []
        for g in GEOS:
            t = tab.get(((eng, K), 'dpcc-t-tightened', g)); a = [e for e in E if e['geo'] == g]
            cells.append(f"{g[:9]}: {t['n_success_and_constraints']:.2f} / {st.mean(e['sc'] for e in a):.2f} (contact {sum(e['contact'] for e in a)}/{len(a)})")
        print(f'{name:9s} {K}  ' + '   '.join(cells))

    print('\n== TABLE 3 · what the failures are (all 240 flights)')
    fail = [e for e in eps if e['succ'] == 0]
    print(f'failed flights {len(fail)}: contact {sum(e["contact"] for e in fail)}, cap {sum(e["cap"] for e in fail)}, other {len(fail) - sum(e["contact"] for e in fail) - sum(e["cap"] for e in fail)}')
    print(f'declared-constraint violations on projected flights that flew to the end: '
          f'{sum(e["viol"] for e in eps if e["var"] != "diffuser" and e["succ"] == 1):.0f} violating steps in {sum(1 for e in eps if e["var"] != "diffuser" and e["succ"] == 1)} flights')
    print(f'failed flights whose COMMANDED path itself passed within the reach ({REACH_A:.4f}) of a pillar: '
          f'{sum(1 for e in fail if e["sp_clear"] <= REACH_A)}/{len(fail)}')
    pill = defaultdict(int)
    for e in fail:
        pill[(e['var'], e['geo'], e['end_pillar'])] += 1
    for k in sorted(pill):
        print(f'   {k[0]:17s} {k[1]:15s} {k[2]:14s} {pill[k]}')
    ok = [e for e in eps if e['var'] != 'diffuser' and e['succ'] == 1]
    print(f'projected flights that survived: commanded-path clearance to the nearest pillar surface, avoiding units: '
          f'p05 {np.percentile([e["sp_clear"] for e in ok], 5):.4f}  median {np.median([e["sp_clear"] for e in ok]):.4f}  '
          f'(reach {REACH_A:.4f}; the table rod is {F.ROD_RADIUS_A:.3f})')
    print(f'failed projected flights: commanded-path clearance p50 {np.median([e["sp_clear"] for e in fail if e["var"] != "diffuser"]):.4f}  '
          f'max {max(e["sp_clear"] for e in fail if e["var"] != "diffuser"):.4f}')

    print('\n== TABLE 4 · the plant (sidecars): tracking and setpoint gap, all flights')
    print(f'track_err mean {st.mean(r["te"] for r in side):.3f} m  p95-of-p95 {np.percentile([r["te95"] for r in side], 95):.3f} m   '
          f'setpoint gap p95 (avoiding units) mean {st.mean(r["gap"] for r in side):.4f}  max {max(r["gap"] for r in side):.4f}   '
          f'|v| max {max(r["v"] for r in side):.2f} m/s   contact flights {sum(1 for r in side if r["contact_steps"] > 0)}')
    print(f'live gap p95 over all flights (npz): mean {st.mean(e["gap95"] for e in eps):.4f}; the table cells\' own gap was 0.02-0.04 (Panda), see U18 L1 note')


if __name__ == '__main__':
    main()
