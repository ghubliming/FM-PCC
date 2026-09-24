#!/usr/bin/env python3
"""D3IL-avoiding: the twenty-episode evaluations that are complete, against the protocol of DPCC.

    python3 logs_in_develop/Writing/Working_Space/ntrial20_appendix_20260924/ntrial20_da.py [RAW_CSV]

Written 2026-09-24 for the rebuilt twenty-episode appendix section (DA_20260924_ntrial20_feasible.md).

Input: the long-format batch the author named,
    temp/23-09/batch_avoiding_combined_20260923_110010(TenpK2D)/candidates_multidimensional_raw.csv
(local drop, not committed). The rows this script uses are written to ntrial20_cells.csv.gz next to it, and
the script re-runs from that file when the batch is absent, so every number stays reproducible.

What it does
  A. Census of every twenty-episode campaign in the batch (tag `_msg20trials`, plus the one-seed
     `_msgafon02_s6` cells): seeds x geometries per variant, episode count per cell (every success rate
     must be k/20), checkpoint identity from Full_Path, and whether the run finished all 13 variants.
     Verdict KEEP / KEEP-DA-ONLY / DROP with the reason.
  B. The kept cells beside their two-episode counterparts (the rows of tab:avoiding-dpcc-protocol and
     tab:avoiding-raw-models), in the chapter's convention: mean per geometry over the seeds, then over
     the geometries; spread = sample standard deviation over the per-seed geometry-means.
  C. Statistics of the two protocols against each other (DA only -- the thesis prints counts/intervals):
     Wilson 95 % intervals on pooled counts; seed-level t intervals; the exact distribution of what a
     two-episode evaluation returns when its episodes are drawn from the twenty-episode ones
     (hypergeometric per seed x geometry, convolved over the 15 cells); Fisher's exact test on the pooled
     counts; control steps and time paired by training seed; the rule ordering at both protocols.
  D. The DA Target (DPCC K20/aw10, dpcc-c-tightened) -- of record at two episodes, and a matched
     diagnostic at twenty episodes on the two geometries where the baseline has all five seeds.

Checkpoint identity (Full_Path, and the tree of 2026-09-23 23:21): MeanFM = the U-Net checkpoints under
H8_Dflow_matcher_v3_meanflow.models.MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5 (state_best.pt
2026-08-06..10; evaluated at two episodes 08-11, at twenty 08-13/14). FM = H8_Dmodels.diffusion.
FlowMatchingODE_a1.5_b1.0_aw10 (state_best.pt 2026-05-06/07; twenty episodes 08-18/19, two 09-17).
Folder_Name does not carry the backbone: the two-episode MeanFlow cells exist under bbunet AND bbmf_dit
with one Folder_Name, so cells are keyed on (Folder_Name, Full_Path substring). Stdlib only.
"""
import csv
import gzip
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(REPO, 'temp/23-09/batch_avoiding_combined_20260923_110010(TenpK2D)/'
                         'candidates_multidimensional_raw.csv')
CELLS_CSV = os.path.join(HERE, 'ntrial20_cells.csv.gz')
OUT_JSON = os.path.join(HERE, 'ntrial20_results.json')

MF = 'Dflow_matcher_v3_meanflow.models.MeanFlowODE'
AF = 'Dflow_matcher_v3_alphaflow.models.AlphaFlowODE'
FMO = 'Dmodels.diffusion.FlowMatchingODE'
MF_CK = 'MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5/'
FM_CK = 'FlowMatchingODE_a1.5_b1.0_aw10/'
CI_CK = 'AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.2'
AF_SIT = 'AlphaFlowODE_aw10_bbsit_tslogit_normal_ai1.0_ae0.0'
DIFF = '/plans/diffusion/'
GEOS = ('top-left-hard', 'top-right-hard', 'both-hard')
SEEDS = ('6', '7', '8', '9', '10')
RULES = ('dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened')
VARIANTS = ('diffuser',) + RULES
METRICS = ('n_success', 'n_success_and_constraints', 'n_violations', 'n_steps', 'avg_time',
           'n_steps_std', 'n_violations_std', 'avg_time_std')
RULE_NAME = {'diffuser': '-', 'dpcc-r-tightened': 'r', 'dpcc-c-tightened': 'c', 'dpcc-t-tightened': 't'}
T975_DF4 = 2.7764451051977987          # Student t, 97.5 %, 4 degrees of freedom (5 seeds)

# id: (model, nfe, episodes per seed and geometry, Folder_Name, Full_Path key)
CAMPAIGNS = {
    # --- twenty episodes per seed and geometry ---------------------------------------------------
    'MF1_20':  ('MeanFM', 1, 20, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', MF_CK),
    'MF2_20':  ('MeanFM', 2, 20, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', MF_CK),
    'MF5_20':  ('MeanFM', 5, 20, f'H8_K5_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', MF_CK),
    'MF10_20': ('MeanFM', 10, 20, f'H8_K10_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', MF_CK),
    'MF20_20': ('MeanFM', 20, 20, f'H8_K20_Meuler_T0.5_A0.5_B1_{MF}_msg20trials', MF_CK),
    'FM1_20':  ('FM', 1, 20, f'H8_K1_Meuler_T0.5_{FMO}_msg20trials', FM_CK),
    'FM2_20':  ('FM', 2, 20, f'H8_K2_Meuler_T0.5_{FMO}_msg20trials', FM_CK),
    'FM5_20':  ('FM', 5, 20, f'H8_K5_Meuler_T0.5_{FMO}_msg20trials', FM_CK),
    'FM20_20': ('FM', 20, 20, f'H8_K20_Meuler_T0.5_{FMO}_msg20trials', FM_CK),
    'AF1_20':  ('alpha-Flow SiT a_end=0', 1, 20, f'H8_K1_Meuler_T0.5_A0.5_B1_{AF}_msg20trials', AF_SIT),
    'AF2_20':  ('alpha-Flow SiT a_end=0', 2, 20, f'H8_K2_Meuler_T0.5_A0.5_B1_{AF}_msg20trials', AF_SIT),
    'CI1_20s6':  ('CI-MeanFM', 1, 20, f'H8_K1_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', CI_CK),
    'CI2_20s6':  ('CI-MeanFM', 2, 20, f'H8_K2_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', CI_CK),
    'CI5_20s6':  ('CI-MeanFM', 5, 20, f'H8_K5_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', CI_CK),
    'CI10_20s6': ('CI-MeanFM', 10, 20, f'H8_K10_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', CI_CK),
    'CI20_20s6': ('CI-MeanFM', 20, 20, f'H8_K20_Meuler_T0.5_A0.5_B4_{AF}_msgafon02_s6', CI_CK),
    'D20_20':  ('Diffusion', 20, 20, 'H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials', DIFF),
    # --- the protocol of DPCC: two episodes per seed and geometry (the chapter's rows) ------------
    'MF1_2':   ('MeanFM', 1, 2, f'H8_K1_Meuler_T0.5_A0.5_B1_{MF}', MF_CK),
    'MF2_2':   ('MeanFM', 2, 2, f'H8_K2_Meuler_T0.5_A0.5_B1_{MF}', MF_CK),
    'MF5_2':   ('MeanFM', 5, 2, f'H8_K5_Meuler_T0.5_A0.5_B1_{MF}', MF_CK),
    'MF10_2':  ('MeanFM', 10, 2, f'H8_K10_Meuler_T0.5_A0.5_B1_{MF}', MF_CK),
    'FM1_2':   ('FM', 1, 2, f'H8_K1_Meuler_T0.5_{FMO}_msgdpccproto', FM_CK),
    'FM2_2':   ('FM', 2, 2, f'H8_K2_Meuler_T0.5_{FMO}_msgdpccproto', FM_CK),
    'CI1_2':   ('CI-MeanFM', 1, 2, f'H8_K1_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', CI_CK),
    'CI2_2':   ('CI-MeanFM', 2, 2, f'H8_K2_Meuler_T0.5_A0.5_B4_{AF}_msgdpccproto', CI_CK),
    'D20_2':   ('Diffusion', 20, 2, 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5', DIFF),
}
# the pairs the section prints: (twenty-episode id, two-episode id)
SECTION_PAIRS = [('MF1_20', 'MF1_2'), ('MF2_20', 'MF2_2'), ('FM1_20', 'FM1_2'), ('FM2_20', 'FM2_2')]
DA_ONLY_PAIRS = [('MF5_20', 'MF5_2'), ('MF10_20', 'MF10_2')]
DA_ONLY_SOLO = ['FM5_20']                  # complete, but no two-episode counterpart on five seeds
TARGET = ('D20_2', 'dpcc-c-tightened')
SCOPE = {   # why a complete, correct campaign is or is not in the thesis section
    'MF1_20': 'section', 'MF2_20': 'section', 'FM1_20': 'section', 'FM2_20': 'section',
    'MF5_20': 'DA only: its two-episode counterpart is an appendix row (tab:app:avoiding-dpcc-full), not a chapter row',
    'FM5_20': 'DA only: no two-episode counterpart on five seeds (seed 6 only)',
    'MF10_20': 'DA only',
}


# ------------------------------------------------------------------------------------------------ load
def load():
    """-> cells {(cid, variant, geo, seed): {metric: value}}, variants_seen {(cid, geo, seed): set}."""
    by_folder = defaultdict(list)
    for cid, (_, _, _, folder, key) in CAMPAIGNS.items():
        by_folder[folder].append((cid, key))
    cells = defaultdict(dict)
    seen = defaultdict(set)
    paths = defaultdict(set)
    src = RAW if os.path.exists(RAW) else CELLS_CSV
    rows_out = []
    opener = gzip.open if src.endswith('.gz') else open
    with opener(src, 'rt', newline='') as fh:
        for r in csv.DictReader(fh):
            hits = [cid for cid, key in by_folder.get(r['Folder_Name'], ()) if key in r['Full_Path']]
            if not hits:
                continue
            cid = hits[0]
            seen[(cid, r['halfspace_variant'], r['seed'])].add(r['variant'])
            paths[cid].add(r['Full_Path'])
            if r['variant'] in VARIANTS and r['metric'] in METRICS:
                try:
                    cells[(cid, r['variant'], r['halfspace_variant'], r['seed'])][r['metric']] = float(r['value'])
                except ValueError:
                    continue
                if src == RAW:
                    rows_out.append({k: r[k] for k in ('Candidate', 'Folder_Name', 'Full_Path', 'seed', 'variant',
                                                        'constraint_type', 'halfspace_variant', 'metric', 'value')})
            elif src == RAW and r['metric'] == 'n_success':
                # keep the variant census reproducible from the extract as well
                rows_out.append({k: r[k] for k in ('Candidate', 'Folder_Name', 'Full_Path', 'seed', 'variant',
                                                    'constraint_type', 'halfspace_variant', 'metric', 'value')})
    if src == RAW:
        with gzip.open(CELLS_CSV, 'wt', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
    return cells, seen, paths, src


# ------------------------------------------------------------------------------------------ statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def hyper_pmf(N, K, n):
    """distribution of successes in n draws without replacement from N holding K successes"""
    tot = math.comb(N, n)
    return [math.comb(K, x) * math.comb(N - K, n - x) / tot for x in range(n + 1)]


def convolve(a, b):
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def fisher_two_sided(a, n1, b, n2):
    """Fisher's exact test on [[a, n1-a], [b, n2-b]]; two-sided = sum of tables no more likely."""
    K = a + b
    N = n1 + n2
    lo, hi = max(0, K - n2), min(K, n1)
    def p(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    p_obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-9)))


def quantile_range(pmf, lo_q=0.025, hi_q=0.975):
    c, lo, hi = 0.0, None, None
    for x, px in enumerate(pmf):
        c += px
        if lo is None and c >= lo_q - 1e-12:
            lo = x
        if hi is None and c >= hi_q - 1e-12:
            hi = x
    return lo, hi


# --------------------------------------------------------------------------------------------- helpers
def cell_rows(cells, cid, variant, keep=None):
    out = {}
    for s in SEEDS:
        for g in GEOS:
            m = cells.get((cid, variant, g, s))
            if m and all(k in m for k in METRICS[:5]) and (keep is None or (g, s) in keep):
                out[(g, s)] = m
    return out


def agg(rows, episodes):
    """chapter convention + counts"""
    if not rows:
        return None
    geos = [g for g in GEOS if any(gg == g for gg, _ in rows)]
    seeds = [s for s in SEEDS if any(ss == s for _, ss in rows)]
    a = {'n_cells': len(rows), 'seeds': seeds, 'geos': geos, 'episodes_per_cell': episodes}
    for k in ('n_success', 'n_success_and_constraints', 'n_violations', 'n_steps', 'avg_time'):
        per_geo = [st.mean(m[k] for (g, s), m in rows.items() if g == gg) for gg in geos]
        per_seed = [st.mean(m[k] for (g, s), m in rows.items() if s == ss) for ss in seeds]
        a[k] = st.mean(per_geo)
        a[k + '_sd'] = st.stdev(per_seed) if len(per_seed) > 1 else 0.0
        a[k + '_per_seed'] = dict(zip(seeds, per_seed))
    for k in ('n_success', 'n_success_and_constraints'):
        a[k + '_count'] = int(round(sum(m[k] * episodes for m in rows.values())))
        a[k + '_per_cell_count'] = {f'{g}|{s}': int(round(m[k] * episodes)) for (g, s), m in rows.items()}
    a['episodes'] = episodes * len(rows)
    return a


def two_episode_draws(rows20, key='n_success_and_constraints', N=20, n=2):
    """exact distribution of the count a 2-episode protocol returns, episodes drawn from the 20"""
    pmf = [1.0]
    for m in rows20.values():
        K = int(round(m[key] * N))
        pmf = convolve(pmf, hyper_pmf(N, K, n))
    return pmf


def steps_se_two_episode(rows20, N=20, n=2):
    """sampling s.e. of the chapter's step mean under a 2-episode protocol, from the per-cell stds"""
    var = sum((m['n_steps_std'] ** 2) * (N - n) / (n * (N - 1)) for m in rows20.values())
    return math.sqrt(var) / len(rows20)


def rank_rules(aggs):
    """rules ordered by S&C (desc), then control steps (asc)"""
    return sorted(aggs, key=lambda r: (-round(aggs[r]['n_success_and_constraints'], 6), aggs[r]['n_steps']))


# ------------------------------------------------------------------------------------------------ main
def census(cells, seen, paths):
    print('=' * 110)
    print('A. CENSUS -- every twenty-episode campaign, and the two-episode rows it is compared with')
    print('=' * 110)
    verdicts = {}
    for cid, (model, K, ep, folder, key) in CAMPAIGNS.items():
        cov = {}
        grain_ok = True
        grain_20 = False
        for v in VARIANTS:
            rows = cell_rows(cells, cid, v)
            cov[v] = len(rows)
            for m in rows.values():
                for k in ('n_success', 'n_success_and_constraints'):
                    x = m[k] * ep
                    if abs(x - round(x)) > 1e-6:
                        grain_ok = False
                    if ep == 20 and abs(m[k] * 2 - round(m[k] * 2)) > 1e-6:
                        grain_20 = True
        nvar = [len(seen.get((cid, g, s), ())) for s in SEEDS for g in GEOS]
        nvar_present = [x for x in nvar if x]
        full_run = bool(nvar_present) and len(nvar_present) == 15 and len(set(nvar_present)) == 1
        missing = {v: 15 - c for v, c in cov.items() if c < 15}
        print(f'\n[{cid}] {model} K={K} episodes={ep}  {folder}')
        for p in sorted(paths.get(cid, ())):
            print(f'      path: {p.split("/plans/")[-1]}')
        print('      cells (of 15) per variant: ' + '  '.join(f'{RULE_NAME[v]}:{cov[v]}' for v in VARIANTS)
              + f'   variants per seed x geometry: {sorted(set(nvar))}   episodes per cell = {ep}: '
              + ('yes' if grain_ok else 'NO') + (' (values off the 1/2 grid present)' if grain_20 else ''))
        if not any(cov.values()):
            verdicts[cid] = ('DROP', 'no rows in the batch')
        elif missing:
            detail = []
            for v in VARIANTS:
                rows = cell_rows(cells, cid, v)
                for g in GEOS:
                    ss = [s for s in SEEDS if (g, s) in rows]
                    if len(ss) < 5:
                        detail.append(f'{RULE_NAME[v]}/{g}:{",".join(ss) or "none"}')
            if len({s for (g, s) in cell_rows(cells, cid, 'diffuser')} | {s for v in RULES for (g, s) in cell_rows(cells, cid, v)}) == 1:
                verdicts[cid] = ('DROP', 'one training seed')
            else:
                verdicts[cid] = ('DROP', 'incomplete: ' + '; '.join(detail))
        elif not grain_ok:
            verdicts[cid] = ('DROP', f'success rates are not multiples of 1/{ep}')
        elif key == AF_SIT:
            verdicts[cid] = ('DROP', 'SiT backbone with alpha annealed to 0: not the U-Net CI-MeanFM (alpha_end 0.2)')
        elif not full_run:
            verdicts[cid] = ('KEEP-DA-ONLY', 'reported cells complete, but the run did not finish every variant '
                             f'(variants per seed x geometry {sorted(set(nvar))})')
        else:
            verdicts[cid] = ('KEEP', 'complete')
        if ep == 20 and verdicts[cid][0].startswith('KEEP') and SCOPE.get(cid, 'section') != 'section':
            verdicts[cid] = ('KEEP-DA-ONLY', verdicts[cid][1] + '; ' + SCOPE[cid])
        print(f'      -> {verdicts[cid][0]}: {verdicts[cid][1]}')
    return verdicts


def fmt_pm(x, sd, dp, scale=1.0):
    return f'{x * scale:.{dp}f}+-{sd * scale:.{dp}f}'


def tables(cells):
    print('\n' + '=' * 110)
    print('B. THE KEPT CELLS BESIDE THE PROTOCOL OF DPCC (chapter convention; spread = sd over the five seeds)')
    print('=' * 110)
    res = {}
    hdr = (f'{"model":7}{"K":>3} {"rule":4}| {"S&C 20ep":>14} {"count":>8} | {"S&C 2ep":>14} {"count":>6} |'
           f' {"steps 20":>11} {"steps 2":>11} | {"ms 20":>12} {"ms 2":>12}')
    for label, pairs in (('SECTION', SECTION_PAIRS), ('DA ONLY', DA_ONLY_PAIRS)):
        print(f'\n-- {label} --')
        print(hdr)
        print('-' * len(hdr))
        for c20, c2 in pairs:
            model, K = CAMPAIGNS[c20][:2]
            for v in VARIANTS:
                a20 = agg(cell_rows(cells, c20, v), 20)
                a2 = agg(cell_rows(cells, c2, v), 2)
                res[(c20, v)] = a20
                res[(c2, v)] = a2
                print(f'{model:7}{K:>3} {RULE_NAME[v]:4}| '
                      f'{fmt_pm(a20["n_success_and_constraints"], a20["n_success_and_constraints_sd"], 3):>14} '
                      f'{a20["n_success_and_constraints_count"]:>4}/{a20["episodes"]:<3} | '
                      f'{fmt_pm(a2["n_success_and_constraints"], a2["n_success_and_constraints_sd"], 3):>14} '
                      f'{a2["n_success_and_constraints_count"]:>3}/{a2["episodes"]:<2} | '
                      f'{fmt_pm(a20["n_steps"], a20["n_steps_sd"], 1):>11} {fmt_pm(a2["n_steps"], a2["n_steps_sd"], 1):>11} | '
                      f'{fmt_pm(a20["avg_time"], a20["avg_time_sd"], 1, 1000):>12} {fmt_pm(a2["avg_time"], a2["avg_time_sd"], 1, 1000):>12}')
            print()
    print('-- DA ONLY, no two-episode counterpart on five seeds --')
    for cid in DA_ONLY_SOLO:
        model, K = CAMPAIGNS[cid][:2]
        for v in VARIANTS:
            a = agg(cell_rows(cells, cid, v), 20)
            res[(cid, v)] = a
            print(f'{model:7}{K:>3} {RULE_NAME[v]:4}| '
                  f'{fmt_pm(a["n_success_and_constraints"], a["n_success_and_constraints_sd"], 3):>14} '
                  f'{a["n_success_and_constraints_count"]:>4}/{a["episodes"]:<3} | {"":>21} | '
                  f'{fmt_pm(a["n_steps"], a["n_steps_sd"], 1):>11} {"":>11} | {fmt_pm(a["avg_time"], a["avg_time_sd"], 1, 1000):>12}')
        print()
    print('per geometry, S&C count (20 ep: of 100 | 2 ep: of 10) and control steps:')
    for c20, c2 in SECTION_PAIRS + DA_ONLY_PAIRS:
        for v in VARIANTS:
            r20, r2 = cell_rows(cells, c20, v), cell_rows(cells, c2, v)
            parts = []
            for g in GEOS:
                k20 = sum(int(round(m['n_success_and_constraints'] * 20)) for (gg, s), m in r20.items() if gg == g)
                k2 = sum(int(round(m['n_success_and_constraints'] * 2)) for (gg, s), m in r2.items() if gg == g)
                st20 = st.mean(m['n_steps'] for (gg, s), m in r20.items() if gg == g)
                st2 = st.mean(m['n_steps'] for (gg, s), m in r2.items() if gg == g)
                parts.append(f'{g[:9]:9} {k20:>3}/100 {k2:>2}/10 {st20:5.1f}/{st2:5.1f}')
            print(f'  {c20:8} {RULE_NAME[v]}  ' + '   '.join(parts))
    print()
    print('unprojected arm (rule "-"): success (goal reached) and violating steps')
    for c20, c2 in SECTION_PAIRS + DA_ONLY_PAIRS:
        a20, a2 = res[(c20, 'diffuser')], res[(c2, 'diffuser')]
        print(f'  {c20:8} success {a20["n_success_count"]}/{a20["episodes"]}  vs {a2["n_success_count"]}/{a2["episodes"]};'
              f'  S&C {a20["n_success_and_constraints_count"]}/{a20["episodes"]} vs {a2["n_success_and_constraints_count"]}/{a2["episodes"]};'
              f'  viol.steps {fmt_pm(a20["n_violations"], a20["n_violations_sd"], 1)} vs {fmt_pm(a2["n_violations"], a2["n_violations_sd"], 1)}')
    return res


def stats(cells, res):
    print('\n' + '=' * 110)
    print('C. STATISTICS -- the two protocols against each other')
    print('=' * 110)
    out = {}
    for c20, c2 in SECTION_PAIRS + DA_ONLY_PAIRS:
        model, K = CAMPAIGNS[c20][:2]
        for v in VARIANTS:
            a20, a2 = res[(c20, v)], res[(c2, v)]
            rows20 = cell_rows(cells, c20, v)
            k20, n20 = a20['n_success_and_constraints_count'], a20['episodes']
            k2, n2 = a2['n_success_and_constraints_count'], a2['episodes']
            w20, w2 = wilson(k20, n20), wilson(k2, n2)
            pmf = two_episode_draws(rows20)
            lo, hi = quantile_range(pmf)
            p_le = sum(pmf[:k2 + 1])
            p_ge = sum(pmf[k2:])
            p_all = pmf[-1]
            mode = max(range(len(pmf)), key=lambda x: pmf[x])
            fisher = fisher_two_sided(k2, n2, k20, n20)
            # seed-paired differences (two-episode minus twenty-episode)
            d_steps = [a2['n_steps_per_seed'][s] - a20['n_steps_per_seed'][s] for s in SEEDS]
            d_sc = [a2['n_success_and_constraints_per_seed'][s] - a20['n_success_and_constraints_per_seed'][s] for s in SEEDS]
            r_ms = [a2['avg_time_per_seed'][s] / a20['avg_time_per_seed'][s] for s in SEEDS]
            md, sd = st.mean(d_steps), st.stdev(d_steps)
            se2 = steps_se_two_episode(rows20)
            z_steps = (a2['n_steps'] - a20['n_steps']) / se2 if se2 > 0 else float('nan')
            sc_seed_ci = (a20['n_success_and_constraints'] - T975_DF4 * a20['n_success_and_constraints_sd'] / math.sqrt(5),
                          a20['n_success_and_constraints'] + T975_DF4 * a20['n_success_and_constraints_sd'] / math.sqrt(5))
            steps_seed_ci20 = (a20['n_steps'] - T975_DF4 * a20['n_steps_sd'] / math.sqrt(5),
                               a20['n_steps'] + T975_DF4 * a20['n_steps_sd'] / math.sqrt(5))
            inside = lo <= k2 <= hi
            rec = dict(model=model, K=K, rule=RULE_NAME[v], k20=k20, n20=n20, k2=k2, n2=n2,
                       wilson20=w20, wilson2=w2, draw_range=(lo, hi), draw_mode=mode, draw_mean=sum(i * p for i, p in enumerate(pmf)),
                       p_all30=p_all, p_le=p_le, p_ge=p_ge, inside=inside, fisher_p=fisher,
                       sc_seed_ci20=sc_seed_ci, d_sc_seed=d_sc,
                       steps20=a20['n_steps'], steps2=a2['n_steps'], steps_se_2ep=se2, z_steps=z_steps,
                       steps_seed_ci20=steps_seed_ci20,
                       d_steps_seed=d_steps, d_steps_mean=md, d_steps_sd=sd,
                       d_steps_tci=(md - T975_DF4 * sd / math.sqrt(5), md + T975_DF4 * sd / math.sqrt(5)),
                       ms20=a20['avg_time'] * 1000, ms2=a2['avg_time'] * 1000, ms_ratio_seed=r_ms)
            out[f'{c20}|{v}'] = rec
            print(f'{model:7}{K:>3} {RULE_NAME[v]}: S&C {k20}/{n20} [{w20[0]:.3f},{w20[1]:.3f}]  vs  {k2}/{n2} [{w2[0]:.3f},{w2[1]:.3f}]'
                  f' | 2-ep draws from the 20: 95% range {lo}-{hi}/30, mode {mode}, P(30/30)={p_all:.3f}, '
                  f'P(<= {k2})={p_le:.3f}, P(>= {k2})={p_ge:.3f} {"IN" if inside else "OUT"} | Fisher p={fisher:.3f}')
            print(f'{"":13}steps {a20["n_steps"]:.1f} vs {a2["n_steps"]:.1f}: diff {a2["n_steps"] - a20["n_steps"]:+.1f}, '
                  f'2-ep s.e. {se2:.2f} (z {z_steps:+.2f}); seed-paired diff {md:+.1f} [{rec["d_steps_tci"][0]:+.1f},{rec["d_steps_tci"][1]:+.1f}], '
                  f'{sum(1 for d in d_steps if d < 0)}/5 seeds shorter at 2 ep | ms {a20["avg_time"] * 1000:.1f} vs {a2["avg_time"] * 1000:.1f} '
                  f'(ratio per seed {min(r_ms):.2f}-{max(r_ms):.2f})')
        print()
    print('unprojected arm, goal reached (n_success):')
    for c20, c2 in SECTION_PAIRS + DA_ONLY_PAIRS:
        a20, a2 = res[(c20, 'diffuser')], res[(c2, 'diffuser')]
        pmf = two_episode_draws(cell_rows(cells, c20, 'diffuser'), key='n_success')
        lo, hi = quantile_range(pmf)
        k2 = a2['n_success_count']
        out[f'{c20}|diffuser|n_success'] = dict(k20=a20['n_success_count'], n20=a20['episodes'], k2=k2, n2=a2['episodes'],
                                              draw_range=(lo, hi), p_all30=pmf[-1], inside=lo <= k2 <= hi,
                                              wilson20=wilson(a20['n_success_count'], a20['episodes']))
        print(f'  {c20:8} {a20["n_success_count"]}/{a20["episodes"]} vs {k2}/{a2["episodes"]}: 2-ep range {lo}-{hi}, '
              f'P(30/30)={pmf[-1]:.3f} {"IN" if lo <= k2 <= hi else "OUT"}')
    print()
    # rule ordering at both protocols
    print('rule ordering (S&C desc, then steps asc):')
    order = {}
    for c20, c2 in SECTION_PAIRS + DA_ONLY_PAIRS:
        o20 = rank_rules({v: res[(c20, v)] for v in RULES})
        o2 = rank_rules({v: res[(c2, v)] for v in RULES})
        order[c20] = ([RULE_NAME[v] for v in o20], [RULE_NAME[v] for v in o2])
        print(f'  {c20:8} 20 ep: {" > ".join(order[c20][0])}    2 ep: {" > ".join(order[c20][1])}'
              f'    best rule {"SAME" if o20[0] == o2[0] else "DIFFERS"}')
    # the stall under cumulative projection cost (sec:res:ablations)
    print('\ncumulative projection cost against temporal consistency (steps c - t, S&C t - c):')
    for c20, c2 in SECTION_PAIRS + DA_ONLY_PAIRS:
        for cid, ep in ((c20, 20), (c2, 2)):
            c, t = res[(cid, 'dpcc-c-tightened')], res[(cid, 'dpcc-t-tightened')]
            print(f'  {cid:8} steps c {c["n_steps"]:.1f} t {t["n_steps"]:.1f} (c-t {c["n_steps"] - t["n_steps"]:+.1f});'
                  f' S&C c {c["n_success_and_constraints_count"]}/{c["episodes"]} t {t["n_success_and_constraints_count"]}/{t["episodes"]}')
    # counts over all section cells
    sec = [k for k in out if k.split('|')[0] in dict(SECTION_PAIRS) and not k.endswith('|n_success')]
    n_in = sum(1 for k in sec if out[k]['inside'])
    n_tot = len(sec)
    print(f'\nsection cells whose 2-ep S&C count lies inside the 95% range of 2-ep draws from the 20-ep evaluation: {n_in}/{n_tot}')
    return out, order


def target(cells, res):
    print('\n' + '=' * 110)
    print('D. DA TARGET -- DPCC K20/aw10 dpcc-c-tightened')
    print('=' * 110)
    t2 = agg(cell_rows(cells, *TARGET), 2)
    print(f'of record (2 ep, 5x3): S&C {t2["n_success_and_constraints_count"]}/{t2["episodes"]} '
          f'({t2["n_success_and_constraints"]:.3f}), steps {t2["n_steps"]:.1f}, ms {t2["avg_time"] * 1000:.1f}')
    keep = {(g, s) for g in ('top-left-hard', 'top-right-hard') for s in SEEDS}
    out = {'record': t2, 'matched': {}}
    print('matched diagnostic at 20 episodes on top-left-hard + top-right-hard, five seeds, 200 episodes per row:')
    for cid, v in (('D20_20', 'dpcc-r-tightened'), ('D20_20', 'dpcc-c-tightened'), ('D20_20', 'dpcc-t-tightened'),
                   ('MF1_20', 'dpcc-t-tightened'), ('MF2_20', 'dpcc-t-tightened'), ('MF1_20', 'dpcc-c-tightened'),
                   ('FM1_20', 'dpcc-c-tightened'), ('FM2_20', 'dpcc-c-tightened'), ('FM1_20', 'dpcc-t-tightened'),
                   ('FM2_20', 'dpcc-t-tightened')):
        rows = cell_rows(cells, cid, v, keep)
        a = agg(rows, 20)
        per_geo = {g: (sum(int(round(m['n_success_and_constraints'] * 20)) for (gg, s), m in rows.items() if gg == g),
                       st.mean(m['n_steps'] for (gg, s), m in rows.items() if gg == g),
                       1000 * st.mean(m['avg_time'] for (gg, s), m in rows.items() if gg == g))
                   for g in ('top-left-hard', 'top-right-hard')}
        out['matched'][f'{cid}|{v}'] = {'agg': a, 'per_geo': per_geo}
        print(f'  {cid:7} {RULE_NAME[v]}  S&C {a["n_success_and_constraints_count"]}/{a["episodes"]} ({a["n_success_and_constraints"]:.3f}+-{a["n_success_and_constraints_sd"]:.3f})'
              f'  steps {a["n_steps"]:.1f}+-{a["n_steps_sd"]:.1f}  ms {a["avg_time"] * 1000:.1f}+-{a["avg_time_sd"] * 1000:.1f}'
              f'   TL {per_geo["top-left-hard"][0]}/100 {per_geo["top-left-hard"][1]:.1f} st   TR {per_geo["top-right-hard"][0]}/100 {per_geo["top-right-hard"][1]:.1f} st')
    return out


def dropped(cells, verdicts):
    print('\n' + '=' * 110)
    print('E. DROPPED CAMPAIGNS over the cells they cover -- for the record, NOT for the thesis')
    print('=' * 110)
    for cid, (model, K, ep, folder, key) in CAMPAIGNS.items():
        if verdicts.get(cid, ('',))[0] != 'DROP':
            continue
        for v in VARIANTS:
            rows = cell_rows(cells, cid, v)
            if not rows:
                continue
            a = agg(rows, ep)
            print(f'  {cid:10} {RULE_NAME[v]}  S&C {a["n_success_and_constraints_count"]}/{a["episodes"]} ({a["n_success_and_constraints"]:.3f})'
                  f'  steps {a["n_steps"]:.1f}  ms {a["avg_time"] * 1000:.1f}  cells {a["n_cells"]}/15  seeds {",".join(a["seeds"])}')
        print()


def main():
    global RAW
    if len(sys.argv) > 1:
        RAW = sys.argv[1]
    cells, seen, paths, src = load()
    print(__doc__)
    print(f'source: {src}\n')
    verdicts = census(cells, seen, paths)
    res = tables(cells)
    stat, order = stats(cells, res)
    tgt = target(cells, res)
    dropped(cells, verdicts)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, float):
            return round(o, 6)
        return o
    with open(OUT_JSON, 'w') as fh:
        json.dump(clean({'source': src, 'verdicts': verdicts,
                         'cells': {f'{k[0]}|{k[1]}': v for k, v in res.items()},
                         'stats': stat, 'rule_order': order, 'target': tgt}), fh, indent=1)
    print(f'\nwrote {os.path.relpath(OUT_JSON, REPO)} and {os.path.relpath(CELLS_CSV, REPO)}')


if __name__ == '__main__':
    main()
