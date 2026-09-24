#!/usr/bin/env python3.14
"""UAV-corridor v3 (Gen15 U19 / R33): every number Chapter 6 §6.3 (UAV-corridor) asks for, from the raw result folders.

    python3.14 Data_Analysis/DA_in_Paper/analysis/corridor_v3_grid.py                  # tables + checks (markdown)
    CORRIDOR_V3_CORPUS=<plans root> python3.14 …/corridor_v3_grid.py --json out.json  # + per-cell JSON

Corpus    : the `plans/` tree of logs/UAV_MIX/uav-corridor on the cluster; default = the 23-09 download
            temp/23-09-Corridor-TEMP/plans (preliminary — see DA_20260923_corridor_v3_preliminary.md §0 for what
            the download lacks). Cells are selected by run tag: p23cv3t (tilt, geometry corridor_v3_tilt) and
            p23cv3ah (hump, geometry corridor_v3_ablation_hump). Nothing else is read (u17cv2 / u7hg / u19smoke*
            live in the same trees and are ignored).
Job logs  : <corpus>/../2026-09-2*/ *uav_mix_eval_*.log — used ONLY to tell "finished on the cluster, not in this
            download" from "still running"; no number is taken from a log.
Protocol  : one training seed (6); the thesis reads TEN flights per cell = trial index 0-9 (routes L,C,R cycling,
            4/3/3). C1-C4 were flown with 12 (the extra two are dropped here), C5 with 10 (runbook §5). Trial i is
            seeded by its index, so the first ten of a 12-flight cell are the flights a 10-flight cell flies.
Metrics   : per flight, the eval's own fields (npz; timing from results.json):
              success          = success_relaxed                 (crossed the finish line)
              violation-free   = constraint_collision_free       (no violating control step, r_drone 0.31 m)
              S&C              = success_relaxed_and_constraints (both)
              violating steps  = constraint_n_violations
              ms/step          = results.json rollouts[i].timing.total_ms_mean (network + projection per action)
            Counts are x/10; means ± sample standard deviation over the ten flights.
Best rule : per (model, budget, projector) the rule with the most S&C; ties → fewer mean violating steps → lower
            ms/step. Stated in the DA so the writing agent does not re-decide it.
Frontier  : as the chapter defines it — S&C (count of 10) against ms/step, a configuration with S&C 0 is not
            eligible, costs within 1 % are one cost; one point per (model, budget, projector) at its best rule.
Checks    : (a) the geometry each run saved (config_snapshot_uav_mix/uav_projection.yaml) equals the repo's; (b) the
            violating-step count recomputed from the flown path with the eval's own scorer equals the stored count;
            (c) divergence aborts / projection circuit-breaker trips; (d) where the violating steps are and whether the
            commanded setpoint itself was clean there; (e) how often the plan sits on the normaliser's box
            (LimitsNormalizer.unnormalize clips plans to the demonstrated range).
Stdlib + numpy + yaml. Author: Claude (Opus 5.5, Claude Code, U19 corridor chat), 2026-09-23.
"""
import argparse
import ast
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import yaml

REPO = '/workspaces/FM-PCC'
CORPUS = os.environ.get('CORRIDOR_V3_CORPUS', os.path.join(REPO, 'temp/23-09-Corridor-TEMP/plans'))
BATCH = os.environ.get('CORRIDOR_V3_BATCH', os.path.join(
    REPO, 'temp/23-09-FULL/24-09-1000/batch_uav_20260924_081422/per_rollout_detail.csv'))   # DA_UAV_v1, full corpus
N_READ = 10
GEOS = {'tilt': ('p23cv3t', 'corridor_v3_tilt'), 'hump': ('p23cv3ah', 'corridor_v3_ablation_hump')}
MODELS = [('MeanFM', 'mf', (1, 2, 3)), ('CI-MeanFM a0.2', 'af', (1, 2, 3)), ('FM', 'fm', (1, 2, 3, 5, 20)),
          ('Diffusion', 'diffusion', (20,))]
TRAIN_FILTER = {'mf': 'dp0.5_bbunet', 'af': 'ae0.2_bbunet', 'fm': 'FlowMatchingODE_9D', 'diffusion': 'GaussianDiffusion_9D_K20'}
STACK = '-bounds_free-pdes-tightened'
PER_STEP = {'r': 'dpcc-r' + STACK, 'c': 'dpcc-c' + STACK, 't': 'dpcc-t' + STACK}
ENDPOINT = {'single': 'hardflow_sls' + STACK, 'r': 'hardflow_sls-r' + STACK, 'c': 'hardflow_sls-c' + STACK,
            't': 'hardflow_sls-t' + STACK}
UNPROJ = 'diffuser'
G3_WINDOW = {'tilt': ((0.5, 2.0), 'min'), 'hump': ((-0.5, 0.5), 'max')}   # where each constraint asks for altitude
R_DRONE = 0.31


# ── the eval's own scorer, extracted from the module (torch-free) ─────────────────────────────────────────────────
def _scorer():
    src = open(os.path.join(REPO, 'mix_uav_test/eval_mix_uav.py')).read()
    ns = {'np': np}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                '_normalize_halfspace', '_hs_plane', '_hs_lean', '_hs_normal3', '_exec_constraint_violations'):
            exec(compile(ast.Module([node], []), 'eval_mix_uav', 'exec'), ns)
    return ns


NS = _scorer()
REPO_YAML = yaml.safe_load(open(os.path.join(REPO, 'config/uav_projection.yaml')))
GEO_CFG = {g['name']: g for g in REPO_YAML['geo_constraint_variants']}


def geo_config(geo):
    g = dict(GEO_CFG[GEOS[geo][1]])
    g['inflation'] = REPO_YAML['inflation']          # the scorer's body radius (0.31); planning_inflation is the projector's
    return g


def per_step_violation(P, cfg):
    """bool per point: the scorer's rule at one position (used for the setpoint check)."""
    return np.array([NS['_exec_constraint_violations']([[0, 0, 0, *p, 0, 0, 0]], cfg)[1] > 0 for p in P])


# ── discovery ─────────────────────────────────────────────────────────────────────────────────────────────────────
def cell_dir(eng, K, tag, variant):
    pat = f'{CORPUS}/mix_uav_{eng}/*/E{eng}_K{K}_mpc4_pid_stopgo_T0.5*_{tag}/6/corridor_cv3*/{variant}'
    ds = [d for d in glob.glob(pat) if TRAIN_FILTER[eng] in d]
    return ds[0] if ds else None


def job_status():
    """{(eval_folder, variant_as_written): 'done'|'running'} from the child logs beside the corpus."""
    out = {}
    logs = glob.glob(os.path.join(os.path.dirname(CORPUS.rstrip('/')), '2026-09-2*', '*uav_mix_eval_*.log'))
    for f in logs:
        s = open(f, errors='replace').read()
        m = re.search(r'(E(?:mf|af|fm|diffusion)_K\d+_mpc4_pid_stopgo_T0\.5(?:_EPlatest)?_p23cv3\w+)', s)
        if not m:
            continue
        want = re.search(r'running \d+/\d+ variants: (\[.*?\])', s)
        want = ast.literal_eval(want.group(1)) if want else []
        done = set(re.findall(r'\[ eval \] corridor variant=([\w-]+) \(', s))
        for w in want:
            wv = w.replace('hardflow_new', 'hardflow_sls')
            out[(m.group(1), wv)] = 'done' if (w in done or wv in done) else 'running'
    return out


JOBS = job_status()


def load_batch():
    """{(geo, eng, K, variant): {rollout_idx: row}} from the DA_UAV_v1 per_rollout_detail.csv (the table source)."""
    out = defaultdict(dict)
    if not BATCH or not os.path.exists(BATCH):
        return out
    import csv
    for r in csv.DictReader(open(BATCH)):
        tag = r.get('run_tag', '')
        if 'p23cv3' not in tag:
            continue
        geo = 'hump' if tag.endswith('p23cv3ah') else 'tilt'
        out[(geo, r['engine'], int(r['K']), r['variant'])][int(float(r['rollout_idx']))] = r
    return out


BATCHROWS = load_batch()


def load_cell(geo, eng, K, variant):
    """Table metrics from the batch CSV when it has the cell (cross-checked equal to the npz, DA §2), else from the npz;
    flown paths / plans from the npz only (`paths` False when the npz is not in the corpus)."""
    tag = GEOS[geo][0]
    d = cell_dir(eng, K, tag, variant)
    ev = f'E{eng}_K{K}_mpc4_pid_stopgo_T0.5{"_EPlatest" if eng == "af" else ""}_{tag}'
    cluster = JOBS.get((ev, variant), 'not submitted')
    base = dict(geo=geo, eng=eng, K=K, variant=variant, cluster=cluster, dir=d)
    z = None
    if d is not None:
        try:
            z = np.load(os.path.join(d, f'{variant}.npz'), allow_pickle=True)
            z['n_steps']
        except Exception:
            z = None
    tms = hom = None
    if d is not None:
        try:
            j = json.load(open(os.path.join(d, 'results.json')))
            tms = np.array([r['timing']['total_ms_mean'] for r in j['rollouts']], float)
            hom = [r.get('homotopy') for r in j['rollouts']]
        except Exception:
            pass
    b = BATCHROWS.get((geo, eng, K, variant))
    if b:
        N = len(b); n = min(N_READ, N); rr = [b[i] for i in range(n)]
        f = lambda col: np.array([float(r[col]) if r.get(col, '') not in ('', 'nan') else np.nan for r in rr])
        out = dict(base, status='ok', source='batch', N=N, n=n, succ=f('success_relaxed'), cf=f('collision_free_completed'),
                   sc=f('n_success_relaxed_and_constraints'), strict=f('n_success'), viol=f('n_violations'),
                   steps=f('n_steps'), div=f('divergence_aborted'), cb=f('projection_cb_tripped'), goal_dist=f('goal_dist'),
                   ms=f('avg_time_ms'), routes=[r.get('homotopy') for r in rr])
    elif z is not None:
        N = len(z['n_steps']); n = min(N_READ, N)
        f = lambda k: np.asarray(z[k][:n], float)
        out = dict(base, status='ok' if tms is not None else 'ok-no-timing', source='npz', N=N, n=n,
                   succ=f('success_relaxed'), cf=f('constraint_collision_free'), sc=f('success_relaxed_and_constraints'),
                   strict=f('success_strict'), viol=f('constraint_n_violations'), steps=f('n_steps'),
                   div=f('divergence_aborted'), cb=f('projection_cb_tripped'), goal_dist=f('goal_dist'),
                   ms=(tms[:n] if tms is not None else np.full(n, np.nan)),
                   routes=(hom[:n] if hom else ['LCR'[i % 3] for i in range(n)]))
    else:
        return dict(base, status={'running': 'running', 'done': 'fetch'}.get(cluster, 'absent'), paths=False, obs=None, plans=None)
    if z is not None:
        n = out['n']
        out['obs'] = [np.asarray(o, float) for o in z['obs_all'][:n]]
        out['plans'] = z['sampled_trajectories_all'][:n] if 'sampled_trajectories_all' in z.files else None
        out['paths'] = True
    else:
        out['obs'] = None; out['plans'] = None; out['paths'] = False
    return out


def all_cells():
    cells = {}
    for geo in GEOS:
        for label, eng, Ks in MODELS:
            for K in Ks:
                vs = [UNPROJ] + list(PER_STEP.values()) + (list(ENDPOINT.values()) if (K >= 3 and eng != 'diffusion') else [])
                for v in vs:
                    cells[(geo, eng, K, v)] = load_cell(geo, eng, K, v)
    return cells


# ── formatting ────────────────────────────────────────────────────────────────────────────────────────────────────
LAB = {eng: lab for lab, eng, _ in MODELS}


def ok(c):
    return c['status'] in ('ok', 'ok-no-timing')


def cnt(c, k):
    return f'{int(c[k].sum())}/{c["n"]}'


def msd(a, nd=1):
    a = np.asarray(a, float)
    if np.all(np.isnan(a)):
        return 'n/a'
    return f'{np.nanmean(a):.{nd}f} ± {np.nanstd(a, ddof=1):.{nd}f}'


def blank(c):
    return {'fetch': 'landed on the cluster, not in this download', 'running': 'still running (C5)',
            'absent': 'no folder', 'not submitted': 'not submitted'}.get(c['status'], c['status'])


def best_rule(cells, geo, eng, K, rules):
    cand = [(r, cells[(geo, eng, K, v)]) for r, v in rules.items() if (geo, eng, K, v) in cells]
    have = [(r, c) for r, c in cand if ok(c)]
    if not have:
        return None, None, [c for _, c in cand]
    key = lambda rc: (-rc[1]['sc'].sum(), rc[1]['viol'].mean(), np.nanmean(rc[1]['ms']) if not np.all(np.isnan(rc[1]['ms'])) else 1e9)
    r, c = sorted(have, key=key)[0]
    missing = [c2 for _, c2 in cand if not ok(c2)]
    return r, c, missing


# ── analyses ──────────────────────────────────────────────────────────────────────────────────────────────────────
def check_snapshots():
    bad = []
    snaps = sorted(glob.glob(f'{CORPUS}/mix_uav_*/*/E*_p23cv3*/6/config_snapshot_uav_mix/uav_projection.yaml'))
    for s in snaps:
        y = yaml.safe_load(open(s))
        g = {x['name']: x for x in y['geo_constraint_variants']}
        for name in ('corridor_v3_tilt', 'corridor_v3_ablation_hump'):
            for k in ('halfspace_constraints', 'obstacle_constraints', 'workspace_bounds', 'planning_inflation', 'scene_xml'):
                if g.get(name, {}).get(k) != GEO_CFG[name].get(k):
                    bad.append((s, name, k))
        for k in ('enlarge_constraints', 'diffusion_timestep_threshold', 'inflation'):
            if y.get(k) != REPO_YAML.get(k):
                bad.append((s, 'global', k))
    return len(snaps), bad


def check_rescore(cells):
    """stored constraint_n_violations vs the eval scorer re-run on the flown path (all ten flights)."""
    n = mism = 0; worst = []
    for (geo, eng, K, v), c in cells.items():
        if not ok(c) or not c['paths']:
            continue
        cfg = geo_config(geo)
        for i, o in enumerate(c['obs']):
            _, nv, _ = NS['_exec_constraint_violations'](o.tolist(), cfg)
            n += 1
            if int(nv) != int(c['viol'][i]):
                mism += 1; worst.append((geo, eng, K, v, i, int(nv), int(c['viol'][i])))
    return n, mism, worst[:5]


def residue(c, geo):
    """violating steps of the ten flights: x range, max depth, share with a clean commanded setpoint."""
    cfg = geo_config(geo)
    xs, depth, clean, tot = [], 0.0, 0, 0
    for o in c['obs']:
        P, PD = o[:, 3:6], o[:, 0:3]
        for k in range(len(P)):
            _, nv, pen = NS['_exec_constraint_violations']([o[k].tolist()], cfg)
            if nv:
                tot += 1; xs.append(P[k, 0]); depth = max(depth, pen)
                if NS['_exec_constraint_violations']([[0, 0, 0, *PD[k], 0, 0, 0]], cfg)[1] == 0:
                    clean += 1
    if not tot:
        return None
    xs = np.array(xs)
    return dict(n=tot, x_lo=float(np.percentile(xs, 5)), x_hi=float(np.percentile(xs, 95)), depth=depth, clean=clean / tot)


def plan_clip(c, zhi=1.2992, zlo=0.9008):
    if c.get('plans') is None:
        return None
    zs = np.concatenate([np.asarray(s, float)[..., [2, 5]].ravel() for s in c['plans']])
    return float(np.mean(np.abs(zs - zhi) < 5e-4)), float(np.mean(np.abs(zs - zlo) < 5e-4))


def altitude(c, geo):
    (lo, hi), stat = G3_WINDOW[geo]
    out = []
    for o in c['obs']:
        x, zz = o[:, 3], o[:, 5]
        m = (x >= lo) & (x <= hi)
        out.append((zz[m].min() if stat == 'min' else zz[m].max()) if m.any() else np.nan)
    return np.array(out)


def frontier(points):
    """points: [(label, sc, ms)] -> the non-dominated set (S&C > 0; costs within 1 % are one cost)."""
    elig = sorted([p for p in points if p[1] > 0 and not math.isnan(p[2])], key=lambda p: (p[2], -p[1]))
    front, best = [], -1
    for p in elig:
        if front and abs(p[2] - front[-1][2]) <= 0.01 * front[-1][2] and p[1] <= front[-1][1]:
            continue
        if p[1] > best:
            front.append(p); best = p[1]
    return front


# ── report ────────────────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    cells = all_cells()
    print(f'# corridor_v3_grid — corpus {CORPUS}\n')

    # inventory
    stat = defaultdict(int)
    for c in cells.values():
        stat[(c['geo'], c['status'])] += 1
    print('## Inventory (68 cells per geometry)\n')
    print('| geometry | readable | readable, no timing | landed, not in download | running | other |')
    print('| :-- | --: | --: | --: | --: | --: |')
    for geo in GEOS:
        s = {k[1]: v for k, v in stat.items() if k[0] == geo}
        other = sum(v for k, v in s.items() if k not in ('ok', 'ok-no-timing', 'fetch', 'running'))
        print(f'| {geo} | {s.get("ok", 0)} | {s.get("ok-no-timing", 0)} | {s.get("fetch", 0)} | {s.get("running", 0)} | {other} |')
    npth = defaultdict(int)
    for c in cells.values():
        if ok(c) and c['paths']:
            npth[c['geo']] += 1
    print(f"\nflown paths (npz) readable: tilt {npth['tilt']}/68, hump {npth['hump']}/68 — table metrics from `{os.path.relpath(BATCH, REPO) if os.path.exists(BATCH) else 'npz only'}`")
    print('\nnot readable here:')
    for (geo, eng, K, v), c in sorted(cells.items()):
        if not ok(c):
            print(f'- {geo} {LAB[eng]} K{K} `{v}` — {blank(c)}')

    # checks
    ns_, bad = check_snapshots()
    n, mism, worst = check_rescore(cells)
    print(f'\n## Checks\n\n- geometry snapshots read: {ns_}; differing from the repo yaml: {len(bad)} {bad[:3]}')
    print(f'- violating steps re-scored from the flown path with the eval scorer: {n} flights, {mism} disagree {worst}')
    dv = sum(int(c["div"].sum()) for c in cells.values() if ok(c)); cb = sum(int(c["cb"].sum()) for c in cells.values() if ok(c))
    print(f'- divergence aborts over all readable flights: {dv}; projection circuit-breaker trips: {cb}')
    nfl = defaultdict(set)
    for c in cells.values():
        if ok(c):
            nfl[(c['variant'] == UNPROJ or 'dpcc' in c['variant'] or 'hardflow' in c['variant'], c['N'])].add(c['eng'])
    print(f'- flights stored per cell (N): {sorted({c["N"] for c in cells.values() if ok(c)})}; the thesis reads the first {N_READ}')

    # Table 6.x raw
    print('\n## tab:uav-corridor-raw — before projection (first ten flights)\n')
    print('| geometry | model | nfe | success | violation-free | S&C | violating steps | ms/step |')
    print('| :-- | :-- | --: | --: | --: | --: | --: | --: |')
    for geo in GEOS:
        for lab, eng, Ks in MODELS:
            for K in Ks:
                c = cells[(geo, eng, K, UNPROJ)]
                if ok(c):
                    print(f'| {geo} | {lab} | {K} | {cnt(c, "succ")} | {cnt(c, "cf")} | {cnt(c, "sc")} | {msd(c["viol"])} | {msd(c["ms"])} |')
                else:
                    print(f'| {geo} | {lab} | {K} | *{blank(c)}* | | | | |')

    # projection methods, every rule, every budget
    print('\n## tab:uav-corridor-projection — S&C per rule (x/10); K1/K2 per-step rows are extra (the chapter prints K ≥ 3)\n')
    print('| geometry | model | nfe | per-step r | c | t | endpoint single | r | c | t |')
    print('| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |')
    for geo in GEOS:
        for lab, eng, Ks in MODELS:
            for K in Ks:
                row = []
                for r, v in list(PER_STEP.items()) + list(ENDPOINT.items()):
                    c = cells.get((geo, eng, K, v))
                    row.append('—' if c is None else (cnt(c, 'sc') if ok(c) else ('*run*' if c['status'] == 'running' else '*fetch*')))
                print(f'| {geo} | {lab} | {K} | ' + ' | '.join(row) + ' |')

    # best rule per projector
    print('\n## tab:uav-corridor — best rule per projector (S&C, rule, ms/step)\n')
    print('| geometry | model | nfe | per-step S&C | rule | ms/step | endpoint S&C | rule | ms/step |')
    print('| :-- | :-- | --: | --: | :-- | --: | --: | :-- | --: |')
    fpts = defaultdict(list)
    for geo in GEOS:
        for lab, eng, Ks in MODELS:
            for K in Ks:
                out = []
                for proj, rules in (('per-step', PER_STEP), ('endpoint', ENDPOINT)):
                    if proj == 'endpoint' and (K < 3 or eng == 'diffusion'):
                        out.append('— | — | —'); continue
                    r, c, missing = best_rule(cells, geo, eng, K, rules)
                    if c is None:
                        out.append(f'*{blank(missing[0]) if missing else "?"}* | | '); continue
                    note = f' ({len(missing)} rule(s) not readable)' if missing else ''
                    out.append(f'{cnt(c, "sc")} | {r}{note} | {np.nanmean(c["ms"]):.1f}')
                    fpts[geo].append((f'{lab} K{K} {proj} {r}', int(c['sc'].sum()), float(np.nanmean(c['ms'])), proj))
                print(f'| {geo} | {lab} | {K} | ' + ' | '.join(out) + ' |')

    # diagnostic detail, every projected cell
    print('\n## Detail — every readable projected cell: outcome, residue location, setpoint, plan box\n')
    print('| geometry | model | nfe | variant | success | viol-free | S&C | strict | viol. steps | steps | residue x (5–95 %) | max depth m | setpoint clean | plan on box top / bottom |')
    print('| :-- | :-- | --: | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: | :-- |')
    detail = {}
    for (geo, eng, K, v), c in sorted(cells.items(), key=lambda kv: (kv[0][0], [m[1] for m in MODELS].index(kv[0][1]), kv[0][2], kv[0][3])):
        if not ok(c):
            continue
        if not c['paths']:
            short = v.replace(STACK, '').replace('hardflow_sls', 'hf')
            print(f'| {geo} | {LAB[eng]} | {K} | {short} | {cnt(c, "succ")} | {cnt(c, "cf")} | {cnt(c, "sc")} | {cnt(c, "strict")} | '
                  f'{msd(c["viol"])} | {np.mean(c["steps"]):.0f} | *paths: fetch* | | | |')
            continue
        rs = residue(c, geo); pc = plan_clip(c)
        detail[(geo, eng, K, v)] = (rs, pc)
        rtxt = '—' if rs is None else f'[{rs["x_lo"]:+.2f}, {rs["x_hi"]:+.2f}]'
        dtxt = '—' if rs is None else f'{rs["depth"]:.3f}'
        ctxt = '—' if rs is None else f'{rs["clean"]:.2f}'
        ptxt = 'n/a' if pc is None else f'{pc[0]:.2f} / {pc[1]:.2f}'
        short = v.replace(STACK, '').replace('hardflow_sls', 'hf')
        print(f'| {geo} | {LAB[eng]} | {K} | {short} | {cnt(c, "succ")} | {cnt(c, "cf")} | {cnt(c, "sc")} | {cnt(c, "strict")} | '
              f'{msd(c["viol"])} | {np.mean(c["steps"]):.0f} | {rtxt} | {dtxt} | {ctxt} | {ptxt} |')

    # altitude, paired by trial index
    print('\n## Altitude — the flown extreme in the constraint window, projected − unprojected, same trial\n')
    print('tilt: min z over x ∈ [0.5, 2.0] (the descent); hump: max z over x ∈ [−0.5, 0.5] (the climb). Unprojected band first.\n')
    print('| geometry | model | nfe | unprojected band (m) | projected variant | Δz per flight: min / median / max (m) |')
    print('| :-- | :-- | --: | :-- | :-- | :-- |')
    for geo in GEOS:
        for lab, eng, Ks in MODELS:
            for K in Ks:
                u = cells[(geo, eng, K, UNPROJ)]
                if not ok(u) or not u['paths']:
                    print(f'| {geo} | {lab} | {K} | *paths: fetch* | | |')
                    continue
                zu = altitude(u, geo)
                for v in list(PER_STEP.values()) + list(ENDPOINT.values()):
                    c = cells.get((geo, eng, K, v))
                    if c is None or not ok(c):
                        continue
                    if not c['paths']:
                        print(f'| {geo} | {lab} | {K} | | {v.replace(STACK, "").replace("hardflow_sls", "hf")} | *paths: fetch* |')
                        continue
                    zp = altitude(c, geo); m = min(len(zu), len(zp)); dz = zp[:m] - zu[:m]
                    if np.all(np.isnan(dz)):
                        txt = 'flights never enter the window'
                    else:
                        txt = f'{np.nanmin(dz):+.2f} / {np.nanmedian(dz):+.2f} / {np.nanmax(dz):+.2f}  (entered {int(np.sum(~np.isnan(zp)))}/10)'
                    print(f'| {geo} | {lab} | {K} | {np.nanmin(zu):.2f}–{np.nanmax(zu):.2f} | {v.replace(STACK, "").replace("hardflow_sls", "hf")} | {txt} |')

    # where projected flights stop when they do not cross
    print('\n## Flights that do not cross the finish line — where they stop\n')
    print('| geometry | model | nfe | variant | not crossed | final x (median) | final altitude (median, m) | ended on the step cap |')
    print('| :-- | :-- | --: | :-- | --: | --: | --: | --: |')
    for (geo, eng, K, v), c in sorted(cells.items()):
        if not ok(c) or c['succ'].sum() == c['n']:
            continue
        fail = [i for i in range(c['n']) if not c['succ'][i]]
        if not c['paths']:
            cap = sum(1 for i in fail if c['steps'][i] >= 396)
            print(f'| {geo} | {LAB[eng]} | {K} | {v.replace(STACK, "").replace("hardflow_sls", "hf")} | {len(fail)}/{c["n"]} | *fetch* | *fetch* | {cap}/{len(fail)} |')
            continue
        fx = np.median([c['obs'][i][-1, 3] for i in fail]); fz = np.median([c['obs'][i][-1, 5] for i in fail])
        cap = sum(1 for i in fail if c['steps'][i] >= 396)
        print(f'| {geo} | {LAB[eng]} | {K} | {v.replace(STACK, "").replace("hardflow_sls", "hf")} | {len(fail)}/{c["n"]} | {fx:+.2f} | {fz:.2f} | {cap}/{len(fail)} |')

    # frontier
    print('\n## Frontier (S&C of 10 against ms/step; S&C 0 not eligible; one point per model × budget × projector at its best rule)\n')
    for geo in GEOS:
        fr = frontier(fpts[geo])
        elig = [p for p in fpts[geo] if p[1] > 0]
        print(f'- **{geo}**: {len(elig)} eligible of {len(fpts[geo])} readable points; frontier: '
              + ('none — no configuration succeeds violation-free' if not fr else '; '.join(f'{p[0]} ({p[1]}/10, {p[2]:.1f} ms)' for p in fr)))

    if a.json:
        dump = {}
        for k, c in cells.items():
            d = {kk: vv for kk, vv in c.items() if kk not in ('obs', 'plans')}
            for kk, vv in list(d.items()):
                if isinstance(vv, np.ndarray):
                    d[kk] = vv.tolist()
            dump['|'.join(map(str, k))] = d
        json.dump(dump, open(a.json, 'w'), indent=1, default=str)
        print(f'\n[json] {a.json}')


if __name__ == '__main__':
    main()
