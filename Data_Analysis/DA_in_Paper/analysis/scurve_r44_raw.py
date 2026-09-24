#!/usr/bin/env python3.14
"""UAV-s-curve R44 (a, b, c): every number Chapter 6's UAV-s-curve section asks for, from the RAW result folders.

    python3.14 Data_Analysis/DA_in_Paper/analysis/scurve_r44_raw.py                    # markdown to stdout
    SCURVE_RAW=<folder holding logs/ and Slurm_Codes/> python3.14 …/scurve_r44_raw.py  # another fetch

Corpus  : the fetch of 2026-09-24 16:17 (Slurm_Codes/temp_bash/fetch_20260924_R44_scurve.sh, git 5092e056), unpacked in
          temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747 -- 621 files, every md5 of its manifest verified.
          Tags p23scgrid (A, jobs 26167-26176), p23scproj (B1, job 26195), p23scmjpc (C0, job 26196; C1 when it lands).
          No DA_UAV_v1 batch is read (author, 24-09): results.json + npz + rollout logs + the job records only.
Metrics : per flight, the evaluation's own fields (results.json rollouts[i], index-aligned with the npz):
            success         success.relaxed = crossed the finish line AND safe (contact within the scene limit, airborne:
                            min z > 0.2 m, not ended by the divergence guard)             eval_mix_uav.py:1994-2027
            violating steps constraint.n_violations -- control steps on which the FLOWN position is within the 0.31 m rotor
                            reach of a wall / corner / the workspace box, counted until the flight ends (mean +- sample sd)
            distance        goal.dist (mean);  aborted  divergence.aborted (count);  ms/step  timing.total_ms_mean (mean,
                            network + projection per action);  tracking error  track_err_mean (mean)
Checks  : (a) each run's geometry snapshot (config_snapshot_uav_mix/uav_projection.yaml) equals the repo's s_curve_hg entry;
          (b) violating steps re-scored from the flown path with the evaluation's own scorer (extracted torch-free from
              eval_mix_uav.py) equal the stored count, flight by flight;  (c) projection health (cut-off trips, backstop);
          (d) determinism: the B1 `diffuser` cell against A7 (same configuration), flight by flight.
Tracking: at every violating step, is the COMMANDED setpoint (obs[:, 0:3]) itself inside the rotor reach (the plan is
          infeasible there) or clean (the vehicle left a feasible command)?  Same scorer, applied to the setpoint.
Paths   : lead = |p_des - p| (what the evaluation calls tracking error: mostly the setpoint running ahead along the
          path); cross-track = distance of the flown position to the polyline of the flight's commanded setpoints (the
          sideways deviation from the commanded path); corner clearance = closest approach of the commanded path and of the
          flown path to the two inside corners of the crossover minus their keep-out radius (0.05 m + 0.31 m rotor reach);
          negative = inside.  The projector's default binding (no `-pdes` token) constrains the plan's measured position p,
          not the commanded setpoint p_des (eval_mix_uav.py:1400-1418); corridor v3 runs use `-pdes`, s-curve cells do not.
Cost    : the chapter's decomposition (tab:uav-controller-cost): elapsed time of a flight (job record, per-trial
          "trial k/10 done (Xs elapsed this variant)") / executed control steps (n_fm_steps), minus the planner time
          (total_ms_mean) = the controller and the simulator step.
Rule    : C1's per-step rule = the one with the most successes in B1; ties -> fewer aborted -> fewer violating steps ->
          smaller distance (data_status/PENDING_20260923_uav_scurve_R44_raw_first.md §4).
Author  : Claude (Opus 5.5, Claude Code, R44 run chat), 2026-09-24.
"""
import ast
import glob
import json
import os
import re
import statistics as st
from collections import Counter

import numpy as np
import yaml

REPO = '/workspaces/FM-PCC'
RAW = os.environ.get('SCURVE_RAW', os.path.join(REPO, 'temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747'))
PLANS = os.path.join(RAW, 'logs/UAV_MIX/uav-s_curve/plans')
JOBLOGS = os.path.join(RAW, 'Slurm_Codes/logs')
GEO = 's_curve_hg'
TRAIN = {'mf': 'H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet',
         'af': 'H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet',
         'fm': 'H8_Dmodels.diffusion.FlowMatchingODE_9D',
         'diffusion': 'H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20'}
LABEL = {'mf': 'MeanFM', 'af': 'CI-MeanFM', 'fm': 'FM', 'diffusion': 'Diffusion'}
RULES = [('dpcc-r-tightened', 'random'), ('dpcc-c-tightened', 'cumulative cost'),
         ('dpcc-t-tightened', 'temporal consistency')]


# ── the evaluation's own scorer, extracted torch-free (same method as corridor_v3_grid.py) ─────────────────────────
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


def geo_cfg(yml):
    g = dict(next(e for e in yml['geo_constraint_variants'] if e['name'] == GEO))
    g['inflation'] = yml['inflation']        # the scorer's rotor reach (0.31); planning_inflation is the projector's
    return g


CFG = geo_cfg(REPO_YAML)


def violates(p, cfg=CFG):
    """True if a single position is inside the rotor reach of any spatial constraint (the scorer's rule)."""
    return NS['_exec_constraint_violations']([[0, 0, 0, *p]], cfg)[1] > 0


def families(p, cfg=CFG):
    """Which constraint families a violating position touches: 'box', 'wall', 'corner' (mirrors the scorer)."""
    out = set(); r = cfg['inflation']['r_drone']
    ws = cfg['workspace_bounds']; lb = np.array(ws['lb'], float); ub = np.array(ws['ub'], float)
    if np.any((p < lb + r)[np.isfinite(lb)]) or np.any((p > ub - r)[np.isfinite(ub)]):
        out.add('box')
    for hs in cfg['halfspace_constraints']:
        if NS['_exec_constraint_violations']([[0, 0, 0, *p]], dict(cfg, obstacle_constraints=[], workspace_bounds=None,
                                                                   halfspace_constraints=[hs]))[1]:
            out.add('wall')
    for ob in cfg['obstacle_constraints']:
        if NS['_exec_constraint_violations']([[0, 0, 0, *p]], dict(cfg, halfspace_constraints=[], workspace_bounds=None,
                                                                   obstacle_constraints=[ob]))[1]:
            out.add('corner')
    return out


# ── discovery ─────────────────────────────────────────────────────────────────────────────────────────────────────
def cell_dir(eng, K, ctrl, tag, variant):
    ep = '_EPlatest' if eng == 'af' else ''
    d = os.path.join(PLANS, f'mix_uav_{eng}', TRAIN[eng], f'E{eng}_K{K}_mpc4_{ctrl}_T0.5{ep}_{tag}', '6')
    hits = glob.glob(os.path.join(d, f'{GEO}_*', variant))
    return hits[0] if hits else None


def job_log(name):
    hits = sorted(glob.glob(os.path.join(JOBLOGS, '*', f'*_{name}_[0-9]*.log')))
    return hits[-1] if hits else None


def load(eng, K, ctrl, tag, variant, job=None):
    d = cell_dir(eng, K, ctrl, tag, variant)
    if d is None:
        return None
    res = json.load(open(os.path.join(d, 'results.json')))
    R = res['rollouts']
    z = np.load(os.path.join(d, f'{variant}.npz'), allow_pickle=True)
    obs = [np.asarray(o, float) for o in z['obs_all']]
    c = dict(eng=eng, K=K, ctrl=ctrl, tag=tag, variant=variant, dir=d, R=R, obs=obs, summary=res['summary'], job=job)
    c['n'] = len(R)
    c['succ'] = sum(bool(r['success']['relaxed']) for r in R)
    c['sc'] = sum(bool(r['success']['relaxed_and_constraints']) for r in R)
    c['vfree'] = sum(bool(r['constraint']['collision_free']) for r in R)
    c['viol'] = [int(r['constraint']['n_violations']) for r in R]
    c['dist'] = [float(r['goal']['dist']) for r in R]
    c['ab'] = sum(bool(r['divergence']['aborted']) for r in R)
    c['ab_steps'] = [int(r['divergence']['step']) for r in R if r['divergence']['aborted']]
    c['ms'] = [float(r['timing']['total_ms_mean']) for r in R]
    c['proj_ms'] = [float(r['timing']['proj_ms_mean']) for r in R]
    c['track'] = [float(r['track_err_mean']) for r in R]
    c['steps'] = [int(r['n_fm_steps']) for r in R]
    c['crossed_unsafe'] = sum(1 for r in R if r['goal']['crossed_line'] and not r['success']['relaxed'])
    c['reached'] = sum(bool(r['goal']['reached']) for r in R)
    c['contact'] = [float(r['physical']['contact_frac']) for r in R]
    c['min_z'] = [float(r['physical']['min_z']) for r in R]
    c['unsafe_why'] = Counter(
        ('aborted' if r['divergence']['aborted'] else 'floor' if r['physical']['min_z'] <= 0.2 else 'contact')
        for r in R if not r['physical']['safe'])
    c['cb'] = sum(int(r['projection_health'].get('cb_trips', 0)) for r in R)
    c['backstop'] = sum(int(r['projection_health'].get('backstop_hits', 0)) for r in R)
    return c


def mean_sd(xs):
    return f'{st.mean(xs):.1f} ± {st.stdev(xs):.1f}' if len(xs) > 1 else f'{xs[0]:.1f}'


# ── checks ────────────────────────────────────────────────────────────────────────────────────────────────────────
def check_snapshot(c):
    snap = os.path.join(os.path.dirname(os.path.dirname(c['dir'])), 'config_snapshot_uav_mix', 'uav_projection.yaml')
    if not os.path.exists(snap):
        return 'no snapshot'
    y = yaml.safe_load(open(snap))
    return 'same' if geo_cfg(y) == CFG else 'DIFFERS'


def rescore(c):
    """(flights whose re-scored violating steps differ from the stored count, violating-step analysis)."""
    mism = 0; tot = 0; clean_sp = 0; sp_viol_steps = 0; fam = Counter(); xs = []
    for o, r in zip(c['obs'], c['R']):
        _, nv, _ = NS['_exec_constraint_violations'](o.tolist(), CFG)
        if nv != int(r['constraint']['n_violations']):
            mism += 1
        for k in range(len(o)):
            pd, p = o[k, 0:3], o[k, 3:6]
            sp_bad = violates(pd)
            sp_viol_steps += int(sp_bad)
            if violates(p):
                tot += 1; xs.append(p[0])
                clean_sp += int(not sp_bad)
                for f in families(p):
                    fam[f] += 1
    return dict(mismatch=mism, viol_steps=tot, clean_setpoint=(clean_sp / tot if tot else float('nan')),
                setpoint_viol_steps=sp_viol_steps, families=fam,
                x_lo=(float(np.percentile(xs, 5)) if xs else float('nan')),
                x_hi=(float(np.percentile(xs, 95)) if xs else float('nan')))


CORNERS = [np.asarray(o['center'][:2], float) for o in CFG['obstacle_constraints']]
R_KEEP = [float(o['radius']) + float(CFG['inflation']['r_drone']) for o in CFG['obstacle_constraints']]


def _poly_dist(p, A, B):
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
    return float(np.min(np.linalg.norm(A + t[:, None] * AB - p, axis=1)))


def paths(c):
    """Per flight: lead, cross-track to the commanded path, corner clearance of commanded and flown paths."""
    lead, xt, sp_cl, fl_cl, sp_in, fl_in, which = [], [], [], [], 0, 0, Counter()
    for o in c['obs']:
        PD, P = o[:, 0:2], o[:, 3:5]
        lead.append(float(np.linalg.norm(PD - P, axis=1).mean()))
        xt.append(float(np.mean([_poly_dist(P[k], PD[:-1], PD[1:]) for k in range(0, len(P), 3)])))
        s_ = [float(np.linalg.norm(PD - cc, axis=1).min()) - r for cc, r in zip(CORNERS, R_KEEP)]
        f_ = [float(np.linalg.norm(P - cc, axis=1).min()) - r for cc, r in zip(CORNERS, R_KEEP)]
        sp_cl.append(min(s_)); fl_cl.append(min(f_))
        sp_in += min(s_) < 0; fl_in += min(f_) < 0
        which[int(np.argmin(f_))] += 1
    return dict(lead=st.mean(lead), xtrack=st.mean(xt), xtrack_max=max(xt), sp_cl=st.median(sp_cl),
                fl_cl=st.median(fl_cl), sp_in=sp_in, fl_in=fl_in, corner=dict(which))


def abort_xy(c):
    return [tuple(np.round(o[-1, 3:5], 2)) for o, r in zip(c['obs'], c['R']) if r['divergence']['aborted']]


X_WALL_END = 3.0   # eval_mix_uav.py SCENE_CLEAR_LINE_X['s_curve'] (R44, 24-09): the end of the walls


def wall_end(c):
    """Success re-scored with the finish line at the end of the walls: crossed = the stored goal-plane/sphere latch OR the
    flown x reaching 3.0 at a control step; success = crossed AND safe (unchanged). Near = within 3 cm of the line without
    crossing (the eval tests after every physics step, the npz holds control steps)."""
    new, changed, near = 0, [], 0
    for i, (o, r) in enumerate(zip(c['obs'], c['R'])):
        xmax = float(o[:, 3].max())
        crossed = bool(r['goal']['crossed_line']) or xmax >= X_WALL_END
        ok = crossed and bool(r['physical']['safe'])
        new += ok
        if ok != bool(r['success']['relaxed']):
            changed.append(i)
        if not crossed and xmax >= X_WALL_END - 0.03:
            near += 1
    return new, changed, near


def per_trial_elapsed(logpath, variant):
    """Seconds per flight for one variant from the job record's cumulative 'trial k/10 done (Xs elapsed ...)' lines."""
    if logpath is None:
        return None
    cum = []; active = False
    for line in open(logpath, errors='replace'):
        m = re.search(r">>> variant \d+/\d+: '([^']+)'", line)
        if m:
            active = (m.group(1) == variant)
            continue
        m = re.search(r'variant=([\w.-]+): trial (\d+)/(\d+) done\s+\(([\d.]+)s elapsed this variant', line)
        if m and active and m.group(1) == variant:
            cum.append(float(m.group(4)))
    return [b - a for a, b in zip([0.0] + cum[:-1], cum)] if cum else None


def cost(c):
    el = per_trial_elapsed(job_log(c['job']), c['variant']) if c.get('job') else None
    if not el or len(el) != c['n']:
        return None
    loop = [1000.0 * e / s for e, s in zip(el, c['steps'])]
    return dict(loop=st.mean(loop), planner=st.mean(c['ms']), rest=st.mean(loop) - st.mean(c['ms']),
                first=loop[0], median=st.median(loop))


# ── report ────────────────────────────────────────────────────────────────────────────────────────────────────────
def main():
    A = {(e, K): load(e, K, 'pid_stopgo', 'p23scgrid', 'diffuser', f'p23scgrid_A{i}_{e}_K{K}')
         for i, e, K in ((1, 'mf', 1), (2, 'mf', 2), (3, 'mf', 20), (4, 'af', 1), (5, 'af', 2), (6, 'af', 20),
                         (7, 'fm', 1), (8, 'fm', 2), (9, 'fm', 20), (10, 'diffusion', 20))}
    B = {v: load('fm', 1, 'pid_stopgo', 'p23scproj', v, 'p23scproj_B1_fm_K1') for v in
         ['diffuser'] + [r for r, _ in RULES]}
    C0 = load('fm', 1, 'mjpc', 'p23scmjpc', 'diffuser', 'p23scmjpc_C0_fm_K1')
    C1 = {r: load('fm', 1, 'mjpc', 'p23scmjpc', r, 'p23scmjpc_C1_fm_K1') for r, _ in RULES}
    C1 = {r: c for r, c in C1.items() if c is not None}
    cells = [c for c in list(A.values()) + list(B.values()) + [C0] + list(C1.values()) if c is not None]

    print(f'# UAV-s-curve R44 — raw result folders\n\ncorpus `{os.path.relpath(RAW, REPO)}`\n')
    print('## 1 · Checks\n')
    print('| cell | flights | geometry snapshot | re-scored ≠ stored | cut-off trips | backstop | job log |')
    print('| :-- | --: | :-- | --: | --: | --: | :-- |')
    RS = {}
    for c in cells:
        RS[id(c)] = rescore(c)
        jl = job_log(c['job']) if c.get('job') else None
        print(f"| {c['tag']} {c['eng']} K{c['K']} {c['ctrl']} {c['variant']} | {c['n']} | {check_snapshot(c)} | "
              f"{RS[id(c)]['mismatch']} | {c['cb']} | {c['backstop']} | {os.path.basename(jl) if jl else '—'} |")
    a7, b0 = A[('fm', 1)], B['diffuser']
    same = sum(1 for i in range(10) for m in ('success', 'constraint', 'goal', 'divergence', 'n_fm_steps', 'track_err_mean')
               if a7['R'][i][m] != b0['R'][i][m])
    print(f"\nDeterminism: B1 `diffuser` against A7 (same configuration, different job): {same} differing outcome "
          f"fields over 10 flights; ms/step {st.mean(a7['ms']):.1f} vs {st.mean(b0['ms']):.1f}.")

    print('\n## 2 · Table 6.14 `tab:uav-scurve`, recomputed from the raw folders\n')
    print('| model | nfe | success | violating steps | distance [m] | aborted | ms/step | tracking error [m] |')
    print('| :-- | --: | --: | --: | --: | --: | --: | --: |')
    for (e, K), c in A.items():
        print(f"| {LABEL[e]} | {K} | {c['succ']}/10 | {mean_sd(c['viol'])} | {st.mean(c['dist']):.3f} | {c['ab']}/10 | "
              f"{st.mean(c['ms']):.1f} | {st.mean(c['track']):.3f} |")

    print('\n## 3 · Table 6.15 `tab:uav-scurve-projection` — FM, nfe 1, cascaded geometric controller\n')
    print('| projection | rule | success | violating steps | distance [m] | aborted | ms/step | S&C | crossed but not safe | not safe (why) | tracking error [m] |')
    print('| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: |')
    rows = [('none', '—', B['diffuser'])] + [('per-step', lab, B[r]) for r, lab in RULES]
    for proj, lab, c in rows:
        print(f"| {proj} | {lab} | {c['succ']}/10 | {mean_sd(c['viol'])} | {st.mean(c['dist']):.3f} | {c['ab']}/10 | "
              f"{st.mean(c['ms']):.1f} | {c['sc']}/10 | {c['crossed_unsafe']} | {dict(c['unsafe_why'])} | {st.mean(c['track']):.3f} |")
    print('| endpoint | — | no guiding step at nfe 1 | | | | | | | | |')
    rank = sorted(RULES, key=lambda rl: (-B[rl[0]]['succ'], B[rl[0]]['ab'], st.mean(B[rl[0]]['viol']),
                                         st.mean(B[rl[0]]['dist'])))
    print(f"\n**C1 rule:** `{rank[0][0]}` ({rank[0][1]}) — order: " + ' > '.join(
        f"{lab} (succ {B[r]['succ']}, aborted {B[r]['ab']}, viol {st.mean(B[r]['viol']):.1f}, dist {st.mean(B[r]['dist']):.3f})"
        for r, lab in rank))

    print('\n## 4 · Table 6.16 `tab:uav-controller` — FM, nfe 1, the same plans under the two controllers (10 vs 10)\n')
    print('| projection | controller | success | violating steps | distance [m] | aborted | S&C | goal within 0.3 m | tracking error [m] | not safe (why) |')
    print('| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- |')
    best = rank[0][0]
    for proj, ctl, c in (('none', 'cascaded geometric', a7), ('none', 'MuJoCo MPC', C0),
                         (f'per-step ({dict(RULES)[best]})', 'cascaded geometric', B[best]),
                         (f'per-step ({dict(RULES)[best]})', 'MuJoCo MPC', C1.get(best))):
        if c is None:
            print(f'| {proj} | {ctl} | pending (C1) | | | | | | | |'); continue
        print(f"| {proj} | {ctl} | {c['succ']}/10 | {mean_sd(c['viol'])} | {st.mean(c['dist']):.3f} | {c['ab']}/10 | "
              f"{c['sc']}/10 | {c['reached']}/10 | {st.mean(c['track']):.3f} | {dict(c['unsafe_why'])} |")

    print('\n## 5 · Which constraints the flown path violates (families per violating step; a step can touch several)\n')
    print('| cell | violating steps (all flights) | families at violating steps | x range (5–95 %) |')
    print('| :-- | --: | :-- | :-- |')
    for c in [a7, C0] + [B[r] for r, _ in RULES] + list(C1.values()) + [A[('mf', 1)], A[('af', 1)], A[('diffusion', 20)]]:
        s = RS[id(c)]
        print(f"| {c['tag']} {LABEL[c['eng']]} K{c['K']} {c['ctrl']} {c['variant']} | {s['viol_steps']} | "
              f"{dict(s['families'])} | {s['x_lo']:.2f} … {s['x_hi']:.2f} |")
    print('\n(box = the workspace box: in the unprojected cells its ceiling, z > 1.49 m, almost only on flights that go on to invert; in the projected cells also the floor after a drop and the far end / side after the line; wall = the four corridor walls; '
          'corner = the two inside corners of the crossover. Setpoint and flown position are not compared step by step here: '
          'the setpoint leads by ~0.3 m, so a same-step comparison mislabels lag as tracking; §5b compares the paths.)')

    print('\n## 5b · Lead, cross-track and the inside corners (commanded path vs flown path)\n')
    print('| cell | lead \\|p_des−p\\| [m] | cross-track to commanded path [m] (max flight) | corner clearance, commanded path '
          '(median) | …flown path | commanded path inside, flights | flown path inside, flights | nearest corner (flights) |')
    print('| :-- | --: | --: | --: | --: | --: | --: | :-- |')
    for c in [a7, C0] + [B[r] for r, _ in RULES] + list(C1.values()) + [A[('mf', 1)], A[('af', 1)], A[('fm', 20)],
                                                                         A[('diffusion', 20)]]:
        q = paths(c)
        print(f"| {c['tag']} {LABEL[c['eng']]} K{c['K']} {c['ctrl']} {c['variant']} | {q['lead']:.3f} | "
              f"{q['xtrack']:.3f} ({q['xtrack_max']:.3f}) | {q['sp_cl']:+.3f} | {q['fl_cl']:+.3f} | {q['sp_in']}/10 | "
              f"{q['fl_in']}/10 | {q['corner']} |")
    print('\nCorners: 0 = (−0.5, −0.3), the first inside corner; 1 = (0.5, 0.3), the second. Keep-out radius 0.36 m.')
    print('\nWhere aborted flights end (flown x, y at the abort):')
    for (e, K), c in A.items():
        if c['ab']:
            xy = abort_xy(c)
            print(f"- {LABEL[e]} K{K}: x {min(x for x, _ in xy):.2f} … {max(x for x, _ in xy):.2f}; y {min(y for _, y in xy):.2f} … "
                  f"{max(y for _, y in xy):.2f}  ({len(xy)} flights)")

    print('\n## 6 · Controller cost per executed control step, FM nfe 1 (job records; 10 vs 10)\n')
    print('| cell | the control loop [ms] | the planner [ms] | controller + simulator step [ms] | first flight | median |')
    print('| :-- | --: | --: | --: | --: | --: |')
    for lab, c in (('unprojected, cascaded geometric (A7)', a7), ('unprojected, MuJoCo MPC (C0)', C0),
                   (f'per-step {dict(RULES)[best]}, cascaded geometric (B1)', B[best]),
                   (f'per-step {dict(RULES)[best]}, MuJoCo MPC (C1)', C1.get(best))):
        k = cost(c) if c is not None else None
        if k is None:
            print(f'| {lab} | pending | | | | |'); continue
        print(f"| {lab} | {k['loop']:.1f} | {k['planner']:.1f} | {k['rest']:.1f} | {k['first']:.1f} | {k['median']:.1f} |")

    print('\n## 6b · Finish line at the end of the walls (x = 3.0 m, eval code since 24-09) — every s-curve cell re-scored\n')
    print('| cell | success, goal point (as flown) | success, end of the walls | flights that change | within 3 cm, not crossed |')
    print('| :-- | --: | --: | :-- | --: |')
    tot_changed = 0
    for c in cells:
        n_new, ch, near = wall_end(c)
        tot_changed += len(ch)
        print(f"| {c['tag']} {LABEL[c['eng']]} K{c['K']} {c['ctrl']} {c['variant']} | {c['succ']}/10 | {n_new}/10 | "
              f"{ch or '—'} | {near} |")
    print(f'\nFlights whose success changes: {tot_changed} of {10 * len(cells)}.')

    print('\n## 7 · Aborts and the flight end, FM nfe 1\n')
    for lab, c in (('A7 cascaded', a7), ('C0 MPC', C0)) + tuple((f'B1 {r}', B[r]) for r, _ in RULES):
        print(f"- {lab}: aborted at steps {c['ab_steps'] or '—'}; steps per flight {st.mean(c['steps']):.0f}; "
              f"contact fraction {st.mean(c['contact']):.3f} (max {max(c['contact']):.3f}); lowest altitude {min(c['min_z']):.2f} m")


if __name__ == '__main__':
    main()
