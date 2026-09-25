#!/usr/bin/env python3.14
"""Extract the expert data of the two environments D3IL-avoiding already has drawn,
and test it against the constraint set each is evaluated against.

    python3.14 plotting/extract/expert_paths.py

Why this exists: `tab:avoiding-geometries` and `fig:constraints-avoiding` show, for the
benchmark, what the demonstrations do against the constraints -- that they cross them, so
satisfying them is the projection's job and not the model's. The alignment task and the
three quadrotor scenes had no such statement. This produces it for both.

What each side is, and what it is NOT:

* **Quadrotor.** The demonstration generator drives the vehicle along an analytic reference
  path per scene and homotopy class (`uav_expert_data_collect/trajectories.py`, selected in
  `generator.py::_build_traj_and_init`). That reference is reproduced here exactly, for every
  homotopy class the collection cycles through, and tested against the same constraint set the
  projection sees -- obstacles inflated by the vehicle radius, and the tightened variant.
  It is the commanded path, not the flown one: the flown demonstrations live on the cluster.
  Altitude does not enter, because every constraint of `sources.UAV_CONSTRAINTS` is in x-y.

* **Alignment.** The demonstration trajectories are on the cluster, but the *contexts* they
  were recorded from are in the repo (`d3il/environments/dataset/data/aligning/
  {train,test}_contexts.pkl`, 60 each): the initial box pose and the target pose, which is
  exactly what a demonstration has to get from one to the other. Tested here is the direct
  push, the straight segment from the box centre to the target centre -- the shortest thing a
  demonstration could do -- against the keep-out region and the halfspace.

Needs numpy (python3.14 in the AI container has it); the builders read only the JSON.
"""
import json
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import sources as S                                            # noqa: E402

OUT = os.path.join(S.REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'expert_paths.json')
NDP = 4
N_SAMPLES = 400

# The homotopy classes the collection cycles through, and the path each one builds.
# uav_expert_data_collect/generator.py::HOMOTOPY_CLASSES and _build_traj_and_init.
UAV_ROUTES = {
    'UAV-corridor': [('L', 'corridor_path', ('L', 1.1, 8.0)),
                     ('C', 'corridor_path', ('C', 1.1, 8.0)),
                     ('R', 'corridor_path', ('R', 1.1, 8.0))],
    'UAV-pillars': [(lab, 'pillar_path', (seq, 1.1, 13.0)) for lab, seq in (
        ('(L,L,L)', ('L', 'L', 'L')), ('(L,R,L)', ('L', 'R', 'L')),
        ('(R,L,R)', ('R', 'L', 'R')), ('(R,R,R)', ('R', 'R', 'R')))],
    'UAV-s-curve': [('default', 's_curve_scene_path', (1.1, 19.0))],
}


# ---------------------------------------------------------------- constraints --
def _halfspace_violation(p, hs, margin):
    """Signed depth into the excluded side of an inflated halfspace, 0 if outside it.

    The boundary moves toward the free side by `margin` (the vehicle radius, plus the
    tightening for the tightened set), which is what inflating the wall by the radius
    of a disk-shaped vehicle amounts to.
    """
    x, y = p
    xa = hs.get('x_active')
    if xa and not (xa[0] <= x <= xa[1]):
        return 0.0
    (ax, ay), (bx, by) = hs['p0'], hs['p1']
    dx, dy = bx - ax, by - ay
    n = math.hypot(dx, dy)
    # unit normal pointing to the side the plan must stay on
    nx, ny = -dy / n, dx / n
    signed = (x - ax) * nx + (y - ay) * ny
    if hs['side'] == 'below':
        signed = -signed
    # signed > 0 means on the allowed side; the boundary is offset by the margin
    return max(0.0, margin - signed)


def _disk_violation(p, dk, margin):
    d = math.hypot(p[0] - dk['c'][0], p[1] - dk['c'][1])
    return max(0.0, dk['r'] + margin - d)


def uav_violation(p, scene, margin):
    """Deepest violation of the inflated constraint set at one point, 0 if clean."""
    v = 0.0
    for hs in scene['halfspaces']:
        v = max(v, _halfspace_violation(p, hs, margin))
    for dk in scene['disks']:
        v = max(v, _disk_violation(p, dk, margin))
    return v


def uav_clearance(p, scene):
    """Distance from the point to the nearest obstacle SURFACE, before any inflation.

    Negative inside an obstacle. The vehicle of radius r touches that obstacle wherever
    this drops below r, which is what makes it the right quantity for choosing where to
    draw the vehicle: the tightest moment of the route, whether or not it violates.
    """
    best = float('inf')
    x, y = p
    for hs in scene['halfspaces']:
        xa = hs.get('x_active')
        if xa and not (xa[0] <= x <= xa[1]):
            continue
        (ax, ay), (bx, by) = hs['p0'], hs['p1']
        dx, dy = bx - ax, by - ay
        n = math.hypot(dx, dy)
        nx, ny = -dy / n, dx / n
        signed = (x - ax) * nx + (y - ay) * ny
        if hs['side'] == 'below':
            signed = -signed
        best = min(best, signed)
    for dk in scene['disks']:
        best = min(best, math.hypot(x - dk['c'][0], y - dk['c'][1]) - dk['r'])
    return best


# ------------------------------------------------------------------ quadrotor --
def uav_reference_paths():
    sys.path.insert(0, os.path.join(S.REPO, 'uav_expert_data_collect'))
    import trajectories as T

    r = S.UAV_CONSTRAINTS['r_drone']
    tight = r + S.UAV_CONSTRAINTS['tightening']
    scenes = {s['name']: s for s in S.UAV_CONSTRAINTS['scenes']}

    out = []
    for scene_name, routes in UAV_ROUTES.items():
        sc = scenes[scene_name]
        drawn = []
        for label, fn_name, args in routes:
            fn = getattr(T, fn_name)(*args)
            dur = args[-1]
            xy = []
            for t in np.linspace(0.0, dur, N_SAMPLES):
                p = fn(t)
                p = p[0] if isinstance(p, tuple) else p
                xy.append((float(p[0]), float(p[1])))
            viol = [uav_violation(p, sc, r) for p in xy]
            # Where the figure draws the vehicle to scale: the tightest moment of the route,
            # least clearance to any obstacle SURFACE, inflation aside. Two refinements, both
            # because the raw argmin picked uninteresting points:
            #   * ties are broken toward the obstacle. On UAV-s-curve the whole straight run
            #     down corridor 1 ties at 0.45 m, so the raw argmin landed on the very first
            #     sample, at the arena edge, instead of on the corner where the route leaves
            #     the passage -- which is the moment the reader wants to see.
            #   * the full clearance ORDER is kept, so the builder can move a vehicle to its
            #     next-tightest moment when two would be drawn on top of each other.
            clear = [uav_clearance(p, sc) for p in xy]
            near = [min([math.hypot(p[0] - dk['c'][0], p[1] - dk['c'][1])
                         for dk in sc['disks']] or [0.0]) for p in xy]
            order = sorted(range(len(xy)), key=lambda i: (round(clear[i], 3), near[i]))
            worst = order[0]
            viol_t = [uav_violation(p, sc, tight) for p in xy]
            drawn.append({
                'label': label,
                'xy': [[round(x, NDP), round(y, NDP)] for x, y in xy],
                'n_violating': int(sum(v > 0 for v in viol)),
                'n_violating_tightened': int(sum(v > 0 for v in viol_t)),
                'max_depth': round(max(viol), NDP),
                'max_depth_tightened': round(max(viol_t), NDP),
                'worst_xy': [round(xy[worst][0], NDP), round(xy[worst][1], NDP)],
                # candidate points in increasing clearance, for the de-cluttering above
                'tight_order': [[round(xy[i][0], NDP), round(xy[i][1], NDP),
                                 round(max(0.0, r - clear[i]), NDP)] for i in order[:120]],
                'worst_clearance': round(clear[worst], NDP),
                'worst_overlap': round(max(0.0, r - clear[worst]), NDP),
                'clean': bool(max(viol) <= 0.0),
                'clean_tightened': bool(max(viol_t) <= 0.0),
            })
            print(f'  {scene_name:13s} {label:9s} '
                  f'{"clean" if drawn[-1]["clean"] else "VIOLATES":9s} '
                  f'tightened {"clean" if drawn[-1]["clean_tightened"] else "VIOLATES"}')
        out.append({'scene': scene_name, 'routes': drawn,
                    'n_clean': sum(d['clean'] for d in drawn),
                    'n_clean_tightened': sum(d['clean_tightened'] for d in drawn),
                    'n_routes': len(drawn)})
    return out


# ------------------------------------------------------------------ alignment --
def _segment_hits_disk(p0, p1, c, r):
    """Does the straight segment p0->p1 come within r of c?"""
    p0, p1, c = np.asarray(p0, float), np.asarray(p1, float), np.asarray(c, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = 0.0 if L2 == 0 else float(np.clip((c - p0) @ d / L2, 0.0, 1.0))
    return float(np.linalg.norm(p0 + t * d - c)) <= r


def aligning_contexts():
    C = S.ALIGNING_CONSTRAINTS
    (ax, ay), (bx, by) = C['halfspace']['p0'], C['halfspace']['p1']
    m = (by - ay) / (bx - ax)
    b = ay - m * ax
    cx, cy = C['disk']['c']
    r, t = C['disk']['r'], C['tightening']
    # the tightened halfspace boundary, moved toward the allowed side by t (as the builder draws it)
    shift = -t * (1 + m * m) ** 0.5

    base = os.path.join(S.REPO, 'd3il', 'environments', 'dataset', 'data', 'aligning')
    out = []
    for split in ('train', 'test'):
        path = os.path.join(base, f'{split}_contexts.pkl')
        if not os.path.isfile(path):
            continue
        rows = []
        for ctx in pickle.load(open(path, 'rb')):
            box, _bq, tgt, _tq = ctx
            bpos = (float(box[0]), float(box[1]))
            tpos = (float(tgt[0]), float(tgt[1]))
            rows.append({
                'box': [round(v, NDP) for v in bpos],
                'box_yaw_deg': round(float(box[2]), 2),
                'target': [round(v, NDP) for v in tpos],
                'target_yaw_deg': round(float(tgt[2]), 2),
                # the shortest push a demonstration could make
                'push_hits_disk': _segment_hits_disk(bpos, tpos, (cx, cy), r),
                'push_hits_disk_tightened': _segment_hits_disk(bpos, tpos, (cx, cy), r + t),
                # excluded side of the halfspace (above the line)
                'box_excluded': bool(bpos[1] > m * bpos[0] + b),
                'target_excluded': bool(tpos[1] > m * tpos[0] + b),
                # v3.100 (audit F13.1, author): the push tested against the halfspace too. Its excluded
                # side is a halfplane, convex, so a segment enters it iff one of its ends lies in it.
                'push_hits_halfspace': bool(bpos[1] > m * bpos[0] + b or tpos[1] > m * tpos[0] + b),
                'push_hits_halfspace_tightened': bool(bpos[1] > m * bpos[0] + b + shift
                                                      or tpos[1] > m * tpos[0] + b + shift),
            })
        n = len(rows)
        print(f'  aligning {split:5s} {n} contexts · direct push crosses the keep-out region in '
              f'{sum(r_["push_hits_disk"] for r_ in rows)} of them '
              f'({sum(r_["push_hits_disk_tightened"] for r_ in rows)} tightened); the halfspace in '
              f'{sum(r_["push_hits_halfspace"] for r_ in rows)} '
              f'({sum(r_["push_hits_halfspace_tightened"] for r_ in rows)} tightened)')
        out.append({'split': split, 'n': n, 'contexts': rows,
                    'n_push_hits': sum(r_['push_hits_disk'] for r_ in rows),
                    'n_push_hits_tightened': sum(r_['push_hits_disk_tightened'] for r_ in rows),
                    'n_push_hits_halfspace': sum(r_['push_hits_halfspace'] for r_ in rows),
                    'n_push_hits_halfspace_tightened': sum(r_['push_hits_halfspace_tightened'] for r_ in rows)})
    return out


def main():
    print('quadrotor reference paths:')
    uav = uav_reference_paths()
    print('alignment contexts:')
    align = aligning_contexts()
    doc = {
        'generated': __import__('datetime').date.today().isoformat(),
        'note': ('Quadrotor: the demonstration generator\'s own reference path per homotopy '
                 'class, tested against the inflated and tightened constraint set. Alignment: '
                 'the recorded contexts, and whether the direct push from box to target crosses '
                 'the keep-out region or the halfspace. Neither is a flown demonstration; those are on the cluster.'),
        'uav': uav,
        'aligning': align,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    print(f'\nwrote {os.path.relpath(OUT, S.REPO)}  ({os.path.getsize(OUT) // 1024} KiB)')


if __name__ == '__main__':
    main()
