#!/usr/bin/env python3
"""Flown-path figures: what the vehicle actually did, over the constraint set it flew in.

The three result figures of v3 Chapter 6 that were `\\hole`s until the rollout artefacts
were downloaded. Each is the SAME picture as `scenes.fig_constraints_uav` -- the scene from
above with its inflated and tightened obstacles -- with the executed paths drawn on top and
a mark where each flight ended. Reusing `scenes._uav_constraint_panel` is deliberate: the
constraint set under a path is then guaranteed to be the one the projection saw, because it
is drawn by the same code from the same `sources.UAV_CONSTRAINTS`.

The paths come from `../../data/uav_paths.json`, written by `extract/uav_paths.py` (numpy);
this module, like every builder, is standard library only.

For corridor and pillars, two independent outcomes are drawn:

    colour   collision-free (green) or entered an inflated obstacle (red)
    stroke   solid = passed the goal, dashed = did not
    end mark filled = passed the goal, hollow = did not

For the s-curve controller comparison alone, the author requested the simpler key:
green means success (crossed the finish line), red means no success. Panel subtitles still
print the separate clean counts. Corridor and pillars retain the independent two-channel key.

v3.63 (author): a flight that crashed (`safe` false: contact or loss of altitude) or that
the divergence guard ended (`aborted`) gets a CROSS at its end instead of a dot, in every
figure that has such a flight. The flags come from the extract, not from the drawing.
Wording: the chapter calls a flight that crosses the finish line a *success*, so the
subtitles and legends say success, not passed.
"""
import json
import os

import sources as S
from svg.fmpcc_svg import Fig, save_grid
from .scenes import _uav_constraint_panel, FONT_CONSTRAINT

CLEAN = '#1e8449'        # never entered an inflated obstacle
VIOLATING = '#c0392b'    # entered one
START = '#2471a3'

# Wider than the 470 of fig_constraints_uav: these panels carry an outcome count in the
# subtitle, which is clipped at 470 once a projection variant is named alongside it.
PANEL_W = 560


def _load():
    if not os.path.isfile(S.UAV_PATHS):
        return None
    with open(S.UAV_PATHS) as f:
        return json.load(f)


def _scene(name):
    for scn in S.UAV_CONSTRAINTS['scenes']:
        if scn['name'] == name:
            return scn
    return None


def _panel(scn, panel, first, r_drone, tight, colour_by_pass=False):
    """One constraint panel with its flights drawn over it."""
    eps = panel['episodes']
    n, passed, clean = panel['n'], panel['passed'], panel['clean']
    # The panel's own title and counts replace the scene's, so the reader never has to
    # count paths to learn the outcome. The scene dict is copied, not mutated: it is
    # shared with fig_constraints_uav.
    # Kept short on purpose: at FONT_CONSTRAINT a longer subtitle runs off the panel.
    local = dict(scn, title=panel['title'],
                 sub=f"{panel['sub']} · {passed}/{n} success · {clean}/{n} violation-free")  # v3.79: one word
    f = _uav_constraint_panel(local, PANEL_W, first, r_drone, tight)

    f.clip_to_box()
    for ep in eps:
        xy = ep['xy']
        if len(xy) < 2:
            continue
        colour = CLEAN if (ep['passed'] if colour_by_pass else ep['clean']) else VIOLATING
        pts = [(f.X(x), f.Y(y)) for x, y in xy]
        # [v3.60] A dash at '6,4' is destroyed by superposition: twelve near-coincident
        # not-passed flights fill in each other's gaps and the bundle reads as solid, which
        # is exactly how the projected diffusion panel came to look like a success. A sparse
        # dot pattern on a thinner stroke survives the overlap, and the end mark is enlarged
        # because on this scene it is the only per-flight evidence of where the flight
        # stopped. Colour still reports the constraint, not the goal. (Author, 2026-09-21.)
        missed = not (colour_by_pass or ep['passed'])
        f.poly(pts, colour, w=1.5 if missed else 1.9, dash='2,7' if missed else '')
        if _failed(ep):
            _xmark(f, pts[-1][0], pts[-1][1], colour)
        else:
            f.marker(pts[-1][0], pts[-1][1], 'o', colour,
                     filled=True if colour_by_pass else ep['passed'],
                     r=5.6 if missed else 4.2, ew=2.0 if missed else 1.3)
    # one start mark for the whole panel: every flight of a cell launches from the same pose
    sx, sy = eps[0]['xy'][0] if eps else (None, None)
    if sx is not None:
        f.marker(f.X(sx), f.Y(sy), 's', START, filled=True, r=4.0, ew=1.2)
    f.end_clip()
    return f


def _failed(ep):
    """Crashed, or ended by the divergence guard. Absent flags (older extracts) count as no."""
    return bool(ep.get('aborted')) or ep.get('safe') is False


def _xmark(f, x, y, colour, r=6.0, w=2.4):
    """A cross where a flight ended: the vehicle crashed or lost control there."""
    k = f.font or 1.0
    r, w = r * k, w * k
    for dx, dy in ((-1, -1), (-1, 1)):
        f.s.append(f'<line x1="{x + dx * r:.1f}" y1="{y + dy * r:.1f}" x2="{x - dx * r:.1f}" '
                   f'y2="{y - dy * r:.1f}" stroke="{colour}" stroke-width="{w:.1f}" stroke-linecap="round"/>')


def _legend(width, colour_by_pass=False, failures=False, failure_label='collided or lost control'):
    # v3.79 (author: "keep the words the same across the thesis"): violation = crossing a declared
    # constraint (violation-free, violating steps, S&C); collision = the vehicle hitting an obstacle.
    h = Fig(width, 104, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    items = [('line', CLEAN, '', 'success: crossed the finish line' if colour_by_pass else 'violation-free flight'),
             ('line', VIOLATING, '', 'no success' if colour_by_pass else 'violated a constraint')]
    if not colour_by_pass:
        items.append(('line', '#5d6d7e', '2,7', 'no success (dotted, hollow end)'))
    items.append(('mark', START, '', 'launch pose · flight end'))
    if failures:
        items.append(('x', '#5d6d7e', '', failure_label))
    if len(items) > 4:
        h = Fig(width, 150, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    row_y, col_x, size = (30, 76, 122), (16, width // 2 + 16), 18
    for i, (kind, colour, dash, lab) in enumerate(items):
        x, y = col_x[i % 2], row_y[i // 2]
        if kind == 'line':
            h.poly([(x - 14, y), (x + 14, y)], colour, w=3.0, dash=dash)
        elif kind == 'x':
            _xmark(h, x, y, colour)
        else:
            h.marker(x - 8, y, 's', colour, filled=True, r=4.0, ew=1.2)
            h.marker(x + 8, y, 'o', '#5d6d7e', filled=True, r=4.2, ew=1.3)
        h.text(x + 26, y + 7, lab, size, '#111')
    return h


def _build(key, filename, cols, outdir, colour_by_pass=False):
    D = _load()
    if not D or key not in D['figures']:
        return None
    fig = D['figures'][key]
    panels = [p for p in fig['panels'] if p['n']]
    if not panels:
        return None
    scn = _scene(fig['scene'])
    if not scn:
        return None
    C = S.UAV_CONSTRAINTS
    r, t = C['r_drone'], C['tightening']

    # [U18, 2026-09-23] UAV-pillars panels each name their avoiding geometry: the constraint set drawn under a
    # panel is that geometry's (sources.UAV_PILLARS_GEOMETRIES), not the figure-level scene.
    G = getattr(S, 'UAV_PILLARS_GEOMETRIES', {}) or {}
    drawn = [_panel(G.get(p.get('geometry')) or scn, p, i % cols == 0, r, t, colour_by_pass)
             for i, p in enumerate(panels)]
    width = max(sum(d.w for d in drawn[i:i + cols]) + 8 * (min(cols, len(drawn) - i) - 1)
                for i in range(0, len(drawn), cols))
    failures = any(_failed(e) for q in panels for e in q['episodes'])
    path = save_grid(drawn, os.path.join(outdir, filename), cols=cols, gap=8,
                     header=_legend(width, colour_by_pass, failures,
                                    'collision with a pillar: the flight ends' if key.startswith('pillars')
                                    else 'collided or lost control'))
    flights = sum(p['n'] for p in panels)
    srcs = ', '.join(sorted({p['tag'] for p in panels}))
    ext = ('extract/scurve_r44_paths.py' if key == 'scurve'            # R44, 2026-09-24: FM nfe 1, both controllers
           else 'extract/pillars_v2_paths.py' if key.startswith('pillars') else 'extract/uav_paths.py')
    return path, (f"data/uav_paths.json ({ext}) | {len(panels)} cells, "
                  f"{flights} flights; runs {srcs}; scene and constraints from "
                  f"sources.UAV_CONSTRAINTS, drawn by scenes._uav_constraint_panel")


def fig_uav_scurve_paths(outdir):
    """The s-curve under both controllers (companion of tab:uav-controller): FM, nfe 1, ten flights per panel, the
    unprojected plans (top) and the per-step projected ones (random rule, bottom) -- R44, extract/scurve_r44_paths.py."""
    return _build('scurve', 'fig_uav_scurve_paths.svg', 2, outdir, colour_by_pass=True)


def fig_uav_pillars_paths(outdir):
    """[U18] UAV-pillars flown live by MeanFM at nfe 1: rows = before / after projection (dpcc-t-tightened),
    columns = the three avoiding geometries; ten flights per panel (5 seeds x 2 episodes). A cross marks a
    pillar contact (the plant ends the flight there). Data: extract/pillars_v2_paths.py."""
    return _build('pillars', 'fig_uav_pillars_paths.svg', 3, outdir)


def fig_uav_pillars_paths_cimf(outdir):
    """[U18] The same figure for CI-MeanFM (alpha_end 0.2) at nfe 1."""
    return _build('pillars_cimf', 'fig_uav_pillars_paths_cimf.svg', 3, outdir)


def fig_uav_corridor_paths(outdir):
    """The slide flown across the budget ladder; the baseline is the last row, at its one budget."""
    return _build('corridor', 'fig_uav_corridor_paths.svg', 3, outdir)


ALL = [
    ('fig_uav_scurve_paths', 'da', fig_uav_scurve_paths),
    ('fig_uav_pillars_paths', 'da', fig_uav_pillars_paths),
    ('fig_uav_pillars_paths_cimf', 'da', fig_uav_pillars_paths_cimf),
    ('fig_uav_corridor_paths', 'da', fig_uav_corridor_paths),
]
