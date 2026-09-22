#!/usr/bin/env python3
"""Data-analysis figure builders for the obstacle-avoidance environment.

Part of Data_Analysis/DA_in_Paper/plotting -- the official figure pipeline of the
thesis. One function per thesis figure.

Every builder has the same shape and the same contract:

    def fig_<name>(outdir) -> (path, provenance_string) | None

* It takes its data from ``sources.CORPORA`` and nothing else -- no path is
  written here, so new data is absorbed by editing ``sources.py`` alone.
* It returns ``None`` when its corpus is absent, so a partial checkout builds
  the figures it can rather than failing.
* Its subtitle names the protocol. TARGET section 6 rule 7: never a bare *n*.
* Its labels use thesis names (``Auxiliary/Naming/NAMING_20260910_master_table.md``),
  never code tokens. Panel tags inside a plot are defined in the caption.

To add a figure: write the builder, add it to ``ALL`` at the bottom, run
``make_figs.py`` (it writes into ``../figures/da/``), then
``export_to_draft.py`` for the draft that uses it.
"""
import os
import statistics as st

import sources as S
from svg.fmpcc_svg import Fig, dec_ticks, fmt_num, legend

# The pinned baseline, fixed once for the whole thesis and never renegotiated
# per figure (TARGET section 6 rule 1): the diffusion engine at K = 20,
# action weight 10, at its own best projection variant.
TARGET = ('diffusion', 20, 'dpcc-c-tightened')

RULES = ['dpcc-c-tightened', 'dpcc-t-tightened']
RULE_MARK = {'dpcc-c-tightened': 'o', 'dpcc-t-tightened': 's'}
RULE_SHORT = {'dpcc-c-tightened': 'c', 'dpcc-t-tightened': 't'}
RULE_NAME = {'dpcc-c-tightened': 'cumulative projection cost, tightened',
             'dpcc-t-tightened': 'temporal consistency, tightened'}
KS = [1, 2, 5, 10, 20]
PARETO_BAND = 0.05


def _avoiding_cells_dpcc(engines=('mf', 'af', 'fm', 'diffusion')):
    """The DPCC protocol -- 5 training seeds x 2 episodes -- reduced to geometry-means.

    Since v3.55 this is what the budget figures of section 6.1 are drawn at. Three reasons,
    all of them properties of the data rather than of the argument:

      * it is the corpus the result TABLES are computed from, so a figure and the table
        beside it are the same evaluation;
      * every cell of it carries five seeds and three geometries, which the 20-episode
        campaign does not at K = 20;
      * the consistency-interpolated model has five seeds here and one there, so it can
        appear as a peer of the other three instead of being drawn apart.

    -> (corpus, {'AGG': rows, geometry: rows}) with rows keyed (engine, K, rule).
    """
    c = S.CORPORA['avoiding_dpcc']
    if not c.available:
        return None, None
    cells = {}
    for eng in engines:
        cells.update(S.load_exact(c, S.AVOIDING_DPCC_FOLDERS[eng], eng,
                                  backbone=S.AVOIDING_DPCC_BACKBONE[eng]))
    agg = S.geometry_mean(cells, lambda k: (k[0], k[1], k[4]), geom_index=3)
    per_geom = {g: S.geometry_mean(cells, lambda k, g=g: (k[0], k[1], k[4]) if k[3] == g else None,
                                   geom_index=3, geometries=[g])
                for g in S.GEOMETRIES}
    return c, {'AGG': agg, **per_geom}


def _avoiding_cells(engines=('diffusion', 'fm', 'mf')):
    """Load the Tier-2 corpus and reduce it to geometry-means per (engine, K, rule).

    ``engines`` defaults to the three architecture-matched U-Net rows. The
    bootstrapped-target row in this corpus is SiT and is a confounded comparison,
    so it is opt-in rather than default.
    """
    c = S.CORPORA['avoiding_t2']
    if not c.available:
        return None, None
    cells = S.load_avoiding(c, {k: v for k, v in S.AVOIDING_T2_FOLDERS.items() if k in engines})
    agg = S.geometry_mean(cells, lambda k: (k[0], k[1], k[4]), geom_index=3)
    per_geom = {}
    for g in S.GEOMETRIES:
        per_geom[g] = S.geometry_mean(cells, lambda k, g=g: (k[0], k[1], k[4]) if k[3] == g else None,
                                      geom_index=3, geometries=[g])
    return c, {'AGG': agg, **per_geom}


# ═══════════════════════════════════════════════════════════════════════════
#  Trade-off between control steps and time per step, per geometry (2 x 2 grid)
# ═══════════════════════════════════════════════════════════════════════════
# v3.55: four models, not three. At the DPCC protocol the consistency-interpolated model
# has all five training seeds, so it is a peer of the other three and no longer has to be
# left out of the frontier or drawn apart in the ladder.
MODELS = ('mf', 'af', 'fm', 'diffusion')
FONT = 1.5          # trade-off grid, printed at \textwidth
FONT_ENV = 1.75     # three-panel constraint figure, printed at \textwidth
DEMO_CLEAN = '#1e8449'      # a demonstration that satisfies the geometry it is drawn on
DEMO_VIOLATING = '#c0392b'  # one that crosses it -- same two colours as the path figures


Y_BREAK = 80          # control steps; the axis is compressed above this value
Y_LOWER_SHARE = 0.74  # share of the panel height given to the range below the break


def _broken(ylo, yhi):
    """Value -> virtual axis value for a y axis broken at Y_BREAK.

    Below the break the map is the identity; above it the slope is chosen so that
    [ylo, Y_BREAK] takes Y_LOWER_SHARE of the height and [Y_BREAK, yhi] the rest. When
    nothing lies above the break the map is the identity throughout.
    """
    if yhi <= Y_BREAK:
        return lambda y: y
    s = ((1.0 - Y_LOWER_SHARE) / Y_LOWER_SHARE) * (Y_BREAK - ylo) / (yhi - Y_BREAK)
    return lambda y: y if y <= Y_BREAK else Y_BREAK + (y - Y_BREAK) * s


def _break_marks(f, y):
    """Two short slanted strokes on each spine, the conventional sign of a broken axis,
    and a faint dashed line across the panel at the same height."""
    k = f.font or 1.0
    d, g = 5.0 * k, 2.6 * k
    f.s.append(f'<line x1="{f.L}" y1="{y:.1f}" x2="{f.R}" y2="{y:.1f}" stroke="#bbb" '
               f'stroke-width="1" stroke-dasharray="2,4"/>')
    for x in (f.L, f.R):
        for dy in (-g, g):
            f.s.append(f'<line x1="{x - d:.1f}" y1="{y + dy + d * 0.6:.1f}" x2="{x + d:.1f}" '
                       f'y2="{y + dy - d * 0.6:.1f}" stroke="#222" stroke-width="{1.6 * k:.1f}"/>')


def _tradeoff_panel(rows, title, sub, ylab=True):
    """One panel: time per control step (log x) against control steps (y).

    A point is drawn filled when its success with constraint satisfaction is
    within PARETO_BAND of the best on the panel; among those, the dashed line and
    rings mark the configurations for which no other filled point has both fewer
    control steps and less time per step. Each point is labelled with its K.
    """
    pts = []
    for (eng, K, rule), r in rows.items():
        if rule in RULES and eng in MODELS:
            pts.append(dict(engine=eng, K=K, rule=rule, avg_time=r['avg_time'], n_steps=r['n_steps'],
                            n_success_and_constraints=r['n_success_and_constraints']))
    if not pts:
        return None
    pts, front = S.pareto_front(pts, band=PARETO_BAND)
    xlo, xhi = min(p['avg_time'] for p in pts) * 0.6, max(p['avg_time'] for p in pts) * 1.7
    ylo, yhi = min(p['n_steps'] for p in pts) - 3, max(p['n_steps'] for p in pts) + 8
    # v3.62 (author): the step axis is BROKEN at Y_BREAK. Every configuration that solves
    # the task sits between the high fifties and the high seventies; only the two hollow
    # K=2 points and one diffusion cell lie above, and on a plain axis they pushed the
    # cluster into a third of the panel (a log axis was tried first and changed nothing,
    # the range being under a factor of two). Below the break the axis is linear and
    # takes Y_LOWER_SHARE of the height; above it the axis is linear again but
    # compressed. The break is marked on both spines and named in the caption.
    v = _broken(ylo, yhi)
    f = Fig(540, 424, ml=86, mr=18, mt=48, mb=72, font=FONT)   # mt: geometry heading only, no subtitle
    f.axes((xlo, xhi), (ylo, v(yhi)), xlog=True)
    lower = [t for t in range(0, 300, 5) if ylo <= t <= Y_BREAK]
    upper = [t for t in range(0, 300, 10) if Y_BREAK < t <= yhi]
    labels = {v(t): f'{t:.0f}' for t in lower + upper}
    f.frame(dec_ticks(xlo, xhi), sorted(labels),
            'time per control step [s] (log)', 'control steps' if ylab else '', title, sub,
            xfmt=fmt_num, yfmt=lambda t: labels[t])
    _break_marks(f, f.Y(v(Y_BREAK)))
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q['avg_time']), f.Y(v(q['n_steps']))))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['avg_time']), f.Y(v(q['n_steps']))))
        f.poly(st, '#222', dash='6,4', w=1.6)
    for q in front:
        f.ring(f.X(q['avg_time']), f.Y(v(q['n_steps'])), r=13)
    for q in pts:
        x, y = f.X(q['avg_time']), f.Y(v(q['n_steps']))
        f.marker(x, y, RULE_MARK[q['rule']], S.ENGINE_COLOUR_DISTINCT[q['engine']], filled=q['eligible'], r=6.5, ew=1.6)
        # label above-right for one selection rule and below-right for the other, so the
        # two rules of one model at the same budget do not print on top of each other
        ly = y - 9 if q['rule'] == 'dpcc-c-tightened' else y + 19
        f.text(x + 9, ly, q['K'], 11, '#111' if q['eligible'] else '#999')
    return f


def _legend_strip(width, protocol):
    h = Fig(width, 44, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 20
    for eng in MODELS:
        h.marker(x, 22, 'o', S.ENGINE_COLOUR_DISTINCT[eng], r=6.5)
        h.text(x + 14, 26, S.ENGINE_LABEL[eng], 11, '#111')
        x += 32 + len(S.ENGINE_LABEL[eng]) * 10.5
    x += 20
    for rule, lab in (('dpcc-c-tightened', 'cumulative projection cost'), ('dpcc-t-tightened', 'temporal consistency')):
        h.marker(x, 22, RULE_MARK[rule], '#777', r=6.5)
        h.text(x + 14, 26, lab, 11, '#111')
        x += 28 + len(lab) * 9.0
    # v3.62: the boxed direction key is drawn ONCE, here, instead of inside each panel,
    # where its box cost a quarter of the height (the axis is broken instead, see
    # _tradeoff_panel). Same drawing as the frontier module's boxed key, at legend size.
    x += 10
    col = '#34495e'
    h.s.append(f'<rect x="{x - 4}" y="4" width="34" height="36" rx="3" fill="#fff" '
               f'stroke="{col}" stroke-width="1.2"/>')
    cx, cy, L = x + 13, 22, 9
    h.s.append(f'<line x1="{cx + L}" y1="{cy - L}" x2="{cx - L}" y2="{cy + L}" stroke="{col}" '
               'stroke-width="2.2" stroke-linecap="round"/>')
    for hx, hy in ((cx - L + 8, cy + L), (cx - L, cy + L - 8)):
        h.s.append(f'<line x1="{cx - L}" y1="{cy + L}" x2="{hx}" y2="{hy}" stroke="{col}" '
                   'stroke-width="2.2" stroke-linecap="round"/>')
    h.text(x + 38, 26, 'better', 11, col, bold=True)
    return h


def fig_avoiding_tradeoff(outdir):
    c, data = _avoiding_cells_dpcc()
    if c is None:
        return None
    panels = [
        # Panel headings name the geometry and nothing else. How the panels are averaged and
        # which of them carries a single-seed baseline are protocol, and protocol belongs in the
        # caption, not on the page (author, v3.49).
        _tradeoff_panel(data['AGG'], 'All three geometries', '', ylab=True),
        _tradeoff_panel(data['top-left-hard'], 'top-left-hard', '', ylab=False),
        _tradeoff_panel(data['top-right-hard'], 'top-right-hard', '', ylab=True),
        _tradeoff_panel(data['both-hard'], 'both-hard', '', ylab=False),
    ]
    if any(p is None for p in panels):
        return None
    from svg.fmpcc_svg import save_grid
    width = 2 * 540 + 10
    header = _legend_strip(width, 'tightened constraints · temporal U-Net, 4.0 M')
    path = save_grid(panels, os.path.join(outdir, 'fig_avoiding_tradeoff.svg'), cols=2, gap=10, header=header)
    return path, f'{c.rel} | {c.protocol}'


# ═══════════════════════════════════════════════════════════════════════════
#  The environment: scene with demonstrations, and the three constraint geometries
# ═══════════════════════════════════════════════════════════════════════════
def _scene():
    import json
    if not os.path.isfile(S.AVOIDING_SCENE):
        return None
    with open(S.AVOIDING_SCENE) as fh:
        return json.load(fh)


def _scene_panel(sc, w, title, sub, ylab, font):
    (x0, x1), (y0, y1) = sc['ax_limits']
    # No heading -> no room reserved for one. A panel that stands alone in a figure is named by
    # its caption, not by text inside the drawing (Writing_Hints/HINT_20260920_figure_and_caption_style.md).
    mt = int((45 if (title or sub) else 14) * font)
    ml, mr, mb = int(52 * font), int(12 * font), int(48 * font)
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)                 # equal aspect: 1 m is the same length on both axes
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=font)
    f.axes((x0, x1), (y0, y1))
    ticks_x = [round(x0 + 0.1 * i, 1) for i in range(int(round((x1 - x0) / 0.1)) + 1)]
    ticks_y = [round(y0 + 0.1 * i, 1) for i in range(int(round((y1 - y0) / 0.1)) + 1)]
    f.frame(ticks_x, ticks_y, 'x [m]', 'y [m]' if ylab else '', title, sub,
            xfmt=lambda v: f'{v:.1f}', yfmt=lambda v: f'{v:.1f}')
    return f


def _demo_satisfies(xy, g, obstacle_radius):
    """Does this demonstration satisfy geometry `g` at every recorded step?

    The same test extract/avoiding_scene.py applies when it writes
    `demonstrations_satisfying` -- repeated here, from the numbers already in the JSON,
    so that the figure can colour a single demonstration rather than only count them.
    fig_constraints_avoiding asserts the two agree, so this cannot drift from the count
    tab:avoiding-geometries prints.
    """
    for hs in g['halfspaces']:
        (ax_, ay), (bx, by) = hs['p0'], hs['p1']
        m = (by - ay) / (bx - ax_)
        b = ay - m * ax_
        below = hs['feasible_side'] == 'below'
        for x, y in xy:
            line = m * x + b
            if (y > line) if below else (y < line):
                return False
    cx, cy = g['disk']['center']
    r = g['disk']['radius'] + obstacle_radius
    for x, y in xy:
        if ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 < r:
            return False
    return True


def _draw_common(f, sc, demo_colour, demo_opacity, per_demo=None):
    """`per_demo`, when given, is one (colour, opacity, width) per demonstration."""
    f.clip_to_box()
    for i, xy in enumerate(sc['demonstrations']):
        if per_demo is None:
            f.dline(xy, demo_colour, w=1.0, opacity=demo_opacity)
        else:
            col, op, w = per_demo[i]
            f.dline(xy, col, w=w, opacity=op)
    (x0, x1), _ = sc['ax_limits']
    f.dline([(x0, sc['goal_y']), (x1, sc['goal_y'])], '#27ae60', w=4.5)
    for cx, cy in sc['obstacles']['centers']:
        f.circle(cx, cy, sc['obstacles']['radius'], fill='#c0392b', stroke='#7b241c', w=1.0)
    f.end_clip()


def fig_env_avoiding(outdir):
    sc = _scene()
    if sc is None:
        return None
    n = sc['n_demonstrations']
    f = _scene_panel(sc, 560, '', '', True, FONT)
    _draw_common(f, sc, '#2471a3', 0.35)
    for xy in sc['demonstrations']:
        f.circle(xy[0][0], xy[0][1], 0.004, fill='#1b4f72')
    (x0, x1), _ = sc['ax_limits']
    f.text(f.X(x1) - 6, f.Y(sc['goal_y']) - 10, 'goal line', 11, '#1e8449', anchor='end', bold=True)
    sx = sum(xy[0][0] for xy in sc['demonstrations']) / n
    sy = sum(xy[0][1] for xy in sc['demonstrations']) / n
    f.text(f.X(sx) + 12, f.Y(sy) + 4, 'start', 11, '#1b4f72', bold=True)
    path = f.save(os.path.join(outdir, 'fig_env_avoiding.svg'))
    return path, f'Data_Analysis/DA_in_Paper/data/avoiding_scene.json | {n} D3IL demonstrations (measured end-effector position)'


def geometry_panel(sc, name, w, title, sub, ylab, font):
    """An empty panel of one D3IL-avoiding geometry: excluded region, tightened boundary,
    keep-out disk, obstacles and goal line -- everything except the demonstrations.

    Factored out of fig_constraints_avoiding (v3.50) so that the executed-path figure of
    Chapter 6 draws its constraint set with the SAME code from the SAME scene file. That is
    the guarantee the quadrotor path figures already have through
    scenes._uav_constraint_panel: what lies under a path is the geometry the projection saw.
    Obstacles and the goal line are drawn here; the caller adds its own lines and then calls
    `close_geometry_panel`.
    """
    from svg.fmpcc_svg import clip_halfplane
    (x0, x1), (y0, y1) = sc['ax_limits']
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    d = sc['tightening']
    g = sc['geometries'][name]
    f = _scene_panel(sc, w, title, sub, ylab, font)
    f.clip_to_box()
    for hs in g['halfspaces']:
        (ax_, ay), (bx, by) = hs['p0'], hs['p1']
        m = (by - ay) / (bx - ax_)
        b = ay - m * ax_
        below = hs['feasible_side'] == 'below'
        keep = (lambda x, y, m=m, b=b: y - (m * x + b)) if below else (lambda x, y, m=m, b=b: (m * x + b) - y)
        f.polygon(clip_halfplane(box, keep), '#5d6d7e', opacity=0.32)
        shift = d * (1 + m * m) ** 0.5 * (-1 if below else 1)
        xs = (x0 - 0.1, x1 + 0.1)
        f.dline([(x, m * x + b) for x in xs], '#34495e', w=1.8)
        f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.6, dash='7,5')
    disk = g['disk']
    f.circle(disk['center'][0], disk['center'][1], disk['radius'], fill='#5d6d7e', opacity=0.32,
             stroke='#34495e', w=1.6)
    f.circle(disk['center'][0], disk['center'][1], disk['radius'] + d, stroke='#34495e', w=1.6, dash='7,5')
    f.dline([(x0, sc['goal_y']), (x1, sc['goal_y'])], '#27ae60', w=4.5)
    for cx, cy in sc['obstacles']['centers']:
        f.circle(cx, cy, sc['obstacles']['radius'], fill='#c0392b', stroke='#7b241c', w=1.0)
    return f


def fig_constraints_avoiding(outdir):
    from svg.fmpcc_svg import clip_halfplane, save_grid
    sc = _scene()
    if sc is None:
        return None
    (x0, x1), (y0, y1) = sc['ax_limits']
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    d = sc['tightening']
    n = sc['n_demonstrations']
    panels = []
    for i, name in enumerate(('top-left-hard', 'top-right-hard', 'both-hard')):
        g = sc['geometries'][name]
        f = _scene_panel(sc, 470, name, f"{g['demonstrations_satisfying']} of {n} demonstrations satisfy it",
                         i == 0, FONT_ENV)
        f.clip_to_box()
        for hs in g['halfspaces']:
            (ax_, ay), (bx, by) = hs['p0'], hs['p1']
            m = (by - ay) / (bx - ax_)
            b = ay - m * ax_
            below = hs['feasible_side'] == 'below'
            # excluded side: y > m x + b when the plan must stay below the line
            keep = (lambda x, y, m=m, b=b: y - (m * x + b)) if below else (lambda x, y, m=m, b=b: (m * x + b) - y)
            f.polygon(clip_halfplane(box, keep), '#5d6d7e', opacity=0.32)
            shift = d * (1 + m * m) ** 0.5 * (-1 if below else 1)
            xs = (x0 - 0.1, x1 + 0.1)
            f.dline([(x, m * x + b) for x in xs], '#34495e', w=1.8)
            f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.6, dash='7,5')
        disk = g['disk']
        f.circle(disk['center'][0], disk['center'][1], disk['radius'], fill='#5d6d7e', opacity=0.32,
                 stroke='#34495e', w=1.6)
        f.circle(disk['center'][0], disk['center'][1], disk['radius'] + d, stroke='#34495e', w=1.6, dash='7,5')
        f.end_clip()
        # Colour carries the one statement this figure exists to make: which recorded
        # demonstrations satisfy this geometry and which cross it (author, v3.50). Both are
        # drawn translucent so the shaded constraint region stays readable underneath; the
        # satisfying ones are given a little more weight because in two panels there is one
        # of them against ninety-five.
        ok = [_demo_satisfies(xy, g, sc['obstacles']['radius']) for xy in sc['demonstrations']]
        assert sum(ok) == g['demonstrations_satisfying'], (
            f"{name}: coloured {sum(ok)} satisfying demonstrations but the scene file counts "
            f"{g['demonstrations_satisfying']}")
        _draw_common(f, sc, None, None,
                     per_demo=[(DEMO_CLEAN, 0.80, 1.9) if v else (DEMO_VIOLATING, 0.24, 1.0)
                               for v in ok])
        panels.append(f)
    width = sum(p.w for p in panels) + 2 * 8
    # Two rows: at FONT_ENV six entries on one line run past the right edge of the figure.
    h = Fig(width, 92, ml=0, mr=0, mt=0, mb=0, font=FONT_ENV)
    rows = [[('area', 'excluded by the constraints'), ('dash', f'tightened by {d:g} m'),
             ('obst', 'obstacle'), ('goal', 'goal line')],
            [('demo_bad', 'demonstration that crosses this geometry'),
             ('demo_ok', 'demonstration that satisfies it')]]
    for r, items in enumerate(rows):
        x, y = 16, 24 + r * 40
        for kind, lab in items:
            if kind == 'area':
                h.s.append(f'<rect x="{x - 8}" y="{y - 8}" width="16" height="16" fill="#5d6d7e" fill-opacity="0.32" stroke="#34495e"/>')
            elif kind == 'dash':
                h.poly([(x - 10, y), (x + 10, y)], '#34495e', dash='7,5', w=2)
            elif kind == 'obst':
                h.marker(x, y, 'o', '#c0392b', r=6.5)
            elif kind == 'goal':
                h.poly([(x - 10, y), (x + 10, y)], '#27ae60', w=5)
            elif kind == 'demo_bad':
                h.s.append(f'<line x1="{x - 10}" y1="{y}" x2="{x + 10}" y2="{y}" stroke="{DEMO_VIOLATING}" '
                           f'stroke-width="2.6" stroke-opacity="0.55"/>')
            else:
                h.s.append(f'<line x1="{x - 10}" y1="{y}" x2="{x + 10}" y2="{y}" stroke="{DEMO_CLEAN}" '
                           f'stroke-width="2.6" stroke-opacity="0.9"/>')
            h.text(x + 20, y + 7, lab, 11, '#111')
            x += 60 + len(lab) * 12.0
    path = save_grid(panels, os.path.join(outdir, 'fig_constraints_avoiding.svg'), cols=3, gap=8, header=h)
    return path, f'Data_Analysis/DA_in_Paper/data/avoiding_scene.json | config/projection_eval.yaml geometries; {n} D3IL demonstrations'

# ═══════════════════════════════════════════════════════════════════════════
#  Fig 5 -- the step-budget ladder. THE figure the engine chapter turns on.
# ═══════════════════════════════════════════════════════════════════════════
def fig_avoiding_k_ladder(outdir):
    """Quality against step budget: who can walk K down and who cannot.

    Two vertical axes would be a lie here, so quality is on its own panel: the point is
    not that the flow-based engines are cheaper (the frontier shows that), it is that
    their quality is FLAT in K while the diffusion engine's collapses below its training
    budget. K is inference-time for one family and training-time for the other.

    v3.55: drawn at the DPCC protocol, from the corpus the tables are computed from. Every
    series is then the same sample -- five training seeds, two episodes, three geometries --
    so the dashed-hollow baseline series and the dotted single-seed CI-MeanFM series that
    earlier versions needed are gone. A marker is hollow only where a cell is short of a
    geometry, which at this protocol never happens.
    """
    c, data = _avoiding_cells_dpcc()
    if c is None:
        return None
    rows = data['AGG']
    f = Fig(760, 452, ml=76, mr=152, mt=26)   # mt: no title line
    f.axes((0.8, 26), (0.55, 1.04), xlog=True)
    f.frame([1, 2, 5, 10, 20], [0.6, 0.7, 0.8, 0.9, 1.0],
            'step budget K  [ network evaluations per plan ]   (log)',
            'success and constraint satisfaction',
            '', '',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.2f}')

    # CI-MeanFM and FM are both at 1.000 at both of their budgets, so one line lies exactly
    # on the other. Nothing is moved to fix that -- the coincidence is the result. Instead
    # CI-MeanFM is drawn LAST, dashed, with a small marker that sits inside FM's, so both
    # series are legible at a point they genuinely share.
    MARK = {'mf': 's', 'af': '^', 'fm': 'o', 'diffusion': 's'}
    DASH = {'af': '7,4'}
    RAD = {'af': 3.2}
    DRAW_ORDER = ('mf', 'fm', 'diffusion', 'af')
    drawn = []
    for eng in DRAW_ORDER:
        rule = S.AVOIDING_DPCC_RULE[eng]
        pts = [(K, rows[(eng, K, rule)]) for K in KS if (eng, K, rule) in rows]
        if not pts:
            continue
        if len(pts) > 1:
            f.poly([(f.X(K), f.Y(r['n_success_and_constraints'])) for K, r in pts],
                   S.ENGINE_COLOUR_DISTINCT[eng], w=2.0, dash=DASH.get(eng, ''))
        for K, r in pts:
            f.marker(f.X(K), f.Y(r['n_success_and_constraints']), MARK[eng],
                     S.ENGINE_COLOUR_DISTINCT[eng], filled=(r['n_geometries'] == 3),
                     r=RAD.get(eng, 5.0))
        drawn.append(eng)

    lx = f.R + 14
    f.text(lx, f.T + 4, 'model', 10, '#111', bold=True)
    legend(f, lx, f.T + 20,
           [(S.ENGINE_COLOUR_DISTINCT[k], S.ENGINE_LABEL[k], MARK[k])
            for k in MODELS if k in drawn])
    path = f.save(os.path.join(outdir, 'fig_avoiding_k_ladder.svg'))
    return path, f'{c.rel} | {c.protocol}'


# ═══════════════════════════════════════════════════════════════════════════
#  Fig 6 -- the two constraint arms, and where their costs cross
# ═══════════════════════════════════════════════════════════════════════════
def fig_avoiding_projector_cost(outdir):
    """Per-step cost of per-step projection vs. endpoint projection against K.

    The shaded band is the budget at which the comparison is not well posed: at
    K = 2 the endpoint method has ONE guiding step, so it is barely running the
    mechanism it names. Greying it is the figure's job, not the caption's.

    Note which series the hollow marker belongs to. Degeneracy is a property of
    ENDPOINT projection only -- per-step projection acts on every step and is
    well posed at every budget -- so only the endpoint series is drawn hollow
    there. An earlier version drew both hollow and implied a limitation that
    per-step projection does not have.
    """
    c = S.CORPORA['avoiding_minK']
    if not c.available:
        return None
    cells = S.load_by_candidate(c, ['hfmink_A1_mfunet'])
    rows = S.geometry_mean(cells, lambda k: (k[0], k[3]), geom_index=2)  # -> (K, variant)

    # (variant, colour, label, marker, degeneracy applies to this series)
    ARMS = [('dpcc-t-tightened', '#34495e', 'per-step projection', 'o', False),
            ('hardflow_sls-t-tightened', '#5d6d7e', 'endpoint projection', 's', True)]
    ks = sorted({k for k, _v in rows})
    if not ks:
        return None

    # v3.48: printed at \textwidth, so the font is scaled up and everything the caption
    # already says is taken out of the panel -- the title, the seed-and-episode subtitle
    # and the axis annotations that were repo vocabulary ("wall clock", "trials",
    # "rollouts"). What is left is the measurement: the metric of Ch 5 against the budget.
    f = Fig(880, 470, ml=104, mr=210, mt=26, mb=70, font=1.45)
    ymax = max(r['avg_time'] for r in rows.values()) * 1.35
    f.axes((1.6, 6.0), (0.02, ymax), ylog=True)
    f.vspan(f.X(1.6), f.X(2.5))
    f.frame([2, 3, 5], dec_ticks(0.02, ymax),
            'step budget K',
            'time to compute one action [s] (log)',
            '', '',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:g}')
    for var, col, lab, mk, degen in ARMS:
        pts = [(K, rows[(K, var)]['avg_time']) for K in ks if (K, var) in rows]
        if not pts:
            continue
        f.poly([(f.X(K), f.Y(t)) for K, t in pts], col, w=2.4)
        for K, t in pts:
            f.marker(f.X(K), f.Y(t), mk, col, filled=not (degen and K < 3), r=7.0)
            f.text(f.X(K) + 13, f.Y(t) - 10, f'{t * 1000:.0f} ms', 10.5, '#111')

    lx = f.R + 18
    f.text(lx, f.T + 14, 'projection method', 11, '#111', bold=True)
    legend(f, lx, f.T + 40, [(c_, l, m) for _v, c_, l, m, _d in ARMS], dy=26)
    path = f.save(os.path.join(outdir, 'fig_avoiding_projector_cost.svg'))
    return path, f'{c.rel} | {c.protocol}'



# ═══════════════════════════════════════════════════════════════════════════
#  The four generative models with the projection switched OFF.
# ═══════════════════════════════════════════════════════════════════════════
def fig_avoiding_raw_models(outdir):
    """Goal reached without projection, all four models, one network evaluation.

    This is the only figure in which the generative models are compared with
    nothing downstream of them. It exists because the projection and the tracking
    controller absorb so much that the projected numbers cannot separate the
    models (DPCC's own Table 2: a dynamics model wrong by 4x still satisfies the
    constraints in 0.77 of episodes).

    ONE geometry, not three. Without projection the plan does not depend on the
    constraint set: the unprojected rows of top-left-hard and both-hard are equal
    to the decimal and every model reaches the goal in 20 of 20 on both, so a
    three-panel version would show two duplicate panels and one real one. See
    sources.AVOIDING_RAW_GEOMETRY.
    """
    c = S.CORPORA['avoiding_af_unet']
    if not c.available:
        return None
    order = ['af02', 'af05', 'mf', 'fm', 'diffusion']
    rows = {}
    for eng, pat in S.AVOIDING_AF_FOLDERS.items():
        K = 20 if eng == 'diffusion' else 1
        # SEED 6 ONLY, and that is not a detail. The consistency-interpolated
        # folders carry the _s6 tag and exist for seed 6 alone, while the analytic
        # average-velocity, instantaneous-velocity and diffusion folders in this
        # same batch carry all five seeds.
        # Averaging each model over the seeds it happens to have would compare a
        # one-seed number with a five-seed number: it reads 0.97 / 0.97 / 0.92
        # instead of 0.85 / 0.85 / 0.60 and silently flatters the comparators.
        # load_by_candidate takes a LIST of substrings; keys are (K, seed, geometry, variant).
        got = S.load_by_candidate(c, [pat % K], seeds=S.AVOIDING_AF_SEEDS)
        cell = [m for (_K, _seed, geom, variant), m in got.items()
                if variant == 'diffuser' and geom == S.AVOIDING_RAW_GEOMETRY]
        if not cell:
            continue
        rows[eng] = (K, {k: st.mean([m[k] for m in cell if k in m])
                         for k in ('n_success', 'n_steps')})
    rows = {e: rows[e] for e in order if e in rows}
    if len(rows) < 2:
        return None

    f = Fig(900, 560, ml=92, mr=24, mt=78, mb=124, font=1.45)   # v3.37: printed at \linewidth, so the
    # 1.0 scale put 8 pt type on the page at about 5 pt. Everything below is sized against it.
    f.axes((-0.5, len(rows) - 0.5), (0.0, 1.16))
    f.frame([], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], '', 'episodes reaching the goal',
            'Without projection: the plan the model itself produces',
            f'{c.protocol}; {S.AVOIDING_RAW_GEOMETRY}, no projection; U-Net 4.0M',
            yfmt=lambda v: f'{v:.1f}')

    bw = 0.56
    for i, (eng, (K, m)) in enumerate(rows.items()):
        v = m['n_success']
        x0, x1 = f.X(i - bw / 2), f.X(i + bw / 2)
        f.bar(x0, f.Y(v), x1 - x0, f.Y(0.0) - f.Y(v), S.AVOIDING_AF_COLOUR[eng])
        f.text((x0 + x1) / 2, f.Y(v) - 10, f'{v:.2f}', 12.5, '#111', anchor='middle', bold=True)
        f.text((x0 + x1) / 2, f.Y(v) - 27, f"{m['n_steps']:.1f} steps", 10.5, '#666', anchor='middle')
        lab = S.AVOIDING_AF_LABEL[eng]
        for j, part in enumerate(lab.split('\n')):
            f.text((x0 + x1) / 2, f.B + 22 + 19 * j, part, 10.5, '#222', anchor='middle')
        f.text((x0 + x1) / 2, f.B + 24 + 19 * len(lab.split('\n')),
               f'K = {K}', 10.5, '#666', anchor='middle')

    path = f.save(os.path.join(outdir, 'fig_avoiding_raw_models.svg'))
    return path, f'{c.rel} | {c.protocol}, {S.AVOIDING_RAW_GEOMETRY}, no projection'


# Registry. Order is the order they appear in the thesis.
ALL = [
    # (name, group, builder), in the order the figures appear in the thesis
    ('fig_env_avoiding', 'env', fig_env_avoiding),
    ('fig_constraints_avoiding', 'env', fig_constraints_avoiding),
    ('fig_avoiding_tradeoff', 'da', fig_avoiding_tradeoff),
    ('fig_avoiding_k_ladder', 'da', fig_avoiding_k_ladder),
    ('fig_avoiding_raw_models', 'da', fig_avoiding_raw_models),
    ('fig_avoiding_projector_cost', 'da', fig_avoiding_projector_cost),
]
