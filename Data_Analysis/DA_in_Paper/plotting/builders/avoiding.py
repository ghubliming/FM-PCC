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
MODELS = ('mf', 'fm', 'diffusion')
FONT = 1.5          # trade-off grid, printed at \textwidth
FONT_ENV = 1.75     # three-panel constraint figure, printed at \textwidth


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
    ylo, yhi = min(p['n_steps'] for p in pts) - 4, max(p['n_steps'] for p in pts) + 6
    step = 5 if yhi - ylo <= 45 else 10
    f = Fig(540, 440, ml=86, mr=18, mt=64, mb=72, font=FONT)
    f.axes((xlo, xhi), (ylo, yhi), xlog=True)
    f.frame(dec_ticks(xlo, xhi), [t for t in range(0, 300, step) if ylo <= t <= yhi],
            'time per control step [s] (log)', 'control steps' if ylab else '', title, sub,
            xfmt=fmt_num, yfmt=lambda v: f'{v:.0f}')
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q['avg_time']), f.Y(q['n_steps'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['avg_time']), f.Y(q['n_steps'])))
        f.poly(st, '#222', dash='6,4', w=1.6)
    for q in front:
        f.ring(f.X(q['avg_time']), f.Y(q['n_steps']), r=13)
    for q in pts:
        x, y = f.X(q['avg_time']), f.Y(q['n_steps'])
        f.marker(x, y, RULE_MARK[q['rule']], S.ENGINE_COLOUR[q['engine']], filled=q['eligible'], r=6.5, ew=1.6)
        # label above-right for one selection rule and below-right for the other, so the
        # two rules of one model at the same budget do not print on top of each other
        ly = y - 9 if q['rule'] == 'dpcc-c-tightened' else y + 19
        f.text(x + 9, ly, q['K'], 11, '#111' if q['eligible'] else '#999')
    return f


def _legend_strip(width, protocol):
    h = Fig(width, 44, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 20
    for eng in MODELS:
        h.marker(x, 22, 'o', S.ENGINE_COLOUR[eng], r=6.5)
        h.text(x + 14, 26, S.ENGINE_LABEL[eng], 11, '#111')
        x += 34 + len(S.ENGINE_LABEL[eng]) * 10.5
    x += 20
    for rule, lab in (('dpcc-c-tightened', 'cumulative projection cost'), ('dpcc-t-tightened', 'temporal consistency')):
        h.marker(x, 22, RULE_MARK[rule], '#777', r=6.5)
        h.text(x + 14, 26, lab, 11, '#111')
        x += 34 + len(lab) * 10.5
    return h


def fig_avoiding_tradeoff(outdir):
    c, data = _avoiding_cells()
    if c is None:
        return None
    panels = [
        _tradeoff_panel(data['AGG'], 'All three geometries', 'mean per geometry, then across', ylab=True),
        _tradeoff_panel(data['top-left-hard'], 'top-left-hard', '', ylab=False),
        _tradeoff_panel(data['top-right-hard'], 'top-right-hard', '', ylab=True),
        _tradeoff_panel(data['both-hard'], 'both-hard', 'baseline: training seed 6 only', ylab=False),
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
    ml, mr, mt, mb = int(52 * font), int(12 * font), int(45 * font), int(48 * font)
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)                 # equal aspect: 1 m is the same length on both axes
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=font)
    f.axes((x0, x1), (y0, y1))
    ticks_x = [round(x0 + 0.1 * i, 1) for i in range(int(round((x1 - x0) / 0.1)) + 1)]
    ticks_y = [round(y0 + 0.1 * i, 1) for i in range(int(round((y1 - y0) / 0.1)) + 1)]
    f.frame(ticks_x, ticks_y, 'x [m]', 'y [m]' if ylab else '', title, sub,
            xfmt=lambda v: f'{v:.1f}', yfmt=lambda v: f'{v:.1f}')
    return f


def _draw_common(f, sc, demo_colour, demo_opacity):
    f.clip_to_box()
    for xy in sc['demonstrations']:
        f.dline(xy, demo_colour, w=1.0, opacity=demo_opacity)
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
    f = _scene_panel(sc, 560, 'Obstacle-avoidance task', f'{n} demonstrations of D3IL', True, FONT)
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
        _draw_common(f, sc, '#7f8c8d', 0.28)
        panels.append(f)
    width = sum(p.w for p in panels) + 2 * 8
    h = Fig(width, 56, ml=0, mr=0, mt=0, mb=0, font=FONT_ENV)
    x = 16
    items = [('area', 'excluded by the constraints'), ('dash', f'tightened by {d:g} m'),
             ('obst', 'obstacle'), ('goal', 'goal line'), ('demo', 'demonstration')]
    for kind, lab in items:
        if kind == 'area':
            h.s.append(f'<rect x="{x - 8}" y="16" width="16" height="16" fill="#5d6d7e" fill-opacity="0.32" stroke="#34495e"/>')
        elif kind == 'dash':
            h.poly([(x - 10, 24), (x + 10, 24)], '#34495e', dash='7,5', w=2)
        elif kind == 'obst':
            h.marker(x, 24, 'o', '#c0392b', r=6.5)
        elif kind == 'goal':
            h.poly([(x - 10, 24), (x + 10, 24)], '#27ae60', w=5)
        else:
            h.poly([(x - 10, 24), (x + 10, 24)], '#7f8c8d', w=2)
        h.text(x + 20, 31, lab, 11, '#111')
        x += 50 + len(lab) * 10.6
    path = save_grid(panels, os.path.join(outdir, 'fig_constraints_avoiding.svg'), cols=3, gap=8, header=h)
    return path, f'Data_Analysis/DA_in_Paper/data/avoiding_scene.json | config/projection_eval.yaml geometries; {n} D3IL demonstrations'

# ═══════════════════════════════════════════════════════════════════════════
#  Fig 5 -- the step-budget ladder. THE figure the engine chapter turns on.
# ═══════════════════════════════════════════════════════════════════════════
def fig_avoiding_k_ladder(outdir):
    """Quality against step budget: who can walk K down and who cannot.

    Two vertical axes would be a lie here, so quality is on its own panel: the
    point is not that the transport engines are cheaper (Fig 1-4 show that), it
    is that their quality is FLAT in K while the diffusion engine's collapses
    below its training budget. K is inference-time for one family and
    training-time for the other -- that asymmetry is the mechanism.
    """
    c, data = _avoiding_cells()
    if c is None:
        return None
    rows = data['AGG']
    f = Fig(720, 470, ml=76, mr=110)
    f.axes((0.8, 26), (0.55, 1.04), xlog=True)
    f.frame([1, 2, 5, 10, 20], [0.6, 0.7, 0.8, 0.9, 1.0],
            'step budget K  [ network evaluations per plan ]   (log)',
            'success and constraint satisfaction',
            'Success against step budget, obstacle avoidance',
            '5 seeds (6-10) x 20 episodes; geometry mean; tightened; U-Net 4.0M.',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.2f}')

    # The baseline in a smaller 5 x 2 sample: the only measurement of
    # what diffusion does below its training budget. Drawn dashed and hollow, and
    # named as a different protocol in the legend, because it is one -- mixing it
    # into the solid series would be the single most misleading thing this figure
    # could do.
    t1_cells = S.load_exact(c, S.AVOIDING_T1_DIFFUSION_FOLDERS, 'diffusion')
    t1_rows = S.geometry_mean(t1_cells, lambda k: (k[0], k[1], k[4]), geom_index=3)
    t1_pts = [(K, t1_rows[('diffusion', K, 'dpcc-c-tightened')])
              for K in sorted(S.AVOIDING_T1_DIFFUSION_FOLDERS)
              if ('diffusion', K, 'dpcc-c-tightened') in t1_rows]
    if len(t1_pts) > 1:
        f.poly([(f.X(K), f.Y(r['n_success_and_constraints'])) for K, r in t1_pts],
               S.ENGINE_COLOUR['diffusion'], dash='5,4', w=1.6)
        for K, r in t1_pts:
            f.marker(f.X(K), f.Y(r['n_success_and_constraints']), 'o',
                     S.ENGINE_COLOUR['diffusion'], filled=False, r=5.0)

    for eng in ('mf', 'fm', 'diffusion'):
        # The baseline has no temporal-consistency row on both-hard, so it is
        # plotted on the cumulative-cost rule -- which is also the rule its
        # pinned configuration uses. Stated in the caption, not hidden.
        rule = 'dpcc-c-tightened' if eng == 'diffusion' else 'dpcc-t-tightened'
        pts = [(K, rows[(eng, K, rule)]) for K in KS if (eng, K, rule) in rows]
        if not pts:
            continue
        f.poly([(f.X(K), f.Y(r['n_success_and_constraints'])) for K, r in pts],
               S.ENGINE_COLOUR[eng], w=1.9)
        for K, r in pts:
            f.marker(f.X(K), f.Y(r['n_success_and_constraints']), 's',
                     S.ENGINE_COLOUR[eng], filled=(r['n_geometries'] == 3), r=5.0)

    lx = f.R + 14
    f.text(lx, f.T + 4, 'model', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(S.ENGINE_COLOUR[k], S.ENGINE_LABEL[k], 's')
                             for k in ('mf', 'fm', 'diffusion')])
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

    f = Fig(760, 470, ml=80, mr=165)
    ymax = max(r['avg_time'] for r in rows.values()) * 1.35
    f.axes((1.6, 6.0), (0.02, ymax), ylog=True)
    f.vspan(f.X(1.6), f.X(2.5))
    f.frame([2, 3, 5], dec_ticks(0.02, ymax),
            'step budget K  [ network evaluations per plan ]',
            'wall clock per control step  [ s ]   (log)',
            'Wall-clock time of the two projection methods, obstacle avoidance',
            # Kept short: at 800 px this line is clipped past about 100 characters,
            # and the model details are in the thesis caption anyway.
            f'{c.protocol}; MeanFM, U-Net 4.0M, tightened.',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:g}')
    for var, col, lab, mk, degen in ARMS:
        pts = [(K, rows[(K, var)]['avg_time']) for K in ks if (K, var) in rows]
        if not pts:
            continue
        f.poly([(f.X(K), f.Y(t)) for K, t in pts], col, w=1.9)
        for K, t in pts:
            f.marker(f.X(K), f.Y(t), mk, col, filled=not (degen and K < 3), r=5.5)
            f.text(f.X(K) + 9, f.Y(t) - 6, f'{t * 1000:.0f} ms', 8.0, '#111')

    lx = f.R + 14
    f.text(lx, f.T + 4, 'projection method', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(c_, l, m) for _v, c_, l, m, _d in ARMS])
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

    f = Fig(680, 480, ml=76, mr=20, mb=95)
    f.axes((-0.5, len(rows) - 0.5), (0.0, 1.16))
    f.frame([], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], '', 'episodes reaching the goal',
            'Without projection: the plan the model itself produces',
            f'{c.protocol}; {S.AVOIDING_RAW_GEOMETRY}, no projection. '
            f'U-Net 4.0M throughout; the baseline at its own budget.',
            yfmt=lambda v: f'{v:.1f}')

    bw = 0.56
    for i, (eng, (K, m)) in enumerate(rows.items()):
        v = m['n_success']
        x0, x1 = f.X(i - bw / 2), f.X(i + bw / 2)
        f.bar(x0, f.Y(v), x1 - x0, f.Y(0.0) - f.Y(v), S.AVOIDING_AF_COLOUR[eng])
        f.text((x0 + x1) / 2, f.Y(v) - 8, f'{v:.2f}', 11.5, '#111', anchor='middle', bold=True)
        f.text((x0 + x1) / 2, f.Y(v) - 22, f"{m['n_steps']:.1f} steps", 9.5, '#666', anchor='middle')
        lab = S.AVOIDING_AF_LABEL[eng]
        for j, part in enumerate(lab.split('\n')):
            f.text((x0 + x1) / 2, f.B + 14 + 11 * j, part, 7.8, '#222', anchor='middle')
        f.text((x0 + x1) / 2, f.B + 16 + 11 * len(lab.split('\n')),
               f'K = {K}', 8.6, '#666', anchor='middle')

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
