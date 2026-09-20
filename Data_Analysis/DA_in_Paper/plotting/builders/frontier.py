#!/usr/bin/env python3
"""Cost frontiers for the two environments beyond D3IL-avoiding.

`avoiding.fig_avoiding_tradeoff` draws the frontier of that benchmark as cost against
cost -- control steps against time per control step -- because there every eligible
configuration solves the task and only the price differs. Neither of the other two
environments is like that:

* on D3IL-aligning nothing "succeeds" or "fails"; what a model delivers is a distance,
  and control steps are bounded by the episode limit rather than by the task, so the
  frontier is **outcome against cost** -- the final box-to-target distance against the
  time one action costs. Down and to the left is better on both.
* on UAV-corridor the outcome is success with constraint satisfaction and, again, the
  control-step count is capped by the episode limit rather than by arrival, so the
  frontier is **outcome against cost** there too: S&C against time per control step.
  Up and to the left is better.

Both are genuine non-dominated sets: a configuration is on the frontier when no other
configuration in the panel is at the same time better in outcome and cheaper per action.
The frontier is computed here rather than through `sources.pareto_front`, whose
eligibility band is the avoiding benchmark's definition (quality within 0.05 of the best,
then compete on two costs) and does not apply to a continuous outcome.

Stdlib only, like every builder.
"""
import csv
import os
import statistics as stats

import sources as S
from svg.fmpcc_svg import Fig, dec_ticks, fmt_num

FONT = 1.5
MODEL_ORDER = ('mf', 'af', 'fm', 'diffusion')      # the house order: MeanFM, CI-MeanFM, FM, Diffusion


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _context_key(r):
    """The position-only context identity used by the alignment analysis.

    Angle fields are inconsistent across these runs, while the initial box and target
    positions identify the same ten situations in every thesis cell.
    """
    values = [_f(r.get(c)) for c in ('context_box_init_xy_x', 'context_box_init_xy_y',
                                      'context_target_xy_x', 'context_target_xy_y')]
    return tuple(round(v, 3) for v in values) if all(v is not None for v in values) else None


def _dodge(f, pts, gap=15.0):
    """Pixel position per point, with coincident points nudged apart horizontally.

    Two models tied at the same outcome and within a millisecond of each other land on
    the same pixel of a log cost axis, and one then hides the other. Where that happens
    the later point is displaced by a fixed few pixels, alternating side. The DATA is
    untouched -- only the drawing moves -- and every figure that uses this says so in its
    caption.
    """
    placed, out = [], {}
    for p in sorted(pts, key=lambda q: (q['ms'], q['y'])):
        x, y = f.X(p['ms']), f.Y(p['y'])
        k = 0
        while any(abs(x - px) < gap and abs(y - py) < gap for px, py in placed):
            k += 1
            x = f.X(p['ms']) + (gap * ((k + 1) // 2)) * (1 if k % 2 else -1)
        placed.append((x, y))
        out[id(p)] = (x, y)
    return out


def _frontier(pts, better):
    """Non-dominated set, ordered by cost. `better(a, b)` is True when a's outcome beats b's."""
    front = []
    for p in sorted(pts, key=lambda q: q['ms']):
        if all(not (better(q, p) and q['ms'] <= p['ms']) for q in pts if q is not p):
            front.append(p)
    return front


def _scatter(f, pts, front, label_key='K'):
    """Points, rings on the non-dominated ones, and a budget label beside each.

    The label sits above the marker for the diffusion baseline and below it for every
    other model, so that two models at the same budget and nearly the same cost -- which
    is exactly where FM and the baseline land on D3IL-aligning -- do not overprint."""
    fx = {id(p) for p in front}
    pos = _dodge(f, pts)
    for p in pts:
        x, y = pos[id(p)]
        f.marker(x, y, 'o', S.ENGINE_COLOUR_DISTINCT[p['engine']], filled=id(p) in fx, r=6.5, ew=1.6)
        dy = -11 if p['engine'] == 'diffusion' else 19
        f.text(x + 17, y + dy, p[label_key], 11, '#111' if id(p) in fx else '#777')
    for p in front:
        f.ring(*pos[id(p)], r=13)


def _legend_strip(width, engines):
    h = Fig(width, 42, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 20
    for eng in engines:
        h.marker(x, 21, 'o', S.ENGINE_COLOUR_DISTINCT[eng], r=6.5)
        h.text(x + 14, 25, S.ENGINE_LABEL[eng], 11, '#111')
        x += 34 + len(S.ENGINE_LABEL[eng]) * 10.5
    h.marker(x + 6, 21, 'o', '#777', filled=False, r=6.5)
    h.text(x + 20, 25, 'dominated', 11, '#111')
    return h


# ═══════════════════════════════════════════════════════════════════════════
#  D3IL-aligning: final box-to-target distance against time per control step
# ═══════════════════════════════════════════════════════════════════════════
def _aligning_cells(corpus):
    """-> {(engine, K): {'y': median final distance, 'ms': mean time per control step, 'n': contexts}}

    A context's value is the median over its rollouts and a cell's value the median over
    its contexts, which is what analysis/va_results.py does; contexts are matched by their
    initial box and target position, because `context_index` is empty on these rows.
    """
    geo, variant = S.ALIGNING_UNPROJECTED
    rows = {k: [] for k in S.ALIGNING_CELLS}
    with open(os.path.join(corpus.path, 'per_rollout_detail.csv')) as fh:
        for r in csv.DictReader(fh):
            if r['geo'] != geo or r['variant'] != variant:
                continue
            for key, (prefix, suffix) in S.ALIGNING_CELLS.items():
                fn = r['FolderName']
                if fn.startswith(prefix) and (suffix is None or fn.endswith(suffix)):
                    rows[key].append(r)
    # MeanFM at K=20 is the ten-context protocol cell named in Chapter 5. Some raw
    # cells contain a thirty-context superset; reduce every plotted cell to these same
    # ten before comparing them. This keeps the figure paired with Tables 6.9--6.11.
    reference = {_context_key(r) for r in rows.get(('mf', 20), [])}
    reference.discard(None)
    if len(reference) != 10:
        return {}

    out = {}
    for key, rs in rows.items():
        if not rs:
            continue
        by_ctx = {}
        for r in rs:
            fp = _context_key(r)
            d = _f(r.get('context_final_xy_dist'))
            ms = _f(r.get('avg_time_ms'))
            if fp in reference and d is not None:
                by_ctx.setdefault(fp, {'distance': [], 'ms': []})['distance'].append(d)
                if ms is not None:
                    by_ctx[fp]['ms'].append(ms)
        complete = [v for v in by_ctx.values() if v['distance'] and v['ms']]
        if len(complete) != len(reference):
            continue
        out[key] = {'y': stats.median([stats.median(v['distance']) for v in complete]),
                    'ms': stats.mean([stats.mean(v['ms']) for v in complete]),
                    'n': len(complete)}
    return out


def fig_aligning_tradeoff(outdir):
    c = S.CORPORA['visual_aligning_15_09']
    if not c.available:
        return None
    cells = _aligning_cells(c)
    if len(cells) < 4:
        return None
    pts = [dict(engine=e, K=K, ms=v['ms'], y=v['y']) for (e, K), v in cells.items()
           if (e, K) in S.ALIGNING_REPORTED]
    front = _frontier(pts, lambda a, b: a['y'] < b['y'])
    xlo, xhi = min(p['ms'] for p in pts) * 0.6, max(p['ms'] for p in pts) * 1.9
    # No title and no protocol line inside the drawing: which run, which seed and which
    # projection state this is belongs to the caption, not to the page (author, v3.49).
    f = Fig(760, 430, ml=92, mr=22, mt=26, mb=74, font=FONT)
    f.axes((xlo, xhi), (0.0, 0.52), xlog=True)
    f.frame(dec_ticks(xlo, xhi), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'time per control step [ms] (log)', 'final box-to-target distance [m]',
            '', '',
            xfmt=fmt_num, yfmt=lambda v: f'{v:.1f}')
    # where the box started: everything above this line moved the box less than half way
    y0 = f.Y(S.ALIGNING_INITIAL_DISTANCE)
    f.s.append(f'<line x1="{f.L}" y1="{y0:.1f}" x2="{f.R}" y2="{y0:.1f}" stroke="#c0392b" '
               f'stroke-width="1.4" stroke-dasharray="6,4"/>')
    f.text(f.L + 8, y0 - 9, 'box not moved', 11, '#c0392b', anchor='start', bold=True)
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q['ms']), f.Y(q['y'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['ms']), f.Y(q['y'])))
        f.poly(st, '#222', dash='6,4', w=1.6)
    _dirarrow(f, -1, +1, y0=f.B - 92)   # less time, closer to the target; clear of the start line
    _scatter(f, pts, front)
    hdr = _legend_strip(760, [e for e in MODEL_ORDER if any(k[0] == e for k in cells)])
    from svg.fmpcc_svg import save_grid
    path = save_grid([f], os.path.join(outdir, 'fig_aligning_tradeoff.svg'), cols=1, gap=8, header=hdr)
    return path, f'{c.rel} | {c.protocol} | unprojected, {"/".join(S.ALIGNING_UNPROJECTED)}'


# ═══════════════════════════════════════════════════════════════════════════
#  UAV-corridor: success with constraint satisfaction against time per control step
# ═══════════════════════════════════════════════════════════════════════════
def _corridor_cells(corpus):
    """-> {(engine, K, rule): {'y': S&C, 'ms': mean time per control step, 'n': flights}}"""
    cfg = S.UAV_CORRIDOR
    want = {f'dpcc-{r}{cfg["suffix"]}': r for r in ('r', 'c', 't')}
    rows = {}
    with open(os.path.join(corpus.path, 'per_rollout_detail.csv')) as fh:
        for r in csv.DictReader(fh):
            fn = r['FolderName']
            if not r['geo'].startswith(cfg['geo_prefix']) or not fn.rsplit('@', 1)[-1].endswith(cfg['tag']):
                continue
            rule = want.get(r['variant'])
            if rule is None:
                continue
            eng, K = r['engine'], int(float(r['K']))
            if K not in cfg['budgets'].get(eng, ()):
                continue
            rows.setdefault((eng, K, rule), []).append(r)
    out = {}
    for key, rs in rows.items():
        sc = [_f(r.get(S.UAV_SC)) for r in rs]
        ms = [_f(r.get('avg_time_ms')) for r in rs]
        sc = [v for v in sc if v is not None]
        ms = [v for v in ms if v is not None]
        if sc and ms:
            out[key] = {'y': stats.mean(sc), 'ms': stats.mean(ms), 'n': len(rs)}
    return out


RULE_MARK = {'r': '^', 'c': 'o', 't': 's'}


def fig_uav_corridor_tradeoff(outdir):
    c = S.CORPORA['uav_19_09']
    if not c.available:
        return None
    cells = _corridor_cells(c)
    if len(cells) < 6:
        return None
    # One point per (model, budget), at its BEST selection rule -- the convention this
    # chapter compares models under. All three rules are tabled in tab:uav-corridor-projection;
    # drawn as thirty points they would be six coincident blots, because six of them are
    # tied at 1.00 within five milliseconds of each other.
    best = {}
    for (e, K, rule), v in cells.items():
        key = (e, K)
        if key not in best or (v['y'], -v['ms']) > (best[key]['y'], -best[key]['ms']):
            best[key] = dict(engine=e, K=K, rule=rule, ms=v['ms'], y=v['y'])
    pts = list(best.values())
    front = _frontier(pts, lambda a, b: a['y'] > b['y'])
    xlo, xhi = min(p['ms'] for p in pts) * 0.6, max(p['ms'] for p in pts) * 1.9
    # Title and protocol line removed for the same reason as in fig_aligning_tradeoff
    # (author, v3.49): the caption states the scene, the seed and the flights per cell.
    f = Fig(760, 430, ml=92, mr=22, mt=26, mb=74, font=FONT)
    f.axes((xlo, xhi), (-0.06, 1.12), xlog=True)
    f.frame(dec_ticks(xlo, xhi), [0.0, 0.25, 0.5, 0.75, 1.0],
            'time per control step [ms] (log)', 'success with constraint satisfaction',
            '', '',
            xfmt=fmt_num, yfmt=lambda v: f'{v:.2f}')
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q['ms']), f.Y(q['y'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['ms']), f.Y(q['y'])))
        f.poly(st, '#222', dash='6,4', w=1.6)
    # bottom right is where the baseline point and its budget label sit, so the key goes
    # to the top right instead; inside a box its placement no longer has to mirror the arrow.
    _dirarrow(f, -1, -1, y0=f.T + 78)   # less time, higher success
    fx = {id(p) for p in front}
    pos = _dodge(f, pts)
    for p in pts:
        x, y = pos[id(p)]
        f.marker(x, y, RULE_MARK[p['rule']], S.ENGINE_COLOUR_DISTINCT[p['engine']],
                 filled=id(p) in fx, r=6.5, ew=1.6)
        f.text(x + 17, y + (-11 if p['engine'] == 'diffusion' else 19), p['K'], 11,
               '#111' if id(p) in fx else '#777')
    for p in front:
        f.ring(*pos[id(p)], r=13)
    hdr = Fig(760, 62, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 20
    for eng in MODEL_ORDER:
        if not any(k[0] == eng for k in cells):
            continue
        hdr.marker(x, 18, 'o', S.ENGINE_COLOUR_DISTINCT[eng], r=6.5)
        hdr.text(x + 14, 22, S.ENGINE_LABEL[eng], 11, '#111')
        x += 34 + len(S.ENGINE_LABEL[eng]) * 10.5
    x = 20
    for rule, lab in (('r', 'random'), ('c', 'cumulative projection cost'), ('t', 'temporal consistency')):
        hdr.marker(x, 45, RULE_MARK[rule], '#777', r=6.5)
        hdr.text(x + 14, 49, lab, 11, '#111')
        x += 34 + len(lab) * 10.5
    from svg.fmpcc_svg import save_grid
    path = save_grid([f], os.path.join(outdir, 'fig_uav_corridor_tradeoff.svg'), cols=1, gap=8, header=hdr)
    return path, f'{c.rel} | {c.protocol} | tag {S.UAV_CORRIDOR["tag"]}, dpcc-{{r,c,t}}{S.UAV_CORRIDOR["suffix"]}'




# ═══════════════════════════════════════════════════════════════════════════
#  D3IL-aligning, per context: how far the box got, under each projection method
# ═══════════════════════════════════════════════════════════════════════════
# The counterpart of `fig_uav_corridor_paths` that this task can actually have.
#
# WHAT IS AND IS NOT IN THE COMMITTED DATA, because it decides the design:
#   available per rollout  context_box_init_xy_{x,y}, context_target_xy_{x,y},
#                          context_init_xy_dist, context_final_xy_dist,
#                          constraint_exec_zero_violation
#   NOT available          the box POSITION at any step, the final box position included --
#                          `context_final_box_xy_*` is empty on every thesis cell (populated
#                          on 13 unrelated cells out of 14,102 rollouts), and no per-step box
#                          pose is logged at all.
#
# So the box's path cannot be drawn and neither can its end POINT. What is known exactly is
# how far the box ended FROM ITS TARGET, which is the number tab:va-projection reports as a
# median. That is drawn as a circle of exactly that radius around the target: the box ended
# somewhere on it. A small circle is a delivered box; a circle that reaches back to the start
# is a box that never moved. Nothing here is interpolated or assumed.
ALIGN_PANELS = [('diffuser', 'no projection'),
                ('dpcc-r', 'per-step projection'),
                ('hardflow_sls-r', 'endpoint projection')]
ALIGN_GEO = 'combined_5-tightened'
MOVED_EPS = 1e-6


def _aligning_outcomes(corpus, cell=('mf', 20)):
    """-> {variant: [ {start, target, d0, d1, clean, moved}, ... ] }, one entry per context."""
    prefix, suffix = S.ALIGNING_CELLS[cell]
    want = {v for v, _ in ALIGN_PANELS}
    out = {v: {} for v in want}
    with open(os.path.join(corpus.path, 'per_rollout_detail.csv')) as fh:
        for r in csv.DictReader(fh):
            fn = r['FolderName']
            if r['geo'] != ALIGN_GEO or r['variant'] not in want:
                continue
            if not fn.startswith(prefix) or (suffix is not None and not fn.endswith(suffix)):
                continue
            g = [_f(r.get(k)) for k in ('context_box_init_xy_x', 'context_box_init_xy_y',
                                        'context_target_xy_x', 'context_target_xy_y',
                                        'context_init_xy_dist', 'context_final_xy_dist')]
            if any(v is None for v in g):
                continue
            bx, by, tx, ty, d0, d1 = g
            out[r['variant']][(round(bx, 3), round(by, 3), round(tx, 3), round(ty, 3))] = dict(
                start=(bx, by), target=(tx, ty), d0=d0, d1=d1,
                clean=_f(r.get('constraint_exec_zero_violation')) == 1.0,
                moved=abs(d1 - d0) > MOVED_EPS)
    return {v: list(d.values()) for v, d in out.items() if d}


def _dirarrow(f, dx, dy, label='better', x0=None, y0=None):
    """A boxed key giving the preferred direction on both axes.

    Drawn as a LEGEND ELEMENT -- a bordered white box holding the arrow and the word
    `better` -- rather than as a bare arrow, so that a reader takes it for part of the
    key and not for an annotation of a data point (author, v3.48).

    The box sits in the corner the arrow points AWAY from, so it never lands on the
    frontier. dx, dy are pixel directions; (-1, +1) is left-and-down on screen, which is
    down-left on axes whose y grows upwards. x0, y0 override the box CENTRE.
    """
    import math as _m
    k = f.font or 1.0
    W, H = 76 * k, 74 * k                       # box, scaled with the figure's font factor
    pad = 14 * k
    cx = x0 if x0 is not None else ((f.L + pad + W / 2) if dx > 0 else (f.R - pad - W / 2))
    cy = y0 if y0 is not None else ((f.T + pad + H / 2) if dy > 0 else (f.B - pad - H / 2))
    col = '#34495e'
    f.s.append(f'<rect x="{cx - W / 2:.1f}" y="{cy - H / 2:.1f}" width="{W:.1f}" height="{H:.1f}" '
               f'rx="{4 * k:.1f}" fill="#ffffff" fill-opacity="0.94" stroke="{col}" '
               f'stroke-width="{1.2 * k:.1f}"/>')
    # arrow in the upper part of the box, label under it
    ax, ay = cx, cy - 9 * k
    L = 22 * k
    x0a, y0a, x1a, y1a = ax - dx * L, ay - dy * L, ax + dx * L, ay + dy * L
    f.s.append(f'<line x1="{x0a:.1f}" y1="{y0a:.1f}" x2="{x1a:.1f}" y2="{y1a:.1f}" stroke="{col}" '
               f'stroke-width="{2.4 * k:.1f}" stroke-linecap="round"/>')
    a = _m.atan2(y1a - y0a, x1a - x0a)
    for da in (2.6, -2.6):
        hx, hy = x1a + 9 * k * _m.cos(a + da), y1a + 9 * k * _m.sin(a + da)
        f.s.append(f'<line x1="{x1a:.1f}" y1="{y1a:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{col}" '
                   f'stroke-width="{2.4 * k:.1f}" stroke-linecap="round"/>')
    f.text(cx, cy + H / 2 - 9 * k, label, 11.0, col, anchor='middle', bold=True)


def fig_aligning_outcomes(outdir):
    """The ten contexts in ONE panel, three stems each -- unprojected, per-step, endpoint.

    Three separate panels put ten stems into a fifth of the text width and nothing could be
    read. Side by side within a context is also the comparison the reader wants: the same
    situation under the three treatments.
    """
    c = S.CORPORA['visual_aligning_15_09']
    if not c.available:
        return None
    data = _aligning_outcomes(c)
    if len(data) < len(ALIGN_PANELS):
        return None
    ctx = sorted({(e['start'], e['target']) for e in data[ALIGN_PANELS[0][0]]})
    by = {v: {(e['start'], e['target']): e for e in rows} for v, rows in data.items()}
    n = len(ctx)
    COL = {'diffuser': '#b0b7bd', 'dpcc-r': '#1f6fb2', 'hardflow_sls-r': '#0f8b8d'}
    fs = 1.58
    f = Fig(1020, 400, ml=int(58 * fs), mr=int(14 * fs), mt=52, mb=int(46 * fs), font=fs)
    f.axes((-0.7, n - 0.3), (0.0, 0.62))
    f.frame([], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'context',
            'box-to-target distance [m]',
            '',
            yfmt=lambda v: f'{v:.1f}')
    w = 0.22
    for i, k in enumerate(ctx):
        # where the box started in this context, spanning the three stems
        d0 = by[ALIGN_PANELS[0][0]][k]['d0']
        f.poly([(f.X(i - 1.35 * w), f.Y(d0)), (f.X(i + 1.35 * w), f.Y(d0))], '#777', w=2.2)
        for s_, (var, _lab) in enumerate(ALIGN_PANELS):
            e = by[var].get(k)
            if e is None:
                continue
            x = f.X(i + (s_ - 1) * w)
            col = COL[var]
            f.poly([(x, f.Y(0.0)), (x, f.Y(e['d1']))], col, w=5.0)
            if not e['clean']:      # a violating context is crossed through
                f.poly([(x - 7, f.Y(e['d1']) - 7), (x + 7, f.Y(e['d1']) + 7)], '#c0392b', w=2.4)
                f.poly([(x - 7, f.Y(e['d1']) + 7), (x + 7, f.Y(e['d1']) - 7)], '#c0392b', w=2.4)
            else:
                f.marker(x, f.Y(e['d1']), 'o', col, filled=e['moved'], r=5.0, ew=1.6)
        f.text(f.X(i), f.B + 17, i + 1, 11, '#444', anchor='middle')
    # One compact treatment legend. Marker semantics belong in the caption.
    lx, ly = f.L + 6, 24
    for s_, (var, lab) in enumerate(ALIGN_PANELS):
        xx = lx + s_ * 210
        f.poly([(xx, ly), (xx + 20, ly)], COL[var], w=5.0)
        f.text(xx + 28, ly + 5, lab, 11, '#111')
    # Which model these stems belong to was only in the caption; it belongs on the drawing
    # too, because three generative models appear in this section (author, v3.49).
    f.text(f.R, ly + 5, 'MeanFM, K = 20', 11.5, '#111', anchor='end', bold=True)
    path = f.save(os.path.join(outdir, 'fig_aligning_outcomes.svg'))
    return path, (f'{c.rel} | {c.protocol} | geometry {ALIGN_GEO}, variants '
                  + ', '.join(v for v, _ in ALIGN_PANELS)
                  + ' | stems are context_final_xy_dist, the column tab:va-projection medians')


ALL = [
    ('fig_aligning_tradeoff', 'da', fig_aligning_tradeoff),
    ('fig_aligning_outcomes', 'da', fig_aligning_outcomes),
    ('fig_uav_corridor_tradeoff', 'da', fig_uav_corridor_tradeoff),
]
