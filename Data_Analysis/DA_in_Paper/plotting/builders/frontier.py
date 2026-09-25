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
import math
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


def _scatter(f, pts, front, label_key='K', avoid=()):
    """Points, rings on the non-dominated ones, and a budget label beside each.

    The label sits below the marker for the diffusion baseline and above it for every
    other model, so that two models at the same budget and nearly the same cost -- which
    is exactly where FM and the baseline land on D3IL-aligning, the baseline the lower
    of the two on the closed-share axis (v3.62) -- do not overprint."""
    fx = {id(p) for p in front}
    pos = _dodge(f, pts)
    # v3.80: the label goes to the first of four sides that is clear of every marker and of the labels
    # already placed (the complete tab:va-models ladder put two labels on neighbouring markers).
    marks = [pos[id(q)] for q in pts]
    placed = list(avoid)          # boxes already on the page (x0, y0, x1, y1)

    def clear(box):
        x0, y0, x1, y1 = box
        if any(x0 - 8 <= mx <= x1 + 8 and y0 - 8 <= my <= y1 + 8 for mx, my in marks):
            return False
        return all(x1 < a0 or a1 < x0 or y1 < b0 or b1 < y0 for a0, b0, a1, b1 in placed)

    for p in pts:
        x, y = pos[id(p)]
        f.marker(x, y, 'o', S.ENGINE_COLOUR_DISTINCT[p['engine']], filled=id(p) in fx, r=6.5, ew=1.6)
        lab = str(p[label_key])
        w, h = 7.0 * len(lab), 11.0
        up, down = (x + 17, y - 11), (x + 17, y + 19)
        cands = [down, up] if p['engine'] == 'diffusion' else [up, down]
        cands += [(x - 17 - w, c[1]) for c in cands]
        tx, ty = next(((cx, cy) for cx, cy in cands if clear((cx, cy - h, cx + w, cy))), cands[0])
        placed.append((tx, ty - h, tx + w, ty))
        f.text(tx, ty, p[label_key], 11, '#111' if id(p) in fx else '#777')
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
# The outcome axis of both alignment frontiers (v3.62, author): the share of the starting
# distance the box was moved TOWARDS its target, on a linear axis. 100 % is a box delivered
# to the target and 0 % a median final distance equal to the mean initial one: the printed
# number is 100*(1 - median(d_final)/mean(d_init)), not a median of per-context shares, so 0 %
# is NOT 'a box not moved' (v3.100, audit F03; the zero-line label reads '0 % reference'). It is
# the bracketed number of
# tab:va-models. v3.61 drew the complement (final distance as % of the start, log axis),
# which put 100 % at the bad end and read as inverted.
PCT_LABEL = 'distance closed, % of the start (log scale)'
# v3.68d (author): the axis is STRETCHED towards 100 %: a point is placed at
# -log10(distance left / start), so 0 % sits at 0, 50 % at 0.30, 80 % at 0.70, 90 % at 1.0,
# and the crowded top of the drawing (84 % vs 85 %) opens up. Tick labels stay in %.
PCT_TICKS = [0, 50, 80, 90, 95]
PCT_YLIM = (-0.07, 1.36)


def _pct_closed(d):
    return 100.0 * (1.0 - d / S.ALIGNING_INITIAL_DISTANCE)


def _ypos(share):
    """Axis position of a share-closed value: log in the distance left."""
    import math
    left = max(1.0 - share / 100.0, 1e-3)
    return -math.log10(left)


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
    # v3.80: the rows only another corpus holds (sources.ALIGNING_UNPROJECTED_EXTRA), same filter
    for key, (ckey, prefix, suffix) in getattr(S, 'ALIGNING_UNPROJECTED_EXTRA', {}).items():
        cx = S.CORPORA[ckey]
        if key in rows and rows[key] or not cx.available:
            continue
        rows[key] = []
        with open(os.path.join(cx.path, 'per_rollout_detail.csv')) as fh:
            for r in csv.DictReader(fh):
                fn = r['FolderName']
                if r['geo'] == geo and r['variant'] == variant and fn.startswith(prefix) and fn.endswith(suffix):
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
    # v3.62 (author): the outcome axis is the share of the starting distance the box was
    # moved towards its target, linear, 100 % at the top -- the bracketed number of
    # tab:va-models. (v3.61 drew final distance as % of the start on a log axis.)
    for p in pts:
        p['y'] = _ypos(_pct_closed(p['y']))
    tick_at = {_ypos(t): t for t in PCT_TICKS}
    f = Fig(760, 430, ml=92, mr=22, mt=26, mb=74, font=FONT)
    f.axes((xlo, xhi), PCT_YLIM, xlog=True)
    f.frame(dec_ticks(xlo, xhi), sorted(tick_at),
            'time per control step [ms] (log)', PCT_LABEL,
            '', '',
            xfmt=fmt_num, yfmt=lambda v: f'{tick_at[v]:g}')
    # where the box started: everything on this line did not move the box
    y0 = f.Y(0.0)
    f.s.append(f'<line x1="{f.L}" y1="{y0:.1f}" x2="{f.R}" y2="{y0:.1f}" stroke="#c0392b" '
               f'stroke-width="1.4" stroke-dasharray="6,4"/>')
    # v3.80: the label sits in the widest empty stretch of the 0 % line (points near it in pixels)
    near = sorted(f.X(p['ms']) for p in pts if abs(f.Y(p['y']) - y0) < 40)
    edges = [f.L] + near + [f.R]
    ga, gb = max(zip(edges, edges[1:]), key=lambda ab: ab[1] - ab[0])
    lx = ga + 14 if gb - ga > 160 else f.L + 8
    f.text(lx, y0 - 9, '0 % reference', 11, '#c0392b', anchor='start', bold=True)
    zero_label = (lx, y0 - 22, lx + 140, y0 - 6)     # handed to _scatter so no budget label lands on it
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q['ms']), f.Y(q['y'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['ms']), f.Y(q['y'])))
        f.poly(st, '#222', dash='6,4', w=1.6)
    # The direction key was dropped when K=100 was added because it covered the two
    # right-hand points; the author asked for it back (v3.61). With the closed share on
    # the axis, better is up-left; the top-left corner is empty, and it is the corner
    # the arrow points INTO.
    _dirarrow(f, -1, -1, x0=f.L + 14 * FONT + 38 * FONT, y0=f.T + 14 * FONT + 37 * FONT)
    _scatter(f, pts, front, avoid=[zero_label])
    hdr = _legend_strip(760, [e for e in MODEL_ORDER if any(k[0] == e for k in cells)])
    from svg.fmpcc_svg import save_grid
    path = save_grid([f], os.path.join(outdir, 'fig_aligning_tradeoff.svg'), cols=1, gap=8, header=hdr)
    return path, f'{c.rel} | {c.protocol} | unprojected, {"/".join(S.ALIGNING_UNPROJECTED)}'


def fig_aligning_projected_tradeoff(outdir):
    """Projected alignment: distance against cost, with constraint coverage explicit.

    Random selection and the ten shared contexts are held fixed. A point enters
    the two-axis frontier only when at least nine contexts are violation-free;
    hollow points remain visible but are not described as successful trade-offs.

    v3.66 (author): the figure shows the SIMPLIFIED tab:va-projection-models -- the
    operating point only, K=20 at threshold 0.2, three models x two projectors (CI-MeanFM
    has no endpoint cell). The K=2 and K=10 points are no longer drawn; the budget ladder
    is tab:va-projection and tab:va-threshold. PROJECTED_K lists what is drawn.

    v3.77 (author, 23-09): the table is complete, so the figure draws it -- the diffusion
    baseline's per-step cell and CI-MeanFM's endpoint cell come in through
    sources.ALIGNING_PROJECTED_EXTRA (each from the corpus that holds it). Both keep fewer
    than nine contexts violation-free, so both are hollow and the frontier does not move.
    """
    PROJECTED_K = (20,)
    colours = {'mf': '#1F4E79', 'af': '#8B3F71', 'fm': '#C45B24',
               'diffusion': S.ENGINE_COLOUR_DISTINCT['diffusion']}   # the baseline stays near-black
    c = S.CORPORA['visual_aligning_15_09']
    if not c.available:
        return None
    reference = _aligning_outcomes(c, ('mf', 20)).get('diffuser', [])
    contexts = {_context_key({'context_box_init_xy_x': e['start'][0],
                              'context_box_init_xy_y': e['start'][1],
                              'context_target_xy_x': e['target'][0],
                              'context_target_xy_y': e['target'][1]}) for e in reference}
    if len(contexts) != 10:
        return None
    variants = ('dpcc-r', 'hardflow_sls-r')
    grouped = {}
    with open(os.path.join(c.path, 'per_rollout_detail.csv')) as fh:
        for r in csv.DictReader(fh):
            if r['geo'] != ALIGN_GEO or r['variant'] not in variants:
                continue
            context = _context_key(r)
            if context not in contexts:
                continue
            for cell, (prefix, suffix) in S.ALIGNING_CELLS.items():
                if cell not in S.ALIGNING_REPORTED or cell[0] == 'diffusion' or cell[1] not in PROJECTED_K:
                    continue
                if not r['FolderName'].startswith(prefix) or (suffix and not r['FolderName'].endswith(suffix)):
                    continue
                d, ms, clean = (_f(r.get(k)) for k in
                                ('context_final_xy_dist', 'avg_time_ms', 'constraint_exec_zero_violation'))
                if None not in (d, ms, clean):
                    grouped.setdefault((*cell, r['variant']), {}).setdefault(context, []).append((d, ms, clean))
    # v3.77: the cells ALIGNING_CELLS cannot reach, read from the corpus that holds each one,
    # on the same geometry, variant filter and ten contexts as everything above.
    for (eng, K, variant), (ckey, prefix, suffix) in S.ALIGNING_PROJECTED_EXTRA.items():
        cx = S.CORPORA[ckey]
        if K not in PROJECTED_K or not cx.available:
            continue
        with open(os.path.join(cx.path, 'per_rollout_detail.csv')) as fh:
            for r in csv.DictReader(fh):
                fn = r['FolderName']
                if r['geo'] != ALIGN_GEO or r['variant'] != variant:
                    continue
                if not fn.startswith(prefix) or not fn.endswith(suffix):
                    continue
                context = _context_key(r)
                if context not in contexts:
                    continue
                d, ms, clean = (_f(r.get(k)) for k in
                                ('context_final_xy_dist', 'avg_time_ms', 'constraint_exec_zero_violation'))
                if None not in (d, ms, clean):
                    grouped.setdefault((eng, K, variant), {}).setdefault(context, []).append((d, ms, clean))
    pts = []
    for (eng, K, variant), by_context in grouped.items():
        if set(by_context) != contexts:
            continue
        distance = stats.median([stats.median(v[0] for v in rows) for rows in by_context.values()])
        ms = stats.mean([stats.mean(v[1] for v in rows) for rows in by_context.values()])
        clean = sum(stats.median(v[2] for v in rows) == 1 for rows in by_context.values())
        pts.append(dict(engine=eng, K=K, variant=variant, ms=ms, y=distance, clean=clean))
    if len(pts) < 5:
        return None
    eligible = [p for p in pts if p['clean'] >= 9]
    front = sorted((p for p in eligible if not any(
        q is not p and q['ms'] <= p['ms'] and q['y'] <= p['y']
        and (q['ms'] < p['ms'] or q['y'] < p['y']) for q in eligible)),
        key=lambda p: p['ms'])
    xlo, xhi = min(p['ms'] for p in pts) * 0.65, max(p['ms'] for p in pts) * 1.55
    # v3.62 (author): share of the starting distance closed, linear, as in
    # fig_aligning_tradeoff. The frontier is unchanged in substance -- it is computed
    # on the eligible (>= 9/10 violation-free) points only, on the distances, before
    # the axis conversion -- and the legend says what a hollow marker means.
    for p in pts:
        p['y'] = _ypos(_pct_closed(p['y']))
    tick_at = {_ypos(t): t for t in PCT_TICKS}
    # v3.63 (author): say on the page that this is AFTER projection and show what it is
    # read against -- the unprojected MeanFM cells of fig_aligning_tradeoff are drawn as
    # faint grey rings at the same budgets, so the reader sees the price of projection
    # (further from the target, dearer per step) and its purchase (filled = at least
    # nine of ten contexts free of violations, which is this task's success with
    # constraint satisfaction). An unprojected plan does not depend on the constraint
    # set, so those points are the same on the untightened and the tightened set.
    before = {K: v for (e, K), v in _aligning_cells(c).items() if e == 'mf' and K in PROJECTED_K}
    xlo = min(xlo, min(v['ms'] for v in before.values()) * 0.65) if before else xlo
    f = Fig(760, 430, ml=92, mr=22, mt=26, mb=74, font=FONT)
    f.axes((xlo, xhi), PCT_YLIM, xlog=True)
    f.frame(dec_ticks(xlo, xhi), sorted(tick_at),
            'time per control step [ms] (log)', PCT_LABEL, '', '',
            xfmt=fmt_num, yfmt=lambda v: f'{tick_at[v]:g}')
    f.text(f.R - 8, f.T + 18, 'after projection, tightened constraints', 11, '#333', anchor='end', bold=True)
    y0 = f.Y(0.0)
    f.s.append(f'<line x1="{f.L}" y1="{y0:.1f}" x2="{f.R}" y2="{y0:.1f}" '
               'stroke="#777" stroke-width="1.4" stroke-dasharray="6,4"/>')
    for K, v in before.items():
        bx, by = f.X(v['ms']), f.Y(_ypos(_pct_closed(v['y'])))
        f.s.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="6.5" fill="none" stroke="#9a9a9a" '
                   'stroke-width="1.6" stroke-dasharray="2,2"/>')
        f.text(bx - 10, by - 10, K, 11, '#9a9a9a', anchor='end')
    # v3.77: above the line at the left end. Under the line (v3.6x) the label sat on the bottom
    # frame, which struck it through; the left end is empty near the line (the cheapest
    # projected point is ~266 ms, the axis starts below the unprojected ring at ~190 ms).
    f.text(f.L + 8, y0 - 9, '0 % reference', 11, '#777', anchor='start', bold=True)
    if len(front) > 1:
        staircase = []
        for i, p in enumerate(front):
            staircase.append((f.X(p['ms']), f.Y(p['y'])))
            if i + 1 < len(front):
                staircase.append((f.X(front[i + 1]['ms']), f.Y(p['y'])))
        f.poly(staircase, '#34495e', dash='6,4', w=1.6)
    # v3.69 (author): the key sat on the unprojected MeanFM ring at the top left; the left half of
    # the panel below 80 % is empty, so it goes there.
    # v3.79 (author, reading Fig 6.4): the key goes to the RIGHT of the panel -- the right half above
    # the hollow band is empty since v3.77 widened the axis to the diffusion cell, and the left half
    # holds the unprojected ring and the frontier.
    _dirarrow(f, -1, -1, x0=f.R - 14 * FONT - 38 * FONT, y0=f.T + 0.42 * (f.B - f.T))
    pos = _dodge(f, pts, gap=19)
    front_ids = {id(p) for p in front}
    for p in pts:
        x, y = pos[id(p)]
        kind = 'o' if p['variant'] == 'dpcc-r' else 's'
        f.marker(x, y, kind, colours[p['engine']],
                 filled=p['clean'] >= 9, r=6.5, ew=1.6)
        if id(p) in front_ids:
            f.ring(x, y, r=13)
            f.text(x - 16, y - 16, p['K'], 11, '#333', anchor='end')
    h = Fig(760, 62, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 20
    for eng in ('mf', 'af', 'fm', 'diffusion'):
        h.marker(x, 18, 'o', colours[eng], r=6.5)
        h.text(x + 14, 22, S.ENGINE_LABEL[eng], 11, '#111')
        x += 36 + 11.5 * len(S.ENGINE_LABEL[eng])     # v3.77: spaced by label length, four entries
    for kind, name in (('o', 'per-step'), ('s', 'endpoint')):
        h.marker(x, 18, kind, '#777', r=6.5)
        h.text(x + 14, 22, name, 11, '#111')
        x += 108
    x = 20
    h.marker(x, 45, 'o', '#777', filled=False, r=6.5)
    h.text(x + 14, 49, '< 9/10 contexts violation-free', 11, '#111')
    x += 300
    h.s.append(f'<circle cx="{x}" cy="45" r="6.5" fill="none" stroke="#9a9a9a" stroke-width="1.6" '
               'stroke-dasharray="2,2"/>')
    h.text(x + 14, 49, 'MeanFM before projection, same budget', 11, '#111')
    from svg.fmpcc_svg import save_grid
    path = save_grid([f], os.path.join(outdir, 'fig_aligning_projected_tradeoff.svg'),
                     cols=1, gap=8, header=h)
    return path, (f'{c.rel} | {c.protocol} | {ALIGN_GEO}, random selection, '
                  f'10 shared contexts; {len(pts)} complete cells at K in {PROJECTED_K}; '
                  'frontier requires at least 9/10 violation-free contexts')


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


def _fig_uav_corridor_tradeoff_v2(outdir):
    """ARCHIVED (v3.83): the corridor v2 frontier on the 19-09 corpus. Superseded by fig_uav_corridor_tradeoff below."""
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
    n_flights = max(v['n'] for v in cells.values())
    # v3.63 (author): the success axis is LOGARITHMIC in the number of successful flights.
    # Zero cannot be drawn on a log axis, so the configurations that never succeed sit on
    # a floor band under an axis break. Two rules go with that: a configuration with no
    # successful flight is not a trade-off and is not eligible for the frontier (before,
    # FM at K=1 was ringed for being the cheapest point at 0.00); and two costs within one
    # per cent of each other are one cost, so a 0.1 ms tie at K=1 cannot put a worse
    # point on the frontier. Ties in success do not dominate, as before, which is why the
    # four configurations at 1.00 all stay on it.
    eligible = [p for p in pts if p['y'] > 0]
    front = sorted((p for p in eligible if not any(
        q is not p and q['y'] > p['y'] and q['ms'] <= p['ms'] * 1.01 for q in eligible)),
        key=lambda p: p['ms'])
    xlo, xhi = min(p['ms'] for p in pts) * 0.6, max(p['ms'] for p in pts) * 1.9
    lo = 1.0 / n_flights                       # one successful flight
    floor = math.log10(lo) - 0.30              # where the zero-success points are drawn
    v = lambda y: math.log10(y) if y > 0 else floor
    ticks = [k / n_flights for k in (1, 2, 4, 6, n_flights) if k <= n_flights]
    labels = {floor: f'0/{n_flights}', **{v(t): f'{round(t * n_flights)}/{n_flights}' for t in ticks}}
    # Title and protocol line removed for the same reason as in fig_aligning_tradeoff
    # (author, v3.49): the caption states the scene, the seed and the flights per cell.
    f = Fig(760, 430, ml=92, mr=22, mt=26, mb=74, font=FONT)
    f.axes((xlo, xhi), (floor - 0.10, 0.08), xlog=True)
    f.frame(dec_ticks(xlo, xhi), sorted(labels),
            'time per control step [ms] (log)', 'success with constraint satisfaction (log)',
            '', '',
            xfmt=fmt_num, yfmt=lambda t: labels[t])
    from .avoiding import _break_marks
    _break_marks(f, f.Y((floor + math.log10(lo)) / 2))
    for p in pts:
        p['y'] = v(p['y'])
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
#  UAV-corridor v3 (v3.83): control steps to the finish line against time per action,
#  before and after projection -- the frontier of D3IL-avoiding (fig_avoiding_tradeoff),
#  on this scene
# ═══════════════════════════════════════════════════════════════════════════
# Author (v3.83): "I need the pareto for pre and post projection", in the convention of the
# avoiding section. So the rule is sources.pareto_front, unchanged: a point is eligible when
# its quality is within 0.05 of the best in its panel, and eligible points then compete on
# two costs, control steps and time per action. Quality before projection is the strict
# S&C (0 for every configuration, so every point competes); after projection it is the
# near-satisfaction rate, the corridor's frontier bar (Ch 5, >= 95 % violation-free steps).
# The strict reading after projection leaves the baseline alone in both panels; the chapter
# says that in words. One point per model x budget x projector, at its best rule for the bar.
# Steps = control steps to the finish line (x = 2.0 m), mean over the flights that reach it;
# per-step projection at K <= 2 over the hump never reaches it and has no point.
CORRIDOR_PROJ_MARK = {'none': 'o', 'per-step': 'o', 'endpoint': 's'}


def _corridor_v3_panel(pts, title, xr, yr, ylab, xlab, sc_labels):
    f = Fig(560, 440, ml=86, mr=18, mt=44, mb=70 if xlab else 40, font=FONT)
    f.axes(xr, yr, xlog=True)
    yt = [t for t in range(0, 400, 10) if yr[0] <= t <= yr[1]]
    f.frame(dec_ticks(*xr), yt, 'time per action [ms] (log)' if xlab else '',
            'control steps to the finish line' if ylab else '', title, '', xfmt=fmt_num, yfmt=lambda v: f'{v:.0f}')
    _, front = S.pareto_front(pts, x='ms', y='y', quality='q', band=0.05)
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q['ms']), f.Y(q['y'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['ms']), f.Y(q['y'])))
        f.poly(st, '#222', dash='6,4', w=1.6)
    pos = _dodge(f, pts, gap=12.0)
    marks = [pos[id(q)] for q in pts]
    placed = []

    def clear(box):
        x0, y0, x1, y1 = box
        if any(x0 - 7 <= mx <= x1 + 7 and y0 - 7 <= my <= y1 + 7 for mx, my in marks):
            return False
        # 5 px between two labels, so that two budget labels never read as one number ("3" "3" -> "33")
        return all(x1 + 5 < a0 or a1 + 5 < x0 or y1 + 3 < b0 or b1 + 3 < y0 for a0, b0, a1, b1 in placed)

    fx = {id(p) for p in front}
    for p in pts:
        x, y = pos[id(p)]
        f.marker(x, y, CORRIDOR_PROJ_MARK[p['proj']], S.ENGINE_COLOUR_DISTINCT[p['engine']],
                 filled=p['eligible'], r=6.5, ew=1.6)
        lab = str(p['K'])
        w, h = 7.0 * len(lab), 11.0
        cands = [(x + 12, y - 9), (x + 12, y + 18), (x - 12 - w, y - 9), (x - 12 - w, y + 18)]
        tx, ty = next(((cx, cy) for cx, cy in cands if clear((cx, cy - h, cx + w, cy))), cands[0])
        placed.append((tx, ty - h, tx + w, ty))
        f.text(tx, ty, lab, 11, '#111' if p['eligible'] else '#999')
        if sc_labels and p['sc'] > 0:
            lab2 = f"S&C {p['sc']:.1f}"
            w2 = 7.2 * len(lab2)
            c2 = [(x - w2 / 2, y + 30), (x - w2 / 2, y - 22), (x + 14, y + 30), (x - 14 - w2, y + 30)]
            sx, sy = next(((cx, cy) for cx, cy in c2 if clear((cx, cy - h, cx + w2, cy))), c2[0])
            placed.append((sx, sy - h, sx + w2, sy))
            f.text(sx, sy, lab2, 10.5, '#111', bold=True)
    for q in front:
        f.ring(*pos[id(q)], r=13)
    return f


def _corridor_v3_legend(width):
    h = Fig(width, 46, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 18
    for eng in MODEL_ORDER:
        h.marker(x, 23, 'o', S.ENGINE_COLOUR_DISTINCT[eng], r=6.5)
        h.text(x + 13, 27, S.ENGINE_LABEL[eng], 11, '#111')
        x += 30 + len(S.ENGINE_LABEL[eng]) * 10.0
    x += 8
    for mk, lab in (('o', 'per-step'), ('s', 'endpoint')):
        h.marker(x, 23, mk, '#777', r=6.5)
        h.text(x + 13, 27, lab, 11, '#111')
        x += 26 + len(lab) * 9.0
    h.marker(x, 23, 'o', '#777', filled=False, r=6.5)
    h.text(x + 13, 27, 'not eligible', 11, '#111')
    x += 26 + 12 * 9.0
    col = '#34495e'
    h.s.append(f'<rect x="{x - 4}" y="5" width="34" height="36" rx="3" fill="#fff" stroke="{col}" stroke-width="1.2"/>')
    cx, cy, L = x + 13, 23, 9
    h.s.append(f'<line x1="{cx + L}" y1="{cy - L}" x2="{cx - L}" y2="{cy + L}" stroke="{col}" '
               'stroke-width="2.2" stroke-linecap="round"/>')
    for hx, hy in ((cx - L + 8, cy + L), (cx - L, cy + L - 8)):
        h.s.append(f'<line x1="{cx - L}" y1="{cy + L}" x2="{hx}" y2="{hy}" stroke="{col}" '
                   'stroke-width="2.2" stroke-linecap="round"/>')
    h.text(x + 38, 27, 'better', 11, col, bold=True)
    return h


VIOL_GAP = 2.0          # v3.85: a new violation group starts where the sorted violating steps jump by more than this
GROUP_BAND = '#eef2f7'
GROUP_BAND_ALT = '#dde5ef'
GROUP_TXT = '#5d6d7e'
ROMAN = ('I', 'II', 'III', 'IV', 'V')


def _violation_groups(pts):
    """Eligible points grouped where their sorted violating steps jump by more than VIOL_GAP; I = the most."""
    el = sorted((p for p in pts if p['eligible']), key=lambda p: p['y'])
    groups, cur = [], []
    for p in el:
        if cur and p['y'] - cur[-1]['y'] > VIOL_GAP:
            groups.append(cur)
            cur = []
        cur.append(p)
    if cur:
        groups.append(cur)
    groups = groups[::-1]                       # I = the group with the most violating steps
    for gi, g in enumerate(groups):
        for p in g:
            p['group'] = gi
    return groups


def _label_points(f, pts, pos, grey=lambda p: not p['eligible']):
    marks = [pos[id(q)] for q in pts]
    placed = []

    def clear(box):
        x0, y0, x1, y1 = box
        if any(x0 - 7 <= mx <= x1 + 7 and y0 - 7 <= my <= y1 + 7 for mx, my in marks):
            return False
        return all(x1 + 5 < a0 or a1 + 5 < x0 or y1 + 3 < b0 or b1 + 3 < y0 for a0, b0, a1, b1 in placed)

    for p in pts:
        x, y = pos[id(p)]
        txt = str(p['K'])
        w, h = 7.0 * len(txt), 11.0
        cands = [(x + 12, y - 9), (x + 12, y + 18), (x - 12 - w, y - 9), (x - 12 - w, y + 18),
                 (x - w / 2, y - 14), (x - w / 2, y + 24)]
        tx, ty = next(((cx, cy) for cx, cy in cands if clear((cx, cy - h, cx + w, cy))), cands[0])
        placed.append((tx, ty - h, tx + w, ty))
        f.text(tx, ty, txt, 11, '#999' if grey(p) else '#111')


def _staircase(f, front, xk, yk, pos=None):
    if len(front) > 1:
        st = []
        for i, q in enumerate(front):
            st.append((f.X(q[xk]), f.Y(q[yk])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1][xk]), f.Y(q[yk])))
        f.poly(st, '#222', dash='6,4', w=1.6)


def fig_uav_corridor_tradeoff(outdir):
    """v3.85 (author): UAV-corridor after projection, 2 x 2.
    (a, b) violating control steps per flight against the time to compute one action -- the frontier of v3.84 --
           with the violation groups shaded: configurations whose sorted violating steps lie within VIOL_GAP of
           one another form a group (the clusters the author read off the v3.84 figure).
    (c, d) control steps to the finish line against time, D3IL-avoiding's frontier, drawn once per violation group:
           within a group the configurations violate about equally and compete on steps and time only.
    Eligible = success within 0.05 of the best (every flight reaches the end of the corridor); hollow points of (a, b)
    have no place in (c, d). One point per model x budget x projector at its best rule (best_viol)."""
    import json
    if not os.path.isfile(S.CORRIDOR_V3_FRONTIER):
        return None
    with open(S.CORRIDOR_V3_FRONTIER) as fh:
        D = json.load(fh)
    C = {(r['geo'], r['engine'], r['K'], r['variant']): r for r in D['cells']}
    data = []
    for geo in ('tilt', 'hump'):
        pts = []
        for k in D['best_viol']:
            if k[0] != geo:
                continue
            r = C[tuple(k)]
            pts.append(dict(engine=r['engine'], K=r['K'], proj=r['projector'], q=r['success'][0], ms=r['ms'][0],
                            y=r['viol'][0], steps=r['steps'][0], group=None))
        S.pareto_front(pts, x='ms', y='y', quality='q', band=0.05)      # sets p['eligible']
        data.append((geo, pts, _violation_groups(pts)))
    xr = (min(p['ms'] for _, P, _ in data for p in P) * 0.7, max(p['ms'] for _, P, _ in data for p in P) * 1.6)
    top, bot = [], []
    for (geo, pts, groups), lab in zip(data, ('tilt', 'hump')):
        # ---- (a, b): violating steps against time, groups as horizontal bands
        yr = (-0.6, math.ceil(max(p['y'] for p in pts) + 1.0))
        f = Fig(560, 430, ml=86, mr=18, mt=44, mb=70, font=FONT)
        f.axes(xr, yr, xlog=True)
        yt = [t for t in range(0, 40, 2) if yr[0] <= t <= yr[1]]
        f.frame(dec_ticks(*xr), yt, 'time per action [ms] (log)', 'violating steps per flight' if geo == 'tilt' else '',
                f"({'a' if geo == 'tilt' else 'b'}) {lab}: violating steps", '', xfmt=fmt_num, yfmt=lambda v: f'{v:.0f}')
        f.clip_to_box()
        for gi, g in enumerate(groups):
            lo, hi = min(p['y'] for p in g) - 0.45, max(p['y'] for p in g) + 0.45
            f.s.append(f'<rect x="{f.L:.1f}" y="{f.Y(hi):.1f}" width="{f.R - f.L:.1f}" height="{f.Y(lo) - f.Y(hi):.1f}" '
                       f'fill="{GROUP_BAND if gi % 2 == 0 else GROUP_BAND_ALT}"/>')
            f.text(f.R - 8, f.Y(hi) + 15, ROMAN[gi], 11, GROUP_TXT, anchor='end', bold=True)
        f.end_clip()
        _, front = S.pareto_front(pts, x='ms', y='y', quality='q', band=0.05)
        _staircase(f, front, 'ms', 'y')
        pos = _dodge(f, pts, gap=12.0)
        for p in pts:
            x, y = pos[id(p)]
            f.marker(x, y, CORRIDOR_PROJ_MARK[p['proj']], S.ENGINE_COLOUR_DISTINCT[p['engine']],
                     filled=p['eligible'], r=6.5, ew=1.6)
        _label_points(f, pts, pos)
        for q in front:
            f.ring(*pos[id(q)], r=13)
        top.append(f)
        # ---- (c, d): steps to the finish line against time, one frontier per violation group
        gp = [p for p in pts if p['eligible']]
        ylo = math.floor(min(p['steps'] for p in gp) / 10) * 10 - 5
        yhi = math.ceil(max(p['steps'] for p in gp) / 10) * 10 + 5
        g2 = Fig(560, 430, ml=86, mr=18, mt=44, mb=70, font=FONT)
        g2.axes(xr, (ylo, yhi), xlog=True)
        g2.frame(dec_ticks(*xr), [t for t in range(0, 400, 10) if ylo <= t <= yhi], 'time per action [ms] (log)',
                 'control steps to the finish line' if geo == 'tilt' else '',
                 f"({'c' if geo == 'tilt' else 'd'}) {lab}: steps within each group", '', xfmt=fmt_num,
                 yfmt=lambda v: f'{v:.0f}')
        g2.clip_to_box()
        # bands in time order; where two would overlap (hump: group I ends at 72.1 ms, II starts at 77.1) they
        # meet at the geometric mean between the two groups, so every band and its numeral stay visible
        spans = sorted(((min(p['ms'] for p in g), max(p['ms'] for p in g), gi) for gi, g in enumerate(groups)))
        edges = []
        for k, (lo_, hi_, gi) in enumerate(spans):
            x0, x1 = lo_ * 0.92, hi_ * 1.08
            if k > 0:
                x0 = max(x0, math.sqrt(spans[k - 1][1] * lo_))
            if k + 1 < len(spans):
                x1 = min(x1, math.sqrt(hi_ * spans[k + 1][0]))
            edges.append((x0, x1, gi))
        for k, (x0, x1, gi) in enumerate(edges):
            # neighbouring bands alternate two shades and keep a 3 px gap, so two adjacent groups read as two
            g2.vspan(g2.X(x0) + 1.5, g2.X(x1) - 1.5, GROUP_BAND if k % 2 == 0 else GROUP_BAND_ALT)
            g2.text((g2.X(x0) + g2.X(x1)) / 2, g2.T + 16, ROMAN[gi], 11, GROUP_TXT, anchor='middle', bold=True)
        g2.end_clip()
        rings = []
        for g in groups:
            for p in g:
                p['one'] = 1.0
            _, fr = S.pareto_front(g, x='ms', y='steps', quality='one', band=0.05)
            _staircase(g2, fr, 'ms', 'steps')
            rings += fr
        for p in gp:
            p['_y'] = p['y']
            p['y'] = p['steps']                       # _dodge reads 'y'
        pos2 = _dodge(g2, gp, gap=12.0)
        for p in gp:
            x, y = pos2[id(p)]
            g2.marker(x, y, CORRIDOR_PROJ_MARK[p['proj']], S.ENGINE_COLOUR_DISTINCT[p['engine']], filled=True, r=6.5, ew=1.6)
        _label_points(g2, gp, pos2, grey=lambda p: False)
        for q in rings:
            g2.ring(*pos2[id(q)], r=13)
        for p in gp:
            p['y'] = p['_y']
        bot.append(g2)
    from svg.fmpcc_svg import save_grid
    width = 2 * 560 + 10
    path = save_grid(top + bot, os.path.join(outdir, 'fig_uav_corridor_tradeoff.svg'), cols=2, gap=10,
                     header=_corridor_v3_legend(width))
    m = D['meta']
    gtxt = '; '.join(f"{geo}: " + ', '.join(f"{ROMAN[i]} {min(p['y'] for p in g):.1f}-{max(p['y'] for p in g):.1f}"
                                          for i, g in enumerate(groups)) for geo, _, groups in data)
    return path, (f"data/corridor_v3_frontier.json (extract/corridor_v3_frontier.py) | corridor v3 after projection, first "
                  f"{m['n_read']} flights per cell, success at x' = {m['x_clear']} m; (a,b) pareto_front on violating steps "
                  f"and ms, band 0.05 on success; (c,d) pareto_front on steps and ms within each violation group (gap > "
                  f"{VIOL_GAP} violating steps): {gtxt}")



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
    ('fig_aligning_projected_tradeoff', 'da', fig_aligning_projected_tradeoff),
    ('fig_aligning_outcomes', 'da', fig_aligning_outcomes),
    ('fig_uav_corridor_tradeoff', 'da', fig_uav_corridor_tradeoff),
]
