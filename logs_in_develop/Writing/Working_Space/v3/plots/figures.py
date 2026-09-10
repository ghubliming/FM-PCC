#!/usr/bin/env python3
"""Figure builders for thesis v3. One function per thesis figure.

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
``make_figs.py``. Nothing else needs to change.
"""
import os

import sources as S
from fmpcc_svg import Fig, dec_ticks, fmt_num, legend

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
#  Fig 1-4 -- the cost/quality frontier on the state-based benchmark
# ═══════════════════════════════════════════════════════════════════════════
def _pareto_panel(outdir, geometry, fname, title):
    c, data = _avoiding_cells()
    if c is None:
        return None
    rows = data[geometry]
    pts = []
    for (eng, K, rule), r in rows.items():
        if rule not in RULES:
            continue
        pts.append(dict(engine=eng, K=K, rule=rule,
                        tag=f'{eng[0].upper()}{K}{RULE_SHORT[rule]}',
                        avg_time=r['avg_time'], n_steps=r['n_steps'],
                        n_success_and_constraints=r['n_success_and_constraints']))
    if not pts:
        return None
    pts, front = S.pareto_front(pts, band=PARETO_BAND)

    ylo = min(p['n_steps'] for p in pts) - 5
    yhi = max(p['n_steps'] for p in pts) + 8
    xlo = min(p['avg_time'] for p in pts) * 0.7
    xhi = max(p['avg_time'] for p in pts) * 1.5
    f = Fig(760, 470, ml=76, mr=190)
    f.axes((xlo, xhi), (ylo, yhi), xlog=True)
    sub = (f'{c.protocol}; geometry-mean over '
           f'{"all three geometries" if geometry == "AGG" else geometry}. '
           f'Backbone U-Net 4.0M throughout.')
    f.frame(dec_ticks(xlo, xhi),
            [t for t in range(40, 200, 5) if ylo <= t <= yhi],
            'per-step wall clock  [ s ]   (log, lower is better)',
            'control steps to task completion   (lower is better)',
            title, sub, xfmt=fmt_num, yfmt=lambda v: f'{v:.0f}')

    if len(front) > 1:
        st = []
        for i, p in enumerate(front):
            st.append((f.X(p['avg_time']), f.Y(p['n_steps'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['avg_time']), f.Y(p['n_steps'])))
        f.poly(st, '#222', dash='5,4', w=1.3)
    for p in front:
        f.ring(f.X(p['avg_time']), f.Y(p['n_steps']))
    for p in pts:
        x, y = f.X(p['avg_time']), f.Y(p['n_steps'])
        f.marker(x, y, RULE_MARK[p['rule']], S.ENGINE_COLOUR[p['engine']], filled=p['eligible'])
        f.text(x + 8, y - 7, p['tag'], 8.0,
               '#111' if p['eligible'] else '#aaa', bold=p in front)
        if (p['engine'], p['K'], p['rule']) == TARGET:
            f.text(x + 8, y + 12, 'pinned baseline', 8.0, '#c0392b', bold=True)

    lx = f.R + 14
    f.text(lx, f.T + 4, 'engine', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(S.ENGINE_COLOUR[k], S.ENGINE_LABEL[k], 'o')
                             for k in ('mf', 'fm', 'diffusion')])
    f.text(lx, f.T + 88, 'selection rule', 10, '#111', bold=True)
    legend(f, lx, f.T + 104, [('#777', 'cumulative cost', 'o'),
                              ('#777', 'temporal consistency', 's')])
    f.text(lx, f.T + 148, 'tag = engine, K, rule', 9, '#555')
    f.text(lx, f.T + 166, f'hollow = outside the', 9, '#555')
    f.text(lx, f.T + 178, f'{PARETO_BAND:g} success band', 9, '#555')
    f.text(lx, f.T + 196, 'ring = on the frontier', 9, '#555')
    path = f.save(os.path.join(outdir, fname))
    return path, f'{c.rel} | {c.protocol}'


def fig_avoiding_pareto_aggregate(outdir):
    return _pareto_panel(outdir, 'AGG', 'fig_avoiding_pareto_aggregate.svg',
                         'Cost/quality frontier, state-based manipulation (all geometries)')


def fig_avoiding_pareto_tl(outdir):
    return _pareto_panel(outdir, 'top-left-hard', 'fig_avoiding_pareto_top-left-hard.svg',
                         'Cost/quality frontier, state-based manipulation (top-left-hard)')


def fig_avoiding_pareto_tr(outdir):
    return _pareto_panel(outdir, 'top-right-hard', 'fig_avoiding_pareto_top-right-hard.svg',
                         'Cost/quality frontier, state-based manipulation (top-right-hard)')


def fig_avoiding_pareto_bh(outdir):
    return _pareto_panel(outdir, 'both-hard', 'fig_avoiding_pareto_both-hard.svg',
                         'Cost/quality frontier, state-based manipulation (both-hard)')


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
    f = Fig(760, 470, ml=76, mr=190)
    f.axes((0.8, 26), (0.55, 1.04), xlog=True)
    f.frame([1, 2, 5, 10, 20], [0.6, 0.7, 0.8, 0.9, 1.0],
            'step budget K  [ network evaluations per plan ]   (log)',
            'success and constraint satisfaction',
            'Quality against step budget, state-based manipulation',
            f'{c.protocol}; geometry-mean over all three geometries, '
            f'temporal-consistency rule, tightened. U-Net 4.0M throughout.',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.2f}')

    # The baseline at the published 10-episode protocol: the only measurement of
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
    f.text(lx, f.T + 4, 'engine', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(S.ENGINE_COLOUR[k], S.ENGINE_LABEL[k], 's')
                             for k in ('mf', 'fm', 'diffusion')])
    f.text(lx, f.T + 84, 'dashed + hollow circles:', 9, '#555')
    f.text(lx, f.T + 96, 'the baseline at the', 9, '#555')
    f.text(lx, f.T + 108, 'PUBLISHED protocol,', 9, '#555')
    f.text(lx, f.T + 120, '10 episodes per cell.', 9, '#555')
    f.text(lx, f.T + 132, 'Mechanism, not a', 9, '#555')
    f.text(lx, f.T + 144, 'powered comparison.', 9, '#555')
    f.text(lx, f.T + 168, 'K is inference-time', 9.5, '#111', bold=True)
    f.text(lx, f.T + 180, 'for transport and', 9.5, '#111', bold=True)
    f.text(lx, f.T + 192, 'training-time for', 9.5, '#111', bold=True)
    f.text(lx, f.T + 204, 'diffusion.', 9.5, '#111', bold=True)
    path = f.save(os.path.join(outdir, 'fig_avoiding_k_ladder.svg'))
    return path, f'{c.rel} | {c.protocol}'


# ═══════════════════════════════════════════════════════════════════════════
#  Fig 6 -- the two constraint arms, and where their costs cross
# ═══════════════════════════════════════════════════════════════════════════
def fig_avoiding_projector_cost(outdir):
    """Per-step cost of iterate projection vs. endpoint projection against K.

    The shaded band is the regime where the comparison is not well posed: at
    K = 2 with a fully-open activation threshold the endpoint arm executes one
    genuine step, so it is not running the mechanism it names. Greying it is the
    figure's job, not the caption's.
    """
    c = S.CORPORA['avoiding_minK']
    if not c.available:
        return None
    cells = S.load_by_candidate(c, ['hfmink_A1_mfunet'])
    rows = S.geometry_mean(cells, lambda k: (k[0], k[3]), geom_index=2)  # -> (K, variant)

    ARMS = [('dpcc-t-tightened', '#2471a3', 'per-step iterate projection', 'o'),
            ('hardflow_sls-t-tightened', '#1e8449', 'in-ODE endpoint projection', 's')]
    ks = sorted({k for k, _v in rows})
    if not ks:
        return None

    f = Fig(760, 470, ml=80, mr=196)
    ymax = max(r['avg_time'] for r in rows.values()) * 1.35
    f.axes((1.6, 6.0), (0.02, ymax), ylog=True)
    f.vspan(f.X(1.6), f.X(2.5))
    f.frame([2, 3, 5], dec_ticks(0.02, ymax),
            'step budget K',
            'per-step wall clock  [ s ]   (log, lower is better)',
            'Cost of the two constraint arms, state-based manipulation',
            f'{c.protocol}; MeanFlow U-Net 4.0M, activation threshold 1.0, '
            f'temporal-consistency rule, tightened.',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:g}')
    f.text(f.X(2.0), f.T + 14, 'K = 2:', 9.0, '#a04000', anchor='middle', bold=True)
    f.text(f.X(2.0), f.T + 26, 'one genuine', 9.0, '#a04000', anchor='middle')
    f.text(f.X(2.0), f.T + 38, 'step -- not', 9.0, '#a04000', anchor='middle')
    f.text(f.X(2.0), f.T + 50, 'citable', 9.0, '#a04000', anchor='middle')

    for var, col, lab, mk in ARMS:
        pts = [(K, rows[(K, var)]['avg_time']) for K in ks if (K, var) in rows]
        if not pts:
            continue
        f.poly([(f.X(K), f.Y(t)) for K, t in pts], col, w=1.9)
        for K, t in pts:
            f.marker(f.X(K), f.Y(t), mk, col, filled=(K >= 3), r=5.5)
            f.text(f.X(K) + 9, f.Y(t) - 6, f'{t * 1000:.0f} ms', 8.0, '#111')

    lx = f.R + 14
    f.text(lx, f.T + 4, 'constraint arm', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(c_, l, m) for _v, c_, l, m in ARMS])
    f.text(lx, f.T + 62, 'Same solver, same', 9, '#555')
    f.text(lx, f.T + 74, 'constraint set, same', 9, '#555')
    f.text(lx, f.T + 86, 'checkpoint, one job.', 9, '#555')
    f.text(lx, f.T + 104, 'The endpoint arm does', 9, '#555')
    f.text(lx, f.T + 116, 'MORE solves and costs', 9, '#555')
    f.text(lx, f.T + 128, 'LESS: it is projecting', 9, '#555')
    f.text(lx, f.T + 140, 'a near-feasible point.', 9, '#555')
    f.text(lx, f.T + 164, 'hollow = degenerate', 9, '#a04000')
    path = f.save(os.path.join(outdir, 'fig_avoiding_projector_cost.svg'))
    return path, f'{c.rel} | {c.protocol}'


# Registry. Order is the order they appear in the thesis.
ALL = [
    ('fig_avoiding_pareto_aggregate', fig_avoiding_pareto_aggregate),
    ('fig_avoiding_pareto_top-left-hard', fig_avoiding_pareto_tl),
    ('fig_avoiding_pareto_top-right-hard', fig_avoiding_pareto_tr),
    ('fig_avoiding_pareto_both-hard', fig_avoiding_pareto_bh),
    ('fig_avoiding_k_ladder', fig_avoiding_k_ladder),
    ('fig_avoiding_projector_cost', fig_avoiding_projector_cost),
]
