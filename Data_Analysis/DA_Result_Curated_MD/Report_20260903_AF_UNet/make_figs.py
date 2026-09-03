#!/usr/bin/env python3
"""Figures for Report_20260903_AF_UNet.

Dependency-free SVG writer — the AI container has no matplotlib/numpy, so the
canvas class is carried over verbatim from Report_20260819_MF_UNet/make_figs.py.
Pareto semantics match Data_Analysis/Visualizer_VA_v2/index.html:
  * axes          = (avg_time, n_steps), lower is better on both
  * comparability = only points within PARETO_BAND of the best S&C are eligible
  * front         = non-dominated staircase over the eligible points
Run:  python3 make_figs.py [<batch_dir>]
"""
import csv, collections, math, os, sys
import statistics as stt
from xml.sax.saxutils import escape as _esc

BATCH = sys.argv[1] if len(sys.argv) > 1 else \
    '/workspaces/FM-PCC/temp/0309/batch_avoiding_combined_20260903_133730'
OUT = os.path.dirname(os.path.abspath(__file__))
PARETO_BAND, TOL = 0.05, 1e-9
ENVS = ['top-left-hard', 'top-right-hard', 'both-hard']
SEED = '6'

FAM = {'af': '#6c3483', 'mf': '#1e8449', 'fm': '#2471a3', 'dpcc': '#c0392b'}
FAMLAB = {'af': 'α-Flow (AF)', 'mf': 'MeanFlow (MF)',
          'fm': 'naive FM', 'dpcc': 'DPCC diffusion'}
# folder-name match -> (family, tag prefix, label)
RUNS = [(('afon02_s6',),                        'af',   'A', 'AF α→0.2'),
        (('afon005_s6',),                       'af5',  'a', 'AF α→0.05'),
        (('meanflow', 'msg20trials'),           'mf',   'M', 'MF-UNet'),
        (('diffusion.FlowMatchingODE', 'msg20trials'), 'fm', 'F', 'naive FM'),
        (('GaussianDiffusion', '_msg20trials'), 'dpcc', 'D', 'DPCC K20')]
FAMOF = {'af': 'af', 'af5': 'af', 'mf': 'mf', 'fm': 'fm', 'dpcc': 'dpcc'}
VARIANTS = ['dpcc-c-tightened', 'dpcc-t-tightened']
VMARK = {'dpcc-c-tightened': 'o', 'dpcc-t-tightened': 's'}
VSHORT = {'dpcc-c-tightened': 'c', 'dpcc-t-tightened': 't'}
KS = [1, 2, 5, 10, 20]


def load():
    """(fam, K, env, variant) -> {metric: value}, seed 6 only (raw, not aggregated)."""
    D = collections.defaultdict(dict)
    with open(os.path.join(BATCH, 'candidates_multidimensional_raw.csv')) as f:
        for r in csv.DictReader(f):
            if r['seed'] != SEED:
                continue
            fn = r['Folder_Name']
            m = __import__('re').search(r'H8_K(\d+)_', fn)
            if not m:
                continue
            for pats, fam, _tag, _lab in RUNS:
                ok = all(p in fn for p in pats) and (len(pats) == 1 or fn.endswith(pats[-1]))
                if not ok:
                    continue
                try:
                    D[(fam, int(m.group(1)), r['halfspace_variant'], r['variant'])][r['metric']] \
                        = float(r['value'])
                except (TypeError, ValueError):
                    pass
    return D


def cell(D, fam, K, env, var):
    """(avg_time, n_steps, S&C) for one env, or the per-environment mean when env='AGG'."""
    if env != 'AGG':
        d = D.get((fam, K, env, var), {})
        if 'avg_time' not in d or 'n_steps' not in d:
            return None
        return d['avg_time'], d['n_steps'], d.get('n_success_and_constraints')
    ds = [D.get((fam, K, e, var), {}) for e in ENVS]
    if any('avg_time' not in d or 'n_steps' not in d
           or 'n_success_and_constraints' not in d for d in ds):
        return None
    return (stt.mean(d['avg_time'] for d in ds),
            stt.mean(d['n_steps'] for d in ds),
            stt.mean(d['n_success_and_constraints'] for d in ds))


# ───────────────────────── minimal SVG canvas ──────────────────────────────
class Fig:
    def __init__(self, w, h, ml=76, mr=16, mt=44, mb=54):
        self.w, self.h = w, h
        self.L, self.R, self.T, self.B = ml, w - mr, mt, h - mb
        self.s = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>']

    def axes(self, xlim, ylim, xlog=False, ylog=False):
        self.xlog, self.ylog = xlog, ylog
        self.x0, self.x1 = (math.log10(xlim[0]), math.log10(xlim[1])) if xlog else xlim
        self.y0, self.y1 = (math.log10(ylim[0]), math.log10(ylim[1])) if ylog else ylim

    def X(self, v):
        v = math.log10(v) if self.xlog else v
        return self.L + (v - self.x0) / (self.x1 - self.x0) * (self.R - self.L)

    def Y(self, v):
        v = math.log10(v) if self.ylog else v
        return self.B - (v - self.y0) / (self.y1 - self.y0) * (self.B - self.T)

    def frame(self, xt, yt, xlab, ylab, title, sub='', xfmt=str, yfmt=str):
        for v in xt:
            x = self.X(v)
            self.s.append(f'<line x1="{x:.1f}" y1="{self.T}" x2="{x:.1f}" y2="{self.B}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            self.s.append(f'<text x="{x:.1f}" y="{self.B+16}" font-size="10.5" fill="#222" '
                          f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif">'
                          f'{_esc(str(xfmt(v)))}</text>')
        for v in yt:
            y = self.Y(v)
            self.s.append(f'<line x1="{self.L}" y1="{y:.1f}" x2="{self.R}" y2="{y:.1f}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            self.s.append(f'<text x="{self.L-7}" y="{y+3.5:.1f}" font-size="10.5" fill="#222" '
                          f'text-anchor="end" font-family="Helvetica,Arial,sans-serif">'
                          f'{_esc(str(yfmt(v)))}</text>')
        self.s.append(f'<rect x="{self.L}" y="{self.T}" width="{self.R-self.L}" '
                      f'height="{self.B-self.T}" fill="none" stroke="#111" stroke-width="1.2"/>')
        self.s.append(f'<text x="{(self.L+self.R)/2}" y="{self.B+38}" font-size="11.5" fill="#111" '
                      f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif">'
                      f'{_esc(xlab)}</text>')
        my = (self.T + self.B) / 2
        self.s.append(f'<text x="16" y="{my}" font-size="11.5" fill="#111" text-anchor="middle" '
                      f'font-family="Helvetica,Arial,sans-serif" transform="rotate(-90 16 {my})">'
                      f'{_esc(ylab)}</text>')
        self.s.append(f'<text x="{self.L}" y="20" font-size="13" font-weight="bold" fill="#111" '
                      f'font-family="Helvetica,Arial,sans-serif">{_esc(title)}</text>')
        if sub:
            self.s.append(f'<text x="{self.L}" y="35" font-size="10.5" fill="#555" '
                          f'font-family="Helvetica,Arial,sans-serif">{_esc(sub)}</text>')

    def marker(self, x, y, kind, color, filled=True, r=5.5, ew=1.4):
        fill, edge = (color, '#111') if filled else ('#ffffff', color)
        if kind == 'o':
            self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                          f'stroke="{edge}" stroke-width="{ew}"/>')
        else:
            self.s.append(f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r:.1f}" '
                          f'height="{2*r:.1f}" fill="{fill}" stroke="{edge}" stroke-width="{ew}"/>')

    def ring(self, x, y, r=11.0):
        self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="#222" '
                      f'stroke-width="1.1"/>')

    def poly(self, pts, color, dash='', w=1.4):
        d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.s.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{w}"{da}/>')

    def bar(self, x, y, w, h, color, alpha=1.0):
        self.s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                      f'fill="{color}" fill-opacity="{alpha}" stroke="#111" stroke-width="0.8"/>')

    def text(self, x, y, t, size=9.5, color='#333', anchor='start', bold=False):
        b = ' font-weight="bold"' if bold else ''
        self.s.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                      f'text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif"{b}>'
                      f'{_esc(str(t))}</text>')

    def save(self, name):
        p = os.path.join(OUT, name)
        with open(p, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                    f'viewBox="0 0 {self.w} {self.h}">' + ''.join(self.s) + '</svg>')
        print('wrote', os.path.basename(p))


def dec_ticks(lo, hi):
    t, e = [], int(math.floor(math.log10(lo)))
    while 10 ** e <= hi * 1.001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                t.append(v)
        e += 1
    return t


def fmt(v):
    return f'{v:.0f}' if v >= 1 else f'{v:g}'


def legend(f, x, y, entries):
    for i, (col, lab, kind) in enumerate(entries):
        yy = y + i * 17
        f.marker(x + 7, yy, kind, col, filled=True, r=5.0)
        f.text(x + 18, yy + 3.5, lab, 9.5, '#111')


# ─────────────────────────── Pareto panels (fig1-4) ─────────────────────────
def pareto(D, env, fname, title, sub):
    pts = []
    for pats, fam, tag, lab in RUNS:
        for K in KS:
            for var in VARIANTS:
                c = cell(D, fam, K, env, var)
                if c is None or c[2] is None:
                    continue
                pts.append(dict(fam=FAMOF[fam], tag=f'{tag}{K}{VSHORT[var]}', lab=lab,
                                var=var, x=c[0], y=c[1], sc=c[2]))
    if not pts:
        return
    best = max(p['sc'] for p in pts)
    for p in pts:
        p['in'] = p['sc'] >= best - PARETO_BAND - TOL
    front, by = [], None
    for p in sorted((q for q in pts if q['in']), key=lambda q: (q['x'], q['y'])):
        if by is None or p['y'] < by - TOL:
            front.append(p)
            by = p['y']

    ylo = min(p['y'] for p in pts) - 5
    yhi = max(p['y'] for p in pts) + 7
    xlo = min(p['x'] for p in pts) * 0.7
    xhi = max(p['x'] for p in pts) * 1.4
    f = Fig(760, 470, ml=76, mr=176)
    f.axes((xlo, xhi), (ylo, yhi), xlog=True)
    f.frame(dec_ticks(xlo, xhi),
            [t for t in range(50, 130, 5) if ylo <= t <= yhi],
            'avg_time   [ s / control step ]   (log)',
            'n_steps   [ control steps / episode ]', title, sub,
            xfmt=fmt, yfmt=lambda v: f'{v:.0f}')
    if len(front) > 1:
        st = []
        for i, p in enumerate(front):
            st.append((f.X(p['x']), f.Y(p['y'])))
            if i + 1 < len(front):
                st.append((f.X(front[i + 1]['x']), f.Y(p['y'])))
        f.poly(st, '#222', dash='5,4', w=1.3)
    for p in front:
        f.ring(f.X(p['x']), f.Y(p['y']))
    for p in pts:
        f.marker(f.X(p['x']), f.Y(p['y']), VMARK[p['var']], FAM[p['fam']], filled=p['in'])
        f.text(f.X(p['x']) + 8, f.Y(p['y']) - 7, p['tag'], 8.0,
               '#111' if p['in'] else '#aaa', bold=p in front)
    lx = f.R + 14
    f.text(lx, f.T + 4, 'engine', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(FAM[k], FAMLAB[k], 'o') for k in ['af', 'mf', 'fm', 'dpcc']])
    f.text(lx, f.T + 104, 'selection rule', 10, '#111', bold=True)
    legend(f, lx, f.T + 120, [('#777', 'dpcc-c-tightened', 'o'),
                              ('#777', 'dpcc-t-tightened', 's')])
    f.text(lx, f.T + 168, 'hollow = outside', 9, '#555')
    f.text(lx, f.T + 180, f'S&C band ({PARETO_BAND})', 9, '#555')
    f.text(lx, f.T + 198, 'ring = on the', 9, '#555')
    f.text(lx, f.T + 210, 'Pareto front', 9, '#555')
    f.save(fname)


# ───────────────────────── fig5: the K ladder ───────────────────────────────
def k_ladder(D, fname):
    f = Fig(760, 470, ml=76, mr=176)
    f.axes((0.8, 26), (52, 76), xlog=True)
    f.frame([1, 2, 5, 10, 20], list(range(55, 80, 5)),
            'K   [ network evaluations per plan ]   (log)',
            'n_steps   [ control steps / episode ]',
            'Fig 5 — K ladder on `dpcc-t-tightened`, per-environment mean',
            'seed 6, n_trials = 20. Filled = S&C 1.00 on all three environments. '
            'DPCC on `dpcc-c-tightened` (no `-t-` row on both-hard).',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.0f}')
    for pats, fam, tag, lab in RUNS:
        # DPCC has no `dpcc-t-tightened` row on `both-hard`; it is plotted on
        # `dpcc-c-tightened`, which is also the rule its pinned target uses.
        rule = 'dpcc-c-tightened' if fam == 'dpcc' else 'dpcc-t-tightened'
        pts = []
        for K in KS:
            c = cell(D, fam, K, 'AGG', rule)
            if c is None:
                continue
            pts.append((K, c[1], c[2]))
        if not pts:
            continue
        f.poly([(f.X(k), f.Y(y)) for k, y, _ in pts], FAM[FAMOF[fam]], w=1.8)
        for k, y, sc in pts:
            f.marker(f.X(k), f.Y(y), 's', FAM[FAMOF[fam]], filled=(sc >= 1.0 - TOL), r=5.0)
    lx = f.R + 14
    f.text(lx, f.T + 4, 'engine', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(FAM[k], FAMLAB[k], 's') for k in ['af', 'mf', 'fm', 'dpcc']])
    f.text(lx, f.T + 96, 'AF α→0.05 shares', 9, '#555')
    f.text(lx, f.T + 108, 'the α-Flow colour.', 9, '#555')
    f.text(lx, f.T + 126, 'hollow = S&C < 1.00', 9, '#555')
    f.save(fname)


# ────────────── fig6: the ordering — cheapest S&C = 1.00 row ────────────────
def ordering(D, fname):
    ARMS = ['dpcc-r', 'dpcc-r-tightened', 'dpcc-c', 'dpcc-c-tightened',
            'dpcc-t', 'dpcc-t-tightened']
    best = {}
    for pats, fam, tag, lab in RUNS:
        for env in ENVS:
            b = None
            for var in ARMS:
                for K in KS:
                    d = D.get((fam, K, env, var), {})
                    if not {'avg_time', 'n_steps', 'n_success_and_constraints'} <= set(d):
                        continue
                    if d['n_success_and_constraints'] < 1.0 - TOL:
                        continue
                    c = (d['n_steps'] * d['avg_time'], d['n_steps'], var, K)
                    if b is None or c < b:
                        b = c
            best[(fam, env)] = b
    order = ['af', 'af5', 'mf', 'fm', 'dpcc']
    f = Fig(760, 470, ml=86, mr=176, mb=64)
    f.axes((0, 3), (0.6, 60), ylog=True)
    f.frame([], [1, 2, 5, 10, 20, 50],
            'environment', 's / episode   at S&C = 1.00   (log, lower is better)',
            'Fig 6 — cheapest row that reaches S&C 1.00, per engine',
            'seed 6, n_trials = 20; best over all dpcc-* selection rules and all K.',
            yfmt=lambda v: f'{v:g}')
    bw = (f.R - f.L) / 3 / (len(order) + 1.2)
    for ei, env in enumerate(ENVS):
        x0 = f.L + ei * (f.R - f.L) / 3 + bw * 0.6
        f.text(x0 + bw * len(order) / 2, f.B + 16, env, 10, '#111', anchor='middle')
        for oi, fam in enumerate(order):
            b = best.get((fam, env))
            if b is None:
                continue
            x = x0 + oi * bw
            y = f.Y(b[0])
            f.bar(x, y, bw * 0.86, f.B - y, FAM[FAMOF[fam]],
                  alpha=0.55 if fam == 'af5' else 1.0)
            f.text(x + bw * 0.43, y - 5, f'{b[0]:.2f}', 8.0, '#111', anchor='middle', bold=True)
            f.text(x + bw * 0.43, y - 15, f'K{b[3]}', 7.5, '#666', anchor='middle')
    lx = f.R + 14
    f.text(lx, f.T + 4, 'engine (left→right)', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(FAM['af'], 'AF α→0.2', 'o'), (FAM['af'], 'AF α→0.05 (pale)', 'o'),
                             (FAM['mf'], 'MF-UNet', 'o'), (FAM['fm'], 'naive FM', 'o'),
                             (FAM['dpcc'], 'DPCC K20', 'o')])
    f.text(lx, f.T + 122, 'bar label = s/ep;', 9, '#555')
    f.text(lx, f.T + 134, 'grey label = the K', 9, '#555')
    f.text(lx, f.T + 146, 'that achieved it.', 9, '#555')
    f.save(fname)


# ────────── fig7: raw plan quality on the unprojected `diffuser` arm ─────────
def raw_arm(D, fname):
    f = Fig(760, 470, ml=86, mr=176, mb=64)
    f.axes((0, 3), (55, 70))
    f.frame([], list(range(56, 70, 2)),
            'environment', 'n_steps   [ control steps / episode ]',
            'Fig 7 — RAW network output (`diffuser`, no projection), K = 1',
            'seed 6, n_trials = 20. Bar label = goal reached. DPCC shown at its own K = 20.',
            yfmt=lambda v: f'{v:.0f}')
    order = ['af', 'af5', 'mf', 'fm', 'dpcc']
    bw = (f.R - f.L) / 3 / (len(order) + 1.2)
    for ei, env in enumerate(ENVS):
        x0 = f.L + ei * (f.R - f.L) / 3 + bw * 0.6
        f.text(x0 + bw * len(order) / 2, f.B + 16, env, 10, '#111', anchor='middle')
        for oi, fam in enumerate(order):
            K = 20 if fam == 'dpcc' else 1
            d = D.get((fam, K, env, 'diffuser'), {})
            if 'n_steps' not in d:
                continue
            x, y = x0 + oi * bw, f.Y(d['n_steps'])
            f.bar(x, y, bw * 0.86, f.B - y, FAM[FAMOF[fam]],
                  alpha=0.55 if fam == 'af5' else 1.0)
            f.text(x + bw * 0.43, y - 5, f"{d['n_success']:.2f}", 8.0, '#111',
                   anchor='middle', bold=True)
            f.text(x + bw * 0.43, y - 15, f"{d['n_steps']:.1f}", 7.5, '#666', anchor='middle')
    lx = f.R + 14
    f.text(lx, f.T + 4, 'engine (left→right)', 10, '#111', bold=True)
    legend(f, lx, f.T + 20, [(FAM['af'], 'AF α→0.2', 'o'), (FAM['af'], 'AF α→0.05 (pale)', 'o'),
                             (FAM['mf'], 'MF-UNet', 'o'), (FAM['fm'], 'naive FM', 'o'),
                             (FAM['dpcc'], 'DPCC K20', 'o')])
    f.text(lx, f.T + 122, 'bold label = goal', 9, '#555')
    f.text(lx, f.T + 134, 'reached (of 1.00);', 9, '#555')
    f.text(lx, f.T + 146, 'grey = n_steps.', 9, '#555')
    f.save(fname)


if __name__ == '__main__':
    D = load()
    pareto(D, 'top-left-hard', 'fig1_pareto_top-left-hard.svg',
           'Fig 1 — Pareto front, `top-left-hard`', 'seed 6, n_trials = 20; every engine × K × rule.')
    pareto(D, 'top-right-hard', 'fig2_pareto_top-right-hard.svg',
           'Fig 2 — Pareto front, `top-right-hard`', 'seed 6, n_trials = 20; every engine × K × rule.')
    pareto(D, 'both-hard', 'fig3_pareto_both-hard.svg',
           'Fig 3 — Pareto front, `both-hard`', 'seed 6, n_trials = 20; every engine × K × rule.')
    pareto(D, 'AGG', 'fig4_pareto_aggregate.svg',
           'Fig 4 — Pareto front, mean over the three environments',
           'seed 6, n_trials = 20; per-environment mean of avg_time, n_steps and S&C.')
    k_ladder(D, 'fig5_k_ladder.svg')
    ordering(D, 'fig6_ordering_sep_at_sc1.svg')
    raw_arm(D, 'fig7_raw_diffuser_K1.svg')
