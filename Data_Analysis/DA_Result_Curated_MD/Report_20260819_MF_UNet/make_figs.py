#!/usr/bin/env python3
"""Figures for Report_20260819_MF_UNet.

Dependency-free SVG writer: the AI container has no matplotlib, so the DAv3 /
Visualizer_VA_v2 plotting logic is reimplemented here rather than imported.
Pareto semantics are copied from Data_Analysis/Visualizer_VA_v2/index.html:
  * axes          = (avg_time, n_steps), lower is better on both
  * comparability = only points within PARETO_BAND of the best S&C are eligible
  * front         = non-dominated staircase over the eligible points (steps-post)
One panel per halfspace environment plus an aggregate (per-environment mean).
Run:  python3 make_figs.py [<batch_dir>]
"""
import csv, collections, math, os, sys
import statistics as stt
from xml.sax.saxutils import escape as _esc

BATCH = sys.argv[1] if len(sys.argv) > 1 else \
    '/workspaces/FM-PCC/temp/1808/batch_avoiding_combined_20260818_152911'
OUT = os.path.dirname(os.path.abspath(__file__))
PARETO_BAND, TOL = 0.05, 1e-9
ENVS = ['top-left-hard', 'top-right-hard', 'both-hard']

FAM = {'dpcc': '#c0392b', 'fm': '#2471a3', 'mf': '#1e8449'}
# candidate -> (family, tag, label, K); every run below is n_trials = 2, 5 seeds
RUNS = [('8',   'dpcc', 'D1',  'DPCC K1',      1),
        ('7',   'dpcc', 'D10', 'DPCC K10',     10),
        ('15',  'dpcc', 'D20', 'DPCC K20',     20),
        ('156', 'fm',   'F20', 'naive FM K20', 20),
        ('138', 'mf',   'M1',  'MF-UNet K1',   1),
        ('142', 'mf',   'M2',  'MF-UNet K2',   2),
        ('147', 'mf',   'M5',  'MF-UNet K5',   5),
        ('135', 'mf',   'M10', 'MF-UNet K10',  10)]
VARIANTS = ['dpcc-c-tightened', 'dpcc-t-tightened']
VMARK = {'dpcc-c-tightened': 'o', 'dpcc-t-tightened': 's'}
VSHORT = {'dpcc-c-tightened': 'c', 'dpcc-t-tightened': 't'}


def load():
    D = collections.defaultdict(dict)
    with open(os.path.join(BATCH, 'candidates_multidimensional_aggregated.csv')) as f:
        for r in csv.DictReader(f):
            try:
                v = float(r['mean'])
            except (TypeError, ValueError):
                continue
            D[(r['Candidate'], r['halfspace_variant'], r['variant'])][r['metric']] = v
    return D


def cell(D, cand, env, var):
    """(avg_time, n_steps, S&C) for one env, or the per-environment mean when env='AGG'."""
    if env != 'AGG':
        d = D.get((cand, env, var), {})
        if 'avg_time' not in d or 'n_steps' not in d:
            return None
        return d['avg_time'], d['n_steps'], d.get('n_success_and_constraints')
    ds = [D.get((cand, e, var), {}) for e in ENVS]
    if any('avg_time' not in d or 'n_steps' not in d for d in ds):
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


# ─────────────────────────── the Pareto panel ───────────────────────────────
def pareto(D, env, fname, title, sub):
    pts = []
    for cand, fam, tag, lab, K in RUNS:
        for var in VARIANTS:
            c = cell(D, cand, env, var)
            if c is None:
                continue
            pts.append(dict(fam=fam, tag=f'{tag}{VSHORT[var]}', lab=lab, var=var,
                            x=c[0], y=c[1], sc=c[2]))
    best = max(p['sc'] for p in pts if p['sc'] is not None)
    for p in pts:
        p['in'] = p['sc'] is not None and p['sc'] >= best - PARETO_BAND - TOL
    front, by = [], None
    for p in sorted((q for q in pts if q['in']), key=lambda q: (q['x'], q['y'])):
        if by is None or p['y'] < by - TOL:
            front.append(p)
            by = p['y']

    ylo = min(p['y'] for p in pts) - 6
    yhi = max(p['y'] for p in pts) + 8
    f = Fig(720, 450, ml=76, mr=168)
    f.axes((0.013, 0.85), (ylo, yhi), xlog=True)
    f.frame(dec_ticks(0.013, 0.85), [t for t in range(50, 110, 10) if ylo <= t <= yhi],
            'avg_time   [ s / control step ]   (log)', 'n_steps   [ control steps / episode ]',
            title, sub, xfmt=fmt, yfmt=lambda v: f'{v:.0f}')
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
        f.text(f.X(p['x']) + 8, f.Y(p['y']) - 7, p['tag'], 8.5,
               '#111' if p['in'] else '#999', bold=p in front)

    lx, ly = f.R + 12, f.T + 10
    f.text(lx, ly, 'runs  (all n_trials = 2)', 9.5, '#111', bold=True)
    for i, (cand, fam, tag, lab, K) in enumerate(RUNS):
        f.marker(lx + 5, ly + 15 + i * 15, 'o', FAM[fam], r=4.5)
        f.text(lx + 14, ly + 18 + i * 15, f'{tag} = {lab}', 8.5, '#222')
    y2 = ly + 15 + len(RUNS) * 15 + 12
    f.text(lx, y2, 'selection rule', 9.5, '#111', bold=True)
    f.marker(lx + 5, y2 + 15, 'o', '#666', r=4.5)
    f.text(lx + 14, y2 + 18, 'c = min-proj-cost', 8.5, '#222')
    f.marker(lx + 5, y2 + 31, 's', '#666', r=4.5)
    f.text(lx + 14, y2 + 34, 't = temporal-consist.', 8.5, '#222')
    f.marker(lx + 5, y2 + 55, 'o', '#666', filled=False, r=4.5)
    f.text(lx + 14, y2 + 58, f'S&C < {best - PARETO_BAND:.2f} (excl.)', 8.5, '#222')
    f.ring(lx + 5, y2 + 78, 8)
    f.text(lx + 14, y2 + 81, 'on Pareto front', 8.5, '#222')
    f.save(fname)
    return front, best


# ───────────────────────── the K ladder (S&C, avg_time) ─────────────────────
def k_ladder(D):
    ser = {'dpcc': [('8', 1), ('7', 10), ('15', 20)],
           'mf':   [('138', 1), ('142', 2), ('147', 5), ('135', 10)],
           'fm':   [('156', 20)]}
    var = {'dpcc': 'dpcc-c-tightened', 'mf': 'dpcc-t-tightened', 'fm': 'dpcc-c-tightened'}
    name = {'dpcc': 'DPCC (diffusion)', 'mf': 'MF-UNet', 'fm': 'naive FM'}
    f = Fig(720, 600, ml=76, mr=168, mb=54)
    f.B = 296
    f.axes((0.85, 24), (0.10, 1.06), xlog=True)
    f.frame([1, 2, 5, 10, 20], [0.2, 0.4, 0.6, 0.8, 1.0], '', 'S&C   (mean of 3 environments)',
            'Fig 5 — the K (NFE) ladder, aggregated over the three environments',
            'K is inference-only for MF/FM; a TRAINING parameter for diffusion. All n_trials = 2.',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:.1f}')
    for fam, items in ser.items():
        pp = [(f.X(K), f.Y(cell(D, c, 'AGG', var[fam])[2])) for c, K in items]
        if len(pp) > 1:
            f.poly(pp, FAM[fam], w=1.8)
        for x, y in pp:
            f.marker(x, y, 'o', FAM[fam], r=5.5)
    f.text(f.X(1) + 10, f.Y(0.63) + 4, 'diffusion degrades at K=1', 9.5, FAM['dpcc'], bold=True)
    f.text(f.X(1) + 10, f.Y(0.97) - 12, 'MF-UNet holds S&C at K=1', 9.5, FAM['mf'], bold=True)

    f.T, f.B = 348, 546
    f.axes((0.85, 24), (0.014, 0.75), xlog=True, ylog=True)
    f.frame([1, 2, 5, 10, 20], [0.02, 0.05, 0.1, 0.2, 0.5],
            'K   —   sampling steps / NFE per plan   (log)', 'avg_time  [ s / step ]  (log)', '', '',
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v:g}')
    for fam, items in ser.items():
        pp = [(f.X(K), f.Y(cell(D, c, 'AGG', var[fam])[0])) for c, K in items]
        if len(pp) > 1:
            f.poly(pp, FAM[fam], w=1.8)
        for x, y in pp:
            f.marker(x, y, 'o', FAM[fam], r=5.5)
    lx, ly = f.R + 12, 360
    for i, k in enumerate(['dpcc', 'mf', 'fm']):
        f.marker(lx + 5, ly + i * 18, 'o', FAM[k], r=4.5)
        f.text(lx + 14, ly + 3 + i * 18, name[k], 9.5, '#222')
    f.text(lx, ly + 68, 'selection rule shown:', 9, '#111')
    f.text(lx, ly + 82, 'DPCC / FM  = min-proj-cost', 8.5, '#555')
    f.text(lx, ly + 94, 'MF-UNet    = temporal-cons.', 8.5, '#555')
    f.text(lx, ly + 106, '(each its own best)', 8.5, '#555')
    f.save('fig5_k_ladder.svg')


if __name__ == '__main__':
    D = load()
    spec = [('top-left-hard',  'fig1_pareto_top-left-hard.svg',  'Fig 1'),
            ('top-right-hard', 'fig2_pareto_top-right-hard.svg', 'Fig 2'),
            ('both-hard',      'fig3_pareto_both-hard.svg',      'Fig 3'),
            ('AGG',            'fig4_pareto_aggregate.svg',      'Fig 4')]
    for env, fname, n in spec:
        nice = 'aggregate — mean of the three environments' if env == 'AGG' else env
        fr, best = pareto(D, env, fname, f'{n} — Pareto front, {nice}',
                          f'lower is better on both axes; eligible = S&C within '
                          f'{PARETO_BAND} of the best S&C in the panel')
        print('     front:', ' -> '.join(f"{p['tag']} (t={p['x']:.4f}, s={p['y']:.1f}, "
                                         f"S&C={p['sc']:.2f})" for p in fr))
    k_ladder(D)
