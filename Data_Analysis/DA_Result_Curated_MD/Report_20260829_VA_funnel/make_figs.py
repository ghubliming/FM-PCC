#!/usr/bin/env python3
"""Figures for Report_20260829_VA_funnel (visual-aligning, three-stage funnel).

Dependency-free SVG writer: no matplotlib in the AI container. Every number is
recomputed from the batch's per_rollout_detail.csv, so the figures and the
README cannot drift apart.

Lead metric is context_final_xy_dist (raw box->target XY distance, metres).
mean_dist_per_rollout is NOT used anywhere: it is 0.5*(pos + rot/pi), a blend.

Run:  python3 make_figs.py [<batch_dir>]
"""
import csv, collections, math, os, sys
import statistics as stt
from xml.sax.saxutils import escape as _esc

BATCH = sys.argv[1] if len(sys.argv) > 1 else \
    '/workspaces/FM-PCC/temp/2508/batch_va2_20260823_135156'
OUT = os.path.dirname(os.path.abspath(__file__))
NEAR = 0.15            # "near the goal"
TRAIN, UNT, TGT = 'train', 'combined_5', 'combined_5-tightened'
ARMB = ['dpcc-r', 'dpcc-c', 'dpcc-t', 'post_processing']
ARMC = ['hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t']
FAM = {'mf': '#1e8449', 'af': '#7d3c98', 'fm': '#2471a3', 'dz': '#c0392b', 'bl': '#7f8c8d'}


# ───────────────────────────── data ────────────────────────────────────────
def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    cells = collections.defaultdict(dict)
    with open(os.path.join(BATCH, 'per_rollout_detail.csv')) as f:
        for r in csv.DictReader(f):
            k = (round(fl(r['context_box_init_xy_x']), 4),
                 round(fl(r['context_box_init_xy_y']), 4))
            cells[(r['Candidate'], r['split'], r['geo'], r['variant'])][k] = r
    return cells


def S(d):
    """Funnel stats for one cell, on raw XY distance."""
    rs = list(d.values())
    n = len(rs)
    f = [fl(r['context_final_xy_dist']) for r in rs]
    i0 = [fl(r['context_init_xy_dist']) for r in rs]
    zv = [fl(r['collision_free_completed']) == 1.0 for r in rs]
    near = [j for j in range(n) if f[j] <= NEAR]
    return dict(n=n, med=stt.median(f), frac=stt.median(f[j] / i0[j] for j in range(n)),
                unt=sum(1 for j in range(n) if abs(f[j] - i0[j]) < 0.005) / n,
                near=len(near), nc=sum(1 for j in near if zv[j]),
                c5=sum(1 for x in f if x <= 0.05),
                zv=sum(zv) / n,
                viol=stt.mean(fl(r['constraint_exec_total_viol_count']) for r in rs),
                ms=stt.mean(fl(r['avg_time_ms']) for r in rs))


def cell(C, c, sp, g, v):
    d = C.get((c, sp, g, v))
    return S(d) if d and len(d) == 30 else None


def bestB(C, c, sp, g, arm=ARMB):
    """Each model's OWN best projector, chosen on the constraint-clean near tail."""
    got = [(v, cell(C, c, sp, g, v)) for v in arm]
    got = [(v, s) for v, s in got if s]
    if not got:
        return None
    got.sort(key=lambda x: (-x[1]['nc'], x[1]['med']))
    return got[0]


# candidate -> (family, engine label, K, backbone)
ENG = {'14': ('mf', 'MeanFlow', 2), '6': ('af', 'AlphaFlow', 2),
       '11': ('fm', 'FlowMatching', 20), '9': ('dz', 'Diffusion aw10', 20),
       '13': ('mf', 'MeanFlow', 100), '5': ('af', 'AlphaFlow', 100),
       '10': ('fm', 'FlowMatching', 100), '8': ('dz', 'Diffusion', 100)}
V1 = ['14', '6', '11', '9']          # matched bone: UNet FiLM v1, 4.04 M
SHORT = {'mf': 'MF', 'af': 'AF', 'fm': 'FM', 'dz': 'DIFF', 'bl': 'BASE'}


# ────────────────────────── minimal SVG canvas ─────────────────────────────
class Fig:
    def __init__(self, w, h, ml=76, mr=16, mt=46, mb=54):
        self.w, self.h = w, h
        self.L, self.R, self.T, self.B = ml, w - mr, mt, h - mb
        self.s = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>']
        self.xlog = self.ylog = False

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

    def grid(self, xt=(), yt=(), xfmt=str, yfmt=str, xtick=True, ytick=True):
        for v in xt:
            x = self.X(v)
            self.s.append(f'<line x1="{x:.1f}" y1="{self.T}" x2="{x:.1f}" y2="{self.B}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            if xtick:
                self.text(x, self.B + 16, xfmt(v), 10.5, '#222', 'middle')
        for v in yt:
            y = self.Y(v)
            self.s.append(f'<line x1="{self.L}" y1="{y:.1f}" x2="{self.R}" y2="{y:.1f}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            if ytick:
                self.text(self.L - 7, y + 3.5, yfmt(v), 10.5, '#222', 'end')

    def frame(self, xlab, ylab, title, sub=''):
        self.s.append(f'<rect x="{self.L}" y="{self.T}" width="{self.R-self.L}" '
                      f'height="{self.B-self.T}" fill="none" stroke="#111" stroke-width="1.2"/>')
        if xlab:
            self.text((self.L + self.R) / 2, self.B + 40, xlab, 11.5, '#111', 'middle')
        if ylab:
            my = (self.T + self.B) / 2
            self.s.append(f'<text x="16" y="{my}" font-size="11.5" fill="#111" '
                          f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif" '
                          f'transform="rotate(-90 16 {my})">{_esc(ylab)}</text>')
        self.text(self.L, 20, title, 13, '#111', 'start', bold=True)
        if sub:
            self.text(self.L, 36, sub, 10.5, '#555')

    def rect(self, x, y, w, h, fill, stroke='none', sw=1.0, op=1.0):
        self.s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" '
                      f'height="{max(h,0):.1f}" fill="{fill}" fill-opacity="{op}" '
                      f'stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, color='#111', w=1.2, dash=''):
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{color}" stroke-width="{w}"{da}/>')

    def marker(self, x, y, kind, color, filled=True, r=6.0, ew=1.4):
        fill, edge = (color, '#111') if filled else ('#ffffff', color)
        if kind == 'o':
            self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                          f'stroke="{edge}" stroke-width="{ew}"/>')
        elif kind == 's':
            self.s.append(f'<rect x="{x-r:.1f}" y="{y-r:.1f}" width="{2*r:.1f}" '
                          f'height="{2*r:.1f}" fill="{fill}" stroke="{edge}" stroke-width="{ew}"/>')
        else:
            p = f'{x:.1f},{y-r-1:.1f} {x+r+1:.1f},{y:.1f} {x:.1f},{y+r+1:.1f} {x-r-1:.1f},{y:.1f}'
            self.s.append(f'<polygon points="{p}" fill="{fill}" stroke="{edge}" '
                          f'stroke-width="{ew}"/>')

    def text(self, x, y, t, size=9.5, color='#333', anchor='start', bold=False):
        b = ' font-weight="bold"' if bold else ''
        self.s.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                      f'text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif"{b}>'
                      f'{_esc(str(t))}</text>')

    def save(self, name):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                    f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                    + ''.join(self.s) + '</svg>')
        print('wrote', name)


# ───────────────── fig 1 — stage 1: does the box move at all ───────────────
def fig1(C):
    bars = []
    for c in V1:
        fam, lab, K = ENG[c]
        v, s = bestB(C, c, TRAIN, UNT)
        bars.append((f'{lab} K{K}', v, s['frac'], s['unt'], fam, False))
    v, s = bestB(C, '16', 'test', UNT)
    bars.append(('DPCC baseline (Gen6v4)', v, s['frac'], s['unt'], 'bl', True))
    d = C[('17', 'test', 'none', 'd3il_baseline')]
    st = S(d) if False else None
    rs = list(d.values())
    f = [fl(r['context_final_xy_dist']) for r in rs]
    i0 = [fl(r['context_init_xy_dist']) for r in rs]
    bars.append(('d3il ddpm-vision baseline', '(no projector)',
                 stt.median(f[j] / i0[j] for j in range(len(rs))),
                 sum(1 for j in range(len(rs)) if abs(f[j] - i0[j]) < 0.005) / len(rs),
                 'bl', True))

    fg = Fig(760, 340, ml=210, mr=118, mt=52, mb=56)
    fg.axes((0, 1.12), (0, len(bars)))
    fg.grid(xt=[0, .2, .4, .6, .8, 1.0], xfmt=lambda v: f'{v:.1f}x')
    fg.frame('median final distance / starting distance   (1.00x = the box never got closer)', '',
             'Stage 1 - does the box get to the goal?',
             'each engine at its OWN best DPCC projector; UNet FiLM v1, train, combined_5, n=30')
    bh = (fg.B - fg.T) / len(bars) * 0.62
    for i, (lab, var, frac, unt, fam, dashed) in enumerate(bars):
        yc = fg.T + (i + 0.5) * (fg.B - fg.T) / len(bars)
        fg.rect(fg.L, yc - bh / 2, fg.X(frac) - fg.L, bh, FAM[fam],
                op=0.45 if dashed else 0.92)
        fg.text(fg.L - 10, yc - 1, lab, 10.5, '#111', 'end', bold=not dashed)
        fg.text(fg.L - 10, yc + 11, var, 9.0, '#777', 'end')
        fg.text(fg.X(frac) + 6, yc + 4, f'{frac:.2f}x   ({unt*100:.0f}% untouched)',
                9.5, '#111')
    fg.line(fg.X(1.0), fg.T, fg.X(1.0), fg.B, '#111', 1.4, '5,4')
    fg.text(fg.X(1.0) - 5, fg.T - 6, 'did nothing', 9.5, '#111', 'end')
    fg.save('fig1_stage1_distance.svg')


# ───────────── fig 2 — the funnel: 30 -> near -> near & clean ──────────────
def fig2(C):
    cols = []
    for c in V1:
        fam, lab, K = ENG[c]
        for g, gl in ((UNT, 'untightened'), (TGT, 'tightened')):
            b = bestB(C, c, TRAIN, g)
            if b:
                v, s = b
                cols.append((f'{SHORT[fam]} K{K}', gl, v, s, fam))
    fg = Fig(880, 400, ml=64, mr=16, mt=52, mb=86)
    fg.axes((0, len(cols)), (0, 30))
    fg.grid(yt=[0, 5, 10, 15, 20, 25, 30], yfmt=lambda v: f'{v:.0f}')
    fg.frame('', 'rollouts (out of 30)', 'Stages 1 -> 2 - reaching the goal, then reaching it legally',
             'grey = all 30 | mid = within 15 cm | solid = within 15 cm AND zero constraint violations')
    w = (fg.R - fg.L) / len(cols)
    for i, (lab, gl, var, s, fam) in enumerate(cols):
        x = fg.L + i * w
        fg.rect(x + w * .14, fg.Y(30), w * .72, fg.B - fg.Y(30), '#e9e9e9')
        fg.rect(x + w * .14, fg.Y(s['near']), w * .72, fg.B - fg.Y(s['near']), FAM[fam], op=.38)
        fg.rect(x + w * .14, fg.Y(s['nc']), w * .72, fg.B - fg.Y(s['nc']), FAM[fam], op=1.0)
        fg.text(x + w / 2, fg.Y(s['near']) - 14, s['near'], 10, '#555', 'middle')
        fg.text(x + w / 2, fg.Y(s['nc']) - 3, s['nc'], 11, '#111', 'middle', bold=True)
        fg.text(x + w / 2, fg.B + 15, lab, 10, '#111', 'middle', bold=True)
        fg.text(x + w / 2, fg.B + 28, gl, 9, '#666', 'middle')
        fg.text(x + w / 2, fg.B + 40, var, 8.5, '#888', 'middle')
        fg.text(x + w / 2, fg.B + 54, f"0-viol {s['zv']:.2f}", 8.5, '#444', 'middle')
        fg.text(x + w / 2, fg.B + 66, f"{s['ms']:.0f} ms", 8.5, '#444', 'middle')
    fg.save('fig2_funnel.svg')


# ─────────── fig 3 — stage 3: cost of one clean near-goal rollout ──────────
def fig3(C):
    pts = []
    for c in V1:
        fam, lab, K = ENG[c]
        for g, mk in ((UNT, 'o'), (TGT, 's')):
            b = bestB(C, c, TRAIN, g)
            if b:
                v, s = b
                pts.append((f'{SHORT[fam]} K{K}{" tight" if g == TGT else ""}',
                            s['ms'], s['nc'] / 30, fam, mk))
    b = bestB(C, '16', 'test', UNT)
    pts.append(('DPCC-base', b[1]['ms'], b[1]['nc'] / 30, 'bl', 'o'))
    fg = Fig(720, 420, ml=76, mr=26, mt=52, mb=58)
    fg.axes((18, 3000), (-0.012, 0.40), xlog=True)
    fg.grid(xt=[20, 50, 100, 200, 500, 1000, 2000], yt=[0, .1, .2, .3, .4],
            xfmt=lambda v: f'{v:.0f}', yfmt=lambda v: f'{v*100:.0f}%')
    fg.frame('avg_time   [ ms / control step ]   (log)',
             'rollouts within 15 cm AND constraint-clean',
             'Stage 3 - what the clean near-goal rollouts cost',
             'circle = combined_5 | square = combined_5-tightened | up-and-left is better')
    for lab, x, y, fam, mk in pts:
        fg.marker(fg.X(x), fg.Y(y), mk, FAM[fam], filled=True)
        fg.text(fg.X(x) + 9, fg.Y(y) - 8, lab, 9.5, '#111', bold=True)
        fg.text(fg.X(x) + 9, fg.Y(y) + 4, f'{x:.0f} ms', 8.5, '#777')
    fg.save('fig3_stage3_cost.svg')


# ──────────────── fig 4 — HardFlow (IPOPT NLP) vs DPCC ─────────────────────
def fig4(C):
    groups = []
    for c, cl in (('14', 'MeanFlow K2'), ('6', 'AlphaFlow K2')):
        for g, gl in ((UNT, 'untightened'), (TGT, 'tightened')):
            row = []
            for rule in ('r', 'c', 't'):
                a = cell(C, c, TRAIN, g, f'dpcc-{rule}')
                b = cell(C, c, TRAIN, g, f'hardflow_new-{rule}')
                if a and b:
                    row.append((rule, a, b))
            if row:
                groups.append((f'{cl} - {gl}', row))
    fg = Fig(880, 420, ml=64, mr=16, mt=52, mb=92)
    fg.axes((0, 12), (0, 16))
    fg.grid(yt=[0, 4, 8, 12, 16], yfmt=lambda v: f'{v:.0f}')
    fg.frame('', 'rollouts within 15 cm AND clean  (of 30)',
             'Stage 4 - HardFlow (in-loop IPOPT NLP) vs the DPCC projector',
             'paired: same checkpoint, same geometry, same selection rule; light = DPCC, dark = HardFlow')
    w = (fg.R - fg.L) / 12
    i = 0
    for gi, (glab, row) in enumerate(groups):
        x0 = fg.L + i * w
        for rule, a, b in row:
            x = fg.L + i * w
            fg.rect(x + w * .10, fg.Y(a['nc']), w * .36, fg.B - fg.Y(a['nc']), '#95a5a6')
            fg.rect(x + w * .52, fg.Y(b['nc']), w * .36, fg.B - fg.Y(b['nc']), '#8e44ad')
            fg.text(x + w * .28, fg.Y(a['nc']) - 3, a['nc'], 9.5, '#111', 'middle')
            fg.text(x + w * .70, fg.Y(b['nc']) - 3, b['nc'], 9.5, '#111', 'middle', bold=True)
            fg.text(x + w / 2, fg.B + 14, f'-{rule}', 10, '#111', 'middle', bold=True)
            fg.text(x + w / 2, fg.B + 27, f"{a['ms']:.0f}", 8.5, '#666', 'middle')
            fg.text(x + w / 2, fg.B + 38, f"{b['ms']:.0f} ms", 8.5, '#8e44ad', 'middle')
            fg.text(x + w / 2, fg.B + 52, f"{a['zv']:.2f}", 8.5, '#666', 'middle')
            fg.text(x + w / 2, fg.B + 63, f"{b['zv']:.2f}", 8.5, '#8e44ad', 'middle')
            i += 1
        fg.text((x0 + fg.L + i * w) / 2, fg.B + 82, glab, 10.5, '#111', 'middle', bold=True)
        if i < 12:
            fg.line(fg.L + i * w, fg.T, fg.L + i * w, fg.B + 68, '#bbb', 1.0, '3,3')
    fg.text(fg.L + 4, fg.T + 14, 'rows under the axis:  ms/step (DPCC, HardFlow)  then  0-viol rate',
            9, '#666')
    fg.save('fig4_hardflow_vs_dpcc.svg')


# ────────────────── fig 5 — the K ladder (unguided arm) ────────────────────
def fig5(C):
    pairs = [('MeanFlow', '13', '14', 100, 2), ('AlphaFlow', '5', '6', 100, 2),
             ('FlowMatching', '10', '11', 100, 20), ('Diffusion', '8', '9', 100, 20)]
    fg = Fig(760, 360, ml=132, mr=112, mt=52, mb=58)
    fg.axes((0, 1.12), (0, len(pairs)))
    fg.grid(xt=[0, .2, .4, .6, .8, 1.0], xfmt=lambda v: f'{v:.1f}x')
    fg.frame('median final distance / starting distance', '',
             'Sampler steps K do not tune this task - they flip an arm between working and no-op',
             'unguided (diffuser) arm, no projection; UNet FiLM v1, train, combined_5, n=30')
    bh = (fg.B - fg.T) / len(pairs) * 0.30
    for i, (lab, chi, clo, khi, klo) in enumerate(pairs):
        hi = cell(C, chi, TRAIN, UNT, 'diffuser')
        lo = cell(C, clo, TRAIN, UNT, 'diffuser')
        yc = fg.T + (i + 0.5) * (fg.B - fg.T) / len(pairs)
        fam = ENG[chi][0]
        for j, (s, k) in enumerate(((hi, khi), (lo, klo))):
            y = yc - bh * 1.05 + j * bh * 1.25
            fg.rect(fg.L, y, fg.X(s['frac']) - fg.L, bh, FAM[fam], op=0.92 if j else 0.42)
            fg.text(fg.X(s['frac']) + 6, y + bh - 2,
                    f"K={k}   {s['frac']:.2f}x   {s['ms']:.0f} ms", 9.5, '#111')
        fg.text(fg.L - 10, yc + 3, lab, 11, '#111', 'end', bold=True)
    fg.line(fg.X(1.0), fg.T, fg.X(1.0), fg.B, '#111', 1.4, '5,4')
    fg.text(fg.X(1.0) - 5, fg.T - 6, 'did nothing', 9.5, '#111', 'end')
    fg.save('fig5_k_ladder.svg')


if __name__ == '__main__':
    C = load()
    fig1(C); fig2(C); fig3(C); fig4(C); fig5(C)
