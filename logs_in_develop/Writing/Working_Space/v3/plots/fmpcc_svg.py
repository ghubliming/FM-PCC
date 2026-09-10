#!/usr/bin/env python3
"""A dependency-free SVG canvas.

Why not matplotlib: the AI container has no scientific Python stack at all, so a
figure script that imports numpy cannot be run where the thesis is written --
which would make "regenerate the figures" a cluster round-trip. This class is
carried over from
``Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/make_figs.py``
(itself carried from ``Report_20260819_MF_UNet``), so the v3 figures are drawn
by the same code that drew the figures in the reports of record and match them
pixel semantics for pixel semantics.

Stdlib only. Output is plain SVG with no external assets, so a figure renders in
a browser, in a Markdown preview and in LaTeX (after ``tools/svg2pdf.sh``)
without anything else being installed.

Coordinates: ``axes()`` fixes the data range, ``X()``/``Y()`` map data to pixels,
everything else draws in pixels. Log axes are handled inside the mapping, so a
caller never takes a logarithm itself.
"""
import math
import os
from xml.sax.saxutils import escape as _esc

__all__ = ['Fig', 'dec_ticks', 'fmt_num', 'legend']


class Fig:
    """One SVG figure. Build with axes() -> frame() -> marks -> save()."""

    def __init__(self, w, h, ml=76, mr=16, mt=44, mb=54):
        self.w, self.h = w, h
        self.L, self.R, self.T, self.B = ml, w - mr, mt, h - mb
        self.s = [f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>']
        self.xlog = self.ylog = False

    # ---------------------------------------------------------------- axes ---
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
        """Gridlines, tick labels, box, axis labels, title and one subtitle line.

        The subtitle is where the protocol goes -- seeds, trials, corpus. Every
        v3 figure carries it, because TARGET section 6 rule 7 forbids a bare *n*.
        """
        for v in xt:
            x = self.X(v)
            self.s.append(f'<line x1="{x:.1f}" y1="{self.T}" x2="{x:.1f}" y2="{self.B}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            self.s.append(f'<text x="{x:.1f}" y="{self.B + 16}" font-size="10.5" fill="#222" '
                          f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif">'
                          f'{_esc(str(xfmt(v)))}</text>')
        for v in yt:
            y = self.Y(v)
            self.s.append(f'<line x1="{self.L}" y1="{y:.1f}" x2="{self.R}" y2="{y:.1f}" '
                          f'stroke="#e6e6e6" stroke-width="1"/>')
            self.s.append(f'<text x="{self.L - 7}" y="{y + 3.5:.1f}" font-size="10.5" fill="#222" '
                          f'text-anchor="end" font-family="Helvetica,Arial,sans-serif">'
                          f'{_esc(str(yfmt(v)))}</text>')
        self.s.append(f'<rect x="{self.L}" y="{self.T}" width="{self.R - self.L}" '
                      f'height="{self.B - self.T}" fill="none" stroke="#111" stroke-width="1.2"/>')
        self.s.append(f'<text x="{(self.L + self.R) / 2}" y="{self.B + 38}" font-size="11.5" '
                      f'fill="#111" text-anchor="middle" '
                      f'font-family="Helvetica,Arial,sans-serif">{_esc(xlab)}</text>')
        my = (self.T + self.B) / 2
        self.s.append(f'<text x="16" y="{my}" font-size="11.5" fill="#111" text-anchor="middle" '
                      f'font-family="Helvetica,Arial,sans-serif" transform="rotate(-90 16 {my})">'
                      f'{_esc(ylab)}</text>')
        self.s.append(f'<text x="{self.L}" y="20" font-size="13" font-weight="bold" fill="#111" '
                      f'font-family="Helvetica,Arial,sans-serif">{_esc(title)}</text>')
        if sub:
            self.s.append(f'<text x="{self.L}" y="35" font-size="10.5" fill="#555" '
                          f'font-family="Helvetica,Arial,sans-serif">{_esc(sub)}</text>')

    # --------------------------------------------------------------- marks ---
    def marker(self, x, y, kind, color, filled=True, r=5.5, ew=1.4):
        """kind: 'o' circle, 's' square, '^' triangle. Hollow = excluded."""
        fill, edge = (color, '#111') if filled else ('#ffffff', color)
        if kind == 'o':
            self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                          f'stroke="{edge}" stroke-width="{ew}"/>')
        elif kind == '^':
            pts = f'{x:.1f},{y - r:.1f} {x + r:.1f},{y + r * 0.8:.1f} {x - r:.1f},{y + r * 0.8:.1f}'
            self.s.append(f'<polygon points="{pts}" fill="{fill}" stroke="{edge}" '
                          f'stroke-width="{ew}"/>')
        else:
            self.s.append(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" '
                          f'height="{2 * r:.1f}" fill="{fill}" stroke="{edge}" '
                          f'stroke-width="{ew}"/>')

    def ring(self, x, y, r=11.0):
        """Halo marking a point on the Pareto front."""
        self.s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="none" stroke="#222" '
                      f'stroke-width="1.1"/>')

    def poly(self, pts, color, dash='', w=1.4):
        d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ''
        self.s.append(f'<polyline points="{d}" fill="none" stroke="{color}" '
                      f'stroke-width="{w}"{da}/>')

    def bar(self, x, y, w, h, color, alpha=1.0):
        self.s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                      f'fill="{color}" fill-opacity="{alpha}" stroke="#111" stroke-width="0.8"/>')

    def vspan(self, x0, x1, color='#f2f2f2'):
        """Shaded vertical band -- used to grey out a non-citable K regime."""
        self.s.append(f'<rect x="{x0:.1f}" y="{self.T:.1f}" width="{x1 - x0:.1f}" '
                      f'height="{self.B - self.T:.1f}" fill="{color}"/>')

    def text(self, x, y, t, size=9.5, color='#333', anchor='start', bold=False):
        b = ' font-weight="bold"' if bold else ''
        self.s.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                      f'text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif"{b}>'
                      f'{_esc(str(t))}</text>')

    # ---------------------------------------------------------------- save ---
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                    f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                    + ''.join(self.s) + '</svg>')
        return path


def dec_ticks(lo, hi):
    """1-2-5 decade ticks covering [lo, hi]. For log axes."""
    t, e = [], int(math.floor(math.log10(lo)))
    while 10 ** e <= hi * 1.001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                t.append(v)
        e += 1
    return t


def fmt_num(v):
    return f'{v:.0f}' if v >= 1 else f'{v:g}'


def legend(f, x, y, entries, dy=17):
    """entries: [(colour, label, marker_kind), ...] laid out downwards from (x, y)."""
    for i, (col, lab, kind) in enumerate(entries):
        yy = y + i * dy
        f.marker(x + 7, yy, kind, col, filled=True, r=5.0)
        f.text(x + 18, yy + 3.5, lab, 9.5, '#111')
