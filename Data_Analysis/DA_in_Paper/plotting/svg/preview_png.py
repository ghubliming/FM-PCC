#!/usr/bin/env python3.14
"""Render a figure SVG from this pipeline to PNG, for checking it by eye.

    python3.14 plotting/svg/preview_png.py figures/env/fig_constraints_avoiding.svg [out.png] [--scale 2]

Why this exists: the AI container has no SVG rasteriser (no rsvg-convert, inkscape
or cairosvg), so a figure could otherwise only be checked after it reached a
LaTeX build. This renders exactly the subset of SVG that fmpcc_svg.py writes --
rect, line, polyline, polygon, circle, text (incl. rotate(-90)), <g translate>,
rectangular clipPath, fill/stroke opacity and dashes -- with PIL. It is a
preview, not the print path: svg2pdf.sh remains the conversion for the thesis.
"""
import argparse
import os
import re
import xml.etree.ElementTree as ET

from PIL import Image, ImageColor, ImageDraw, ImageFont

try:
    import matplotlib
    FONT_PATH = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
    FONT_BOLD = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans-Bold.ttf')
except ImportError:                                            # pragma: no cover
    FONT_PATH = FONT_BOLD = None

NS = '{http://www.w3.org/2000/svg}'


def colour(v, opacity=1.0):
    if not v or v == 'none':
        return None
    r, g, b = ImageColor.getrgb(v)[:3]
    return (r, g, b, int(round(255 * float(opacity))))


def points(s, dx, dy, k):
    out = []
    for pair in s.split():
        x, y = pair.split(',')
        out.append(((float(x) + dx) * k, (float(y) + dy) * k))
    return out


def dashed(draw, pts, fill, width, dash, k):
    on, off = (float(v) * k for v in dash.replace(',', ' ').split()[:2])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if seg == 0:
            continue
        t, draw_on = 0.0, True
        while t < seg:
            step = min(on if draw_on else off, seg - t)
            if draw_on:
                a, b = t / seg, (t + step) / seg
                draw.line([(x0 + a * (x1 - x0), y0 + a * (y1 - y0)), (x0 + b * (x1 - x0), y0 + b * (y1 - y0))],
                          fill=fill, width=width)
            t += step
            draw_on = not draw_on


class Renderer:
    def __init__(self, root, k):
        self.k = k
        self.w, self.h = int(float(root.get('width')) * k), int(float(root.get('height')) * k)
        self.img = Image.new('RGBA', (self.w, self.h), (255, 255, 255, 255))
        self.clips = {}

    def font(self, size, bold):
        path = FONT_BOLD if bold else FONT_PATH
        return ImageFont.truetype(path, max(1, int(size * self.k))) if path else ImageFont.load_default()

    def layer(self):
        return Image.new('RGBA', (self.w, self.h), (0, 0, 0, 0))

    def commit(self, lay, clip):
        if clip:
            x0, y0, x1, y1 = (int(v) for v in clip)
            mask = Image.new('L', (self.w, self.h), 0)
            ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
            lay.putalpha(Image.composite(lay.getchannel('A'), Image.new('L', lay.size, 0), mask))
        self.img.alpha_composite(lay)

    def walk(self, el, dx=0.0, dy=0.0, clip=None):
        k = self.k
        for ch in el:
            tag = ch.tag.replace(NS, '')
            if tag == 'clipPath':
                r = ch.find(f'{NS}rect')
                if r is not None:
                    self.clips[ch.get('id')] = tuple(float(r.get(a)) for a in ('x', 'y', 'width', 'height'))
                continue
            if tag == 'g':
                ndx, ndy, nclip = dx, dy, clip
                m = re.match(r'translate\(([-\d.]+)[ ,]([-\d.]+)\)', ch.get('transform') or '')
                if m:
                    ndx, ndy = dx + float(m.group(1)), dy + float(m.group(2))
                cp = re.match(r'url\(#(.+)\)', ch.get('clip-path') or '')
                if cp and cp.group(1) in self.clips:
                    x, y, w, h = self.clips[cp.group(1)]
                    nclip = ((x + ndx) * k, (y + ndy) * k, (x + w + ndx) * k, (y + h + ndy) * k)
                self.walk(ch, ndx, ndy, nclip)
                continue
            lay = self.layer()
            d = ImageDraw.Draw(lay)
            sw = max(1, int(round(float(ch.get('stroke-width') or 1) * k)))
            stroke = colour(ch.get('stroke'), ch.get('stroke-opacity') or 1)
            fill = colour(ch.get('fill'), ch.get('fill-opacity') or 1)
            dash = ch.get('stroke-dasharray')
            if tag == 'rect':
                x, y = (float(ch.get('x')) + dx) * k, (float(ch.get('y')) + dy) * k
                w, h = float(ch.get('width')) * k, float(ch.get('height')) * k
                d.rectangle([x, y, x + w, y + h], fill=fill, outline=stroke, width=sw if stroke else 0)
            elif tag == 'line':
                pts = [((float(ch.get('x1')) + dx) * k, (float(ch.get('y1')) + dy) * k),
                       ((float(ch.get('x2')) + dx) * k, (float(ch.get('y2')) + dy) * k)]
                if stroke:
                    (dashed(d, pts, stroke, sw, dash, k) if dash else d.line(pts, fill=stroke, width=sw))
            elif tag in ('polyline', 'polygon'):
                pts = points(ch.get('points'), dx, dy, k)
                if tag == 'polygon' and fill and len(pts) > 2:
                    d.polygon(pts, fill=fill)
                if stroke and len(pts) > 1:
                    if tag == 'polygon':
                        pts = pts + pts[:1]
                    (dashed(d, pts, stroke, sw, dash, k) if dash else d.line(pts, fill=stroke, width=sw, joint='curve'))
            elif tag == 'circle':
                cx, cy, r = (float(ch.get('cx')) + dx) * k, (float(ch.get('cy')) + dy) * k, float(ch.get('r')) * k
                if fill:
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
                if stroke:
                    if dash:
                        import math
                        n = 72
                        pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
                        dashed(d, pts, stroke, sw, dash, k)
                    else:
                        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=stroke, width=sw)
            elif tag == 'text':
                txt = ''.join(ch.itertext())
                size = float(ch.get('font-size') or 10)
                f = self.font(size, ch.get('font-weight') == 'bold')
                x, y = (float(ch.get('x')) + dx) * k, (float(ch.get('y')) + dy) * k
                tw = d.textlength(txt, font=f)
                anchor = ch.get('text-anchor') or 'start'
                if 'rotate(-90' in (ch.get('transform') or ''):
                    tmp = Image.new('RGBA', (int(tw) + 4, int(size * k * 1.4) + 4), (0, 0, 0, 0))
                    ImageDraw.Draw(tmp).text((2, 0), txt, font=f, fill=fill or (0, 0, 0, 255))
                    tmp = tmp.rotate(90, expand=True)
                    lay.alpha_composite(tmp, (int(x - tmp.width / 2), int(y - tmp.height / 2)))
                else:
                    ox = {'start': 0, 'middle': tw / 2, 'end': tw}[anchor]
                    d.text((x - ox, y - size * k * 0.8), txt, font=f, fill=fill or (0, 0, 0, 255))
            else:
                continue
            self.commit(lay, clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--scale', type=float, default=1.5)
    a = ap.parse_args()
    root = ET.parse(a.svg).getroot()
    r = Renderer(root, a.scale)
    r.walk(root)
    out = a.out or os.path.splitext(a.svg)[0] + '.preview.png'
    r.img.convert('RGB').save(out)
    print(f'{out}  ({r.w}x{r.h})')


if __name__ == '__main__':
    main()
