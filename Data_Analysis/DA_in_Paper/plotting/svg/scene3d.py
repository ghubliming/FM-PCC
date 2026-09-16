#!/usr/bin/env python3
"""A small 3D scene renderer, for drawing the simulated environments as they are.

Why this exists: the thesis needs a picture of each environment that shows what
the scene IS -- the corridor the quadrotor flies down, the obstacle field the
manipulator crosses -- and this container has no MuJoCo, no mesh renderer and no
GPU. What it does have is the scene definitions themselves: the quadrotor scenes
are MJCF files in the repository and the D3IL scenes are built from primitives in
the environment code. Both are boxes and cylinders with exact positions.

So the geometry here is READ from those definitions, never invented. What is
drawn is the real scene at the real dimensions, in an orthographic projection
with flat shading -- not a photograph of the simulator, and it does not pretend
to be one. Robot links are the one thing it cannot show: those are STL meshes.

Orthographic rather than perspective on purpose: these figures are read for
distances (how wide is the corridor, how far apart are the pillars), and a
perspective view makes equal lengths unequal on the page.

Conventions: right-handed world, z up. `azimuth` is degrees CCW about z from +x,
`elevation` degrees above the horizon. Faces are painted far to near.
"""
import math

from .fmpcc_svg import Fig


def _shade(hex_colour, f):
    """Multiply an #rrggbb colour by f, clamped. f>1 lightens, f<1 darkens."""
    h = hex_colour.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(v * f)))) for v in (r, g, b))


class Scene3D:
    """Collect faces in world coordinates, then project, sort and draw them."""

    def __init__(self, w, h, azimuth=55.0, elevation=22.0, scale=180.0,
                 target=(0.0, 0.0, 0.0), light=(0.4, -0.7, 0.6), font=1.0):
        # `font` is Fig's numeric text scale, not a family name.
        self.fig = Fig(w, h, ml=0, mr=0, mt=0, mb=0, font=font)
        self.w, self.h = w, h
        self.az, self.el = math.radians(azimuth), math.radians(elevation)
        self.scale, self.target = scale, target
        n = math.sqrt(sum(c * c for c in light))
        self.light = tuple(c / n for c in light)
        self.faces = []                      # (depth, [(x,y)...], fill, stroke, width, opacity)

    # ---- projection -------------------------------------------------------
    def _cam(self, p):
        """World point -> (screen x, screen y, depth). Depth grows towards the camera."""
        x, y, z = (p[i] - self.target[i] for i in range(3))
        ca, sa = math.cos(self.az), math.sin(self.az)
        # rotate about z so the camera looks along +x', then tilt by elevation
        xr, yr = x * ca + y * sa, -x * sa + y * ca
        ce, se = math.cos(self.el), math.sin(self.el)
        depth = xr * ce + z * se
        up = -xr * se + z * ce
        return (self.w / 2 + yr * self.scale,
                self.h / 2 - up * self.scale + getattr(self, '_yshift', 0.0), depth)

    def project(self, p):
        sx, sy, _ = self._cam(p)
        return (sx, sy)

    def fit(self, points, pad=26, top=52, bottom=44):
        """Choose target and scale so `points` fill the canvas without clipping.

        Written because fixed scales were the thing that actually went wrong: a
        camera tuned on one scene clipped the next, and the floor -- always the
        widest thing present -- ran off the page before any of the geometry did.
        `top` and `bottom` reserve room for the title and the note lines.
        """
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        cz = sum(p[2] for p in points) / len(points)
        self.target, self.scale = (cx, cy, cz), 1.0
        xs, ys = zip(*[self.project(p) for p in points])
        dx = max(max(xs) - self.w / 2, self.w / 2 - min(xs)) * 2 or 1e-6
        dy = max(max(ys) - self.h / 2, self.h / 2 - min(ys)) * 2 or 1e-6
        self.scale = min((self.w - 2 * pad) / dx, (self.h - top - bottom) / dy)
        # re-centre vertically inside the band left by the reserved strips
        _xs, ys = zip(*[self.project(p) for p in points])
        self.target = (cx, cy, cz)
        shift = ((top + (self.h - bottom)) / 2) - (min(ys) + max(ys)) / 2
        self._yshift = shift
        return self

    @staticmethod
    def bbox_corners(lo, hi):
        return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]

    # ---- primitives -------------------------------------------------------
    def add_face(self, pts, colour, normal=None, opacity=1.0, stroke=None, w=0.6, flat=False):
        proj = [self._cam(p) for p in pts]
        depth = sum(q[2] for q in proj) / len(proj)
        fill = colour
        if normal is not None and not flat:
            lam = sum(n * l for n, l in zip(normal, self.light))
            fill = _shade(colour, 0.62 + 0.38 * max(0.0, lam))
        self.faces.append((depth, [(q[0], q[1]) for q in proj], fill,
                           stroke if stroke is not None else _shade(colour, 0.55), w, opacity))

    def box(self, centre, half, colour, opacity=1.0, stroke=None, yaw=0.0):
        cx, cy, cz = centre
        hx, hy, hz = half
        ca, sa = math.cos(yaw), math.sin(yaw)
        c = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    dx, dy = sx * hx, sy * hy
                    c.append((cx + dx * ca - dy * sa,
                              cy + dx * sa + dy * ca,
                              cz + sz * hz))
        # (indices into c, outward normal)
        quads = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)),
                 ((0, 4, 5, 1), (0, -1, 0)), ((2, 3, 7, 6), (0, 1, 0)),
                 ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1))]
        for idx, nrm in quads:
            self.add_face([c[i] for i in idx], colour, nrm, opacity, stroke)

    def cylinder(self, centre, radius, half_height, colour, n=28, opacity=1.0, stroke=None):
        """A z-aligned cylinder, as MuJoCo defines one: size = (radius, half-height)."""
        cx, cy, cz = centre
        z0, z1 = cz - half_height, cz + half_height
        ring = [(cx + radius * math.cos(2 * math.pi * i / n),
                 cy + radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
        for i in range(n):
            x0, y0 = ring[i]
            x1, y1 = ring[(i + 1) % n]
            mx, my = (x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy
            ln = math.hypot(mx, my) or 1.0
            self.add_face([(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)],
                          colour, (mx / ln, my / ln, 0.0), opacity, stroke, w=0.0)
        self.add_face([(x, y, z1) for x, y in ring], colour, (0, 0, 1), opacity, stroke, w=0.6)

    def ground(self, x0, x1, y0, y1, colour='#e8e8e8', step=None, grid='#cfcfcf'):
        """The floor, with optional grid lines drawn on it (they are faces too, so
        they sort correctly against anything standing on the floor)."""
        self.add_face([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)],
                      colour, (0, 0, 1), 1.0, '#bdbdbd', w=0.8, flat=True)
        # A single painter-sorted floor polygon has only one average depth.  If
        # left at that value it can cover an object on the far half of the same
        # floor even though the object is above it.  The floor is the backdrop.
        self.faces[-1] = (-1e12,) + self.faces[-1][1:]
        if step:
            v = x0
            while v <= x1 + 1e-9:
                self.line((v, y0, 0.001), (v, y1, 0.001), grid, 0.7)
                v += step
            v = y0
            while v <= y1 + 1e-9:
                self.line((x0, v, 0.001), (x1, v, 0.001), grid, 0.7)
                v += step

    def line(self, p0, p1, colour, w=1.4, dash=''):
        a, b = self._cam(p0), self._cam(p1)
        self.faces.append(((a[2] + b[2]) / 2 + 1e-4,
                           [(a[0], a[1]), (b[0], b[1])], None, colour, w, 1.0, dash))

    def path(self, pts, colour, w=1.6, dash=''):
        for p, q in zip(pts, pts[1:]):
            self.line(p, q, colour, w, dash)

    # ---- output -----------------------------------------------------------
    def draw(self):
        for face in sorted(self.faces, key=lambda t: t[0]):
            if face[2] is None:                                    # a line
                (_d, pts, _f, col, w, _o) = face[:6]
                dash = face[6] if len(face) > 6 else ''
                self.fig.poly(pts, col, dash=dash, w=w)
            else:
                _d, pts, fill, stroke, w, op = face
                # Raw emit, not Fig.polygon: that one maps DATA coordinates through
                # the axes transform, and these points are already in pixels.
                d = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
                st = (f' stroke="{stroke}" stroke-width="{w}" stroke-linejoin="round"'
                      if stroke and w else ' stroke="none"')
                self.fig.s.append(f'<polygon points="{d}" fill="{fill}" '
                                  f'fill-opacity="{op}"{st}/>')
        return self.fig

    def text(self, *a, **kw):
        return self.fig.text(*a, **kw)

    def save(self, path):
        return self.fig.save(path)
