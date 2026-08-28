import math
FONT = "ui-sans-serif,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,'SF Mono',Menlo,Consolas,monospace"
INK  = "var(--fig-ink,#1b1f24)"
MUTE = "var(--fig-mute,#5b6570)"
GRID = "var(--fig-grid,#dfe3e8)"
BG   = "var(--fig-bg,#ffffff)"
C4  = "var(--fig-f4,#b4472e)"   # fan 4
C1  = "var(--fig-f1,#1f6f8b)"   # fan 1
ACC = "var(--fig-acc,#8a8f98)"  # neutral

class Fig:
    def __init__(s, w, h, title=None, sub=None):
        s.w, s.h, s.o = w, h, []
        s.o.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{BG}"/>')
        if title: s.text(24, 30, title, 15.5, INK, weight=650)
        if sub:   s.text(24, 50, sub, 11.5, MUTE)
    def text(s, x, y, t, sz=11, fill=INK, anchor="start", weight=400, font=FONT, rot=None, ls=0):
        t = (t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))
        tr = f' transform="rotate({rot},{x},{y})"' if rot else ''
        s.o.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{sz}" fill="{fill}" '
                   f'text-anchor="{anchor}" font-weight="{weight}" letter-spacing="{ls}"{tr}>{t}</text>')
    def line(s, x1,y1,x2,y2, c=GRID, w=1, dash=None, cap="butt", op=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        s.o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" '
                   f'stroke-width="{w}" stroke-linecap="{cap}" opacity="{op}"{d}/>')
    def rect(s, x,y,w,h, fill="none", stroke=None, sw=1, rx=0, op=1):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
        s.o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0):.2f}" height="{max(h,0):.2f}" '
                   f'fill="{fill}" rx="{rx}" opacity="{op}"{st}/>')
    def circ(s, x,y,r, fill, stroke=None, sw=1.4, op=1):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ''
        s.o.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" opacity="{op}"{st}/>')
    def arrow(s, x1,y1,x2,y2, c=INK, w=1.6, head=6, op=1):
        a = math.atan2(y2-y1, x2-x1); L = math.hypot(x2-x1, y2-y1)
        if L < 1e-6: return
        xe, ye = x2 - head*0.85*math.cos(a), y2 - head*0.85*math.sin(a)
        s.line(x1,y1,xe,ye,c,w,cap="round",op=op)
        p = [(x2,y2),
             (x2-head*math.cos(a-0.42), y2-head*math.sin(a-0.42)),
             (x2-head*math.cos(a+0.42), y2-head*math.sin(a+0.42))]
        s.o.append('<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+f'" fill="{c}" opacity="{op}"/>')
    def path(s, pts, c, w=1.5, dash=None, op=1):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        s.o.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)+
                   f'" fill="none" stroke="{c}" stroke-width="{w}" stroke-linejoin="round" opacity="{op}"{d}/>')
    def save(s, p):
        open(p,'w').write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.w}" height="{s.h}" '
                          f'viewBox="0 0 {s.w} {s.h}" role="img">\n' + "\n".join(s.o) + "\n</svg>\n")
