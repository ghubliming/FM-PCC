import json, math, os, stats
from svglib import Fig, INK, MUTE, GRID, C4, C1, ACC, MONO
D = json.load(open('perseed.json'))
OUT = '/workspaces/FM-PCC/logs_in_develop/HF_Batch_Parity/figs_20260827_mpc_fan'
os.makedirs(OUT, exist_ok=True)
def g(l,v,m): return D.get(f'{l}|{v}|{m}')
def mean(l,v,m,ks=None):
    d=g(l,v,m); ks=ks or sorted(d); return sum(d[k] for k in ks)/len(ks)
def succ(l,v):
    d=g(l,v,'n_success_and_constraints'); return round(sum(d.values())*2), 2*len(d)

# ─────────────────────────────── FIG A: cost decomposition ───────────────────────────────
rows = [('DPCC Gen0','K=20 · UNet','DPCC_f4','DPCC_f1','diffuser','dpcc-c-tightened'),
        ('AlphaFlow','K=2 · SiT','AF_f4','AF_f1','diffuser','dpcc-t-tightened'),
        ('MeanFlow','K=2 · UNet · seed 6','MFu_f4','MFu_f1','diffuser','dpcc-t-tightened')]
W,H=880,430; f=Fig(W,H,'Figure A — The candidate fan scales the projection stage and nothing else',
  'Per-step cost decomposed into generator + projection, normalised to each run\'s own generator cost. avoiding-d3il, tightened arm.')
L,R,T=200,W-150,92; BH,GAP=30,13; xmax=3.6
def X(v): return L+(R-L)*v/xmax
for t in [0,1,2,3]:
    f.line(X(t),T-10,X(t),T+6*(BH+GAP)+16,GRID,1)
    f.text(X(t),T-16,f'{t}×',10,MUTE,'middle')
f.text((L+R)/2,T-34,'per-step cost, in units of that run\'s generator cost',11,MUTE,'middle')
y=T
for name,meta,l4,l1,base,arm in rows:
    f.text(L-16,y+BH+GAP/2+4,name,12,INK,'end',650)
    f.text(L-16,y+BH+GAP/2+18,meta,9.5,MUTE,'end')
    for leg,col,lab in ((l4,C4,'fan 4'),(l1,C1,'fan 1')):
        gt=mean(leg,base,'avg_time'); at=mean(leg,arm,'avg_time')
        gen,proj=1.0,at/gt-1.0
        f.rect(X(0),y,X(gen)-X(0),BH,ACC,rx=2,op=.35)
        f.rect(X(gen),y,X(gen+proj)-X(gen),BH,col,rx=2)
        f.text(X(0)+8,y+BH/2+4,'generator',9.5,'var(--fig-ink,#1b1f24)')
        if proj>0.35: f.text((X(gen)+X(gen+proj))/2,y+BH/2+4,'projection',9.5,'var(--fig-bg,#ffffff)','middle',600)
        f.text(X(gen+proj)+10,y+BH/2+4.5,f'{lab}   {at*1000:.0f} ms  ({gen+proj:.2f}×)',10.5,col,weight=600,font=MONO)
        y+=BH+GAP
    p4=mean(l4,arm,'avg_time')/mean(l4,base,'avg_time')-1
    p1=mean(l1,arm,'avg_time')/mean(l1,base,'avg_time')-1
    f.text(L-16,y+2,f'projection {p4/p1:.1f}× cheaper  ·  end-to-end {(1+p4)/(1+p1):.2f}×',9.5,C1,'end',600)
    y+=26
f.text(24,H-22,'Generator cost is fan-invariant (grey band identical within each model); the entire saving comes from the projection band. '
                'End-to-end gain is therefore set by the projection\'s share of the budget.',10,MUTE)
f.save(f'{OUT}/figA_cost_decomposition.svg')

# ─────────────────────────────── FIG B: safety–cost Pareto ───────────────────────────────
W,H=880,470; f=Fig(W,H,'Figure B — Safety–cost plane: dropping the fan moves DPCC and AlphaFlow in opposite directions',
  'Tightened arms. y = success-and-constraints over 30 paired episodes (Wilson 95% CI). x = per-step cost, log scale.')
L,R,T,B=78,W-190,100,H-72
lo,hi=math.log10(10),math.log10(700)
def PX(ms): return L+(R-L)*(math.log10(ms)-lo)/(hi-lo)
def PY(v):  return B-(B-T)*v
for v in (0,.2,.4,.6,.8,1.0):
    f.line(L,PY(v),R,PY(v),GRID,1); f.text(L-10,PY(v)+4,f'{v:.1f}',10,MUTE,'end')
for ms in (10,20,50,100,200,500):
    f.line(PX(ms),T,PX(ms),B,GRID,1,dash='2 4'); f.text(PX(ms),B+18,f'{ms}',10,MUTE,'middle')
f.text((L+R)/2,B+38,'per-step cost (ms, log)',11,MUTE,'middle')
f.text(20,(T+B)/2,'S&C  (successes / 30)',11,MUTE,'middle',rot=-90)
f.line(L,T,L,B,MUTE,1.2); f.line(L,B,R,B,MUTE,1.2)
pts=[('DPCC  dpcc-c-tight','DPCC_f4','DPCC_f1','dpcc-c-tightened',C4,1),
     ('DPCC  dpcc-t-tight','DPCC_f4','DPCC_f1','dpcc-t-tightened',C4,-1),
     ('AF  dpcc-t-tight','AF_f4','AF_f1','dpcc-t-tightened',C1,1),
     ('AF  dpcc-c-tight','AF_f4','AF_f1','dpcc-c-tightened',C1,-1)]
for lab,l4,l1,arm,col,side in pts:
    a4,n=succ(l4,arm); a1,_=succ(l1,arm)
    x4,x1=PX(mean(l4,arm,'avg_time')*1000),PX(mean(l1,arm,'avg_time')*1000)
    y4,y1=PY(a4/n),PY(a1/n)
    for (x,y,k) in ((x4,y4,a4),(x1,y1,a1)):
        c0,c1_=stats.wilson(k,n); f.line(x,PY(c0),x,PY(c1_),col,1.2,op=.45)
    f.arrow(x4,y4,x1,y1,col,1.8,7,op=.9)
    f.circ(x4,y4,5.5,'var(--fig-bg,#ffffff)',col,2); f.circ(x1,y1,5.5,col,col,1)
    f.text(x1+(10 if side>0 else -10),y1+(-9 if side>0 else 15),lab,9.5,col,'start' if side>0 else 'end',600)
f.circ(R+34,T+8,5.5,'var(--fig-bg,#ffffff)',INK,2); f.text(R+46,T+12,'fan 4',10.5,INK)
f.circ(R+34,T+30,5.5,INK,INK,1);      f.text(R+46,T+34,'fan 1',10.5,INK)
f.text(R+24,T+62,'up  = safer',10,MUTE); f.text(R+24,T+78,'left = cheaper',10,MUTE)
f.text(R+24,T+104,'AlphaFlow moves',10,C1,weight=600); f.text(R+24,T+118,'up AND left.',10,C1,weight=600)
f.text(R+24,T+140,'DPCC moves left,',10,C4,weight=600); f.text(R+24,T+154,'slightly down',10,C4,weight=600)
f.text(R+24,T+168,'(not resolvable,',10,C4); f.text(R+24,T+182,'see Fig C).',10,C4)
f.text(24,H-22,'Hollow = fan 4, filled = fan 1. AlphaFlow\'s dpcc-c-tightened arm is the extreme case: at fan 4 the min-cost rule stalls episodes '
                'to 181 steps (S&C 6/30); at fan 1 there is no candidate to choose and it reaches 30/30.',10,MUTE)
f.save(f'{OUT}/figB_safety_cost_pareto.svg')
print('A,B written')
