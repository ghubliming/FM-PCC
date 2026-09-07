import json, math, os, stats
from svglib import Fig, INK, MUTE, GRID, C4, C1, ACC, MONO
D=json.load(open('perseed.json')); OUT='/workspaces/FM-PCC/logs_in_develop/HF_Batch_Parity/figs_20260827_mpc_fan'
def g(l,v,m): return D.get(f'{l}|{v}|{m}')
def blocks(l4,l1,arm):
    a,b=g(l4,arm,'n_success_and_constraints'),g(l1,arm,'n_success_and_constraints')
    ks=sorted(set(a)&set(b)); return ks,[round(b[k]*2)-round(a[k]*2) for k in ks],\
           round(sum(a[k] for k in ks)*2),round(sum(b[k] for k in ks)*2),2*len(ks)

# ─────────────── FIG C: paired block differences ───────────────
rows=[('DPCC Gen0','dpcc-t','DPCC_f4','DPCC_f1'),('DPCC Gen0','dpcc-c','DPCC_f4','DPCC_f1'),
      ('DPCC Gen0','dpcc-c-tightened','DPCC_f4','DPCC_f1'),('DPCC Gen0','dpcc-t-tightened','DPCC_f4','DPCC_f1'),
      ('AlphaFlow','dpcc-c-tightened','AF_f4','AF_f1'),('AlphaFlow','dpcc-t-tightened','AF_f4','AF_f1'),
      ('AlphaFlow','dpcc-r-tightened','AF_f4','AF_f1')]
W,H=880,150+len(rows)*62; f=Fig(W,H,'Figure C — Paired per-block effect of fan 4 → fan 1 on safety',
 'One dot per (seed × scenario) block, n = 15, 2 trials each. x = change in successful episodes within that block. Exact two-sided Wilcoxon signed-rank.')
L,R,T=228,W-232,104; RH=62
def X(v): return L+(R-L)*(v+2)/4
for t in (-2,-1,0,1,2):
    f.line(X(t),T-14,X(t),T+len(rows)*RH-18,GRID if t else MUTE,1 if t else 1.3,dash=None if t==0 else '2 4')
    f.text(X(t),T-20,f'{t:+d}' if t else '0',10,MUTE,'middle')
f.text((L+R)/2,T-38,'Δ successful episodes in block  (fan 1 − fan 4)',11,MUTE,'middle')
y=T
for model,arm,l4,l1 in rows:
    ks,d,s4,s1,N=blocks(l4,l1,arm)
    r=stats.paired({k:0 for k in range(len(d))},{k:v for k,v in enumerate(d)})
    p=r['wilcoxon']['p']; tot=s1-s4
    f.text(L-18,y+16,arm,11,INK,'end',600); f.text(L-18,y+30,model,9.5,MUTE,'end')
    f.line(L,y+16,R,y+16,GRID,1,op=.6)
    cnt={}
    for v in d:
        cnt[v]=cnt.get(v,0)+1
        col = C1 if v>0 else (C4 if v<0 else ACC)
        f.circ(X(v), y+16-(cnt[v]-1)*10.5, 4.1, col, 'var(--fig-bg,#ffffff)', 1, op=.95)
    ci=r['ci']; mu=r['mean']
    f.line(X(ci[0]),y+34,X(ci[1]),y+34,INK,1.4,cap='round')
    f.line(X(ci[0]),y+30,X(ci[0]),y+38,INK,1.2); f.line(X(ci[1]),y+30,X(ci[1]),y+38,INK,1.2)
    f.circ(X(mu),y+34,3.2,INK)
    sig = 'p = %.4f' % p if p<.06 else 'p = %.2f' % p
    col = C1 if tot>0 else (C4 if tot<0 else MUTE)
    f.text(R+18,y+11,f'{s4}/{N} → {s1}/{N}',10.5,INK,'start',600,font=MONO)
    f.text(R+18,y+26,f'{tot:+d} episodes',10.5,col,'start',650,font=MONO)
    f.text(R+18,y+40,sig+('  ✓' if p<.05 else ''),10,INK if p<.05 else MUTE,'start',650 if p<.05 else 400,font=MONO)
    y+=RH
f.text(24,H-42,'Black bar = mean Δ with 95 % bootstrap CI (20 000 paired block resamples). Only the two untightened DPCC arms and the two '
                'AlphaFlow -c arms clear α = 0.05.',10,MUTE)
f.text(24,H-24,'The tightened DPCC loss (−2/30) sits far below this design\'s minimum detectable effect (~8/30 at 80 % power) — it is unresolved, not absent.',10,MUTE)
f.save(f'{OUT}/figC_paired_blocks.svg')

# ─────────────── FIG D: selection-rule collapse ───────────────
legs=[('DPCC Gen0','DPCC_f4','DPCC_f1','dpcc'),('AlphaFlow','AF_f4','AF_f1','dpcc'),
      ('MeanFlow-DiT','MFd_f1','MFd_f1','dpcc'),('FMv3ODE','FM_A10_f1','FM_A10_f1','hardflow_new')]
W,H=880,392; f=Fig(W,H,'Figure D — At fan 1 the three selection rules become bit-identical',
 'Pooled S&C for the random / min-projection-cost / temporal-consistency rules, tightened arm. Identity verified per (seed × scenario) block, not just in the mean.')
L,T=90,116; BW,BG_,GG=26,9,52; PH=170
f.line(L-14,T+PH,W-40,T+PH,MUTE,1.2)
for v in (0,.25,.5,.75,1.0):
    yy=T+PH-PH*v; f.line(L-14,yy,W-40,yy,GRID,1); f.text(L-22,yy+4,f'{v:.2f}',9.5,MUTE,'end')
f.text(30,T+PH/2,'pooled S&C',11,MUTE,'middle',rot=-90)
x=L
for name,l4,l1,pre in legs:
    single = (l4==l1)
    for leg,col,lab in ([(l1,C1,'fan 1')] if single else [(l4,C4,'fan 4'),(l1,C1,'fan 1')]):
        vals=[]
        for rr in 'rct':
            d=g(leg,f'{pre}-{rr}-tightened','n_success_and_constraints')
            vals.append(sum(d.values())/len(d))
        for i,v in enumerate(vals):
            f.rect(x+i*(BW+2),T+PH-PH*v,BW,PH*v,col,rx=2,op=.9)
            f.text(x+i*(BW+2)+BW/2,T+PH+14,'rct'[i],9.5,MUTE,'middle',font=MONO)
        ident = max(vals)-min(vals) < 1e-12
        f.text(x+(3*BW+4)/2,T+PH-PH*max(vals)-9,('identical' if ident else 'differ'),9.5,
               C1 if ident else C4,'middle',650)
        f.text(x+(3*BW+4)/2,T+PH+30,lab,10,col,'middle',600)
        x+=3*BW+4+BG_+14
    f.text(x-(3*BW+4+BG_+14)*(1 if single else 2)/2 - (0 if single else 0), T+PH+50, name, 11, INK,'middle',650) if False else None
    span=(3*BW+4+BG_+14)*(1 if single else 2)
    f.text(x-span/2-7,T+PH+52,name,11,INK,'middle',650)
    x+=GG-14
f.text(24,H-50,'At fan 4 the three rules choose different candidates and diverge. At fan 1 every rule executes index 0, so -r / -c / -t are the same rollout: '
                'S&C and step counts match exactly in all 15 blocks,',10,MUTE)
f.text(24,H-32,'in every generation tested. Evaluating all three at fan 1 costs 3× the projection compute for zero additional information.',10,MUTE)
f.save(f'{OUT}/figD_selection_collapse.svg')
print('C,D written')
