import json,csv,stats,os
D=json.load(open('perseed.json')); OUT='/workspaces/FM-PCC/logs_in_develop/HF_Batch_Parity/figs_20260827_mpc_fan'
def g(l,v,m): return D.get(f'{l}|{v}|{m}')
PAIRS=[('DPCC Gen0','K20/aw10','UNet','DPCC_f4','DPCC_f1',
        ['diffuser','model_free','dpcc-r','dpcc-c','dpcc-t','dpcc-r-tightened','dpcc-c-tightened','dpcc-t-tightened']),
       ('AlphaFlow','K2/A0.5','SiT','AF_f4','AF_f1',
        ['diffuser','dpcc-r','dpcc-c','dpcc-t','dpcc-r-tightened','dpcc-c-tightened','dpcc-t-tightened',
         'hardflow_new-r','hardflow_new-t-tightened']),
       ('MeanFlow','K2/A0.5','UNet','MFu_f4','MFu_f1',
        ['diffuser','dpcc-r','dpcc-c','dpcc-t','dpcc-r-tightened','dpcc-c-tightened','dpcc-t-tightened'])]
rows=[]
for model,cfg,bb,l4,l1,arms in PAIRS:
    for arm in arms:
        a,b=g(l4,arm,'n_success_and_constraints'),g(l1,arm,'n_success_and_constraints')
        if not a or not b: continue
        ks=sorted(set(a)&set(b)); N=2*len(ks)
        A={k:round(a[k]*2) for k in ks}; B={k:round(b[k]*2) for k in ks}
        rS=stats.paired(A,B); s4,s1=sum(A.values()),sum(B.values())
        w4,w1=stats.wilson(s4,N),stats.wilson(s1,N)
        st4=sum(g(l4,arm,'n_steps')[k] for k in ks)/len(ks); st1=sum(g(l1,arm,'n_steps')[k] for k in ks)/len(ks)
        rT=stats.paired(g(l4,arm,'n_steps'),g(l1,arm,'n_steps'))
        t4=sum(g(l4,arm,'avg_time')[k] for k in ks)/len(ks); t1=sum(g(l1,arm,'avg_time')[k] for k in ks)/len(ks)
        rC=stats.paired(g(l4,arm,'avg_time'),g(l1,arm,'avg_time'))
        rows.append(dict(model=model,config=cfg,backbone=bb,arm=arm,n_blocks=len(ks),n_episodes=N,
            sc4=s4,sc1=s1,sc4_rate=round(s4/N,4),sc1_rate=round(s1/N,4),
            sc4_ci=f'[{w4[0]:.3f},{w4[1]:.3f}]',sc1_ci=f'[{w1[0]:.3f},{w1[1]:.3f}]',
            d_episodes=s1-s4,d_ci_lo=round(rS['ci'][0]*len(ks),1),d_ci_hi=round(rS['ci'][1]*len(ks),1),
            p_wilcoxon=round(rS['wilcoxon']['p'],5),p_sign=round(rS['sign']['p'],5),
            steps4=round(st4,2),steps1=round(st1,2),d_steps=round(rT['mean'],2),p_steps=round(rT['wilcoxon']['p'],5),
            ms4=round(t4*1000,2),ms1=round(t1*1000,2),cost_ratio=round(t4/t1,3),p_cost=round(rC['wilcoxon']['p'],5)))
w=csv.DictWriter(open(f'{OUT}/results_mpc_fan_20260827.csv','w'),fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('csv rows',len(rows))

def esc(s): return s.replace('_','\\_')
key=[r for r in rows if r['arm'] in ('dpcc-c-tightened','dpcc-t-tightened','dpcc-t','dpcc-c')]
L=[r'\begin{tabular}{llrrrrrr}', r'\toprule',
   r'Model & Arm & \multicolumn{2}{c}{S\&C (/30)} & $\Delta$ & $p$ & \multicolumn{2}{c}{ms/step} \\',
   r'\cmidrule(lr){3-4}\cmidrule(lr){7-8}',
   r' &  & fan 4 & fan 1 & (ep.) &  & fan 4 & fan 1 \\', r'\midrule']
cur=None
for r in key:
    if r['model']!=cur: cur=r['model']; L.append(r'\addlinespace')
    star='$^{*}$' if r['p_wilcoxon']<0.05 else ''
    L.append(f"{esc(r['model'])} & \\texttt{{{esc(r['arm'])}}} & {r['sc4']} & {r['sc1']} & "
             f"{r['d_episodes']:+d}{star} & {r['p_wilcoxon']:.3f} & {r['ms4']:.0f} & {r['ms1']:.0f} \\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/table_mpc_fan.tex','w').write('\n'.join(L)+'\n')
print('\n'.join(L))
