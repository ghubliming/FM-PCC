import pickle, re, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT='/workspaces/FM-PCC/temp/0408'
OUT='/workspaces/FM-PCC/logs_in_develop/Gen14/U5/figs'
D={n:pickle.load(open(os.path.join(ROOT,f),'rb')) for n,f in [('mf','mf_losses.pkl'),('af','af)losses.pkl')]}
g=lambda d,k: np.asarray(d[k],dtype=float)

# ---------- FIG 1 : training ----------
fig,ax=plt.subplots(2,2,figsize=(13,8))
c={'mf':'#c0392b','af':'#2471a3'}
a=ax[0,0]
for n in ('mf','af'):
    tr=g(D[n],'training_raw_mse_u_losses'); te=g(D[n],'test_raw_mse_u_losses')
    a.plot(tr[:,0],tr[:,1],c[n],alpha=.35,lw=1,label=f'{n} train')
    a.plot(te[:,0],te[:,1],c[n],lw=2,label=f'{n} test')
al=g(D['af'],'training_alpha_losses')
cliff=al[al[:,1]==0][0,0]
for aa in (ax[0,0],ax[0,1],ax[1,0]):
    aa.axvline(cliff,color='k',ls='--',lw=1.2)
a.set_yscale('log'); a.set_xlabel('train step'); a.set_ylabel('raw MSE (u head)')
a.set_title('u-head regression error  (adaptive-weight-free)'); a.legend(fontsize=8); a.grid(alpha=.3)
a.annotate(f'α clamped to 0\n@ {cliff:.0f}',xy=(cliff,8.5),xytext=(cliff-38000,20),
           arrowprops=dict(arrowstyle='->'),fontsize=9)

a=ax[0,1]
a.plot(al[:,0],al[:,1],'#2471a3',lw=2,label='af  α(t)')
a2=a.twinx()
te=g(D['af'],'test_raw_mse_u_losses'); a2.plot(te[:,0],te[:,1],'#e67e22',lw=2,label='af test MSE_u')
te=g(D['mf'],'test_raw_mse_u_losses'); a2.plot(te[:,0],te[:,1],'#c0392b',lw=1.2,ls=':',label='mf test MSE_u')
a.set_xlabel('train step'); a.set_ylabel('α'); a2.set_ylabel('test raw MSE (u)')
a.set_title('the α cliff:  bootstrapped target → JVP target'); a.grid(alpha=.3)
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=8,loc='upper right')

a=ax[1,0]
for n in ('mf','af'):
    gn=g(D[n],'grad_norm_history'); a.plot(gn[:,0],gn[:,1],c[n],lw=1.4,label=f'{n} pre-clip ‖g‖')
a.axhline(1.0,color='k',lw=2,label='gradient_clip = 1.0')
a.set_yscale('log'); a.set_xlabel('train step'); a.set_ylabel('grad norm')
a.set_title('every step is clipped (median ≈ 67–73 ⇒ ~70× scale-down)'); a.legend(fontsize=8); a.grid(alpha=.3)

a=ax[1,1]
for n in ('mf','af'):
    for k,ls,lab in [('training_h_mse_b0_losses','-','b0 (small h)'),
                     ('training_h_mse_b1_losses','--','b1'),
                     ('training_h_mse_b3_losses',':','b3 (large h)')]:
        v=g(D[n],k)
        if v.size: a.plot(v[:,0],v[:,1],c[n],ls=ls,lw=1.3,label=f'{n} {lab}')
a.set_yscale('log'); a.set_xlabel('train step'); a.set_ylabel('MSE_u by interval bucket')
a.set_title('error vs interval length h'); a.legend(fontsize=7,ncol=2); a.grid(alpha=.3)
fig.suptitle('Gen14 U5 DA — mix_visual_aligning mf / af, seed 6, FiLM v1, U-Net, 100k steps',fontweight='bold')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_training.png',dpi=140); plt.close(fig)

# ---------- eval parse ----------
MF=os.path.join(ROOT,"mix_visual_aligning_mf/H8_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal/H8_K100_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_filmv1_Emf/6/results_train_set/combined_5")
AF=os.path.join(ROOT,"6/results_train_set/combined_5")
def parse(f):
    t=open(f,errors='replace').read()
    bl=re.split(r'\[ Seen Training Context \d+ Finished \]',t)[1:]
    o=[]
    for b in bl:
        gg=lambda p:re.search(p,b).group(1)
        o.append(dict(fmd=float(gg(r'Final Mean Distance:\s*([\d.]+)')),
                      trk=float(gg(r'Max Physical Tracking Error:\s*([\d.]+)')),
                      inf=float(gg(r'Avg Inference Time:\s*([\d.]+)'))))
    rx=re.compile(r'\[ constraints \] sat=([\d.]+)\s+violated=(\d+)steps')
    cs=[(float(x),float(y)) for x,y in rx.findall(t)]
    for i,r in enumerate(o):
        if i<len(cs): r['sat'],r['viol']=cs[i]
    return o
E={}
for arm,base in [('mf',MF),('af',AF)]:
    for v in ['diffuser_train_set','dpcc-r_train_set']:
        E[(arm,v)]=parse(os.path.join(base,v,f'eval_{v}.log'))

fig,ax=plt.subplots(1,3,figsize=(15,4.4))
w=0.2
lbl=['mf diffuser','mf dpcc-r','af diffuser','af dpcc-r']
keys=[('mf','diffuser_train_set'),('mf','dpcc-r_train_set'),('af','diffuser_train_set'),('af','dpcc-r_train_set')]
cols=['#c0392b','#e8837a','#2471a3','#7fb3d5']
for j,(metric,title,ylab) in enumerate([('fmd','final mean distance (lower better)','m'),
                                        ('trk','max physical tracking error','m'),
                                        ('sat','constraint satisfaction rate','rate')]):
    a=ax[j]
    for i,k in enumerate(keys):
        v=[r[metric] for r in E[k] if metric in r]
        a.scatter(np.full(len(v),i)+np.random.uniform(-.13,.13,len(v)),v,s=22,color=cols[i],alpha=.75)
        a.hlines(np.mean(v),i-.28,i+.28,color='k',lw=2.2)
    a.set_xticks(range(4)); a.set_xticklabels(lbl,rotation=18,fontsize=9)
    a.set_title(title); a.set_ylabel(ylab); a.grid(alpha=.3,axis='y')
    if metric=='trk': a.set_yscale('log')
fig.suptitle('Gen14 U5 DA — per-rollout eval, seed 6, combined_5, K=100, mpc=4  '
             '(dpcc-r truncated at 11/30 by the 24 h wall clock)',fontweight='bold')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_eval.png',dpi=140); plt.close(fig)
print('wrote', OUT)
