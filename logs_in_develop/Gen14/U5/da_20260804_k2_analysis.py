"""Gen14 U5 DA — K=2 sweep analysis (temp/0408/minimal_K2_thres0.5).
numpy + matplotlib only; no torch, no GPU. Run from repo root."""
import re, os, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/workspaces/FM-PCC/temp/0408'
K2   = f'{ROOT}/minimal_K2_thres0.5'
OUT  = '/workspaces/FM-PCC/logs_in_develop/Gen14/U5/figs'
ARM  = {'mf': 'H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_filmv1_Emf',
        'af': 'H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow_VTrue_mpc4_filmv1_Eaf'}
ORDER = ['diffuser','gradient','post_processing','dpcc-r','dpcc-c','dpcc-t',
         'model_free','bounds_free','geo_free','geo_free-bounds_free',
         'geo_free-model_free','model_free-bounds_free']

def summary(f):
    t = open(f, errors='replace').read()
    tail = t[t.rfind('--- aligning-d3il-visual'):]
    g = lambda p: (float(re.search(p, tail).group(1)) if re.search(p, tail) else np.nan)
    return dict(succ=g(r'Success rate:\s+([\d.]+)'), fmd=g(r'Avg final mean distance:\s+([\d.]+)'),
                sat=g(r'Execution satisfaction rate:\s+([\d.]+)'), viol=g(r'Violated steps/rollout:\s+([\d.]+)'),
                inf=g(r'Avg inference time/replan:\s+([\d.]+)'))

def per_rollout(f):
    t = open(f, errors='replace').read()
    bl = re.split(r'\[ Seen Training Context \d+ Finished \]', t)[1:]
    fmd = np.array([float(re.search(r'Final Mean Distance:\s*([\d.]+)', b).group(1)) for b in bl])
    cs = re.findall(r'\[ constraints \] sat=([\d.]+)\s+violated=(\d+)steps', t)
    return fmd, np.array([float(x[0]) for x in cs]), np.array([float(x[1]) for x in cs])

R = {}; PR = {}
for a, d in ARM.items():
    for geo in ('combined_5', 'combined_5-tightened'):
        for v in ORDER:
            f = f'{K2}/{d}/6/results_train_set/{geo}/{v}_train_set/eval_{v}_train_set.log'
            if os.path.exists(f):
                R[(a, geo, v)] = summary(f)
                PR[(a, geo, v)] = per_rollout(f)[0]

# noise floor from the `diffuser` replicate (no projector -> same computation twice)
noise = {}
for a, d in ARM.items():
    p = f'{K2}/{d}/6/results_train_set'
    m1, s1, v1 = per_rollout(f'{p}/combined_5/diffuser_train_set/eval_diffuser_train_set.log')
    m2, s2, v2 = per_rollout(f'{p}/combined_5-tightened/diffuser_train_set/eval_diffuser_train_set.log')
    noise[a] = dict(fmd=np.abs(m1-m2).mean(), sat=np.abs(s1-s2).mean(), viol=np.abs(v1-v2).mean(),
                    same=int((m1 == m2).sum()), n=len(m1))

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

# ---- final mean distance by variant (THE metric) ----
ax = fig.add_subplot(gs[0, :])
x = np.arange(len(ORDER)); w = 0.2
cols = {('mf','combined_5'):'#c0392b', ('mf','combined_5-tightened'):'#e8837a',
        ('af','combined_5'):'#2471a3', ('af','combined_5-tightened'):'#7fb3d5'}
for i,(a,geo) in enumerate([('mf','combined_5'),('mf','combined_5-tightened'),
                            ('af','combined_5'),('af','combined_5-tightened')]):
    y  = [np.mean(PR[(a,geo,v)]) for v in ORDER]
    mn = [np.min(PR[(a,geo,v)])  for v in ORDER]
    ax.bar(x + (i-1.5)*w, y, w, color=cols[(a,geo)],
           label=f'{a} {"tightened" if "tight" in geo else "nominal"}')
    ax.scatter(x + (i-1.5)*w, mn, s=16, color='k', zorder=5,
               label='best-case (min over 10)' if i == 0 else None)
nf = np.mean([noise['mf']['fmd'], noise['af']['fmd']])
for a, geo in [('mf','combined_5'), ('af','combined_5')]:
    b = np.mean(PR[(a,geo,'diffuser')])
    ax.axhspan(b-nf, b+nf, color=cols[(a,geo)], alpha=.10, zorder=0)
    ax.axhline(b, color=cols[(a,geo)], ls=':', lw=1)
ax.text(len(ORDER)-0.4, 0.05, f'shaded = unprojected baseline ± run-to-run noise (±{nf:.3f} m)',
        ha='right', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=28, ha='right', fontsize=9)
ax.set_ylabel('final mean distance to goal (m)   — lower better')
ax.set_title('K=2 — distance to goal by projection variant.  Bars = mean over 10 contexts, '
             'dots = best case.\nProjection collapses the mf/af gap from 0.22 m to 0.02 m; '
             'hard SLSQP (dpcc-r / post_processing) has the WORST best case.', fontweight='bold')
ax.legend(fontsize=8, ncol=5); ax.grid(alpha=.3, axis='y')

# ---- latency ----
ax = fig.add_subplot(gs[1, 0])
lab = ['diffuser\nK=100','diffuser\nK=2','dpcc-r\nK=100','dpcc-r\nK=2']
val = [0.893, R[('mf','combined_5','diffuser')]['inf'], 14.99, R[('mf','combined_5','dpcc-r')]['inf']]
b = ax.bar(lab, val, color=['#888','#2e8b57','#888','#2e8b57'])
ax.set_yscale('log'); ax.set_ylabel('s / replan (log)')
ax.set_title('latency: the one unambiguous win', fontweight='bold')
for r, v in zip(b, val):
    ax.text(r.get_x()+r.get_width()/2, v*1.25, f'{v:.3g}', ha='center', fontsize=9)
ax.text(0.5, 0.06, '×32.7', ha='center', transform=ax.get_xaxis_transform(), fontsize=11, color='#2e8b57')
ax.text(2.5, 0.06, '×316', ha='center', transform=ax.get_xaxis_transform(), fontsize=11, color='#2e8b57')
ax.grid(alpha=.3, axis='y')

# ---- reproducibility ----
ax = fig.add_subplot(gs[1, 1])
for a, d in ARM.items():
    p = f'{K2}/{d}/6/results_train_set'
    m1, _, _ = per_rollout(f'{p}/combined_5/diffuser_train_set/eval_diffuser_train_set.log')
    m2, _, _ = per_rollout(f'{p}/combined_5-tightened/diffuser_train_set/eval_diffuser_train_set.log')
    ax.scatter(m1, m2, s=55, label=f'{a}  ({noise[a]["same"]}/{noise[a]["n"]} identical)',
               color='#c0392b' if a == 'mf' else '#2471a3', alpha=.8)
lim = [0, 1.15]; ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('final mean distance, run 1 (m)'); ax.set_ylabel('run 2 (m)')
ax.set_title('same config, no projector, run twice\n→ the eval is NOT deterministic', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

fig.suptitle('Gen14 U5 DA — K=2 projection sweep, seed 6, 10 contexts, mpc=4, T=0.5', fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT}/fig3_k2_projection.png', dpi=140)
print('wrote', f'{OUT}/fig3_k2_projection.png')
print('noise floor:', noise)
