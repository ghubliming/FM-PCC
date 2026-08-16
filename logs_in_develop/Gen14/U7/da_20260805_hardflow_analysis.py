"""Gen14 U7 DA — first HardFlow run (job 24255). numpy+matplotlib only; run from repo root."""
import re, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = ('/workspaces/FM-PCC/temp/0408/H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.'
        'visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_filmv1_Emf/6/results_train_set')
OUT  = '/workspaces/FM-PCC/logs_in_develop/Gen14/U7/figs'
BUDGET_MS = 1000.0 / 30.0          # 30 Hz control loop, from the realtime recorder

ORDER = ['diffuser', 'gradient', 'post_processing', 'dpcc-r', 'dpcc-c', 'dpcc-t',
         'hardflow_new-r', 'dpcc-c-dt0p25', 'dpcc-c-dt0p5', 'dpcc-c-dt2p0', 'dpcc-c-dt4p0',
         'model_free', 'bounds_free', 'geo_free', 'geo_free-bounds_free',
         'geo_free-model_free', 'model_free-bounds_free']
GEOS = ['combined_5', 'combined_5-tightened']

def load(geo, v):
    f = f'{ROOT}/{geo}/{v}_train_set/eval_{v}_train_set.log'
    t = open(f, errors='replace').read()
    tail = t[t.rfind('--- aligning-d3il-visual'):]
    g = lambda p: (float(re.search(p, tail).group(1)) if re.search(p, tail) else np.nan)
    bl = re.split(r'\[ Seen Training Context \d+ Finished \]', t)[1:]
    fmd = np.array([float(re.search(r'Final Mean Distance:\s*([\d.]+)', b).group(1)) for b in bl])
    return dict(fmd=g(r'Avg final mean distance:\s+([\d.]+)'),
                sat=g(r'Execution satisfaction rate:\s+([\d.]+)'),
                inf=g(r'Avg inference time/replan:\s+([\d.]+)'), pr=fmd)

R = {(g, v): load(g, v) for g in GEOS for v in ORDER}

fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
HF, GR = '#8e44ad', '#16a085'
def col(v, tight):
    base = HF if v.startswith('hardflow') else GR if v == 'gradient' else (
        '#7f8c8d' if v == 'diffuser' else '#2471a3')
    return base if not tight else base + '99'

# ── A: distance ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
x, w = np.arange(len(ORDER)), 0.38
for i, geo in enumerate(GEOS):
    tight = 'tight' in geo
    ax.bar(x + (i - .5) * w, [R[(geo, v)]['fmd'] for v in ORDER], w,
           color=[col(v, tight) for v in ORDER],
           edgecolor='k' if tight else 'none', linewidth=.7,
           label='tightened' if tight else 'nominal')
    for j, v in enumerate(ORDER):
        pr = R[(geo, v)]['pr']
        ax.scatter(np.full(len(pr), x[j] + (i - .5) * w), pr, s=14, color='k', zorder=5, alpha=.65)
ax.axhline(R[('combined_5', 'diffuser')]['fmd'], color='#7f8c8d', ls=':', lw=1.4)
ax.text(len(ORDER) - .4, R[('combined_5', 'diffuser')]['fmd'] + .012,
        'unprojected baseline', ha='right', fontsize=8.5, color='#555')
ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=26, ha='right', fontsize=9)
ax.set_ylabel('final mean distance to goal (m)  — lower better')
ax.set_title('A — distance by variant (bars = mean of 3 contexts, dots = each context).  '
             'purple = HardFlow, green = gradient', fontweight='bold')
ax.legend(fontsize=8.5); ax.grid(alpha=.3, axis='y')

# ── B: the trade ──────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for geo in GEOS:
    tight = 'tight' in geo
    for v in ORDER:
        d = R[(geo, v)]
        ax.scatter(d['inf'] * 1000, d['sat'], s=95 if v.startswith(('hardflow', 'gradient')) else 45,
                   color=col(v, tight), marker='s' if tight else 'o',
                   edgecolor='k', linewidth=.5, zorder=4)
for v, geo, dx, dy in [('hardflow_new-r', 'combined_5', -6, -.012),
                       ('gradient', 'combined_5', 6, -.012),
                       ('dpcc-r', 'combined_5', 5, -.016),
                       ('diffuser', 'combined_5', 4, .006),
                       ('model_free', 'combined_5', 4, .004)]:
    d = R[(geo, v)]
    ax.annotate(v, (d['inf'] * 1000, d['sat']), textcoords='offset points',
                xytext=(dx, dy * 400), fontsize=8.5)
ax.axvline(BUDGET_MS, color='crimson', ls='--', lw=1.5)
ax.text(BUDGET_MS * 1.06, .70, f'30 Hz budget\n{BUDGET_MS:.1f} ms', color='crimson', fontsize=8.5)
ax.set_xscale('log'); ax.set_xlabel('inference time per replan (ms, log)')
ax.set_ylabel('execution constraint satisfaction')
ax.set_title('B — safety vs cost.  square = tightened', fontweight='bold')
ax.grid(alpha=.3)

# ── C: determinism ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
pairs = [('diffuser\n(no projector)', ('combined_5', 'diffuser'), ('combined_5-tightened', 'diffuser')),
         ('geo_free\n(geo off)',       ('combined_5', 'geo_free'), ('combined_5-tightened', 'geo_free')),
         ('dpcc-r vs post_proc\n(nominal)',   ('combined_5', 'dpcc-r'), ('combined_5', 'post_processing')),
         ('dpcc-r vs post_proc\n(tightened)', ('combined_5-tightened', 'dpcc-r'),
                                              ('combined_5-tightened', 'post_processing'))]
names, ident, projected = [], [], []
for nm, a, b in pairs:
    m1, m2 = R[a]['pr'], R[b]['pr']
    names.append(nm); ident.append(int((m1 == m2).sum()))
    projected.append('dpcc' in nm)
bars = ax.bar(range(len(names)), ident,
              color=['#16a085' if not p else '#c0392b' for p in projected])
ax.axhline(3, color='k', ls='--', lw=1)
ax.text(len(names) - .5, 3.04, 'must be 3/3', ha='right', fontsize=8.5)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0, 3.5); ax.set_ylabel('rollouts identical (of 3)')
ax.set_title('C — the generator is deterministic; the SLSQP projector is not',
             fontweight='bold')
ax.grid(alpha=.3, axis='y')

fig.suptitle('Gen14 U7 DA — first HardFlow run (job 24255): mf, K=2, seed 6, 3 contexts, mpc4, T=0.5',
             fontweight='bold')
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_hardflow.png', dpi=140)
print('wrote', f'{OUT}/fig1_hardflow.png')
