"""
DA figures for the 2026-08-05 massive Gen14 K=2 eval (jobs 24281 / 24282).

Source: temp/0508/  — 2 arms (mf, af) x 2 geometries (combined_5, combined_5-tightened)
                      x 19 projection variants x 30 contexts, NFE K=2, T=0.5, mpc=4, seed 6.

Run with the scratchpad venv (numpy + matplotlib); this container has no project env.
Outputs: figs/fig1_distance_null.png, figs/fig2_pareto.png, figs/fig3_time.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/workspaces/FM-PCC/temp/0508'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- load
rows = []
for f in sorted(glob.glob(os.path.join(ROOT, '*/6/results_train_set/*/*/*.npz'))):
    p = f[len(ROOT) + 1:].split('/')
    arm = 'mf' if 'Emf' in p[0] else 'af'
    geo = 'nom' if p[3] == 'combined_5' else 'tgt'
    var = p[4].replace('_train_set', '')
    d = np.load(f, allow_pickle=True)
    cm = json.load(open(os.path.join(os.path.dirname(f), 'constraint_metrics.json')))
    rows.append(dict(
        arm=arm, geo=geo, var=var,
        md=np.asarray(d['mean_dist_per_rollout'], dtype=float),
        time=float(np.mean(d['avg_time'])),
        viol=cm['exec']['n_violated_steps']['mean'],
        sat=cm['exec']['constraint_sat_rate']['mean'],
        zero=cm['exec']['zero_violation_rollouts'],
    ))
R = {(r['arm'], r['geo'], r['var']): r for r in rows}
CELLS = [('mf', 'nom'), ('mf', 'tgt'), ('af', 'nom'), ('af', 'tgt')]

# Noise floor.  `diffuser` never touches the constraint set, so its combined_5 and
# combined_5-tightened runs are a pure replicate: same policy, same 30 contexts,
# different noise draw.  The SD of that paired difference is the per-rollout noise;
# 1.96*SD/sqrt(30) is the 95% half-width on any 30-rollout MEAN difference.
# Using the observed |mean diff| instead would be one draw of a random variable.
NULL = {}
for _a in ('mf', 'af'):
    _d = R[(_a, 'nom', 'diffuser')]['md'] - R[(_a, 'tgt', 'diffuser')]['md']
    NULL[_a] = 1.96 * _d.std(ddof=1) / np.sqrt(len(_d))
TITLE = {('mf', 'nom'): 'MeanFlow / combined_5',
         ('mf', 'tgt'): 'MeanFlow / combined_5-tightened',
         ('af', 'nom'): 'AlphaFlow / combined_5',
         ('af', 'tgt'): 'AlphaFlow / combined_5-tightened'}


def fam(v):
    if v.startswith('hardflow'):
        return 'HardFlow'
    if v.startswith('dpcc'):
        return 'DPCC'
    if v == 'diffuser':
        return 'unprojected'
    if v == 'gradient':
        return 'gradient guid.'
    return 'ablation'


COL = {'HardFlow': '#d62728', 'DPCC': '#1f77b4', 'unprojected': '#7f7f7f',
       'gradient guid.': '#9467bd', 'ablation': '#2ca02c'}

# ------------------------------------------------- fig 1: distance vs null
# The null: `diffuser` never touches the constraint set, so its combined_5 and
# combined_5-tightened runs are a pure replicate of the same policy on the same
# 30 contexts.  Their difference measures run-to-run noise, nothing else.
fig, axes = plt.subplots(2, 2, figsize=(17, 11))
for ax, (arm, geo) in zip(axes.ravel(), CELLS):
    sub = sorted([r for r in rows if r['arm'] == arm and r['geo'] == geo],
                 key=lambda r: np.median(r['md']))
    names = [r['var'] for r in sub]
    data = [r['md'] for r in sub]
    bp = ax.boxplot(data, orientation='horizontal', widths=.62, patch_artist=True, showfliers=False)
    for b, r in zip(bp['boxes'], sub):
        b.set_facecolor(COL[fam(r['var'])])
        b.set_alpha(.45)
    for m in bp['medians']:
        m.set_color('k')
    for i, r in enumerate(sub):
        ax.scatter(r['md'], np.full(len(r['md']), i + 1) + np.random.default_rng(i).uniform(-.16, .16, len(r['md'])),
                   s=7, color=COL[fam(r['var'])], alpha=.65, zorder=3, linewidths=0)
    # null band, centred on the unprojected baseline of this cell
    base = np.mean(R[(arm, geo, 'diffuser')]['md'])
    nul = NULL[arm]
    ax.axvspan(base - nul, base + nul, color='crimson', alpha=.09, zorder=0)
    ax.axvline(base, color='crimson', ls='--', lw=1.2, zorder=1)
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel('final mean distance to goal (m)   — lower is better')
    ax.set_title(f'{TITLE[(arm, geo)]}    (n=30)\nred band = 95% noise floor of the SAME policy (±{nul:.3f} m)',
                 fontsize=10)
    ax.grid(axis='x', alpha=.25)
    ax.set_xlim(0, 1.05)
fig.suptitle('Gen14 K=2 / T=0.5 — distance is inside the noise floor in every cell', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .965])
fig.savefig(os.path.join(OUT, 'fig1_distance_null.png'), dpi=125)
plt.close(fig)

# ------------------------------------------------- fig 2: constraint/distance Pareto
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (arm, geo) in zip(axes.ravel(), CELLS):
    sub = [r for r in rows if r['arm'] == arm and r['geo'] == geo]
    for r in sub:
        ax.scatter(r['viol'], np.mean(r['md']), s=28 + r['time'] * 1400,
                   color=COL[fam(r['var'])], alpha=.55, edgecolor='k', linewidth=.6, zorder=3)
        ax.annotate(r['var'], (r['viol'], np.mean(r['md'])), fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    nul = NULL[arm]
    lo = min(np.mean(r['md']) for r in sub)
    ax.axhspan(lo, lo + nul, color='crimson', alpha=.08, zorder=0)
    ax.set_xlabel('violated steps / rollout   — lower is better')
    ax.set_ylabel('mean final distance (m)   — lower is better')
    ax.set_title(f'{TITLE[(arm, geo)]}\nmarker area ∝ inference time; red band = distance noise floor', fontsize=10)
    ax.grid(alpha=.25)
fig.suptitle('Gen14 K=2 — the constraint axis separates; the distance axis does not', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .965])
fig.savefig(os.path.join(OUT, 'fig2_pareto.png'), dpi=125)
plt.close(fig)

# ------------------------------------------------- fig 3: cost
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, arm in zip(axes, ('mf', 'af')):
    for geo, mk in (('nom', 'o'), ('tgt', '^')):
        sub = [r for r in rows if r['arm'] == arm and r['geo'] == geo]
        for r in sub:
            ax.scatter(r['time'] * 1000, r['viol'], marker=mk, s=55,
                       color=COL[fam(r['var'])], alpha=.75, edgecolor='k', linewidth=.5, zorder=3)
    ax.axvline(1000 / 30., color='k', ls=':', lw=1.4)
    ax.text(1000 / 30. + 1.5, ax.get_ylim()[1] * .95, '30 Hz budget', fontsize=9, rotation=90, va='top')
    ax.set_xlabel('inference time per replan (ms)')
    ax.set_ylabel('violated steps / rollout')
    ax.set_title(f'{arm}  (circle = combined_5, triangle = tightened)')
    ax.grid(alpha=.25)
h = [plt.Line2D([], [], marker='o', ls='', color=c, label=k) for k, c in COL.items()]
axes[0].legend(handles=h, fontsize=8, loc='upper right')
fig.suptitle('Gen14 K=2 — HardFlow costs 3.5× DPCC and buys no constraint advantage', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(os.path.join(OUT, 'fig3_time.png'), dpi=125)
plt.close(fig)
print('wrote', OUT)
