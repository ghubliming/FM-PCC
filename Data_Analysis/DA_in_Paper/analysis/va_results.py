#!/usr/bin/env python3
"""Vision-conditioned alignment: the numbers Chapter 6.2 quotes, recomputed from the committed data.

    python3 Data_Analysis/DA_in_Paper/analysis/va_results.py

Source: Data_Analysis/analysis_results_checkpoint/15-09/batch_va2_20260915_100754/per_rollout_detail.csv
(14,102 rollouts, the same corpus CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md analysed).
Stdlib only. Prints every quantity the text uses, so a changed batch shows up as changed output.

Conventions, all taken from the closure and checked here rather than assumed:
* distance  = context_final_xy_dist, metres, box to target in the table plane;
* a context's value is the median over its rollouts (1 rollout per context in these cells);
* every reported generative/projection cell is restricted to the same ten enumerated contexts; cells evaluated on
  a thirty-context superset are reduced to those ten before they are compared;
* paired tests run over the contexts both cells share: exact two-sided sign test, and exact
  sign-flip permutation test on the mean difference (2^n enumerations, n <= 10);
* 'untouched' = frozen (the box never moved);
* split: every thesis model on this task was evaluated on the TRAIN contexts -- reported, not hidden.
"""
import csv
import itertools
import math
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/15-09/'
                         'batch_va2_20260915_100754/per_rollout_detail.csv')

# Thesis cells: (label) -> (folder prefix, required folder substring or None)
CELLS = {
    'mf_K2':    ('H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_fil', None),
    'mf_K10':   ('H8_K10_Meuler_T0.4_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow', None),
    'mf_K20':   ('H8_K20_Meuler_T0.2_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow', None),
    'mf_K100':  ('H8_K100_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow', None),
    'af_K2':    ('H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow', None),
    'af_K20':   ('H8_K20_Meuler_T0.2_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow', None),
    'fm_K20':   ('H8_K20_Meuler_T0.2_Dmix_visual_aligning.models.visual_fm_diffusion.VisualFlowMatching', None),
    'fm_K20_T0.5': ('H8_K20_Meuler_T0.5_Dmix_visual_aligning.models.visual_fm_diffusion.VisualFlowMatching', None),
    'diff_K20': ('H8_K20_T0.5_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion', None),
    'diff_K100': ('H8_K100_T0.5_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion', None),
    'd3il':     ('d3il_baseline_ddpm_encdec_vision', None),
}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    rows = defaultdict(list)        # (cell, geo, variant, run_tag) -> rows
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            for cell, (prefix, _need) in CELLS.items():
                if r['FolderName'].startswith(prefix):
                    rows[(cell, r['geo'], r['variant'], r['FolderName'][len(prefix):])].append(r)
    return rows


def fingerprint(r):
    """The context a rollout was run on. context_index is empty for the thesis cells, so contexts are
    matched by their initial box pose and target pose, as the closure did."""
    # Positions only. The angle columns are not recorded consistently across runs (the K=2 MeanFlow run
    # and the consistency-training run share all ten box and target positions but not the angle
    # values), so including them silently unpairs contexts that are the same.
    keys = ('context_box_init_xy_x', 'context_box_init_xy_y',
            'context_target_xy_x', 'context_target_xy_y')
    vals = [fnum(r.get(k)) for k in keys]
    if any(v is None for v in vals):
        return None
    return tuple(round(v, 3) for v in vals)


def untouched(r):
    """Box never moved: final box-to-target distance equal to the initial one to 1e-6. This is the
    closure's definition, and the one that reproduces its counts (flow matching 4/10, diffusion 10/30,
    D3IL 1712/3884). The `frozen` column is 0 on every thesis row and is not used."""
    f, i = fnum(r.get('context_final_xy_dist')), fnum(r.get('context_init_xy_dist'))
    return f is not None and i is not None and abs(f - i) < 1e-6


def per_context(rs, key='context_final_xy_dist'):
    by = defaultdict(list)
    for r in rs:
        v, fp = fnum(r.get(key)), fingerprint(r)
        if v is not None and fp is not None:
            by[fp].append(v)
    return {c: st.median(v) for c, v in by.items()}


def sign_test(diffs):
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return pos, neg, min(1.0, p)


def perm_test(diffs):
    obs = abs(sum(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12:
            hits += 1
    return hits / total


def summary(label, rs):
    ctx = per_context(rs)
    v = list(ctx.values())
    by = defaultdict(list)
    for r in rs:
        fp = fingerprint(r)
        if fp is not None:
            by[fp].append(r)
    frozen = sum(
        abs(st.median(fnum(r['context_final_xy_dist']) for r in g)
            - st.median(fnum(r['context_init_xy_dist']) for r in g)) < 1e-6
        for g in by.values())
    ms = [st.mean(fnum(r['avg_time_ms']) for r in g if fnum(r.get('avg_time_ms')) is not None)
          for g in by.values()]
    return (f'{label:34s} n_ctx={len(v):3d} rollouts={len(rs):4d} median={st.median(v):.4f} '
            f'mean={st.mean(v):.4f} +/- {st.stdev(v):.4f} min={min(v):.4f} '
            f'unmoved={frozen}/{len(v)} ms/step={st.mean(ms):.1f} +/- {st.stdev(ms):.1f}') \
        if v else f'{label:34s} (no contexts)'


def pair(label, a, b):
    ca, cb = per_context(a), per_context(b)
    common = sorted(set(ca) & set(cb))
    diffs = [ca[c] - cb[c] for c in common]
    if not diffs:
        return f'{label:44s} no shared contexts'
    pos, neg, p = sign_test(diffs)
    return (f'{label:44s} n={len(common):2d} mean diff={st.mean(diffs):+.4f} m '
            f'({pos} worse / {neg} better)  sign p={p:.4f}  perm p={perm_test(diffs):.4f}')


def main():
    rows = load()

    def raw_cell(c, geo, variant='diffuser', tag=None):
        return [r for k, v in rows.items() if k[0] == c and k[1] == geo and k[2] == variant
                and (tag is None or k[3].endswith(tag)) for r in v]

    U, T = 'combined_5', 'combined_5-tightened'
    thesis_contexts = set(per_context(raw_cell('mf_K20', U)))
    if len(thesis_contexts) != 10:
        raise RuntimeError(f'expected the ten-context protocol cell, found {len(thesis_contexts)}')

    def cell(c, geo, variant='diffuser', tag=None):
        """One thesis cell, always reduced to Chapter 5's same ten contexts."""
        return [r for r in raw_cell(c, geo, variant, tag) if fingerprint(r) in thesis_contexts]

    # The thesis cells. Consistency training = floor 0.2 at the final checkpoint (closure: AFAFend0p2,
    # state_100000). MeanFlow at K=2 = FiLM v1, the conditioning of every other cell.
    TH = {
        'MeanFlow K2':             ('mf_K2', 'mv1_Emf'),
        'MeanFlow K10':            ('mf_K10', None),
        'MeanFlow K20':            ('mf_K20', None),
        'consistency tr. K2':      ('af_K2', '_msgafon02_s6'),
        'consistency tr. K20':     ('af_K20', '_msgafon02_s6'),
        'flow matching K20':       ('fm_K20', None),
        'diffusion K20':           ('diff_K20', None),
        'diffusion K100':          ('diff_K100', None),
    }

    splits = sorted({r['split'] for c, t in TH.values() for r in cell(c, U, tag=t)})
    print(f'split of every thesis cell: {splits}')

    print('\n== unprojected plan, untightened (the only geometry with the diffusion model) ==')
    for lab, (c, t) in TH.items():
        print('  ' + summary(lab, cell(c, U, tag=t)))
    d3 = [r for k, v in rows.items() if k[0] == 'd3il' for r in v]
    print('  ' + summary('D3IL image policy (test split)', d3))
    print(f'  initial distance, thesis contexts: '
          f'{st.mean(fnum(r["context_init_xy_dist"]) for r in cell("mf_K20", U)):.4f} m')

    g = lambda lab, geo=U: cell(TH[lab][0], geo, tag=TH[lab][1])
    print('\n== paired over the same ten contexts, unprojected, untightened ==')
    for a, b in (('MeanFlow K20', 'diffusion K20'), ('flow matching K20', 'MeanFlow K20'),
                 ('flow matching K20', 'diffusion K20'), ('consistency tr. K20', 'MeanFlow K20'),
                 ('consistency tr. K20', 'flow matching K20'), ('consistency tr. K20', 'diffusion K20'),
                 ('consistency tr. K2', 'MeanFlow K2')):
        print('  ' + pair(f'{a} - {b}', g(a), g(b)))

    print('\n== the same on tightened geometry (no diffusion cell exists there) ==')
    for a, b in (('flow matching K20', 'MeanFlow K20'), ('consistency tr. K20', 'MeanFlow K20'),
                 ('consistency tr. K2', 'MeanFlow K2')):
        print('  ' + pair(f'{a} - {b}', g(a, T), g(b, T)))
    for lab in ('MeanFlow K2', 'MeanFlow K20', 'consistency tr. K2', 'consistency tr. K20'):
        print('  ' + summary(lab + ' [tightened]', g(lab, T)))

    # Thesis Table tab:va-projection (v3.26): analytic average-velocity matching on tightened constraints,
    # per-step (dpcc-*) against endpoint projection (hardflow_sls-*), every selection rule.
    print('\n== projection methods, MeanFlow, tightened (tab:va-projection) ==')
    print('  violation-free = contexts with constraint_exec_zero_violation == 1; moved = not untouched')
    for c in ('mf_K20', 'mf_K10'):
        for var in ('diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t', 'hardflow_sls-r', 'hardflow_sls-c', 'hardflow_sls-t'):
            rs = cell(c, T, variant=var)
            if not rs:
                continue
            zv = sum(1 for r in rs if fnum(r.get('constraint_exec_zero_violation')) == 1)
            mv = sum(1 for r in rs if not untouched(r))
            print(f'  {c:7s} {var:16s} n={len(rs):2d} violation-free={zv}/{len(rs)} '
                  f'viol={st.mean(fnum(r["n_violations"]) for r in rs):5.1f} moved={mv}/{len(rs)} '
                  f'median={st.median(per_context(rs).values()):.3f} '
                  f'ms/step={st.mean(fnum(r["avg_time_ms"]) for r in rs):.1f}')
    rs = cell('diff_K20', U, variant='dpcc-r')
    print(f'  diffusion K20 dpcc-r [untightened] n={len(rs)} violation-free='
          f'{sum(1 for r in rs if fnum(r.get("constraint_exec_zero_violation")) == 1)}/{len(rs)} '
          f'viol={st.mean(fnum(r["n_violations"]) for r in rs):.1f} '
          f'unmoved={sum(1 for r in rs if untouched(r))}/{len(rs)} '
          f'ms/step={st.mean(fnum(r["avg_time_ms"]) for r in rs):.1f}')


if __name__ == '__main__':
    main()
