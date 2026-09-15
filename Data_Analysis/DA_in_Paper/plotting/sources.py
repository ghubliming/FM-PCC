#!/usr/bin/env python3
"""The single point of truth for *where the thesis numbers and figures come from*.

Part of Data_Analysis/DA_in_Paper/plotting -- the official figure pipeline of the
thesis (live since 2026-09-14). Moved here from Working_Space/v3/plots/.

═══════════════════════════════════════════════════════════════════════════════
 WHEN NEW DATA LANDS, THIS IS THE ONLY FILE YOU EDIT.
═══════════════════════════════════════════════════════════════════════════════
Point a ``Corpus`` at the new batch directory, update its ``protocol`` string,
re-run ``python3 make_figs.py`` and every figure in ``../figures/<group>/`` is rebuilt
against the new data with its subtitle and provenance line updated. No figure
script contains a path.

The registry mirrors, one row for one row, section 10 ("Corpora of record") of
``logs_in_develop/Writing/Working_Space/data_status/DATASTATUS_20260910_v3_entry_readiness.md``,
plus the corpora the thesis has cited since. If the two ever
disagree, that file is right and this one is stale.

A note on aggregation, because it is the one thing easy to get wrong here.
``avoiding`` cells are aggregated **per geometry first, then across the three
geometries** -- never as a flat mean over (seed x geometry) cells. The two differ
whenever a geometry has a different number of seeds, which is exactly the case
for the pinned baseline (``both-hard`` is seed-6-only). Geometry-mean is what the
DA of record used, and it reproduces its published numbers exactly:

    DPCC K20 dpcc-c-tightened  ->  S&C 0.983 | 69.0 steps | 0.5635 s/step
    MeanFlow K1 dpcc-t-tightened -> S&C 0.993 | 61.0 steps | 0.0181 s/step

against DA_20260827 section 10.1's 0.983 / 69.0 / 564 ms and 0.993 / 61.0 /
18.1 ms. A flat cell-mean gives 0.977 / 72.5 / 544 ms and would silently
contradict the text. ``load_avoiding`` therefore returns per-cell values and
``geometry_mean`` does the aggregation; do not reimplement it inline.
"""
import collections
import csv
import os
import re
import statistics as st

REPO = '/workspaces/FM-PCC'

# Plain-JSON extracts written by plotting/extract/*.py (they need numpy / PyYAML, the
# builders do not). Rebuild with: python3.14 plotting/extract/avoiding_scene.py
AVOIDING_SCENE = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'avoiding_scene.json')


class Corpus:
    """One batch directory, with the protocol that produced it.

    ``protocol`` is not decoration: it is printed into every figure subtitle and
    into figures/MANIFEST.md, because TARGET section 6 rule 7 forbids reporting a
    bare *n*. ``status`` is the DATASTATUS grade, carried so a figure can refuse
    to imply more strength than its corpus has.
    """

    def __init__(self, key, path, protocol, status, note=''):
        self.key, self.rel, self.protocol, self.status, self.note = key, path, protocol, status, note
        self.path = os.path.join(REPO, path)

    @property
    def available(self):
        return os.path.isdir(self.path)

    def csv(self, name='candidates_multidimensional_raw.csv'):
        return os.path.join(self.path, name)

    def __repr__(self):
        return f'<Corpus {self.key} {"ok" if self.available else "MISSING"}>'


# ═══════════════════════════════════════════════════════════════════════════
#  THE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════
CORPORA = {
    # ---- entry A: the state-based manipulation benchmark (the foundation) ----
    'avoiding_t2': Corpus(
        'avoiding_t2',
        'temp/2508/batch_avoiding_combined_20260825_143212',
        '5 seeds (6-10) x 20 trials = 100 episodes per cell',
        'ready',
        'Tier 2, "ours". The only multi-seed corpus in the project. '
        'DPCC both-hard is seed-6 only; FM K20 both-hard is seeds 6-7 + a partial 8.'),
    'avoiding_af_unet': Corpus(
        'avoiding_af_unet',
        'temp/0309/batch_avoiding_combined_20260903_133730',
        'seed 6, 20 trials',
        'partial',
        'Consistency training on the architecture-matched U-Net. Single seed: '
        'the mf/af separation here is p = 0.231 and must never be drawn as a ladder.'),
    'avoiding_minK': Corpus(
        'avoiding_minK',
        'temp/0609/I/batch_avoiding_combined_20260906_125724',
        '4 seeds (7-10) x 3 geometries x 2 trials = 24 rollouts per row',
        'partial',
        'Endpoint projection at its budget floor, job 25444. Seed 6 absent, so it is '
        'NOT seed-matched to the pinned baseline. Timing survives the small n; safety does not.'),
    'avoiding_t1': Corpus(
        'avoiding_t1',
        'temp/1808/batch_avoiding_combined_20260818_152911',
        '5 seeds x 2 trials = 10 episodes per cell',
        'partial',
        'Tier 1, "theirs" -- DPCC\'s own published protocol. BASELINE ROWS ONLY: the flow '
        'arms exist only in pre-2026-08-19 batches on an older eval binary, and the '
        'post_processing rows are corrupt (they resolved to the dpcc-r projector). '
        'A clean Tier 1 needs the fresh n_trials=2 run, DATASTATUS section 8 item 3.'),

    # ---- entry B: the vision-conditioned manipulation task -------------------
    'visual_aligning': Corpus(
        'visual_aligning',
        'temp/0609/II/batch_va2_20260907_141036',
        'seed 6, 10 enumerated contexts x 1 trajectory (paired)',
        'partial',
        'No n_trials by construction: contexts are enumerated and held fixed, which is '
        'what makes every comparison paired. n_contexts=50 is the owed upgrade.'),

    # ---- entry C: the aerial task -------------------------------------------
    'uav': Corpus(
        'uav',
        'temp/0909/batch_uav_20260910_092309',
        'seed 6, n = 10 rollouts',
        'blocked',
        'One seed, one scene for the ranking. corridor is constraint-trivial; s_curve is '
        'the controller-limit case, never an engine table; pillars K=5 carries the '
        'ranking and is blocked on seeds.'),
    'uav_corridor_v2': Corpus(
        'uav_corridor_v2',
        'temp/1409/batch_uav_20260914_091148',
        'seed 6, 12 flights per configuration (4 per route L/C/R)',
        'partial',
        'Corridor with walls 1.90 m apart and a 14 deg test-time slide (U16). Projection '
        'configuration differs from pillars/s_curve (-bounds_free-pdes-tightened): never pooled.'),
    'uav_pillars_diffusion': Corpus(
        'uav_pillars_diffusion',
        'temp/1209/batch_uav_20260912_201035',
        'seed 6, 10 flights per configuration',
        'partial',
        'Diffusion baseline reference on pillars, trained with action loss weight 1; budgets '
        'unmatched by construction (training-time K vs inference-time K).'),
}


# ═══════════════════════════════════════════════════════════════════════════
#  VENDORED FIGURES -- assets copied in, not generated here
# ═══════════════════════════════════════════════════════════════════════════
# Some panels were produced by the evaluation's own diagnostics rather than by
# this pipeline, and cannot be regenerated from a CSV. They are copied rather
# than redrawn, and every one carries its PROVENANCE, because a figure whose
# origin is unrecorded is how the baseline authors' own plots nearly ended up in
# this thesis presented as ours (see CHANGELOG v3.3).
#
# RULE: nothing goes in here until it has been checked against
# /workspaces/aux_repo/ and shown to be ours. All four below were.
#
# Entry: name -> (group, source path relative to the repo, provenance).
VENDORED = {
    'fig_raw_plans_meanflow_K1': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/'
        'fig6a_plans_mfunet_K1_seed6_both-hard.png',
        'Report_20260819_MF_UNet fig 6a; per-episode MPC diagnostics, unprojected arm, '
        'seed 6, both-hard, K=1. Ours; verified absent from aux_repo.'),
    'fig_raw_plans_diffusion_K1': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/'
        'fig6b_plans_dpcc_K1_seed6_both-hard.png',
        'Report_20260819_MF_UNet fig 6b; same protocol, diffusion baseline at K=1. '
        'Ours; verified absent from aux_repo.'),
    'fig_raw_plans_meanflow_K2': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/'
        'fig6c_plans_mfunet_K2_seed6_both-hard.png',
        'Report_20260819_MF_UNet fig 6c; same protocol, K=2. '
        'Ours; verified absent from aux_repo.'),
    'fig_raw_goal_reached_K1': ('da',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/fig7_raw_diffuser_K1.svg',
        'Report_20260903_AF_UNet fig 7; goal reached on the unprojected arm at K=1, '
        'top-right-hard, seed 6, 20 trials. Ours; verified absent from aux_repo.'),
}


def vendored_path(key):
    return os.path.join(REPO, VENDORED[key][1])


# ═══════════════════════════════════════════════════════════════════════════
#  PLANNED FIGURES -- specified in a draft, not yet made
# ═══════════════════════════════════════════════════════════════════════════
# Each mirrors a \todofigure box in a draft. When one is made, put the file into
# ../figures/<group>/<name>.<ext>, switch the draft from \todofigure to
# \includegraphics{<name>}, remove the entry here, and export.
# Entry: name -> (group, where the draft asks for it, what it must show / how).
PLANNED = {
    'fig_env_aligning': ('env', 'v3 sec:setup:tasks:aligning (fig:env-aligning)',
        'Alignment scene with box and target pose, next to the two camera images the policy '
        'receives (overhead and in-hand).'),
    'fig_env_uav': ('env', 'v3 sec:setup:tasks:uav (fig:env-uav)',
        'Corridor (walls 1.90 m apart + 14 deg slide, routes L/C/R), pillars, s-curve: overhead, '
        'common scale, vehicle drawn to scale (0.62 m across). Geometry: tab:uav-scenes.'),
}


# ═══════════════════════════════════════════════════════════════════════════
#  EXCLUDED -- existing images that must NOT become thesis figures
# ═══════════════════════════════════════════════════════════════════════════
EXCLUDED = {
    'figures/avoiding.png': 'byte-identical to aux_repo/dpcc/figures/avoiding.png -- the DPCC '
                            "authors' figure (checked 2026-09-11, v3 CHANGELOG v3.3)",
    'figures/avoiding_constraints.png': 'byte-identical to the DPCC copy; regenerate instead (PLANNED)',
    'figures/avoiding_data.png': 'byte-identical to the DPCC copy',
}


# ═══════════════════════════════════════════════════════════════════════════
#  ENGINES, ARMS AND RULES -- the plotting vocabulary
# ═══════════════════════════════════════════════════════════════════════════
# Thesis names, per Auxiliary/Naming/NAMING_20260910_master_table.md. Code tokens
# appear ONLY in the `folder` patterns, never in a label that reaches a figure.
ENGINE_COLOUR = {
    'diffusion': '#c0392b',   # the inherited denoising engine -- the pinned baseline
    'fm':        '#2471a3',   # flow matching        (alpha = 1)
    'mf':        '#1e8449',   # MeanFlow             (alpha = 0)
    'af':        '#6c3483',   # consistency training    (0 < alpha < 1)
}
ENGINE_LABEL = {
    'diffusion': 'diffusion (DPCC)',
    'fm':        'flow matching',
    'mf':        'MeanFlow',
    'af':        'consistency training',
}

# Folder-name pattern per engine, with %d for the step budget K. These ARE code
# tokens and they belong here rather than in a figure script.
AVOIDING_T2_FOLDERS = {
    'diffusion': 'H8_K%d_T0.5_Dmodels.GaussianDiffusion_msg20trials',
    'fm':        'H8_K%d_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials',
    'mf':        'H8_K%d_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials',
    # SiT backbone -- a CONFOUNDED row. Figures that include it must label it so.
    'af':        'H8_K%d_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msg20trials',
}
AVOIDING_T2_BACKBONE = {'diffusion': 'U-Net 4.0M', 'fm': 'U-Net 4.0M',
                        'mf': 'U-Net 4.0M', 'af': 'SiT (confounded)'}

# The baseline's low-budget cells, which live in the SAME batch directory but at
# the PUBLISHED protocol (5 seeds x 2 trials = 10 episodes), not ours. They are
# the only measurement of what the diffusion engine does below its training
# budget, and DATASTATUS section 2.2 claim A4 is explicit about how to use them:
# as the MECHANISM sentence, never as a powered comparison. A clean Tier 1 is an
# owed run (section 8 item 3) and cannot be subsampled out of Tier 2, because the
# avoiding corpus has already collapsed the trial dimension.
#
# Provenance check, not a guess: the K=1 rows here read 0.60 (top-left) and 0.50
# (top-right) on the cumulative-cost tightened rule, which is exactly what
# DA_20260819 section 1b quotes for "DPCC/aw10 at K=1". The K=10 rows read 1.00 /
# 1.00, likewise as quoted. Cell means land on multiples of 0.1 across 5 seeds,
# which is the granularity 2 trials per seed produces -- consistent with n=2 and
# NOT with n=20.
AVOIDING_T1_DIFFUSION_FOLDERS = {
    1:  'H8_K1_T0.5_Dmodels.GaussianDiffusion',
    10: 'H8_K10_Dmodels.GaussianDiffusion_aw10_thres0.5',
    20: 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5',
}
AVOIDING_T1_PROTOCOL = '5 seeds x 2 trials = 10 episodes per cell (the published protocol)'

GEOMETRIES = ['top-left-hard', 'top-right-hard', 'both-hard']
RULE_LABEL = {'dpcc-r': 'random', 'dpcc-c': 'cumulative projection cost',
              'dpcc-t': 'temporal consistency'}
METRICS = ('n_success_and_constraints', 'n_steps', 'avg_time')

RE_K = re.compile(r'H8_K(\d+)[_.]')


# ═══════════════════════════════════════════════════════════════════════════
#  LOADERS
# ═══════════════════════════════════════════════════════════════════════════
def load_avoiding(corpus, folders=None, seeds=None):
    """-> {(engine, K, seed, geometry, variant): {metric: value}}

    One dict entry is one evaluated cell, un-aggregated. Aggregation is the
    caller's job and should go through geometry_mean().
    """
    folders = folders or AVOIDING_T2_FOLDERS
    # invert: exact folder name -> (engine, K)
    index = {}
    for eng, pat in folders.items():
        for K in (1, 2, 3, 5, 10, 20):
            index[pat % K] = (eng, K)
    out = collections.defaultdict(dict)
    with open(corpus.csv()) as f:
        for r in csv.DictReader(f):
            hit = index.get(r['Folder_Name'])
            if hit is None:
                continue
            if seeds and r['seed'] not in seeds:
                continue
            eng, K = hit
            try:
                out[(eng, K, r['seed'], r['halfspace_variant'], r['variant'])][r['metric']] \
                    = float(r['value'])
            except (TypeError, ValueError):
                pass
    return dict(out)


def load_exact(corpus, folders, engine, seeds=None):
    """Load rows for a {K: exact_folder_name} map, under one engine label.

    ``load_avoiding`` takes a ``%d`` pattern, which assumes an engine's folder
    name differs across K only in the budget. That is true for the runs of one
    sweep and false across sweeps -- the baseline's low-budget cells were
    produced by three different jobs with three different folder spellings. This
    loader takes the spellings literally.
    -> {(engine, K, seed, geometry, variant): {metric: value}}
    """
    index = {folder: K for K, folder in folders.items()}
    out = collections.defaultdict(dict)
    with open(corpus.csv()) as f:
        for r in csv.DictReader(f):
            K = index.get(r['Folder_Name'])
            if K is None or (seeds and r['seed'] not in seeds):
                continue
            try:
                out[(engine, K, r['seed'], r['halfspace_variant'], r['variant'])][r['metric']] \
                    = float(r['value'])
            except (TypeError, ValueError):
                pass
    return dict(out)


def load_by_candidate(corpus, folder_contains, seeds=None):
    """Looser loader: match folders by substring, keep K from the folder name.

    Used where a corpus has one engine under several run-message suffixes, e.g.
    the endpoint-projection budget-floor job.
    -> {(K, seed, geometry, variant): {metric: value}}
    """
    out = collections.defaultdict(dict)
    with open(corpus.csv()) as f:
        for r in csv.DictReader(f):
            fn = r['Folder_Name']
            if not all(p in fn for p in folder_contains):
                continue
            m = RE_K.search(fn)
            if not m:
                continue
            if seeds and r['seed'] not in seeds:
                continue
            try:
                out[(int(m.group(1)), r['seed'], r['halfspace_variant'], r['variant'])][r['metric']] \
                    = float(r['value'])
            except (TypeError, ValueError):
                pass
    return dict(out)


def geometry_mean(cells, key_fn, geom_index, geometries=None, metrics=METRICS):
    """Per-geometry mean, then mean across geometries. See the module docstring.

    ``key_fn(cell_key) -> group`` says which rows belong together; rows whose
    key_fn returns None are skipped. ``geom_index`` is the position of the
    geometry in the cell key -- 3 for load_avoiding's 5-tuple, 2 for
    load_by_candidate's 4-tuple. It is a required argument on purpose: inferring
    it from the tuple length silently picked the *variant* column for 4-tuples,
    which produced an empty result rather than a wrong one only by luck.
    Returns {group: {'n_cells', 'n_geometries', metric: value, ...}}.
    """
    geometries = geometries or GEOMETRIES
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, d in cells.items():
        g = key_fn(k)
        if g is None or not set(metrics) <= set(d):
            continue
        geom = k[geom_index]
        per[g][geom].append(d)
    out = {}
    for g, bygeom in per.items():
        usable = {h: ds for h, ds in bygeom.items() if h in geometries and ds}
        if not usable:
            continue
        row = {'n_cells': sum(len(ds) for ds in usable.values()),
               'n_geometries': len(usable)}
        for m in metrics:
            row[m] = st.mean(st.mean(d[m] for d in ds) for ds in usable.values())
        out[g] = row
    return out


def pareto_front(points, x='avg_time', y='n_steps', quality='n_success_and_constraints',
                 band=0.05, tol=1e-9):
    """Non-dominated staircase over the points within ``band`` of the best quality.

    This is the project's standing definition of "good" and it is deliberately
    strict: a point is only eligible if its success-and-constraints score is
    within the band of the best on the panel, and only then may it compete on
    cost. Matches Data_Analysis/Visualizer_VA_v2/index.html.
    Returns (eligible_flags, front) where front is ordered by x.
    """
    best = max(p[quality] for p in points)
    for p in points:
        p['eligible'] = p[quality] >= best - band - tol
    front, ybest = [], None
    for p in sorted((q for q in points if q['eligible']), key=lambda q: (q[x], q[y])):
        if ybest is None or p[y] < ybest - tol:
            front.append(p)
            ybest = p[y]
    return points, front
