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

# The FLOWN quadrotor paths of fig_uav_{scurve,pillars,corridor}_paths. Rebuild with:
#   python3.14 plotting/extract/uav_paths.py [--root <folder holding logs/>]
# Unlike every other corpus here this one does NOT point at a batch directory: the
# executed positions are not in the 15-09 batches at all (those carry per-rollout
# scalars), so they were staged off the cluster on 2026-09-18 by
# Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh into the gitignored
# temp/18-09-2026/. That drop is NOT committed and will not survive a clean checkout --
# this extract is the committed record, and the ledger below is how to recreate it.
#   ledger: Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md
#   request: plotting/REQUEST_20260917_trajectory_figures.md
# The run folder and variant of every panel are declared in extract/uav_paths.py::PANELS.
UAV_PATHS = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'uav_paths.json')
UAV_PATHS_DROP = os.path.join(REPO, 'temp', '18-09-2026')


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
#  SCENE GEOMETRY, for the environment figures
# ═══════════════════════════════════════════════════════════════════════════
# The quadrotor scenes are MJCF and are parsed straight out of this directory, so
# nothing about them is duplicated here.
UAV_SCENE_DIR = 'd3il/environments/d3il/models/mj/robot/quadrotor/scenes'

# The constraint set each quadrotor scene is flown against, transcribed from the
# evaluation configuration so the figure cannot drift from what was projected:
#   config/uav_projection.yaml :: geo_constraint_variants, entries
#   corridor_v2_slide (the corridor the results use), pillars_hg, s_curve_hg.
# Halfspaces are (line through p0,p1; the side the plan must stay on; the x span on
# which the constraint is active). Disks are sphere_outside constraints in (x, y).
# Every boundary is inflated by the vehicle radius before it reaches the solver
# (planning_inflation.r_drone), and a tightened variant shrinks it by
# enlarge_constraints on top of that.
UAV_CONSTRAINTS = {
    'r_drone': 0.31,
    'tightening': 0.025,
    'scenes': [
        {'name': 'UAV-corridor', 'title': 'UAV-corridor',
         'sub': 'two walls and the test-time slide',
         'xlim': (-2.8, 2.8), 'ylim': (-1.4, 1.4),
         'halfspaces': [
             {'p0': (-2.0, -0.95), 'p1': (2.0, -0.95), 'side': 'above', 'x_active': (-2.0, 2.0)},
             {'p0': (-2.0, 0.95), 'p1': (2.0, 0.95), 'side': 'below', 'x_active': (-2.0, 2.0)},
             {'p0': (-2.0, 0.95), 'p1': (2.0, -0.05), 'side': 'below', 'x_active': (-2.0, 2.0),
              'label': 'slide'},
         ],
         'disks': [{'c': (-2.0, -1.0), 'r': 0.05}, {'c': (2.0, -1.0), 'r': 0.05},
                   {'c': (-2.0, 1.0), 'r': 0.05}, {'c': (2.0, 1.0), 'r': 0.05}]},
        # [2026-09-19] UAV-pillars is drawn at the ENLARGED test-time radius of `pillars_xl`
        # (config/uav_projection.yaml, Gen15 U17: radius 0.35 on all six sphere_outside rows),
        # not at the 0.12 of `pillars_hg`. `pillars_hg` enforced exactly the clearance its own
        # demonstration generator was built to keep, so the demonstrated routes satisfied it and
        # the scene measured nothing; it is withdrawn from the thesis and being re-evaluated at
        # 0.35. `r_phys` is the pillar the simulator actually contains, which does NOT change --
        # the MJCF geometry stays at 0.12 so the demonstrations and checkpoints remain valid, and
        # a panel therefore shows a small physical pillar inside a larger enforced keep-out disk.
        # That difference is the point of the redesign and the drawing carries it.
        {'name': 'UAV-pillars', 'title': 'UAV-pillars',
         'sub': 'six pillars, enlarged at test time',
         'xlim': (-3.0, 3.0), 'ylim': (-1.9, 1.9),
         'halfspaces': [],
         'disks': [{'c': (x, y), 'r': 0.35, 'r_phys': 0.12}
                   for x in (-2.0, 0.0, 2.0) for y in (-0.6, 0.6)]},
        {'name': 'UAV-s-curve', 'title': 'UAV-s-curve',
         'sub': 'two offset passages',
         'xlim': (-3.4, 3.4), 'ylim': (-1.7, 1.7),
         'halfspaces': [
             {'p0': (-3.0, -0.35), 'p1': (-0.5, -0.35), 'side': 'below', 'x_active': (-3.0, -0.5)},
             {'p0': (-3.0, -1.25), 'p1': (-0.5, -1.25), 'side': 'above', 'x_active': (-3.0, -0.5)},
             {'p0': (0.5, 0.35), 'p1': (3.0, 0.35), 'side': 'above', 'x_active': (0.5, 3.0)},
             {'p0': (0.5, 1.25), 'p1': (3.0, 1.25), 'side': 'below', 'x_active': (0.5, 3.0)},
         ],
         'disks': [{'c': (-0.5, -0.3), 'r': 0.05}, {'c': (0.5, 0.3), 'r': 0.05}]},
    ],
}

# The two D3IL manipulation scenes are built in Python, not XML, so their
# primitives are transcribed here WITH the symbol they come from. MuJoCo cylinder
# size is (radius, half-height); a cylinder standing on the table therefore has
# its centre one half-height above it.
#
#   avoiding  aux_repo/d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/
#             envs/objects/avoiding_objects.py :: get_obj_list, init_end_eff_pos
#   aligning  .../gym_aligning_env/gym_aligning/envs/objects/aligning_objects.py
#             :: box_pos, init_end_eff_pos, and the box geoms in
#             models/mj/common-objects/robot_push_box/robot_push_box.xml
# The Skydio X2 as the experiments load it, transcribed from
# d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml:
#   rotor geoms at (+/-0.14, +/-0.18, z), rotor disk radius 0.13 (class `rotor`,
#   ellipsoid size .13 .13 .01); body ellipsoid size .16 .04 .02; masses 4 x 0.25
#   (rotors) + 0.325 (body) = 1.325 kg. The inflation radius is the one the
#   projection uses for every obstacle (UAV_CONSTRAINTS['r_drone']).
X2_GEOMETRY = {
    'rotor_xy': [(-0.14, -0.18), (-0.14, 0.18), (0.14, 0.18), (0.14, -0.18)],
    'rotor_radius': 0.13,
    'body_semi_axes': (0.16, 0.04),
    'mass_kg': 1.325,
}

# The constraint set of the alignment task, from the evaluation configuration:
#   config/visual_aligning_eval.yaml :: active_geo_variants [combined_5],
#   combined_5.workspace_bounds, enlarge_constraints, action_bounds: 'auto'.
# The projection enforces a halfspace and a circular keep-out region on the planned
# end-effector position, exactly the two forms D3IL-avoiding uses. The workspace box of
# the same entry is the reachable table area, not a task constraint, and is used here only
# to set the plotted extent -- it is never drawn as a constraint.
ALIGNING_CONSTRAINTS = {
    'name': 'combined-5',
    'extent': {'x': (0.20, 0.80), 'y': (-0.45, 0.45)},
    'halfspace': {'p0': (0.65, 0.45), 'p1': (0.80, -0.45), 'side': 'below'},
    'disk': {'c': (0.50, 0.00), 'r': 0.06},
    'tightening': 0.03,
}

# The ten evaluation contexts of the alignment task (v3.42, Fig 5.5).
# A context is one (initial box pose, target pose) draw; the evaluation replays this
# enumerated set, so these are the exact ten every §6.2 number is averaged over.
# Recovered from the corpus of record rather than re-read from the pickle, so that the
# figure and the results cannot disagree: distinct
# (context_box_init_xy_*, context_target_xy_*) tuples of the mf_K20 cell of
# analysis_results_checkpoint/15-09/batch_va2_20260915_100754/per_rollout_detail.csv.
# Their mean box-to-target distance is 0.4530 m, which is the figure §6.2 quotes.
# Draw ranges are the environment's own: gym_aligning/envs/aligning.py:62-67.
ALIGNING_CONTEXT_DRAW = {
    'box':    {'x': (0.40, 0.60), 'y': (-0.25, -0.10), 'yaw': (-90.0, 90.0)},
    'target': {'x': (0.40, 0.60), 'y': (0.20, 0.35),   'yaw': (-90.0, 90.0)},
}
ALIGNING_CONTEXTS = [
    {'box': (0.4012, -0.2271), 'box_yaw': 77.92, 'target': (0.4599, 0.2034), 'target_yaw': -84.81},
    {'box': (0.4540, -0.1901), 'box_yaw': -41.80, 'target': (0.5260, 0.2785), 'target_yaw': -53.45},
    {'box': (0.4827, -0.2017), 'box_yaw': -6.43, 'target': (0.4327, 0.2056), 'target_yaw': 86.50},
    {'box': (0.4987, -0.1443), 'box_yaw': 82.14, 'target': (0.5570, 0.2582), 'target_yaw': -11.05},
    {'box': (0.5036, -0.2487), 'box_yaw': -60.36, 'target': (0.4275, 0.2680), 'target_yaw': -67.62},
    {'box': (0.5151, -0.2321), 'box_yaw': 79.35, 'target': (0.5773, 0.2227), 'target_yaw': 58.54},
    {'box': (0.5339, -0.1328), 'box_yaw': 73.50, 'target': (0.5925, 0.3158), 'target_yaw': -31.86},
    {'box': (0.5676, -0.1968), 'box_yaw': -5.53, 'target': (0.4852, 0.2803), 'target_yaw': 5.76},
    {'box': (0.5848, -0.1834), 'box_yaw': -37.37, 'target': (0.4773, 0.2173), 'target_yaw': -21.13},
    {'box': (0.5859, -0.1598), 'box_yaw': -89.34, 'target': (0.5632, 0.3109), 'target_yaw': -25.28},
]
# The one context drawn with real footprints rather than as a dot, so the reader sees the
# scale of the object the dots stand for. Index 8 has the box and target well apart in x.
ALIGNING_CONTEXT_SHOWN = 8

D3IL_SCENES = {
    'avoiding': {
        'start': [0.525, -0.28, 0.12],
        # mid 0.5, offset 0.075, first level y -0.1, level distance 0.18
        'obstacles': [
            {'pos': [0.500, -0.10, 0], 'size': [0.030, 0.07]},
            {'pos': [0.425,  0.08, 0], 'size': [0.025, 0.10]},
            {'pos': [0.575,  0.08, 0], 'size': [0.025, 0.10]},
            {'pos': [0.350,  0.26, 0], 'size': [0.025, 0.10]},
            {'pos': [0.500,  0.26, 0], 'size': [0.025, 0.10]},
            {'pos': [0.650,  0.26, 0], 'size': [0.025, 0.10]},
        ],
        'finish_line': {'pos': [0.400, 0.35, 0], 'half': [0.5, 0.01, 0.005]},
    },
    'aligning': {
        'start': [0.525, -0.35, 0.25],
        # The box drawn to scale. MuJoCo half-sizes in
        # d3il/environments/d3il/models/mj/common-objects/robot_push_box/robot_push_box.xml:
        # base geom 0.05 x 0.05 x 0.01, four coloured rims of half-thickness 0.005 at +-0.05.
        # So the base footprint is 0.10 m square and the outer footprint 0.11 m square.
        # target_box.xml carries the same sizes as sites.
        'box_half': 0.05,
        'box_rim_half': 0.055,
        # First held-out context, not the XML's coincident placeholder poses.
        # d3il/environments/dataset/data/aligning/test_contexts.pkl, index 0.
        'box_pos': [0.58057404, -0.20366790, 0.0],
        'box_yaw_deg': -43.513584,
        'target_pos': [0.49818864, 0.33333877, 0.0],
        'target_yaw_deg': -58.283190,
    },
}

# Frames extracted once into data/prepared/ by prep/extract_env_frames.py.  The
# D3IL montage is tracked; the UAV diagnostics are local copies of real cluster
# rollouts.  The prepared PNGs are committed so a fresh checkout can build the
# thesis even when the ignored temp/ tree is absent.
# Quadrotor scenes rendered by MuJoCo itself (prep/render_mujoco_scenes.py), from the
# scene MJCF and the Skydio X2 mesh, with the demonstration generator's reference path
# drawn in. These replace the frames once cut from rollout GIFs, which were 140 px,
# mostly black and carried the diagnostic step counter. Camera values are MuJoCo free
# camera conventions: azimuth/elevation in degrees, distance in metres.
MUJOCO_RENDERS = {
    # Camera behind the start (azimuth ~25 deg looks along +x), so the vehicle is in
    # the foreground and the course runs away from the reader in the direction flown.
    # Path in green: the pillars are orange in the scene file.
    'fig_render_uav_corridor': {
        'scene': 'scene_corridor_v2.xml', 'size': (1600, 1100),
        'path_fn': 'corridor_path', 'path_args': ('C', 1.1, 8.0),
        'lookat': (-0.6, 0.0, 0.8), 'distance': 5.6, 'azimuth': 14.0, 'elevation': -24.0,
        'path_rgba': (0.20, 0.85, 0.35, 1.0),
    },
    'fig_render_uav_pillars': {
        'scene': 'scene_pillars.xml', 'size': (1600, 1100),
        'path_fn': 'pillar_path', 'path_args': (('L', 'R', 'L'), 1.1, 13.0),
        'lookat': (-0.6, 0.0, 0.8), 'distance': 6.6, 'azimuth': 28.0, 'elevation': -26.0,
        'path_rgba': (0.20, 0.85, 0.35, 1.0),
    },
    'fig_render_uav_scurve': {
        'scene': 'scene_s_curve.xml', 'size': (1600, 1100),
        'path_fn': 's_curve_scene_path', 'path_args': (1.1, 19.0),
        'lookat': (-0.6, -0.3, 0.7), 'distance': 6.6, 'azimuth': 12.0, 'elevation': -34.0,
        'path_rgba': (0.20, 0.85, 0.35, 1.0),
    },
}

# The two simulated platforms on their own, for Fig. 5.1: no scene, no floor, no sky.
# Rendered by prep/render_mujoco_scenes.py from the model files the simulators load; the
# background is removed with MuJoCo's segmentation pass (every pixel that hits no geom is
# set to white), and a thin contour is drawn along the mask so a white robot stays visible
# on a white page. Poses are illustrative: the Panda in its standard ready configuration,
# the X2 level.
PLATFORM_RENDERS = {
    'fig_platform_panda': {
        'model': '/workspaces/aux_repo/d3il/environments/d3il/models/mj/robot/panda.xml',
        'wrap_include': True,              # the file is a <mujocoinclude> fragment
        'meshdir': '/workspaces/aux_repo/d3il/environments/d3il/models/mj/robot/assets',
        'qpos_by_joint': {'panda_joint1': 0.0, 'panda_joint2': -0.785, 'panda_joint3': 0.0,
                          'panda_joint4': -2.356, 'panda_joint5': 0.0, 'panda_joint6': 1.571,
                          'panda_joint7': 0.785},
        'lookat': (0.25, 0.0, 0.45), 'distance': 2.0, 'azimuth': 140.0, 'elevation': -18.0,
        'size': (1400, 1400),
    },
    'fig_platform_x2': {
        'model': 'd3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml',
        'wrap_include': False,
        'free_body_pos': (0.0, 0.0, 0.5),
        'lookat': (0.0, 0.0, 0.5), 'distance': 1.1, 'azimuth': 135.0, 'elevation': -28.0,
        'size': (1400, 1400),
    },
}

ENV_RENDER_FRAMES = {
    'fig_render_avoiding': {
        'source': 'd3il/figures/github_readme.gif',
        'frame': 0,
        'crop': (0, 0, 320, 180),
        'what': 'D3IL obstacle-avoidance simulator view',
    },
    'fig_render_aligning': {
        'source': 'd3il/figures/github_readme.gif',
        'frame': 0,
        'crop': (640, 180, 960, 360),
        'what': 'D3IL alignment simulator view',
    },
    # One frame of an expert demonstration recorded by the evaluation pipeline.
    # The GIF is exactly the two 96x96 observations concatenated horizontally:
    # bp-cam first, wrist-mounted inhand-cam second. Rollout 1, frame 163: the
    # only instant in the three recorded rollouts at which the box is complete
    # and centred in the in-hand view (v3.27; frame 60 of rollout 0 cut it at
    # the corner). Ranked by the in-hand box's distance from the tile centre
    # and its clearance from the tile border.
    'fig_aligning_camera_overhead': {
        'source': 'temp/0408/mix_visual_aligning_mf/'
                  'H8_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_'
                  'a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal/'
                  'H8_K100_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.'
                  'VisualMeanFlow_VTrue_mpc4_filmv1_Emf/6/results_train_set/'
                  'expert_references/expert_rollout_1.gif',
        'frame': 163,
        'crop': (0, 0, 96, 96),
        'what': 'D3IL alignment expert demonstration, fixed overhead camera observation',
    },
    'fig_aligning_camera_wrist': {
        'source': 'temp/0408/mix_visual_aligning_mf/'
                  'H8_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_'
                  'a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal/'
                  'H8_K100_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.'
                  'VisualMeanFlow_VTrue_mpc4_filmv1_Emf/6/results_train_set/'
                  'expert_references/expert_rollout_1.gif',
        'frame': 163,
        'crop': (96, 0, 192, 96),
        'what': 'D3IL alignment expert demonstration, wrist-mounted camera observation',
    },



}

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
    # ---- the corpora of record of v3 Chapter 6 (2026-09-20) ------------------
    # These two are what the draft's alignment and quadrotor TABLES are computed from
    # (analysis/va_results.py, analysis/uav_results.py). The entries above are the older
    # batches the earlier figures were built on and are kept so those figures still build.
    'visual_aligning_15_09': Corpus(
        'visual_aligning_15_09',
        'Data_Analysis/analysis_results_checkpoint/15-09/batch_va2_20260915_100754',
        'seed 6, ten shared enumerated contexts (paired; raw cells with 30 contexts are restricted)',
        'partial',
        'The alignment corpus of record. Cells are selected by FolderName prefix, see '
        'ALIGNING_CELLS: the budget and the activation threshold are both in the prefix, so a '
        'prefix without its T token would pool two different thresholds.'),
    'uav_19_09': Corpus(
        'uav_19_09',
        'Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/batch_uav_20260919_111701',
        'seed 6, 12 flights per corridor configuration (4 per route L/C/R)',
        'partial',
        'The quadrotor corpus of record. UAV-pillars is EXCLUDED from it (the scene enforces the '
        'constraint its own demonstrations satisfy); s-curve is in it but scores 0.00 with '
        'constraint satisfaction in every configuration, so it carries no cost frontier.'),
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
# RULE: diagnostic result panels do not go in here until they have been checked
# against /workspaces/aux_repo/ and shown to be ours. The environment stills are
# instead traced to their simulator media by ENV_RENDER_FRAMES above.
#
# Entry: name -> (group, source path relative to the repo, provenance).
VENDORED = {
    'fig_platform_panda': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_platform_panda.png',
        'MuJoCo render of the D3IL Panda model file (aux_repo/d3il .../robot/panda.xml), ready pose, '
        'background removed. prep/render_mujoco_scenes.py, PLATFORM_RENDERS. Rendered here.'),
    'fig_platform_x2': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_platform_x2.png',
        'MuJoCo render of quadrotor_modified.xml (Skydio X2, MuJoCo Menagerie), level, background '
        'removed. prep/render_mujoco_scenes.py, PLATFORM_RENDERS. Rendered here.'),

    'fig_raw_plans_meanflow_K1': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/'
        'fig6a_plans_mfunet_K1_seed6_both-hard.png',
        'Report_20260819_MF_UNet fig 6a; per-episode MPC diagnostics, no projection, '
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

    # The four panels that completed fig:raw-plans on 2026-09-19. They were NOT
    # re-run: REQUEST_20260916's own "CORRECTION 2026-09-18 (v3.39)" found that the
    # August 20-trials evaluations had already written them, under the run tags
    # '*_msg20trials' (FM) and '*_msgafon02_s6' (CI-MeanFM). Staged off the cluster
    # by Slurm_Codes/temp_bash/fetch_20260919_v3_figure_artefacts_wave2.sh and
    # copied here under the names Report_20260903_AF_UNet section 8 already reserved,
    # so the source is committed like the other three and the gitignored drop is not
    # in the build path. Ledger: analysis_results_checkpoint/
    # LEDGER_20260918_v3_figure_artefact_fetch.md, 2026-09-19 section.
    'fig_raw_plans_fm_K1': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/'
        'fig8e_plans_fm_K1_seed6.png',
        'Report_20260903_AF_UNet fig 8e; per-episode MPC diagnostics, no projection, '
        'seed 6, both-hard, K=1, instantaneous-velocity matching (aw10). '
        'Ours; verified absent from aux_repo.'),
    'fig_raw_plans_fm_K2': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/'
        'fig8f_plans_fm_K2_seed6.png',
        'Report_20260903_AF_UNet fig 8f; same protocol, K=2. '
        'Ours; verified absent from aux_repo.'),
    'fig_raw_plans_af_K1': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/'
        'fig8a_plans_af_K1_seed6.png',
        'Report_20260903_AF_UNet fig 8a; same protocol, K=1, consistency-interpolated '
        'average-velocity matching (U-Net, alpha_end 0.2). '
        'Ours; verified absent from aux_repo.'),
    'fig_raw_plans_af_K2': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/'
        'fig8b_plans_af_K2_seed6.png',
        'Report_20260903_AF_UNet fig 8b; same protocol, K=2. '
        'Ours; verified absent from aux_repo.'),
    # The eighth and last panel of fig:raw-plans, landed 2026-09-19. It is the only
    # one of the eight that was a RUN rather than a fetch: no K=2 diffusion checkpoint
    # existed anywhere, because for the diffusion engine K is fixed when the noise
    # schedule is discretised at training (config/avoiding-d3il.py:1074 puts K in
    # diffusion_loadpath). So one was trained -- job 25965, 2 h 41 m on an A5000 -- and
    # evaluated by 25966, both on 2026-09-19. Staged by
    # Slurm_Codes/temp_bash/fetch_20260919_fig63_diffusion_K2.sh.
    # Ledger R24; data_status/PENDING_20260919_fig63_diffusion_K2_panel.md.
    'fig_raw_plans_diffusion_K2': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/'
        'fig8h_plans_dpcc_K2_seed6.png',
        'Report_20260903_AF_UNet fig 8h; per-episode MPC diagnostics, no projection, '
        'seed 6, both-hard, diffusion baseline TRAINED AND RUN at K=2 (aw10). '
        'Ours; verified absent from aux_repo.'),
    # The NINTH panel, landed 2026-09-20: the diffusion baseline at its OWN native
    # budget K=20, which the author asked for because that is the configuration every
    # number of Chapter 6 is measured against. A fetch, not a run -- but NOT from the
    # run REQUEST_20260916's 2026-09-20 addition names. That run
    # ('..._msg20trials') died mid-variant on 2026-08-18 after writing five of
    # thirteen variants, so its seed-6 both-hard 'diffuser' dashboard was never
    # written. This panel comes from the sibling '..._aw10_thres0.5' campaign, which
    # is complete. Same checkpoint, same K=20, same seed 6, same both-hard geometry;
    # 'thres0.5' is a PROJECTION threshold and this is the unprojected arm, so it
    # cannot have touched the plans drawn here. Layout is the 3000x1000 two-episode
    # one, so it takes the 08-19 box and needs NO resize -- verified, not assumed:
    # the cut lands on (56,28)-(388,378) at 403x400 like the other eight.
    # Ledger: LEDGER_20260918_v3_figure_artefact_fetch.md, 2026-09-20 section.
    'fig_raw_plans_diffusion_K20': ('demo',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/'
        'fig8i_plans_dpcc_K20_seed6.png',
        'Report_20260903_AF_UNet fig 8i; per-episode MPC diagnostics, no projection, '
        'seed 6, both-hard, diffusion baseline at its native K=20 (aw10), from the '
        'thres0.5 campaign. Ours; verified absent from aux_repo.'),
    'fig_raw_goal_reached_K1': ('da',
        'Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/fig7_raw_diffuser_K1.svg',
        'Report_20260903_AF_UNet fig 7; goal reached without projection at K=1, '
        'top-right-hard, seed 6, 20 trials. Ours; verified absent from aux_repo.'),
    'fig_render_avoiding': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_render_avoiding.png',
        'D3IL tracked simulator montage, frame 0, top-left 320x180 tile; extracted by '
        'prep/extract_env_frames.py.'),
    'fig_render_aligning': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_render_aligning.png',
        'D3IL tracked simulator montage, frame 0, alignment tile; extracted by '
        'prep/extract_env_frames.py.'),
    'fig_aligning_camera_overhead': ('demo',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_aligning_camera_overhead.png',
        'D3IL expert alignment demonstration, rollout 0 frame 60, left 96x96 tile: the fixed '
        'bp-cam observation consumed by the visual policy; extracted by prep/extract_env_frames.py.'),
    'fig_aligning_camera_wrist': ('demo',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_aligning_camera_wrist.png',
        'Same expert demonstration and instant, right 96x96 tile: the wrist-mounted inhand-cam '
        'observation consumed by the visual policy; extracted by prep/extract_env_frames.py.'),
    'fig_render_uav_corridor': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_render_uav_corridor.png',
        'MuJoCo render of scene_corridor_v2.xml with the Skydio X2 mesh, vehicle placed at the start of '
        'the generator reference path (uav_expert_data_collect/trajectories.py), path drawn in. '
        'prep/render_mujoco_scenes.py, declared in MUJOCO_RENDERS. Ours.'),
    'fig_render_uav_pillars': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_render_uav_pillars.png',
        'MuJoCo render of scene_pillars.xml with the Skydio X2 mesh, vehicle placed at the start of '
        'the generator reference path (uav_expert_data_collect/trajectories.py), path drawn in. '
        'prep/render_mujoco_scenes.py, declared in MUJOCO_RENDERS. Ours.'),
    'fig_render_uav_scurve': ('env',
        'Data_Analysis/DA_in_Paper/data/prepared/fig_render_uav_scurve.png',
        'MuJoCo render of scene_s_curve.xml with the Skydio X2 mesh, vehicle placed at the start of '
        'the generator reference path (uav_expert_data_collect/trajectories.py), path drawn in. '
        'prep/render_mujoco_scenes.py, declared in MUJOCO_RENDERS. Ours.'),
}


def vendored_path(key):
    return os.path.join(REPO, VENDORED[key][1])


# ═══════════════════════════════════════════════════════════════════════════
#  VENDORED FIGURES THAT MUST BE CUT BEFORE USE
# ═══════════════════════════════════════════════════════════════════════════
# A diagnostic dashboard is not a thesis figure. fig6a/b/c of Report_20260819 are
# 2x6 grids -- four per-axis time series, the executed path, and the overlaid
# plans -- at 3000x1000. Dropped whole into a 0.49\linewidth minipage, the panel
# the thesis actually argues from is about 12 mm wide and unreadable; that is what
# reached the first build of fig:raw-plans.
#
# The cut is declared here, not done by hand, so that it is reproducible and so
# that re-copying the source cannot quietly restore the dashboard.
#   name -> ((left, top, right, bottom) in source pixels, what the box keeps
#            [, (width, height) to resize the cut to])
# Boxes were read off the white gutters between panels, not guessed: the plans
# panel spans x 2349-2702 with its y tick labels at 2326-2348, and the top row
# spans y 98-476 including its x tick labels.
#
# THE TWO DASHBOARD LAYOUTS. The 08-19 sources are 3000x1000 -- two episodes, the
# `n_trials: 2` default. The four added on 09-19 come from the 20-trials campaign
# and are 3000x5000: ten episodes, same six columns at the same x, same 500 px row
# pitch, but a title band on top that shifts row 1 down and squeezes its axes.
# Measured rather than assumed -- the plot frame sits at
#   08-19 sources : x 2368-2700, y 620-970   (332 x 350 px)
#   09-19 sources : x 2368-2700, y 600-926   (332 x 326 px)
# Same width, 24 px shorter, over the identical data range (x 0.2-0.8, y -0.3-0.4).
# Dropped into the 2x4 matrix untreated, the four new panels would sit 7% flatter
# than the three old ones and their obstacles would read as ellipses beside circles.
# So the new entries carry a resize: the box is chosen so that scaling the cut to
# 403x400 lands the frame on exactly (56, 28)-(388, 378), where the 08-19 panels
# already have it. All seven panels then share one canvas and one data aspect.
#
# Applied by prep/crop_vendored.py (needs PIL -> python3.14); make_figs.py copies
# the prepared file and REFUSES to fall back to the uncut source.
VENDORED_CROP = {
    'fig_raw_plans_meanflow_K1': ((2312, 92, 2715, 492),
        'top row, last column: every plan of the episode overlaid on the scene'),
    'fig_raw_plans_diffusion_K1': ((2312, 92, 2715, 492),
        'top row, last column: every plan of the episode overlaid on the scene'),
    'fig_raw_plans_meanflow_K2': ((2312, 92, 2715, 492),
        'top row, last column: every plan of the episode overlaid on the scene'),
    # 2026-09-19 run, n_trials 2 -> the same 3000x1000 layout as the three 08-19
    # panels above, so it takes their box and needs NO resize. Checked against
    # fig6b (diffusion K1) before vendoring: identical axes and frame position.
    'fig_raw_plans_diffusion_K2': ((2312, 92, 2715, 492),
        'top row, last column: every plan of the episode overlaid on the scene'),
    # 2026-09-20, thres0.5 campaign: 3000x1000 again (two episodes), measured grid
    # bands cols [(370,709)...(2349,2702)] rows [(98,476),(518,896)] -- the same
    # layout as the four entries above, so the same box and no resize.
    'fig_raw_plans_diffusion_K20': ((2312, 92, 2715, 492),
        'top row, last column: every plan of the episode overlaid on the scene'),
    # 20-trials layout: episode 1 of ten, then stretched onto the 08-19 canvas.
    'fig_raw_plans_fm_K1': ((2312, 574, 2715, 947),
        'episode 1, last column: every plan of the episode overlaid on the scene',
        (403, 400)),
    'fig_raw_plans_fm_K2': ((2312, 574, 2715, 947),
        'episode 1, last column: every plan of the episode overlaid on the scene',
        (403, 400)),
    'fig_raw_plans_af_K1': ((2312, 574, 2715, 947),
        'episode 1, last column: every plan of the episode overlaid on the scene',
        (403, 400)),
    'fig_raw_plans_af_K2': ((2312, 574, 2715, 947),
        'episode 1, last column: every plan of the episode overlaid on the scene',
        (403, 400)),
}

# Where prep/crop_vendored.py writes. Committed, like data/avoiding_scene.json, so
# a fresh clone can build the store without PIL.
PREPARED = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'prepared')


def prepared_path(key):
    """The cut file make_figs.py must copy, or None when the figure needs no cut."""
    if key not in VENDORED_CROP:
        return None
    ext = os.path.splitext(VENDORED[key][1])[1]
    return os.path.join(PREPARED, key + ext)


# ═══════════════════════════════════════════════════════════════════════════
#  PLANNED FIGURES -- specified in a draft, not yet made
# ═══════════════════════════════════════════════════════════════════════════
# Each mirrors a \todofigure box in a draft. When one is made, put the file into
# ../figures/<group>/<name>.<ext>, switch the draft from \todofigure to
# \includegraphics{<name>}, remove the entry here, and export.
# Entry: name -> (group, where the draft asks for it, what it must show / how).
PLANNED = {
    # EMPTY as of 2026-09-19 -- every figure the drafts ask for now exists in the store.
    # The last entry to leave was fig_raw_plans_diffusion_K2, the eighth panel of
    # fig:raw-plans; it is in VENDORED above. It was the only one of the eight that
    # needed a RUN: for the diffusion engine K is fixed at training
    # (config/avoiding-d3il.py:1074 puts K in diffusion_loadpath), so a K=2 checkpoint
    # had to be trained -- jobs 25965 + 25966, 2026-09-19.
    #
    # NOTE for whoever reads the batch CSVs next: a folder named
    # '...Dmodels.diffusion.GaussianDiffusion...' under a flow_matching_v3_* prefix is a
    # FLOW model under its pre-2026-05-26 class name (commit cac7cc6a renamed
    # GaussianDiffusion -> FlowMatchingODE). The diffusion baseline is
    # '...Dmodels.GaussianDiffusion...' under plans/diffusion/. Reading the first as the
    # second is what made the 09-19 audit believe this panel was an evaluation away;
    # job 25964 died proving otherwise.
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
    # Dark slate family used by the constraint figures. Model identity is also
    # carried by labels/markers, so the plots remain legible in grayscale.
    'diffusion': '#17202a',
    'fm':        '#34495e',
    'mf':        '#5d6d7e',
    'af':        '#283747',
}
ENGINE_LABEL = {
    'diffusion': 'Diffusion',
    'fm':        'FM',
    'mf':        'MeanFM',
    'af':        'CI-MeanFM',
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
# a smaller protocol (5 seeds x 2 trials = 10 episodes), not ours. They are
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
# The consistency-training corpus (temp/0309) is a DIFFERENT batch with its own folder
# names: the run tag carries the alpha floor and the seed, and the candidate fan token
# B4 is hf_batch_size, which no arm here uses (the MPC fan was 4 for every run).
# Its `af` rows are the architecture-matched U-Net -- unlike AVOIDING_T2_FOLDERS['af'],
# which is the SiT backbone and confounded. Do not mix the two.
AVOIDING_AF_FOLDERS = {
    'af02': 'H8_K%d_Meuler_T0.5_A0.5_B4_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msgafon02_s6',
    'af05': 'H8_K%d_Meuler_T0.5_A0.5_B4_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msgafon005_s6',
    'mf':   'H8_K%d_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials',
    'fm':   'H8_K%d_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials',
    'diffusion': 'H8_K%d_T0.5_Dmodels.GaussianDiffusion_msg20trials',
}
AVOIDING_AF_LABEL = {
    'af02': 'CI-MeanFM\nα_end = 0.2',
    'af05': 'CI-MeanFM\nα_end = 0.05',
    'mf': 'MeanFM',
    'fm': 'FM',
    'diffusion': 'Diffusion',
}
# v3.37: the two D3IL-avoiding charts that put every model in ONE panel need hues, not
# shades. ENGINE_COLOUR is a slate family -- correct for the constraint figures, where
# colour separates obstacle from vehicle, but in `fig_avoiding_tradeoff` three models sit
# in one scatter and three greys of the same hue cannot be told apart in print. These are
# used by those two figures only; everywhere else ENGINE_COLOUR still rules.
ENGINE_COLOUR_DISTINCT = {
    'diffusion': '#17202a',   # the baseline stays near-black, as in every other figure
    'mf':        '#1f6fb2',   # blue
    'fm':        '#d95f02',   # orange
    'af':        '#0f8b8d',   # teal
}
AVOIDING_AF_COLOUR = {'af02': ENGINE_COLOUR_DISTINCT['af'], 'af05': '#7fc3c4',
                      'mf': ENGINE_COLOUR_DISTINCT['mf'], 'fm': ENGINE_COLOUR_DISTINCT['fm'],
                      'diffusion': ENGINE_COLOUR_DISTINCT['diffusion']}

# Without projection the plan does not depend on the constraint set, so the
# unprojected rows of top-left-hard and both-hard are identical to the decimal and
# every model reaches the goal in 20 of 20 on both. Only top-right-hard separates
# the models, which is why figures of the unprojected output use it alone rather
# than showing three panels of which two are duplicates.
AVOIDING_RAW_GEOMETRY = 'top-right-hard'

# The one seed every model in that batch shares. The consistency-training runs are
# tagged _s6 and exist for seed 6 only; MeanFlow, flow matching and diffusion carry
# all five in the same batch. Any comparison across those models must be pinned
# here, or it compares a one-seed number with a five-seed one.
AVOIDING_AF_SEEDS = ['6']

AVOIDING_T1_DIFFUSION_FOLDERS = {
    1:  'H8_K1_T0.5_Dmodels.GaussianDiffusion',
    10: 'H8_K10_Dmodels.GaussianDiffusion_aw10_thres0.5',
    20: 'H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5',
}
AVOIDING_T1_PROTOCOL = '5 seeds x 2 trials = 10 episodes per cell'
# NOT the published protocol, despite what this constant used to say. DPCC reports
# "five training seeds and ten test seeds for each constraint set formulation"
# (Roemer et al. 2025, Sec. 6.1) = 50 rollouts per geometry. 5 x 2 = 10 is SMALLER
# than theirs, and 5 x 20 = 100 is twice it. Figures must not call either one
# "the published protocol".

# ═══════════════════════════════════════════════════════════════════════════
#  D3IL-ALIGNING and the QUADROTOR: cell selection for the corpora of record
# ═══════════════════════════════════════════════════════════════════════════
# One thesis cell is one FolderName PREFIX. The prefix carries the step budget and
# the activation threshold (`_T0.2`), and both matter: the same model at the same
# budget was also run at T0.5, and pooling the two would average two different
# projections. These are the prefixes analysis/va_results.py selects on, copied
# here so the figures and the tables cannot disagree.
# (engine, K) -> (FolderName prefix, required run-tag suffix or None). The SUFFIX is
# not optional: the K=2 MeanFlow prefix also matches a FiLM v2 run, and the consistency
# prefixes also match the alpha floor 0.05 arm. Dropping it pools two models into one
# cell and moves the median (MeanFM K=2: 0.2354 with the suffix, 0.3678 without).
ALIGNING_CELLS = {
    ('mf', 2):    ('H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_fil', 'mv1_Emf'),
    ('mf', 10):   ('H8_K10_Meuler_T0.4_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow', None),
    ('mf', 20):   ('H8_K20_Meuler_T0.2_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow', None),
    ('af', 2):    ('H8_K2_Meuler_T0.5_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow', '_msgafon02_s6'),
    ('af', 20):   ('H8_K20_Meuler_T0.2_Dmix_visual_aligning.models.visual_af_diffusion.VisualAlphaFlow', '_msgafon02_s6'),
    ('fm', 20):   ('H8_K20_Meuler_T0.2_Dmix_visual_aligning.models.visual_fm_diffusion.VisualFlowMatching', None),
    ('diffusion', 20):  ('H8_K20_T0.5_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion', None),
    ('diffusion', 100): ('H8_K100_T0.5_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion', None),
}
# Budgets the thesis REPORTS on this task. The K=100 diffusion checkpoint exists and is
# `visual_aligning_dpcc`'s own configured chain length (config/aligning-d3il-visual.py:454);
# it is kept in the registry above, and out of the figures, because v3.45 fixed the reported
# budget set to the ones the flow models are operated at. Anything that draws this task filters
# through here, so the figure and the tables cannot disagree about which budgets are shown.
ALIGNING_REPORTED = {('mf', 2), ('mf', 10), ('mf', 20),
                     ('af', 2), ('af', 20),
                     ('fm', 20),
                     ('diffusion', 20)}
# The untightened set is the only geometry on which all four models were evaluated,
# and `diffuser` is the unprojected arm -- the model on its own, which is what the
# generative-model comparison of section 6.2.1 is made on.
ALIGNING_UNPROJECTED = ('combined_5', 'diffuser')
ALIGNING_INITIAL_DISTANCE = 0.4530     # metres, mean over the ten contexts

# UAV-corridor v2: the run tag, the geometry prefix, and the projection suffix that
# every corridor cell of the thesis carries. The corridor's projection configuration
# releases the action box and constrains the commanded position, which is why its
# variant names differ from the other two scenes and why the scenes are never pooled.
UAV_CORRIDOR = {
    'tag': 'u17cv2',
    'geo_prefix': 'corridor_cv2s',
    'suffix': '-bounds_free-pdes-tightened',
    'budgets': {'mf': (1, 3, 5), 'af': (1, 3, 5), 'fm': (1, 3, 5), 'diffusion': (20,)},
}
# S&C on the quadrotor is "passed the goal on a collision-free flight". The strict
# within-0.30 m columns are not what the thesis reports; see analysis/uav_results.py.
UAV_SC = 'n_success_relaxed_and_constraints'

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
