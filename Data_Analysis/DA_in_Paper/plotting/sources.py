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
        {'name': 'UAV-pillars', 'title': 'UAV-pillars',
         'sub': 'six pillars in two rows',
         'xlim': (-3.0, 3.0), 'ylim': (-1.5, 1.5),
         'halfspaces': [],
         'disks': [{'c': (x, y), 'r': 0.12} for x in (-2.0, 0.0, 2.0) for y in (-0.6, 0.6)]},
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
#   name -> ((left, top, right, bottom) in source pixels, what the box keeps)
# Boxes were read off the white gutters between panels, not guessed: the plans
# panel spans x 2349-2702 with its y tick labels at 2326-2348, and the top row
# spans y 98-476 including its x tick labels.
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
    # The five missing cells of the 2x4 matrix fig:raw-plans (analytic average-velocity K1/K2 and
    # diffusion K1 exist).
    # All need saved plans from a cluster evaluation: plotting/REQUEST_20260916_cluster_fm_plan_panels.md.
    'fig_raw_plans_fm_K1': ('demo', 'v3 sec:res:avoiding:raw (fig:raw-plans), instantaneous-velocity matching K=1',
        'Plan fan without projection, seed 6, both-hard. Cluster run: see the request file.'),
    'fig_raw_plans_fm_K2': ('demo', 'v3 sec:res:avoiding:raw (fig:raw-plans), instantaneous-velocity matching K=2',
        'As above at K=2.'),
    'fig_raw_plans_diffusion_K2': ('demo', 'v3 sec:res:avoiding:raw (fig:raw-plans), diffusion K=2',
        'As above for the diffusion model at K=2; the K=1 panel came from a model trained at K=1.'),
    'fig_raw_plans_af_K1': ('demo', 'v3 sec:res:avoiding:raw (fig:raw-plans), consistency-interpolated average-velocity matching K=1',
        'As above for consistency-interpolated average-velocity matching (U-Net, alpha_end 0.2) at K=1; Report_20260903 section 8 panel 8a.'),
    'fig_raw_plans_af_K2': ('demo', 'v3 sec:res:avoiding:raw (fig:raw-plans), consistency-interpolated average-velocity matching K=2',
        'As above at K=2; report panel 8b.'),
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
AVOIDING_AF_COLOUR = {'af02': ENGINE_COLOUR['af'], 'af05': '#566573',
                      'mf': ENGINE_COLOUR['mf'], 'fm': ENGINE_COLOUR['fm'],
                      'diffusion': ENGINE_COLOUR['diffusion']}

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
