#!/usr/bin/env python3.14
"""Extract the FLOWN quadrotor paths into a plain JSON file for the figure builders.

    python3.14 plotting/extract/uav_paths.py [--root <folder holding logs/>]

Needs numpy (python3.14 in the AI container has it; plain python3 does not), which is
why this is a separate step: the SVG builders run on the standard library and read only
``../../data/uav_paths.json``.

Why this exists
---------------
The committed 15-09 batches carry per-rollout SCALARS (`per_rollout_detail.csv`); the
executed positions live only in the rollout artefacts on the cluster. Those were staged
and downloaded on 2026-09-18 by `Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh`
(groups F1-F3) -- see `Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md`.
The drop is NOT committed (it lives under the gitignored `temp/`), so this extract is what
persists: the builders never read the drop again once `data/uav_paths.json` exists.

Sources, read at run time, not typed in:
  <run>/<variant>.npz     obs_all -- 6-D per control step, [des_pos(0:3) | pos(3:6)].
                          The FLOWN position is dims 3:5 (x, y); dim 5 is z. Until v3.65
                          z was read for the altitude check only; since v3.65 it travels
                          with every path as `z`, for the corridor altitude figure (F4).
                          One entry per episode, ragged.
  <run>/results.json      summary + rollouts[]: per-episode outcome, index-aligned with
                          obs_all, and `homotopy` / `homotopy_flown` (L/C/R) on corridor.

What a path is drawn as, decided here rather than in the builder:
  passed      success_relaxed       -- crossed the finish line (the chapter's "success")
  clean       constraint.collision_free -- never entered an inflated obstacle
  aborted     divergence_aborted    -- the divergence guard ended the flight early
  safe        phys_safe             -- no contact and no loss of altitude
The first two are independent: a flight can reach the goal THROUGH a pillar, which is the
whole point of the projection comparison, so both flags travel with every path. The last
two (v3.63) let a builder mark where a flight that crashed or lost control ended.
"""
import argparse
import datetime
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
OUT = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'uav_paths.json')
DEFAULT_ROOT = os.path.join(REPO, 'temp', '18-09-2026')

# Keep the SVG small: a corridor flight is ~270 control steps and there are ten panels of
# twelve. Every STRIDE-th step plus both endpoints keeps the shape at a fraction of the size.
STRIDE = 2
NDP = 3                     # metres, rounded -- 1 mm, far below anything the figure resolves

MF = 'H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet'
FM = 'H8_Dmodels.diffusion.FlowMatchingODE_9D'
AF = 'H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet'
DF = 'H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20'

# figure -> panels, in the order they are drawn. Each panel names one evaluated cell:
#   (panel title, panel subtitle, scene, model folder, run tag, variant)
# The variant IS the npz basename; see the fetch script's header for why that is the key.
PANELS = {
    # F1 -- fig:uav-scurve-paths, the companion of tab:uav-controller. Unprojected on
    # purpose: the figure is about the CONTROLLER, so the plan must not be filtered.
    'scurve': ('UAV-s-curve', [
        ('MuJoCo MPC', 'unprojected plan',
         'uav-s_curve', 'mix_uav_mf', MF, 'Emf_K10_mpc4_mjpc_T0.5_u7hg', 'diffuser'),
        ('Cascaded geometric controller', 'unprojected plan',
         'uav-s_curve', 'mix_uav_mf', MF, 'Emf_K10_mpc4_pid_stopgo_T0.5_u7hg', 'diffuser'),
    ]),
    # F2 -- one panel per model, each under the projection that tab:uav-pillars-best
    # reports as its best configuration. The variants DIFFER by row on purpose.
    'pillars': ('UAV-pillars', [
        ('Average-velocity matching', 'endpoint projection, random',
         'uav-pillars', 'mix_uav_mf', MF, 'Emf_K5_mpc4_pid_stopgo_T0.5_u7hg', 'hardflow_sls-r'),
        ('Instantaneous-velocity matching', 'endpoint projection',
         'uav-pillars', 'mix_uav_fm', FM, 'Efm_K5_mpc4_pid_stopgo_T0.5_u7hg', 'hardflow_sls'),
        ('Consistency-interpolated', 'per-step projection, tightened',
         'uav-pillars', 'mix_uav_af', AF, 'Eaf_K5_mpc4_pid_stopgo_T0.5_EPlatest_u7hg',
         'dpcc-t-tightened'),
        ('Diffusion baseline', 'per-step projection, tightened',
         'uav-pillars', 'mix_uav_diffusion', DF, 'Ediffusion_K20_mpc4_pid_stopgo_T0.5_u7hg',
         'dpcc-t-tightened'),
    ]),
    # F3 -- the slide. Model-major so each row is one model across its budget ladder;
    # the baseline is the last row on its own, at the only budget it has.
    'corridor': ('UAV-corridor', [
        (f'{lab}', f'{k} function evaluation{"s" if k != 1 else ""}',
         'uav-corridor', folder, model, f'{pre}_K{k}_mpc4_pid_stopgo_T0.5{suf}_u17cv2',
         'dpcc-t-bounds_free-pdes-tightened')
        for lab, folder, model, pre, suf in (
            ('Average-velocity matching', 'mix_uav_mf', MF, 'Emf', ''),
            ('Instantaneous-velocity matching', 'mix_uav_fm', FM, 'Efm', ''),
            ('Consistency-interpolated', 'mix_uav_af', AF, 'Eaf', '_EPlatest'),
        ) for k in (1, 3, 5)
    ] + [
        ('Diffusion baseline', '20 function evaluations',
         'uav-corridor', 'mix_uav_diffusion', DF, 'Ediffusion_K20_mpc4_pid_stopgo_T0.5_u17cv2',
         'dpcc-t-bounds_free-pdes-tightened'),
    ]),
    # F4 -- fig:uav-corridor-altitude (v3.65). The SAME flights as F3 seen from the side:
    # for each model, the unprojected arm (`diffuser`) and the projected arm of tab:uav-corridor
    # at one budget, so the builder can pair them by title. Every corridor constraint acts in
    # x-y; this figure exists to show what the altitude does while the slide is being cleared.
    # The author asked (2026-09-22) whether the corridor also slides in z: it does not, and the
    # figure is the evidence. Flow models at K = 3 (the budget the chapter reads), diffusion at 20.
    'corridor_altitude': ('UAV-corridor', [
        (lab, sub, 'uav-corridor', folder, model, f'{pre}_K{k}_mpc4_pid_stopgo_T0.5{suf}_u17cv2', var)
        for lab, folder, model, pre, suf, k in (
            ('Average-velocity matching', 'mix_uav_mf', MF, 'Emf', '', 3),
            ('Instantaneous-velocity matching', 'mix_uav_fm', FM, 'Efm', '', 3),
            ('Consistency-interpolated', 'mix_uav_af', AF, 'Eaf', '_EPlatest', 3),
            ('Diffusion baseline', 'mix_uav_diffusion', DF, 'Ediffusion', '', 20),
        ) for sub, var in ((f'{k} function evaluations, unprojected', 'diffuser'),
                           (f'{k} function evaluations, per-step projection', 'dpcc-t-bounds_free-pdes-tightened'))
    ]),
}


def find_cell(root, scene, folder, model, tag, variant):
    """-> (npz path, results.json path). The geometry folder is not named here: a run
    holds exactly one, and naming it would duplicate a string the eval owns."""
    base = os.path.join(root, 'logs', 'UAV_MIX', scene, 'plans', folder, model, tag)
    if not os.path.isdir(base):
        return None, None
    for dirpath, _dirnames, filenames in os.walk(base):
        if f'{variant}.npz' in filenames:
            res = os.path.join(dirpath, 'results.json')
            return (os.path.join(dirpath, f'{variant}.npz'),
                    res if os.path.isfile(res) else None)
    return None, None


def episodes(npz_path, res_path):
    d = np.load(npz_path, allow_pickle=True)
    obs = d['obs_all']
    passed = d['success_relaxed']          # crossing the finish line -- the reported goal criterion
    clean = d['constraint_collision_free']
    rollouts = []
    if res_path:
        rollouts = json.load(open(res_path)).get('rollouts', [])

    out = []
    for i, ep in enumerate(obs):
        a = np.asarray(ep, dtype=float)
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        xy = a[:, 3:5]                                  # flown position; 0:3 is the reference
        keep = list(range(0, len(xy), STRIDE))
        if keep[-1] != len(xy) - 1:
            keep.append(len(xy) - 1)                    # never lose where the flight ended
        r = rollouts[i] if i < len(rollouts) else {}
        out.append({
            'xy': [[round(float(x), NDP), round(float(y), NDP)] for x, y in xy[keep]],
            # v3.65: the flown altitude at the same kept steps (obs_all dim 5)
            'z': [round(float(v), NDP) for v in a[keep, 5]],
            'passed': bool(passed[i]),
            'clean': bool(clean[i]),
            'homotopy': r.get('homotopy'),
            'homotopy_flown': r.get('homotopy_flown'),
            'min_z': round(float(d['phys_min_z'][i]), NDP),
            # v3.63: the two per-flight failure flags the s-curve figure marks with an X
            'aborted': bool(d['divergence_aborted'][i]) if 'divergence_aborted' in d.files else None,
            'safe': bool(d['phys_safe'][i]) if 'phys_safe' in d.files else None,
            'reason': (str(d['divergence_reason'][i]) if 'divergence_reason' in d.files
                       and bool(d['divergence_aborted'][i]) else None),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT,
                    help='folder holding the staged logs/ tree (default: the 18-09 drop)')
    args = ap.parse_args()

    figures, missing = {}, []
    for fig, (scene_label, panels) in PANELS.items():
        drawn = []
        for title, sub, scene, folder, model, tag, variant in panels:
            npz, res = find_cell(args.root, scene, folder, model, tag, variant)
            if not npz:
                missing.append(f'{fig}/{title} [{tag} :: {variant}]')
                continue
            eps = episodes(npz, res)
            drawn.append({
                'title': title, 'sub': sub, 'variant': variant, 'tag': tag,
                'source': os.path.relpath(npz, args.root),
                'episodes': eps,
                'n': len(eps),
                'passed': sum(e['passed'] for e in eps),
                'clean': sum(e['clean'] for e in eps),
            })
            print(f'  {fig:9s} {title:34s} {len(eps):3d} flights  '
                  f'{sum(e["passed"] for e in eps)} passed, {sum(e["clean"] for e in eps)} clean')
        figures[fig] = {'scene': scene_label, 'panels': drawn}

    payload = {
        'generated': datetime.date.today().isoformat(),
        'stride': STRIDE,
        'source_root': os.path.relpath(args.root, REPO),
        'note': ('Flown positions from obs_all[:, 3:5] (and altitude from dim 5, v3.65) of each run npz, staged from the cluster '
                 'on 2026-09-18 (fetch_20260918_v3_figure_artefacts.sh, groups F1-F3). The drop '
                 'itself is under the gitignored temp/; this extract is the committed record.'),
        'figures': figures,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    print(f'\nwrote {os.path.relpath(OUT, REPO)}  '
          f'({os.path.getsize(OUT) / 1024:.0f} KiB)')
    if missing:
        print('\nNOT FOUND in the drop (panel skipped):')
        for m in missing:
            print(f'  - {m}')


if __name__ == '__main__':
    main()
