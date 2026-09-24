#!/usr/bin/env python3.14
"""UAV-s-curve: final distance to the goal point, mean and sample standard deviation over each cell's ten flights,
for the Distance [m] columns of tab:uav-scurve, tab:uav-scurve-projection and tab:uav-controller (author, 24-09, v3.93:
"Distance [m] Table 6.13 can you add the var?"). Same corpus as DA_20260924_scurve_R44bc_projection_controller.md;
the means must equal the printed ones.

    python3.14 Data_Analysis/DA_in_Paper/analysis/scurve_goal_dist_sd.py
"""
import glob
import os

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
ROOT = os.path.join(REPO, 'temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747/logs/UAV_MIX/uav-s_curve/plans')
CELLS = [
    ('MeanFM K1', 'mix_uav_mf/*/Emf_K1_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('MeanFM K2', 'mix_uav_mf/*/Emf_K2_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('MeanFM K20', 'mix_uav_mf/*/Emf_K20_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('CI-MeanFM K1', 'mix_uav_af/*/Eaf_K1_mpc4_pid_stopgo_T0.5_EPlatest_p23scgrid', 'diffuser'),
    ('CI-MeanFM K2', 'mix_uav_af/*/Eaf_K2_mpc4_pid_stopgo_T0.5_EPlatest_p23scgrid', 'diffuser'),
    ('CI-MeanFM K20', 'mix_uav_af/*/Eaf_K20_mpc4_pid_stopgo_T0.5_EPlatest_p23scgrid', 'diffuser'),
    ('FM K1', 'mix_uav_fm/*/Efm_K1_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('FM K2', 'mix_uav_fm/*/Efm_K2_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('FM K20', 'mix_uav_fm/*/Efm_K20_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('Diffusion K20', 'mix_uav_diffusion/*/Ediffusion_K20_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('FM K1 per-step r (cascaded)', 'mix_uav_fm/*/Efm_K1_mpc4_pid_stopgo_T0.5_p23scproj', 'dpcc-r-tightened'),
    ('FM K1 per-step c (cascaded)', 'mix_uav_fm/*/Efm_K1_mpc4_pid_stopgo_T0.5_p23scproj', 'dpcc-c-tightened'),
    ('FM K1 per-step t (cascaded)', 'mix_uav_fm/*/Efm_K1_mpc4_pid_stopgo_T0.5_p23scproj', 'dpcc-t-tightened'),
    ('FM K1 unprojected (MuJoCo MPC)', 'mix_uav_fm/*/Efm_K1_mpc4_mjpc_T0.5_p23scmjpc', 'diffuser'),
    ('FM K1 per-step r (MuJoCo MPC)', 'mix_uav_fm/*/Efm_K1_mpc4_mjpc_T0.5_p23scmjpc', 'dpcc-r-tightened'),
]
for name, pat, var in CELLS:
    hits = glob.glob(os.path.join(ROOT, pat, '*', '*', var, f'{var}.npz'))
    assert len(hits) == 1, (name, hits)
    g = np.load(hits[0], allow_pickle=True)['goal_dist'][:10].astype(float)
    print(f'{name:32s} n={len(g):2d}  mean {g.mean():.3f}  sd {g.std(ddof=1):.3f}')
