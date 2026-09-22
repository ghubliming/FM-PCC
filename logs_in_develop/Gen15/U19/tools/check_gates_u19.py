#!/usr/bin/env python
"""Gen15 U19 — gates G1 / G2 / G3 of the corridor_v3_hump pilot, read from the eval npz files.

    python logs_in_develop/Gen15/U19/tools/check_gates_u19.py "<...>/Emf_K3_*_u19smokecv3/6/corridor_cv3h*"  [more globs]

Each matched folder is one geo folder: `<geo_tag>/<variant>.npz` (eval_artifacts.save_npz). Needs only numpy.
Plan: logs_in_develop/Gen15/U19/PLAN_20260922_U19_corridor_v3_z_slide.md §3.

  G1  the unprojected arm (`diffuser`) has violating steps on EVERY trial      -> the hump binds
  G2  some projected arm: collision_free >= 2/3 of trials AND success >= 2/3   (strict AND relaxed printed)
  G3  paired executed z, projected - unprojected, same trial index (same route / seed), apex over
      x in [-0.5, 0.5]: > 0.15 m on EVERY projected flight                      -> the plan CLIMBS
      (U12 lesson: a violation count that drops because the flight is shorter is not a pass —
       the per-step rate is printed next to the count for that reason)

obs layout per FM step: [p_des(0:3) | p(3:6) | v(6:9)]  -> executed x = col 3, z = col 5.
"""
import glob
import os
import sys

import numpy as np

X_WIN = (-0.5, 0.5)
G3_MIN_DZ = 0.15
UNPROJECTED = 'diffuser'


def _load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _apex_z(obs, win=X_WIN):
    """max executed z over the x window (nan if the flight never entered it)."""
    o = np.asarray(obs, dtype=float)
    if o.ndim != 2 or o.shape[0] == 0:
        return float('nan')
    x, z = o[:, 3], o[:, 5]
    m = (x >= win[0]) & (x <= win[1])
    return float(z[m].max()) if m.any() else float('nan')


def _frac(a):
    a = np.asarray(a, dtype=float)
    return f'{int(np.nansum(a))}/{len(a)}'


def check_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.npz')))
    if not files:
        print(f'[skip] no npz in {folder}')
        return
    arms = {os.path.splitext(os.path.basename(f))[0]: _load(f) for f in files}
    print(f'\n=== {folder}')
    print(f'    arms: {list(arms)}')
    ref = arms.get(UNPROJECTED)
    if ref is None:
        print(f'    [!] no {UNPROJECTED}.npz here — G1 and G3 need the unprojected arm in the SAME folder')
    # ── per-arm table ──
    print(f'    {"arm":48s} {"n":>3s} {"coll_free":>9s} {"succ_S":>7s} {"succ_R":>7s} {"S&C":>5s} {"viol>0":>7s} {"viol/step":>9s} {"steps":>6s}')
    for name, d in arms.items():
        n = len(d['n_steps'])
        vs = d['constraint_n_violations'].astype(float)
        st = d['n_steps'].astype(float)
        print(f'    {name:48s} {n:3d} {_frac(d["constraint_collision_free"]):>9s} '
              f'{_frac(d["success_strict"]):>7s} {_frac(d["success_relaxed"]):>7s} '
              f'{_frac(d["success_strict_and_constraints"]):>5s} {_frac(vs > 0):>7s} '
              f'{np.nanmean(vs / np.maximum(st, 1)):9.4f} {np.nanmean(st):6.1f}')
    # ── G1 ──
    if ref is not None:
        v = ref['constraint_n_violations'].astype(float)
        g1 = bool(np.all(v > 0))
        print(f'    G1 hump binds on the unprojected arm: violating trials {_frac(v > 0)}  -> {"PASS" if g1 else "FAIL (raise H / check plane xz reached the scorer)"}')
    # ── G2 / G3 per projected arm ──
    for name, d in arms.items():
        if name == UNPROJECTED:
            continue
        n = len(d['n_steps'])
        cf = d['constraint_collision_free'].astype(float)
        ss = d['success_strict'].astype(float)
        sr = d['success_relaxed'].astype(float)
        g2s = (cf.sum() >= 2 * n / 3) and (ss.sum() >= 2 * n / 3)
        g2r = (cf.sum() >= 2 * n / 3) and (sr.sum() >= 2 * n / 3)
        print(f'    G2 {name}: collision_free {_frac(cf)}, success strict {_frac(ss)} / relaxed {_frac(sr)} '
              f'-> {"PASS" if g2s else ("PASS (relaxed only)" if g2r else "FAIL")}')
        if ref is None:
            continue
        m = min(n, len(ref['n_steps']))
        rows = []
        for i in range(m):
            zp = _apex_z(d['obs_all'][i]); zu = _apex_z(ref['obs_all'][i])
            rows.append((i, zu, zp, zp - zu))
        ok = [r[3] > G3_MIN_DZ for r in rows if np.isfinite(r[3])]
        g3 = bool(ok) and all(ok) and len(ok) == m
        print(f'    G3 {name}: apex z over x∈[{X_WIN[0]},{X_WIN[1]}], projected − unprojected, per trial:')
        for i, zu, zp, dz in rows:
            flag = '  ✓' if (np.isfinite(dz) and dz > G3_MIN_DZ) else '  ✗'
            print(f'         trial {i}: unprojected {zu:.3f}  projected {zp:.3f}  Δz {dz:+.3f} m{flag}')
        print(f'       -> {"PASS" if g3 else "FAIL (no climb on demand — the U12 finding again)"}  (needs Δz > {G3_MIN_DZ} m on every flight)')


def main(argv):
    if not argv:
        print(__doc__); sys.exit(1)
    folders = []
    for pat in argv:
        folders += sorted(glob.glob(pat))
    if not folders:
        print('[!] nothing matched', argv); sys.exit(1)
    for f in folders:
        check_folder(f)


if __name__ == '__main__':
    main(sys.argv[1:])
