#!/usr/bin/env python3.14
"""UAV-pillars, recomputed cell by cell from the batch CSV.

    python3.14 analysis/pillars_grid.py

🔴 WITHHELD SINCE 2026-09-18 — this scene is out of the results. `pillars_hg` enforces the
constraint its own demonstration generator was built to satisfy (the demonstrated routes are
already 8 cm inside the feasible set), so its projected configurations measure how little of an
ALREADY-FEASIBLE plan each method disturbs, not whether a method can repair an infeasible one.
The script still runs -- it is the tooling the enlarged geometry will be read with -- but nothing
it prints may enter the draft until `pillars_xl` / `pillars_xxl` (Gen15 U17) have been evaluated.
    logs_in_develop/Writing/Working_Space/data_status/PENDING_20260918_pillars_geometry_redesign.md
    Data_Analysis/DA_in_Paper/analysis/DA_20260919_wave_1718_corridor_endpoint_and_scurve.md section 4

Why this exists (v3.36): `tab:uav-pillars` used to report one budget, $\\nfe=5$, and one mean
over eleven configurations of two different kinds. That mean put instantaneous-velocity
matching ahead of analytic average-velocity matching, which is not what the per-step
comparison says, and it hid the budgets at which the ordering reverses. This script reads
the grid back from the CSV so the table can carry every evaluated budget and keep the two
projection blocks apart.

Three things it is careful about, each of which has produced a wrong table before:

1. **The model of a row is not in the row.** `candidates_per_variant.csv` carries an engine
   and a K, but which checkpoint that was -- backbone included -- lives in
   `candidate_axes.csv` under the same Candidate id, and only `Full_Path` is conclusive.
   Checked here: `mf` and `af` are `bbunet`; `fm`'s folder `…FlowMatchingODE_9D` carries no
   backbone suffix because the quadrotor flow-matching model predates that convention, and
   is the U-Net. All four models are therefore architecture-matched.
2. **Geometry.** Only `pillars_hg_*` is the constraint set the thesis reports. The earlier
   `pillars_*` geometry has more budgets but is a different constraint set; pooling them
   would invent a budget ladder that was never flown against these constraints.
3. **`geo_free` variants switch the obstacle constraints off inside the projection** and are
   excluded from every mean here.
"""
import collections
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'plotting'))
import sources as S                                            # noqa: E402

BATCH = os.path.join(S.REPO, 'Data_Analysis', 'analysis_results_checkpoint', '15-09',
                     'batch_uav_20260915_100816')
GEO_PREFIX = 'pillars_hg'
PER_STEP = ['diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t',
            'dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened']
ENDPOINT = ['hardflow_sls', 'hardflow_sls-r', 'hardflow_sls-c', 'hardflow_sls-t']
NAME = {'mf': 'MeanFM', 'af': 'CI-MeanFM', 'fm': 'FM', 'diffusion': 'Diffusion'}
ORDER = [('mf', 2), ('mf', 5), ('af', 1), ('af', 2), ('af', 5),
         ('fm', 2), ('fm', 5), ('diffusion', 20)]
SC = 'n_success_relaxed_and_constraints'        # the goal passed on a collision-free flight


def load():
    axes = {r['Candidate']: r for r in csv.DictReader(
        open(os.path.join(BATCH, 'candidate_axes.csv')))}
    cells = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(BATCH, 'candidates_per_variant.csv'))):
        if r['scene'] != 'pillars' or not r['geo'].startswith(GEO_PREFIX) or r['mask'] != 'all':
            continue
        a = axes[r['Candidate']]
        cells[(a['engine'], int(float(a['K'])))][r['variant']] = r
    return axes, cells


def block(cells, variants, title):
    print(f'\n== {title} ==')
    print(f"{'model':11}{'K':>4}  " + ' '.join(f'{v.replace("hardflow_sls", "hf"):>17}'
                                               for v in variants) + '     mean')
    for key in ORDER:
        row = cells.get(key, {})
        vals = [float(row[v][SC]) for v in variants if v in row]
        cs = ' '.join(f'{(f"{float(row[v][SC]):.2f}" if v in row else "---"):>17}'
                      for v in variants)
        mean = f'{statistics.mean(vals):.3f}' if vals else '  ---'
        print(f'{NAME[key[0]]:11}{key[1]:>4}  {cs}    {mean}')


def best(cells):
    print('\n== best projected configuration per model (S&C, tie-break lower ms/step) ==')
    for key in ORDER:
        top = None
        for v, r in cells.get(key, {}).items():
            if v == 'diffuser' or 'geo_free' in v:
                continue
            cand = (float(r[SC]), -float(r['avg_time_ms']), v, r)
            if top is None or cand[:2] > top[:2]:
                top = cand
        if top:
            s, _, v, r = top
            print(f'  {NAME[key[0]]:11} K{key[1]:<3} {v:22} S&C={s:.2f} '
                  f"passed={float(r['success_relaxed']):.2f} "
                  f"viol={float(r['n_violations']):6.1f} steps={float(r['n_steps']):.0f} "
                  f"ms={float(r['avg_time_ms']):.1f} n={int(float(r['n_rollouts']))}")


def best_per_block(cells):
    """Best configuration per (model, budget) WITHIN each block.

    `tab:uav-pillars-best` used to take each model's best over both blocks at once. The
    diffusion baseline has no endpoint row on this scene, so that table compared three
    models on eleven configurations against a baseline that had five -- and then marked the
    winner. Reported per block, every column is a set the model was actually evaluated in.
    """
    print('\n== best configuration per model and budget, WITHIN each block ==')
    print(f"{'model':11}{'K':>4}   {'per-step best':26} {'S&C':>5} {'steps':>6} {'ms':>8}   "
          f"{'endpoint best':22} {'S&C':>5} {'steps':>6} {'ms':>8}")
    for key in ORDER:
        row = cells.get(key, {})
        out = [f'{NAME[key[0]]:11}{key[1]:>4}   ']
        for block, width in ((PER_STEP, 26), (ENDPOINT, 22)):
            got = [(float(row[v][SC]), -float(row[v]['avg_time_ms']), v, row[v])
                   for v in block if v in row and v != 'diffuser']
            if not got:
                out.append(f'{"--- not evaluated":{width}} {"---":>5} {"---":>6} {"---":>8}   ')
                continue
            sc, _t, v, r = max(got, key=lambda q: q[:2])
            out.append(f'{v:{width}} {sc:5.2f} {float(r["n_steps"]):6.0f} '
                       f'{float(r["avg_time_ms"]):8.1f}   ')
        print(''.join(out))


def head_to_head(cells):
    """MeanFM against FM at nfe=5, cell by cell over the seven shared configurations."""
    a, b = cells[('mf', 5)], cells[('fm', 5)]
    win_a = win_b = tie = 0
    print('\n== MeanFM vs FM at nfe=5, per-step block ==')
    for v in PER_STEP:
        x, y = float(a[v][SC]), float(b[v][SC])
        who = 'MeanFM' if x > y else ('FM' if y > x else 'level')
        win_a += x > y
        win_b += y > x
        tie += x == y
        print(f'  {v:20} {x:.2f}  {y:.2f}   {who}')
    print(f'  -> MeanFM ahead in {win_a}, FM ahead in {win_b}, level in {tie}')


def main():
    print("\n*** WITHHELD: pillars_hg is out of the results (see the module docstring). ***\n")
    _axes, cells = load()
    block(cells, PER_STEP, 'per-step projection (every evaluated budget)')
    block(cells, ENDPOINT, 'endpoint projection (nfe=5 only)')
    head_to_head(cells)
    best_per_block(cells)
    best(cells)
    print('\nmissing on this geometry: MeanFM and FM at nfe=1; the diffusion baseline has no '
          'endpoint row and no random-selection row.')


if __name__ == '__main__':
    main()
