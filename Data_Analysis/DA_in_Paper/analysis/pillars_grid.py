#!/usr/bin/env python3.14
"""UAV-pillars, recomputed cell by cell from the batch CSV.

    python3.14 analysis/pillars_grid.py

🔴 2026-09-22 — FALLBACK TO `pillars_hg`. Gen15 U17 (`pillars_xl`, the enlarged keep-out) was
ABANDONED the same day: its full wave returned S&C = 0.00 and collision_free = 0.00 in all 94 cells,
unprojected included — every projector routes into the forbidden centre corridor between the pillar
rows (DA_20260922_pillars_xl_wave.md; Gen15/U17/CLOSURE_20260922_U17_abandoned.md). By author
decision the thesis reports `pillars_hg` again, WITH its caveat: on this geometry the unprojected flows
are already feasible, so the projected rows measure how much of a feasible plan each method preserves,
not whether it can repair an infeasible one. The section prose says so first.

    To reproduce the abandoned post-mortem: BATCH -> 22-09-UAV-Pillars/batch_uav_20260922_113112,
    GEO_PREFIX -> 'pillars_xl'. The `lanes()` block below only runs for that geometry.

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
                     'batch_uav_20260915_100816')          # the withheld tables were computed here
GEO_PREFIX = 'pillars_hg'
# ABANDONED alternative (Gen15 U17 post-mortem only):
#   BATCH = .../'22-09-UAV-Pillars', 'batch_uav_20260922_113112';  GEO_PREFIX = 'pillars_xl'
# The verify cell (tag u7xlchk, mf K5 diffuser only) is the pre-flight check, not the campaign; the
# diffusion `dpcc-r-tightened` cell is the 2/10 remnant of the walled job 25951. Both are excluded.
EXCLUDE_TAGS = ('u7xlchk',)
N_FLIGHTS = 10
PER_STEP = ['diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t',
            'dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened']
ENDPOINT = ['hardflow_sls', 'hardflow_sls-r', 'hardflow_sls-c', 'hardflow_sls-t']
NAME = {'mf': 'MeanFM', 'af': 'CI-MeanFM', 'fm': 'FM', 'diffusion': 'Diffusion'}
ORDER = [('mf', 1), ('mf', 2), ('mf', 5), ('af', 1), ('af', 2), ('af', 5),
         ('fm', 1), ('fm', 2), ('fm', 5), ('diffusion', 20)]   # rows absent on a geometry print ---
SC = 'n_success_relaxed_and_constraints'        # the goal passed on a collision-free flight


def load():
    axes = {r['Candidate']: r for r in csv.DictReader(
        open(os.path.join(BATCH, 'candidate_axes.csv')))}
    cells = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(BATCH, 'candidates_per_variant.csv'))):
        if r['scene'] != 'pillars' or not r['geo'].startswith(GEO_PREFIX) or r['mask'] != 'all':
            continue
        a = axes[r['Candidate']]
        if any(t in (a['run_tag'] or '') for t in EXCLUDE_TAGS):
            continue
        if int(float(r['n_rollouts'])) != N_FLIGHTS:          # the walled remnant
            continue
        cells[(a['engine'], int(float(a['K'])))][r['variant']] = r
    return axes, cells


def block(cells, variants, title, metric=SC, fmt='{:.2f}'):
    print(f'\n== {title} ==')
    print(f"{'model':11}{'K':>4}  " + ' '.join(f'{v.replace("hardflow_sls", "hf"):>17}'
                                               for v in variants) + '     mean')
    for key in ORDER:
        row = cells.get(key, {})
        vals = [float(row[v][metric]) for v in variants if v in row]
        cs = ' '.join(f'{(fmt.format(float(row[v][metric])) if v in row else "---"):>17}'
                      for v in variants)
        mean = fmt.format(statistics.mean(vals)) if vals else '  ---'
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


def lanes(axes):
    """Per-flight lane classification (DA_20260922 section 3). Reads the per-rollout rows so the
    'where did it fly' statement is reproducible: demo-lane = the demonstrated route at |y|=1.11;
    centre-lane = between the two pillar rows (physically open, closed by the 0.66 m keep-out);
    contact = MuJoCo contact. Mean depth = total_violations / n_violations, m per violating step."""
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(BATCH, 'per_rollout_detail.csv'))):
        if r['scene'] != 'pillars' or not r['geo'].startswith(GEO_PREFIX):
            continue
        a = axes[r['Candidate']]
        if any(t in (a['run_tag'] or '') for t in EXCLUDE_TAGS):
            continue
        rows[(a['engine'], int(float(a['K'])), r['variant'])].append(r)

    def cls(r):
        nv = float(r['n_violations'])
        depth = float(r['total_violations']) / nv if nv else 0.0
        gd, safe = float(r['goal_dist']), float(r['phys_safe'])
        if safe == 0:
            return 'contact'
        if gd < 0.40 and depth < 0.12:
            return 'demo'
        if 0.80 <= gd <= 1.15 and depth > 0.28:
            return 'centre'
        return 'other'

    print('\n== lane classification per cell (flights of 10): demo / CENTRE / CONTACT / other | '
          'mean depth [m] | median goal dist [m] ==')
    for key in ORDER:
        for v in PER_STEP + ENDPOINT:
            rs = rows.get((key[0], key[1], v), [])
            if len(rs) != N_FLIGHTS:
                continue
            c = collections.Counter(cls(r) for r in rs)
            nv = sum(float(r['n_violations']) for r in rs)
            depth = sum(float(r['total_violations']) for r in rs) / max(nv, 1)
            gd = sorted(float(r['goal_dist']) for r in rs)[N_FLIGHTS // 2]
            print(f"  {NAME[key[0]]:11}{key[1]:>3} {v:18} {c['demo']:>3} {c['centre']:>5} "
                  f"{c['contact']:>6} {c['other']:>4} | {depth:5.3f} | {gd:5.2f}")


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
    print(f"\n*** geometry {GEO_PREFIX} — batch {os.path.basename(BATCH)} — "
          f"{N_FLIGHTS} flights per cell, seed 6 ***\n")
    _axes, cells = load()
    block(cells, PER_STEP, 'S&C — per-step projection (every evaluated budget)')
    block(cells, ENDPOINT, 'S&C — endpoint projection (nfe=5 only)')
    block(cells, PER_STEP, 'collision-free rate (constraint) — per-step', 'collision_free_completed')
    block(cells, ENDPOINT, 'collision-free rate (constraint) — endpoint', 'collision_free_completed')
    block(cells, PER_STEP, 'violating steps, mean per flight — per-step', 'n_violations', '{:.1f}')
    block(cells, ENDPOINT, 'violating steps, mean per flight — endpoint', 'n_violations', '{:.1f}')
    block(cells, PER_STEP, 'strict success (goal within 0.30 m, contact-free) — per-step',
          'n_success', '{:.2f}')
    block(cells, PER_STEP, 'goal distance at episode end, mean [m] — per-step', 'goal_dist', '{:.2f}')
    block(cells, ENDPOINT, 'goal distance at episode end, mean [m] — endpoint', 'goal_dist', '{:.2f}')
    block(cells, PER_STEP, 'executed steps (634 = episode budget exhausted) — per-step',
          'n_steps', '{:.0f}')
    block(cells, PER_STEP, 'ms per control step — per-step', 'avg_time_ms', '{:.1f}')
    block(cells, ENDPOINT, 'ms per control step — endpoint', 'avg_time_ms', '{:.1f}')
    if GEO_PREFIX.startswith('pillars_x'):        # the centre-corridor diagnosis of U17 only
        lanes(_axes)
    head_to_head(cells)
    best_per_block(cells)
    best(cells)
    if GEO_PREFIX == 'pillars_hg':
        print('\nmissing on this geometry: MeanFM and FM at nfe=1; the diffusion baseline has no '
              'endpoint row and no random-selection row. (pillars_xl — Gen15 U17 — is ABANDONED.)')
    else:
        print('\nnot on this geometry: the diffusion baseline has only its unprojected row and `dpcc-r` '
              '(job 25951 reached the 24 h limit; the remainder was closed by decision, runbook §3d); '
              'HardFlow is degenerate at nfe<=2 and is not evaluated there.')


if __name__ == '__main__':
    main()
