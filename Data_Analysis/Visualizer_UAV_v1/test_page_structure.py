"""Visualizer_UAV_v1 — structural check of the generated page.

    python Data_Analysis/Visualizer_UAV_v1/test_page_structure.py

Stdlib only, so it runs in the AI-coding container. It is the counterpart to
`Visualizer_VA_v2/test_page_offline.py`, which drives the page's handlers
against real CSVs and therefore needs pandas + the cluster.

This one answers the question a GENERATED page raises that a hand-written one
does not: **did every edit actually land, and is the result still valid Python?**
`build_from_va2.py` exits on a missing anchor, so a failed edit cannot be silent
— but an edit whose anchor matched something ALMOST right, or that introduced a
syntax error inside the `<script type="py">` block, would only surface in the
browser console. Both are caught here:

  1. the embedded PyScript block compiles;
  2. every UAV-specific string the layer promises is present;
  3. no visual-aligning identifier that would now be WRONG survives (a stale
     `frozen` mask column reads every rollout as unmasked; a stale
     `va2_aggregated_long.csv` makes the page fetch a file this pipeline never
     writes);
  4. the page still carries the inherited features it is built to inherit, so a
     future rebuild against a changed ancestor cannot quietly drop half of them.

It does NOT check rendering or data handling — for that, run the batch on the
cluster and open the page.
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAGE = HERE / 'index.html'

FAILURES = []


def check(label, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {label}' + (f'  — {detail}' if detail else ''))
    if not ok:
        FAILURES.append(label)


def main():
    if not PAGE.exists():
        print(f'{PAGE} does not exist — run build_from_va2.py first')
        return 2
    html = PAGE.read_text()

    # ── 1. the PyScript block is valid Python ────────────────────────────────
    print('\n[embedded python]')
    match = re.search(r'<script type="py">\n(.*?)\n</script>', html, re.S)
    check('py block found', match is not None)
    if match is None:
        return 1
    source = match.group(1)
    try:
        compile(source, '<page py-script>', 'exec')
        check('py block compiles', True, f'{len(source.splitlines())} lines')
    except SyntaxError as exc:
        check('py block compiles', False, f'line {exc.lineno}: {exc.msg}')

    # The plain-JS block too — the presets and view switching live there.
    js = re.search(r'<script>\n(.*?)\n</script>', html, re.S)
    check('js block found', js is not None)

    # ── 2. UAV identity and wiring ───────────────────────────────────────────
    print('\n[UAV identity]')
    for label, needle in (
            ('title', '<title>FM-PCC — UAV Mix-ML Explorer</title>'),
            ('header', '<h1>FM-PCC UAV EXPLORER</h1>'),
            ('suite tag', 'DA_UAV_v1'),
            ('aggregated CSV', "'agg': 'uav_aggregated_long.csv'"),
            ('units CSV', "'units': 'uav_units_long.csv'"),
            ('rollout CSV', "'roll': 'per_rollout_detail.csv'"),
            ('quality CSV', "'qual': 'data_quality.csv'"),
            ('run picker regex', 'batch_uav_[^/"]+'),
            ('manifest prefix', '"batch_uav_"'),
            ('custom-path placeholder', 'batch_uav_'),
    ):
        check(label, needle in html)

    print('\n[UAV mask semantics]')
    for label, needle in (
            ('mask flag constant', "MASK_FLAG = 'projection_cb_tripped'"),
            ('mask option', 'value="proj_valid"'),
            ('mask slice branch', "mask == 'proj_valid' and MASK_FLAG in out.columns"),
            ('banner class', 'mask-banner proj-valid'),
            ('rollout tint class', 'class="cb-row"'),
            ('quality flag columns', "('n_cb_tripped', 'cb_sentinel', 'timing_missing')"),
            ('timing caveat spelled out', 'npz carries NO timing group'),
    ):
        check(label, needle in html)

    print('\n[UAV axis panel]')
    for label, needle in (
            ('axis filter table', "AXIS_FILTERS = (('scene', 'scene-check')"),
            ('scene list node', 'id="scene-list"'),
            ('engine list node', 'id="engine-list"'),
            ('K list node', 'id="kaxis-list"'),
            ('populate function', 'def populate_axis_filters():'),
            ('populate before derive', 'populate_axis_filters()      # must precede derive_frames'),
            ('axis slice applied in _slice', 'return _axis_slice(out)'),
            ('K normalised to int spelling', 'def _k_str(value):'),
            ('axis summary in the banner', 'def _axis_summary():'),
            ('matched-budget warning', 'Matched budget or nothing'),
    ):
        check(label, needle in html)

    print('\n[UAV metrics]')
    for label, needle in (
            ('rollout default columns', "'steps_to_goal'"),
            ('rollout sort options', "('proj_ms', 'Projection ms / replan')"),
            ('matrix: phys_safe', '("phys_safe", "PHYS_SAFE'),
            ('matrix: proj_ms', '("proj_ms", "PROJ_MS'),
            ('matrix: over_budget_frac', '("over_budget_frac", "OVER_BUDGET'),
            ('reference row = step budget', "REF_METRIC = 'max_episode_length'"),
            ('reference row applies to n_steps', "REF_FOR = 'n_steps'"),
            ('compare default y', "_opts('goal_dist')"),
            ('compare default x', "_opts('avg_time_ms')"),
            ('ablation preset', "('ablations', 'Constraint ablations'"),
            ('ablation regex', 'PRESET_ABLATION_RE'),
    ):
        check(label, needle in html)

    # ── 3. nothing visual-aligning survived that would now be WRONG ──────────
    print('\n[no stale visual-aligning wiring]')
    for label, needle in (
            ('no va2 CSV names', 'va2_aggregated_long.csv'),
            ('no va2 units CSV', 'va2_units_long.csv'),
            ('no batch_va2 picker', 'batch_va2_'),
            ('no frozen mask value', "'unfrozen'"),
            ('no frozen-row tint', 'frozen-row'),
            ('no n_frozen quality column', 'n_frozen'),
            ('no box-target metric in the matrices', 'mean_dist_per_rollout'),
            ('no D1 box-obstacle wording', 'box-obstacle'),
            ('no constraint_exec_* metric', 'constraint_exec_sat_rate'),
    ):
        check(label, needle not in html, f'found {needle!r}' if needle in html else '')

    # ── 4. the inherited features are still there ────────────────────────────
    # A rebuild against a changed ancestor is the realistic way these vanish, and
    # they are the reason this page is derived rather than written.
    print('\n[inherited from DAv3 / VA v2]')
    for label, needle in (
            ('per-rollout view', 'def show_rollout_table('),
            ('compare view', 'def trigger_compare('),
            ('quality view', 'def render_quality('),
            ('result matrices', 'SUMMARY_TABLES'),
            ('LaTeX export', 'tex'),
            ('folder ZIP download', 'def download_folder('),
            ('candidate highlight', 'CLEAR HIGHLIGHTS'),
            ('(G, C) plot flags', 'FLAG_INPUTS'),
            ('Last Run column', 'Latest_Snapshot'),
            ('seed coverage', 'NOT FULL'),
            ('variant presets', 'VARIANT_PRESETS'),
            ('custom seed compare', 'seed-mode-select'),
    ):
        check(label, needle in html)

    print()
    if FAILURES:
        print(f'FAILED — {len(FAILURES)} check(s): {FAILURES}')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
