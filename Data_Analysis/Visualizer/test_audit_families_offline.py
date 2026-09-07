"""Check the Path Audit Map's model-family parser (U19) against every real batch.

    python Data_Analysis/Visualizer/test_audit_families_offline.py

Stdlib only (csv + re) — the family rules are pure string work over `Full_Path`, so
unlike test_page_offline.py in Visualizer_VA_v2 this one runs anywhere, including the
AI-coding container.

What it guards: the tabs in the audit map are PARSED off the run path, not configured.
That is only safe while the parser actually recognises the folders the cluster writes —
so this asserts that no run in any batch under Data_Analysis/analysis_results/ falls
through to OTHER, and prints the folder names behind each tab so a new model folder
showing up in the wrong family is visible in the diff of this output.
"""
import collections
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, 'index.html')
BATCHES = os.path.join(HERE, os.pardir, 'analysis_results')

# The parser is read out of the page itself rather than copied here: a second copy would
# drift, and then this test would be certifying code the page does not run.
src = open(HTML, encoding='utf-8').read()
block = re.search(r'<script type="py">\n(.*?)\n</script>', src, re.S).group(1)
frag = block[block.index('FAMILY_ORDER = ['):block.index('def _active_audit_tab')]
ns = {'re': re}
exec(compile(frag, HTML, 'exec'), ns)
model_folder, model_family = ns['_model_folder'], ns['_model_family']
FAMILY_LABELS = ns['FAMILY_LABELS']

# Folder -> family, spelled out for the shapes that are easy to get wrong: `imeanflow`
# contains `meanflow`, `hardflow` contains `af` as a substring, `plans(Bf_U8)` is a
# versioned plans dir, and the DA_VA bridge folders have no plans/ segment at all.
FIXTURES = [
    ('/l/plans/flow_matching_v3_meanflow(Bf_Fix5)/train/eval', 'mf'),
    ('/l/plans/flow_matching_v3_imeanflow(cfgO_0_in_eval_DiT_1e5)/train/eval', 'imf'),
    ('/l/plans/flow_matching_v3_alphaflow(Bf_U3)/train/eval', 'af'),
    ('/l/plans/flow_matching_v3_hardflow(Gen12_Bf_U5)/train/eval', 'hardflow'),
    ('/l/plans/flow_matching_v3_ode_selectable/train/eval', 'fm'),
    ('/l/plans/flow_matching_v3_drifting/train/eval', 'drifting'),
    ('/l/plans/diffusion/train/eval', 'diffusion'),
    ('/l/plans/visual_avoiding_dpcc(unfull)/train/eval', 'diffusion'),
    ('/l/plans/mix_uav_mf/train/eval', 'mf'),
    ('/l/plans/mix_uav_af/train/eval', 'af'),
    ('/l/plans/mix_uav_fm/train/eval', 'fm'),
    ('/l/plans/mix_visual_aligning_diffusion/train/eval', 'diffusion'),
    ('/l/plans/fm_visual_avoiding/train/eval', 'fm'),
    ('/l/uav-pillars/plans(Bf_U8)/flow_matching_v3_uav/train/eval', 'fm'),
    ('/l/uav-s_curve/plans(Bf_DC-FIX)/flow_matching_v3_meanflow/train/eval', 'mf'),
    ('/l/logs/d3il_visual_aligning_baseline/_DA_VA_BRIDGE_d3il_baseline/'
     'd3il_baseline_ddpm_encdec_vision', 'diffusion'),
]

failures = []
for path, want in FIXTURES:
    got = model_family(path)
    if got != want:
        failures.append(f'FIXTURE {path}\n    want {want}, got {got} '
                        f'(model folder = {model_folder(path)!r})')

paths = set()
for csv_path in sorted(glob.glob(os.path.join(BATCHES, '*', 'candidates_multidimensional_aggregated.csv'))):
    with open(csv_path, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            p = row.get('Full_Path') or row.get('Folder_Name') or ''
            if p:
                paths.add(p)

if not paths:
    print('SKIP — no batches under', BATCHES)
    sys.exit(0)

by_family = collections.defaultdict(set)
for p in paths:
    by_family[model_family(p)].add(model_folder(p))

print(f'{len(paths)} distinct run paths across {len(glob.glob(os.path.join(BATCHES, "*", "candidates_multidimensional_aggregated.csv")))} batches\n')
for key, label in ns['FAMILY_ORDER']:
    folders = sorted(by_family.get(key, ()))
    if not folders:
        continue
    print(f'{label:18s} {len(folders):3d} folder(s)')
    for f in folders:
        print(f'    {f}')

if by_family.get('other'):
    failures.append('these run folders fell through to OTHER — add a rule or check the path shape:\n    '
                    + '\n    '.join(sorted(by_family['other'])))

print()
if failures:
    print('FAIL')
    for f in failures:
        print(' -', f)
    sys.exit(1)
print('PASS — every run path classifies into a named model family')
