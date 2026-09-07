"""DA_UAV_v1 — discovery + axis-parser regression test.

    python Data_Analysis/DA_UAV_v1/test_discovery_offline.py

Stdlib only, so it runs in the AI-coding container (no numpy/pandas there). It
builds a synthetic UAV tree in a temp dir with the exact shape
`eval_mix_uav.py` writes, plus a Gen11 tree and a visual-aligning-shaped tree
beside it, and asserts:

  * the candidate is the EVAL-TAG folder, not the model folder and not a seed;
  * the geo/variant split is right for `{seed}/{geo}/{variant}/{variant}.npz`
    (the UAV shape, with NO `results/` level) and still right for the four
    shapes that do have one;
  * scene / engine / K / mpc / controller / threshold / backbone come out of the
    path correctly, including the underscore-bearing controller token and the
    Gen11 spelling with no `E{engine}` prefix;
  * `.partial.npz` sidecars, `expert_references/` and `config_snapshot_*` are
    not mistaken for results;
  * a variant with only `diagnostics/` and no npz is still found (JSON source);
  * the config-snapshot stamps and the PROJECTION_CB_TRIPPED sentinel are seen.

It does NOT test the loaders — those need the science stack and therefore the
cluster.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import discovery  # noqa: E402

FAILURES = []


def check(label, got, want):
    if got == want:
        print(f'  PASS  {label}')
    else:
        print(f'  FAIL  {label}\n          got:  {got!r}\n          want: {want!r}')
        FAILURES.append(label)


def check_true(label, value):
    check(label, bool(value), True)


# ──────────────────────────────────────────────────────────────────────────────
# fixture
# ──────────────────────────────────────────────────────────────────────────────

CORRIDOR_GEO = 'corridor_bounds+dynamics+geo_bounds+halfspace+obstacles'
PILLARS_GEO = 'pillars_bounds+dynamics+geo_bounds+obstacles'


def _touch(path, text=''):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def build_tree(root):
    """A miniature but structurally exact copy of what the evals write."""

    # ── Gen15: two engines x two K values, one scene, plus a second scene ─────
    for engine, model in (('mf', 'H8_DMeanFlowODE_9D_dp0.5_bbunet'),
                          ('fm', 'H8_DFlowMatchingODE_9D')):
        for k in (4, 20):
            tag = f'E{engine}_K{k}_mpc4_pid_stopgo_T0.5'
            base = os.path.join(root, 'logs/UAV_MIX/uav-corridor/plans',
                                f'mix_uav_{engine}', model, tag)
            for seed in (6, 7):
                seed_dir = os.path.join(base, str(seed))
                for variant in ('diffuser', 'dpcc-c', 'dpcc-c-tightened'):
                    vdir = os.path.join(seed_dir, CORRIDOR_GEO, variant)
                    _touch(os.path.join(vdir, f'{variant}.npz'))
                    _touch(os.path.join(vdir, f'eval_{variant}.log'))
                    _touch(os.path.join(vdir, 'results.json'), '{}')
                    _touch(os.path.join(vdir, 'diagnostics',
                                        'rollout_0_stats.json'), '{}')
                # crash-safety sidecar + the eval's own geo schematic: neither is
                # a result, and the schematic sits directly in the geo folder.
                _touch(os.path.join(seed_dir, CORRIDOR_GEO,
                                    'dpcc-t', 'dpcc-t.partial.npz'))
                _touch(os.path.join(seed_dir, CORRIDOR_GEO,
                                    'constraint_overview.png'))
                # config snapshot markers (two launches for seed 6, one for 7)
                snap = os.path.join(seed_dir, 'config_snapshot_uav_mix')
                _touch(os.path.join(snap, 'snapshot_20260814_101500'))
                if seed == 6:
                    _touch(os.path.join(snap, 'snapshot_20260815_090000'))

    # a variant that died before its final npz write: diagnostics only
    dead = os.path.join(root, 'logs/UAV_MIX/uav-corridor/plans/mix_uav_mf',
                        'H8_DMeanFlowODE_9D_dp0.5_bbunet',
                        'Emf_K4_mpc4_pid_stopgo_T0.5', '6', CORRIDOR_GEO,
                        'hardflow_new-c')
    _touch(os.path.join(dead, 'diagnostics', 'rollout_0_stats.json'), '{}')

    # a variant whose projection circuit breaker opened
    tripped = os.path.join(root, 'logs/UAV_MIX/uav-corridor/plans/mix_uav_fm',
                           'H8_DFlowMatchingODE_9D', 'Efm_K20_mpc4_pid_stopgo_T0.5',
                           '6', CORRIDOR_GEO, 'dpcc-c')
    _touch(os.path.join(tripped, 'PROJECTION_CB_TRIPPED.txt'), 'tripped')

    # second scene, second geo tag
    pillars = os.path.join(root, 'logs/UAV_MIX/uav-pillars/plans/mix_uav_af',
                           'H8_DAlphaFlowODE_9D_as1.0_ae0.0_bbdit',
                           'Eaf_K2_mpc8_mjpc_T0.25', '6', PILLARS_GEO, 'dpcc-r')
    _touch(os.path.join(pillars, 'dpcc-r.npz'))

    # ── Gen11: same artifacts, different root, eval tag with no E-token ───────
    gen11 = os.path.join(root, 'logs/UAV_FM/uav-corridor/plans/flow_matching_v3_uav',
                         'H8_DFlowMatchingODE_9D', 'K20_mpc4_pid_T0.5',
                         '6', CORRIDOR_GEO, 'dpcc-c')
    _touch(os.path.join(gen11, 'dpcc-c.npz'))

    # ── an aligning-shaped tree (has a results/ level) beside them ────────────
    va = os.path.join(root, 'logs/aligning-d3il-visual/plans/mix_visual_aligning_mf',
                      'H8_K2_Emf', '6', 'results', 'combined_5', 'dpcc-c')
    _touch(os.path.join(va, 'dpcc-c.npz'))
    _touch(os.path.join(root, 'logs/aligning-d3il-visual/plans/mix_visual_aligning_mf',
                        'H8_K2_Emf', '6', 'results', 'expert_references', 'ref.npz'))


# ──────────────────────────────────────────────────────────────────────────────
# tests
# ──────────────────────────────────────────────────────────────────────────────

def test_eval_tag_parse():
    print('\n[eval-tag parser]')
    got = discovery.parse_eval_tag('Emf_K4_mpc4_pid_stopgo_T0.5')
    check('Gen15 tag: engine', got.get('engine'), 'mf')
    check('Gen15 tag: K', got.get('K'), 4)
    check('Gen15 tag: mpc_batch', got.get('mpc_batch'), 4)
    # The controller token contains an underscore — the reason the regex is
    # greedy between `mpc{n}_` and the trailing `_T{thresh}`.
    check('Gen15 tag: controller (underscored)', got.get('controller'), 'pid_stopgo')
    check('Gen15 tag: threshold', got.get('threshold'), 0.5)

    got = discovery.parse_eval_tag('Eaf_K2_mpc8_mjpc_T0.25')
    check('Gen15 tag: single-word controller', got.get('controller'), 'mjpc')
    check('Gen15 tag: threshold 0.25', got.get('threshold'), 0.25)

    got = discovery.parse_eval_tag('K20_mpc4_pid_T0.5')
    check('Gen11 tag: engine blank', got.get('engine'), '')
    check('Gen11 tag: K', got.get('K'), 20)
    check('Gen11 tag: controller', got.get('controller'), 'pid')

    check('unparsable tag yields {}', discovery.parse_eval_tag('random_folder'), {})

    # ── trailing run tag (FMPCC_UAV_EVAL_TAG) ────────────────────────────────
    # Regression: before Fix_16 DA (2026-09-03) these folders did not match at
    # all, K came back None, and the axis groupbys dropped every one of their
    # rollouts on the NaN key — silently.
    got = discovery.parse_eval_tag('Emf_K1_mpc4_pid_stopgo_T0.5_fix16scaled')
    check('run tag: engine', got.get('engine'), 'mf')
    check('run tag: K parsed (was None)', got.get('K'), 1)
    check('run tag: controller still right', got.get('controller'), 'pid_stopgo')
    check('run tag: threshold still right', got.get('threshold'), 0.5)
    check('run tag: captured', got.get('run_tag'), 'fix16scaled')

    got = discovery.parse_eval_tag('Emf_K5_mpc4_pid_stopgo_T0.5_fix16legacy')
    check('run tag: other arm', got.get('run_tag'), 'fix16legacy')
    check('run tag: other arm K', got.get('K'), 5)

    # untagged folders must stay untagged, not absorb the controller or threshold
    got = discovery.parse_eval_tag('Emf_K4_mpc4_pid_stopgo_T0.5')
    check('no run tag → empty', got.get('run_tag'), '')
    check('no run tag: controller intact', got.get('controller'), 'pid_stopgo')

    # tag containing separators the sanitiser allows (`[^A-Za-z0-9._-]` → `-`)
    got = discovery.parse_eval_tag('Efm_K20_mpc4_pid_const_v_T0.5_ab-1.2_x')
    check('dotted/dashed tag', got.get('run_tag'), 'ab-1.2_x')
    check('dotted/dashed tag: controller', got.get('controller'), 'pid_const_v')
    check('dotted/dashed tag: threshold', got.get('threshold'), 0.5)

    # Gen11 spelling (no E-token) takes a tag too
    got = discovery.parse_eval_tag('K20_mpc4_pid_stopgo_anchorP_T0.5_probe')
    check('Gen11 + run tag: K', got.get('K'), 20)
    check('Gen11 + run tag: controller', got.get('controller'), 'pid_stopgo_anchorP')
    check('Gen11 + run tag: tag', got.get('run_tag'), 'probe')

    # display_name must keep the two arms of an A/B apart
    base = {'scene': 'pillars', 'engine': 'mf', 'K': 1, 'backbone': 'unet',
            'data_proportion': '0.5'}
    a = discovery.display_name(dict(base, run_tag='fix16scaled'), 'x')
    b = discovery.display_name(dict(base, run_tag='fix16legacy'), 'x')
    c = discovery.display_name(dict(base, run_tag=''), 'x')
    check('display_name: arms differ', a != b, True)
    check('display_name: arm vs untagged differ', a != c, True)
    check('display_name: untagged unchanged', c, 'pillars|mf|K1|bbunet|dp0.5')
    check('display_name: tag suffix', a, 'pillars|mf|K1|bbunet|dp0.5|@fix16scaled')

    # A folder that is not eval-tag-shaped is the ordinary legacy case (a Gen11
    # MODEL folder used as the candidate) and must stay QUIET; one that IS
    # eval-tag-shaped but does not parse is a parser bug and must be LOUD.
    import logging

    class _Catch(logging.Handler):
        def __init__(self):
            super().__init__()
            self.seen = []

        def emit(self, record):
            self.seen.append(record.getMessage())

    catcher = _Catch()
    discovery.logger.addHandler(catcher)
    # main() sets the ROOT level to ERROR to keep this suite's output readable;
    # discovery.logger inherits it, so a WARNING would be filtered before it ever
    # reaches the handler. Pin the level for the duration of this check.
    previous_level = discovery.logger.level
    discovery.logger.setLevel(logging.WARNING)
    discovery._WARNED_UNPARSED_TAGS.clear()
    discovery.parse_eval_tag('H8_Dmodels.diffusion.FlowMatchingODE_9D')
    check('legacy model folder warns: no', len(catcher.seen), 0)
    discovery.parse_eval_tag('Emf_K1_mpc4_pid_stopgo_TWOPOINTFIVE')
    check('tag-shaped miss warns: yes', len(catcher.seen), 1)
    discovery.parse_eval_tag('Emf_K1_mpc4_pid_stopgo_TWOPOINTFIVE')
    check('and only once per name', len(catcher.seen), 1)
    discovery.logger.setLevel(previous_level)
    discovery.logger.removeHandler(catcher)


def test_model_dir_parse():
    print('\n[model-dir parser]')
    got = discovery.parse_model_dir('mix_uav_mf/H8_DMeanFlowODE_9D_dp0.5_bbunet')
    check('engine from prefix folder', got.get('engine'), 'mf')
    check('horizon', got.get('horizon'), '8')
    check('diffusion class', got.get('diffusion_cls'), 'MeanFlowODE')
    check('obs dim tag', got.get('obs_dim_tag'), '9D')
    check('data proportion', got.get('data_proportion'), '0.5')
    check('backbone', got.get('backbone'), 'unet')

    got = discovery.parse_model_dir('mix_uav_af/H8_DAlphaFlowODE_9D_as1.0_ae0.0_bbdit')
    check('alpha init', got.get('alpha_init'), '1.0')
    check('alpha end', got.get('alpha_end'), '0.0')
    check('backbone dit', got.get('backbone'), 'dit')

    # The DDPM arm is the only one carrying a train-time K in the CHECKPOINT name.
    got = discovery.parse_model_dir('mix_uav_diffusion/H8_DGaussianDiffusion_9D_K20')
    check('train-time K', got.get('train_K'), '20')
    check('engine diffusion', got.get('engine'), 'diffusion')


def test_geo_scene():
    print('\n[geo_tag -> scene]')
    # The split is on the LAST underscore because scene names contain them.
    check('s_curve keeps its underscore', discovery.geo_scene('s_curve_dynamics'), 's_curve')
    check('corridor full stack', discovery.geo_scene(CORRIDOR_GEO), 'corridor')
    check('unconstrained', discovery.geo_scene('empty_unconstrained'), 'empty')
    check('no suffix passes through', discovery.geo_scene('corridor'), 'corridor')


def test_variant_parts():
    print('\n[variant tightening]')
    check('tightened split', discovery.variant_parts('dpcc-c-tightened'),
          ('dpcc-c', True))
    check('plain variant', discovery.variant_parts('dpcc-c'), ('dpcc-c', False))
    check('multi-toggle tightened',
          discovery.variant_parts('model_free-bounds_free-tightened'),
          ('model_free-bounds_free', True))


def test_discovery(root):
    print('\n[candidate discovery]')
    candidates = discovery.discover_candidates(os.path.join(root, 'logs'))
    by_tag = {info['name']: (key, info) for key, info in candidates.items()}

    check('candidate count', len(candidates), 7)
    for tag in ('Emf_K4_mpc4_pid_stopgo_T0.5', 'Emf_K20_mpc4_pid_stopgo_T0.5',
                'Efm_K4_mpc4_pid_stopgo_T0.5', 'Efm_K20_mpc4_pid_stopgo_T0.5',
                'Eaf_K2_mpc8_mjpc_T0.25', 'K20_mpc4_pid_T0.5', 'H8_K2_Emf'):
        check_true(f'candidate present: {tag}', tag in by_tag)

    _, mf4 = by_tag['Emf_K4_mpc4_pid_stopgo_T0.5']
    check('mf/K4 seeds', mf4['seeds'], [6, 7])
    check('mf/K4 scene', mf4['axes']['scene'], 'corridor')
    check('mf/K4 engine', mf4['axes']['engine'], 'mf')
    check('mf/K4 K', mf4['axes']['K'], 4)
    check('mf/K4 backbone', mf4['axes']['backbone'], 'unet')
    check('mf/K4 generation', mf4['axes']['generation'], 'Gen15')
    check('mf/K4 display name', mf4['display'], 'corridor|mf|K4|bbunet|dp0.5')
    check('mf/K4 newest snapshot', mf4['snapshots']['latest'], '20260815_090000')
    check('mf/K4 oldest snapshot', mf4['snapshots']['first'], '20260814_101500')
    check('mf/K4 snapshot count', mf4['snapshots']['count'], 3)
    check('mf/K4 per-seed snapshots', mf4['snapshots']['per_seed'],
          {6: '20260815_090000', 7: '20260814_101500'})

    _, gen11 = by_tag['K20_mpc4_pid_T0.5']
    check('Gen11 generation', gen11['axes']['generation'], 'Gen11')
    check('Gen11 engine (unparsable, no prefix folder)', gen11['axes']['engine'], '')
    check('Gen11 K', gen11['axes']['K'], 20)
    check('Gen11 display falls back to axes it has',
          gen11['display'], 'corridor|K20|Gen11')

    _, af = by_tag['Eaf_K2_mpc8_mjpc_T0.25']
    check('af scene', af['axes']['scene'], 'pillars')
    check('af mpc batch', af['axes']['mpc_batch'], 8)
    check('af controller', af['axes']['controller'], 'mjpc')
    check('af threshold', af['axes']['threshold'], 0.25)

    _, fm20 = by_tag['Efm_K20_mpc4_pid_stopgo_T0.5']
    check('circuit-breaker sentinel counted', fm20['cb_sentinels'], 1)

    return candidates, by_tag


def test_units(candidates, by_tag):
    print('\n[unit discovery]')
    key, mf4 = by_tag['Emf_K4_mpc4_pid_stopgo_T0.5']
    units = discovery.discover_units(key, mf4)
    names = sorted((u['seed'], u['geo'], u['variant']) for u in units)

    # 2 seeds x 3 npz variants + the npz-less hardflow variant on seed 6.
    check('mf/K4 unit count', len(units), 7)
    check('mf/K4 geo is the geo_tag, not "none"',
          sorted({u['geo'] for u in units}), [CORRIDOR_GEO])
    check('mf/K4 variants',
          sorted({u['variant'] for u in units}),
          ['diffuser', 'dpcc-c', 'dpcc-c-tightened', 'hardflow_new-c'])
    check('partial npz not a unit',
          any(u['variant'] == 'dpcc-t' for u in units), False)
    check('geo-level png not a unit',
          any(u['variant'] == 'constraint_overview' for u in units), False)

    json_only = [u for u in units if u['npz_path'] is None]
    check('json-only unit found', len(json_only), 1)
    check('json-only unit is the hardflow one', json_only[0]['variant'],
          'hardflow_new-c')
    check('json-only unit keeps its geo', json_only[0]['geo'], CORRIDOR_GEO)

    tightened = [u for u in units if u['variant'] == 'dpcc-c-tightened'][0]
    check('tightened flag on the VARIANT axis', tightened['tightened'], True)
    check('variant_base strips -tightened', tightened['variant_base'], 'dpcc-c')
    check('geo axis carries no tightening', tightened['geo'], CORRIDOR_GEO)

    check('axes ride on the unit', (tightened['scene'], tightened['engine'],
                                    tightened['K']), ('corridor', 'mf', 4))
    check('cb sentinel flag off here', tightened['cb_sentinel'], False)
    check('split is test', sorted({u['split'] for u in units}), ['test'])
    check('unit sort order is (seed, split, geo, variant)',
          names[0], (6, CORRIDOR_GEO, 'diffuser'))

    print('\n[unit discovery — aligning-shaped tree with a results/ level]')
    key, va = by_tag['H8_K2_Emf']
    units = discovery.discover_units(key, va)
    check('aligning unit count', len(units), 1)
    check('aligning geo below results/', units[0]['geo'], 'combined_5')
    check('aligning variant', units[0]['variant'], 'dpcc-c')
    check('expert_references skipped',
          any(u['variant'] == 'ref' for u in units), False)

    print('\n[unit discovery — circuit-breaker sentinel on a unit]')
    key, fm20 = by_tag['Efm_K20_mpc4_pid_stopgo_T0.5']
    units = discovery.discover_units(key, fm20)
    tripped = [u for u in units if u['cb_sentinel']]
    check('exactly one unit carries the sentinel', len(tripped), 1)
    check('and it is dpcc-c', tripped[0]['variant'], 'dpcc-c')


def test_axis_filters(candidates):
    print('\n[axis filters]')
    # 4 Gen15 corridor candidates + the Gen11 one. The aligning-shaped tree has
    # no `uav-<scene>` folder, so its scene is blank and it is correctly dropped.
    check('scene filter',
          len(discovery.filter_by_axes(candidates, scenes=['corridor'])), 5)
    check('engine filter',
          len(discovery.filter_by_axes(candidates, engines=['mf'])), 2)
    check('K filter',
          len(discovery.filter_by_axes(candidates, k_values=[4])), 2)
    check('combined filter',
          len(discovery.filter_by_axes(candidates, scenes=['corridor'],
                                       engines=['mf'], k_values=[4])), 1)
    check('no filter is a no-op',
          len(discovery.filter_by_axes(candidates)), len(candidates))


def test_snapshot_formatting():
    print('\n[snapshot formatting]')
    check('stamp to human', discovery.format_snapshot_ts('20260815_090000'),
          '2026-08-15 09:00:00')
    check('non-stamp passes through', discovery.format_snapshot_ts('n/a'), 'n/a')
    check('per-seed string',
          discovery.snapshot_by_seed_str({7: 'b', 6: 'a'}), '6:a | 7:b')
    check('empty per-seed string', discovery.snapshot_by_seed_str({}), '')


def main():
    root = tempfile.mkdtemp(prefix='da_uav_v1_test_')
    try:
        build_tree(root)
        test_eval_tag_parse()
        test_model_dir_parse()
        test_geo_scene()
        test_variant_parts()
        candidates, by_tag = test_discovery(root)
        test_units(candidates, by_tag)
        test_axis_filters(candidates)
        test_snapshot_formatting()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAILURES:
        print(f'FAILED — {len(FAILURES)} check(s): {FAILURES}')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s %(message)s')
    sys.exit(main())
