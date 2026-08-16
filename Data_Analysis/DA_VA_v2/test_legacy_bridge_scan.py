"""Regression test for legacy (bridged) tree handling — U4.2.

    python Data_Analysis/DA_VA_v2/test_legacy_bridge_scan.py

Stdlib only, on a synthetic tree, so it runs in the AI-coding container as well
as on the cluster (no numpy/pandas — the npz half of the bridge is exercised only
through its `--json-only` path).

What it pins down, end to end:

  1. `bridge_d3il_va_to_da_va_v2.py` turns a legacy D3IL baseline run into the
     DA_VA_v2 layout, and recomputes the entropy the old eval always reported
     as 0.0 (U4.1 bug B1).
  2. `discovery.detect_legacy()` flags the bridged tree (`_`-prefixed root +
     `_bridge_manifest.json`) and does NOT flag a natively exported one.
  3. Discovery resolves the bridged unit to (geo=none, variant=d3il_baseline).
  4. Every JSON key path `data_loader._load_json()` reads resolves in the
     rollout stats the bridge writes — checked with a local mirror of `_dig`,
     since importing the loader would require pandas.
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))                       # repo root
BASELINE = os.path.join(ROOT, 'd3il_visual_aligning_baseline_test')

sys.path.insert(0, BASELINE)
sys.path.insert(0, HERE)

import discovery as va2                                             # noqa: E402
import da_va_export as api                                          # noqa: E402
import bridge_d3il_va_to_da_va_v2 as bridge                         # noqa: E402

failures = []

N_CONTEXTS, N_TRAJS = 4, 3
AGENT = 'ddpm_encdec_vision'
SEED = 42


def check(label, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {label}' + (f'   ({detail})' if detail and not ok else ''))
    if not ok:
        failures.append(label)


def eq(label, got, want):
    check(label, got == want, f'got {got!r}, want {want!r}')


def dig(node, path):
    """Mirror of data_loader._dig — nested lookup, None counts as missing."""
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def write_legacy_run(root, successes):
    """The ORIGINAL baseline layout: results json + flat diagnostics + rt logs."""
    seed_dir = os.path.join(root, AGENT, f'seed_{SEED}')
    diag = os.path.join(seed_dir, 'diagnostics')
    os.makedirs(diag)
    for ctx in range(N_CONTEXTS):
        for traj in range(N_TRAJS):
            ridx = ctx * N_TRAJS + traj
            stats = {
                'rollout_index': ridx,
                'success': ridx in successes,
                'steps': 400,
                'mean_distance': 0.40,
                'mode': traj % 2,
                'context_info': {
                    'context_idx': ctx,
                    'box_init_xy': [0.5, -0.2], 'box_init_angle_deg': -43.5,
                    'target_xy': [0.5, 0.33], 'target_angle_deg': -58.3,
                    'init_xy_dist': 0.54, 'final_box_xy': [0.51, -0.18],
                    'final_box_angle_deg': -40.0, 'final_xy_dist': 0.51,
                },
            }
            with open(os.path.join(diag, f'rollout_{ridx}_stats.json'), 'w') as f:
                json.dump(stats, f)
            with open(os.path.join(seed_dir,
                                   f'realtime_baseline_ctx{ctx}_traj{traj}.log'), 'w') as f:
                f.write('#            total_ms mean=20.0  max=30.0  p95=25.0\n')
    with open(os.path.join(seed_dir, f'results_seed_{SEED}.json'), 'w') as f:
        json.dump({'agent_name': AGENT, 'seed': SEED,
                   'n_contexts': N_CONTEXTS, 'n_trajectories_per_context': N_TRAJS,
                   'success_rate': len(successes) / (N_CONTEXTS * N_TRAJS),
                   'entropy': 0.0}, f)                    # ← the always-0.0 legacy value
    return seed_dir


root = tempfile.mkdtemp(prefix='da_va_v2_legacy_test_')
try:
    source_root = os.path.join(root, 'logs', 'd3il_visual_aligning_baseline')
    os.makedirs(source_root)

    # Context 0 succeeds in BOTH modes (traj 0 → mode 0, traj 1 → mode 1) and
    # context 1 in one mode only ⇒ entropy = (1.0 + 0 + 0 + 0)/4 = 0.25 exactly.
    write_legacy_run(source_root, successes={0, 1, 3})

    print('\n[1] bridge (--json-only)')
    code = bridge.main(['--source-root', source_root, '--json-only'])
    eq('bridge exits 0', code, 0)

    bridged_root = os.path.join(source_root, api.DA_BRIDGE_ROOT_NAME)
    unit = api.unit_dir(bridged_root, AGENT, SEED)
    check('unit directory exists', os.path.isdir(unit), unit)
    check('manifest written',
          os.path.isfile(os.path.join(bridged_root, api.BRIDGE_MANIFEST_NAME)))

    with open(os.path.join(unit, 'unit_meta.json')) as f:
        meta = json.load(f)
    eq('rollouts bridged', meta['scalars']['n_rollouts'], N_CONTEXTS * N_TRAJS)
    eq('success_rate recomputed', round(meta['scalars']['success_rate'], 6), 0.25)
    check('entropy recomputed, not the legacy 0.0',
          abs(meta['scalars']['entropy'] - 0.25) < 1e-6,
          f'got {meta["scalars"]["entropy"]}')
    eq('legacy entropy kept for the audit trail',
       meta['args']['entropy_legacy_broken'], 0.0)
    eq('timing parsed from every realtime log',
       meta['args']['rollouts_with_timing'], N_CONTEXTS * N_TRAJS)

    print('\n[2] discovery: legacy vs native')
    candidates = va2.discover_candidates(bridged_root)
    eq('one bridged candidate', len(candidates), 1)
    info = candidates[1]
    check('flagged legacy', info.get('legacy') is True)
    eq('legacy kind from the manifest', info.get('legacy_kind'),
       'd3il_visual_aligning_baseline')
    eq('manifest says: no projector', info.get('has_projector'), False)
    eq('candidate name', info['name'], api.candidate_name(AGENT))

    native_root = os.path.join(source_root, api.DA_NATIVE_ROOT_NAME)
    api.write_unit(native_root, AGENT, SEED,
                   [api.make_record(0, 0, 0, True, 400, 0.4, 0, {'context_idx': 0})],
                   api.summarise([], 1, 1), write_npz=False)
    native = va2.discover_candidates(native_root)
    eq('one native candidate', len(native), 1)
    check('native tree NOT flagged legacy', native[1].get('legacy') is False)

    print('\n[3] discovery: unit resolution')
    units = va2.discover_units(1, info)
    eq('one unit', len(units), 1)
    eq('geo', units[0]['geo'], va2.GEO_NONE)
    eq('variant', units[0]['variant'], api.DA_VARIANT_NAME)
    eq('split', units[0]['split'], 'test')
    check('legacy flag rides on the unit', units[0]['legacy'] is True)
    check('json source (no npz in --json-only mode)', units[0]['npz_path'] is None)
    check('diagnostics dir found', os.path.isdir(units[0]['diagnostics_dir']))

    print('\n[4] rollout JSON answers every data_loader path')
    with open(os.path.join(units[0]['diagnostics_dir'], 'rollout_0_stats.json')) as f:
        row = json.load(f)
    for label, paths, want_missing in [
        ('success.strict', [['success', 'strict'], ['success']], False),
        ('outcome.mean_distance', [['outcome', 'mean_distance'], ['mean_distance']], False),
        ('timing.steps', [['timing', 'steps'], ['steps']], False),
        ('timing.avg_inference_time_per_replan',
         [['timing', 'avg_inference_time_per_replan'],
          ['avg_inference_time_per_replan']], False),
        ('mode', [['mode']], False),
        ('context.init_xy_dist',
         [['context', 'init_xy_dist'], ['context_info', 'init_xy_dist']], False),
        ('context.final_xy_dist', [['context', 'final_xy_dist']], False),
        ('context.box_init_xy', [['context', 'box_init_xy']], False),
        # Must stay absent: a truthy value here would mark the rollout D1-frozen.
        ('context.box_obstacle_conflict',
         [['context', 'box_obstacle_conflict']], True),
        # `constraint: null` must resolve to missing, not crash the walk.
        ('constraint.exec.sat_rate',
         [['constraint', 'exec', 'constraint_sat_rate']], True),
    ]:
        hit = next((v for v in (dig(row, p) for p in paths) if v is not None), None)
        check(f'{label} {"absent" if want_missing else "resolves"}',
              (hit is None) if want_missing else (hit is not None),
              f'got {hit!r}')

    print('\n[5] a second source tree with the same agent+seed')
    # The real case: logs/d3il_visual_aligning_baseline/ and
    # logs/d3il_visual_aligning_baseline(Bf_U3)/ both hold ddpm_encdec_vision/seed_42.
    # That older tree is also a KILLED run: no results_seed_*.json (so the scale
    # has to be inferred or pinned) and no realtime logs (so avg_time is NaN).
    other_root = os.path.join(root, 'logs', 'd3il_visual_aligning_baseline(older)')
    os.makedirs(other_root)
    other_seed = write_legacy_run(other_root, successes=set())
    os.remove(os.path.join(other_seed, f'results_seed_{SEED}.json'))
    for name in os.listdir(other_seed):
        if name.startswith('realtime_'):
            os.remove(os.path.join(other_seed, name))

    code = bridge.main(['--source-root', other_root, '--out-root', bridged_root,
                        '--json-only'])
    eq('collision: nothing bridged, exit 1', code, 1)
    with open(os.path.join(unit, 'unit_meta.json')) as f:
        after = json.load(f)
    eq('the original unit was NOT overwritten',
       after['args']['legacy_source_dir'], meta['args']['legacy_source_dir'])

    labelled_root = os.path.join(root, 'bridge_labelled')
    code = bridge.main(['--source-root', other_root, '--out-root', labelled_root,
                        '--label', 'older', '--n-contexts', str(N_CONTEXTS),
                        '--n-trajectories', str(N_TRAJS), '--json-only'])
    eq('labelled bridge exits 0', code, 0)
    labelled = api.unit_dir(labelled_root, AGENT, SEED, label='older')
    check('labelled candidate is a separate folder',
          os.path.isdir(labelled) and '__older' in labelled, labelled)
    with open(os.path.join(labelled, 'unit_meta.json')) as f:
        lmeta = json.load(f)
    eq('n_contexts pinned from the CLI, not inferred',
       lmeta['args']['n_contexts'], N_CONTEXTS)
    eq('n_trajs pinned from the CLI, not inferred',
       lmeta['args']['n_trajectories_per_context'], N_TRAJS)
    eq('no realtime logs ⇒ no rollout carries timing',
       lmeta['args']['rollouts_with_timing'], 0)
    with open(os.path.join(labelled, 'diagnostics', 'rollout_0_stats.json')) as f:
        eq('missing timing serialises as null, not NaN',
           json.load(f)['timing']['avg_inference_time_per_replan'], None)

    both = va2.discover_candidates(f'{bridged_root},{labelled_root}')
    eq('the two trees are two candidates', len(both), 2)
    eq('distinct candidate names',
       len({c['name'] for c in both.values()}), 2)
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f'{len(failures)} FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('all legacy-bridge checks passed')
