"""
Bridge: legacy D3IL visual-aligning baseline runs → DA_VA_v2-readable tree (U4.2).

Reads a finished run in the ORIGINAL baseline layout

    logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/
        results_seed_{s}.json
        diagnostics/rollout_{r}_stats.json
        realtime_baseline_ctx{c}_traj{t}.log        (optional — timing lives here)

and re-emits it in the layout DA_VA_v2 already understands (see
`da_va_export.py` for the contract):

    logs/d3il_visual_aligning_baseline/_DA_VA_BRIDGE_d3il_baseline/
        _bridge_manifest.json
        d3il_baseline_{agent_name}/{seed}/results/d3il_baseline/…

The leading underscore on the root is load-bearing: DA_VA_v2 treats a candidate
under a `_`-prefixed folder as legacy (`discovery.py::is_legacy_path`) and turns
on the legacy reader. Natively exported runs (`DA_VA_d3il_baseline/`, written by
`eval_d3il_visual_aligning.py --da-export`) carry no underscore and are read on
the normal Gen14 path.

Nothing is read from the source tree except JSON and `.log` text, and nothing in
it is modified — the bridge is a pure copy-forward.

Three things the bridge fixes or fills that the legacy files do not have:

  1. **entropy** — recomputed from the per-rollout modes. The legacy
     `results_seed_{s}.json` entropy is a fixed 0.0 for every run ever made
     (U4.1 §2 bug B1); it is carried over as `entropy_legacy_broken` for the
     record and never used.
  2. **avg_time** — parsed out of the `realtime_baseline_ctx*_traj*.log` SUMMARY
     block (`total_ms mean=…`). Absent logs ⇒ NaN, never a guess.
  3. **traj/context indices** — reconstructed from the rollout ordering
     (`rollout = ctx * n_trajs + traj`, the eval's own loop order) and
     cross-checked against `context_info.context_idx` in each stats JSON.

Usage (cluster, where the logs and numpy live):

    python d3il_visual_aligning_baseline_test/bridge_d3il_va_to_da_va_v2.py \
        --source-root logs/d3il_visual_aligning_baseline \
        --agent-name ddpm_encdec_vision            # omit for every agent found

    # path/JSON logic only, no numpy needed (skips the npz; DA_VA_v2 then reads
    # the diagnostics JSONs) — this is what runs in the AI container
    python ... --json-only

Then point DA_VA_v2 at it:

    python Data_Analysis/DA_VA_v2/main_da_batch.py \
        --parent-path logs/d3il_visual_aligning_baseline/_DA_VA_BRIDGE_d3il_baseline
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import da_va_export as api            # noqa: E402  (path set above)

# `#            total_ms mean=12.3  max=…` in the RTRecorder SUMMARY block.
_TOTAL_MS_RE = re.compile(r'total_ms\s+mean=([0-9.eE+-]+)')
_RT_LOG_RE = re.compile(r'realtime_baseline_ctx(\d+)_traj(\d+)\.log$')


# ──────────────────────────────────────────────────────────────────────────────
# reading the legacy tree
# ──────────────────────────────────────────────────────────────────────────────

def find_seed_dirs(agent_dir):
    """`{agent}/seed_{s}` directories that hold at least one result artefact."""
    out = []
    for entry in sorted(os.listdir(agent_dir)):
        if not entry.startswith('seed_'):
            continue
        path = os.path.join(agent_dir, entry)
        if not os.path.isdir(path):
            continue
        try:
            seed = int(entry[len('seed_'):])
        except ValueError:
            continue
        has_results = glob.glob(os.path.join(path, 'results_seed_*.json'))
        has_diag = os.path.isdir(os.path.join(path, 'diagnostics'))
        if has_results or has_diag:
            out.append((seed, path))
    return out


def find_agent_dirs(source_root, agent_name=None):
    """Agent folders under the legacy root, skipping our own export roots."""
    skip = {api.DA_NATIVE_ROOT_NAME, api.DA_BRIDGE_ROOT_NAME}
    out = []
    for entry in sorted(os.listdir(source_root)):
        if entry in skip or entry.startswith('.'):
            continue
        if agent_name and entry != agent_name:
            continue
        path = os.path.join(source_root, entry)
        if os.path.isdir(path) and find_seed_dirs(path):
            out.append((entry, path))
    return out


def read_results_json(seed_path, seed):
    path = os.path.join(seed_path, f'results_seed_{seed}.json')
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        print(f'[ WARN ] unreadable {path}: {exc}')
        return {}


def read_rollout_stats(seed_path):
    """Legacy `diagnostics/rollout_{r}_stats.json`, ordered by rollout index."""
    diag = os.path.join(seed_path, 'diagnostics')
    if not os.path.isdir(diag):
        return []
    rows = []
    for path in glob.glob(os.path.join(diag, 'rollout_*_stats.json')):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except (OSError, ValueError) as exc:
            print(f'[ WARN ] unreadable {path}: {exc}')
    rows.sort(key=lambda r: int(r.get('rollout_index', 10 ** 9)))
    return rows


def read_realtime_timings(seed_path):
    """{(ctx, traj) -> mean seconds per control step} from the RTRecorder logs.

    The per-step lines are ignored; only the SUMMARY `total_ms mean=` is used, so
    a truncated log (job killed mid-rollout) simply yields no entry.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(seed_path, 'realtime_baseline_ctx*_traj*.log'))):
        match = _RT_LOG_RE.search(os.path.basename(path))
        if not match:
            continue
        try:
            with open(path, errors='replace') as f:
                text = f.read()
        except OSError:
            continue
        hits = _TOTAL_MS_RE.findall(text)
        if not hits:
            continue
        try:
            out[(int(match.group(1)), int(match.group(2)))] = float(hits[-1]) / 1000.0
        except ValueError:
            continue
    return out


# ──────────────────────────────────────────────────────────────────────────────
# legacy → canonical records
# ──────────────────────────────────────────────────────────────────────────────

def build_records(rows, timings, n_trajs):
    """Legacy stats rows → `da_va_export.make_record()` records.

    Legacy schema (written by `eval_d3il_visual_aligning._export_rollout_realtime`):
        {rollout_index, success, steps, mean_distance, mode, context_info{…}}
    """
    records = []
    for position, row in enumerate(rows):
        ridx = int(row.get('rollout_index', position))
        ctx_info = dict(row.get('context_info') or {})

        # The eval loops context-major (`for ctx: for traj:`), so the rollout
        # index carries both coordinates. `context_idx` in the JSON is
        # authoritative when present; the derived one is the fallback and the
        # cross-check.
        derived_ctx = ridx // n_trajs if n_trajs > 0 else 0
        traj_idx = ridx % n_trajs if n_trajs > 0 else 0
        ctx_idx = int(ctx_info.get('context_idx', derived_ctx))
        if ctx_idx != derived_ctx:
            print(f'[ WARN ] rollout {ridx}: context_idx={ctx_idx} but ordering '
                  f'implies {derived_ctx} — trusting the JSON')

        records.append(api.make_record(
            rollout_index=ridx,
            context_idx=ctx_idx,
            traj_idx=traj_idx,
            success=bool(row.get('success', False)),
            steps=int(row.get('steps', 0) or 0),
            mean_distance=row.get('mean_distance'),
            mode=row.get('mode', -1),
            context_info=ctx_info,
            avg_time_s=timings.get((ctx_idx, traj_idx)),
        ))
    records.sort(key=lambda r: r['rollout_index'])
    return records


def infer_scale(results, rows, force_contexts=None, force_trajs=None):
    """(n_contexts, n_trajs) — from the CLI, else the results JSON, else the rows.

    Both are needed before records can be built: n_trajs is what splits a rollout
    index into (context, traj), and n_contexts sizes the entropy table.

    A killed run has no `results_seed_*.json` at all, and its diagnostics cover
    only the contexts it reached — inference then reports the contexts actually
    present (26 of 60, say), which is the honest denominator for entropy but not
    the scale the run was launched at. Pass `--n-contexts/--n-trajs` to pin it.
    """
    if force_contexts and force_trajs:
        return int(force_contexts), int(force_trajs)

    n_contexts = int(force_contexts or results.get('n_contexts') or 0)
    n_trajs = int(force_trajs or results.get('n_trajectories_per_context') or 0)
    if n_contexts > 0 and n_trajs > 0:
        return n_contexts, n_trajs

    contexts = set()
    for row in rows:
        ctx = (row.get('context_info') or {}).get('context_idx')
        if ctx is not None:
            contexts.add(int(ctx))
    n_contexts = n_contexts or len(contexts) or 1
    n_trajs = n_trajs or max(1, round(len(rows) / n_contexts))
    print(f'[ note ] results JSON lacked the eval scale — inferred '
          f'n_contexts={n_contexts}, n_trajectories_per_context={n_trajs} '
          f'from {len(rows)} rollout stats')
    return n_contexts, n_trajs


# ──────────────────────────────────────────────────────────────────────────────
# one seed
# ──────────────────────────────────────────────────────────────────────────────

def occupied_by_other(out_root, agent_name, seed, split, label, seed_path):
    """Is this candidate/seed slot already filled from a DIFFERENT source dir?

    Two source trees can hold the same agent+seed (`…/ddpm_encdec_vision/seed_42`
    exists in both the current logs root and the `(Bf_U3)` one). Same agent+seed
    ⇒ same output path ⇒ the second bridge would overwrite the first without a
    trace. `--label` separates them; this check makes forgetting it loud.
    """
    meta_path = os.path.join(api.unit_dir(out_root, agent_name, seed, split, label),
                             'unit_meta.json')
    if not os.path.isfile(meta_path):
        return ''
    try:
        with open(meta_path) as f:
            previous = (json.load(f).get('args') or {}).get('legacy_source_dir', '')
    except (OSError, ValueError):
        return ''
    if previous and os.path.abspath(previous) != os.path.abspath(seed_path):
        return previous
    return ''


def bridge_seed(out_root, agent_name, seed, seed_path, write_npz=True, split='test',
                label='', force_contexts=None, force_trajs=None, force=False):
    clash = occupied_by_other(out_root, agent_name, seed, split, label, seed_path)
    if clash and not force:
        print(f'[ SKIP ] {agent_name}/seed_{seed}: target already holds a bridge of a '
              f'DIFFERENT source\n'
              f'          existing: {clash}\n'
              f'          new:      {os.path.abspath(seed_path)}\n'
              f'          → rerun this source with --label <name> (e.g. --label Bf_U3), '
              f'or --force to overwrite')
        return None

    results = read_results_json(seed_path, seed)
    rows = read_rollout_stats(seed_path)
    if not rows:
        print(f'[ SKIP ] {agent_name}/seed_{seed}: no diagnostics/rollout_*_stats.json')
        return None

    n_contexts, n_trajs = infer_scale(results, rows, force_contexts, force_trajs)
    timings = read_realtime_timings(seed_path)
    records = build_records(rows, timings, n_trajs)

    scalars = api.summarise(records, n_contexts, n_trajs)
    n_timed = sum(1 for r in records if r['avg_time_s'] == r['avg_time_s'])

    # A run killed mid-sweep: fewer rollouts than the scale calls for. Not an
    # error — but every rate below is over the rollouts that finished, and the
    # contexts are the LOW-index ones, so it is not a random subsample.
    expected = n_contexts * n_trajs
    partial = len(records) < expected
    if partial:
        print(f'[ PARTIAL ] {agent_name}/seed_{seed}: {len(records)}/{expected} rollouts '
              f'({len(records) / expected * 100:.0f}%) — contexts '
              f'{min(r["context_idx"] for r in records)}..'
              f'{max(r["context_idx"] for r in records)} of 0..{n_contexts - 1}')

    args_extra = {
        'export_source': 'bridge',
        'legacy_source_dir': os.path.abspath(seed_path),
        'n_contexts': n_contexts,
        'n_trajectories_per_context': n_trajs,
        'label': label,
        'partial_run': bool(partial),
        'n_rollouts_expected': expected,
        # Kept for the audit trail — this is the value the buggy eval reported.
        'entropy_legacy_broken': results.get('entropy'),
        'success_rate_legacy': results.get('success_rate'),
        'rollouts_with_timing': n_timed,
    }
    if results.get('agent_name'):
        args_extra['legacy_agent_name'] = results['agent_name']

    stamp = _stamp_of(seed_path, seed)
    out_dir = api.write_unit(out_root, agent_name, seed, records, scalars,
                             args_extra=args_extra, split=split,
                             write_npz=write_npz, snapshot_stamp=stamp, label=label)

    legacy_sr = results.get('success_rate')
    mismatch = ''
    if legacy_sr is not None and abs(float(legacy_sr) - scalars['success_rate']) > 1e-6:
        mismatch = (f'  [ !! ] recomputed success_rate {scalars["success_rate"]:.4f} '
                    f'!= results_seed json {float(legacy_sr):.4f} — partial diagnostics?')
    print(f'[ ok ] {agent_name}/seed_{seed}: {len(records)} rollouts  '
          f'success={scalars["success_rate"]:.4f}  entropy={scalars["entropy"]:.4f}  '
          f'timing={n_timed}/{len(records)}  → {out_dir}{mismatch}')
    return {
        'agent_name': agent_name,
        'label': label,
        'candidate': api.candidate_name(agent_name, label),
        'seed': seed,
        'source': os.path.abspath(seed_path),
        'n_rollouts': len(records),
        'n_rollouts_expected': expected,
        'partial_run': bool(partial),
        'n_contexts': n_contexts,
        'n_trajectories_per_context': n_trajs,
        'success_rate': scalars['success_rate'],
        'entropy_recomputed': scalars['entropy'],
        'entropy_legacy_broken': results.get('entropy'),
        'rollouts_with_timing': n_timed,
        'npz': bool(write_npz),
    }


def _stamp_of(seed_path, seed):
    """Snapshot stamp = mtime of the run's own results JSON (else the folder)."""
    from datetime import datetime
    for candidate in (os.path.join(seed_path, f'results_seed_{seed}.json'), seed_path):
        try:
            return datetime.fromtimestamp(os.path.getmtime(candidate)).strftime('%Y%m%d_%H%M%S')
        except OSError:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Bridge legacy D3IL visual-aligning baseline runs into a '
                    'DA_VA_v2-readable tree',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('--source-root', default='logs/d3il_visual_aligning_baseline',
                   help='Root holding {agent_name}/seed_{s}/ (default: %(default)s)')
    p.add_argument('--out-root', default=None,
                   help=f'Bridged tree root (default: '
                        f'<source-root>/{api.DA_BRIDGE_ROOT_NAME})')
    p.add_argument('--agent-name', default=None,
                   help='Only bridge this agent folder (default: every one found)')
    p.add_argument('--seeds', default=None,
                   help='Comma-separated seeds (default: every seed found)')
    p.add_argument('--split', default='test', choices=['test', 'train'],
                   help='Which results root the source run wrote (default: test)')
    p.add_argument('--label', default='',
                   help='Suffix for the candidate folder (d3il_baseline_{agent}__{label}). '
                        'REQUIRED when bridging two source trees that share an '
                        'agent+seed — e.g. --label Bf_U3 for the pre-U3 runs — '
                        'otherwise the second bridge overwrites the first.')
    p.add_argument('--n-contexts', type=int, default=None,
                   help='Pin n_contexts instead of reading/inferring it. Use for a '
                        'killed run with no results_seed_*.json, where inference '
                        'reports only the contexts that were reached.')
    p.add_argument('--n-trajectories', dest='n_trajs', type=int, default=None,
                   help='Pin n_trajectories_per_context (see --n-contexts)')
    p.add_argument('--force', action='store_true',
                   help='Overwrite a unit already bridged from a different source '
                        'directory (default: skip it and tell you to use --label)')
    p.add_argument('--json-only', action='store_true',
                   help='Skip the npz (no numpy needed). DA_VA_v2 then reads the '
                        'diagnostics JSONs for these units.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_root = args.source_root
    if not os.path.isdir(source_root):
        print(f'[ FATAL ] source root not found: {source_root}')
        return 1

    out_root = args.out_root or os.path.join(source_root, api.DA_BRIDGE_ROOT_NAME)
    wanted = None
    if args.seeds:
        wanted = {int(s) for s in args.seeds.split(',') if s.strip()}

    agents = find_agent_dirs(source_root, args.agent_name)
    if not agents:
        print(f'[ FATAL ] no legacy agent folders under {source_root}'
              + (f' matching --agent-name {args.agent_name}' if args.agent_name else ''))
        return 1

    print(f'[ bridge ] source: {os.path.abspath(source_root)}')
    print(f'[ bridge ] out:    {os.path.abspath(out_root)}')
    print(f'[ bridge ] agents: {[a for a, _ in agents]}')

    bridged = []
    for agent_name, agent_dir in agents:
        for seed, seed_path in find_seed_dirs(agent_dir):
            if wanted is not None and seed not in wanted:
                continue
            entry = bridge_seed(out_root, agent_name, seed, seed_path,
                                write_npz=not args.json_only, split=args.split,
                                label=args.label, force_contexts=args.n_contexts,
                                force_trajs=args.n_trajs, force=args.force)
            if entry:
                bridged.append(entry)

    if not bridged:
        print('[ FATAL ] nothing bridged — no seed had rollout stats JSONs.')
        return 1

    api.write_bridge_manifest(out_root, {
        'source_root': os.path.abspath(source_root),
        'bridge_script': os.path.basename(__file__),
        'npz_written': not args.json_only,
        'label': args.label,
        'units': bridged,
        'known_legacy_issues': [
            'entropy in results_seed_*.json is always 0.0 (U4.1 bug B1) — '
            'recomputed here from per-rollout modes',
            'avg_time exists only in realtime_baseline_*.log — parsed here, '
            'NaN where the log is missing',
            'no projector: every constraint_* metric is absent by design',
        ],
    })

    print(f'\n[ done ] {len(bridged)} unit(s) bridged into {out_root}')
    print('[ next ] python Data_Analysis/DA_VA_v2/main_da_batch.py '
          f'--parent-path {out_root}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
