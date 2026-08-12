"""
D3IL Visual-Aligning baseline → DA_VA_v2 export schema (U4).

ONE definition of the on-disk contract, used by two producers:

  * `eval_d3il_visual_aligning.py`  — native export, written while the eval runs
                                      (root `DA_VA_d3il_baseline/`, no underscore)
  * `bridge_d3il_va_to_da_va_v2.py` — legacy bridge, re-reads finished runs from
                                      the old layout (root `_DA_VA_BRIDGE_…/`,
                                      leading underscore = "legacy, read me with
                                      the legacy path")

Both emit exactly the same files; the only difference is the root folder name and
`args['export_source']`. That is the whole point: once every future run exports
natively, DA_VA_v2 reads D3IL exactly like it reads Gen14, and the bridge stays
only for data already on disk.

Layout written (Gen14 spec §1 "shape C" — no geometry level, because the D3IL
baseline has no projector and therefore no constraint geometry):

    <root>/
      _bridge_manifest.json                     (bridge root only)
      d3il_baseline_{agent_name}/               ← DA_VA_v2 "candidate"
        {seed}/
          config_snapshot_d3il_baseline/
            snapshot_{YYYYMMDD_HHMMSS}          ← "when was this run" marker
          results[_train_set]/
            d3il_baseline/                      ← DA_VA_v2 "variant"
              d3il_baseline.npz                 ← source of truth
              unit_meta.json                    ← provenance, human-readable
              diagnostics/
                rollout_{r}_stats.json          ← Gen14 nested schema

What is deliberately NOT written: every `constraint_*` key. The D3IL baseline
runs no projector, so those metrics do not exist. Absent ⇒ NaN in the DA tables,
which is the honest answer; writing 0 would make the baseline look perfectly
constraint-satisfying and would poison `n_success_and_constraints`.

Timing semantics (read this before comparing `avg_time` to Gen14): D3IL calls
`agent.predict()` on EVERY control step and the action-chunk is served from an
internal buffer, so `avg_time` here is **mean seconds per control step**, not per
model replan. Gen14's `avg_time` is per replan. `args['timing_semantics']` records
which one the file holds.

Stdlib only at import time — numpy is imported lazily inside `write_unit()` so the
JSON half of this module can run in the AI container (no scientific stack).
"""

import json
import math
import os
from datetime import datetime

SCHEMA_VERSION = '1.0'

# ── names DA_VA_v2 will see ───────────────────────────────────────────────────
DA_VARIANT_NAME = 'd3il_baseline'
DA_CANDIDATE_PREFIX = 'd3il_baseline_'
DA_SNAPSHOT_DIR = 'config_snapshot_d3il_baseline'
DA_NATIVE_ROOT_NAME = 'DA_VA_d3il_baseline'
DA_BRIDGE_ROOT_NAME = '_DA_VA_BRIDGE_d3il_baseline'
BRIDGE_MANIFEST_NAME = '_bridge_manifest.json'

TIMING_SEMANTICS = 'seconds_per_control_step'


# ──────────────────────────────────────────────────────────────────────────────
# paths
# ──────────────────────────────────────────────────────────────────────────────

def candidate_name(agent_name, label=''):
    """`d3il_baseline_{agent}`, or `d3il_baseline_{agent}__{label}` when labelled.

    The label exists because the same agent name is re-used across run
    generations (`logs/d3il_visual_aligning_baseline/` and
    `logs/d3il_visual_aligning_baseline(Bf_U3)/` both hold a
    `ddpm_encdec_vision/seed_42`). Bridging both without a label would have the
    second silently overwrite the first, since the candidate/seed path is
    identical. With a label they land side by side as two DA candidates.
    """
    base = f'{DA_CANDIDATE_PREFIX}{agent_name}'
    return f'{base}__{label}' if label else base


def results_dir_name(split='test'):
    return 'results' if split == 'test' else 'results_train_set'


def unit_dir(root, agent_name, seed, split='test', label=''):
    """`<root>/d3il_baseline_{agent}[__{label}]/{seed}/results[_train_set]/d3il_baseline/`."""
    return os.path.join(root, candidate_name(agent_name, label), str(seed),
                        results_dir_name(split), DA_VARIANT_NAME)


def seed_dir(root, agent_name, seed, label=''):
    return os.path.join(root, candidate_name(agent_name, label), str(seed))


# ──────────────────────────────────────────────────────────────────────────────
# the canonical per-rollout record
# ──────────────────────────────────────────────────────────────────────────────

def make_record(rollout_index, context_idx, traj_idx, success, steps,
                mean_distance, mode, context_info, avg_time_s=None):
    """The one record shape both producers build.

    `context_info` is the D3IL eval's own dict: box_init_xy, box_init_angle_deg,
    target_xy, target_angle_deg, init_xy_dist and (after the rollout)
    final_box_xy, final_box_angle_deg, final_xy_dist.
    """
    return {
        'rollout_index': int(rollout_index),
        'context_idx': int(context_idx),
        'traj_idx': int(traj_idx),
        'success': bool(success),
        'steps': int(steps),
        'mean_distance': _f(mean_distance),
        'mode': int(mode) if mode is not None else -1,
        'avg_time_s': _f(avg_time_s),
        'context_info': dict(context_info or {}),
    }


def rollout_stats_json(record):
    """One `diagnostics/rollout_{r}_stats.json` in the Gen14 NESTED schema.

    Key paths are chosen to match `DA_VA_v2/data_loader._load_json` exactly:
    `success.strict`, `outcome.mean_distance`, `timing.steps`,
    `timing.avg_inference_time_per_replan`, `context.*`, top-level `mode`.
    """
    ci = dict(record.get('context_info') or {})
    ci.setdefault('context_idx', record['context_idx'])
    ci['traj_idx'] = record['traj_idx']
    return {
        'rollout_index': record['rollout_index'],
        'mode': record['mode'],
        'success': {
            'strict': 1 if record['success'] else 0,
            # No relaxed criterion exists for the D3IL baseline: the env latches
            # success on the strict (pos_min_dist, rot_min_dist) gate only.
            'relaxed': None,
        },
        'outcome': {
            'mean_distance': _json_num(record['mean_distance']),
            'mode': record['mode'],
        },
        'timing': {
            'steps': record['steps'],
            'avg_inference_time_per_replan': _json_num(record['avg_time_s']),
            'timing_semantics': TIMING_SEMANTICS,
        },
        'context': ci,
        # Explicit, not merely absent: this pipeline has no projector at all.
        'constraint': None,
        'projector': 'none',
    }


# ──────────────────────────────────────────────────────────────────────────────
# metrics
# ──────────────────────────────────────────────────────────────────────────────

def behavior_entropy(records, n_contexts, n_trajs, n_modes=2):
    """Conditional behavior entropy — D3IL paper Eq. 2 / `aligning_sim.py:226-242`.

    Per context, count the modes of the SUCCESSFUL rollouts, divide by n_trajs,
    row-normalise, take the base-|B| entropy, average over contexts.

    Contexts are read from `record['context_idx']` — the U4.1 audit found the eval's
    own copy of this function looking for a `context` key its records never carried,
    which pinned entropy to exactly 0.0 for every run ever made.
    """
    if n_contexts <= 0 or n_trajs <= 0:
        return 0.0
    counts = [[0.0] * n_modes for _ in range(n_contexts)]
    for rec in records:
        ctx = int(rec.get('context_idx', -1))
        if not 0 <= ctx < n_contexts:
            continue
        if not rec.get('success'):
            continue                      # success-conditioned, like the paper
        mode = int(rec.get('mode', -1))
        if 0 <= mode < n_modes:
            counts[ctx][mode] += 1.0

    total = 0.0
    for row in counts:
        probs = [c / float(n_trajs) for c in row]
        norm = sum(probs) + 1e-12
        probs = [p / norm for p in probs]
        total += -sum(p * math.log(p + 1e-12) / math.log(n_modes) for p in probs)
    return float(total / n_contexts)


def summarise(records, n_contexts, n_trajs):
    """Run-level scalars, computed the same way on both producers."""
    n = len(records)
    successes = [1.0 if r['success'] else 0.0 for r in records]
    success_rate = (sum(successes) / n) if n else float('nan')
    entropy = behavior_entropy(records, n_contexts, n_trajs)
    return {
        'success_rate': success_rate,
        'entropy': entropy,
        'score': 0.5 * (success_rate + entropy) if n else float('nan'),
        'n_rollouts': n,
        'n_contexts': int(n_contexts),
        'n_trajectories_per_context': int(n_trajs),
    }


# ──────────────────────────────────────────────────────────────────────────────
# npz assembly
# ──────────────────────────────────────────────────────────────────────────────

# Per-rollout arrays. Names are Gen14's (spec §2) so DA_VA_v2 picks them up with
# no per-pipeline special-casing; `_classify()` turns each into one column.
def build_arrays(records):
    """{npz key -> list} — plain Python, so this is testable without numpy."""
    def ctx(rec, field, default=float('nan')):
        return _f(rec.get('context_info', {}).get(field, default))

    def xy(rec, field):
        value = rec.get('context_info', {}).get(field)
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, IndexError, ValueError):
            return [float('nan'), float('nan')]

    return {
        'n_success': [1.0 if r['success'] else 0.0 for r in records],
        'success_strict': [1.0 if r['success'] else 0.0 for r in records],
        'n_steps': [float(r['steps']) for r in records],
        'avg_time': [_f(r['avg_time_s']) for r in records],
        'mean_dist_per_rollout': [_f(r['mean_distance']) for r in records],
        'mode_encoding': [[float(r['mode'])] for r in records],
        'context_index': [float(r['context_idx']) for r in records],
        'context_box_init_xy': [xy(r, 'box_init_xy') for r in records],
        'context_target_xy': [xy(r, 'target_xy') for r in records],
        'context_final_box_xy': [xy(r, 'final_box_xy') for r in records],
        'context_box_angle_deg': [ctx(r, 'box_init_angle_deg') for r in records],
        'context_target_angle_deg': [ctx(r, 'target_angle_deg') for r in records],
        'context_init_xy_dist': [ctx(r, 'init_xy_dist') for r in records],
        'context_final_xy_dist': [ctx(r, 'final_xy_dist') for r in records],
        'context_final_box_angle_deg': [ctx(r, 'final_box_angle_deg') for r in records],
    }


def build_args(agent_name, seed, split, extra=None):
    """The `args` object DA_VA_v2 lifts into run_config.csv."""
    args = {
        'engine': 'd3il_baseline',
        'agent_name': agent_name,
        'seed': int(seed),
        'split': split,
        'if_vision': True,
        'projection_variant': 'none',
        'projector': 'none',
        'schema_version': SCHEMA_VERSION,
        'timing_semantics': TIMING_SEMANTICS,
        'pipeline': 'd3il_visual_aligning_baseline',
    }
    args.update(extra or {})
    return args


# ──────────────────────────────────────────────────────────────────────────────
# writing
# ──────────────────────────────────────────────────────────────────────────────

def require_numpy():
    """Import numpy, or raise with the actual fix.

    Called as a PREFLIGHT before anything is written: the npz step is the only
    numpy-dependent one, and discovering that at the end of a 1080-rollout unit
    leaves a half-written folder behind.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            'numpy is required to write the npz. On the cluster the base conda '
            'env does not have it:\n'
            '    source ~/miniconda3/etc/profile.d/conda.sh && conda activate FMPCC\n'
            'Or run with --json-only to skip the npz entirely (DA_VA_v2 then '
            'reads the diagnostics JSONs for these units).') from exc
    return np


def write_unit(root, agent_name, seed, records, scalars, args_extra=None,
               split='test', write_npz=True, snapshot_stamp=None, label=''):
    """Write one complete DA_VA_v2 unit. Returns the unit directory.

    `write_npz=False` skips the only numpy-dependent step, which keeps the JSON
    half runnable in the AI-coding container; DA_VA_v2 then falls back to its
    diagnostics-JSON reader for that unit.

    `label` disambiguates two runs of the same agent+seed (see candidate_name).
    """
    # Preflight: fail before writing anything rather than after N rollout JSONs.
    if write_npz:
        require_numpy()

    out_dir = unit_dir(root, agent_name, seed, split, label)
    diag_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    for record in records:
        path = os.path.join(diag_dir, f'rollout_{record["rollout_index"]}_stats.json')
        with open(path, 'w') as f:
            json.dump(json_safe(rollout_stats_json(record)), f, indent=2)

    args = build_args(agent_name, seed, split, args_extra)
    meta = {
        'schema_version': SCHEMA_VERSION,
        'variant': DA_VARIANT_NAME,
        'candidate': candidate_name(agent_name, label),
        'label': label,
        'agent_name': agent_name,
        'seed': int(seed),
        'split': split,
        'written_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scalars': scalars,
        'args': args,
        'npz_written': bool(write_npz),
        'note': ('No projector in this pipeline: every constraint_* metric is '
                 'intentionally absent (NaN in DA_VA_v2), never zero.'),
    }

    # npz BEFORE the metadata: unit_meta.json is the human-readable record of what
    # this folder contains, so it must never claim an npz that failed to write.
    if write_npz:
        _write_npz(os.path.join(out_dir, f'{DA_VARIANT_NAME}.npz'),
                   records, scalars, args)

    with open(os.path.join(out_dir, 'unit_meta.json'), 'w') as f:
        json.dump(json_safe(meta), f, indent=2)

    write_snapshot_marker(root, agent_name, seed, snapshot_stamp, label)
    return out_dir


def _write_npz(path, records, scalars, args):
    np = require_numpy()                    # lazy: only the npz step needs it

    payload = {key: np.asarray(value, dtype=float)
               for key, value in build_arrays(records).items()}
    for key, value in scalars.items():
        payload[key] = np.asarray(value, dtype=float)
    payload['seed'] = np.asarray(float(args.get('seed', -1)))
    payload['complete'] = np.asarray(1.0)
    payload['args'] = np.asarray(args, dtype=object)
    np.savez_compressed(path, **payload)


def write_snapshot_marker(root, agent_name, seed, stamp=None, label=''):
    """`{seed}/config_snapshot_d3il_baseline/snapshot_<YYYYMMDD_HHMMSS>`.

    DA_VA_v2's `scan_snapshot_timestamps()` reads these to answer "when was this
    folder last run" in every report; without one the candidate shows a blank
    Last-Run column.
    """
    stamp = stamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    snap_dir = os.path.join(seed_dir(root, agent_name, seed, label), DA_SNAPSHOT_DIR)
    os.makedirs(snap_dir, exist_ok=True)
    marker = os.path.join(snap_dir, f'snapshot_{stamp}')
    if not os.path.exists(marker):
        with open(marker, 'w') as f:
            f.write(f'd3il_visual_aligning_baseline  agent={agent_name}  '
                    f'seed={seed}  schema={SCHEMA_VERSION}\n')
    return marker


def write_bridge_manifest(root, payload):
    """Root-level marker that tells DA_VA_v2 "this tree was bridged, not run"."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, BRIDGE_MANIFEST_NAME)
    body = {
        'schema_version': SCHEMA_VERSION,
        'legacy_kind': 'd3il_visual_aligning_baseline',
        'variant': DA_VARIANT_NAME,
        'has_projector': False,
        'written_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    body.update(payload or {})
    with open(path, 'w') as f:
        json.dump(json_safe(body), f, indent=2)
    return path


def _f(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def _json_num(value):
    """NaN/inf → None, so the JSON stays valid for strict parsers (JS viewers).

    `null` is what the consumers already treat as "absent" — `data_loader._dig`
    returns its default for None — so nothing downstream changes.
    """
    number = _f(value)
    return number if math.isfinite(number) else None


def json_safe(payload):
    """Recursively replace non-finite floats with None."""
    if isinstance(payload, dict):
        return {k: json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(v) for v in payload]
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    return payload
