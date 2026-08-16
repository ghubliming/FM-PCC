"""
DA_VA_v2 — candidate + result-unit discovery.

Deliberately dependency-free (stdlib only, no numpy/pandas) so the path-shape
logic can be exercised on a machine without the science stack.

Two levels:

  discover_candidates()   parent tree  → candidate folders (a folder holding
                                          numeric seed subdirs that contain a
                                          results root)
  discover_units()        candidate    → one ResultUnit per
                                          (seed, split, geo, variant) npz

Path shapes handled (all four seen across Gen7 / Gen14 / state-only):

  A  {seed}/results/{variant}.npz                      state-only flat
  B  {seed}/results/halfspace_{geo}/{variant}.npz      state-only avoiding (DA_Code_v3)
  C  {seed}/results/{variant}/{variant}.npz            Gen7 flat visual-aligning
  D  {seed}/results[_train_set]/{geo}/{variant}/{variant}.npz   Gen7+geo and Gen14

Variant names get the `_train_set` suffix stripped (Gen14 spec §4 L4); the split
is preserved as its own column instead, so a train-set run and a test run of the
same variant line up in the tables rather than masquerading as two variants.

Legacy trees: a candidate sitting under a `_`-prefixed folder (or under a
`_bridge_manifest.json`) is flagged `legacy=True` and the flag is carried on every
unit, so the loader can apply the per-pipeline rescues an old layout needs. See
`detect_legacy()` and `config.LEGACY_DIR_PREFIX`.
"""

import json
import logging
import os
import re

from config import (
    BRIDGE_MANIFEST_NAME,
    GEO_DIR_PREFIXES,
    GEO_NONE,
    LEGACY_DIR_PREFIX,
    NON_VARIANT_DIRS,
    RESULTS_DIR_NAMES,
    SKIP_NPZ_SUFFIXES,
    SNAPSHOT_DIR_PREFIX,
    TIGHTENED_SUFFIX,
    TRAIN_SET_SUFFIX,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_FILE_RE = re.compile(r'^snapshot_(\d{8}_\d{6})$')


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def split_of(results_dir_name):
    """`results` → 'test', `results_train_set` → 'train'."""
    return 'train' if results_dir_name.endswith(TRAIN_SET_SUFFIX) else 'test'


def clean_variant(raw_name):
    """Strip the `--eval-on-train` suffix from a variant name."""
    if raw_name.endswith(TRAIN_SET_SUFFIX):
        return raw_name[: -len(TRAIN_SET_SUFFIX)]
    return raw_name


def clean_geo(raw_dir):
    """Normalise a geometry directory name.

    `halfspace_top-right-hard` → `top-right-hard` so the state-only trees keep
    the same geo labels DA_Code_v3 produced; everything else is verbatim.
    """
    for prefix in GEO_DIR_PREFIXES:
        if raw_dir.startswith(prefix):
            return raw_dir[len(prefix):]
    return raw_dir


def geo_parts(geo):
    """`combined_5-tightened` → ('combined_5', True)."""
    if geo.endswith(TIGHTENED_SUFFIX):
        return geo[: -len(TIGHTENED_SUFFIX)], True
    return geo, False


def _seed_dirs(candidate_path):
    """Numeric subdirectories of a candidate folder that hold a results root."""
    out = []
    try:
        entries = sorted(os.listdir(candidate_path))
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        seed_path = os.path.join(candidate_path, entry)
        if not os.path.isdir(seed_path):
            continue
        if any(os.path.isdir(os.path.join(seed_path, r)) for r in RESULTS_DIR_NAMES):
            out.append(int(entry))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# legacy (bridged) trees
# ──────────────────────────────────────────────────────────────────────────────

def detect_legacy(candidate_path, parent_root=None):
    """Is this candidate a bridged legacy tree, and what do we know about it?

    Two independent signals, either is enough:

      * a `_`-prefixed directory anywhere between the parent root and the
        candidate (the naming contract — see config.LEGACY_DIR_PREFIX);
      * a `_bridge_manifest.json` at the candidate or any ancestor up to the
        parent root (the bridge's own marker, which also carries `legacy_kind`
        and `has_projector`).

    Returns {legacy, legacy_kind, has_projector, bridge_manifest_path, manifest}.
    A tree written by a normal Gen14-shaped eval returns legacy=False and is
    then read with no special handling at all.
    """
    candidate_path = os.path.abspath(candidate_path)
    root = os.path.abspath(parent_root) if parent_root else None

    manifest, manifest_path = {}, ''
    node = candidate_path
    while True:
        probe = os.path.join(node, BRIDGE_MANIFEST_NAME)
        if os.path.isfile(probe):
            manifest_path = probe
            try:
                with open(probe) as f:
                    manifest = json.load(f) or {}
            except (OSError, ValueError) as exc:
                logger.warning(f'Unreadable bridge manifest {probe}: {exc}')
            break
        parent = os.path.dirname(node)
        if node == root or parent == node:
            break
        node = parent

    names = [os.path.basename(candidate_path)]
    if root:
        names.append(os.path.basename(root))
        rel = os.path.relpath(candidate_path, root)
        names.extend(part for part in rel.split(os.sep) if part not in ('.', '..'))
    underscored = any(name.startswith(LEGACY_DIR_PREFIX) for name in names if name)

    legacy = bool(manifest) or underscored
    return {
        'legacy': legacy,
        'legacy_kind': manifest.get('legacy_kind', 'unknown' if legacy else ''),
        # Absent manifest + legacy naming: assume a projector until told
        # otherwise, so nothing silently claims constraint-freedom.
        'has_projector': bool(manifest.get('has_projector', True)) if manifest else True,
        'bridge_manifest_path': manifest_path,
        'manifest': manifest,
    }


# ──────────────────────────────────────────────────────────────────────────────
# config-snapshot timestamps — "when was this folder last run?"
# ──────────────────────────────────────────────────────────────────────────────

def _snapshot_stamps_in_seed(seed_dir):
    """Every `snapshot_<ts>` stamp under `<seed_dir>/config_snapshot*/`."""
    stamps = []
    try:
        entries = sorted(os.listdir(seed_dir))
    except OSError:
        return stamps
    for entry in entries:
        if not entry.startswith(SNAPSHOT_DIR_PREFIX):
            continue
        snapshot_dir = os.path.join(seed_dir, entry)
        if not os.path.isdir(snapshot_dir):
            continue
        try:
            names = os.listdir(snapshot_dir)
        except OSError:
            continue
        for name in names:
            match = _SNAPSHOT_FILE_RE.match(name)
            if match:
                stamps.append(match.group(1))
    return stamps


def scan_snapshot_timestamps(candidate_path, seeds=None):
    """Collect the config-snapshot timestamps of one candidate folder.

    Args:
        candidate_path: candidate folder (the one holding the seed subdirs).
        seeds: seeds to inspect (None = every numeric subdirectory).

    Returns:
        latest           newest stamp over all seeds, `YYYYMMDD_HHMMSS` ('' if none)
        first            oldest stamp over all seeds ('' if none)
        count            total stamp files found
        per_seed         {seed:int -> newest stamp of that seed}
        n_seeds_stamped  how many seeds carried at least one stamp

    Never raises: a tree written before config snapshots existed (or assembled
    by hand) just reports empty strings and count 0.
    """
    if seeds is None:
        try:
            seeds = sorted(int(entry) for entry in os.listdir(candidate_path)
                           if entry.isdigit()
                           and os.path.isdir(os.path.join(candidate_path, entry)))
        except OSError:
            seeds = []

    per_seed = {}
    all_stamps = []
    for seed in seeds:
        stamps = _snapshot_stamps_in_seed(os.path.join(candidate_path, str(seed)))
        if not stamps:
            continue
        all_stamps.extend(stamps)
        per_seed[int(seed)] = max(stamps)

    return {
        'latest': max(all_stamps) if all_stamps else '',
        'first': min(all_stamps) if all_stamps else '',
        'count': len(all_stamps),
        'per_seed': per_seed,
        'n_seeds_stamped': len(per_seed),
    }


def format_snapshot_ts(stamp):
    """'20260506_034806' -> '2026-05-06 03:48:06'. Anything else passes through."""
    text = str(stamp or '').strip()
    if len(text) == 15 and text[8] == '_' and text.replace('_', '').isdigit():
        return (f'{text[0:4]}-{text[4:6]}-{text[6:8]} '
                f'{text[9:11]}:{text[11:13]}:{text[13:15]}')
    return text


def snapshot_by_seed_str(per_seed):
    """'6:20260506_034806 | 7:20260506_041233' — the per-seed audit trail.

    Worth its own column: the seeds of one candidate are usually launched by
    separate jobs, so a single stale seed hides behind a fresh `latest`.
    """
    if not per_seed:
        return ''
    return ' | '.join(f'{seed}:{per_seed[seed]}' for seed in sorted(per_seed))


# ──────────────────────────────────────────────────────────────────────────────
# candidates
# ──────────────────────────────────────────────────────────────────────────────

def discover_candidates(parent_paths, seed_list=None, max_depth=10):
    """Find every candidate folder under one or more parent trees.

    A candidate is a folder with at least one numeric seed subdir that itself
    contains `results/` or `results_train_set/`. Search does not descend into a
    folder once it is recognised as a candidate.

    Args:
        parent_paths: path string, comma-separated string, or list of paths.
        seed_list:    restrict to these seeds (None = every seed found).
        max_depth:    how deep below each parent to look.

    Returns:
        {index (1-based int): {path, name, parent, seeds, missing_seeds}}
    """
    if isinstance(parent_paths, str):
        parent_paths = [p.strip() for p in parent_paths.split(',') if p.strip()]

    found = []

    def _walk(path, depth, parent_root):
        if depth > max_depth or not os.path.isdir(path):
            return
        seeds = _seed_dirs(path)
        if seed_list is not None:
            seeds = [s for s in seeds if s in seed_list]
        if seeds:
            entry = {
                'path': os.path.abspath(path),
                'name': os.path.basename(os.path.normpath(path)),
                'parent': parent_root,
                'seeds': sorted(seeds),
                'missing_seeds': ([s for s in seed_list if s not in seeds]
                                  if seed_list else []),
                'snapshots': scan_snapshot_timestamps(path, seeds),
            }
            entry.update(detect_legacy(path, parent_root))
            found.append(entry)
            return   # a candidate is a leaf — do not recurse into its seeds
        try:
            entries = sorted(os.listdir(path))
        except OSError:
            return
        for entry in entries:
            if entry.startswith('.'):
                continue
            sub = os.path.join(path, entry)
            if os.path.isdir(sub):
                _walk(sub, depth + 1, parent_root)

    for parent in parent_paths:
        if not os.path.isdir(parent):
            logger.warning(f'Parent path does not exist, skipped: {parent}')
            continue
        _walk(parent, 1, os.path.abspath(parent))

    # Stable ordering across runs and across multi-tree merges.
    found.sort(key=lambda c: c['path'])
    candidates = {i + 1: info for i, info in enumerate(found)}

    for idx, info in candidates.items():
        tag = f'  [legacy: {info.get("legacy_kind") or "unknown"}]' if info.get('legacy') else ''
        logger.info(f'Candidate {idx}: {info["name"]}  seeds={info["seeds"]}{tag}')
        if info['missing_seeds']:
            logger.warning(f'Candidate {idx} ({info["name"]}) missing seeds: '
                           f'{info["missing_seeds"]}')
    if not candidates:
        logger.warning(f'No candidates found under: {parent_paths}')
    else:
        logger.info(f'Total candidates discovered: {len(candidates)}')
    return candidates


def filter_candidates(candidates, selected):
    """Keep only the listed candidate indices ("1,3,5" or [1, 3, 5])."""
    if isinstance(selected, str):
        wanted = {s.strip() for s in selected.split(',') if s.strip()}
    else:
        wanted = {str(s) for s in selected}
    filtered = {k: v for k, v in candidates.items() if str(k) in wanted}
    logger.info(f'Filtered candidates: {sorted(filtered.keys())}')
    return filtered


def assign_custom_names(candidates, custom_names):
    """Attach display names ("fm,mf,af,diffusion") in candidate index order."""
    if isinstance(custom_names, str):
        names = [n.strip() for n in custom_names.split(',')]
    else:
        names = list(custom_names)
    keys = sorted(candidates.keys())
    if len(names) != len(keys):
        logger.warning(f'{len(names)} names for {len(keys)} candidates — '
                       f'keeping auto names.')
        return candidates
    out = {}
    for key, name in zip(keys, names):
        out[key] = dict(candidates[key])
        out[key]['custom_name'] = name
    logger.info(f'Applied custom names: {names}')
    return out


def get_candidate_summary(candidates):
    lines = ['=== Candidates Discovered ===', f'Total: {len(candidates)}', '']
    for key in sorted(candidates.keys()):
        info = candidates[key]
        lines.append(f'  {key}: {info["name"]}')
        lines.append(f'      Path:  {info["path"]}')
        lines.append(f'      Seeds: {info["seeds"]}')
        snapshots = info.get('snapshots') or {}
        if snapshots.get('latest'):
            lines.append(f'      Last run: {format_snapshot_ts(snapshots["latest"])}'
                         f'  ({snapshots["count"]} config snapshot(s) over '
                         f'{snapshots["n_seeds_stamped"]} seed(s))')
        if info.get('legacy'):
            lines.append(f'      LEGACY (bridged): kind={info.get("legacy_kind") or "unknown"}'
                         f'  projector={"yes" if info.get("has_projector", True) else "no"}')
        if info.get('missing_seeds'):
            lines.append(f'      WARNING missing seeds: {info["missing_seeds"]}')
        if info.get('custom_name'):
            lines.append(f'      Display: {info["custom_name"]}')
        lines.append('')
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# result units
# ──────────────────────────────────────────────────────────────────────────────

def parse_unit_path(results_root, npz_path):
    """Map an npz path under a results root onto (geo_raw, variant_raw).

    Returns None when the path is not a variant result (expert references etc.).
    """
    rel_dir = os.path.relpath(os.path.dirname(npz_path), results_root)
    stem = os.path.basename(npz_path)[: -len('.npz')]

    if rel_dir in ('.', ''):
        return GEO_NONE, stem                                    # shape A

    parts = [p for p in rel_dir.split(os.sep) if p not in ('.',)]
    if any(p in NON_VARIANT_DIRS for p in parts):
        return None

    if len(parts) == 1:
        # `{variant}/{variant}.npz` (shape C) vs `{geo}/{variant}.npz` (shape B).
        if clean_variant(parts[0]) == clean_variant(stem):
            return GEO_NONE, stem
        return parts[0], stem

    # shape D — the last directory is the variant folder, everything above is geo.
    return os.sep.join(parts[:-1]).replace(os.sep, '/'), stem


def discover_units(candidate_idx, candidate_info, seeds=None, splits=None,
                   geos=None, variants=None):
    """Enumerate every result unit of one candidate.

    Args:
        candidate_idx:  1-based candidate key.
        candidate_info: entry from discover_candidates().
        seeds/splits/geos/variants: optional whitelists (cleaned names).

    Returns:
        list of unit dicts. `npz_path` is None when only diagnostics JSONs exist
        (pre-npz Gen7 runs) — the loader falls back to the JSON source there.
    """
    units = []
    root = candidate_info['path']
    use_seeds = candidate_info['seeds'] if seeds is None else \
        [s for s in candidate_info['seeds'] if s in seeds]

    for seed in use_seeds:
        for results_name in RESULTS_DIR_NAMES:
            results_root = os.path.join(root, str(seed), results_name)
            if not os.path.isdir(results_root):
                continue
            split = split_of(results_name)
            if splits and split not in splits:
                continue

            seen_dirs = set()

            for dirpath, dirnames, filenames in os.walk(results_root):
                dirnames[:] = [d for d in sorted(dirnames)
                               if d not in NON_VARIANT_DIRS and not d.startswith('.')]
                for filename in sorted(filenames):
                    if not filename.endswith('.npz'):
                        continue
                    if any(filename.endswith(s) for s in SKIP_NPZ_SUFFIXES):
                        continue
                    npz_path = os.path.join(dirpath, filename)
                    parsed = parse_unit_path(results_root, npz_path)
                    if parsed is None:
                        continue
                    geo_raw, variant_raw = parsed
                    unit = _make_unit(candidate_idx, candidate_info, seed, split,
                                      geo_raw, variant_raw, npz_path,
                                      os.path.dirname(npz_path))
                    if _keep(unit, geos, variants):
                        units.append(unit)
                        seen_dirs.add(os.path.dirname(npz_path))

            # JSON-only variants: a diagnostics folder whose sibling npz is absent
            # (pre-npz Gen7 runs, or a variant killed before the final write).
            for dirpath, dirnames, _ in os.walk(results_root):
                dirnames[:] = [d for d in sorted(dirnames) if not d.startswith('.')]
                if os.path.basename(dirpath) != 'diagnostics':
                    continue
                variant_dir = os.path.dirname(dirpath)
                if variant_dir in seen_dirs:
                    continue
                variant_raw = os.path.basename(variant_dir)
                pseudo_npz = os.path.join(variant_dir, f'{variant_raw}.npz')
                parsed = parse_unit_path(results_root, pseudo_npz)
                if parsed is None:
                    continue
                geo_raw, variant_raw = parsed
                unit = _make_unit(candidate_idx, candidate_info, seed, split,
                                  geo_raw, variant_raw, None, variant_dir)
                if _keep(unit, geos, variants):
                    units.append(unit)

    units.sort(key=lambda u: (u['seed'], u['split'], u['geo'], u['variant']))
    return units


def _make_unit(candidate_idx, candidate_info, seed, split, geo_raw, variant_raw,
               npz_path, variant_dir):
    geo = clean_geo(geo_raw)
    geo_base, tightened = geo_parts(geo)
    return {
        'Candidate': candidate_idx,
        'FolderName': candidate_info.get('custom_name') or candidate_info['name'],
        'RawFolderName': candidate_info['name'],
        'FullPath': candidate_info['path'],
        'seed': seed,
        'split': split,
        'geo': geo,
        'geo_raw': geo_raw,
        'geo_base': geo_base,
        'tightened': bool(tightened),
        'variant': clean_variant(variant_raw),
        'variant_raw': variant_raw,
        'npz_path': npz_path,
        'variant_dir': variant_dir,
        'diagnostics_dir': os.path.join(variant_dir, 'diagnostics'),
        'constraint_metrics_json': os.path.join(variant_dir, 'constraint_metrics.json'),
        # Legacy flags ride on the unit so the loader never has to re-stat the tree.
        'legacy': bool(candidate_info.get('legacy', False)),
        'legacy_kind': candidate_info.get('legacy_kind', ''),
        'has_projector': bool(candidate_info.get('has_projector', True)),
    }


def _keep(unit, geos, variants):
    if geos and unit['geo'] not in geos:
        return False
    if variants and unit['variant'] not in variants:
        return False
    return True


def write_discovery_manifest(path, candidates, units):
    """Dump what discovery saw — the first thing to read when a run looks short."""
    payload = {
        'n_candidates': len(candidates),
        'n_units': len(units),
        'candidates': {
            str(k): {
                'name': v['name'],
                'display_name': v.get('custom_name') or v['name'],
                'path': v['path'],
                'seeds': v['seeds'],
                'missing_seeds': v.get('missing_seeds', []),
                'snapshots': v.get('snapshots', {}),
                'legacy': bool(v.get('legacy', False)),
                'legacy_kind': v.get('legacy_kind', ''),
                'has_projector': bool(v.get('has_projector', True)),
                'bridge_manifest': v.get('bridge_manifest_path', ''),
            } for k, v in sorted(candidates.items())
        },
        'units': [
            {
                'Candidate': u['Candidate'], 'seed': u['seed'], 'split': u['split'],
                'geo': u['geo'], 'variant': u['variant'],
                'source': 'npz' if u['npz_path'] else 'json',
                'legacy': bool(u.get('legacy', False)),
            } for u in units
        ],
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    logger.info(f'Discovery manifest written: {path}')


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if len(sys.argv) < 2:
        print('Usage: python discovery.py <parent_path>[,<parent_path>...]')
        raise SystemExit(1)
    cands = discover_candidates(sys.argv[1])
    print(get_candidate_summary(cands))
    total = 0
    for idx, info in sorted(cands.items()):
        found = discover_units(idx, info)
        total += len(found)
        print(f'Candidate {idx}: {len(found)} units')
        for unit in found[:5]:
            print(f'   seed={unit["seed"]} split={unit["split"]} '
                  f'geo={unit["geo"]} variant={unit["variant"]} '
                  f'src={"npz" if unit["npz_path"] else "json"}')
        if len(found) > 5:
            print(f'   ... +{len(found) - 5} more')
    print(f'TOTAL units: {total}')
