"""
DA_UAV_v1 — candidate + result-unit discovery, and the path-encoded axis parser.

Deliberately dependency-free (stdlib only, no numpy/pandas) so the path-shape
logic can be exercised in the AI-coding container, which has no science stack.

    python Data_Analysis/DA_UAV_v1/discovery.py logs/UAV_MIX

Three levels:

  discover_candidates()   parent tree  → candidate folders (a folder holding
                                          numeric seed subdirs that contain
                                          result data)
  discover_units()        candidate    → one ResultUnit per
                                          (seed, split, geo, variant) npz
  parse_axes()            candidate path → scene / engine / K / controller / …

Path shapes handled
-------------------

  U  {seed}/{geo_tag}/{variant}/{variant}.npz          UAV (Gen11 + Gen15)  ← the one
  A  {seed}/results/{variant}.npz                      state-only flat
  B  {seed}/results/halfspace_{geo}/{variant}.npz      state-only avoiding
  C  {seed}/results/{variant}/{variant}.npz            flat visual-aligning
  D  {seed}/results[_train_set]/{geo}/{variant}/{variant}.npz

Shape U is the reason this is not DA_VA_v2 with a flag: UAV writes the geometry
folder DIRECTLY under the seed, with no `results/` level at all
(`eval_mix_uav.py` builds `seed_dir/geo_dir/out_dir` by hand at lines 1438-1444).
The other four are accepted anyway — the cost is one extra `isdir` per seed and
it means a Gen11 or state-only tree can be merged into a UAV comparison instead
of needing a second tool.

Why the axis parser lives here
------------------------------

The Gen15 experiment IS the K sweep (PLAN §7.3), and K, engine, controller and
threshold are encoded in the CANDIDATE FOLDER NAME (`_uav_eval_tag`), while the
scene is encoded in the dataset folder (`uav-<scene>`) and the backbone / data
proportion / alpha schedule in the model folder (`_uav_mix_exp_name`). Discovery
is the only stage that sees those path segments, so it is the only place that can
turn them into columns. Leaving them inside an opaque candidate name would make
"success vs K, per engine" — the plot the generation exists to produce —
impossible to draw without hand-editing a CSV.
"""

import json
import logging
import os
import re

from config import (
    CB_SENTINEL_NAME,
    ENGINE_LABELS,
    EVAL_TAG_PREFIX_RE,
    EVAL_TAG_RE,
    EVAL_TAG_RE_GEN11,
    GEO_DIR_PREFIXES,
    GEO_NONE,
    MODEL_PREFIX_RE,
    MODEL_TOKEN_RES,
    NON_VARIANT_DIRS,
    RESULTS_DIR_NAMES,
    RESULTS_JSON_NAME,
    SCENE_DIR_RE,
    SCENES,
    SKIP_NPZ_SUFFIXES,
    SNAPSHOT_DIR_PREFIX,
    TIGHTENED_SUFFIX,
    TRAIN_SET_SUFFIX,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_FILE_RE = re.compile(r'^snapshot_(\d{8}_\d{6})$')
_SCENE_RE = re.compile(SCENE_DIR_RE)
_EVAL_TAG_RE = re.compile(EVAL_TAG_RE)
_EVAL_TAG_RE_GEN11 = re.compile(EVAL_TAG_RE_GEN11)
_EVAL_TAG_PREFIX_RE = re.compile(EVAL_TAG_PREFIX_RE)
_WARNED_UNPARSED_TAGS = set()
_MODEL_PREFIX_RE = re.compile(MODEL_PREFIX_RE)
_MODEL_TOKEN_RES = {name: re.compile(pattern)
                    for name, pattern in MODEL_TOKEN_RES.items()}


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

    UAV geo tags (`corridor_bounds+dynamics+geo_bounds+halfspace+obstacles`,
    `empty_unconstrained`) pass through verbatim; the avoiding family's
    `halfspace_top-right-hard` loses its prefix so the labels stay comparable
    with the old state-only CSVs.
    """
    for prefix in GEO_DIR_PREFIXES:
        if raw_dir.startswith(prefix):
            return raw_dir[len(prefix):]
    return raw_dir


def geo_scene(geo):
    """Leading scene token of a UAV geo_tag.

    `corridor_bounds+dynamics+geo_bounds+halfspace+obstacles` → 'corridor'
    `s_curve_dynamics`                                        → 's_curve'
    `empty_unconstrained`                                     → 'empty'

    The eval builds the tag as `f'{scene}_{"+".join(sorted(ctypes))}'` (or
    `f'{scene}_unconstrained'`). Neither underscore split works on it: the SCENE
    can contain one (`s_curve`) and so can a CONSTRAINT FAMILY (`geo_bounds`),
    which is why this matches the known scene names instead — longest first, so
    a future `s_curve_hard` never loses to `s_curve`. An unrecognised scene falls
    back to the leading token, which is right for every name without an
    underscore in it.
    """
    text = str(geo or '')
    for scene in sorted(SCENES, key=len, reverse=True):
        if text == scene or text.startswith(scene + '_'):
            return scene
    return text.split('_', 1)[0] if '_' in text else text


def variant_parts(variant):
    """`dpcc-c-tightened` → ('dpcc-c', True).

    UAV tightening is a per-VARIANT margin modifier enumerated explicitly in
    config/uav_projection.yaml — unlike visual-aligning, where the geo loop
    auto-generates a tightened geometry twin. So `tightened` is read off the
    variant name here and the geometry axis never carries it.
    """
    if variant.endswith(TIGHTENED_SUFFIX):
        return variant[: -len(TIGHTENED_SUFFIX)], True
    return variant, False


def _has_result_data(path, depth=0, max_depth=3):
    """Is there a finished result anywhere under `path`? (npz or a diagnostics dir)"""
    if depth > max_depth:
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    subdirs = []
    for entry in entries:
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            if entry == 'diagnostics':
                return True
            if entry.startswith('.') or entry.startswith(SNAPSHOT_DIR_PREFIX):
                continue
            subdirs.append(full)
        elif entry.endswith('.npz') and not any(entry.endswith(s)
                                                for s in SKIP_NPZ_SUFFIXES):
            return True
    return any(_has_result_data(sub, depth + 1, max_depth) for sub in subdirs)


def _seed_dirs(candidate_path):
    """Numeric subdirectories of a candidate folder that actually hold results.

    Unlike DA_VA_v2 this cannot key on a `results/` folder — UAV has none — so it
    asks the honest question instead: does this numeric folder contain a finished
    result within a few levels? The `config_snapshot_*` folder is skipped so a
    seed dir that was snapshotted and then died before writing anything is not
    mistaken for a result.
    """
    out = []
    try:
        entries = sorted(os.listdir(candidate_path))
    except OSError:
        return out
    for entry in entries:
        if not entry.isdigit():
            continue
        seed_path = os.path.join(candidate_path, entry)
        if os.path.isdir(seed_path) and _has_result_data(seed_path):
            out.append(int(entry))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# path-encoded axes — scene / engine / K / controller / threshold / backbone
# ──────────────────────────────────────────────────────────────────────────────

def parse_eval_tag(name):
    """`Emf_K4_mpc4_pid_stopgo_T0.5` → engine/K/mpc_batch/controller/threshold.

    A trailing `_{run_tag}` (`FMPCC_UAV_EVAL_TAG`, e.g.
    `Emf_K1_mpc4_pid_stopgo_T0.5_fix16scaled`) is parsed into `run_tag` and is a
    first-class axis — two runs that differ only in it are DIFFERENT experiments
    and must never pool. Untagged folders get `run_tag == ''`.

    Falls back to the Gen11 spelling (no `E{engine}` token) so a Gen11 tree can
    sit in the same comparison; `engine` is then left empty for `parse_axes` to
    fill from the model folder. An unrecognised folder name yields empty fields
    rather than raising — a hand-made or renamed folder must still be analysable.
    """
    match = _EVAL_TAG_RE.match(str(name or ''))
    if match:
        found = match.groupdict()
    else:
        match = _EVAL_TAG_RE_GEN11.match(str(name or ''))
        if not match:
            # A folder that is SHAPED like an eval tag but does not parse is a
            # parser bug, and one that costs whole runs: every axis comes back
            # empty, K is None, and the aggregator groupbys then drop the rows on
            # the NaN key with no error. (Exactly how the six Fix_16 A/B runs went
            # missing on 2026-09-03, before EVAL_TAG_RE learned the run-tag
            # suffix.) A folder that is not eval-tag-shaped at all is the ordinary
            # legacy case — a Gen11 MODEL folder used as the candidate — and stays
            # quiet.
            text = str(name or '')
            if _EVAL_TAG_PREFIX_RE.match(text) and text not in _WARNED_UNPARSED_TAGS:
                _WARNED_UNPARSED_TAGS.add(text)
                logger.warning(
                    f'  eval-tag folder {text!r} looks like an eval tag but does NOT '
                    f'match EVAL_TAG_RE — every path-encoded axis (engine/K/mpc/'
                    f'controller/threshold/run_tag) will be EMPTY for it, and rows '
                    f'with no K are excluded from uav_k_sweep.csv. Fix EVAL_TAG_RE '
                    f'in DA_UAV_v1/config.py rather than reading the run as missing.')
            return {}
        found = dict(match.groupdict())
        found['engine'] = ''
    out = {
        'engine': found.get('engine', '') or '',
        'K': _int_or_none(found.get('K')),
        'mpc_batch': _int_or_none(found.get('mpc_batch')),
        'controller': found.get('controller', '') or '',
        'threshold': _float_or_none(found.get('threshold')),
        'run_tag': found.get('run_tag', '') or '',
    }
    return out


def parse_model_dir(model_dir):
    """`mix_uav_mf/H8_DMeanFlowODE_9D_dp0.5_bbunet` → engine + identity tokens.

    `model_dir` is the path segment(s) between the scene root and the eval tag,
    i.e. what `eval_mix_uav.py` calls `_model_dir`. Both halves are optional:
    a tree with no `mix_uav_<engine>/` prefix folder still yields its H/D tokens.
    """
    parts = [p for p in str(model_dir or '').replace('\\', '/').split('/') if p]
    out = {'engine': '', 'model_name': parts[-1] if parts else ''}
    for part in parts:
        prefix = _MODEL_PREFIX_RE.match(part)
        if prefix:
            out['engine'] = prefix.group('engine')
    name = out['model_name']
    for field, pattern in _MODEL_TOKEN_RES.items():
        match = pattern.search(name)
        out[field] = match.group('v') if match else ''
    return out


def parse_axes(candidate_path):
    """Every axis encoded in one candidate's path.

    Returns a flat dict — scene, engine, engine_label, K, mpc_batch, controller,
    threshold, model_name, backbone, data_proportion, alpha_init/alpha_end,
    train_K, horizon, diffusion_cls, obs_dim_tag, generation.

    `K` is the NFE budget the eval actually ran at. It is truthful ONLY because
    `_load_base_cfg` injects `flow_steps_v3` into the config from the plan block
    (see the 🔴 note in `_uav_eval_tag`): in Gen11 that key was never present, so
    every folder was labelled K20 regardless of the real budget. A Gen11-era
    folder's K is therefore NOT to be trusted, which is what `generation` is for.
    """
    path = os.path.abspath(str(candidate_path))
    parts = [p for p in path.replace('\\', '/').split('/') if p]

    scene = ''
    scene_at = -1
    for i, part in enumerate(parts):
        match = _SCENE_RE.match(part)
        if match:
            scene = match.group('scene')
            scene_at = i
    tag = parse_eval_tag(parts[-1] if parts else '')

    # Everything between the scene root (+ optional `plans`) and the eval tag.
    model_parts = []
    if scene_at >= 0:
        model_parts = [p for p in parts[scene_at + 1: len(parts) - 1] if p != 'plans']
    model = parse_model_dir('/'.join(model_parts))

    engine = tag.get('engine') or model.get('engine') or ''
    axes = {
        'scene': scene,
        'engine': engine,
        'engine_label': ENGINE_LABELS.get(engine, engine),
        'K': tag.get('K'),
        'mpc_batch': tag.get('mpc_batch'),
        'controller': tag.get('controller', ''),
        'threshold': tag.get('threshold'),
        # Trailing `_{run_tag}` from FMPCC_UAV_EVAL_TAG. An A/B axis, not cosmetic.
        'run_tag': tag.get('run_tag', ''),
        'eval_tag': parts[-1] if parts else '',
        'model_name': model.get('model_name', ''),
        'horizon': model.get('horizon', ''),
        'diffusion_cls': model.get('diffusion_cls', ''),
        'obs_dim_tag': model.get('obs_dim_tag', ''),
        'backbone': model.get('backbone', ''),
        'data_proportion': model.get('data_proportion', ''),
        'alpha_init': model.get('alpha_init', ''),
        'alpha_end': model.get('alpha_end', ''),
        'train_K': model.get('train_K', ''),
        # Gen15 lives under logs/UAV_MIX, Gen11 under logs/UAV_FM. The roots are
        # different ON PURPOSE (PLAN §1.2 / gate G2b) so a Gen15 job can never
        # overwrite a Gen11 checkpoint; that also makes the root the one reliable
        # generation marker, since both write identical artifacts.
        'generation': ('Gen15' if 'UAV_MIX' in path
                       else ('Gen11' if 'UAV_FM' in path else '')),
    }
    return axes


def display_name(axes, folder_name):
    """A candidate label that says what the run IS, not where it sits on disk.

    The raw folder name is the eval tag (`Emf_K4_mpc4_pid_stopgo_T0.5`), which
    omits the two axes a reader needs first — which scene, which model — and
    repeats four they rarely do. This builds `corridor|mf|K4|bbunet` instead, so
    a K sweep is readable straight off the candidate checkbox list.
    """
    bits = []
    if axes.get('scene'):
        bits.append(axes['scene'])
    if axes.get('engine'):
        bits.append(axes['engine'])
    if axes.get('K') is not None:
        bits.append(f"K{axes['K']}")
    for field, prefix in (('backbone', 'bb'), ('data_proportion', 'dp')):
        if axes.get(field):
            bits.append(f'{prefix}{axes[field]}')
    if axes.get('generation') == 'Gen11':
        bits.append('Gen11')
    # 🔴 Must be in the label. Without it the A and B arms of an A/B collapse to
    # the same display name and read as one candidate in every by-name table.
    if axes.get('run_tag'):
        bits.append(f"@{axes['run_tag']}")
    return '|'.join(bits) if bits else folder_name


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    Returns latest / first / count / per_seed / n_seeds_stamped. Never raises: a
    tree written before config snapshots existed just reports empty strings.

    `Snapshot_By_Seed` is the column to read when a candidate looks half-stale —
    UAV seeds are launched as separate SLURM jobs, so one fresh seed can hide
    four old ones behind a recent `Latest_Snapshot`.
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
    """'20260806_034806' -> '2026-08-06 03:48:06'. Anything else passes through."""
    text = str(stamp or '').strip()
    if len(text) == 15 and text[8] == '_' and text.replace('_', '').isdigit():
        return (f'{text[0:4]}-{text[4:6]}-{text[6:8]} '
                f'{text[9:11]}:{text[11:13]}:{text[13:15]}')
    return text


def snapshot_by_seed_str(per_seed):
    """'6:20260806_034806 | 7:20260806_041233' — the per-seed audit trail."""
    if not per_seed:
        return ''
    return ' | '.join(f'{seed}:{per_seed[seed]}' for seed in sorted(per_seed))


# ──────────────────────────────────────────────────────────────────────────────
# candidates
# ──────────────────────────────────────────────────────────────────────────────

def discover_candidates(parent_paths, seed_list=None, max_depth=10):
    """Find every candidate folder under one or more parent trees.

    A candidate is a folder with at least one numeric seed subdir holding result
    data. Search does not descend into a folder once it is recognised as one.

    Args:
        parent_paths: path string, comma-separated string, or list of paths.
        seed_list:    restrict to these seeds (None = every seed found).
        max_depth:    how deep below each parent to look. The UAV tree is
                      `<root>/uav-<scene>/plans/mix_uav_<engine>/<model>/<tag>/`
                      = 5 levels below `logs/UAV_MIX`, so the default has room
                      for a root passed one or two levels higher.

    Returns:
        {index (1-based int): {path, name, display, axes, seeds, missing_seeds,
                               snapshots, cb_sentinels}}
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
            axes = parse_axes(path)
            name = os.path.basename(os.path.normpath(path))
            entry = {
                'path': os.path.abspath(path),
                'name': name,
                'display': display_name(axes, name),
                'axes': axes,
                'parent': parent_root,
                'seeds': sorted(seeds),
                'missing_seeds': ([s for s in seed_list if s not in seeds]
                                  if seed_list else []),
                'snapshots': scan_snapshot_timestamps(path, seeds),
                'cb_sentinels': _count_cb_sentinels(path, seeds),
            }
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
        axes = info['axes']
        logger.info(f'Candidate {idx}: {info["display"]}  seeds={info["seeds"]}  '
                    f'(scene={axes.get("scene") or "?"} engine={axes.get("engine") or "?"} '
                    f'K={axes.get("K")})')
        if info['missing_seeds']:
            logger.warning(f'Candidate {idx} ({info["display"]}) missing seeds: '
                           f'{info["missing_seeds"]}')
        if info['cb_sentinels']:
            logger.warning(f'Candidate {idx} ({info["display"]}): '
                           f'{info["cb_sentinels"]} variant folder(s) carry a '
                           f'{CB_SENTINEL_NAME} sentinel — projection was ABANDONED '
                           f'there, their constraint metrics are not valid')
    if not candidates:
        logger.warning(f'No candidates found under: {parent_paths}')
    else:
        logger.info(f'Total candidates discovered: {len(candidates)}')
    return candidates


def _count_cb_sentinels(candidate_path, seeds):
    """How many variant folders of this candidate carry PROJECTION_CB_TRIPPED.txt.

    The eval drops that file whenever the sustained-slowness breaker opened
    (eval_mix_uav.py Fix_15.3), precisely so a broken run is visible from the file
    tree without opening an npz. Counting it at discovery time means the warning
    lands in the log BEFORE the numbers do.
    """
    total = 0
    for seed in seeds:
        for dirpath, dirnames, filenames in os.walk(
                os.path.join(candidate_path, str(seed))):
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            if CB_SENTINEL_NAME in filenames:
                total += 1
    return total


def filter_candidates(candidates, selected):
    """Keep only the listed candidate indices ("1,3,5" or [1, 3, 5])."""
    if isinstance(selected, str):
        wanted = {s.strip() for s in selected.split(',') if s.strip()}
    else:
        wanted = {str(s) for s in selected}
    filtered = {k: v for k, v in candidates.items() if str(k) in wanted}
    logger.info(f'Filtered candidates: {sorted(filtered.keys())}')
    return filtered


def filter_by_axes(candidates, scenes=None, engines=None, k_values=None):
    """Keep candidates matching the path-encoded axis whitelists.

    The axis filters that make a K sweep tractable: `--scenes corridor
    --engines mf,af --k 1,2,5` cuts a 4-scene x 4-engine x 5-K tree of 80
    candidates down to the six rows of the plot being drawn. Applied BEFORE unit
    enumeration, so the units of a rejected candidate are never even listed.
    """
    if not any((scenes, engines, k_values)):
        return candidates
    out = {}
    for key, info in candidates.items():
        axes = info.get('axes', {})
        if scenes and axes.get('scene') not in scenes:
            continue
        if engines and axes.get('engine') not in engines:
            continue
        if k_values and axes.get('K') not in k_values:
            continue
        out[key] = info
    logger.info(f'Axis filter kept {len(out)}/{len(candidates)} candidates '
                f'(scenes={scenes} engines={engines} K={k_values})')
    return out


def assign_custom_names(candidates, custom_names):
    """Attach display names ("fm,mf,af,dm") in candidate index order."""
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
        axes = info.get('axes', {})
        lines.append(f'  {key}: {info.get("custom_name") or info["display"]}')
        lines.append(f'      Path:  {info["path"]}')
        lines.append(f'      Axes:  scene={axes.get("scene") or "?"}  '
                     f'engine={axes.get("engine") or "?"}  K={axes.get("K")}  '
                     f'mpc={axes.get("mpc_batch")}  ctrl={axes.get("controller") or "?"}  '
                     f'T={axes.get("threshold")}  gen={axes.get("generation") or "?"}')
        if axes.get('backbone') or axes.get('data_proportion'):
            lines.append(f'      Model: {axes.get("model_name")}  '
                         f'bb={axes.get("backbone") or "-"}  '
                         f'dp={axes.get("data_proportion") or "-"}')
        lines.append(f'      Seeds: {info["seeds"]}')
        snapshots = info.get('snapshots') or {}
        if snapshots.get('latest'):
            lines.append(f'      Last run: {format_snapshot_ts(snapshots["latest"])}'
                         f'  ({snapshots["count"]} config snapshot(s) over '
                         f'{snapshots["n_seeds_stamped"]} seed(s))')
        if info.get('cb_sentinels'):
            lines.append(f'      WARNING {info["cb_sentinels"]} variant folder(s) '
                         f'marked {CB_SENTINEL_NAME}')
        if info.get('missing_seeds'):
            lines.append(f'      WARNING missing seeds: {info["missing_seeds"]}')
        lines.append('')
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# result units
# ──────────────────────────────────────────────────────────────────────────────

def parse_unit_path(unit_root, npz_path):
    """Map an npz path under a unit root onto (geo_raw, variant_raw).

    Returns None when the path is not a variant result (expert references etc.).
    Handles all five shapes listed in the module docstring; the UAV one is the
    two-level `{geo}/{variant}/{variant}.npz`.
    """
    rel_dir = os.path.relpath(os.path.dirname(npz_path), unit_root)
    stem = os.path.basename(npz_path)[: -len('.npz')]

    if rel_dir in ('.', ''):
        return GEO_NONE, stem                                    # shape A

    parts = [p for p in rel_dir.split(os.sep) if p != '.']
    if any(p in NON_VARIANT_DIRS for p in parts):
        return None

    if len(parts) == 1:
        # `{variant}/{variant}.npz` (shape C) vs `{geo}/{variant}.npz` (shape B).
        if clean_variant(parts[0]) == clean_variant(stem):
            return GEO_NONE, stem
        return parts[0], stem

    # shapes U and D — last directory is the variant, everything above is geo.
    return os.sep.join(parts[:-1]).replace(os.sep, '/'), stem


def unit_roots(seed_dir):
    """The (root, split) pairs to scan inside one seed folder.

    UAV writes its geometry folders DIRECTLY under the seed, so the seed folder
    itself is the root and the split is always 'test'. When a `results/` /
    `results_train_set/` level IS present (the aligning/avoiding family, or a
    future UAV eval that grows one) those are used instead — never both, or every
    unit under `results/` would be discovered twice, once through each root.
    """
    roots = []
    for name in RESULTS_DIR_NAMES:
        path = os.path.join(seed_dir, name)
        if os.path.isdir(path):
            roots.append((path, split_of(name)))
    if roots:
        return roots
    return [(seed_dir, 'test')]


def discover_units(candidate_idx, candidate_info, seeds=None, splits=None,
                   geos=None, variants=None):
    """Enumerate every result unit of one candidate.

    Args:
        candidate_idx:  1-based candidate key.
        candidate_info: entry from discover_candidates().
        seeds/splits/geos/variants: optional whitelists (cleaned names).

    Returns:
        list of unit dicts. `npz_path` is None when only diagnostics JSONs exist
        (a variant killed before its final write) — the loader falls back to the
        JSON source there, which on UAV loses almost nothing since the timing
        block is JSON-only anyway.
    """
    units = []
    root = candidate_info['path']
    use_seeds = candidate_info['seeds'] if seeds is None else \
        [s for s in candidate_info['seeds'] if s in seeds]

    for seed in use_seeds:
        seed_dir = os.path.join(root, str(seed))
        for results_root, split in unit_roots(seed_dir):
            if splits and split not in splits:
                continue

            seen_dirs = set()

            for dirpath, dirnames, filenames in os.walk(results_root):
                dirnames[:] = [d for d in sorted(dirnames)
                               if d not in NON_VARIANT_DIRS
                               and not d.startswith('.')
                               and not d.startswith(SNAPSHOT_DIR_PREFIX)]
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

            # JSON-only variants: a diagnostics folder whose sibling npz is
            # absent (a variant that died before its final write — common on the
            # s_curve scene, which brushes the 24 h SLURM limit).
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
    variant = clean_variant(variant_raw)
    variant_base, tightened = variant_parts(variant)
    axes = candidate_info.get('axes', {}) or {}
    unit = {
        'Candidate': candidate_idx,
        'FolderName': (candidate_info.get('custom_name')
                       or candidate_info.get('display')
                       or candidate_info['name']),
        'RawFolderName': candidate_info['name'],
        'FullPath': candidate_info['path'],
        'seed': seed,
        'split': split,
        'geo': geo,
        'geo_raw': geo_raw,
        # The scene the geo_tag names. Usually equal to the path-derived scene;
        # they differ only if a geo entry from another scene was run here, which
        # is exactly the mix-up worth being able to see.
        'geo_scene': geo_scene(geo),
        'variant': variant,
        'variant_raw': variant_raw,
        'variant_base': variant_base,
        'tightened': bool(tightened),
        'npz_path': npz_path,
        'variant_dir': variant_dir,
        'diagnostics_dir': os.path.join(variant_dir, 'diagnostics'),
        'results_json': os.path.join(variant_dir, RESULTS_JSON_NAME),
        'cb_sentinel': os.path.isfile(os.path.join(variant_dir, CB_SENTINEL_NAME)),
    }
    # Path-encoded axes ride on every unit so the aggregator never has to look a
    # candidate back up to group by scene / engine / K.
    for field in ('scene', 'engine', 'engine_label', 'K', 'mpc_batch',
                  'controller', 'threshold', 'run_tag', 'backbone',
                  'data_proportion', 'model_name', 'generation'):
        unit[field] = axes.get(field, '')
    return unit


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
                'display_name': v.get('custom_name') or v.get('display') or v['name'],
                'path': v['path'],
                'axes': v.get('axes', {}),
                'seeds': v['seeds'],
                'missing_seeds': v.get('missing_seeds', []),
                'snapshots': v.get('snapshots', {}),
                'cb_sentinels': v.get('cb_sentinels', 0),
            } for k, v in sorted(candidates.items())
        },
        'units': [
            {
                'Candidate': u['Candidate'], 'seed': u['seed'], 'split': u['split'],
                'geo': u['geo'], 'variant': u['variant'],
                'scene': u.get('scene', ''), 'engine': u.get('engine', ''),
                'K': u.get('K'),
                'source': 'npz' if u['npz_path'] else 'json',
                'cb_sentinel': bool(u.get('cb_sentinel')),
            } for u in units
        ],
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
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
            print(f'   seed={unit["seed"]} geo={unit["geo"]} '
                  f'variant={unit["variant"]} '
                  f'src={"npz" if unit["npz_path"] else "json"}')
        if len(found) > 5:
            print(f'   ... +{len(found) - 5} more')
    print(f'TOTAL units: {total}')
