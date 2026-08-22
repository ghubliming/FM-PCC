#!/usr/bin/env python3
"""Gen16 gates — run these on the cluster BEFORE believing any visual-avoiding number.

    python -m mix_visual_avoiding_test.gates_mix_visual_avoiding --gate all
    python -m mix_visual_avoiding_test.gates_mix_visual_avoiding --gate offline   # no torch

A0  copy fidelity        every file that should be a pure rename of Gen14's still is
A1  spec coherence       visual_spec's derived constants actually derive
A2  no stray literals    nothing outside visual_spec.py names a camera or a dim
A3  registry wiring      the four arms resolve to Gen16 classes + the right trainers
A4  path round-trip      diffusion_loadpath reproduces the train block's exp_name, per arm
A5  backbones agree      all three bones report the SAME dims/latent (one spec, one task)
A6  dataset <-> spec     the dataset's constants and condition keys match the spec
A7  four arms train      one loss step per engine, if_vision=True, single camera, finite
A8  hardflow hosts       'diffusion' is refused; fm gets images, mf/af get a latent
A9  yaml <-> config      arm-C fan == arm-B fan, and the two thresholds are what they claim

A0/A1/A2/A9 need NOTHING but the stdlib — they run in the AI-coding container
(`--gate offline`). A3/A4/A6 need the FMPCC env (torch import) but no GPU
(`--gate static` adds them). A5/A7/A8 build models: GPU.

🔴 A0 IS THE LOAD-BEARING GATE. Gen16's whole claim is "Gen14's frame, one task swapped".
   If A0 fails, that claim is false and every cross-generation comparison is suspect. Do not
   patch over it — re-open the plan.
"""

import argparse
import ast
import difflib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PKG = 'mix_visual_avoiding'
SRC_PKG = 'mix_visual_aligning'


# ══════════════════════════════════════════════════════════════════════════════
# A0 — the copy ledger
# ══════════════════════════════════════════════════════════════════════════════
# Gen16 = Gen14 @ HEAD with the package renamed, MINUS the files listed under EDITED.
# Anything not in EDITED must compare byte-equal after reversing the rename.
#
# Keeping this list exhaustive is what makes "Gen16 reproduces Gen14's engines" a property
# of the file layout rather than something a numerical test has to establish.

EDITED = {
    # ── the task spec, and the files that now read it ──
    f'{PKG}/models/visual_spec.py':
        'NEW — the single source of truth for cameras + dims',
    f'{PKG}/models/visual_unet.py':
        'encoder + dims from visual_spec; encode_visual is variadic over cameras',
    f'{PKG}/models/visual_unet_twotime.py':
        'same three edits as visual_unet.py',
    f'{PKG}/models/visual_dit_twotime.py':
        'same three edits as visual_unet.py',
    f'{PKG}/models/visual_gaussian_diffusion.py':
        'cond packing via visual_spec (no hardcoded 2-camera tuple)',
    f'{PKG}/models/visual_fm_diffusion.py':
        'cond packing via visual_spec',
    f'{PKG}/models/visual_mf_diffusion.py':
        'cond packing via visual_spec; _encode_once is variadic',
    f'{PKG}/models/visual_af_diffusion.py':
        'cond packing via visual_spec; _encode_once is variadic',
    f'{PKG}/models/__init__.py':
        'exports visual_spec',
    f'{PKG}/models/engine_registry.py':
        'provenance strings only (the table itself is Gen14 verbatim)',
    # ── the data ──
    f'{PKG}/datasets/sequence.py':
        'REPLACED — ParityAvoidingDataset (Gen9 Epoch 2) + episode_split, spec-driven',
    f'{PKG}/datasets/__init__.py':
        'exports ParityAvoidingDataset',
    # ── the trainers ──
    f'{PKG}/utils/training.py':
        'episode-level split + EMA-consistent test() + final save (Gen9 U4 Fix1 / B8)',
    f'{PKG}/utils/training_twotime.py':
        'the SAME three edits, so all four arms share one split and one selection criterion',
    f'{PKG}/utils/serialization.py':
        "DiffusionExperiment gains 'ema' (mirrors Gen3v6 Fix2/U6)",
    f'{PKG}/utils/setup.py':
        'the snapshotted eval yaml follows FMPCC_PROJ_CFG instead of a hardcoded name',
    # ── the samplers ──
    f'{PKG}/sampling/policies.py':
        'NEW — VisualPolicy / VisualHardFlowPolicy, the Policy surface over visual engines',
    f'{PKG}/sampling/hardflow_projection.py':
        'encode_visual_cond via visual_spec + the fm raw-image path',
}


def _norm(path):
    """Read a Gen16 file and reverse the package rename, so a pure copy compares equal."""
    with open(os.path.join(REPO, path), 'rb') as f:
        return f.read().decode('utf-8').replace('\r\n', '\n').replace(PKG, SRC_PKG)


def _walk_pkg(pkg):
    out = []
    for root, _dirs, files in os.walk(os.path.join(REPO, pkg)):
        if '__pycache__' in root:
            continue
        for fn in sorted(files):
            if fn.endswith('.py'):
                out.append(os.path.relpath(os.path.join(root, fn), REPO))
    return sorted(out)


def gate_a0(**_):
    """Every file not in EDITED is byte-identical to Gen14's, modulo the package rename."""
    print('\n=== A0: copy fidelity (Gen16 vs Gen14) ===')
    failures, clean, skipped = [], 0, 0

    gen16_files = _walk_pkg(PKG)
    gen14_files = {p.replace(PKG, SRC_PKG) for p in gen16_files}

    for dst in gen16_files:
        src = dst.replace(PKG, SRC_PKG)
        src_abs = os.path.join(REPO, src)
        if dst in EDITED:
            print(f'  edit {dst}\n         <- {EDITED[dst]}')
            skipped += 1
            continue
        if not os.path.exists(src_abs):
            failures.append(f'{dst}: no Gen14 counterpart at {src} — either it is a NEW '
                            f'Gen16 file (add it to EDITED with a reason) or Gen14 moved it.')
            continue
        with open(src_abs, 'rb') as f:
            src_text = f.read().decode('utf-8').replace('\r\n', '\n')
        if _norm(dst) != src_text:
            diff = list(difflib.unified_diff(
                src_text.splitlines(), _norm(dst).splitlines(),
                fromfile=src, tofile=dst, lineterm='', n=0))
            failures.append(f'{dst}: DIFFERS from {src} ({len(diff)} diff lines) — '
                            f'either declare the edit in EDITED with a reason, or revert it.')
            continue
        clean += 1

    # A Gen14 file with no Gen16 counterpart is a silently dropped arm.
    for src in _walk_pkg(SRC_PKG):
        if src.replace(SRC_PKG, PKG) not in set(gen16_files):
            failures.append(f'{src}: present in Gen14 but MISSING from Gen16 — an arm or a '
                            f'backbone was dropped. Copy it or record why it is absent.')

    # Every EDITED entry must name a file that exists (a stale entry hides a real diff).
    for dst in EDITED:
        if not os.path.exists(os.path.join(REPO, dst)):
            failures.append(f'{dst}: listed in EDITED but does not exist — stale ledger entry.')

    if failures:
        print(f'\n  A0 FAIL — the copy assumption broke ({len(failures)} problems):')
        for f_ in failures:
            print(f'    ! {f_}')
        return False
    print(f'\n  A0 PASS — {clean} files byte-identical to Gen14, {skipped} declared edits.')
    return True


# ══════════════════════════════════════════════════════════════════════════════
# A1 — the spec derives what it claims to derive
# ══════════════════════════════════════════════════════════════════════════════

def _load_spec_module():
    """Import visual_spec WITHOUT importing torch (it has no torch-level imports)."""
    import importlib.util
    path = os.path.join(REPO, PKG, 'models', 'visual_spec.py')
    spec = importlib.util.spec_from_file_location('_gen16_visual_spec', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gate_a1(**_):
    """visual_spec's derived constants actually derive; the camera lists agree in length."""
    print('\n=== A1: spec coherence ===')
    vs = _load_spec_module()
    checks = [
        ('LATENT_DIM == N_CAMERAS * RGB_OUTPUT_SIZE',
         vs.LATENT_DIM == vs.N_CAMERAS * vs.RGB_OUTPUT_SIZE),
        ('TRANSITION_DIM == ACTION_DIM + OBS_DIM',
         vs.TRANSITION_DIM == vs.ACTION_DIM + vs.OBS_DIM),
        ('len(CAMERA_KEYS) == len(COND_IMG_KEYS)',
         len(vs.CAMERA_KEYS) == len(vs.COND_IMG_KEYS) == vs.N_CAMERAS),
        ('shape_meta() has one entry per camera',
         len(vs.shape_meta()['obs']) == vs.N_CAMERAS),
        ('shape_meta() keys == CAMERA_KEYS',
         tuple(vs.shape_meta()['obs']) == tuple(vs.CAMERA_KEYS)),
        ('avoiding is 6-D', vs.TRANSITION_DIM == 6),
        ('avoiding is single-camera', vs.N_CAMERAS == 1),
    ]
    # split_visual round-trips whatever pack_visual builds
    payload = vs.pack_visual(tuple(f'img{i}' for i in range(vs.N_CAMERAS)), 'obs_seq')
    cams, obs = vs.split_visual(payload)
    checks.append(('pack_visual/split_visual round-trip',
                   len(cams) == vs.N_CAMERAS and obs == 'obs_seq'))
    # a short payload must RAISE, never silently drop a camera
    try:
        vs.split_visual(('only_one_thing',)[:max(0, vs.N_CAMERAS - 1)])
        short_raises = (vs.N_CAMERAS == 0)
    except (ValueError, TypeError):
        short_raises = True
    checks.append(('split_visual refuses a short payload', short_raises))

    ok = True
    for label, passed in checks:
        print(f'  {"ok  " if passed else "FAIL"} {label}')
        ok &= bool(passed)
    print(f'  spec: {vs.LAYOUT}')
    print(f'  A1 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A2 — no module outside visual_spec.py names a camera or a dim
# ══════════════════════════════════════════════════════════════════════════════
# This is the gate that keeps the hoist honest. Gen14 spread these literals across nine
# files; if one creeps back, the next ML bone silently inherits the wrong camera count.

_BANNED = [
    (re.compile(r"['\"]in_hand_image['\"]"), "camera key 'in_hand_image'"),
    (re.compile(r"['\"]wrist_img['\"]"),     "condition key 'wrist_img'"),
    (re.compile(r'^\s*TRANSITION_DIM\s*=\s*\d'), 'a literal TRANSITION_DIM'),
    (re.compile(r'^\s*LATENT_DIM\s*=\s*\d'),     'a literal LATENT_DIM'),
]
# Comments are exempt: the files legitimately EXPLAIN what they no longer hardcode.
_EXEMPT_FILES = {f'{PKG}/models/visual_spec.py'}


def _strip_comments_and_docstrings(src):
    """Return only the executable text of a module, so prose cannot trip the scan."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc_lines.add(ln)
    out = []
    for i, line in enumerate(src.splitlines(), start=1):
        if i in doc_lines:
            out.append('')
            continue
        out.append(line.split('#', 1)[0] if '#' in line else line)
    return '\n'.join(out)


def gate_a2(**_):
    """No executable line outside visual_spec.py hardcodes a camera name or a dim."""
    print('\n=== A2: no stray camera/dim literals ===')
    failures = []
    for path in _walk_pkg(PKG):
        if path in _EXEMPT_FILES:
            continue
        with open(os.path.join(REPO, path), 'rb') as f:
            src = f.read().decode('utf-8')
        code = _strip_comments_and_docstrings(src)
        for lineno, line in enumerate(code.splitlines(), start=1):
            for pattern, what in _BANNED:
                if pattern.search(line):
                    failures.append(f'{path}:{lineno} carries {what} -> {line.strip()!r}')
    if failures:
        print(f'  A2 FAIL — the spec hoist leaked ({len(failures)} sites). Every one of these '
              f'belongs in visual_spec.py:')
        for f_ in failures:
            print(f'    ! {f_}')
        return False
    print(f'  A2 PASS — {len(_walk_pkg(PKG)) - len(_EXEMPT_FILES)} modules are spec-driven.')
    return True


# ══════════════════════════════════════════════════════════════════════════════
# A3 — registry wiring
# ══════════════════════════════════════════════════════════════════════════════

def gate_a3(**_):
    """The four arms resolve to Gen16 classes and the right Trainer."""
    print('\n=== A3: registry wiring ===')
    from mix_visual_avoiding.models.engine_registry import ENGINES, resolve, ENGINE_KEYS
    expected = {
        'diffusion': ('VisualGaussianDiffusion', 'VisualUNet',      'training.Trainer',         False),
        'fm':        ('VisualFlowMatching',      'VisualUNet',      'training.Trainer',         False),
        'mf':        ('VisualMeanFlow',          'MeanFlowEngine',  'training_twotime.Trainer', True),
        'af':        ('VisualAlphaFlow',         'AlphaFlowEngine', 'training_twotime.Trainer', True),
    }
    ok = tuple(sorted(ENGINE_KEYS)) == tuple(sorted(expected))
    if not ok:
        print(f'  FAIL engine set is {sorted(ENGINE_KEYS)}, expected {sorted(expected)}')
    for eng, (diff_cls, model_cls, trainer, two_time) in expected.items():
        spec = resolve(eng)
        good = (spec['diffusion'].endswith(diff_cls)
                and spec['model'].endswith(model_cls)
                and spec['trainer'].endswith(trainer)
                and spec['two_time'] == two_time
                and spec['diffusion'].startswith(PKG)
                and spec['model'].startswith(PKG))
        print(f'  {"ok  " if good else "FAIL"} {eng:<9} {spec["diffusion"]}')
        ok &= good
    # 🔴 The structural rule: no diffusion/fm entry may reach a two-time module.
    for eng in ('diffusion', 'fm'):
        spec = resolve(eng)
        leaked = [v for k, v in spec.items()
                  if isinstance(v, str) and ('twotime' in v or 'two_time' in v)]
        if leaked:
            print(f'  FAIL {eng} reaches a two-time module: {leaked} — the reference arms '
                  f'must import ONLY verbatim single-time copies.')
            ok = False
    print(f'  A3 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A4 — the path round-trip (the oldest trap in this repo)
# ══════════════════════════════════════════════════════════════════════════════

def gate_a4(**_):
    """`diffusion_loadpath` reproduces the TRAIN block's `exp_name`, key for key, per arm.

    A plan block whose loadpath does not match resolves to a directory that does not exist,
    and the eval dies minutes into a GPU allocation. The config derives both strings from
    one watch list precisely so this cannot happen — this gate proves the derivation.
    """
    print('\n=== A4: checkpoint-path round-trip ===')
    import importlib
    cfg = importlib.import_module('config.avoiding-d3il-visual-mix')

    class _Args:
        def __init__(self, d):
            self.__dict__.update(d)

    ok = True
    for eng in ('diffusion', 'fm', 'mf', 'af'):
        train_blk = cfg.base[f'mix_visual_avoiding_{eng}']
        plan_blk  = cfg.base[f'plan_mix_visual_avoiding_{eng}']

        # What watch() emits at TRAIN time. NOTE this ALREADY carries the
        # 'mix_visual_avoiding_<engine>/' fragment: `prefix` is the first entry of
        # args_to_watch_mix_visual_train with an empty label, and watch() collapses the
        # '/_' join back to '/'. So the loadpath must equal train_name outright — do not
        # re-prepend the prefix here.
        train_name = train_blk['exp_name'](_Args(train_blk))
        # What the plan block asks the loader to look for, resolved against the PLAN args.
        loadpath = plan_blk['diffusion_loadpath']
        assert loadpath.startswith('f:')
        try:
            resolved = loadpath[2:].format(**plan_blk)
        except KeyError as e:
            print(f'  FAIL {eng:<9} loadpath references {e} which the plan block lacks')
            ok = False
            continue
        good = (resolved == train_name)
        print(f'  {"ok  " if good else "FAIL"} {eng:<9} {resolved}')
        if not good:
            print(f'         train exp_name -> {train_name}')
        ok &= good

        # The plan block's own `prefix` must resolve too — it is the CHECKPOINT identity
        # re-pointed into the plans/ namespace, and an unresolvable placeholder there fails
        # only after the model has loaded.
        try:
            plan_blk['prefix'][2:].format(**plan_blk)
        except KeyError as e:
            print(f'  FAIL {eng:<9} plan prefix references {e} which the plan block lacks')
            ok = False

        # ml_bone / film_mode must not appear on a plan block unless the TRAIN block has it
        for key in ('ml_bone', 'film_mode'):
            if key in plan_blk and key not in train_blk:
                print(f'  FAIL {eng:<9} plan block carries {key!r} but the train block does '
                      f'not — the results folder would claim an architecture the weights lack')
                ok = False
        # ONE fan, two names
        if plan_blk.get('batch_size') != plan_blk.get('mpc_batch_size'):
            print(f'  FAIL {eng:<9} batch_size={plan_blk.get("batch_size")} != '
                  f'mpc_batch_size={plan_blk.get("mpc_batch_size")}')
            ok = False
        # matched K across arms B and C
        if eng != 'diffusion' and plan_blk.get('flow_steps_v3') != plan_blk.get('flow_steps'):
            print(f'  FAIL {eng:<9} arm-B K={plan_blk.get("flow_steps_v3")} != arm-C '
                  f'K={plan_blk.get("flow_steps")} — unmatched NFE budget')
            ok = False
    print(f'  A4 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A5 — the three bones agree on the task
# ══════════════════════════════════════════════════════════════════════════════

def gate_a5(**_):
    """All three visual bones report the SAME dims and latent width.

    They now read one module, so this is true by construction — the gate makes the property
    visible and catches a future bone that reintroduces its own constants.
    """
    print('\n=== A5: backbones agree on the task ===')
    from mix_visual_avoiding.models import visual_spec
    from mix_visual_avoiding.models.visual_unet import VisualUNet
    from mix_visual_avoiding.models.visual_unet_twotime import VisualUNetTwoTime
    from mix_visual_avoiding.models.visual_dit_twotime import VisualDiTTwoTime

    ok = True
    for cls in (VisualUNet, VisualUNetTwoTime, VisualDiTTwoTime):
        good = (cls.TRANSITION_DIM == visual_spec.TRANSITION_DIM
                and cls.LATENT_DIM == visual_spec.LATENT_DIM)
        print(f'  {"ok  " if good else "FAIL"} {cls.__name__:<20} '
              f'TRANSITION_DIM={cls.TRANSITION_DIM} LATENT_DIM={cls.LATENT_DIM}')
        ok &= good
    print(f'  A5 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A6 — dataset <-> spec
# ══════════════════════════════════════════════════════════════════════════════

def gate_a6(**_):
    """The dataset's constants, camera plumbing and DATA PATHS are self-consistent.

    Static: no data is loaded (the pickles live on the cluster's data volume, and this gate
    must be runnable before a single frame is read).

    🔴 The path half was added after job 24857. The train script had its own copy of the
    dataset-path literal and it was WRONG — `.../avoiding/train_files.pkl` instead of
    `.../avoiding/all_data/train_files.pkl`, copied from the aligning layout where the
    episode list sits one level higher. Nothing caught it until a GPU allocation did, and
    only after the config pkl had already been written to the checkpoint dir. `sequence.py`
    now defines the paths once; this gate pins that there is no second copy.
    """
    print('\n=== A6: dataset <-> spec, and the data paths ===')
    from mix_visual_avoiding.models import visual_spec
    from mix_visual_avoiding.datasets import sequence as seq
    from mix_visual_avoiding.datasets.sequence import ParityAvoidingDataset as DS

    checks = [
        ('ACTION_DIM',     DS.ACTION_DIM == visual_spec.ACTION_DIM),
        ('OBS_DIM',        DS.OBS_DIM    == visual_spec.OBS_DIM),
        ('TRAJ_DIM',       DS.TRAJ_DIM   == visual_spec.TRANSITION_DIM),
        ('one CAM_DIR per condition key',
         tuple(DS.CAM_DIRS) == tuple(visual_spec.COND_IMG_KEYS)),
        ('episode_split exists (no window leakage)', hasattr(DS, 'episode_split')),
        # ── paths: the episode list and the directories it names share one root ──
        ('STATE_DIR is under DATA_ROOT',
         seq.STATE_DIR.startswith(seq.DATA_ROOT + '/')),
        ('DEFAULT_DATASET_PATH is under DATA_ROOT',
         seq.DEFAULT_DATASET_PATH.startswith(seq.DATA_ROOT + '/')),
        ('DATA_ROOT names the avoiding task, not aligning',
         '/avoiding/' in seq.DATA_ROOT + '/' and 'aligning' not in seq.DATA_ROOT),
        ('the class re-export matches the module definition',
         DS.DEFAULT_DATASET_PATH == seq.DEFAULT_DATASET_PATH
         and DS.DATA_ROOT == seq.DATA_ROOT),
    ]

    # ── no SECOND copy of the path anywhere in the generation ────────────────────────
    # The train script must import it. A literal here is the exact defect job 24857 hit.
    stray = []
    scan = _walk_pkg(PKG) + [f'{PKG}_test/{n}' for n in
                             ('train_mix_visual_avoiding.py', 'eval_mix_visual_avoiding.py')]
    for path in scan:
        abs_path = os.path.join(REPO, path)
        if not os.path.exists(abs_path) or path.endswith('datasets/sequence.py'):
            continue
        with open(abs_path, 'rb') as f:
            code = _strip_comments_and_docstrings(f.read().decode('utf-8'))
        for lineno, line in enumerate(code.splitlines(), start=1):
            if re.search(r"['\"]environments/dataset/data/", line):
                stray.append(f'{path}:{lineno} -> {line.strip()!r}')
    checks.append((f'no hardcoded data path outside sequence.py ({len(stray)} found)',
                   not stray))

    ok = True
    for label, passed in checks:
        print(f'  {"ok  " if passed else "FAIL"} {label}')
        ok &= bool(passed)
    for s_ in stray:
        print(f'    ! {s_}   (import sequence.DEFAULT_DATASET_PATH instead)')
    print(f'  paths: root={seq.DATA_ROOT}  list={seq.DEFAULT_DATASET_PATH}')
    print(f'  A6 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A7 — all four arms take one training step on single-camera visual data
# ══════════════════════════════════════════════════════════════════════════════

def gate_a7(device='cuda'):
    """One loss step per engine, if_vision=True, ONE camera. Finite loss, no shape error.

    🔴 THE GATE THAT ANSWERS THE ACTUAL QUESTION — "do mf/af work in visual avoiding?".
    The mf/af arms differentiate the network with a forward-mode JVP; the vision encoder is
    kept OUT of it by pre-encoding the latent (visual_mf_diffusion.py). If that repack is
    wrong for a single-camera payload, this is where it surfaces — not 9 hours into a run.

    🔴 CONSTRUCTION GOES THROUGH `utils.Config`, exactly as the train script does. An
    earlier version of this gate called `DiffusionCls(model, device=..., **kwargs)` by hand
    and every arm died on `unexpected keyword argument 'device'` (job 24853) — `device` is a
    `Config` kwarg that drives `.to(device)`, never an engine kwarg. Hand-rolling the
    constructor also meant the gate was not testing the path training actually takes. It
    does now: `savepath=None` so nothing is pickled, `verbose=False` so the log stays short.
    """
    print('\n=== A7: four arms, one visual training step ===')
    import torch
    import mix_visual_avoiding.utils as utils
    from mix_visual_avoiding.models import visual_spec
    from mix_visual_avoiding.models.engine_registry import resolve

    H, B = 8, 2

    class _Cfg:
        """Stand-in for the Tap args object the config block produces."""

    ok = True
    for eng in ('diffusion', 'fm', 'mf', 'af'):
        spec = resolve(eng)
        cfg = _Cfg()
        cfg.device = device
        cfg.if_vision = True
        cfg.horizon = H
        cfg.action_dim = visual_spec.ACTION_DIM
        cfg.obs_dim = visual_spec.OBS_DIM
        cfg.dim = 32
        cfg.dim_mults = (1, 2, 4, 8)
        cfg.condition_dropout = 0.1
        cfg.returns_condition = False
        cfg.film_mode = 'v1'
        cfg.engine = eng

        # ── the backbone / engine wrapper, per ENGINE_SPEC['wraps_unet'] ──────────────
        if spec['wraps_unet']:
            model_config = utils.Config(
                spec['model'], verbose=False, savepath=None,
                state_dim=visual_spec.TRANSITION_DIM, seq_len=H, freq_dim=32,
                dropout_rate=0.1, device=device, if_vision=True, vis_config=cfg,
                dual_head=True, interval_cfg=False, imf_backbone='unet')
        else:
            model_config = utils.Config(
                spec['model'], verbose=False, savepath=None, config=cfg)

        # ── the engine's kwargs, mirroring the train script's assembly ────────────────
        kwargs = dict(horizon=H, observation_dim=visual_spec.OBS_DIM,
                      action_dim=visual_spec.ACTION_DIM, goal_dim=0, n_timesteps=20,
                      loss_type='l2', clip_denoised=False, predict_epsilon=True,
                      action_weight=1.0)
        if eng == 'diffusion':
            kwargs.update(loss_discount=1.0)
        else:
            kwargs.update(time_beta_alpha_v3=1.5, time_beta_beta_v3=1.0, flow_steps_v3=2,
                          ode_solver_backend_v3='legacy_euler',
                          ode_solver_method_v3='euler')
        if spec['two_time']:
            kwargs.update(if_vision=True, mf_freeze_vision_encoder=False,
                          t_schedule='logit_normal', p_mean=-0.4, p_std=1.0)
        if eng == 'mf':
            kwargs.update(meanflow_data_proportion=0.5, mf_adp_p=1.0, mf_adp_eps=0.01)
        if eng == 'af':
            # The alpha anneal must span the ACTUAL budget; both keys derive from one number
            # here for the same reason the train script derives them from n_train_steps.
            _steps = 100
            kwargs.update(af_ratio_fm=0.5, af_adp_eps=1e-3, af_clamp_utgt=4.0,
                          af_alpha_scheduler='sigmoid', af_alpha_init=1.0, af_alpha_end=0.0,
                          af_alpha_init_step=0, af_alpha_end_step=_steps,
                          af_alpha_gamma=25.0, af_alpha_clamp=0.005,
                          af_n_train_steps=_steps)

        try:
            model = model_config()
            diffusion_config = utils.Config(
                spec['diffusion'], verbose=False, savepath=None, device=device, **kwargs)
            engine = diffusion_config(model)

            trajectories = torch.randn(B, H, visual_spec.TRANSITION_DIM, device=device)
            conditions = {0: torch.randn(B, visual_spec.OBS_DIM, device=device)}
            for key in visual_spec.COND_IMG_KEYS:
                conditions[key] = torch.rand(B, *visual_spec.IMG_SHAPE, device=device)

            loss, infos = engine.loss(trajectories, conditions)
            finite = bool(torch.isfinite(loss).item())
            # A backward pass is part of the question for the two-time arms: the JVP is a
            # FORWARD-mode graph, and whether reverse-mode can then differentiate through it
            # is exactly what a training step needs and what a loss value alone does not show.
            grad_ok = True
            try:
                loss.backward()
                grads = [p.grad for p in engine.parameters() if p.grad is not None]
                grad_ok = bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
            except Exception as e:
                print(f'       backward failed: {type(e).__name__}: {e}')
                grad_ok = False
            good = finite and grad_ok
            print(f'  {"ok  " if good else "FAIL"} {eng:<9} loss={loss.item():.4f}  '
                  f'grads={"finite" if grad_ok else "BAD"}  '
                  f'(cameras={visual_spec.N_CAMERAS}, traj_dim={visual_spec.TRANSITION_DIM})')
            ok &= good
        except Exception as e:
            import traceback
            print(f'  FAIL {eng:<9} {type(e).__name__}: {e}')
            traceback.print_exc()
            ok = False

    print(f'  A7 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A8 — HardFlow host policy + the fm raw-image path
# ══════════════════════════════════════════════════════════════════════════════

def gate_a8(device='cpu'):
    """'diffusion' is refused as a HardFlow host; fm gets images, mf/af get a latent.

    The fm branch is a Gen16 fix, not a port: Gen14's `encode_visual_cond` called
    `model._encode_once(...)` unconditionally, but only the TWO-TIME wrappers define that
    method — so arm C on the fm host would have raised AttributeError on its first step.
    """
    print('\n=== A8: hardflow hosts + fm raw-image path ===')
    import torch
    from mix_visual_avoiding.models import visual_spec
    from mix_visual_avoiding.sampling.hardflow_projection import (
        encode_visual_cond, resolve_engine_hf, ENGINE_INIT_NOISE, ENGINE_TWO_TIME)

    ok = True

    # 1. the DDPM arm has no velocity field, so it is not a host
    try:
        resolve_engine_hf('diffusion')
        print('  FAIL resolve_engine_hf accepted the diffusion arm (no velocity field exists)')
        ok = False
    except ValueError:
        print('  ok   diffusion refused as a HardFlow host')

    # 2. the three flow arms are hosts, with the noise scale read off their OWN samplers
    for eng, want_noise, want_two_time in (('fm', 0.5, False), ('mf', 1.0, True),
                                           ('af', 1.0, True)):
        noise, two_time = resolve_engine_hf(eng)
        good = (noise == want_noise == ENGINE_INIT_NOISE[eng]
                and two_time == want_two_time == ENGINE_TWO_TIME[eng])
        print(f'  {"ok  " if good else "FAIL"} {eng:<3} init_noise={noise} two_time={two_time}')
        ok &= good

    # 3. the cond repack, on stubs (no GPU, no real backbone)
    B, T = 2, 1
    cams = tuple(torch.rand(B, T, *visual_spec.IMG_SHAPE) for _ in range(visual_spec.N_CAMERAS))
    obs_seq = torch.randn(B, T, visual_spec.OBS_DIM)
    cond = {0: visual_spec.pack_visual(cams, obs_seq)}

    class _TwoTimeStub:
        def _encode_once(self, *imgs):
            assert len(imgs) == visual_spec.N_CAMERAS
            return torch.zeros(B, visual_spec.LATENT_DIM)

    class _SingleTimeStub:   # no _encode_once — this IS the fm arm's shape
        pass

    out_two = encode_visual_cond(_TwoTimeStub(), cond)
    good = ('visual_latent' in out_two and out_two['visual_latent'].shape[-1]
            == visual_spec.LATENT_DIM and out_two[0].shape == (B, visual_spec.OBS_DIM))
    print(f'  {"ok  " if good else "FAIL"} two-time host -> visual_latent '
          f'{tuple(out_two.get("visual_latent", torch.empty(0)).shape)}')
    ok &= good

    out_one = encode_visual_cond(_SingleTimeStub(), cond)
    good = ('visual' in out_one and len(out_one['visual']) == visual_spec.N_CAMERAS + 1
            and 'visual_latent' not in out_one)
    print(f'  {"ok  " if good else "FAIL"} single-time host -> raw images '
          f'({len(out_one.get("visual", ()))}-tuple)')
    ok &= good

    # an already-encoded cond passes through untouched
    same = encode_visual_cond(_TwoTimeStub(), out_two)
    good = (same is out_two)
    print(f'  {"ok  " if good else "FAIL"} already-encoded cond passes through')
    ok &= good

    print(f'  A8 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# A9 — eval yaml <-> config agreement
# ══════════════════════════════════════════════════════════════════════════════

def _parse_scalar(text):
    text = text.strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _yaml_lite(path):
    """A deliberately tiny reader for the two scalars this gate needs.

    The AI-coding container has no pyyaml, and A9 is the gate most worth being able to run
    BEFORE submitting a job — a fan mismatch here silently voids every arm-B-vs-arm-C timing
    number (B4_PARITY). So it reads the two keys by hand rather than not running at all.
    """
    top, hardflow = {}, {}
    section = None
    with open(path) as f:
        for raw in f:
            line = raw.split('#', 1)[0].rstrip()
            if not line.strip():
                continue
            if not line[0].isspace():
                section = None
                if line.rstrip().endswith(':') and ':' in line:
                    key = line.split(':', 1)[0].strip()
                    if key == 'hardflow':
                        section = 'hardflow'
                    continue
                if ':' in line:
                    k, v = line.split(':', 1)
                    if v.strip():
                        top[k.strip()] = _parse_scalar(v)
            elif section == 'hardflow' and ':' in line:
                k, v = line.split(':', 1)
                if v.strip():
                    hardflow[k.strip()] = _parse_scalar(v)
    return top, hardflow


def gate_a9(**_):
    """Arm C's fan matches arm B's, and the two activation gates are what they claim."""
    print('\n=== A9: eval yaml <-> config agreement ===')
    yaml_path = os.path.join(REPO, 'config', 'visual_avoiding_mix_eval.yaml')
    top, hf = _yaml_lite(yaml_path)

    # The plan block's mpc_batch_size, read WITHOUT importing torch: a plain regex over the
    # config module would be fragile, so the value is read from _mix_plan_common's literal.
    cfg_src = open(os.path.join(REPO, 'config', 'avoiding-d3il-visual-mix.py')).read()
    m = re.search(r"'mpc_batch_size':\s*(\d+)", cfg_src)
    cfg_fan = int(m.group(1)) if m else None

    checks = [
        (f'arm-C fan (hardflow.batch_size={hf.get("batch_size")}) == arm-B fan '
         f'(mpc_batch_size={cfg_fan})',
         hf.get('batch_size') == cfg_fan),
        (f'arm-C gate (activation_threshold={hf.get("activation_threshold")}) == arm-B gate '
         f'(diffusion_timestep_threshold={top.get("diffusion_timestep_threshold")})',
         hf.get('activation_threshold') == top.get('diffusion_timestep_threshold')),
        (f'replan_steps={top.get("replan_steps")} is the historic default (1)',
         top.get('replan_steps') == 1),
        ('the yaml declares a hardflow section', bool(hf)),
    ]
    ok = True
    for label, passed in checks:
        print(f'  {"ok  " if passed else "WARN" if "gate" in label else "FAIL"} {label}')
        ok &= bool(passed)
    if not checks[1][1]:
        print('       (a deliberate threshold sweep sets these apart — but then run it with '
              'DPCC_THRESHOLD / HFFM_ACT_THRESHOLD so both land in the results path)')
    print(f'  A9 {"PASS" if ok else "FAIL"}')
    return ok


# ══════════════════════════════════════════════════════════════════════════════

GATES = {'a0': gate_a0, 'a1': gate_a1, 'a2': gate_a2, 'a3': gate_a3, 'a4': gate_a4,
         'a5': gate_a5, 'a6': gate_a6, 'a7': gate_a7, 'a8': gate_a8, 'a9': gate_a9}
NEEDS_TORCH = {'a3', 'a4', 'a5', 'a6', 'a7', 'a8'}
NEEDS_GPU   = {'a5', 'a7'}
OFFLINE     = [g for g in GATES if g not in NEEDS_TORCH]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', default='all',
                    choices=['all', 'offline', 'static'] + list(GATES),
                    help="'offline' = stdlib only (runs in the AI container); "
                         "'static' = everything that needs no GPU")
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    if a.gate == 'all':
        names = list(GATES)
    elif a.gate == 'offline':
        names = OFFLINE
    elif a.gate == 'static':
        names = [g for g in GATES if g not in NEEDS_GPU]
    else:
        names = [a.gate]

    results = {}
    for name in names:
        try:
            results[name] = GATES[name](device=a.device)
        except Exception as e:      # a gate that crashes is a gate that failed
            print(f'  {name.upper()} ERROR: {type(e).__name__}: {e}')
            results[name] = False

    print('\n' + '=' * 60)
    for name, passed in results.items():
        print(f'  {name.upper()}: {"PASS" if passed else "FAIL"}')
    print('=' * 60)
    sys.exit(0 if all(results.values()) else 1)
