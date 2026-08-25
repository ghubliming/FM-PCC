#!/usr/bin/env python3
"""Gen14 gates — run these on the cluster BEFORE believing any Visual-Mix-ML number.

    python -m mix_visual_aligning_test.gates_mix_visual --gate all
    python -m mix_visual_aligning_test.gates_mix_visual --gate g0      # no torch needed

G0  copy fidelity          every verbatim/sed file still matches its source
G1  reference-arm identity diffusion/fm still resolve to Gen6V4/Gen7 classes and Gen7's Trainer
G2  JVP survives vision    one mf training step, if_vision=True, no NotImplementedError
G3  MeanFlow identity h=0   u_target == v_inst when every sample is an FM anchor
G4  alpha spans the budget  alpha(0)~1, alpha(N)~0, monotone
G5  alpha->0 == MeanFlow    af with alpha pinned to 0 matches the mf objective
G6  projector fires at K=1  the DPCC projection is not silently skipped at 1 NFE
G7  film_mode=v2 everywhere all four arms build at v2; two-time keeps h_mlp; JVP survives

G0 needs no torch at all. G1 and G4 need torch but no GPU. G6 builds small state-only
models and runs on CPU. G2/G3/G5/G7 build visual models and need a GPU (`--gate static`
runs everything except those four).

Read `raw_mse_u`, never `diffusion_loss`: the adaptive loss sits at its ceiling by
construction and says nothing about convergence.

fix_2 (2026-08-01): G6 was a substring search that could never fail — see its docstring.
It is now a runtime spy-projector test.
"""

import argparse
import os
import subprocess
import sys
import difflib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── G0: the copy ledger ───────────────────────────────────────────────────────
# (gen14_path, source_path, source_package_name)
# 'source_package_name' is sed-reversed before comparing, so a pure rename shows as clean.
VERBATIM = [
    # ── from Gen7 (fm_visual_aligning) — the frame + the fm arm ──
    ('mix_visual_aligning/models/helpers.py',              'fm_visual_aligning/models/helpers.py',              'fm_visual_aligning'),
    ('mix_visual_aligning/models/unet1d_temporal_cond.py', 'fm_visual_aligning/models/unet1d_temporal_cond.py', 'fm_visual_aligning'),
    ('mix_visual_aligning/models/unet1d_temporal_film.py', 'fm_visual_aligning/models/unet1d_temporal_film.py', 'fm_visual_aligning'),
    ('mix_visual_aligning/models/visual_unet.py',          'fm_visual_aligning/models/visual_unet.py',          'fm_visual_aligning'),
    ('mix_visual_aligning/models/fm_diffusion.py',         'fm_visual_aligning/models/diffusion.py',            'fm_visual_aligning'),
    ('mix_visual_aligning/sampling/projection.py',         'fm_visual_aligning/sampling/projection.py',         'fm_visual_aligning'),
    ('mix_visual_aligning/datasets/sequence.py',           'fm_visual_aligning/datasets/sequence.py',           'fm_visual_aligning'),
    ('mix_visual_aligning/datasets/normalization.py',      'fm_visual_aligning/datasets/normalization.py',      'fm_visual_aligning'),
    ('mix_visual_aligning/utils/arrays.py',                'fm_visual_aligning/utils/arrays.py',                'fm_visual_aligning'),
    ('mix_visual_aligning/utils/serialization.py',         'fm_visual_aligning/utils/serialization.py',         'fm_visual_aligning'),
    ('mix_visual_aligning/utils/setup.py',                 'fm_visual_aligning/utils/setup.py',                 'fm_visual_aligning'),
    ('mix_visual_aligning/utils/config.py',                'fm_visual_aligning/utils/config.py',                'fm_visual_aligning'),
    # ── from Gen6V4 (diffuser_visual_aligning) — the diffusion arm ──
    ('mix_visual_aligning/models/diffusion.py',                'diffuser_visual_aligning/models/diffusion.py',                'diffuser_visual_aligning'),
    ('mix_visual_aligning/models/visual_gaussian_diffusion.py', 'diffuser_visual_aligning/models/visual_gaussian_diffusion.py', 'diffuser_visual_aligning'),
    # ── from Gen3v6 (flow_matcher_v3_meanflow) — the mf arm ──
    ('mix_visual_aligning/models/mf_diffusion.py',              'flow_matcher_v3_meanflow/models/mf_diffusion.py',              'flow_matcher_v3_meanflow'),
    ('mix_visual_aligning/models/mlp.py',                       'flow_matcher_v3_meanflow/models/mlp.py',                       'flow_matcher_v3_meanflow'),
    # ── from Gen3v7 (flow_matcher_v3_alphaflow) — the af arm + the two-time trainer ──
    ('mix_visual_aligning/models/af_diffusion.py',        'flow_matcher_v3_alphaflow/models/af_diffusion.py',        'flow_matcher_v3_alphaflow'),
]

# ── Gen14 U8 ── grafted files that STILL have a live upstream to check against.
#
# The four DiT/SiT bones were VERBATIM copies until U8 added the visual prefix token. A
# plain existence entry (GRAFTED below) would drop them out of G0's coverage entirely,
# which is the wrong trade: their upstreams (Gen3v6 / Gen3v7) are still actively edited,
# and a divergence in the transformer internals is exactly what G0 exists to catch.
#
# So they keep a real check, weakened only where U8 needed it: the graft must remain
# ADDITIVE. `removed` is the number of source lines U8 legitimately rewrote —
#   RoPE bones (mf_dit/af_dit): 4 = prefix_tokens sum, _build_sequence signature,
#                                   2-line forward docstring+call
#   adaLN bones (official/sit): 3 = num_tokens/pos_embed sizing and the forward hunk
# If that count moves, either someone edited a Gen14 bone by hand or upstream changed
# underneath it. Both mean: re-open the plan, do not bump the number.
# (gen14_path, source_path, source_package, removed_lines, why)
GRAFTED_DIFF = [
    ('mix_visual_aligning/models/mf_dit_trajectory.py',
     'flow_matcher_v3_meanflow/models/mf_dit_trajectory.py', 'flow_matcher_v3_meanflow', 4,
     'Gen3v6 + U8 cond_dim visual PREFIX TOKEN (RoPE bone)'),
    ('mix_visual_aligning/models/mf_dit_official_trajectory.py',
     'flow_matcher_v3_meanflow/models/mf_dit_official_trajectory.py', 'flow_matcher_v3_meanflow', 3,
     'Gen3v6 + U8 cond_dim visual token prepended before pos_embed (adaLN bone)'),
    ('mix_visual_aligning/models/af_dit_trajectory.py',
     'flow_matcher_v3_alphaflow/models/af_dit_trajectory.py', 'flow_matcher_v3_alphaflow', 4,
     'Gen3v7 + U8 cond_dim visual PREFIX TOKEN (RoPE bone, same graft as mf)'),
    ('mix_visual_aligning/models/af_sit_trajectory.py',
     'flow_matcher_v3_alphaflow/models/af_sit_trajectory.py', 'flow_matcher_v3_alphaflow', 3,
     'Gen3v7 + U8 cond_dim visual token; pos_embed stays frozen sincos (SiT fidelity)'),
    # ── Fix_10 ── the two trainers. These were VERBATIM until Fix_10; they are moved here
    # rather than to GRAFTED (which drops a file out of G0's coverage entirely) because the
    # upstreams are actively edited and the training loop is exactly what G0 must keep
    # watching. The graft is ADDITIVE and the 3 rewritten lines are enumerated below, so a
    # 4th means someone changed something that is not Fix_10.
    #
    #   1x  self.save_freq = n_train_steps // 5      -> honours the save_freq argument
    #   2x  torch.save(<payload>, savepath)          -> _atomic_torch_save(...)
    #
    # Everything else Fix_10 adds is insertion only: the `save_freq=None` kwarg and the
    # _atomic_torch_save helper. See logs_in_develop/Gen14/Fix_10/.
    ('mix_visual_aligning/utils/training.py',
     'fm_visual_aligning/utils/training.py', 'fm_visual_aligning', 3,
     'Gen7 + Fix_10 save_freq knob and atomic checkpoint writes'),
    ('mix_visual_aligning/utils/training_twotime.py',
     'flow_matcher_v3_alphaflow/utils/training.py', 'flow_matcher_v3_alphaflow', 3,
     'Gen3v7 + Fix_10 save_freq knob and atomic checkpoint writes'),
]

# Files that are deliberately grafted — a diff here is EXPECTED. Listed so the ledger is
# complete and nobody mistakes an unlisted file for an audited one.
GRAFTED = {
    'mix_visual_aligning/models/unet1d_twotime_cond.py':
        'Gen3v6 Flow_matcher_U_Net_v2 + Gen7 cond_mlp block (2 additive hunks)',
    'mix_visual_aligning/models/mf_trajectory_model.py':
        'Gen3v6 + if_vision/vis_config visual branch',
    'mix_visual_aligning/models/af_trajectory_model.py':
        'Gen3v7 + if_vision/vis_config visual branch (same graft, copied)',
    'mix_visual_aligning/models/mf_engine.py':  'Gen3v6 + if_vision/vis_config passthrough',
    'mix_visual_aligning/models/af_engine.py':  'Gen3v7 + if_vision/vis_config passthrough',
}
NEW_FILES = [
    'mix_visual_aligning/models/engine_registry.py',
    'mix_visual_aligning/models/visual_unet_twotime.py',
    # U5 — two-time TRUE-FiLM backbone. Imports Gen7's verbatim FiLMResidualTemporalBlock
    # rather than reimplementing it, so both v2 arms share one FiLM definition.
    'mix_visual_aligning/models/unet1d_twotime_film.py',
    'mix_visual_aligning/models/visual_mf_diffusion.py',
    'mix_visual_aligning/models/visual_af_diffusion.py',
    'mix_visual_aligning/models/__init__.py',
]


def _norm(path, src_pkg):
    """Read a file, strip CR, and reverse the package rename so a pure copy compares equal."""
    with open(os.path.join(REPO, path), 'rb') as f:
        text = f.read().decode('utf-8')
    return text.replace('\r\n', '\n').replace('mix_visual_aligning', src_pkg)


def gate_g0(verbose=True):
    """Every verbatim/sed file still matches its source, modulo the package rename."""
    print('\n=== G0: copy fidelity ===')
    failures = []
    for dst, src, pkg in VERBATIM:
        dst_abs, src_abs = os.path.join(REPO, dst), os.path.join(REPO, src)
        if not os.path.exists(dst_abs):
            failures.append(f'{dst}: MISSING in Gen14'); continue
        if not os.path.exists(src_abs):
            failures.append(f'{src}: MISSING source (upstream moved?)'); continue
        with open(src_abs, 'rb') as f:
            src_text = f.read().decode('utf-8').replace('\r\n', '\n')
        if _norm(dst, pkg) != src_text:
            failures.append(f'{dst}: DIFFERS from {src}')
        elif verbose:
            print(f'  ok   {dst}')
    # U8 — the additive-graft check (see GRAFTED_DIFF).
    print(f'\n  grafted, additive-only (U8 bones, {len(GRAFTED_DIFF)} files):')
    for dst, src, pkg, want_removed, why in GRAFTED_DIFF:
        dst_abs, src_abs = os.path.join(REPO, dst), os.path.join(REPO, src)
        if not os.path.exists(dst_abs):
            failures.append(f'{dst}: MISSING in Gen14'); continue
        if not os.path.exists(src_abs):
            failures.append(f'{src}: MISSING source (upstream moved?)'); continue
        with open(src_abs, 'rb') as f:
            src_lines = f.read().decode('utf-8').replace('\r\n', '\n').splitlines()
        dst_lines = _norm(dst, pkg).splitlines()
        sm = difflib.SequenceMatcher(None, src_lines, dst_lines, autojunk=False)
        removed = added = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'delete'):
                removed += i2 - i1
            if tag in ('replace', 'insert'):
                added += j2 - j1
        if removed != want_removed:
            failures.append(
                f'{dst}: graft is no longer additive — {removed} source lines removed/rewritten, '
                f'expected {want_removed}. Either a Gen14 bone was hand-edited or {src} '
                f'changed upstream. Diff them before touching this number.')
            continue
        print(f'    ok   {dst}  <- {why}  (+{added} lines, -{removed})')
    print(f'\n  grafted (diff expected, {len(GRAFTED)} files):')
    for f_, why in GRAFTED.items():
        mark = 'ok  ' if os.path.exists(os.path.join(REPO, f_)) else 'MISS'
        print(f'    {mark} {f_}  <- {why}')
    print(f'  new ({len(NEW_FILES)} files):')
    for f_ in NEW_FILES:
        mark = 'ok  ' if os.path.exists(os.path.join(REPO, f_)) else 'MISS'
        print(f'    {mark} {f_}')
    if failures:
        print('\n  G0 FAIL — the copy assumption broke. Re-open the plan; do NOT patch over it:')
        for f_ in failures:
            print(f'    ! {f_}')
        return False
    print(f'\n  G0 PASS — {len(VERBATIM)} verbatim files match their sources.')
    return True


def gate_g1():
    """diffusion/fm resolve to Gen6V4/Gen7 classes and Gen7's Trainer; mf/af to the two-time one.

    This is the STRUCTURAL half of the reference-arm guarantee (PLAN §3.1). The numerical
    half — 50 training steps compared against Gen7 — needs a GPU and is driven by the
    sbatch wrapper, not from here.
    """
    print('\n=== G1: reference-arm wiring ===')
    from mix_visual_aligning.models.engine_registry import ENGINES, resolve
    expected = {
        'diffusion': ('VisualGaussianDiffusion', 'VisualUNet',       'training.Trainer',          False),
        'fm':   ('VisualFlowMatching',      'VisualUNet',       'training.Trainer',          False),
        'mf':   ('VisualMeanFlow',          'MeanFlowEngine',   'training_twotime.Trainer',  True),
        'af':   ('VisualAlphaFlow',         'AlphaFlowEngine',  'training_twotime.Trainer',  True),
    }
    ok = True
    for eng, (diff_cls, model_cls, trainer, two_time) in expected.items():
        spec = resolve(eng)
        checks = [
            (spec['diffusion'].endswith(diff_cls), f"diffusion={spec['diffusion']}"),
            (spec['model'].endswith(model_cls),    f"model={spec['model']}"),
            (spec['trainer'].endswith(trainer),    f"trainer={spec['trainer']}"),
            (spec['two_time'] == two_time,         f"two_time={spec['two_time']}"),
            (spec['wraps_unet'] == two_time,       f"wraps_unet={spec['wraps_unet']}"),
        ]
        bad = [msg for good, msg in checks if not good]
        if bad:
            ok = False
            print(f'  FAIL {eng}: {"; ".join(bad)}')
        else:
            print(f'  ok   {eng}: {spec["label"]}')
    # The diffusion/fm arms must not reach any Gen14-authored module.
    for eng in ('diffusion', 'fm'):
        spec = ENGINES[eng]
        for key in ('diffusion', 'model', 'trainer'):
            if 'twotime' in spec[key]:
                ok = False
                print(f'  FAIL {eng}.{key} reaches a two-time module ({spec[key]}) — '
                      f'this breaks the PLAN §3.1 structural guarantee.')
    print('  G1 PASS' if ok else '  G1 FAIL')
    return ok


def _build(engine, if_vision=True, horizon=8, batch=2, device='cuda', film_mode='v1',
           ml_bone='unet', dit_hidden_size=160, dit_depth=8,
           vis_pretrained=False, vis_cond_mode='token'):
    """Minimal in-memory build of one arm — no dataset, no checkpoint.

    Gen14 U8: `ml_bone` selects the generative backbone ('unet' | 'mf_dit' | 'sit' | 'dit').
    On a transformer bone `film_mode` is not set on the cfg at all, mirroring what
    `_mix_bone_keys()` does in the real config — FiLM is a U-Net concept.
    """
    import torch
    from mix_visual_aligning.models.engine_registry import resolve, import_class

    class _Cfg:  # stand-in for the parsed args object VisualUNet* reads
        pass
    cfg = _Cfg()
    cfg.device, cfg.if_vision, cfg.horizon = device, if_vision, horizon
    cfg.action_dim, cfg.obs_dim, cfg.dim = 3, 6, 32
    cfg.dim_mults, cfg.condition_dropout, cfg.returns_condition = (1, 2, 4, 8), 0.1, False
    # ── Gen14 U9 ── both default to the pre-U9 value, so every U8 gate that calls _build()
    # without them builds exactly the model it built before.
    cfg.vis_pretrained = bool(vis_pretrained)
    cfg.vis_cond_mode = str(vis_cond_mode)
    bone_kw = {}
    if ml_bone == 'unet':
        cfg.film_mode = film_mode
    else:
        # 🔴 The geometry MUST travel as engine kwargs, not as cfg attributes.
        # VisualDiTTwoTime._knob() gives the engine-passed value precedence (single source
        # of truth = the config block -> train script -> engine chain), and the engine's own
        # default is the STATE-ONLY 256. Setting only `cfg.dit_hidden_size` therefore loses
        # to that default and silently builds a 2.5x model — which is exactly what the
        # 2026-08-21 gate run produced. Mirror train_mix_visual_aligning.py:418-422 instead.
        bone_kw = dict(dit_hidden_size=dit_hidden_size, dit_depth=dit_depth,
                       dit_num_heads=4, dit_patch_size=1)
        for k, v in bone_kw.items():
            setattr(cfg, k, v)          # kept in sync so the cfg fallback can never disagree

    spec = resolve(engine)
    ModelCls, DiffCls = import_class(spec['model']), import_class(spec['diffusion'])
    obs_dim = 6 if if_vision else 20
    model = ModelCls(state_dim=3 + obs_dim, seq_len=horizon, freq_dim=32,
                     dropout_rate=0.1, device=device, if_vision=if_vision, vis_config=cfg,
                     dual_head=True, interval_cfg=False, imf_backbone=ml_bone, **bone_kw)
    return cfg, model, DiffCls, obs_dim


def _vnet(m):
    """The velocity network, whatever wrapper `_build` handed back.

    mf/af are `wraps_unet=True` arms: `_build` returns the ENGINE (MeanFlowEngine /
    AlphaFlowEngine), and the trajectory model that owns `velocity_net` sits one level
    down at `.model`. Reaching for `engine.velocity_net` is what made G-B2/G-B3 die with
    an AttributeError on 2026-08-21.
    """
    if hasattr(m, 'velocity_net'):
        return m.velocity_net
    return m.model.velocity_net


# ══════════════════════════════════════════════════════════════════════════════
# Gen14 U8 — ML-bone gates (visual DiT/SiT). Plan: logs_in_develop/Gen14/U8/
# ══════════════════════════════════════════════════════════════════════════════

# (arm, ml_bone) pairs that must work after U8.
_U8_BONES = (('mf', 'mf_dit'), ('mf', 'dit'), ('af', 'sit'), ('af', 'dit'))

# module -> class, for the state-only regression check
_U8_BONE_CLASSES = (
    ('mix_visual_aligning.models.mf_dit_official_trajectory', 'MFDiTOfficialTrajectory'),
    ('mix_visual_aligning.models.af_sit_trajectory',          'AFSiTTrajectory'),
    ('mix_visual_aligning.models.mf_dit_trajectory',          'MFDiTTrajectory'),
    ('mix_visual_aligning.models.af_dit_trajectory',          'AFDiTTrajectory'),
)


def gate_gb1():
    """G-B1 — at cond_dim=0 every bone is byte-identical to its pre-U8 self.

    The four transformer ports are shared with the STATE-ONLY generations (Gen3v4/v6/v7).
    U8 must be provably additive there: no new parameter, no changed shape, when the
    visual token is off. CPU-only, no vision encoder built.
    """
    print('\n=== G-B1: cond_dim=0 leaves every bone unchanged ===')
    import importlib, torch
    ok = True
    for mod, cls_name in _U8_BONE_CLASSES:
        Cls = getattr(importlib.import_module(mod), cls_name)
        m0 = Cls(horizon=8, transition_dim=6, hidden_size=64, depth=2, num_heads=4)
        keys0 = {k: tuple(v.shape) for k, v in m0.state_dict().items()}
        new_keys = [k for k in keys0 if 'vis' in k.lower()]
        if new_keys:
            ok = False
            print(f'  FAIL {cls_name}: state-only build leaked visual params {new_keys}')
            continue
        if getattr(m0, 'use_visual', False):
            ok = False
            print(f'  FAIL {cls_name}: use_visual is True at cond_dim=0')
            continue
        m1 = Cls(horizon=8, transition_dim=6, hidden_size=64, depth=2, num_heads=4, cond_dim=128)
        keys1 = {k: tuple(v.shape) for k, v in m1.state_dict().items()}
        added = set(keys1) - set(keys0)
        changed = {k for k in set(keys0) & set(keys1) if keys0[k] != keys1[k]}
        # pos_embed legitimately grows by one row on the adaLN pair (it owns the visual
        # position); the RoPE pair keeps its table in a persistent=False buffer.
        if changed - {'pos_embed'}:
            ok = False
            print(f'  FAIL {cls_name}: cond_dim>0 changed non-visual shapes {changed}')
            continue
        print(f'  ok   {cls_name}: {len(keys0)} params state-only; '
              f'+{len(added)} visual ({sorted(added)}), grew={sorted(changed)}')
    print('  G-B1 PASS' if ok else '  G-B1 FAIL')
    return ok


def gate_gb6():
    """G-B6 — the prefix/RoPE bookkeeping, the one silent failure mode of Option 2.

    On the RoPE bones `prefix_tokens` strips the prefix before the u/v heads and the RoPE
    table is sized from `prefix_tokens + num_patches`. A half-applied +1 trains fine and
    reads the WRONG positions. On the adaLN bones the same risk lives in `pos_embed`.
    Both are checked here, plus the only thing that ultimately matters: output shape.
    """
    print('\n=== G-B6: prefix / pos-embed bookkeeping ===')
    import importlib, torch
    ok, H, D, B = True, 8, 9, 2
    for mod, cls_name in _U8_BONE_CLASSES:
        Cls = getattr(importlib.import_module(mod), cls_name)
        for cond_dim in (0, 128):
            m = Cls(horizon=H, transition_dim=D, hidden_size=64, depth=4, num_heads=4,
                    cond_dim=cond_dim)
            n_vis = 1 if cond_dim else 0
            if hasattr(m, 'rope_cos'):                     # RoPE bones
                want = m.prefix_tokens + m.num_patches
                got = m.rope_cos.shape[0]
                if got != want:
                    ok = False
                    print(f'  FAIL {cls_name} cond_dim={cond_dim}: RoPE table {got} != '
                          f'prefix_tokens+num_patches {want} — HALF-APPLIED TOKEN BUMP')
                    continue
                base = 7                                    # class+omega+tmin+tmax+time
                if m.prefix_tokens != base + n_vis:
                    ok = False
                    print(f'  FAIL {cls_name}: prefix_tokens={m.prefix_tokens}, want {base + n_vis}')
                    continue
            else:                                           # adaLN bones
                if m.pos_embed.shape[1] != n_vis + m.num_patches:
                    ok = False
                    print(f'  FAIL {cls_name}: pos_embed len {m.pos_embed.shape[1]} != '
                          f'{n_vis} + {m.num_patches}')
                    continue
            x = torch.randn(B, H, D)
            cond = torch.randn(B, 128) if cond_dim else None
            u, v = m(x, cond, torch.rand(B), h=torch.rand(B), return_v=True)
            if tuple(u.shape) != (B, H, D) or tuple(v.shape) != (B, H, D):
                ok = False
                print(f'  FAIL {cls_name} cond_dim={cond_dim}: output {tuple(u.shape)} != '
                      f'{(B, H, D)} — the prefix was not stripped correctly')
                continue
            print(f'  ok   {cls_name} cond_dim={cond_dim}: shapes clean, out={tuple(u.shape)}')
    print('  G-B6 PASS' if ok else '  G-B6 FAIL')
    return ok


def gate_gb2(device='cuda'):
    """G-B2 — every visual bone CONSTRUCTS, and lands near the U-Net's parameter count.

    🔴 The parameter check is the whole point. `bb_unet_ablation` (2026-07-25) reported a
    DiT beating a U-Net 3.5-7x; the 2026-08-19 STUDY showed its 'U-Net' was the 253 M
    Fix_8 build and retracted the result. An unmatched visual A/B would repeat that.
    The visual bone target is the ~4.0 M VisualUNetTwoTime (dim=32).
    """
    print('\n=== G-B2: visual bones build, parameter-matched ===')
    ok = True
    _, unet_model, _, _ = _build('mf', True, 8, 2, device, ml_bone='unet')
    n_unet = sum(p.numel() for p in _vnet(unet_model).backbone.parameters())
    print(f'  reference: VisualUNetTwoTime bone = {n_unet / 1e6:.2f} M')
    for arm, bone in _U8_BONES:
        try:
            _, model, _, _ = _build(arm, True, 8, 2, device, ml_bone=bone)
        except Exception as e:
            ok = False
            print(f'  FAIL {arm}@{bone}: construction raised {type(e).__name__}: {e}')
            continue
        vnet = _vnet(model)
        if type(vnet).__name__ != 'VisualDiTTwoTime':
            ok = False
            print(f'  FAIL {arm}@{bone}: velocity_net is {type(vnet).__name__}, not VisualDiTTwoTime')
            continue
        if not getattr(vnet.backbone, 'use_visual', False):
            ok = False
            print(f'  FAIL {arm}@{bone}: bone built with cond_dim=0 — it would train IMAGE-BLIND')
            continue
        n = sum(p.numel() for p in vnet.backbone.parameters())
        ratio = n / n_unet
        flag = 'ok  ' if 0.75 <= ratio <= 1.35 else 'WARN'
        if flag == 'WARN':
            ok = False
            print(f'  FAIL {arm}@{bone}: bone {n / 1e6:.2f} M = {ratio:.2f}x the U-Net — '
                  f'NOT parameter-matched. Fix dit_hidden_size (160 is the matched width).')
            continue
        print(f'  {flag} {arm}@{bone}: bone {n / 1e6:.2f} M ({ratio:.2f}x U-Net)')
    print('  G-B2 PASS' if ok else '  G-B2 FAIL')
    return ok


def gate_gb3(device='cuda'):
    """G-B3 — vision is LIVE: the visual latent actually reaches the output.

    The U5 lesson: a zero-initialised conditioning path can look perfectly wired and be
    inert. Construction proving `vis_projector` exists proves nothing.

    🔴 WHY THIS GATE WARMS UP FIRST (2026-08-21, job 24834)
    The original one-step version FAILED on all four bones with "grad is all zero", and it
    was the GATE that was wrong, not the model. Every one of these bones is a DiT: the
    module docstrings say "zero-init final layers", and `initialize_weights()` sets
    `final_layer.linear.weight = 0` plus every `adaLN_modulation[-1] = 0`. At step 0 the
    network therefore outputs exactly 0 for ANY input, and
        dL/d(final-layer input) = W_final^T . dL/dout = 0
    so EVERY parameter upstream of it — vis_projector included — has an exactly-zero
    gradient. That is adaLN-zero working as designed, not an image-blind model. The U-Net
    has no zero-init final layer, which is why G2/G3/G7 never saw this.
    Measuring at step 0 on a zero-init transformer measures nothing. Do not "simplify"
    the warm-up away.

    Two independent checks are made after warm-up, because the gradient test alone cannot
    tell "zero because zero-init" from "zero because disconnected":
      (a) gradient reaches `vis_projector` and the ResNet encoder;
      (b) SENSITIVITY — the backbone's output actually MOVES when the latent changes.
          (b) involves no autograd at all, so it is the check that survives any future
          init convention.
    """
    print('\n=== G-B3: the visual token receives gradient ===')
    import torch
    ok = True
    # Gate-only optimiser settings, chosen to move DECISIVELY off the zero-init: Adam's
    # step is ~lr regardless of gradient scale, so 5 steps at 1e-2 puts the final layer and
    # the adaLN gates around 5e-2. At lr=1e-3 they would sit near 5e-3 and the measured
    # gradient would be ~1e-5 — nonzero, but close enough to underflow that a PASS would be
    # luck. These numbers train nothing; they only make the measurement well-conditioned.
    WARMUP, WARMUP_LR = 5, 1e-2
    for arm, bone in _U8_BONES:
        cfg, model, DiffCls, obs_dim = _build(arm, True, 8, 2, device, ml_bone=bone)
        kw = dict(horizon=8, observation_dim=obs_dim, action_dim=3, goal_dim=0,
                  n_timesteps=100, loss_type='l2', if_vision=True,
                  t_schedule='logit_normal', p_mean=-0.4, p_std=1.0)
        if arm == 'mf':
            kw.update(meanflow_data_proportion=0.5, mf_adp_p=1.0, mf_adp_eps=0.01)
        else:
            kw.update(af_ratio_fm=0.5, af_adp_eps=1e-3, af_clamp_utgt=4.0)
        diffusion = DiffCls(model, **kw).to(device)
        opt = torch.optim.Adam(diffusion.parameters(), lr=WARMUP_LR)

        traj, cond = _fake_visual_batch(2, 8, device)
        for _ in range(WARMUP):                     # break the zero-init, then measure
            opt.zero_grad(set_to_none=True)
            loss, _ = diffusion.loss(traj, cond)
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        loss, _ = diffusion.loss(traj, cond)
        loss.backward()

        vnet = _vnet(model)
        proj = vnet.backbone.vis_projector
        w = proj.linear.weight if hasattr(proj, 'linear') else proj.weight
        g = w.grad
        live = g is not None and float(g.abs().sum()) > 0
        if not live:
            ok = False
            print(f'  FAIL {arm}@{bone}: vis_projector grad is {"None" if g is None else "all zero"} '
                  f'AFTER {WARMUP} warm-up steps — the model is IMAGE-BLIND despite '
                  f'reporting if_vision=True')
            continue
        # the encoder itself must be training end-to-end, as in Gen6V4/Gen7
        enc_g = [p.grad for p in vnet.obs_encoder.parameters() if p.grad is not None]
        enc_live = any(float(x.abs().sum()) > 0 for x in enc_g)
        if not enc_live:
            ok = False
            print(f'  FAIL {arm}@{bone}: the ResNet encoder receives NO gradient — the latent '
                  f'reached the bone detached. Gen14 trains the encoder end-to-end.')
            continue

        # (b) gradient-free sensitivity: same x/t, two different latents -> different output.
        with torch.no_grad():
            B, H = 2, 8
            x = torch.randn(B, H, 9, device=device)
            t = torch.rand(B, device=device)
            h = torch.rand(B, device=device) * 0.1
            l1 = torch.randn(B, vnet.backbone.cond_dim, device=device)
            l2 = torch.randn(B, vnet.backbone.cond_dim, device=device)
            o1 = vnet.backbone(x, l1, t, h=h)
            o2 = vnet.backbone(x, l2, t, h=h)
            o1 = o1[0] if isinstance(o1, tuple) else o1
            o2 = o2[0] if isinstance(o2, tuple) else o2
            delta = float((o1 - o2).abs().max())
        if delta == 0.0:
            ok = False
            print(f'  FAIL {arm}@{bone}: output IDENTICAL for two different visual latents — '
                  f'the token is being discarded downstream.')
            continue
        print(f'  ok   {arm}@{bone}: |grad vis_projector|={float(g.abs().sum()):.3e}, '
              f'encoder trains={enc_live}, d(out)/d(latent) max={delta:.3e}, loss={float(loss):.4f}')
    print('  G-B3 PASS' if ok else '  G-B3 FAIL')
    return ok


def gate_gb45(device='cuda'):
    """G-B4/G-B5 — one loss step per (arm, bone) is FINITE under forward-mode AD.

    mf takes a literal `torch.func.jvp`; af re-enters the backbone for its bootstrap
    target. Both must survive the prepended token. This should hold by construction (the
    latent is a captured constant, softmax/RoPE/adaLN are all forward-AD friendly) — gate
    it anyway, because 'should' is what gates exist to disprove.
    """
    print('\n=== G-B4/5: JVP + bootstrap survive the visual token ===')
    import torch
    ok = True
    for arm, bone in _U8_BONES:
        cfg, model, DiffCls, obs_dim = _build(arm, True, 8, 2, device, ml_bone=bone)
        kw = dict(horizon=8, observation_dim=obs_dim, action_dim=3, goal_dim=0,
                  n_timesteps=100, loss_type='l2', if_vision=True,
                  t_schedule='logit_normal', p_mean=-0.4, p_std=1.0)
        if arm == 'mf':
            kw.update(meanflow_data_proportion=0.5, mf_adp_p=1.0, mf_adp_eps=0.01)
        else:
            kw.update(af_ratio_fm=0.5, af_adp_eps=1e-3, af_clamp_utgt=4.0)
        diffusion = DiffCls(model, **kw).to(device)
        traj, cond = _fake_visual_batch(2, 8, device)
        try:
            loss, info = diffusion.loss(traj, cond)
            loss.backward()
        except Exception as e:
            ok = False
            print(f'  FAIL {arm}@{bone}: loss step raised {type(e).__name__}: {e}')
            continue
        finite = bool(torch.isfinite(loss))
        if not finite:
            ok = False
            print(f'  FAIL {arm}@{bone}: non-finite loss')
            continue
        raw = float(info['raw_mse_u']) if 'raw_mse_u' in info else float('nan')
        print(f'  ok   {arm}@{bone}: loss={float(loss):.6f} finite raw_mse_u={raw:.6f}')
    print('  G-B4/5 PASS' if ok else '  G-B4/5 FAIL')
    return ok


def gate_gb7():
    """G-B7 — path identity: two bones must NOT collide in one checkpoint directory.

    This is the trap CHANGELOG_Gen14_U5...md:208 flagged in advance. It also checks that
    a DiT block carries NO film_mode fragment (FiLM is a U-Net concept; the fragment would
    be a lying directory name).
    """
    print('\n=== G-B7: bone is a checkpoint-path key ===')
    import os, importlib.util, sys
    ok = True
    path = os.path.join('config', 'aligning-d3il-visual.py')

    def _load(env):
        for k in ('MIX_BONE', 'MIX_BONE_MF', 'MIX_BONE_AF'):
            os.environ.pop(k, None)
        os.environ.update(env)
        spec = importlib.util.spec_from_file_location('_cfg_probe', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.base

    try:
        b_unet = _load({})['mix_visual_aligning_mf']
        b_dit = _load({'MIX_BONE_MF': 'mf_dit'})['mix_visual_aligning_mf']
    finally:
        for k in ('MIX_BONE', 'MIX_BONE_MF', 'MIX_BONE_AF'):
            os.environ.pop(k, None)
        sys.modules.pop('_cfg_probe', None)

    # 🔴 The U-Net block must NOT define ml_bone: watch() skips undefined keys, which is what
    # keeps every pre-U8 checkpoint path byte-identical. The DiT block must define it.
    if 'ml_bone' in b_unet:
        ok = False
        print(f"  FAIL the unet block defines ml_bone={b_unet['ml_bone']!r} — that adds a "
              f"'_Bunet' fragment and ORPHANS every existing Gen14 U-Net checkpoint.")
    else:
        print('  ok   unet block omits ml_bone (pre-U8 paths preserved)')
    if b_dit.get('ml_bone') != 'mf_dit':
        ok = False
        print(f"  FAIL the DiT block did not resolve ml_bone: {b_dit.get('ml_bone')!r}")
    if 'film_mode' in b_dit:
        ok = False
        print(f"  FAIL the mf_dit block still carries film_mode={b_dit['film_mode']!r} — "
              f"that fragment would put a lying '_film..' in the DiT checkpoint path.")
    else:
        print('  ok   film_mode absent from the DiT block')
    if 'film_mode' not in b_unet:
        ok = False
        print('  FAIL the unet block LOST film_mode — existing U-Net paths would change.')
    else:
        print(f"  ok   unet block keeps film_mode={b_unet['film_mode']!r}")

    exp_unet, exp_dit = str(b_unet['exp_name']), str(b_dit['exp_name'])
    if exp_unet == exp_dit:
        ok = False
        print('  FAIL both bones produce the SAME exp_name — checkpoints WILL collide.')
    else:
        print(f'  ok   distinct exp_name templates (bone fragment present)')
    print('  G-B7 PASS' if ok else '  G-B7 FAIL')
    return ok





def _fake_visual_batch(batch, horizon, device):
    import torch
    traj = torch.randn(batch, horizon, 9, device=device)
    cond = {
        0: torch.randn(batch, 6, device=device),
        'primary_img': torch.rand(batch, 3, 96, 96, device=device),
        'wrist_img':   torch.rand(batch, 3, 96, 96, device=device),
    }
    return traj, cond


def gate_g2(device='cuda'):
    """One mf step with if_vision=True: the JVP must survive, and must NOT differentiate
    the ResNets (peak memory within 1.3x of the fm arm proves the pre-encode is live)."""
    print('\n=== G2: JVP survives the visual path ===')
    import torch
    horizon, batch = 8, 2
    cfg, model, DiffCls, obs_dim = _build('mf', True, horizon, batch, device)
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        meanflow_data_proportion=0.5, mf_adp_p=1.0, mf_adp_eps=0.01).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    torch.cuda.reset_peak_memory_stats()
    loss, info = diffusion.loss(traj, cond)
    loss.backward()
    peak_mf = torch.cuda.max_memory_allocated() / 2 ** 20
    finite = bool(torch.isfinite(loss))
    raw = float(info['raw_mse_u']) if 'raw_mse_u' in info else float('nan')
    print(f'  loss={float(loss):.6f} finite={finite}  raw_mse_u={raw:.6f}  peak={peak_mf:.0f} MiB')
    ok = finite
    print('  G2 PASS' if ok else '  G2 FAIL — JVP produced a non-finite loss')
    print('  NOTE: compare peak against the fm arm by hand; >1.3x means the pre-encoded '
          'latent (PLAN §6.1) is NOT live and the ResNets are inside the JVP.')
    return ok


def gate_g3(device='cuda'):
    """meanflow_data_proportion=1.0 forces every sample to h=0, where the MeanFlow target
    collapses to the FM velocity. raw_mse_u must then match a plain FM residual."""
    print('\n=== G3: MeanFlow identity at h=0 ===')
    import torch
    horizon, batch = 8, 4
    cfg, model, DiffCls, obs_dim = _build('mf', True, horizon, batch, device)
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        meanflow_data_proportion=1.0,   # ← every sample is an FM anchor
                        mf_adp_p=1.0, mf_adp_eps=0.01).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    loss, info = diffusion.loss(traj, cond)
    h_mean   = float(info['h_mean'])   if 'h_mean'   in info else float('nan')
    fm_frac  = float(info['fm_frac'])  if 'fm_frac'  in info else float('nan')
    ok = (abs(h_mean) < 1e-6) and (abs(fm_frac - 1.0) < 1e-6)
    print(f'  h_mean={h_mean:.3e} (want 0)   fm_frac={fm_frac:.3f} (want 1.0)')
    print('  G3 PASS' if ok else '  G3 FAIL — h is not pinned to 0; the anchor branch is wrong')
    return ok


def gate_g4():
    """alpha(0)~1, alpha(N)~0, monotone. A run whose alpha never moved trained plain flow
    matching and is otherwise indistinguishable from a working one."""
    print('\n=== G4: alpha spans the real budget ===')
    from mix_visual_aligning.models.af_diffusion import AlphaFlowODE
    n = 100000
    vals = []
    for step in (0, n // 4, n // 2, 3 * n // 4, n):
        a = AlphaFlowODE._get_ratio('sigmoid', 1.0, 0.0, 0, n, 25.0, 0.005, step)
        vals.append((step, float(a)))
    for step, a in vals:
        print(f'  step {step:>7} -> alpha {a:.4f}')
    alphas = [a for _s, a in vals]
    monotone = all(alphas[i] >= alphas[i + 1] - 1e-9 for i in range(len(alphas) - 1))
    ok = abs(alphas[0] - 1.0) < 1e-3 and abs(alphas[-1]) < 1e-3 and monotone
    print('  G4 PASS' if ok else '  G4 FAIL — alpha does not span [1,0] monotonically')
    return ok


def gate_g5(device='cuda'):
    """alpha pinned to 0 makes alpha-Flow's objective the MeanFlow objective (PLAN §3.4)."""
    print('\n=== G5: alpha->0 limit is MeanFlow ===')
    import torch
    horizon, batch = 8, 4
    cfg, model, DiffCls, obs_dim = _build('af', True, horizon, batch, device)
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        af_ratio_fm=0.5, af_adp_eps=1e-3, af_clamp_utgt=4.0,
                        af_alpha_scheduler='constant',
                        af_alpha_init=0.0, af_alpha_end=0.0).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    loss, info = diffusion.loss(traj, cond)
    alpha = float(info['alpha']) if 'alpha' in info else float('nan')
    disc  = float(info['discrete_frac']) if 'discrete_frac' in info else float('nan')
    ok = abs(alpha) < 1e-9 and abs(disc) < 1e-9 and bool(torch.isfinite(loss))
    print(f'  alpha={alpha:.6f} (want 0)   discrete_frac={disc:.6f} (want 0 -> JVP branch only)')
    print('  G5 PASS' if ok else '  G5 FAIL — the bootstrap branch is still active at alpha=0')
    return ok


class _SpyProjector:
    """Records whether the sampler actually called project() — the whole point of G6.

    Mimics the surface `Projector` exposes to p_sample_loop: `.gradient`,
    `.diffusion_timestep_threshold`, `.project()`. `gradient=False` selects the
    SLSQP branch, which is the one the eval pipeline uses.
    """

    def __init__(self, threshold):
        self.diffusion_timestep_threshold = threshold
        self.gradient = False
        self.n_calls = 0
        self.called_at = []

    def project(self, trajectory, constraints=None):
        self.n_calls += 1
        return trajectory, 0.0          # identity projection: we only count the call

    def compute_cost(self, trajectory, constraints=None):
        return 0.0


def _eval_threshold(default=0.5):
    """The threshold the eval pipeline actually deploys, read from the same YAML the
    config block reads. Hard-coding it here would let the gate drift away from reality."""
    path = os.path.join(REPO, 'config', 'visual_aligning_eval.yaml')
    try:
        with open(path) as f:
            for line in f:
                if line.strip().startswith('diffusion_timestep_threshold:'):
                    return float(line.split(':', 1)[1].strip())
    except Exception:
        pass
    return default


def gate_g6(device='cpu'):
    """🔴 RUNTIME check: at K=1, does the DPCC projector actually get called?

    ── fix_2 ──────────────────────────────────────────────────────────────────
    The original G6 was a substring search for 'flow_steps - 1' over the whole
    module. That string also appears in the unrelated `repeat_last` clamp
    (`loop_idx = min(i, flow_steps - 1)`) which is present in ALL THREE engines, so
    the check could never fail: mf/af passed for the wrong reason and fm was
    reported as having a fallback it does not have. Cluster run 24082 printed
    `DIFF fm: terminal-step fallback present` — a false pass.

    Replaced with a behavioural test: run p_sample_loop at K=1 with a spy projector
    and count the calls. No source-string heuristics, nothing to drift.
    ───────────────────────────────────────────────────────────────────────────

    EXPECTED RESULT (with the deployed threshold of 0.5):
      mf, af -> project() IS called. Gen3v6's guard is
                `(loop_idx >= int((1-thr)*K)) or (loop_idx == K-1)`; at K=1 both
                the int() truncation and the explicit fallback fire.
      fm     -> project() is NOT called. Gen7's guard is only
                `loop_idx >= (1-thr)*K` = `0 >= 0.5` -> False. The DPCC projection
                is skipped ENTIRELY and the run silently reports FM as unsafe when
                nothing was ever projected.

    The fm leg does NOT fail this gate: it is a known Gen7/Gen6V4 upstream defect,
    not a Gen14 regression, and failing here would block the pipeline's
    `--dependency=afterok` chain forever. It is printed as a loud banner instead.
    The fix belongs upstream in Gen7, after which Gen14 re-copies (PLAN §6.4).
    """
    print('\n=== G6: projector fires at K=1 (RUNTIME) ===')
    import torch
    from mix_visual_aligning.models.engine_registry import resolve, import_class

    thr = _eval_threshold()
    print(f'  threshold = {thr} (from config/visual_aligning_eval.yaml — the deployed value)')
    print(f'  building state-only (if_vision=False) models on {device}: no vision encoder needed\n')

    horizon, batch, action_dim, obs_dim = 8, 2, 3, 6
    transition_dim = action_dim + obs_dim

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.device, cfg.if_vision, cfg.horizon = device, False, horizon
    cfg.action_dim, cfg.obs_dim, cfg.dim = action_dim, obs_dim, 32
    cfg.dim_mults, cfg.condition_dropout, cfg.returns_condition = (1, 2, 4, 8), 0.1, False
    cfg.film_mode = 'v1'

    ok = True
    upstream_finding = False
    for arm in ('mf', 'af', 'fm'):
        spec = resolve(arm)
        ModelCls, DiffCls = import_class(spec['model']), import_class(spec['diffusion'])

        if spec['wraps_unet']:
            model = ModelCls(state_dim=transition_dim, seq_len=horizon, freq_dim=cfg.dim,
                             dropout_rate=0.1, device=device, if_vision=False, vis_config=cfg,
                             dual_head=True, interval_cfg=False)
            extra = dict(if_vision=False, t_schedule='logit_normal', p_mean=-0.4, p_std=1.0)
        else:
            model = ModelCls(cfg)
            extra = {}

        diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim,
                            action_dim=action_dim, goal_dim=0, n_timesteps=100,
                            loss_type='l2', flow_steps_v3=1, **extra).to(device)
        diffusion.flow_steps_v3 = 1                      # K = 1: the low-NFE regime
        diffusion.ode_inference_steps_v3 = 1

        spy = _SpyProjector(thr)
        cond = {0: torch.zeros(batch, obs_dim, device=device)}
        with torch.no_grad():
            diffusion.p_sample_loop((batch, horizon, transition_dim), cond,
                                    projector=spy, constraints=None)

        fired = spy.n_calls > 0
        if arm in ('mf', 'af'):
            if fired:
                print(f'  ok   {arm}: project() called {spy.n_calls}x at K=1')
            else:
                ok = False
                print(f'  FAIL {arm}: project() NEVER called at K=1 — the DPCC cage is OFF. '
                      f'The terminal-step fallback was lost.')
        else:
            if fired:
                print(f'  ok   {arm}: project() called {spy.n_calls}x at K=1 '
                      f'(upstream Gen7 appears to have been fixed — update this gate)')
            else:
                upstream_finding = True
                print(f'  !!   {arm}: project() NEVER called at K=1  <-- KNOWN UPSTREAM DEFECT')

    if upstream_finding:
        print('\n  ' + '!' * 68)
        print('  !! Gen7/Gen6V4 UPSTREAM DEFECT CONFIRMED AT RUNTIME (not a Gen14 regression)')
        print(f'  !! fm arm at K=1, threshold={thr}: `loop_idx >= (1-thr)*K` = `0 >= {1-thr}`')
        print('  !! is False, so the DPCC projection NEVER RUNS. Any K=1 result from the fm')
        print('  !! arm showing constraint violations is measuring an UNPROJECTED trajectory.')
        print('  !! Only K=1 is affected (at K=2, `1 >= 1.0` is True).')
        print('  !! Fix belongs in fm_visual_aligning/models/diffusion.py, THEN re-copy here.')
        print('  ' + '!' * 68)

    print('\n  G6 PASS (runtime)' if ok else '\n  G6 FAIL')
    print('  End-to-end confirmation (optional): run eval with --engine mf and '
          'flow_steps_v3=1 and check projection_costs[0] is populated and non-trivial.')
    return ok


def gate_g7(device='cuda'):
    """film_mode='v2' builds on ALL FOUR arms, keeps h_mlp, and survives the JVP (U5).

    Until U5 the True-FiLM backbone existed only for the diffusion/fm arms, and
    `VisualUNetTwoTime` raised on `film_mode='v2'` — Gen7's v2 file has no `h_mlp`, so
    routing mf/af through it would have silently un-conditioned the model on the
    interval h. `unet1d_twotime_film.py` closes that. Four things must hold:

      1. every arm CONSTRUCTS at v2, and the two legacy arms are unchanged at v1
      2. FiLM heads are actually present and live (`use_film`) on every arm at v2
      3. the two-time v2 backbone still owns an `h_mlp`, and its block time-path is
         TIME-ONLY (`in_features == dim`, not `2*dim`) — that width IS the v1/v2
         difference, so it is the cheapest proof the visual latent left the time path
      4. one mf loss step at v2 is finite: forward-mode AD survives the multiplicative
         gate `out = (1+γ)·f + β`. γ/β are constants under the JVP (cond is captured,
         not differentiated), so this should hold by construction — gate it anyway,
         because "should" is what G2 exists to disprove.

    Needs a GPU: the FiLM heads only exist when cond_dim > 0, i.e. if_vision=True,
    which builds the two ResNet-18 encoders.
    """
    print('\n=== G7: film_mode=v2 on all four arms ===')
    import torch
    from mix_visual_aligning.models.engine_registry import resolve, import_class

    horizon, batch, dim = 8, 2, 32
    ok = True

    class _Cfg:
        pass

    for arm in ('diffusion', 'fm', 'mf', 'af'):
        spec = resolve(arm)
        ModelCls = import_class(spec['model'])

        cfg = _Cfg()
        cfg.device, cfg.if_vision, cfg.horizon = device, True, horizon
        cfg.action_dim, cfg.obs_dim, cfg.dim = 3, 6, dim
        cfg.dim_mults, cfg.condition_dropout, cfg.returns_condition = (1, 2, 4, 8), 0.1, False
        cfg.film_mode = 'v2'

        try:
            if spec['wraps_unet']:
                model = ModelCls(state_dim=9, seq_len=horizon, freq_dim=dim,
                                 dropout_rate=0.1, device=device, if_vision=True,
                                 vis_config=cfg, dual_head=True, interval_cfg=False)
            else:
                model = ModelCls(cfg)
        except Exception as e:
            ok = False
            print(f'  FAIL {arm}: construction at v2 raised {type(e).__name__}: {e}')
            continue

        # (2) FiLM heads present and live. Class identity is checked by NAME so this
        # still passes if the block is re-exported, but it must be THE Gen7 block.
        blocks = [m for m in model.modules()
                  if type(m).__name__ == 'FiLMResidualTemporalBlock']
        if not blocks:
            ok = False
            print(f'  FAIL {arm}: no FiLMResidualTemporalBlock in the module tree — '
                  f'film_mode=v2 was accepted but silently ignored.')
            continue
        if not all(b.use_film for b in blocks):
            ok = False
            print(f'  FAIL {arm}: {sum(not b.use_film for b in blocks)}/{len(blocks)} FiLM '
                  f'heads are inert (cond_dim==0) — the visual latent is not reaching them.')
            continue

        # (3) time path is TIME-ONLY, and h_mlp survived on the two-time arms.
        width = blocks[0].time_mlp[1].in_features
        width_ok = (width == dim)
        if not width_ok:
            ok = False
            print(f'  FAIL {arm}: block time_mlp in_features={width}, want {dim}. '
                  f'{2 * dim} means the cond concat is still in the time path (v1 shape).')
            continue

        if spec['two_time']:
            carriers = [m for m in model.modules() if hasattr(m, 'h_mlp')]
            if not carriers:
                ok = False
                print(f'  FAIL {arm}: v2 backbone has NO h_mlp — MeanFlow/alpha-Flow '
                      f'h-conditioning would be silently dropped. This is exactly the '
                      f'failure the pre-U5 guard was preventing; do not remove the guard '
                      f'without this gate passing.')
                continue

        print(f'  ok   {arm}: {len(blocks)} FiLM heads, time-only embed ({width}), '
              f'h_mlp={"yes" if spec["two_time"] else "n/a"}')

    # (4) the JVP, on the arm that actually takes one.
    cfg, model, DiffCls, obs_dim = _build('mf', True, horizon, batch, device, film_mode='v2')
    diffusion = DiffCls(model, horizon=horizon, observation_dim=obs_dim, action_dim=3,
                        goal_dim=0, n_timesteps=100, loss_type='l2', if_vision=True,
                        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
                        meanflow_data_proportion=0.5, mf_adp_p=1.0, mf_adp_eps=0.01).to(device)
    traj, cond = _fake_visual_batch(batch, horizon, device)
    loss, info = diffusion.loss(traj, cond)
    loss.backward()
    finite = bool(torch.isfinite(loss))
    raw = float(info['raw_mse_u']) if 'raw_mse_u' in info else float('nan')
    print(f'  mf@v2 JVP: loss={float(loss):.6f} finite={finite}  raw_mse_u={raw:.6f}')
    if not finite:
        ok = False
        print('  FAIL mf@v2: JVP produced a non-finite loss through the FiLM gate.')

    print('\n  G7 PASS' if ok else '\n  G7 FAIL')
    print('  NOTE: v1 and v2 state_dicts are NOT interchangeable (embed_dim differs and '
          'film_proj is new). film_mode is a path key, so they land in parallel dirs — '
          'but a v2 config pointed at a v1 checkpoint still dies in load_state_dict.')
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Gen14 U9 — perception-first gates. Plan: logs_in_develop/Gen14/U9/
#
# U9 adds three ML-side knobs, all defaulting to the pre-U9 value:
#   vis_pretrained  (bool)  ImageNet init of the dual ResNet-18
#   vis_lr_scale    (float) encoder LR multiplier, trainer-side
#   vis_cond_mode   (str)   'token' (U8) | 'adaln' | 'both'
# These gates exist to prove the ADDITIVITY claim on hardware rather than by reading.
# ══════════════════════════════════════════════════════════════════════════════

# The adaLN pair. The RoPE bones (dit) have no adaLN pathway and are deliberately excluded —
# VisualDiTTwoTime RAISES if vis_cond_mode != 'token' reaches them.
_U9_ADALN_BONES = (('mf', 'mf_dit'), ('af', 'sit'))


def _encoder_spec_block(path):
    """The rgb_model spec as written, for the byte-identity check in G-B8."""
    import re
    src = open(os.path.join(REPO, path), 'rb').read().decode('utf-8').replace('\r\n', '\n')
    m = re.search(r"'rgb_model':\s*\{(.*?)\}", src, re.S)
    if m is None:
        return None
    return re.sub(r'\s+', ' ', m.group(1)).strip()


def gate_gb8():
    """G-B8 — the two two-time encoder specs are still identical.

    🔴 WHY THIS GATE EXISTS. visual_unet_twotime.py and visual_dit_twotime.py each build their
    own MultiImageObsEncoder config, and both carry a red comment saying the block is
    BYTE-IDENTICAL by design: the whole U-Net-vs-DiT comparison assumes the two arms differ
    only in the trajectory bone. Before U9 that was a comment. U9 makes the block carry a
    FLAG, which is exactly the kind of thing that gets added to one file and forgotten in the
    other — and the failure is silent, because both models still build and train fine.

    visual_unet.py (the fm/diffusion arm) is NOT compared here: it is a G0 VERBATIM file with
    a different upstream, it is untouched by U9, and folding it in would force it out of the
    VERBATIM ledger for no experimental gain.
    """
    print('\n=== G-B8: encoder spec identical across the two-time wrappers ===')
    paths = ('mix_visual_aligning/models/visual_unet_twotime.py',
             'mix_visual_aligning/models/visual_dit_twotime.py')
    specs = {p: _encoder_spec_block(p) for p in paths}
    ok = True
    for p, spec in specs.items():
        if spec is None:
            print(f'  ! {p}: no rgb_model block found'); ok = False
    if ok:
        a, b = (specs[p] for p in paths)
        if a != b:
            ok = False
            print('  ! the two rgb_model specs DIFFER — the U-Net-vs-DiT comparison is void')
            print(f'    unet_twotime: {a}')
            print(f'    dit_twotime : {b}')
        else:
            print(f'  ok   both wrappers: {a}')
        if "'pretrained'" not in a:
            ok = False
            print("  ! 'pretrained' missing from the rgb_model spec — U9 C1 is not wired")
    print('\n  G-B8 PASS' if ok else '\n  G-B8 FAIL')
    return ok


def gate_gb9(device='cuda'):
    """G-B9 — vis_cond_mode: 'token' is a bit-identical no-op, 'adaln' really moves the latent.

    Three claims, one per mode:
      (a) token  — same seed, same weights, same output as a build that never heard of U9.
                   This is THE additivity guarantee; if it fails, every U8 result is at risk.
      (b) adaln  — the sequence is one token shorter (num_visual_tokens == 0), vis_token is
                   gone from the state_dict, and two different latents still produce different
                   outputs (the latent reaches the output through `c`, not through attention).
      (c) both   — keeps the token AND responds to the latent.

    🔴 The sensitivity check WARMS UP first, for the same reason G-B3 does: these are
    adaLN-ZERO bones, every adaLN_modulation[-1] and both final layers start at exactly 0, so
    at step 0 the network emits 0 for any input and a naive difference reads 0.0 in every
    mode. See gate_gb3's note (job 24834, 2026-08-21).
    """
    import torch
    print('\n=== G-B9: vis_cond_mode token/adaln/both ===')
    WARMUP, WARMUP_LR = 5, 1e-2
    ok = True

    for arm, bone in _U9_ADALN_BONES:
        # (a) bit-identity of the default path
        torch.manual_seed(0)
        _, m_ref, _, _ = _build(arm, ml_bone=bone, device=device)
        torch.manual_seed(0)
        _, m_tok, _, _ = _build(arm, ml_bone=bone, device=device, vis_cond_mode='token')
        sd_ref, sd_tok = _vnet(m_ref).state_dict(), _vnet(m_tok).state_dict()
        same = (sd_ref.keys() == sd_tok.keys()) and all(
            torch.equal(sd_ref[k], sd_tok[k]) for k in sd_ref)
        print(f"  {'ok  ' if same else 'FAIL'} {arm}@{bone} token: state_dict identical to the "
              f"pre-U9 build ({len(sd_ref)} tensors)")
        ok &= same

        # (b)/(c) structure + sensitivity
        for mode, want_tokens, want_vis_token in (('adaln', 0, False), ('both', 1, True)):
            torch.manual_seed(0)
            _, model, DiffCls, obs_dim = _build(arm, ml_bone=bone, device=device,
                                                vis_cond_mode=mode)
            bb = _vnet(model).backbone
            n_tok = int(getattr(bb, 'num_visual_tokens', -1))
            has_vt = hasattr(bb, 'vis_token')
            struct = (n_tok == want_tokens) and (has_vt == want_vis_token)
            print(f"  {'ok  ' if struct else 'FAIL'} {arm}@{bone} {mode}: "
                  f"num_visual_tokens={n_tok} (want {want_tokens}), "
                  f"vis_token present={has_vt} (want {want_vis_token})")
            ok &= struct

            diffusion = DiffCls(model, horizon=8, observation_dim=obs_dim, action_dim=3,
                                if_vision=True).to(device)
            opt = torch.optim.Adam(diffusion.parameters(), lr=WARMUP_LR)
            traj, cond = _fake_visual_batch(2, 8, device)
            for _ in range(WARMUP):
                opt.zero_grad(set_to_none=True)
                loss, _ = diffusion.loss(traj, cond)
                loss.backward(); opt.step()

            B, H = 2, 8
            x = torch.randn(B, H, 9, device=device)
            t = torch.rand(B, device=device)
            h = torch.rand(B, device=device)
            with torch.no_grad():
                l1 = torch.randn(B, bb.cond_dim, device=device)
                l2 = torch.randn(B, bb.cond_dim, device=device)
                o1, o2 = bb(x, l1, t, h=h), bb(x, l2, t, h=h)
                o1 = o1[0] if isinstance(o1, tuple) else o1
                o2 = o2[0] if isinstance(o2, tuple) else o2
                delta = float((o1 - o2).abs().max())
            live = delta > 0.0
            print(f"  {'ok  ' if live else 'FAIL'} {arm}@{bone} {mode}: "
                  f"d(out)/d(latent) max = {delta:.4e}"
                  + ('' if live else '   <-- the latent does NOT reach the output'))
            ok &= live
            del model, diffusion, opt
            torch.cuda.empty_cache() if device == 'cuda' else None

    print('\n  G-B9 PASS' if ok else '\n  G-B9 FAIL')
    return ok


def gate_gb11():
    """G-B11 — vis_pretrained=True really loaded weights from disk, and did not fall back.

    🔴 WHY THIS IS NOT PARANOIA. `pretrained=True` makes torchvision fetch ImageNet weights
    into ~/.cache/torch/hub/checkpoints/, and COMPUTE NODES HAVE NO INTERNET. A run whose
    download failed, or an env on torchvision >= 0.15 where the `pretrained=` kwarg was
    removed, can end up training a randomly-initialised encoder while every log line says
    `vis_pretrained=True`. That does not crash. It produces a null result that looks like an
    architecture finding, and it would take a week to catch by any other means.

    The test needs no checksum and no network: build the encoder TWICE under two different
    torch seeds.
      * loaded from a file -> the two builds agree exactly (weights are seed-independent)
      * random init        -> the two builds differ (weights come from the RNG)
    The control (pretrained=False) must show the opposite, otherwise the test itself is
    vacuous — e.g. if some caller had already made init deterministic.
    """
    import torch
    print('\n=== G-B11: pretrained weights actually loaded ===')
    sys.path.insert(0, os.path.join(REPO, 'd3il'))
    from agents.models.vision.model_getter import get_resnet

    def _trunk(pretrained, seed):
        torch.manual_seed(seed)
        net = get_resnet(input_shape=[3, 96, 96], output_size=64, pretrained=pretrained)
        return {k: v.detach().clone() for k, v in net.backbone.state_dict().items()}

    ok = True
    try:
        a, b = _trunk(True, 0), _trunk(True, 12345)
    except TypeError as e:
        print(f'  ! get_resnet rejected pretrained=: {e}')
        print('    -> torchvision >= 0.15 removed the `pretrained` kwarg; switch '
              "base_nets.py:510 to weights='IMAGENET1K_V1'.")
        print('\n  G-B11 FAIL')
        return False
    except Exception as e:
        print(f'  ! could not build a pretrained trunk: {type(e).__name__}: {e}')
        print('    -> almost certainly the weights are not in ~/.cache/torch/hub/checkpoints/.')
        print('    -> pre-fetch ONCE on the login node (compute nodes have no internet):')
        print('       python -c "import torchvision as tv; tv.models.resnet18(pretrained=True)"')
        print('\n  G-B11 FAIL')
        return False

    det = all(torch.equal(a[k], b[k]) for k in a)
    print(f"  {'ok  ' if det else 'FAIL'} pretrained=True is seed-independent "
          f"({len(a)} tensors) -> weights came from a file, not the RNG")
    ok &= det

    c, d = _trunk(False, 0), _trunk(False, 12345)
    rnd = any(not torch.equal(c[k], d[k]) for k in c)
    print(f"  {'ok  ' if rnd else 'FAIL'} pretrained=False IS seed-dependent "
          f"-> the control is meaningful (a passing test above is not vacuous)")
    ok &= rnd

    if det and rnd:
        diff = max(float((a[k].float() - c[k].float()).abs().max())
                   for k in a if a[k].dtype.is_floating_point)
        print(f'  ok   pretrained and random trunks differ (max |dw| = {diff:.4e})')

    print('\n  G-B11 PASS' if ok else '\n  G-B11 FAIL')
    return ok


GATES = {'gb1': gate_gb1, 'gb6': gate_gb6, 'gb7': gate_gb7,
         'gb2': gate_gb2, 'gb3': gate_gb3, 'gb45': gate_gb45,
         # Gen14 U9
         'gb8': gate_gb8, 'gb9': gate_gb9, 'gb11': gate_gb11,
         'g0': gate_g0, 'g1': gate_g1, 'g2': gate_g2,
         'g3': gate_g3, 'g4': gate_g4, 'g5': gate_g5, 'g6': gate_g6,
         'g7': gate_g7}
NEEDS_GPU = {'g2', 'g3', 'g5', 'g7', 'gb2', 'gb3', 'gb45', 'gb9'}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', default='all',
                    choices=['all', 'static', 'bone'] + list(GATES))
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    if a.gate == 'all':
        names = list(GATES)
    elif a.gate == 'static':
        names = [g for g in GATES if g not in NEEDS_GPU]
    elif a.gate == 'bone':
        # Gen14 U8 — the ML-bone gates (G-B1..G-B7)
        # Gen14 U9 — plus the perception/conditioning gates (G-B8, G-B9, G-B11)
        names = ['gb1', 'gb6', 'gb7', 'gb2', 'gb3', 'gb45', 'gb8', 'gb9', 'gb11']
    else:
        names = [a.gate]

    results = {}
    for name in names:
        fn = GATES[name]
        try:
            results[name] = fn(device=a.device) if name in NEEDS_GPU else fn()
        except Exception as e:              # a gate that crashes is a gate that failed
            print(f'  {name.upper()} ERROR: {type(e).__name__}: {e}')
            results[name] = False

    print('\n' + '=' * 60)
    for name, passed in results.items():
        print(f'  {name.upper()}: {"PASS" if passed else "FAIL"}')
    print('=' * 60)
    sys.exit(0 if all(results.values()) else 1)
