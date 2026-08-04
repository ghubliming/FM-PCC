"""Gen14 U7 — pre-flight gates for the HardFlow (arm C) port.

Mirrors `FM_v3_alphaflow_test/gates_hardflow_alphaflow.py` (H0/H1/H3) and adds the two
checks that are specific to Gen14: the visual-latent survival (U7 delta 3) and the
diffusion-arm refusal (U7 delta 2/4).

H0/H2/H4 are static and run anywhere (no numpy/torch). H1/H3 need the cluster env:

    python mix_visual_aligning_test/gates_hardflow_mix_visual.py            # all
    python mix_visual_aligning_test/gates_hardflow_mix_visual.py --gate h0  # one

Exit code is non-zero if any selected gate fails.
"""
import argparse
import ast
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SRC = os.path.join(REPO, 'flow_matcher_v3_alphaflow', 'sampling', 'hardflow_projection.py')
DST = os.path.join(REPO, 'mix_visual_aligning', 'sampling', 'hardflow_projection.py')

NEEDS_GPU = {'h1', 'h3'}


def _lines(path):
    with open(path, encoding='utf-8') as f:
        return f.read().splitlines()


def _block(lines, start_pat, end_pats):
    i = next(n for n, l in enumerate(lines) if re.match(start_pat, l))
    ends = [n for n, l in enumerate(lines) if n > i and any(re.match(p, l) for p in end_pats)]
    return lines[i:(ends[0] if ends else len(lines))]


# ── H0 ────────────────────────────────────────────────────────────────────────────────
def gate_h0(**_):
    """The copied regions are BYTE-IDENTICAL to the Gen3v7 source.

    This is the whole point of the port being a `cp` + 4 documented edits. If someone
    "improves" the NLP or the dof layout here instead of upstream, the two generations
    silently stop solving the same problem and every cross-generation comparison is void.
    """
    src, dst = _lines(SRC), _lines(DST)
    regions = [
        ('TrajectoryLayout', r'^class TrajectoryLayout', [r'^class HardFlowNLP']),
        ('HardFlowNLP',      r'^class HardFlowNLP',      [r'^def resolve_activation_threshold']),
        ('resolve_activation_threshold',
         r'^def resolve_activation_threshold',
         [r'^class HardFlowSampler', r'^# ── U7 delta']),
    ]
    ok = True
    for name, sp, ep in regions:
        a, b = _block(src, sp, ep), _block(dst, sp, ep)
        same = (a == b)
        print(f'  {"ok  " if same else "FAIL"} {name:32s} src={len(a):4d} dst={len(b):4d}')
        ok &= same
    if not ok:
        print('  → a copied region drifted. Fix it UPSTREAM (Gen3v7) and re-copy.')
    return ok


# ── H2 ────────────────────────────────────────────────────────────────────────────────
def gate_h2(**_):
    """U7 delta 3: the visual latent must survive the cond filter, and a blind visual
    rollout must RAISE rather than silently produce numbers.

    Static (AST/text) so it runs without torch. The failure it guards is invisible at
    runtime — `_project_cond` returns None and the model just conditions on nothing.
    """
    txt = open(DST, encoding='utf-8').read()
    # The module docstring QUOTES the old blind filter as documentation, so a raw
    # substring search would always trip. Strip docstrings and comments first and search
    # only executable text.
    _doc = ast.get_docstring(ast.parse(txt)) or ''
    code = '\n'.join(l for l in txt.replace(_doc, '').splitlines()
                     if not l.lstrip().startswith('#'))
    checks = [
        ("_VISUAL_COND_KEYS defined",
         "_VISUAL_COND_KEYS = frozenset({'visual_latent', 'visual'})" in code),
        ("cond filter keeps the allow-list",
         "if not isinstance(k, str) or k in _VISUAL_COND_KEYS" in code),
        ("verbatim (blind) filter is GONE from executable code",
         "cond_net = {k: v for k, v in cond.items() if not isinstance(k, str)}" not in code),
        ("blind visual rollout raises",
         "if self.model_is_visual and not any(k in cond_net for k in _VISUAL_COND_KEYS)" in code),
        ("if_vision probed on the BACKBONE, not just the wrapper",
         "getattr(getattr(model, 'model', None), 'if_vision', False)" in code),
        ("encode_visual_cond exists",
         "def encode_visual_cond(" in code),
    ]
    ok = True
    for name, passed in checks:
        print(f'  {"ok  " if passed else "FAIL"} {name}')
        ok &= passed
    return ok


# ── H4 ────────────────────────────────────────────────────────────────────────────────
def gate_h4(**_):
    """U7 delta 2/4: engine table is correct and `diffusion` is refused.

    Reads the constants by AST so it needs no numpy/torch and runs in the AI-coding
    container as well as on the cluster. The init-noise scales are cross-checked against
    the Gen14 engines' OWN `p_sample_loop` source, so changing a sampler there fails this
    gate instead of silently running arm C at the wrong tau=0 distribution — the Gen3v6
    fix_4 mistake, which cost a full K-sweep.
    """
    tree = ast.parse(open(DST, encoding='utf-8').read())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('ENGINE_INIT_NOISE', 'ENGINE_TWO_TIME'):
                consts[name] = ast.literal_eval(node.value)
    ok = ('ENGINE_INIT_NOISE' in consts and 'ENGINE_TWO_TIME' in consts)
    if not ok:
        print('  FAIL ENGINE_INIT_NOISE / ENGINE_TWO_TIME not found as module constants')
        return False
    noise, two = consts['ENGINE_INIT_NOISE'], consts['ENGINE_TWO_TIME']

    expect_noise = {'fm': 0.5, 'mf': 1.0, 'af': 1.0}
    expect_two   = {'fm': False, 'mf': True, 'af': True}
    for e in ('fm', 'mf', 'af'):
        good = (noise.get(e) == expect_noise[e] and two.get(e) == expect_two[e])
        print(f'  {"ok  " if good else "FAIL"} {e}: init_noise={noise.get(e)} '
              f'two_time={two.get(e)} (want {expect_noise[e]}, {expect_two[e]})')
        ok &= good

    refused = ('diffusion' not in noise and 'diffusion' not in two)
    print(f'  {"ok  " if refused else "FAIL"} diffusion absent from both tables '
          f'(no velocity field -> hardflow_new undefined)')
    ok &= refused

    src = open(DST, encoding='utf-8').read()
    raises = 'raise ValueError(' in src and 'HardFlow is not available for engine' in src
    print(f'  {"ok  " if raises else "FAIL"} resolve_engine_hf raises on an unknown engine')
    ok &= raises

    # Cross-check the scales against the engines' own samplers.
    for eng, path, pat, want in [
        ('fm', 'mix_visual_aligning/models/fm_diffusion.py', r'x = 0\.5 \* torch\.randn\(shape', 0.5),
        ('mf', 'mix_visual_aligning/models/mf_diffusion.py', r'x = torch\.randn\(shape',           1.0),
        ('af', 'mix_visual_aligning/models/af_diffusion.py', r'x = torch\.randn\(shape',           1.0),
    ]:
        body = open(os.path.join(REPO, path), encoding='utf-8').read()
        found = re.search(pat, body) is not None
        agrees = found and noise[eng] == want
        print(f'  {"ok  " if agrees else "FAIL"} {eng}: sampler in {os.path.basename(path)} '
              f'is sigma={want} and ENGINE_INIT_NOISE agrees ({noise[eng]})')
        ok &= agrees
    return ok


# ── H1 / H3 (GPU) ─────────────────────────────────────────────────────────────────────
def gate_h1(device='cuda', **_):
    """The two-time hosts are queried at h=0 EXACTLY (the grounding u(x,t,0)=v(x,t)),
    and the one-time host is queried without `h` at all."""
    import inspect
    import torch  # noqa: F401  — import here so the static gates run without it
    from mix_visual_aligning.models import mf_diffusion, af_diffusion, fm_diffusion

    ok = True
    for name, mod, want_h in [('mf', mf_diffusion, True), ('af', af_diffusion, True),
                              ('fm', fm_diffusion, False)]:
        cls = next(c for c in vars(mod).values()
                   if inspect.isclass(c) and hasattr(c, '_predict_velocity'))
        sig = inspect.signature(cls._predict_velocity)
        has_h = 'h' in sig.parameters
        good = (has_h == want_h)
        print(f'  {"ok  " if good else "FAIL"} {name}._predict_velocity has h= {has_h} '
              f'(want {want_h}) — {list(sig.parameters)}')
        ok &= good
    txt = open(DST, encoding='utf-8').read()
    explicit = 'h=torch.zeros_like(t)' in txt
    print(f'  {"ok  " if explicit else "FAIL"} h=0 passed EXPLICITLY (never via an h=None default)')
    return ok and explicit


def gate_h3(device='cuda', **_):
    """`draw_init_noise` really produces the host's sigma — the fix_4 regression pin."""
    import torch
    from mix_visual_aligning.sampling.hardflow_projection import (
        HardFlowSampler, TrajectoryLayout, ENGINE_INIT_NOISE)

    layout = TrajectoryLayout(horizon=8, action_dim=3, state_dim=6)
    ok = True
    for eng, sigma in ENGINE_INIT_NOISE.items():
        s = HardFlowSampler.__new__(HardFlowSampler)
        s.init_noise_scale, s.layout, s.device = sigma, layout, device
        torch.manual_seed(0)
        got = s.draw_init_noise(4096).std().item()
        good = abs(got - sigma) < 0.05
        print(f'  {"ok  " if good else "FAIL"} {eng}: measured sigma={got:.4f} want {sigma}')
        ok &= good
    return ok


GATES = {'h0': gate_h0, 'h1': gate_h1, 'h2': gate_h2, 'h3': gate_h3, 'h4': gate_h4}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', choices=sorted(GATES), default=None)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    names = [a.gate] if a.gate else sorted(GATES)
    results = {}
    for n in names:
        print('=' * 60)
        print(f'  {n.upper()}: {GATES[n].__doc__.splitlines()[0]}')
        print('=' * 60)
        try:
            results[n] = bool(GATES[n](device=a.device))
        except Exception as e:
            print(f'  ERROR: {type(e).__name__}: {e}')
            results[n] = False
        print(f'\n  {n.upper()}: {"PASS" if results[n] else "FAIL"}\n')
    print('=' * 60)
    print('  ' + '  '.join(f'{k.upper()}={"PASS" if v else "FAIL"}' for k, v in results.items()))
    print('=' * 60)
    sys.exit(0 if all(results.values()) else 1)
