"""Gen15 (UAV Mix-ML) gates — run this BEFORE any science run.

    python mix_uav_test/gates_mix_uav.py              # G0, G2, G3, G4, G5, G6, G7, G8
    python mix_uav_test/gates_mix_uav.py --gates G3 G4
    python mix_uav_test/gates_mix_uav.py --gen11-savepath <dir> --gen15-savepath <dir>   # + G1

Every gate here is cheap (seconds, no dataset, no MuJoCo) EXCEPT G1, which needs two real
checkpoints and is skipped unless both paths are given. The point is that a wiring mistake in
this generation is discovered in a 30-second job, not 18 hours into a training sweep.

Gate list (PLAN §8):
  G0  every arm builds a model + diffusion + trainer with no missing config key
  G1  `fm` parity vs Gen11 (needs checkpoints; skipped otherwise)
  G2  path collision + Gen11 isolation
  G3  backbone identity — param counts equal across arms
  G4  two-time sampling stays inside the trained (t, h) domain
  G5  projector receives the FULL trajectory once goal_dim is forced to 0
  G6  per-plan wall clock is measured and reported per arm
  G7  every arm returns the same `infos` contract AND actually times the projector (Fix_1)
  G8  HardFlow arm: per-engine init_noise_scale + two_time match the engine source (U2)
"""

import argparse
import os
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

import torch

from mix_uav.models import engine_registry

ARMS = list(engine_registry.ENGINE_KEYS)

# UAV E8 'pos_only' schema: obs=[p_des|p] 6-D, action=Δp_des 3-D, transition=9-D, H=8.
OBS_DIM, ACT_DIM, HORIZON = 6, 3, 8
TRANSITION_DIM = OBS_DIM + ACT_DIM


class _FakeDataset:
    """Just the three attributes the registry's kwarg builders read off a dataset.

    Deliberately NOT the real SequenceDataset: these gates must run on a node with no UAV data
    on disk, and every shape below is fixed by the E8 schema anyway.
    """
    observation_dim = OBS_DIM
    action_dim = ACT_DIM
    goal_dim = 0


class _StubProjector:
    """Records the trajectory shape the sampler hands to a projector (G5, G7).

    `sleep_s` burns a known amount of wall-clock inside each projector call so G7 can prove the
    sampler actually TIMED the call rather than reporting a hard 0.0.
    """

    def __init__(self, gradient=False, threshold=0.5, sleep_s=0.0):
        self.gradient = gradient
        self.diffusion_timestep_threshold = threshold
        self.sleep_s = sleep_s
        self.seen_shapes = []

    def _tick(self, x):
        self.seen_shapes.append(tuple(x.shape))
        if self.sleep_s:
            import time as _t
            _t.sleep(self.sleep_s)

    def project(self, x, constraints):
        self._tick(x)
        import numpy as np
        return x, np.zeros(x.shape[0])

    def compute_gradient(self, x, constraints):
        self._tick(x)
        return torch.zeros_like(x)


def _args_for(engine, device='cpu'):
    """The parsed-config surface the registry expects, without touching config/uav_mix.py.

    G2 exercises the REAL config; G0/G3/G4/G5/G6 only need the values, and building them here
    keeps those gates runnable on a CPU-only login node.
    """
    class A:
        pass

    a = A()
    a.horizon = HORIZON
    a.loss_type = 'l2'
    a.loss_discount = 1.0
    a.action_weight = 1
    a.predict_epsilon = True
    a.clip_denoised = False
    a.returns_condition = False
    a.device = device
    a.n_train_steps = 100000
    a.n_steps_per_epoch = 1000
    a.batch_size = 8
    a.learning_rate = 1e-4
    a.gradient_accumulate_every = 2
    a.ema_decay = 0.995
    a.train_test_split = 0.9
    a.savepath = os.path.join('/tmp', f'gen15_gates_{engine}')
    a.flow_steps_v3 = 10
    a.ode_inference_steps_v3 = 10
    if engine in ('fm', 'diffusion'):
        a.dim = 32
        a.dim_mults = (1, 2, 4, 8)
        a.condition_dropout = 0.25
        a.condition_guidance_w = 1.2
        a.time_beta_alpha_v3 = 1.5
        a.time_beta_beta_v3 = 1.0
        a.n_diffusion_steps = 20      # `diffusion` arm only: K, baked into the checkpoint
    else:
        a.freq_dim = 32
        a.depth = 8
        a.num_heads = 4
        a.mlp_dim = 256
        a.time_dim = 256
        a.dropout_rate = 0.1
        a.dual_head = True
        a.interval_cfg = False
        a.imf_backbone = 'unet'
        a.condition_guidance_w = 0.0
        a.u_loss_weight = 1.0
        a.v_loss_weight = 1.0
        a.loss_schedule = 'balanced'
        a.warmup_epochs = 0
        a.transition_epochs = 0
        a.t_schedule = 'logit_normal'
        a.p_mean = -0.4
        a.p_std = 1.0
        a.time_beta_alpha_v3 = 1.0
        a.time_beta_beta_v3 = 1.0
        a.gradient_clip = 1.0
        a.split_seed = 42
        if engine == 'mf':
            a.mf_objective = 'meanflow'
            a.meanflow_data_proportion = 0.5
            a.mf_adp_p = 1.0
            a.mf_adp_eps = 0.01
        else:
            a.af_alpha_scheduler = 'sigmoid'
            a.af_alpha_init = 1.0
            a.af_alpha_end = 0.0
            a.af_alpha_init_step = 0
            a.af_alpha_end_step = 100000
            a.af_alpha_gamma = 25.0
            a.af_alpha_clamp = 0.005
            a.af_ratio_fm = 0.5
            a.af_clamp_utgt = 4.0
            a.af_adp_eps = 1e-3
    return a


def _build(engine, device='cpu'):
    """(model, diffusion) for one arm, built exactly the way train_mix_uav.py builds them."""
    import mix_uav.utils as utils

    row = engine_registry.get(engine)
    args = _args_for(engine, device)
    dataset = _FakeDataset()
    model_cls = utils.import_class(row['model'])
    diff_cls = utils.import_class(row['diffusion'])
    model = model_cls(**row['model_kwargs'](args, dataset)).to(device)
    diffusion = diff_cls(model, **row['diffusion_kwargs'](args, dataset)).to(device)
    return model, diffusion, args


# ── G0 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G0(device='cpu'):
    """Every arm builds model + diffusion + trainer with no missing config key."""
    import mix_uav.utils as utils

    ok = True
    for engine in ARMS:
        row = engine_registry.get(engine)
        try:
            model, diffusion, args = _build(engine, device)
            trainer_cls = utils.import_class(row['trainer'])
            os.makedirs(args.savepath, exist_ok=True)
            # Trainer construction needs a dataset it can len()/index; skip the dataloader by
            # building the object only far enough to prove the kwarg set matches the signature.
            import inspect
            sig = inspect.signature(trainer_cls.__init__)
            unknown = [k for k in row['trainer_kwargs'](args) if k not in sig.parameters]
            if unknown:
                print(f'  ✗ {engine}: trainer kwargs not in {trainer_cls.__module__}.'
                      f'{trainer_cls.__name__} signature: {unknown}')
                ok = False
                continue
            n = sum(p.numel() for p in model.parameters())
            print(f'  ✓ {engine:3s} {row["label"]:42s} params={n:,} '
                  f'({n / 1e6:.2f} M)  diffusion={type(diffusion).__name__}')
        except Exception as e:
            print(f'  ✗ {engine}: {type(e).__name__}: {e}')
            ok = False
    return ok


# ── G1 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G1(gen11_savepath=None, gen15_savepath=None):
    """`fm` parity vs Gen11.

    Structural half only: the two checkpoints must declare the same architecture and the same
    training hyper-parameters. The BEHAVIOURAL half (same scene/seed/variant → same success,
    steps, trajectory) is a rollout comparison and belongs in a real eval job — see the
    changelog's G1 procedure.

    ⚠️ Assert parity on `diffuser` + `dpcc-c` variants only. The `dpcc-t*` (temporal
    consistency) variants are EXPECTED to differ: Gen15 grafts Gen3v6's `fix_5` executed_idx
    correction into policies.py, which changes which candidate seeds `prev_observations`.
    """
    if not (gen11_savepath and gen15_savepath):
        print('  ⊘ SKIPPED — needs --gen11-savepath and --gen15-savepath '
              '(a trained Gen11 run and the matching Gen15 `fm` run)')
        return None

    import pickle

    def _load(savepath, name):
        with open(os.path.join(savepath, name), 'rb') as f:
            return pickle.load(f)

    ok = True
    for cfg_name, ignore in (('model_config.pkl', set()),
                             ('diffusion_config.pkl', {'flow_steps_v3', 'ode_inference_steps_v3'})):
        a, b = _load(gen11_savepath, cfg_name), _load(gen15_savepath, cfg_name)
        ad, bd = dict(a._dict), dict(b._dict)
        keys = (set(ad) | set(bd)) - ignore
        diffs = [(k, ad.get(k, '<missing>'), bd.get(k, '<missing>'))
                 for k in sorted(keys) if ad.get(k, '<missing>') != bd.get(k, '<missing>')]
        if diffs:
            ok = False
            print(f'  ✗ {cfg_name} differs:')
            for k, v11, v15 in diffs:
                print(f'      {k}: Gen11={v11!r}  Gen15={v15!r}')
        else:
            print(f'  ✓ {cfg_name} identical ({len(keys)} keys compared)')
    return ok


# ── G2 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G2():
    """(a) distinct paths across the knob cross-product, (b) Gen11 isolation, (c) config/uav.py
    untouched.

    (a) is the live risk. Two `mf` runs differing only in `meanflow_data_proportion` or
    `imf_backbone` would land in the same checkpoint directory under Gen11's `_uav_exp_name`,
    which has no slot for either.
    """
    from config.uav_mix import _uav_mix_exp_name, logbase

    class A:
        pass

    seen = {}
    ok = True

    def _name(engine, **kw):
        a = A()
        a.horizon = HORIZON
        a.cond_mode = 'pos_only'
        a.engine = engine
        a.prefix = f'mix_uav_{engine}/'
        a.diffusion = engine_registry.get(engine)['diffusion']
        for k, v in kw.items():
            setattr(a, k, v)
        return _uav_mix_exp_name(a)

    combos = [('fm', {})]
    for k in (1, 5, 10, 20):          # U3: `diffusion` K is train-time → it IS a folder token
        combos.append(('diffusion', dict(n_diffusion_steps=k)))
    for dp in (0.25, 0.5, 0.75):
        for bb in ('unet', 'dit', 'mf_dit'):
            combos.append(('mf', dict(meanflow_data_proportion=dp, imf_backbone=bb)))
    for ae in (0.0, 0.5):
        for bb in ('unet', 'dit', 'sit'):
            combos.append(('af', dict(af_alpha_init=1.0, af_alpha_end=ae, imf_backbone=bb)))

    for engine, kw in combos:
        name = _name(engine, **kw)
        if name in seen:
            print(f'  ✗ COLLISION: {engine} {kw} and {seen[name]} both resolve to {name!r}')
            ok = False
        seen[name] = (engine, kw)
    print(f'  {"✓" if ok else "✗"} (a) {len(combos)} knob combinations → {len(seen)} distinct exp_names')

    # (d) 🔴 U3 — every declared exp_name token must actually EXIST in that engine's train
    # block. `_uav_mix_exp_name` renders a token only `if hasattr(args, key)`, so a misspelled
    # key is SILENT: the token simply never appears and two runs differing only in that knob
    # share a checkpoint directory. This gate found exactly that — the `af` row declared
    # `af_alpha_start` while the config key is `af_alpha_init`, so the `as` token had never
    # rendered on any α-Flow path.
    try:
        import importlib
        _cfg = importlib.import_module('config.uav_mix')
        for _eng in engine_registry.ENGINE_KEYS:
            _blk = _cfg.base[engine_registry.experiment_name(_eng)]
            _missing = [k for k, _lbl in engine_registry.get(_eng)['exp_name_tokens']
                        if k not in _blk]
            if _missing:
                print(f'  ✗ (d) engine {_eng!r} declares exp_name tokens absent from its train '
                      f'block: {_missing} — these render as NOTHING and cause silent collisions')
                ok = False
            else:
                print(f'  ✓ (d) {_eng}: all exp_name token keys present in the train block')
    except Exception as e:
        print(f'  ⊘ (d) could not import config.uav_mix ({type(e).__name__}: {e})')

    # (b) Gen15 never writes under Gen11's root.
    if logbase != 'logs/UAV_MIX':
        print(f'  ✗ (b) logbase is {logbase!r}, expected \'logs/UAV_MIX\' — Gen15 must not share '
              f'Gen11\'s logs/UAV_FM root')
        ok = False
    else:
        print(f'  ✓ (b) logbase={logbase} (Gen11 keeps logs/UAV_FM)')

    # (c) config/uav.py is byte-untouched.
    try:
        r = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'config/uav.py'],
                           cwd=_REPO, capture_output=True)
        if r.returncode == 0:
            print('  ✓ (c) config/uav.py unmodified vs HEAD')
        else:
            print('  ✗ (c) config/uav.py has uncommitted changes — Gen15 must not edit Gen11\'s config')
            ok = False
    except Exception as e:
        print(f'  ⊘ (c) could not run git diff: {e}')
    return ok


# ── G3 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G3(device='cpu'):
    """Backbone identity: the velocity net must have the SAME parameter count on every arm.

    This is the FIX_8_UNET_WIDTH lesson made executable. `freq_dim` is the two-time U-Net's
    channel width (32 => 3.97 M, 256 => 253 M) and nothing in a training log states a parameter
    count, so a width defect is invisible for months. If this gate fails, the three-way
    comparison is confounded and no result from it means anything.
    """
    counts = {}
    for engine in ARMS:
        model, _diffusion, _args = _build(engine, device)
        # On the two-time arms the engine wraps the backbone; count the velocity net itself.
        net = model
        for attr in ('model', 'velocity_net'):
            net = getattr(net, attr, net)
        counts[engine] = sum(p.numel() for p in net.parameters())
        print(f'  {engine:3s} velocity-net params: {counts[engine]:,} ({counts[engine] / 1e6:.2f} M)')

    uniq = set(counts.values())
    if len(uniq) == 1:
        print(f'  ✓ all {len(ARMS)} arms parameter-identical')
        return True
    # The two-time U-Net carries an extra h_mlp (and a v-head when dual_head=True), so a small
    # positive delta vs `fm` is EXPECTED and is reported, not failed. A large one is a width bug.
    base = counts['fm']
    worst = max(abs(c - base) / base for c in counts.values())
    if worst < 0.25:
        print(f'  ✓ arms within {worst:.1%} of the fm baseline '
              f'(h_mlp + dual v-head account for the delta; a width defect would be ~60x)')
        return True
    print(f'  ✗ parameter counts diverge by {worst:.1%} — check freq_dim (must be 32, not 256)')
    return False


# ── G4 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G4(device='cpu'):
    """Two-time sampling stays inside the trained (t, h) domain: t, h ∈ [0,1] and t + h ≤ 1.

    The two-time engines were trained on the JVP primal (x_r, r, h) with h = t − r and t ≤ 1.
    A sampler that queries outside that box is asking the model to extrapolate, which shows up
    as jitter rather than as an error — so it must be asserted, not eyeballed.
    """
    ok = True
    for engine in [e for e in ARMS if engine_registry.get(e)['two_time']]:
        model, diffusion, _args = _build(engine, device)
        seen = []

        target = model.model            # the trajectory model wrapping the velocity net
        original_forward = target.forward

        def spy(x, t, h=None, cond=None, *a, **kw):
            tv = float(torch.as_tensor(t).flatten()[0])
            hv = 0.0 if h is None else float(torch.as_tensor(h).flatten()[0])
            seen.append((tv, hv))
            return original_forward(x, t, h=h, cond=cond, *a, **kw)

        target.forward = spy
        try:
            cond = {0: torch.zeros(2, OBS_DIM, device=device)}
            with torch.no_grad():
                diffusion.conditional_sample(cond, horizon=HORIZON, num_steps=4)
        finally:
            target.forward = original_forward

        bad = [(t, h) for t, h in seen
               if not (-1e-6 <= t <= 1 + 1e-6 and -1e-6 <= h <= 1 + 1e-6 and t + h <= 1 + 1e-6)]
        if bad:
            print(f'  ✗ {engine}: {len(bad)}/{len(seen)} queries outside the trained domain, '
                  f'e.g. {bad[:3]}')
            ok = False
        else:
            print(f'  ✓ {engine}: all {len(seen)} (t, h) queries inside [0,1] with t+h ≤ 1')
    return ok


# ── G5 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G5(device='cpu'):
    """With goal_dim forced to 0, the projector must receive the FULL 9-D trajectory.

    UAV has no semantic goal columns, but SequenceDataset.get_goal_dim() false-positives on
    incidentally-constant channels (constant p_des, corridor altitude). A non-zero goal_dim
    makes p_sample_loop slice the trajectory before handing it over, and the dynamics
    constraints — which touch p indices 6,7,8 — then index out of bounds inside build_matrices.
    The eval patches model.goal_dim = 0; this gate proves the patch has the intended effect on
    every engine.
    """
    ok = True
    for engine in ARMS:
        _model, diffusion, _args = _build(engine, device)
        diffusion.goal_dim = 0                       # what eval_mix_uav.py does before Policy()
        proj = _StubProjector(gradient=False, threshold=0.5)
        cond = {0: torch.zeros(2, OBS_DIM, device=device)}
        with torch.no_grad():
            diffusion.conditional_sample(cond, horizon=HORIZON, projector=proj, constraints={})
        if not proj.seen_shapes:
            print(f'  ✗ {engine}: projector was never called')
            ok = False
        elif any(s[-1] != TRANSITION_DIM for s in proj.seen_shapes):
            print(f'  ✗ {engine}: projector saw {set(proj.seen_shapes)}, expected last dim '
                  f'{TRANSITION_DIM}')
            ok = False
        else:
            print(f'  ✓ {engine}: projector called {len(proj.seen_shapes)}x with full '
                  f'{TRANSITION_DIM}-D trajectory')
    return ok


# ── G6 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G6(device='cpu', k_values=(1, 2, 5, 10, 20)):
    """Per-plan wall clock, per arm, per K.

    NOT a pass/fail on the 33 Hz deadline — whether an arm meets it IS the experiment
    (PLAN §7.3). This gate exists so the number is measured and printed rather than assumed,
    and so an obviously broken arm (10x slower than its siblings) is caught before a sweep.
    On CPU the absolute numbers mean nothing; run it on a GPU node for a real reading.
    """
    import time

    budget_ms = 1000.0 / 33.0
    print(f'  control budget at 33 Hz: {budget_ms:.1f} ms per replan '
          f'(device={device}; CPU numbers are indicative only)')
    for engine in ARMS:
        _model, diffusion, _args = _build(engine, device)
        row = engine_registry.get(engine)
        cond = {0: torch.zeros(4, OBS_DIM, device=device)}   # mpc_batch_size=4
        line = []
        if engine_registry.get(engine)['nfe_is_train_time']:
            # U3: K is baked into this checkpoint's beta schedule; sweeping it here would time a
            # sampler that cannot exist. Report the single budget it actually has.
            k_values_eng = (int(getattr(diffusion, 'n_timesteps', 20)),)
        else:
            k_values_eng = k_values
        for k in k_values_eng:
            engine_registry.apply_nfe(diffusion, k, engine=engine)
            kw = engine_registry.sample_kwargs_for(engine, k)
            with torch.no_grad():
                diffusion.conditional_sample(cond, horizon=HORIZON, **kw)   # warm-up
                t0 = time.perf_counter()
                for _ in range(5):
                    diffusion.conditional_sample(cond, horizon=HORIZON, **kw)
                ms = (time.perf_counter() - t0) * 1e3 / 5
            line.append(f'K={k}: {ms:7.2f} ms')
        print(f'  {engine:3s} ({row["label"][:28]:28s}) ' + '  '.join(line))
    return True


# ── G7 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G7(device='cpu'):
    """Every arm's sampler returns the SAME `infos` contract, and actually times the projector.

    🔴 This gate exists because Gen15 Fix_1 was exactly this bug. `FlowMatchingODE` emitted
    `infos['projection_ms']`; `MeanFlowODE`/`AlphaFlowODE`, written in Gen3v6/v7 where
    `policies.py` had no real-time logging, did not. `Policy` falls back to
    `infos.get('projection_ms', 0.0)`, so the two-time arms silently reported `proj_ms=0.0`
    while `fm` reported the real number — a fabricated cross-arm difference in the metric this
    generation is built to measure, visible only by reading a log line closely.

    A `.get(..., default)` on a cross-engine boundary is invisible when it goes wrong. Assert
    the contract instead: same keys everywhere, and a projector that burns 2 ms per call must
    show up as a non-zero `projection_ms`.
    """
    REQUIRED = {'projection_costs', 'projection_ms'}
    ok = True
    keysets = {}
    for engine in ARMS:
        _model, diffusion, _args = _build(engine, device)
        diffusion.goal_dim = 0
        proj = _StubProjector(gradient=False, threshold=0.5, sleep_s=0.002)
        cond = {0: torch.zeros(2, OBS_DIM, device=device)}
        with torch.no_grad():
            _x, infos = diffusion.conditional_sample(
                cond, horizon=HORIZON, projector=proj, constraints={})
        keysets[engine] = set(infos)

        missing = REQUIRED - set(infos)
        if missing:
            print(f'  ✗ {engine}: infos is missing {sorted(missing)} — the UAV frame reads these')
            ok = False
            continue
        measured = float(infos['projection_ms'])
        expected_floor = 1000 * proj.sleep_s * len(proj.seen_shapes) * 0.5   # generous margin
        if measured <= expected_floor:
            print(f'  ✗ {engine}: projection_ms={measured:.2f} ms after {len(proj.seen_shapes)} '
                  f'projector calls of {proj.sleep_s * 1e3:.0f} ms — the call is not being timed')
            ok = False
        else:
            print(f'  ✓ {engine}: projection_ms={measured:.2f} ms over '
                  f'{len(proj.seen_shapes)} projector calls; keys={sorted(infos)}')

    if len(set(map(frozenset, keysets.values()))) > 1:
        print(f'  ✗ arms disagree on the infos key set: '
              + '; '.join(f'{e}={sorted(k)}' for e, k in keysets.items()))
        ok = False
    elif ok:
        print(f'  ✓ all {len(ARMS)} arms return an identical infos contract')
    return ok



# ── G8 ──────────────────────────────────────────────────────────────────────────────────────

def gate_G8(device='cpu'):
    """Gen15 U2 — the HardFlow arm's two engine-specific values are correct per arm.

    🔴 Both are silent when wrong, which is why they are asserted rather than reviewed:

      `init_noise_scale`  the HardFlow sampler must start its ODE from the SAME distribution
                          the host engine trains and deploys with. `fm` uses `0.5 * randn`
                          (diffusion.py:184); `mf`/`af` use `randn` (sigma=1.0). Gen3v6's port
                          defaulted this to 1.0 because it hosted one engine — on Gen15's `fm`
                          arm that default would start at TWICE the trained scale and merely
                          look like a slightly worse model. This is the fix_4 bug, re-armed by
                          multi-engine hosting.

      `two_time`          decides which FIELD the NLP is handed. `mf`/`af` emit the interval
                          average u(x,r,t) and must be queried at h=0, where the mean-flow
                          identity gives u(x,t,0) = v(x,t) exactly. `fm` emits v directly and
                          its `_predict_velocity` has NO `h` parameter — passing one is a
                          TypeError, omitting it on a two-time arm silently swaps the field.

    Checked against the engines' own `p_sample_loop` source, so the gate cannot drift from the
    thing it is guarding.
    """
    import inspect
    import mix_uav.utils as utils

    EXPECTED = {'fm': (0.5, False), 'mf': (1.0, True), 'af': (1.0, True),
                'diffusion': (0.5, False)}
    ok = True
    for engine in ARMS:
        row = engine_registry.get(engine)
        if not row['supports_hardflow']:
            # U3: the `diffusion` arm has no `_predict_velocity` at all — HardFlow cannot host
            # it, so there is nothing here to get wrong. Assert the exclusion instead.
            import mix_uav.utils as _u
            _cls = _u.import_class(row['diffusion'])
            if hasattr(_cls, '_predict_velocity'):
                print(f'  ✗ {engine}: supports_hardflow=False but {_cls.__name__} HAS '
                      f'_predict_velocity — the exclusion may be wrong')
                ok = False
            else:
                print(f'  ✓ {engine}: supports_hardflow=False and {_cls.__name__} has no '
                      f'velocity field — correctly excluded from the HardFlow arm')
            continue
        want_sigma, want_tt = EXPECTED[engine]
        got_sigma = float(row.get('init_noise_scale', -1))
        got_tt = bool(row.get('two_time', False))

        if got_sigma != want_sigma or got_tt != want_tt:
            print(f'  ✗ {engine}: registry says init_noise_scale={got_sigma}, two_time={got_tt}; '
                  f'expected {want_sigma}, {want_tt}')
            ok = False
            continue

        # Cross-check sigma against the engine's ACTUAL sampler source.
        diff_cls = utils.import_class(row['diffusion'])
        src = inspect.getsource(diff_cls.p_sample_loop)
        has_half = '0.5 * torch.randn(shape' in src
        src_sigma = 0.5 if has_half else 1.0
        if src_sigma != want_sigma:
            print(f'  ✗ {engine}: {diff_cls.__name__}.p_sample_loop starts at sigma={src_sigma} '
                  f'but the registry says {want_sigma} — the port would run off-distribution')
            ok = False
            continue

        # Cross-check two_time against the velocity signature.
        params = inspect.signature(diff_cls._predict_velocity).parameters
        src_tt = 'h' in params
        if src_tt != want_tt:
            print(f'  ✗ {engine}: {diff_cls.__name__}._predict_velocity '
                  f"{'has' if src_tt else 'lacks'} an `h` arg but two_time={want_tt}")
            ok = False
            continue

        print(f'  ✓ {engine}: init_noise_scale={got_sigma} and two_time={got_tt} '
              f'both match {diff_cls.__name__} source')

    # The NLP itself needs casadi, which lives in the cluster env only.
    try:
        import casadi  # noqa: F401
        print('  ✓ casadi importable — the HardFlow NLP can be built')
    except ImportError:
        print('  ⊘ casadi not installed here (cluster-only dep); NLP construction not exercised')
    return ok


def gate_G9():
    """G9 — Gen15 U6: what each of the three af knobs is ALLOWED to move.

    🔴 WHY. U6 flips the af arm's default backbone from 'sit' to 'unet' and adds two knobs.
    Three separate ways that can go wrong, all silent:

      * the BONE and ALPHA knobs must reach the CHECKPOINT path, or two different models share
        one directory and the second overwrites the first (that is what `bb`/`ae` are for, and
        G2 already checks the cross-product — this gate checks the knobs actually RESOLVE);
      * the EPOCH knob must NOT reach the checkpoint path — it selects among files already in
        one tree — but it MUST reach the eval-params folder, or a `latest` pass overwrites the
        `best` pass of the same weights. On the af arm those are different models: `best` is
        chosen on an alpha-weighted test_loss and prefers a MID-CURRICULUM checkpoint;
      * the TRAIN and PLAN blocks must agree on bone and alpha, or eval rebuilds a savepath the
        trainer never wrote and dies on a missing checkpoint after the GPU is allocated.
    """
    import importlib.util
    ok = True
    path = os.path.join(_REPO, 'config', 'uav_mix.py')

    def _load(env):
        for k in ('UAV_MIX_BONE_AF', 'UAV_MIX_AF_ALPHA_END', 'UAV_MIX_EPOCH'):
            os.environ.pop(k, None)
        os.environ.update(env)
        spec = importlib.util.spec_from_file_location('_uav_cfg_probe', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    class A:
        pass

    def _ckpt(mod, blk):
        a = A()
        for k, v in blk.items():
            setattr(a, k, v)
        return mod._uav_mix_exp_name(a)

    try:
        m_def  = _load({})
        m_sit  = _load({'UAV_MIX_BONE_AF': 'sit'})
        m_a02  = _load({'UAV_MIX_AF_ALPHA_END': '0.2'})
        m_ep   = _load({'UAV_MIX_EPOCH': 'latest'})
    finally:
        for k in ('UAV_MIX_BONE_AF', 'UAV_MIX_AF_ALPHA_END', 'UAV_MIX_EPOCH'):
            os.environ.pop(k, None)
        sys.modules.pop('_uav_cfg_probe', None)

    # (a) the config's bone whitelist must not drift from the registry's.
    _reg = tuple(engine_registry.get('af')['backbones'])
    if tuple(m_def._UAV_AF_BONES) != _reg:
        ok = False
        print(f'  ✗ (a) config _UAV_AF_BONES={m_def._UAV_AF_BONES} != registry {_reg} — the '
              f'config would accept or reject a bone the model layer disagrees about')
    else:
        print(f'  ✓ (a) bone whitelist agrees with engine_registry: {list(_reg)}')

    # (b) the U6 defaults, and train/plan agreement on every env-resolved key.
    tb, pb = m_def.base['mix_uav_af'], m_def.base['plan_mix_uav_af']
    if tb['imf_backbone'] != 'unet':
        ok = False
        print(f"  ✗ (b) default af bone is {tb['imf_backbone']!r}, expected 'unet' — U6 exists "
              f"to make the af arm parameter-matched to the 4.0 M fm/mf U-Net")
    else:
        print("  ✓ (b) default af bone = 'unet' (was 'sit'; ~9.4 M and NOT param-matched)")
    if float(tb['af_alpha_end']) != 0.0 or pb['diffusion_epoch'] != 'best':
        ok = False
        print(f"  ✗ (b) shipped defaults moved: af_alpha_end={tb['af_alpha_end']}, "
              f"diffusion_epoch={pb['diffusion_epoch']!r} (want 0.0 / 'best')")
    else:
        print("  ✓ (b) shipped defaults intact: af_alpha_end=0.0, diffusion_epoch='best'")
    for mod, label in ((m_def, 'default'), (m_sit, 'sit'), (m_a02, 'alpha0.2')):
        _t, _p = mod.base['mix_uav_af'], mod.base['plan_mix_uav_af']
        _bad = [k for k in ('imf_backbone', 'af_alpha_end') if _t[k] != _p[k]]
        if _bad:
            ok = False
            print(f'  ✗ (b) {label}: train/plan disagree on {_bad} — eval would rebuild a '
                  f'savepath the trainer never wrote')
    if ok:
        print('  ✓ (b) train and plan blocks agree on bone + alpha in every configuration')

    # (c) bone and alpha DO move the checkpoint tree, and each value gets its own.
    names = {lbl: _ckpt(mod, mod.base['mix_uav_af'])
             for lbl, mod in (('default(unet)', m_def), ('sit', m_sit), ('alpha0.2', m_a02))}
    if len(set(names.values())) != 3:
        ok = False
        print(f'  ✗ (c) checkpoint-path COLLISION across bone/alpha values: {names}')
    else:
        print('  ✓ (c) bone and alpha each land in their own checkpoint tree:')
        for lbl, nm in names.items():
            print(f'          {lbl:14s} {nm}')

    # (d) the epoch knob must NOT touch the checkpoint tree.
    if _ckpt(m_ep, m_ep.base['mix_uav_af']) != names['default(unet)']:
        ok = False
        print('  ✗ (d) UAV_MIX_EPOCH moved the CHECKPOINT path — it is an EVAL-only selector '
              'and must never orphan a checkpoint')
    elif m_ep.base['plan_mix_uav_af']['diffusion_epoch'] != 'latest':
        ok = False
        print(f"  ✗ (d) UAV_MIX_EPOCH=latest did not reach the plan block "
              f"({m_ep.base['plan_mix_uav_af']['diffusion_epoch']!r})")
    else:
        print("  ✓ (d) UAV_MIX_EPOCH reaches the plan block and leaves the checkpoint path alone")

    # (e) ...but it MUST reach the eval-params folder, or `latest` overwrites `best`.
    try:
        from mix_uav_test.eval_mix_uav import _uav_eval_tag
    except Exception as e:
        print(f'  ⊘ (e) could not import _uav_eval_tag ({type(e).__name__}: {e}) — '
              f'the eval-folder token is UNCHECKED in this run')
    else:
        _base = dict(flow_steps_v3=20, mpc_batch_size=4, diffusion_timestep_threshold=0.5)
        t_best = _uav_eval_tag({**_base, 'diffusion_epoch': 'best'}, 'pid_stopgo', engine='af')
        t_late = _uav_eval_tag({**_base, 'diffusion_epoch': 'latest'}, 'pid_stopgo', engine='af')
        t_none = _uav_eval_tag(_base, 'pid_stopgo', engine='af')
        if t_best != t_none:
            ok = False
            print(f"  ✗ (e) an explicit 'best' changed the folder name ({t_none} -> {t_best}) — "
                  f"every existing UAV results path would move")
        elif t_late != t_best + '_EPlatest':
            ok = False
            print(f'  ✗ (e) eval-params folder is not a strict extension.\n'
                  f'          best  : {t_best}\n          latest: {t_late}')
        else:
            print(f"  ✓ (e) eval folder: {t_best}  (+'_EPlatest' when non-default; "
                  f"pre-U6 names unchanged)")

    # (f) a typo must die at config time, not as state_<garbage>.pt inside a GPU allocation.
    _bad_ok = True
    for env in ({'UAV_MIX_BONE_AF': 'mf_dit'},        # the mf arm's class, not af's
                {'UAV_MIX_BONE_AF': 'unett'},
                {'UAV_MIX_AF_ALPHA_END': '1.5'},      # outside [0, 1]
                {'UAV_MIX_AF_ALPHA_END': '0.001'},    # below af_alpha_clamp -> snaps to 0
                {'UAV_MIX_AF_ALPHA_END': 'abc'},
                {'UAV_MIX_EPOCH': 'lastest'}):
        try:
            _load(env)
        except ValueError:
            continue
        except Exception as e:
            print(f'  ⊘ (f) {env} raised {type(e).__name__} instead of ValueError: {e}')
            continue
        finally:
            for k in ('UAV_MIX_BONE_AF', 'UAV_MIX_AF_ALPHA_END', 'UAV_MIX_EPOCH'):
                os.environ.pop(k, None)
            sys.modules.pop('_uav_cfg_probe', None)
        ok = _bad_ok = False
        print(f'  ✗ (f) {env} was ACCEPTED — it would surface as a missing checkpoint or a '
              f'lying folder name hours later')
    if _bad_ok:
        print('  ✓ (f) malformed bone / alpha / epoch values rejected at config-import time')

    return ok


GATES = {
    'G0': ('every arm builds', lambda a: gate_G0(a.device)),
    'G1': ('fm parity vs Gen11', lambda a: gate_G1(a.gen11_savepath, a.gen15_savepath)),
    'G2': ('path collision + Gen11 isolation', lambda a: gate_G2()),
    'G3': ('backbone identity', lambda a: gate_G3(a.device)),
    'G4': ('two-time (t, h) domain', lambda a: gate_G4(a.device)),
    'G5': ('projector sees full trajectory', lambda a: gate_G5(a.device)),
    'G6': ('per-plan wall clock', lambda a: gate_G6(a.device)),
    'G7': ('infos contract + projector timing', lambda a: gate_G7(a.device)),
    'G8': ('hardflow arm: noise scale + field', lambda a: gate_G8(a.device)),
    'G9': ('U6 af knobs: bone / alpha / epoch path safety', lambda a: gate_G9()),
}


def main():
    p = argparse.ArgumentParser(description='Gen15 UAV Mix-ML gates.')
    p.add_argument('--gates', nargs='+', default=list(GATES.keys()),
                   choices=list(GATES.keys()), help='Subset of gates to run.')
    p.add_argument('--device', type=str, default='cpu',
                   help="'cpu' (default, works on a login node) or 'cuda' for real G6 timings.")
    p.add_argument('--gen11-savepath', type=str, default=None,
                   help='A trained Gen11 run directory (enables G1).')
    p.add_argument('--gen15-savepath', type=str, default=None,
                   help='The matching Gen15 `fm` run directory (enables G1).')
    args = p.parse_args()

    results = {}
    for name in args.gates:
        title, fn = GATES[name]
        print(f'\n── {name}: {title} ' + '─' * max(0, 60 - len(title)))
        try:
            results[name] = fn(args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  ✗ {name} raised {type(e).__name__}: {e}')
            results[name] = False

    print('\n' + '=' * 72)
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    for k in args.gates:
        mark = {True: 'PASS', False: 'FAIL', None: 'SKIP'}[results.get(k)]
        print(f'  {k}  {mark:4s}  {GATES[k][0]}')
    print('=' * 72)
    if skipped:
        print(f'skipped: {", ".join(skipped)}')
    if failed:
        print(f'FAILED: {", ".join(failed)}')
        sys.exit(1)
    print('all requested gates passed')


if __name__ == '__main__':
    main()
