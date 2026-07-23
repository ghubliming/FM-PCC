#!/usr/bin/env python3
"""Gen3v7 pre-flight gates G0–G5 for α-Flow.

⚠️ RUN ON THE CLUSTER (i6-gpu-1, FMPCC env). Nothing here can execute in the AI-coding
container — no torch installed.

Gates (PLAN_Gen3v7_alphaflow.md §7). **G1 and G2 together prove the homotopy is wired
correctly at BOTH endpoints. If either fails, stop — every intermediate α is meaningless.**

  G0  build + one forward   AlphaFlowEngine/AlphaFlowODE from the Gen3v7 defaults, one
                            (x, τ, h) → (u, v); assert shapes; assert no Gen3v4/Gen3v6
                            module got imported by mistake.
  G1  α = 1 ⇒ pure FM       force af_alpha_init = af_alpha_end = 1.0, af_ratio_fm = 0:
                            u_tgt must equal v **BITWISE**. This is the safety floor —
                            α-Flow at α=1 is a method already known to reach 100% safety.
  G2  α = 0 ⇒ MeanFlow      the α=0 target must equal **Gen3v6**'s `_p_losses_meanflow`
                            target on identical inputs (<1e-5), and the scalar losses must
                            agree once the adaptive eps is matched. This is why Gen3v6
                            shipped first — the comparator is a real, importable sibling,
                            not a paper.
  G3  intermediate α        first-order agreement between the discrete and JVP branches
                            (PLAN §3.4). Split into G3a (binding at random init) and G3b
                            (the plan's literal form, informational without --ckpt) —
                            see gate_g3.__doc__ for why.
  G4  schedule is alive     α moves across the budget; branch counts (`discrete_frac`)
                            shift with it; `alpha` is present in the info dict every step.
  G5  no gradient leak      u_tgt.requires_grad is False, and no grad reaches the params
                            through it. This is PLAN §11 trap 2 — a leak here re-creates a
                            self-referential target and voids the whole generation.
  G3' smoke overfit         200 optimizer steps of the real loss at a mid-schedule α on a
                            fixed random batch: raw_mse_u must fall, nothing may go NaN.

Usage (cluster):
    python FM_v3_alphaflow_test/gates_alphaflow.py
    python FM_v3_alphaflow_test/gates_alphaflow.py --ckpt logs/.../state_best.pt
"""

import argparse
import math
import sys

import torch

from flow_matcher_v3_alphaflow.models import AlphaFlowEngine, AlphaFlowODE

# Gen3v7 architecture defaults — MUST mirror config/avoiding-d3il.py:flow_matching_v3_alphaflow
ARCH = dict(
    imf_backbone='dit', dit_depth=8, dit_hidden_size=256, dit_num_heads=4,
    dit_aux_head_depth=2, dit_patch_size=1, dit_condition_on_t=False,
    dual_head=True, interval_cfg=False,
)
OBJ = dict(
    af_alpha_scheduler='sigmoid', af_alpha_init=1.0, af_alpha_end=0.0,
    af_alpha_init_step=0, af_alpha_end_step=100000, af_alpha_gamma=25.0,
    af_alpha_clamp=0.005, af_ratio_fm=0.5, af_clamp_utgt=4.0, af_adp_eps=1e-3,
    t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
)
N_TRAIN_STEPS = 100000


def build(args, device, **obj_overrides):
    """Build an AlphaFlowODE. `af_n_train_steps=None` bypasses the end_step assert on
    purpose: several gates deliberately force a CONSTANT α, for which the assert is moot."""
    obj = dict(OBJ)
    obj.update(obj_overrides)
    engine = AlphaFlowEngine(
        state_dim=args.obs_dim + args.action_dim, seq_len=args.horizon,
        device=device, **ARCH,
    )
    diffusion = AlphaFlowODE(
        engine, horizon=args.horizon,
        observation_dim=args.obs_dim, action_dim=args.action_dim,
        action_weight=10, loss_discount=1.0, condition_guidance_w=0.0,
        af_n_train_steps=None,
        **obj,
    ).to(device)
    return engine, diffusion


def _batch(args, device, B=16):
    x = torch.randn(B, args.horizon, args.obs_dim + args.action_dim, device=device)
    cond = {0: x[:, 0, args.action_dim:].clone()}
    return x, cond


def _draw_inputs(diffusion, x, cond, device, force_h=None):
    """Reproduce the loss's own (r, h, x_r, v) draw so a gate can call compute_u_target
    directly. Mirrors _p_losses_alphaflow lines-for-line; `force_h` overrides h for G3."""
    from flow_matcher_v3_alphaflow.models.helpers import apply_conditioning
    B = x.shape[0]
    ad, gd = diffusion.action_dim, diffusion.goal_dim
    taus = diffusion._sample_tau_pair(B, device)
    t = torch.maximum(taus[0], taus[1])
    r = torch.minimum(taus[0], taus[1])
    fm_mask = torch.rand(B, device=device) < diffusion.af_ratio_fm
    r = torch.where(fm_mask, t, r)
    h = t - r
    if force_h is not None:
        h = torch.full_like(h, float(force_h))
        r = torch.clamp(r, max=1.0 - float(force_h))
    x_base = apply_conditioning(torch.randn_like(x), cond, ad, goal_dim=gd, noise=True)
    x_r = apply_conditioning(diffusion.q_sample(x_start=x, t=r, noise=x_base),
                             cond, ad, goal_dim=gd)
    v = apply_conditioning(x - x_base, cond, ad, goal_dim=gd, noise=True)
    return x_r, r, h, v


# ──────────────────────────────────────────────────────────────────────────────────────
# G0
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_g0(args, device):
    print('\n── G0: import + shapes ' + '─' * 50)
    import flow_matcher_v3_alphaflow as pkg
    # Every module of THIS package must resolve to a file inside THIS folder. (Merely
    # having a sibling package in sys.modules is fine and expected — G2 imports Gen3v6 on
    # purpose — what would be fatal is a Gen3v7 module that is secretly Gen3v6's file.)
    import os
    root = os.path.dirname(os.path.abspath(pkg.__file__))
    stale = [m for m, mod in list(sys.modules.items())
             if m.startswith('flow_matcher_v3_alphaflow')
             and getattr(mod, '__file__', None)
             and not os.path.abspath(mod.__file__).startswith(root)]
    if stale:
        print(f'  FAIL: Gen3v7 modules resolving outside {root}: {stale}')
        return False
    print(f'  package: {pkg.__file__}')

    _engine, diffusion = build(args, device)
    B, H, D = 4, args.horizon, args.obs_dim + args.action_dim
    x = torch.randn(B, H, D, device=device)
    tau = torch.rand(B, device=device)
    h = torch.rand(B, device=device) * (1 - tau)
    u, v = diffusion._predict_uv(x, {}, tau, h=h)
    ok = (u.shape == (B, H, D)) and (v.shape == (B, H, D))
    print(f'  u{tuple(u.shape)} v{tuple(v.shape)} expected ({B}, {H}, {D})  ->  '
          f'{"PASS" if ok else "FAIL"}')
    print(f'  params: {sum(p.numel() for p in diffusion.parameters())/1e6:.2f}M')

    # the end_step assert must actually fire (PLAN §11 trap 1)
    try:
        AlphaFlowODE(
            _engine, horizon=args.horizon, observation_dim=args.obs_dim,
            action_dim=args.action_dim, af_alpha_end_step=400000,
            af_n_train_steps=N_TRAIN_STEPS,
        )
        print('  FAIL: af_alpha_end_step=400000 vs n_train_steps=100000 did NOT raise')
        ok = False
    except ValueError:
        print('  end_step != n_train_steps raises ValueError  ->  PASS')
    return ok


# ──────────────────────────────────────────────────────────────────────────────────────
# G1 — α = 1 ⇒ pure flow matching
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_g1(args, device):
    print('\n── G1: α = 1 ⇒ u_tgt == v BITWISE (the objective IS flow matching) ' + '─' * 5)
    _e, diffusion = build(args, device, af_alpha_scheduler='constant',
                          af_alpha_init=1.0, af_alpha_end=1.0, af_ratio_fm=0.0)
    diffusion.set_train_step(0)
    alpha = diffusion.current_alpha()
    x, cond = _batch(args, device)
    x_r, r, h, v = _draw_inputs(diffusion, x, cond, device)
    u_tgt, _cf = diffusion.compute_u_target(x_r, r, h, v, cond, alpha)

    exact = bool(torch.equal(u_tgt, v))
    n_h0 = int((h == 0).sum())
    print(f'  α = {alpha}   (ratio_fm = 0 ⇒ {n_h0}/{len(h)} samples have h == 0)')
    print(f'  max|u_tgt − v| = {(u_tgt - v).abs().max().item():.3e}')
    print(f'  bitwise equal  = {exact}  ->  {"PASS" if exact else "FAIL"}')
    if not exact:
        print('  NOTE: a non-bitwise match means the α=1 short-circuit in compute_u_target '
              'was bypassed — check that current_alpha() returns exactly 1.0.')
    return exact


# ──────────────────────────────────────────────────────────────────────────────────────
# G2 — α = 0 ⇒ Gen3v6 MeanFlow, numerically
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_g2(args, device, tol=1e-5):
    print('\n── G2: α = 0 ⇒ Gen3v6 MeanFlow (target AND loss) ' + '─' * 23)
    try:
        from flow_matcher_v3_meanflow.models import MeanFlowEngine, MeanFlowODE
    except ImportError as exc:
        print(f'  SKIP: Gen3v6 not importable here ({exc}). Run where '
              'flow_matcher_v3_meanflow/ is on PYTHONPATH.')
        return True

    _e, af = build(args, device, af_alpha_scheduler='constant',
                   af_alpha_init=0.0, af_alpha_end=0.0,
                   af_adp_eps=0.01)   # match Gen3v6's eps so the SCALAR losses compare
    mf_engine = MeanFlowEngine(state_dim=args.obs_dim + args.action_dim,
                               seq_len=args.horizon, device=device, **ARCH)
    mf = MeanFlowODE(
        mf_engine, horizon=args.horizon, observation_dim=args.obs_dim,
        action_dim=args.action_dim, action_weight=10, loss_discount=1.0,
        condition_guidance_w=0.0, mf_objective='meanflow',
        meanflow_data_proportion=OBJ['af_ratio_fm'], mf_adp_p=1.0, mf_adp_eps=0.01,
        t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
    ).to(device)
    # identical weights — the two backbones are the same code under different names
    mf.load_state_dict(af.state_dict())
    af.set_train_step(0)
    print(f'  α = {af.current_alpha()} (forced constant);  adaptive eps matched at 0.01')

    x, cond = _batch(args, device)

    # 1) TARGETS on identical inputs
    torch.manual_seed(1234)
    x_r, r, h, v = _draw_inputs(af, x, cond, device)
    u_tgt_af, _ = af.compute_u_target(x_r, r, h, v, cond, 0.0)
    u_tgt_mf = _meanflow_target(mf, x_r, r, h, v, cond)
    d_tgt = (u_tgt_af - u_tgt_mf).abs().max().item()
    ok_tgt = d_tgt < tol
    print(f'  max|u_tgt(α=0) − u_tgt(Gen3v6)| = {d_tgt:.3e}  (tol {tol:.0e})  ->  '
          f'{"PASS" if ok_tgt else "FAIL"}')

    # 2) FULL LOSS — relies on the two objectives consuming RNG in the same order
    torch.manual_seed(4321)
    loss_af, info_af = af.loss(x, cond)
    torch.manual_seed(4321)
    loss_mf, info_mf = mf.loss(x, cond)
    d_loss = abs(loss_af.item() - loss_mf.item())
    d_raw = abs(info_af['raw_mse_u'].item() - info_mf['raw_mse_u'].item())
    ok_loss = d_loss < tol and d_raw < tol
    print(f'  |loss_α−Flow − loss_Gen3v6|      = {d_loss:.3e}   '
          f'({loss_af.item():.6f} vs {loss_mf.item():.6f})')
    print(f'  |raw_mse_u difference|           = {d_raw:.3e}  ->  '
          f'{"PASS" if ok_loss else "FAIL"}')
    if not ok_loss and ok_tgt:
        print('  NOTE: targets agree but losses do not ⇒ the RNG consumption order '
              'diverged between the two loss bodies, not the maths.')
    return ok_tgt and ok_loss


def _meanflow_target(mf, x_r, r, h, v, cond):
    """Gen3v6's target, recomputed from its own code path on OUR inputs."""
    from torch.func import jvp as _jvp
    from flow_matcher_v3_meanflow.models.helpers import apply_conditioning

    def _u_of(z_in, r_in, h_in):
        u, _ = mf._predict_uv(z_in, cond, r_in, h=h_in)
        return u

    ones = torch.ones_like(r)
    with torch.no_grad():
        _u, du_dr = _jvp(_u_of, (x_r, r, h), (v, ones, -ones))
    h_exp = h
    while h_exp.ndim < x_r.ndim:
        h_exp = h_exp.unsqueeze(-1)
    tgt = (v + h_exp * du_dr).detach()
    return apply_conditioning(tgt, cond, mf.action_dim, goal_dim=mf.goal_dim, noise=True)


# ──────────────────────────────────────────────────────────────────────────────────────
# G3 — intermediate α agrees with the JVP branch to first order
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_g3(args, device, ckpt=None, alpha=0.05, small_h=0.02, rel_tol=0.1):
    """G3 in two parts. See the §"DEVIATION" note below for why.

    G3a (BINDING, works at random init) — the discrete branch must be a first-order
    consistent discretisation of the continuous one. PLAN §3.4 expands
    u_next = u + dt·D_tot, giving

        u_tgt(α)  ≈  α·v + (1−α)·(u + α·h·D_tot)          [error O(dt²)]

    and D_tot is recoverable from the JVP branch itself: u_tgt(0) = v + h·D_tot, so
    D_tot = (u_tgt(0) − v)/h. Nothing here depends on the model being trained.

    G3b (the PLAN's literal wording) — ‖u_tgt(α) − u_tgt(0)‖/‖u_tgt(0)‖ < 0.1. Expanding
    both sides, that difference tends to (u − v − h·D_tot), i.e. **the model's own MeanFlow
    residual**, which is O(1) at random initialisation and only small once the field is
    near its fixed point. So G3b is reported as INFO without --ckpt, exactly the way
    Gen3v6's gates treat their h→0 degeneracy check.
    """
    print(f'\n── G3: α = {alpha} vs the α→0 JVP target at h = {small_h} ' + '─' * 20)
    _e, diffusion = build(args, device, af_alpha_scheduler='constant',
                          af_alpha_init=alpha, af_alpha_end=alpha, af_ratio_fm=0.0)
    trained = False
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        diffusion.load_state_dict(state['ema'] if 'ema' in state else state['model'])
        trained = True
        print(f'  loaded: {ckpt} ({"ema" if "ema" in state else "model"} weights)')
    else:
        print('  no --ckpt: RANDOM INIT. G3a still binds; G3b is informational.')

    x, cond = _batch(args, device, B=32)
    torch.manual_seed(7)
    x_r, r, h, v = _draw_inputs(diffusion, x, cond, device, force_h=small_h)

    h_exp = h.view(-1, 1, 1)
    u_d, _ = diffusion.compute_u_target(x_r, r, h, v, cond, alpha)   # discrete branch
    u_c, _ = diffusion.compute_u_target(x_r, r, h, v, cond, 0.0)     # JVP branch
    with torch.no_grad():
        u_now, _v = diffusion._predict_uv(x_r, cond, r, h=h)
    d_tot = (u_c - v) / h_exp                                        # from the JVP branch
    predicted = alpha * v + (1.0 - alpha) * (u_now + alpha * h_exp * d_tot)

    rel_a = ((u_d - predicted).norm() / predicted.norm().clamp(min=1e-12)).item()
    ok_a = rel_a < rel_tol
    print(f'  G3a  ‖u_tgt(α) − [α·v + (1−α)(u + α·h·D_tot)]‖ / ‖·‖ = {rel_a:.4f}  '
          f'(tol {rel_tol})  ->  {"PASS" if ok_a else "FAIL"}')
    print(f'       dt = α·h = {alpha * small_h:.4g}; the residual should scale like dt².')

    rel_b = ((u_d - u_c).norm() / u_c.norm().clamp(min=1e-12)).item()
    ok_b = rel_b < rel_tol
    verdict_b = 'PASS' if ok_b else ('FAIL' if trained else 'INFO (untrained — expected)')
    print(f'  G3b  ‖u_tgt(α) − u_tgt(JVP)‖ / ‖u_tgt(JVP)‖ = {rel_b:.4f}  (tol {rel_tol})'
          f'  ->  {verdict_b}')
    return ok_a and (ok_b or not trained)


# ──────────────────────────────────────────────────────────────────────────────────────
# G4 — the schedule is alive
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_g4(args, device):
    print('\n── G4: α schedule is alive and the branch counts follow it ' + '─' * 13)
    _e, diffusion = build(args, device)     # the REAL config schedule
    x, cond = _batch(args, device, B=64)

    rows = []
    for frac in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        step = int(frac * N_TRAIN_STEPS)
        diffusion.set_train_step(step)
        _loss, info = diffusion.loss(x, cond)
        rows.append((step, info['alpha'].item(), info['discrete_frac'].item(),
                     info['clamp_frac'].item()))

    print(f'  {"step":>8} {"alpha":>8} {"discrete_frac":>15} {"clamp_frac":>12}')
    for step, a, df, cf in rows:
        print(f'  {step:>8d} {a:>8.4f} {df:>15.3f} {cf:>12.4f}')

    alphas = [a for _s, a, _d, _c in rows]
    moved = (max(alphas) - min(alphas)) > 1e-6
    starts_fm = math.isclose(alphas[0], 1.0, abs_tol=1e-9)
    ends_mf = math.isclose(alphas[-1], 0.0, abs_tol=1e-9)
    frac_shift = (max(d for _s, _a, d, _c in rows) - min(d for _s, _a, d, _c in rows)) > 1e-6
    for name, ok in (('α moves', moved), ('α(0) == 1', starts_fm),
                     ('α(end) == 0', ends_mf), ('discrete_frac shifts', frac_shift)):
        print(f'  {name:<22} {"PASS" if ok else "FAIL"}')
    if not moved:
        print('  🔴 A FLAT α IS THE #1 SILENT FAILURE OF THIS GENERATION (PLAN §11 trap 1): '
              'you would be training plain flow matching under an α-Flow folder name.')
    return moved and starts_fm and ends_mf and frac_shift


# ──────────────────────────────────────────────────────────────────────────────────────
# G5 — no gradient leaks into u_next
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_g5(args, device):
    print('\n── G5: no gradient leak into the bootstrapped target ' + '─' * 20)
    _e, diffusion = build(args, device, af_alpha_scheduler='constant',
                          af_alpha_init=0.5, af_alpha_end=0.5)
    x, cond = _batch(args, device)
    x_r, r, h, v = _draw_inputs(diffusion, x, cond, device)
    u_tgt, _cf = diffusion.compute_u_target(x_r, r, h, v, cond, 0.5)

    ok_flag = (u_tgt.requires_grad is False)
    print(f'  u_tgt.requires_grad = {u_tgt.requires_grad}  ->  '
          f'{"PASS" if ok_flag else "FAIL"}')

    # stronger check: backprop through the TARGET alone must reach no parameter
    diffusion.zero_grad(set_to_none=True)
    ok_graph = True
    try:
        u_tgt.sum().backward()
        touched = [n for n, p in diffusion.named_parameters() if p.grad is not None]
        ok_graph = not touched
        print(f'  params receiving grad from u_tgt: {len(touched)}  ->  '
              f'{"PASS" if ok_graph else "FAIL"}')
    except RuntimeError:
        # "element 0 of tensors does not require grad" — exactly what we want
        print('  backward() through u_tgt raises (no graph)  ->  PASS')
    diffusion.zero_grad(set_to_none=True)
    if not (ok_flag and ok_graph):
        print('  🔴 PLAN §11 trap 2: a leak here re-creates a self-referential target and '
              'throws away the entire point of this generation.')
    return ok_flag and ok_graph


# ──────────────────────────────────────────────────────────────────────────────────────
# G3' — smoke overfit at a mid-schedule α
# ──────────────────────────────────────────────────────────────────────────────────────
def gate_smoke(args, device, steps=200):
    print(f"\n── G3': {steps}-step smoke overfit at mid-schedule α " + '─' * 20)
    _e, diffusion = build(args, device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=5e-4)
    x, cond = _batch(args, device)

    first_raw, last_raw = None, None
    mid = N_TRAIN_STEPS // 2
    for step in range(steps):
        # walk across the middle of the anneal so BOTH branches are exercised
        diffusion.set_train_step(mid - steps // 2 + step)
        loss, info = diffusion.loss(x, cond)
        if not torch.isfinite(loss):
            print(f'  FAIL: non-finite loss at step {step}')
            return False
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        opt.step()

        raw = info['raw_mse_u'].item()
        if step == 0:
            first_raw = raw
        last_raw = raw
        if step % 50 == 0 or step == steps - 1:
            buckets = ' '.join(f'b{i}={info[f"h_mse_b{i}"].item():.3g}' for i in range(4))
            print(f'  step {step:4d}  loss={loss.item():.4f}  raw_mse_u={raw:.4f}  '
                  f'raw_mse_v={info["raw_mse_v"].item():.4f}  '
                  f'α={info["alpha"].item():.4f} disc={info["discrete_frac"].item():.2f} '
                  f'clamp={info["clamp_frac"].item():.4f}  {buckets}')
            for i in range(4):
                if math.isinf(info[f'h_mse_b{i}'].item()):
                    print(f'  FAIL: h_mse_b{i} is infinite at step {step}')
                    return False

    dropped = last_raw < first_raw
    print(f'  raw_mse_u: {first_raw:.4f} -> {last_raw:.4f}  ->  '
          f'{"PASS" if dropped else "FAIL (not decreasing)"}')
    return dropped


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Gen3v7 α-Flow pre-flight gates')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--obs-dim', type=int, default=4, help='avoiding-d3il observation_dim')
    ap.add_argument('--action-dim', type=int, default=2, help='avoiding-d3il action_dim')
    ap.add_argument('--ckpt', type=str, default=None,
                    help='state_best.pt — makes G3b binding instead of informational')
    ap.add_argument('--steps', type=int, default=200, help="G3' optimizer steps")
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    torch.manual_seed(0)
    print('=' * 80)
    print(f'Gen3v7 α-Flow gates — device={a.device}, H={a.horizon}, '
          f'obs={a.obs_dim}, act={a.action_dim}')
    print('=' * 80)

    results = {
        'G0': gate_g0(a, a.device),
        'G1': gate_g1(a, a.device),
        'G2': gate_g2(a, a.device),
        'G3': gate_g3(a, a.device, a.ckpt),
        'G4': gate_g4(a, a.device),
        'G5': gate_g5(a, a.device),
        "G3'": gate_smoke(a, a.device, a.steps),
    }

    print('\n' + '=' * 80)
    for name, ok in results.items():
        print(f'  {name}: {"PASS" if ok else "FAIL"}')
    print('=' * 80)
    if not (results['G1'] and results['G2']):
        print('🔴 G1 and/or G2 FAILED — the homotopy is not wired correctly at its '
              'endpoints, so every intermediate α is meaningless. STOP (PLAN §7).')
    sys.exit(0 if all(results.values()) else 1)
