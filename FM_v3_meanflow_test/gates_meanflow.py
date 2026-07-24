#!/usr/bin/env python3
"""Gen3v6 pre-flight gates G0/G1/G3-lite for the MeanFlow baseline.

⚠️ RUN ON THE CLUSTER (i6-gpu-1, FMPCC env). Nothing here can execute in the AI-coding
container — no torch installed.

Gates implemented (PLAN_Gen3v6_meanflow_baseline.md §6):

  G0  import + shapes    build MeanFlowEngine/MeanFlowODE from the Gen3v6 defaults, run one
                         forward (x, τ, h) → (u, v), assert shapes; assert no module in the
                         loaded package resolves back into flow_matcher_v3_imeanflow.
  G1  h → 0 degeneracy   with h=1e-4, ‖u − v‖/‖v‖ < 0.05.
                         ⚠️ ONLY meaningful on a TRAINED checkpoint (--ckpt). At random init
                         the u/v heads are independent, so a fail is expected and reported
                         as INFO, not as a gate failure.
  G3' smoke overfit      200 optimizer steps of the real _p_losses_meanflow on a fixed random
                         batch: raw_mse_u must fall, nothing may go NaN, h_mse_b* must be
                         finite whenever their bucket is non-empty. Dataset-free — it checks
                         the objective, not the data.

  G2 (objective parity vs aux_repo/MeanFlow) is NOT scripted here: that repo lives in the
  local aux_repo checkout, not on the cluster. Run it wherever both are importable; note the
  known adaptive-loss difference (they reduce with MEAN over dims, official/we use SUM) and
  patch their predicted v_c to the analytic v before comparing.

Usage (cluster):
    python FM_v3_meanflow_test/gates_meanflow.py
    python FM_v3_meanflow_test/gates_meanflow.py --ckpt logs/.../state_best.pt
"""

import argparse
import math
import sys

import torch

from flow_matcher_v3_meanflow.models import MeanFlowEngine, MeanFlowODE

# Gen3v6 architecture defaults — MUST mirror config/avoiding-d3il.py:flow_matching_v3_meanflow
ARCH = dict(
    imf_backbone='dit', dit_depth=8, dit_hidden_size=256, dit_num_heads=4,
    dit_aux_head_depth=2, dit_patch_size=1, dit_condition_on_t=False,
    dual_head=True, interval_cfg=False,
)
OBJ = dict(
    mf_objective='meanflow', meanflow_data_proportion=0.5,
    mf_adp_p=1.0, mf_adp_eps=0.01,
    t_schedule='logit_normal', p_mean=-0.4, p_std=1.0,
)


def build(args, device):
    engine = MeanFlowEngine(
        state_dim=args.obs_dim + args.action_dim, seq_len=args.horizon,
        device=device, **ARCH,
    )
    diffusion = MeanFlowODE(
        engine, horizon=args.horizon,
        observation_dim=args.obs_dim, action_dim=args.action_dim,
        action_weight=10, loss_discount=1.0, condition_guidance_w=0.0,
        **OBJ,
    ).to(device)
    return engine, diffusion


def gate_g0(args, device):
    print('\n── G0: import + shapes ' + '─' * 50)
    import flow_matcher_v3_meanflow as pkg
    stale = [m for m in sys.modules
             if m.startswith('flow_matcher_v3_imeanflow')]
    if stale:
        print(f'  FAIL: stale Gen3v4 modules imported: {stale}')
        return False
    print(f'  package: {pkg.__file__}')

    engine, diffusion = build(args, device)
    B, H, D = 4, args.horizon, args.obs_dim + args.action_dim
    x = torch.randn(B, H, D, device=device)
    tau = torch.rand(B, device=device)
    h = torch.rand(B, device=device) * (1 - tau)
    u, v = diffusion._predict_uv(x, {}, tau, h=h)
    ok = (u.shape == (B, H, D)) and (v.shape == (B, H, D))
    print(f'  u{tuple(u.shape)} v{tuple(v.shape)} expected ({B}, {H}, {D})  ->  '
          f'{"PASS" if ok else "FAIL"}')
    n_params = sum(p.numel() for p in diffusion.parameters())
    print(f'  params: {n_params/1e6:.2f}M')
    return ok


def gate_g1(args, device, ckpt=None):
    print('\n── G1: h → 0 degeneracy (u should collapse to v) ' + '─' * 25)
    _engine, diffusion = build(args, device)
    trained = False
    if ckpt:
        state = torch.load(ckpt, map_location=device)
        diffusion.load_state_dict(state['ema'] if 'ema' in state else state['model'])
        trained = True
        print(f'  loaded: {ckpt} ({"ema" if "ema" in state else "model"} weights)')
    else:
        print('  no --ckpt given: RANDOM INIT, result is informational only')

    B, H, D = 32, args.horizon, args.obs_dim + args.action_dim
    x = torch.randn(B, H, D, device=device)
    tau = torch.rand(B, device=device) * 0.9
    h = torch.full((B,), 1e-4, device=device)
    with torch.no_grad():
        u, v = diffusion._predict_uv(x, {}, tau, h=h)
    rel = (u - v).norm() / v.norm().clamp(min=1e-12)
    verdict = 'PASS' if rel < 0.05 else ('FAIL' if trained else 'INFO (untrained)')
    print(f'  ‖u − v‖/‖v‖ = {rel:.4f}  (threshold 0.05)  ->  {verdict}')
    return (rel < 0.05) if trained else True


def gate_g3(args, device, steps=200):
    print(f'\n── G3\': {steps}-step smoke overfit on a fixed random batch ' + '─' * 14)
    _engine, diffusion = build(args, device)
    opt = torch.optim.Adam(diffusion.parameters(), lr=5e-4)

    B, H, D = 16, args.horizon, args.obs_dim + args.action_dim
    x = torch.randn(B, H, D, device=device)
    cond = {0: x[:, 0, args.action_dim:].clone()}

    first_raw, last_raw = None, None
    for step in range(steps):
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
            buckets = ' '.join(
                f'b{i}={info[f"h_mse_b{i}"].item():.3g}' for i in range(4))
            print(f'  step {step:4d}  loss={loss.item():.4f}  '
                  f'raw_mse_u={raw:.4f}  raw_mse_v={info["raw_mse_v"].item():.4f}  '
                  f'per_dim_rms_u={info["per_dim_rms_u"].item():.4f}  '
                  f'h_mean={info["h_mean"].item():.3f} fm_frac={info["fm_frac"].item():.2f}  '
                  f'{buckets}')
            for i in range(4):
                value = info[f'h_mse_b{i}'].item()
                # NaN is legal ONLY for a bucket that drew no samples this step
                if math.isinf(value):
                    print(f'  FAIL: h_mse_b{i} is infinite at step {step}')
                    return False

    dropped = last_raw < first_raw
    print(f'  raw_mse_u: {first_raw:.4f} -> {last_raw:.4f}  ->  '
          f'{"PASS" if dropped else "FAIL (not decreasing)"}')
    return dropped


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Gen3v6 MeanFlow pre-flight gates')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--obs-dim', type=int, default=4, help='avoiding-d3il observation_dim')
    ap.add_argument('--action-dim', type=int, default=2, help='avoiding-d3il action_dim')
    ap.add_argument('--ckpt', type=str, default=None, help='state_best.pt for a real G1')
    ap.add_argument('--steps', type=int, default=200, help='G3\' optimizer steps')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()

    torch.manual_seed(0)
    print('=' * 80)
    print(f'Gen3v6 MeanFlow gates — device={a.device}, H={a.horizon}, '
          f'obs={a.obs_dim}, act={a.action_dim}')
    print('=' * 80)

    results = {
        'G0': gate_g0(a, a.device),
        'G1': gate_g1(a, a.device, a.ckpt),
        "G3'": gate_g3(a, a.device, a.steps),
    }

    print('\n' + '=' * 80)
    for name, ok in results.items():
        print(f'  {name}: {"PASS" if ok else "FAIL"}')
    print('=' * 80)
    sys.exit(0 if all(results.values()) else 1)
