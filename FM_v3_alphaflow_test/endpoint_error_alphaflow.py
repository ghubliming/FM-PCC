#!/usr/bin/env python3
"""Gen3v7 §8.2 — ENDPOINT ERROR AT THE SAMPLER'S OWN GRID.

⚠️ RUN ON THE CLUSTER. Needs torch + the trained checkpoint.

**This is the single most decisive number in the generation** (PLAN §8, diagnostic 2).

The whole MeanFlow-family rationale is that `x̂_t = z_r + h·u(z_r, r, h)` is an *exact*
endpoint map, versus FM's first-order Euler shot `z_r + h·v(z_r, r)` with O(h²) error. Gen13
fix_7.3 §4 measured that claim for iMF and found the error **flat in K at ~0.155** — the
signature of a fixed FIELD error, not a discretisation error: the u-field's own training
error dwarfed the Euler error it was meant to eliminate (FM: 0.026–0.038).

    🔴 IF α-FLOW'S ERROR IS FLAT IN K TOO, THE OBJECTIVE CHANGE DID NOT HELP,
       and the honest reading is fix_7.3's: the blind direction was never the binding cause.

Two complementary measurements, both reported per K ∈ {1, 2, 5, 10}:

  T-A  INTERVAL endpoint error, on the data coupling. For each query (τ, h) the sampler
       would actually make at that K, draw x₀ ~ N(0,I), form z_τ = τ·x₁ + (1−τ)·x₀ and
       compare the model's one-shot jump against the TRUE endpoint of that coupling:

           err_u = ‖(z_τ + h·u(z_τ, τ, h)) − z_{τ+h}‖      (the average-velocity map)
           err_v = ‖(z_τ + h·v(z_τ, τ, h)) − z_{τ+h}‖      (the Euler shot, same model)

       ⚠️ The floor is NOT zero: the best possible field predicts E[z_{τ+h} | z_τ], not the
       particular paired sample. That floor is a property of the DATA and is identical for
       Gen3v4 / Gen3v6 / Gen3v7 / FM, so these numbers are comparable ACROSS models — which
       is the point — but must never be read as an absolute accuracy.
       The `err_v` column is the intra-model control: u beating v is the entire claim.
       Grid {(τ=0,h=1)} is K=1; {(0,0.5),(0.5,0.5)} is K=2; etc.

  T-B  TERMINAL prediction error, self-consistency — the exact metric of fix_7.3 §4, so the
       numbers are directly comparable to iMF's 0.1539/0.1538/0.1595/0.1572. Roll the REAL
       sampler out for K steps from the same x₀, keep x_final, and at each step k form
       x̂₁(k) = x_k + (1−τ_k)·u(x_k, τ_k, 1−τ_k), reporting ‖x̂₁(k) − x_final‖.
       ⚠️ err at τ=1 is 0 by construction for every method — only err(τ=0) and the shape of
       the curve carry information (fix_7.3 §4 metric caveat).

Both are computed in NORMALISED trajectory space (the space the model is trained in), over
held-out-by-index windows of the avoiding-d3il dataset. The window-level split leak
(POST_U10_III §4.2) applies to any "val" reading here — see --split-note.

Usage (cluster):
    python FM_v3_alphaflow_test/endpoint_error_alphaflow.py --seed 6
    python FM_v3_alphaflow_test/endpoint_error_alphaflow.py --seed 6 \
        --loadpath logs/avoiding-d3il/flow_matching_v3_meanflow/<exp>/6 --label Gen3v6
"""

import argparse
import json
import os

import numpy as np
import torch

import flow_matcher_v3_alphaflow.utils as utils

K_GRID = (1, 2, 5, 10)


class Parser(utils.Parser):
    dataset: str = 'avoiding-d3il'
    config: str = 'config.avoiding-d3il'


def resolve_loadpath(cli):
    """Explicit --loadpath wins; otherwise ask the plan config where the checkpoint is."""
    if cli.loadpath:
        return cli.loadpath, 'cli --loadpath'
    args = Parser().parse_args(experiment='plan_fm_v3_alphaflow', seed=cli.seed)
    loadpath = os.path.join(args.loadbase or args.logbase, args.dataset,
                            args.diffusion_loadpath, str(cli.seed))
    return loadpath, 'plan_fm_v3_alphaflow'


def get_batch(dataset, n, device, seed=0):
    """A batch of normalised trajectory windows + their conditioning."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    xs = np.stack([dataset[int(i)].trajectories for i in idx])
    x1 = torch.as_tensor(xs, dtype=torch.float32, device=device)
    cond = {0: x1[:, 0, dataset.action_dim:].clone()}
    return x1, cond


@torch.no_grad()
def interval_endpoint_error(model, x1, cond, K):
    """T-A. Mean L2 (per window, summed over horizon dims) at every (τ, h) the K-step
    sampler queries. Returns (err_u, err_v) averaged over the grid, plus the per-τ lists."""
    from flow_matcher_v3_alphaflow.models.helpers import apply_conditioning
    B = x1.shape[0]
    ad, gd = model.action_dim, model.goal_dim
    h_val = 1.0 / K

    x0 = apply_conditioning(torch.randn_like(x1), cond, ad, goal_dim=gd, noise=True)
    per_tau_u, per_tau_v, taus = [], [], []
    for i in range(K):
        tau_val = i / K
        tau = torch.full((B,), tau_val, device=x1.device)
        h = torch.full((B,), h_val, device=x1.device)

        z_tau = apply_conditioning(model.q_sample(x_start=x1, t=tau, noise=x0),
                                   cond, ad, goal_dim=gd)
        z_next = apply_conditioning(
            model.q_sample(x_start=x1, t=tau + h, noise=x0), cond, ad, goal_dim=gd)

        u, v = model._predict_uv(z_tau, cond, tau, h=h)
        pred_u = apply_conditioning(z_tau + h_val * u, cond, ad, goal_dim=gd)
        pred_v = apply_conditioning(z_tau + h_val * v, cond, ad, goal_dim=gd)

        per_tau_u.append(float((pred_u - z_next).pow(2).sum(dim=(1, 2)).sqrt().mean()))
        per_tau_v.append(float((pred_v - z_next).pow(2).sum(dim=(1, 2)).sqrt().mean()))
        taus.append(tau_val)
    return float(np.mean(per_tau_u)), float(np.mean(per_tau_v)), taus, per_tau_u, per_tau_v


@torch.no_grad()
def terminal_prediction_error(model, x1, cond, K):
    """T-B. fix_7.3 §4's metric: ‖x̂₁(k) − x_final‖ along the model's OWN K-step rollout.

    Reproduces the sampler's update exactly (x += dt·u with h = dt = 1/K, conditioning
    re-applied every step) rather than calling p_sample_loop, because we need the
    intermediate x_k — and a projector must NOT be involved: this measures the raw field.
    """
    from flow_matcher_v3_alphaflow.models.helpers import apply_conditioning
    B = x1.shape[0]
    ad, gd = model.action_dim, model.goal_dim
    dt = 1.0 / K

    x = apply_conditioning(torch.randn_like(x1), cond, ad, goal_dim=gd)
    chain, x1_hat, taus = [x.clone()], [], []
    for i in range(K):
        tau_val = i / K
        tau = torch.full((B,), tau_val, device=x1.device)
        # the endpoint map at THIS point: jump the whole remaining interval in one query
        h_rem = torch.full((B,), 1.0 - tau_val, device=x1.device)
        u_rem, _v = model._predict_uv(x, cond, tau, h=h_rem)
        x1_hat.append(apply_conditioning(x + (1.0 - tau_val) * u_rem, cond, ad, goal_dim=gd))
        taus.append(tau_val)

        # the sampler's actual step
        h = torch.full((B,), dt, device=x1.device)
        u, _ = model._predict_uv(x, cond, tau, h=h)
        x = apply_conditioning(x + dt * u, cond, ad, goal_dim=gd)
        chain.append(x.clone())

    x_final = chain[-1]
    errs = [float((xh - x_final).pow(2).sum(dim=(1, 2)).sqrt().mean()) for xh in x1_hat]
    return taus, errs


def main():
    ap = argparse.ArgumentParser(description='Gen3v7 endpoint-error diagnostic (PLAN §8.2)')
    ap.add_argument('--seed', type=int, default=6)
    ap.add_argument('--loadpath', type=str, default=None,
                    help='explicit checkpoint dir; also lets you point at Gen3v4/Gen3v6 '
                         'for the cross-generation table')
    ap.add_argument('--label', type=str, default=None, help='row label in the output')
    ap.add_argument('--epoch', type=str, default='best')
    ap.add_argument('--n', type=int, default=256, help='windows to average over')
    ap.add_argument('--use-ema', action='store_true', default=True)
    ap.add_argument('--no-ema', dest='use_ema', action='store_false')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', type=str, default=None, help='write JSON here')
    cli = ap.parse_args()

    loadpath, source = resolve_loadpath(cli)
    label = cli.label or os.path.basename(os.path.dirname(loadpath.rstrip('/')))
    print('=' * 88)
    print(f'Endpoint error at the sampler grid — {label}')
    print(f'  checkpoint : {loadpath}  (epoch={cli.epoch}, via {source})')
    print(f'  weights    : {"EMA" if cli.use_ema else "raw/live"}')
    print('=' * 88)

    exp = utils.load_diffusion(loadpath, epoch=cli.epoch, device=cli.device)
    model = (exp.ema if cli.use_ema else exp.diffusion).to(cli.device)
    model.eval()

    x1, cond = get_batch(exp.dataset, cli.n, cli.device, seed=cli.seed)
    print(f'  batch      : {tuple(x1.shape)} normalised windows\n')

    report = {'label': label, 'loadpath': loadpath, 'epoch': str(cli.epoch),
              'use_ema': cli.use_ema, 'n_windows': int(x1.shape[0]), 'K': {}}

    print('T-A  INTERVAL endpoint error on the data coupling  (lower is better;')
    print('     err_v is the SAME model\'s Euler shot — u must beat it, that is the claim)')
    print(f'  {"K":>4} {"err_u":>10} {"err_v":>10} {"u/v":>8}   per-τ err_u')
    for K in K_GRID:
        eu, ev, taus, pu, _pv = interval_endpoint_error(model, x1, cond, K)
        ratio = eu / max(ev, 1e-12)
        per = ' '.join(f'{t:.2f}:{e:.4f}' for t, e in zip(taus, pu))
        print(f'  {K:>4d} {eu:>10.4f} {ev:>10.4f} {ratio:>8.2f}   {per}')
        report['K'][str(K)] = {'interval_err_u': eu, 'interval_err_v': ev,
                               'interval_ratio_u_over_v': ratio,
                               'interval_taus': taus, 'interval_err_u_per_tau': pu}

    print('\nT-B  TERMINAL prediction error ‖x̂₁(k) − x_final‖  (fix_7.3 §4 metric)')
    print('     ⚠️ err(τ=1) == 0 for every method by construction — read err(τ=0) only.')
    print(f'  {"K":>4} {"err(τ=0)":>10}   full curve')
    tau0 = {}
    for K in K_GRID:
        taus, errs = terminal_prediction_error(model, x1, cond, K)
        curve = ' '.join(f'{t:.2f}:{e:.4f}' for t, e in zip(taus, errs))
        print(f'  {K:>4d} {errs[0]:>10.4f}   {curve}')
        tau0[K] = errs[0]
        report['K'][str(K)].update({'terminal_taus': taus, 'terminal_errs': errs,
                                    'terminal_err_tau0': errs[0]})

    values = list(tau0.values())
    spread = (max(values) - min(values)) / max(np.mean(values), 1e-12)
    report['terminal_err_tau0_relative_spread'] = float(spread)
    print('\n' + '─' * 88)
    print(f'READ (PLAN §8.2): err(τ=0) across K = '
          f'{", ".join(f"{k}:{v:.4f}" for k, v in tau0.items())}   '
          f'relative spread = {spread:.1%}')
    if spread < 0.15:
        print('  → FLAT IN K. That is the signature of a fixed FIELD error, not a')
        print('    discretisation error — the same verdict fix_7.3 reached for iMF')
        print('    (0.1539/0.1538/0.1595/0.1572). The objective change did NOT fix the field.')
    else:
        print('  → NOT flat in K: the error shrinks with the budget, i.e. discretisation is')
        print('    still the binding term. Compare the magnitude against FM (~0.026–0.038')
        print('    at τ=0 in fix_7.3) before claiming a win.')
    print('  ⚠️ Windows come from the full dataset: the inherited train/test split is')
    print('     WINDOW-level (POST_U10_III §4.2), so nothing here is a generalisation claim.')
    print('─' * 88)

    out = cli.out or os.path.join(loadpath, 'endpoint_error.json')
    try:
        with open(out, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nwrote {out}')
    except OSError as exc:
        print(f'\ncould not write {out}: {exc}')


if __name__ == '__main__':
    main()
