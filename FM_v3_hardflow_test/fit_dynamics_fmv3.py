"""Gen12 — refit HardFlow's linear dynamics model on FMv3's data pipeline.

PLAN §3.1, the trap this file exists to close
--------------------------------------------
`hardflow_new`'s NLP can enforce linear dynamics `s' = A s + B a + c`.  Upstream
reads those matrices from `HardFlow/logs/avoiding-v0/dynamics/linear_model.npz`,
which was fit on HardFlow's `SequenceDataset` with HardFlow's normalizer:

                     HardFlow          FMv3 (Gen12)
    env id           avoiding-v0       avoiding-d3il
    max_path_length  200               150
    normalizer       LimitsNormalizer  LimitsNormalizer
    horizon          8 / 16            8

A, B and c live in NORMALIZED units.  Two LimitsNormalizers fit on different
episode sets have different mins/maxs, so reusing HardFlow's .npz against
FMv3-normalized trajectories enforces the wrong physics — and the NLP still
converges, so nothing looks broken.  Hence: refit here, and write the
normalizer limits into the .npz so the sampler can *verify* the match at load
time instead of trusting the filename (see `load_linear_dynamics`).

Usage (cluster only — this container has no Python deps):
    python FM_v3_hardflow_test/fit_dynamics_fmv3.py
    python FM_v3_hardflow_test/fit_dynamics_fmv3.py --env avoiding-d3il --horizon 8

Gate (PLAN §4, step 3): the run prints held-out one-step R^2 / RMSE. The split
is by EPISODE, not by window — a window-level split at H=8 shares 7 of 8 frames
between neighbours and would report a train error dressed up as a test error.
"""

import argparse
import os

import numpy as np

from flow_matcher_v3_hardflow.datasets.sequence import SequenceDataset


DEFAULT_OUTPUT_ROOT = 'logs'


def output_path(env, horizon, max_path_length, root=DEFAULT_OUTPUT_ROOT):
    """Provenance-encoding path (PLAN §3.6): env + horizon + max_path_length."""
    return os.path.join(root, env, 'dynamics_gen12',
                        f'linear_model_H{horizon}_mpl{max_path_length}.npz')


def _least_squares(features, targets):
    """Closed-form multi-output least squares with an intercept column."""
    n = features.shape[0]
    design = np.hstack([features, np.ones((n, 1))])
    coeffs, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return coeffs[:-1].T, coeffs[-1]          # weights (out x in), intercept (out,)


def _metrics(y_true, y_pred):
    err = y_true - y_pred
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2_per_dim = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    return {
        'mse': float(np.mean(err ** 2)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'r2': float(1.0 - ss_res.sum() / ss_tot.sum()) if ss_tot.sum() > 0 else float('nan'),
        'r2_per_dim': r2_per_dim.tolist(),
        'rmse_per_dim': np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        'n': int(len(y_true)),
    }


def collect_transitions(dataset, episode_ids):
    """(s_t, a_t) -> s_{t+1} pairs in NORMALIZED units, for the given episodes."""
    fields = dataset.fields
    states, actions, next_states = [], [], []
    for i in episode_ids:
        length = fields.path_lengths[i]
        obs = fields.normed_observations[i, :length]
        act = fields.normed_actions[i, :length]
        if length < 2:
            continue
        states.append(obs[:-1])
        actions.append(act[:-1])
        next_states.append(obs[1:])
    if not states:
        raise RuntimeError('no usable episodes in split')
    return (np.concatenate(states), np.concatenate(actions), np.concatenate(next_states))


def fit(env='avoiding-d3il', horizon=8, max_path_length=150, max_n_episodes=100000,
        train_fraction=0.9, seed=0, output_root=DEFAULT_OUTPUT_ROOT, force=False):
    dest = output_path(env, horizon, max_path_length, output_root)
    if os.path.exists(dest) and not force:
        raise FileExistsError(
            f'{dest} already exists. Re-run with --force to overwrite '
            '(PLAN §3.6: never silently replace a finished artefact).')

    print(f'[ fit_dynamics_fmv3 ] loading {env} through FMv3\'s SequenceDataset')
    dataset = SequenceDataset(
        env=env,
        horizon=horizon,
        normalizer='LimitsNormalizer',
        preprocess_fns=[],
        max_path_length=max_path_length,
        max_n_episodes=max_n_episodes,
        termination_penalty=0,
        use_padding=True,
        include_returns=False,
    )

    n_episodes = dataset.fields.n_episodes
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_episodes)
    n_train = max(1, int(round(train_fraction * n_episodes)))
    train_ids, test_ids = order[:n_train], order[n_train:]
    print(f'[ fit_dynamics_fmv3 ] {n_episodes} episodes -> '
          f'{len(train_ids)} train / {len(test_ids)} held out (EPISODE-level split)')

    s_tr, a_tr, sn_tr = collect_transitions(dataset, train_ids)
    features = np.hstack([s_tr, a_tr])
    W, c = _least_squares(features, sn_tr)
    obs_dim = s_tr.shape[1]
    A, B = W[:, :obs_dim], W[:, obs_dim:]

    train_metrics = _metrics(sn_tr, s_tr @ A.T + a_tr @ B.T + c)
    if len(test_ids) > 0:
        s_te, a_te, sn_te = collect_transitions(dataset, test_ids)
        test_metrics = _metrics(sn_te, s_te @ A.T + a_te @ B.T + c)
    else:
        test_metrics = None

    eig = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eig)))

    print('\n=== Gen12 linear dynamics: s\' = A s + B a + c (normalized units) ===')
    print(f'  transitions   : {train_metrics["n"]} train'
          + (f', {test_metrics["n"]} held out' if test_metrics else ''))
    print(f'  obs dim / act dim : {obs_dim} / {a_tr.shape[1]}')
    print(f'  spectral radius(A): {spectral_radius:.4f}'
          + ('  (stable)' if spectral_radius < 1.0 else '  (>= 1.0 — UNSTABLE)'))
    print(f'  ||B||_F           : {np.linalg.norm(B):.4f}')
    print(f'  TRAIN  R2 = {train_metrics["r2"]:.6f}   RMSE = {train_metrics["rmse"]:.6f}')
    if test_metrics:
        print(f'  HELD-OUT R2 = {test_metrics["r2"]:.6f}   RMSE = {test_metrics["rmse"]:.6f}')
        print(f'  HELD-OUT per-dim R2 = '
              + ', '.join(f'{v:.4f}' for v in test_metrics['r2_per_dim']))
        # PLAN §6: this is the gate. A near-1 R2 means the linear surrogate is
        # a fair stand-in for the true dynamics; anything low means the NLP
        # would be enforcing physics the data does not support.
        verdict = 'PASS' if test_metrics['r2'] > 0.99 else 'INSPECT'
        print(f'  GATE (held-out R2 > 0.99): {verdict}')

    obs_norm = dataset.normalizer.normalizers['observations']
    act_norm = dataset.normalizer.normalizers['actions']

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(
        dest,
        A=A, B=B, c=c,
        env=env, horizon=horizon, max_path_length=max_path_length,
        normalizer='LimitsNormalizer',
        obs_mins=obs_norm.mins, obs_maxs=obs_norm.maxs,
        action_mins=act_norm.mins, action_maxs=act_norm.maxs,
        n_episodes=n_episodes, train_episodes=train_ids, test_episodes=test_ids,
        train_metrics=train_metrics, test_metrics=test_metrics,
        spectral_radius=spectral_radius,
    )
    print(f'\n[ fit_dynamics_fmv3 ] wrote {dest}')
    return dest


def load_linear_dynamics(path, normalizer, atol=1e-8):
    """Load A/B/c and REFUSE to return them if the normalizer does not match.

    This is the runtime half of PLAN §3.1. Passing dynamics fit under different
    limits produces a converging NLP that enforces the wrong physics, which is
    invisible in the results — so it is made a hard error here.
    """
    data = np.load(path, allow_pickle=True)
    checks = {
        'obs_mins': normalizer.normalizers['observations'].mins,
        'obs_maxs': normalizer.normalizers['observations'].maxs,
        'action_mins': normalizer.normalizers['actions'].mins,
        'action_maxs': normalizer.normalizers['actions'].maxs,
    }
    for key, current in checks.items():
        stored = np.asarray(data[key], dtype=float)
        if stored.shape != np.shape(current) or not np.allclose(stored, current, atol=atol):
            raise ValueError(
                f'Linear dynamics at {path} were fit under a DIFFERENT normalizer.\n'
                f'  {key}: stored={stored}  current={np.asarray(current)}\n'
                'A, B and c are in normalized units and are meaningless here. '
                'Re-run FM_v3_hardflow_test/fit_dynamics_fmv3.py (PLAN §3.1).')
    return {'A': data['A'], 'B': data['B'], 'c': data['c']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='avoiding-d3il')
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--max-path-length', type=int, default=150)
    parser.add_argument('--max-n-episodes', type=int, default=100000)
    parser.add_argument('--train-fraction', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    fit(env=args.env, horizon=args.horizon, max_path_length=args.max_path_length,
        max_n_episodes=args.max_n_episodes, train_fraction=args.train_fraction,
        seed=args.seed, output_root=args.output_root, force=args.force)


if __name__ == '__main__':
    main()
