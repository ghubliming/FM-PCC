"""Epoch 5 WS-C — Mini-FM sanity gate.

Train a tiny Flow Matching model on ≤100 empty-scene episodes from Epoch 4 to
verify that the dataset schema, action convention (Δp_des), and horizon config
(H=8, D=9) are all correct before investing in full-scale training.

This is a standalone script that does NOT depend on the full FM-PCC training
pipeline. It implements a minimal training loop with a small 1D temporal UNet
to validate data flow, not to achieve SoTA performance.

Pass criteria (from EPOCH5_PLAN.md §4.3)  — U2: D updated 9→12
-----------------------------------------
- Tensor shape through dataloader: (B, H=8, D=12) confirmed
- RMS position error (held-out): < 0.1 m
- Action delta norm (predicted vs GT): within 2× of GT mean

Usage
-----
python uav_expert_data_collect/mini_fm_sanity.py
python uav_expert_data_collect/mini_fm_sanity.py --data-dir logs/uav_expert_data/empty
python uav_expert_data_collect/mini_fm_sanity.py --n-episodes 50 --n-steps 500

See: logs_in_develop/Gen11/Epoch5_visual_and_validation/EPOCH5_PLAN.md §4
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

_DEFAULT_DATA_DIR = os.path.join(_REPO, 'logs', 'uav_expert_data', 'empty')

# ── FM constants ──────────────────────────────────────────────────────────────
HORIZON = 8       # prediction horizon (number of future steps)
OBS_DIM = 9       # [p_des(3), p(3), v(3)]  U2: was 6 ([p(3), v(3)])
ACTION_DIM = 3    # [Δp_des(3)]
DATA_DIM = 12     # ACTION_DIM + OBS_DIM = 3 + 9  U2: was 9 (3+6)
T_FLOW = 20       # number of flow ODE steps (kept small for speed)


def parse_args():
    p = argparse.ArgumentParser(
        description='Epoch 5 WS-C: mini-FM sanity gate on UAV expert data.')
    p.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                   help='Directory containing episode pickles (default: empty scene)')
    p.add_argument('--n-episodes', type=int, default=100,
                   help='Max episodes to load. Default: 100')
    p.add_argument('--n-steps', type=int, default=1000,
                   help='Training steps. Default: 1000')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Batch size. Default: 32')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate. Default: 1e-3')
    p.add_argument('--train-ratio', type=float, default=0.8,
                   help='Train/eval split ratio. Default: 0.8')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed. Default: 42')
    p.add_argument('--output-json', default=None,
                   help='Save results JSON here (default: data_dir/mini_fm_results.json)')
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────

def load_episodes(data_dir, max_episodes=100):
    """Load episode pickles from data_dir (walks subdirs)."""
    episodes = []
    for root, _, files in os.walk(data_dir):
        for fn in sorted(files):
            if fn.endswith('.pkl') and not fn.startswith('run_summary'):
                try:
                    with open(os.path.join(root, fn), 'rb') as f:
                        ep = pickle.load(f)
                    episodes.append(ep)
                except Exception:
                    pass
            if len(episodes) >= max_episodes:
                return episodes
    return episodes


def episodes_to_chunks(episodes, horizon=HORIZON):
    """Convert episodes to (N, H, D) training chunks.

    Each chunk is [actions(3) ‖ obs(9)] for H consecutive timesteps.
    This matches the FM-PCC dataloader format: (B, H=8, D=12).  U2: was D=9.
    """
    chunks = []
    for ep in episodes:
        obs = ep['obs']          # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
        actions = ep['actions']  # (T-1, 3)
        T_act = len(actions)

        for t in range(T_act - horizon + 1):
            act_chunk = actions[t:t + horizon]     # (H, 3)
            obs_chunk = obs[t:t + horizon]          # (H, 9)
            chunk = np.concatenate([act_chunk, obs_chunk], axis=1)  # (H, 12)
            chunks.append(chunk)

    return np.array(chunks, dtype=np.float32)  # (N, H, D)


# ── minimal flow matching model (numpy-only, no PyTorch dependency) ───────────

class TinyFlowModel:
    """A minimal MLP-based flow matching model for sanity checking.

    This is intentionally simple: a 2-layer MLP that learns the velocity
    field v(x_t, t) for the flow ODE from noise to data. Not meant to be
    performant — just correct enough to test data pipeline integrity.
    """

    def __init__(self, data_dim, hidden=128, seed=42):
        rng = np.random.default_rng(seed)
        self.d = data_dim
        self.h = hidden
        # Layer 1: (D+1) -> H   (D features + 1 time input)
        self.W1 = rng.normal(0, 0.02, (data_dim + 1, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        # Layer 2: H -> D
        self.W2 = rng.normal(0, 0.02, (hidden, data_dim)).astype(np.float32)
        self.b2 = np.zeros(data_dim, dtype=np.float32)

    def forward(self, x_t, t_scalar):
        """Predict velocity v(x_t, t). x_t: (B, D), t_scalar: float."""
        B = x_t.shape[0]
        t_col = np.full((B, 1), t_scalar, dtype=np.float32)
        inp = np.concatenate([x_t, t_col], axis=1)  # (B, D+1)
        h = inp @ self.W1 + self.b1                   # (B, H)
        h = np.maximum(h, 0)                           # ReLU
        out = h @ self.W2 + self.b2                    # (B, D)
        return out

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]


def fm_loss(model, x1, x0, t):
    """Conditional flow matching loss: ||v_θ(x_t, t) - (x1 - x0)||²

    x1: (B, D) data samples
    x0: (B, D) noise samples ~ N(0, I) — passed in, NOT redrawn here, so that
        finite-difference gradient evals (+eps/-eps) see the same noise sample.
    t:  (B,) uniform in [0, 1] — same reasoning.
    x_t = (1-t)*x0 + t*x1  (linear interpolation)
    target = x1 - x0
    """
    t_col = t[:, None]  # (B, 1)
    x_t = (1 - t_col) * x0 + t_col * x1  # (B, D)
    target = x1 - x0                       # (B, D)

    # We use a single t for the whole batch for simplicity in the MLP
    t_mean = float(t.mean())
    pred = model.forward(x_t, t_mean)

    loss = np.mean((pred - target) ** 2)
    return loss, pred, target, x_t, t_mean


def train_step_numerical(model, x1_batch, rng, lr=1e-3, eps=1e-5):
    """One training step using numerical gradient (no autograd needed).

    This is intentionally slow but dependency-free. For a real sanity gate,
    use PyTorch — this is just to verify data flow without any framework.
    """
    params = model.parameters()
    B, D = x1_batch.shape

    # Freeze the random draw for this whole step: every loss eval below
    # (base, +eps, -eps for every parameter) must see the SAME (x0, t) or the
    # finite-difference gradient is dominated by noise between draws, not the
    # true gradient — that noise gets amplified by 1/(2*eps) and explodes.
    x0 = rng.normal(0, 1, x1_batch.shape).astype(np.float32)
    t = rng.uniform(0, 1, (B,)).astype(np.float32)

    # Forward: compute loss
    loss_val, _, _, _, _ = fm_loss(model, x1_batch, x0, t)

    # Numerical gradient for each parameter (very slow but correct)
    for p in params:
        grad = np.zeros_like(p)
        flat = p.ravel()
        for i in range(min(len(flat), 200)):  # limit gradient computation
            old = flat[i]
            flat[i] = old + eps
            loss_plus, _, _, _, _ = fm_loss(model, x1_batch, x0, t)
            flat[i] = old - eps
            loss_minus, _, _, _, _ = fm_loss(model, x1_batch, x0, t)
            flat[i] = old
            grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
        p -= lr * grad

    return loss_val


def sample_ode(model, shape, n_steps=T_FLOW, seed=0):
    """Euler ODE integration from noise to data.

    x_0 ~ N(0, I)
    dx/dt = v_θ(x_t, t)
    Integrate t: 0 → 1
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, shape).astype(np.float32)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        v = model.forward(x, t)
        x = x + v * dt
    return x


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, eval_chunks, data_mean, data_std, n_eval=20, seed=99):
    """Sample from model and compare against ground-truth chunks.

    The model is trained in normalised (z-scored) space, so its samples come
    out normalised. De-normalise them back to metres before comparing against
    the unnormalised ground-truth eval_chunks, otherwise the RMS is computed
    across mismatched scales and always fails.

    Returns dict with RMS error and action norm comparison.
    """
    n = min(n_eval, len(eval_chunks))
    gt = eval_chunks[:n]  # (n, H, D) — unnormalised (metres)

    # Flatten to (n*H, D) for sampling
    gt_flat = gt.reshape(-1, DATA_DIM)  # (n*H, D)

    # Sample same number of points (normalised space) then de-normalise to metres
    pred_norm = sample_ode(model, gt_flat.shape, seed=seed)
    pred_flat = pred_norm * data_std + data_mean

    # Position error: first 3 dims are actions (Δp_des), next 3 are position
    gt_pos = gt_flat[:, ACTION_DIM:ACTION_DIM + 3]    # p(3)
    pred_pos = pred_flat[:, ACTION_DIM:ACTION_DIM + 3]
    rms_pos = float(np.sqrt(np.mean((gt_pos - pred_pos) ** 2)))

    # Action norm comparison
    gt_act_norm = float(np.mean(np.linalg.norm(gt_flat[:, :ACTION_DIM], axis=1)))
    pred_act_norm = float(np.mean(np.linalg.norm(pred_flat[:, :ACTION_DIM], axis=1)))

    return {
        'rms_position_error_m': round(rms_pos, 5),
        'gt_action_norm_mean': round(gt_act_norm, 5),
        'pred_action_norm_mean': round(pred_act_norm, 5),
        'action_norm_ratio': round(pred_act_norm / max(gt_act_norm, 1e-8), 3),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f'[ mini-fm ] Loading episodes from {args.data_dir} …')
    episodes = load_episodes(args.data_dir, max_episodes=args.n_episodes)
    if not episodes:
        print('[ mini-fm ] ERROR: no episodes found.')
        sys.exit(1)
    print(f'[ mini-fm ] Loaded {len(episodes)} episodes')

    # ── Build chunks ──────────────────────────────────────────────────────────
    all_chunks = episodes_to_chunks(episodes, horizon=HORIZON)
    N = len(all_chunks)
    print(f'[ mini-fm ] {N} chunks of shape (H={HORIZON}, D={DATA_DIM})')
    print(f'[ mini-fm ] Chunk tensor shape: {all_chunks.shape}')

    # ── Shape gate ────────────────────────────────────────────────────────────
    expected_shape = (HORIZON, DATA_DIM)
    actual_shape = all_chunks.shape[1:]
    shape_ok = actual_shape == expected_shape
    print(f'[ mini-fm ] Shape check: expected {expected_shape}, '
          f'got {actual_shape} → {"✅ PASS" if shape_ok else "❌ FAIL"}')

    if not shape_ok:
        print('[ mini-fm ] CRITICAL: Shape mismatch. Data pipeline is broken.')
        print(f'           Expected (H={HORIZON}, D={DATA_DIM}) = '
              f'(H, actions(3) + obs(6))')
        sys.exit(1)

    # ── Train/eval split ──────────────────────────────────────────────────────
    n_train = int(N * args.train_ratio)
    train_chunks = all_chunks[:n_train]
    eval_chunks = all_chunks[n_train:]
    print(f'[ mini-fm ] Train: {len(train_chunks)}, Eval: {len(eval_chunks)}')

    # ── Data stats ────────────────────────────────────────────────────────────
    act_norms = np.linalg.norm(
        all_chunks[:, :, :ACTION_DIM].reshape(-1, ACTION_DIM), axis=1)
    print(f'[ mini-fm ] Action Δp_des norm: '
          f'mean={act_norms.mean():.5f}, p95={np.percentile(act_norms, 95):.5f}')

    # ── Normalise data ────────────────────────────────────────────────────────
    flat_train = train_chunks.reshape(-1, DATA_DIM)
    data_mean = flat_train.mean(axis=0)
    data_std = flat_train.std(axis=0) + 1e-8
    train_norm = (train_chunks - data_mean) / data_std
    eval_norm = (eval_chunks - data_mean) / data_std

    # ── Train ─────────────────────────────────────────────────────────────────
    model = TinyFlowModel(data_dim=DATA_DIM, hidden=128, seed=args.seed)
    flat_train_norm = train_norm.reshape(-1, DATA_DIM)

    print(f'[ mini-fm ] Training {args.n_steps} steps '
          f'(B={args.batch_size}, lr={args.lr}) …')

    losses = []
    for step in range(args.n_steps):
        idx = rng.choice(len(flat_train_norm), args.batch_size, replace=True)
        batch = flat_train_norm[idx]

        loss = train_step_numerical(model, batch, rng, lr=args.lr)
        losses.append(loss)

        if (step + 1) % 100 == 0 or step == 0:
            avg = np.mean(losses[-100:])
            print(f'  step {step+1:5d}/{args.n_steps}  loss={avg:.5f}')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print('[ mini-fm ] Evaluating on held-out chunks …')
    flat_eval_norm = eval_norm.reshape(-1, DATA_DIM)
    metrics = evaluate(model, eval_chunks, data_mean, data_std,
                       n_eval=50, seed=args.seed + 1)

    elapsed = time.time() - t0

    # ── Verdict ───────────────────────────────────────────────────────────────
    rms = metrics['rms_position_error_m']
    act_ratio = metrics['action_norm_ratio']
    rms_pass = rms < 0.1
    act_pass = 0.5 <= act_ratio <= 2.0

    print()
    print('=' * 60)
    print('  MINI-FM SANITY GATE RESULTS')
    print('=' * 60)
    print(f'  Episodes loaded:       {len(episodes)}')
    print(f'  Training chunks:       {len(train_chunks)}')
    print(f'  Eval chunks:           {len(eval_chunks)}')
    print(f'  Tensor shape:          {all_chunks.shape} '
          f'→ (N, H={HORIZON}, D={DATA_DIM})')
    print(f'  Shape check:           {"✅ PASS" if shape_ok else "❌ FAIL"}')
    print()
    print(f'  RMS position error:    {rms:.5f} m  '
          f'(threshold < 0.1 m) → {"✅ PASS" if rms_pass else "❌ FAIL"}')
    print(f'  GT action norm mean:   {metrics["gt_action_norm_mean"]:.5f} m/step')
    print(f'  Pred action norm mean: {metrics["pred_action_norm_mean"]:.5f} m/step')
    print(f'  Action norm ratio:     {act_ratio:.3f}  '
          f'(threshold 0.5–2.0) → {"✅ PASS" if act_pass else "❌ FAIL"}')
    print()

    all_pass = shape_ok and rms_pass and act_pass
    verdict = '✅ GO — proceed to Epoch 6' if all_pass else '❌ NO-GO — fix data pipeline'
    print(f'  OVERALL VERDICT:       {verdict}')
    print(f'  Elapsed:               {elapsed:.1f}s')
    print('=' * 60)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'verdict': 'GO' if all_pass else 'NO-GO',
        'n_episodes': len(episodes),
        'n_chunks_train': len(train_chunks),
        'n_chunks_eval': len(eval_chunks),
        'chunk_shape': list(all_chunks.shape),
        'shape_ok': shape_ok,
        'rms_position_error_m': rms,
        'rms_pass': rms_pass,
        'action_norm_ratio': act_ratio,
        'action_norm_pass': act_pass,
        'n_steps': args.n_steps,
        'final_loss': float(np.mean(losses[-50:])),
        'elapsed_s': round(elapsed, 1),
    }

    out_path = args.output_json or os.path.join(args.data_dir,
                                                 'mini_fm_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[ mini-fm ] Results saved → {out_path}')


if __name__ == '__main__':
    main()
