"""
Gen14 U9 — P2: is the 128-D visual latent informative?

    python mix_visual_aligning_test/probe_latent_informativeness.py \
        --checkpoint logs/.../state_100000.pt

WHY THIS EXISTS
───────────────
Every Gen14 visual-aligning result so far lands in the same 0.29-0.47 m band against a
0.4547 m do-nothing baseline, whatever the bone, engine or projector (ARCH §13.3a). One
explanation is that 22.36 M of the 26.4 M trainable parameters are a dual ResNet-18 fitted
from scratch to 900 episodes, and the 128-D latent it produces carries little the model can
use (ARCH §13.1). That is a claim about a tensor, so it should be MEASURED, not argued.

Gate G-B3 already proves gradient reaches `vis_projector`. It does NOT prove the latent is
INFORMATIVE — a latent can receive gradient and still encode nothing task-relevant.

WHAT IS MEASURED
────────────────
Ridge-regress three feature sets onto the behaviour-cloning target (the H x 3 action chunk),
on a held-out split:

    state           cond[0], the 6-D [des_c_pos | c_pos] anchor the model gets for free
    latent          the 128-D dual-camera encoding
    state + latent  both

The number that matters is the LAST COLUMN: the incremental R^2 of the latent OVER the state.
The image is only worth its 22.36 M parameters if it explains variance the state does not.

    incremental R^2 ~ 0      -> the encoder adds nothing; ARCH §13.1 confirmed; the fix is
                               perception (U9 C1/C2), or input resolution if pretraining does
                               not move it (the trunk emits 512x3x3 at 96x96 — 32 keypoints
                               over NINE spatial positions, base_nets.py:535-537)
    incremental R^2 clearly >0 -> perception is doing real work; the bottleneck is downstream
                               (horizon, single-frame conditioning, the 400-step cap)

🔴 SCOPE, STATED HONESTLY. ParityAligningDataset yields only (trajectory, cond[0], images) —
it carries NO box or target pose (sequence.py:42-47), so this probe cannot regress onto box
pose directly. Predicting the action chunk is the closest available proxy and is arguably the
more relevant target anyway: it is what the policy is actually asked to produce. A low score
here does not prove the image lacks box information in principle; it proves the trained
encoder does not expose it linearly for the thing the policy must do.

Read-only. Loads a checkpoint, touches no training state, writes nothing.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'd3il'))


def _build_encoder(device, pretrained=False):
    """The SAME MultiImageObsEncoder the wrappers build (visual_*_twotime.py).

    Built standalone rather than through a wrapper so the probe works on a checkpoint from
    ANY bone — obs_encoder.* keys are identical across unet / mf_dit / sit / dit.
    """
    import hydra
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        '_target_': 'agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder',
        'shape_meta': {'obs': {
            'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'},
            'in_hand_image':   {'shape': [3, 96, 96], 'type': 'rgb'},
        }},
        'rgb_model': {
            '_target_': 'agents.models.vision.model_getter.get_resnet',
            'input_shape': [3, 96, 96],
            'output_size': 64,
            'pretrained': bool(pretrained),
        },
        'resize_shape': None, 'random_crop': False, 'use_group_norm': True,
        'share_rgb_model': False, 'imagenet_norm': True,
    })
    return hydra.utils.instantiate(cfg).to(device)


def _load_encoder_weights(encoder, ckpt_path, device):
    """Pull obs_encoder.* out of a training checkpoint, wherever the wrapper nested it."""
    import torch
    blob = torch.load(ckpt_path, map_location=device)
    sd = blob.get('model', blob.get('ema', blob)) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict):
        raise ValueError(f'{ckpt_path}: no state_dict found in the checkpoint.')
    hits = {}
    for k, v in sd.items():
        i = k.find('obs_encoder.')
        if i >= 0:
            hits[k[i + len('obs_encoder.'):]] = v
    if not hits:
        raise ValueError(
            f'{ckpt_path}: no obs_encoder.* keys. Is this a VISUAL run (if_vision=True)?')
    missing, unexpected = encoder.load_state_dict(hits, strict=False)
    print(f'  loaded {len(hits)} encoder tensors  '
          f'(missing {len(missing)}, unexpected {len(unexpected)})')
    if missing:
        raise ValueError(
            f'{len(missing)} encoder tensors were NOT in the checkpoint, e.g. {missing[:3]}. '
            'Refusing to probe a partially-random encoder — the number would be meaningless.')
    return encoder


def _ridge_r2(X_tr, Y_tr, X_te, Y_te, lam=1e-3):
    """Closed-form ridge + R^2 on the held-out half. torch only, no sklearn/numpy needed."""
    import torch
    mx, my = X_tr.mean(0, keepdim=True), Y_tr.mean(0, keepdim=True)
    sx = X_tr.std(0, keepdim=True).clamp_min(1e-6)
    Xt, Xe = (X_tr - mx) / sx, (X_te - mx) / sx
    Yt = Y_tr - my
    A = Xt.T @ Xt + lam * Xt.shape[0] * torch.eye(Xt.shape[1], device=Xt.device, dtype=Xt.dtype)
    W = torch.linalg.solve(A, Xt.T @ Yt)
    pred = Xe @ W + my
    ss_res = ((Y_te - pred) ** 2).sum()
    ss_tot = ((Y_te - Y_te.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='state_<step>.pt from a visual run')
    ap.add_argument('--data', default='environments/dataset/data/aligning/train_files.pkl')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--n-samples', type=int, default=4096)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--lam', type=float, default=1e-3)
    a = ap.parse_args()

    import torch
    from mix_visual_aligning.datasets.sequence import ParityAligningDataset

    print('=' * 78)
    print('Gen14 U9 P2 — latent informativeness probe')
    print(f'  checkpoint : {a.checkpoint}')
    print(f'  data       : {a.data}')
    print('=' * 78)

    enc = _build_encoder(a.device)
    _load_encoder_weights(enc, a.checkpoint, a.device)
    enc.eval()

    ds = ParityAligningDataset(dataset_path=a.data, horizon=a.horizon)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=2)

    S, L, Y = [], [], []
    seen = 0
    with torch.no_grad():
        for batch in dl:
            traj, cond = batch[0], batch[1]
            state = cond[0].to(a.device).float()
            imgs = {'agentview_image': cond['primary_img'].to(a.device).float(),
                    'in_hand_image':   cond['wrist_img'].to(a.device).float()}
            lat = enc(imgs)
            if lat.ndim == 3:                       # (B, T_win, C) -> pool, as the wrappers do
                lat = lat.mean(dim=1)
            y = traj.to(a.device).float()[:, :, :3].reshape(traj.shape[0], -1)   # action chunk
            S.append(state); L.append(lat); Y.append(y)
            seen += state.shape[0]
            if seen >= a.n_samples:
                break

    S, L, Y = torch.cat(S), torch.cat(L), torch.cat(Y)
    n = S.shape[0]; cut = n // 2
    print(f'\n  n = {n} samples   state {tuple(S.shape[1:])}   '
          f'latent {tuple(L.shape[1:])}   target {tuple(Y.shape[1:])} (H x 3 actions)')

    def r2(X):
        return _ridge_r2(X[:cut], Y[:cut], X[cut:], Y[cut:], lam=a.lam)

    r_state = r2(S)
    r_lat = r2(L)
    r_both = r2(torch.cat([S, L], dim=1))
    incr = r_both - r_state

    print('\n  held-out R^2 predicting the action chunk')
    print('  ' + '-' * 58)
    print(f'    state only  (6-D, free to the model)   : {r_state: .4f}')
    print(f'    latent only (128-D, costs 22.36 M)     : {r_lat: .4f}')
    print(f'    state + latent                         : {r_both: .4f}')
    print('  ' + '-' * 58)
    print(f'    INCREMENTAL R^2 of the latent over state: {incr: .4f}')
    print('  ' + '-' * 58)

    # A threshold, stated once, so the reading does not drift between runs.
    if incr < 0.01:
        print('\n  READING: the latent adds essentially NOTHING the state does not already give.')
        print('  ARCH §13.1 is supported: 85 % of the trainable model is buying ~no signal.')
        print('  Next: U9 C1/C2 (ImageNet init + reduced encoder LR). If those do not move')
        print('  this number, suspect INPUT RESOLUTION, not the weights — the trunk emits')
        print('  512x3x3 at 96x96, i.e. 32 keypoints over nine spatial positions.')
    elif incr < 0.05:
        print('\n  READING: the latent adds a little. Perception is weak but not inert.')
        print('  U9 C1/C2 are still the right next move; expect a modest, not decisive, gain.')
    else:
        print('\n  READING: the latent carries real, state-independent signal.')
        print('  Perception is NOT the primary bottleneck. Look downstream instead:')
        print('  horizon (ARCH §5), single-frame conditioning (§13.3c), or the 400-step cap.')
    print()


if __name__ == '__main__':
    main()
