"""Gen13 iMF convention module — THE single place that owns time-convention logic.

Everything iMF in this package is expressed NATIVELY in HardFlow's convention, so
no sign flip exists anywhere at sampling time. This file documents the mapping to
the official iMeanFlow (aux repo) convention and derives the training identity
used by `imf_matcher.py`. If a sign bug is ever suspected, audit THIS file only.

HardFlow convention (used throughout this package)
--------------------------------------------------
    z_tau = tau * x1 + (1 - tau) * x0      tau=0 noise  ->  tau=1 data
    v(z, tau)      = instantaneous velocity, conditional target v = x1 - x0
    u(z, tau, h)   = average velocity over [tau, tau + h]:
                         z_{tau+h} = z_tau + h * u(z_tau, tau, h)     (exact jump)
    endpoint (terminal sample) prediction:
                         x1_hat = z + (1 - tau) * u(z, tau, 1 - tau)  (1 NFE)

Official iMeanFlow convention (aux repo /workspaces/aux_repo/imeanflow, imf.py)
-------------------------------------------------------------------------------
    z_t = (1 - t) * x + t * e              t=0 data  ->  t=1 noise,  v = e - x
    mapping:  tau = 1 - t,   u_HF = -u_iMF,   h = t - r = (tau'-tau)  (equal)

MeanFlow identity in HardFlow convention (derivation)
-----------------------------------------------------
Fix the interval endpoint s = tau + h and define the displacement
    D(tau) = z_s - z_tau = h * u(z_tau, tau, h),   h = s - tau.
Along the flow dz/dtau = v, so dD/dtau = -v. Expanding the left side with
dh/dtau = -1:
    -u + h * (du/dz . v + du/dtau - du/dh) = -v
    =>   u = v - h * D_tot,
    D_tot = JVP of u(z, tau, h) with tangents (v, +1, -1).
This mirrors the official identity u = v - (t - r) * du/dt (aux imf.py forward()),
with the (+1, -1) tangent pair replacing their (dtdt=1, dtdr=0) because our net
takes (tau, h) directly instead of (t, r).

Training compound (official iMF style, CFG dropped — Gen13 decision D3):
    V = u + h * stopgrad(D_tot)        loss_u = ||V - v_target||^2
    v-head aux:                        loss_v = ||v_pred - v_target||^2
    adaptive weighting:                L / stopgrad((L + eps)^p)
JVP tangent for z is the PREDICTED v (v_c, detached) — the "improved" MeanFlow
feature (predicted-v tangent), per aux imf.py line ~373.

(t, r) -> (tau, h) sampling mapping
-----------------------------------
Official: t, r ~ logit-normal(p_mean, p_std); t = max, r = min; a data_proportion
fraction gets r = t (h=0 flow-matching anchors). Under tau = 1 - t the same
distribution is produced by sampling sigmoid(N(-p_mean, p_std)) and taking
tau = min, s = max, h = s - tau. Config keeps (p_mean, p_std) in the OFFICIAL
convention; the negation happens here.
"""

import torch


def pad_t_like_x(t, x):
    """Reshape a (B,) time tensor to broadcast against x (B, H, T)."""
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


def sample_tau_h(batch_size, p_mean, p_std, data_proportion, device):
    """Sample (tau, h, fm_mask) in HardFlow convention.

    Mirrors aux imf.py sample_tr() through the tau = 1 - t mapping (see module
    docstring). fm_mask marks h=0 flow-matching anchor samples.

    Returns:
        tau: (B,) interval start (query point), in (0, 1)
        h:   (B,) interval width, tau + h <= 1, h >= 0 (0 on fm_mask)
        fm_mask: (B,) bool
    """
    # official-convention logit-normals, negated into HF convention
    a = torch.sigmoid(-(p_mean + p_std * torch.randn(batch_size, device=device)))
    b = torch.sigmoid(-(p_mean + p_std * torch.randn(batch_size, device=device)))
    tau = torch.minimum(a, b)
    s = torch.maximum(a, b)

    data_size = int(batch_size * data_proportion)
    fm_mask = torch.arange(batch_size, device=device) < data_size
    s = torch.where(fm_mask, tau, s)  # h = 0 on anchors

    h = s - tau
    return tau, h, fm_mask


def jvp_tangents(v_c, tau):
    """The identity's JVP tangent triple for primals (z, tau, h): (v_c, +1, -1)."""
    ones = torch.ones_like(tau)
    return (v_c, ones, -ones)


def endpoint_from_u(z, tau, u):
    """Terminal (data-side) sample prediction: x1_hat = z + (1 - tau) * u."""
    return z + pad_t_like_x(1.0 - tau, z) * u if torch.is_tensor(tau) else z + (1.0 - tau) * u


def jump_from_u(z, u, h):
    """Exact interval jump: z_{tau+h} = z + h * u."""
    return z + pad_t_like_x(h, z) * u if torch.is_tensor(h) else z + h * u
