"""Gen13 — ImfMatcher: the improved-MeanFlow training objective, HF convention.

Port of the official iMeanFlow loss (aux repo imeanflow, JAX main branch,
imf.py forward()) to PyTorch, in HardFlow's native time convention, with CFG
dropped entirely (Gen13 decision D3 — HardFlow conditions by state-inpainting).

Mirrors the interface of hardflow.models_flow.flow_matcher.FlowMatcher:
    matcher = ImfMatcher(model=..., action_dim=...)
    loss, infos = matcher.loss(x, cond)

Objective (see convention.py for the derivation):
    z      = tau * x1 + (1 - tau) * x0,     v_target = x1 - x0
    (u, du_tot, v) via torch.func.jvp with tangents (v_c, +1, -1),
        v_c = detached predicted v (the "improved" predicted-v tangent)
    V      = u + h * stopgrad(du_tot)
    loss_u = adp(sum_dims (V - v_target)^2)         # MeanFlow identity, v-form
    loss_v = adp(sum_dims (v - v_target)^2)         # auxiliary v-head
    adp(L) = L / stopgrad((L + eps)^p)              # adaptive normalization
    loss   = (loss_u + loss_v).mean()

IMPORTANT (Gen3v4 lesson, ANALYSIS_imf_official_K2 §0): the adaptive `loss` is
bounded and flat BY CONSTRUCTION — convergence must be judged on the raw MSEs
(`raw_mse_u`, `raw_mse_v`, `a0_mse`) reported in infos, never on `loss`.
"""

import torch

from hardflow.models_flow.flow_matcher import apply_conditioning

from .convention import jvp_tangents, pad_t_like_x, sample_tau_h


class ImfMatcher:

    def __init__(
        self,
        model,
        action_dim,
        p_mean=-0.4,
        p_std=1.4,
        data_proportion=0.25,
        adp_p=1.0,
        adp_eps=0.01,
    ):
        """
        Args:
            model: TemporalImfUnet — callable (x, tau, h) -> (u, v).
            action_dim: action dimension (for state-inpainting conditioning).
            p_mean, p_std: logit-normal (t, r) parameters, OFFICIAL convention
                (mapped to HF inside sample_tau_h). Defaults per Gen13 D5
                (Gen3v4 §6 recommendation: more large-h mass than the official
                -0.4/1.0).
            data_proportion: fraction of h=0 flow-matching anchors (D5: 0.25).
            adp_p, adp_eps: adaptive-loss exponent / epsilon (official: 1, 0.01).
        """
        self.model = model
        self.action_dim = action_dim
        self.p_mean = p_mean
        self.p_std = p_std
        self.data_proportion = data_proportion
        self.adp_p = adp_p
        self.adp_eps = adp_eps

    def _adp(self, loss_per_sample):
        wt = (loss_per_sample + self.adp_eps) ** self.adp_p
        return loss_per_sample / wt.detach()

    def loss(self, x, cond):
        """
        x: (batch_size, planning_horizon, transition_dim), normalized data traj.
        cond: {timestep: state_value} state-inpainting conditions.
        """
        x1 = x
        x0 = torch.randn_like(x1)
        bz = x1.shape[0]

        tau, h, fm_mask = sample_tau_h(
            bz, self.p_mean, self.p_std, self.data_proportion, x1.device
        )

        z = pad_t_like_x(tau, x1) * x1 + pad_t_like_x(1.0 - tau, x1) * x0
        v_target = x1 - x0

        # state-inpainting conditioning, exactly as the FM path does (flow_matcher.loss)
        z = apply_conditioning(z, cond, self.action_dim)

        # predicted-v tangent (the "improved" MeanFlow feature), detached
        with torch.no_grad():
            _, v_c = self.model(z, tau, h)

        def u_fn(z_, tau_, h_):
            u_, v_ = self.model(z_, tau_, h_)
            return u_, v_

        u, du_tot, v = torch.func.jvp(
            u_fn, (z, tau, h), jvp_tangents(v_c, tau), has_aux=True
        )

        V = u + pad_t_like_x(h, u) * du_tot.detach()
        v_target = v_target.detach()

        # per-sample summed squared error over (horizon, transition) dims
        err_u = ((V - v_target) ** 2).sum(dim=(1, 2))
        err_v = ((v - v_target) ** 2).sum(dim=(1, 2))

        loss = (self._adp(err_u) + self._adp(err_v)).mean()

        infos = {
            "loss": loss.item(),                    # adaptive — flat by design
            "raw_mse_u": err_u.mean().item(),       # judge convergence on THESE
            "raw_mse_v": err_v.mean().item(),
            "a0_mse": ((V - v_target)[:, 0, : self.action_dim] ** 2).mean().item(),
            "fm_frac": fm_mask.float().mean().item(),
            "h_mean": h.mean().item(),
        }
        return loss, infos
