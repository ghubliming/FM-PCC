"""Gen13 U11 — MfMatcher: the faithful MeanFlow (Gen3v6) objective, HF convention.

Ported from `flow_matcher_v3_meanflow/models/mf_diffusion.py::_p_losses_meanflow`
into HardFlow's native convention + state-inpainting conditioning, mirroring
`imf/imf_matcher.py` EXACTLY so the iMF↔MF comparison is controlled to a SINGLE line:

    iMF (imf_matcher.py):  JVP z-tangent = PREDICTED v_c   (detached model v-head)
    MF  (this file):       JVP z-tangent = ANALYTIC  v_target = x1 - x0   ← the ONLY diff

Both HardFlow and Gen3v6 use the DATA-AT-1 convention (tau=0 noise, tau=1 data), so
NO sign flip is involved — see `imf/convention.py`. (The JAX iMeanFlow flip that iMF
needed does NOT apply here; Gen3v6 is already the torch/DATA-AT-1 dialect.)

Objective (u-target form, faithful to Gen3v6):
    z         = tau*x1 + (1-tau)*x0,       v_target = x1 - x0
    u,du_dr,v = jvp(model, (z,tau,h), tangents=(v_target, +1, -1), has_aux=True)
    u_target  = stopgrad(v_target + h*du_dr)          # MeanFlow identity u = v + h*D_tot
    loss_u    = adp( sum (u - u_target)^2 )
    loss_v    = adp( sum (v - v_target)^2 )            # aux v-head (identical to iMF)
    loss      = (loss_u + loss_v).mean()

IMPORTANT (same lesson as iMF): the adaptive `loss` is flat BY CONSTRUCTION — judge
convergence on `raw_mse_u` / `raw_mse_v`, never on `loss`.
"""

import torch

from hardflow.models_flow.flow_matcher import apply_conditioning

from ..imf.convention import jvp_tangents, pad_t_like_x, sample_tau_h


class MfMatcher:

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
            model: TemporalImfUnet — callable (x, tau, h) -> (u, v). Shared backbone
                with iMF (U11: architecture held constant across MLbones).
            action_dim: action dimension (for state-inpainting conditioning).
            p_mean, p_std: logit-normal (t, r) parameters, OFFICIAL convention
                (mapped to HF inside sample_tau_h). Gen3v6 uses the same log-normal
                family as iMF.
            data_proportion: fraction of h=0 flow-matching anchors (Gen3v6 `dp`).
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

        # state-inpainting conditioning, exactly as iMF / the FM path do
        z = apply_conditioning(z, cond, self.action_dim)

        # ════════════════════════════════════════════════════════════════════════════
        # 🔴 THE ONE LINE THAT MAKES THIS MEANFLOW AND NOT iMF: the JVP z-tangent is the
        # ANALYTIC velocity v_target = x1 - x0 (the true dz/dtau along the flow), NOT a
        # predicted v_c. Replacing it with the model's v-head turns MF into Gen3v4 iMF
        # and destroys the U11 A/B. (Mirror of mf_diffusion.py's red banner.) NOT a knob.
        # ════════════════════════════════════════════════════════════════════════════
        def u_fn(z_, tau_, h_):
            u_, v_ = self.model(z_, tau_, h_)
            return u_, v_

        u, du_dr, v = torch.func.jvp(
            u_fn, (z, tau, h), jvp_tangents(v_target, tau), has_aux=True
        )

        # MeanFlow identity (u-target form): regress u -> stopgrad(v + h*D_tot).
        u_target = (v_target + pad_t_like_x(h, u) * du_dr).detach()
        v_target = v_target.detach()

        # per-sample summed squared error over (horizon, transition) dims
        err_u = ((u - u_target) ** 2).sum(dim=(1, 2))
        err_v = ((v - v_target) ** 2).sum(dim=(1, 2))

        loss = (self._adp(err_u) + self._adp(err_v)).mean()

        infos = {
            "loss": loss.item(),                    # adaptive — flat by design
            "raw_mse_u": err_u.mean().item(),       # judge convergence on THESE
            "raw_mse_v": err_v.mean().item(),
            "a0_mse": ((u - u_target)[:, 0, : self.action_dim] ** 2).mean().item(),
            "fm_frac": fm_mask.float().mean().item(),
            "h_mean": h.mean().item(),
        }
        return loss, infos
