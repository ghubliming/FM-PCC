"""MeanFlow diffusion adapter (Gen3v6) with FMv3ODE-compatible training/sampling APIs.

Gen3v6 is a copy-modify sibling of Gen3v4 (`flow_matcher_v3_imeanflow`). It implements the
**original MeanFlow paper** objective (arXiv 2505.13447), *not* improved-MeanFlow (iMF):

    | variant                         | JVP z-tangent   | regression target             |
    |---------------------------------|-----------------|-------------------------------|
    | MeanFlow paper  ← THIS FILE     | ANALYTIC v=x1−x0| u ← sg(v + h·du/dr)           |
    | iMF (Gen3v4 `imf_official`)     | PREDICTED v_c   | V-form, guided v_g target     |

The single scientific payload of this generation is the **analytic-vs-predicted JVP tangent**
A/B against Gen3v4-`imf_official`, measured on constrained control instead of FID. Everything
that only serves iMF (interval-CFG, guided v_g, cond_drop null token, the fm_equivalent
finite-difference arm) has been deleted so this file reads as one objective.

See logs_in_develop/Gen3v6_MeanFlow/init/PLAN_Gen3v6_meanflow_baseline.md.
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple
import time
import warnings

import torch
from torch import nn

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None

from .mf_engine import MeanFlowEngine
from .helpers import apply_conditioning, Losses


class MeanFlowODE(nn.Module):
    """MeanFlow wrapper that preserves FM-PCC/FMv3ODE diffusion interfaces."""

    def __init__(
        self,
        model: MeanFlowEngine,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        goal_dim: int = 0,
        n_timesteps: int = 1000,
        loss_type: str = "l2",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        # ── FIX-3: action_weight / loss_discount are KEPT (utils + folder naming read them)
        #    but are deliberately NOT applied to the MeanFlow loss. They are a DPCC idea with
        #    no counterpart in MeanFlow and they distort the gradient geometry of the identity.
        #    DO NOT "fix" this back — see PLAN §3.2 FIX-3.
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights: Optional[Dict] = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 0.0,
        u_loss_weight: float = 1.0,
        v_loss_weight: float = 1.0,
        loss_schedule: str = "balanced",
        warmup_epochs: int = 0,
        transition_epochs: int = 0,
        # ── Gen3v6 objective knobs ────────────────────────────────────────────────────
        mf_objective: str = 'meanflow',        # only value for now; folder-name slot for future arms
        meanflow_data_proportion: float = 0.5, # fraction of the batch forced to r==t (FM anchors)
        mf_adp_p: float = 1.0,                 # official adaptive-loss exponent  w = (‖Δ‖²+eps)^(−p)
        mf_adp_eps: float = 0.01,              # official adaptive-loss epsilon
        time_beta_alpha_v3: float = 1.0,
        time_beta_beta_v3: float = 1.0,
        t_schedule: str = 'logit_normal',   # 'logit_normal' (official) | 'beta' (legacy ablation)
        p_mean: float = -0.4,               # logit-normal: mean in logit space (official convention)
        p_std: float = 1.0,                 # logit-normal: std in logit space
        flow_steps_v3: Optional[int] = None,
        ode_inference_steps_v3: int = 50,
        ode_solver_backend_v3: str = 'legacy_euler',
        ode_solver_method_v3: str = 'euler',
        ode_solver_rtol_v3: Optional[float] = None,
        ode_solver_atol_v3: Optional[float] = None,
        ode_solver_step_size_v3: Optional[float] = None,
    ):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.loss_type = loss_type

        self.time_beta_alpha_v3 = float(time_beta_alpha_v3)
        self.time_beta_beta_v3 = float(time_beta_beta_v3)
        self.t_schedule = str(t_schedule)
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        resolved_flow_steps = flow_steps_v3 if flow_steps_v3 is not None else ode_inference_steps_v3
        self.flow_steps_v3 = int(resolved_flow_steps)
        self.ode_inference_steps_v3 = int(self.flow_steps_v3)
        self.ode_solver_backend_v3 = str(ode_solver_backend_v3)
        self.ode_solver_method_v3 = str(ode_solver_method_v3)
        self.ode_solver_rtol_v3 = ode_solver_rtol_v3
        self.ode_solver_atol_v3 = ode_solver_atol_v3
        self.ode_solver_step_size_v3 = ode_solver_step_size_v3

        # Gen3v6 objective
        self.mf_objective = str(mf_objective)
        if self.mf_objective != 'meanflow':
            raise ValueError(
                f"mf_objective='{self.mf_objective}' is not implemented in Gen3v6 "
                "(the only value is 'meanflow'). iMF lives in Gen3v4 (flow_matcher_v3_imeanflow)."
            )
        self.meanflow_data_proportion = float(meanflow_data_proportion)
        self.mf_adp_p = float(mf_adp_p)
        self.mf_adp_eps = float(mf_adp_eps)

        # Kept for backward compatibility with existing configs / utils.
        self.loss_schedule = loss_schedule
        self.warmup_epochs = int(warmup_epochs)
        self.transition_epochs = int(transition_epochs)
        self.u_mix = 1.0
        self.v_mix = 1.0

        # NOTE (FIX-3): loss_fn is still constructed — `loss_fn.weights` is part of the
        # state_dict and several utils/serialization paths expect it — but the MeanFlow
        # objective below never multiplies by it.
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        # Buffers retained for FM-PCC compatibility.
        self.register_buffer('betas', torch.linspace(1.0, 0.0, n_timesteps, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.ones(n_timesteps, dtype=torch.float32))

        # Keep the wrapper, buffers, and loss weights on the same device as the backbone.
        model_device = next(self.model.parameters()).device
        self.to(model_device)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    def _predict_uv(self, x, cond, t, h=None, returns=None, force_dropout=False):
        """(u, v) from the two-time backbone.

        Gen3v6 has NO interval-CFG: the (ω, t_min, t_max) net inputs are never fed, so the
        DiT's ω/interval tokens receive their constant default (ω→0 ⇒ w_arg=0, guidance off)
        at BOTH train and sample time. Constant conditioning ⇒ the tokens are inert, and the
        architecture stays byte-identical to Gen3v4's DiT so the A/B is controlled.
        """
        return self.model.forward_train(x, t, h=h, cond=cond, force_dropout=force_dropout)

    def _predict_velocity(self, x, cond, t, h=None, returns=None):
        """Sampling-time velocity: the u (mean-velocity) head ONLY.

        Deviation A (fix_2/REFERENCE_IMF_AUDIT.md §5.3): the v head is discarded at sampling
        time, matching the reference implementations. Mixing it in produced step-to-step
        jitter (the post-fix_1 residual symptom).
        """
        velocity, _v = self._predict_uv(x, cond, t, h=h, returns=returns)
        # DPCC returns-CFG output mix. Inert in Gen3v6 (condition_guidance_w = 0.0); kept so
        # the returns-conditioned eval path stays available without a code change.
        if self.returns_condition and returns is not None and self.condition_guidance_w > 0:
            uncond_vel, _ = self._predict_uv(x, cond, t, h=h, returns=returns, force_dropout=True)
            velocity = (1 + self.condition_guidance_w) * velocity - self.condition_guidance_w * uncond_vel
        return velocity

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t_cont = t
        while t_cont.ndim < x_start.ndim:
            t_cont = t_cont.unsqueeze(-1)
        return (1.0 - t_cont) * noise + t_cont * x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        returns=None,
        return_diffusion=False,
        projector=None,
        constraints=None,
        repeat_last=0,
        num_steps=None,
    ):
        device = self.betas.device
        batch_size = shape[0]
        flow_steps = int(num_steps) if num_steps is not None else self.flow_steps_v3

        x = torch.randn(shape, device=device)  # sigma=1.0 to match q_sample training noise
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        diffusion = [x] if return_diffusion else None
        costs = {}
        # 🔴 Gen15 Fix_1 — wall-clock ms spent inside projector calls during THIS sample.
        # Mirrors FlowMatchingODE.p_sample_loop so all three arms report the same field.
        proj_ms = 0.0

        total_steps = flow_steps + int(repeat_last)
        dt = 1.0 / max(flow_steps, 1)
        h_batch = torch.full((batch_size,), dt, device=device, dtype=torch.float32)

        # ⚠️ SAMPLER — DO NOT TOUCH (PLAN §3.6). The interval-jump update x += dt·u with
        # h = dt = 1/N is already faithful (U10 audit F1, re-verified); MeanFlow and iMF share
        # the identical sampler. This is the unconditional path — Gen3v6 has no CFG special-case.
        #
        # U8 — torchdiffeq dispatch. No flow_steps floor: every sub-stage query computes h
        # dynamically (see ode_rhs below) as the remaining distance to the current macro step's
        # own declared end t1, so every query stays inside the (t,h) domain the model was
        # actually trained on (t,h∈[0,1], t+h<=1 — see _p_losses_meanflow's JVP primal
        # (x_r, r, h) with h=t-r, t<=1). This has nothing to do with the constraint projector
        # below (which runs once per macro step, after this entire block).
        use_torchdiffeq = self.ode_solver_backend_v3 == 'torchdiffeq'
        if use_torchdiffeq and torchdiffeq_odeint is None:
            raise RuntimeError(
                "ode_solver_backend_v3='torchdiffeq' but torchdiffeq is not installed. "
                "Install torchdiffeq or switch backend to 'legacy_euler'."
            )

        # U3-B1 guardrail: NEVER freeze t for this architecture. Training always passed the
        # true t at each step, so the learned weights encode u(x, t, h). A frozen t at
        # inference converts that into a biased h-only function, which is out-of-distribution.
        # Use t_i = loop_idx / flow_steps: the position the sampler is currently AT.
        for i in range(total_steps):
            loop_idx = min(i, flow_steps - 1)
            tau = loop_idx / max(flow_steps, 1)
            t_i = torch.full((batch_size,), tau, device=device, dtype=torch.float32)

            if use_torchdiffeq:
                # "Homing missile": h is recomputed PER SUB-STAGE as the remaining distance to
                # this macro step's own declared end t1, NOT held fixed at dt. h_sub = t1 −
                # t_scalar keeps every sub-stage's query at (t_scalar, h_sub) with
                # t_scalar+h_sub = t1 exactly, always in-domain, for any solver, any
                # flow_steps_v3 — including flow_steps_v3=1.
                t0, t1 = float(loop_idx) * dt, float(loop_idx) * dt + dt
                t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)

                def ode_rhs(t_scalar, state):
                    # t_scalar is a 0-dim tensor from torchdiffeq; multiply rather than
                    # torch.full(..., fill_value=t_scalar), which some torch versions reject.
                    ones = torch.ones(batch_size, device=device, dtype=torch.float32)
                    t_batch = ones * t_scalar
                    h_sub = ones * (t1 - t_scalar)
                    return self._predict_velocity(state, cond, t_batch, h=h_sub, returns=returns)

                odeint_kwargs = {'method': self.ode_solver_method_v3}
                if self.ode_solver_rtol_v3 is not None:
                    odeint_kwargs['rtol'] = float(self.ode_solver_rtol_v3)
                if self.ode_solver_atol_v3 is not None:
                    odeint_kwargs['atol'] = float(self.ode_solver_atol_v3)
                if self.ode_solver_step_size_v3 is not None:
                    fixed_step_methods = {
                        'euler', 'midpoint', 'heun2', 'heun3', 'rk4',
                        'explicit_adams', 'fixed_adams',
                    }
                    if self.ode_solver_method_v3 in fixed_step_methods:
                        odeint_kwargs['options'] = {'step_size': float(self.ode_solver_step_size_v3)}
                    else:
                        warnings.warn(
                            f"Ignoring ode_solver_step_size_v3 for method '{self.ode_solver_method_v3}' "
                            "because it is not a fixed-step solver method.",
                            RuntimeWarning,
                        )

                x = torchdiffeq_odeint(ode_rhs, x, t_span, **odeint_kwargs)[-1]
            else:
                velocity = self._predict_velocity(x, cond, t_i, h=h_batch, returns=returns)
                x = x + velocity * dt

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if projector is not None:
                snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * flow_steps)
                near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)
                if near_end and projector.gradient:
                    # 🔴 Gen15 Fix_1 — accumulate CPU projector wall-time. See the `proj_ms`
                    # comment where it is initialised, above.
                    _t_proj = time.perf_counter()
                    if self.goal_dim > 0:
                        grad = projector.compute_gradient(x[:, :, :-self.goal_dim], constraints)
                    else:
                        grad = projector.compute_gradient(x, constraints)
                    proj_ms += (time.perf_counter() - _t_proj) * 1e3
                    x = x + grad
                    if hasattr(projector, 'compute_cost'):
                        costs[loop_idx] = projector.compute_cost(x, constraints)

                if near_end and not projector.gradient:
                    _t_proj = time.perf_counter()
                    if self.goal_dim > 0:
                        x[:, :, :-self.goal_dim], step_cost = projector.project(x[:, :, :-self.goal_dim], constraints)
                    else:
                        x, step_cost = projector.project(x, constraints)
                    proj_ms += (time.perf_counter() - _t_proj) * 1e3
                    costs[loop_idx] = step_cost

                x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

        infos = {}
        if return_diffusion:
            infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs
        # 🔴 Gen15 Fix_1 — `projection_ms` is part of the infos CONTRACT the UAV frame relies on:
        # sampling/policies.py reads it into `policy.last_proj_ms`, which eval_mix_uav.py prints
        # as the `proj_ms=` field of the per-variant TIMING line. Gen3v6/Gen3v7 never emitted it
        # (their own policies.py has no real-time logging), so before this fix the mf/af arms
        # silently reported proj_ms=0.0 while the `fm` arm reported the real number — a fake
        # cross-arm difference in exactly the metric this generation is built to measure.
        # NOTE: total wall-clock was never wrong; the projector runs inside this loop, so its
        # cost was always inside `fm_ms`. What was missing is the split.
        infos['projection_ms'] = proj_ms
        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, num_steps=None, *args, **kwargs):
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, returns=returns, num_steps=num_steps, *args, **kwargs)

    def sample(
        self,
        batch_size: int,
        returns: Optional[torch.Tensor] = None,
        conditions: Optional[Dict] = None,
        returns_condition: Optional[bool] = None,
        guidance_weight: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        # num_steps is forwarded without mutating self.flow_steps_v3 (BUG-08 fix).
        if conditions is None:
            cond = {0: torch.zeros(batch_size, self.observation_dim, device=self.betas.device)}
        else:
            cond = conditions

        sampled, _ = self.conditional_sample(
            cond=cond, returns=returns, horizon=self.horizon, num_steps=num_steps
        )
        return sampled

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Trainer entrypoint matching FM-PCC's expected `model.loss(*batch)` contract.

        FIX-1 consequence: MeanFlow samples its OWN (t, r) pair from two independent
        logit-normals, so there is no pre-sampled single `t` and no `p_losses()` hop.
        """
        return self._p_losses_meanflow(x, cond, returns=returns)

    def _sample_tau_pair(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Two i.i.d. draws on the τ axis (DATA-AT-1), shape [2, B].

        Official MeanFlow draws s ~ sigmoid(N(P_mean, P_std)), which puts mass near the DATA
        end (s→0) in their noise-at-1 convention. Under our τ = 1 − s that is
        τ ~ sigmoid(N(−P_mean, P_std)), i.e. mass near τ→1 (data).

        🔴 TRAP: using **+p_mean** here puts the mass near NOISE and looks *almost* fine.
        See flow_matcher_v3_imeanflow/models/imf_diffusion.py:666 for the same warning.
        """
        if self.t_schedule == 'logit_normal':
            return torch.sigmoid(torch.randn(2, batch_size, device=device) * self.p_std - self.p_mean)
        # legacy 1−Beta(α,β) ablation arm (α=β=1 ⇒ uniform)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=device)
        beta = torch.tensor(self.time_beta_beta_v3, device=device)
        beta_dist = torch.distributions.Beta(alpha, beta)
        return 1.0 - beta_dist.sample((2, batch_size)).to(device)

    def _adaptive(self, err: torch.Tensor) -> torch.Tensor:
        """Official MeanFlow adaptive L2: err / sg((err + eps)^p), p=1.0, eps=0.01.

        `err` is the PER-SAMPLE SUM over (H, D) — official imeanflow convention. (The
        unofficial aux_repo/MeanFlow uses a per-sample MEAN instead; that only rescales err
        by H·D, but it changes what `eps` means relative to the error scale. G2 parity checks
        must account for this — see PLAN §3.2 FIX-2 / §6 G2.)
        """
        return err / (err + self.mf_adp_eps).pow(self.mf_adp_p).detach()

    def _p_losses_meanflow(
        self,
        x_start: torch.Tensor,
        cond: Dict,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Faithful MeanFlow (2505.13447) objective via a forward-mode JVP.

        Derivation (DATA-AT-1: τ=0 noise, τ=1 data; integrate forward; the sampler anchors at
        the current noise-side point z_r at time r and steps to t=r+h):

            (t − r)·u(z_r, r, t) = z_t − z_r          [u = average velocity over [r,t]]

        Differentiate w.r.t. r at FIXED t, following the trajectory (dz_r/dr = v):

            u(z_r, r, t) = v(z_r, r) + (t − r)·d/dr u        (d/dr = ∂_z u·v + ∂_r u − ∂_h u)

        so the START-anchored target is  u_target = v_inst + h·(JVP)  with tangents
        (∂z = v_inst, ∂time_r = +1, ∂h = −1)  since h = t − r ⇒ dh/dr = −1.
        At the r==t anchor (h=0) this reduces to u_target = v_inst — the FM velocity — which
        grounds the field. The target is stop-gradiented: we regress to it, never backprop
        through the JVP (that would need 2nd-order grads).

        Cross-check vs aux_repo/MeanFlow (noise-at-1, anchor at their t, tangents (v,+1,0)):
        their identity u = v − h·du/dt maps onto ours under u_ours=−u_theirs, v_ours=−v_theirs,
        d/ds = −d/dτ, giving u = v + h·du/dr exactly. Signs agree.

        NOTE (no local runtime here): correctness must be verified on the cluster — gates
        G0–G3 in PLAN §6.
        """
        try:
            from torch.func import jvp as _jvp
        except ImportError:  # older torch
            from functorch import jvp as _jvp

        device = x_start.device
        B = x_start.shape[0]
        ad, gd = self.action_dim, self.goal_dim

        # ── FIX-1: (t, r) from two INDEPENDENT draws on the τ axis ───────────────────────
        # The pre-fix code used r = t·U(0,1), which forces h ≤ t and starves large-h
        # (few-NFE) queries exactly where 1–2-step sampling lives.
        taus = self._sample_tau_pair(B, device)
        t = torch.maximum(taus[0], taus[1])      # data-side end
        r = torch.minimum(taus[0], taus[1])      # noise-side anchor = the network's query point
        fm_mask = torch.rand(B, device=device) < self.meanflow_data_proportion
        r = torch.where(fm_mask, t, r)           # FM anchors: h=0 ⇒ u_target = v_inst
        h = t - r                                # ≥ 0

        # noise (DATA-AT-1 τ=0 side), pinned to 0 at conditioned dims
        x_base = torch.randn_like(x_start)
        x_base = apply_conditioning(x_base, cond, ad, goal_dim=gd, noise=True)

        # anchor point z_r at time r (noise side) — matches the sampler's query convention
        x_r = self.q_sample(x_start=x_start, t=r, noise=x_base)
        x_r = apply_conditioning(x_r, cond, ad, goal_dim=gd)

        # instantaneous (FM) velocity v = x_data − noise, pinned to 0 at conditioned dims.
        # This is BOTH the identity's v(z_r, r) AND the z-tangent for the JVP.
        v_inst = x_start - x_base
        v_inst = apply_conditioning(v_inst, cond, ad, goal_dim=gd, noise=True)

        def _u_of(z_in, r_in, h_in):
            u, _v = self._predict_uv(z_in, cond, r_in, h=h_in, returns=returns)
            return u

        ones = torch.ones_like(r)
        # ══════════════════════════════════════════════════════════════════════════════
        # 🔴 DO NOT CHANGE THE z-TANGENT. `v_inst` is the ANALYTIC velocity x1 − x0 and it
        # IS the Gen3v6 hypothesis. Feeding a PREDICTED v_c here turns Gen3v6 into
        # Gen3v4-iMF and destroys the entire ablation. A future agent reading the iMF
        # audit will be tempted to "fix" this — do not.
        # ══════════════════════════════════════════════════════════════════════════════
        u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))

        h_exp = h
        while h_exp.ndim < x_start.ndim:
            h_exp = h_exp.unsqueeze(-1)

        # MeanFlow-Identity target (START-anchored): u ← v + h·du/dr ; stop-gradient.
        u_target = (v_inst + h_exp * du_dr).detach()
        u_target = apply_conditioning(u_target, cond, ad, goal_dim=gd, noise=True)

        # ── FIX-4: the v head is a FULL second loss, not a 0.05 stabiliser ───────────────
        # Same footing as the u loss, matching MeanFlow's dual head (aux_repo/MeanFlow's
        # `fm_loss`, which uses the v_p from the same query point).
        _u2, v_pred = self._predict_uv(x_r, cond, r, h=h, returns=returns)

        # ── FIX-2 / FIX-3: official adaptive loss, per-sample SUM, NO DPCC loss_weights ──
        reduce_dims = tuple(range(1, u_pred.ndim))
        err_u = (u_pred - u_target).pow(2).sum(dim=reduce_dims)     # [B]
        err_v = (v_pred - v_inst.detach()).pow(2).sum(dim=reduce_dims)
        loss = (self._adaptive(err_u) + self._adaptive(err_v)).mean()

        # ── Metrics (PLAN §3.4). NEVER read `diffusion_loss` as convergence: the adaptive
        # loss is pinned at its ceiling by construction (COMPARE §7.1). Read raw_mse_u.
        info = self._build_info(
            loss, err_u, err_v, (u_pred - u_target).detach(), h, fm_mask, x_start)
        return loss, info

    def _build_info(self, loss, err_u, err_v, delta_u, h, fm_mask, x_start) -> Dict:
        device = x_start.device
        ad = self.action_dim
        n_elem = float(x_start.shape[1] * x_start.shape[2])   # H·D

        with torch.no_grad():
            err_u_d, err_v_d = err_u.detach(), err_v.detach()
            raw_mse_u = err_u_d.mean()
            raw_mse_v = err_v_d.mean()
            a0 = delta_u.detach()[:, 0, :ad].pow(2).mean() if ad > 0 else torch.tensor(0.0, device=device)

            info = {
                # kept for pipeline compat — DO NOT read as a convergence signal
                'diffusion_loss': loss.detach(),
                'total_loss': loss,
                'a0_loss': a0,
                # the REAL convergence signals
                'raw_mse_u': raw_mse_u,
                'raw_mse_v': raw_mse_v,
                # per-dim RMS — comparable across horizons and generations
                'per_dim_rms_u': torch.sqrt(raw_mse_u / n_elem),
                # back-compat aliases for the inherited pkl/W&B keys
                'raw_mse': raw_mse_u,
                'aux_loss': raw_mse_v,
                'u_weight': torch.tensor(1.0, device=device),
                'v_weight': torch.tensor(1.0, device=device),
                # sampler sanity
                'h_mean': h.detach().mean(),
                'fm_frac': fm_mask.float().mean(),
            }

            # ⭐ h-stratified residual — the single highest-value metric here (COMPARE §7.4.1,
            # never implemented in any previous generation). It answers directly: is the field
            # bad only at large h, which is exactly where few-NFE sampling lives?
            # Empty buckets emit NaN; the trainer drops NaN points (sparse curves), it does not
            # log them as zeros.
            hd = h.detach()
            buckets = (
                hd == 0,
                (hd > 0) & (hd < 0.3),
                (hd >= 0.3) & (hd < 0.6),
                hd >= 0.6,
            )
            nan = torch.tensor(float('nan'), device=device)
            for i, mask in enumerate(buckets):
                info[f'h_mse_b{i}'] = err_u_d[mask].mean() if bool(mask.any()) else nan

        return info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with compatibility for legacy inner-engine checkpoints."""
        remapped_state_dict, was_legacy = self._remap_state_dict_for_compatibility(state_dict)

        if was_legacy:
            incompatible_keys = super().load_state_dict(remapped_state_dict, strict=False)
            allowed_missing = {'betas', 'alphas_cumprod', 'loss_fn.weights'}
            missing_keys = [key for key in incompatible_keys.missing_keys if key not in allowed_missing]
            unexpected_keys = list(incompatible_keys.unexpected_keys)

            if strict and (missing_keys or unexpected_keys):
                raise RuntimeError(
                    'Error(s) in loading state_dict for MeanFlowODE:\n'
                    f'\tMissing key(s) in state_dict: {missing_keys}\n'
                    f'\tUnexpected key(s) in state_dict: {unexpected_keys}'
                )

            return incompatible_keys

        return super().load_state_dict(remapped_state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the full wrapper state so future checkpoints stay self-describing."""
        return super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    @staticmethod
    def _remap_state_dict_for_compatibility(state_dict):
        """Translate legacy checkpoint keys saved from the inner engine."""
        if not isinstance(state_dict, dict):
            return state_dict, False

        if any(key.startswith('model.velocity_net.') or key.startswith('model.aux_head.') for key in state_dict):
            remapped = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith('model.') and not key.startswith('model.model.'):
                    remapped[f'model.{key}'] = value
                else:
                    remapped[key] = value
            return remapped, True

        return state_dict, False
