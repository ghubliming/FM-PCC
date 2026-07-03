"""iMeanFlow diffusion adapter with FMv3ODE-compatible training and sampling APIs."""

from collections import OrderedDict
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None

from .imf_engine import iMeanFlowEngine
from .helpers import apply_conditioning, Losses


class iMeanFlowODE(nn.Module):
    """iMeanFlow wrapper that preserves FM-PCC/FMv3ODE diffusion interfaces."""
    
    def __init__(
        self,
        model: iMeanFlowEngine,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        goal_dim: int = 0,
        n_timesteps: int = 1000,
        loss_type: str = "l2",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights: Optional[Dict] = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 0.1,
        u_loss_weight: float = 1.0,
        v_loss_weight: float = 0.1,
        loss_schedule: str = "balanced",
        warmup_epochs: int = 0,
        transition_epochs: int = 0,
        # ── U4 imfv2: flag-gated training objective (see logs_in_develop/Gen3v4_imf/U4) ──
        # 'fm_equivalent' : legacy finite-difference target (collapses to FM, the A/B baseline)
        # 'meanflow_jvp'  : real MeanFlow-Identity target via JVP (the imfv2 objective)
        imf_objective: str = 'fm_equivalent',
        meanflow_r_equals_t_frac: float = 0.25,   # fraction of batch forced to r==t (FM anchor)
        meanflow_adaptive_p: float = 0.5,         # adaptive-weight exponent  w=(‖Δ‖²+c)^(−p)
        meanflow_adaptive_c: float = 1e-3,        # adaptive-weight epsilon
        meanflow_aux_weight: float = 0.0,         # aux v-head stabilizer (shares backbone iff dual_head)
        # U5 Phase 1c — interval-CFG (off when omega=0). Needs the engine built with interval_cfg=True.
        # Fix2 (U6): meanflow_cfg_omega/t_min/t_max are now the EVAL-time operating point only
        # (the single (ω,τ_min,τ_max) p_sample_loop guides with). At TRAIN time they no longer
        # fill a constant — ω/t_min/t_max are sampled per-batch-element (see
        # _sample_cfg_scale/_sample_cfg_interval below, matching official imf.py:140-175), with
        # meanflow_cfg_omega reused as the training distribution's ceiling (s_max). This is what
        # lets the eval-time operating point be in-distribution instead of a single overfit point.
        meanflow_cfg_omega: float = 0.0,          # CFG scale ω: train s_max ceiling AND eval operating point (0 ⇒ off)
        meanflow_cfg_t_min: float = 0.0,          # guidance interval lower bound (τ) — eval operating point only
        meanflow_cfg_t_max: float = 1.0,          # guidance interval upper bound (τ) — eval operating point only
        meanflow_cfg_beta: float = 1.0,           # power-law shape for ω sampling (1.0 = official default branch)
        time_beta_alpha_v3: float = 1.5,
        time_beta_beta_v3: float = 1.0,
        # U7: time-schedule selector. 'logit_normal' = canonical iMF (reference imf.py L120-124).
        # 'beta' = legacy 1-Beta(α,β) — keep for A-B ablation (uniform = beta with α=β=1).
        t_schedule: str = 'logit_normal',   # DEFAULT: logit-normal (iMF paper default)
        p_mean: float = -0.4,               # logit-normal: mean in logit space (sigmoid median ≈ 0.40)
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
        self.t_schedule = str(t_schedule)   # U7
        self.p_mean = float(p_mean)         # U7
        self.p_std = float(p_std)           # U7
        resolved_flow_steps = flow_steps_v3 if flow_steps_v3 is not None else ode_inference_steps_v3
        self.flow_steps_v3 = int(resolved_flow_steps)
        self.ode_inference_steps_v3 = int(self.flow_steps_v3)
        self.ode_solver_backend_v3 = str(ode_solver_backend_v3)
        self.ode_solver_method_v3 = str(ode_solver_method_v3)
        self.ode_solver_rtol_v3 = ode_solver_rtol_v3
        self.ode_solver_atol_v3 = ode_solver_atol_v3
        self.ode_solver_step_size_v3 = ode_solver_step_size_v3

        # U4 imfv2: training-objective selector + MeanFlow-JVP hyperparameters.
        self.imf_objective = str(imf_objective)
        self.meanflow_r_equals_t_frac = float(meanflow_r_equals_t_frac)
        self.meanflow_adaptive_p = float(meanflow_adaptive_p)
        self.meanflow_adaptive_c = float(meanflow_adaptive_c)
        self.meanflow_aux_weight = float(meanflow_aux_weight)
        # U5 interval-CFG knobs — meanflow_cfg_omega/t_min/t_max are the EVAL operating point
        # (p_sample_loop guides with exactly these); meanflow_cfg_omega doubles as the TRAIN
        # sampling ceiling (s_max) for _sample_cfg_scale. See Fix2 changelog.
        self.meanflow_cfg_omega = float(meanflow_cfg_omega)
        self.meanflow_cfg_t_min = float(meanflow_cfg_t_min)
        self.meanflow_cfg_t_max = float(meanflow_cfg_t_max)
        self.meanflow_cfg_beta = float(meanflow_cfg_beta)

        # Keep parameters for backward compatibility with existing configs.
        self.loss_schedule = loss_schedule
        self.warmup_epochs = int(warmup_epochs)
        self.transition_epochs = int(transition_epochs)
        total_w = float(u_loss_weight) + float(v_loss_weight)
        if total_w <= 0:
            self.u_mix = 1.0
            self.v_mix = 0.1
        else:
            self.u_mix = float(u_loss_weight) / total_w
            self.v_mix = float(v_loss_weight) / total_w
        self.sample_aux_weight = 0.1 * self.v_mix
        self.aux_loss_weight = max(0.01, 0.1 * float(v_loss_weight))

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

    def _predict_uv(self, x, cond, t, h=None, returns=None, force_dropout=False,
                    omega=None, t_min=None, t_max=None):
        return self.model.forward_train(
            x, t, h=h, cond=cond, force_dropout=force_dropout,
            omega=omega, t_min=t_min, t_max=t_max,
        )

    def _predict_velocity(self, x, cond, t, h=None, returns=None,
                          omega=None, t_min=None, t_max=None, cfg_scale=0.0):
        # FIX-3 / Deviation A (per fix_2/REFERENCE_IMF_AUDIT.md §5.3 + §7.1):
        # reference iMF's inference uses ONLY the u (mean-velocity) head and
        # explicitly DISCARDS the v (instantaneous-velocity) head at sampling
        # time. See imeanflow/imf.py:93 (`u_fn(...)[0]`) and imfDiT.py:282-288
        # (v_heads are not even instantiated when eval_mode=True).
        #
        # The aux head is an MLP on the current latent x; as x drifts during
        # integration, its output varies step-to-step even though the true
        # mean-flow target is a CONSTANT v_const = x_data − noise. Mixing
        # `sample_aux_weight * aux` into the sampling velocity introduced
        # step-to-step jitter that was the post-fix_1 residual symptom.
        velocity, _aux = self._predict_uv(x, cond, t, h=h, returns=returns,
                                          omega=omega, t_min=t_min, t_max=t_max)
        if self.returns_condition and returns is not None and self.condition_guidance_w > 0:
            uncond_vel, _ = self._predict_uv(x, cond, t, h=h, returns=returns, force_dropout=True,
                                             omega=omega, t_min=t_min, t_max=t_max)
            velocity = (1 + self.condition_guidance_w) * velocity - self.condition_guidance_w * uncond_vel
        # U5 Phase 1c — interval-CFG: u_cfg = u_uncond + ω·(u_cond − u_uncond).
        # The caller (p_sample_loop) gates cfg_scale>0 to within [t_min, t_max].
        if cfg_scale > 0:
            uncond, _ = self._predict_uv(x, cond, t, h=h, returns=returns, force_dropout=True,
                                         omega=omega, t_min=t_min, t_max=t_max)
            velocity = uncond + cfg_scale * (velocity - uncond)
        return velocity   # was: velocity + self.sample_aux_weight * aux

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

        total_steps = flow_steps + int(repeat_last)
        dt = 1.0 / max(flow_steps, 1)
        h_batch = torch.full((batch_size,), dt, device=device, dtype=torch.float32)

        # U8 — torchdiffeq dispatch (recovered from FMv3ODE / flow_matcher_v3_imeanflow's own
        # sibling diffusion.py:FlowMatchingIMF). No flow_steps floor: h for every sub-stage query
        # is computed dynamically per-substage (see ode_rhs below) as the remaining distance to
        # this macro step's own declared end, so every query is domain-valid (t+h<=1) regardless
        # of flow_steps_v3, including flow_steps_v3=1. Whether treating u(x,t,h) as a stand-in for
        # the instantaneous derivative RK4/dopri5 expect at each sub-stage gives a GOOD estimate at
        # small flow_steps_v3 is a separate, genuinely empirical question (model- and
        # task-dependent curvature of the true velocity field) — not something with a derivable
        # numeric cutoff, so no threshold is hard-coded here. Validate empirically per the U8
        # plan's §6 sequence before trusting results at any given flow_steps_v3.
        use_torchdiffeq = self.ode_solver_backend_v3 == 'torchdiffeq'
        if use_torchdiffeq and torchdiffeq_odeint is None:
            raise RuntimeError(
                "ode_solver_backend_v3='torchdiffeq' but torchdiffeq is not installed. "
                "Install torchdiffeq or switch backend to 'legacy_euler'."
            )

        # U5 Phase 1c — interval-CFG setup (active only when ω>0; net ignores knobs if not built
        # with interval_cfg). omega/t_min/t_max are conditioning inputs to the backbone.
        cfg_on = self.meanflow_cfg_omega > 0
        omega_b = t_min_b = t_max_b = None
        if cfg_on:
            omega_b = torch.full((batch_size,), self.meanflow_cfg_omega, device=device, dtype=torch.float32)
            t_min_b = torch.full((batch_size,), self.meanflow_cfg_t_min, device=device, dtype=torch.float32)
            t_max_b = torch.full((batch_size,), self.meanflow_cfg_t_max, device=device, dtype=torch.float32)

        # U3-B1 guardrail: NEVER freeze t for this architecture. Both time_mlp(t)
        # and h_mlp(h) are active and additively combined in the UNet backbone
        # (unet1d_temporal_cond.py:118-123, :249). Training always passed the true
        # t at each step, so the learned weights encode u(x, t, h). A frozen t at
        # inference converts that into a biased h-only function; every step receives
        # (x@t_i, t=constant), which is out-of-distribution and produces chaotic
        # rollouts. (This was Deviation B from fix_3 — reverted here.)
        # Use t_i = loop_idx / flow_steps: the position the sampler is currently AT.
        # See Gen8E1F2_Problem&Solution_Fable.md §B1 for the full derivation.

        for i in range(total_steps):
            loop_idx = min(i, flow_steps - 1)
            tau = loop_idx / max(flow_steps, 1)
            t_i = torch.full(
                (batch_size,), tau,
                device=device, dtype=torch.float32,
            )
            # apply guidance only inside the [t_min, t_max] interval (official interval-CFG)
            step_cfg = self.meanflow_cfg_omega if (cfg_on and self.meanflow_cfg_t_min <= tau <= self.meanflow_cfg_t_max) else 0.0

            if use_torchdiffeq:
                # h is recomputed PER SUB-STAGE as the remaining distance to this macro step's
                # declared end t1, NOT held fixed at dt. Fixed multi-stage methods (rk4, etc.)
                # evaluate the RHS at interior points (e.g. rk4: t_i, t_i+dt/2, t_i+dt/2, t_i+dt);
                # holding h=dt for all of them would ask u(x, t_i+dt/2, h=dt) — i.e. "average
                # velocity over [t_i+dt/2, t_i+1.5dt]" — which overshoots this macro step's own
                # boundary by dt/2, and for the final sub-stage (t=t1) overshoots by a full dt,
                # querying past t=1 on the last macro step regardless of how large flow_steps is.
                # h_sub = t1 - t_scalar keeps every sub-stage's implied interval [t_scalar, t1] —
                # always ending exactly at the macro boundary, for any solver, any K. At the
                # first sub-stage h_sub=dt (matches Euler); at t_scalar=t1, h_sub=0, which is the
                # valid base case u(x,t,h=0)=v(x,t), not an out-of-domain query.
                t0, t1 = float(loop_idx) * dt, float(loop_idx) * dt + dt
                t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)

                def ode_rhs(t_scalar, state):
                    # t_scalar is a 0-dim tensor from torchdiffeq; multiply rather than
                    # torch.full(..., fill_value=t_scalar), which some torch versions reject.
                    ones = torch.ones(batch_size, device=device, dtype=torch.float32)
                    t_batch = ones * t_scalar
                    h_sub = ones * (t1 - t_scalar)
                    return self._predict_velocity(
                        state, cond, t_batch, h=h_sub, returns=returns,
                        omega=omega_b, t_min=t_min_b, t_max=t_max_b, cfg_scale=step_cfg,
                    )

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
                velocity = self._predict_velocity(
                    x, cond, t_i, h=h_batch, returns=returns,
                    omega=omega_b, t_min=t_min_b, t_max=t_max_b, cfg_scale=step_cfg,
                )
                x = x + velocity * dt

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if projector is not None:
                snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * flow_steps)
                near_end = (loop_idx >= snapping_start_idx) or (loop_idx == flow_steps - 1)
                if near_end and projector.gradient:
                    if self.goal_dim > 0:
                        grad = projector.compute_gradient(x[:, :, :-self.goal_dim], constraints)
                    else:
                        grad = projector.compute_gradient(x, constraints)
                    x = x + grad
                    if hasattr(projector, 'compute_cost'):
                        costs[loop_idx] = projector.compute_cost(x, constraints)

                if near_end and not projector.gradient:
                    if self.goal_dim > 0:
                        x[:, :, :-self.goal_dim], step_cost = projector.project(x[:, :, :-self.goal_dim], constraints)
                    else:
                        x, step_cost = projector.project(x, constraints)
                    costs[loop_idx] = step_cost

                x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

        infos = {}
        if return_diffusion:
            infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs
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
        """Trainer entrypoint matching FM-PCC's expected `model.loss(*batch)` contract."""
        batch_size = x.shape[0]
        # U7: time-schedule selector (t_schedule set in config / __init__).
        if self.t_schedule == 'logit_normal':
            # Canonical iMF schedule — matches reference imf.py L120-124:
            #   sigmoid(randn * P_std + P_mean), P_mean=-0.4, P_std=1.0 → median ≈ 0.40
            t = torch.sigmoid(torch.randn(batch_size, device=x.device) * self.p_std + self.p_mean)
        else:  # 'beta' — legacy 1-Beta(α,β). Set α=β=1 for uniform. (pre-U7 default)
            alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
            beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
            beta_dist = torch.distributions.Beta(alpha, beta)
            t = 1.0 - beta_dist.sample((batch_size,))
        return self.p_losses(x, cond, t, returns=returns)
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Dict,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # U4 imfv2 dispatch: 'meanflow_jvp' runs the real MeanFlow-Identity objective;
        # 'fm_equivalent' (default) keeps the legacy finite-difference path below unchanged.
        if self.imf_objective == 'meanflow_jvp':
            return self._p_losses_meanflow_jvp(x_start, cond, t, returns=returns)

        # Sample noise with sigma=1.0 (matches q_sample training distribution at t=0)
        x_base = torch.randn_like(x_start)
        x_base = apply_conditioning(x_base, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        # Sample r ~ Uniform(0, t) per sample to define the mean-flow interval [r, t]
        r = t * torch.rand_like(t)
        h = t - r  # step size h = t - r > 0

        # Interpolants at times t and r (DATA-AT-1: t=0 is noise, t=1 is data)
        x_t = self.q_sample(x_start=x_start, t=t, noise=x_base)   # data-biased (target side)
        x_r = self.q_sample(x_start=x_start, t=r, noise=x_base)   # noise-biased (model input)
        # U3-B2: condition on x_r (noise-side endpoint) — matches the sampler, which presents
        # the current noise-side x at each step. Old code conditioned on x_t (data-side), which
        # the sampler never has access to at inference. apply_conditioning moves to x_r.
        x_r = apply_conditioning(x_r, cond, self.action_dim, goal_dim=self.goal_dim)

        # Expand h for broadcasting against [batch, horizon, dim] tensors
        h_expand = h
        while h_expand.ndim < x_start.ndim:
            h_expand = h_expand.unsqueeze(-1)

        # Mean flow target: (x_t - x_r) / h — average velocity over interval [r, t].
        # For the linear interpolant this equals the constant v = x_data − noise,
        # independent of which endpoint we condition on. Target direction unchanged.
        u_target = (x_t - x_r) / (h_expand + 1e-8)
        u_target = apply_conditioning(u_target, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        # Instantaneous FM velocity target for the aux (v) branch
        v_target = x_start - x_base
        v_target = apply_conditioning(v_target, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        # U3-B2: predict from x_r at time r (noise-side) — consistent with inference.
        velocity_pred, aux_pred = self._predict_uv(x_r, cond, r, h=h, returns=returns)
        if not self.predict_epsilon:
            velocity_pred = apply_conditioning(velocity_pred, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        main_loss, info = self.loss_fn(velocity_pred, u_target)
        aux_loss = F.mse_loss(aux_pred, v_target)  # real v_target, not zero
        # Apply u_mix to main loss so the u/v weighting is actually applied
        total_loss = self.u_mix * main_loss + self.aux_loss_weight * aux_loss

        info['diffusion_loss'] = main_loss
        info['a0_loss'] = info.get('a0_loss', torch.tensor(0.0, device=x_start.device))
        info['aux_loss'] = aux_loss
        info['u_weight'] = torch.tensor(self.u_mix, device=x_start.device)
        info['v_weight'] = torch.tensor(self.v_mix, device=x_start.device)
        info['total_loss'] = total_loss

        return total_loss, info

    def _sample_cfg_scale(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Per-sample CFG scale ω ~ power-law on (0, s_max], s_max=meanflow_cfg_omega.

        Direct port of the official iMF `sample_cfg_scale` (imeanflow/imf.py:140-159): drawing a
        FRESH ω per training sample (instead of FM-PCC's old fixed constant) is what lets the
        network learn the guidance manifold instead of overfitting to one operating point.
        """
        s_max = torch.as_tensor(self.meanflow_cfg_omega, device=device, dtype=dtype)
        u = torch.rand(shape, device=device, dtype=dtype)
        if self.meanflow_cfg_beta == 1.0:
            return torch.exp(u * torch.log1p(s_max))
        beta = torch.as_tensor(self.meanflow_cfg_beta, device=device, dtype=dtype)
        log_base = (1.0 - beta) * torch.log1p(s_max)
        log_inner = torch.log1p(u * torch.expm1(log_base))
        return torch.exp(log_inner / (1.0 - beta))

    def _sample_cfg_interval(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample CFG interval [t_min, t_max] ~ U(0,0.5) x U(0.5,1).

        Direct port of official `sample_cfg_interval` (imeanflow/imf.py:163-178). Caller is
        responsible for overriding to [0,1] on FM-anchor (r==t) samples, matching the official
        `fm_mask` special-case ("for flow matching samples, there is no CFG interval").
        """
        t_min = torch.rand(shape, device=device, dtype=dtype) * 0.5
        t_max = 0.5 + torch.rand(shape, device=device, dtype=dtype) * 0.5
        return t_min, t_max

    def _p_losses_meanflow_jvp(
        self,
        x_start: torch.Tensor,
        cond: Dict,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Real iMF objective — MeanFlow Identity via a forward-mode JVP (U4 imfv2).

        Derivation (Gen3v4 DATA-AT-1: τ=0 noise, τ=1 data; integrate forward; the sampler
        anchors at the current noise-side point z_r at time r and steps to t=r+h):

            (t − r)·u(z_r, r, t) = z_t − z_r          [u = average velocity over [r,t]]

        Differentiate w.r.t. r at FIXED t, following the trajectory (dz_r/dr = v):

            u(z_r, r, t) = v(z_r, r) + (t − r)·d/dr u        (d/dr = ∂_z u·v + ∂_r u − ∂_h u)

        so the START-anchored target is  u_target = v_inst + h·(JVP)  with tangents
        (∂z = v_inst, ∂time_r = +1, ∂h = −1)  since h = t − r ⇒ dh/dr = −1.
        At the r==t anchor (h=0) this reduces to u_target = v_inst — the FM velocity — which
        grounds the field. The target is stop-gradiented: we regress to it, never backprop
        through the JVP (that would need 2nd-order grads).

        NOTE (no local runtime here): correctness must be verified on the cluster by a 1-NFE
        reconstruction check. If it diverges, the prime suspect is the JVP sign / the h-tangent.
        """
        try:
            from torch.func import jvp as _jvp
        except ImportError:  # older torch
            from functorch import jvp as _jvp

        # noise (DATA-AT-1 t=0 side), masked at conditioned dims
        x_base = torch.randn_like(x_start)
        x_base = apply_conditioning(x_base, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        # interval [r, t]; force a fraction to r==t so those samples anchor to the FM velocity
        r = t * torch.rand_like(t)
        anchor = torch.rand_like(t) < self.meanflow_r_equals_t_frac
        r = torch.where(anchor, t, r)
        h = (t - r)

        # anchor point z_r at time r (noise-side) — matches the inference query convention
        x_r = self.q_sample(x_start=x_start, t=r, noise=x_base)
        x_r = apply_conditioning(x_r, cond, self.action_dim, goal_dim=self.goal_dim)

        # instantaneous (FM) velocity v = x_data − noise, masked at conditioned dims.
        # This is BOTH the identity's v(z_r,r) AND the z-tangent for the JVP.
        v_inst = x_start - x_base
        v_inst = apply_conditioning(v_inst, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        # U5 Phase 1c / Fix2 (U6) — interval-CFG conditioning during training. ω/t_min/t_max are
        # SAMPLED PER BATCH ELEMENT (held constant only through the JVP itself — these are
        # guidance knobs, not differentiated inputs), matching official imf.py's per-sample
        # sample_cfg_scale/sample_cfg_interval. Previously this filled a literal constant
        # (meanflow_cfg_omega/t_min/t_max) for every sample, every step — the network only ever
        # saw one point on the guidance manifold (see Fix2_CFG&EMA changelog). Off when omega=0.
        omega_c = t_min_c = t_max_c = None
        if self.meanflow_cfg_omega > 0:
            omega_c = self._sample_cfg_scale(t.shape, t.device, t.dtype)
            t_min_c, t_max_c = self._sample_cfg_interval(t.shape, t.device, t.dtype)
            # FM-anchor (r==t) samples get the full interval — no CFG restriction — matching
            # official `fm_mask`: "for flow matching samples, there is no CFG interval".
            t_min_c = torch.where(anchor, torch.zeros_like(t_min_c), t_min_c)
            t_max_c = torch.where(anchor, torch.ones_like(t_max_c), t_max_c)

        # u-head as a pure function of the differentiated inputs (cond/returns/cfg held constant).
        def _u_of(z_in, t_in, h_in):
            u, _aux = self._predict_uv(z_in, cond, t_in, h=h_in, returns=returns,
                                       omega=omega_c, t_min=t_min_c, t_max=t_max_c)
            return u

        ones = torch.ones_like(r)
        # tangents: dz=v_inst, d(time r)=+1, d(h=t−r)/dr = −1
        u_pred, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))

        h_expand = h
        while h_expand.ndim < x_start.ndim:
            h_expand = h_expand.unsqueeze(-1)

        # MeanFlow-Identity target (START-anchored): u = v + h·du/dr ; stop-gradient.
        u_target = v_inst + h_expand * du_dr
        u_target = apply_conditioning(u_target, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)
        u_target = u_target.detach()

        # weighted squared error (keep the trajectory/action loss_weights) + adaptive per-sample weight
        delta = u_pred - u_target
        sq = delta.pow(2) * self.loss_fn.weights          # [B,H,D]; weights buffer is on-device
        reduce_dims = tuple(range(1, sq.ndim))
        per_sample = sq.mean(dim=reduce_dims)             # [B]
        w = 1.0 / (per_sample.detach() + self.meanflow_adaptive_c).pow(self.meanflow_adaptive_p)
        main_loss = (w * per_sample).mean()

        total_loss = main_loss
        info = {}
        # aux v-head stabilizer on the instantaneous velocity (discarded at sampling).
        # With dual_head=True this v-head SHARES the backbone, so this term now actually
        # regularizes the trunk that produces u (U5 Phase 1b).
        if self.meanflow_aux_weight > 0:
            _u2, aux_pred = self._predict_uv(x_r, cond, r, h=h, returns=returns,
                                             omega=omega_c, t_min=t_min_c, t_max=t_max_c)
            aux_loss = F.mse_loss(aux_pred, v_inst)
            total_loss = total_loss + self.meanflow_aux_weight * aux_loss
            info['aux_loss'] = aux_loss

        a0_loss = delta[:, 0, :self.action_dim].pow(2).mean() if self.action_dim > 0 \
            else torch.tensor(0.0, device=x_start.device)
        info['diffusion_loss'] = main_loss
        info['a0_loss'] = a0_loss
        info['total_loss'] = total_loss
        info['u_weight'] = torch.tensor(1.0, device=x_start.device)
        info['v_weight'] = torch.tensor(self.meanflow_aux_weight, device=x_start.device)
        return total_loss, info

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
                    'Error(s) in loading state_dict for iMeanFlowODE:\n'
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
