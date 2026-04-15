import time
import warnings
import torch
from torch import nn

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None

import diffuser.utils as utils
from .helpers import (
    apply_conditioning,
    Losses,
)

# ODE solver configuration contract (read from config/avoiding-d3il.py):
# - ode_solver_backend_v3:
#     * "legacy_euler" -> use the built-in explicit Euler step (default/fallback)
#     * "torchdiffeq"  -> use torchdiffeq.odeint when torchdiffeq is installed
# - ode_solver_method_v3 and optional tolerances/step-size are config-driven.

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, goal_dim=0, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
        loss_discount=1.0, loss_weights=None, returns_condition=False, condition_guidance_w=0.1,
        time_beta_alpha_v3=1.5, time_beta_beta_v3=1.0,
        flow_steps_v3=None, ode_inference_steps_v3=None,
        ode_solver_backend_v3='legacy_euler', ode_solver_method_v3='euler',
        ode_solver_rtol_v3=None, ode_solver_atol_v3=None, ode_solver_step_size_v3=None):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        # Flow Matching uses continuous time; buffers are retained for interface compatibility.
        betas = torch.linspace(1.0, 0.0, n_timesteps, dtype=torch.float32)
        alphas_cumprod = torch.ones(n_timesteps, dtype=torch.float32)
        alphas_cumprod_prev = torch.ones(n_timesteps, dtype=torch.float32)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.time_beta_alpha_v3 = float(time_beta_alpha_v3)
        self.time_beta_beta_v3 = float(time_beta_beta_v3)
        resolved_flow_steps = flow_steps_v3 if flow_steps_v3 is not None else ode_inference_steps_v3
        if resolved_flow_steps is None:
            resolved_flow_steps = n_timesteps
        self.flow_steps_v3 = int(resolved_flow_steps)
        # Keep this alias for compatibility with older call sites.
        self.ode_inference_steps_v3 = int(self.flow_steps_v3)
        self.ode_solver_backend_v3 = str(ode_solver_backend_v3)
        self.ode_solver_method_v3 = str(ode_solver_method_v3)
        self.ode_solver_rtol_v3 = ode_solver_rtol_v3
        self.ode_solver_atol_v3 = ode_solver_atol_v3
        self.ode_solver_step_size_v3 = ode_solver_step_size_v3

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Flow Matching does not use diffusion posterior buffers; keep zero tensors for compatibility.
        zeros = torch.zeros(n_timesteps, dtype=torch.float32)
        self.register_buffer('sqrt_alphas_cumprod', zeros)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', zeros)
        self.register_buffer('log_one_minus_alphas_cumprod', zeros)
        self.register_buffer('sqrt_recip_alphas_cumprod', zeros)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', zeros)
        self.register_buffer('posterior_variance', zeros)
        self.register_buffer('posterior_log_variance_clipped', zeros)
        self.register_buffer('posterior_mean_coef1', zeros)
        self.register_buffer('posterior_mean_coef2', zeros)

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def _time_from_timestep(self, t):
        if t.is_floating_point():
            return t.clamp(0.0, 1.0)
        t_float = t.float()
        denom = max(self.n_timesteps - 1, 1)
        return (t_float / denom).clamp_(0.0, 1.0)

    def _predict_velocity(self, x, cond, t, returns=None):
        if self.returns_condition:
            v_cond = self.model(x, cond, t, returns, use_dropout=False)
            v_uncond = self.model(x, cond, t, returns, force_dropout=True)
            return v_uncond + self.condition_guidance_w * (v_cond - v_uncond)
        return self.model(x, cond, t)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # In Flow Matching with linear interpolation, x_t = x_0 + t * v where v = x_1 - x_0.
        t_cont = self._time_from_timestep(t)
        while t_cont.ndim < x_t.ndim:
            t_cont = t_cont.unsqueeze(-1)
        return x_t - t_cont * noise

    def q_posterior(self, x_start, x_t, t):
        # Flow Matching has deterministic dynamics under the ODE sampler.
        zeros = torch.zeros_like(x_t)
        return x_start, zeros, zeros

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        # if self.model.calc_energy:
        #     assert self.predict_epsilon
        #     x = torch.tensor(x, requires_grad=True)
        #     t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        #     returns = torch.tensor(returns, requires_grad=True)

        velocity = self._predict_velocity(x, cond, t, returns=returns)
        dt = 1.0 / max(self.flow_steps_v3, 1)
        # FIX: + not -, integrate forward along the flow
        model_mean = x + velocity * dt

        if projector is not None and projector.gradient:
            if self.goal_dim > 0:
                grad = projector.compute_gradient(model_mean[:, :, :-self.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(model_mean, constraints)
            model_mean = model_mean + grad

        zeros = torch.zeros_like(x)
        return model_mean, zeros, zeros

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, projector=None, constraints=None):
        model_mean, _, _ = self.p_mean_variance(
            x=x,
            cond=cond,
            t=t,
            returns=returns,
            projector=projector,
            constraints=constraints,
        )
        return model_mean

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion:
            diffusion = [x]
        costs = {}

        use_torchdiffeq = self.ode_solver_backend_v3 == 'torchdiffeq'
        if use_torchdiffeq and torchdiffeq_odeint is None:
            raise RuntimeError(
                "ode_solver_backend_v3='torchdiffeq' but torchdiffeq is not installed. "
                "Install torchdiffeq or switch backend to 'legacy_euler'."
            )

        # Forward integration t=0 → t=T (noise → data)
        total_steps = self.flow_steps_v3 + repeat_last
        for i in range(total_steps):
            loop_idx = min(i, self.flow_steps_v3 - 1)   # clamp for repeat_last extra steps
            t_cont = torch.full((batch_size,), loop_idx / max(self.flow_steps_v3, 1), device=device, dtype=torch.float32)
            dt = 1.0 / max(self.flow_steps_v3, 1)

            # Apply projector near the END of integration (near t=1, near data)
            near_end = loop_idx >= (1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3 \
                       if projector is not None else False

            if use_torchdiffeq:
                t0 = float(loop_idx) * dt
                t1 = t0 + dt
                t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)

                def ode_rhs(t_scalar, state):
                    t_batch = torch.ones(batch_size, device=device, dtype=torch.float32) * t_scalar
                    return self._predict_velocity(state, cond, t_batch, returns=returns)

                odeint_kwargs = {
                    'method': self.ode_solver_method_v3,
                }
                if self.ode_solver_rtol_v3 is not None:
                    odeint_kwargs['rtol'] = float(self.ode_solver_rtol_v3)
                if self.ode_solver_atol_v3 is not None:
                    odeint_kwargs['atol'] = float(self.ode_solver_atol_v3)
                if self.ode_solver_step_size_v3 is not None:
                    fixed_step_methods = {
                        'euler', 'midpoint', 'heun2', 'heun3', 'rk4',
                        'explicit_adams', 'fixed_adams'
                    }
                    if self.ode_solver_method_v3 in fixed_step_methods:
                        odeint_kwargs['options'] = {'step_size': float(self.ode_solver_step_size_v3)}
                    else:
                        warnings.warn(
                            f"Ignoring ode_solver_step_size_v3 for method '{self.ode_solver_method_v3}' "
                            "because it is not a fixed-step solver method.",
                            RuntimeWarning,
                        )

                x = torchdiffeq_odeint(
                    ode_rhs,
                    x,
                    t_span,
                    **odeint_kwargs,
                )[-1]

                if projector is not None and projector.gradient and near_end:
                    if self.goal_dim > 0:
                        grad = projector.compute_gradient(x[:, :, :-self.goal_dim], constraints)
                    else:
                        grad = projector.compute_gradient(x, constraints)
                    x = x + grad
            else:
                if projector is not None and projector.gradient and near_end:
                    x = self.p_sample(x, cond, t_cont, returns, projector=projector, constraints=constraints)
                else:
                    x = self.p_sample(x, cond, t_cont, returns)

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if projector is not None and not projector.gradient and near_end:
                if self.goal_dim > 0:
                    x[:, :, :-self.goal_dim], projection_costs = projector.project(x[:, :, :-self.goal_dim], constraints)
                    costs[i] = projection_costs
                else:
                    x, projection_costs = projector.project(x, constraints)
                    costs[i] = projection_costs

            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

        infos = {}
        if return_diffusion:
            infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = costs

        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        return self.p_sample(x, cond, t, returns=returns)

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        if return_diffusion:
            diffusion = [x]

        # FIX: forward integration t=0 → t=T (noise → data)
        for i in range(0, self.flow_steps_v3):
            t_cont = torch.full((batch_size,), i / max(self.flow_steps_v3, 1), device=device, dtype=torch.float32)
            x = self.grad_p_sample(x, cond, t_cont, returns)
            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # Linear interpolation path: x_t = (1 - t) * x_0_base + t * x_1_data.
        t_cont = self._time_from_timestep(t)
        while t_cont.ndim < x_start.ndim:
            t_cont = t_cont.unsqueeze(-1)
        return (1.0 - t_cont) * noise + t_cont * x_start

    def p_losses(self, x_start, cond, t, returns=None):
        x_base = torch.randn_like(x_start)
        x_base = apply_conditioning(x_base, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        x_t = self.q_sample(x_start=x_start, t=t, noise=x_base)
        x_t = apply_conditioning(x_t, cond, self.action_dim, goal_dim=self.goal_dim)

        v_target = x_start - x_base
        v_target = apply_conditioning(v_target, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        v_pred = self._predict_velocity(x_t, cond, t, returns=returns)
        if not self.predict_epsilon:
            v_pred = apply_conditioning(v_pred, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        loss, info = self.loss_fn(v_pred, v_target)
        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
        beta_dist = torch.distributions.Beta(alpha, beta)
        t = beta_dist.sample((batch_size,))
        t = 1.0 - t
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)