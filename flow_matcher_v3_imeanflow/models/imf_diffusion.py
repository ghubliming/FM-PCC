"""iMeanFlow diffusion adapter with FMv3ODE-compatible training and sampling APIs."""

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .imf_engine import iMeanFlowEngine
from .helpers import apply_conditioning, Losses


class iMFDiffusion(nn.Module):
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
        time_beta_alpha_v3: float = 1.5,
        time_beta_beta_v3: float = 1.0,
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
        resolved_flow_steps = flow_steps_v3 if flow_steps_v3 is not None else ode_inference_steps_v3
        self.flow_steps_v3 = int(resolved_flow_steps)
        self.ode_inference_steps_v3 = int(self.flow_steps_v3)
        self.ode_solver_backend_v3 = str(ode_solver_backend_v3)
        self.ode_solver_method_v3 = str(ode_solver_method_v3)
        self.ode_solver_rtol_v3 = ode_solver_rtol_v3
        self.ode_solver_atol_v3 = ode_solver_atol_v3
        self.ode_solver_step_size_v3 = ode_solver_step_size_v3

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

    def _predict_uv(self, x, cond, t, returns=None):
        # Returns-conditioning is intentionally ignored here because
        # iMeanFlowEngine does not model classifier-free guidance.
        return self.model.forward_train(x, t, cond)

    def _predict_velocity(self, x, cond, t, returns=None):
        velocity, aux = self._predict_uv(x, cond, t, returns=returns)
        return velocity + self.sample_aux_weight * aux

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
    ):
        device = self.betas.device
        batch_size = shape[0]
        x = 0.5 * torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

        diffusion = [x] if return_diffusion else None

        total_steps = self.flow_steps_v3 + int(repeat_last)
        for i in range(total_steps):
            loop_idx = min(i, self.flow_steps_v3 - 1)
            t_cont = torch.full(
                (batch_size,),
                loop_idx / max(self.flow_steps_v3, 1),
                device=device,
                dtype=torch.float32,
            )
            velocity = self._predict_velocity(x, cond, t_cont, returns=returns)
            dt = 1.0 / max(self.flow_steps_v3, 1)
            x = x + velocity * dt
            x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if projector is not None:
                snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
                near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
                if near_end and projector.gradient:
                    if self.goal_dim > 0:
                        grad = projector.compute_gradient(x[:, :, :-self.goal_dim], constraints)
                    else:
                        grad = projector.compute_gradient(x, constraints)
                    x = x + grad

                if near_end and not projector.gradient:
                    if self.goal_dim > 0:
                        x[:, :, :-self.goal_dim], _ = projector.project(x[:, :, :-self.goal_dim], constraints)
                    else:
                        x, _ = projector.project(x, constraints)

                x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)

            if return_diffusion:
                diffusion.append(x)

        infos = {}
        if return_diffusion:
            infos['diffusion'] = torch.stack(diffusion, dim=1)
        infos['projection_costs'] = {}
        return x, infos

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, returns=returns, *args, **kwargs)
    
    def sample(
        self,
        batch_size: int,
        returns: Optional[torch.Tensor] = None,
        conditions: Optional[Dict] = None,
        returns_condition: Optional[bool] = None,
        guidance_weight: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        # Keep compatibility with the existing eval script API.
        if num_steps is not None:
            self.flow_steps_v3 = int(num_steps)
            self.ode_inference_steps_v3 = int(num_steps)

        if conditions is None:
            cond = {0: torch.zeros(batch_size, self.observation_dim, device=self.betas.device)}
        else:
            cond = conditions

        sampled, _ = self.conditional_sample(cond=cond, returns=returns, horizon=self.horizon)
        return sampled

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Trainer entrypoint matching FM-PCC's expected `model.loss(*batch)` contract."""
        batch_size = x.shape[0]
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
        x_base = torch.randn_like(x_start)
        x_base = apply_conditioning(x_base, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        x_t = self.q_sample(x_start=x_start, t=t, noise=x_base)
        x_t = apply_conditioning(x_t, cond, self.action_dim, goal_dim=self.goal_dim)

        v_target = x_start - x_base
        v_target = apply_conditioning(v_target, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        velocity_pred, aux_pred = self._predict_uv(x_t, cond, t, returns=returns)
        if not self.predict_epsilon:
            velocity_pred = apply_conditioning(velocity_pred, cond, self.action_dim, goal_dim=self.goal_dim, noise=True)

        main_loss, info = self.loss_fn(velocity_pred, v_target)
        aux_loss = F.mse_loss(aux_pred, torch.zeros_like(aux_pred))
        total_loss = main_loss + self.aux_loss_weight * aux_loss

        info['diffusion_loss'] = main_loss
        info['a0_loss'] = info.get('a0_loss', torch.tensor(0.0, device=x_start.device))
        info['aux_loss'] = aux_loss
        info['u_weight'] = torch.tensor(self.u_mix, device=x_start.device)
        info['v_weight'] = torch.tensor(self.v_mix, device=x_start.device)
        info['total_loss'] = total_loss

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
                    'Error(s) in loading state_dict for iMFDiffusion:\n'
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
