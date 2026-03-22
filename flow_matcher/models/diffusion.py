import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, goal_dim=0, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1.0, 
        loss_discount=1.0, loss_weights=None, returns_condition=False, condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        # TODO: Implement Flow Matching logic here
        # (Replaces: diffusion beta/alpha schedule construction and cumulative products.)
        # Compatibility buffers are initialized so non-diffusion code can still access expected attributes.
        betas = torch.zeros(n_timesteps, dtype=torch.float32)
        alphas_cumprod = torch.ones(n_timesteps, dtype=torch.float32)
        alphas_cumprod_prev = torch.ones(n_timesteps, dtype=torch.float32)

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # TODO: Implement Flow Matching logic here
        # (Replaces: diffusion q(x_t|x_0), reciprocal alpha transforms, and posterior q(x_{t-1}|x_t,x_0) buffers.)
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
        # TODO: Implement Flow Matching logic here
        # (Replaces: epsilon-to-x0 inversion using diffusion reciprocal alpha coefficients.)
        raise NotImplementedError("Flow Matching not yet implemented")

    def q_posterior(self, x_start, x_t, t):
        # TODO: Implement Flow Matching logic here
        # (Replaces: closed-form DDPM posterior mean/variance computation.)
        raise NotImplementedError("Flow Matching not yet implemented")

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        # if self.model.calc_energy:
        #     assert self.predict_epsilon
        #     x = torch.tensor(x, requires_grad=True)
        #     t = torch.tensor(t, dtype=torch.float, requires_grad=True)
        #     returns = torch.tensor(returns, requires_grad=True)

        # TODO: Implement Flow Matching logic here
        # (Replaces: classifier-free guided epsilon prediction, x0 reconstruction, posterior moments, and projection gradient merge.)
        raise NotImplementedError("Flow Matching not yet implemented")

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, projector=None, constraints=None):
        # TODO: Implement Flow Matching logic here
        # (Replaces: single reverse DDPM sampling step with variance-scaled Gaussian noise.)
        raise NotImplementedError("Flow Matching not yet implemented")

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, return_diffusion=False, projector=None, constraints=None, repeat_last=0):
        # TODO: Implement Flow Matching logic here
        # (Replaces: full reverse-time denoising loop with optional projection during DDPM trajectory generation.)
        raise NotImplementedError("Flow Matching not yet implemented")

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
        # TODO: Implement Flow Matching logic here
        # (Replaces: gradient-based reverse DDPM sampling step.)
        raise NotImplementedError("Flow Matching not yet implemented")

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        # TODO: Implement Flow Matching logic here
        # (Replaces: iterative gradient-driven reverse diffusion trajectory generation.)
        raise NotImplementedError("Flow Matching not yet implemented")

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
        # TODO: Implement Flow Matching logic here
        # (Replaces: forward noising process q(x_t|x_0) for diffusion training.)
        raise NotImplementedError("Flow Matching not yet implemented")

    def p_losses(self, x_start, cond, t, returns=None):
        # TODO: Implement Flow Matching logic here
        # (Replaces: diffusion objective that predicts noise/x0 on noised trajectories.)
        raise NotImplementedError("Flow Matching not yet implemented")

    def loss(self, x, cond, returns=None):
        # TODO: Implement Flow Matching logic here
        # (Replaces: random diffusion timestep sampling and diffusion loss dispatch.)
        raise NotImplementedError("Flow Matching not yet implemented")

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)