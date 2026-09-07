"""α-Flow diffusion adapter (Gen3v7) with FMv3ODE-compatible training/sampling APIs.

Gen3v7 is a copy-modify sibling of Gen3v6 (`flow_matcher_v3_meanflow`). It implements
**α-Flow** (arXiv 2510.20771, snap-research/alphaflow @ b0fef77), which replaces MeanFlow's
JVP target with a **self-bootstrapped, no-grad** target and anneals a scalar `α: 1 → 0`:

    | generation                    | regression target for u                        |
    |-------------------------------|------------------------------------------------|
    | Gen3v4 iMF  (`imf_official`)  | V-form, guided v_g, PREDICTED v_c JVP tangent   |
    | Gen3v6 MeanFlow               | u ← sg(v + h·du/dr)   (JVP, ANALYTIC v)         |
    | Gen3v7 α-Flow ← THIS FILE     | u ← sg(α·v + (1−α)·u_next)  (bootstrapped)      |

Why it exists (PLAN §0): the MeanFlow residual only sees `δ_u − h·δ_D`, so any error with
`δ_u = h·δ_D` is **invisible to the loss** while the sampler uses `u` alone — worst exactly
as `h → 1`, which is where 1–2-NFE sampling lives. A *fixed* (no-grad) target has no such
blind direction: the loss measures `u` directly.

Two endpoints of the homotopy, both proved/checked by the gates:
  α = 1  ⇒  `u_tgt = v` exactly            ⇒ the objective IS flow matching (gate G1)
  α = 0  ⇒  the JVP branch, byte-identical to Gen3v6's `_p_losses_meanflow` (gate G2;
            first-order equivalence derived in PLAN §3.4)

🔴 CONVENTION (PLAN §2). α-Flow uses NOISE-AT-1 (`t=1` noise) and calls the query point `t`.
We use DATA-AT-1 (`τ=1` data) and call the query point `r`. The mapping `τ = 1 − t` is done
ONCE, here, and the code below uses ONLY our names:

    theirs:  x_t, t (query), t_next (destination), dt, step x -= (t−t_next)·u
    ours:    z_r, r (query), t      (destination), dt, step z += h·u,   h = t − r

Naming a variable `t` while meaning α-Flow's `t` is the single most likely bug in this
generation. Do not do it.

See logs_in_develop/Gen3v7_AlphaFlow/init/PLAN_Gen3v7_alphaflow.md.
"""

from collections import OrderedDict
import math
from typing import Dict, Optional, Tuple
import warnings

import torch
from torch import nn

try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None

from .af_engine import AlphaFlowEngine
from .helpers import apply_conditioning, Losses


class AlphaFlowODE(nn.Module):
    """α-Flow wrapper that preserves FM-PCC/FMv3ODE diffusion interfaces."""

    def __init__(
        self,
        model: AlphaFlowEngine,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        goal_dim: int = 0,
        n_timesteps: int = 1000,
        loss_type: str = "l2",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        # ── FIX-3: action_weight / loss_discount are KEPT (utils + folder naming read them)
        #    but are deliberately NOT applied to the α-Flow loss. They are a DPCC idea with
        #    no counterpart in the MeanFlow family and they distort the gradient geometry of
        #    the identity. DO NOT "fix" this back — inherited from Gen3v6 PLAN §3.2 FIX-3.
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
        # ── Gen3v7 α-Flow objective knobs (PLAN §5.2) ─────────────────────────────────
        # α schedule — upstream `configs/loss/alphaflow.yaml:alpha` +
        # `infra/experiments/experiments-alphaflow.yaml:155`, RESCALED to our budget.
        af_alpha_scheduler: str = 'sigmoid',   # 'sigmoid' | 'linear' | 'exponential' | 'log'
                                               #  | 'constant' | 'step'
        af_alpha_init: float = 1.0,            # α at step 0        (1.0 ⇒ start as pure FM)
        af_alpha_end: float = 0.0,             # α at the end       (0.0 ⇒ end as MeanFlow)
        af_alpha_init_step: int = 0,
        af_alpha_end_step: int = 100000,       # 🔴 MUST equal n_train_steps — see the assert
                                               #    below and PLAN §11 trap 1. Upstream's
                                               #    400000 would leave α≈1 for our whole run.
        af_alpha_gamma: float = 25.0,          # sigmoid sharpness
        af_alpha_clamp: float = 0.005,         # snap to exactly 0.0 / 1.0 near the ends
        af_ratio_fm: float = 0.5,              # fraction of the batch forced to r==t (h=0)
        af_clamp_utgt: float = 4.0,            # upstream `clamp_utgt`; no prior generation
                                               # in this repo clamps its target
        af_adp_eps: float = 1e-3,              # ⚠️ α-Flow's `adaptive_loss_weight_eps`.
                                               # DELIBERATELY ≠ MeanFlow/iMF's 0.01 — different
                                               # method, different constant. Do NOT "harmonise"
                                               # it (PLAN §11 trap 7). The exponent p is fixed
                                               # at 1 upstream, so there is no `p` knob here.
        af_n_train_steps: Optional[int] = None,  # only for the end_step assert; not used in math
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

        # ── Gen3v7 α-Flow objective ───────────────────────────────────────────────────
        self.af_alpha_scheduler = str(af_alpha_scheduler)
        self.af_alpha_init = float(af_alpha_init)
        self.af_alpha_end = float(af_alpha_end)
        self.af_alpha_init_step = int(af_alpha_init_step)
        self.af_alpha_end_step = int(af_alpha_end_step)
        self.af_alpha_gamma = float(af_alpha_gamma)
        self.af_alpha_clamp = float(af_alpha_clamp)
        self.af_ratio_fm = float(af_ratio_fm)
        self.af_clamp_utgt = float(af_clamp_utgt)
        self.af_adp_eps = float(af_adp_eps)

        # 🔴 PLAN §11 trap 1 — THE #1 SILENT FAILURE OF THIS GENERATION.
        # Upstream anneals over 400 k steps; we train 100 k. Copying their number verbatim
        # leaves α pinned at ~1.0 for the entire run, i.e. you trained plain flow matching
        # for 100 k steps and called it α-Flow — and nothing in the logs would say so.
        if af_n_train_steps is not None and self.af_alpha_scheduler not in ('constant', 'step'):
            if int(af_n_train_steps) != self.af_alpha_end_step:
                raise ValueError(
                    f"af_alpha_end_step={self.af_alpha_end_step} != n_train_steps="
                    f"{int(af_n_train_steps)}. The α anneal must span the ACTUAL budget "
                    "(PLAN §3.6 / §11 trap 1). Set af_alpha_end_step = n_train_steps in "
                    "config/avoiding-d3il.py, or pass af_n_train_steps=None to bypass "
                    "deliberately (e.g. a gate that forces a constant α)."
                )

        # Global optimizer step, pushed in by the Trainer via set_train_step() before every
        # loss() call (PLAN §3.7). Never read the inner gradient-accumulation counter: with
        # gradient_accumulate_every=2 that would halve the effective schedule length.
        self._train_step = 0

        # Kept for backward compatibility with existing configs / utils.
        self.loss_schedule = loss_schedule
        self.warmup_epochs = int(warmup_epochs)
        self.transition_epochs = int(transition_epochs)
        self.u_mix = 1.0
        self.v_mix = 1.0

        # NOTE (FIX-3): loss_fn is still constructed — `loss_fn.weights` is part of the
        # state_dict and several utils/serialization paths expect it — but the α-Flow
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

        Gen3v7 has NO interval-CFG (α-Flow's own headline config is non-cfg, ω=1/κ=0): the
        (ω, t_min, t_max) net inputs are never fed, so the
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
        # DPCC returns-CFG output mix. Inert in Gen3v7 (condition_guidance_w = 0.0); kept so
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

        total_steps = flow_steps + int(repeat_last)
        dt = 1.0 / max(flow_steps, 1)
        h_batch = torch.full((batch_size,), dt, device=device, dtype=torch.float32)

        # ⚠️ SAMPLER — DO NOT TOUCH. The interval-jump update x += dt·u with h = dt = 1/N is
        # already faithful (U10 audit F1, re-verified); iMF, MeanFlow and α-Flow all share
        # the identical sampler. ✅ α is TRAINING-ONLY — it does not appear here, which is
        # exactly what makes the three-way comparison clean (PLAN §5.3).
        # This is the unconditional path — Gen3v7 has no CFG special-case.
        #
        # U8 — torchdiffeq dispatch. No flow_steps floor: every sub-stage query computes h
        # dynamically (see ode_rhs below) as the remaining distance to the current macro step's
        # own declared end t1, so every query stays inside the (t,h) domain the model was
        # actually trained on (t,h∈[0,1], t+h<=1 — see _p_losses_alphaflow's query point
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
        """Trainer entrypoint matching FM-PCC's expected `model.loss(*batch)` contract.

        Inherited from Gen3v6: the (t, r) pair comes from two independent logit-normals
        sampled inside the loss, so there is no pre-sampled single `t` and no `p_losses()`
        hop. α-Flow adds the step-dependent α on top (see set_train_step).
        """
        return self._p_losses_alphaflow(x, cond, returns=returns)

    # ──────────────────────────────────────────────────────────────────────────────────
    # α schedule (PLAN §3.6)
    # ──────────────────────────────────────────────────────────────────────────────────

    def set_train_step(self, step: int):
        """Push the global OPTIMIZER step in from the Trainer (PLAN §3.7).

        🔴 Must be the optimizer-step counter (`Trainer.step`), NOT the inner
        gradient-accumulation loop index. With gradient_accumulate_every=2 the inner loop
        runs twice per optimizer step; using it would halve the effective schedule length.

        Resume safety: `Trainer.load()` restores `self.step` before training resumes and
        `train_epoch` calls this on every iteration, so α picks up where it left off. If α
        restarted at 1.0 on every requeue the model would unlearn (PLAN §11 trap 6).
        """
        self._train_step = int(step)

    @staticmethod
    def _get_ratio(scheduler, initial_value, end_value,
                   init_step, end_step, gamma, clamp_value, cur_step) -> float:
        """Port of α-Flow's `AlphaFlowLoss.get_ratio` (src/training/loss.py:390-427).

        A staticmethod on purpose: the train script prints the whole α curve before building
        anything, and `gates_alphaflow.py` checks the schedule without a model.

        The `clamp_value` snap is NOT cosmetic: without it α becomes a tiny-but-nonzero
        number and EVERY sample takes the discrete branch with dt ≈ 0, i.e. `u_next`
        evaluated essentially at the query point — a degenerate near-identity target. The
        snap routes those samples to the exact JVP branch instead (PLAN §3.6 / §11 trap 4).
        """
        if scheduler == 'constant':
            ratio = initial_value
        elif scheduler == 'step':
            if init_step != end_step:
                raise ValueError("af_alpha_scheduler='step' requires init_step == end_step")
            ratio = initial_value if cur_step < init_step else end_value
        elif scheduler in ('linear', 'exponential', 'log', 'sigmoid'):
            if cur_step < init_step:
                ratio = initial_value
            elif cur_step > end_step:
                ratio = end_value
            else:
                span = max(end_step - init_step, 1)
                if scheduler == 'sigmoid':
                    middle = init_step + (end_step - init_step) / 2.0
                    progress = (cur_step - middle) / span          # centred, ∈ [−0.5, 0.5]
                    ratio = initial_value + (end_value - initial_value) * (
                        1.0 / (1.0 + math.exp(-progress * gamma)))
                else:
                    progress = (cur_step - init_step) / span
                    if scheduler == 'linear':
                        ratio = initial_value + (end_value - initial_value) * progress
                    elif scheduler == 'exponential':
                        ratio = initial_value * ((end_value / initial_value) ** (progress ** gamma))
                    else:  # log
                        log_progress = math.log(1 + progress * 9) / math.log(10)
                        ratio = initial_value + (end_value - initial_value) * log_progress
        else:
            raise NotImplementedError(f"Unknown α scheduler: {scheduler}")

        if ratio < clamp_value:
            ratio = 0.0
        elif ratio > 1.0 - clamp_value:
            ratio = 1.0
        return float(ratio)

    def current_alpha(self) -> float:
        """α at the current global step. 1.0 ⇒ pure FM, 0.0 ⇒ MeanFlow."""
        return self._get_ratio(
            self.af_alpha_scheduler, self.af_alpha_init, self.af_alpha_end,
            self.af_alpha_init_step, self.af_alpha_end_step,
            self.af_alpha_gamma, self.af_alpha_clamp, self._train_step,
        )

    def _sample_tau_pair(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Two i.i.d. draws on the τ axis (DATA-AT-1), shape [2, B]. Gen3v6 verbatim.

        Matches α-Flow's own `distrib_t_t_next_mf: {type: minmax, logit_norm(-0.4, 1.0)}`
        (configs/loss/alphaflow.yaml:16) as well as MeanFlow's — the two upstreams agree
        here, so nothing had to change.

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
        """α-Flow adaptive L2: err / sg(err + eps), eps = `af_adp_eps` (upstream 1e-3).

        Upstream (`AlphaFlowLoss.forward`, loss.py:589-593):
            loss_unscaled = ((pred − tgt)**2).flatten(1).mean(1)
            weight        = w_branch / (loss_unscaled.detach() + adaptive_loss_weight_eps)
        There is no exponent knob — upstream's `p` is fixed at 1.

        ⚠️ REDUCTION DISCREPANCY, stated on purpose. `err` here is the per-sample **SUM**
        over (H, D), inherited from Gen3v6/official-imeanflow; upstream α-Flow reduces with
        **MEAN**. That is an H·D = 8·6 = 48× rescale of `err`, so upstream's eps=1e-3 sits at
        a different point relative to the error scale than ours does. PLAN §3.5 specifies SUM
        **and** eps=1e-3, and SUM is what keeps the Gen3v6 A/B controlled, so that is what is
        implemented. Practical consequence: with SUM, `err ≫ eps` almost always, so the
        adaptive weight is ≈1 and this term is near-inert — exactly as in Gen3v6, where
        `diffusion_loss` is pinned at its ceiling by construction. **Read `raw_mse_u`, never
        `diffusion_loss`.** (To match upstream's balance instead, either switch to MEAN or
        scale eps by H·D ⇒ 0.048; do not do it silently.)
        """
        return err / (err + self.af_adp_eps).detach()

    def compute_u_target(self, x_r, r, h, v_inst, cond, alpha, returns=None):
        """The α-Flow regression target for u. Returns `(u_target, clamp_frac)`.

        Factored out of the loss so `gates_alphaflow.py` can interrogate it directly:
        G1 (α=1 ⇒ u_tgt == v bitwise), G2 (α=0 ⇒ equals Gen3v6's JVP target), G3
        (first-order agreement at small α) and G5 (`requires_grad is False`) are all
        statements about THIS function, and a gate that re-implemented it would prove
        nothing.

        Always returns a **detached** tensor — see G5 / PLAN §11 trap 2.
        """
        try:
            from torch.func import jvp as _jvp
        except ImportError:  # older torch
            from functorch import jvp as _jvp

        ad, gd = self.action_dim, self.goal_dim
        clamp_frac = torch.zeros((), device=x_r.device)

        h_exp = h
        while h_exp.ndim < x_r.ndim:
            h_exp = h_exp.unsqueeze(-1)

        if alpha <= 0.0:
            # ══════════════════════════════════════════════════════════════════════════
            # CONTINUOUS BRANCH (α = 0) — Gen3v6's `_p_losses_meanflow` body, UNMODIFIED.
            # Do not re-derive it (PLAN §3.2). Run on the FULL batch: at h = 0 the target
            # collapses to v_inst on its own, which is exactly the FM-anchor target, so no
            # masking is needed and the α→0 limit stays bit-identical to Gen3v6.
            #
            # 🔴 DO NOT CHANGE THE z-TANGENT. `v_inst` is the ANALYTIC velocity x1 − x0.
            # Feeding a PREDICTED v_c here turns this branch into Gen3v4-iMF and destroys
            # the three-way A/B (Gen3v4 / Gen3v6 / Gen3v7).
            # ══════════════════════════════════════════════════════════════════════════
            def _u_of(z_in, r_in, h_in):
                u, _v = self._predict_uv(z_in, cond, r_in, h=h_in, returns=returns)
                return u

            ones = torch.ones_like(r)
            # 🔴 NOT wrapped in torch.no_grad(). Forward-mode AD shares the GradMode guard
            # with reverse mode in several torch versions, so a jvp inside no_grad can come
            # back with null/zero tangents — which would silently degrade this branch to
            # `u_tgt = v` (plain FM) and quietly break G2. Gen3v6 computes the JVP with grad
            # enabled and detaches the RESULT; that is the path this lineage has actually
            # run, so it is the path kept here. The graph is discarded immediately by the
            # .detach() at the end of this function (gate G5 checks that).
            _u_primal, du_dr = _jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
            u_target = (v_inst + h_exp * du_dr).detach()
        else:
            # ══════════════════════════════════════════════════════════════════════════
            # DISCRETE / BOOTSTRAPPED BRANCH (α > 0) — THE NEW THING IN THIS GENERATION.
            # ══════════════════════════════════════════════════════════════════════════
            if alpha >= 1.0:
                # α = 1 ⇒ dt = h ⇒ u_tgt = v exactly. Short-circuit: computing
                # (h·v + 0·u_next)/h instead would be v only up to float round-off, and G1
                # demands BITWISE equality. Also saves a whole forward pass. (Upstream does
                # the same via its `isclose(1 − dt/(t−t_next), 0)` guard, loss.py:536.)
                # 🔴 .clone() is REQUIRED: apply_conditioning() below writes IN PLACE
                # (models/helpers.py:161). Aliasing v_inst here would (a) mutate the
                # caller's v and (b) make gate G1's `torch.equal(u_tgt, v)` tautological
                # by comparing an object with itself.
                u_target = v_inst.clone()
            else:
                dt = alpha * h                                    # [B]
                dt_exp = dt
                while dt_exp.ndim < x_r.ndim:
                    dt_exp = dt_exp.unsqueeze(-1)

                # step toward DATA by dt at velocity v. The conditioned dims of v_inst are
                # pinned to 0, so z_shift keeps x_r's conditioning untouched by construction.
                z_shift = x_r + dt_exp * v_inst

                # 🔴 no_grad: this is what makes the target a FIXED tensor (gate G5).
                # DEVIATION from upstream, deliberate: α-Flow splits the batch with boolean
                # masks and runs each branch on its own sub-batch. We run the FULL batch and
                # select with torch.where instead — the h==0 FM anchors ride along in this
                # forward and are discarded below. That costs ≤ af_ratio_fm of one extra
                # forward per step on a small DiT, and it buys us not having to index `cond`
                # (a dict of per-timestep tensors) by mask, which is where a subtle
                # conditioning bug would live. Both queries stay in-domain: r+dt ≤ t ≤ 1 and
                # h−dt ≥ 0 always.
                with torch.no_grad():
                    u_next, _v_next = self._predict_uv(
                        z_shift, cond, r + dt, h=h - dt, returns=returns)

                    # (dt·v + (h−dt)·u_next) / h  ==  α·v + (1−α)·u_next.
                    # h_safe only guards the h==0 FM anchors, which the torch.where below
                    # overwrites anyway; it exists so no NaN is ever created in the first
                    # place.
                    h_safe = h_exp.clamp(min=1e-12)
                    u_target = (dt_exp * v_inst + (h_exp - dt_exp) * u_next) / h_safe

                    # Upstream clamps ONLY this branch (loss.py:542) — the FM anchors and
                    # the JVP target are left alone. Kept faithful.
                    u_clamped = u_target.clamp(-self.af_clamp_utgt, self.af_clamp_utgt)
                    # measured over the DISCRETE rows only — the h==0 rows below are
                    # discarded, so counting them would dilute the diagnostic by ratio_fm.
                    disc = (h_exp > 0).expand_as(u_target)
                    clamp_frac = (((u_clamped != u_target) & disc).float().sum()
                                  / disc.float().sum().clamp(min=1.0))
                    u_target = u_clamped

                    # FM anchors (h == 0) keep the plain FM target.
                    u_target = torch.where(h_exp > 0, u_target, v_inst)

        u_target = apply_conditioning(u_target, cond, ad, goal_dim=gd, noise=True).detach()
        return u_target, clamp_frac

    def _p_losses_alphaflow(
        self,
        x_start: torch.Tensor,
        cond: Dict,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """α-Flow (2510.20771) objective: a homotopy from flow matching to MeanFlow.

        DATA-AT-1 convention (τ=0 noise, τ=1 data). The sampler anchors at the noise-side
        point z_r at time r and steps to t = r + h, so u(z_r, r, h) is the average velocity
        over [r, t] and the exact endpoint map is x̂_t = z_r + h·u.

        Three-way batch routing (PLAN §3.1; upstream `sample_traj_params` + `forward`):

          | branch              | when                | target                             |
          |---------------------|---------------------|------------------------------------|
          | FM anchor           | h == 0 (r forced=t) | u_tgt = v                          |
          | discrete/bootstrap  | h > 0 and α > 0     | u_tgt = α·v + (1−α)·u_next  ⭐      |
          | continuous (JVP)    | h > 0 and α == 0    | u_tgt = v + h·du/dr  (= Gen3v6)    |

        ⭐ The discrete target, transported from upstream `_compute_mean_velocity_d`
        (loss.py:531-543) into our convention with dt = α·h:

            z_shift = z_r + dt·v                       # travel dt toward data at velocity v
            u_next  = u(z_shift, r+dt, h−dt)           # under no_grad — a FIXED tensor
            u_tgt   = (dt·v + (h−dt)·u_next) / h  ==  α·v + (1−α)·u_next

        i.e. a one-step interval-composition identity: traverse [r, r+dt] at the
        instantaneous velocity v, then [r+dt, r+h] at the model's own average velocity, and
        average by arc length. It needs NO derivative of the network — which is the point:
        a fixed target has no blind direction, so unlike the MeanFlow residual it cannot hide
        an error with δ_u = h·δ_D (COMPARE §8.2).

        🔴 `u_next` MUST be under torch.no_grad(). If gradient flows into it, the target
        becomes self-referential again and the entire generation is void (gate G5).

        Endpoints (the two gates that prove the homotopy is wired correctly):
          α = 1 ⇒ dt = h ⇒ u_tgt = v EXACTLY. Short-circuited below so it is bitwise-exact
                  and costs no forward pass (upstream does the same via its
                  `isclose(1 − dt/(t−t_next), 0)` guard).                        [G1]
          α = 0 ⇒ dt = 0 ⇒ the JVP branch, character-for-character Gen3v6's
                  `_p_losses_meanflow`. First-order expansion of u_next recovers the
                  MeanFlow identity u = v + h·D_tot — PLAN §3.4.                 [G2]
        """
        device = x_start.device
        B = x_start.shape[0]
        ad, gd = self.action_dim, self.goal_dim

        alpha = self.current_alpha()

        # ── (t, r) from two INDEPENDENT draws on the τ axis (Gen3v6 FIX-1, kept) ─────────
        # 🔴 RNG ORDER IS PART OF THE CONTRACT: _sample_tau_pair → rand(fm_mask) →
        # randn_like(noise), identical to Gen3v6. gates_alphaflow.py's G2 reseeds both
        # models with the same seed and relies on this to compare like with like.
        taus = self._sample_tau_pair(B, device)
        t = torch.maximum(taus[0], taus[1])      # data-side end
        r = torch.minimum(taus[0], taus[1])      # noise-side anchor = the network's query point
        fm_mask = torch.rand(B, device=device) < self.af_ratio_fm
        r = torch.where(fm_mask, t, r)           # FM anchors: h=0 ⇒ u_target = v_inst
        h = t - r                                # ≥ 0

        # noise (DATA-AT-1 τ=0 side), pinned to 0 at conditioned dims
        x_base = torch.randn_like(x_start)
        x_base = apply_conditioning(x_base, cond, ad, goal_dim=gd, noise=True)

        # anchor point z_r at time r (noise side) — matches the sampler's query convention
        x_r = self.q_sample(x_start=x_start, t=r, noise=x_base)
        x_r = apply_conditioning(x_r, cond, ad, goal_dim=gd)

        # instantaneous (FM) velocity v = x_data − noise, pinned to 0 at conditioned dims.
        # This is the identity's v(z_r, r), the bootstrap's transport velocity, AND the
        # z-tangent for the α=0 JVP branch.
        v_inst = x_start - x_base
        v_inst = apply_conditioning(v_inst, cond, ad, goal_dim=gd, noise=True)

        mf_mask = h > 0                                   # not an FM anchor
        discrete_mask = mf_mask & (alpha > 0.0)           # took the bootstrapped branch

        u_target, clamp_frac = self.compute_u_target(
            x_r, r, h, v_inst, cond, alpha, returns=returns)

        # ── Prediction: ONE forward, both heads, at the sampler's own query point ────────
        # Upstream also predicts with a separate forward rather than reusing a JVP primal
        # (loss.py:579). Here it additionally gives the v head for free at the same query
        # point, so this is one forward CHEAPER than Gen3v6's JVP-primal + second-forward
        # arrangement. The DiT arm has no stochastic dropout (its `dropout_rate` only feeds
        # the deterministic null-class CFG token, and force_dropout=False here), so the
        # separate forward is numerically identical to reusing a primal — which is what
        # makes gate G2's comparison against Gen3v6 exact rather than approximate.
        u_pred, v_pred = self._predict_uv(x_r, cond, r, h=h, returns=returns)

        # ── Loss: adaptive L2, per-sample SUM; discrete samples weighted by α ────────────
        # Upstream: weight_c = 1 for continuous/FM-anchor samples, weight_d = α for the
        # bootstrapped ones (loss.py:590-593). As α → 0 the discrete branch is switched off
        # continuously rather than abruptly — the discrete samples stop existing (α snaps to
        # 0 and everything routes to the JVP) at the same time as their weight vanishes.
        reduce_dims = tuple(range(1, u_pred.ndim))
        err_u = (u_pred - u_target).pow(2).sum(dim=reduce_dims)     # [B]
        err_v = (v_pred - v_inst.detach()).pow(2).sum(dim=reduce_dims)
        w_br = torch.where(discrete_mask,
                           torch.full_like(err_u, alpha), torch.ones_like(err_u))
        loss = (w_br * self._adaptive(err_u) + self._adaptive(err_v)).mean()

        # ── Metrics. NEVER read `diffusion_loss` as convergence: the adaptive loss is
        # pinned at its ceiling by construction (COMPARE §7.1). Read raw_mse_u.
        info = self._build_info(
            loss, err_u, err_v, (u_pred - u_target).detach(), h, fm_mask, x_start,
            alpha=alpha, discrete_mask=discrete_mask, clamp_frac=clamp_frac)
        return loss, info

    def _build_info(self, loss, err_u, err_v, delta_u, h, fm_mask, x_start,
                    alpha=0.0, discrete_mask=None, clamp_frac=None) -> Dict:
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
                # ⭐ Gen3v7 schedule telemetry (PLAN §3.7 / gate G4). A run whose α never
                # moved is otherwise INDISTINGUISHABLE from a working one — that is the
                # #1 silent failure of this generation, so these three are not optional.
                'alpha': torch.tensor(float(alpha), device=device),
                'discrete_frac': (discrete_mask.float().mean() if discrete_mask is not None
                                  else torch.zeros((), device=device)),
                # fraction of target ELEMENTS hit by af_clamp_utgt; 0 in the JVP branch
                # (upstream clamps only the bootstrapped target). A rising clamp_frac means
                # the bootstrap is diverging before the α-anneal reaches it.
                'clamp_frac': (clamp_frac.detach() if torch.is_tensor(clamp_frac)
                               else torch.zeros((), device=device)),
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
                    'Error(s) in loading state_dict for AlphaFlowODE:\n'
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
