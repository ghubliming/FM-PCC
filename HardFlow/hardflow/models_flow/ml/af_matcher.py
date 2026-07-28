"""Gen13 U11 — AfMatcher: the α-Flow (Gen3v7) objective, HardFlow convention.

Ported from `flow_matcher_v3_alphaflow/models/af_diffusion.py`
(`_p_losses_alphaflow` + `compute_u_target` + `_get_ratio`) into HardFlow's native
convention + state-inpainting conditioning, mirroring `imf/imf_matcher.py`'s
interface so it drops into the shared TemporalImfUnet + ImfFlowPolicy stack unchanged.

α-Flow (arXiv 2510.20771) replaces MeanFlow's JVP target with a SELF-BOOTSTRAPPED,
no-grad target and anneals a scalar α: 1 → 0 over training:

    | branch              | when                | u_target                           |
    |---------------------|---------------------|------------------------------------|
    | FM anchor           | h == 0              | v_target                           |
    | discrete/bootstrap  | h > 0 and α > 0     | α·v + (1−α)·u_next        ⭐        |
    | continuous (JVP)    | h > 0 and α == 0    | v + h·du/dr   (== Gen3v6 MeanFlow)  |

    α = 1 ⇒ pure flow matching (u_target = v_target, bitwise).
    α = 0 ⇒ the MeanFlow JVP branch (analytic-v tangent — identical to MfMatcher).

The bootstrap needs NO derivative of the network: a fixed target has no blind
direction, which is the whole α-Flow thesis (COMPARE §8.2). `u_next` is under
torch.no_grad — if gradient flowed into it the target would be self-referential and
the generation void (Gen3v7 gate G5).

Convention: HardFlow and Gen3v7 are both DATA-AT-1 (tau=0 noise, tau=1 data) ⇒ no
sign flip (see `imf/convention.py`). The one HardFlow-idiom adaptation: the bootstrap
query point `z_shift` is re-pinned with apply_conditioning (Gen3v7 instead zeroes
v_inst at conditioned dims; the two are equivalent — both keep the network query's
conditioned state fixed).

Judge convergence on `raw_mse_u`; the adaptive `loss` is flat by construction. The
α telemetry (`alpha`, `discrete_frac`, `clamp_frac`) is NOT optional — a run whose α
never moved is otherwise indistinguishable from a working one (Gen3v7 gate G4).
"""

import math

import torch

from hardflow.models_flow.flow_matcher import apply_conditioning

from ..imf.convention import pad_t_like_x, sample_tau_h


class AfMatcher:

    def __init__(
        self,
        model,
        action_dim,
        p_mean=-0.4,
        p_std=1.4,
        ratio_fm=0.5,
        adp_eps=1e-3,
        clamp_utgt=4.0,
        alpha_scheduler="sigmoid",
        alpha_init=1.0,
        alpha_end=0.0,
        alpha_init_step=0,
        alpha_end_step=100000,
        alpha_gamma=25.0,
        alpha_clamp=0.005,
        n_train_steps=None,
    ):
        """
        Args mirror Gen3v7's α knobs (PLAN §6.3). `ratio_fm` is α-Flow's FM-anchor
        fraction (its analogue of iMF/MF `data_proportion`, default 0.5).

        🔴 The α anneal MUST span the ACTUAL training budget: `alpha_end_step` has to
        equal `n_train_steps` (Gen3v7 trap 1). Enforced below.
        """
        self.model = model
        self.action_dim = action_dim
        self.p_mean = p_mean
        self.p_std = p_std
        self.ratio_fm = ratio_fm
        self.adp_eps = adp_eps
        self.clamp_utgt = clamp_utgt

        self.alpha_scheduler = str(alpha_scheduler)
        self.alpha_init = float(alpha_init)
        self.alpha_end = float(alpha_end)
        self.alpha_init_step = int(alpha_init_step)
        self.alpha_end_step = int(alpha_end_step)
        self.alpha_gamma = float(alpha_gamma)
        self.alpha_clamp = float(alpha_clamp)

        if (
            n_train_steps is not None
            and self.alpha_scheduler not in ("constant", "step")
            and int(alpha_end_step) != int(n_train_steps)
        ):
            raise ValueError(
                f"af_alpha_end_step={alpha_end_step} != n_train_steps={n_train_steps}. "
                "The α anneal must span the ACTUAL budget (Gen3v7 trap 1). Set "
                "af_alpha_end_step = n_train_steps (train_ml.sh does this automatically)."
            )

        # pushed in from the training loop each step (set_step); drives current_alpha()
        self._train_step = 0

    # ── α schedule ───────────────────────────────────────────────────────────────

    def set_step(self, step):
        """Push the global optimizer step in from the trainer (drives the α anneal).

        Resume safety: the trainer restores its step counter before resuming and calls
        this every iteration, so α picks up where it left off (Gen3v7 trap 6).
        """
        self._train_step = int(step)

    @staticmethod
    def _get_ratio(scheduler, initial_value, end_value,
                   init_step, end_step, gamma, clamp_value, cur_step):
        """Port of α-Flow's `AlphaFlowLoss.get_ratio` (verbatim from Gen3v7).

        The `clamp_value` snap is NOT cosmetic: without it α becomes tiny-but-nonzero
        and every sample takes the discrete branch with dt≈0 (a degenerate near-identity
        target). The snap routes those to the exact JVP branch instead.
        """
        if scheduler == 'constant':
            ratio = initial_value
        elif scheduler == 'step':
            if init_step != end_step:
                raise ValueError("alpha_scheduler='step' requires init_step == end_step")
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

    def current_alpha(self):
        """α at the current global step. 1.0 ⇒ pure FM, 0.0 ⇒ MeanFlow."""
        return self._get_ratio(
            self.alpha_scheduler, self.alpha_init, self.alpha_end,
            self.alpha_init_step, self.alpha_end_step,
            self.alpha_gamma, self.alpha_clamp, self._train_step,
        )

    # ── loss ─────────────────────────────────────────────────────────────────────

    def _adp(self, err):
        """α-Flow adaptive L2: err / sg(err + eps) (upstream p fixed at 1)."""
        return err / (err + self.adp_eps).detach()

    def _compute_u_target(self, z, tau, h, v_target, cond, alpha):
        """The α-Flow regression target for u. Returns (u_target, clamp_frac), detached.

        α ≤ 0 : Gen3v6 JVP branch (analytic-v tangent — identical to MfMatcher).
        α ≥ 1 : pure FM (u_target = v_target, bitwise).
        else  : bootstrap  u_target = α·v + (1−α)·u_next, u_next under no_grad.
        """
        clamp_frac = torch.zeros((), device=z.device)
        h_exp = pad_t_like_x(h, z)

        if alpha <= 0.0:
            # ── CONTINUOUS BRANCH (α=0) == Gen3v6 MeanFlow ───────────────────────────
            # 🔴 analytic v_target is the z-tangent (NOT a predicted v_c). See mf_matcher.
            def u_fn(z_, tau_, h_):
                u_, _v = self.model(z_, tau_, h_)
                return u_

            ones = torch.ones_like(tau)
            _u_primal, du_dr = torch.func.jvp(u_fn, (z, tau, h), (v_target, ones, -ones))
            u_target = (v_target + h_exp * du_dr).detach()

        elif alpha >= 1.0:
            # α=1 ⇒ pure FM. .clone() so the caller's v_target is never aliased/mutated.
            u_target = v_target.clone().detach()

        else:
            # ── DISCRETE / BOOTSTRAPPED BRANCH (0 < α < 1) — the α-Flow contribution ──
            dt = alpha * h                                   # [B]
            dt_exp = pad_t_like_x(dt, z)

            # travel dt toward DATA at velocity v, then RE-PIN the conditioned state so
            # the bootstrap query keeps HardFlow's inpainting fixed (Gen3v7 zeroes v at
            # conditioned dims to the same effect).
            z_shift = z + dt_exp * v_target
            z_shift = apply_conditioning(z_shift, cond, self.action_dim)

            with torch.no_grad():                            # 🔴 fixed target (gate G5)
                u_next, _v_next = self.model(z_shift, tau + dt, h - dt)

                # (dt·v + (h−dt)·u_next) / h  ==  α·v + (1−α)·u_next.
                h_safe = h_exp.clamp(min=1e-12)              # guards only h==0 anchors
                u_target = (dt_exp * v_target + (h_exp - dt_exp) * u_next) / h_safe

                # clamp ONLY the bootstrapped target (upstream leaves JVP/FM alone).
                u_clamped = u_target.clamp(-self.clamp_utgt, self.clamp_utgt)
                disc = (h_exp > 0).expand_as(u_target)
                clamp_frac = (((u_clamped != u_target) & disc).float().sum()
                              / disc.float().sum().clamp(min=1.0))
                u_target = u_clamped

                # FM anchors (h == 0) keep the plain FM target.
                u_target = torch.where(h_exp > 0, u_target, v_target)

            u_target = u_target.detach()

        return u_target, clamp_frac

    def loss(self, x, cond):
        """
        x: (batch_size, planning_horizon, transition_dim), normalized data traj.
        cond: {timestep: state_value} state-inpainting conditions.
        """
        x1 = x
        x0 = torch.randn_like(x1)
        bz = x1.shape[0]

        # ratio_fm is α-Flow's FM-anchor fraction (its `data_proportion`).
        tau, h, fm_mask = sample_tau_h(
            bz, self.p_mean, self.p_std, self.ratio_fm, x1.device
        )

        z = pad_t_like_x(tau, x1) * x1 + pad_t_like_x(1.0 - tau, x1) * x0
        v_target = x1 - x0
        z = apply_conditioning(z, cond, self.action_dim)

        alpha = self.current_alpha()
        mf_mask = h > 0                                   # not an FM anchor
        discrete_mask = mf_mask & (alpha > 0.0)           # took the bootstrapped branch

        u_target, clamp_frac = self._compute_u_target(z, tau, h, v_target, cond, alpha)

        # one forward, both heads, at the sampler's own query point
        u_pred, v_pred = self.model(z, tau, h)

        err_u = ((u_pred - u_target) ** 2).sum(dim=(1, 2))
        err_v = ((v_pred - v_target.detach()) ** 2).sum(dim=(1, 2))
        # discrete samples weighted by α (upstream weight_d = α, weight_c = 1)
        w_br = torch.where(
            discrete_mask, torch.full_like(err_u, float(alpha)), torch.ones_like(err_u)
        )
        loss = (w_br * self._adp(err_u) + self._adp(err_v)).mean()

        infos = {
            "loss": loss.item(),                    # adaptive — flat by design
            "raw_mse_u": err_u.mean().item(),       # judge convergence on THESE
            "raw_mse_v": err_v.mean().item(),
            "a0_mse": ((u_pred - u_target)[:, 0, : self.action_dim] ** 2).mean().item(),
            "fm_frac": fm_mask.float().mean().item(),
            "h_mean": h.mean().item(),
            # ⭐ Gen3v7 schedule telemetry — NOT optional (gate G4).
            "alpha": float(alpha),
            "discrete_frac": discrete_mask.float().mean().item(),
            "clamp_frac": float(clamp_frac),
        }
        return loss, infos
