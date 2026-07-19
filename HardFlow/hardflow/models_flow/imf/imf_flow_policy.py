"""Gen13 — ImfFlowPolicy: HardFlow's constrained sampler on an iMF backbone.

Additive subclass of hardflow.models_flow.flow_policy.FlowPolicy (imported,
never modified). Two guidance methods:

  * "original_imf"      — unguided K-step iMF sampling (K = cfg.ode_t_steps;
                          K=1/2 is the paper regime). Mirrors the base
                          "original" branch line-for-line, sampler swapped.
  * "hardflow_new_imf"  — HardFlow's prox-NLP + pull-back loop with THE SEAM
                          swapped (Gen13 plan §2, BLEND §2, THEORY Thm 1):

        FM  (base):   x1_hat = z_ref + (1 - tau) * v(z_ref, tau)   # Euler shot
        iMF (here):   x1_hat = z_ref + (1 - tau) * u(z_ref, tau, 1 - tau)  # exact

    and the reference step taken as an exact interval jump
        z_ref = z + dt * u(z, tau, dt)
    instead of the Euler step z + dt * v(z, tau). The CasADi NLP, obstacle/
    dynamics constraints, and the tau-weighted pull-back are UNCHANGED
    (Gen13 decision D8, Level 1).

The FM guidance methods of the base class are deliberately NOT reachable with
an iMF model (different forward signature) — dispatch raises on them.

NFE accounting: info carries nfe_warmstart / nfe_sampling / nfe_diag and total.
computation_time mirrors the base timing exactly (warmstart excluded, x1
diagnostics included) so numbers are comparable with the frozen FM baselines.
"""

import time

import numpy as np
import torch

from hardflow import utils
from hardflow.models_flow.flow_matcher import (
    apply_conditioning,
    apply_conditioning_from_conditioned_x,
)
from hardflow.models_flow.flow_policy import FlowPolicy, Trajectories, to_np
from hardflow.utils.arrays import to_torch

from .imf_sampler import imf_sample


class ImfFlowPolicy(FlowPolicy):
    """FlowPolicy whose flow_model is a TemporalImfUnet: (x, tau, h) -> (u, v)."""

    IMF_GUIDANCE_METHODS = ("original_imf", "hardflow_new_imf")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nfe = {"warmstart": 0, "sampling": 0, "diag": 0}
        # fix_4: NLP health counters. With IPOPT's own output silenced
        # (solver_print_level=0), these preserve the ONLY signal that block
        # carried: whether each prox-NLP actually solved. Cumulative across the
        # episode (NOT reset per plan) so the per-episode summary is meaningful.
        self._nlp_solves = 0
        self._nlp_failures = 0

    # ------------------------------------------------------------------ utils

    def reset_nlp_stats(self):
        """Zero the cumulative NLP counters (call at episode start)."""
        self._nlp_solves = 0
        self._nlp_failures = 0

    def nlp_stats(self):
        return {"nlp_solves": self._nlp_solves, "nlp_failures": self._nlp_failures}

    def _reset_nfe(self):
        self._nfe = {"warmstart": 0, "sampling": 0, "diag": 0}

    def _nfe_info(self):
        total = sum(self._nfe.values())
        return {
            "nfe_warmstart": self._nfe["warmstart"],
            "nfe_sampling": self._nfe["sampling"],
            "nfe_diag": self._nfe["diag"],
            "nfe_total": total,
        }

    def _u(self, x, tau, h, bucket="sampling"):
        """Single u-field evaluation with NFE accounting."""
        u, _ = self.flow_model(x, tau, h)
        self._nfe[bucket] += 1
        return u

    # ------------------------------------------------------------- dispatch

    def __call__(self, conditions, batch_size=1, verbose=True):
        if self.cfg.guidance_method == "original_imf":
            return self.imf_original_forward(conditions, batch_size)
        elif self.cfg.guidance_method == "hardflow_new_imf":
            return self.imf_hardflow_new_forward(conditions, batch_size)
        raise ValueError(
            f"ImfFlowPolicy supports only {self.IMF_GUIDANCE_METHODS}, got "
            f"'{self.cfg.guidance_method}'. FM guidance methods need the FM "
            f"backbone (run/eval.py)."
        )

    # ------------------------------------------------- overridden primitives

    def x1_estimate(self, x_chain, conditions):
        """Terminal prediction along a chain — iMF exact endpoint per state.

        Same signature/semantics as the base (diagnostic x1 chain), with the
        Euler shot x + (1-t) * v replaced by the exact x + (1-t) * u(x, t, 1-t).
        """
        batch_size, n_steps, horizon, transition_dim = x_chain.shape
        x1_estimation = torch.zeros_like(x_chain)

        for k in range(n_steps):
            t = k / (n_steps - 1)
            current_x = x_chain[:, k, :, :]

            with torch.no_grad():
                current_u = self._u(current_x, t, 1.0 - t, bucket="diag")
            current_u = apply_conditioning_from_conditioned_x(
                current_u, torch.zeros_like(current_u), conditions, self.action_dim
            )

            x1_estimation[:, k, :, :] = current_x + (1.0 - t) * current_u

        return x1_estimation

    def warmstart(self, conditions):
        """Base warmstart with Euler v-steps replaced by exact u-jumps (h=dt)."""
        x = torch.randn(
            self.cfg.warmstart_batch,
            self.horizon,
            self.action_dim + self.state_dim,
            device=self.cfg.device,
        )
        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )
        x_chain = [x.clone()]
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t = k / self.oc_N_steps
            with torch.no_grad():
                u = self._u(x, t, dt, bucket="warmstart")
            u = apply_conditioning_from_conditioned_x(
                u, torch.zeros_like(x), conditions, self.action_dim
            )
            x = x + u * dt
            x_chain.append(x)

        values = self.value_model(x_chain[-1])
        best_idx = torch.argmax(values, dim=0)

        best_x_chain = torch.stack(
            [x_step[best_idx].squeeze(0) for x_step in x_chain], dim=0
        )
        best_x_chain = best_x_chain.reshape(self.oc_N_steps + 1, -1)
        best_s0 = best_x_chain[0, self.action_dim : self.state_dim + self.action_dim]
        best_dof_chain_part_1 = best_x_chain[:, : self.action_dim]
        best_dof_chain_part_2 = best_x_chain[:, self.action_dim + self.state_dim :]
        best_dof_chain = torch.cat(
            [best_dof_chain_part_1, best_dof_chain_part_2], dim=-1
        )
        return to_np(best_s0), to_np(best_dof_chain)

    def hardflow_formulate(self, *args, **kwargs):
        """Reuse the base CasADi formulation (purely algebraic — no net inside).

        The base method asserts guidance_method in (hardflow, hardflow_new);
        shim the name around the super() call, restore afterwards. No base
        code is modified.
        """
        original_method = self.cfg.guidance_method
        assert original_method == "hardflow_new_imf", (
            f"hardflow_formulate on ImfFlowPolicy expects hardflow_new_imf, "
            f"got {original_method}"
        )
        self.cfg.guidance_method = "hardflow_new"
        try:
            super().hardflow_formulate(*args, **kwargs)
        finally:
            self.cfg.guidance_method = original_method

    def constrained_u_fn_torch(self, dof, t, h, s0):
        """dof-space u-field evaluation (mirror of base constrained_flow_fn_torch)."""
        device = dof.device
        assert dof.shape == (1, self.oc_dof)

        full_flat = torch.cat(
            [dof[:, : self.action_dim], s0, dof[:, self.action_dim :]], dim=1
        )
        full_traj = full_flat.view(1, self.horizon, self.action_dim + self.state_dim)

        u = self._u(full_traj, t, h, bucket="sampling")
        u_flat = u.reshape(1, -1)

        dof_u = torch.cat(
            [
                u_flat[:, : self.action_dim],
                u_flat[:, self.action_dim + self.state_dim :],
            ],
            dim=1,
        )
        return dof_u

    # ------------------------------------------------------ forward variants

    def imf_original_forward(self, conditions, batch_size=1):
        """Unguided K-step iMF sampling; mirrors the base 'original' branch."""
        assert batch_size == 1, "batch_size must be 1 for no guidance"
        self._reset_nfe()

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        x = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim + self.state_dim,
            device=self.cfg.device,
        )
        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )

        k_steps = self.cfg.ode_t_steps  # K == ode_t_steps (K=1/2: paper regime)
        x, x_chain = imf_sample(
            self.flow_model,
            x,
            to_torch(conditions, device=x.device),
            self.action_dim,
            k_steps,
            return_chain=True,
        )
        self._nfe["sampling"] += k_steps

        normed_actions = x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(
            to_np(normed_observations), "observations"
        )

        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)

        trajectories = Trajectories(actions, observations, values)
        action = actions[0, 0]

        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)
        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        info = {"computation_time": t_end - t_start}
        info.update(self._nfe_info())
        return action, trajectories, x_chain, x1_estimation, info

    def imf_hardflow_new_forward(self, conditions, batch_size=1):
        """hardflow_new with the iMF seam (adapted copy of the base method).

        Differences vs base hardflow_new_forward — ONLY these:
          1. reference step: exact u-jump  (base: Euler v-step)
          2. terminal prediction: exact u-endpoint  (base: Euler v-shot)
          3. NFE accounting.
        Warmstart, NLP, pull-back, bookkeeping, timing: identical.
        """
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "hardflow_new_imf"
        ), f"guidance_method must be hardflow_new_imf, got {self.cfg.guidance_method}"
        self._reset_nfe()

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        best_s0_np, best_dof_chain_np = self.warmstart(conditions)

        t_start = time.time()

        device = self.cfg.device
        best_s0_torch = to_torch(best_s0_np, device=device).reshape(1, self.state_dim)

        def u_eval_np(x_np, t, h):
            x_torch = to_torch(x_np, device=device).reshape(1, self.oc_dof)
            with torch.no_grad():
                u = self.constrained_u_fn_torch(x_torch, t, h, best_s0_torch)
            return to_np(u).reshape(-1)

        self.oc_cs_opti.set_value(self.oc_s0_param, best_s0_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]

            # (1) SEAM: exact interval jump replaces the Euler reference step
            u_k = u_eval_np(x_k, t_k, dt)
            x_next_ref = x_k + u_k * dt

            control_flag = True
            if self.cfg.hardflow_activation == "all":
                pass
            elif self.cfg.hardflow_activation == "late":
                if k < self.oc_N_steps // 2:
                    control_flag = False
            else:
                raise ValueError(
                    f"Unsupported hardflow_activation: {self.cfg.hardflow_activation}"
                )

            if control_flag:
                # (2) SEAM: exact u-endpoint replaces the Euler terminal shot
                t_next = t_k + dt
                h_terminal = 1.0 - t_next
                u_terminal = u_eval_np(x_next_ref, t_next, h_terminal)
                x_terminal_predicted_ref = x_next_ref + h_terminal * u_terminal

                self.oc_cs_opti.set_value(self.oc_t_param, t_next)
                self.oc_cs_opti.set_value(
                    self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref
                )
                self.oc_cs_opti.set_initial(
                    self.oc_X_terminal_predicted, x_terminal_predicted_ref
                )

                self._nlp_solves += 1
                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)
                except RuntimeError:
                    # fix_4: keep this WARNING loud even with IPOPT silenced —
                    # a failed projection is the one thing that can silently
                    # explain a constraint violation downstream.
                    self._nlp_failures += 1
                    print(
                        "[ eval_imf ] WARNING: NLP solve failed "
                        f"(step k={k}) — using last available value.",
                        flush=True,
                    )
                    x_terminal_predicted = self.oc_cs_opti.debug.value(
                        self.oc_X_terminal_predicted
                    )

                x_next = x_next_ref + t_next * (
                    x_terminal_predicted - x_terminal_predicted_ref
                )
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_ctrl = (x_next - x_next_ref) / dt
            U_optimized.append(u_ctrl)
            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        self.value_model.previous_sample = to_np(optimized_final_x.cpu())

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)
        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)
        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)
        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=self.cfg.device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)
        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()

        info = {"computation_time": t_end - t_start}
        info.update(self._nfe_info())
        return action, trajectories, x_chain, x1_estimation, info
