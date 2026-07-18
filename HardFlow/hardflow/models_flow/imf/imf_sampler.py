"""Gen13 — iMF K-step sampler (exact interval jumps), HF convention.

Port of aux imeanflow torch-branch imf.py sample_one_step()/generate() into
HardFlow's convention and conditioning scheme. Per step i of K:

    tau_i = i / K,   dt = 1 / K
    x <- x + dt * u(x, tau_i, dt)          # exact jump over [tau_i, tau_i + dt]

This is the textbook MeanFlow composition validated in the Gen3v4 K2 analysis
(§7: anchor at the interval start, width = dt). K=1 is the paper's one-step
generation; K=2 its refined mode. Conditioned entries are handled exactly like
HardFlow's ConditionedODESolver: the update is zeroed at conditioned indices
(conditions were already baked into x at init via apply_conditioning).
"""

import torch

from hardflow.models_flow.flow_matcher import apply_conditioning_from_conditioned_x


@torch.no_grad()
def imf_sample(model, x, conditions, action_dim, k_steps, return_chain=False):
    """
    Args:
        model: TemporalImfUnet — (x, tau, h) -> (u, v).
        x: (batch, horizon, transition_dim) initial noise WITH conditions applied.
        conditions: {timestep: state} dict (normalized), for update masking.
        action_dim: action dimension.
        k_steps: K — the number of function evaluations (NFE = K).
    Returns:
        x_final  or  (x_final, x_chain (batch, K+1, horizon, transition)).
    """
    assert k_steps >= 1
    dt = 1.0 / k_steps
    x_chain = [x.clone()]

    for i in range(k_steps):
        tau = i * dt
        u, _ = model(x, tau, dt)
        u = apply_conditioning_from_conditioned_x(
            u, torch.zeros_like(x), conditions, action_dim
        )
        x = x + u * dt
        x = x.detach()
        x_chain.append(x.clone())

    if return_chain:
        return x, torch.stack(x_chain, dim=1).detach()
    return x
