from collections import namedtuple
import torch
import time
import einops
import numpy as np
import diffuser.utils as utils
from diffuser.models.helpers import apply_conditioning
from diffuser.utils.arrays import to_device
from diffuser.datasets.preprocessing import get_policy_preprocess_fn

Trajectories = namedtuple('Trajectories', 'actions observations')


class Policy:

    def __init__(self, model, normalizer, scheduler=None, preprocess_fns=[], test_ret=0, projector=None, 
                 trajectory_selection='random', **sample_kwargs):
        self.model = model
        self.scheduler = scheduler,   # 'DDPM' or 'DDIM'
        self.normalizer = normalizer
        self.action_dim = model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.test_ret = test_ret
        self.sample_kwargs = sample_kwargs

        self.inverse_dynamics = False

        # Projector
        self.projector = projector

        # Trajectory selection
        self.trajectory_selection = trajectory_selection        # 'random' or 'temporal_consistency' or 'minimum_projection_cost'

        # Previous observations
        self.prev_observations = None

        # Real-time logging: last-call diagnostics (set every __call__; read by eval rollout)
        self.last_infos = {}            # raw infos from the model (projection_ms, projection_costs, ...)
        self.last_proj_ms = 0.0         # CPU projection wall-time for the last sample (ms)
        self.last_proj_cost = 0.0       # projection cost of the SELECTED trajectory (0.0 if no projector)
        self.last_which_trajectory = 0  # index of the executed candidate in the batch

    def __call__(self, conditions, batch_size=1, horizon=16, test_ret=None, constraints=None, disable_projection=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), 'cuda')

        # Engine-agnostic (Gen15): `self.model` is FlowMatchingODE | MeanFlowODE | AlphaFlowODE.
        # All three expose forward(cond, returns=, projector=, constraints=, horizon=) -> (x, infos).
        # `sample_kwargs` carries per-engine extras — notably `num_steps=K` for the two-time
        # engines, which FlowMatchingODE does NOT accept (see models/engine_registry.py).
        projector = self.projector if not disable_projection else None
        samples, infos = self.model(conditions, returns=returns, projector=projector, constraints=constraints, horizon=horizon, **self.sample_kwargs)

        trajectories = utils.to_np(samples)

        ## extract observations [ batch_size x horizon x observation_dim ]
        if not 'diffusion' in infos:
            normed_observations = trajectories[:, :, self.action_dim:]
            observations = self.normalizer.unnormalize(normed_observations, 'observations')
        if 'diffusion' in infos:
            diffusion_trajectories = utils.to_np(infos['diffusion'])         # Shape: batch_size x T x horizon x transition_dim     
            observations = self.normalizer.unnormalize(diffusion_trajectories[:, :, :, self.action_dim:], 'observations')
        
        # Sort according to similarity with previous observations
        # Gen15 G1 — `fix_5` grafted VERBATIM from flow_matcher_v3_meanflow/sampling/policies.py.
        # `which_trajectory` always indexes the ORIGINAL (unsorted) batch, because `actions` is
        # never reordered (see the `actions[which_trajectory, 0]` below). The temporal-consistency
        # branch DOES reorder `observations`, so indexing it with `which_trajectory` picks a
        # different candidate than the one being executed. `executed_idx` is the index of the
        # executed plan *within `observations` as it stands here* — 0 after the sort,
        # `which_trajectory` otherwise. This restores the invariant MPC_NPZ_PATCH intended:
        # prev_observations is the plan that was actually executed.
        # ⚠️ This is a BEHAVIOUR CHANGE vs Gen11 on `dpcc-t*` (temporal_consistency) variants,
        # applied identically to all three arms so the comparison stays internally consistent.
        # Parity gate G1 is therefore asserted on `diffuser` + `dpcc-c` only. See PLAN §5 G1.
        if self.trajectory_selection == 'temporal_consistency' and not disable_projection and self.prev_observations is not None:   # Temporal consistency
            order = np.argsort(np.linalg.norm(observations[:,:-1,:] - self.prev_observations[:,1:,:], axis=(1,2)))
            which_trajectory = order[0]
            observations = observations[order]
            executed_idx = 0                                                                                                        # observations is now sorted; best match sits at 0
        elif self.trajectory_selection == 'minimum_projection_cost' and not disable_projection:                                     # Minimum projection cost
            costs_total = np.zeros(batch_size)
            for timestep, cost in infos['projection_costs'].items():
                costs_total += cost
            which_trajectory = np.argmin(costs_total)
            executed_idx = which_trajectory                                                                                         # observations not reordered
        else:                                                                                                                       # Random selection
            which_trajectory = 0
            executed_idx = 0
        self.prev_observations = np.repeat(np.expand_dims(observations[executed_idx], axis=0), batch_size, axis=0)  # MPC_NPZ_PATCH + fix_5

        ## Extract or calculate action
        if self.inverse_dynamics:
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*samples.shape[-1])
            actions = self.inv_model(obs_comb)
            actions = utils.to_np(actions)
            actions = self.normalizer.unnormalize(actions, 'actions')
            action = actions[which_trajectory]
        else:
            ## extract action [ batch_size x horizon x action_dim ]
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[which_trajectory, 0]

        trajectories = Trajectories(actions, observations)

        # Real-time logging diagnostics for the eval rollout (no signature change; read via attrs).
        self.last_infos = infos
        self.last_proj_ms = float(infos.get('projection_ms', 0.0))
        self.last_which_trajectory = int(which_trajectory)
        proj_costs = infos.get('projection_costs', {})       # {fm_step: per-batch cost array}
        if proj_costs:
            total = np.zeros(batch_size)
            for _, cost in proj_costs.items():
                total += cost
            self.last_proj_cost = float(total[which_trajectory])
        else:
            self.last_proj_cost = 0.0

        return action, trajectories

    @property
    def device(self):
        parameters = list(self.model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cpu')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    