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

        # ── H8+8 (U10) — receding-horizon cadence ──────────────────────────────────────
        # How many actions of each plan the CALLER executes before asking for a new one.
        # The policy itself does not loop; it needs this for exactly one thing — the
        # temporal-consistency shift below, which assumes the previous plan advanced by
        # this many steps. DEFAULT 1 == the historic behaviour (replan every env step).
        # The eval sets it (MF_REPLAN_STEPS); nothing else touches it.
        self.replan_steps = 1
        # The FULL action/observation sequence of the plan that was actually executed
        # (i.e. of candidate `which_trajectory`, after any -c/-t selection). Published
        # because `which_trajectory` is not visible to the caller, so a replan>1 caller
        # cannot otherwise know WHICH candidate's remaining actions to replay.
        self.last_executed_actions = None
        self.last_executed_observations = None

    def __call__(self, conditions, batch_size=1, horizon=16, test_ret=None, constraints=None, disable_projection=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), 'cuda')

        # Use FlowMatchingIMF model
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
        # fix_5: `which_trajectory` always indexes the ORIGINAL (unsorted) batch, because
        # `actions` is never reordered (see the `actions[which_trajectory, 0]` below). The
        # temporal-consistency branch DOES reorder `observations`, so indexing it with
        # `which_trajectory` picks a different candidate than the one being executed.
        # `executed_idx` is the index of the executed plan *within `observations` as it
        # stands at line 70* — 0 after the sort, `which_trajectory` otherwise. This restores
        # the invariant MPC_NPZ_PATCH (JOB C) intended: prev_observations is the plan that
        # was actually executed.
        if self.trajectory_selection == 'temporal_consistency' and not disable_projection and self.prev_observations is not None:   # Temporal consistency
            # 🔵 U10 — the shift is the REPLAN CADENCE, not a constant 1. The previous plan
            # was executed for `replan_steps` steps before this call, so its step `n` is the
            # new plan's step 0. At the default replan_steps=1 this is the original
            # `[:, :-1] vs [:, 1:]` expression, byte-for-byte.
            _n = max(1, int(getattr(self, 'replan_steps', 1)))
            _n = min(_n, observations.shape[1] - 1)   # keep at least one overlapping step
            order = np.argsort(np.linalg.norm(observations[:,:-_n,:] - self.prev_observations[:,_n:,:], axis=(1,2)))
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

        # 🔵 U10 — publish the executed candidate's FULL plan (see __init__). `observations`
        # may have been reordered by the -t branch, which is why the two use different
        # indices: `executed_idx` indexes `observations` AS IT STANDS NOW, `which_trajectory`
        # indexes `actions`, which is never reordered (the fix_5 invariant, above).
        # The inverse-dynamics branch flattens `actions` and so has no per-candidate
        # sequence to publish; it is hard-False in this generation.
        self.last_executed_observations = observations[executed_idx]
        self.last_executed_actions = None if self.inverse_dynamics else actions[which_trajectory]

        trajectories = Trajectories(actions, observations)

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
    