"""Gen16 — the closed-loop policies for visual avoiding.

WHY THIS FILE EXISTS (and why Gen14 has no equivalent)
────────────────────────────────────────────────────────────────────────────────
Gen14's aligning eval drives its engines through `VisualAgentWrapper`, a ~700-line class
shaped by D3IL's `Aligning_Sim` callback protocol (`reset()` / `predict(state)` /
`update_rollout_info(info)`). The avoiding task has no such harness: `ObstacleAvoidanceEnv`
is a plain gym loop, and the whole state-only avoiding lineage — Gen3v2, Gen3v6, Gen3v7,
Gen12 — drives it through diffuser's `Policy`, whose surface is

    action, Trajectories(actions, observations) = policy(
        conditions={0: obs}, batch_size=B, horizon=H, disable_projection=bool)

That surface is what the mature avoiding rollout loop is written against: the receding-
horizon cadence reads `policy.last_executed_actions`, the tracking reference reads
`samples.observations`, the `-c` selection reads the projector's per-candidate costs.

So Gen16 does NOT port Gen14's agent. It keeps the avoiding lineage's loop and supplies
the SAME `Policy` surface over the visual engines. Two classes, matching arms B and C:

    VisualPolicy          arms A/B — `diffuser` (no projection) and `dpcc-*` (DPCC Projector)
    VisualHardFlowPolicy  arm C    — HardFlow's in-loop constrained sampler

🔴 The candidate-selection block in `VisualPolicy.__call__` is a faithful copy of
`flow_matcher_v3_meanflow/sampling/policies.py` — including the `which_trajectory` /
`executed_idx` split that Gen3v6 fix_5 introduced. Do not "simplify" it: the two indices
address DIFFERENT arrays (`actions` is never reordered, `observations` is, under `-t`),
and collapsing them re-creates the bug where the plan that gets executed is not the plan
that gets recorded.

WHAT IS VISUAL HERE, AND ONLY HERE
────────────────────────────────────────────────────────────────────────────────
1. `conditions` carries images alongside the obs anchor:
       {0: obs_raw (obs_dim,), 'primary_img': (C, H, W)}     <- visual_spec.COND_IMG_KEYS
2. Only the obs anchor is normalized; images are already in [0, 1] from the env.
3. The engine is called with the tuple form its wrapper expects,
   `cond = {0: (*camera_seqs, obs_seq)}`, built by `visual_spec.pack_visual`.
4. Normalization goes through `VisualNormalizer`, which adapts the two separate
   `LimitsNormalizer`s the visual train script pickles into the single-object interface
   the Projector, the HardFlow builder and this file all expect.
"""

from collections import namedtuple

import numpy as np
import torch

from ..models import visual_spec

Trajectories = namedtuple('Trajectories', 'actions observations')


# ─── normalizer adapter ───────────────────────────────────────────────────────

class VisualNormalizer:
    """One object exposing BOTH normalizer interfaces this pipeline needs.

    The visual train script pickles two independent `LimitsNormalizer`s
    (`obs_normalizer.pkl`, `act_normalizer.pkl`) because the visual dataset fits them
    separately. But three consumers want a single object:

      * `Projector.__init__`      reads `normalizer.normalizers['observations'|'actions']`
      * `build_hardflow_sampler`  reads the same `.normalizers` dict for mins/maxs
      * this file / the eval      want `normalize(x, 'observations')` like diffuser's
                                  `DatasetNormalizer`

    Gen9's eval had a `ProjectorNormalizer` that covered only the first. Covering all
    three in one adapter is what lets the avoiding rollout loop stay unmodified.
    """

    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {'observations': obs_normalizer, 'actions': act_normalizer}

    def __call__(self, x, key):
        return self.normalize(x, key)

    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def __repr__(self):
        return (f'[ VisualNormalizer ] '
                f'obs={self.normalizers["observations"]} '
                f'act={self.normalizers["actions"]}')


# ─── shared machinery ─────────────────────────────────────────────────────────

class _VisualPolicyBase:
    """Condition packing + candidate selection, shared by arms A/B and arm C.

    Subclasses implement `_sample(cond, batch_size, horizon, disable_projection)` and
    return `(trajectories_normalized_np, infos)`.
    """

    def __init__(self, model, normalizer, device='cuda:0', trajectory_selection='random'):
        self.model = model
        self.normalizer = normalizer
        self.device = device
        self.action_dim = int(model.action_dim)
        self.trajectory_selection = trajectory_selection

        self.prev_observations = None
        # ── receding-horizon cadence (Gen3v6 U10) ─────────────────────────────
        # How many actions of each plan the CALLER executes before asking for a new one.
        # This class does not loop; it needs the number for exactly one thing — the
        # temporal-consistency shift below, which assumes the previous plan advanced by
        # this many steps. DEFAULT 1 == replan every env step (the historic behaviour).
        self.replan_steps = 1
        # The FULL action/observation sequence of the plan that was actually executed
        # (i.e. of candidate `which_trajectory`, after any -c/-t selection). Published
        # because `which_trajectory` is not visible to the caller, so a replan>1 caller
        # cannot otherwise know WHICH candidate's remaining actions to replay.
        self.last_executed_actions = None
        self.last_executed_observations = None
        self.last_info = {}

    # ── condition packing ─────────────────────────────────────────────────────

    def _format_conditions(self, conditions, batch_size):
        """{0: obs_raw, <img keys>} -> the engine's `{0: (*cam_seqs, obs_seq)}` form.

        The obs anchor is normalized (the network works in normalized space); the images
        are not (they arrive in [0, 1] from the env, exactly as the dataset stored them).
        Everything is repeated across the candidate fan so the ODE starts from `batch_size`
        independent noise draws under one shared condition — the visual analogue of the
        state-only `Policy(batch_size=B)` fan.
        """
        obs_raw = np.asarray(conditions[0], dtype=np.float32).reshape(1, -1)
        obs_norm = self.normalizer.normalize(obs_raw, 'observations').astype(np.float32)

        obs_t = torch.from_numpy(obs_norm).to(self.device)            # (1, obs_dim)
        obs_b = obs_t.unsqueeze(1).repeat(batch_size, 1, 1)           # (B, 1, obs_dim)

        cam_seqs = []
        for key in visual_spec.COND_IMG_KEYS:
            if key not in conditions:
                raise KeyError(
                    f'[ VisualPolicy ] conditions has no {key!r}; this task needs '
                    f'{list(visual_spec.COND_IMG_KEYS)} (one per camera). Running the '
                    f'policy without an image would silently plan blind.')
            img = conditions[key]
            if not torch.is_tensor(img):
                img = torch.from_numpy(np.asarray(img, dtype=np.float32))
            img = img.to(self.device).float()
            # (C, H, W) -> (B, 1, C, H, W): one window step, repeated across the fan.
            cam_seqs.append(img.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1))

        return {0: visual_spec.pack_visual(tuple(cam_seqs), obs_b)}

    # ── selection + unpacking ─────────────────────────────────────────────────

    def _finish(self, trajectories, infos, batch_size, disable_projection):
        """Normalized (B, H, transition_dim) array -> (action, Trajectories).

        Copied from the state-only `Policy.__call__` tail, including the fix_5
        `which_trajectory` / `executed_idx` split — see this module's header.
        """
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        if (self.trajectory_selection == 'temporal_consistency'
                and not disable_projection and self.prev_observations is not None):
            # U10 — the shift is the REPLAN CADENCE, not a constant 1. The previous plan
            # ran for `replan_steps` steps before this call, so its step n is the new
            # plan's step 0. At replan_steps=1 this is the original [:, :-1] vs [:, 1:].
            _n = max(1, int(getattr(self, 'replan_steps', 1)))
            _n = min(_n, observations.shape[1] - 1)   # keep >= 1 overlapping step
            order = np.argsort(np.linalg.norm(
                observations[:, :-_n, :] - self.prev_observations[:, _n:, :], axis=(1, 2)))
            which_trajectory = order[0]
            observations = observations[order]
            executed_idx = 0                       # observations is sorted; best match at 0
        elif (self.trajectory_selection == 'minimum_projection_cost'
                and not disable_projection and infos.get('projection_costs')):
            costs_total = np.zeros(batch_size)
            for _timestep, cost in infos['projection_costs'].items():
                costs_total += cost
            which_trajectory = int(np.argmin(costs_total))
            executed_idx = which_trajectory        # observations not reordered
        else:
            which_trajectory = 0
            executed_idx = 0

        self.prev_observations = np.repeat(
            np.expand_dims(observations[executed_idx], axis=0), batch_size, axis=0)

        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        action = actions[which_trajectory, 0]

        # U10 — publish the executed candidate's FULL plan. The two indices differ on
        # purpose: `executed_idx` indexes `observations` AS IT STANDS NOW (possibly
        # reordered by the -t branch), `which_trajectory` indexes `actions`, which is
        # never reordered.
        self.last_executed_observations = observations[executed_idx]
        self.last_executed_actions = actions[which_trajectory]

        return action, Trajectories(actions, observations)

    # ── entry point ───────────────────────────────────────────────────────────

    def __call__(self, conditions, batch_size=1, horizon=8, disable_projection=False,
                 constraints=None, **_ignored):
        cond = self._format_conditions(conditions, batch_size)
        self.model.eval()
        with torch.no_grad():
            trajectories, infos = self._sample(
                cond, batch_size=batch_size, horizon=horizon,
                constraints=constraints, disable_projection=disable_projection)
        trajectories = trajectories.detach().cpu().numpy()
        return self._finish(trajectories, infos, batch_size, disable_projection)

    @property
    def device_of_model(self):
        return next(self.model.parameters()).device


# ─── arm A / B — plain sampler, optionally DPCC-projected ─────────────────────

class VisualPolicy(_VisualPolicyBase):
    """The `diffuser` and `dpcc-*` arms.

    `projector=None` is arm A (unguided); a `Projector` is arm B. Which of the two runs
    is decided by the eval driver, exactly as in the state-only lineage — this class only
    forwards the object into the engine's sampler, where the per-step activation gate
    (`diffusion_timestep_threshold`) lives.
    """

    def __init__(self, model, normalizer, projector=None, device='cuda:0',
                 trajectory_selection='random'):
        super().__init__(model, normalizer, device=device,
                         trajectory_selection=trajectory_selection)
        self.projector = projector

    def _sample(self, cond, batch_size, horizon, constraints, disable_projection):
        projector = None if disable_projection else self.projector
        return self.model(cond, projector=projector, constraints=constraints,
                          horizon=horizon)


# ─── arm C — HardFlow in-loop constrained sampler ─────────────────────────────

class VisualHardFlowPolicy(_VisualPolicyBase):
    """The `hardflow_new*` arm.

    HardFlow REPLACES `p_sample_loop` (its NLP runs INSIDE the ODE), so it is mutually
    exclusive with the DPCC `Projector` — both being active would double-project. The
    eval driver constructs one or the other, never both, and this class has no
    `projector` attribute at all so the mistake is not expressible here.

    `disable_projection` maps to the sampler's `activation_threshold`: with the NLP
    switched off there is nothing left but the plain ODE, so we fall back to the host
    model's own sampler rather than pretending the arm ran.
    """

    def __init__(self, model, normalizer, sampler, flow_steps, device='cuda:0',
                 trajectory_selection='random', candidate_cost='prox'):
        super().__init__(model, normalizer, device=device,
                         trajectory_selection=trajectory_selection)
        self.sampler = sampler
        # 🔴 MATCHED-K. `HardFlowSampler.sample()` takes K as an argument (the horizon is
        # baked into its layout), so K must be handed in here and it must be the SAME K
        # arms A/B run at. The eval resolves one `flow_steps` for every arm and passes it
        # to both policy classes; a mismatch would compare arms at different NFE budgets,
        # which is the defect Gen3v6 fix_7.3 §9 cost a generation over.
        self.flow_steps = int(flow_steps)
        self.candidate_cost = candidate_cost
        self.nfe = 0

    def _sample(self, cond, batch_size, horizon, constraints, disable_projection):
        if disable_projection:
            # No NLP => this is the unguided host sampler. Say so rather than reporting
            # an arm-C number that had no arm C in it.
            trajectories, infos = self.model(cond, projector=None, horizon=horizon)
            self.last_info = {'nlp_solves': 0, 'nlp_failures': 0, 'hardflow_active': False}
            return trajectories, infos

        from .hardflow_projection import encode_visual_cond
        # The sampler bypasses `model.forward()`, so the visual repack the engine wrapper
        # would have done must happen here. See `encode_visual_cond`'s docstring.
        hf_cond = encode_visual_cond(self.model, cond)
        # `horizon` is not passed: it is fixed in the sampler's TrajectoryLayout at
        # construction. The eval asserts layout.horizon == args.horizon there.
        trajectories, info = self.sampler.sample(
            hf_cond, flow_steps=self.flow_steps, batch_size=batch_size)

        self.nfe = int(getattr(self.sampler, 'nfe', self.nfe))
        self.last_info = dict(info or {})
        self.last_info.setdefault('hardflow_active', True)

        # Map HardFlow's per-candidate costs onto the key `_finish`'s `-c` branch reads,
        # so DPCC-style selection works identically on both arms. `candidate_costs` is a
        # (B,) array, whereas the DPCC projector publishes {step: (B,)} — wrapping it in a
        # single pseudo-step keeps one summation path.
        costs = self.last_info.get('candidate_costs')
        infos = dict(info or {})
        infos['projection_costs'] = {0: np.asarray(costs)} if costs is not None else {}
        return trajectories, infos
