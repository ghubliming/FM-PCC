#!/usr/bin/env python3
"""Gen12 pre-flight gates for the HardFlow -> FMv3 sampler port.

⚠️ RUN ON THE CLUSTER (i6-gpu-1, FMPCC env). Nothing here can execute in the
AI-coding container — no torch, no casadi.

No checkpoint and no dataset are needed: every gate runs against a stub
velocity field. That is deliberate — these gates test the *seams* (PLAN §3.2,
§3.3), and a seam bug is exactly what a real checkpoint would hide behind
plausible-looking trajectories.

  G0  layout        The Gen12 dof indexing is index-by-index identical to
                    HardFlow's `2*action_dim + i*(action_dim+state_dim)`, and
                    to_dof/from_dof round-trips exactly. PLAN §3.2: "write a
                    shape/layout assertion test FIRST".
  G1  direction     FMv3 integrates tau = 0 (noise) -> 1 (data) with x += v*dt,
                    same as HardFlow, so `x1 = x + (1-tau)*v` is the right
                    terminal predictor. PLAN §3.3.
  G2  NLP           The prox-NLP returns a FEASIBLE terminal trajectory from an
                    infeasible reference, and the tau bookkeeping behaves as
                    claimed (including one place where the plan is wrong — see
                    gate_g2.__doc__).
  G3  end-to-end    The full constrained sampler runs, is feasible at the end,
                    and reports NFE = K + n_active - 1 with n_active NLP solves
                    (A = 1.0, so n_active = K).
  G4  threshold     Final-step invariant + solve count, both read off the shared
                    `hardflow_step_budget` helper (DPCC floor rounding, fix_8).
  G6  HFK1 low-K    The low-K degeneracy is pinned: the budget helper matches the
                    literal loop gate, K=1 is threshold-invariant and carries ZERO
                    genuine HardFlow steps, and NFE excludes the skipped terminal
                    lookahead call. See HF_Study/DEGENERACY_HardFlow_at_low_K.md.

Usage (cluster):
    python FM_v3_hardflow_test/gates_hardflow.py
    python FM_v3_hardflow_test/gates_hardflow.py --flow-steps 5
"""

import argparse
import sys

import numpy as np
import torch
import yaml

import flow_matcher_v3_hardflow.utils as utils
from flow_matcher_v3_hardflow.sampling.hardflow_projection import (
    HardFlowNLP,
    HardFlowSampler,
    TrajectoryLayout,
    hardflow_step_budget,           # HFK1 (2026-08-24)
    hardflow_regime, HF_OK, HF_THIN, HF_DEGENERATE,   # HFK1b (2026-08-24)
)

EXP = 'avoiding-d3il'
HORIZON = 8
ACTION_DIM = 2
STATE_DIM = 4

# Gen12 reads its OWN config, never config/projection_eval.yaml. The gates build
# their constraint list from the same file eval_FM_v3_hardflow.py uses, so a gate
# that passes proves the exact geometry the eval will enforce.
CONFIG_PATH = 'config/hardflow_projection_eval.yaml'

# Stand-in normalizer limits for [vx, vy, x_des, y_des, x, y]. The gates test
# geometry and index bookkeeping, not dataset statistics; the real limits come
# from the checkpoint's normalizer at eval time.
STUB_MINS = np.array([-0.02, -0.02, 0.2, -0.3, 0.2, -0.3])
STUB_MAXS = np.array([0.02, 0.02, 0.8, 0.4, 0.8, 0.4])


class StubVelocity:
    """A velocity field with a closed form, so gate outcomes are checkable.

    `v(x, t) = target - x` is NOT used; a constant field is, because then the
    Euler solution after K steps is exactly `x0 + v` for ANY K, and the terminal
    predictor `x + (1-t)v` is exact at every t. Any deviation is a seam bug.
    """

    def __init__(self, constant):
        self.constant = constant
        self.calls = 0

    def _predict_velocity(self, x, cond, t, returns=None):
        self.calls += 1
        return self.constant.expand_as(x).clone()

    def parameters(self):
        return iter(())


def build_constraints(halfspace_variant='both-hard', config_path=CONFIG_PATH):
    """The real avoiding-d3il constraint list, exactly as eval builds it."""
    with open(config_path, 'r') as fh:
        config = yaml.safe_load(fh)

    if halfspace_variant == 'top-left-hard':
        polytopic = [config['halfspace_constraints'][EXP][0]]
        obstacles = [config['obstacle_constraints'][EXP][3]]
    elif halfspace_variant == 'top-right-hard':
        polytopic = [config['halfspace_constraints'][EXP][1]]
        obstacles = [config['obstacle_constraints'][EXP][4]]
    else:
        polytopic = [config['halfspace_constraints'][EXP][2],
                     config['halfspace_constraints'][EXP][3]]
        obstacles = [config['obstacle_constraints'][EXP][5]]

    trajectory_dim = ACTION_DIM + STATE_DIM
    obs_indices = config['observation_indices']['avoiding']
    act_indices = config['action_indices']['avoiding']
    act_obs_indices = {**act_indices,
                       **{k: v + ACTION_DIM for k, v in obs_indices.items()}}

    constraint_list = []
    for constraint in polytopic:
        constraint_list.append(
            ('ineq', utils.formulate_halfspace_constraints(
                constraint, 0, trajectory_dim, act_obs_indices)))
    lb, ub = utils.formulate_bounds_constraints(
        ['bounds'], config['bounds'][EXP], trajectory_dim, act_obs_indices)
    constraint_list.extend([['lb', lb], ['ub', ub]])
    for constr in obstacles:
        constraint_list.append([
            constr['type'],
            [act_obs_indices[constr['dimensions'][0]],
             act_obs_indices[constr['dimensions'][1]]],
            constr['center'], constr['radius']])
    constraint_list.extend(
        utils.formulate_dynamics_constraints(EXP, act_obs_indices, ACTION_DIM))
    return constraint_list, obstacles, act_obs_indices


def unnormalize(x_norm, mins=STUB_MINS, maxs=STUB_MAXS):
    return (x_norm + 1.0) * (maxs - mins) / 2.0 + mins


# ---------------------------------------------------------------------------#
# G0 — layout
# ---------------------------------------------------------------------------#
def gate_g0():
    """PLAN §3.2: verify the layout, do not assume it.

    HardFlow's `oc_dof = horizon*(state+action) - state` implies the i-th
    constrained state sits at `2*action_dim + i*(action_dim+state_dim)`.
    FMv3 uses diffuser's `[action | observation]` per-step convention with
    `apply_conditioning` pinning step 0's observation. The claim is that these
    coincide. This gate proves it rather than restating it.
    """
    print('\n-- G0: dof layout vs HardFlow, index by index ' + '-' * 30)
    L = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    ok = True

    expected_dof = HORIZON * (ACTION_DIM + STATE_DIM) - STATE_DIM
    print(f'  dof = {L.dof}   expected {expected_dof}')
    ok &= L.dof == expected_dof

    print('  step | Gen12 a_t | Gen12 s_t | HardFlow s_t (2*ad + i*(ad+sd))')
    for t in range(HORIZON):
        a_i = L.action_index(t)
        if t == 0:
            print(f'   {t:>3}  |    {a_i:>3}     |   pinned  |   n/a (s_0 is the condition)')
            continue
        s_i = L.state_index(t)
        hf = 2 * ACTION_DIM + (t - 1) * (ACTION_DIM + STATE_DIM)
        match = s_i == hf
        ok &= match
        print(f'   {t:>3}  |    {a_i:>3}     |    {s_i:>3}    |   {hf:>3}   '
              f'{"OK" if match else "MISMATCH"}')

    # round trip
    rng = np.random.RandomState(0)
    traj = rng.randn(HORIZON * (ACTION_DIM + STATE_DIM))
    s0 = traj[ACTION_DIM:ACTION_DIM + STATE_DIM]
    rt = L.from_dof(L.to_dof(traj), s0)
    exact = np.array_equal(traj, rt)
    ok &= exact
    print(f'  to_dof -> from_dof round trip exact: {exact}')

    # column semantics, both repos: [vx, vy, x_des, y_des, x, y]
    print('  transition columns : 0=vx 1=vy 2=x_des 3=y_des 4=x 5=y  (both repos)')
    print('  HardFlow reads x,y at obs offsets +2,+3 -> transition cols 4,5  OK')

    print(f'  G0 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


# ---------------------------------------------------------------------------#
# G1 — time direction
# ---------------------------------------------------------------------------#
def gate_g1(flow_steps, device):
    """PLAN §3.3: confirm FMv3 integrates tau = 0 -> 1 before wiring x1 = z + (1-tau)v.

    Gen13 lost real time to this class of bug. With a constant field v, forward
    Euler over tau in [0,1] gives x_K = x_0 + v exactly, and the terminal
    predictor x + (1-tau)*v equals x_0 + v at EVERY step. If FMv3 integrated
    backwards, or if t were an integer timestep rather than continuous tau, both
    identities break.
    """
    print('\n-- G1: time direction tau = 0 (noise) -> 1 (data) ' + '-' * 26)
    T = ACTION_DIM + STATE_DIM
    const = torch.full((1, HORIZON, T), 0.3, device=device)
    model = StubVelocity(const)

    x = torch.zeros(1, HORIZON, T, device=device)
    dt = 1.0 / flow_steps
    ok = True
    for k in range(flow_steps):
        tau = k * dt
        v = model._predict_velocity(x, {}, torch.full((1,), tau, device=device))
        x1_pred = x + (1.0 - tau) * v
        err = (x1_pred - 0.3).abs().max().item()
        if err > 1e-5:
            ok = False
        x = x + v * dt
    terminal_err = (x - 0.3).abs().max().item()
    ok &= terminal_err < 1e-5
    print(f'  K = {flow_steps}, constant field v = 0.3')
    print(f'  max |x_K - (x_0 + v)|                = {terminal_err:.3e}')
    print(f'  terminal predictor exact at every k  = {ok}')

    # the same convention, read straight off the shipped sampler
    from flow_matcher_v3_hardflow.models.diffusion import GaussianDiffusion
    src = GaussianDiffusion.p_sample_loop.__doc__ or ''
    import inspect
    body = inspect.getsource(GaussianDiffusion.p_sample_loop)
    forward = 'loop_idx / max(self.flow_steps_v3, 1)' in body
    plus = 'x + velocity * dt' in inspect.getsource(GaussianDiffusion.p_mean_variance)
    ok &= forward and plus
    print(f'  FMv3 p_sample_loop uses t = k/K (increasing) : {forward}')
    print(f'  FMv3 p_mean_variance uses x + v*dt           : {plus}')
    print(f'  G1 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


# ---------------------------------------------------------------------------#
# G2 — the NLP
# ---------------------------------------------------------------------------#
def gate_g2(halfspace_variant='both-hard', config_path=CONFIG_PATH):
    """Feasibility of the prox solve, plus one correction to the plan.

    PLAN §1.2 describes the pull-back as "weighted by tau^2
    (oc_control_cost ... * self.oc_t_param**2)". Those are two different things.
    With a PURE proximal objective, multiplying the cost by tau^2 does not move
    the argmin at all — it is a positive scalar on the only term. The actual
    tau-gating of the trajectory is the LINEAR factor in the pull-back
    `x_next = x_ref + tau*(x1_proj - x1_ref)`. The tau^2 factor only becomes
    live if a competing objective (upstream's `distance` term) is added, so it
    is kept in the code but must not be described as the schedule.

    This gate asserts: the solve is feasible, and the BINDING behaviour (feasibility
    / min-obstacle distance) is tau-invariant. It does NOT assert bitwise DOF
    invariance — the TRUE argmin is tau-invariant, but IPOPT is an iterative
    interior-point solver and rescaling the objective by tau^2 (0.0625 vs 1.0)
    shifts its convergence/stopping, so non-binding DOFs drift within solver
    tolerance. Asserting bitwise equality would test IPOPT numerics, not the port.
    """
    print('\n-- G2: prox-NLP feasibility and tau bookkeeping ' + '-' * 29)
    constraint_list, obstacles, idx = build_constraints(halfspace_variant, config_path)
    L = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    nlp = HardFlowNLP(layout=L, constraint_list=constraint_list,
                      mins=STUB_MINS, maxs=STUB_MAXS, dt=1.0,
                      reg_scale=1.0, dynamics_mode='deriv', print_level=0)

    # A deliberately infeasible reference: park every step on the obstacle centre.
    centre = np.asarray(obstacles[0]['center'], dtype=float)
    radius = float(obstacles[0]['radius'])
    s0_phys = np.array([centre[0], centre[1] - 0.15, centre[0], centre[1] - 0.15])
    to_norm = lambda phys, dims: (
        (phys - STUB_MINS[dims]) / (STUB_MAXS[dims] - STUB_MINS[dims]) * 2 - 1)
    s0 = to_norm(s0_phys, np.array([2, 3, 4, 5]))
    nlp.set_s0(s0)

    x1_ref = np.zeros(L.dof)
    for t in range(1, HORIZON):
        si = L.state_index(t)
        x1_ref[si:si + STATE_DIM] = to_norm(
            np.array([centre[0], centre[1], centre[0], centre[1]]),
            np.array([2, 3, 4, 5]))

    ok = True
    sols = {}
    worst_by_tau = {}
    for tau in (0.25, 1.0):
        sol = nlp.solve(x1_ref, tau)
        sols[tau] = sol
        dists = []
        for t in range(1, HORIZON):
            si = L.state_index(t)
            s_phys = unnormalize(
                np.concatenate([np.zeros(ACTION_DIM), sol[si:si + STATE_DIM]]))
            dists.append(np.linalg.norm(s_phys[[idx['x'], idx['y']]] - centre))
        worst = min(dists)
        worst_by_tau[tau] = worst
        feasible = worst >= radius - 1e-3
        ok &= feasible
        print(f'  tau = {tau:<5} min obstacle distance = {worst:.4f} '
              f'(radius {radius:.3f})  -> {"feasible" if feasible else "VIOLATED"}')

    # tau-invariance of the BINDING behaviour (NOT bitwise DOF equality). The true
    # argmin is invariant to the tau^2 scalar, but IPOPT does not reproduce its iterate
    # bitwise under objective rescaling — non-binding DOFs drift within solver tolerance.
    # So we require the feasibility-relevant quantity (min-obstacle distance) to be
    # tau-invariant, and report the raw DOF drift only as INFO.
    binding_gap = abs(worst_by_tau[0.25] - worst_by_tau[1.0])
    raw_drift = float(np.max(np.abs(sols[0.25] - sols[1.0])))
    binding_invariant = binding_gap < 1e-3
    print(f'  binding (min-dist) invariant to tau: {binding_invariant}  (|Δ| = {binding_gap:.2e})')
    print(f'  raw DOF drift (INFO only): {raw_drift:.2e}')
    print('    ^ the TRUE argmin is tau-invariant (tau^2 scales the ONLY cost term); any raw')
    print('      drift is IPOPT numerics on non-binding DOFs, not a bug. The real schedule is')
    print('      the LINEAR tau in the pull-back x_next = x_ref + tau*(x1_proj - x1_ref).')
    ok &= binding_invariant
    print(f'  solves = {nlp.n_solves}, failures = {nlp.n_failures}')
    ok &= nlp.n_failures == 0
    print(f'  G2 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


# ---------------------------------------------------------------------------#
# G3 — end to end
# ---------------------------------------------------------------------------#
def gate_g3(flow_steps, device, halfspace_variant='both-hard', config_path=CONFIG_PATH):
    """The whole sampler on a stub field: runs, ends feasible, costs what we say."""
    print('\n-- G3: end-to-end constrained sampler ' + '-' * 38)
    constraint_list, obstacles, idx = build_constraints(halfspace_variant, config_path)
    L = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    nlp = HardFlowNLP(layout=L, constraint_list=constraint_list,
                      mins=STUB_MINS, maxs=STUB_MAXS, dt=1.0,
                      reg_scale=1.0, dynamics_mode='deriv', print_level=0)

    T = ACTION_DIM + STATE_DIM
    model = StubVelocity(torch.full((1, HORIZON, T), 0.2, device=device))
    # [HFK1 2026-08-24] Was `activation_threshold=0.0`. Under fix_6's DPCC polarity 0.0 means
    # TERMINAL-ONLY (n_active = 1), so this gate has been building a 1-solve sampler while
    # asserting K solves and 2K NFE below — a pre-fix_6 contract that no longer holds. G3's
    # subject is the "every step is solved" end-to-end path, which is A = 1.0.
    G3_THRESHOLD = 1.0
    sampler = HardFlowSampler(model=model, layout=L, nlp=nlp, device=device,
                              activation_threshold=G3_THRESHOLD)

    centre = np.asarray(obstacles[0]['center'], dtype=float)
    radius = float(obstacles[0]['radius'])
    s0_phys = np.array([centre[0], centre[1] - 0.15, centre[0], centre[1] - 0.15])
    s0_norm = (s0_phys - STUB_MINS[2:]) / (STUB_MAXS[2:] - STUB_MINS[2:]) * 2 - 1
    cond = {0: torch.as_tensor(s0_norm, dtype=torch.float32,
                               device=device).unsqueeze(0)}

    torch.manual_seed(0)
    x, infos = sampler.sample(cond, flow_steps=flow_steps, batch_size=1)

    ok = tuple(x.shape) == (1, HORIZON, T)
    print(f'  output shape {tuple(x.shape)} expected (1, {HORIZON}, {T})')

    traj = x[0].detach().cpu().numpy()
    dists = [np.linalg.norm(unnormalize(traj[t])[[idx['x'], idx['y']]] - centre)
             for t in range(1, HORIZON)]
    feasible = min(dists) >= radius - 1e-3
    ok &= feasible
    print(f'  min obstacle distance = {min(dists):.4f} (radius {radius:.3f}) '
          f'-> {"feasible" if feasible else "VIOLATED"}')

    # [HFK1 2026-08-24] NFE is now K + n_active - 1, not K + n_active: the terminal step's
    # lookahead call was multiplied by (1 - tau_next) == 0 and is no longer made.
    n_active, n_genuine = hardflow_step_budget(flow_steps, G3_THRESHOLD)
    exp_nfe = flow_steps + n_active - 1
    nfe_ok = infos['nfe'] == exp_nfe
    solves_ok = infos['nlp_solves'] == n_active
    ok &= nfe_ok and solves_ok
    print(f'  NFE = {infos["nfe"]} (expected {exp_nfe})   '
          f'NLP solves = {infos["nlp_solves"]} (expected {n_active})   '
          f'failures = {infos["nlp_failures"]}')
    print(f'    ^ at A={G3_THRESHOLD} arm C costs ~2 network evals per ODE step vs arm B\'s 1 '
          f'(minus the\n      skipped terminal lookahead). Any matched-K comparison must '
          f'report this (PLAN §5).\n      genuine (non-terminal) HardFlow steps here: '
          f'{n_genuine} of {flow_steps}.')
    print(f'  G3 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


# ---------------------------------------------------------------------------#
# G4 — U4 activation threshold: final-step invariant + monotone solve count
# ---------------------------------------------------------------------------#
def _make_sampler(device, activation_threshold, halfspace_variant='both-hard',
                  config_path=CONFIG_PATH):
    constraint_list, obstacles, idx = build_constraints(halfspace_variant, config_path)
    L = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    nlp = HardFlowNLP(layout=L, constraint_list=constraint_list, mins=STUB_MINS,
                      maxs=STUB_MAXS, dt=1.0, reg_scale=1.0, dynamics_mode='deriv',
                      print_level=0)
    T = ACTION_DIM + STATE_DIM
    model = StubVelocity(torch.full((1, HORIZON, T), 0.2, device=device))
    sampler = HardFlowSampler(model=model, layout=L, nlp=nlp, device=device,
                              activation_threshold=activation_threshold)
    centre = np.asarray(obstacles[0]['center'], dtype=float)
    radius = float(obstacles[0]['radius'])
    s0_phys = np.array([centre[0], centre[1] - 0.15, centre[0], centre[1] - 0.15])
    s0_norm = (s0_phys - STUB_MINS[2:]) / (STUB_MAXS[2:] - STUB_MINS[2:]) * 2 - 1
    cond = {0: torch.as_tensor(s0_norm, dtype=torch.float32, device=device).unsqueeze(0)}
    return sampler, nlp, L, cond, centre, radius, idx


def gate_g4(device, halfspace_variant='both-hard', config_path=CONFIG_PATH):
    """U4 + fix_6 (DPCC polarity): the final step is ALWAYS solved (safety), and the
    solve count is monotone NON-DECREASING in the threshold (higher thr = more
    projection) and matches #{k: tau_next >= (1 - thr)} plus the forced final step."""
    print('\n-- G4: U4 activation threshold — final-step invariant + count ' + '-' * 14)
    ok = True
    for K in (2, 5, 10):
        prev_solves = None
        for thr in (0.0, 0.5, 0.9, 1.0):
            sampler, nlp, L, cond, centre, radius, idx = _make_sampler(
                device, thr, halfspace_variant, config_path)
            torch.manual_seed(0)
            x, infos = sampler.sample(cond, flow_steps=K, batch_size=1)
            # [HFK1 2026-08-24] Was the RAW-FLOAT form `k >= (1.0 - thr) * K`, i.e. CEIL.
            # Gen12 fix_8 moved the sampler to DPCC's FLOOR `int((1 - thr) * K)` but never
            # updated this gate, so G4 has been asserting the pre-fix_8 count ever since —
            # it disagrees with the shipped sampler at (K=2, thr=0.9), (K=5, thr=0.5),
            # (K=5, thr=0.9) and (K=10, thr=0.9): 4 of its 12 cells. Both the gate and the
            # loop now read the SAME helper, so they cannot drift apart again.
            expected, n_genuine = hardflow_step_budget(K, thr)
            solves = infos['nlp_solves']
            # terminal feasibility (safety guarantee) must hold at every threshold
            traj = x[0].detach().cpu().numpy()
            dmin = min(np.linalg.norm(unnormalize(traj[t])[[idx['x'], idx['y']]] - centre)
                       for t in range(1, HORIZON))
            feasible = dmin >= radius - 1e-3
            count_ok = solves == expected
            mono_ok = prev_solves is None or solves >= prev_solves  # DPCC polarity: ↑thr ⇒ ↑solves
            prev_solves = solves
            ok &= feasible and count_ok and mono_ok
            print(f'  K={K:>2} thr={thr:<4} solves={solves:>2} (exp {expected:>2})  '
                  f'genuine={n_genuine:>2}{" DEGENERATE" if n_genuine == 0 else "          "}  '
                  f'min_d={dmin:.3f} {"feasible" if feasible else "VIOLATED"}  '
                  f'{"OK" if (feasible and count_ok and mono_ok) else "FAIL"}')
    print(f'  G4 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


# ---------------------------------------------------------------------------#
# G5 — U4.2 candidate fan + DPCC-style selection
# ---------------------------------------------------------------------------#
def gate_g5(device, halfspace_variant='both-hard', config_path=CONFIG_PATH):
    """U4.2: a batch fan produces per-candidate costs; minimum_projection_cost
    returns argmin(candidate_costs); random returns 0; each terminal feasible."""
    print('\n-- G5: U4.2 candidate fan + selection ' + '-' * 38)
    from flow_matcher_v3_hardflow.sampling.hardflow_projection import HardFlowPolicy
    ok = True
    sampler, nlp, L, cond, centre, radius, idx = _make_sampler(
        device, 0.0, halfspace_variant, config_path)
    torch.manual_seed(0)
    x, infos = sampler.sample(cond, flow_steps=5, batch_size=4)

    costs = infos.get('candidate_costs')
    has_costs = costs is not None and len(costs) == 4
    print(f'  candidate_costs present, len==4: {has_costs}  values={np.round(costs, 4) if has_costs else None}')
    ok &= has_costs

    # every candidate's terminal must be feasible (safety holds per candidate)
    feas = True
    for b in range(4):
        traj = x[b].detach().cpu().numpy()
        dmin = min(np.linalg.norm(unnormalize(traj[t])[[idx['x'], idx['y']]] - centre)
                   for t in range(1, HORIZON))
        feas &= dmin >= radius - 1e-3
    print(f'  all 4 candidate terminals feasible: {feas}')
    ok &= feas

    # selection rules (mirror HardFlowPolicy._select on stub infos)
    class _P:
        trajectory_selection = 'minimum_projection_cost'
        candidate_cost = 'prox'
        prev_observations = None
    sel_c = HardFlowPolicy._select(_P(), np.zeros((4, HORIZON, STATE_DIM)), infos, 4, False)
    _P.trajectory_selection = 'random'
    sel_r = HardFlowPolicy._select(_P(), np.zeros((4, HORIZON, STATE_DIM)), infos, 4, False)
    c_ok = sel_c == int(np.argmin(costs))
    r_ok = sel_r == 0
    ok &= c_ok and r_ok
    print(f'  minimum_projection_cost -> {sel_c} (argmin {int(np.argmin(costs))})  {"OK" if c_ok else "FAIL"}')
    print(f'  random -> {sel_r} (expect 0)  {"OK" if r_ok else "FAIL"}')
    print(f'  G5 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


# ---------------------------------------------------------------------------#
# G6 — HFK1: low-K degeneracy is detected, announced, and costed correctly
# ---------------------------------------------------------------------------#
def gate_g6(device, halfspace_variant='both-hard', config_path=CONFIG_PATH):
    """HFK1 (2026-08-24): the low-K degeneracy is pinned, not rediscovered.

    A step does real HardFlow work only if it is ACTIVE and NOT the terminal step —
    at k = K-1 the flow time is exactly 1, which kills the endpoint lookahead, turns
    the pull-back into a full snap, and leaves no successor call to react. So:

      (a) `hardflow_step_budget` must agree with the LITERAL loop gate on a K x A grid
          (it is the single source both the sampler and G4 now read);
      (b) K=1 is degenerate at EVERY threshold, and — since the gate cannot fire
          anywhere else — its output must be BIT-IDENTICAL across thresholds;
      (c) NFE is K + n_active - 1: the terminal lookahead call is no longer made.

    See logs_in_develop/HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md.
    """
    print('\n-- G6: HFK1 low-K degeneracy + NFE accounting ' + '-' * 30)
    ok = True

    # (a) helpers vs the literal gate expression, over the whole grid we ever ship
    grid_ok = True
    for K in (1, 2, 3, 5, 6, 10, 14, 20):
        for A in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
            active = [k for k in range(K)
                      if (k >= int((1.0 - A) * K)) or (k == K - 1)]
            genuine = [k for k in active if k != K - 1]
            if hardflow_step_budget(K, A) != (len(active), len(genuine)):
                grid_ok = False
                print(f'    MISMATCH K={K} A={A}: helper={hardflow_step_budget(K, A)} '
                      f'loop=({len(active)}, {len(genuine)})')
            # HFK1b: the regime classifier must agree with the same literal loop, and
            # `first_lookahead` must be (1 - tau_next) at the FIRST genuine step — the
            # least trustworthy guided step, and the one Thm. 4's bound is worst at.
            tier, n_a, n_g, la = hardflow_regime(K, A)
            exp_tier = (HF_DEGENERATE if not genuine
                        else HF_THIN if len(genuine) == 1 else HF_OK)
            exp_la = (1.0 - float(genuine[0] + 1) / K) if genuine else 0.0
            if (tier, n_a, n_g) != (exp_tier, len(active), len(genuine)) \
                    or abs(la - exp_la) > 1e-12:
                grid_ok = False
                print(f'    REGIME MISMATCH K={K} A={A}: got ({tier}, {n_a}, {n_g}, '
                      f'{la:.6f}) expected ({exp_tier}, {len(active)}, {len(genuine)}, '
                      f'{exp_la:.6f})')
    ok &= grid_ok
    print(f'  (a) hardflow_step_budget + hardflow_regime == literal loop gate on 8x6 grid: '
          f'{"OK" if grid_ok else "FAIL"}')
    # the two tiers that must warn, and the one that must not
    tier_ref = {(1, 0.5): HF_DEGENERATE, (2, 0.5): HF_DEGENERATE, (3, 0.5): HF_THIN,
                (5, 0.5): HF_OK, (1, 1.0): HF_DEGENERATE, (2, 1.0): HF_THIN,
                (5, 1.0): HF_OK, (20, 0.0): HF_DEGENERATE}
    tier_ok = all(hardflow_regime(K, A)[0] == t for (K, A), t in tier_ref.items())
    ok &= tier_ok
    print(f'  (a) tier reference table {{(K, A): tier}} -> '
          f'{"OK" if tier_ok else "FAIL"}')

    # the documented reference row (DEGENERACY §4.1), A = 0.5 as shipped
    ref = {1: (1, 0), 2: (1, 0), 5: (3, 2), 10: (5, 4), 20: (10, 9)}
    ref_ok = all(hardflow_step_budget(K, 0.5) == v for K, v in ref.items())
    ok &= ref_ok
    print(f'  (a) A=0.5 reference row {ref} -> {"OK" if ref_ok else "FAIL"}')

    # (b) K=1 is threshold-invariant, bit for bit
    outs = {}
    for A in (0.0, 0.5, 1.0):
        sampler, _nlp, _L, cond, _c, _r, _i = _make_sampler(
            device, A, halfspace_variant, config_path)
        torch.manual_seed(0)
        x, infos = sampler.sample(cond, flow_steps=1, batch_size=1)
        outs[A] = (x.detach().cpu().numpy(), infos['nlp_solves'],
                   infos['nfe'], infos['n_genuine'])
    base = outs[0.0][0]
    inv_ok = all(np.array_equal(base, outs[A][0]) for A in outs)
    deg_ok = all(outs[A][3] == 0 and outs[A][1] == 1 for A in outs)
    ok &= inv_ok and deg_ok
    print(f'  (b) K=1 output identical across A in {{0.0, 0.5, 1.0}}: '
          f'{"OK" if inv_ok else "FAIL"}   n_genuine==0 & 1 solve everywhere: '
          f'{"OK" if deg_ok else "FAIL"}')
    print(f'      ^ at K=1 the arm is Pi_S(one Euler step) — sample-then-project, '
          f'NOT HardFlow.')

    # (c) NFE accounting: the terminal lookahead call is skipped
    nfe_ok = True
    for K in (1, 2, 5):
        for A in (0.5, 1.0):
            sampler, _nlp, _L, cond, _c, _r, _i = _make_sampler(
                device, A, halfspace_variant, config_path)
            torch.manual_seed(0)
            _x, infos = sampler.sample(cond, flow_steps=K, batch_size=1)
            n_active, n_genuine = hardflow_step_budget(K, A)
            exp = K + n_active - 1
            cell = (infos['nfe'] == exp and infos['nlp_solves'] == n_active
                    and infos['n_active'] == n_active
                    and infos['n_genuine'] == n_genuine)
            nfe_ok &= cell
            print(f'      K={K:>2} A={A:<4} NFE={infos["nfe"]:>2} (exp {exp:>2})  '
                  f'solves={infos["nlp_solves"]:>2} (exp {n_active:>2})  '
                  f'genuine={n_genuine}  {"OK" if cell else "FAIL"}')
    ok &= nfe_ok
    print(f'  (c) NFE == K + n_active - 1: {"OK" if nfe_ok else "FAIL"}')

    print(f'  G6 -> {"PASS" if ok else "FAIL"}')
    return bool(ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow-steps', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--halfspace-variant', default='both-hard')
    parser.add_argument('--config', default=CONFIG_PATH,
                        help='Gen12 eval config the gates read constraints from.')
    args = parser.parse_args()

    print(f'Gen12 gates | device = {args.device} | K = {args.flow_steps} | config = {args.config}')
    results = {
        'G0 layout': gate_g0(),
        'G1 direction': gate_g1(args.flow_steps, args.device),
        'G2 NLP': gate_g2(args.halfspace_variant, args.config),
        'G3 end-to-end': gate_g3(args.flow_steps, args.device, args.halfspace_variant, args.config),
        'G4 U4 threshold': gate_g4(args.device, args.halfspace_variant, args.config),
        'G5 U4.2 selection': gate_g5(args.device, args.halfspace_variant, args.config),
        'G6 HFK1 low-K': gate_g6(args.device, args.halfspace_variant, args.config),
    }

    print('\n' + '=' * 60)
    for name, passed in results.items():
        print(f'  {name:<16} {"PASS" if passed else "FAIL"}')
    print('=' * 60)
    if not all(results.values()):
        print('\nA gate failed. PLAN §4: do not proceed past a failing step.')
        sys.exit(1)


if __name__ == '__main__':
    main()
