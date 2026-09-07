#!/usr/bin/env python3
"""Gen12 solver bench — HardFlow's IPOPT vs DPCC's SLSQP on the SAME NLP.

⚠️ RUN ON THE CLUSTER (i6-gpu-1, FMPCC env). Nothing here executes in the
AI-coding container — no casadi, no scipy, no torch.

WHY THIS EXISTS
---------------
On `avoiding-d3il` our arm C (HardFlow) is 1.4-14x more expensive per plan than
arm B (DPCC), while HardFlow's own paper reports the opposite sign. The audit
(`Data_Analysis/DA_Result_Curated_MD/AUDIT_20260827_hardflow_paper_timing_and_
baselines.md` §0) traced that to the SOLVER, not the algorithm: every row of
their table is IPOPT, ours pits IPOPT against scipy SLSQP, and ~81% of an H8
IPOPT solve looks like fixed per-call overhead rather than optimisation work.

This script measures that directly, and it does NOT change any production code
path -- it builds both solvers side by side in-process and times them.

WHAT IT MEASURES
----------------
For each repetition, on the SAME reference trajectory:

  1. `HardFlowNLP.solve`   -- IPOPT/CasADi, the arm-C projector
  2. `Projector.project`   -- scipy SLSQP, the arm-B projector

and reports per-solve wall time for each, plus (free, and separately useful):

  3. ||Pi_IPOPT - Pi_SLSQP||  -- how far apart the two answers are
  4. max obstacle/bound residual of EACH output -- is either INFEASIBLE?
  5. failure counts on both sides

(3)-(5) are the offline gate proposed in `HF_Study/DEGENERACY_HardFlow_at_low_K.md`
§0.3 and never built: if one projector returns infeasible output, that is a bug
worth more than the whole cost comparison.

TWO REFERENCE REGIMES
---------------------
The audit's claim is that HardFlow's NLP is cheaper because of WHAT it projects,
not how often. `--ref` reproduces both cases on the same solver:

  --ref endpoint   near-feasible reference  (HardFlow projects a predicted CLEAN
                   endpoint, which sits near the data manifold)
  --ref iterate    far-from-feasible reference (Projection-All/Late project the
                   NOISY intermediate ODE iterate)
  --ref both       run both regimes (default)

So the output answers two questions at once: which solver is cheaper, and how
much of the cost is the difficulty of the reference.

NO CHECKPOINT, NO DATASET, NO ENV. Geometry comes from the real
`config/hardflow_projection_eval.yaml` via `gates_hardflow.build_constraints`,
so the constraint set is byte-identical to what the eval enforces.

Usage (cluster):
    python FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py
    python FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py --reps 100 --horizon 16
    python FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py --ref iterate --csv out.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from FM_v3_hardflow_test.gates_hardflow import (       # noqa: E402
    build_constraints, STUB_MINS, STUB_MAXS, EXP, ACTION_DIM, STATE_DIM,
)
from flow_matcher_v3_hardflow.sampling.hardflow_projection import (  # noqa: E402
    HardFlowNLP, TrajectoryLayout,
)
from flow_matcher_v3_hardflow.sampling.projection import Projector   # noqa: E402


# ---------------------------------------------------------------------------#
# stub normalizer -- both projectors must see the SAME geometry
# ---------------------------------------------------------------------------#
class _Limits:
    def __init__(self, mins, maxs):
        self.mins, self.maxs = np.asarray(mins), np.asarray(maxs)


class StubNormalizer:
    """Minimal stand-in for the checkpoint normalizer.

    `Projector` only ever reads `.normalizers['observations'|'actions'].mins/.maxs`
    (via ProjectionNormalizer.get_limits), so this is sufficient AND it pins both
    arms to the same limits -- which is the whole point of the comparison.
    """

    def __init__(self, mins, maxs, action_dim):
        self.normalizers = {
            'actions': _Limits(mins[:action_dim], maxs[:action_dim]),
            'observations': _Limits(mins[action_dim:], maxs[action_dim:]),
        }


def unnormalize(x_norm, mins, maxs):
    return (x_norm + 1.0) * (maxs - mins) / 2.0 + mins


# ---------------------------------------------------------------------------#
# feasibility -- obstacles and box bounds, in UNNORMALIZED units
# ---------------------------------------------------------------------------#
def max_residual(traj_norm, constraint_list, mins, maxs, horizon, transition_dim):
    """Largest constraint violation of `traj_norm` (>0 means INFEASIBLE).

    Checks the two constraint families whose tuple shape is unambiguous:
    `sphere_outside`/`sphere_inside` and `lb`/`ub`. Linear halfspace and `deriv`
    rows are enforced as hard equality/inequality rows by BOTH solvers and are
    not re-derived here; see the CHANGELOG for why that is sufficient.
    """
    x = np.asarray(traj_norm).reshape(horizon, transition_dim)
    xu = unnormalize(x, mins, maxs)
    worst = 0.0
    for spec in constraint_list:
        kind = spec[0]
        if kind in ('sphere_outside', 'sphere_inside'):
            (ix, iy), center, radius = spec[1], spec[2], float(spec[3])
            cx, cy = float(center[0]), float(center[1])
            for t in range(1, horizon):
                sq = (xu[t, int(ix)] - cx) ** 2 + (xu[t, int(iy)] - cy) ** 2
                v = (radius ** 2 - sq) if kind == 'sphere_outside' else (sq - radius ** 2)
                worst = max(worst, float(v))
        elif kind in ('lb', 'ub'):
            bound = np.asarray(spec[1], dtype=float)
            for dim in range(transition_dim):
                if not np.isfinite(bound[dim]):
                    continue
                t0 = 0 if dim < ACTION_DIM else 1
                for t in range(t0, horizon):
                    v = (bound[dim] - xu[t, dim]) if kind == 'lb' else (xu[t, dim] - bound[dim])
                    worst = max(worst, float(v))
    return worst


# ---------------------------------------------------------------------------#
# references
# ---------------------------------------------------------------------------#
def make_reference(rng, regime, horizon, transition_dim):
    """A normalized reference trajectory in [-1, 1]^(H*T).

    endpoint : small perturbation of a smooth, mostly-feasible path -- stands in
               for HardFlow's predicted CLEAN terminal sample.
    iterate  : heavy noise -- stands in for a mid-ODE NOISY iterate, which is
               what Projection-All/Late hand to their solver.
    """
    base = np.linspace(-0.6, 0.6, horizon)[:, None] * np.ones((1, transition_dim))
    if regime == 'endpoint':
        traj = base + 0.05 * rng.standard_normal((horizon, transition_dim))
    elif regime == 'iterate':
        traj = base + 0.60 * rng.standard_normal((horizon, transition_dim))
    else:
        raise ValueError(regime)
    return np.clip(traj, -1.0, 1.0).reshape(-1)


# ---------------------------------------------------------------------------#
def run(args):
    horizon = args.horizon
    transition_dim = ACTION_DIM + STATE_DIM
    layout = TrajectoryLayout(horizon, ACTION_DIM, STATE_DIM)

    constraint_list, _obstacles, _idx = build_constraints(args.halfspace)
    normalizer = StubNormalizer(STUB_MINS, STUB_MAXS, ACTION_DIM)

    print('=' * 78)
    print('Gen12 solver bench -- HardFlow IPOPT vs DPCC SLSQP, same NLP')
    print(f'  horizon={horizon}  dof={layout.dof}  vars(DPCC)={horizon*transition_dim}')
    print(f'  halfspace={args.halfspace}  reps={args.reps}  tau={args.tau}')
    print('=' * 78)

    # --- arm C: HardFlow's IPOPT NLP (operates on the dof vector, s_0 pinned)
    # [SolverSwap] PIN to ipopt. The shipped default is now 'slsqp', so without
    # this the bench would compare SLSQP against SLSQP and report ~1.0x.
    nlp = HardFlowNLP(layout, constraint_list, STUB_MINS, STUB_MAXS,
                      nlp_backend='ipopt',
                      dt=args.dt, reg_scale=1.0, print_level=0, print_time=False)

    # --- arm B: DPCC's projector, scipy SLSQP, same constraint_list
    projector = Projector(horizon=horizon, transition_dim=transition_dim,
                          action_dim=ACTION_DIM, goal_dim=0,
                          constraint_list=constraint_list, normalizer=normalizer,
                          variant='states_actions', dt=args.dt,
                          skip_initial_state=True, device='cpu', solver='scipy',
                          parallelize=False)

    import torch  # local: only needed for the Projector's tensor interface

    regimes = ['endpoint', 'iterate'] if args.ref == 'both' else [args.ref]
    rows = []

    for regime in regimes:
        rng = np.random.default_rng(args.seed)

        # WARM-UP -- the first solve of each backend pays CasADi codegen / scipy
        # import / BLAS thread spin-up. Excluding it is the difference between
        # measuring a solver and measuring an import.
        warm = make_reference(rng, regime, horizon, transition_dim)
        s0 = warm.reshape(horizon, transition_dim)[0, ACTION_DIM:]
        nlp.set_s0(s0)
        for _ in range(args.warmup):
            nlp.solve(layout.to_dof(warm), args.tau)
            projector.project(torch.tensor(warm.reshape(1, horizon, transition_dim),
                                           dtype=torch.float32, device='cpu'))

        t_ipopt, t_slsqp, deltas, res_i, res_s = [], [], [], [], []
        fail_i0, fail_i = nlp.n_failures, None

        for _ in range(args.reps):
            ref = make_reference(rng, regime, horizon, transition_dim)
            s0 = ref.reshape(horizon, transition_dim)[0, ACTION_DIM:]
            ref_dof = layout.to_dof(ref)

            nlp.set_s0(s0)
            t0 = time.perf_counter()
            out_dof = nlp.solve(ref_dof, args.tau)
            t_ipopt.append((time.perf_counter() - t0) * 1e3)

            ref_t = torch.tensor(ref.reshape(1, horizon, transition_dim),
                                 dtype=torch.float32, device='cpu')
            t0 = time.perf_counter()
            out_dpcc, _cost = projector.project(ref_t)
            t_slsqp.append((time.perf_counter() - t0) * 1e3)

            hf_full = layout.from_dof(np.asarray(out_dof).reshape(-1), s0)
            dp_full = out_dpcc.detach().cpu().numpy().reshape(-1)
            deltas.append(float(np.linalg.norm(hf_full - dp_full)))
            res_i.append(max_residual(hf_full, constraint_list, STUB_MINS, STUB_MAXS,
                                      horizon, transition_dim))
            res_s.append(max_residual(dp_full, constraint_list, STUB_MINS, STUB_MAXS,
                                      horizon, transition_dim))

        fail_i = nlp.n_failures - fail_i0
        ti, ts = np.array(t_ipopt), np.array(t_slsqp)
        row = dict(
            regime=regime, horizon=horizon, dof=layout.dof, reps=args.reps,
            ipopt_ms_mean=float(ti.mean()), ipopt_ms_median=float(np.median(ti)),
            ipopt_ms_std=float(ti.std()),
            slsqp_ms_mean=float(ts.mean()), slsqp_ms_median=float(np.median(ts)),
            slsqp_ms_std=float(ts.std()),
            ratio_median=float(np.median(ti) / max(np.median(ts), 1e-9)),
            delta_mean=float(np.mean(deltas)), delta_max=float(np.max(deltas)),
            resid_ipopt_max=float(np.max(res_i)), resid_slsqp_max=float(np.max(res_s)),
            ipopt_failures=int(fail_i),
        )
        rows.append(row)

        print(f'\n--- reference regime: {regime}')
        print(f'  IPOPT (HardFlow) : {row["ipopt_ms_median"]:8.3f} ms median '
              f'({row["ipopt_ms_mean"]:.3f} +- {row["ipopt_ms_std"]:.3f} mean)')
        print(f'  SLSQP (DPCC)     : {row["slsqp_ms_median"]:8.3f} ms median '
              f'({row["slsqp_ms_mean"]:.3f} +- {row["slsqp_ms_std"]:.3f} mean)')
        print(f'  IPOPT / SLSQP    : {row["ratio_median"]:8.2f}x')
        print(f'  ||Pi_I - Pi_S||  : mean {row["delta_mean"]:.5f}  max {row["delta_max"]:.5f}')
        print(f'  max residual     : IPOPT {row["resid_ipopt_max"]:+.2e}   '
              f'SLSQP {row["resid_slsqp_max"]:+.2e}   (>0 == INFEASIBLE)')
        print(f'  IPOPT failures   : {row["ipopt_failures"]} / {args.reps}')

    if len(rows) == 2:
        e, i = rows[0], rows[1]
        print('\n--- endpoint vs iterate (the audit §0.1 claim)')
        print(f'  IPOPT: {e["ipopt_ms_median"]:.3f} -> {i["ipopt_ms_median"]:.3f} ms '
              f'= {i["ipopt_ms_median"]/max(e["ipopt_ms_median"],1e-9):.2f}x harder on a noisy reference')
        print(f'  SLSQP: {e["slsqp_ms_median"]:.3f} -> {i["slsqp_ms_median"]:.3f} ms '
              f'= {i["slsqp_ms_median"]/max(e["slsqp_ms_median"],1e-9):.2f}x')

    if args.csv:
        import csv as _csv
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f'\n[ bench ] appended {len(rows)} row(s) -> {args.csv}')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(rows, fh, indent=2)
        print(f'[ bench ] wrote {args.json}')

    bad = [r for r in rows if max(r['resid_ipopt_max'], r['resid_slsqp_max']) > 1e-4]
    if bad:
        print('\n🔴 [ bench ] a projector returned INFEASIBLE output -- this is a bug, '
              'and it matters more than the timing above.')
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reps', type=int, default=50, help='timed solves per regime')
    p.add_argument('--warmup', type=int, default=3, help='untimed solves first')
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--dt', type=float, default=1.0)
    p.add_argument('--tau', type=float, default=1.0,
                   help='flow time handed to the HardFlow NLP (1.0 = terminal solve)')
    p.add_argument('--halfspace', default='both-hard',
                   choices=['both-hard', 'top-left-hard', 'top-right-hard'])
    p.add_argument('--ref', default='both', choices=['endpoint', 'iterate', 'both'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--csv', default=None, help='append results to this CSV')
    p.add_argument('--json', default=None, help='write results to this JSON')
    sys.exit(run(p.parse_args()))


if __name__ == '__main__':
    main()
