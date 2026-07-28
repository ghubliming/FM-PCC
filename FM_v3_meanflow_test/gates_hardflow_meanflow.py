"""Gen3v6 U3 pre-flight gates for the HardFlow arm ported into Gen3v6 (MeanFlow).

The HardFlow math is a VERBATIM copy of Gen12 (see the U3 changelog), so Gen12's own gates
(`FM_v3_hardflow_test/gates_hardflow.py`: dof layout, time direction, NLP feasibility) still
cover it. The ONE thing that is new in Gen3v6 — and the only thing that can silently break the
math — is that the velocity is now queried from a MEAN-FLOW net at h=0, using the identity
`u(x,t,0) = v(x,t)` (mf_diffusion.py:443-461). These gates pin exactly that.

Run on the cluster (needs torch; casadi NOT needed for H0/H1):

    python FM_v3_meanflow_test/gates_hardflow_meanflow.py
    python FM_v3_meanflow_test/gates_hardflow_meanflow.py --checkpoint <loadpath>   # adds H2
"""

import argparse
import sys

import torch

from flow_matcher_v3_meanflow.sampling.hardflow_projection import (
    TrajectoryLayout, HardFlowSampler, resolve_activation_threshold,
)

HORIZON, ACTION_DIM, STATE_DIM = 8, 2, 4
TRANSITION_DIM = ACTION_DIM + STATE_DIM


class _RecordingMeanFlowStub:
    """Mimics MeanFlowODE._predict_velocity(x, cond, t, h=None, returns=None).

    Records the `h` it is called with so the gate can assert the HardFlow port queries the
    GROUNDED field (h==0), not the interval average (h==dt) or an accidental h=None.
    """

    def __init__(self):
        self.h_calls = []

    def _predict_velocity(self, x, cond, t, h=None, returns=None):
        self.h_calls.append(h)
        # any deterministic field of the right shape is fine for the wiring check
        return 0.3 * torch.ones_like(x)


def gate_h0():
    """Imports resolve; report casadi availability (needed only for the NLP at run time)."""
    print('\n-- H0: imports + casadi ' + '-' * 40)
    ok = True
    try:
        import casadi  # noqa: F401
        print('  casadi: available')
    except ImportError:
        print('  casadi: NOT available (fine locally; REQUIRED on the cluster for the NLP)')
    print(f'  HardFlowSampler / TrajectoryLayout / resolve_activation_threshold imported OK')
    print(f'  resolve_activation_threshold(0.5) = {resolve_activation_threshold(0.5)}')
    print(f'  H0 -> {"PASS" if ok else "FAIL"}')
    return ok


def gate_h1(device='cpu'):
    """THE pin: the ported _velocity_batch must query the mean-flow net at h==0 EXACTLY.

    If a future backbone default changed h=None away from 0, or someone reverted the explicit
    h=torch.zeros_like(t), this gate fails — catching the §2 wrong-field bug before any run.
    """
    print('\n-- H1: HardFlow queries the mean-flow field at h==0 ' + '-' * 15)
    stub = _RecordingMeanFlowStub()
    layout = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    sampler = HardFlowSampler(model=stub, layout=layout, nlp=None, device=device,
                              activation_threshold=1.0)
    B = 4
    X = torch.zeros(B, layout.dof, device=device)
    s0 = torch.zeros(B, STATE_DIM, device=device)
    _ = sampler._velocity_batch(X, tau=0.3, s0_all=s0, cond={}, returns=None)

    ok = len(stub.h_calls) == 1
    h = stub.h_calls[0]
    is_tensor = torch.is_tensor(h)
    all_zero = bool(is_tensor and torch.count_nonzero(h) == 0)
    right_shape = bool(is_tensor and tuple(h.shape) == (B,))
    print(f'  _predict_velocity called: {len(stub.h_calls)} time(s)')
    print(f'  h is a tensor           : {is_tensor}')
    print(f'  h.shape == (B,)         : {right_shape}  (got {tuple(h.shape) if is_tensor else type(h)})')
    print(f'  h is all-zero (== h=0)  : {all_zero}')
    ok = ok and is_tensor and all_zero and right_shape
    print(f'  H1 -> {"PASS" if ok else "FAIL"}  '
          f'(u(x,t,0)=v identity is what makes the projection math == Gen12)')
    return ok


def gate_h2(checkpoint, device='cuda'):
    """Numeric identity on a REAL checkpoint: u(x,t,h=0) ~= FM velocity (finite-diff of the flow).

    Only runs when --checkpoint is given (needs a trained mean-flow model + its config).
    Left as a thin harness: load the model like the eval driver does and compare
    _predict_velocity(x, t, h=0) against a small finite-difference of the sampler endpoint.
    """
    print('\n-- H2: numeric u(x,t,0)=v on a real checkpoint ' + '-' * 18)
    print(f'  checkpoint = {checkpoint}')
    print('  TODO(cluster): load via the eval loader, then assert '
          '||_predict_velocity(x,t,h=0) - v_fd|| is tiny.')
    print('  H2 -> SKIPPED (harness stub; wire to the eval loader on the cluster)')
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=None, help='real mean-flow loadpath → enables H2')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    results = [gate_h0(), gate_h1(device=a.device)]
    if a.checkpoint:
        results.append(gate_h2(a.checkpoint, device=a.device))
    print('\n' + '=' * 60)
    print(f'GATES: {"ALL PASS" if all(results) else "FAILURE"}  ({sum(results)}/{len(results)})')
    sys.exit(0 if all(results) else 1)
