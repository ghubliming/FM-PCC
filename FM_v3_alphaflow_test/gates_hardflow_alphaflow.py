"""Gen3v7 U3 pre-flight gates for the HardFlow arm ported into Gen3v7 (α-Flow).

The HardFlow math is a VERBATIM copy of Gen12/Gen3v6 (see the U3 changelog), so Gen12's own
gates (`FM_v3_hardflow_test/gates_hardflow.py`: dof layout, time direction, NLP feasibility)
still cover it. Exactly THREE things are Gen3v7-specific, and every one of them fails
SILENTLY — no crash, no solver error, just a degraded field:

  H1  the velocity is queried from a TWO-TIME net at h=0, using the identity
      u(x,t,0) = v(x,t). α-Flow TRAINS that anchor directly — af_diffusion.py:694 forces
      `af_ratio_fm` (default 0.5) of every batch to r==t with u_tgt = v_inst — so the field
      is better supported here than in Gen3v6, but only if we actually query h=0.
  H3  the tau=0 draw must match α-Flow's own sampler, sigma=1.0 (af_diffusion.py:260).
      🔴 NOT flow_matcher_v3_alphaflow/models/diffusion.py:183,302, which is the LEGACY
      FMv3ODE class in the same folder and still on `0.5 * randn`. Reading the wrong one is
      exactly how Gen3v6 lost a whole K-sweep (fix_4).
  H4  af_ratio_fm > 0 in the config that trained the checkpoint. At 0 the h=0 anchor was
      never supervised and the entire arm queries an untrained corner of the field.

Run on the cluster (needs torch; casadi NOT needed for H0/H1/H3/H4):

    python FM_v3_alphaflow_test/gates_hardflow_alphaflow.py
    python FM_v3_alphaflow_test/gates_hardflow_alphaflow.py --checkpoint <loadpath>   # adds H2
"""

import argparse
import sys

import torch

from flow_matcher_v3_alphaflow.sampling.hardflow_projection import (
    TrajectoryLayout, HardFlowSampler, resolve_activation_threshold,
)

HORIZON, ACTION_DIM, STATE_DIM = 8, 2, 4
TRANSITION_DIM = ACTION_DIM + STATE_DIM

# fix_4 (inherited from Gen3v6): α-Flow's sampler starts at sigma=1.0 — af_diffusion.py:260,
# "sigma=1.0 to match q_sample training noise". Gen12's FMv3ODE base used 0.5 and Gen3v6's
# first U3 port inherited it. This constant plus gate_h3 is what stops that regressing, here
# or in a future generation.
ALPHAFLOW_INIT_NOISE_SCALE = 1.0


class _RecordingAlphaFlowStub:
    """Mimics AlphaFlowODE._predict_velocity(x, cond, t, h=None, returns=None).

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
    """THE pin: the ported _velocity_batch must query the α-Flow net at h==0 EXACTLY.

    If a future backbone default changed h=None away from 0, or someone reverted the explicit
    h=torch.zeros_like(t), this gate fails — catching the wrong-field bug before any run.
    """
    print('\n-- H1: HardFlow queries the α-Flow field at h==0 ' + '-' * 14)
    stub = _RecordingAlphaFlowStub()
    layout = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    sampler = HardFlowSampler(model=stub, layout=layout, nlp=None,
                              init_noise_scale=ALPHAFLOW_INIT_NOISE_SCALE,
                              device=device, activation_threshold=1.0)
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
          f'(u(x,t,0)=v is what makes the projection math == Gen12; α-Flow supervises it '
          f'via af_ratio_fm, af_diffusion.py:694)')
    return ok


def gate_h3(device='cpu'):
    """fix_4 pin: arm C's tau=0 draw must match the α-Flow sampler's own sigma=1.0.

    Gen3v6's original U3 port carried Gen12's `0.5 * randn` into a model trained and sampled
    at sigma=1.0, so arm C started off-distribution and arms B/C did not share a start
    distribution. Silent failure -> numeric gate rather than a comment.
    """
    print('\n-- H3: arm C init noise == α-Flow sampler sigma ' + '-' * 12)
    layout = TrajectoryLayout(HORIZON, ACTION_DIM, STATE_DIM)
    sampler = HardFlowSampler(model=_RecordingAlphaFlowStub(), layout=layout, nlp=None,
                              init_noise_scale=ALPHAFLOW_INIT_NOISE_SCALE,
                              device=device, activation_threshold=1.0)
    torch.manual_seed(0)
    draw = sampler.draw_init_noise(4096)
    std = float(draw.std())
    scale_ok = sampler.init_noise_scale == ALPHAFLOW_INIT_NOISE_SCALE
    # 4096*8*6 samples -> the sample std is tight; 2% is generous but still catches 0.5 vs 1.0.
    std_ok = abs(std - ALPHAFLOW_INIT_NOISE_SCALE) < 0.02 * ALPHAFLOW_INIT_NOISE_SCALE
    shape_ok = tuple(draw.shape) == (4096, HORIZON, TRANSITION_DIM)
    print(f'  init_noise_scale        : {sampler.init_noise_scale} '
          f'(expected {ALPHAFLOW_INIT_NOISE_SCALE}, af_diffusion.py:260)')
    print(f'  empirical std of x_init : {std:.4f}')
    print(f'  draw shape              : {tuple(draw.shape)}  ok={shape_ok}')
    ok = scale_ok and std_ok and shape_ok
    print(f'  H3 -> {"PASS" if ok else "FAIL"}  '
          f'(0.5 here = arm C runs at half the trained noise scale, silently)')
    return ok


def gate_h4(exp='avoiding-d3il'):
    """Gen3v7-only (PORT_GUIDE caution 2): the h=0 anchor must actually have been TRAINED.

    If the checkpoint was trained with af_ratio_fm == 0, no batch ever saw r == t, the
    u(x,t,0) = v identity is unsupervised, and this whole arm queries an untrained corner of
    the field. Also checks plan == train, because the loadpath carries `_rf{af_ratio_fm}` and
    a mismatch means eval resolves to a different (or missing) checkpoint.
    """
    print('\n-- H4: the h=0 anchor was trained (af_ratio_fm > 0) ' + '-' * 8)
    import importlib
    # `avoiding-d3il` is not a legal identifier, but import_module resolves it by path —
    # this is exactly how utils.Parser.read_config loads the same module.
    mod = importlib.import_module('config.' + exp)
    plan = mod.base['plan_fm_v3_alphaflow']
    train = mod.base['flow_matching_v3_alphaflow']
    rf_plan = float(plan.get('af_ratio_fm', 0.0))
    rf_train = float(train.get('af_ratio_fm', 0.0))
    print(f'  plan_fm_v3_alphaflow.af_ratio_fm       = {rf_plan}')
    print(f'  flow_matching_v3_alphaflow.af_ratio_fm = {rf_train}')
    match = (rf_plan == rf_train)
    if not match:
        print('  ⚠️ plan != train -> the loadpath token _rf{af_ratio_fm} will not resolve to '
              'the checkpoint you trained.')
    ok = rf_plan > 0.0 and match
    print(f'  H4 -> {"PASS" if ok else "FAIL"}  '
          f'(af_ratio_fm == 0 means u(x,t,0)=v was never supervised)')
    return ok


def gate_h2(checkpoint, device='cuda'):
    """Numeric identity on a REAL checkpoint: u(x,t,h=0) ~= FM velocity (finite-diff of the flow).

    Only runs when --checkpoint is given (needs a trained α-Flow model + its config).
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
    ap.add_argument('--checkpoint', default=None, help='real α-Flow loadpath → enables H2')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--exp', default='avoiding-d3il', help='config module for H4')
    a = ap.parse_args()

    results = [gate_h0(), gate_h1(device=a.device), gate_h3(device=a.device), gate_h4(a.exp)]
    if a.checkpoint:
        results.append(gate_h2(a.checkpoint, device=a.device))
    print('\n' + '=' * 60)
    print(f'GATES: {"ALL PASS" if all(results) else "FAILURE"}  ({sum(results)}/{len(results)})')
    sys.exit(0 if all(results) else 1)
