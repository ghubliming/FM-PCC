"""Hypothesis-A test for fix_2: disable the aux head's contribution to
_predict_velocity at inference, without editing source and without retraining.

Mechanism: monkey-patch `iMeanFlowODE._predict_velocity` to ignore the aux
branch. Trained weights are untouched; only the inference combiner changes.

Usage on the cluster (from repo root):

    # Wraps the existing eval command so the patch is active for that run.
    python logs_in_develop/Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_2/\
disable_aux_at_inference.py <forwarded args to eval_flow_matching_v3_imeanflow.py>

Example (matches Slurm_Codes/sbatch/iMF/eval_imf.sh's invocation):

    python logs_in_develop/Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_2/\
disable_aux_at_inference.py --seed 6

After the patch fires you'll see `[ fix_2/HypA ] aux disabled at inference`
printed once. Then the eval driver runs as usual. Compare the rollout PNG
against the prior with-aux run to see if trajectories are smoother.

To run inside Slurm: copy your existing eval sbatch to a temp file, swap
the `python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py ...`
line with `python logs_in_develop/.../fix_2/disable_aux_at_inference.py ...`
and submit.

This script does NOT modify any tracked source file. To revert, simply
stop running it — the unmodified eval invocation produces the original
(with-aux) behaviour.
"""

import os
import sys
from pathlib import Path

# Repo root on sys.path so the monkey-patched module can be imported.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[4]   # logs_in_develop/Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_2/<this>
sys.path.insert(0, str(_REPO))


def _install_patch():
    from flow_matcher_v3_imeanflow.models import imf_diffusion as _imf

    _original = _imf.iMeanFlowODE._predict_velocity

    def _predict_velocity_no_aux(self, x, cond, t, h=None, returns=None):
        # Re-implement the original but drop the aux term.
        velocity, _aux = self._predict_uv(x, cond, t, h=h, returns=returns)
        if self.returns_condition and returns is not None and self.condition_guidance_w > 0:
            uncond_vel, _ = self._predict_uv(
                x, cond, t, h=h, returns=returns, force_dropout=True
            )
            velocity = (
                (1 + self.condition_guidance_w) * velocity
                - self.condition_guidance_w * uncond_vel
            )
        return velocity   # ← was: velocity + self.sample_aux_weight * aux

    _imf.iMeanFlowODE._predict_velocity = _predict_velocity_no_aux
    print('[ fix_2/HypA ] aux disabled at inference (sample_aux_weight contribution = 0)')


def _run_eval():
    """Re-exec the iMF eval driver with the patch installed. argv passes through."""
    eval_path = os.path.join(_REPO, 'FM_v3_imeanflow_test',
                             'eval_flow_matching_v3_imeanflow.py')
    if not os.path.exists(eval_path):
        print(f'[ fix_2/HypA ] ERROR: eval driver not found at {eval_path}', file=sys.stderr)
        sys.exit(1)

    # Replace argv[0] with the eval script path; rest of argv flows through.
    sys.argv[0] = eval_path

    # Execute as if directly invoked (preserves __name__ == '__main__' semantics).
    import runpy
    runpy.run_path(eval_path, run_name='__main__')


if __name__ == '__main__':
    _install_patch()
    _run_eval()
