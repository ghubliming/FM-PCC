"""Mode L switch for the five avoiding eval scripts (Gen15 U18).

    from uav_avoiding_bridge.factory import make_avoiding_env
    env = make_avoiding_env(ObstacleAvoidanceEnv)      # default: the Panda env, byte-identical behaviour

FMPCC_AVOIDING_PLANT=uav   -> UavAvoidingPlant (quadrotor in the scaled pillars_v2 scene) instead of the Panda.
   Refuses unless FMPCC_RUN_MSG contains 'uav': the results land in the SAME layout as the Panda runs, and only
   the message tag on the eval folder keeps them apart (…_msg<tag>). Without it the Panda npz would be overwritten.
Knobs (all optional): FMPCC_AVOID_UAV_HZ (5), FMPCC_AVOID_UAV_FF (1), FMPCC_AVOID_UAV_GAIN (pid_default),
   FMPCC_AVOID_UAV_REPLAY (clock|settle), FMPCC_AVOID_UAV_SCALE / _ALT (read by frame.py).
"""
import os


def _truthy(v):
    return str(v).strip().lower() not in ('', '0', 'false', 'no', 'off')


def make_avoiding_env(default_cls=None):
    mode = os.environ.get('FMPCC_AVOIDING_PLANT', 'panda').strip().lower()
    if mode in ('', 'panda'):
        if default_cls is None:
            from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv as default_cls
        return default_cls()
    if mode != 'uav':
        raise ValueError(f'FMPCC_AVOIDING_PLANT={mode!r}: expected "panda" or "uav"')
    tag = os.environ.get('FMPCC_RUN_MSG', '')
    if 'uav' not in tag.lower():
        raise RuntimeError('FMPCC_AVOIDING_PLANT=uav needs FMPCC_RUN_MSG containing "uav" (e.g. uavpv2s10): '
                           'the UAV results share the Panda result layout and only the message tag separates them.')
    from uav_avoiding_bridge.plant import UavAvoidingPlant
    return UavAvoidingPlant(control_hz=float(os.environ.get('FMPCC_AVOID_UAV_HZ', '5')),
                            feedforward=_truthy(os.environ.get('FMPCC_AVOID_UAV_FF', '1')),
                            gain=os.environ.get('FMPCC_AVOID_UAV_GAIN', 'pid_default'),
                            replay=os.environ.get('FMPCC_AVOID_UAV_REPLAY', 'clock'))
