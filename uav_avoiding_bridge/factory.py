"""Mode L switch for the five avoiding eval scripts (Gen15 U18).

    from uav_avoiding_bridge.factory import make_avoiding_env
    env = make_avoiding_env(ObstacleAvoidanceEnv)      # default: the Panda env, byte-identical behaviour

FMPCC_AVOIDING_PLANT=uav   -> UavAvoidingPlant (quadrotor in the scaled pillars_v2 scene) instead of the Panda.
   Refuses unless FMPCC_RUN_MSG contains 'uav': the results land in the SAME layout as the Panda runs, and only
   the message tag on the eval folder keeps them apart (…_msg<tag>). Without it the Panda npz would be overwritten.
Knobs (all optional): FMPCC_AVOID_UAV_HZ (1), FMPCC_AVOID_UAV_FF (0), FMPCC_AVOID_UAV_VMAX (1.0), FMPCC_AVOID_UAV_GAIN
   (pid_default), FMPCC_AVOID_UAV_REPLAY (clock|settle), FMPCC_AVOID_UAV_SCALE / _ALT (read by frame.py),
   FMPCC_AVOID_UAV_SIDECAR_DIR (dump the plant's per-episode records there on close).
"""
import os
import sys
import time

_SIDECAR_COUNT = [0]                                  # process-wide (fix8)
_PROC_STAMP = time.strftime('%Y%m%d_%H%M%S')
_CTX_ENV = ('FMPCC_RUN_MSG', 'FMPCC_PROJ_CFG', 'AF_SEEDS', 'AF_NTRIALS', 'MF_BACKBONE', 'FMPCC_AVOID_UAV_SCALE',
            'FMPCC_AVOID_UAV_HZ', 'FMPCC_AVOID_UAV_FF', 'FMPCC_AVOID_UAV_VMAX', 'FMPCC_AVOID_UAV_REPLAY', 'SLURM_JOB_ID')


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
    plant = UavAvoidingPlant(control_hz=float(os.environ.get('FMPCC_AVOID_UAV_HZ', '1')),
                             feedforward=_truthy(os.environ.get('FMPCC_AVOID_UAV_FF', '0')),
                             gain=os.environ.get('FMPCC_AVOID_UAV_GAIN', 'pid_default'),
                             replay=os.environ.get('FMPCC_AVOID_UAV_REPLAY', 'clock'),
                             v_max=float(os.environ.get('FMPCC_AVOID_UAV_VMAX', '1.0')))
    side = os.environ.get('FMPCC_AVOID_UAV_SIDECAR_DIR', '')
    if side:                                   # Mode L: dump the plant sidecar when the eval closes the env
        os.makedirs(side, exist_ok=True)
        _close = plant.close
        def _close_and_dump():
            _close()
            # fix8: the evals build a NEW env (= new plant) per geometry x seed, so the counter must be process-wide,
            # and the name must carry the process: MeanFM runs one python call per seed into the same directory.
            # (fix5's per-plant counter wrote `_01` every time -> each close overwrote the previous file.)
            _SIDECAR_COUNT[0] += 1
            k = _SIDECAR_COUNT[0]
            name = f'uav_plant_records_{tag}_{_PROC_STAMP}_p{os.getpid()}_{k:02d}.json'
            ctx = {'close_index': k, 'pid': os.getpid(), 'process_start': _PROC_STAMP, 'argv': list(sys.argv),
                   'env': {e: os.environ.get(e) for e in _CTX_ENV},
                   'note': 'close order inside one process = the eval loops: geometry outer, seed inner '
                           '(avoiding_halfspace_variants order of the yaml); match episodes to the npz obs_all by path'}
            try:
                path = plant.save_records(os.path.join(side, name), extra={'process': ctx})
                print(f'[ uav-plant ] sidecar -> {path}')
            except Exception as exc:           # pragma: no cover
                print(f'[ uav-plant ] sidecar dump failed: {exc}')
        plant.close = _close_and_dump
    return plant
