import os
import pickle
import glob
import torch
import pdb

from collections import namedtuple

# DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
DiffusionExperiment = namedtuple('Diffusion', 'dataset model diffusion trainer epoch losses')

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        try:
            epoch = int(state.replace('state_', '').replace('.pt', ''))
        except ValueError:
            epoch = -1
        # epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    # print(f'[ utils/serialization ] Loaded config from {loadpath}')
    # print(config)
    return config

def load_losses(*loadpath):
    loadpath = os.path.join(*loadpath)
    if os.path.exists(loadpath):
        losses = pickle.load(open(loadpath, 'rb'))
        # print(f'[ utils/serialization ] Loaded losses from {loadpath}')
        return losses
    else:
        # print(f'[ utils/serialization ] File {loadpath} does not exist')
        return None

def load_diffusion(*loadpath, epoch='latest', device='cuda:0', seed=None, override_args=None):
    print(f'\n[ utils/serialization ] Loading model from {os.path.join(*loadpath)}\n')

    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    # CONFIG-OVERRIDES-PKL (fix_1, 2026-07-14): the pkl PRESERVES training-time params; the eval
    # config is compared against it and reconciled in TWO tiers (see
    # logs_in_develop/config_override_pkl/fix_1/):
    #   - SAMPLING knobs (operating point, safe to change at eval): eval config OVERRIDES the pkl, [INFO].
    #   - identity/architecture keys (must match the checkpoint): pkl value is KEPT to protect the
    #     state_dict; a loud [WARNING] fires if the eval config disagrees.
    _SAMPLING_OVERRIDE_KEYS = {
        'flow_steps_v3', 'ode_inference_steps_v3', 'ode_solver_backend_v3',
        'ode_solver_method_v3', 'ode_solver_rtol_v3', 'ode_solver_atol_v3',
        'ode_solver_step_size_v3', 'meanflow_cfg_omega', 'meanflow_cfg_t_min',
        'meanflow_cfg_t_max', 'condition_guidance_w', 'clip_denoised',
        'diffusion_timestep_threshold',
    }
    if override_args is not None:
        for _k in list(diffusion_config._dict.keys()):
            if not hasattr(override_args, _k):
                continue
            _new, _old = getattr(override_args, _k), diffusion_config._dict[_k]
            try:
                _same = bool(_new == _old)
            except Exception:
                _same = False
            if _same:
                continue
            if _k in _SAMPLING_OVERRIDE_KEYS:
                print(f"[ config->pkl ] INFO  {_k}: train={_old!r} -> eval={_new!r}  (sampling knob; applied)")
                diffusion_config._dict[_k] = _new
            else:
                print(f"[ config->pkl ] WARNING  {_k}: train-pkl={_old!r} vs eval-config={_new!r} -- "
                      f"identity/architecture key; KEEPING the train value to protect the checkpoint "
                      f"(fix the config to match the checkpoint, or retrain).")

    dataset = dataset_config()
    model = model_config().to(device)
    diffusion = diffusion_config(model).to(device)
    trainer = trainer_config(diffusion_model=diffusion, dataset=dataset)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    # print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)

    losses = load_losses(*loadpath, 'losses.pkl')

    return DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, losses)

def check_compatibility(experiment_1, experiment_2):
    '''
        returns True if `experiment_1 and `experiment_2` have
        the same normalizers and number of diffusion steps
    '''
    normalizers_1 = experiment_1.dataset.normalizer.get_field_normalizers()
    normalizers_2 = experiment_2.dataset.normalizer.get_field_normalizers()
    for key in normalizers_1:
        norm_1 = type(normalizers_1[key])
        norm_2 = type(normalizers_2[key])
        assert norm_1 == norm_2, \
            f'Normalizers should be identical, found {norm_1} and {norm_2} for field {key}'

    n_steps_1 = experiment_1.diffusion.n_timesteps
    n_steps_2 = experiment_2.diffusion.n_timesteps
    assert n_steps_1 == n_steps_2, \
        ('Number of timesteps should match between diffusion experiments, '
        f'found {n_steps_1} and {n_steps_2}')
