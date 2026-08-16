"""Gen12 fix_5 — FMv3ODE-style output path layout for the HardFlow eval.

The FMv3ODE / visual family lays results out as

    logs/<dataset>/plans/<gen>/<TRAIN-NAME>/<EVAL-NAME>/<seed>/results/halfspace_<hv>/<variant>.{npz,png}

where the eval knobs (K, activation threshold, MPC candidate count, n_trials) live in
the **EVAL-NAME folder** — e.g. `...T0.5_D..._mpc4/` — NOT buried under `results/`.

Before fix_5, Gen12 wrote `<exp>/<seed>/results/halfspace_<hv>/K..._thres..._mpc.../<variant>.npz`
(the knobs as a subdir under results). This module builds the FMv3ODE-style path instead,
and both the eval and the aggregator import it so they cannot drift.
"""

import os

GEN = 'flow_matching_v3_hardflow'


def train_name(checkpoint_dir, diffusion_loadpath):
    """Checkpoint identity = the loaded folder's basename.

    e.g. 'H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10'. Uses the direct
    `checkpoint_dir` if set, else the templated `diffusion_loadpath`.
    """
    src = checkpoint_dir if checkpoint_dir else diffusion_loadpath
    return os.path.basename(str(src).rstrip('/'))


def eval_name(flow_steps, activation_threshold, batch_size, n_trials):
    """The eval-run identity folder: K, activation threshold, MPC candidates, n."""
    return (f'K{flow_steps}_thres{activation_threshold:g}'
            f'_mpc{batch_size}_n{n_trials}')


def eval_root(logbase, dataset, train, evalname, seed):
    """<logbase>/<dataset>/plans/<gen>/<train>/<eval>/<seed>."""
    return os.path.join(logbase, dataset, 'plans', GEN, train, evalname, str(seed))
