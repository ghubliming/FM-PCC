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


def sanitize_msg(text):
    """Filesystem-safe token. Deliberately byte-identical to config/avoiding-d3il.py's
    `_sanitize_msg`, so the tag Gen12 puts in its own path matches the `_msg…` suffix the
    other generations get from `watch_plan`."""
    raw = str(text if text is not None else '').strip()
    if not raw:
        return ''
    out = ''.join(ch if (ch.isalnum() or ch in '._-') else '-' for ch in raw)
    while '--' in out:
        out = out.replace('--', '-')
    return out.strip('-._')[:40]


def resolve_run_msg():
    """The run tag for this process, from FMPCC_RUN_MSG (explicit) or FMPCC_MPC_BATCH (auto).

    Gen12 builds savepath itself instead of going through `Parser.mkdir`, so `custom_msg` from
    config/avoiding-d3il.py never reaches this layout — the tag has to be resolved here, and
    identically in the eval and the aggregator, which is what this module is for.

    Why a tag is needed at all: `eval_name`'s `mpc<N>` token has ALWAYS described arm C only
    (it is fed `hf_batch_size`), because the arms-A/B fan was a hardcoded 4 and therefore
    invisible. Now that FMPCC_MPC_BATCH can move it, an `mpc=1` arms-A/B run would otherwise
    land in the same `K…_mpc1_n…` folder as the historic arm-C-only B1 runs, whose DPCC arms
    ran at 4 — two different controllers in one directory.
    """
    msg = sanitize_msg(os.environ.get('FMPCC_RUN_MSG', ''))
    if msg:
        return msg
    mpc_batch = int(os.environ.get('FMPCC_MPC_BATCH', 4))
    return f'mpc{mpc_batch}' if mpc_batch != 4 else ''


def eval_name(flow_steps, activation_threshold, batch_size, n_trials, run_msg=None):
    """The eval-run identity folder: K, activation threshold, MPC candidates, n [, run tag].

    `batch_size` is the ARM-C fan (`hf_batch_size`). The arms-A/B fan (FMPCC_MPC_BATCH) is
    not a token of its own — a non-default value arrives through the `_msg…` suffix, see
    `resolve_run_msg`. Pass `run_msg=''` to force the historic, untagged name.
    """
    msg = resolve_run_msg() if run_msg is None else sanitize_msg(run_msg)
    return (f'K{flow_steps}_thres{activation_threshold:g}'
            f'_mpc{batch_size}_n{n_trials}' + (f'_msg{msg}' if msg else ''))


def eval_root(logbase, dataset, train, evalname, seed):
    """<logbase>/<dataset>/plans/<gen>/<train>/<eval>/<seed>."""
    return os.path.join(logbase, dataset, 'plans', GEN, train, evalname, str(seed))
