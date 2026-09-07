"""
U10.1 — run provenance for ENV/CLI-overridden runs. SHARED across generations.

WHY THIS EXISTS
---------------
Several generations resolve experiment knobs from the environment rather than from
literals in `config/*.py` — HFFM_ACT_THRESHOLD / HFFM_BATCH / HFFM_FLOW_STEPS (Gen12,
Gen3v6, Gen3v7), DPCC_THRESHOLD, FMPCC_PROJ_CFG, FMPCC_RUN_MSG, MIX_FILM_MODE (Gen7),
UAV_MIX_* (Gen15), MF_HORIZON / MF_BACKBONE / MF_REPLAN_STEPS (Gen3v6 U10).
That pattern removed a class of half-applied-edit bugs, but it cost provenance:

  * `Parser.snapshot_configs` copies the config module VERBATIM, so a config that reads
    `'horizon': _mf_horizon` snapshots identical bytes whether the run was H8 or H16.
  * `Parser.save` writes `args.json` only when `experiment == 'train'` (setup.py:85),
    so an EVAL run records its resolved args nowhere.
  * Knobs consumed by the eval SCRIPT rather than the config (MF_REPLAN_STEPS,
    HFFM_VARIANTS, UAV_MIX_HF_OFF, ...) never reach `args` at all, so they appear in no
    snapshot — at best in a `_msg` path tag.

The results-folder tokens (H16_, bbunet, K1, T0.5, A0.5, B1, _msg...) remain the primary
record and are unchanged. This module is the SECOND, machine-readable copy that also
captures what the path cannot: which values were set explicitly versus inherited from a
fallback, which yaml was actually read, and which commit produced the numbers.

DESIGN RULES
------------
1. NEVER break a run. Every failure path here is swallowed with a warning — provenance is
   diagnostic, and losing a 6-hour eval to a metadata bug would be absurd.
2. Write next to the results (`args.savepath`), not into the config snapshot directory,
   which `Data_Analysis` discovery code enumerates by name.
3. De-duplicate by CONTENT. The Gen3v6 eval calls parse_args once per halfspace variant
   against the same savepath; three identical payloads should be one file, not three.
   A genuinely different configuration writes a numbered sibling instead of overwriting
   (same convention as `args.json` -> `args_resume_1.json`).
4. OPT-IN. Deliberately NOT exported from `diffuser/utils/__init__.py` and never called
   from `Parser.mkdir`, so adding this file changed the behaviour of exactly zero existing
   runs. A generation opts in with one import and one `provenance.write(...)` call.

Wired into (2026-08-16): Gen3v6 MeanFlow train+eval, Gen3v7 AlphaFlow eval, Gen12 HardFlow
eval, Gen7 mix_visual_aligning eval, Gen15 mix_uav eval.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

SCHEMA = 'fmpcc.run_provenance/1'
FILENAME = 'run_provenance.json'

# Env vars whose PRESENCE is itself the experiment configuration. Recorded verbatim when
# set, and listed under `env_absent` when not — the difference between "A=0.5 because the
# submitter asked for it" and "A=0.5 because the yaml said so" is invisible in the results
# path, and it is exactly the ambiguity that made the Gen12 threshold sweep unreadable.
# Keep this list a superset across generations: a var that is meaningless to one
# generation simply never appears in its `env_set`, and listing it costs nothing.
TRACKED_ENV = (
    # Gen3v6 U10 (MeanFlow horizon / backbone / receding-horizon cadence)
    'MF_HORIZON', 'MF_BACKBONE', 'MF_REPLAN_STEPS', 'MF_FLOW_STEPS',
    # HardFlow arm — Gen12, Gen3v6, Gen3v7
    'HFFM_FLOW_STEPS', 'HFFM_ACT_THRESHOLD', 'HFFM_BATCH', 'HFFM_VARIANTS',
    # shared projection / naming knobs
    'DPCC_THRESHOLD', 'FMPCC_PROJ_CFG', 'FMPCC_RUN_MSG', 'FMPCC_DPCC_THRESHOLD',
    # Gen7 mix_visual_aligning
    'MIX_FILM_MODE', 'MIX_FILM_MODE_MF', 'MIX_FILM_MODE_AF', 'MIX_FILM_MODE_FM',
    'MIX_FILM_MODE_DIFFUSION',
    'FMPCC_BOX_OBS_GUARD', 'FMPCC_BOX_OBS_MAX_OVERLAP_M', 'FMPCC_BOX_HALF_SIDE_M',
    # Gen15 mix_uav
    'UAV_MIX_FLOW_STEPS', 'UAV_MIX_HF_OFF',
    # Gen15 U6 — the af arm's backbone (default flipped 'sit' -> 'unet'), the terminal alpha
    # (>0 = alpha-Flow actually trains the final weights) and the checkpoint the eval deploys.
    # The first two are checkpoint-path keys and the third is a results-path key, but a run
    # that leaves them all at the default is indistinguishable from a pre-U6 one by name; this
    # is where that gets recorded.
    'UAV_MIX_BONE_AF', 'UAV_MIX_AF_ALPHA_END', 'UAV_MIX_EPOCH',
    # Gen14 U10/U11 mix_visual_aligning — bone, perception, alpha schedule, projection budget
    'MIX_BONE', 'MIX_BONE_MF', 'MIX_BONE_AF',
    'MIX_VIS_PRETRAINED', 'MIX_VIS_LR_SCALE', 'MIX_VIS_COND',
    'MIX_TRAIN_STEPS', 'MIX_PROJ_T',
    'MIX_AF_ALPHA_SCHED', 'MIX_AF_ALPHA_INIT', 'MIX_AF_ALPHA_END',
    'MIX_AF_ALPHA_CLAMP', 'MIX_AF_ALPHA_GAMMA',
    # Gen14 U12 — which checkpoint the eval deployed ('best' | 'latest' | <step>). The
    # results path carries '_EP<sel>' only when non-default, so a plain 'best' run is
    # indistinguishable from a pre-U12 one by name alone; this is where that is recorded.
    'MIX_EPOCH',
    'FMPCC_HF_NLP_BACKEND',
    # job-level
    'TRAIN_SEEDS', 'AUTO_RESUME', 'FORCE_OVERWRITE',
)

_SLURM_ENV = ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES')


def _git_state(cwd):
    """Commit + dirty flag. Returns partial info rather than failing."""
    out = {}
    try:
        out['commit'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True,
            timeout=10).stdout.strip() or None
        # --porcelain is empty iff the tree is clean; a dirty tree means the snapshot
        # commit does NOT fully describe the code that ran.
        status = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=cwd, capture_output=True, text=True,
            timeout=10).stdout
        out['dirty'] = bool(status.strip())
    except Exception as exc:                                    # noqa: BLE001
        out['error'] = f'{type(exc).__name__}: {exc}'
    return out


def _file_digest(path):
    try:
        import hashlib
        with open(path, 'rb') as fh:
            return 'sha256:' + hashlib.sha256(fh.read()).hexdigest()
    except Exception:                                           # noqa: BLE001
        return None


def _jsonable(obj):
    """Best-effort coercion; provenance must never die on an exotic value."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return repr(obj)


def build_payload(role, resolved, yaml_path=None, repo_root=None, extra=None):
    """Assemble the record. Split into `config` (the comparison key) and `runtime`
    (volatile: timestamps, job ids), so re-runs of the SAME configuration dedupe."""
    repo_root = repo_root or os.getcwd()
    env_set = {k: os.environ[k] for k in TRACKED_ENV if k in os.environ}
    config = {
        'role': role,
        'resolved': _jsonable(resolved),
        'env_set': env_set,
        # Everything tracked that was NOT set — i.e. every value in `resolved` that came
        # from a yaml/config fallback rather than from the submit line.
        'env_absent': [k for k in TRACKED_ENV if k not in os.environ],
        'git': _git_state(repo_root),
    }
    if yaml_path:
        config['yaml'] = {'path': yaml_path, 'digest': _file_digest(yaml_path)}
    if extra:
        config['extra'] = _jsonable(extra)
    return {
        'schema': SCHEMA,
        'config': config,
        'runtime': {
            'written_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'argv': list(sys.argv),
            'cwd': os.getcwd(),
            'python': sys.version.split()[0],
            'slurm': {k: os.environ[k] for k in _SLURM_ENV if k in os.environ},
        },
    }


def write(savepath, role, resolved, yaml_path=None, repo_root=None, extra=None,
          verbose=True):
    """Write `run_provenance.json` into `savepath`.

    Returns the path written, or None when nothing was written (identical record already
    present, or an error — both are non-fatal by design).
    """
    try:
        payload = build_payload(role, resolved, yaml_path=yaml_path,
                                repo_root=repo_root, extra=extra)
        os.makedirs(savepath, exist_ok=True)
        target = os.path.join(savepath, FILENAME)

        # De-dup / version: identical config -> keep the existing file; different config
        # -> land beside it as _2, _3, ... so a re-run never silently rewrites history.
        candidates = [target]
        idx = 2
        while os.path.exists(candidates[-1]):
            try:
                with open(candidates[-1], 'r') as fh:
                    if json.load(fh).get('config') == payload['config']:
                        if verbose:
                            print(f'[ provenance ] unchanged, kept {candidates[-1]}')
                        return None
            except Exception:                                   # noqa: BLE001
                pass          # unreadable/corrupt -> fall through and write a sibling
            candidates.append(os.path.join(
                savepath, FILENAME.replace('.json', f'_{idx}.json')))
            idx += 1

        with open(candidates[-1], 'w') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        if verbose:
            print(f'[ provenance ] wrote {candidates[-1]}')
        return candidates[-1]
    except Exception as exc:                                    # noqa: BLE001
        # Rule 1: never break a run over metadata.
        print(f'[ provenance ] WARNING: could not write provenance ({type(exc).__name__}: {exc})',
              file=sys.stderr)
        return None
