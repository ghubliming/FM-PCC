"""Minimal closed-loop UAV Mix-ML eval (Gen15; forked from Gen11 Epoch 6's eval_fm_uav.py).

Gen15 change vs Gen11: the ML engine is a CLI/config switch (`--engine {fm,mf,af}`), resolved
through mix_uav/models/engine_registry.py. Everything else in this file — the control loop,
the scene geometry, the DPCC projector wiring, the success criteria, the artifacts — is
Gen11's, unchanged. There is no `if engine == ...` chain anywhere below.

🔴 K (NFE budget) IS A REAL AXIS HERE, unlike in Gen11. Gen11's `flow_steps_v3` was inert in
both directions and Gen15 depends on it, so both paths are closed (see `build_experiment` and
`_load_base_cfg`):
  • the plan block's K now reaches the SAMPLER (engine_registry.apply_nfe on the loaded model,
    plus `num_steps=K` in the Policy's sample kwargs on the two-time arms), and
  • the plan block's K now reaches the FOLDER NAME (`cfg['flow_steps_v3']` → `_uav_eval_tag`).
In Gen11 neither happened: the model sampled at the pickled training value (10) while every
output folder was labelled `K20`. Gen11 is NOT patched by this file — see the Gen15 changelog.

Deliberately NOT forked from the 700-line D3IL/minari-coupled FMv3-ODE eval — that base
is the wrong shape for UAV and impossible to debug. Instead this mirrors the *known-good*
expert control loop (`uav_expert_data_collect/generator.run_trial`) and swaps the expert
trajectory for the trained FM policy. One scene at a time, receding-horizon (MPC) execution.

Multi-rate control (IMPORTANT):
  • physics + PID run every `dt = model.opt.timestep`
  • the FM predicts Δp_des at the DATASET rate (DATASET_HZ from dataset_writer.py)
  → the FM is queried every `decim = round(1/(dt·33))` physics steps; p_des is zero-order
    held between queries while the PID tracks it. This matches how the data was recorded.

Per FM step:
  obs = [p_des | p | v]  (9-D, raw) → policy → first Δp_des (3-D)
  → p_des += Δp_des  (free-running Euler in commanded space)
  → PID tracks p_des for `decim` physics steps → obs updated from new (p, v).

SUCCESS CRITERION (Fix2_metrics, scene-aware):
  • GOAL-PATH scenes (corridor, s_curve, pillars — fixed start + geometry route):
      `success = goal_reached AND safe` — must REACH the route endpoint (final position
      within `--goal-radius`) AND fly cleanly (contact-free + airborne, `min_z > 0.2`).
  • `empty` (RANDOM per-episode start→goal the state-only FM is never told → goal-reaching
      ill-defined): `success = safe` — just stay stable (contact-free + airborne).
  The old contact-free+airborne proxy is always reported as `safe` / `safe_rate`, and
  `goal_reached` / `goal_dist` are reported for every scene regardless. A drone that flies
  around a goal-path scene without reaching the target is NOT a success (the prior global
  definition scored that as success — a bug).

SUCCESS_RELAXED (U7): episodes never terminate early on goal-reach — they always run the
  full fixed FM-step budget. So `success` (which only checks the FINAL position) scores an
  outright FAIL for a rollout that reaches the goal and then drifts/overshoots for the rest
  of a fixed-length episode, identical to one that never got close. `success_relaxed` fixes
  this by treating the goal like a race finish line: a vertical plane (xy line, any z)
  through the goal, oriented perpendicular to the expert path's final approach heading.
  `crossed_line` latches true the first time the drone is EVER on the goal side of that
  line, regardless of what happens afterward. `success_relaxed = crossed_line AND safe`
  (goal-path scenes); `success ⇒ success_relaxed` always. See
  logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/U7_Succes_realaxed/.

No torch/MuJoCo in the Docker dev env — this is cluster-only; here it is syntax-checked.
"""

import os
import re
import sys
import json
import time
import argparse

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

import mix_uav.utils as utils
from mix_uav.sampling.policies import Policy
from diffuser.utils import provenance   # U10.1 — env-override provenance (shared)
# Gen15 U2 — the HardFlow arm. Imported lazily-tolerant: casadi lives in the cluster env, so a
# machine without it can still run every DPCC variant. The ImportError only fires if a
# `hardflow*` variant is actually requested.
from mix_uav.sampling.hardflow_projection import (
    HardFlowPolicy, resolve_activation_threshold, resolve_hf_batch_size,
    hardflow_step_budget,          # HFK1 (2026-08-24)
    hardflow_guard, hardflow_skip_note,   # HFK1c (2026-08-30) — the degeneracy guard
    # [SolverSwap] artifact naming — keeps an SLSQP run from overwriting IPOPT data.
    artifact_variant_label, resolve_nlp_backend)
from mix_uav.models import engine_registry
import mix_uav_test.eval_artifacts as artifacts
from uav_expert_data_collect.dataset_writer import DATASET_HZ   # authoritative 33 Hz source

# Selected ML engine for this process. Set once from --engine in main(); every consumer reads
# it through the registry, so nothing below branches on the value itself.
ENGINE = engine_registry.DEFAULT_ENGINE

def _uav_eval_tag(config, controller, engine=None):
    """Eval-parameter folder name — mirrors args_to_watch_fm_visual_plan style.

    Format:  E{engine}_K{flow_steps}_mpc{B}_{controller}_T{thresh}
    Aligning analogue: K{flow_steps}_M{solver}_T{thresh}_mpc{B}_film{mode}

    Sits BETWEEN the train-identity folder (H8_D...ODE_9D) and the seed,
    so the projection variant (diffuser / dpcc-c) remains a pure leaf name.
    e.g.  mix_uav_mf/H8_D...MeanFlowODE_9D_dp0.5_bbunet / Emf_K4_mpc4_pid_stopgo_T0.5 / 0 / dpcc-c /

    Gen15 adds the `E{engine}` token. It is defence in depth rather than the primary guard —
    the arms already separate two levels up via their own `prefix` and via `D{diffusion}` in
    exp_name, and Gen11 is separated further up still by `logbase` (logs/UAV_MIX vs
    logs/UAV_FM). It exists so the leaf is self-describing when results are lifted out of the
    tree by hand or aggregated across arms.

    🔴 `K{k}` is only truthful because `_load_base_cfg` now injects `flow_steps_v3` into
    `config` from the plan block. In Gen11 that key was never present, so `config.get(...)`
    always fell through to the default and EVERY folder was labelled K20 regardless of the
    actual budget. Do not remove that injection.
    """
    k      = int(config.get('flow_steps_v3', 20))
    mpc_b  = int(config.get('mpc_batch_size', config.get('batch_size', 4)))
    thresh = config.get('diffusion_timestep_threshold', 0.5)
    parts  = [f'E{engine or ENGINE}', f'K{k}', f'mpc{mpc_b}', controller]
    parts.append(f'T{thresh:g}')
    # 🔴 Gen15 Fix_16 — optional run tag, appended last. Two evals that differ ONLY in an
    # environment knob (e.g. FMPCC_SAFE_EPS_MODE=scaled vs legacy) produce an identical
    # folder name and the second SILENTLY OVERWRITES the first. Set FMPCC_UAV_EVAL_TAG to
    # keep A/B runs side by side. Sanitised to [A-Za-z0-9._-] so it cannot escape the path.
    run_tag = os.environ.get('FMPCC_UAV_EVAL_TAG', '').strip()
    if run_tag:
        parts.append(re.sub(r'[^A-Za-z0-9._-]+', '-', run_tag).strip('-'))
    return '_'.join(parts)


def _report_degenerate_dims(dataset, config):
    """🔴 Gen15 Fix_16 — say out loud which channels carry no data, and what they can command.

    A dimension with `min == max` has no scale and no gradient signal; the normalizer has to
    invent a width for it, and that width becomes BOTH the channel's physical output scale and
    the `action_bounds:'auto'` cap the projector derives from `mins`/`maxs`. Before this fix a
    single terse `Constant data in dimension 2` line was the only trace, and the +/-1 m ceiling
    it implied was invisible. Print the actual numbers so the next reader does not have to
    reverse-engineer them from the artifacts.
    """
    try:
        norms = dataset.normalizer.normalizers
    except Exception:
        return
    for key in ('actions', 'observations'):
        n = norms.get(key)
        dims = list(getattr(n, 'degenerate_dims', []) or [])
        if not dims:
            continue
        eps_map = getattr(n, 'degenerate_eps', {}) or {}
        for i in dims:
            w = eps_map.get(i, float('nan'))
            print(f'[ eval ] Fix_16 DEGENERATE {key}[{i}]: constant in the expert data — '
                  f'no training signal for this channel. Widened by eps={w:.3e}; a saturated '
                  f'model output on it commands +/-{w:.3e} data-units/step.')
        if key == 'actions' and config.get('action_bounds', 'auto') == 'auto':
            lb = np.asarray(n.mins, dtype=float); ub = np.asarray(n.maxs, dtype=float)
            print(f'[ eval ] Fix_16 projector action_bounds=auto → lb={np.round(lb, 6)} '
                  f'ub={np.round(ub, 6)}  (degenerate dims now bounded, not +/-1)')


SCENES = ['empty', 'corridor', 's_curve', 'pillars']
# Scenes with a FIXED start + geometry-determined route endpoint → success REQUIRES reaching
# the goal. `empty` is excluded: it has a RANDOM per-episode start→goal that the state-only FM
# is never told (generator._build_traj_and_init), so goal-reaching is ill-defined there — its
# success is stable/safe flight only (Fix2_metrics scene-aware refinement).
GOAL_PATH_SCENES = {'corridor', 's_curve', 'pillars'}
GOAL_RADIUS = 0.30                   # m — secondary goal-reach tolerance (constrained scenes)

# U_13: FIXED per-scene episode budget (steps), replacing the per-trial RANDOM
# n_fm = round(dur * DATASET_HZ). The old random `dur` (generator._build_traj_and_init,
# e.g. corridor U(6,10)s) made the step budget — and therefore whether a fixed-speed policy
# could reach the goal in time — vary trial-to-trial for reasons unrelated to policy quality
# (see logs_in_develop/Gen11/Epoch9_PCC_Constraints/U_13/). This mirrors DPCC d3il-avoiding's
# single fixed `max_episode_length` (aux_repo/dpcc/config/avoiding-d3il.py:68 = 200) + its
# early-terminate-on-success loop (aux_repo/dpcc/scripts/eval.py:203-268).
#
# Value = ceil(scene_max_expert_dur * DATASET_HZ * SAFETY), SAFETY=1.2 so a policy flying up
# to 1.2x SLOWER than the SLOWEST expert of that scene can still finish. The budget is already
# keyed to the SLOWEST (max-dur) expert, so 1.2x is ample headroom without ballooning compute
# on the many rollouts that miss and run the FULL budget (jobs already brush the 24h SLURM
# limit — Fix_11). `dur` is still sampled (for the fixed goal endpoint + initial pose) but NO
# LONGER sets the budget. CLI --max-episode-length overrides all scenes with one value
# (DPCC-style single knob); yaml `max_episode_length` (scalar or per-scene dict) also overrides.
SCENE_MAX_EPISODE_LENGTH = {
    'empty':    504,   # dur = max(4, sep/0.4) ≤ ~12.7s → 419 * 1.2
    'corridor': 396,   # dur U(6,10)s,  max 10s → 330 * 1.2
    'pillars':  634,   # dur U(10,16)s, max 16s → 528 * 1.2
    's_curve':  871,   # dur U(16,22)s, max 22s → 726 * 1.2
}

# Fix_8: tracks which seed_dirs have been config-snapshotted THIS PROCESS (not "ever on disk").
# See _run_variant's snapshot call — a filesystem-existence check would let a stale, wrong-yaml
# snapshot from before a `snapshot_configs` fix persist forever across job re-runs.
_SNAPSHOTTED_DIRS = set()

# ── Div_Abort: divergence detection + episode abort ──────────────────────────
# A quadrotor that has lost control does not merely fail the task — it flies an ABNORMAL
# ROUTE: it leaves the volume every expert trajectory of its scene lives in, and it gets
# there fast. Nothing in the loop stopped that, so the episode burned its whole step budget
# while the drone tumbled. This guard ends such an episode on the step the flight is provably
# lost and records WHEN / WHERE / WHY into every artifact.
#
# What this guard is NOT
#   • NOT what keeps the plots readable. `p_des` excursions used to autoscale the
#     `*_mpc_foresight.svg` into two near-empty panels dominated by one hike. That is fixed
#     independently by `eval_artifacts.view_window()`, which scales every panel to the FLOWN
#     PATH and lets `p_des` widen it by at most VIEW_MAX_GROW core spans — a commanded point
#     at -600 m can no longer shrink a plot, aborted or not. Aborting is about not burning
#     800 steps of compute on a dead aircraft, and about scoring it as the miss it is.
#   • NOT a constraint check. Leaving the declared workspace box is a NORMAL, measured
#     violation (`_exec_constraint_violations`). That box also SHRINKS per geo ablation, so
#     it is deliberately NOT used here: the envelope below is a fixed physical property of
#     the SCENE, identical across every projection variant, so an abort can never be an
#     artefact of which ablation happens to be running.
#   • Does NOT look at `p_des`. The commanded-point lead (`|p_des - p| > 5 m`) was tried and
#     REMOVED — it is direction-blind. `p_des` 5 m BELOW the drone means required thrust has
#     gone negative (saturation → free fall); 5 m ABOVE means an ordinary lagging climb the
#     drone recovers from at max climb accel; 5 m sideways means a 64° tilt, aggressive but
#     upright and flying. The old check aborted all three identically. Loss of control is now
#     read off the AIRCRAFT state only.
#
# What the abort DOES do for the plots: the clamp above rescues `p_des` excursions, but it
# cannot rescue a DRONE excursion — the flown path IS the core the window scales to, so a
# fly-away sets a huge percentile band and compresses the real flight anyway. Ending the
# episode on the step the aircraft is lost keeps the core band on the arena. That is why
# `off_route` and `overspeed` fire INDEPENDENTLY (OR, not AND): either one on its own is
# already a sign the rollout has failed, and either one on its own already wrecks the SVG.
#
# Env overrides (all optional; the defaults are what the cluster runs):
#   FMPCC_UAV_DIVERGENCE_ABORT=0  → disable entirely (exact pre-Div_Abort behaviour)
#   FMPCC_UAV_DIV_SLACK_M         → how far outside its scene envelope counts as abnormal
#   FMPCC_UAV_DIV_SPEED_MS        → what counts as "very fast"
#   FMPCC_UAV_DIV_MAP_XY_M / _MAP_Z_M → the hard off-the-world bound
DIVERGENCE_ABORT = os.environ.get('FMPCC_UAV_DIVERGENCE_ABORT', '1').lower() not in ('0', 'false', 'no')

# Per-scene FLIGHT ENVELOPE — the box containing every expert trajectory of that scene.
# Sources: d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_<scene>.xml (floor
# plane spans ±10 m; walls are 1.5 m tall) and uav_expert_data_collect/generator.py (altitude
# drawn U(0.90, 1.30) at the start, U(0.70, 1.10) at the goal):
#   empty     start/goal drawn in U(-1.8, 1.8) on x and y; no walls, no obstacles.
#   corridor  path spans x = ±2.8; wall inner faces at y = ∓0.45.
#   pillars   path spans x = ±3.2; outermost trained channel centred y = ±1.11.
#   s_curve   path spans x = ±3.2; wall corridor band |y| <= 1.25.
SCENE_FLIGHT_ENVELOPE = {
    'empty':    ((-1.8, -1.80, 0.70), (1.8, 1.80, 1.30)),
    'corridor': ((-2.8, -0.45, 0.70), (2.8, 0.45, 1.30)),
    'pillars':  ((-3.2, -1.11, 0.70), (3.2, 1.11, 1.30)),
    's_curve':  ((-3.2, -1.25, 0.70), (3.2, 1.25, 1.30)),
}
# Unknown scene → the union of the four above, so a new scene can never abort spuriously
# before someone measures its real envelope and adds a row.
SCENE_FLIGHT_ENVELOPE_DEFAULT = ((-3.6, -1.80, 0.70), (3.6, 1.80, 1.30))

# `off_route`: how far outside its scene envelope the drone has to be. 2.0 m is wider than the
# whole corridor/s_curve wall gap, and puts the ceiling trigger at z = 3.30 m — 1.8 m clear of
# the tallest wall (1.5 m) and 2.0 m above any altitude the expert ever flies. Fires ALONE.
DIV_ENVELOPE_SLACK_M = float(os.environ.get('FMPCC_UAV_DIV_SLACK_M', '2.0'))
# `overspeed`: the expert covers <= 8 m of path in 6-22 s — ~0.4-0.9 m/s mean, well under
# 2 m/s peak. Fires ALONE, so it is set above every speed the arena can produce innocently:
# a free fall from the top of the altitude draw (1.30 m) lands at sqrt(2*9.81*1.30) = 5.05 m/s,
# so 6.0 m/s cannot be reached by merely dropping out of cruise — it takes powered divergence.
# ~3x the expert peak. THIS is the threshold to sanity-check first on the cluster (§5).
DIV_SPEED_MAX_MS = float(os.environ.get('FMPCC_UAV_DIV_SPEED_MS', '6.0'))
# Hard off-the-world bound: the MuJoCo floor plane is 10 x 10 m. Fires on POSITION ALONE (no
# speed term) — a drone that coasts out over the void and slows down is still gone.
DIV_MAP_XY_M = float(os.environ.get('FMPCC_UAV_DIV_MAP_XY_M', '10.0'))
DIV_MAP_Z_M = float(os.environ.get('FMPCC_UAV_DIV_MAP_Z_M', '10.0'))


def _flight_envelope(scene):
    """(lb, ub) of the box every EXPERT trajectory of `scene` stays inside.

    A fixed physical property of the scene. Deliberately independent of `geo_config`: its
    `workspace_bounds` shrink per geo ablation (`geo_bounds_only`, tightened `combined_*`),
    which would make the same flight abort under one variant and survive under another.
    """
    lb, ub = SCENE_FLIGHT_ENVELOPE.get(scene, SCENE_FLIGHT_ENVELOPE_DEFAULT)
    return np.array(lb, dtype=float), np.array(ub, dtype=float)


def _check_divergence(p, v, envelope_lb, envelope_ub, quat=None):
    """First lost-the-aircraft condition this state trips → (reason, detail); else (None, '').

    Reads the AIRCRAFT state only — p, v, attitude. Never `p_des`. `reason` is a short
    greppable tag that lands in results.json / the npz / the eval log / the foresight SVG;
    `detail` is the human sentence.
    """
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
        return 'nan_state', 'non-finite p / v — the integrator blew up'

    if (np.any(np.abs(np.asarray(p, dtype=float)[:2]) > DIV_MAP_XY_M)
            or p[2] > DIV_MAP_Z_M or p[2] < -0.5):
        return 'off_map', (f'p={np.round(p, 2).tolist()} is off the map — the floor plane '
                           f'spans ±{DIV_MAP_XY_M:.0f} m and the ceiling trigger is '
                           f'{DIV_MAP_Z_M:.0f} m; the drone has left the world')

    # off_route: somewhere no expert trajectory of this scene ever goes — too far from the
    # walls / pillars the route is defined by. Independent of speed: a drone that drifts far
    # off-route slowly has still failed, and its trace still sets the plot scale.
    lo = np.asarray(envelope_lb, dtype=float) - DIV_ENVELOPE_SLACK_M
    hi = np.asarray(envelope_ub, dtype=float) + DIV_ENVELOPE_SLACK_M
    off_route = [ax for ax, c, l, h in zip('xyz', p, lo, hi) if c < l or c > h]
    if off_route:
        return 'off_route', (f'p={np.round(p, 2).tolist()} is outside the scene flight envelope '
                             f'{np.round(lo, 2).tolist()}..{np.round(hi, 2).tolist()} on '
                             f'{"/".join(off_route)} (expert envelope ⊕ '
                             f'{DIV_ENVELOPE_SLACK_M:.1f} m) — too far off the trained route')

    # overspeed: independent of position. A speed no expert ever flies is itself the failure
    # signature, and by the time it shows up the trace is already leaving the arena.
    speed = float(np.linalg.norm(v))
    if speed > DIV_SPEED_MAX_MS:
        return 'overspeed', (f'|v|={speed:.2f} m/s > {DIV_SPEED_MAX_MS:.1f} m/s — the expert '
                             f'cruises at 0.4-0.9 m/s and a free fall from cruise altitude '
                             f'lands at 5.05 m/s; this is powered divergence')

    if quat is not None:
        q = np.asarray(quat, dtype=float).reshape(-1)
        if q.size == 4 and np.all(np.isfinite(q)):
            cos_tilt = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])   # body z-axis · world z
            if cos_tilt < 0.0:
                return 'inverted', f'body z-axis · world z = {cos_tilt:.2f} < 0 — the drone is upside down'
    return None, ''


def parse_args():
    p = argparse.ArgumentParser(description='Closed-loop UAV Mix-ML evaluation.')
    p.add_argument('--engine', type=str, default=engine_registry.DEFAULT_ENGINE,
                   choices=list(engine_registry.ENGINE_KEYS),
                   help="ML engine: 'fm' (Gen11 flow matching), 'mf' (Gen3v6 MeanFlow), "
                        "'af' (Gen3v7 alpha-Flow). Selects the config block AND the checkpoint tree.")
    p.add_argument('--flow-steps', type=int, default=None, metavar='K',
                   help='Override the NFE budget K (flow_steps_v3) from the plan block. K is a '
                        'first-class axis in Gen15 and appears in the output path as K{n}, so '
                        'distinct-K runs never collide. MATCHED BUDGET OR NOTHING: when comparing '
                        'arms, pass the SAME K to every one of them.')
    p.add_argument('--scene', type=str, default='all', choices=['all', *SCENES],
                   help="Scene(s) to eval: 'all' runs each scene and rolls up SUMMARY.json.")
    p.add_argument('--seed', type=int, default=None,
                   help='Trained-model checkpoint seed to load. Default: seed from config/uav_projection.yaml.')
    p.add_argument('--n-trials', type=int, default=None,
                   help='Closed-loop rollouts per scene. Default: n_trials from config/uav_projection.yaml.')
    p.add_argument('--goal-radius', type=float, default=GOAL_RADIUS,
                   help='Goal-reach tolerance (m). success now REQUIRES goal_dist < this (Fix2_metrics).')
    p.add_argument('--max-episode-length', type=int, default=None,
                   help='U_13: FIXED step budget for ALL scenes/trials (overrides the per-scene '
                        'SCENE_MAX_EPISODE_LENGTH defaults). Mirrors DPCC avoiding max_episode_length. '
                        'Episodes early-stop on goal-reach; goal-path scenes that never reach run the '
                        'full budget. Default: per-scene value.')
    p.add_argument('--epoch', type=str, default='best', help="Checkpoint to load: 'best' (lowest val loss; the default and the ONLY choice that matches the D3IL arms), "
                   "'latest' (last PERIODIC save -- with save_freq=n_train_steps//5 that is "
                   "step 80000 of 100000, i.e. 80%% trained, NOT the final model), or an int.")
    p.add_argument('--projection', type=str, default='fm_only',
                   help="Projection variant for the output subfolder. 'fm_only' (state-only FM, no DPCC); "
                        "DPCC variants (dpcc-c, …) slot in here when Phase-3 lands.")
    p.add_argument('--record', type=str, default='none', choices=['none', 'gif', 'all'],
                   help="Overhead-render GIFs per rollout. 'none' (default, fast) adds ~0 overhead; "
                        "'gif'/'all' render frames and write diagnostics/rollout_<r>.gif.")
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_known_args()


def build_experiment(scene, seed, epoch, device, flow_steps=None):
    """Resolve + load the trained model & dataset (the per-variant Policy is built later)."""
    class Parser(utils.Parser):
        dataset: str = 'uav'              # overridden on the instance below
        config: str = 'config.uav_mix'
    p = Parser()
    p.dataset = f'uav-{scene}'            # → data branch + output path segregation
    args = p.parse_args(experiment=engine_registry.experiment_name(ENGINE), seed=seed)

    ep = epoch if epoch in ('latest', 'best') else int(epoch)
    # CONFIG-OVERRIDES-PKL (2026-07-13): pass parsed config args so same-named pickled
    # diffusion kwargs are overridden by the current config (warn-on-change). See
    # logs_in_develop/config_override_pkl/CHANGELOG_config_overrides_pkl.md
    experiment = utils.load_diffusion(args.savepath, epoch=ep, device=device, override_args=args)
    diffusion = experiment.diffusion

    # 🔴 Gen15 — pin K onto the LOADED model. This is one of the two halves of the Gen11 K bug:
    # `override_args` here is the TRAIN block's args, which carry no `flow_steps_v3`, so the
    # reconciliation loop in load_diffusion skips the key entirely and the pickled TRAINING
    # value (10) survives. The plan block's K therefore never reached the sampler in Gen11.
    # Setting it explicitly after load closes that path for every engine.
    if flow_steps is not None:
        k = engine_registry.apply_nfe(diffusion, flow_steps, engine=ENGINE)
        print(f'[ eval ] engine={ENGINE}  NFE budget K={k} pinned on the loaded model '
              f'(flow_steps_v3 / ode_inference_steps_v3)')

    return diffusion, experiment.dataset, args, int(getattr(args, 'horizon', 8))


def _resolve_active_geo_matches(scene, cfg):
    """All `geo_constraint_variants` entries active for this scene (Fix_4 pattern): matched by
    `scene:` field (falls back to `name`) AND listed in `active_geo_variants` (or all, if that
    key is null). Returns a list — 0 (no active geo for this scene), 1 (the common case), or
    many (Fix_6: eval_scene runs every one of them in a single job)."""
    _all_geo = [g for g in (cfg.get('geo_constraint_variants') or []) if 'name' in g]
    _active = cfg.get('active_geo_variants')
    return [g for g in _all_geo
            if g.get('scene', g['name']) == scene and (_active is None or g['name'] in _active)]


def _apply_geo_entry(cfg, scene, entry):
    """Return a COPY of cfg with one geo_constraint_variants entry's constraint_types/geometry
    applied (or the dynamics-only global fallback if entry is None), plus its geo_tag (Fix_1).
    Shared by load_pcc_config (single-match) and eval_scene's multi-match loop (Fix_6)."""
    cfg = dict(cfg)
    if entry is not None:
        cfg['constraint_types']      = list(entry.get('constraint_types', cfg['constraint_types']))
        cfg['workspace_bounds']      = entry.get('workspace_bounds', None)
        cfg['halfspace_constraints'] = entry.get('halfspace_constraints', [])
        cfg['obstacle_constraints']  = entry.get('obstacle_constraints', [])
        print(f"[ eval ] E9 geo '{scene}' ← variant '{entry['name']}': "
              f"constraint_types={cfg['constraint_types']} "
              f"(bounds={cfg['workspace_bounds'] is not None}, "
              f"hs={len(cfg['halfspace_constraints'])}, obs={len(cfg['obstacle_constraints'])})")
    elif cfg.get('geo_constraint_variants'):
        print(f"[ eval ] E9: scene '{scene}' has no active geo variant → dynamics-only fallback")

    # E9 fix1: `geo_tag` — a second, swappable output-path axis mirroring the old avoiding-task
    # `results/halfspace_<halfspace_variant>/` folder level. Encodes WHICH geometry/constraint
    # combo produced a run (resolved geo entry name + its actually-active constraint_types),
    # so re-running the same scene under a different constraint_types subset (e.g. an ablation
    # like obstacles-only vs the full stack) lands in a DIFFERENT folder instead of overwriting
    # the previous run. `empty` (constraint_types=[]) tags as '<scene>_unconstrained'.
    _ctypes = cfg.get('constraint_types') or []
    cfg['geo_tag'] = f'{scene}_unconstrained' if not _ctypes else f"{scene}_{'+'.join(sorted(_ctypes))}"
    return cfg


def _load_base_cfg(scene, seed):
    """Build the merged config WITHOUT resolving per-scene geometry (Fix_6): yaml load +
    defaults + eval control params (batch_size, thresholds, U4 knobs, logging) from the
    plan_flow_matching_v3_uav block in config/uav.py. Geo resolution is a separate step
    (`_resolve_active_geo_matches` + `_apply_geo_entry`) so callers can run it once
    (`load_pcc_config`, single-match) or in a loop (`eval_scene`, Fix_6, possibly multi-match)."""
    import yaml

    yaml_path = os.path.join(_REPO, 'config', 'uav_projection.yaml')
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f'[ eval ] {yaml_path} not found → diffuser-only fallback')
        cfg = {}
    cfg.setdefault('write_to_file', True)
    cfg.setdefault('projection_variants', ['diffuser'])
    cfg.setdefault('constraint_types', ['dynamics'])
    cfg.setdefault('dt', 1.0)
    cfg.setdefault('diffusion_timestep_threshold', 0.5)
    cfg.setdefault('enlarge_constraints', 0.0)
    cfg.setdefault('workspace_bounds', None)
    cfg.setdefault('halfspace_constraints', [])
    cfg.setdefault('obstacle_constraints', [])
    cfg.setdefault('inflation', {'r_drone': 0.0, 'margin_base': 0.0})
    cfg.setdefault('action_bounds', None)

    class PlanParser(utils.Parser):
        dataset: str = 'uav'
        config: str = 'config.uav_mix'
    pp = PlanParser()
    pp.dataset = f'uav-{scene}'
    plan_args = pp.parse_args(experiment=engine_registry.experiment_name(ENGINE, plan=True), seed=seed)
    cfg['mpc_batch_size']               = int(getattr(plan_args, 'mpc_batch_size', getattr(plan_args, 'batch_size', 4)))
    cfg['diffusion_timestep_threshold'] = float(getattr(plan_args, 'diffusion_timestep_threshold', 0.5))

    # 🔴 Gen15 — the second half of the Gen11 K bug. `_uav_eval_tag` reads `flow_steps_v3` off
    # this cfg dict, which is built from uav_projection.yaml (where the key does not exist) —
    # so in Gen11 every folder was labelled with the `K20` default no matter what K ran, and
    # different-K runs silently overwrote each other. Inject it from the plan block.
    cfg['flow_steps_v3']                = int(os.environ.get(
        'UAV_MIX_FLOW_STEPS', getattr(plan_args, 'flow_steps_v3', 20)))
    cfg['engine']                       = ENGINE

    # ── Gen15 U2: the HardFlow arm is declared in Gen15's OWN config, never in the yaml ──────
    # `config/uav_projection.yaml` is SHARED READ-ONLY with Gen11 (init plan §1.9 / drift-scan
    # §11.4). Adding `hardflow_new` to its `projection_variants` would make Gen11's next eval
    # try to run an arm Gen11 has no code for. So the extra variants and the solver knobs come
    # from `plan_mix_uav_<engine>` instead. The CONSTRAINTS still come from the shared yaml —
    # which is the half that has to match for the comparison to mean anything.
    cfg['hardflow'] = dict(getattr(plan_args, 'hardflow', {}) or {})
    # [HFK1c 2026-08-30 / R4] `HFFM_ACT_THRESHOLD` is the per-job override for A, and until now
    # it was wired into Slurm_Codes/sbatch/{MeanFlow,AlphaFlow,mix_visual_aligning} but NOT the
    # UAV path — `config/uav_mix.py` hardcoded 0.5, so A was the one HardFlow knob this
    # generation could not sweep. It has to be settable here because the supported way to
    # recover the projector-only control is `A=0.0 at K>=5` (terminal-only at ANY K), which is
    # the clean replacement for the low-K degenerate rows this guard now blocks. Same env name
    # and same polarity as the sibling generations. See AUDIT_20260830 §6.
    _hf_env_A = os.environ.get('HFFM_ACT_THRESHOLD')
    if _hf_env_A not in (None, ''):
        cfg['hardflow']['activation_threshold'] = _hf_env_A
        print(f'[ eval ][hardflow] HFFM_ACT_THRESHOLD={_hf_env_A} → overriding '
              f'activation_threshold for this job')
    _hf_variants = list(getattr(plan_args, 'hardflow_variants', []) or [])
    if not engine_registry.get(ENGINE)['supports_hardflow']:
        # e.g. the `diffusion` arm: HardFlow's NLP needs v = f(x, t) and a DDPM has no velocity
        # field. Drop the arm rather than crash inside the sampler hours into a job.
        if _hf_variants:
            print(f"[ eval ] engine '{ENGINE}' does not support HardFlow (no instantaneous "
                  f'velocity field) → dropping {_hf_variants}')
        _hf_variants = []
    if os.environ.get('UAV_MIX_HF_OFF'):
        _hf_variants = []
        print('[ eval ] UAV_MIX_HF_OFF set → HardFlow arm disabled for this run')
    # [HFK1c 2026-08-30] Degeneracy guard. Same principle as the `supports_hardflow` drop
    # above — drop the arm at config time rather than crash (or, worse, silently produce an
    # uncitable row) inside the sampler hours into a job. A DEGENERATE arm runs no HardFlow
    # arithmetic at all, so there is nothing for it to measure; see
    # logs_in_develop/aggregated_hardflow_lowK/AUDIT_20260830_*.md
    if _hf_variants:
        # Resolve A through the SAME fallback chain the policy build uses (line ~1633), or
        # the guard would judge a different arm than the one that runs.
        _hf_A = resolve_activation_threshold(cfg['hardflow'].get(
            'activation_threshold', cfg.get('diffusion_timestep_threshold', 0.5)))
        _hf_ok, _hf_reason, _hf_tier, _, _hf_ngen, _ = hardflow_guard(
            int(cfg['flow_steps_v3']), _hf_A)
        if not _hf_ok:
            print(f'[ eval ][hardflow][BLOCKED] dropping {len(_hf_variants)} HardFlow '
                  f'variant(s) {_hf_variants}: {_hf_reason}')
            cfg['hardflow_skipped'] = {'variants': list(_hf_variants),
                                       'reason': _hf_reason,
                                       'tier': _hf_tier,
                                       'n_genuine': int(_hf_ngen),
                                       'K': int(cfg['flow_steps_v3']),
                                       'A': float(_hf_A)}
            _hf_variants = []
    if _hf_variants:
        _existing = list(cfg.get('projection_variants') or [])
        cfg['projection_variants'] = _existing + [v for v in _hf_variants if v not in _existing]
        print(f'[ eval ] HardFlow arm: +{len(_hf_variants)} variants {_hf_variants} '
              f'(from config/uav_mix.py, NOT the shared yaml)')

    cfg['control_hz']                   = float(getattr(plan_args, 'control_hz', DATASET_HZ))
    cfg['behavior_log']                 = bool(getattr(plan_args, 'behavior_log', True))

    # E8 (Epoch8) — observation layout + tracker selection. Defaults = E7 (p_des / pid).
    cfg['cond_mode']                    = str(getattr(plan_args, 'cond_mode', 'p_des'))
    cfg['controller']                   = str(getattr(plan_args, 'controller', 'pid'))
    # U6: MJX predictive-sampling params (replaces gRPC mjpc_task_id/planner_steps).
    cfg['mjx_n_samples']                = int(getattr(plan_args, 'mjx_n_samples', 16))
    cfg['mjx_horizon']                  = float(getattr(plan_args, 'mjx_horizon', 0.3))
    cfg['mjx_n_improve']                = int(getattr(plan_args, 'mjx_n_improve', 5))
    cfg['mjx_vel_weight']               = float(getattr(plan_args, 'mjx_vel_weight', 0.1))

    return cfg


def load_pcc_config(scene, seed):
    """Merged eval config matching the avoiding-d3il.py pattern (single-geo-match convenience
    wrapper around `_load_base_cfg` + `_resolve_active_geo_matches` + `_apply_geo_entry`).

    Raises if the scene has MORE THAN ONE active geo_constraint_variants entry — that case is
    handled by `eval_scene`'s Fix_6 loop, which runs every active entry in one invocation
    instead of erroring. Use `eval_scene` (not this function) when a scene may have several
    active variants (e.g. testing dynamics_only vs dynamics_bounds_only vs combined_1)."""
    cfg = _load_base_cfg(scene, seed)
    _matches = _resolve_active_geo_matches(scene, cfg)
    if len(_matches) > 1:
        raise ValueError(
            f"E9: scene '{scene}' matches MULTIPLE active geo_constraint_variants "
            f"({[g['name'] for g in _matches]}) — load_pcc_config() only resolves ONE. "
            f"eval_scene() runs all of them in one job (Fix_6) — call that instead, or narrow "
            f"active_geo_variants in config/uav_projection.yaml to exactly one for this scene.")
    return _apply_geo_entry(cfg, scene, _matches[0] if _matches else None)


# ── DPCC projector — copied from fm_visual_aligning_test/eval_fm_visual_aligning.py and
#    adapted to the UAV 12-D transition. Only the DYNAMICS constraint is active this epoch;
#    bounds/halfspace/obstacle blocks are kept verbatim as PLACEHOLDERS (fire only if their
#    config keys are enabled — they are not this epoch). ────────────────────────────────────

class ProjectorNormalizer:
    """Wrap obs + act LimitsNormalizers into the dict Projector('states_actions') expects
    (verbatim from the visual-aligning eval)."""
    def __init__(self, obs_normalizer, act_normalizer):
        self.normalizers = {'observations': obs_normalizer, 'actions': act_normalizer}


def _exec_constraint_violations(obs_traj, config):
    """E9 exec-time violation metrics: check the FLOWN path against the RAW geometry ⊕ r_drone
    (physical collision truth — NOT the planning margin, which includes margin_base + enlarge).

    obs layout is [p_des | p | (v)] in both cond_modes → actual position p is cols 3:6.
    Returns (collision_free: bool, n_violations: int, total_violations: float).
    A step counts once toward n_violations if it violates ANY active spatial family; the
    magnitude sum accumulates every family's penetration depth (metres).
    """
    ctypes = config.get('constraint_types', []) or []
    spatial = {'geo_bounds', 'halfspace', 'obstacles'} & set(ctypes)
    if not spatial or not obs_traj:
        return True, 0, 0.0
    P = np.asarray(obs_traj, dtype=float)[:, 3:6]          # executed p, (T,3)
    r_drone = float((config.get('inflation') or {}).get('r_drone', 0.0))

    n_steps_viol = 0
    total = 0.0
    ws = config.get('workspace_bounds')
    halfspaces = config.get('halfspace_constraints', []) if 'halfspace' in spatial else []
    obstacles  = config.get('obstacle_constraints', [])  if 'obstacles'  in spatial else []
    for p in P:
        step_pen = 0.0
        if 'geo_bounds' in spatial and ws is not None:
            lb = np.array(ws['lb'], dtype=float); ub = np.array(ws['ub'], dtype=float)
            # physical box is raw lb/ub; body clears when p ∈ [lb+r, ub-r]
            below = (lb + r_drone) - p; above = p - (ub - r_drone)
            step_pen += float(np.clip(below, 0, None)[np.isfinite(lb)].sum())
            step_pen += float(np.clip(above, 0, None)[np.isfinite(ub)].sum())
        for hs in halfspaces:
            triple, x_active = _normalize_halfspace(hs)
            if x_active is not None and not (x_active[0] <= p[0] <= x_active[1]):
                continue                                    # wall not live at this x
            (x1, y1), (x2, y2), side = triple[0], triple[1], triple[2]
            dx, dy = x2 - x1, y2 - y1
            nrm = np.hypot(dx, dy)
            if nrm < 1e-9:
                continue
            nx, ny = (-dy / nrm, dx / nrm)                  # left normal of the segment
            signed = nx * (p[0] - x1) + ny * (p[1] - y1)    # + on the 'above'/left side
            feasible = signed if side == 'above' else -signed
            step_pen += max(0.0, r_drone - feasible)        # clear when feasible >= r_drone
        for ob in obstacles:
            dims = ob['dimensions']; c = ob['center']
            didx = [{'x': 0, 'y': 1, 'z': 2}[d] if isinstance(d, str) else int(d) for d in dims]
            dist = float(np.linalg.norm(p[didx] - np.asarray(c, dtype=float)))
            if ob['type'] == 'sphere_outside':
                step_pen += max(0.0, (ob['radius'] + r_drone) - dist)
            else:                                            # sphere_inside
                step_pen += max(0.0, dist - (ob['radius'] - r_drone))
        if step_pen > 1e-9:
            n_steps_viol += 1
            total += step_pen
    return (n_steps_viol == 0), int(n_steps_viol), float(total)


def _realized_homotopy(scene, obs_traj):
    """Fix_12: the homotopy class the drone ACTUALLY flew, read off the flown path.

    The `homotopy` label cycled through trials is the EXPERT route's class, but the FM
    policy is unconditioned and never tracks that route, so the commanded label need not
    match what was flown — this reads the realized class from the flown positions instead.

    Returns None only for scenes whose homotopy set has a SINGLE class, where there is
    nothing to disambiguate: s_curve (['default']) and empty (['N/A']). For those the
    commanded `homotopy` already tells the whole story, so a `homotopy_flown` field would be
    noise (Fix_12 follow-up: earlier this returned None for corridor too, leaving a
    confusing `homotopy_flown: null` next to a meaningful `homotopy: L/C/R`).

    pillars ((L/R)³): at each pillar column x∈{-2,0,2}, which side (y<0 → 'L' matching
        trajectories._Y_L<0, else 'R') the path crossed, interpolated at the first crossing;
        '?' if the column was never reached.
    corridor (L/C/R): nearest expert channel (trajectories.CORRIDOR_CHANNELS
        L=-0.12/C=0/R=+0.12) to the MEDIAN flown y over the walled section x∈[-2,2] (median,
        not mean, so a transient at entry/exit doesn't swing the label). '?' if the drone
        never entered the walled section.
    """
    if not obs_traj:
        return None
    xy = np.asarray(obs_traj, dtype=float)[:, 3:5]         # flown (x, y)

    if scene == 'pillars':
        labels = []
        for px in (-2.0, 0.0, 2.0):
            cross = np.where(np.diff(np.sign(xy[:, 0] - px)) != 0)[0]
            if cross.size == 0:
                labels.append('?')
                continue
            i = int(cross[0])
            (x0, y0), (x1, y1) = xy[i], xy[i + 1]
            t = 0.0 if x1 == x0 else (px - x0) / (x1 - x0)
            y = y0 + t * (y1 - y0)
            labels.append('L' if y < 0 else 'R')
        return '(' + ','.join(labels) + ')'

    if scene == 'corridor':
        in_walls = (xy[:, 0] >= -2.0) & (xy[:, 0] <= 2.0)  # x-span of the corridor walls (XML)
        ys = xy[in_walls, 1] if in_walls.any() else np.empty(0)
        if ys.size == 0:
            return '?'
        y_med = float(np.median(ys))
        channels = {'L': -0.12, 'C': 0.0, 'R': 0.12}       # trajectories.CORRIDOR_CHANNELS
        return min(channels, key=lambda k: abs(channels[k] - y_med))

    return None                                             # single-class scenes (s_curve, empty)


def _warn_expert_route_infeasibility(scene, config, homotopies, n_samples=200):
    """Fix_12: cheap sanity gate, run once per geo entry BEFORE any rollout.

    Samples each homotopy's expert reference route and checks it against the PLANNING
    constraint set (surfaces at the full r_drone+margin_base offset — reuses
    `_exec_constraint_violations` by substituting the planning margin for its r_drone).
    A warning here means the projector will fight the trained behavior on every replan —
    exactly the failure mode Fix_12 diagnosed (pillars: both trained channels closed by
    over-inflation; boxes excluding start/goal). Base-variant margin only; `-tightened`
    adds enlarge_constraints on top, so treat a near-zero-slack PASS here as a tightened
    FAIL. Print-only: never blocks the run (ablation variants may relax the violated
    family anyway).
    """
    ctypes = set(config.get('constraint_types') or [])
    if not ({'geo_bounds', 'halfspace', 'obstacles'} & ctypes):
        return
    import uav_expert_data_collect.generator as gen
    _infl = config.get('inflation') or {}
    margin = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    probe_cfg = dict(config, inflation={'r_drone': margin, 'margin_base': 0.0})
    rng = np.random.default_rng(0)                         # deterministic probe routes
    for h in homotopies:
        traj_fn, _init, dur = gen._build_traj_and_init(scene, h, rng)
        ts = np.linspace(0.0, dur, n_samples)
        obs_like = []
        for t in ts:
            p = np.asarray(traj_fn(t)[0], dtype=float)
            obs_like.append(np.concatenate([p, p, np.zeros(3)]))   # p in cols 3:6
        ok, n_bad, total = _exec_constraint_violations(obs_like, probe_cfg)
        if ok:
            print(f'[ eval ] {scene} feasibility check: homotopy={h} expert route OK '
                  f'under planning margin {margin:.2f} m')
        else:
            print(f'[ eval ] WARNING {scene} homotopy={h}: expert route violates the '
                  f'PLANNING constraint set at {n_bad}/{n_samples} samples '
                  f'(total penetration {total:.2f} m·samples, margin {margin:.2f} m) — the '
                  f'projector will fight the trained behavior; check geometry/inflation in '
                  f'config/uav_projection.yaml (Fix_12).')


def plot_geo_constraints(geo_name, config, out_dir, is_tightened=False, basename='constraint_overview'):
    """E9 U2: constraint-geometry schematic — the UAV equivalent of the visual-aligning
    `constraint_overview.png` (`fm_visual_aligning_test/eval_fm_visual_aligning.py
    plot_geo_constraints`), which we were missing entirely (not a faithful port).

    3-panel: 3D wireframe | XY top-down | XZ side. Shows the workspace box (steelblue),
    halfspace boundaries + feasible-side arrow (darkorange; x_active segments drawn only
    over their live x-range, s_curve), obstacle balls (tomato). Boundaries are drawn at the
    TRUE enforced margin (r_drone + margin_base [+ enlarge if tightened]) — the same `margin`
    setup_dpcc_projector uses — not the raw scene geometry, so the figure shows what the
    projector actually believes, not just the XML.

    Saved as BOTH <basename>.png (raster, quick viewing) AND .svg (vector — the visual-aligning
    original only had .png; added here since a vector schematic is what a paper/thesis figure
    actually wants). Idempotent (skipped if both files already exist).

    `basename` distinguishes the tightened twin: unlike visual-aligning (where `-tightened`
    is baked into a whole separate NAMED geo entry with its own results/<name>/ folder), UAV's
    `-tightened` is a per-VARIANT margin modifier sharing the same geo_tag/geo_dir as its base
    sibling (matching the older DPCC-avoiding convention) — so the two margins need two
    filenames (`constraint_overview.png` vs `constraint_overview_tightened.png`) inside the
    SAME folder, not two folders.
    """
    out_png = os.path.join(out_dir, f'{basename}.png')
    out_svg = os.path.join(out_dir, f'{basename}.svg')
    if os.path.exists(out_png) and os.path.exists(out_svg):
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as _mpa
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _P3C

    ctypes = config.get('constraint_types', [])
    _infl = config.get('inflation') or {}
    inflation_base = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    enlarge = float(config.get('enlarge_constraints') or 0.0) if is_tightened else 0.0
    margin = inflation_base + enlarge          # the TRUE enforced offset (matches setup_dpcc_projector)

    has_bounds = 'geo_bounds' in ctypes and config.get('workspace_bounds') is not None
    ws_lb = ws_ub = lb_d = ub_d = None
    _Z_DISP = (0.0, 2.0)        # UAV flight band default display when z is unconstrained

    # Gen15 Fix_2: these two are resolved HERE, before the display-bounds block, because that
    # block now frames an unbounded axis from the geometry that actually lives on it.
    halfspace_list = config.get('halfspace_constraints', []) if 'halfspace' in ctypes else []
    obstacle_list  = config.get('obstacle_constraints', [])  if 'obstacles' in ctypes else []

    def _geo_extent(axis):
        """(min, max) over every drawn feature on `axis` (0=x, 1=y), or None if nothing is
        drawn there. Used to frame an axis the workspace box does not bound. Halfspaces use
        their RAW endpoints — x_active only shortens a segment, so this stays a superset."""
        vals = []
        for hs in halfspace_list:
            _triple, _xa = _normalize_halfspace(hs)
            vals += [float(_triple[0][axis]), float(_triple[1][axis])]
        for obs in obstacle_list:
            _c = float(obs['center'][axis]); _r = float(obs['radius']) + margin
            vals += [_c - _r, _c + _r]
        return (min(vals), max(vals)) if vals else None

    if has_bounds:
        ws_lb = np.array(config['workspace_bounds']['lb'], dtype=float)
        ws_ub = np.array(config['workspace_bounds']['ub'], dtype=float)
        ws_lb = ws_lb + margin; ws_ub = ws_ub - margin
        lb_d = ws_lb.copy(); ub_d = ws_ub.copy()
        # 🔴 Gen15 Fix_2 — CLAMP PER AXIS, NEVER WITH A BOOLEAN MASK. This used to read
        #     lb_d[np.isinf(lb_d)] = _Z_DISP[0]; ub_d[np.isinf(ub_d)] = _Z_DISP[1]
        # which stamps the *z* flight band (0.0, 2.0) onto ANY infinite axis. corridor declares
        # y = ±inf deliberately ('y handled by the wall halfspaces', config/uav_projection.yaml),
        # so its y frame became [0.0, 2.0] — the walls at y = ±0.45 and the cap balls at y = ∓0.5
        # fell outside _ylim(), and every drawn object appeared shoved to the bottom edge under a
        # bounds rectangle floating over empty space. The mask also made the x/y fallback loop
        # that followed it DEAD CODE: nothing is inf after the mask, so np.isinf never fires.
        # Display only — setup_dpcc_projector keeps the ±inf rows (−inf + margin = −inf) and SLSQP
        # reads them as unbounded, so no evaluated number was ever affected. See
        # logs_in_develop/Gen15/fix_2/.
        for _i, _fb in ((0, (-3.6, 3.6)), (1, (-2.0, 2.0)), (2, _Z_DISP)):
            # Prefer the real geometry on an unbounded axis; the constants are a last resort
            # (they frame corridor's ±0.45 walls inside a ±2 m box, which is legible but poor).
            _geo = _geo_extent(_i) if _i < 2 else None
            if np.isinf(lb_d[_i]): lb_d[_i] = (_geo[0] - 0.35) if _geo else _fb[0]
            if np.isinf(ub_d[_i]): ub_d[_i] = (_geo[1] + 0.35) if _geo else _fb[1]

    def _xlim(): return (lb_d[0]-0.3, ub_d[0]+0.3) if lb_d is not None else (-3.5, 3.5)
    def _ylim(): return (lb_d[1]-0.3, ub_d[1]+0.3) if lb_d is not None else (-2.0, 2.0)
    def _zlim(): return (lb_d[2]-0.1, ub_d[2]+0.2) if lb_d is not None else (_Z_DISP[0]-0.1, _Z_DISP[1]+0.2)

    def _wall_xy(hs):
        """Resolve a halfspace to (p1, p2, side) clipped to its x_active range (if any)."""
        triple, x_active = _normalize_halfspace(hs)
        (x1, y1), (x2, y2), side = triple
        if x_active is not None:
            lo, hi = x_active
            if abs(x2 - x1) > 1e-9:
                t0 = (lo - x1) / (x2 - x1); t1 = (hi - x1) / (x2 - x1)
                t0, t1 = sorted((max(0.0, min(1.0, t0)), max(0.0, min(1.0, t1))))
                y1n = y1 + t0 * (y2 - y1); y2n = y1 + t1 * (y2 - y1)
                x1n = x1 + t0 * (x2 - x1); x2n = x1 + t1 * (x2 - x1)
                return (x1n, y1n), (x2n, y2n), side, (lo, hi)
        return (x1, y1), (x2, y2), side, None

    fig = plt.figure(figsize=(16, 5))
    _tstr = ' [tightened]' if is_tightened else ''
    fig.suptitle(f'{geo_name}{_tstr}  |  types: {ctypes}  |  margin(r_drone+base'
                 f'{"+" + str(enlarge) if enlarge else ""})={margin:.3f} m',
                 fontsize=11, fontweight='bold', y=0.98)

    # ── 3D panel ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(131, projection='3d')
    ax3.set_title('3D view', fontsize=9)
    ax3.set_xlabel('x (m)', fontsize=7); ax3.set_ylabel('y (m)', fontsize=7); ax3.set_zlabel('z (m)', fontsize=7)
    ax3.tick_params(labelsize=6)
    if lb_d is not None:
        x0, y0, z0 = lb_d; x1v, y1v, z1v = ub_d
        for xs, ys, zs in [
            ([x0,x1v],[y0,y0],[z0,z0]), ([x0,x1v],[y1v,y1v],[z0,z0]),
            ([x0,x1v],[y0,y0],[z1v,z1v]), ([x0,x1v],[y1v,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y1v],[z0,z0]), ([x1v,x1v],[y0,y1v],[z0,z0]),
            ([x0,x0],[y0,y1v],[z1v,z1v]), ([x1v,x1v],[y0,y1v],[z1v,z1v]),
            ([x0,x0],[y0,y0],[z0,z1v]), ([x1v,x1v],[y0,y0],[z0,z1v]),
            ([x0,x0],[y1v,y1v],[z0,z1v]), ([x1v,x1v],[y1v,y1v],[z0,z1v]),
        ]:
            ax3.plot(xs, ys, zs, color='steelblue', alpha=0.7, lw=1.2)
    for obs in obstacle_list:
        dims = obs.get('dimensions', ['x', 'y'])
        cx, cy = float(obs['center'][0]), float(obs['center'][1])
        cz = float(obs['center'][2]) if ('z' in dims and len(obs['center']) > 2) else (
            (lb_d[2]+ub_d[2])/2 if lb_d is not None else 1.0)
        r = obs['radius'] + margin
        u = np.linspace(0, 2*np.pi, 20); v = np.linspace(0, np.pi, 10)
        ax3.plot_surface(cx + r*np.outer(np.cos(u), np.sin(v)), cy + r*np.outer(np.sin(u), np.sin(v)),
                          cz + r*np.outer(np.ones_like(u), np.cos(v)), color='tomato', alpha=0.25, linewidth=0)
    _hs_zlo, _hs_zhi = (lb_d[2], ub_d[2]) if lb_d is not None else _Z_DISP
    for hs in halfspace_list:
        (hx1, hy1), (hx2, hy2), side, _ = _wall_xy(hs)
        ax3.add_collection3d(_P3C([[
            [hx1, hy1, _hs_zlo], [hx2, hy2, _hs_zlo], [hx2, hy2, _hs_zhi], [hx1, hy1, _hs_zhi],
        ]], alpha=0.25, facecolor='darkorange', edgecolor='darkorange', lw=0.8))
    if not has_bounds and not obstacle_list and not halfspace_list:
        ax3.text2D(0.5, 0.5, 'no geometric\nconstraints', ha='center', va='center',
                   transform=ax3.transAxes, fontsize=9, color='gray')
    ax3.set_xlim(*_xlim()); ax3.set_ylim(*_ylim()); ax3.set_zlim(*_zlim())

    # ── XY top-down panel ────────────────────────────────────────────────────
    ax_xy = fig.add_subplot(132)
    ax_xy.set_title('XY top-down (z projected)', fontsize=9)
    ax_xy.set_xlabel('x (m)', fontsize=7); ax_xy.set_ylabel('y (m)', fontsize=7)
    ax_xy.set_aspect('equal'); ax_xy.grid(True, linestyle='--', alpha=0.4); ax_xy.tick_params(labelsize=6)
    if lb_d is not None:
        ax_xy.add_patch(_mpa.Rectangle((lb_d[0], lb_d[1]), ub_d[0]-lb_d[0], ub_d[1]-lb_d[1],
                                        lw=1.5, edgecolor='steelblue', facecolor='steelblue',
                                        alpha=0.12, label='bounds (enforced)'))
    else:
        ax_xy.text(0.5, 0.5, 'no bounds', ha='center', va='center', transform=ax_xy.transAxes,
                   fontsize=9, color='gray')
    for hs in halfspace_list:
        (hx1, hy1), (hx2, hy2), side, x_active = _wall_xy(hs)
        ax_xy.plot([hx1, hx2], [hy1, hy2], color='darkorange', lw=2.0,
                   label='halfspace wall' if hs is halfspace_list[0] else None)
        dx, dy = hx2-hx1, hy2-hy1; nrm = np.hypot(dx, dy) or 1.0
        nx, ny = (-dy/nrm, dx/nrm) if side == 'above' else (dy/nrm, -dx/nrm)
        mx, my = (hx1+hx2)/2, (hy1+hy2)/2
        ax_xy.annotate('', xy=(mx+nx*0.2, my+ny*0.2), xytext=(mx, my),
                       arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.3))
        if x_active is not None:
            ax_xy.text(mx, my, f'x∈[{x_active[0]:.1f},{x_active[1]:.1f}]', fontsize=5,
                       color='saddlebrown', ha='center', va='bottom')
    for obs in obstacle_list:
        ax_xy.add_patch(_mpa.Circle((float(obs['center'][0]), float(obs['center'][1])),
                                     obs['radius']+margin, lw=1.5, edgecolor='tomato',
                                     facecolor='tomato', alpha=0.2, label='obstacle (enforced)'))
        ax_xy.plot(float(obs['center'][0]), float(obs['center'][1]), 'r+', ms=6)
    ax_xy.set_xlim(*_xlim()); ax_xy.set_ylim(*_ylim())
    _handles, _labels = ax_xy.get_legend_handles_labels()
    if _handles:
        _seen = dict(zip(_labels, _handles))
        ax_xy.legend(_seen.values(), _seen.keys(), fontsize=6, loc='upper right')

    # ── XZ side panel ────────────────────────────────────────────────────────
    ax_xz = fig.add_subplot(133)
    ax_xz.set_title('XZ side (y projected)', fontsize=9)
    ax_xz.set_xlabel('x (m)', fontsize=7); ax_xz.set_ylabel('z (m)', fontsize=7)
    ax_xz.grid(True, linestyle='--', alpha=0.4); ax_xz.tick_params(labelsize=6)
    if lb_d is not None:
        ax_xz.add_patch(_mpa.Rectangle((lb_d[0], lb_d[2]), ub_d[0]-lb_d[0], ub_d[2]-lb_d[2],
                                        lw=1.5, edgecolor='steelblue', facecolor='steelblue', alpha=0.12))
        ax_xz.axhline(lb_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7, label=f'floor z={lb_d[2]:.2f} m')
        ax_xz.axhline(ub_d[2], color='steelblue', ls='--', lw=0.9, alpha=0.7, label=f'ceiling z={ub_d[2]:.2f} m')
        ax_xz.legend(fontsize=6, loc='upper right')
    else:
        ax_xz.text(0.5, 0.5, 'no bounds', ha='center', va='center', transform=ax_xz.transAxes,
                   fontsize=9, color='gray')
    for obs in obstacle_list:
        cz_mid = (lb_d[2]+ub_d[2])/2 if lb_d is not None else 1.0
        ax_xz.add_patch(_mpa.Circle((float(obs['center'][0]), cz_mid), obs['radius']+margin,
                                     lw=1.2, edgecolor='tomato', facecolor='tomato',
                                     alpha=0.25, linestyle='--'))
    for hs in halfspace_list:
        (hx1, hy1), (hx2, hy2), side, x_active = _wall_xy(hs)
        xb_lo, xb_hi = sorted((hx1, hx2))
        ax_xz.axvspan(xb_lo, xb_hi, color='darkorange', alpha=0.13, zorder=1)
        ax_xz.axvline(xb_lo, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
        ax_xz.axvline(xb_hi, color='darkorange', lw=1.0, ls='--', alpha=0.8, zorder=2)
    ax_xz.set_xlim(*_xlim()); ax_xz.set_ylim(*_zlim())

    if 'dynamics' in ctypes:
        fig.text(0.5, 0.01, 'Dynamics: p_des[t+1]=p_des[t]+act[t], p[t+1]=p[t]+act[t]  '
                 '(Euler link — no geometric shape)', ha='center', fontsize=7,
                 color='dimgray', style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)
    print(f'[ geo ] Constraint overview → {out_png} (+ .svg)')


def _normalize_halfspace(hs):
    """E9: accept both halfspace formats and return (constraint_triple, x_active).
      - list form  [[x1,y1], [x2,y2], 'side']                     → x_active=None (always live)
      - dict form  {line: [[..],[..]], side: '..', x_active: [lo,hi]}  → per-segment switching
    `constraint_triple` is the [p0, p1, side] that formulate_halfspace_constraints consumes.
    """
    if isinstance(hs, dict):
        line = hs['line']
        return [line[0], line[1], hs['side']], hs.get('x_active')
    return [hs[0], hs[1], hs[2]], None


def setup_dpcc_projector(args, config, obs_normalizer, act_normalizer, variant,
                         trajectory_dim=12, current_x=None, return_constraint_list=False):
    """Build the DPCC projector (mirrors visual-aligning `setup_dpcc_projector`).

    UAV 12-D transition: [dx(0) dy(1) dz(2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)].
    Both position channels are real and anchored with 6 rows (DC_FIX), mirroring DPCC avoiding.
    p_des(3,4,5): commanded setpoint. p(6,7,8): actual drone position from qpos[:3].

    E9 additions:
      - GEOMETRIC constraints (halfspace, obstacles, workspace box) bind to ACTUAL p (6,7,8)
        ONLY — DPCC-faithful (see STUDY_DPCC_constraint_dim_binding.md). Velocity (9,10,11)
        is never touched.
      - `bounds` builds TWO orthogonal row-sets (§2.2): the workspace box on p AND the shared
        action-magnitude bound on the action dims (0,1,2 = Δp_des). The action bound is NOT
        inflated (it is a dataset-range cap, not a spatial surface). SETTABLE via
        `config['action_bounds']` (Fix_3): `'auto'` (default, recommended) self-derives from
        `act_normalizer.mins/maxs` (this dataset's own observed action range); an explicit
        `{lb,ub}` dict overrides it; `None` disables the action bound entirely.
      - Inflation (r_drone + margin_base) offsets every spatial surface so the BODY clears,
        always-on; the -tightened enlarge is added on top.
      - Halfspaces may carry `x_active` (s_curve): a wall is included only if `current_x` is
        inside its interval; `current_x=None` → all walls active (build-time fallback).

    Variant semantics: gradient / post_processing / model_free / tightened (unchanged), plus
    U8's `bounds_free` and `geo_free` (new) — three independent, composable variant-name
    toggles, one per constraint "group": `model_free`→no dynamics, `bounds_free`→no action
    bound, `geo_free`→no geo_bounds+halfspace+obstacles (the three geometric families move
    together as one group, same as `model_free` already grouped nothing since dynamics was
    already singular). Combine by substring, e.g. `dpcc-c-geo_free-bounds_free` = dynamics-only
    projection with selection `dpcc-c`. This REPLACES the earlier per-scene
    `<scene>_dynamics_only`/`<scene>_dynamics_bounds_only` geo_constraint_variants entries
    (Fix_5) — those are removed; the SAME ablations are now variant-level toggles on the one
    remaining full-stack geo entry per scene. See
    logs_in_develop/Gen11/Epoch9_PCC_Constraints/U_8_new_projection_var_upgrade/.
    """
    from mix_uav.sampling.projection import Projector

    _DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'x': 6, 'y': 7, 'z': 8}   # x,y,z = actual position p
    pad = trajectory_dim - 9
    is_tightened = 'tightened' in variant
    tightening   = float(config.get('enlarge_constraints') or 0.0)
    enlarge      = tightening if is_tightened else 0.0
    # E9 inflation: always-on offset so the drone body (not just its center) clears geometry.
    _infl = config.get('inflation') or {}
    inflation_base = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    margin = inflation_base + enlarge                 # total spatial offset (surfaces only)
    ctypes = config.get('constraint_types', [])
    constraint_list = []

    if 'geo_bounds' in ctypes and 'geo_free' not in variant:
        # Workspace box on ACTUAL p (6,7,8), shrunk inward by the spatial margin.
        # Patch_Constraints_C3: renamed from 'bounds' — this flag now ONLY means the geo box;
        # 'bounds' below is a SEPARATE, independent family (the restored DPCC action limit).
        # U8: 'geo_free' (variant-level, mirrors 'model_free') skips this + halfspace +
        # obstacles TOGETHER — the "geometric/spatial" group — same pattern as 'model_free'
        # skipping 'dynamics' alone. See U_8_new_projection_var_upgrade changelog.
        ws = config.get('workspace_bounds')
        if ws is not None:
            ws_lb = np.array(ws['lb'], dtype=float); ws_ub = np.array(ws['ub'], dtype=float)
            lb = np.concatenate([np.full(6, -np.inf), ws_lb + margin, np.full(pad, -np.inf)])
            ub = np.concatenate([np.full(6,  np.inf), ws_ub - margin, np.full(pad,  np.inf)])
            constraint_list += [['lb', lb], ['ub', ub]]

    if 'bounds' in ctypes and 'bounds_free' not in variant:
        # Shared action-magnitude bound on the ACTION dims (0,1,2) — NOT inflated (§2.2).
        # Patch_Constraints_C3: independent of 'geo_bounds' above (was conflated under one
        # 'bounds' flag until this split; mirrors the same fix applied to visual-aligning's
        # eval scripts — see logs_in_develop/Gen7_FMPCC_Viusal_Aligning/Patch_Constraints_C3/).
        # U8: 'bounds_free' (variant-level) skips just this family, mirroring 'model_free'
        # (skips 'dynamics') and 'geo_free' (skips geo_bounds+halfspace+obstacles). The three
        # toggles are independent and composable in the variant name (e.g.
        # 'geo_free-bounds_free' = dynamics alone; 'geo_free-model_free' = bounds alone) — see
        # U_8_new_projection_var_upgrade changelog for the full truth table.
        # SETTABLE in the yaml (`action_bounds`), default `'auto'` (Fix_3):
        #   'auto' (RECOMMENDED, default) → SELF-DERIVE from act_normalizer.mins/.maxs, the
        #     dataset's OWN observed Δp_des range (LimitsNormalizer: mins/maxs = X.min/max
        #     over the training data, fit at load time). This is what DPCC-avoiding's
        #     hardcoded ['vx','vy'] bound approximated BY HAND for ITS OWN dataset ("need to
        #     be within the limits of the dataset due to the normalization") — copying
        #     avoiding's NUMBER would be wrong here (different robot, different workspace
        #     scale, different expert speed); reusing the SAME METHOD (derive from THIS
        #     dataset's own action range) is the faithful equivalent, and it's already
        #     computed by code that runs at eval time — no placeholder, no cluster-side
        #     manual measurement needed. Mirrors the `pid_const_v` self-calibration precedent
        #     (`_run_variant`: `v_des_magnitude` derived from `dataset.fields.actions`).
        #   explicit {lb: [...], ub: [...]} → override with a hand-picked cap (e.g. to test a
        #     tighter/looser action limit than the dataset's own range) instead of 'auto'.
        ab = config.get('action_bounds', 'auto')
        if ab is None:
            a_lb = a_ub = None
        elif ab == 'auto':
            a_lb = np.asarray(act_normalizer.mins, dtype=float)
            a_ub = np.asarray(act_normalizer.maxs, dtype=float)
        else:
            a_lb = np.array(ab['lb'], dtype=float)
            a_ub = np.array(ab['ub'], dtype=float)
        if a_lb is not None:
            a_lb = np.concatenate([a_lb, np.full(trajectory_dim - 3, -np.inf)])
            a_ub = np.concatenate([a_ub, np.full(trajectory_dim - 3,  np.inf)])
            constraint_list += [['lb', a_lb], ['ub', a_ub]]

    if 'dynamics' in ctypes and 'model_free' not in variant:
        # DC_FIX: both real channels anchored — 6 rows (DPCC avoiding 4-row pattern scaled to 3D).
        # Traj layout: [act(0,1,2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)]
        constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # DC_FIX p_des ← act
        constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]  # DC_FIX p     ← act

    if 'halfspace' in ctypes and 'geo_free' not in variant:
        _hs = {'x': _DIM['x'], 'y': _DIM['y']}
        for hs in config.get('halfspace_constraints', []):
            triple, x_active = _normalize_halfspace(hs)
            if x_active is not None and current_x is not None:
                if not (x_active[0] <= float(current_x) <= x_active[1]):
                    continue                              # wall not live in this x-segment
            C_row, d = utils.formulate_halfspace_constraints(triple, margin, trajectory_dim, _hs)
            constraint_list.append(('ineq', (C_row, d)))

    if 'obstacles' in ctypes and 'geo_free' not in variant:
        for obs in config.get('obstacle_constraints', []):
            dims = [_DIM[d] if isinstance(d, str) else int(d) for d in obs['dimensions']]
            constraint_list.append((obs['type'], dims, obs['center'], obs['radius'] + margin))

    is_gradient      = 'gradient' in variant
    is_post_proc     = 'post_processing' in variant
    threshold        = 0.0 if is_post_proc else config.get('diffusion_timestep_threshold', 0.5)

    # Gen15 U2 — the HardFlow NLP is built from the SAME `constraint_list` the DPCC Projector
    # consumes. That is not a convenience: if the two arms enforced different constraint sets
    # the comparison would be void (the Gen12 port's first design rule).
    if return_constraint_list:
        return constraint_list

    return Projector(
        horizon=int(getattr(args, 'horizon', 8)),
        transition_dim=trajectory_dim,
        action_dim=3,
        goal_dim=0,
        constraint_list=constraint_list,
        normalizer=ProjectorNormalizer(obs_normalizer, act_normalizer),
        diffusion_timestep_threshold=threshold,
        variant='states_actions',
        dt=config.get('dt', 1.0),                   # action IS Δp_des → Euler dt=1.0 (NOT 1/33)
        gradient=is_gradient,
        gradient_weights=[1, 0.5, 2] if is_gradient else None,
        solver='scipy',
        device=getattr(args, 'device', 'cuda'),
    )


def _is_hardflow(variant):
    """True for the Gen15 U2 arm-C variants (`hardflow_new`, `hardflow_new-c`, ...).

    These are NOT projection variants of the DPCC projector — they select a different
    guidance MECHANISM (an in-loop prox-NLP inside each ODE step, instead of generate-then-
    project). The DPCC `Projector` is never constructed for them.
    """
    return str(variant).startswith('hardflow')


def _selection_for(variant):
    """FMv3ODE variant → trajectory_selection (verbatim semantics).

    Gen15 U2: the `-c` / `-t` suffixes compose with the hardflow arm too, so
    `hardflow_new-c` gets minimum-projection-cost selection exactly like `dpcc-c`. The
    substring tests below already cover it — `hardflow_new-c` contains neither 'dpcc-t' nor
    'dpcc-c', so it is spelled out here rather than left to luck.
    """
    if 'dpcc-t' in variant or variant.endswith('-t') or '-t-' in variant:
        return 'temporal_consistency'
    if 'dpcc-c' in variant or variant.endswith('-c') or '-c-' in variant:
        return 'minimum_projection_cost'
    return 'random'


def _make_overhead_renderer(mujoco, model, res=140):
    """Headless overhead renderer; None if rendering is unavailable (no hard dep).

    ONE renderer is created per scene and reused across rollouts — never one per
    rollout — so we allocate exactly one EGL/GL context instead of leaking N of them.

    Fix_7: res lowered 360→200 (~3x fewer pixels/frame). Fix_9: lowered again 200→140
    (~2x more on top, ~5x total vs the original 360). This render is ONLY used for the
    debug/visualization GIF — never fed to a policy (unlike the arm's bp_cam/inhand_cam,
    which render at 96x96 because that resolution IS the trained vision-model input) — so
    there is no accuracy tradeoff from shrinking it, only a smaller file. 140px is close to
    the arm's own 96px scale while staying legible for a multi-obstacle overhead scene. See
    logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_7_Gif_lower_size/ and .../Fix_9_.../.
    """
    try:
        return mujoco.Renderer(model, height=res, width=res)
    except Exception as exc:                               # pragma: no cover - cluster-only
        print(f'[ eval ] render unavailable ({exc}); GIF skipped')
        return None


def _free_renderer(renderer):
    """Release the renderer's GL context *now*, while EGL is still initialized.

    MuJoCo's GLContext.__del__ calls eglMakeCurrent; if it runs at interpreter
    shutdown (after EGL is torn down) it raises EGL_NOT_INITIALIZED. Freeing here —
    plus reusing a single renderer per scene — prevents both that teardown error and
    the per-rollout GL-context leak. Tolerant of mujoco versions with/without close().
    """
    if renderer is None:
        return
    try:
        if hasattr(renderer, 'close'):        # mujoco >= 3.x
            renderer.close()
    except Exception:                         # pragma: no cover
        pass
    try:
        import gc
        gc.collect()                          # force GLContext.__del__ while EGL is up
    except Exception:                         # pragma: no cover
        pass


def _render_overhead(mujoco, model, data, renderer):
    """Single top-down frame. Reuses the PROVEN overhead camera from the expert GIF
    tool (uav_expert_data_collect/generate_trajectory_gifs._render_overhead); falls
    back to the same camera inline only if that import is unavailable."""
    try:
        from uav_expert_data_collect.generate_trajectory_gifs import (
            _render_overhead as _proven_overhead)
        return _proven_overhead(model, data, renderer)
    except Exception:                                      # pragma: no cover
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = data.qpos[:3]
        cam.distance = 5.0
        cam.azimuth = 0.0
        cam.elevation = -90.0
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()


def rollout_one(model, scene, homotopy, trial_seed, policy, horizon,
                renderer=None, frame_stride=3, goal_radius=GOAL_RADIUS, batch_size=1,
                variant='diffuser', log_dir=None, control_hz=DATASET_HZ, text_log=True,
                controller='pid', cond_mode='p_des', mjpc_kwargs=None,
                v_des_magnitude=0.0, geo_config=None, rebuild_projector=None,
                max_episode_length=None):
    """One closed-loop MuJoCo rollout. Mirrors generator.run_trial; FM replaces traj_fn.

    `model` and `renderer` are owned by eval_scene and shared across rollouts (one
    GL context per scene, not per rollout). `batch_size` = MPC candidate-fan size: the
    policy samples a batch and (per its trajectory_selection) returns the chosen
    candidate's first action; `plans` stores the whole fan. Buffers obs/action/plan per
    FM step (U3 npz schema); if `renderer` is given, also captures overhead frames.
    Heavy arrays/frames are returned under HEAVY_KEYS and stripped from results.json.

    U_13: `max_episode_length` is the FIXED step budget (same for every trial of a scene),
    replacing the old per-trial random `n_fm = round(dur*DATASET_HZ)`. Goal-path scenes
    EARLY-STOP the instant the drone reaches within `goal_radius` of the goal (DPCC
    avoiding pattern); a run that never reaches uses the full budget. So `n_fm_steps` is now
    a deterministic time-to-goal (success) or the full budget (miss), not a random draw.
    """
    import mujoco
    import uav_expert_data_collect.generator as gen

    rng = np.random.default_rng(trial_seed)
    data = mujoco.MjData(model)

    traj_fn, init_pos, dur = gen._build_traj_and_init(scene, homotopy, rng)
    goal = np.asarray(traj_fn(dur)[0], dtype=float)        # expert path endpoint (secondary metric)
    # U7: finish-line crossing test (success_relaxed) — a vertical plane (xy line, any z)
    # through `goal`, oriented perpendicular to the expert path's final approach heading.
    # `crossed_line` latches true the first time the drone's xy position is ever on the
    # goal side of that line, independent of where it ends up afterward.
    _p_before_goal = np.asarray(traj_fn(max(dur - 0.1, 0.0))[0], dtype=float)
    _line_dir_xy = (goal - _p_before_goal)[:2]
    _line_norm = np.linalg.norm(_line_dir_xy)
    line_dir_xy = _line_dir_xy / _line_norm if _line_norm > 1e-9 else np.array([1.0, 0.0])
    crossed_line = False

    data.qpos[:3] = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # E8: tracker selection.
    #   'pid'         (default) — E7 cascaded PID, v_des = action/dt_fm.
    #   'pid_stopgo'  (U2)      — same CascadedPID, v_des = 0 (strict stop-and-go).
    #   'pid_const_v' (U3)      — same CascadedPID, v_des = unit(action)*v_des_magnitude (constant speed).
    #   'mjpc'                  — MJPC optimal-control thrust tracker (cluster-only).
    # All four expose the same .compute(p,q,v,om,p_des,v_des) API.
    pid = gen._make_pid(model, 'pid_default')
    tracker = pid                              # pid_stopgo also uses CascadedPID (v_des differs)
    if controller == 'mjpc':
        from mix_uav_test.mjpc_tracker import MJPCTracker
        mjpc_kwargs = mjpc_kwargs or {}
        tracker = MJPCTracker(model, scene=scene, **mjpc_kwargs)
    dt = float(model.opt.timestep)
    dt_fm = 1.0 / DATASET_HZ
    decim = max(1, int(round(1.0 / (dt * DATASET_HZ))))    # physics steps per FM query
    # U_13: FIXED budget (not the per-trial random round(dur*DATASET_HZ)). Falls back to the
    # old behaviour only if no budget was passed (defensive; callers always pass one now).
    n_fm = int(max_episode_length) if max_episode_length else int(round(dur * DATASET_HZ))

    frames = []

    # Real-time behaviour logger (digital-twin audit). ALWAYS ON — independent of --record;
    # near-zero cost (wraps timings the loop already takes). See Real_Time_eval_loggging/PLAN.md.
    from mix_uav_test.behavior_logger import BehaviorLogger
    episode_id = f'{scene}_{homotopy}_{trial_seed}'
    blog = BehaviorLogger(episode_id, variant, scene, homotopy,
                          control_hz=control_hz, batch_size=batch_size, horizon=horizon,
                          text_log=text_log)
    proj_on = (variant != 'diffuser')

    p_des = np.asarray(init_pos, dtype=float).copy()
    n_hit = 0
    n_phys = 0
    min_z = float('inf')
    track_err = []
    fm_ms = []           # PURE FM inference ms (projection time subtracted out — Real_Time logging)
    proj_ms = []         # PCC projector wall-time ms per FM step
    total_ms = []        # fm_ms + proj_ms  → the real-time budget number
    proj_cb_skipped_steps = 0   # Fix_15.3: FM steps whose projection was SKIPPED by the tripped breaker
    obs_traj = []        # realized [p_des|p|v] per FM step  → npz obs_all
    act_traj = []        # FM Δp_des per FM step             → npz act_all
    plans = []           # FM H-step predicted obs plan      → npz sampled_trajectories_all

    # U_13: goal-reach latch (goal-path scenes only) + the FM step the episode actually
    # stopped on. `empty` has a random ill-defined goal → never early-stops (runs full budget).
    goal_reached_latch = False
    steps_run = n_fm     # overwritten at an early break; == full budget on a miss
    # Div_Abort: divergence/abort bookkeeping for THIS rollout. Every field is persisted
    # (results.json `divergence` group, npz `divergence_*`, eval log, foresight SVG) so a
    # lost flight can be read back — WHEN (step/time/physics step), WHERE (p/p_des/v) and
    # WHY (reason/detail/thresholds) — without re-running anything.
    arena_lb, arena_ub = _flight_envelope(scene)
    divergence = {
        'enabled': bool(DIVERGENCE_ABORT),
        'aborted': False, 'reason': None, 'detail': '',
        'step': -1, 'time_s': float('nan'), 'physics_step': -1, 'executed_steps': 0,
        'p': None, 'p_des': None, 'v': None,
        'speed': float('nan'), 'p_des_lead': float('nan'),
        'arena_lb': [float(c) for c in arena_lb], 'arena_ub': [float(c) for c in arena_ub],
        'thresholds': {'envelope_slack_m': DIV_ENVELOPE_SLACK_M,
                       'speed_max_ms': DIV_SPEED_MAX_MS,
                       'map_xy_m': DIV_MAP_XY_M, 'map_z_m': DIV_MAP_Z_M},
    }

    for k in range(n_fm):
        p = data.qpos[:3].copy()
        v = data.qvel[:3].copy()
        # E8: obs layout MUST match how the model was trained (dataset cond_mode).
        #   'pos_only' → [p_des|p] (6D, velocity dropped → 9D transition; FM→MJPC).
        #   'p_des' (default) → [p_des|p|v] (9D → 12D transition; E7 PID).
        if cond_mode == 'pos_only':
            obs = np.concatenate([p_des, p]).astype(np.float32)      # [p_des | p] (6,) raw
        else:
            obs = np.concatenate([p_des, p, v]).astype(np.float32)   # [p_des | p | v] (9,) raw

        # E9 s_curve: re-select the active wall set from the drone's current x before this
        # replan (only when the scene declares x_active halfspaces — else rebuild_projector
        # is None and this is skipped, preserving the build-once path exactly).
        if rebuild_projector is not None:
            policy.projector = rebuild_projector(float(p[0]))

        t0 = time.perf_counter()
        action, traj = policy({0: obs}, batch_size=batch_size, horizon=horizon)
        step_total_ms = (time.perf_counter() - t0) * 1e3         # bundled FM + projection
        step_proj_ms = float(getattr(policy, 'last_proj_ms', 0.0))
        step_fm_ms = max(step_total_ms - step_proj_ms, 0.0)      # PURE inference
        fm_ms.append(step_fm_ms)
        proj_ms.append(step_proj_ms)
        total_ms.append(step_total_ms)
        # Fix_15.3: the projector marks each call it SKIPS while its sustained-slowness circuit
        # breaker is OPEN (this step's trajectory is UNPROJECTED). Count them so the artifacts
        # can flag that projection was abandoned for high SLSQP cost (see projection.py Fix_15.2).
        if getattr(getattr(policy, 'projector', None), 'last_proj_skipped', False):
            proj_cb_skipped_steps += 1

        action = np.asarray(action, dtype=float).reshape(-1)[:3]  # first Δp_des
        obs_traj.append(obs)
        act_traj.append(action.astype(np.float32))
        # traj.observations = FM's unnormalized H-step plan in obs space (the foresight).
        plan = getattr(traj, 'observations', None)
        if plan is not None:
            plans.append(np.asarray(plan, dtype=np.float32))
        # FM Δp_des H-step foresight of the EXECUTED candidate (for the log's `horizon=` field).
        which = int(getattr(policy, 'last_which_trajectory', 0))
        fm_horizon = None
        if getattr(traj, 'actions', None) is not None:
            acts = np.asarray(traj.actions)
            if acts.ndim == 3 and which < acts.shape[0]:
                fm_horizon = acts[which]

        p_des = p_des + action
        # v_des feedforward to PID — source depends on controller:
        #   pid         (default): action / dt_fm  (E7, timing-derived).
        #   pid_stopgo  (U2):      zero → PID brakes to zero each FM step (stop-and-go).
        #   pid_const_v (U3):      unit(action)*v_des_magnitude → constant speed, timing-free.
        #   mjpc:                  v_des accepted for API parity but ignored internally.
        if controller == 'pid_stopgo':
            v_des = np.zeros(3)
        elif controller == 'pid_const_v':
            norm = float(np.linalg.norm(action))
            v_des = (action / norm) * v_des_magnitude if norm > 1e-6 else np.zeros(3)
        else:                                    # 'pid' default (and 'mjpc')
            v_des = action / dt_fm

        hit_before = n_hit
        for _ in range(decim):
            p = data.qpos[:3].copy()
            v = data.qvel[:3].copy()
            q = data.qpos[3:7].copy()
            om = data.qvel[3:6].copy()
            u = tracker.compute(p, q, v, om, p_des, v_des)   # E8: pid OR mjpc (same API)
            data.ctrl[:4] = u
            mujoco.mj_step(model, data)
            n_phys += 1
            if any(gen._is_obstacle_contact(model, data.contact[ci]) for ci in range(data.ncon)):
                n_hit += 1
            min_z = min(min_z, float(data.qpos[2]))
            track_err.append(float(np.linalg.norm(data.qpos[:3] - p_des)))
            # U7: one-way latch — true the instant the drone is ever on the goal side
            # of the finish line, regardless of what it does for the rest of the episode.
            # Fix_10: the directional half-plane alone is NOT guaranteed to fire whenever the
            # drone ends up within goal_radius of the goal — its orientation is fixed from the
            # EXPERT's final approach heading, computed once upfront, and a rough rollout
            # (high track_err / obstacle contacts, e.g. s_curve under load) can arrive at the
            # goal from a genuinely different bearing that never crosses that specific
            # fixed-orientation plane, even while ending up physically adjacent to the goal.
            # Observed: success=True (goal_reached + safe) but crossed_line=False forever,
            # violating the documented "success ⇒ success_relaxed" invariant. Fix: ALSO latch
            # on raw proximity — this uses the exact same qpos/goal/threshold `goal_reached`
            # uses on the final step, so goal_reached ⇒ crossed_line is now guaranteed by
            # construction (the final step's proximity check runs in THIS loop, same as every
            # other step), not just true for the common/typical-approach case.
            _side = float(np.dot(data.qpos[:2] - goal[:2], line_dir_xy))
            _dist_now = float(np.linalg.norm(data.qpos[:3] - goal))
            crossed_line = crossed_line or (_side >= 0.0) or (_dist_now < goal_radius)
            # U_13: strict goal-reach latch (within goal_radius of the actual goal) — the
            # early-stop trigger for goal-path scenes. Same qpos/goal/threshold the final
            # `goal_reached` uses, so it is exactly "reached the goal at some step".
            if scene in GOAL_PATH_SCENES and _dist_now < goal_radius:
                goal_reached_latch = True

        # ── one structured log line per FM control step ──
        te_step = float(np.linalg.norm(data.qpos[:3] - p_des))
        blog.step(
            t=k / DATASET_HZ, step_idx=f'{k}/{n_fm}', obs=obs, fm_horizon=fm_horizon,
            fm_ms=step_fm_ms, proj_ms=step_proj_ms,
            proj_cost=float(getattr(policy, 'last_proj_cost', 0.0)), proj_active=proj_on,
            state_p=data.qpos[:3].copy(), state_v=data.qvel[:3].copy(),
            contact='obstacle' if n_hit > hit_before else None, track_err=te_step,
        )

        if renderer is not None and (k % frame_stride == 0):
            try:
                frame = _render_overhead(mujoco, model, data, renderer)
                # U2b: GIF step-count overlay ('sK', top-left) — ported from visual-aligning's
                # Aligning_Sim.capture_frame (`cv2.putText(frame, f's{self.step_counter}', ...)`),
                # which the UAV GIFs were missing entirely. Same style: yellow, top-left, FONT_HERSHEY_PLAIN.
                import cv2
                cv2.putText(frame, f's{k}', (5, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1)
                frames.append(frame)
            except Exception as exc:                       # pragma: no cover
                print(f'[ eval ] frame render failed ({exc}); stopping capture')
                renderer = None     # stop capturing for THIS rollout; eval_scene still owns/frees it

        # ── Div_Abort: stop the episode the step the flight is provably lost ──
        # Checked AFTER the physics decimation (so it reads the freshly integrated state) and
        # only while the goal has NOT been latched — a rollout that already reached the goal
        # exits through the normal break below and is never re-labelled an abort.
        if DIVERGENCE_ABORT and not goal_reached_latch:
            _p_now = data.qpos[:3].copy()
            _v_now = data.qvel[:3].copy()
            _reason, _detail = _check_divergence(_p_now, _v_now, arena_lb, arena_ub,
                                                 quat=data.qpos[3:7])
            if _reason is not None:
                divergence.update({
                    'aborted': True, 'reason': _reason, 'detail': _detail,
                    'step': int(k), 'time_s': float(k / DATASET_HZ),
                    'physics_step': int(n_phys), 'executed_steps': int(k + 1),
                    'p': [float(c) for c in _p_now],
                    'p_des': [float(c) for c in np.asarray(p_des, dtype=float).reshape(-1)],
                    'v': [float(c) for c in _v_now],
                    'speed': float(np.linalg.norm(_v_now)),
                    'p_des_lead': float(np.linalg.norm(
                        np.asarray(p_des, dtype=float).reshape(-1) - _p_now)),
                })
                blog.note(f'DIVERGENCE ABORT  reason={_reason}  step={k}/{n_fm}  '
                          f't={k / DATASET_HZ:.3f}s  p={np.round(_p_now, 3).tolist()}  '
                          f'p_des={np.round(np.asarray(p_des, dtype=float), 3).tolist()}  '
                          f'|v|={np.linalg.norm(_v_now):.2f}m/s  |  {_detail}')
                print(f'[ eval ] {scene} variant={variant} trial_seed={trial_seed}: '
                      f'⚠ DIVERGENCE ABORT at FM step {k}/{n_fm} (t={k / DATASET_HZ:.2f}s) — '
                      f'reason={_reason}: {_detail}', flush=True)
                # U_13 step accounting: an abort is a MISS, and a miss costs the FULL budget
                # (DPCC convention). Charging it the truncated count would make a lost flight
                # look like a fast one in steps_mean. The true executed count lives in
                # divergence['executed_steps'].
                steps_run = n_fm
                break

        # U_13: DPCC avoiding-style early termination (aux_repo/dpcc/scripts/eval.py:264) —
        # stop the instant the goal is reached (goal-path scenes) or the fixed budget is
        # exhausted. `steps_run` (the FM step count at stop) is the deterministic time-to-goal
        # on success / the full budget on a miss — no longer a random draw. `empty` never
        # latches (no goal) so it always runs the full budget.
        if goal_reached_latch or k == n_fm - 1:
            steps_run = k + 1
            break

    # E8: release the MJPC gRPC agent server (no-op for the PID path).
    if controller == 'mjpc' and hasattr(tracker, 'close'):
        tracker.close()

    p_final = data.qpos[:3].copy()
    contact_frac = n_hit / max(n_phys, 1)
    limit = gen.SCENE_MAX_CONTACT_FRACTION.get(scene, gen.MAX_CONTACT_FRACTION)
    airborne = bool(min_z > 0.2)                           # crude floor gate
    goal_dist = float(np.linalg.norm(p_final - goal))
    # U_13: for goal-path scenes, goal_reached is the early-stop latch (reached within
    # goal_radius at SOME step — identical trigger to the break), so a rollout that reaches
    # then would-have-drifted is no longer scored a miss (the old final-position-only check).
    # `empty` (random, ill-defined goal; reported only) keeps the final-position check.
    if scene in GOAL_PATH_SCENES:
        goal_reached = bool(goal_reached_latch)
    else:
        goal_reached = bool(goal_dist < goal_radius)
    safe = bool(contact_frac <= limit and airborne)       # contact-free + airborne
    # Div_Abort: a rollout that flew away is NOT "safe" whatever contact_frac/min_z say — it
    # may well have left the arena without ever touching an obstacle or dropping to the floor,
    # and on `empty` (where success == safe) that would have been scored a SUCCESS. Force the
    # physical-safety axis false so every downstream success flag collapses to 0.
    if divergence['aborted']:
        safe = False
    # Scene-aware success (Fix2_metrics): fixed-route scenes must REACH the goal AND be safe;
    # `empty` has a RANDOM goal the unconditioned FM can't be expected to hit, so there
    # success = stable/safe flight only. A goal-path drone that flies around without reaching
    # the target is NOT a success.
    if scene in GOAL_PATH_SCENES:
        success = bool(goal_reached and safe)
    else:                                                 # empty (random goal): stay stable
        success = bool(safe)

    # U7 (success_relaxed): "crossed the finish line" instead of "ended exactly on it".
    # U_13 note: episodes now DO early-stop on strict goal-reach, so the drift-after-arrival
    # case `success` used to miss can no longer happen for goal-reaching rollouts. crossed_line
    # is retained (it also latches on the half-plane crossing, so it still catches a rollout
    # that grazed past the finish line but stopped just outside goal_radius). success ⇒
    # success_relaxed still holds by construction.
    if scene in GOAL_PATH_SCENES:
        success_relaxed = bool(crossed_line and safe)
    else:                                                  # empty: no fixed goal to cross
        success_relaxed = success

    # Constraint-aware metrics (FMv3ODE schema). E9: computed from the FLOWN path against the
    # scene's RAW spatial geometry ⊕ r_drone (physical collision truth). Dynamics-only /
    # unconstrained (`empty`) scenes have no spatial families → trivially clean.
    collision_free, n_violations, total_violations = _exec_constraint_violations(obs_traj, geo_config or {})
    success_and_constraints = bool(success and collision_free)
    success_and_constraints_relaxed = bool(success_relaxed and collision_free)

    # ── persist the real-time behaviour log + capture its timing summary ──
    behaviour = {
        'result': 'SUCCESS' if success else ('FAIL(goal)' if (scene in GOAL_PATH_SCENES and safe and not goal_reached) else 'FAIL'),
        'result_relaxed': 'SUCCESS' if success_relaxed else 'FAIL',
        'goal_dist': f'{goal_dist:.3f}m', 'safe': safe, 'min_z': f'{min_z:.3f}',
        'contact_frac': f'{contact_frac:.3f}',
    }
    if divergence['aborted']:
        behaviour['result'] = f'ABORT({divergence["reason"]})'
        behaviour['abort_when'] = f'step {divergence["step"]}/{n_fm} (t={divergence["time_s"]:.3f}s)'
        behaviour['abort_where'] = f'p={divergence["p"]} p_des={divergence["p_des"]}'
        behaviour['abort_why'] = divergence['detail']
    blog_summary = blog.summary_dict()
    if log_dir is not None:
        blog.save(os.path.join(log_dir, f'rollout_{episode_id}.log'), behaviour=behaviour)

    # Fix_10 (2/2): grouped schema — replaces the old flat dict where near-synonymous names
    # (`safe` vs `collision_free`) sat side by side with no signal that they measure different
    # things (physical MuJoCo contact truth vs. the projector's own softer declared-constraint
    # margin), and the 4 success booleans were an unlabelled 2×2 matrix. See
    # logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_10_json_metrics/PLAN_fix10_2_json_schema_redesign.md
    return {
        'scene': scene, 'homotopy': homotopy,
        # Fix_12: the class the drone ACTUALLY flew (pillars; None elsewhere). `homotopy`
        # above is only the expert route's label — the unconditioned FM picks its own route,
        # and for pillars all four labels share the same start/goal, so they must not be
        # read as the flown route (they were being, via results.json/plot colors).
        'homotopy_flown': _realized_homotopy(scene, obs_traj),
        # Axis A — physical ground truth (hard MuJoCo mesh contact detection).
        'physical': {
            'safe': safe,
            'contact_frac': contact_frac,
            'min_z': min_z,
            'final_z': float(p_final[2]),
        },
        # Axis B — declared-constraint margin truth (softer than physical; the flown path vs.
        # the projector's own inflated geo_bounds/halfspace/obstacles boundary).
        'constraint': {
            'collision_free': collision_free,
            'n_violations': n_violations,
            'total_violations': total_violations,
        },
        'goal': {
            'reached': goal_reached,
            'dist': goal_dist,
            'crossed_line': crossed_line,
        },
        # 2x2 matrix: {strict, relaxed} goal-reach x {with, without} Axis-B compliance.
        'success': {
            'strict': success,
            'relaxed': success_relaxed,
            'strict_and_constraints': success_and_constraints,
            'relaxed_and_constraints': success_and_constraints_relaxed,
        },
        'timing': {
            'fm_ms_mean': float(np.mean(fm_ms)) if fm_ms else float('nan'),   # PURE inference (proj subtracted)
            'fm_ms_p95': float(np.percentile(fm_ms, 95)) if fm_ms else float('nan'),
            'proj_ms_mean': float(np.mean(proj_ms)) if proj_ms else 0.0,
            'total_ms_mean': float(np.mean(total_ms)) if total_ms else float('nan'),
            'total_ms_p95': float(np.percentile(total_ms, 95)) if total_ms else float('nan'),
            'total_over_budget': int(blog_summary['total_over_budget']),
            'budget_ms': blog_summary['budget_ms'],
        },
        'track_err_mean': float(np.mean(track_err)) if track_err else float('nan'),
        # Fix_15.3: projection-circuit-breaker health for THIS rollout. `cb_tripped` means the
        # sustained-slowness breaker (projection.py Fix_15.2) OPENED and projection was SKIPPED
        # for `cb_skipped_steps` FM steps — those steps ran on the UNPROJECTED trajectory, so the
        # constraint results for this rollout are NOT trustworthy (high SLSQP cost bailout).
        # `cb_trips`/`backstop_hits` read the projector's cumulative counters (may under-count on
        # x_active/s_curve scenes where the projector is rebuilt each step and its state resets).
        'projection_health': {
            'cb_tripped': bool(proj_cb_skipped_steps > 0),
            'cb_skipped_steps': int(proj_cb_skipped_steps),
            'cb_trips': int(getattr(getattr(policy, 'projector', None), '_cb_trips', 0)),
            'backstop_hits': int(getattr(getattr(policy, 'projector', None), '_cost_exploded_count', 0)),
            'n_proj_steps': int(len(proj_ms)),
        },
        # U_13: actual FM steps executed (deterministic time-to-goal on success, full budget
        # on a miss) — was the random round(dur*HZ) budget. `max_episode_length` = the budget.
        # Div_Abort: WHEN/WHERE/WHY this flight was declared lost (all-False group when it
        # was not). `n_fm_steps` below is charged the FULL budget for an abort (miss
        # convention); `divergence['executed_steps']` is what actually ran.
        'divergence': divergence,
        'n_fm_steps': steps_run, 'max_episode_length': n_fm, 'decim': decim, 'dt': dt,
        # ── heavy (npz / gif only; stripped from results.json) ──
        'obs_traj': np.asarray(obs_traj),
        'act_traj': np.asarray(act_traj),
        'plans': plans,
        'frames': frames,
    }


def _run_variant(scene, variant, model_fm, dataset, parsed, horizon, config, args,
                 mj_model, mujoco, homotopies):
    """Run all trials for ONE projection variant → write its plans/<variant>/ artifacts.

    Mirrors the FMv3ODE per-variant block: `projector = None` for `diffuser`, else the DPCC
    projector; `trajectory_selection` per variant; one Policy built per variant (persists
    across trials, exactly as FMv3ODE)."""
    # E8: tracker + obs-layout selection (defaults preserve E7).
    controller      = str(config.get('controller', 'pid'))
    cond_mode       = str(config.get('cond_mode', 'p_des'))
    # U3: pid_const_v speed — auto-derived from dataset so it self-calibrates to any
    # dataset/scene without a magic number.  mean(|action|) × DATASET_HZ ≡ mean(action/dt_fm)
    # i.e. the same value the default 'pid' controller produces on average.
    # Zero-padding (at-goal steps) is filtered before averaging.
    if controller == 'pid_const_v':
        _all_acts = dataset.fields.actions.reshape(-1, 3)
        _act_norms = np.linalg.norm(_all_acts, axis=-1)
        _valid = _act_norms > 1e-4
        v_des_magnitude = float(np.mean(_act_norms[_valid])) * DATASET_HZ if _valid.any() else 0.4
        print(f'[ eval ] pid_const_v: v_des_magnitude={v_des_magnitude:.3f} m/s '
              f'(mean_act={np.mean(_act_norms[_valid]):.4f} m × {DATASET_HZ} Hz)')
    else:
        v_des_magnitude = 0.0   # unused by other controllers
    # U6: MJX predictive-sampling kwargs (task_id/planner_steps removed — MJX needs neither).
    mjpc_kwargs = {
        'n_trajectories': config.get('mjx_n_samples', 16),
        'horizon':        config.get('mjx_horizon', 0.3),
        'n_improve':      config.get('mjx_n_improve', 5),
        'vel_weight':     config.get('mjx_vel_weight', 0.1),
    } if controller == 'mjpc' else None
    # Eval-parameter folder — mirrors args_to_watch_fm_visual_plan naming convention.
    # Sits BETWEEN train-identity and seed; keeps variant name pure.
    # e.g.  flow_matching_v3_uav/H8_D..._9D / mpc4_pid_stopgo_T0.5 / 0 / diffuser /
    eval_params_dir = _uav_eval_tag(config, controller)

    projector = None
    if variant != 'diffuser' and not _is_hardflow(variant):
        # UAV has no semantic goal columns. SequenceDataset.get_goal_dim() can false-positive
        # on incidentally-constant channels (e.g. corridor altitude, constant p_des).
        # DC_FIX dynamics constraints touch p indices 6,7,8 — if goal_dim>0 shrinks traj_dim
        # below 9, those indices go out-of-bounds in build_matrices (IndexError: index 64).
        # Fix: always force goal_dim=0 for UAV and patch the loaded model so p_sample_loop
        # doesn't slice the trajectory before handing it to the projector.
        _detected_goal_dim = int(getattr(model_fm, 'goal_dim', 0))
        if _detected_goal_dim != 0:
            print(f'[ eval ] UAV: overriding model_fm.goal_dim {_detected_goal_dim} → 0 '
                  f'(false-positive constant channel; UAV has no goal dims)')
            model_fm.goal_dim = 0
        traj_dim = int(dataset.observation_dim + dataset.action_dim)
        _report_degenerate_dims(dataset, config)
        projector = setup_dpcc_projector(
            parsed, config,
            dataset.normalizer.normalizers['observations'],
            dataset.normalizer.normalizers['actions'],
            variant, trajectory_dim=traj_dim)
    # Gen15: `sample_kwargs` carries the engine's per-call extras. The two-time engines
    # (mf/af) take an explicit `num_steps=K` on p_sample_loop; FlowMatchingODE has no such
    # parameter and would raise TypeError, so the registry returns {} for `fm` (K is already
    # pinned on the model itself by build_experiment). This is the only engine-dependent line
    # in the whole rollout path.
    _sample_kwargs = engine_registry.sample_kwargs_for(ENGINE, config['flow_steps_v3'])

    if _is_hardflow(variant):
        # ── Gen15 U2: arm C — HardFlow's in-loop constrained sampler ────────────────────────
        # A DIFFERENT guidance mechanism, not a projection variant: DPCC generates then
        # projects; HardFlow solves a prox-NLP INSIDE each ODE step. It replaces the Policy
        # wholesale (drop-in: same call signature, same (action, Trajectories) return).
        _hf = config.get('hardflow', {}) or {}
        _row = engine_registry.get(ENGINE)
        _traj_dim = int(dataset.observation_dim + dataset.action_dim)
        _clist = setup_dpcc_projector(
            parsed, config,
            dataset.normalizer.normalizers['observations'],
            dataset.normalizer.normalizers['actions'],
            variant, trajectory_dim=_traj_dim, return_constraint_list=True)
        policy = HardFlowPolicy(
            model=model_fm, normalizer=dataset.normalizer,
            horizon=int(getattr(parsed, 'horizon', 8)),
            transition_dim=_traj_dim, action_dim=int(dataset.action_dim),
            constraint_list=_clist,
            dt=float(config.get('dt', 1.0)),          # action IS Δp_des → Euler dt=1.0
            flow_steps=int(config['flow_steps_v3']),
            preprocess_fns=getattr(parsed, 'preprocess_fns', []),
            test_ret=getattr(parsed, 'test_ret', 0),
            reg_scale=float(_hf.get('reg_scale', 1.0)),
            activation_threshold=resolve_activation_threshold(
                _hf.get('activation_threshold', config.get('diffusion_timestep_threshold', 0.5))),
            trajectory_selection=_selection_for(variant),
            candidate_cost=_hf.get('candidate_cost', 'prox'),
            dynamics_mode=_hf.get('dynamics_mode', 'deriv'),
            linear_dynamics=None,
            print_level=int(_hf.get('ipopt_print_level', 0)),
            print_time=bool(_hf.get('casadi_print_time', False)),
            device=getattr(parsed, 'device', 'cuda'),
            goal_dim=0,
            # 🔴 Both REQUIRED and both engine-specific — see engine_registry. A defaulted
            # init_noise_scale would start the `fm` arm at 2x its trained noise, silently.
            init_noise_scale=float(_row['init_noise_scale']),
            two_time=bool(_row['two_time']),
        )
        print(f'[hardflow] engine={ENGINE} K={config["flow_steps_v3"]} '
              f'noise_sigma={_row["init_noise_scale"]} two_time={_row["two_time"]} '
              f'dyn={_hf.get("dynamics_mode", "deriv")} '
              f'A={policy.sampler.activation_threshold} sel={_selection_for(variant)}')
    else:
        policy = Policy(model=model_fm, normalizer=dataset.normalizer,
                        preprocess_fns=getattr(parsed, 'preprocess_fns', []),
                        test_ret=getattr(parsed, 'test_ret', 0),
                        projector=projector, trajectory_selection=_selection_for(variant),
                        **_sample_kwargs)

    # E9 s_curve: per-replan active-set switching. If the scene declares any `x_active`
    # halfspaces, the active wall set depends on the drone's current x, so the projector must
    # be rebuilt each FM step. This closure does that (cheap matrix rebuild, small H×T); it is
    # passed to rollout_one only for such scenes — every other scene builds ONCE (rebuild=None,
    # behaviour byte-identical to before). Non-switching walls (no x_active) are unaffected.
    _has_x_active = (variant != 'diffuser') and any(
        isinstance(hs, dict) and hs.get('x_active') is not None
        for hs in (config.get('halfspace_constraints') or []))
    rebuild_projector = None
    if _has_x_active:
        def rebuild_projector(current_x, _tdim=int(dataset.observation_dim + dataset.action_dim)):
            return setup_dpcc_projector(
                parsed, config,
                dataset.normalizer.normalizers['observations'],
                dataset.normalizer.normalizers['actions'],
                variant, trajectory_dim=_tdim, current_x=current_x)

    # Path: scene_root / plans / <model_exp_noseed> / <eval_params> / <seed> / <geo_tag> / <variant> /
    # savepath = scene_root / mix_uav_<engine> / H8_...9D[_dp..._bb...] / <seed>
    # NOTE: the eval output path is built HERE, by hand, from the TRAIN block's savepath — the
    # plan block's `exp_name` / `prefix` are DEAD for pathing in the whole UAV family. Anything
    # that wants to tag the results folder must go into `_uav_eval_tag` (or the yaml it reads),
    # never into a plan-block exp_name. See logs_in_develop/more_trials_folder_Path/
    # CHANGELOG_custom_msg_path_token.md §6.
    # E9 fix1: `<geo_tag>` restores the old avoiding-task `results/halfspace_<variant>/` path
    # level — a second, swappable axis (which geometry/constraint-combo produced this run)
    # alongside the projection `<variant>` folder. Without it, two runs of the SAME scene under
    # different constraint_types (e.g. an ablation subset) would collide in one output folder.
    scene_root  = os.path.join(parsed.logbase, parsed.dataset)
    _model_dir  = os.path.relpath(os.path.dirname(parsed.savepath), scene_root)  # strip seed
    _seed_str   = os.path.basename(parsed.savepath)
    seed_dir    = os.path.join(scene_root, 'plans', _model_dir, eval_params_dir, _seed_str)
    geo_dir     = os.path.join(seed_dir, config.get('geo_tag', scene))
    # [SolverSwap] 🔴 The output folder carries the NLP backend, so an SLSQP run lands
    # BESIDE the IPOPT corpus instead of overwriting it. Under 'ipopt' the label is the
    # old name unchanged, so nothing already on disk moves. Isolating the DIRECTORY
    # isolates the npz, the eval log, the plots and the diagnostics in one move.
    variant_out = artifact_variant_label(variant, resolve_nlp_backend())
    out_dir     = os.path.join(geo_dir, variant_out)
    diag_dir    = os.path.join(out_dir, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    # ── U10.1 RUN PROVENANCE ──────────────────────────────────────────────────────────
    # Gen15 builds its output path BY HAND from the train savepath (see the note above),
    # so no Parser.mkdir runs here and neither args.json nor a config snapshot lands with
    # the results. Meanwhile UAV_MIX_FLOW_STEPS and UAV_MIX_HF_OFF silently reshape the
    # run — the latter DELETES the HardFlow arm, which is invisible in the path.
    # Written at seed_dir (not out_dir) so one record covers every geometry/variant under
    # this eval-params folder; identical configs de-duplicate. Never fatal.
    provenance.write(
        seed_dir, role='eval',
        yaml_path=os.path.join(_REPO, 'config', 'uav_projection.yaml'),
        resolved={
            'engine': ENGINE,
            'scene': scene,
            'flow_steps_v3': config.get('flow_steps_v3'),
            'diffusion_timestep_threshold': config.get('diffusion_timestep_threshold'),
            'projection_variants': config.get('projection_variants'),
            'hardflow': config.get('hardflow'),
            'hardflow_arm_disabled_by_env': bool(os.environ.get('UAV_MIX_HF_OFF')),
            'constraint_types': config.get('constraint_types'),
            'geo_tag': config.get('geo_tag'),
            'n_trials': config.get('n_trials'),
            'model_savepath': parsed.savepath,
            'eval_params_dir': eval_params_dir,
            'seed': _seed_str,
        })

    # E9 U2: constraint-geometry schematic (constraint_overview.png + .svg), mirroring
    # visual-aligning's `plot_geo_constraints` call site — once per geo_dir, before any
    # trajectory rollouts. `plot_geo_constraints` itself is idempotent (skips once both
    # files exist), so it's safe to call once per (base, tightened) margin encountered.
    # UAV's `-tightened` is a per-variant margin modifier sharing this geo_dir with its base
    # sibling (not a separate named geo entry as in visual-aligning) — so the base and
    # tightened schematics get distinct filenames in the SAME folder (see docstring).
    _variants = config['projection_variants']
    _is_this_tightened = 'tightened' in variant
    _first_of_kind = variant == next(
        (v for v in _variants if ('tightened' in v) == _is_this_tightened), variant)
    if _first_of_kind:
        os.makedirs(geo_dir, exist_ok=True)
        _basename = 'constraint_overview_tightened' if _is_this_tightened else 'constraint_overview'
        plot_geo_constraints(config.get('geo_tag', scene), config, geo_dir,
                             is_tightened=_is_this_tightened, basename=_basename)
        # [HFK1c 2026-08-30] If the degeneracy guard dropped the HardFlow arm, leave the record
        # HERE, beside the variant folders — a reader browsing the results tree must be able to
        # see that the arm was deliberately SKIPPED, not that it was never configured or that it
        # crashed. Mirrors the PROJECTION_CB_TRIPPED.txt / DIVERGENCE_ABORT.txt sentinel pattern.
        _hf_skip = config.get('hardflow_skipped')
        if _hf_skip:
            with open(os.path.join(geo_dir, 'HF_DEGENERATE_SKIPPED.txt'), 'w') as _f:
                _f.write(hardflow_skip_note(', '.join(_hf_skip['variants']),
                                            _hf_skip['K'], _hf_skip['A'], _hf_skip['reason']))

    # Write config snapshot at the eval-tag-aware seed dir (once per PROCESS, on first
    # variant/geo_tag — Fix_8). setup.py's mkdir() no longer auto-snapshots during eval
    # (save=False path); we do it here where eval_params_dir is known. Kept ABOVE geo_dir
    # (model/eval-param snapshot, not geometry-specific) so it's written once per seed
    # regardless of how many geo_tags run.
    #
    # Fix_8: guard on an in-memory set, NOT `os.path.exists(_snap_dir)`. The old filesystem
    # check meant a snapshot folder created by an OLDER run — e.g. before
    # yaml_config_snapshot_patch fixed snapshot_configs's hardcoded wrong yaml path — would
    # never be touched again by any LATER run, permanently freezing the stale/wrong file in
    # place. The in-memory set still avoids redundant re-copies WITHIN this run's
    # variant/geo_tag loop, but a fresh process (i.e. every new job submission) always
    # re-snapshots and `shutil.copy` naturally overwrites whatever was there before.
    _snap_dir = os.path.join(seed_dir, f'config_snapshot_{parsed.config.split(".")[-1]}')
    if _snap_dir not in _SNAPSHOTTED_DIRS:
        import types as _t
        _snap_args = _t.SimpleNamespace(config=parsed.config, savepath=seed_dir)
        utils.Parser().snapshot_configs(_snap_args)
        # Fix_8 cleanup: delete the leftover 'projection_eval.yaml' the OLD (pre-
        # yaml_config_snapshot_patch) bug wrote here — different filename than the fixed
        # code's 'uav_projection.yaml', so it isn't overwritten, just left as confusing
        # clutter. Safe ONLY here (config_snapshot_uav/ is UAV-specific — that file is never
        # legitimately correct content for a UAV run, unlike avoiding-family packages where
        # projection_eval.yaml IS the right file).
        _stale = os.path.join(_snap_dir, 'projection_eval.yaml')
        if os.path.exists(_stale):
            os.remove(_stale)
            print(f'[ eval ] Fix_8: removed stale wrong-package snapshot {_stale}')
        _SNAPSHOTTED_DIRS.add(_snap_dir)

    record = (args.record != 'none')
    renderer = _make_overhead_renderer(mujoco, mj_model) if record else None
    batch_size = int(config.get('mpc_batch_size', config.get('batch_size', 4)))
    # 🔴 B4_PARITY (2026-08-20) — arm C's fan comes from the variant NAME. Gen15 already gave
    # every arm the same `mpc_batch_size` (so it never had the Gen3v6/v7/Gen12 timing
    # confound), but that also made bare `hardflow_new` byte-identical to `hardflow_new-r`
    # at B=4 — both select index 0 — i.e. duplicated compute under two names. The bare name
    # now means what it says upstream: the faithful batch-1 control.
    if _is_hardflow(variant):
        batch_size = resolve_hf_batch_size(variant, batch_size)

    # U_13: FIXED episode budget for every trial of this scene — CLI --max-episode-length
    # wins, else the yaml `max_episode_length` (scalar-all or per-scene dict), else the
    # per-scene SCENE_MAX_EPISODE_LENGTH default. Replaces the per-trial random round(dur*HZ).
    _cli_mel = getattr(args, 'max_episode_length', None)
    _yaml_mel = config.get('max_episode_length')
    if isinstance(_yaml_mel, dict):
        _yaml_mel = _yaml_mel.get(scene)
    max_episode_length = int(_cli_mel or _yaml_mel or SCENE_MAX_EPISODE_LENGTH.get(scene, 500))
    print(f'[ eval ] {scene} variant={variant}: max_episode_length={max_episode_length} '
          f'(U_13 fixed budget; source: {"CLI" if _cli_mel else ("yaml" if _yaml_mel else "SCENE default")})')

    rollouts = []
    _variant_t0 = time.time()   # Fix_11: per-trial progress/ETA, so a killed job shows where
    try:
        for i in range(args.n_trials):
            homotopy = homotopies[i % len(homotopies)]
            r = rollout_one(mj_model, scene, homotopy, 10_000 + i, policy, horizon,
                            renderer=renderer, goal_radius=args.goal_radius, batch_size=batch_size,
                            variant=variant, log_dir=out_dir,
                            control_hz=config.get('control_hz', DATASET_HZ),
                            text_log=config.get('behavior_log', True),
                            controller=controller, cond_mode=cond_mode, mjpc_kwargs=mjpc_kwargs,
                            v_des_magnitude=v_des_magnitude, geo_config=config,
                            rebuild_projector=rebuild_projector,
                            max_episode_length=max_episode_length)
            artifacts.save_rollout_stats(diag_dir, i, r)
            artifacts.write_mpc_foresight(diag_dir, i, r, scene,   # real candidate-fan plot (E7)
                                          geo_config=config, variant=variant)   # Fix_14: overlay enforced constraints
            if record:
                artifacts.save_rollout_gif(diag_dir, i, r.pop('frames', None))
            else:
                r.pop('frames', None)
            rollouts.append(r)
            # Fix_11: one line per completed trial — elapsed/ETA lets you tell, from the log
            # alone, whether a variant/scene will finish before the SLURM time limit.
            _elapsed = time.time() - _variant_t0
            _avg = _elapsed / (i + 1)
            _eta = _avg * (args.n_trials - (i + 1))
            print(f'[ eval ] {scene} variant={variant}: trial {i + 1}/{args.n_trials} done  '
                  f'({_elapsed:.1f}s elapsed this variant, ~{_eta:.1f}s to go)')
    finally:
        _free_renderer(renderer)
        renderer = None

    # Fix_10 (2/2): summary mirrors rollout_one's grouped schema — same group names, `_rate`/
    # `_mean` suffixes inside each group instead of flat top-level keys.
    succ = np.mean([r['success']['strict'] for r in rollouts])
    # HFK1 (2026-08-24) — (n_active, n_genuine) for this arm; (0, 0) for the non-HardFlow arms,
    # which have no in-loop NLP and therefore no notion of a genuine step.
    _hf_budget = (hardflow_step_budget(int(config['flow_steps_v3']),
                                       float(getattr(policy.sampler, 'activation_threshold', 0.0)))
                  if _is_hardflow(variant) else (0, 0))
    summary = {
        'scene': scene, 'seed': args.seed, 'n_trials': len(rollouts), 'variant': variant,
        'physical': {
            'safe_rate': float(np.mean([r['physical']['safe'] for r in rollouts])),          # contact-free + airborne
            'contact_frac_mean': float(np.mean([r['physical']['contact_frac'] for r in rollouts])),
        },
        'constraint': {
            'collision_free_rate': float(np.mean([r['constraint']['collision_free'] for r in rollouts])),
            'n_violations_mean': float(np.mean([r['constraint']['n_violations'] for r in rollouts])),
            'total_violations_mean': float(np.mean([r['constraint']['total_violations'] for r in rollouts])),
        },
        'goal': {
            'dist_mean': float(np.mean([r['goal']['dist'] for r in rollouts])),
            'reached_rate': float(np.mean([r['goal']['reached'] for r in rollouts])),
        },
        # U_13: DPCC-style step accounting (eval.py:315 "Avg number of steps"). steps_mean over
        # ALL trials; steps_to_goal_mean over reaching trials only (the true time-to-goal, since
        # misses run the full budget and would otherwise dominate the average).
        'steps': {
            'mean': float(np.mean([r['n_fm_steps'] for r in rollouts])),
            'to_goal_mean': (float(np.mean([r['n_fm_steps'] for r in rollouts if r['goal']['reached']]))
                             if any(r['goal']['reached'] for r in rollouts) else float('nan')),
            'max_episode_length': max_episode_length,
        },
        'success': {
            'strict_rate': float(succ),                                                       # task success: goal+safe (scene-aware)
            'relaxed_rate': float(np.mean([r['success']['relaxed'] for r in rollouts])),        # U7: crossed finish line
            'strict_and_constraints_rate': float(np.mean([r['success']['strict_and_constraints'] for r in rollouts])),
            'relaxed_and_constraints_rate': float(np.mean([r['success']['relaxed_and_constraints'] for r in rollouts])),
        },
        'timing': {
            'fm_ms_mean': float(np.mean([r['timing']['fm_ms_mean'] for r in rollouts])),
            'fm_ms_p95': float(np.max([r['timing']['fm_ms_p95'] for r in rollouts])),
            'proj_ms_mean': float(np.mean([r['timing']['proj_ms_mean'] for r in rollouts])),
            'total_ms_mean': float(np.mean([r['timing']['total_ms_mean'] for r in rollouts])),
            'total_ms_p95': float(np.max([r['timing']['total_ms_p95'] for r in rollouts])),
            'total_over_budget': int(np.sum([r['timing']['total_over_budget'] for r in rollouts])),
            'budget_ms': rollouts[0]['timing']['budget_ms'] if rollouts else float('nan'),
        },
        'track_err_mean': float(np.mean([r['track_err_mean'] for r in rollouts])),
        'projection': variant,
        # Div_Abort: variant-level rollup. n_aborted > 0 means some trials were cut short
        # because the drone lost control — those rows are misses by construction (safe forced
        # False) and their constraint counts cover fewer steps, so a DA comparing violation
        # COUNTS across variants must account for them.
        'divergence': {
            'n_aborted_trials': int(sum(1 for r in rollouts if r.get('divergence', {}).get('aborted'))),
            'aborted_trials': [i for i, r in enumerate(rollouts) if r.get('divergence', {}).get('aborted')],
            'reasons': {i: r['divergence']['reason'] for i, r in enumerate(rollouts)
                        if r.get('divergence', {}).get('aborted')},
            'enabled': bool(DIVERGENCE_ABORT),
        },
        # Fix_15.3: variant-level projection-circuit-breaker rollup. `n_tripped_trials` > 0 means
        # the sustained-slowness breaker (projection.py Fix_15.2) OPENED on some trials, which ran
        # (partly) UNPROJECTED — treat this variant as "projection broken for this geometry", not a
        # valid result. Also dropped as a PROJECTION_CB_TRIPPED.txt sentinel in the variant dir.
        'projection_health': {
            'n_tripped_trials': int(sum(1 for r in rollouts if r.get('projection_health', {}).get('cb_tripped'))),
            'total_skipped_steps': int(sum(r.get('projection_health', {}).get('cb_skipped_steps', 0) for r in rollouts)),
            'tripped_trials': [i for i, r in enumerate(rollouts) if r.get('projection_health', {}).get('cb_tripped')],
        },
        # ── Gen15 U2 — HardFlow accounting. ⚠️ FAIRNESS: `hardflow_new` evaluates the network
        # twice per ACTIVE ODE step (reference step + terminal predict), so an arm-C run costs
        # K + n_active - 1 network evals against a DPCC arm's K. Comparing at "the same K" is
        # therefore comparing a smaller generation budget on the DPCC side — EXCEPT at K=1,
        # where the two are now equal (HFK1 2026-08-24 removed the terminal lookahead call,
        # whose weight (1 - tau) was exactly zero). Record the real count so the DA can
        # normalise; `nfe_per_plan` is the number to quote.
        #
        # ⚠️ HFK1 (2026-08-24): `n_genuine` is the honest "is this HardFlow at all?" field.
        # A step is genuinely HardFlow only if it is active AND non-terminal — at the terminal
        # step tau=1 kills the endpoint lookahead, the damped pull-back and the feedback alike.
        # n_genuine == 0 => this row is Pi_S(Euler sample): sample-then-project, == DPCC modulo
        # solver/variable-scope, and it must NOT be reported as a HardFlow result. Always true
        # at K=1; also at K=2 under the shipped A=0.5.
        # See logs_in_develop/aggregated_hardflow_lowK/
        'hardflow': ({
            'is_hardflow': True,
            'nfe_total': int(getattr(policy.sampler, 'nfe', 0)),
            'nlp_solves_total': int(getattr(policy.nlp, 'n_solves', 0)),
            'nlp_failures_total': int(getattr(policy.nlp, 'n_failures', 0)),
            # [SolverSwap] 'slsqp' (DPCC scipy) or 'ipopt' (original CasADi). Read off
            # the live NLP object so it reflects what RAN, env override included.
            'nlp_backend': str(getattr(policy.nlp, 'nlp_backend', 'n/a')),
            'nlp_backend_slsqp': float(getattr(policy.nlp, 'nlp_backend', '') == 'slsqp'),
            # one 'plan' = one outer FM/MPC step; summed over all rollouts of this variant.
            'nfe_per_plan': (float(getattr(policy.sampler, 'nfe', 0))
                             / max(sum(int(r['n_fm_steps']) for r in rollouts), 1)),
            'activation_threshold': float(getattr(policy.sampler, 'activation_threshold', 0.0)),
            'init_noise_scale': float(getattr(policy.sampler, 'init_noise_scale', 0.0)),
            'two_time': bool(getattr(policy.sampler, 'two_time', False)),
            'n_active': int(_hf_budget[0]),
            'n_genuine': int(_hf_budget[1]),
            'is_degenerate': bool(_hf_budget[1] == 0),
        } if _is_hardflow(variant) else {'is_hardflow': False}),
    }

    # ── Artifacts (legacy schema): results.json + npz + log + 2-D overview ──
    json_rollouts = artifacts.json_safe_rollouts(rollouts)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'rollouts': json_rollouts}, f, indent=2)
    npz_path = artifacts.save_npz(out_dir, variant_out, rollouts, vars(args))
    artifacts.write_eval_log(out_dir, variant_out, summary, rollouts)
    artifacts.plot_overview(out_dir, variant_out, scene, rollouts)

    # Fix_15.3: drop a greppable sentinel in the variant dir when the projection circuit breaker
    # tripped, so a tripped (UNPROJECTED, invalid-constraint) result is obvious from the file tree
    # alone without opening the npz/log. The artifacts themselves are still written (partial run is
    # kept), just clearly marked. See projection.py Fix_15.2.
    _ph = summary['projection_health']
    if _ph['n_tripped_trials'] > 0:
        with open(os.path.join(out_dir, 'PROJECTION_CB_TRIPPED.txt'), 'w') as _f:
            _f.write(f"PROJECTION CIRCUIT-BREAKER TRIPPED — {scene} variant={variant}\n")
            _f.write(f"tripped_trials={_ph['tripped_trials']}  "
                     f"({_ph['n_tripped_trials']}/{len(rollouts)})\n")
            _f.write(f"total_skipped_steps={_ph['total_skipped_steps']}\n")
            _f.write("Cause: sustained SLSQP slowness (projection.py Fix_15.2 breaker OPENED).\n")
            _f.write("These trials ran (partly) UNPROJECTED — constraint metrics are NOT valid;\n")
            _f.write("treat this variant as 'projection broken for this geometry'.\n")
        print(f'[ eval ] {scene} variant={variant}: ⚠ PROJECTION CIRCUIT-BREAKER TRIPPED on '
              f'{_ph["n_tripped_trials"]}/{len(rollouts)} trials ({_ph["total_skipped_steps"]} '
              f'steps skipped) — results marked UNPROJECTED. See PROJECTION_CB_TRIPPED.txt.', flush=True)

    # [HFK1c 2026-08-30] A DEGENERATE HardFlow row can only reach this point under an explicit
    # FMPCC_HF_ALLOW_DEGENERATE=1 opt-in. Mark it in the file tree too, so the row is
    # identifiable without opening results.json — same reasoning as the CB sentinel above.
    _hfs = summary.get('hardflow') or {}
    if _hfs.get('is_hardflow') and _hfs.get('is_degenerate'):
        with open(os.path.join(out_dir, 'HF_DEGENERATE.txt'), 'w') as _f:
            _f.write(f"DEGENERATE HardFlow arm — {scene} variant={variant}\n")
            _f.write(f"K={config['flow_steps_v3']}  A={_hfs.get('activation_threshold')}  "
                     f"n_active={_hfs.get('n_active')}  n_genuine=0\n\n")
            _f.write("NO HardFlow arithmetic ran: every NLP solve is the terminal tau=1 solve,\n")
            _f.write("so this row is Pi_S(Euler sample) = sample-then-project, == DPCC modulo\n")
            _f.write("solver/variable-scope. It is a valid SOLVER comparison; it is NOT a\n")
            _f.write("HardFlow result and must not carry a HardFlow claim.\n\n")
            _f.write("It exists because FMPCC_HF_ALLOW_DEGENERATE=1 was set for this job.\n")
            _f.write("See logs_in_develop/aggregated_hardflow_lowK/\n")
        print(f'[ eval ] {scene} variant={variant}: ⚠ DEGENERATE HardFlow arm (n_genuine=0) — '
              f'ran under FMPCC_HF_ALLOW_DEGENERATE=1. NOT a HardFlow result. '
              f'See HF_DEGENERATE.txt.', flush=True)

    # Div_Abort: greppable sentinel when any trial of this variant lost control, mirroring the
    # PROJECTION_CB_TRIPPED.txt convention — visible from the file tree without opening artifacts.
    _dv = summary['divergence']
    if _dv['n_aborted_trials'] > 0:
        with open(os.path.join(out_dir, 'DIVERGENCE_ABORT.txt'), 'w') as _f:
            _f.write(f"DIVERGENCE ABORT — {scene} variant={variant}\n")
            _f.write(f"aborted_trials={_dv['aborted_trials']} "
                     f"({_dv['n_aborted_trials']}/{len(rollouts)})\n\n")
            for _i in _dv['aborted_trials']:
                _d = rollouts[_i]['divergence']
                _f.write(f"  trial {_i}: reason={_d['reason']}  step={_d['step']}/"
                         f"{rollouts[_i]['max_episode_length']}  t={_d['time_s']:.3f}s\n")
                _f.write(f"            p={_d['p']}  p_des={_d['p_des']}  "
                         f"|v|={_d['speed']:.2f} m/s  |p_des-p|={_d['p_des_lead']:.2f} m\n")
                _f.write(f"            why: {_d['detail']}\n")
            _f.write("\nThese rollouts were STOPPED early (the drone had lost control). They are\n")
            _f.write("scored as misses (physical.safe forced False) and their step count is charged\n")
            _f.write("the full budget; constraint counts cover only the steps actually flown.\n")
        print(f'[ eval ] {scene} variant={variant}: ⚠ DIVERGENCE ABORT on '
              f'{_dv["n_aborted_trials"]}/{len(rollouts)} trials {_dv["aborted_trials"]} — '
              f'see DIVERGENCE_ABORT.txt', flush=True)

    _steps_tg = summary['steps']['to_goal_mean']
    print(f'[ eval ] {scene} variant={variant} (B={batch_size}, proj={"on" if projector else "off"}, '
          f'sel={_selection_for(variant)}): success={succ:.3f}  success_relaxed={summary["success"]["relaxed_rate"]:.3f}  '
          f'safe={summary["physical"]["safe_rate"]:.3f}  goal_reached={summary["goal"]["reached_rate"]:.3f}  '
          f'steps_to_goal={_steps_tg:.0f}/{max_episode_length}  '
          f'track_err={summary["track_err_mean"]:.3f}  → {os.path.dirname(npz_path)}/')
    # Real-time timing verdict echoed to stdout (per-step detail stays in the .log files).
    # Fix_10 follow-up: these were left reading the old flat top-level keys after the
    # nested-schema migration moved everything under summary['timing'] — KeyError on
    # 'budget_ms' at runtime. Updated to match the print statement just above.
    _budget = summary['timing']['budget_ms']
    _rt = 'SAFE' if summary['timing']['total_over_budget'] == 0 else f'OVER×{summary["timing"]["total_over_budget"]}'
    print(f'[ eval ] {scene} variant={variant} TIMING: fm_ms={summary["timing"]["fm_ms_mean"]:.1f} '
          f'proj_ms={summary["timing"]["proj_ms_mean"]:.1f} total_ms={summary["timing"]["total_ms_mean"]:.1f} '
          f'(p95={summary["timing"]["total_ms_p95"]:.1f}) budget={_budget}ms → real_time_{_rt}')
    return summary


def eval_scene(scene, args):
    """Run EVERY projection variant (diffuser, dpcc-r/-c/-t) for EVERY active geo variant of
    one scene (Fix_6). A scene may have several active `geo_constraint_variants` entries at
    once (e.g. `active_geo_variants` listing `s_curve_dynamics_only`,
    `s_curve_dynamics_bounds_only`, AND `s_curve_combined_1` together — exactly the case that
    used to raise in `load_pcc_config`) — this now runs ALL of them in one job submission,
    each writing to its own `geo_tag`-named output folder (Fix_1), instead of erroring.

    Model+dataset loaded ONCE and shared across geo variants (constraint geometry doesn't
    change them); a Policy is (re)built per (geo variant, projection variant) pair, same as
    before. Returns {variant: summary} when exactly one geo variant is active for this scene
    (the common case, byte-identical to the pre-Fix_6 return shape) — else
    {geo_variant_name: {variant: summary}}."""
    import mujoco
    import uav_expert_data_collect.generator as gen
    # Gen15: base_cfg is resolved BEFORE the model so the plan block's K can be pinned onto the
    # loaded diffusion object (Gen11 built the model first and K never reached it — see the
    # module docstring). _load_base_cfg needs only (scene, seed), so the reorder is safe.
    base_cfg = _load_base_cfg(scene, args.seed)
    model_fm, dataset, parsed, horizon = build_experiment(
        scene, args.seed, args.epoch, args.device, flow_steps=base_cfg['flow_steps_v3'])
    homotopies = gen.HOMOTOPY_CLASSES[scene]
    mj_model = mujoco.MjModel.from_xml_path(gen.SCENE_XMLS[scene])

    _matches = _resolve_active_geo_matches(scene, base_cfg)
    _entries = _matches if _matches else [None]   # None → dynamics-only global fallback (unchanged)
    if len(_entries) > 1:
        print(f"[ eval ] {scene}: {len(_entries)} active geo variants in one job (Fix_6): "
              f"{[e['name'] for e in _entries]}")

    all_summaries = {}
    for _gi, entry in enumerate(_entries):
        if len(_entries) > 1:
            print(f'[ eval ] {scene}: geo entry {_gi + 1}/{len(_entries)}: '
                  f'\'{entry["name"] if entry is not None else "(none)"}\'')
        config = _apply_geo_entry(base_cfg, scene, entry)

        # cond_mode is a MODEL property (obs layout baked into the normalizer at train time).
        # Lock it to what the checkpoint was actually trained with — ignore the plan block
        # value, which is user-editable and can silently mismatch (crash: shapes (9,) vs (6,)).
        config['cond_mode'] = str(getattr(parsed, 'cond_mode', config.get('cond_mode', 'p_des')))
        print(f'[ eval ] cond_mode={config["cond_mode"]}  (source: train checkpoint args)')

        # Tightened variants only differ from their base siblings when spatial constraints
        # (geo_bounds/halfspace/obstacles) are active — enlarge_constraints is applied there.
        # 'bounds' (the action-magnitude family, Patch_Constraints_C3) is NEVER tightened — it's
        # a dataset-range cap, not a spatial surface — so it's excluded from this set on purpose.
        # With only 'dynamics'+'bounds' in constraint_types the enlarge margin is computed but
        # never used, so tightened == non-tightened == wasted compute. Skip them and say why.
        _spatial = {'geo_bounds', 'halfspace', 'obstacles'}
        _has_spatial = bool(_spatial & set(config.get('constraint_types', [])))
        if not _has_spatial:
            _skip = [v for v in config['projection_variants'] if 'tightened' in v]
            if _skip:
                print(f'[ eval ] {scene}: skipping {len(_skip)} tightened variants '
                      f'(no spatial constraints in constraint_types — enlarge has no effect): {_skip}')
            config['projection_variants'] = [v for v in config['projection_variants'] if 'tightened' not in v]

        print(f'[ eval ] {scene} [geo_tag={config["geo_tag"]}]: variants={config["projection_variants"]}  '
              f'constraints={config["constraint_types"]}  batch_size={config.get("mpc_batch_size", config.get("batch_size", 4))}')

        # Fix_12: sanity-gate the geometry BEFORE burning GPU time — warn if any expert
        # route is infeasible under the planning margin (the bug class this fix repaired).
        _warn_expert_route_infeasibility(scene, config, homotopies)
        summaries = {}
        _n_variants = len(config['projection_variants'])
        for _vi, variant in enumerate(config['projection_variants']):
            # Fix_11: this is the line to grep for after a 24h-timeout job — the LAST one
            # printed tells you which variant was running (or just-finished) when it got cut.
            print(f'[ eval ] {scene} [geo_tag={config["geo_tag"]}] '
                  f'>>> variant {_vi + 1}/{_n_variants}: \'{variant}\'  (n_trials={args.n_trials})')
            summaries[variant] = _run_variant(scene, variant, model_fm, dataset, parsed, horizon,
                                              config, args, mj_model, mujoco, homotopies)
        all_summaries[entry['name'] if entry is not None else config['geo_tag']] = summaries

    # Preserve the pre-Fix_6 flat {variant: summary} shape for the single-geo-variant case
    # (the overwhelming common case, and what any external caller/aggregator expects).
    if len(_entries) == 1:
        return next(iter(all_summaries.values()))
    return all_summaries


def main():
    global ENGINE
    args, remaining = parse_args()

    # Publish the engine selection to module scope BEFORE anything resolves a config block or
    # a savepath — build_experiment, _load_base_cfg and _uav_eval_tag all read it.
    ENGINE = args.engine
    _row = engine_registry.get(ENGINE)
    print(f'[ eval ] Gen15 UAV Mix-ML — engine: {ENGINE}  ({_row["label"]})')

    # --flow-steps K overrides the plan block for this process. Exported via the environment so
    # _load_base_cfg picks it up without threading an extra argument through eval_scene.
    if args.flow_steps is not None:
        os.environ['UAV_MIX_FLOW_STEPS'] = str(int(args.flow_steps))
        print(f'[ eval ] K override from --flow-steps: {args.flow_steps}')

    # ── Resolve seed and n_trials: CLI wins; else read from config/uav_projection.yaml ──
    # We do this BEFORE stripping sys.argv so the yaml path is resolved cleanly.
    import yaml as _yaml
    _yaml_path = os.path.join(_REPO, 'config', 'uav_projection.yaml')
    try:
        with open(_yaml_path) as _fh:
            _proj_defaults = _yaml.safe_load(_fh) or {}
    except FileNotFoundError:
        _proj_defaults = {}

    _seed_from_cli = args.seed is not None
    args.seed = args.seed if _seed_from_cli else int(_proj_defaults.get('seed', 6))
    print(f'[ eval ] seed={args.seed}  (source: {"--seed CLI" if _seed_from_cli else _yaml_path})')

    _trials_from_cli = args.n_trials is not None
    args.n_trials = args.n_trials if _trials_from_cli else int(_proj_defaults.get('n_trials', 20))
    print(f'[ eval ] n_trials={args.n_trials}  (source: {"--n-trials CLI" if _trials_from_cli else _yaml_path})')

    # utils.Parser.parse_args() (called inside build_experiment) re-parses sys.argv with its
    # own argparse that only knows --config/--seed — strip our already-consumed flags
    # first or it chokes on --scene/--n-trials/--projection/--device (mirrors train_fm_uav.py).
    sys.argv = [sys.argv[0], *remaining]
    scenes = SCENES if args.scene == 'all' else [args.scene]
    # Fix_11: progress breadcrumb — with `--scene all` this is the outermost of 4 nested
    # loops (scene → geo entry → projection variant → trial); if a 24h SLURM job gets killed
    # mid-run, this line (plus the ones added in eval_scene/_run_variant) is what tells you
    # WHERE, instead of the log just going silent after the last completed variant.
    summaries = {}
    for _si, _s in enumerate(scenes):
        if len(scenes) > 1:
            print(f'[ eval ] ══ scene {_si + 1}/{len(scenes)}: \'{_s}\' ══')
        summaries[_s] = eval_scene(_s, args)

    if len(scenes) > 1:
        # experimental --scene all path; per-scene runs use aggregate_scene_summaries.py instead.
        # Gen15 writes under logs/UAV_MIX (never logs/UAV_FM — that root is Gen11's), and the
        # engine is part of the rollup path so a cross-scene run of one arm can never land on
        # another arm's SUMMARY.json.
        roll = os.path.join('logs', 'UAV_MIX', 'uav-all', 'plans',
                            f'E{ENGINE}', args.projection, 'SUMMARY.json')
        os.makedirs(os.path.dirname(roll), exist_ok=True)
        with open(roll, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f'[ eval ] cross-scene rollup → {roll}')
    print(f'UAV Mix-ML eval complete (engine={ENGINE}).')


if __name__ == '__main__':
    main()
