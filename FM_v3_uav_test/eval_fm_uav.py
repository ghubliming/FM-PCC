"""Minimal closed-loop UAV FM eval (Gen11 Epoch 6).

Deliberately NOT forked from the 700-line D3IL/minari-coupled FMv3-ODE eval — that base
is the wrong shape for UAV and impossible to debug. Instead this mirrors the *known-good*
expert control loop (`uav_expert_data_collect/generator.run_trial`) and swaps the expert
trajectory for the trained FM policy. One scene at a time, receding-horizon (MPC) execution.

Multi-rate control (IMPORTANT):
  • physics + PID run every `dt = model.opt.timestep`
  • the FM predicts Δp_des at the DATASET rate (DATASET_HZ = 33 Hz)
  → the FM is queried every `decim = round(1/(dt·33))` physics steps; p_des is zero-order
    held between queries while the PID tracks it. This matches how the data was recorded.

Per FM step:
  obs = [p_des | p | v]  (9-D, raw) → policy → first Δp_des (3-D) → p_des += Δp_des →
  PID tracks p_des for `decim` physics steps → obs updated from new (p, v).

SUCCESS CRITERION — DESIGN DECISION (please confirm):
  The state-only FM is conditioned on obs[0]=[p_des,p,v] only, so for the random-goal
  `empty` scene it is NOT goal-conditioned → "reach the goal" is ill-defined there. We
  therefore default `success` to the expert's OWN acceptance gate — **contact-free +
  airborne** (contact_frac < scene limit AND airborne) — which is well-defined for every
  scene. For the geometry-constrained scenes we ALSO report distance to the expert path
  endpoint (`goal_dist`). If you want goal-reaching as the headline metric, the FM must be
  made goal-conditioned first (next Epoch / Gen7).

No torch/MuJoCo in the Docker dev env — this is cluster-only; here it is syntax-checked.
"""

import os
import sys
import json
import time
import argparse

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

import flow_matcher_v3_uav.utils as utils
from flow_matcher_v3_uav.sampling.policies import Policy
import FM_v3_uav_test.eval_artifacts as artifacts

SCENES = ['empty', 'corridor', 's_curve', 'pillars']
DATASET_HZ = 33                      # must match uav_expert_data_collect/dataset_writer.py
GOAL_RADIUS = 0.30                   # m — secondary goal-reach tolerance (constrained scenes)


def parse_args():
    p = argparse.ArgumentParser(description='Closed-loop UAV FM evaluation.')
    p.add_argument('--scene', type=str, default='all', choices=['all', *SCENES],
                   help="Scene(s) to eval: 'all' runs each scene and rolls up SUMMARY.json.")
    p.add_argument('--seed', type=int, default=5, help='Trained-model seed to load.')
    p.add_argument('--n-trials', type=int, default=20, help='Closed-loop rollouts per scene.')
    p.add_argument('--epoch', type=str, default='latest', help="Checkpoint epoch ('latest' or int).")
    p.add_argument('--projection', type=str, default='fm_only',
                   help="Projection variant for the output subfolder. 'fm_only' (state-only FM, no DPCC); "
                        "DPCC variants (dpcc-c, …) slot in here when Phase-3 lands.")
    p.add_argument('--record', type=str, default='none', choices=['none', 'gif', 'all'],
                   help="Overhead-render GIFs per rollout. 'none' (default, fast) adds ~0 overhead; "
                        "'gif'/'all' render frames and write diagnostics/rollout_<r>.gif.")
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_known_args()


def build_policy(scene, seed, epoch, device):
    """Resolve the trained model dir via the same Parser wiring as train, then load it."""
    class Parser(utils.Parser):
        dataset: str = 'uav'              # overridden on the instance below
        config: str = 'config.uav'
    p = Parser()
    p.dataset = f'uav-{scene}'            # → data branch + output path segregation
    args = p.parse_args(experiment='flow_matching_v3_uav', seed=seed)

    ep = epoch if epoch == 'latest' else int(epoch)
    experiment = utils.load_diffusion(args.savepath, epoch=ep, device=device)
    fm_model = experiment.diffusion
    dataset = experiment.dataset
    policy = Policy(
        model=fm_model,
        normalizer=dataset.normalizer,
        preprocess_fns=getattr(args, 'preprocess_fns', []),
        test_ret=getattr(args, 'test_ret', 0),
    )
    return policy, args, int(getattr(args, 'horizon', 8))


def _make_overhead_renderer(mujoco, model, res=360):
    """Headless overhead renderer; None if rendering is unavailable (no hard dep).

    ONE renderer is created per scene and reused across rollouts — never one per
    rollout — so we allocate exactly one EGL/GL context instead of leaking N of them.
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


def rollout_one(model, scene, homotopy, trial_seed, policy, horizon, renderer=None, frame_stride=2):
    """One closed-loop MuJoCo rollout. Mirrors generator.run_trial; FM replaces traj_fn.

    `model` and `renderer` are owned by eval_scene and shared across rollouts (one
    GL context per scene, not per rollout). Buffers obs/action/plan per FM step (U3
    npz schema); if `renderer` is given, also captures overhead frames for a GIF.
    Heavy arrays/frames are returned under HEAVY_KEYS and stripped from results.json.
    """
    import mujoco
    import uav_expert_data_collect.generator as gen

    rng = np.random.default_rng(trial_seed)
    data = mujoco.MjData(model)

    traj_fn, init_pos, dur = gen._build_traj_and_init(scene, homotopy, rng)
    goal = np.asarray(traj_fn(dur)[0], dtype=float)        # expert path endpoint (secondary metric)

    data.qpos[:3] = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pid = gen._make_pid(model, 'pid_default')
    dt = float(model.opt.timestep)
    dt_fm = 1.0 / DATASET_HZ
    decim = max(1, int(round(1.0 / (dt * DATASET_HZ))))    # physics steps per FM query
    n_fm = int(round(dur * DATASET_HZ))

    frames = []

    p_des = np.asarray(init_pos, dtype=float).copy()
    n_hit = 0
    n_phys = 0
    min_z = float('inf')
    track_err = []
    fm_ms = []
    obs_traj = []        # realized [p_des|p|v] per FM step  → npz obs_all
    act_traj = []        # FM Δp_des per FM step             → npz act_all
    plans = []           # FM H-step predicted obs plan      → npz sampled_trajectories_all

    for k in range(n_fm):
        p = data.qpos[:3].copy()
        v = data.qvel[:3].copy()
        obs = np.concatenate([p_des, p, v]).astype(np.float32)   # [p_des | p | v] (9,) raw

        t0 = time.perf_counter()
        action, traj = policy({0: obs}, batch_size=1, horizon=horizon)
        fm_ms.append((time.perf_counter() - t0) * 1e3)

        action = np.asarray(action, dtype=float).reshape(-1)[:3]  # first Δp_des
        obs_traj.append(obs)
        act_traj.append(action.astype(np.float32))
        # traj.observations = FM's unnormalized H-step plan in obs space (the foresight).
        plan = getattr(traj, 'observations', None)
        if plan is not None:
            plans.append(np.asarray(plan, dtype=np.float32))

        p_des = p_des + action
        v_des = action / dt_fm

        for _ in range(decim):
            p = data.qpos[:3].copy()
            v = data.qvel[:3].copy()
            q = data.qpos[3:7].copy()
            om = data.qvel[3:6].copy()
            u = pid.compute(p, q, v, om, p_des, v_des)
            data.ctrl[:4] = u
            mujoco.mj_step(model, data)
            n_phys += 1
            if any(gen._is_obstacle_contact(model, data.contact[ci]) for ci in range(data.ncon)):
                n_hit += 1
            min_z = min(min_z, float(data.qpos[2]))
            track_err.append(float(np.linalg.norm(data.qpos[:3] - p_des)))

        if renderer is not None and (k % frame_stride == 0):
            try:
                frames.append(_render_overhead(mujoco, model, data, renderer))
            except Exception as exc:                       # pragma: no cover
                print(f'[ eval ] frame render failed ({exc}); stopping capture')
                renderer = None     # stop capturing for THIS rollout; eval_scene still owns/frees it

    p_final = data.qpos[:3].copy()
    contact_frac = n_hit / max(n_phys, 1)
    limit = gen.SCENE_MAX_CONTACT_FRACTION.get(scene, gen.MAX_CONTACT_FRACTION)
    airborne = bool(min_z > 0.2)                           # crude floor gate
    success = bool(contact_frac <= limit and airborne)
    goal_dist = float(np.linalg.norm(p_final - goal))

    return {
        'scene': scene, 'homotopy': homotopy,
        'success': success,
        'contact_frac': contact_frac,
        'goal_dist': goal_dist,
        'goal_reached': bool(goal_dist < GOAL_RADIUS),
        'min_z': min_z,
        'final_z': float(p_final[2]),
        'track_err_mean': float(np.mean(track_err)) if track_err else float('nan'),
        'fm_ms_mean': float(np.mean(fm_ms)) if fm_ms else float('nan'),
        'fm_ms_p95': float(np.percentile(fm_ms, 95)) if fm_ms else float('nan'),
        'n_fm_steps': n_fm, 'decim': decim, 'dt': dt,
        # ── heavy (npz / gif only; stripped from results.json) ──
        'obs_traj': np.asarray(obs_traj),
        'act_traj': np.asarray(act_traj),
        'plans': plans,
        'frames': frames,
    }


def eval_scene(scene, args):
    import mujoco
    import uav_expert_data_collect.generator as gen
    policy, parsed, horizon = build_policy(scene, args.seed, args.epoch, args.device)
    homotopies = gen.HOMOTOPY_CLASSES[scene]

    variant = args.projection
    # Eval outputs go under a sibling `plans/` tree (like the other FM-PCC models),
    # NOT nested inside the train model folder. Train weights stay at parsed.savepath;
    # eval lands at  <logbase>/<dataset>/plans/<exp_name>/<seed>/<projection>/ .
    scene_root = os.path.join(parsed.logbase, parsed.dataset)        # logs/UAV_FM/uav-<scene>
    sub = os.path.relpath(parsed.savepath, scene_root)               # flow_matching_v3_uav/<exp>/<seed>
    out_dir = os.path.join(scene_root, 'plans', sub, variant)        # …/plans/…/<seed>/<projection>
    diag_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    record = (args.record != 'none')

    # ONE model + ONE renderer per scene (NOT per rollout) → exactly one GL context.
    model = mujoco.MjModel.from_xml_path(gen.SCENE_XMLS[scene])
    renderer = _make_overhead_renderer(mujoco, model) if record else None

    rollouts = []
    try:
        for i in range(args.n_trials):
            homotopy = homotopies[i % len(homotopies)]
            r = rollout_one(model, scene, homotopy, 10_000 + i, policy, horizon, renderer=renderer)
            # Per-rollout diagnostics while this rollout's heavy data is still in hand.
            artifacts.save_rollout_stats(diag_dir, i, r)
            artifacts.write_pcc_placeholder(diag_dir, i)    # PCC foresight stub (Epoch 7)
            if record:
                artifacts.save_rollout_gif(diag_dir, i, r.pop('frames', None))
            else:
                r.pop('frames', None)
            rollouts.append(r)
    finally:
        # Free the GL context here, while EGL is still initialized (avoids the
        # EGL_NOT_INITIALIZED teardown error); renderer is no longer needed below.
        _free_renderer(renderer)
        renderer = None

    succ = np.mean([r['success'] for r in rollouts])
    summary = {
        'scene': scene, 'seed': args.seed, 'n_trials': len(rollouts),
        'success_rate': float(succ),
        'contact_frac_mean': float(np.mean([r['contact_frac'] for r in rollouts])),
        'goal_dist_mean': float(np.mean([r['goal_dist'] for r in rollouts])),
        'goal_reached_rate': float(np.mean([r['goal_reached'] for r in rollouts])),
        'track_err_mean': float(np.mean([r['track_err_mean'] for r in rollouts])),
        'fm_ms_mean': float(np.mean([r['fm_ms_mean'] for r in rollouts])),
        'fm_ms_p95': float(np.max([r['fm_ms_p95'] for r in rollouts])),
        'projection': variant,
    }

    # ── Artifacts (legacy schema): results.json (summary) + npz + log + 2-D overview ──
    json_rollouts = artifacts.json_safe_rollouts(rollouts)
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'rollouts': json_rollouts}, f, indent=2)
    npz_path = artifacts.save_npz(out_dir, variant, rollouts, vars(args))
    artifacts.write_eval_log(out_dir, variant, summary, rollouts)
    png_path = artifacts.plot_overview(out_dir, variant, scene, rollouts)

    print(f'[ eval ] {scene} seed={args.seed} proj={variant}: '
          f'success={succ:.3f}  contact={summary["contact_frac_mean"]:.3f}  '
          f'fm_ms(mean/p95)={summary["fm_ms_mean"]:.1f}/{summary["fm_ms_p95"]:.1f}\n'
          f'         → {out_dir}/  (results.json, {os.path.basename(npz_path)}, '
          f'{os.path.basename(png_path)}, eval_{variant}.log, diagnostics/)')
    return summary


def main():
    args, remaining = parse_args()
    # utils.Parser.parse_args() (called inside build_policy) re-parses sys.argv with its
    # own argparse that only knows --config/--seed — strip our already-consumed flags
    # first or it chokes on --scene/--n-trials/--projection/--device (mirrors train_fm_uav.py).
    sys.argv = [sys.argv[0], *remaining]
    scenes = SCENES if args.scene == 'all' else [args.scene]
    summaries = {s: eval_scene(s, args) for s in scenes}

    if len(scenes) > 1:
        # experimental --scene all path; per-scene runs use aggregate_scene_summaries.py instead.
        roll = os.path.join('logs', 'UAV_FM', 'uav-all', 'plans', args.projection, 'SUMMARY.json')
        os.makedirs(os.path.dirname(roll), exist_ok=True)
        with open(roll, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f'[ eval ] cross-scene rollup → {roll}')
    print('UAV FM eval complete.')


if __name__ == '__main__':
    main()
