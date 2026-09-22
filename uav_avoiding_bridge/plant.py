"""`UavAvoidingPlant` — the quadrotor stand-in for `ObstacleAvoidanceEnv` (Gen15 U18).

Honours the five-call contract every avoiding eval loop uses, in avoiding units in and out:

    env.start(); obs = env.reset()                       # obs: (2,) float32, mapped drone xy
    action = env.robot_state()[:2]; fixed_z = env.robot_state()[2:]
    obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0])))
    success = info[1]                                    # info = (mode_encoding, success)
    env.close()

Inside: the setpoint [x_des, y_des] is mapped by frame.to_world_xy, z is held at frame.ALTITUDE, and the existing
CascadedPID (uav_env_test/flight_controller.py, built via uav_expert_data_collect.generator._make_pid) tracks it for
one control period. Success = mapped y_a > frame.GOAL_Y_A (the Panda's check_success). A drone-pillar contact ends
the episode as a failure (the Panda's check_failure on the rod), so does leaving the arena or a runaway speed.

Replay policies (also used by turbo.py):
    clock   fixed control period 1/control_hz per setpoint, velocity feed-forward (p_des - p_des_prev)/T   [default]
    settle  hold the setpoint until ||p - p_des|| < settle_eps (cap settle_max_s), no feed-forward: geometric flyability

Nothing here touches the model, the projector or the constraints — those stay in the avoiding frame.
"""
import json
import os
import time

import numpy as np

from uav_avoiding_bridge import frame as F
from uav_avoiding_bridge.scene import scene_xml_path, build_scene_xml

# divergence thresholds (values copied from mix_uav_test/eval_mix_uav.py Div_Abort; not imported on purpose)
DIV_ENVELOPE_SLACK_M = 2.0
DIV_SPEED_MAX_MS = 6.0
Z_PLACEHOLDER = 0.12          # what robot_state()[2] returns; the loop passes it back untouched and we ignore it


def _drone_reach_m():
    return 0.31               # rotor reach of the X2 (trajectories.PILLAR_ROTOR_REACH); diagnostics only


class UavAvoidingPlant:
    def __init__(self, scale=None, control_hz=5.0, altitude=None, feedforward=True, gain='pid_default',
                 replay='clock', settle_eps=0.05, settle_max_s=2.0, terminate_on_contact=True,
                 max_steps_per_episode=250, xml_path=None, verbose=True):
        import mujoco
        import uav_expert_data_collect.generator as gen
        self._mujoco, self._gen = mujoco, gen
        self.scale = float(F.SCALE if scale is None else scale)
        self.altitude = float(F.ALTITUDE if altitude is None else altitude)
        self.control_hz = float(control_hz)
        self.feedforward = bool(feedforward)
        self.gain = gain
        assert replay in ('clock', 'settle'), replay
        self.replay = replay
        self.settle_eps, self.settle_max_s = float(settle_eps), float(settle_max_s)
        self.terminate_on_contact = bool(terminate_on_contact)
        self.max_steps_per_episode = int(max_steps_per_episode)

        path = xml_path or scene_xml_path(self.scale)
        if not os.path.exists(path):                       # a scale variant nobody generated yet
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as fh:
                fh.write(build_scene_xml(self.scale))
        self.xml_path = path
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self._check_scene_matches_frame()
        self.pid = gen._make_pid(self.model, gain)
        self.dt = float(self.model.opt.timestep)
        self.n_sub = max(1, int(round(1.0 / (self.dt * self.control_hz))))
        self.period_s = self.n_sub * self.dt
        (x0, x1), (y0, y1) = F.world_field(self.scale)
        self.arena_lb = np.array([x0 - DIV_ENVELOPE_SLACK_M, y0 - DIV_ENVELOPE_SLACK_M, 0.15])
        self.arena_ub = np.array([x1 + DIV_ENVELOPE_SLACK_M, y1 + DIV_ENVELOPE_SLACK_M, self.altitude + DIV_ENVELOPE_SLACK_M])
        self._start_w = np.array([*F.to_world_xy(*F.START_A, self.scale), self.altitude])
        self.episode = -1
        self.records = []           # per-episode dicts (sidecar)
        self._rec = None
        self._init_episode_state()
        if verbose:
            print(f'[ uav-plant ] scene={os.path.basename(path)} scale={self.scale:g} alt={self.altitude:g} '
                  f'control {self.control_hz:g} Hz -> {self.n_sub} physics steps of {self.dt:g} s per setpoint '
                  f'(period {self.period_s:.3f} s), replay={self.replay}, feedforward={self.feedforward}, '
                  f'gain={gain}, contact terminates={self.terminate_on_contact}', flush=True)

    # ── contract ────────────────────────────────────────────────────────────────────────────────
    def start(self):
        pass

    def reset(self, random=True, context=None):
        self._finish_record()
        self.episode += 1
        self._init_episode_state()
        d, m = self.data, self.model
        self._mujoco.mj_resetData(m, d)
        d.qpos[:3] = self._start_w
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[:] = 0.0
        self._mujoco.mj_forward(m, d)
        # settle burst at hover so the first observation is at rest (the Panda's start() does a 0.5 s goto)
        for _ in range(int(round(0.5 / self.dt))):
            self._ctrl_step(self._start_w, np.zeros(3))
        self._p_des_w = self._start_w.copy()
        self._rec = {'episode': self.episode, 'steps': [], 'contact_steps': 0, 'diverged': None,
                     'success': False, 'ended': None, 'start_w': self._start_w.tolist()}
        return self._obs2()

    def robot_state(self):
        x, y = self._obs2()
        return np.array([x, y, Z_PLACEHOLDER], dtype=float)

    def step(self, action, gripper_width=None):
        sp = np.asarray(action, float)[:2]
        info = self.track(sp)
        obs = self._obs2()
        success = bool(obs[1] > F.GOAL_Y_A)
        self._check_mode(obs)
        done = (success or (info['contact'] and self.terminate_on_contact) or info['diverged'] is not None
                or self.env_step_counter >= self.max_steps_per_episode - 1)
        self.env_step_counter += 1
        self.success = self.success or success
        if done:
            self.terminated = True
            self.last_reason = ('success' if success else 'contact' if (info['contact'] and self.terminate_on_contact)
                                else info['diverged'] if info['diverged'] else 'cap')
            self._rec['success'], self._rec['ended'] = bool(self.success), self.last_reason
        return obs, 0.0, bool(done), (self.mode_encoding.copy(), self.success)

    def close(self):
        self._finish_record()

    # ── the core: track one avoiding-frame setpoint for one control period ──────────────────────
    def track(self, sp_a):
        P = np.array([*F.to_world_xy(float(sp_a[0]), float(sp_a[1]), self.scale), self.altitude])
        if self.replay == 'clock':
            v_des = (P - self._p_des_w) / self.period_s if self.feedforward else np.zeros(3)
            n_steps, settled = self.n_sub, None
        else:
            v_des, n_steps, settled = np.zeros(3), int(round(self.settle_max_s / self.dt)), False
        contact, errs, diverged, t0 = False, [], None, time.perf_counter()
        for i in range(n_steps):
            hit = self._ctrl_step(P, v_des)
            contact = contact or hit
            errs.append(float(np.linalg.norm(self.data.qpos[:3] - P)))
            if hit and self.terminate_on_contact:
                break
            diverged = self._divergence()
            if diverged:
                break
            if self.replay == 'settle' and errs[-1] < self.settle_eps:
                settled = True
                break
        self._p_des_w = P
        p = self.data.qpos[:3].copy()
        gap_a = float(np.linalg.norm(np.asarray(sp_a, float) - np.asarray(F.to_avoiding_xy(p[0], p[1], self.scale))))
        info = {'contact': bool(contact), 'diverged': diverged, 'track_err_end': errs[-1] if errs else 0.0,
                'track_err_max': max(errs) if errs else 0.0, 'gap_a': gap_a, 'settled': settled,
                'phys_steps': len(errs), 'p_w': p.tolist(), 'p_des_w': P.tolist(),
                'speed': float(np.linalg.norm(self.data.qvel[:3])), 'wall_ms': (time.perf_counter() - t0) * 1e3}
        if self._rec is not None:
            row = {k: info[k] for k in ('track_err_end', 'track_err_max', 'gap_a', 'phys_steps', 'speed')}
            row.update({'p_w': info['p_w'], 'contact': info['contact']})
            self._rec['steps'].append(row)
            self._rec['contact_steps'] += int(contact)
            if diverged:
                self._rec['diverged'] = diverged
        return info

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────
    def _ctrl_step(self, P, v_des):
        d, m = self.data, self.model
        u = self.pid.compute(d.qpos[:3].copy(), d.qpos[3:7].copy(), d.qvel[:3].copy(), d.qvel[3:6].copy(), P, v_des)
        d.ctrl[:4] = u
        self._mujoco.mj_step(m, d)
        return any(self._gen._is_obstacle_contact(m, d.contact[ci]) for ci in range(d.ncon))

    def _divergence(self):
        p, v = self.data.qpos[:3], self.data.qvel[:3]
        if np.any(p < self.arena_lb) or np.any(p > self.arena_ub):
            return f'left_arena p={np.round(p, 2).tolist()}'
        if np.linalg.norm(v) > DIV_SPEED_MAX_MS:
            return f'speed {np.linalg.norm(v):.1f} m/s'
        if not np.all(np.isfinite(p)):
            return 'nan_state'
        return None

    def _obs2(self):
        p = self.data.qpos[:3]
        return np.asarray(F.to_avoiding_xy(float(p[0]), float(p[1]), self.scale), dtype=np.float32)

    def obs4(self):
        """[x_des, y_des, x, y] with x_des = the current setpoint (what the eval loop assembles)."""
        sp = F.to_avoiding_xy(self._p_des_w[0], self._p_des_w[1], self.scale)
        return np.concatenate((np.asarray(sp, np.float32), self._obs2()))

    def _init_episode_state(self):
        self.env_step_counter, self.terminated, self.success, self.last_reason = 0, False, False, None
        self.l1_passed = self.l2_passed = self.l3_passed = False
        self.mode_encoding = np.zeros(2 + 3 + 4)
        self._p_des_w = self._start_w.copy() if hasattr(self, '_start_w') else None

    def _check_mode(self, obs):
        # verbatim ObstacleAvoidanceEnv.check_mode, on avoiding-frame coordinates
        r_x_pos, r_y_pos = float(obs[0]), float(obs[1])
        l1_y, l2_y, l3_y = -0.1, -0.1 + 0.18, -0.1 + 2 * 0.18
        l1_x, l2t, l2b, l3t, l3m, l3b = 0.5, 0.425, 0.575, 0.35, 0.5, 0.65
        if r_y_pos - 0.03 <= l1_y <= r_y_pos + 0.03 and not self.l1_passed:
            if r_x_pos < l1_x: self.mode_encoding[0] = 1
            elif r_x_pos > l1_x: self.mode_encoding[1] = 1
            self.l1_passed = True
        if r_y_pos - 0.03 <= l2_y <= r_y_pos + 0.03 and not self.l2_passed:
            if r_x_pos < l2t: self.mode_encoding[2] = 1
            elif l2t < r_x_pos < l2b: self.mode_encoding[3] = 1
            elif r_x_pos > l2b: self.mode_encoding[4] = 1
            self.l2_passed = True
        if r_y_pos >= l3_y and not self.l3_passed:
            if r_x_pos < l3t: self.mode_encoding[5] = 1
            if l3t < r_x_pos < l3m: self.mode_encoding[6] = 1
            elif l3m < r_x_pos < l3b: self.mode_encoding[7] = 1
            elif r_x_pos > l3t: self.mode_encoding[8] = 1
            self.l3_passed = True

    def _check_scene_matches_frame(self):
        m, mj = self.model, self._mujoco
        for n, xa, ya, r in F.OBSTACLES_A:
            gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f'pillar_{n}')
            assert gid >= 0, f'scene {self.xml_path} lacks geom pillar_{n}'
            X, Y = F.to_world_xy(xa, ya, self.scale)
            assert np.allclose(m.geom_pos[gid][:2], [X, Y], atol=1e-3), (n, m.geom_pos[gid][:2], (X, Y))
            assert abs(float(m.geom_size[gid][0]) - self.scale * r) < 1e-3, (n, m.geom_size[gid][0], self.scale * r)

    def _finish_record(self):
        if self._rec is not None and self._rec['steps']:
            st = self._rec['steps']
            te = np.array([s['track_err_end'] for s in st]); gap = np.array([s['gap_a'] for s in st])
            self._rec['summary'] = {
                'n_steps': len(st), 'track_err_mean_m': float(te.mean()), 'track_err_p95_m': float(np.percentile(te, 95)),
                'track_err_max_m': float(max(s['track_err_max'] for s in st)),
                'gap_a_mean': float(gap.mean()), 'gap_a_p95': float(np.percentile(gap, 95)), 'gap_a_max': float(gap.max()),
                'contact_steps': int(self._rec['contact_steps']), 'diverged': self._rec['diverged'],
                'success': bool(self._rec['success']), 'ended': self._rec['ended'],
                'speed_max_ms': float(max(s['speed'] for s in st)),
            }
            self.records.append(self._rec)
        self._rec = None

    def settings(self):
        return {'scale': self.scale, 'altitude': self.altitude, 'control_hz': self.control_hz, 'period_s': self.period_s,
                'physics_dt': self.dt, 'feedforward': self.feedforward, 'gain': self.gain, 'replay': self.replay,
                'settle_eps': self.settle_eps, 'settle_max_s': self.settle_max_s,
                'terminate_on_contact': self.terminate_on_contact, 'scene_xml': self.xml_path,
                'origin_a': list(F.ORIGIN_A), 'drone_reach_m': _drone_reach_m()}

    def save_records(self, path, extra=None):
        self._finish_record()
        eps = []
        for r in self.records:
            e = {k: v for k, v in r.items() if k != 'steps'}
            e['path_w'] = [s['p_w'] for s in r['steps']]
            eps.append(e)
        out = {'settings': self.settings(), 'episodes': eps}
        if extra:
            out.update(extra)
        with open(path, 'w') as fh:
            json.dump(out, fh, indent=1)
        return path
