"""Real-Time Behaviour Text Logger — digital-twin audit for UAV FM-PCC eval.

Plan: logs_in_develop/Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/PLAN.md
Idea: logs_in_develop/REALTIME_RECORDING/IDEAS.md

The headline product is TIMING: can FM-PCC close the 33 Hz / 30 ms control loop, and how
much of that budget is FM inference vs the PCC projector? A GIF captures none of this; a
structured text log captures all of it.

System-agnostic by design (IDEAS.md §"Implementation approach"): the `diffuser` variant has
no projector, so its PCC block prints `status=OFF` and `proj_ms=0` — same grammar, empty
fields. The per-step `.step(...)` call wraps measurements the eval loop already takes, so
the logger adds ~zero latency (it never inserts new compute, only formats existing numbers).

ARCHITECTURAL NOTE (differs from IDEAS.md): our PCC projects INSIDE the flow-matching ODE
loop, not as a post-FM QP filter. So there is no clean `fm_cmd -> qp_cmd` override on the
executed action. We log `proj_ms` (projector wall-time) and `proj_cost` (the projection
cost of the selected candidate) as the honest analogs. Spatial-constraint margin fields are
scaffolded but blank this epoch (dynamics-only; no scene geometry yet — see U3 CHANGELOG).
"""

import os
import numpy as np


def _fmt_vec(v, p=3):
    if v is None:
        return 'None'
    a = np.asarray(v, dtype=float).reshape(-1)
    return '(' + ','.join(f'{x:.{p}f}' for x in a) + ')'


def _fmt_horizon(h, max_steps=4, p=3):
    """Format the FM H-step Δp_des foresight as a compact list, truncated for readability."""
    if h is None:
        return 'None'
    a = np.asarray(h, dtype=float)
    if a.ndim == 1:                         # single (3,) action → wrap
        a = a.reshape(1, -1)
    rows = [_fmt_vec(a[i], p) for i in range(min(len(a), max_steps))]
    tail = ',...' if len(a) > max_steps else ''
    return '[' + ','.join(rows) + tail + ']'


class BehaviorLogger:
    """Accumulates per-step lines + a summary block for ONE rollout episode.

    Usage (in rollout_one):
        logger = BehaviorLogger(episode_id, variant, scene, homotopy,
                                control_hz=33, batch_size=4, horizon=8)
        for k in range(n_fm):
            ... measure fm_ms_total, proj_ms ...
            logger.step(t=..., obs=..., fm_horizon=..., fm_ms=..., proj_ms=...,
                        proj_cost=..., proj_active=..., state_p=..., state_v=...,
                        contact=..., track_err=...)
        logger.save(path); summary = logger.summary_dict()
    """

    def __init__(self, episode_id, variant, scene, homotopy, system='Gen11E7_UAV_FMPCC',
                 control_hz=33, batch_size=1, horizon=8, node=None, text_log=True):
        self.episode_id = episode_id
        self.variant = variant
        self.scene = scene
        self.homotopy = homotopy
        self.system = system
        self.control_hz = float(control_hz)
        self.budget_ms = 1000.0 / float(control_hz)
        self.batch_size = batch_size
        self.horizon = horizon
        self.node = node or os.environ.get('SLURMD_NODENAME') or os.environ.get('HOSTNAME', 'unknown')
        # text_log=False: skip per-step string formatting (the only loop-path overhead).
        # Raw timing stats (fm_ms/proj_ms/total_ms) are always collected — they land in
        # results.json regardless. text_log only gates the .log file and per-step text blocks.
        self.text_log = text_log

        self._lines = []
        self.fm_ms = []
        self.proj_ms = []
        self.total_ms = []
        self.proj_active_steps = 0
        self.n_contacts = 0
        self._contact_lines = []
        self.track_errs = []

    def step(self, t, obs, fm_horizon, fm_ms, proj_ms, proj_cost,
             proj_active, state_p, state_v, contact, track_err,
             constraint='dynamics', step_idx=None):
        """Record one FM control step. `contact` is a truthy contact descriptor or falsy/None.

        Raw timing stats are ALWAYS collected (they land in results.json regardless of
        text_log). Per-step string formatting is skipped when text_log=False — this is the
        only loop-path overhead that could add noise to timing measurements.
        """
        total = float(fm_ms) + float(proj_ms)
        self.fm_ms.append(float(fm_ms))
        self.proj_ms.append(float(proj_ms))
        self.total_ms.append(total)
        if track_err is not None and np.isfinite(track_err):
            self.track_errs.append(float(track_err))
        if proj_active:
            self.proj_active_steps += 1
        if contact:
            self.n_contacts += 1

        if not self.text_log:
            return   # skip all string formatting — stats already recorded above

        verdict = '✅' if total <= self.budget_ms else '❌ OVER'
        sidx = '' if step_idx is None else f'step={step_idx}  '
        head = (f'═══ T={t:.3f}s  {sidx}total_ms={total:.1f}  '
                f'[BUDGET={self.budget_ms:.1f}ms {verdict}] ═══════════════')
        obs = np.asarray(obs, dtype=float).reshape(-1)
        p_des, p, v = obs[0:3], obs[3:6], obs[6:9]
        block = [
            head,
            f'OBS       p_des={_fmt_vec(p_des)}  p={_fmt_vec(p)}  v={_fmt_vec(v)}',
            f'          track_err={track_err:.3f}m' if track_err is not None else '          track_err=NA',
            f'FM        fm_ms={fm_ms:.1f}   horizon={_fmt_horizon(fm_horizon)}   (H={self.horizon}, B={self.batch_size} fan)',
        ]
        if proj_active:
            block.append(f'PCC       proj_ms={proj_ms:.1f}  constraint={constraint}  status=ON  proj_cost={proj_cost:.4f}')
        else:
            block.append('PCC       proj_ms=0.0  status=OFF (no projector)')
        ct = 'NONE'
        if contact:
            ct = str(contact)
            self._contact_lines.append(f'  T={t:.3f}s  contact={ct}  fm_horizon0={_fmt_horizon(fm_horizon, 1)}  track_err={track_err:.3f}m')
        block.append(f'STATE     p={_fmt_vec(state_p)}  v={_fmt_vec(state_v)}  contact={ct}')
        block.append('')
        self._lines.extend(block)

    # ── summary ──────────────────────────────────────────────────────────────
    def _stat(self, arr):
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return (float('nan'),) * 3 + (0,)
        over = int(np.sum(a > self.budget_ms))
        return float(np.mean(a)), float(np.max(a)), float(np.percentile(a, 95)), over

    def summary_dict(self):
        fm_mean, fm_max, fm_p95, fm_over = self._stat(self.fm_ms)
        pj_mean, pj_max, pj_p95, _ = self._stat(self.proj_ms)
        tt_mean, tt_max, tt_p95, tt_over = self._stat(self.total_ms)
        n = len(self.total_ms)
        return {
            'episode': self.episode_id, 'variant': self.variant, 'system': self.system,
            'scene': self.scene, 'homotopy': self.homotopy, 'node': self.node,
            'steps': n, 'control_hz': self.control_hz, 'budget_ms': round(self.budget_ms, 1),
            'fm_ms_mean': fm_mean, 'fm_ms_max': fm_max, 'fm_ms_p95': fm_p95, 'fm_over_budget': fm_over,
            'proj_ms_mean': pj_mean, 'proj_ms_max': pj_max, 'proj_ms_p95': pj_p95,
            'total_ms_mean': tt_mean, 'total_ms_max': tt_max, 'total_ms_p95': tt_p95,
            'total_over_budget': tt_over,
            'proj_active_steps': self.proj_active_steps,
            'contacts': self.n_contacts,
            'max_track_err': float(np.max(self.track_errs)) if self.track_errs else float('nan'),
        }

    def summary_block(self, behaviour=None):
        """Render the `# SUMMARY` text block. `behaviour` = optional dict of result fields."""
        s = self.summary_dict()
        n = max(s['steps'], 1)
        L = [
            '# ' + '─' * 70,
            f'# SUMMARY  episode={s["episode"]}  variant={s["variant"]}  system={s["system"]}',
            f'#          scene={s["scene"]}  homotopy={s["homotopy"]}  node={s["node"]}',
            f'#          steps={s["steps"]}  control_hz={s["control_hz"]:.0f}  budget_ms={s["budget_ms"]}',
            '#          TIMING:',
            f'#            fm_ms    mean={s["fm_ms_mean"]:.1f}  max={s["fm_ms_max"]:.1f}  p95={s["fm_ms_p95"]:.1f}  over_budget={s["fm_over_budget"]}/{n}',
            f'#            proj_ms  mean={s["proj_ms_mean"]:.1f}  max={s["proj_ms_max"]:.1f}  p95={s["proj_ms_p95"]:.1f}',
            f'#            total_ms mean={s["total_ms_mean"]:.1f}  max={s["total_ms_max"]:.1f}  p95={s["total_ms_p95"]:.1f}  over_budget={s["total_over_budget"]}/{n} ({100.0*s["total_over_budget"]/n:.1f}%)',
            f'#            real_time_safe={"YES" if s["total_over_budget"] == 0 else "NO"}  (measured on {s["node"]} — cluster latency, NOT target drone)',
        ]
        if behaviour:
            L.append('#          BEHAVIOUR:')
            bl = '  '.join(f'{k}={v}' for k, v in behaviour.items())
            L.append(f'#            {bl}')
        L.append(f'#          proj_active_steps={s["proj_active_steps"]}/{n}  contacts={s["contacts"]}  max_track_err={s["max_track_err"]:.3f}m')
        if self._contact_lines:
            L.append('#          CONTACTS:')
            L.extend('# ' + c for c in self._contact_lines)
        L.append('# ' + '─' * 70)
        return '\n'.join(L)

    def save(self, path, behaviour=None):
        if not self.text_log:
            return None   # stats still live in summary_dict(); no file written
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(self._lines))
            f.write('\n')
            f.write(self.summary_block(behaviour))
            f.write('\n')
        return path
