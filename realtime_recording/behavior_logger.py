# REAL_TIME_RECORDING_UPDATE
"""Portable Real-Time Behaviour Recorder — cross-gen digital-twin audit.

This is the shared, obs-layout-agnostic generalisation of the Gen11 UAV logger
(`FM_v3_uav_test/behavior_logger.BehaviorLogger`). Same grammar, same SUMMARY block,
same headline question: **can this system close its control loop inside the real-time
budget, and how much of the budget is FM inference vs the PCC projector?**

Reference / spec:
  logs_in_develop/REALTIME_RECORDING/IDEAS.md
  logs_in_develop/REALTIME_RECORDING/PATCH_TODO_RealTime_Recording.md

Why a separate portable class (vs reusing the UAV one verbatim):
  * The UAV logger hardcodes the 9-D obs layout [p_des|p|v]. Other gens are 2-D position
    tasks (avoiding/aligning) or manipulation (D3IL) with different obs vectors.
  * Most non-UAV evals measure ONE bundled `policy()` wall-time (FM + in-loop projection
    together) because their `policies.py` does not expose `projection_ms` separately. This
    recorder accepts that: pass `total_ms` and leave `fm_ms`/`proj_ms` unset → it records
    `total_ms` as the headline number and marks the split as bundled. When the diffuser /
    no-projector variant runs, `total_ms` IS pure FM time (proj=0).

Design contract (IDEAS.md §"Implementation approach"): the recorder only FORMATS numbers the
eval loop already measured — it never inserts new compute, so it adds ~zero loop latency.
`text_log=False` skips per-step string building entirely; raw timing stats are still kept.
"""

import os
import numpy as np


def _fmt_vec(v, p=3):
    if v is None:
        return 'None'
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 0:
        return '()'
    if a.size > 9:                       # keep long obs vectors readable
        a = a[:9]
        return '(' + ','.join(f'{x:.{p}f}' for x in a) + ',...)'
    return '(' + ','.join(f'{x:.{p}f}' for x in a) + ')'


def _fmt_horizon(h, max_steps=4, p=3):
    """Format an FM H-step foresight (or single action) as a compact, truncated list."""
    if h is None:
        return 'None'
    a = np.asarray(h, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    rows = [_fmt_vec(a[i], p) for i in range(min(len(a), max_steps))]
    tail = ',...' if len(a) > max_steps else ''
    return '[' + ','.join(rows) + tail + ']'


class RTRecorder:
    """Accumulates per-step timing lines + a SUMMARY block for ONE rollout episode.

    Usage (drop into any eval rollout loop, around the EXISTING timing measurement):

        rec = RTRecorder(episode_id, variant, scene, system='Gen7_VisualAligning_FM',
                         control_hz=control_hz, batch_size=B, horizon=H)
        for _ in range(max_steps):
            start = time.time()
            action, samples = policy(...)          # FM (+ in-loop projection)
            total_ms = (time.time() - start) * 1e3
            rec.step(t=_/control_hz, total_ms=total_ms, obs=obs, action=action,
                     pos=obs[[xi, yi]], proj_active=(variant != 'diffuser'),
                     track_err=err, contact=violated, step_idx=_)
        rec.save(f'{save_path}/realtime_{variant}.log', behaviour={...})
        stats = rec.summary_dict()                 # also stash into npz/json if desired

    `total_ms` is mandatory (the headline). `fm_ms`/`proj_ms` are optional:
      * both unset  → split unknown; total recorded, fm shown as bundled(=total), proj=0.
      * proj_ms set → fm_ms = total - proj_ms (Gen11 in-loop convention).
      * fm_ms set   → used verbatim; proj defaults 0.
    """

    def __init__(self, episode_id, variant, scene='', system='FMPCC',
                 control_hz=33, batch_size=1, horizon=8, node=None, text_log=True):
        self.episode_id = episode_id
        self.variant = variant
        self.scene = scene
        self.system = system
        self.control_hz = float(control_hz) if control_hz else 0.0
        self.budget_ms = (1000.0 / self.control_hz) if self.control_hz > 0 else float('inf')
        self.batch_size = batch_size
        self.horizon = horizon
        self.node = node or os.environ.get('SLURMD_NODENAME') or os.environ.get('HOSTNAME', 'unknown')
        self.text_log = text_log

        self._lines = []
        self.fm_ms = []
        self.proj_ms = []
        self.total_ms = []
        self.bundled = False            # True if any step had no fm/proj split
        self.proj_active_steps = 0
        self.n_contacts = 0
        self._contact_lines = []
        self.track_errs = []

    def step(self, t, total_ms, obs=None, action=None, pos=None,
             fm_ms=None, proj_ms=None, proj_active=False, proj_cost=0.0,
             horizon=None, contact=False, track_err=None,
             constraint='dynamics', step_idx=None):
        """Record one control step. Raw timing stats are ALWAYS kept; per-step text is
        gated by `text_log`. `contact` is any truthy descriptor (or count > 0)."""
        total = float(total_ms)
        if proj_ms is None and fm_ms is None:
            fm = total           # bundled: cannot split FM vs projector
            pj = 0.0
            self.bundled = True
        elif fm_ms is None:
            pj = float(proj_ms)
            fm = max(total - pj, 0.0)
        else:
            fm = float(fm_ms)
            pj = float(proj_ms) if proj_ms is not None else 0.0

        self.fm_ms.append(fm)
        self.proj_ms.append(pj)
        self.total_ms.append(total)
        if track_err is not None and np.isfinite(track_err):
            self.track_errs.append(float(track_err))
        if proj_active:
            self.proj_active_steps += 1
        if contact:
            self.n_contacts += 1

        if not self.text_log:
            return

        verdict = '✅' if total <= self.budget_ms else '❌ OVER'
        sidx = '' if step_idx is None else f'step={step_idx}  '
        head = (f'═══ T={t:.3f}s  {sidx}total_ms={total:.1f}  '
                f'[BUDGET={self.budget_ms:.1f}ms {verdict}] ═══════════════')
        block = [head]
        if obs is not None:
            block.append(f'OBS       {_fmt_vec(obs)}' +
                         (f'   pos={_fmt_vec(pos)}' if pos is not None else ''))
        if self.bundled and fm_ms is None and proj_ms is None:
            block.append(f'FM+PCC    total_ms={total:.1f} (bundled — policy() FM+projection)   '
                         f'horizon={_fmt_horizon(horizon if horizon is not None else action)}   '
                         f'(H={self.horizon}, B={self.batch_size})')
        else:
            block.append(f'FM        fm_ms={fm:.1f}   horizon={_fmt_horizon(horizon if horizon is not None else action)}   '
                         f'(H={self.horizon}, B={self.batch_size})')
            if proj_active:
                block.append(f'PCC       proj_ms={pj:.1f}  constraint={constraint}  status=ON  proj_cost={proj_cost:.4f}')
            else:
                block.append('PCC       proj_ms=0.0  status=OFF (no projector)')
        if track_err is not None:
            block.append(f'          track_err={track_err:.3f}')
        ct = 'NONE'
        if contact:
            ct = str(contact)
            self._contact_lines.append(
                f'  T={t:.3f}s  contact={ct}  track_err='
                f'{(track_err if track_err is not None else float("nan")):.3f}')
        if action is not None:
            block.append(f'ACT       {_fmt_vec(action)}  contact={ct}')
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
            'scene': self.scene, 'node': self.node, 'bundled_timing': self.bundled,
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
        s = self.summary_dict()
        n = max(s['steps'], 1)
        bnote = '  (timing BUNDLED: FM+projection measured together)' if s['bundled_timing'] else ''
        L = [
            '# ' + '─' * 70,
            f'# SUMMARY  episode={s["episode"]}  variant={s["variant"]}  system={s["system"]}',
            f'#          scene={s["scene"]}  node={s["node"]}',
            f'#          steps={s["steps"]}  control_hz={s["control_hz"]:.0f}  budget_ms={s["budget_ms"]}',
            f'#          TIMING:{bnote}',
            f'#            fm_ms    mean={s["fm_ms_mean"]:.1f}  max={s["fm_ms_max"]:.1f}  p95={s["fm_ms_p95"]:.1f}  over_budget={s["fm_over_budget"]}/{n}',
            f'#            proj_ms  mean={s["proj_ms_mean"]:.1f}  max={s["proj_ms_max"]:.1f}  p95={s["proj_ms_p95"]:.1f}',
            f'#            total_ms mean={s["total_ms_mean"]:.1f}  max={s["total_ms_max"]:.1f}  p95={s["total_ms_p95"]:.1f}  over_budget={s["total_over_budget"]}/{n} ({100.0*s["total_over_budget"]/n:.1f}%)',
            f'#            real_time_safe={"YES" if s["total_over_budget"] == 0 else "NO"}  (measured on {s["node"]} — host latency, NOT target hardware)',
        ]
        if behaviour:
            L.append('#          BEHAVIOUR:')
            L.append('#            ' + '  '.join(f'{k}={v}' for k, v in behaviour.items()))
        L.append(f'#          proj_active_steps={s["proj_active_steps"]}/{n}  contacts={s["contacts"]}  max_track_err={s["max_track_err"]:.3f}')
        if self._contact_lines:
            L.append('#          CONTACTS:')
            L.extend('# ' + c for c in self._contact_lines)
        L.append('# ' + '─' * 70)
        return '\n'.join(L)

    def save(self, path, behaviour=None):
        if not self.text_log:
            return None
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(self._lines))
            f.write('\n')
            f.write(self.summary_block(behaviour))
            f.write('\n')
        return path
