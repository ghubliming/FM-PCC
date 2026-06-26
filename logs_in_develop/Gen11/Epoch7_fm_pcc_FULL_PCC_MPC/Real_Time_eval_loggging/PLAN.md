# PLAN — Real-Time Behaviour Text Logging for Gen11 E7 UAV FM-PCC

**Status:** IMPLEMENTED 2026-06-25 — see [`CHANGELOG.md`](CHANGELOG.md) (Phases 0–4 done; act_raw→act_proj deferred)
**Date:** 2026-06-25
**Source idea:** [`logs_in_develop/REALTIME_RECORDING/IDEAS.md`](../../../REALTIME_RECORDING/IDEAS.md) — "Real-Time Behavior Recording — Digital Twin Audit Framework"
**Target code:** `FM_v3_uav_test/eval_fm_uav.py` (`rollout_one`), building on the U3 variant suite ([`../U3/CHANGELOG.md`](../U3/CHANGELOG.md))

---

## Goal (one sentence)

Add a **per-step structured text log** (NOT a GIF) to the Gen11 E7 UAV eval that answers
the headline question from IDEAS.md: **can FM-PCC close the 33 Hz / 30 ms control loop on
real hardware, and how much of that budget is FM inference vs the PCC projector?**

The GIF/SVG stays as-is (intuition). The text log is the **evidence / digital twin**.

---

## Why this is cheap for us: half of it already exists

`rollout_one` (`eval_fm_uav.py:297–384`) already:

- runs at the real control rate (`DATASET_HZ = 33` → budget = `1000/33 = 30.3 ms`)
- times the policy call per FM step: `t0 = time.perf_counter(); policy(...); fm_ms.append(...)` (lines 302–304)
- buffers `obs_traj`, `act_traj`, `plans` (the FM H-step foresight) per step
- tracks `track_err`, `min_z`, contact (`n_hit`), and already emits `fm_ms_mean` / `fm_ms_p95` in the summary (lines 376–377)

So we are NOT building timing from scratch — we are **splitting, structuring, and emitting**
what the loop already measures.

---

## The ONE architectural divergence from IDEAS.md (must be honest about this)

IDEAS.md models the pipeline as a **post-FM QP safety filter**:

```
obs → FM infer (fm_ms) → DPCC/QP solve (qp_ms) → fm_cmd OVERRIDDEN to qp_cmd → execute
```

That is the classic DPCC architecture. **Our FM-PCC is different:** the projector runs
**inside** the flow-matching ODE sampling loop (`flow_matcher_v3_uav/models/diffusion.py:263–269`
calls `projector.project()` on the tail FM ODE steps), not as a separate filter on the
executed action. Consequences for the log:

| IDEAS.md field | Our reality | What we log instead |
|---|---|---|
| `fm_ms` (inference only) | bundled with projection inside `policy(...)` | **split required** — see Phase 1 |
| `qp_ms` (separate QP) | projection is interleaved into ODE steps | sum of per-`project()` wall-times within the step |
| `fm_cmd → qp_cmd` override delta | no post-hoc override exists; projection reshapes the *sampled trajectory*, not a final action | log **projection cost** (`projection_costs` already returned by `project()`) + raw-vs-projected first-action delta captured inside sampling |
| DPCC `status=ACTIVE/IDLE`, `nearest_constraint`, `margin` | only the **dynamics** constraint is active this epoch; no spatial geometry (U3) → no obstacle margins yet | log `constraint=dynamics status=ALWAYS_ON`; spatial margins become real only when geometry is designed (future epoch) |

**Bottom line:** the timing block (the headline product) ports cleanly. The
override/constraint-margin block is partially N/A this epoch and is scaffolded for when
spatial constraints land — exactly mirroring the U3 "wired but no-op" pattern.

---

## Per-step log format (adapted to our FM-PCC, dynamics-only epoch)

```
═══ T=0.061s  step=2/207  total_ms=6.4  [BUDGET=30.3ms ✅] ═══════════════
OBS       p_des=(−3.158,0.002,0.901)  p=(−3.171,0.001,0.900)  v=(0.42,0.01,0.00)
          track_err=0.013m
FM        fm_ms=5.1   horizon=[(0.022,0.000,0.000),(0.021,0.000,0.000),...]   (H=8, B=4 fan)
PCC       proj_ms=1.3  constraint=dynamics  status=ON  proj_cost=0.004
          act_raw=(0.024,0.001,−0.002) → act_proj=(0.022,0.000,0.000)   ← Euler-coherence snap
STATE     p=(−3.149,0.001,0.901)  v=(0.45,0.01,0.00)  contact=NONE
```

`diffuser` variant: `PCC` block prints `status=OFF (no projector)`, `proj_ms=0.0`, and no
`act_raw→act_proj` line — same grammar, fields just empty (system-agnostic, per IDEAS.md).

### Summary block (the first thing to read)

```
# SUMMARY  episode=pillars_RLR_seed6_t0  variant=dpcc-t  system=Gen11E7_UAV_FMPCC
#          steps=207  duration=6.27s  control_hz=33  budget_ms=30.3
#          TIMING:
#            fm_ms    mean=5.1  max=28.7  p95=6.2  over_budget=0/207
#            proj_ms  mean=1.3  max=3.0   p95=2.1
#            total_ms mean=6.4  max=30.9  p95=8.0  over_budget=1/207 (0.5%)
#          vs diffuser baseline (same seed/scene):
#            fm_overhead  mean=+1.3ms/step (projector)   fits_budget=YES
#          BEHAVIOUR:
#            result=FAIL(goal)  goal_dist=0.975m  safe=True  contacts=0  contact_frac=0.0
#            min_z=1.107  max_track_err=0.05m
```

---

## Implementation phases

### Phase 0 — `BehaviorLogger` module (new file)
- `FM_v3_uav_test/behavior_logger.py`: a system-agnostic logger with `.step(**fields)` and
  `.save(path)` / `.summary()` — exactly the grammar in IDEAS.md §"Implementation approach".
- Pure formatting + percentile math; zero added latency (wraps existing measurements).
- Emits `<episode>.log` next to the existing artifacts in `plans/<variant>/`.

### Phase 1 — split `fm_ms` into `fm_ms` + `proj_ms` (the only real plumbing)
- Today line 302–304 times the whole `policy(...)` call (FM + projection bundled).
- Add an optional timing hook so `Projector.project()` accumulates its own wall-time per
  policy call (a counter on the projector reset each `policy()` invocation), returned via
  the `infos`/`traj` already passed back from `policy(...)`.
- `fm_ms_pure = fm_ms_bundled − proj_ms`. No behaviour change; purely instrumentation.

### Phase 2 — capture raw-vs-projected first action + projection cost
- `project()` already returns `projection_costs` (`diffusion.py:265`). Thread it out to
  `rollout_one` alongside the chosen action.
- Capture the FM's pre-projection first action vs post-projection first action to populate
  the `act_raw → act_proj` line (the closest honest analog to IDEAS.md's override delta).

### Phase 3 — wire logger into `rollout_one`
- Instantiate `BehaviorLogger(episode_id, variant, scene, homotopy)` at the top of the loop.
- Call `logger.step(...)` each FM step with obs / horizon / timings / state / contact.
- `logger.save(...)` after the loop; attach summary fields into the existing return dict so
  they also land in `results.json` (alongside `fm_ms_mean`/`fm_ms_p95` already there).

#### Activation — ALWAYS ON, no `.sh` change required
- The text log is **unconditional** — it is NOT gated behind `--record`. Per IDEAS.md it is
  "the standard output of every evaluation run", and it is near-zero cost (wraps timings the
  loop already takes). `--record gif/all` stays opt-in **only** for the expensive overhead
  GIF; the `.log` is independent of it.
- Because the logger lives inside `rollout_one`, the existing submit script
  `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` needs **no edit** — it already calls
  `python eval_fm_uav.py ...`, and every rollout will now also write its `.log`.
- **Output location:** `logs/UAV_FM/<scene>/plans/<...>/<seed>/<variant>/<episode>.log`,
  i.e. right next to the existing `results.json` / `.npz` / GIF for that variant.
- **SLURM stdout** (e.g. `temp/Gen11E7/output1`): keep the existing one-line per-variant
  print; additionally echo the compact `# SUMMARY TIMING` block to stdout so the
  over-budget verdict is visible in the job log without opening the `.log` file. Per-step
  lines go to the `.log` file only (too verbose for stdout).

### Phase 4 — baseline comparison + grep one-liners
- After all variants run, emit the `vs diffuser baseline` overhead line per variant
  (we already loop every variant in `eval_scene` — diff their summaries).
- Document the IDEAS.md grep one-liner adapted to our paths:
  `grep "total_ms mean" logs/UAV_FM/**/plans/**/*.log`.

---

## Scope / non-goals this epoch

- **No spatial constraint margins** (no obstacle/halfspace geometry yet — U3 left them as
  wired no-ops). The `PCC` block logs `constraint=dynamics status=ON` only; spatial
  `nearest/margin` fields are scaffolded but blank until geometry is designed.
- **Hardware caveat (from IDEAS.md §"Can we load…")**: timing is cluster-GPU-bound. Every
  `.log` must be stamped with the node it ran on (the SLURM header already prints `NODE`);
  cluster latency ≠ real-drone latency, label accordingly.
- **No retroactive logs**: timing can only be measured live → this attaches to the next
  eval pass, it does not reprocess old `.npz`/GIF outputs.

---

## Acceptance check

1. Run `eval_fm_uav.py --scene pillars --seed 6 --n-trials 2` → each `plans/<variant>/`
   contains an `<episode>.log` with per-step + `# SUMMARY` blocks.
2. `diffuser` log shows `proj_ms=0.0`, `total_ms ≈ fm_ms`; `dpcc-*` logs show `proj_ms > 0`.
3. `# SUMMARY TIMING` reports `over_budget` count against the 30.3 ms budget.
4. `grep "total_ms mean"` across logs sorts variants by per-step cost.

---

## What was actually implemented — porting reference for other models

This section documents the **exact pattern we shipped** so it can be copied faithfully to
FMv3ODE, Visual Aligning, DPCC baseline, and any future model. Each step below names the
file, the function, and the minimal change required.

### Step 1 — time the projector INSIDE the model sampling loop

**File:** `flow_matcher_v3_uav/models/diffusion.py` → `p_sample_loop`

Add a `proj_ms = 0.0` accumulator before the ODE loop. Wrap every `projector.project()` and
`projector.compute_gradient()` call with `time.perf_counter()` and accumulate the delta.
Return it in `infos`:

```python
costs = {}
proj_ms = 0.0                     # ← add this

# inside the loop, around the project() call:
_t_proj = time.perf_counter()
x, projection_costs = projector.project(x, constraints)
proj_ms += (time.perf_counter() - _t_proj) * 1e3

# at the end, before return:
infos['projection_ms'] = proj_ms  # ← add this
```

**Why inside the model:** our PCC projects during ODE integration, not as a post-FM filter.
The outer `policy(...)` timer therefore bundles FM + projection. The only way to split them
is to time from inside the model and expose the split via `infos`.

**Goal-dim shape fix (UAV-specific):** when `self.goal_dim > 0`, apply gradient only to the
non-goal slice — `x[:,:,:-goal_dim] += grad` not `x += grad`. The full `x` is 12-D but
`grad` is computed on the 11-D slice → shape crash without this.

---

### Step 2 — expose per-call diagnostics on the Policy object

**File:** `flow_matcher_v3_uav/sampling/policies.py` → `Policy.__call__`

No signature change. After each call, store read-by-eval attributes:

```python
self.last_proj_ms = float(infos.get('projection_ms', 0.0))
self.last_proj_cost = float(total_projection_cost_of_selected_candidate)
self.last_which_trajectory = int(which_trajectory)
self.last_infos = infos
```

The eval rollout reads these after `action, traj = policy(...)` — no return-value change.

---

### Step 3 — split bundled timing in the eval rollout

**File:** `<model>_test/eval_<model>.py` → rollout loop

Replace:
```python
t0 = time.perf_counter()
action, traj = policy({0: obs}, ...)
fm_ms.append((time.perf_counter() - t0) * 1e3)
```
With:
```python
t0 = time.perf_counter()
action, traj = policy({0: obs}, ...)
step_total_ms = (time.perf_counter() - t0) * 1e3
step_proj_ms  = float(getattr(policy, 'last_proj_ms', 0.0))
step_fm_ms    = max(step_total_ms - step_proj_ms, 0.0)   # PURE inference
fm_ms.append(step_fm_ms)
proj_ms.append(step_proj_ms)
total_ms.append(step_total_ms)
```

For models without a projector (`diffuser` / plain FMv3ODE): `last_proj_ms` is always 0 →
`step_fm_ms == step_total_ms`. No code branching needed.

---

### Step 4 — instantiate BehaviorLogger at top of rollout

**File:** `<model>_test/eval_<model>.py` → rollout function signature + body

Add params: `variant='diffuser', log_dir=None, control_hz=33, text_log=True`

```python
from FM_v3_uav_test.behavior_logger import BehaviorLogger

episode_id = f'{scene}_{homotopy}_{trial_seed}'
blog = BehaviorLogger(episode_id, variant, scene, homotopy,
                      control_hz=control_hz, batch_size=batch_size, horizon=horizon,
                      text_log=text_log)
proj_on = (variant != 'diffuser')
```

`BehaviorLogger` lives in `FM_v3_uav_test/behavior_logger.py` and is system-agnostic —
import it from there for all models. No copy needed.

---

### Step 5 — call `blog.step()` once per control step

At the END of each FM control step (after physics), call:

```python
blog.step(
    t=k / control_hz, step_idx=f'{k}/{n_fm}', obs=obs,
    fm_horizon=<H-step action array of executed candidate>,
    fm_ms=step_fm_ms, proj_ms=step_proj_ms,
    proj_cost=float(getattr(policy, 'last_proj_cost', 0.0)),
    proj_active=proj_on,
    state_p=<pos after physics>, state_v=<vel after physics>,
    contact=<contact descriptor or None>, track_err=<float>,
)
```

**`text_log=False` path:** `step()` records raw stats and returns immediately — no string
formatting, no memory growth. Timing numbers are still accurate and land in `results.json`.

---

### Step 6 — save log + add timing to return dict

After the rollout loop:

```python
blog_summary = blog.summary_dict()
if log_dir is not None:
    blog.save(os.path.join(log_dir, f'rollout_{episode_id}.log'), behaviour=behaviour_dict)
```

Add to the rollout return dict:
```python
'proj_ms_mean':        blog_summary['proj_ms_mean'],
'total_ms_mean':       blog_summary['total_ms_mean'],
'total_ms_p95':        blog_summary['total_ms_p95'],
'total_over_budget':   blog_summary['total_over_budget'],
'budget_ms':           blog_summary['budget_ms'],
# fm_ms_mean now means PURE inference (projection subtracted)
```

---

### Step 7 — wire from `_run_variant` and echo to SLURM stdout

Pass `variant`, `log_dir`, `control_hz`, `text_log` from `_run_variant` to the rollout.
After all rollouts, echo the timing verdict:

```python
print(f'[ eval ] {scene} variant={variant} TIMING: fm_ms={fm_ms_mean:.1f} '
      f'proj_ms={proj_ms_mean:.1f} total_ms={total_ms_mean:.1f} '
      f'(p95={total_ms_p95:.1f}) budget={budget_ms}ms → real_time_{verdict}')
```

---

### Step 8 — add config keys to the model's yaml

```yaml
control_hz: 33        # Hz; budget_ms = 1000/control_hz computed by BehaviorLogger
behavior_log: true    # false = timing stats only (no string fmt in loop, no .log file)
```

Read in `_run_variant`:
```python
control_hz = config.get('control_hz', 33)
text_log   = config.get('behavior_log', True)
```

---

### Model-specific adaptation notes

| Model | Sampling | Projector location | Key difference from UAV |
|---|---|---|---|
| **UAV FM-PCC** (this) | ODE | inside `p_sample_loop` | proj bundled with FM; split via `infos['projection_ms']` |
| **FMv3ODE** (visual aligning) | ODE | same pattern | same Steps 1–8; `control_hz` from its yaml |
| **DPCC baseline** | DDPM/ODE | same projection.py | `fm_ms` is near-zero or absent; `proj_ms` is the whole cost |
| **Gen3v4 iMF** (state FM) | — | no projector | `proj_ms=0` always; Steps 1–2 not needed; Steps 3–8 still apply |

**No changes to the `.sh` submit scripts** — the logger is fully inside the Python eval
loop and is activated by the yaml config, not the shell.

---

## References

- Idea source: [`../../../REALTIME_RECORDING/IDEAS.md`](../../../REALTIME_RECORDING/IDEAS.md)
- Target loop: `FM_v3_uav_test/eval_fm_uav.py:297–384` (`rollout_one`)
- Projection call site: `flow_matcher_v3_uav/models/diffusion.py:263–269`
- Variant suite this builds on: [`../U3/CHANGELOG.md`](../U3/CHANGELOG.md)
- Why dynamics-only this epoch: [`../U1&2/WHY_PCC_WORKS_PILLARS.md`](../U1&2/WHY_PCC_WORKS_PILLARS.md)
