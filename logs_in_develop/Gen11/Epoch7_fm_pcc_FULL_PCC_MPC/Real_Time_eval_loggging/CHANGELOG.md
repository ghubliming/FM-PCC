# CHANGELOG — Real-Time Behaviour Text Logging (implemented)

**Date:** 2026-06-25
**Plan:** [`PLAN.md`](PLAN.md) · **Idea:** [`../../../REALTIME_RECORDING/IDEAS.md`](../../../REALTIME_RECORDING/IDEAS.md)
**Builds on:** [`../U3/CHANGELOG.md`](../U3/CHANGELOG.md) (full variant suite)

Implements the digital-twin **text** logger (NOT a GIF) for the Gen11 E7 UAV FM-PCC eval.
Headline product: per-step + per-episode **timing** — can FM-PCC close the 33 Hz / 30 ms
loop, and how much budget is FM inference vs the PCC projector?

---

## Files changed

### NEW `FM_v3_uav_test/behavior_logger.py`
System-agnostic `BehaviorLogger` (per IDEAS.md §"Implementation approach"):
- `.step(t, obs, fm_horizon, fm_ms, proj_ms, proj_cost, proj_active, state_p, state_v, contact, track_err, ...)`
  → one structured block per FM control step.
- `.summary_dict()` / `.summary_block()` → `# SUMMARY` with `fm_ms / proj_ms / total_ms`
  mean·max·p95, `over_budget` count vs `budget_ms = 1000/33 = 30.3`, `real_time_safe` verdict,
  contacts, and a `CONTACTS:` appendix.
- `.save(path, behaviour=...)` → writes the `.log`.
- `diffuser` (no projector) prints `PCC status=OFF`, `proj_ms=0.0` — same grammar, empty fields.
- Stamps `node` (SLURM `$SLURMD_NODENAME`/`$HOSTNAME`) — timing is cluster-latency, explicitly
  labelled as NOT the target-drone budget (IDEAS.md §"hardware-bound").

### `flow_matcher_v3_uav/models/diffusion.py` — `p_sample_loop`
Split projection time out of FM inference. The projector runs **inside** the FM ODE loop,
so the eval's outer timer bundled FM+projection. Now:
- `proj_ms` accumulator wraps both `projector.project()` (SLSQP) and `projector.compute_gradient()`
  (gradient variant) calls with `time.perf_counter()`.
- Exposed via `infos['projection_ms']`. (Accurate because the projector is CPU/scipy SLSQP —
  synchronous wall-time, not async GPU dispatch.)

### `flow_matcher_v3_uav/sampling/policies.py` — `Policy.__call__`
No signature change. After each call, stores read-by-eval attributes:
- `last_proj_ms` (from `infos['projection_ms']`)
- `last_proj_cost` (summed projection cost of the **selected** candidate)
- `last_which_trajectory`, `last_infos`

### `FM_v3_uav_test/eval_fm_uav.py` — `rollout_one` + `_run_variant` + `eval_scene`
- `rollout_one(..., variant=, log_dir=)`: instantiates the logger, **always on** (independent
  of `--record`). Per FM step splits `total_ms = fm_ms + proj_ms`
  (`fm_ms = bundled − policy.last_proj_ms`), logs obs / FM Δp_des horizon of the executed
  candidate / timings / proj_cost / state / per-step contact / track_err.
- Saves `rollout_<scene>_<homotopy>_<seed>.log` into the variant's `plans/.../<variant>/` dir
  (next to `results.json`).
- Return dict + per-variant `summary` gain `proj_ms_mean`, `total_ms_mean`, `total_ms_p95`,
  `total_over_budget`, `budget_ms`. **`fm_ms_mean`/`fm_ms_p95` now mean PURE inference**
  (projection subtracted) — meaning changed; documented here.
- `eval_scene` echoes a compact `TIMING:` line per variant to **stdout** (the SLURM job log):
  `fm_ms / proj_ms / total_ms (p95) / budget → real_time_SAFE|OVER×N`.

---

## No `.sh` change required
`Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` is untouched — it already runs
`python eval_fm_uav.py ...`, and the logger lives inside `rollout_one`, so every rollout
auto-writes its `.log`. `--record` still gates only the expensive GIF.

---

## Hotfixes applied after first cluster run (2026-06-26)

### Bug 1 — `gradient` variant shape crash (`tensor a (12) must match tensor b (11)`)
Pre-existing latent bug in `p_mean_variance` and the torchdiffeq branch of `p_sample_loop`:
`grad` was computed on `x[:,:,:-goal_dim]` (11 dims) but added to the full `x` (12 dims).
Never triggered before because `gradient` wasn't in `projection_variants` until U3.

Fix: apply gradient only to the non-goal slice in both locations:
```python
# before (both p_mean_variance and p_sample_loop torchdiffeq branch):
x = x + grad                           # 12 + 11 → crash

# after:
x[:, :, :-self.goal_dim] = x[:, :, :-self.goal_dim] + grad   # 11 + 11 → OK
```

### Bug 2 — `behavior_log` flag added; `budget_ms` / `control_hz` moved to config
`control_hz=DATASET_HZ` was hardcoded in `rollout_one`. User-reported: should come from
`uav_eval.yaml` so it can be changed without touching code.

**`behavior_log` flag** — per-step string formatting inside the control loop can add noise to
timing measurements. New `behavior_log: true` key in `uav_eval.yaml`:

| `behavior_log` | per-step `.log` file | string formatting in loop | timing stats in `results.json` |
|---|---|---|---|
| `true` (default) | ✅ written | ✅ runs | ✅ always |
| `false` | ✗ skipped | ✗ skipped (early return in `step()`) | ✅ always |

Raw timing stats (`fm_ms`, `proj_ms`, `total_ms`, `total_over_budget`) are **always**
collected and land in `results.json` regardless of the flag — only the string formatting
and file I/O are gated. Set `behavior_log: false` when timing accuracy is critical.

**`control_hz` + `budget_ms` from config** — was hardcoded as `DATASET_HZ=33`:
- Added `control_hz: 33` to `config/uav_eval.yaml`
- `rollout_one(..., control_hz=DATASET_HZ, text_log=True)` — new params with safe defaults
- `_run_variant` passes `config.get('control_hz', DATASET_HZ)` and `config.get('behavior_log', True)`

---

## Verification (Docker, no cluster/GPU)
- `python -m py_compile` on all 4 changed files → OK.
- Isolated logger smoke test (numpy only): per-step blocks + `# SUMMARY` render correctly;
  an injected 28 ms FM spike flips `real_time_safe=NO` with `over_budget=1/5`; `diffuser`
  path yields `proj_ms=0.0`, `over_budget=0`. Contact appendix lists the contact step.
- End-to-end timing (`fm_ms`/`proj_ms` magnitudes) is GPU/cluster-bound → validated on next
  Slurm eval, not in Docker (no Python runtime for the model here).

Example summary (synthetic smoke test):
```
#          TIMING:
#            fm_ms    mean=10.7  max=33.1  p95=27.5  over_budget=1/5
#            proj_ms  mean=1.3   max=1.3   p95=1.3
#            total_ms mean=12.0  max=34.4  p95=28.8  over_budget=1/5 (20.0%)
#            real_time_safe=NO  (measured on <node> — cluster latency, NOT target drone)
```

---

## Scope / deferred (faithful to this dynamics-only epoch)
- **`act_raw → act_proj` override delta** (IDEAS.md): NOT implemented. Our PCC reshapes the
  sampled trajectory *inside* the ODE loop rather than filtering the executed action, so a
  clean pre/post first-action delta needs a second projection-disabled forward pass (doubles
  inference, corrupts timing). Logged proxy: `proj_cost` of the selected candidate. Deferred.
- **Spatial constraint margins** (`nearest_constraint`, `margin`, ACTIVE/IDLE): blank — only
  the dynamics constraint exists this epoch (no scene geometry; see U3). The `PCC` block logs
  `constraint=dynamics status=ON`; spatial fields are scaffolded for when geometry lands.

---

## After running on the cluster
Sort variants by per-step real-time cost across all logs:
```bash
grep "total_ms mean" logs/UAV_FM/**/plans/**/*.log
```
Or read the per-variant `TIMING:` lines straight from the SLURM job stdout.
