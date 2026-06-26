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

## References

- Idea source: [`../../../REALTIME_RECORDING/IDEAS.md`](../../../REALTIME_RECORDING/IDEAS.md)
- Target loop: `FM_v3_uav_test/eval_fm_uav.py:297–384` (`rollout_one`)
- Projection call site: `flow_matcher_v3_uav/models/diffusion.py:263–269`
- Variant suite this builds on: [`../U3/CHANGELOG.md`](../U3/CHANGELOG.md)
- Why dynamics-only this epoch: [`../U1&2/WHY_PCC_WORKS_PILLARS.md`](../U1&2/WHY_PCC_WORKS_PILLARS.md)
