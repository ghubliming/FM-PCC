# Discussion — UAV `diffuser`: MPC foresight decoupled from the executed waypoint (Fix15, pillars)

**Gen:** Gen11 / Epoch9_PCC_Constraints (Fix_15 context) · **Scene:** `pillars_bounds+dynamics+geo_bounds+obstacles` · **Variant:** `diffuser`
**Date:** 2026-07-13 · **Status:** discussion / root-cause hypotheses — **no code changed yet**
**Artifact inspected:** `temp/pillars_bounds+dynamics+geo_bounds+obstacles/_traj_viz/scene.json`

> **Reference convention:** pointers are `file · function/logic`, not line numbers (lines rot). This is a **discussion**, not a fix — it ranks candidate causes and proposes diagnostics.

---

## 1. The observation (user)

In the `diffuser` variant, the receding-horizon **foresight** (the plotted H-step plan fan) points a **different direction than the executed backbone / next waypoint** — "waypoint heading north, the backbone's next waypoint heading east." In a correct receding-horizon MPC the executed path should be tangent to each fan at its `h=0` anchor. They are decoupled.

---

## 2. Critical framing — `scene.json` is a *derived* artifact, not eval output

`scene.json` is **not** written by the UAV eval. It is reconstructed by the visualizer/export tool
`npz_analysis/npz_traj_visualizer/npz_traj_export.py` from the npz. So a "decoupling" seen in `scene.json` has **two possible layers**:

- **(L1) the data** — what the eval actually recorded into the npz (`obs_all` = executed; `sampled_trajectories_all` = plan fans), and whether the model's plan and its execution genuinely diverge; and
- **(L2) the reconstruction** — how the export tool aligns/columns those two arrays when drawing the fans.

A wrong picture can come from **either**. This must be disambiguated before blaming the policy.

---

## 3. Candidate root causes (ranked)

### A — [STRUCTURAL, most fundamental] Foresight and execution are *two different model outputs*, coupled only by projection

The UAV control loop (`FM_v3_uav_test/eval_fm_uav.py`, main rollout in `run_episode`/`build_experiment` region):

- **Execution** is driven by the **action head**: `action = policy(...) → first Δp_des`, then `p_des = p_des + action`; the PID/MJPC `tracker.compute(...)` makes the drone's **physical** position `p` chase `p_des`. So the executed backbone = **actual `p`** produced by *action-head → PID → physics*.
- **Foresight** is the **observation head**: `plan = traj.observations` (the H-step obs-space rollout), stored into `plans` → `sampled_trajectories_all`. The fan plotted is the model's **predicted `p`** channel.

These are **separate outputs of the generative model**. The model's predicted-`p` trajectory (the fan) is *not* what physics does with its predicted-Δp_des (the executed path). In the `dpcc*` variants, `iMeanFlowODE.p_sample_loop`'s projection block snaps the plan into the constraint-feasible set each step, which **re-couples** the plan to a physically consistent path. **`diffuser` sets `projector = None`** (eval main loop) → nothing reconciles obs-head vs action-head → the fan can head one way while physics goes another. **This is decoupling by construction, not a bug per se** — it is what "no projection" means for a dual-head model. Same family as the Gen3v4 iMF finding (projection was silently masking an unreconciled generative output).

*Prediction if this dominates:* the divergence grows where the plan is least physically realizable (tight turns around pillars, saturated Δp_des) and is **absent/small in `dpcc*` variants** on the same npz.

### B — [VIZ ARTIFACT, likely co-cause] Recording-phase offset mis-detection in the export tool

`npz_traj_export.py` explicitly knows the executed buffer and the plan's conditioning are recorded with **different phase**, and *auto-detects* the offset rather than hard-coding it:
- `_recording_offset(...)` — "the single structural step-offset between the plan's conditioning and the executed buffer … eval loops record the executed buffer with DIFFERENT phase … never stores the initial reset obs → executed is shifted +1 control step."
- `_nearest_executed(...)` / `snapshot_analytics(...)` — pin each fan's `h=0` onto the nearest executed sample within `±2` of a guessed step; `save_every = round((T-1)/(n-1))`.

If the detected offset is wrong (or `save_every` rounding is off for this scene's step counts — pillars `diffuser` shows `n_steps≈634`, `plan_steps≈212`, ratio 3, `plan_every=3`), every fan is pinned to the **wrong executed step**. On a curving path (north→east around a pillar) a 1–3 step mispin rotates the fan's apparent heading relative to the local backbone — exactly the reported symptom. **This code is under active change** — recent commits `npz viz fix2: detect and correct recording-phase misalignment in trajectory exports` — so it is a live suspect, not settled.

*Prediction if this dominates:* the fan shape is *correct* but *shifted*; overlaying the raw `sampled_trajectories_all[t]` fan at its true recording index (no offset correction) puts it back on the backbone.

### C — [LOW, but has a known trap] Column / channel mapping

Checked and mostly consistent: `env_col_map('uav')` → `col_map = {x:3, y:4, z:5}` and panels `[('xy',0,1),('xz',0,2)]`; **both** executed (`obs_all[:, 3:6]`) and plan (`mean_plan(snap, dims)` with the same col_map dims) use cols 3,4,5 = **actual `p`**. So x/y/z are not transposed between the two.
**The trap** (tool's own header): UAV obs = `[p_des(0:3) | p(3:6) | v(6:9)]`; **col 2 is `p_des_z`, which can run away** — the header notes "the `p_des_z` runaway in col 2 is caught by the executed-explosion curve, not the scene." If any panel or a stale export path ever slices `0:3` (p_des) instead of `3:6` (p) for one of the two series, the two series become different channels and diverge. Worth a 1-check but not the leading hypothesis.

---

## 4. Why it shows up here specifically

- **`diffuser`** = the only variant with `projector = None` → cause **A** is unmasked (dpcc/gradient/model_free all re-couple via projection).
- **pillars** = a scene with real turns → both **A** (hard-to-realize plans near obstacles) and **B** (mispin rotates heading on curves) become visually large; on a straight scene the decoupling would be nearly invisible.
- The scene.json metrics corroborate a *physically* rough diffuser rollout: `goal_reached=0`, `goal_dist≈1.21`, `constraint_n_violations=508`, `phys_min_z≈-0.01` — i.e. the executed path itself is poor, consistent with A.

---

## 5. Diagnostics to disambiguate (do these before any fix)

1. **A vs B — projection A/B on the SAME npz.** Render `dpcc` (or `gradient`) from the same run's npz through the *same* export tool. If those fans sit on their backbones but `diffuser` doesn't → **A** (model/projection). If *all* variants show the same offset-shaped decoupling → **B** (viz).
2. **B — bypass the offset correction.** Plot `sampled_trajectories_all[t]` fans directly at their nominal recording index (`step = i * plan_every`), no `_recording_offset` shift, over `obs_all[t][:, 3:6]`. If they snap back onto the backbone → **B** confirmed (the auto-offset is mis-firing for this scene).
3. **A — head-consistency check in the eval.** For one replan, compare the plan's first-segment heading `unit(traj.observations[·,1,3:6] − traj.observations[·,0,3:6])` against the executed heading `unit(Δp_des)=unit(action)`. If these already disagree at `h=0→h=1` *inside the eval* (before any viz), the decoupling is real (A), not the tool.
4. **C — channel sanity.** Assert both series are read from cols 3,4,5; confirm no path reads `0:3`. Confirm `obs_all` last-dim (6 vs 9) matches `env_col_map`'s assumption via `infer_env`.
5. **Cross-pipeline reference (uses the Issue-2 patch).** DPCC `scripts/eval.py` now writes `sampled_trajectories_all` too (see `Gen0_FMPCC_DPCC_Code_Updates`); render a *state-only* FMv3ODE/DPCC avoiding npz (2D, `col_map={x:2,y:3}`, well-understood coupling) through the tool. If 2D fans couple correctly but UAV 3D don't → the fault is **UAV-specific** (env col layout or the 3D offset path), not the shared tool.

---

## 6. What is *not* the cause (ruled out this pass)

- **Not** an x/y transpose — executed and plan use the same col_map (cause C mostly cleared).
- **Not** the Issue-2 npz gap — that was DPCC-only and is patched; UAV already writes `sampled_trajectories_all`.
- **Not** (necessarily) a projection/circuit-breaker bug from Fix_15 — `diffuser` runs with `projector = None`, so Fix_15's SLSQP circuit breaker is not even in the loop for this variant. (It *would* be relevant for the `dpcc*` variants' foresight, a separate question.)

---

## 7. The decision this discussion needs

- If **A** dominates: this is **expected** for `diffuser` (no projection ⇒ obs-head and action-head uncoupled). The "fix" is framing, not code — either (i) document that `diffuser` foresight is the raw obs-head prediction and is *not* expected to match execution, or (ii) plot the **action-head foresight** (`traj.actions` integrated) instead of the obs-head plan for the `diffuser` panel, so the fan reflects what is actually executed.
- If **B** dominates: fix `npz_traj_export.py`'s `_recording_offset` / `save_every` for the UAV step-count regime (continue the `npz viz fix2` line).
- Most likely it is **A + B stacked** — confirm with diagnostic #1 and #2, then decide per-layer.

---

## 8. VERIFICATION (2026-07-13) — the viz/npz-analyzer is EXONERATED; the decoupling is real data

Ran a **pure-data check on `scene.json` itself** (it already stores `executed`, `plans` = `[4 cand, 8 H, 3 xyz]`, and `plan_steps`), comparing the unprojected `diffuser` against a projected variant (`geo_free`) rendered by the **same** export tool. No cluster needed — this is L1-vs-L2 disambiguation from the artifact.

**Test — does each fan's `h=0` (mean over candidates) sit on the executed path at its anchor, and can any constant step-offset put it there?**

| metric (trial 0) | `diffuser` (projector=None) | `geo_free` (projected) |
|---|---|---|
| `‖fan_h0 − executed[plan_step]‖`, median | **0.534** | **0.066** |
| best residual over **any** offset k∈[−25,25], median | **0.502** (shift doesn't help) | 0.065 |
| min dist from `fan_h0` to **ANY** executed point, median | **0.503** | 0.073 |
| fans on-path (`<1e-2`) at best-k | 0.19 | 0.36 |
| best-k distribution | scattered (0, −25 boundary, 1, 2, …) → **no phase** | sharply k=0 (127/212) |

**Reading:**
1. **The tool is correct.** With the *same* export code, `geo_free`'s fans sit on the backbone (median residual 0.066 ≈ one executed step of ~0.05). A column/offset bug in `npz_traj_export.py` would misalign *all* variants equally — it does not. → **cause B (viz recording-offset) and cause C (columns) are ruled out** for this artifact.
2. **`diffuser`'s decoupling is genuine L1 data.** Its fan `h=0` is ~**0.5 off the executed path**, and **no step-offset fixes it** (best-offset residual 0.50 ≈ offset-0 residual 0.53; best-k is scattered, not a constant phase). The fan is ~0.5 from *any* executed point → **off-path, not mis-phased**.
3. **Heading (XY, plan `h0→h1` vs executed over one replan):** median **24.6°**, p90 **72.7°**, **26% > 45°**, max 173° — real directional divergence, not lag jitter (consistent with the reported "north vs east").
4. **Timescale/trackability:** plan step `h0→h1` median **0.238** vs executed motion per replan median **0.054** → the plan sprints ~**4.4×** faster than the drone actually moves.

**Localization verdict:** **NOT the viz, NOT the npz analyzer, NOT the columns.** It is a **real, data-level decoupling specific to the unprojected `diffuser`**, and it is **cause A**: the model's obs-head plan (predicted `p`) is not reconciled with what the action-head + PID + physics actually execute. The projected `geo_free` proves the point — **projection is what pins the plan's `h=0` back onto the true state** (residual 0.066 vs 0.53); with `projector = None` nothing does.

**Beyond "just tracking error":** a pure tracking lag would show a *small, roughly tangential* offset that a small phase-shift would absorb — instead the offset is large (~0.5), off-path at every shift, with wrong heading and a 4.4× speed mismatch. So it is **not merely tracking error** — the diffuser plan is fundamentally disconnected from its own execution.

**One open sub-question (A refinement, worth a code check):** since `geo_free` (projection) pins `h0` but `diffuser` does not, the `h0=current-state` pinning appears to come from **projection**, not from `apply_conditioning`. Check whether the UAV plan's **`p` channel (cols 3:6)** is actually inpainted to the current obs at `t=0` in `p_sample_loop`/`apply_conditioning` — if only `p_des` (cols 0:3) is pinned and `p` is left free, that is a genuine **code** gap (h0 predicted-`p` is unconstrained), independent of projection.

*Awaiting instruction before implementing any of the above.*
