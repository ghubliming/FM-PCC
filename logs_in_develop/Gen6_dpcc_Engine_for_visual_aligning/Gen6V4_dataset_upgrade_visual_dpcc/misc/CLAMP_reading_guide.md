# The Action Clamp — What It Is and How to Read the Outputs

**Fix 4 — added 2026-05-20**

---

## What is the clamp?

The clamp is a per-step **safety gate** on the output of the diffusion/flow-matching model.
After the model predicts the next action `a` (a 3D position delta `[dx, dy, dz]` in metres),
the clamp checks:

```
raw_mag = |a|   (Euclidean norm across all 3 axes)

if raw_mag > max_action_delta:
    a = a * (max_action_delta / raw_mag)   # rescale, keep direction
```

The key property: **direction is preserved**, only magnitude is reduced.
This is different from per-axis `np.clip`, which would distort the direction vector and
cause the robot to approach from a wrong angle.

### Why is it needed?

The ODE solver inside Flow Matching (Gen7) or the DDPM reverse chain (Gen6V4) can
occasionally produce an exploding step — a single `a` with `|a| >> 1 m`, usually when
the denoising trajectory drifts into an unstable region. Without the clamp:

1. MuJoCo receives a 1-metre position command in one step → simulation goes to NaN.
2. `LimitsNormalizer` hard-clamps all dimensions to `[min, max]` simultaneously.
3. All 3 axes become identical values. Every subsequent action is a copy of this
   saturated vector. The episode "runs" 400 steps but the robot is frozen.

With the clamp, the exploding step is reduced to `max_action_delta` in that direction,
the simulation stays valid, and diagnostics capture the true episode behavior.

### Current setting

```yaml
# config/visual_aligning_eval.yaml
max_action_delta: 0.01   # metres per step; set to null to disable
```

`0.01 m = 10 mm/step`. The healthy Gen6V4 step size is ≈ 0.22 mm — so the threshold
is **45× the normal step size**. It will not touch a healthy run.

---

## How to read the debug outputs

### Terminal output — `[ CLAMP ... ]` lines

```
[ CLAMP step=47 ] raw|a|=2.3142 m → clamped to 0.01 m
[ CLAMP step=48 ] raw|a|=1.8800 m → clamped to 0.01 m
```

| What you see | What it means |
|---|---|
| First 5 clamp events, then every 50 steps | The policy is exploding. `raw|a|` shows the magnitude of the raw ODE output. |
| No `[ CLAMP ]` lines at all | Policy is healthy — all steps are well within 0.01 m. |
| Occasional single clamp event mid-episode | Transient instability; check if the episode recovers. |
| Clamp events every step from step N onward | ODE divergence at step N; check `[ DIAG replan=N ]` just before that step. |

### PNG — panel `[2,2]` — Clamp Events Scatter

- **X axis**: step number (0–400)
- **Y axis**: raw action magnitude (metres) that triggered the clamp
- **Each dot**: one clamped step

**Healthy run**: panel is empty (title shows "Clamp Events: 0").

**Exploding ODE**: dense cluster of dots early in the episode, all at `raw|a| >> 0.01 m`.

**Partial divergence**: sparse dots scattered after step N, with `raw|a|` growing over time.

### PNG — panel `[2,0]` — Action Magnitude per Step

This panel shows the **raw magnitude before clamping** is applied (logged first for
diagnostic purposes). Use it alongside the Clamp Events panel:

| Pattern | Interpretation |
|---|---|
| Flat ~0.0002 m throughout | Healthy, very small steps (may be "too slow" hypothesis) |
| Flat ~0.0002 m, then sudden spike to >> 0.01 m | ODE diverges at that step |
| Gradually increasing from 0.0002 → 0.01 m | Policy is accelerating toward target — expected on approach |
| All steps exactly at 0.01 m | Clamp active every step — model output is consistently exploded |

### PNG — panel `[1,0]` — Distance to Target over Steps

Cross-reference with Act Magnitude and Clamp Events to distinguish hypotheses:

| Distance curve | Action magnitude | Clamp events | Conclusion |
|---|---|---|---|
| Steady decrease → 0 | Small, healthy | None | **Success** — robot reached box |
| Flat, no change | Small, healthy | None | **Too slow** — 0.22 mm/step can't close 91 mm in 400 steps |
| Flat, no change | Spike then flat | Yes, from step N | **ODE exploded** at step N; clamped correctly; but policy has no healthy replanning |
| Flat, no change | All at max delta | Dense, all steps | **Model weights wrong** or completely diverged — check checkpoint |
| Decrease then plateau | Normal | None | **Near-miss** — policy ran out of steps before reaching threshold |

---

## The two hypotheses for Gen6V4 failure

The first diagnostic run showed:
- Healthy first-replan: normalized `|a0|` = 0.24, physical step ≈ 0.22 mm
- Mode 0 at episode end (robot was near the box at the final step)
- `Success: False`, final mean distance = 0.091 m

The two candidate explanations:

**Hypothesis 1 — Too slow**: Policy is correct but generates steps too small.
0.22 mm/step × 400 steps = 88 mm maximum travel ≈ 0.088 m. The box was 0.091 m away.
The robot almost made it but ran out of budget.

**Hypothesis 2 — ODE explosion mid-episode**: Policy starts healthy, then diverges
partway through. LimitsNormalizer saturates; remaining steps are wasted.

### How Fix 4 distinguishes them

| Panel | Hypothesis 1 signature | Hypothesis 2 signature |
|---|---|---|
| Clamp Events `[2,2]` | Empty | Dots from step N onward |
| Act Magnitude `[2,0]` | Flat ~0.0002 m | Spike at step N |
| Distance to Target `[1,0]` | Steady decrease, stops at ~0.09 m | Decrease, then flat from step N |

---

## Quick reference

```
[ CLAMP step=N ] raw|a|=X m → clamped to 0.01 m
```
→ Step N had a raw action X times larger than the threshold.
   Direction was preserved; magnitude was reduced to 0.01 m.
   If X >> 1.0, ODE is diverging. If X is just slightly over 0.01, policy is
   accelerating normally and may not need the clamp at all.
