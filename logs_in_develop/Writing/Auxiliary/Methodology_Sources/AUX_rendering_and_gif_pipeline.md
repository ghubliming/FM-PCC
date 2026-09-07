# AUX — offscreen rendering, state injection, and the GIF/overlay artefacts

**Thesis home:** `sec:setup:tasks` (one paragraph), `app:repro` (the detail) · **TARGET** §5.1, §5.2
**Primary dev logs:** `Gen11/Epoch5_visual_and_validation/` (`init_0/METHODOLOGY.md`, `U3/`,
`U4_Stress_Test_Rendering/`, `U5_Overview_Plots/PLOT_EXPLAINER.md`) ·
`Gen11/Epoch9_PCC_Constraints/Fix_7_Gif_lower_size/`, `Fix_9_Gif_lower_size_again/`,
`Fix_14/CHANGELOG_fix14_foresight_constraint_overlay_and_allpng.md` ·
`Gen11/Epoch10_Visual_UAV/PLAN_E10_uav_visual_mode.md` · `Clean_Gifs/`

---

## 1. What the thesis must say

Rendering is infrastructure, not a contribution — but **one design decision in it is a genuine
methodological point** and deserves a short paragraph rather than an appendix line.

### 🔑 The two-stage split: physics once, pixels many times

Epoch 4 ran physics and recorded `(p, v, p_des)` to pickles. Camera rendering was **deliberately
excluded** from that stage. Offscreen EGL rendering needs a GPU and is ~10× slower per step than
headless physics; coupling it to the physics loop would mean re-flying all 1769 trajectories every
time the resolution, camera, or overlay changed.

So Stage 2 **replays the stored states**, at zero physics cost:

```python
data.qpos[:3]  = obs[t, :3]              # 3-D position, world frame
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]    # quaternion — forced identity (level)
data.qvel[:3]  = obs[t, 3:6]             # linear velocity, world frame
data.qvel[3:6] = 0.0                     # angular velocity — forced zero
mujoco.mj_forward(model, data)           # kinematics only; no physics step
```

`mj_forward` propagates `qpos`/`qvel` through the kinematic chain without advancing the ODE solver,
contacts, or rotor forces — so the replay is exactly the recorded trajectory, not a re-simulation
that could diverge.

**⚠️ The caveat that must be stated with it.** Attitude is *not* replayed: the quaternion is forced
to identity and angular velocity to zero, because the stored state vector is `[p(3), v(3)]` and
carries no attitude. **Every rendered image and GIF therefore shows a level drone regardless of its
true attitude.** For visual *inspection* of paths this is harmless; for anything that claims to
depend on orientation it is not. Say this once, plainly — a reader who spots a level drone in a
banking turn will otherwise assume the physics is wrong.

---

## 2. What the artefacts are, and what they are for

| artefact | purpose | where |
|---|---|---|
| camera images | training input for the visual UAV mode (Epoch 10) | Epoch 5 WS-A |
| inspection GIFs | qualitative check that a rollout did what the metrics say | Epoch 5 WS-B; `Clean_Gifs/` |
| constraint overlays | the constraint geometry drawn on the rollout — obstacles, bounds, the foresight window | `Epoch9/Fix_14`, `Epoch9/U2` |
| overview plots | multi-rollout summary panels | `Epoch5/U5_Overview_Plots/PLOT_EXPLAINER.md` |

**GIF size was a recurring operational problem** (`Fix_7`, then `Fix_9` — "round 2"). It is not
thesis material, but it is why recording mode is a knob (`Recording mode set to: all`) and why some
runs carry artefacts others do not. Mention only if a figure's provenance needs it.

**Figures for the thesis should come from the constraint-overlay path, not the plain GIF path** —
the overlay is the one that shows the geometry the projector actually enforced, which is the thing a
reader needs to see.

---

## 3. The sanity-gate pattern, worth borrowing in the write-up

Epoch 5 ran a third workstream (WS-C) in parallel with rendering: a **mini-FM training run whose
only job was to confirm the data was correct before Epoch 6 committed to full training**. This
"cheap gate before expensive commitment" pattern recurs throughout the project (smoke seeds before
powered runs, `n_trials=2` wiring checks before `n_trials=20`). It is worth one sentence in
`sec:setup` as a stated methodology rather than leaving it implicit in the run history.

---

## 4. Holes — fill before writing

- [ ] **Render settings are not recorded**: resolution, camera pose(s), FOV, lighting, EGL device
      handling. Needed for `app:repro`.
- [ ] **Which V_A figures in the thesis will come from where** — the V_A env has its own recording
      path (D3IL's), separate from the UAV replay path described here. Do not conflate them; the
      state-injection caveat above is **UAV-only**.
- [ ] **Whether the Epoch 10 visual-UAV mode was ever used in a reported result.** If not, it is
      scoped-out infrastructure and should be said so once rather than described at length.
- [ ] **Attitude in the stored schema** — confirm no later revision added quaternion to the pickle;
      if one did, the level-drone caveat may be narrower than stated.
