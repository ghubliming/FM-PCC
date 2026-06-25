# Epoch 7 PLAN — Restore the FULL PCC/MPC bone for the UAV task (single pass)

**Status:** plan only (no code). **Build it all in ONE pass.**
**Goal:** make the UAV eval a faithful, full-feature replica of the proven FM PCC/MPC bone
(projection + candidate selection + MPC candidate fan + constraint-aware metrics), adapted
to the 3-D UAV plant. **Only the DYNAMICS constraint is really projected.** Bounds /
halfspace / obstacle (free-space) constraints are wired but left as **empty placeholders** —
we will design their geometry per scene later, and **we will not run them** this epoch.
Everything else must be **exactly the same as the old bone**.

---

## 0. Primary template — Gen7 visual-aligning eval (same bone, already 3-D, already gated)
Use **`fm_visual_aligning_test/eval_fm_visual_aligning.py`** as the reference, not the 2-D
FMv3ODE eval. It is the *same* PCC bone but already 3-D and **config-gated**, so it already
supports exactly what we want — run with **only dynamics active**, bounds/halfspace/obstacles
as empty placeholders:

```python
# fm_visual_aligning_test/eval_fm_visual_aligning.py  (build_projector)
constraint_list = []
if 'bounds'    in config['constraint_types']: constraint_list += [['lb',lb], ['ub',ub]]   # ← placeholder for us
if 'dynamics'  in config['constraint_types'] and 'model_free' not in variant:
    constraint_list += [('deriv',[6,0]), ('deriv',[7,1]), ('deriv',[8,2])]                 # ← the ONLY real one
if 'halfspace' in config['constraint_types']: ...                                          # ← placeholder
if 'obstacles' in config['constraint_types']: ...                                          # ← placeholder
```
Set `constraint_types = ['dynamics']` and bounds/halfspace/obstacle code never fires — but
the paths exist for the per-scene follow-up. The UAV port = **this eval minus the vision/image
conditioning**, with **obs 9-D instead of 6-D**.

The compute engine is **already forked and identical** — reuse as-is:
- `flow_matcher_v3_uav/sampling/policies.py` (selection: random / temporal_consistency /
  minimum_projection_cost).
- `flow_matcher_v3_uav/sampling/projection.py` (`Projector` with Dynamic/Safety/Obstacle
  constraints, SLSQP).

## 1. Why this epoch (E6 recap)
E6 ran only the pure-ML `diffuser` baseline (`batch_size=1`, no projector, no selection),
verified faithful to the old `diffuser` variant. Pure FM flies on single-mode scenes but
explodes on multi-mode (un-selected sample oscillates → 2-nd-order drone destabilises). The
candidate selection + projection that fix this are exactly the bone this epoch restores.

## 2. Build in ONE pass — the component list (all at once)
1. **Variant loop** over `projection_variants` = `['diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t']`
   (exact old semantics): `projector = None if variant=='diffuser'`;
   `trajectory_selection` = `temporal_consistency` (dpcc-t) / `minimum_projection_cost`
   (dpcc-c) / `random` (else). One `plans/<variant>/` output subfolder each.
2. **Multi-candidate sampling** — `batch_size > 1` (e.g. 4) in the policy call → a real MPC
   **candidate fan**. `sampled_trajectories_all` (already saved since U3) now holds the batch
   → `npz_analysis --replot-plans` / `plan_cand_spread` become meaningful.
3. **Projector wired into the policy call** (a `build_projector(variant, config, …)` helper
   mirroring the visual-aligning one). It projects the predicted plan; selection picks one
   candidate; the existing MuJoCo+PID loop executes its first action. **No change to the
   physics loop.**
4. **Constraints — DYNAMICS REAL, the rest placeholder:**
   - `constraint_types = ['dynamics']` (only). 
   - **Dynamics `deriv` mapping for UAV** (the resolved design call, see §3): bind **`p_des`**
     to the action, NOT the actual `p`.
   - `bounds` / `halfspace` / `obstacles` blocks present but their config lists are **empty**
     (`workspace_bounds`/`halfspace_constraints`/`obstacle_constraints` = placeholders), and
     their `constraint_types` keys are **off** → never built, never run.
5. **Constraint-aware metrics restored** to rollout/summary/npz:
   `n_success_and_constraints`, `n_violations`, `total_violations`,
   `collision_free_completed` (with only dynamics active these are ≈ trivial/zero — correct
   for a free-space, no-safety-constraint run). Keep E6's scene-aware `success` + `safe`.
6. **`config/uav.py` PCC block** (mirror the visual-aligning config shape):
   `projection_variants`, `constraint_types=['dynamics']`, `dt`, `batch_size`,
   `diffusion_timestep_threshold`, and **placeholder** `workspace_bounds` / per-scene
   `halfspace_constraints={}` / `obstacle_constraints={}` (all empty for now).
7. **Free-space run** of all 4 variants on `empty` (no obstacles), dynamics-only — the
   end-to-end acceptance of the restored bone.

## 3. UAV-specific adaptations (vs the visual-aligning template)
- **No vision.** Drop all image/ResNet conditioning; UAV is state-only (`returns_condition=
  False`, `test_ret=0`). The policy/projection engine is unaffected.
- **Transition layout:** UAV transition = `[action(3) | obs(9)]` = **12-D**; obs =
  `[p_des(3) | p(3) | v(3)]`. So indices: action `0:3`, `p_des` `3:6`, `p` `6:9`, `v` `9:12`.
  Projector `action_dim=3`, `transition_dim=12`, `variant='states_actions'`.
- **Dynamics `deriv` mapping — RESOLVED: bind `p_des`, not `p`.** In the visual-aligning arm,
  `('deriv',[6,0..2])` binds the *actual* `c_pos` because the arm tracks perfectly
  (`c_pos[t+1]=c_pos[t]+act`). On the drone the actual `p` **lags** (`p ≠ ∫act`), but
  `p_des` **is** the exact integrator (`p_des[t+1]=p_des[t]+act`, the dataset action
  convention). So bind `p_des`:
  `('deriv',[3,0]), ('deriv',[4,1]), ('deriv',[5,2])`. (This was the open P3 question in the
  prior draft — now answered.)
- **3-D bounds/obstacles (later):** placeholders must be 3-D ready — bounds = x,y,z box;
  pillars = cylinders → `sphere_outside` in x,y at altitude; corridor/s_curve walls = `ineq`
  halfspaces in x,y. Shape the placeholder config to accept these without an eval rewrite.

## 4. Output / metrics parity (acceptance checklist)
- [ ] Variants `diffuser` / `dpcc-r` / `dpcc-c` / `dpcc-t` each produce `plans/<variant>/`
      (npz + overview + log + diagnostics), same artifact set as E6 U3.
- [ ] `batch_size>1` → candidate fan present; `plan_cand_spread` non-NaN;
      `dpcc-c`/`dpcc-t` show committed/lower-spread plans vs `diffuser`/`dpcc-r`.
- [ ] Metrics back: `n_success_and_constraints`, `n_violations`, `total_violations`,
      `collision_free_completed` (≈ 0 violations, free-space dynamics-only).
- [ ] **Only `('deriv', …)` on `p_des` is active**; bounds/halfspace/obstacle lists empty.
- [ ] Free-space run of all variants completes without error.
- [ ] Bone-validation: on a multi-mode scene, `dpcc-t`/`dpcc-c` (with selection) **stop the
      E6 explosion** — direct confirmation the restored selection is what was missing.

## 5. Explicitly OUT of scope (placeholders, designed later, NOT run this epoch)
- Real geometry for **bounds / halfspace (free-space) / obstacle** constraints, per scene
  (pillar cylinders, corridor/s_curve walls, arena box, altitude floor). Empty stubs only.
- Any run that **activates** obstacle/free-space constraints. This epoch runs **dynamics-only**.
- Goal-conditioning / one-shot full-horizon architecture (separate track).

## 6. Risks
- **Projector compute:** SLSQP per step × `batch_size` × ~hundreds of FM steps may be slow;
  profile, and reuse the visual-aligning `diffusion_timestep_threshold` / snapshot cadence
  knobs to bound cost.
- **`states_actions` normalizer wiring:** the projector needs both obs and action
  normalizers (`ProjectorNormalizer`); copy the visual-aligning `_wrap_normalizers` pattern,
  adjusted for the 9-D obs.
- **`deriv` dt scaling:** confirm the UAV `dt` used in the `deriv` constraint matches the FM
  step rate (33 Hz, `dt_fm`), not the physics dt — the action is Δp_des per FM step.

## 7. References
- **Primary template:** `fm_visual_aligning_test/eval_fm_visual_aligning.py`
  (`build_projector`, variant loop, `deriv` mapping, constraint gating, placeholder drawing).
- Forked engine (reuse): `flow_matcher_v3_uav/sampling/{policies,projection}.py`.
- Old reference (2-D): `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`.
- Selection mechanics: `../../npz_analysis_tool/MPC_Candidate_Selection_Explained.md`.
- E6 baseline + the gap this closes: `../Epoch6_fm_pcc_training/U3/FINDING_homotopy_ambiguity_4scene_AB.md`.
- UAV eval to extend: `FM_v3_uav_test/eval_fm_uav.py`; config: `config/uav.py`.
