# Gen11 Epoch 6 — FM-PCC Training: IDEAS / What To Do

**Date:** 2026-06-13
**Status:** Blueprint — no code in this document.
**Predecessors:**
- [Epoch4 CLOSURE / U9 results](../Epoch4_expert_data/U9_Smooth_Trajectories/U9_EVAL_RESULTS.md) — expert dataset (3 scenes clean, pillars to recollect)
- [Epoch5 CLOSURE](../Epoch5_visual_and_validation/init_0/CLOSURE.md) — GIFs done, cameras + mini-FM not run
- [Path & prep](../Gen11_Path&Preperations/temp_Ideas.md) — the 7-step project spine

---

## 0. One-line summary

> Epoch 6 is the payoff: take the E4 expert dataset + E5 visual replay and **train the
> FM-PCC policy** for the MuJoCo quadrotor — state-only first, then DPCC safety projection,
> then visual. This is "step 4/5/6" of the project spine, finally executed.

---

## 1. Where we actually stand entering Epoch 6 (honest inventory)

| Asset | State | Blocker for E6? |
|---|---|---|
| UAV model + 4 scenes (empty/corridor/s_curve/pillars) | ✅ done (E1–E3) | No |
| Expert data — empty/corridor/s_curve | ✅ 500 ep each, clean, stop-and-go removed (U9 `blended_path`) | No |
| Expert data — **pillars** | ❌ 274 ep, 45% rejection, LRL/RLR homotopy collapse (14:1 imbalance) | **YES — recollect** |
| GIFs / physics GIFs / overview plots | ✅ done (E5) | No (inspection only) |
| **WS-A camera images** | ❌ never collected | YES for *visual* training |
| **WS-C mini-FM sanity gate** | ❌ never run | **YES — run before scaling** |
| DPCC quadrotor projector | ❌ does not exist yet | YES for safety stage |

**Two carry-over tasks from E4/E5 are hard prerequisites**: the pillars recollect and the
mini-FM sanity gate. Neither is optional if E6 training is to be trusted.

---

## 2. The plan — phased, each phase gates the next

### Phase 0 — Finalise the dataset (carry-over from E4 U9)

1. Apply `BLEND_RADIUS = 0.45` in `uav_expert_data_collect/trajectories.py`
   (was 0.30 → drops peak fillet accel 8.6 → ~5.7 m/s²; scales 1/r).
2. `python uav_expert_data_collect/verify_blends.py` — confirm 0.43 m clearance gate holds.
3. Recollect **pillars only** (`scene=pillars n_trials=500`). If rejection still >30%, also
   raise duration floor to `(12.0, 16.0)` in `generator.py` and rerun once.
4. **Discard** the old 274 pillars episodes — do not mix. Combine clean pillars with the
   ready empty/corridor/s_curve → **final E4 dataset**.

> Gate: all 4 scenes <30% rejection, homotopy balance within ~2:1. Until then, do not train
> anything beyond the empty-scene sanity model.

### Phase 1 — Mini-FM sanity gate (carry-over from E5 WS-C) — CHEAP, DO FIRST

Train a tiny FM on ≤100 empty-scene episodes; eval RMS vs ground-truth PID on held-out.
This is the cheapest possible catch for data-convention bugs (action delta vs absolute,
schema↔dataloader shape, normalisation) **before** spending cluster budget on full training.

- Pass: held-out RMS < 0.1 m, dataloader yields `(B, H, D)` cleanly.
- Fail RMS-diverging → **action convention wrong** (re-examine E4 Decision 1: `actions[t] =
  targets[t+1] - targets[t]`, position-delta). Fix before anything else.

> Gate: mini-FM passes → data pipeline is trustworthy → proceed to Phase 2.

### Phase 2 — State-only FM-PCC training (the real milestone)

Train the FM trajectory model on the full 4-scene state dataset. **No vision, no DPCC yet** —
just confirm FM closes the loop and flies without crashing at target Hz.

Key decisions to lock (see §3):
- **Which codebase to fork** as the UAV FM template.
- **Obs/action dims**: UAV `obs=(T,6)=[p(3)|v(3)]`, `actions=(T-1,3)=Δp_des`, `targets=(T,3)=p_des`.
  Note this is *not* the D3IL avoiding `(T,9)` schema — dims must be re-wired, not copied.
- **Horizon** `H=8`, flow steps `T_flow≈20` (fast ODE) as the mini-FM used.

> Gate: closed-loop rollout in MuJoCo reaches goal, contact-free on empty/corridor, at ≥33 Hz
> inference. Compare FM-only success vs the PID expert.

### Phase 3 — DPCC safety projection (the genuinely new wiring)

Add the DPCC/SLSQP projector on top of the FM prior, with **quadrotor dynamics** as the
projection model and scene geoms as half-space / signed-distance constraints `Z_f^t`.

- Reuse the projector from `diffuser/` (state DPCC) / `flow_matcher_v3_ode_selectable/`.
- Swap the prediction model to the quadrotor (differential-flatness or linearised — position
  trajectory → thrust/torque, no learning needed here).
- Constraints from `SCENE_OBSTACLES`: corridor walls, pillar cylinders, s_curve walls.
- Validate: given an FM action chunk, projection keeps the rollout within accel + obstacle
  constraints. This is where pillars (the hard scene) earns its keep.

> Gate: FM+DPCC beats FM-only on contact rate in pillars/s_curve, with bounded projection cost.

### Phase 4 — Visual FM-PCC (needs WS-A first)

1. Run **WS-A camera collection** (deferred from E5): replay state pickles via `mj_forward()`
   state injection, capture `bp-cam` + `fpv-cam` at 96×96. Watch the known RGB↔BGR swap
   (Gen9 Bugfix 4 / Gen7 Fix 18.6.1) and FPV-cam orientation (E5 U3 fix).
2. Add visual encoder (X-IL / FiLM-ResNet, dual camera) → condition the FM, à la the Gen9
   visual avoiding stack.
3. Retrain FM-PCC with visual conditioning; constraints can now come (partly) from rendered
   geometry rather than ground-truth geoms.

> Gate: visual FM-PCC matches state FM-PCC success within tolerance → vision is not degrading.

---

## 3. Decisions to make before coding Phase 2

| # | Decision | Options | Lean |
|---|---|---|---|
| D1 | UAV FM codebase template | (a) state DPCC `diffuser/` + `flow_matcher_v3_ode_selectable/`; (b) visual avoiding `fm_visual_avoiding/` stripped to state | **(a)** for state-only first — it is the paper-faithful DPCC stack and already pairs with the SLSQP projector |
| D2 | FM vs DPCC arms | Train both `diffuser`-style (diffusion) and `flow_matcher`-style (FM) for the thesis A/B | Keep **both common-mode** like Gen9 — same data, same infra |
| D3 | Action space the FM predicts | position-delta `Δp_des` (matches E4 schema) vs velocity/accel | **Δp_des** — already what the dataset stores; PID consumes `p_des` |
| D4 | DPCC prediction model | differential-flatness analytic vs learned/linearised quadrotor | **differential-flatness** — no learning, exact for the X2 |
| D5 | Pillars stop-and-go residue | FM will learn the conservative expert; accept or post-smooth | Accept for E6; revisit only if smoothness is a deployment requirement (E5 CLOSURE open Q) |

---

## 4. Evaluation — bring in the realtime framework

Epoch 6 is the first time we have a *policy* to time, so this is where the
[REALTIME_RECORDING framework](../../REALTIME_RECORDING/IDEAS.md) finally applies:

- **Primary metric — timing.** `total_ms = fm_ms + qp_ms` vs the 33 Hz / 30 ms budget. A
  policy that flies perfectly but takes >30 ms/step is not deployable. Timing can only be
  measured live (never loaded) → every eval runs the `BehaviorLogger`.
- **Secondary — behaviour.** success rate, contact fraction, DPCC active-step fraction,
  tracking error. Compare FM-PCC vs DPCC-baseline vs PID expert on the same seeds/scenes.
- **Zero-shot scenes.** Hold out a scene topology (e.g. wall-with-holes, denser pillars) to
  test generalisation, as flagged in the path doc's "Env zero-shot eval task".

---

## 5. Risk register (E6-specific)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Train on un-fixed pillars data → policy learns to crash at fillets | 🔴 | Phase 0 gate — recollect first, never mix old 274 ep |
| 2 | Action-convention bug surfaces only after full training | 🔴 | Phase 1 mini-FM gate catches it for ~2 h of compute |
| 3 | DPCC projector too slow → blows 30 ms budget | 🟠 | Time `qp_ms` early (Phase 3); cap SLSQP iters; warm-start from FM |
| 4 | Quadrotor dynamics in projector mismatch MuJoCo physics → projection useless | 🟠 | Validate one chunk: project → roll out in MuJoCo → check constraint held |
| 5 | Visual RGB↔BGR / FPV orientation regression | 🟠 | Reuse Gen9/Gen7 known fixes at WS-A (Phase 4) |
| 6 | Dim mismatch (UAV 6-D obs vs D3IL 9-D template) | 🟡 | Re-wire dims explicitly in dataset + model config; don't copy D3IL shapes |

---

## 6. Suggested deliverables for the Epoch 6 folder

| File | Purpose |
|---|---|
| `IDEAS.md` | This document |
| `EPOCH6_PLAN.md` | Detailed phase-by-phase execution plan (when Phase 0 unblocks) |
| `CHANGELOG.md` | Per-phase implementation log |
| `CLOSURE.md` | Final results: timing table, success/contact metrics, FM vs DPCC verdict |

---

## 7. Cross-references

| Document | Content |
|---|---|
| [Epoch4 U9 EVAL](../Epoch4_expert_data/U9_Smooth_Trajectories/U9_EVAL_RESULTS.md) | Pillars recollect spec (`BLEND_RADIUS=0.45`) |
| [Epoch5 CLOSURE](../Epoch5_visual_and_validation/init_0/CLOSURE.md) | WS-A / WS-C deferred, stop-and-go open question |
| [Epoch5 PLAN](../Epoch5_visual_and_validation/init_0/EPOCH5_PLAN.md) | WS-A camera spec, WS-C mini-FM spec (carry-overs) |
| [Path & prep](../Gen11_Path&Preperations/temp_Ideas.md) | 7-step spine; DPCC/FM/UAV-Flow reuse map |
| [REALTIME IDEAS](../../REALTIME_RECORDING/IDEAS.md) | Timing-first eval framework (applies from E6) |
