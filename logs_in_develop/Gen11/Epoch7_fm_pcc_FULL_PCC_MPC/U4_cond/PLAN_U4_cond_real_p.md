# PLAN — U4: two opt-in grounding options for the UAV (re-anchor + cond-on-real-`p`)

**Date**: 2026-06-28
**Status**: PLAN ONLY (no code yet)
**Authorized scope**: add **two** opt-in options, selectable from the config train/eval block, that
ground the command in the **measured** position so it can't run away from the lagging drone (the §8
`corridor_C` failure). **Keep ALL existing behavior as the default.**
Analysis: [../Real_Time_eval_loggging/data_example_anlysis/TRACKING_ERROR_Gen11E7.md](../Real_Time_eval_loggging/data_example_anlysis/TRACKING_ERROR_Gen11E7.md) §8–§9,
[CRITIQUE_three_layer_absurdity.md](../Real_Time_eval_loggging/data_example_anlysis/CRITIQUE_three_layer_absurdity.md) §9–§10.

---

## 0. TL;DR — two options, one default

| Option | What it does | Retrain? | Uses which checkpoint |
|---|---|---|---|
| **default** (`cond_mode='p_des'`, `reanchor_alpha=0.0`) | exactly today | — | existing |
| **A — Re-anchor** (`cond_mode='p_des'`, `reanchor_alpha>0`) | bleed the command back toward measured `p` at eval: `p_des = (1−α)(p_des+act) + α·p` | **NO** | **existing** (command-space FM, unchanged) |
| **B — Cond on real `p`** (`cond_mode='real_p'`) | plan in real position: obs `[p｜v]`, action `Δp`, setpoint `= p + k·Δp` | **YES** | **new** 9D FM |

Both attack the **same root** (the command running away → OOD → crash) by re-grounding to measured
`p`. **A is the free first test** (no retrain). **B is the principled version** (retrain).

> **In scope:** A (re-anchor, no retrain) + B (cond-on-real-`p`, retrain).
> **OUT of scope (per instruction):** ❌ #3 real drone dynamics model in the projector; ❌ #4
> termination on `track_err` — **we allow tracking error and log it.**

---

## 1. The two options are MUTUALLY EXCLUSIVE modes (answering "can I only turn on one?")

**Yes — you turn on one, the other's knob is moot.** The selector is `cond_mode`. Each mode has its
own knob; the other knob does nothing in that mode:

| `cond_mode` | active knob | the other knob | why |
|---|---|---|---|
| `'p_des'` (default family) | **`reanchor_alpha`** ∈ [0,1] | `lead_gain` ignored | command-space FM; you can bleed `p_des` toward `p` |
| `'real_p'` | **`lead_gain`** ≥ 1.0 | `reanchor_alpha` ignored | real-`p` FM; **already grounded by construction** |

> [!IMPORTANT]
> **Phase 1 (`real_p`) does NOT reuse Phase 0's re-anchor math — you are correct.** In `real_p` the
> eval integration is `setpoint = p + lead_gain·Δp`, i.e. the command is **rebuilt from measured `p`
> every step**. That *is* a hard re-anchor (equivalent to `α = 1`), but expressed structurally — so
> the `reanchor_alpha` formula simply **does not apply** in `real_p`. The two grounding mechanisms
> live in different modes and never combine. Pick `p_des`+`reanchor_alpha`, **or** `real_p`+`lead_gain`.

The three usable configurations:

```
1. cond_mode='p_des', reanchor_alpha=0.0   → EXACT current behavior (default)         [existing ckpt]
2. cond_mode='p_des', reanchor_alpha∈(0,1] → Option A: re-anchor (no retrain)         [existing ckpt]
3. cond_mode='real_p' (+ lead_gain)        → Option B: plan in real p (retrain)       [new 9D ckpt]
```

---

## 2. Why A needs NO retrain but B needs a retrain (train-vs-eval)

The deciding fact: **`p_des` and the action `Δp_des` are baked into the FM's weights at training
time.** The dataset trains the FM on the 12D tensor `[Δp_des｜p_des｜p｜v]`, conditioned on
`[p_des｜p｜v]` at t=0. So the model learned the contract *"given `[p_des｜p｜v]`, emit a `Δp_des`
trajectory."*

- **Option A changes only the EVAL integration** (`p_des += act` → `p_des = (1−α)(p_des+act)+α·p`).
  The FM's input (`[p_des｜p｜v]`), output (`Δp_des`), and weights are **untouched** → **no retrain**;
  runs on the **existing checkpoint**. The re-anchor is a post-hoc tweak to how the command accumulates.
- **Option B changes the TRAINING contract** (obs `[p｜v]`, action `Δp`, 9D tensor). The FM must
  *learn* to emit `Δp` from `[p｜v]` → **retrain from scratch** (and a different `transition_dim`, so
  a fresh, isolated checkpoint).

> The `dynamics` *constraint* is eval-time, but re-pointing it to real `p` **alone** is **not** a
> valid no-retrain lever: the trained `p` channel is the *lagging* real position and does not obey
> `p=∫act`, so binding it fights the model (§9.7); it's also near-vacuous in free space (§9.8). **The
> real no-retrain lever is the eval *integration* (Option A), not the constraint.**

---

## 3. Option A — Re-anchor (no retrain, existing checkpoint)

**Change (eval only):**
```python
reanchor_alpha = getattr(config, 'reanchor_alpha', 0.0)   # 0.0 = today; (0,1] = re-anchor
# was:  p_des = p_des + action
p_des = (1.0 - reanchor_alpha) * (p_des + action) + reanchor_alpha * p   # p = measured position
```
- `α=0.0` → identical to today. `α=1.0` → hard reset (`p_des = p + action`, command rebuilt from
  measured `p` each step → cannot run away). `α∈(0,1)` → partial bleed (keeps some lead, caps drift).
- **Train side: nothing.** Obs `[p_des｜p｜v]`, action `Δp_des`, FM weights — all unchanged.
- Projector/`dynamics`: unchanged (still binds `p_des`; near-vacuous, leave as-is).
- This is the **cheap hypothesis test**: does grounding the command stop the `corridor_C` spiral,
  using the model we already trained?

---

## 4. Option B — Cond on real `p` (retrain, new 9D checkpoint)

**Change set:**

```
            cond_mode='p_des' (today)                  cond_mode='real_p' (new)
TRAIN   traj = [Δp_des | p_des | p | v] (12D)      traj = [Δp | p | v] (9D)
        action = Δp_des (from pkl)                 action = Δp = diff(p)   (recomputed from pkl, §7)
        obs    = [p_des | p | v]                   obs    = [p | v]
EVAL    obs = concat([p_des, p, v])                obs = concat([p, v])
        p_des += action                            setpoint = p + lead_gain * Δp   (structurally α=1)
        v_des = action/dt                          v_des = action/dt   (formula unchanged)
PROJ    deriv binds p_des (idx 3–5)               deriv binds real p (idx 3–5 of the 9D traj) — feasible
                                                   because action=Δp makes p=∫act tautological; NOT a model
```

- **`lead_gain` (default 1.0):** `setpoint = p + lead_gain·Δp`. `1.0` = pure grounded (may
  **under-track**: the setpoint doesn't lead the drone by the controller lag → safe-but-sluggish).
  Raise to ~1.5–2.0 if it under-reaches goals. (A *learned* lead = the #3 model route, which we are
  **not** doing.)
- **Why binding real `p` is OK here without a model:** `action=Δp` ⇒ `p=∫act` is a tautology ⇒ the
  deriv is feasible, and `skip_initial_state` pins `p[0]=measured` → grounded. This is **not** a
  drone model (so it respects "no #3"); it's the same Euler relation, re-pointed.

---

## 5. Files to modify (small, backward-compatible)

| File | Option A | Option B |
|---|---|---|
| `FM_v3_uav_test/eval_fm_uav.py` | branch the integration (`:338`) with `reanchor_alpha` | branch obs assembly (`:312` → `[p,v]`), integration (`:338` → `p + lead_gain·Δp`), and `deriv` target (`:157` → real-`p` channel) |
| `flow_matcher_v3_uav/datasets/sequence.py` (or preprocessing/buffer) | — (unchanged) | `cond_mode=='real_p'`: build `observations=[p｜v]`, `actions=diff(p)` from the loaded pkl |
| `FM_v3_uav_test/train_fm_uav.py` | — | pass `cond_mode`; model **auto-sizes** (`transition_dim=obs_dim+action_dim`, `cond_dim=obs_dim`, lines 259–260) — no model edit |
| `config/<uav config>` (train+eval) | add `'reanchor_alpha': 0.0` | add `'cond_mode': 'p_des'`, `'lead_gain': 1.0` + path-isolation fragment (§7) |

All defaults (`cond_mode='p_des'`, `reanchor_alpha=0.0`) → **byte-identical to today**.

---

## 6. Backward-compatibility

1. ✅ No config change → defaults → identical to today; every existing checkpoint loads & evals unchanged.
2. ✅ Option A reuses the **existing** checkpoint (eval-only tweak).
3. ✅ Option B is a **fresh 9D** model, isolated on disk (§7); cannot collide with `p_des` checkpoints.
4. ✅ FM backbone, solver, real-time loop: unchanged.

---

## 7. No data regen + checkpoint isolation

- **No regeneration:** the pkls already store **both** `p_des` and real `p`
  (`dataset_writer.py: obs=[p_des｜p｜v]`). Option B builds `action=diff(p)`, `obs=[p｜v]` at load time
  from existing episodes.
- **Isolation:** append a `_cond{cond_mode}` (and optionally `_a{reanchor_alpha}` for eval-only runs)
  fragment to the train `exp_name`/eval `diffusion_loadpath` so `p_des` vs `real_p` save in parallel
  dirs — same mechanism as `film_mode`. (Re-anchor runs reuse the `p_des` checkpoint but should still
  tag eval outputs with `α` so A/B/default results don't overwrite.)

---

## 8. What we deliberately DON'T do

- ❌ **No real plant model (#3).** Both A and B ground via re-anchoring/tautology, not a drone model.
- ❌ **No termination (#4).** Tracking error is **permitted and logged** — the experiment is to see
  whether grounding the command (A or B) reduces drift on its own. (Note: if A/B prevent the
  *runaway*, the post-crash "frozen drone, FM keeps planning" situation largely disappears anyway,
  so #4 becomes far less necessary — but we are not adding it.)

---

## 9. Verification — Phase 0 (A, free) → Phase 1 (B, retrain)

**Phase 0 — Option A, NO retrain (do this first):**
1. On the existing checkpoint, sweep `reanchor_alpha ∈ {0.0, 0.5, 1.0}`.
2. Re-run `corridor_C` (the §8 failure). Compare `result`, `goal_dist`, `contacts`, `max_track_err`,
   and the `track_err(t)` curve vs the `α=0.0` baseline.
3. **If `track_err` stays bounded and the crash stops → hypothesis confirmed for free.** Decide
   whether A is "good enough" or whether B's principled version is worth a retrain.

**Phase 1 — Option B, retrain (only if needed / for the clean version):**
4. Train the 9D `real_p` model; smoke-test forward shapes.
5. Eval `real_p` vs `p_des` on the same seeds/scenes; check the same metrics + per-scene.
6. **Under-tracking check:** does `real_p` (or A with high α) under-reach goals? If yes, raise
   `lead_gain` (B) / lower `α` (A).

**Success = the command stays anchored to measured `p`, `track_err` stays bounded, `corridor_C`
stops crashing — without a model and without a terminator.**

---

## 10. Risks & open decisions

| Item | Note |
|---|---|
| Under-tracking (sluggish, miss goals) | both A (high α) and B (k=1) can under-lead; mitigate with α<1 / `lead_gain>1`. We accept some tracking error by instruction. |
| `deriv` index reuse (3–5 = `p_des` in p_des-mode, real `p` in real_p-mode) | branch on `cond_mode` so the projector binds the right channel. |
| `goal_dim` for the 9D layout | confirm `goal_dim` inference still holds when obs=`[p｜v]`. |
| Re-anchor `α` is eval-only; tag results | so A/default/B outputs don't overwrite each other. |
| Mutual exclusivity enforced | assert in code: if `cond_mode='real_p'`, ignore/forbid `reanchor_alpha`; if `'p_des'`, ignore `lead_gain`. |

---

## 11. Task breakdown

1. **T1 (Option A, no retrain):** add `reanchor_alpha` branch to `eval_fm_uav.py` integration + config key; tag eval outputs with `α`.
2. **T2 (Phase 0 test):** sweep `α∈{0,0.5,1.0}` on `corridor_C` with the existing checkpoint; decide if B is needed.
3. **T3 (Option B dataset):** `cond_mode` branch building `obs=[p｜v]`, `action=diff(p)` (sequence/preprocessing).
4. **T4 (Option B train/eval):** thread `cond_mode`; branch eval obs/integration/`deriv`; add `lead_gain`; path isolation.
5. **T5 (Phase 1 test):** train + eval `real_p` vs `p_des`; under-tracking sweep.

> **Definition of done:** defaults → identical to today; **Option A** runs on the existing checkpoint
> and bounds `track_err` on `corridor_C` with no retrain; **Option B** trains a 9D real-`p` model with
> re-anchored setpoints, isolated on disk; **the two modes are mutually exclusive** (one knob each);
> **no** real model and **no** terminator added.
