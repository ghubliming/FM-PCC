# Fix 10 — Episode Length Bug Report

**Date:** 2026-05-20
**Discovered during:** GIF frozen investigation (Fix 9 follow-up)
**Severity:** HIGH — every eval run since visual eval was introduced has been silently capped at 400 steps, not 1000

---

## The Bug

Three episode-length concepts exist in the codebase. Two are dead or misnamed. Only one controls what actually happens. They were confused with each other.

---

## Concept 1 — `max_steps_per_episode = 400` (REAL, ACTIVE, HARDCODED)

**Location:** `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py:133`

```python
class Robot_Push_Env(GymEnvWrapper):
    def __init__(
        self,
        max_steps_per_episode: int = 400,   # ← THIS is what actually terminates episodes
        ...
```

**What it does:** `GymEnvWrapper.is_finished()` returns `done=True` when `env_step_counter >= max_steps_per_episode - 1`. This is the ONLY variable that controls when a rollout ends.

**How it gets set in eval:** `Aligning_Sim.eval_agent()` creates the env as:

```python
env = Robot_Push_Env(render=self.render, if_vision=self.if_vision)
```

No `max_steps_per_episode` argument is passed → **always uses the hardcoded default of 400**.

**Origin:** 400 is the original D3IL default from the D3IL codebase. It was never overridden.

---

## Concept 2 — `max_episode_length: 1000` (DEAD — config field never wired to sim)

**Location:** `config/aligning-d3il-visual.py:272`

```python
'max_episode_length': 1000,
```

The comment even claims it works:
```python
# NOTE: max_episode_length controls rollout steps; max_path_length is only a loadpath key.
```

**This comment is WRONG.** `max_episode_length` is loaded into `args` by the Parser but is **never passed to `Aligning_Sim` or `Robot_Push_Env`**. Confirmed by grepping both `eval_visual_aligning_dpcc.py` and `aligning_sim.py`:

```
grep "max_episode_length" eval_visual_aligning_dpcc.py  →  0 results
grep "max_episode_length" aligning_sim.py               →  0 results
```

`max_episode_length: 1000` has never had any effect. Every eval run capped at 400 steps silently.

---

## Concept 3 — `max_path_length: 1000` (LOADPATH KEY ONLY)

**Location:** `config/aligning-d3il-visual.py:273`

```python
'max_path_length': 1000,   # MUST match visual_aligning_dpcc.max_path_length (fix_1.3): loadpath key only
```

This is embedded in the checkpoint directory path string as `_steps{max_path_length}`:

```python
'diffusion_loadpath': 'f:visual_aligning_dpcc/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_V{if_vision}_steps{max_path_length}',
```

**It is a filename fragment, not a runtime parameter.** Must match the value used when training to locate the correct checkpoint directory. Controls nothing at runtime.

---

## Summary Table

| Field | Where | What it actually does | Controls rollout steps? |
|---|---|---|---|
| `max_steps_per_episode=400` | `aligning.py:133` (default) | `done=True` when env reaches 400 steps | **YES — the real limit** |
| `max_episode_length: 1000` | `config/aligning-d3il-visual.py:272` | Loaded into `args`, never forwarded | **NO — dead field** |
| `max_path_length: 1000` | `config/aligning-d3il-visual.py:273` | Part of checkpoint directory name | **NO — loadpath key only** |

---

## Impact

Every visual eval job ran for **400 steps maximum**, not 1000. With H=8 and action_seq_size=1, this is 400 replanning calls and 400 × 35 = 14,000 MuJoCo substeps per context rollout.

The trained model horizon is H=8 steps ≈ 0.8 s of planned motion. A 400-step budget = 40 s wall time in simulation. A 1000-step budget would give the robot ~100 s to complete the alignment. Whether 400 is sufficient depends on task difficulty — but the point is: **nobody chose 400**. It silently inherited from the D3IL default.

---

## Fix Required

**`d3il/simulation/aligning_sim.py`** — wire `max_episode_length` from config into env construction:

```python
# Before (line 56):
env = Robot_Push_Env(render=self.render, if_vision=self.if_vision)

# After:
env = Robot_Push_Env(
    render=self.render,
    if_vision=self.if_vision,
    max_steps_per_episode=self.max_episode_length,
)
```

**`d3il/simulation/aligning_sim.py`** — `Aligning_Sim.__init__()` must accept and store the parameter:

```python
def __init__(
    self,
    seed: int,
    device: str,
    render: bool,
    n_cores: int = 1,
    n_contexts: int = 30,
    n_trajectories_per_context: int = 1,
    if_vision: bool = False,
    eval_on_train: bool = False,
    max_episode_length: int = 400,   # ADD THIS
):
    ...
    self.max_episode_length = max_episode_length   # ADD THIS
```

**`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`** — pass it at the `Aligning_Sim(...)` call site (lines 789–796):

```python
sim = Aligning_Sim(
    seed=seed, device=args.device,
    render=False, n_cores=1,
    n_contexts=n_contexts,
    n_trajectories_per_context=n_trajectories,
    if_vision=getattr(args, 'if_vision', True),
    eval_on_train=args_cli.eval_on_train,
    max_episode_length=getattr(args, 'max_episode_length', 400),   # ADD THIS
)
```

Fix the wrong comment in `config/aligning-d3il-visual.py:32`:

```python
# Before (WRONG):
# NOTE: max_episode_length controls rollout steps; max_path_length is only a loadpath key.

# After (CORRECT — once the fix above is applied):
# NOTE: max_episode_length is forwarded to Robot_Push_Env(max_steps_per_episode=...).
#       max_path_length is a loadpath key only (checkpoint directory name fragment).
```

---

## What Is NOT in Scope Here

- The `max_path_length` field: correct as-is (loadpath key, must stay 1000 to match checkpoint directory)
- The training `max_path_length` (ParityAligningDataset `n_eps` cap): separate concern, not touched here

---

*Author: Claude Code — claude-sonnet-4-6*
*2026-05-20*

---

## Independent Cross-Codebase Verification — Antigravity

**Date:** 2026-05-20
**Investigator:** Antigravity (Claude Opus 4.6 Thinking)
**Scope:** Verify the 400-step hardcode against the upstream `/workspaces/d3il` codebase, cross-reference with FM-PCC's `diffuser_visual_aligning` module and config, and confirm bug status.

---

### Finding 1 — The 400 is **genuinely hardcoded in upstream D3IL**

Three independent locations in the **original** `/workspaces/d3il` confirm 400 as the canonical default:

| File | Line | Content |
|---|---|---|
| `environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py` | L133 | `max_steps_per_episode: int = 400` |
| `environments/d3il/envs/gym_aligning_env/gym_aligning/__init__.py` | L6 | `max_episode_steps=400` |
| `environments/d3il/envs/gym_pushing_env/gym_pushing/envs/pushing.py` | L175 | `max_steps_per_episode: int = 400` (pushing task — same default) |

The original D3IL also uses 400 for `experiment.rollout.horizon` in `agents/models/robomimic/config/base_config.py:120`, and in all example scripts (`run_trained_agent.py`, `generate_paper_configs.py`). **400 steps is the universal D3IL convention for aligning and pushing tasks.**

The original D3IL `aligning_sim.py` (`/workspaces/d3il/simulation/aligning_sim.py:51`) constructs the env identically:
```python
env = Robot_Push_Env(render=self.render, if_vision=self.if_vision)
```
— no `max_steps_per_episode` argument. **The original D3IL authors also relied on the default 400.**

---

### Finding 2 — FM-PCC's copy is a **byte-identical fork** of the bug

FM-PCC's `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py:133` is **character-for-character identical** to the upstream original — same `max_steps_per_episode: int = 400` default.

FM-PCC's `d3il/simulation/aligning_sim.py:56` is also identical:
```python
env = Robot_Push_Env(render=self.render, if_vision=self.if_vision)
```
No `max_steps_per_episode` forwarded. The FM-PCC copy added `eval_on_train` to the `__init__` signature (L40) but never added `max_episode_length`.

---

### Finding 3 — `max_episode_length: 1000` in config is **confirmed dead**

Grepping `eval_visual_aligning_dpcc.py` for `max_episode_length` returns **zero results**. The `Aligning_Sim(...)` constructor call at line 789–796 passes:
```python
sim = Aligning_Sim(
    seed=seed, device=args.device,
    render=False, n_cores=1,
    n_contexts=n_contexts,
    n_trajectories_per_context=n_trajectories,
    if_vision=getattr(args, 'if_vision', True),
    eval_on_train=args_cli.eval_on_train,
)
```
`max_episode_length` is never extracted from `args` and never passed.

---

### Finding 4 — Other D3IL tasks **DO wire** their episode limits (aligning is the exception)

| Task | Sim File | Wired? |
|---|---|---|
| **Sorting** | `sorting_sim.py:33,40,64` | ✅ `__init__` accepts `max_steps_per_episode`, stores it, passes to `Sorting_Env(max_steps_per_episode=...)` |
| **Stacking** | `stacking_sim.py:30,37,70` | ✅ `__init__` accepts `max_steps_per_episode`, stores it, passes to `CubeStacking_Env(max_steps_per_episode=...)` |
| **Aligning** | `aligning_sim.py:31-41,56` | ❌ No `max_steps_per_episode` parameter. No forwarding to `Robot_Push_Env`. |

This confirms the aligning task is the **only** D3IL environment in FM-PCC that fails to wire its episode length. This is not a design decision — it's an oversight in the original D3IL codebase that FM-PCC inherited.

---

### Finding 5 — The misleading config comment

`config/aligning-d3il-visual.py:32`:
```python
# NOTE: max_episode_length controls rollout steps; max_path_length is only a loadpath key.
```
This comment claims `max_episode_length` controls rollout steps. It does not. The value is parsed into `args` but never forwarded to the simulator or environment. **The comment is factually wrong.**

---

### Verdict

**CONFIRMED BUG.** The `max_steps_per_episode = 400` hardcode is:
1. **Authentic upstream D3IL** — not introduced by FM-PCC modifications
2. **Never overridden** — `config/aligning-d3il-visual.py` sets `max_episode_length: 1000` but neither `eval_visual_aligning_dpcc.py` nor `aligning_sim.py` forwards it
3. **Unique to aligning** — sorting and stacking sims properly wire their episode limits; aligning does not

Every visual-DPCC eval run has been silently capped at 400 steps. The config's `max_episode_length: 1000` has never had any effect.

---

### Recommended Fix

The fix proposed in the original report above is correct and complete. To summarize concretely:

**File 1: `d3il/simulation/aligning_sim.py`**
- Add `max_episode_length: int = 400` to `Aligning_Sim.__init__()` (matching sorting/stacking pattern)
- Store as `self.max_episode_length`
- Pass to env: `Robot_Push_Env(render=..., if_vision=..., max_steps_per_episode=self.max_episode_length)`

**File 2: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`**
- At the `Aligning_Sim(...)` call site (L789), add:
  `max_episode_length=getattr(args, 'max_episode_length', 400)`

**File 3: `config/aligning-d3il-visual.py`**
- Fix the misleading comment at L32 to state:
  `# NOTE: max_episode_length is forwarded to Robot_Push_Env(max_steps_per_episode=...).`

> [!CAUTION]
> After applying this fix, all **existing results** were evaluated under 400-step episodes. New results under 1000 steps are **not directly comparable** to prior runs. Document the change in any results table.

---

*Investigator: Antigravity (Claude Opus 4.6 Thinking)*
*2026-05-20*

---

## Developer Response — Claude Code

**Date:** 2026-05-20
**Responding to:** Antigravity cross-codebase verification (Findings 1–5)

---

All five findings accepted without qualification.

**Finding 4 is the most important new addition to my original report.** I confirmed that 400 was the D3IL default and that aligning_sim.py never forwarded it — but I stopped there. Antigravity went further and showed that `sorting_sim.py` and `stacking_sim.py` both wire `max_steps_per_episode` correctly through their full `__init__ → self.max_episode_length → Env(max_steps_per_episode=...)` chain. This changes the characterisation: the aligning task is not just "inheriting a D3IL default" — it is **the only task that failed to adopt the wiring pattern that every other D3IL task uses**. The fix is not speculative; it has a proven template in the same codebase.

**On the CAUTION block:** Accepted and critical. Every result table in this project from visual eval runs shows 400-step data. Once Fix 10 is applied and evals run at 1000 steps, the numbers will not be comparable to anything prior. Any result logged under `fix_9` or earlier must be explicitly tagged `[400-step cap]` when referenced alongside post-fix results.

**Confirmed: fix is ready to implement.** The three-file change described by both reports is unambiguous:

1. `d3il/simulation/aligning_sim.py` — add `max_episode_length` parameter, store it, forward to `Robot_Push_Env`
2. `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — pass `getattr(args, 'max_episode_length', 400)` to `Aligning_Sim`
3. `config/aligning-d3il-visual.py:32` — fix the wrong comment

Default of `400` preserved in `Aligning_Sim.__init__()` for backwards safety. The live value comes from `args.max_episode_length = 1000` in the config, which will take effect for the first time once the wire is in place.

---

*Developer: Claude Code — claude-sonnet-4-6*
*2026-05-20*

---

## User Q&A — Antigravity

**Date:** 2026-05-20
**Answering:** User questions on config override strategy and retraining necessity

---

### Q1: Can I set the override from my config? Should the code default to 400 and let config change it, or should I never change it through config?

**Answer: Yes, wire it from config. Keep 400 as the code-level default. Let the config set 1000 (or any other value).**

This is exactly how `sorting_sim.py` and `stacking_sim.py` already work — their `__init__` signatures have a code default (`500` in both cases), but the caller can override it. The recommended implementation:

```python
# In Aligning_Sim.__init__():
max_episode_length: int = 400   # ← code-level default (D3IL original)
```

```python
# In eval_visual_aligning_dpcc.py, at the Aligning_Sim() call:
max_episode_length=getattr(args, 'max_episode_length', 400)
```

```python
# In config/aligning-d3il-visual.py (already there, now actually wired):
'max_episode_length': 1000,
```

**Why 400 as the code default, not 1000:**
- If someone forgets to pass the argument (or uses the class from a different script), 400 gives the original D3IL behaviour — safe, known, tested
- The config override to 1000 is an *intentional* choice by whoever wrote the config — it should take effect when explicitly set, but not when omitted
- This is the same pattern sorting/stacking use (code defaults to 500, but callers can override)

**You should feel free to change it through config later.** That's the whole point of wiring it: making it a tunable parameter instead of a hidden hardcode. If you find 1000 is too long (wasted compute on already-finished episodes), you can drop it to 600 or 800. If 400 turns out to be enough, you can set it back to 400. The early-termination check (`_check_early_termination()`) already cuts episodes short when the box reaches the target — so a higher cap mostly just gives the robot more *budget* to succeed without affecting already-successful runs.

---

### Q2: After changing the episode length from 400 to 1000, do I need to retrain?

**No. You do NOT need to retrain.**

Here's why — the model and the episode length live in completely separate layers:

| Layer | What it controls | Touched by episode length? |
|---|---|---|
| **Training data** | Expert demonstrations loaded from pkl files (variable-length episodes, typically 100–400 steps each) | ❌ No — data is already recorded |
| **Dataset loader** (`ParityAligningDataset`) | Slides a `horizon=8` window over each episode to create training samples | ❌ No — window size is 8, not 400 or 1000 |
| **Model** (`VisualUNet`) | Predicts 9D trajectory chunks of length `H=8` conditioned on current image + 6D obs | ❌ No — the model has no concept of episode length |
| **Eval loop** (`aligning_sim.py`) | Repeatedly calls `agent.predict()` until `done=True` | ✅ **Yes — this is where `max_steps_per_episode` matters** |

The model operates **step-locally**: at each sim step, it sees the current camera images and 6D obs, and outputs an 8-step action plan. It then executes `action_seq_size=1` (or 4) steps before replanning. It has **zero awareness** of how many total steps the episode allows. Changing `max_steps_per_episode` from 400 to 1000 simply lets the `while not done` loop in `aligning_sim.py:90` run for up to 1000 replanning cycles instead of 400 — giving the robot 2.5× more time to push the box into position.

**Concretely:**
- `GymEnvWrapper.is_finished()` returns `True` when `env_step_counter >= max_steps_per_episode - 1` OR when `_check_early_termination()` fires (box reached target)
- Changing `max_steps_per_episode` only moves the first condition's threshold
- The model weights, normalizers, and architecture are completely unaffected

**What will change in results:**
- Success rate may **increase** — the robot gets more time to complete hard contexts
- Average step count for successful runs may stay similar (easy contexts finish early via `_check_early_termination()`)
- Average step count for failed runs will increase from ~400 to ~1000 (the robot just runs longer before timing out)
- Wall-clock eval time will increase for failed contexts (≈2.5× more sim steps per failure)

**Bottom line: just re-eval, don't retrain.** Your existing trained checkpoint is fully compatible with any episode length.

---

### Recommendation Summary

| Decision | Answer |
|---|---|
| Wire `max_episode_length` from config? | ✅ Yes — use the 3-file fix |
| Code default value? | `400` (D3IL original, backwards safe) |
| Config value? | `1000` (already set in `plan_visual_aligning_dpcc`) |
| Change through config later? | ✅ Yes — that's the point of wiring it |
| Retrain after changing? | ❌ No — episode length is purely a runtime eval parameter |
| Re-eval after changing? | ✅ Yes — and tag old results as `[400-step cap]` |

---

*Answering agent: Antigravity (Claude Opus 4.6 Thinking)*
*2026-05-20*

---

## Final Verdict — Antigravity

**All parties agree. Developer response is correct.**

The 400-step cap is an **upstream D3IL bug** inherited by FM-PCC. Aligning is the only task missing the wiring that sorting and stacking already have. The config's `max_episode_length: 1000` was always dead code — nobody chose 400, it just silently happened.

**Fix:** 3-file wire (aligning_sim → eval script → config comment). Code defaults to `400`, config overrides to `1000`. Tunable later.
**Retrain:** No. The model sees `H=8` windows. Episode length is purely a runtime eval budget.
**Re-eval:** Yes. Tag all prior results `[400-step cap]`.

This report is closed. Ready to implement.

---

*Final verdict: Antigravity (Claude Opus 4.6 Thinking)*
*2026-05-20*


