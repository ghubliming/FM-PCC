# How the MPC picks the next-step candidate (and the "dots every N steps" in the plot)

**Date:** 2026-06-19
**Question:** at each control step we sample a *batch* of candidate trajectories — how exactly is **one**
chosen to act on, and what are the periodic plan snapshots ("dots every ~4 steps") in the eval figure?
**Files:** `E` = `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`,
`P` = `flow_matcher_v3_imeanflow/sampling/policies.py`.

---

## 1. The receding-horizon loop (one control step)

Per env step (`E:283` loop), for the current observation `obs`:

```
E:305   action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon)
E:308   next_pos_des = action + obs[:2]            # action is a Δ on (x,y)
E:309   env.step( concat(next_pos_des, fixed_z, quat) )
        → new obs → repeat
```

So each step: **sample a batch of candidate plans → select ONE → execute its first action → re-plan from
the new observation.** Classic receding-horizon MPC; only the first action of the chosen plan is ever used.

---

## 2. Inside the policy — sample, select, extract (`P:37–90`)

### 2a. Sample a batch of candidate plans
```
P:46   samples, infos = self.model(conditions, returns, projector, constraints, horizon)
P:48   trajectories = to_np(samples)               # [batch_size, horizon, transition_dim]
```
`batch_size` candidate trajectories are generated in parallel (plan block `batch_size`, e.g. 4). Each is a
full H-step plan; the projector (if active) has already snapped each near its end.

### 2b. Select ONE candidate — three rules (`P:58–69`)
The rule is `self.trajectory_selection`, set per **variant** in eval (`E:242–244`):

| variant contains | `trajectory_selection` | how it picks `which_trajectory` | code |
|---|---|---|---|
| `dpcc-t` | `temporal_consistency` | the plan **closest to the previous step's plan** (smoothest in time) | `P:59–62` |
| `dpcc-c` | `minimum_projection_cost` | the plan with the **lowest constraint-projection cost** | `P:63–67` |
| (else, incl. `diffuser`, `dpcc-r`) | `random` | **trajectory 0** (no preference) | `P:68–69` |

- **temporal_consistency** (`P:60`): `order = argsort‖observations[:, :-1] − prev_observations[:, 1:]‖`
  over the whole horizon → pick the candidate whose path best continues last step's path;
  `which_trajectory = order[0]`, and `observations` is **reordered** so index 0 = the chosen one.
- **minimum_projection_cost** (`P:64–67`): sum `infos['projection_costs']` per candidate, take `argmin` →
  the plan the projector had to bend least (most constraint-consistent).
- **random** (`P:69`): `which_trajectory = 0`. ("dpcc-**r**" = projector ON but **random** pick among the
  batch — the `r` is the selection, not "no projector".)

### 2c. Execute the first action of the chosen plan (`P:82–86`)
```
P:82   actions = trajectories[:, :, :action_dim]
P:83   actions = normalizer.unnormalize(actions, 'actions')
P:86   action  = actions[which_trajectory, 0]       # ← FIRST action of the SELECTED candidate
```
Only `[which_trajectory, 0]` is returned to the env. Waypoints `1…H-1` of even the chosen plan are
discarded (re-planned next step).

### 2d. Remember this step for the next temporal match (`P:70`)
```
P:70   self.prev_observations = repeat(observations[0])   # used by 2b temporal_consistency next step
```

---

## 3. The "dots every ~4 steps" in the plot = plan snapshots, not execution

Execution happens **every** step, but the **open-loop plans** are **snapshotted periodically** for the
figure:

```
E:248   save_samples_every = args.horizon // 2          # H=8 ⇒ every 4 executed steps
E:323   if _ % save_samples_every == 0:
E:324       sampled_trajectories.append(samples.observations[:, :, :])   # stash the whole batch's plans
...
E:353-357  plot each stashed plan (blue) for up to min(batch_size, 4) candidates + green start dots
```

So the **blue plan overlays / their start dots appear every `horizon//2` steps** (4 for H=8) — that is the
"every 4 steps" cadence you remember. It is purely a **visualization sampling rate**, unrelated to how
often the controller acts (every step) or selects (every step).

The **black executed path** (`E:346`) is the continuous closed-loop result; the **green dots** mark plan
starts (`E:347, E:357`).

### 3b. Why you see MANY candidates per point — plotted ≠ executed
The figure draws **every** candidate of the batch, not just the chosen one:
```
E:354   for ___ in range(min(args.batch_size, 4)):     # up to 4 candidate plans …
E:356       curr_ax.plot( sampled_trajectories[___][:horizon, x], [.., y], 'b')   # … all drawn blue
```
So the "lots of MPC candidates at each point" you see is the **full batch** (`batch_size` plans from
`batch_size` independent noise samples) — **visualization only**. The controller still executes **exactly
one** of them: `action = actions[which_trajectory, 0]` (`P:86`).

**For `diffuser` there is no real selection** — `trajectory_selection='random'` ⇒ `which_trajectory = 0`
(`P:68–69`), i.e. it simply takes **candidate 0** and runs its first action; the other blue plans are
drawn but discarded. Intelligent picking (`temporal_consistency`, `minimum_projection_cost`) happens only
for `dpcc-t` / `dpcc-c`. So: **the plot shows the whole fan of candidates; execution commits to one (for
diffuser, the 0th).**

---

## 4. Tracking reference + one subtle inconsistency

Tracking error uses the chosen plan's **second** waypoint as the desired next position:
```
E:322   desired_next_pos = samples.observations[0, 1, [x, y]]     # waypoint 1 of trajectory index 0
E:320-321  pos_tracking_errors = ‖actual_next_obs(x,y) − desired_next_pos‖
```

**Caveat (latent inconsistency):** `E:322` always indexes trajectory **0**, but the *executed* action used
`which_trajectory`:
- `random`: `which=0` ✔ consistent (traj 0 is the selected one).
- `temporal_consistency`: `observations` was **reordered** (`P:62`) so index 0 **is** the selected ✔.
- `minimum_projection_cost`: `observations` is **NOT** reordered (`P:63–67` only sets `which_trajectory`),
  so `observations[0]` is **not** the executed candidate ✘ — the tracking reference (and `prev_observations`
  at `P:70`) then refer to trajectory 0, not the one actually driven. Minor (affects only the `dpcc-c`
  tracking/temporal bookkeeping, not the executed control), but worth knowing when reading `dpcc-c` plots.

---

## 5. One-paragraph summary

Every control step samples `batch_size` full-horizon candidate plans; **one** is chosen — `random` (traj 0),
`temporal_consistency` (closest to last step's plan), or `minimum_projection_cost` (least-bent by the
constraint projector) — per the variant (`E:242–244`, `P:58–69`); the **first action** of that chosen plan
is executed (`P:86`) and everything else is re-planned next step. The periodic blue plan overlays in the
figure are open-loop snapshots taken every `horizon//2 = 4` steps (`E:248, E:323`), a plotting cadence
only. The executed black path is the continuous closed-loop result; tracking compares against waypoint 1 of
the (for `dpcc-c`, nominally) chosen plan (`E:322`).
