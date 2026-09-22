# PLAN — Gen15 U18 · `pillars_v2` = the D3IL-avoiding planner flown by the quadrotor in a similarity-scaled scene

**2026-09-22 · assessment + implementation plan · NO CODE WRITTEN YET — waiting for approval.**
Internal name `pillars_v2` (a.k.a. "pillars-v2 avoiding"); in the thesis simply *pillars*.
Predecessor: [U17](../U17/CLOSURE_20260922_U17_abandoned.md) (abandoned 2026-09-22, all-zero grid).
**rev 2 (same day):** adds **Mode T — bridging-turbo** (§3.3): replay the *stored* avoiding executions through the
drone plant, no network, no NLP. Mode L (live closed loop) is kept unchanged. Both share one plant.
**rev 4 (same day, after pilot 26073):** the scale is **36**, not 10 — the Panda paths hug the obstacles at the rod
radius (0.009–0.02 units), so the faithful map sends the rod radius (0.01) onto the drone's radial reach (0.36 m);
at 10× the drone touched a pillar in 15–18 of 20 episodes with perfect tracking. Clock mode: feed-forward off
(this PID flips above ≈ 0.5 m/s of `v_des`), rate-limited reference 1 m/s, 1 setpoint/s. Details and evidence:
[`CHANGELOG_…fix3_scale36_clock_mode.md`](CHANGELOG_20260922_U18_fix3_scale36_clock_mode.md). §2.3–2.4 below are
superseded where they say 10× / 5 Hz / feed-forward.
**rev 3 (same day, coded):** §2.2–2.3 corrected — each avoiding geometry selects **one** keep-out disk (not a disk per
pillar); the physical margin comes from the demonstrated gap clearance. Implementation: `uav_avoiding_bridge/`,
changelog [`CHANGELOG_20260922_U18_pillars_v2_bridge_coding1.md`](CHANGELOG_20260922_U18_pillars_v2_bridge_coding1.md).

> **The idea in one line.** Keep every trained D3IL-avoiding model, its normaliser, its DPCC / HardFlow projector,
> its three test-time halfspace geometries and its scorer **exactly as they are**, and swap only the *plant*: instead
> of the Panda arm tracking the 2-D setpoint through inverse kinematics, a quadrotor tracks it with the existing
> cascaded PID in a MuJoCo scene that is the avoiding obstacle field scaled up by a factor $s$ and extruded into
> tall pillars. The planner never learns it is flying. No training, no new constraints, no new metric.

---

## 0 · Verdict of the assessment

| question | answer |
| :-- | :-- |
| Does it reuse everything? | Yes. The avoiding eval loop talks to the environment through exactly five calls (`start`, `reset`, `robot_state`, `step`, `close`) and reads back a 2-D position plus a `(mode, success)` tuple. A drop-in plant with that contract leaves the planner, projector, selection rules, violation scorer, npz schema, `load_results_*`, the batch reporter and the DA readers untouched. |
| Does it fix what killed U7 and U17? | Yes, by construction. On avoiding the constraints are introduced *at test time* and 0 / 1 / 2 of 96 demonstrations satisfy them, so the projector has real repair work (the U7 problem), and the projector's behaviour in that set is already known and tabulated from the Panda runs (the U17 problem: nothing new can go degenerate in the planner). The only new physics is tracking error. |
| Is "2×, 3×" enough? | **No.** The X2 spans 0.62 m rotor-tip to rotor-tip. At 3× the physical opening between the second-row pair is 0.30 m and a constraint-legal plan already touches a pillar (§2.3). **The drone fits from ≈ 8×; the plan proposes 10×** (6.0 × 7.0 m arena, pillars r = 0.25 m, keep-out 0.80 m, 0.24 m of tracking error allowed before a legal plan can touch anything). |
| Cost | Evaluation only. **Mode T** replays the whole existing avoiding corpus (every model, rule, budget, seed, geometry that has an npz) on **CPU in well under an hour** — no GPU, no NN, no NLP. **Mode L** (live closed loop) adds ~1 ms of physics per control step to a loop whose planner already costs 20–550 ms per step, so a live replica of the avoiding protocol costs what the avoiding waves cost: hours, not GPU-days. |
| Is the turbo idea sound? | Yes, with one sentence of honesty. The stored `obs_all` of every avoiding cell is the exact setpoint sequence the planner + projector produced and the Panda tracked, so flying it through the drone plant answers "is the plan the projector delivered flyable?" — without re-solving anything. It cannot show the planner *reacting* to the drone (the loop is open); Mode L on a subset shows that the two agree. Timing columns carry over verbatim, because the plant's time is not counted in either mode. |
| Main risk | Distribution shift in the *setpoint-minus-position* gap the model conditions on (obs = `[x_des, y_des, x, y]`). The Panda tracks to millimetres; a lagging drone at 10× would show the model gaps it never saw. Mitigated by velocity feed-forward, a modest speed, and a hard gate (§4, G1/G3) *before* any model wave. |
| New code | ≈ 850 lines, one new root folder, one generated scene XML, a 5-line hook in each of the five avoiding eval scripts, one Slurm driver. Nothing in `mix_uav*`, `flow_matcher_v3*`, the projector or the configs changes. |

---

## 1 · Why this is the right scene after U17

U7 (`pillars_hg`): the constraint was the one the demonstrations were built to satisfy → nothing to repair.
U17 (`pillars_xl`): the keep-out closed a physically open lane by centimetres → every local projector took it.
Both were *new* geometries whose interaction with the projector was unknown until 5 GPU-days had been spent.

`pillars_v2` inverts the order. The geometry and the projector's behaviour in it are the avoiding results the
thesis already has (`tab:avoiding-dpcc-protocol`, `fig:avoiding-tradeoff`, `tab:avoiding-budget20`): for every
model, selection rule and budget the S&C, violating steps and steps-to-goal are known. The UAV run then asks a
sharper and cheaper question — *does that ordering survive execution by a physical vehicle with a real tracker?* —
and the answer is a diff against known numbers, not a search in the dark.

For the thesis this is a **cross-embodiment execution** of the avoiding planner, and it must be presented as that:
the planner is the avoiding-trained model, the drone executes it in a scene similar (in the geometric sense) to the
avoiding table. It is *not* a UAV-trained model. Said plainly once in the setup chapter, it is a clean story:
Chapter 6 already has "the projector repairs an infeasible plan" (avoiding, corridor); pillars adds "…and the
repaired plan is flyable".

---

## 2 · The mapping

### 2.1 Frames

Avoiding frame $a = (x_a, y_a)$: table coordinates, progress along $+y_a$ from the start
$(0.525, -0.28)$ to the finish line $y_a = 0.35$; field $x_a \in [0.2, 0.8]$, $y_a \in [-0.3, 0.4]$.
World frame $w = (X, Y, Z)$: the quadrotor scene, progress along $+X$ (as every UAV scene in the repo).

Similarity transform (scale $s$, proper rotation by $-90°$, translation $c$), one function and its inverse:

$$X = s\,(y_a - c_y), \qquad Y = -\,s\,(x_a - c_x), \qquad Z = Z_0 \;(\text{constant})$$

with $c = (0.5,\ 0.035)$ (the field's centre line and the midpoint between start and finish). It is a proper
rotation, so left/right are preserved; the halfspace names (`top-left-hard` = `\`, `top-right-hard` = `/`)
keep their meaning in the avoiding frame, which is the only frame the planner and the tables ever see.

### 2.2 The scene at $s = 10$

| object | avoiding $(x_a, y_a)$, physical r | world $(X, Y)$ | pillar r (m) |
| :-- | :-- | :-- | --: |
| start | (0.525, −0.28) | (−3.15, −0.25) | |
| row 1 `l1` | (0.500, −0.10), 0.030 | (−1.35, 0.00) | 0.30 |
| row 2 `l2_top` / `l2_bottom` | (0.425, 0.08) / (0.575, 0.08), 0.025 | (0.45, +0.75) / (0.45, −0.75) | 0.25 |
| row 3 `l3_top` / `l3_mid` / `l3_bottom` | (0.35 / 0.50 / 0.65, 0.26), 0.025 | (2.25, +1.5 / 0 / −1.5) | 0.25 |
| finish line | $y_a = 0.35$ | $X = 3.15$ | |
| field | [0.2, 0.8] × [−0.3, 0.4] | $Y \in [-3, 3]$, $X \in [-3.35, 3.65]$ | |

**The planner's constraint set is not "a keep-out per pillar".** Each geometry name selects (eval loop, verbatim)
one or two halfspaces plus **one** keep-out disk of radius 0.08 (0.80 m in the world):

| geometry | halfspaces (avoiding frame) | keep-out disk centre | world centre |
| :-- | :-- | :-- | :-- |
| `top-left-hard` | `\` hard: (0.8, −0.5)–(0.4, 0.5) below | (0.40, 0.08) | (0.45, +1.00) |
| `top-right-hard` | `/` hard: (0.2, −0.5)–(0.6, 0.5) below | (0.60, 0.08) | (0.45, −1.00) |
| `both-hard` | `\` and `/` easier: (0.8, −0.3)–(0.575, 0.5), (0.2, −0.3)–(0.425, 0.5) | (0.50, −0.09) | (−1.25, 0.00) |

The six physical pillars are avoided by the *learned behaviour* (the demonstrations thread the gaps), exactly as on
the table where a rod–obstacle contact ends the episode. That is what the drone's physical size has to fit.

Altitude $Z_0 = 1.0$ m, pillars 2 m tall (z ∈ [0, 2]) — the "infinite" extrusion. The drone's z-loop holds
$Z_0$; the plan has no z. Constraint halfspaces and the 7 keep-out entries of `projection_eval.yaml` are **not**
mapped anywhere: they stay in the avoiding frame, where the projector and the scorer read them.

### 2.3 Why the scale must be ≥ 8 (and 10 is proposed)

The demonstrated routes thread gaps of 0.15 (centre to centre) between the row-2 and row-3 cylinders of radius
0.025, i.e. a centred pass has **0.05 avoiding-units of clearance** to each pillar surface (the rod is 0.01). In the
world that clearance is $s \cdot 0.05$ and must absorb the drone's own reach (0.31 m) **and** its tracking error:

| $s$ | arena (m) | pillar r | row-2/3 physical opening | tracking error a centred pass may have before touching | step at the action bound | speed at 5 Hz |
| --: | :-- | --: | --: | --: | --: | --: |
| 2 | 1.2 × 1.4 | 0.050 | 0.20 | −0.21 | 0.02 | 0.10 |
| 3 | 1.8 × 2.1 | 0.075 | 0.30 | −0.16 | 0.03 | 0.15 |
| 6 | 3.6 × 4.2 | 0.150 | 0.60 | −0.01 | 0.06 | 0.30 |
| 8 | 4.8 × 5.6 | 0.200 | 0.80 | +0.09 | 0.08 | 0.40 |
| **10** | **6.0 × 7.0** | **0.250** | **1.00** | **+0.19** | **0.10** | **0.50** |
| 12 | 7.2 × 8.4 | 0.300 | 1.20 | +0.29 | 0.12 | 0.60 |

Negative = the vehicle does not fit through the gap even with perfect tracking. At 2–3× no route of the data is
flyable; the drone fits from 8×; 10× leaves 0.19 m of tracking slack in the tightest gaps (0.39 m in the row-1 to
row-2 diagonal). The planner's keep-out disk (0.80 m at 10×) and halfspaces are unaffected by the choice: they
scale with everything else, and the rescoring reads them in the avoiding frame. $s$ is one constant
(`frame.SCALE`, env `FMPCC_AVOID_UAV_SCALE`); 8 and 12 are one-line variants if the gate asks.

### 2.4 Time, speed and the tracker

The avoiding action is a position increment bounded at 0.010–0.012 per step, with `dt = 1` (no clock). The
plant therefore owns the clock: one planner step = one control period $T_c$. Proposed default **$T_c = 0.2$ s
(5 Hz)** → ≤ 0.10 m per step, ≤ 0.5 m/s, the speed the UAV expert data was flown at (0.4 m/s). The setpoint is
tracked by `CascadedPID` (Kp = [4, 4, 8], Kd = [3, 3, 4]) with **velocity feed-forward** $v_{des} = \Delta p / T_c$
(the E7 `pid` mode): with $v_{des} = 0$ (`pid_stopgo`, used on `pillars_hg`) a PD loop trails a ramp by
$K_d v / K_p = 0.75\,v$ = 0.37 m at 0.5 m/s, which is the 0.34 m `track_err` the old pillars scene measured and
which would become 0.037 avoiding units of gap — out of the model's distribution (§4 G3). With feed-forward the
ramp lag vanishes and only turn transients remain. Both are knobs; nothing else in the PID is touched.

---

## 3 · Design — the plant and where it plugs in

### 3.1 Contract the plant has to honour (read off the five eval loops; identical in all of them)

```
env = ObstacleAvoidanceEnv(); env.start()
obs = env.reset()                          # -> (2,) float32  = robot xy   [avoiding frame]
action = env.robot_state()[:2]; fixed_z = env.robot_state()[2:]
obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0])))
success = info[1]                          # info = (mode_encoding, success)
env.close()
```
`next_pos_des` is integrated by the *loop* (`action + obs[:2]`) — the plant is never asked for it; the loop then
builds `obs = [x_des, y_des, x, y]`. Success = `y > 0.35`; a rod–obstacle contact terminates the episode as a
failure; the step budget is `max_episode_length = 200` (config) and the env's own cap 250.

### 3.2 `UavAvoidingPlant` (new module, ~250 lines)

Same five methods, same shapes and semantics, avoiding units in and out:

- `__init__(scale=10, control_hz=5, altitude=1.0, feedforward=True, gain='pid_default', terminate_on_contact=True)`
  loads `scene_avoiding_pillars_s10.xml`, builds `CascadedPID` via `generator._make_pid`, asserts the XML's pillar
  centres equal `T(avoiding centres)` (single source of truth is the constants, §3.3).
- `reset()` → drone at `T(start)`, hover state, one settle burst (0.5 s) so the first observation is at rest;
  returns mapped xy. `robot_state()` → `[x_a, y_a, Z_placeholder]`.
- `step(a7)` → `p_des = T(a7[:2])`, `v_des = (p_des - p_des_prev)/T_c` (or 0), run `round(T_c/dt)` MuJoCo steps
  with `tracker.compute(p, q, v, ω, p_des, v_des)`; per physics step: obstacle contact (`generator._is_obstacle_contact`,
  non-floor contact), arena / speed divergence guard (thresholds copied from `eval_mix_uav.py`, not imported).
  Returns `(T⁻¹(p_xy), 0.0, terminated, (mode_encoding, success))` with `success = (y_a > 0.35)`, `terminated =
  success or contact or diverged or step ≥ 250`. `mode_encoding` is the avoiding `check_mode` copied verbatim (gives
  the flown-lane label for free).
- Sidecar record per episode (**new information only**, never in the npz): world path, setpoint path, `track_err`
  (mean/p95, m and avoiding units), the *gap* `‖(x_des,y_des) − (x,y)‖` statistics, contacts, divergence, control
  period — written by the plant to `<results>/uav_plant_trial{i}.json` when the loop calls `reset()` for the next
  trial / `close()`. The DA reads it next to `{variant}.npz`.
- Optional overhead PNG of the world frame per trial (`FMPCC_AVOID_UAV_PNG=1`), drawing the physical pillars **and**
  the mapped keep-outs (U17 lesson: a picture with only the physical pillar misleads). No GIFs by default.

### 3.3 Mode T — bridging-turbo (replay the stored executions; nothing is solved)

Every avoiding npz (all five arms: FM, MeanFlow, CI-MeanFlow, diffusion DPCC, HardFlow — checked) stores
`obs_all`: per episode the executed `[x_des, y_des, x, y]` sequence, i.e. **the setpoint path the planner and the
projector actually produced**, plus `act_all`, `n_steps`, `avg_time`, `args`. Mode T takes `obs_all[i][:, :2]`
(the setpoint sequence), maps it with `to_world()` and drives the plant with it, one setpoint per control period,
exactly as Mode L would — but the setpoints come from the file, not from a network + NLP call.

```
for every  logs/avoiding-d3il/plans/<engine>/<train>/<eval>/<seed>/results/halfspace_<geo>/<variant>.npz :
    for episode i:  plant.reset(); for p_des in obs_all[i][:, :2]: plant.step(p_des)   # 5 Hz, feed-forward
                     stop at: finish crossed | contact | divergence | sequence exhausted
    rescore on the DRONE's mapped path (same scorer arithmetic as the eval loop, copied):
        n_success            = finish line crossed (y_a > 0.35)
        collision_free_completed / n_violations / total_violations  = halfspace + keep-out checks per step
        n_success_and_constraints = both
        n_steps              = steps flown (== recorded unless stopped early)
        avg_time             = COPIED from the source npz (planner time; the plant's time is not a metric)
    write  …/<eval>_msguavpv2s10turbo/<seed>/results/halfspace_<geo>/<variant>.npz   (same keys, same dtypes,
           obs_all = drone-mapped [x_des, y_des, x, y], act_all/args/sampled_trajectories_all copied)
           + uav_plant_trial{i}.json sidecar (world path, track_err, gap, contact, divergence, source npz path)
```

What it buys: the **complete** pillars grid — every model × selection rule × budget × seed × geometry the thesis
already has for avoiding, including the diffusion baseline at K = 20 and the HardFlow arms — from one CPU job, with
no risk of a planner-side surprise (the plan is the recorded one). What it cannot show: the planner correcting for
the drone's lag mid-flight (open loop). That is exactly why Mode L stays: a live subset (§5) shows that open-loop
and closed-loop numbers agree, after which Mode T's full grid is the table and Mode L is its check.

Two replay policies, one knob (`FMPCC_AVOID_UAV_REPLAY=clock|settle`):
- `clock` (default): fixed 0.2 s per setpoint, as Mode L — the realistic one, the gap can grow if the PID lags.
- `settle`: advance to the next setpoint only when `‖p − p_des‖ < ε` (cap 2 s) — pure geometric flyability, time is
  not counted anyway. Useful to separate "the path is unflyable" from "the path is flown too fast".

Early-stop semantics are the avoiding ones: the recorded episode that *terminated* on a Panda contact is replayed to
that same step; the drone's own contact stops it earlier. A recorded success whose replay hits a pillar becomes a
failure — that difference is the result.

### 3.4 Files

| file | status | lines | what |
| :-- | :-- | --: | :-- |
| `uav_avoiding_bridge/frame.py` | new | ~60 | constants (`SCALE`, `ORIGIN`, `ALTITUDE`, avoiding obstacle table copied from `avoiding_objects.py`), `to_world()`, `to_avoiding()` |
| `uav_avoiding_bridge/scene.py` | new | ~80 | `build_scene_xml(scale)` → MJCF string (drone include, floor, N pillars, visual finish line, arena size from the field); `write_scene_xml()` |
| `d3il/…/quadrotor/scenes/scene_avoiding_pillars_s10.xml` | new, generated & committed | — | so it sits beside the other scenes and can be viewed; header comment carries the generator constants |
| `uav_avoiding_bridge/plant.py` | new | ~250 | `UavAvoidingPlant` (§3.2) |
| `uav_avoiding_bridge/factory.py` | new | ~40 | `make_avoiding_env()` — returns `ObstacleAvoidanceEnv()` unless `FMPCC_AVOIDING_PLANT=uav`; in `uav` mode **refuses to run unless `FMPCC_RUN_MSG` contains `uav`** (otherwise the run would overwrite the Panda npz in the same results folder), and reads `FMPCC_AVOID_UAV_SCALE / _HZ / _FF / _GAIN` |
| `uav_avoiding_bridge/turbo.py` | new | ~220 | **Mode T** (§3.3): walks `logs/avoiding-d3il/plans/**/<variant>.npz` (filters: engine, eval-folder substring, seeds, geometries, variants), replays each episode through the plant, rescores on the drone path, writes the mirrored `_msg…turbo` npz + sidecars, prints a Panda-vs-drone agreement table; `--dry-run` lists what it would replay; idempotent (skips cells whose output exists unless `--force`) |
| `uav_avoiding_bridge/scoring.py` | new | ~80 | the avoiding halfspace / keep-out / bounds violation arithmetic, copied from the eval loop, so Mode T scores exactly as Mode L (reads `config/projection_eval.yaml`) |
| `uav_avoiding_bridge/README.md` | new | — | contract, knobs, how to add a scale |
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | hook | +4 | `env = make_avoiding_env()` in place of `ObstacleAvoidanceEnv()`; default path byte-identical |
| `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py` | hook | +4 | same |
| `FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py` | hook | +4 | same |
| `scripts/eval.py` (diffusion DPCC baseline) | hook | +4 | same |
| `FM_v3_hardflow_test/eval_FM_v3_hardflow.py` | hook | +4 | same (its run-tag goes through `hf_paths.resolve_run_msg`, also honoured) |
| `Slurm_Codes/temp_bash/eval_2026MMDD_u18_pillars_v2.sh` | new, gitignored | ~180 | plan-by-default driver: group **T** (turbo, CPU partition, no GPU) and the live groups below; copies the U17 driver's pre-flight pattern |
| `logs_in_develop/Gen15/U18/CHANGELOG_…md` | new | — | after coding |

Result folders: Mode L writes in the avoiding tree with the message tag —
`logs/avoiding-d3il/plans/<engine>/<train>/<eval>_msguavpv2s10/<seed>/results/halfspace_<geo>/<variant>.npz`;
Mode T (fix2) mirrors that layout under the UAV-pillars scene folder —
`logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/<engine>/<train>/<eval>_msguavpv2s10turbo/<seed>/results/halfspace_<geo>/<variant>.npz`.
The batch reporter and `avoiding_rules_by_protocol.py` select on `Folder_Name`, so a `pillars_v2` DA script is that
file with the suffix changed.

Nothing in `mix_uav/`, `mix_uav_test/`, `config/uav_projection.yaml` (old pillars stays as is, U17 banner stays),
`config/projection_eval.yaml`, the projector or any model folder is modified.

---

## 4 · Gates before any wave (the U16/U17 lesson: validate the *plant* and the *projector reach* first)

| gate | what | pass rule | cost |
| :-- | :-- | :-- | :-- |
| **G0** offline | `scene.py` XML compiles in MuJoCo; `to_world∘to_avoiding = id`; XML pillars = `T(centres)` | assert | seconds, CPU |
| **G1** turbo on one cell | `turbo.py` on one existing cell (FM K20 `dpcc-r-tightened`, both-hard, seed 6, 20 episodes). Compare per episode: success (finish crossed), `collision_free` recomputed on the *drone's* path, physical contact. | ≥ 95 % of episodes reproduce the Panda's `(success, collision_free)`; contact on 0 legal episodes; gap p95 < the G3 bound. Else: try `control_hz` 3, gain `pid_high_gain`, `settle` replay, then $s = 12$. | minutes, CPU, MuJoCo → cluster |
| **G2** one live cell | FM seed 6, K = 20, `diffuser` + `dpcc-r-tightened`, both-hard, 2 episodes, tag `uavpv2chk` | numbers within the Panda cell's spread; log shows plant stats | ~10 min |
| **G3** gap in distribution | the plant logs the setpoint-minus-position gap; the dataset's own gap range is read from the normaliser (`dataset.normalizer` mins/maxs of `x_des−x`, `y_des−y`) at G2 time | p95 of the drone's gap inside the training range | free (in G2 log) |

Only after G1–G3: the waves.

## 5 · Waves

**Order: T first (the whole grid, CPU), then the live subset (the closed-loop check), then more live only if the
subset disagrees with T.**

| group | mode | arm / script | cells | protocol | est. wall |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **T** | turbo | `turbo.py` over the entire `logs/avoiding-d3il/plans` corpus (FM K 1/2/5/10/20, MeanFlow, CI-MeanFlow, diffusion K20, HardFlow; DPCC protocol *and* the 20-episode extended cells) | every existing cell | as recorded | **< 1 h, CPU, no GPU** |
| **L1** | live | FM K20 + MeanFlow K2 (`dpcc-r/c/t-tightened` + `diffuser`), 3 geometries, seed 6, 2 ep | 2 × 4 × 3 | closed-loop check vs T | ~20 min |

Live full replica (only if L1 and T disagree beyond the seed spread — otherwise not run):

| group | arm / script | cells | protocol | est. wall |
| :-- | :-- | :-- | :-- | :-- |
| A | FM (`eval_fmv3_ode_job.sh`, `FMV3_FLOW_STEPS="1 2 20"`) | 3 K × 3 geo × 7 variants | 5 seeds × 2 ep (DPCC protocol) | ~1 h |
| B | MeanFlow U-Net (`MeanFlow/eval_meanflow.sh`, K = 1 2) | 2 × 3 × 7 | 5 × 2 | ~40 min |
| C | CI-MeanFlow U-Net ae0.2 (`AlphaFlow/eval_alphaflow.sh`, K = 1 2) | 2 × 3 × 7 | seed 6 × 2 (as in the thesis) | ~15 min |
| D | Diffusion DPCC K20 (`eval_dpcc_job.sh`) | 1 × 3 × 7 | 5 × 2 | ~1 h |
| E | HardFlow arms (`hardflow_fmv3/eval_fmv3_hardflow_job.sh`) | as the avoiding HardFlow table | as there | ~1 h |
| F (opt-in) | extended protocol, 20 episodes, FM + MeanFlow | | 5 × 20 | ~4 h |

Env for every live job: `FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=uavpv2s10` (plus `AF_NTRIALS`, K grids as today).
Mode T needs no env and no model checkpoint — only the npz files, which are on the cluster already.
Disk: the avoiding npz are small (no frames); no GIFs → no U17 disk problem. Time cap: every group is far
under 24 h; per-step planner cost is the avoiding one (K20 FM ≈ 0.55 s, K1/2 ≈ 0.02–0.1 s).

## 6 · DA and the thesis (after the waves; not before G2)

- `DA_in_Paper/analysis/pillars_v2_grid.py` = `avoiding_rules_by_protocol.py` with the `_msguavpv2s10turbo`
  suffix (and `_msguavpv2s10` for the live cells), printing **Panda vs drone side by side** per model / rule /
  budget (S&C, violating steps, steps) plus the plant columns from the sidecar (track_err, gap p95, contacts), and
  a T-vs-L agreement row for the live subset. The headline is the *difference*; the timing column is the avoiding one.
- One world-frame figure (drone paths over physical pillars + keep-outs) and the avoiding-frame path panels the
  eval already draws.
- v3 is **not** told anything until G2 passes; `pillars_hg` stays restored as the section of record until then.
  If U18 delivers, the pillars section becomes the cross-embodiment section (setup: the mapping in §2, one
  paragraph; results: the diff table) and `pillars_hg` is withdrawn again — that is a second reversal for v3, so
  it happens once, with data in hand.

## 7 · Risks and what is *not* being promised

1. **Gap distribution shift** (§2.4, G3) — the one real risk; if feed-forward + 5 Hz do not bring the gap inside
   the training range, the fallback is a "settle-to-tolerance" plant (run the PID until `‖p − p_des‖ < ε` before
   observing, variable time per step): exact transfer, but then the drone is a slow Panda and the run says nothing
   about real-time tracking. Decide at G1, not now.
2. **Turn transients** near row 2/3 can eat the 0.19–0.24 m allowance → physical contact on a legal plan. The
   sidecar separates *constraint* violations (avoiding frame) from *physical* contact, so this shows up as its own
   column, never silently as "S&C = 0".
3. **Open loop is open loop.** Mode T can only fail a plan, never rescue it; a lagging drone in `clock` mode
   never gets the planner's correction. The thesis says "the plans executed on the table were flown by the drone";
   Mode L on the subset is the evidence that this is not hiding anything.
4. **HardFlow** arms report `[NLP-FAILURE]` on some avoiding cells already; unchanged here.
5. **Naming**: `pillars` in the current draft means `pillars_hg`. Until the swap, every new file says `pillars_v2`.
6. This plan does **not** touch `MASTER_TEST_HISTORY.md` (offer only), does not commit, and writes no v3 note.

## 8 · Decisions needed before coding

| # | decision | proposal |
| :-- | :-- | :-- |
| 1 | scale $s$ | **10** (8 / 12 kept as one-line variants for G1) |
| 2 | control period / tracker | 5 Hz, `CascadedPID` `pid_default`, velocity feed-forward on |
| 3 | contact rule | first drone–pillar contact terminates the episode as a failure (mirrors the Panda rod rule) |
| 4 | which runs | **T over the whole corpus + live subset L1**; the live full replica A–F only if L1 disagrees with T |
| 4b | turbo replay policy | `clock` (0.2 s per setpoint) as the reported one; `settle` run alongside as the geometric bound |
| 5 | code location | new root folder `uav_avoiding_bridge/` + 5-line hooks in the five eval scripts (no forks of the evals) |
| 6 | run tags | live `uavpv2s10`, turbo `uavpv2s10turbo` (a tag must contain `uav`; the factory / turbo writer enforce it, and turbo never writes into a source folder) |

Approve (or amend) and I code it; the changelog lands in this folder when the code does.
