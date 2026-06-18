# U6 — Eval Call Chain: from Trajectory Generation to the Plot

**Scope:** the current iMF (U6, backbone `unet` *or* `dit`) on `avoiding-d3il`. Traces every function
hop from "ask the model for a trajectory" to "trajectory drawn in the saved figure," and shows exactly
where the **`diffuser`** variant (no projection) and the **`dpcc-r`** / projection variant (DPCC
post-processing) diverge.
**Entry script:** `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`.

---

## 1. The chain at a glance (file → function)

```
eval_flow_matching_v3_imeanflow.py   (main loops: exp → halfspace → seed → variant → trial → env-step)
│
├─ load_diffusion_with_override(...)                         # build the model from the checkpoint
│     └─ utils.load_config(... model_config.pkl / diffusion_config.pkl ...)
│           model     = iMeanFlowEngine(...)                 # backbone inside: UNet or IMFDiTTrajectory  (imf_engine.py)
│           diffusion = iMeanFlowODE(model, ...)             # the sampler wrapper            (imf_diffusion.py)
│           trainer.load(epoch)                              # load weights
│
├─ Projector(...)                                            # SLSQP/gradient projector       (sampling/projection.py)
│     projector = None  if  variant == 'diffuser'            # ◄── DIVERGENCE POINT (see §3)
│
├─ Policy(model=diffusion, projector=projector, trajectory_selection=...)   (sampling/policies.py)
│
└─ for each env step:
      action, samples = policy(conditions={0: obs}, batch_size, horizon, disable_projection)
      obs, ... = env.step(action)        # ObstacleAvoidanceEnv.step  (d3il avoiding.py)
      ...record metrics & buffers...
   then: ax.plot(obs_buffer x,y) + plot sampled trajectories + draw constraints → fig.savefig(variant.png)
```

---

## 2. The generation hop (Policy → DiT/UNet → projector)

```
Policy.__call__                                                     (policies.py:37)
│  normalize conditions {0: obs}  →  returns vector
│
└─ samples, infos = self.model(conditions, returns, projector, constraints, horizon)
        = iMeanFlowODE.forward                                      (imf_diffusion.py:478)
        └─ conditional_sample                                       (imf_diffusion.py:268)
           └─ p_sample_loop  ── the Euler flow noise→data ──        (imf_diffusion.py:177)
              │
              │   for i in range(flow_steps):          # NFE, e.g. 2
              │     ┌ _predict_velocity                              (imf_diffusion.py:142)
              │     │   └ _predict_uv → model.forward_train          (imf_diffusion.py:135)
              │     │        └ iMFTrajectoryModel.forward            (imf_trajectory_model.py:103)
              │     │             └ velocity_net(...)  ──►  u (,v)
              │     │                  • imf_backbone='dit' → IMFDiTTrajectory.forward  (imf_dit_trajectory.py)
              │     │                  • imf_backbone='unet'→ Flow_matcher_U_Net_v2.forward
              │     │   interval-CFG: u = u_uncond + ω·(u_cond−u_uncond)   (only if τ∈[t_min,t_max])
              │     └ x ← x + u·dt ;  x ← apply_conditioning(x)      # re-pin the start observation
              │
              │     ┌ if projector is not None and near_end:         # ◄── DPCC POST-PROCESSING (see §3)
              │     │    grad variant : x ← x + projector.compute_gradient(x, constraints)
              │     │    proj variant : x, cost ← projector.project(x, constraints)   # SLSQP snap
              │     └    costs[loop_idx] = cost
              │
              └ return x, infos                  # infos['projection_costs'], (optional) infos['diffusion']
```

Back in `Policy.__call__`:
```
trajectories = to_np(samples)                                       (policies.py:48)
observations = normalizer.unnormalize(traj[:, :, action_dim:])     # to world coords
which_trajectory = select(...)        # 'random' | 'temporal_consistency' | 'minimum_projection_cost'
action = actions[which_trajectory, 0]  # first action of the chosen trajectory
return action, Trajectories(actions, observations)
```

The eval loop sends `action` to `env.step(...)`, appends the resulting `obs` to `obs_buffer`, and (every
`horizon//2` steps) stashes `samples.observations` into `sampled_trajectories` for plotting.

---

## 3. Where `diffuser` vs `dpcc-r` (and friends) diverge

The **variant string** (from `config/projection_eval.yaml`) controls two switches set in the eval script
(`:221`, `:241–245`):

| Variant | `projector` | `trajectory_selection` | Effect in `p_sample_loop` |
|---|---|---|---|
| **`diffuser`** | **None** | random | **No projection.** Pure model rollout; trajectory = raw DiT/UNet output. Only variant that prints **Tracking error**. |
| **`dpcc-r`** | Projector (SLSQP) | **random** | DPCC **post-processing**: `projector.project` snaps `x` to the constraint manifold near the rollout end; pick a random batch member. |
| `dpcc-c` | Projector | **minimum_projection_cost** | Same projection, but pick the batch member with the **lowest** summed `infos['projection_costs']`. |
| `dpcc-t` | Projector | **temporal_consistency** | Same projection, but pick the member closest to last step's trajectory. |
| `*gradient*` | Projector (`gradient=True`) | per name | Soft push `x ← x + ∇` instead of a hard SLSQP snap. |

**"DPCC post-processing" = the projection block inside `p_sample_loop`.** It is *post-processing of the
velocity step*: after `x ← x + u·dt`, when `loop_idx ≥ (1−threshold)·flow_steps` (the tail of the
rollout), `projector.project(x, constraints)` solves
`min‖x−x_raw‖² s.t. obstacle/halfspace constraints` (SLSQP) and replaces `x` with the snapped result.
`diffuser` skips this entirely (`projector=None`), so its trajectory can cut through obstacles — which is
exactly what the metrics below measure.

> Backbone-agnostic: whether `u` came from the **DiT** or the **UNet**, the projector acts on `x` after
> the step and never inspects the network. Switching `imf_backbone` changes only the `velocity_net`
> hop in §2; the projection/selection/plot path is identical.

---

## 4. From buffers to metrics & plot (per variant)

After each trial's env loop ends, the eval script computes from `obs_buffer` / counters:

```
Success rate                         = mean(n_success)                       # env reported goal hit
Constraints satisfied                = mean(collision_free_completed)        # never entered an obstacle
Success (goal AND constraints)       = mean(n_success_and_constraints)
Avg #steps / #violations / total viol / comp-time                           # per-trial counters
Tracking error                       = max(pos_tracking_errors)              # printed ONLY for 'diffuser'
```
Violations are accumulated live each env step by testing `obs` against `obstacle_constraints` /
halfspaces (`:285–303`) — independent of the projector, so they fairly compare `diffuser` (unprojected)
vs `dpcc-*` (projected).

**Plotting** (`:340–366`), per trial `i`:
- `ax[i,0..3]` = x, y, x_des, y_des time series.
- `ax[i,4]` & `ax_all[i,variant]` = executed path `obs_buffer(x,y)` (black) + green start dot.
- `ax[i,5]` = the **sampled trajectories** stashed during the rollout (blue) — these show what the
  model proposed (and, for `dpcc-*`, what projection produced).
- `utils.plot_environment_constraints` + obstacle circles drawn on each path axis.

**Saved artifacts** (`:376–414`):
```
{savepath}/results/halfspace_{variant}/{variant}.png      # per-seed grid
{savepath}/.../{variant}.npz                              # n_success, violations, obs_all, ... (for --aggregate-only)
{savepath}/.../all.png                                    # all variants, this seed
{...}/all_seeds/{halfspace}/{variant}.png|.pdf            # overlay across seeds (one color per seed)
```

---

## 5. One-line summary per metric arm

- **`diffuser`** — model only (DiT/UNet u-field, no projector). Plot = raw few-step iMF trajectory;
  reports tracking error; expect more constraint violations.
- **`dpcc-r`** — same model, **+ DPCC SLSQP projection** snapped near the rollout end (post-processing),
  random pick of the batch. Plot = constraint-respecting trajectory; reports the violation/cost metrics
  that justify the projector.

> Everything above is identical for `imf_backbone='dit'` and `'unet'`; the only changed hop is
> `velocity_net.forward` in §2.

---

## 6. DEEP WALKTHROUGH — the `diffuser` metric, eval → plot, line by line

This follows **one full env step** of the `diffuser` variant (no projector) and what happens to the data
at every hop. Files: `E` = `eval_flow_matching_v3_imeanflow.py`, `P` = `sampling/policies.py`,
`D` = `models/imf_diffusion.py`, `T` = `models/imf_trajectory_model.py`, `N` = the backbone
(`imf_dit_trajectory.py` or `unet1d_temporal_cond.py`).

### Stage A — the variant makes `diffuser` projector-free  `E:221–245`
```
E:221  gradient = 'gradient' in variant            → False  (variant == 'diffuser')
E:239  projector = Projector(... )                 # the object is still constructed …
E:241  projector = None if variant == 'diffuser' else projector   ◄── for 'diffuser' it becomes None
E:242  trajectory_selection = 'random'             # no dpcc-t/dpcc-c override
E:245  policy = Policy(model=fm_model, ..., projector=None, trajectory_selection='random')
```
**Consequence carried through the whole step:** `projector=None` ⇒ the projection block in
`p_sample_loop` (`D:238`) never runs, and `infos['projection_costs']` stays `{}`.

### Stage B — reset env, build the observation vector  `E:264–279`
```
E:268  obs   = env.reset()                          # ObstacleAvoidanceEnv
E:269  action = env.robot_state()[:2]               # current TCP (x,y)
E:270  fixed_z = env.robot_state()[2:]              # frozen z/orientation for the step command
E:278  obs   = concatenate(action[:2], obs)         # prepend (x,y) → the conditioning observation
```
`obs` is now the raw world-space observation the planner conditions on.

### Stage C — ask the policy for an action  `E:304–305`
```
E:304  start = time.time()
E:305  action, samples = policy(conditions={0: obs}, batch_size, horizon, disable_projection=False)
```
`{0: obs}` means "pin trajectory timestep 0 to this observation."

### Stage D — Policy normalizes & calls the model  `P:37–46`
```
P:38   conditions = preprocess(conditions)
P:39   conditions = self._format_conditions(conditions, batch_size)     # P:97
P:99       normalizer.normalize(obs,'observations')   # world → normalized units
P:104      to_torch(...) ; einops.repeat 'd -> repeat d'  → [batch, obs_dim]
P:42   returns = test_ret * ones(batch,1)             # returns-conditioning vector
P:45   projector = self.projector if not disable_projection else None   # = None (Stage A)
P:46   samples, infos = self.model(conditions, returns=returns, projector=None, constraints=None, horizon)
```

### Stage E — the sampler: forward → conditional_sample → p_sample_loop  `D:478→269→178`
```
D:478  forward(cond, ...)          → conditional_sample(...)
D:269  conditional_sample          → builds shape (batch, horizon, transition_dim) → p_sample_loop
D:178  p_sample_loop(shape, cond, returns, projector=None, ...)
```

### Stage F — inside `p_sample_loop`, the few-step Euler rollout  `D:193–266`
```
D:193  x = randn(shape)                              # start = pure noise (τ=0)
D:194  x = apply_conditioning(x, cond, action_dim)   # pin obs at timestep 0
D:200  dt = 1/flow_steps                             # e.g. 1/2
D:205  cfg_on = meanflow_cfg_omega > 0               # interval-CFG armed (U5/U6 default ω=4)

for i in range(flow_steps):                          # D:222
   D:224  tau = i/flow_steps
   D:230  step_cfg = ω if (cfg_on and t_min ≤ tau ≤ t_max) else 0
   D:231  velocity = self._predict_velocity(x, cond, t_i, h, returns, omega,t_min,t_max, cfg_scale=step_cfg)
          └─ D:142 _predict_velocity → D:135 _predict_uv → T:103 iMFTrajectoryModel.forward
                                                       → N velocity_net.forward  ► u (DiT or UNet)
             (if step_cfg>0: u = u_uncond + ω·(u_cond − u_uncond))               # D:163–166
   D:235  x = x + velocity * dt                       # Euler step toward data
   D:236  x = apply_conditioning(x, cond, action_dim) # re-pin timestep 0

   D:238  if projector is not None:   ◄── SKIPPED ENTIRELY for 'diffuser' (projector is None)
          #  → no SLSQP snap, no gradient push, costs stay empty

D:262  infos = {}
D:265  infos['projection_costs'] = {}                 # empty (no projector ran)
D:266  return x, infos                                # x = final normalized trajectory [batch,H,Dtr]
```
So for `diffuser`, `x` is the **raw model trajectory** — whatever the DiT/UNet u-field integrated to, with
**zero constraint correction**.

### Stage G — Policy turns samples into a world-space action  `P:48–90`
```
P:48   trajectories = to_np(samples)                                  # [batch,H,Dtr]
P:51   'diffusion' not in infos  → True                               # return_diffusion was False
P:52     normed_obs = trajectories[:, :, action_dim:]                 # strip action dims
P:53     observations = normalizer.unnormalize(normed_obs)            # normalized → world (x,y,…)
P:68   trajectory_selection == 'random' → which_trajectory = 0        # pick batch member 0
P:82   actions = trajectories[:, :, :action_dim]
P:83   actions = normalizer.unnormalize(actions,'actions')
P:86   action = actions[0, 0]                                         # first action of chosen traj
P:88   return action, Trajectories(actions, observations)
```
`samples` returned to eval = the `Trajectories` namedtuple; `samples.observations` is `[batch,H,obs_dim]`
in world coords.

### Stage H — eval applies the action, records buffers & metrics  `E:305–333`
```
E:306  avg_time[i] += time.time() - start                            # per-step latency
E:308  next_pos_des = action + obs[:2]                               # action is a Δ on (x,y)
E:309  obs, rew, terminated, info = env.step(concat(next_pos_des, fixed_z, [0,1,0,0]))
E:310  success = info[1]
E:319  obs = concat(next_pos_des[:2], obs)                           # rebuild conditioning obs
E:320  pos_tracking_errors[i,_-1] = ‖obs(x,y) − desired_next_pos‖     # tracking error accumulation
E:322  desired_next_pos = samples.observations[0, 1, (x,y)]          # next planned waypoint (traj member 0, step 1)
E:323  if _ % (H//2)==0: sampled_trajectories.append(samples.observations[:, :, :])   # stash for the plot
E:325  obs_buffer.append(obs) ; action_buffer.append(action)
E:327  if success: n_success[i]=1
E:329  break when success/terminated/max steps  → n_steps[i], avg_time[i]/=_
```
Violations were checked at the **top** of the same loop (`E:285–303`) by testing `obs` against the
obstacle/halfspace constraints — this runs for `diffuser` too, so an unprojected path that clips an
obstacle is counted here.

### Stage I — plotting this trial  `E:339–388`
```
E:342  ax[i,0..3].plot( obs_buffer[:, x|y|x_des|y_des] )             # state time-series
E:346  ax[i,4] & ax_all[i,variant].plot( obs_buffer x, y , 'k')      # EXECUTED path (black)
E:347     + green 'Start' dot
E:351  axes_all_seeds[variant].plot(obs_buffer x,y, color=seed)      # cross-seed overlay
E:353  for stashed sampled_trajectories:                             # the model's PROPOSED rollouts
E:356     ax[i,5] & ax_all.plot( sampled_traj[member, :H, x], [.., y], 'b')   # blue proposals
E:362  utils.plot_environment_constraints(exp, ax) + obstacle circles  # draw the scene
E:367  print Success rate / Constraints satisfied / …
E:374  if variant=='diffuser': print('Tracking error', max(pos_tracking_errors))  ◄── diffuser-only line
E:377  np.savez('{...}/diffuser.npz', n_success, total_violations, obs_all, …)
E:388  fig.savefig('{...}/diffuser.png')
```
For `diffuser`, `ax[i,5]` (blue) and the projected path coincide because there is no projector — the
proposed and executed geometries are the model's own output. (For `dpcc-r`, the blue proposals would show
the **post-projection** snapped trajectories instead.)

### Stage J — cross-seed aggregate figure  `E:400–415`
```
E:401  path = '{savepath}/../all_seeds/{halfspace}'
E:403  for each variant's figs_all_seeds:                            # one figure per variant, all seeds overlaid
E:407     plot_environment_constraints + obstacle circles
E:413     fig.savefig('{path}/diffuser.png')  + '.pdf'
```

### Net: what makes `diffuser` special in this trace
1. `E:241` nulls the projector → the entire `D:238–257` projection block is dead code for this run.
2. `infos['projection_costs']` is `{}` → `minimum_projection_cost` selection can't apply (and isn't asked
   to; selection is `random`, `which_trajectory=0`).
3. `E:374` is the only place **Tracking error** is printed — it quantifies how well the executed path
   followed the model's *own* planned waypoint, with no projector cleaning it up.
4. The blue "sampled" curves in `ax[i,5]` are the unmodified DiT/UNet output; in `dpcc-*` the same axis
   shows the projected (constraint-satisfying) version, which is the visual difference you see between
   the two metrics' plots.

---

### 6.7 The Python mechanism behind each "jump" (how control crosses each boundary)

Each hop in the stages above is a specific Python control-transfer mechanism — not magic. The flowcharts
below retrace the **same `diffuser` step**, but the annotation **between each pair of boxes names *how*
Python jumps there**, so you can see exactly what kind of call crosses each file/object boundary.

#### 6.7a — Build phase (config string → live objects)

```
 config dict:  'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE'   (a STRING)
        │
        │  ⟶ import_class(_class)                         diffuser/utils/config.py:6
        │     mechanism: dotted-string DYNAMIC IMPORT
        │     importlib.import_module(repo.module) → getattr(module, 'iMeanFlowODE')
        ▼
 the real class object  iMeanFlowODE
        │
        │  ⟶ diffusion_config(model)                      Config.__call__  config.py:92
        │     mechanism: CALLABLE INSTANCE (__call__ on a Config)
        │     runs  self._class(model, **self._dict)  → constructs the nn.Module
        ▼
 live diffusion object  (iMeanFlowODE instance, with DiT or UNet inside)
```

#### 6.7b — Run phase (one `diffuser` env step — who calls whom, and via what)

```
 eval loop:  action, samples = policy(conditions={0: obs}, ...)        E:305
        │
        │  ⟶ mechanism: CALLABLE OBJECT  →  Policy.__call__            P:37
        │     (Policy defines __call__, so "policy(...)" runs that method)
        ▼
 Policy.__call__:  samples, infos = self.model(conditions, ..., projector=None)   P:46
        │
        │  ⟶ mechanism: nn.Module.__call__  →  forward                 D:478
        │     (self.model is an nn.Module; calling the instance fires PyTorch's
        │      __call__ which dispatches to YOUR forward)
        ▼
 iMeanFlowODE.forward → conditional_sample → p_sample_loop            D:478→269→178
        │
        │  ⟶ mechanism: BOUND-METHOD calls (self.xxx())  — same object, internal delegation
        ▼
 p_sample_loop, per Euler step:  velocity = self._predict_velocity(...)   D:231
        │
        │  ⟶ mechanism: BOUND METHOD  →  _predict_velocity → _predict_uv   D:142→135
        ▼
 _predict_uv:  self.model.forward_train(...)                          D:137
        │
        │  ⟶ mechanism: BOUND METHOD on a DIFFERENT object
        │     (here self.model is the iMeanFlowEngine — attribute access hops object)
        ▼
 iMeanFlowEngine.forward_train:  self.model(...)                      (engine)
        │
        │  ⟶ mechanism: nn.Module.__call__  →  forward                 T:103
        │     (self.model is the iMFTrajectoryModel)
        ▼
 iMFTrajectoryModel.forward:  self.velocity_net(...)                  T:117
        │
        │  ⟶ mechanism: nn.Module.__call__  →  forward  +  RUNTIME POLYMORPHISM
        │     self.velocity_net was bound at __init__ to ONE of:
        │        • IMFDiTTrajectory.forward         (imf_dit_trajectory.py)   ← imf_backbone='dit'
        │        • Flow_matcher_U_Net_v2.forward    (unet1d_temporal_cond.py) ← imf_backbone='unet'
        │     SAME call site, different concrete forward  ── the U5/U6 swap point
        ▼
 returns u  →  (Euler) x = x + u·dt                                   D:235
        │
        │  ⟶ mechanism: DUCK-TYPED OPTIONAL COLLABORATOR
        │     if projector is not None:   ← for 'diffuser' projector IS None  → block skipped  D:238
        ▼
 p_sample_loop returns (x, infos)   →   back up the SAME bound-method stack to Policy   D:266
```

#### 6.7c — Back in Policy → eval → plot (the remaining jumps)

```
 Policy: observations = self.normalizer.unnormalize(...)              P:53   (bound method, plain object)
         which_trajectory = 0   # 'random'                            P:68
         return action, Trajectories(actions, observations)           P:88
        │
        │  ⟶ mechanism: NAMEDTUPLE construction
        │     Trajectories = namedtuple('Trajectories','actions observations')   P:11
        ▼
 eval:  desired_next_pos = samples.observations[0, 1, (x,y)]          E:322
        │  ⟶ mechanism: NAMEDTUPLE FIELD ACCESS  (.observations = field 1 by name)
        │
        │  ⟶ env.step(...)            E:309   mechanism: bound method on the gym env (ObstacleAvoidanceEnv)
        │  ⟶ getattr(args,'flow_steps_v3',…)  E:139  mechanism: dynamic attr lookup w/ default (plan override)
        ▼
 plot:  ax.plot(obs_buffer x,y,'k')   /   fig.savefig('diffuser.png')   E:346 / E:388
           ⟶ mechanism: bound methods on Matplotlib Axes / Figure objects
```

#### The two patterns worth remembering
- **`__call__` is the workhorse.** Every "I called the thing like a function" hop — `Config(...)`,
  `policy(...)`, `self.model(...)`, `velocity_net(...)` — is an object implementing `__call__`. For
  `nn.Module` subclasses (`iMeanFlowODE`, `iMeanFlowEngine`, `iMFTrajectoryModel`, the DiT/UNet) that
  `__call__` is PyTorch's, which then calls **your** `forward`. So "calling the module" always means
  "run its `forward`."
- **The backbone swap is one polymorphic call site.** `self.velocity_net(...)` in `T:117` is *textually
  fixed*, but the object it resolves to (`IMFDiTTrajectory` vs the UNet) was chosen at construction by
  `imf_backbone`. Nothing else in the chain knows or cares which backbone ran — that is the whole point
  of the U5/U6 contract.
