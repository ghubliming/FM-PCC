# One-Shot Generation (K=1 DDPM / ODE-1 FM / iMF) vs. Horizon, Across Domains

**Date**: 2026-06-01
**Companion**: [`horizon_and_ode_step_relationship.md`](horizon_and_ode_step_relationship.md) —
the strict math for `H` vs `N` in the FM-PCC framework. This note is a
broader **conceptual** discussion that compares how the same DGM machinery
maps onto image generation (where iMF was born) vs. trajectory
generation (what we do).

---

## TL;DR — the four claims this note defends

1. **`H` (horizon) and `N`/`NFE` (diffusion integration steps) are
   independent axes.** Increasing one does not change the other.
   One-shot (NFE=1) is about collapsing diffusion time, not the
   trajectory's internal time.

2. **Image-domain iMF has no horizon concept at all.** "One-shot" =
   "noise → full image in one model evaluation." The image's spatial
   shape is not analogous to our trajectory's H.

3. **Our DPCC/FM at H=8 with iMF is correct.** H is an MPC design
   parameter; it's orthogonal to whether the DGM is one-shot.

4. **D3IL aligning vision uses a HETEROGENEOUS set of agents** with
   different per-call output sizes:
   - MLP single-step agents: H=1 (replans every env step, ~300 model
     calls per episode)
   - Encoder-decoder chunked: H=4 or 8
   - "No horizon" in D3IL means **H=1 (max replan)**, NOT
     "open-loop full trajectory" (which doesn't exist in D3IL).

---

## Common misconceptions (FAQ)

> **"If a D3IL agent has no horizon, it predicts the whole 300-step
> episode in one shot, right?"**

**Wrong.** "No horizon concept" in this note means **H = 1 (single-step
predictor)**, not "H = episode length." A single-step MLP agent
predicts ONE action per model call, then executes one env step,
observes, predicts the next one action, etc. It calls the model ~300
times per 300-step episode — the OPPOSITE of one-shot full-trajectory.

> **"Then is `bc_vision` doing pure open-loop?"**

**No, it's the opposite — pure closed-loop reactive control.** Every
single env step, it re-observes and re-predicts. Maximum reactivity to
disturbances; no look-ahead beyond a single action.

> **"How can a single-step MLP solve aligning then? Aligning is
> multi-step."**

By being a pure reactive policy: `state → action` mapping, applied
repeatedly. The training dataset has many `(state, action)` pairs
across many demonstrations. The MLP learns a function approximator of
the demonstrators' policy. At deployment it executes that policy step
by step. This is the simplest form of behavioral cloning and is what
the `bc_vision_agent` is.

> **"Why does D3IL ship MLP single-step agents alongside chunked
> sequence agents? Aren't they obsolete?"**

D3IL is a **benchmark** — it intentionally evaluates many architectural
paradigms on the same task. The MLP agents are baselines; the
transformers/chunked DDPM/etc are comparison points. They're not meant
to all be the "best" — they're meant to characterize what each design
choice buys you.

> **"Is our DPCC/FM doing 'one-shot' generation?"**

Depends on which axis:
- **Diffusion-time one-shot (NFE=1)?** No, our default is K=100 (DPCC)
  or several Euler steps (FM). Only the K=1 / ODE=1 experiments are
  diffusion-time-one-shot.
- **Trajectory-time one-shot (full episode in one call)?** No. We
  generate an 8-step chunk per call, execute 4 of those steps, then
  replan. ~38 model calls per 300-step episode.

> **"Does iMF in the paper / on image generation have an H?"**

No. The image data has spatial dimensions but no "time to execute" —
the image just IS the output. iMF's one-shot promise is purely about
the diffusion-time axis.

> **"So when we apply iMF to trajectories at H=8, are we abusing iMF?"**

No. iMF's velocity field doesn't care what shape the data is — it
works on any tensor. The model's 1D temporal U-Net processes the H
axis with convolutions; the diffusion-time axis is what iMF's one-shot
claim addresses.

---

## 1. There are TWO orthogonal time axes — they're often conflated

In every diffusion / flow-matching pipeline that operates on
**trajectories**, two completely different "times" coexist:

| Axis | Symbol | Range | Meaning |
|---|---|---|---|
| **Diffusion / flow time** | `t` (or `τ`) | continuous in `[0, 1]` | the integration variable of the probability flow ODE. `t=0` = pure noise, `t=1` = data. NFE / `K` / `ODE-steps` discretize this axis. |
| **Trajectory horizon time** | `h_idx` (or `k`) | discrete in `[0, H)` | the index into the produced trajectory tensor of shape `(B, H, D)`. **This is "real-world time"** — the timesteps the robot executes. |

These have nothing to do with each other:

- Increasing `H` makes the model produce a LONGER chunk per inference call. Costs more memory; same number of diffusion-time integration steps.
- Increasing `N` (NFE) makes diffusion integration finer. Costs more model evaluations; same chunk length produced.

K=1 / ODE=1 / NFE=1 / iMF one-shot all refer to **collapsing the diffusion-time axis to a single step.** They have nothing to do with `H`.

A picture:

```
                  noise                                   data
diffusion time:   t=0  ──────────────────────────────────── t=1
                       (multi-step DDPM does N=100 here)
                       (FM / iMF one-shot does N=1 here)

                                                            │
                                                            ▼
                                                  ┌─────────────────┐
                                                  │ trajectory chunk│
                                                  │  shape (H, D)   │
                                                  │                 │
                                                  │   h_idx=0       │  ← step 0 of real-world execution
                                                  │   h_idx=1       │  ← step 1
                                                  │     ...         │
                                                  │   h_idx=7       │  ← step 7 (H=8)
                                                  └─────────────────┘
```

The full task is then executed by chaining many such (H, D) chunks
together in an MPC loop (next section).

---

## 2. How the produced (H, D) chunk gets used — the MPC contract

We don't generate the FULL task trajectory in one inference call (a full
aligning episode is 300+ steps; our H=8). Instead:

```
loop until done:
    obs = env.observe()
    chunk = model.sample(cond=obs)          # produces (H, D) tensor
    for i in 0..action_seq_size:
        env.step(chunk[i, :action_dim])      # execute first few steps
    # discard chunk[action_seq_size:, :]; re-plan next iteration
```

So the **planning horizon** `H` is just "how far ahead we plan each
replan call." We execute a smaller `action_seq_size` (typically the
first 4 of 8) before re-planning. This is **Model Predictive Control
(MPC)** — short-horizon forward generation with frequent replanning.

The opposite extreme would be: train with `H = full_episode_length`
(e.g. 300), generate the entire episode once, execute it open-loop. No
replanning. This is feasible but brittle — any state drift over 300
steps compounds. MPC with H=8 + replanning is the standard robotics
choice.

**`H` is fundamentally an MPC-design parameter, not a DGM parameter.**

---

## 3. Does D3IL visual aligning use H=8? (Corrected — depends on the agent)

**My earlier "horizon=4" claim was WRONG as a generalization.** D3IL
ships ~11 vision agents for aligning, and they fall into two
fundamentally different groups regarding horizon/chunking. The
top-level config `aligning_vision_config.yaml` sets `window_size: 8`,
but **this is a DATASET window, not a model output dimension** — how
each agent consumes it varies.

### 3.1 D3IL dataset side

The dataset (`Aligning_Img_Dataset`) returns per sample:
- `obs`: shape `(window_size, obs_dim)` — 8 consecutive timesteps of obs
- `act`: shape `(window_size, action_dim)` — 8 consecutive timesteps of action
- `mask`: shape `(window_size,)`

`window_size=8` means "load 8 consecutive timesteps per training sample."
Whether the MODEL uses all 8, only the last 1, or a 5+3 split is
agent-specific.

### 3.2 D3IL agent landscape (verified by `head` on each yaml)

| Agent | Backbone | Output per inference | Horizon / chunking? |
|---|---|---|---|
| `bc_vision_agent` | `ResidualMLPNetwork`, `output_dim=action_dim` | **single action** | ❌ No |
| `bet_mlp_vision_agent` | MLP head with vocab + offsets | **single action** (categorical+offset) | ❌ No |
| `ddpm_vision_agent` (non-encdec) | `DiffusionMLPNetwork`, t_dim=4 | **single action** (diffuses one action) | ❌ No |
| `ibc_vision_agent` | Implicit BC score model | **single action** (via energy sampling) | ❌ No |
| `cvae_vision_agent` | conditional VAE | **single action** (typical) | ❌ No |
| `ddpm_encdec_vision_agent` | DiffusionEncDec | **sequence**: `action_seq_size=4`, conditioned on `obs_seq_len=5` | ✅ Chunked (H=4) |
| `act_vision_agent` (Action Chunking Transformer) | Transformer | **sequence**: `action_seq_size = window_size = 8` | ✅ Chunked (H=8) |
| `ddpm_transformer_vision_agent` | Transformer DDPM | **sequence** (window_size) | ✅ Chunked |
| `gpt_vision_agent` | GPT-style transformer | **sequence** | ✅ Chunked |
| `bet_vision_agent` | BeT transformer | **sequence** | ✅ Chunked |
| `beso_vision_agent` | BESO score model | varies (sequence in typical config) | ✅ Chunked |

### 3.3 What this means — "no horizon" ≠ "open-loop full trajectory"

**Critical clarification (easy to misread):** when MLP-based agents are
described as having "no horizon concept," it does NOT mean they
generate the entire 300-step task trajectory in one shot. **It means
the OPPOSITE — they have horizon = 1, replanning every single step.**

The rollout loop for a single-step MLP agent (e.g., `bc_vision`) is:

```python
for step in range(max_episode_length):    # ~300 steps for aligning
    obs = env.get_obs()                   # fresh observation
    action = agent.predict(obs)           # ONE action only
    env.step(action)                      # execute one step
    # next iteration → fresh obs → next predict() → next action
```

So these agents call the model **~300 times per episode** — the most
aggressive MPC possible (replan every step). The `window_size=8` from
the dataset config provides observation CONTEXT to the encoder/MLP
(past obs may be aggregated, or just the last obs used — depends on
the agent), but the OUTPUT is always one action.

D3IL is heterogeneous by design — it benchmarks many architectural
paradigms on the same task:

- **MLP-based agents** (bc, bet_mlp, ddpm non-encdec, ibc, cvae) are
  **single-step predictors with horizon=1**. They MPC-replan every
  environment step. ~300 model calls per episode.
- **Sequence-based agents** (ddpm_encdec, act, ddpm_transformer, gpt,
  bet, beso) output **action chunks**. ACT outputs chunk=8,
  ddpm_encdec outputs `action_seq_size=4`. They MPC-replan every
  chunk-prefix (~38 to ~75 model calls per episode).

### 3.3.0 Horizon ≠ Episode length — distinct quantities

| Quantity | Definition | Set by | Independent of |
|---|---|---|---|
| **Horizon `H`** | actions per single inference call | model architecture | task length |
| **Episode length** | env steps per full task | task / env config | model architecture |
| **Replans per episode** | `≈ episode / (H · execute_fraction)` | derived from the two above | — |

For aligning task (~300 env steps per episode):

| Agent | H | episode | model calls per episode |
|---|---|---|---|
| `bc_vision` (MLP) | 1 | 300 | ~300 |
| `ddpm_encdec_vision` | 4 | 300 | ~75 |
| `act_vision`; **ours DPCC/FM** | 8 | 300 | ~38 |
| (Hypothetical open-loop) | 300 | 300 | 1 |

**A single-step MLP agent has H=1, NOT H=episode.** It just gets called
300 times.

### 3.3.1 The horizon-vs-replan spectrum

All these styles live on one axis: **how many steps per inference
call.** Different points on this axis trade off different things:

| Style | Output per inference | Model calls per episode (≈300) | Example | Trade-off |
|---|---|---|---|---|
| Single-step (max MPC) | 1 action | ~300 | bc, ddpm, bet_mlp, ibc, cvae | Maximum reactivity to disturbances. High inference cost. No multi-step coherence. |
| Small-chunk MPC | 4 actions | ~75 | ddpm_encdec_vision (chunk=4) | Modest look-ahead, balance reactivity/cost. |
| Medium-chunk MPC (ours) | 8 actions | ~38 | Ours (H=8, replan every 4); ACT (chunk=8) | More coherent multi-step planning. Slightly less reactive. |
| Full open-loop | full episode (~300 actions) | 1 | (hypothetical — not in D3IL) | Maximum coherence but brittle to disturbances; one model call per episode. |

D3IL covers the three left columns with different agents. Our DPCC/FM
sits at the third column. The fourth column (full open-loop) is rarely
seen in robotics for the brittleness reason.

**None of these styles uses "one-shot full-trajectory from start to
end."** Single-step MLP is the OPPOSITE of that — it's the maximum
amount of replanning possible.

| Comparison axis | DPCC/FM (us) | D3IL MLP family | D3IL sequence family |
|---|---|---|---|
| Trajectory tensor produced per inference | (H=8, D=9) — joint action+state | None (just one action) | (action_seq_size, action_dim) — actions only |
| Horizon / chunk concept | 8 (joint) | 1 (degenerate) | 4 to 8 depending on agent |
| MPC replan frequency | every 4 steps (action_seq_size of our policy.py loop) | every step | every chunk-prefix |
| Closest D3IL analog architecturally | **`act_vision_agent`** (chunk = 8, sequence transformer) and **`ddpm_encdec_vision_agent`** (encdec) | — | — |

### 3.3.2 Rollout loops side-by-side — concrete pseudocode

**(A) Single-step MLP** (D3IL `bc_vision`, `ddpm_vision`, `bet_mlp_vision`, ...):

```python
obs_history = deque(maxlen=window_size)         # 8 past obs for context
for step in range(episode_length):              # ~300 iterations
    obs_history.append(env.observe())
    feats = obs_encoder(obs_history)            # window_size=8 → 128-D latent
    action = mlp_head(feats)                    # → SINGLE action, action_dim
    env.step(action)
# Total: ~300 model calls. H=1. Replans every env step.
```

The `window_size=8` is consumed by the obs encoder as past context.
Output is one action.

**(B) Chunked sequence agent** (D3IL `ddpm_encdec_vision`, `act_vision`, ours):

```python
obs_history = deque(maxlen=obs_seq_len)         # 5 past obs (D3IL), 1 (ours)
step = 0
while step < episode_length:                    # ~300
    obs_seq = list(obs_history)
    action_chunk = model.sample(obs_seq)        # → (H, action_dim) sequence
                                                 #    H=4 (ddpm_encdec) or 8 (ACT, us)
    for k in range(execute_count):              # execute_count = 4 (ours), 4 (ddpm_encdec),
                                                 # or H itself (ACT, "execute the full chunk")
        env.step(action_chunk[k])
        obs_history.append(env.observe())
        step += 1
# Total: ~38 to ~75 model calls. H>1. Replans every execute_count steps.
```

The chunk's first `execute_count` actions are executed, the rest are
discarded, then the model re-plans from the new state.

**(C) Hypothetical open-loop** (not used by anyone in D3IL):

```python
obs0 = env.observe()
full_traj = model.sample(obs0, H=episode_length)   # (300, action_dim) in one shot
for k in range(episode_length):
    env.step(full_traj[k])
# Total: 1 model call. H = episode_length. No replanning.
```

This requires training a model with `H = episode_length` and is
brittle to disturbances over the full 300-step rollout. Nobody does
this for aligning.

**Mapping to our DPCC/FM:** style (B) with `obs_seq_len=1` (just the
current obs anchor pinned at trajectory step 0), `H=8`, `execute_count=4`
(so 4-of-8 → replan every 4 env steps).

### 3.4 Closest comparable D3IL agent to ours

`act_vision_agent` (Action Chunking Transformer): outputs `(8, action_dim)`
per inference, matches our window. Difference: ACT predicts ONLY actions
(action_dim), we predict joint (action + state) per timestep (9D or
23D). The "chunk size" is the same; the channel content differs.

`ddpm_encdec_vision_agent` is similar but with `action_seq_size=4`
(half our window). And it uses `obs_seq_len=5` of observation context
where we use a single obs anchor at step 0.

### 3.5 Bottom line correction

When discussing "horizon" across the D3IL agent zoo, **specify which
agent**. The bare statement "D3IL uses horizon X" is misleading because
~half of D3IL's agents have no horizon concept (they're single-step
MLPs).

Our DPCC/FM at H=8 is closest to D3IL's `act_vision_agent`
architecturally on the horizon axis (both predict 8-step chunks),
though the model class and what's predicted differ.

---

## 3.6 Why does no one actually ship H = episode_length?

If H and N are independent, and the "open-loop full-episode" cell of
the matrix in §2 is mathematically valid, why doesn't D3IL (or anyone)
include it? Six concrete reasons:

### 3.6.1 Compounding state drift

Open-loop execution of a 300-step trajectory means: any small error in
the predicted action at step 5 propagates a small error into the
actual state at step 6. The model predicted step 6's action assuming
step 6's state would match the demonstrator's. It won't. Now step 6's
executed action is computed on a hypothesis that's already false.

With **MPC + small H**: at step 6, we look at the ACTUAL observed state
and re-plan. Error from step 5 is absorbed.

With **open-loop H=300**: error compounds over all 300 steps with no
correction. After ~50 steps, the model is essentially predicting
actions for a state that no longer exists in the data manifold.

### 3.6.2 Out-of-distribution latents at long horizons

Training data has `(initial_state, demonstrator_trajectory)` pairs. The
demonstrator's trajectory was generated by a *reactive* controller
(human, MPC, etc.) — at every step the demonstrator saw the current
state and acted. Any single demonstration is one path consistent with
that policy.

A model trained to "predict the full 300-step trajectory given initial
state" is being asked to predict the demonstrator's reactive behavior
without observing any of the reactive feedback. It can only output the
mean trajectory of all demonstrations consistent with the initial
state — which is generally NOT a valid execution path because real
demonstrations branch into multiple modes after a few steps.

### 3.6.3 Multi-modality collapse

Aligning has multiple expert modes: push box A first or box B first;
go around the left or right; etc. Single-step BC at H=1 can replicate
multi-modal demonstrations because at every step there are multiple
valid actions and the dataset has examples of each. With H=episode,
the model must commit to ONE trajectory at step 0, but the data has
many. The MSE loss averages them → "mode collapse" → a single
unrealistic average trajectory that doesn't accomplish any of the
modes.

(Diffusion models partially mitigate this via stochastic sampling, but
the brittleness reasons above still bite even for diffusion at long H.)

### 3.6.4 Architecture constraints (the small reason)

Our temporal U-Net halves the H axis 3 times (`dim_mults=(1,2,4,8)`),
so H must be a multiple of 8. For H=episode≈300, you'd round up to
H=320 (next multiple of 8 ≥ 300). Tensors and memory grow linearly,
which is doable but wasteful — and doesn't solve §3.6.1–3.6.3.

Other architectures (transformers) don't have this constraint but
suffer the same fundamental issues.

### 3.6.5 The robotics tradition

Robotics has decades of empirical evidence that **closed-loop control
beats open-loop control whenever there's any uncertainty**. Even
classical motion planners (RRT, A*, etc.) typically pair with
trajectory tracking and replanning. The community defaults to MPC for
this reason. Imitation learning inherits the assumption.

### 3.6.6 Some methods DO push H toward episode-length — with nuance

A class of methods DOES train and generate at long H:

- **Diffuser** (Janner et al. 2022): diffusion on full-episode
  trajectories. H up to 256 on D4RL Maze2D, H up to 100 on Hopper.
  Evaluated **both** open-loop and with MPC-style replanning;
  replanning consistently wins by a measurable margin.
- **Decision Diffuser** (Ajay et al. 2023): conditional diffusion over
  trajectory sequences with returns/goals.
- **Decision Transformer** (Chen et al. 2021), **Trajectory Transformer**
  (Janner et al. 2021): autoregressive over full trajectory tokens.
  Can run open-loop or autoregressively step-by-step.
- **AdaptDiffuser, PlanDiffuser**, etc.: variants that push the
  trajectory diffusion paradigm further.

So **the H=episode regime is not empty** — it's a research direction
with active interest. But there are caveats:

1. **The published numbers prefer MPC.** Even Diffuser's own paper
   reports better results with replanning than with pure open-loop.
   Open-loop is treated as an ablation or simplicity baseline, not
   the recommended use.

2. **The "long H" is often a training/architecture choice, not an
   inference choice.** The model is trained on long sequences so it
   learns coherent multi-step planning, but at inference time many
   of these systems still replan periodically.

3. **Tasks differ.** Long-H diffusion works better on:
   - Smooth, low-disturbance environments (Maze2D, simulated locomotion).
   - Tasks where the demonstrator trajectory is itself relatively
     deterministic given initial state.

   It works less well on:
   - High-contact, high-uncertainty tasks like manipulation (which is
     why D3IL — a contact-rich aligning task — doesn't ship any
     pure-open-loop variant).
   - Multi-modal tasks where mode collapse over long horizons is severe.

4. **D3IL specifically.** Across all 11 D3IL vision agents, NONE is
   trained or evaluated at H = episode_length. The benchmark
   deliberately covers H=1 (MLP/BC/IBC/CVAE), H=4 (ddpm_encdec),
   H=8 (ACT, transformer DDPM), but not H=300. That's a reasoned
   choice by the D3IL authors based on what the field knows works
   for manipulation.

**Bottom line:** H = episode_length is **not absent from the
literature**, but it's also **not the dominant paradigm** for tasks
like ours. The community has converged on small-H + MPC for
manipulation specifically. Aligning the H to a "manageable" 4-8 range
is where the empirical winners live.

The closest things — open-loop trajectory optimization methods (DDP,
iLQR, MPPI) — all assume access to a dynamics model and do their own
internal "closed-loop" via optimization. They aren't pure imitation
learning, so they don't really live on this spectrum.

#### Summary table of long-H literature

| Method | Max H reported | Open-loop vs MPC at inference (in their own paper) |
|---|---|---|
| Diffuser (Janner 2022) | 256 (Maze2D), 100 (Hopper) | MPC-replan wins by measurable margin |
| Decision Diffuser (Ajay 2023) | full episode | Replanning preferred |
| Decision Transformer (Chen 2021) | full trajectory autoregressively | Step-by-step at inference (effectively H=1 at execution time) |
| Trajectory Transformer (Janner 2021) | full trajectory tokens | Same |
| AdaptDiffuser, PlanDiffuser, ... | long | Same pattern |
| **D3IL aligning (any agent)** | **8 max** | **Pure open-loop not shipped — manipulation-specific judgment** |
| **Our DPCC/FM** | 8 | MPC (replan every 4 of 8 executed) |

### 3.6.7 Three qualifications when reading the long-H literature

1. **Training long, executing short.** "Trained at H=256" often means
   the model SEES long sequences during training so it learns coherent
   multi-step plans. At inference, the same model is frequently run
   with MPC replanning. The "long H" is an architectural/data choice,
   not an inference recommendation.

2. **Task type drives the right H.** Long-H diffusion works better on:
   - Smooth, low-disturbance environments (Maze2D, simulated locomotion).
   - Tasks where the demonstrator's trajectory is itself relatively
     deterministic given the initial state.

   It works less well on:
   - High-contact, high-uncertainty manipulation (D3IL aligning,
     pushing, stacking).
   - Multi-modal tasks where mode collapse over long horizons is
     severe.

3. **No "production" pure-open-loop full-episode IL system exists**
   that we know of, for manipulation. The closest are research
   experiments inside diffusion-planning papers, where open-loop is
   typically shown as a baseline that the recommended (MPC) inference
   beats.

### 3.6.8 Why D3IL specifically draws the line at H=8

D3IL's H-coverage across its agent zoo (re-stated):

| H | Agents |
|---|---|
| 1 | `bc_vision`, `bet_mlp_vision`, `ddpm_vision`, `ibc_vision`, `cvae_vision` |
| 4 | `ddpm_encdec_vision` |
| 8 | `act_vision`, `ddpm_transformer_vision`, `gpt_vision`, `bet_vision` |
| episode (≈300) | — (nothing) |

The top row at H=episode is **conspicuously empty**. Two reasons,
both visible by reading D3IL's task choices:

1. **D3IL focuses on contact-rich manipulation** (aligning, pushing,
   sorting, stacking, avoiding). These are exactly the tasks where
   §3.6.1–3.6.3 issues bite hardest. The D3IL authors implicitly
   judged that long-H open-loop won't help here.
2. **Multi-modal demonstrations are a core D3IL theme.** Long-H
   open-loop is especially bad at multi-modality (§3.6.3). D3IL would
   show such a method underperforming severely; including it would
   confuse the benchmark.

So D3IL's omission of H=episode isn't an oversight — it's a reasoned
benchmark design choice. The right read of "D3IL has no H=episode
agent" is: *the manipulation-IL community has empirically converged
that this regime doesn't work well for these tasks*.

### 3.6.9 Bottom line for OUR DPCC/FM choices

Two paths if we ever consider pushing H beyond 8:

| Path | What | Risk |
|---|---|---|
| **Conservative** — stay MPC, increase H to 16 or 24 | Slightly more look-ahead per replan; same closed-loop reactivity | Low. Just need U-Net depth or `dim_mults` compatible with new H. Modest gain expected. |
| **Aggressive** — train at H=episode (320 padded), execute open-loop | Pure-imitation full-trajectory generation; no replan | High. Hits §3.6.1–3.6.3 head-on. Would be a research experiment, not a working policy. Mode collapse + state drift make this unlikely to succeed on aligning. |

Most of the field stays in **small-H + MPC**. That's the conservative,
empirically-validated path. **What we already do.** Our H=8 +
replan-every-4 is a sensible point in the design space; pushing H
slightly higher is safe-to-explore, pushing all the way to
episode-length is essentially writing a research paper.

### 3.6.10 Mapping to our framework

Our DPCC/FM at H=8 sits squarely in the MPC paradigm. We deliberately
chose H small enough to fit the U-Net's stride-2 constraints (H=8) but
large enough to give the policy some look-ahead (8 > 1). The replan
frequency (every 4 steps) was tuned to balance plan coherence and
reactivity.

We could push H up to 16 or 24 (next valid multiples of 8) for richer
multi-step planning — but going much further (H=80, H=160, ...) hits
all the issues in §3.6.1–3.6.3 unless paired with MPC replanning
(which defeats the purpose of going longer). The conservative path in
§3.6.9 covers this; the aggressive path is the research-only territory.

---

## 4. iMF in its native image domain — does it have a horizon?

**No.** iMF (and DDPM, FM, etc.) was developed for **image generation**,
where the "data" is a single 2D image of shape `(C, H_img, W_img)`.
There's only one integration: from noise to a full image.

```
diffusion time:  t=0  ──[N=1 or N=many]──→  t=1
                 ↓                           ↓
              noise pixels              data image (C, H_img, W_img)
```

The 2D image dimensions `(H_img, W_img)` are **NOT** an MPC-style time
axis. They're spatial. The model treats the entire image as the
"thing to generate," not as a sequence to execute step-by-step.

In image-domain iMF / DDPM / FM:
- `NFE = 1` means "one model evaluation to produce the entire image."
- The image is "the full task." There's no replanning, no MPC loop,
  no execution.
- The image's spatial dimensions are processed by the U-Net's 2D
  convolutions, the same way our trajectory's `H` axis is processed
  by our 1D temporal convolutions.

So when the iMF paper says "one-shot generation," it means **one
diffusion-time forward pass producing the full output object.** In
image-land the output is the full image. In our trajectory-land the
output is one `(H, D)` chunk — which is **NOT** the full task, it's
one MPC step's worth of plan.

---

## 5. Is our H=8 wrong when we apply iMF?

**No, it's correct.** Here's why this confusion sometimes arises:

iMF's NFE=1 collapses the **diffusion-time** axis. It says nothing
about what shape the data lives in. The model can output:
- A `(C, H_img, W_img)` image (image domain) — full image in one shot
- A `(H, D)` trajectory chunk (our domain) — full chunk in one shot

In our domain, "one-shot" means **noise → (H=8, D) trajectory chunk in
one model evaluation.** That chunk still has 8 internal sequence
positions — those are not iterated over in any DGM-related sense; the
1D temporal U-Net processes them in parallel via convolutions.

The MPC loop around this one-shot generation is a SEPARATE concern.
You can have:
- One-shot DGM + MPC replanning (our application): 1 NFE per chunk, many chunks per episode.
- Multi-step DGM + MPC replanning (e.g. DPCC at K=100): 100 NFE per chunk, many chunks per episode.
- One-shot DGM + open-loop (no replan): 1 NFE per episode, but H = episode length.
- Multi-step DGM + open-loop: 100 NFE per episode, H = episode length.

The four combinations are all valid; one-shot vs. multi-step (NFE) is
independent of full-task vs. MPC-chunked (H).

**Our H=8 is an MPC-design choice unrelated to iMF's one-shot claim.**
iMF's one-shot promise is: the chunk it produces is reachable in one
diffusion-time step. That promise is independent of whether the chunk
covers 8 steps or 300 steps of real-world time.

---

## 6. Conceptual lookup table

For someone porting iMF from images to robot trajectories:

| Question | Image domain (iMF native) | Trajectory domain (our use) |
|---|---|---|
| What is the data? | A single image `(C, H_img, W_img)` | A trajectory chunk `(H, D)` |
| Does the data have an "internal time"? | No (spatial only) | Yes (`H` axis is real-world time) |
| Does diffusion time `t` interact with internal data structure? | No (whole image evolves together) | No (whole chunk evolves together — same `t` for all `H` slots) |
| What does NFE=1 produce? | One full image in one forward pass | One full `(H, D)` chunk in one forward pass |
| Is the full task = one DGM call? | Yes — task = generate image | **No** — task = execute many chunks via MPC |
| Where does "horizon" come from? | Doesn't exist as a concept | MPC design parameter; orthogonal to NFE |
| Can NFE=1 be applied to longer `H`? | N/A | Yes — H=100 + NFE=1 = one massive chunk in one shot, no replanning. Brittle but valid. |

---

## 7. Practical implications for FM-PCC

### 7.1 Our H=8 iMF is correct in design

The Gen3v4 iMF code applies iMF's mean-flow loss to `(H=8, D=9)` data
(visual avoiding's trajectory chunk). NFE=1 means producing this
8-step chunk in one model evaluation. The chunk is then MPC-executed
(some prefix executed, then replan). This is a clean reinterpretation
of iMF for sequential decision-making.

If iMF's one-shot promise holds, the chunk will be on-distribution
(coherent 8-step trajectory) in one Euler step. This is what
`fix_1/INVESTIGATION.md` validated (post-target-fix), and what
`fix_2/INVESTIGATION.md` is still investigating for residual
jitter.

### 7.2 What would "NFE=1, H=episode_length" look like?

If we wanted truly one-shot full-task generation (no replanning), we
could train with `H = episode_length` (e.g., 300). Then:
- One model evaluation produces the entire 300-step trajectory.
- Execute open-loop, no replanning.

Two practical problems:
- U-Net depth constraint: `H mod 2^L = 0`, so we'd need H ∈ {256, 320, ...}.
- Compounding error: any noise in the predicted 300-step trajectory
  has no opportunity to be corrected. Performance would degrade
  rapidly.

This is **why MPC (small H + replan) is the standard choice in
robotics**, regardless of whether the DGM is one-shot or multi-step.

### 7.3 Why iMF success at images doesn't automatically mean iMF success at our chunks

In image domain, "data manifold" is the set of natural images. iMF's
one-shot Euler step lands on (or close to) the manifold for that domain.

In our trajectory domain, "data manifold" is the set of expert-style
8-step trajectory chunks. Whether iMF's one-shot step lands on THIS
manifold depends on:
- Whether the data distribution is sufficiently regular (mean-flow
  assumption: integration is roughly linear over h).
- Whether the model has enough capacity to learn the mean-flow
  velocity field across the (x, t, h) parameter space.

For images, iMF was shown to work in the paper. For 8-step
trajectories: unknown until empirically validated. Our `fix_1/fix_2`
investigations are precisely the validation. **The "H concept" is not
a barrier — iMF works on any data shape.**

---

## 8. One-line summary

`H` (trajectory horizon) and `NFE` (diffusion integration steps) are
**orthogonal axes**. iMF / one-shot / NFE=1 collapses the latter, not
the former. Our H=8 is an MPC design choice; it's correct to use with
iMF and doesn't conflict with iMF's one-shot promise. In image-domain
iMF, "H" simply doesn't exist as a concept — the data has no internal
time — but the diffusion-time axis is identical between image and
trajectory applications.
