# D3IL paper metrics — how Success Rate & Entropy are computed (aligning), and did we replicate them?

**Date:** 2026-06-20
**Question:** where in the d3il repo are the paper's **success rate** and **entropy** actually computed?
Is any of it manual/off-PC? Did our FM-PCC baseline run replicate them?
**Short answer:** **fully in-code, no manual analysis.** The env defines success + mode; the simulation
harness aggregates success-rate + a **mode-coverage entropy** over contexts. **We already replicate it** —
our `eval_imf_visual_aligning.py` reproduces the exact formula and our last baseline run produced these
numbers. Source lines below.

---

## 1. Where the numbers come from (the chain)

```
aligning env  ──per-rollout──►  Aligning_Sim  ──aggregate over contexts──►  success_rate, entropy, distance
(success,mode,                  (test_agent /                                (+ wandb 'score')
 mean_distance)                  eval_agent)
```

- **Env** = `/workspaces/d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py`
- **Sim harness** = `/workspaces/d3il/simulation/aligning_sim.py`

No spreadsheet / manual step — everything is computed in `aligning_sim.test_agent()` and printed/logged.

---

## 2. Per-rollout signals (defined in the ENV)

Each env step returns `info = {'mode', 'success', 'mean_distance'}` (`aligning.py:293`).

### 2a. `success` — task completion (binary, sticky) — `aligning.py:334–353`
```
box_goal_pos_dist = ‖box_pos − target_pos‖
box_goal_rot_dist = rotation_distance(box_quat, target_quat) / π
success = (box_goal_pos_dist ≤ pos_min_dist) AND (box_goal_rot_dist ≤ rot_min_dist)
```
i.e. the pushed box is **close enough in both position and orientation** to the target. Once true it
**terminates** the episode (early-termination).

### 2b. `mode` — which strategy the rollout used — `aligning.py:295–319`
```
robot_box_dist = ‖box_xy − robot_xy‖
mode = 0  if robot_box_dist < self.robot_box_dist   # approached/pushed from one side
       1  otherwise                                 # the other side
```
The aligning task is **multimodal**: the box can be aligned via **2 distinct strategies** (`n_modes = 2`).
`mode` records which one this rollout expressed.

### 2c. `mean_distance` — continuous quality — `aligning.py:317`
```
mean_distance = 0.5 · (box_goal_pos_dist + box_goal_rot_dist)
```
Average of positional + (normalized) rotational distance to the target. Lower = better; reported as a
secondary metric (`aligning_sim.py:199–201`).

---

## 3. Aggregation (defined in the SIM HARNESS) — `aligning_sim.py:128–205`

`test_agent` runs `n_contexts × n_trajectories_per_context` rollouts (default 30 contexts), filling
`successes`, `mode_encoding`, `mean_distance` tensors (`:128–173`).

### 3a. Success rate — `aligning_sim.py:177`
```
success_rate = mean(successes)            # fraction of all rollouts that succeeded
```
Plain mean of the binary success flag over every context × trajectory.

### 3b. Entropy — **mode-coverage entropy**, the paper's diversity metric — `aligning_sim.py:178–194`
This is the subtle one. It measures **how evenly the agent covers the 2 task modes**, per context, among
**successful** rollouts:

```
for each context c:
    mode_probs[c] = [ #(successful rollouts in c with mode==0),
                      #(successful rollouts in c with mode==1) ] / n_trajectories_per_context
mode_probs /= row-sum (+1e-12)            # normalize → p(mode | context)

entropy = mean_over_c [ − Σ_m  p(m|c) · log(p(m|c)) / log(n_modes) ]      # n_modes = 2
```

- It's **base-`n_modes` normalized Shannon entropy** (`/ log(2)`), so **entropy ∈ [0, 1]**.
- **entropy = 1** ⇒ the agent reaches **both** modes equally often (full multimodal coverage — the goal of
  imitating a multimodal demo set).
- **entropy = 0** ⇒ mode collapse (always the same strategy), even if success_rate is high.
- Only **successful** rollouts contribute to the mode counts (`successes[c,:]==1`), so entropy is a
  *quality-conditioned diversity* score.

### 3c. Combined score + logging — `aligning_sim.py:196–203`
```
score = 0.5 · (success_rate + entropy)        # the single headline number (wandb 'score')
wandb: Metrics/successes, Metrics/entropy, Metrics/distance
print: 'Successrate', 'entropy', 'Mean Distance'
```

> **Why entropy at all?** D3IL is a *multimodal imitation* benchmark — a good policy must not just succeed
> but reproduce the **diversity** of human strategies. Success rate alone rewards mode collapse; entropy
> is what penalizes it. The paper reports both (and the `0.5·(SR+H)` score).

---

## 4. Did WE replicate it? — yes

**Our eval reproduces the exact harness + formula:** `imf_visual_aligning_test/eval_imf_visual_aligning.py`
- `:2068` `success_rate, mode_encoding, successes, mean_dist = sim.test_agent(agent)` — calls the **same**
  `Aligning_Sim.test_agent` (so success/mode/distance are computed by the **native d3il code**, not a
  re-implementation).
- `:2072–2081` recomputes **entropy** with the identical `n_modes=2`, per-context `mode_probs`,
  `/ log(n_modes)` normalization — byte-for-byte the `aligning_sim.py:178–194` formula.
- `:2110, :2134` writes `success_rate`, `entropy`, `mean_distance` into the eval `.npz` / pickle.

So our last baseline run **did** produce the paper metrics, the same way the paper does — and they land in
our result files (analyzable with `npz_analysis/analyze_npz.py`: keys `success_rate`, `entropy`,
`mean_distance`).

**No manual/off-PC analysis is needed or used** — the metrics are fully computed in the eval run.

---

## 4B. What about `d3il_visual_aligning_baseline_test/`? — partial: success_rate ✅, **entropy ✗**

This folder is a **separate, lightweight baseline eval** (`eval_d3il_visual_aligning.py`,
`train_d3il_visual_aligning.py`) — **not** the same path as §4's `imf_visual_aligning_test`. It runs the
agent and **does** save metrics, but it does **NOT** reproduce the paper's entropy.

**What it computes & saves** (`eval_d3il_visual_aligning.py:461–489`) → `results_seed_{s}.json`
(+ `aggregate_results.json` across seeds, `:542–562`):
```
success_rate        = mean(successes)                         # ✅ matches the paper (:461)
mean_distance_mean/std                                        # ✅ secondary metric
final_xy_dist, final_angle_deg, n_steps                       # extra diagnostics
mode_0_rate         = #(mode==0) / n_rollouts                 # ✗ NOT the paper entropy (:475)
```

**The gap — `mode_0_rate` ≠ entropy:**
- It is the **raw fraction of rollouts that used mode 0**, over **all** rollouts (not success-conditioned,
  not per-context, not Shannon).
- The paper's **entropy** (§3b) is the **base-2-normalized Shannon entropy of `p(mode|context)` over
  successful rollouts**, averaged across contexts. `mode_0_rate` is a single scalar that doesn't capture
  per-context balance and isn't on the [0,1] entropy scale.
- This eval also **does not call** the native `Aligning_Sim.test_agent` (it wraps the agent directly,
  `D3ILBaselineWrapper`), so it never goes through `aligning_sim.py:178–194` where entropy is computed.

**So:**

| Eval path | success_rate | entropy (paper) | mean_distance | calls native `test_agent`? |
|---|---|---|---|---|
| `imf_visual_aligning_test/eval_imf_visual_aligning.py` | ✅ | ✅ (`:2072–2081`) | ✅ | ✅ (`:2068`) |
| `d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py` | ✅ (`:461`) | ✗ (only `mode_0_rate`) | ✅ | ✗ (direct wrapper) |

**Did it produce outputs?** It **writes** `results_seed_{s}.json` / `aggregate_results.json` **when run**
(`:487–489, :559–562`) — but **no such files are committed in the repo** (searched; none found). So this
baseline's metric outputs exist only on whatever machine last ran it, and even then **without entropy**.

**To make this folder paper-complete:** either (a) add the §3b entropy formula on its per-rollout
`mode` + `success` arrays (need per-context grouping + success filtering), or (b) route it through the
native `Aligning_Sim.test_agent` like the `imf_visual_aligning_test` path already does. **Recommended: use
the `imf_visual_aligning_test` numbers for any entropy claim**, since they are the faithful replication.

---

## 4D. Deep dive — the EXACT original d3il code + math (visual aligning)

Verbatim from the d3il repo. This is the ground truth both metrics are read off.

### 4D.1 Per-rollout: `success` & `mode` — `aligning.py`

The success criterion uses a quaternion **rotation distance** helper (`aligning.py:22–32`):
```python
def rotation_distance(p, q):                 # p, q are quaternions
    theta = 2 * np.arccos(abs(p @ q))        # angular distance (rad) between two orientations
    return theta
```

Success (early-termination) — `aligning.py:334–353`:
```python
def _check_early_termination(self):
    box_goal_pos_dist = np.linalg.norm(box_pos - target_pos)          # metres
    box_goal_rot_dist = rotation_distance(box_quat, target_quat)/np.pi  # ∈[0,1], normalized by π
    if (box_goal_pos_dist <= self.pos_min_dist) and (box_goal_rot_dist <= self.rot_min_dist):
        self.terminated = True
        return True                          # success == box matched target in BOTH pos & rot
    return False
```

Mode (which of the 2 strategies) + `mean_distance` — `aligning.py:295–319`:
```python
def check_mode(self):
    robot_box_dist = np.linalg.norm(box_pos[:2] - robot_pos[:2])
    mode = 0 if robot_box_dist < self.robot_box_dist else 1    # which side the robot engaged
    mean_distance = 0.5 * (box_goal_pos_dist + box_goal_rot_dist)
    return mode, mean_distance
```

These are returned every step in `info` (`aligning.py:291–293`):
```python
self.success = self._check_early_termination()
mode, mean_distance = self.check_mode()
return observation, reward, done, {'mode': mode, 'success': self.success, 'mean_distance': mean_distance}
```

### 4D.2 The rollout loop fills 3 tensors — `aligning_sim.py:46–123`

`eval_agent` runs every (context, trajectory) and stores the **last** `info` of each rollout into
`[n_contexts, n_trajectories_per_context]` tensors:
```python
mode_encoding[context, i] = torch.tensor(info['mode'])        # 0 or 1
successes[context, i]     = torch.tensor(info['success'])     # 0 or 1
mean_distance[context, i] = torch.tensor(info['mean_distance'])
```

### 4D.3 The EXACT aggregation — `aligning_sim.py:175–203` (verbatim)

```python
n_modes = 2
success_rate = torch.mean(successes).item()                          # ── SUCCESS RATE

mode_probs = torch.zeros([self.n_contexts, n_modes])
for c in range(self.n_contexts):                                     # per context c
    mode_probs[c, :] = torch.tensor(
        [sum(mode_encoding[c, successes[c, :] == 1] == 0) / self.n_trajectories_per_context,   # count mode-0 among SUCCESSES
         sum(mode_encoding[c, successes[c, :] == 1] == 1) / self.n_trajectories_per_context])  # count mode-1 among SUCCESSES

mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)             # normalize rows → p(m|c)

entropy = - (mode_probs * torch.log(mode_probs + 1e-12)
             / torch.log(torch.tensor(n_modes))).sum(1).mean()       # ── ENTROPY

wandb.log({'score': 0.5 * (success_rate + entropy)})                 # headline score
```

### 4D.4 The math, written out

Let `S_{c,i} ∈ {0,1}` = success, `M_{c,i} ∈ {0,1}` = mode, for context `c`, trajectory `i`;
`C` = n_contexts, `T` = n_trajectories_per_context, `K = n_modes = 2`.

**Success rate** — plain mean over all rollouts:
```
                1
success_rate = ───  Σ_c Σ_i  S_{c,i}
               C·T
```

**Entropy** — per-context mode distribution **over successful rollouts only**, then base-K-normalized
Shannon entropy, averaged over contexts:
```
                    Σ_i  1[S_{c,i}=1 ∧ M_{c,i}=m]                    (unnormalized count/T)
   p̃(m|c)  =       ─────────────────────────────
                              T

   p(m|c)  =  p̃(m|c) / Σ_{m'} p̃(m'|c)            (row-normalize → distribution over the 2 modes)

                1          K-1   p(m|c) · ln p(m|c)
   entropy =  ───  Σ_c  − Σ      ──────────────────          ∈ [0, 1]
                C         m=0          ln K
```

Reading it:
- The `/ ln K` makes it **normalized** entropy: `K=2` ⇒ divide by `ln 2` ⇒ range exactly **[0,1]**.
- **entropy = 1** ⇔ every context's successes split **50/50** across the two modes (max diversity).
- **entropy = 0** ⇔ each context's successes all use **one** mode (mode collapse).
- Built on `successes==1` only ⇒ a policy gets diversity credit **only for modes it can reach
  successfully**. A context with **zero** successes contributes `p≈0,0 → 0/0` guarded by `+1e-12` →
  contributes ~0 entropy (no diversity credit), which is the intended behaviour.

**Headline score** = `0.5 · (success_rate + entropy)` (`aligning_sim.py:196`) — the single number the
paper/leaderboard ranks on; it rewards **both** completing the task **and** covering both human strategies.

> **Why two metrics, restated with the math in hand:** `success_rate` alone is maximized by a collapsed
> policy that always pushes from the same side; `entropy` is the term that forces a policy to reproduce the
> **bimodal** human demonstration distribution. D3IL is a *multimodal imitation* benchmark, so both are
> required — and the `0.5·(SR+H)` score bakes that in.

---

## 4E. What does success/failure mean IRL? + the entropy meaning, deeper

### 4E.1 The aligning task, physically
A robot arm with a rod pushes a **box** (`push_box`) on a table so it lands on a **target box** outline
(`target_box`) — matching both **where** it sits and **how** it's rotated. Two human strategies exist
(push from the left vs the right side to rotate it into place) → the **bimodal** nature the entropy
measures.

### 4E.2 When exactly is a rollout judged success vs failure? — `aligning.py:198–199, 334–353`

Success fires the instant **both** of these hold (hard thresholds, in SI units):

```
self.pos_min_dist = 0.018      # 1.8 cm
self.rot_min_dist = 0.048      # 0.048 · π ≈ 0.151 rad ≈ 8.6°   (rot_dist is normalized by π)

success  ⇔   ‖box_pos − target_pos‖ ≤ 0.018   AND   rotation_distance(box,target)/π ≤ 0.048
             └──────── within 1.8 cm ────────┘       └──────── within ~8.6° ────────┘
```

**IRL meaning:** *"the box is parked on the target to within **1.8 cm** of position **and** **~8.6°** of
orientation."* Both must be satisfied **simultaneously** (logical AND) — getting the location right but
the angle wrong is a **failure**, and vice-versa. That is what makes aligning harder than pure reaching:
it's a full **SE(2) pose** match (x, y, yaw), not just a point.

**Binary + sticky + early-terminating:**
- It is **0/1**, not graded — `mean_distance` is the graded version, but `success` itself is a threshold.
- The moment it's true, `self.terminated = True` (`aligning.py:351`) → the episode **ends**. So a rollout
  is a success iff it reaches the pose box **at any point** before the step budget runs out; if the budget
  expires first, `success` stays 0 ⇒ **failure**.
- No partial credit: a rollout that ends at 1.9 cm / 9° is exactly as "unsuccessful" as one that wandered
  off — both contribute `0` to `success_rate`.

**The math of the decision (per rollout):**
```
                    ⎧ 1   if  d_pos ≤ 0.018  ∧  d_rot ≤ 0.048      (and reached before timeout)
S_{c,i}  =          ⎨
                    ⎩ 0   otherwise
   where  d_pos = ‖box_pos − target_pos‖₂ ,   d_rot = (2·arccos|q_box·q_tgt|)/π
```
`d_rot` uses the quaternion dot → angle, normalized to [0,1] by dividing by π; the `|·|` makes it the
**shorter** of the two equivalent rotations (q and −q are the same orientation).

### 4E.3 Entropy — what it *means*, with the math intuition

**Meaning in one sentence:** entropy asks *"across the successful runs, did the policy use **both** human
push-strategies in balance, or did it collapse to one?"* — it is a **behavioural-diversity** score, not a
quality score.

**Why a plain mean isn't enough (the failure it catches):** imagine a policy that **always** pushes from
the left and succeeds 100% of the time. Its `success_rate = 1.0` — looks perfect. But it has **thrown away
half the demonstrated behaviour** (the right-side strategy). For an *imitation* benchmark that is a real
failure: the policy did not learn the human's **distribution**, only one slice of it. **Entropy is the
term that exposes this** — that left-only policy scores `entropy = 0`.

**The math, built up intuitively:**
1. For each context `c`, look only at the **successful** rollouts and count how many used mode 0 vs mode 1:
   ```
   p(0|c), p(1|c)     with  p(0|c)+p(1|c) = 1     (a coin describing "which strategy, given context c")
   ```
2. Shannon entropy of that coin measures its **balance**:
   ```
   H_c = −[ p(0|c)·ln p(0|c) + p(1|c)·ln p(1|c) ]
   ```
   - all one mode → `p = (1,0)` → `H_c = 0` (a certain coin carries no diversity)
   - even split → `p = (½,½)` → `H_c = ln 2` (maximum for 2 outcomes)
3. **Normalize** by `ln K = ln 2` so the scale is **[0,1]** regardless of #modes:
   ```
   Ĥ_c = H_c / ln 2 ∈ [0,1]
   ```
4. **Average over contexts:**
   ```
   entropy = (1/C) Σ_c Ĥ_c
   ```

**How to read a value:**
| entropy | meaning |
|---|---|
| **1.0** | every context's successes are a perfect 50/50 mix of both push-strategies — full multimodal coverage |
| **~0.5** | partial coverage — some contexts bimodal, many lean one way |
| **0.0** | **mode collapse** — within each context, successes all use the same single strategy |

**Subtle but important — it's conditioned on success:** the counts use `mode_encoding[c, successes[c,:]==1]`.
A mode the policy attempts but **fails** at gives **no** entropy credit. So entropy rewards diversity *that
actually works* — you can't game it by failing in creative ways. A context with **zero** successes yields
`p̃=(0,0)`, and the `+1e-12` guard makes its normalized `p≈(0,0)` → `H_c≈0` (no diversity credit, as
intended).

**Why combine as `0.5·(SR+H)`:** the two are **orthogonal** — `SR` rewards *getting there*, `H` rewards
*getting there in all the demonstrated ways*. A policy must do **both** to score high; maxing one while
ignoring the other caps the headline `score` at 0.5. That is exactly the property a multimodal-imitation
benchmark wants.

---

## 4F. Train + eval protocol (real config values) & reading the paper's 0.278 / 0.139

> **Source:** this section now uses the **actual D3IL paper PDF** (ICLR 2024, openreview `6pPYRXKPpw`,
> extracted Tables 3/4/5 + Eq. 2) **cross-checked** against the repo configs. Verified facts are cited as
> *(paper, Table N)* or *(config)*. The `0.278 / 0.139` figures are confirmed below as the paper's
> **Image Data, Aligning, DDPM-ACT** row.

### 4F.1 The real eval protocol — paper vs config (⚠ they differ)

**Paper (Table 4):** Aligning (T2) uses **S₀ = 60** sampled initial states and **Nsim = 18·S₀ = 1080
simulations** (18 rollouts per initial state). Mean ± std is over **6 random seeds** (Table 3 caption).

**Repo config** (`aligning_vision_config.yaml:86–93`) ships:
```yaml
simulation:
  n_contexts: 60                 # = S₀, matches the paper ✓
  n_trajectories_per_context: 8  # ⚠ paper Table 4 says 18, not 8
```

> ⚠️ **Discrepancy to know:** the paper aggregates **18 rollouts per context (1080 total)**; the shipped
> config uses **8 (480 total)**. Both share `n_contexts=60`. For a *faithful* reproduction of the paper
> number, set `n_trajectories_per_context: 18`. The `8` still gives a valid estimate but with higher
> variance on entropy (fewer samples to resolve the per-context mode split). Either way you need **many**
> rollouts per context — entropy is undefined with 1.

A separate **in-training** eval (`train_simulation:`, `:75–82`) uses `n_contexts: 1,
n_trajectories_per_context: 1` — a single-rollout smoke check every `eval_every_n_epochs`, **not** the
headline metric.

**Entropy formula confirmed against the paper (Eq. 2):**
```
E_{s0}[ H(π(β|s0)) ] ≈ − (1/S₀) Σ_{s0} Σ_{β∈B}  π(β|s0) · log_{|B|} π(β|s0)
```
This is **exactly** the code in §3b: per-initial-state behaviour distribution `π(β|s0)`, Shannon entropy
with **log base `|B|`** (= `/ log(n_modes)`), Monte-Carlo-averaged over the `S₀=60` initial states. For
Aligning, **`|B| = 2`** (two push-strategy modes). ✓ paper ≡ repo ≡ our replication.

### 4F.2 Training protocol (configs)

| Setting | Vision aligning (`aligning_vision_config.yaml`) | State aligning (`aligning_config.yaml`) |
|---|---|---|
| `epoch` | **4** ⚠️ (see note) | **500** |
| `eval_every_n_epochs` | 2 | 50 |
| `train_batch_size` | 64 | 1024 |
| `window_size` (horizon) | 8 | — |
| `seed` | 42 | — |
| final-eval `n_contexts × n_traj` | 60 × 8 = **480** | 60 × 8 = **480** |
| `n_cores` (final eval) | 5 | 30 |

> ⚠️ **`epoch: 4` in the vision config is a debug placeholder**, not the paper's training length — the
> pushing/sorting **vision** configs are *also* `epoch: 4`, while every **state** config is 200–500. Treat
> the vision `4` as "left at a smoke-test value" and use a realistic schedule (state-aligning's `500` is
> the reference).

**Paper model-selection detail (image experiments, §4.2):** *"For image-based experiments we evaluate the
task performance frequently (after every 1/10th of total training) and chose the model with the best
task-performance."* So the reported image number is a **best-checkpoint-over-training** selection, not the
final epoch — eval every 10% of training and keep the best. Replicate that selection rule, not just a
fixed final-epoch eval.

### 4F.3 How to replicate with OUR baseline code

Map the above onto our pipeline (`imf_visual_aligning_test/` — the path that computes entropy faithfully,
§4):
1. **Use the native sim** — our `eval_imf_visual_aligning.py:2068` already calls `Aligning_Sim.test_agent`,
   so set the sim block to **`n_contexts=60`** and **`n_trajectories_per_context=18`** to match the paper
   (Table 4: Nsim=18·S₀); the shipped config's `8` works but is noisier on entropy (§4F.1).
2. **Train long enough + best-checkpoint selection** — do **not** copy `epoch: 4`; train a realistic
   schedule and **eval every 1/10 of training, keep the best task-performance checkpoint** (paper §4.2).
3. **6 seeds, report mean ± std** (paper Table 3); fix the **test contexts**
   (`environments/dataset/data/aligning/test_contexts.pkl`) so contexts match.
4. **Report `success_rate`, `entropy`, and `score = 0.5·(SR+H)`** — all land in our npz and are summarized
   by `npz_analysis/analyze_npz.py`.

### 4F.4 Reading the paper number `0.278 / 0.139` (DDPM-ACT, Image Data, Aligning)

**Verbatim from the paper (Table 3, Image Data row):**
```
Image Data   Aligning (T2)
DDPM-ACT     Success 0.278 ± 0.071     Entropy 0.139 ± 0.054
```
**The decisive context — the same agent on STATE vs IMAGE (paper Table 3):**

| DDPM-ACT, Aligning | Success Rate | Entropy |
|---|---|---|
| **State Data** | **0.849 ± 0.023** | **0.749 ± 0.041** |
| **Image Data** | **0.278 ± 0.071** | **0.139 ± 0.054** |

So `0.278 / 0.139` is **not "DDPM-ACT is weak"** — the *same* model scores 0.849 / 0.749 from state. The
collapse is the **vision modality**: inferring SE(2) box pose from two 96×96 images is dramatically harder
than from the 20-D handcrafted state. **Read both numbers as "image aligning is hard," not "the method
failed."**

**`success_rate = 0.278 ± 0.071`** → ~28% of rollouts parked the box within **1.8 cm & ~8.6°** (the §4E.2
gate), down from 85% on state. Pixels + tight SE(2) gate is the difficulty.

**`entropy = 0.139 ± 0.054`** (on **[0,1]**) → **near mode-collapse**, down from 0.749 on state. Recall
`1.0` = even 50/50 use of both push-strategies per context, `0.0` = always one. At `0.139` the (few)
successful image rollouts overwhelmingly use **one** strategy. Intuition: when the policy can barely solve
the task from pixels, it clings to its single most-reliable mode and loses the diversity it *had* on state.

**Combined `score = 0.5·(0.278 + 0.139) = 0.209`** (vs `0.5·(0.849+0.749)=0.799` on state). The image
modality knocks the headline from ~0.80 to ~0.21 — the benchmark's way of showing vision aligning is both
**hard to solve and hard to solve diversely**.

> Note: the in-text Figure-1 entropy/SR numbers (e.g. 0.75 at history=1) are the **state** ablation; don't
> confuse them with the 0.139 image result.

**How to use these as YOUR baseline target:** DDPM-ACT is the **most directly comparable** baseline to our
generative (flow/diffusion) visual-aligning policy — same modality (images), same task, same generative
family (diffusion). So `0.278 / 0.139` is the **apples-to-apples reference row**. When our
`imf_visual_aligning` baseline runs under the §4F.1 protocol (60×8, native sim), compare our printed
`success_rate` / `entropy` directly to it:
- **match `~0.278` success** ⇒ our pipeline reproduces DDPM-ACT-level task competence (sanity that the
  port is correct);
- **beat `0.139` entropy** at comparable success ⇒ the meaningful win — our flow model covers the **two
  modes** more evenly than DDPM-ACT, which is exactly what a *distribution-matching* generative policy
  should buy on a multimodal task.

> **Two caveats that DON'T change the row, only the comparison:**
> 1. The paper's DDPM-ACT used **its own** training length / hparams + best-checkpoint selection; our
>    number is only comparable if we train to convergence (not `epoch: 4`) under the **same 60×18 eval**
>    with the same model-selection rule (§4F.2–3).
> 2. The metric *definitions* are identical for every agent (§3–§4E), so any agent's row reads the same —
>    DDPM-ACT (image) is just the architecturally-closest, hence fairest, target. The full image row also
>    reports Sorting-4/6 if you want harder points of comparison.

---

## 4G. Replicating the paper's `0.278 / 0.139` with `d3il_visual_aligning_baseline_test/`

**Good news:** this baseline trains the **exact paper agent** — `ddpm_encdec_vision` (a Transformer
encoder-decoder DDPM = action-chunking diffusion on images) = the paper's **DDPM-ACT (image)**, the very
row that reports `0.278 / 0.139`. So it *is* the right harness to reproduce that number. **But** as shipped
it has **three gaps** vs the paper protocol; close them and it reproduces.

### 4G.1 Agent match — ✅ already correct
`d3il_eval_config.yaml`: `agent_name: ddpm_encdec_vision`, `agent_cfg_group: ddpm_encdec_vision_agent`
(Hydra-composed from `d3il/configs/agents/`). Encoder-decoder + diffusion + vision = **DDPM-ACT image**.
Nothing to change here.

### 4G.2 The 3 gaps to close (shipped = smoke-test, not paper protocol)

| # | Paper protocol (verified) | This baseline as shipped | Fix |
|---|---|---|---|
| **G1 — eval scale** | `n_contexts=60`, `n_traj_per_ctx=18` ⇒ **1080 rollouts** (Table 4) | `n_contexts: 3`, `n_trajectories_per_context: 1` (smoke) | set `n_contexts: 60`, `n_trajectories_per_context: 18` in `d3il_eval_config.yaml` |
| **G2 — entropy not computed** | `entropy` = base-`\|B\|` Shannon over modes (Eq. 2) | eval saves only `success_rate`, `mean_distance`, **`mode_0_rate`** (§4B) — **no entropy** | add the §3b/§4D.3 entropy block, **or** route eval through native `Aligning_Sim.test_agent` (as `imf_visual_aligning_test` does) |
| **G3 — model selection** | eval every 1/10 training, keep **best task-performance** ckpt (paper §4.2) | trains with **val-loss** checkpointing (`eval_best_*.pth` = lowest val loss) | either accept val-loss ckpt (approx) or add periodic sim-eval + best-success selection like `run_vision.py` |

> **Why G2 matters most:** without entropy you literally cannot compare to `0.139`. `mode_0_rate` (raw
> fraction of mode-0 rollouts) is **not** the paper metric (it's not per-context, not success-conditioned,
> not base-`|B|` normalized). This is the one true blocker.

### 4G.3 Concrete steps

**1. Train the DDPM-ACT image agent** (mirror the documented invocation in `train_d3il_visual_aligning.py`):
```bash
python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
    "agents=ddpm_encdec_vision_agent" \
    "agent_name=ddpm_encdec_vision" \
    "hydra.run.dir=logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_42/weights"
```
- Train a **realistic** length (the vision config's `epoch: 4` is a debug value — §4F.2; use the paper-grade
  schedule). Repeat for **6 seeds** to match the paper's mean ± std.
- (Cluster: there is an sbatch dir `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/`.)

**2. Set the eval to paper scale** — edit `d3il_eval_config.yaml`:
```yaml
seeds:                       [0,1,2,3,4,5]   # 6 seeds (paper)
n_contexts:                  60              # G1
n_trajectories_per_context:  18              # G1  → 1080 rollouts
```

**3. Make eval emit entropy (G2)** — recommended: **delegate to the native sim** so you get the paper
formula for free (this is exactly what our `imf_visual_aligning_test/eval_imf_visual_aligning.py:2068`
already does):
```python
sim = Aligning_Sim(seed=seed, device=..., render=False,
                   n_contexts=60, n_trajectories_per_context=18, if_vision=True)
success_rate, mode_encoding, successes, mean_distance = sim.test_agent(agent)
# then the §4D.3 entropy block on (mode_encoding, successes)
```
…instead of the current hand-rolled rollout loop that only yields `mode_0_rate`. Alternatively, keep the
current loop but add the §3b entropy computation on its per-rollout `mode` + `success` arrays.

**4. Run eval + read results:**
```bash
python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py \
    --config d3il_visual_aligning_baseline_test/d3il_eval_config.yaml
# → results_seed_{s}.json + aggregate_results.json  (success_rate, entropy[after G2], mean_distance)
```

### 4G.4 What "replicated" looks like
With G1–G3 closed and paper-grade training, expect the DDPM-ACT image aligning numbers to land **near**:
```
success_rate ≈ 0.278 ± 0.071        entropy ≈ 0.139 ± 0.054        score ≈ 0.21
```
(±std across 6 seeds; exact match is unlikely — sim stochasticity, seed set, MuJoCo version, and the
val-loss-vs-best-task model-selection difference (G3) all shift it.) **Acceptance:** both numbers within
~1 std of the paper, *and* the state-vs-image gap reproduced (state should be ~0.85/0.75 — §4F.4) — that
confirms the harness, not luck.

> **Two-tier recommendation:**
> - **For a faithful paper-replication** of `0.278/0.139`, use **this** `d3il_visual_aligning_baseline_test`
>   (it's the native DDPM-ACT agent) with G1–G3 fixed.
> - **For OUR method's number** (flow/iMF visual aligning), use `imf_visual_aligning_test` (already emits
>   entropy correctly, §4) and compare *its* `success_rate`/`entropy` against the replicated baseline.
>   Both feed `npz_analysis/analyze_npz.py` for the side-by-side.

---

## 5. Quick reference — exact source lines

| Metric | Definition | File:line |
|---|---|---|
| `success` (per rollout) | pos_dist ≤ pos_min AND rot_dist ≤ rot_min | `d3il/.../aligning.py:334–353` |
| `mode` (per rollout) | `robot_box_dist < robot_box_dist` ? 0 : 1 | `d3il/.../aligning.py:295–319` |
| `mean_distance` | `0.5·(pos_dist + rot_dist/π-norm)` | `d3il/.../aligning.py:317` |
| `success_rate` | `mean(successes)` | `d3il/simulation/aligning_sim.py:177` |
| `entropy` | base-2-normalized Shannon of `p(mode\|context)` over successes | `d3il/simulation/aligning_sim.py:178–194` |
| `score` | `0.5·(success_rate + entropy)` | `d3il/simulation/aligning_sim.py:196` |
| **our replication** | same `test_agent` + entropy formula → npz | `FM-PCC/imf_visual_aligning_test/eval_imf_visual_aligning.py:2068–2081, 2110` |

---

## 6. Related docs
- Pipeline (how the eval runs end-to-end): [D3IL_Native_Visual_Aligning_Pipeline_Guide.md](./D3IL_Native_Visual_Aligning_Pipeline_Guide.md)
- npz metric extraction: [../npz_analysis_tool/CHANGELOG.md](../npz_analysis_tool/CHANGELOG.md) (`success_rate`, `entropy` are auto-summarized)

> **Caveat:** the above is the **aligning** task (the visual-aligning baseline we run). The **avoiding**
> task computes its own entropy differently — over **24 trajectory "modes"** via end-point binning
> (`gym_avoiding_env/.../avoiding.py:279–280`, `entropy = −Σ p·log(p)/log(24)`) — not the 2-mode
> push-side scheme here. Don't mix the two.
