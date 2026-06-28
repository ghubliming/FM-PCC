# Why Non-Visual Aligning Uses 20D State (NOT 9D)

**Date**: 2026-06-27
**Verdict**: The 20D state for **non-visual aligning is CORRECT, not a bug.** Reducing it to the
9D visual layout would make the task **impossible**. The folder name "Non_Visual_9D" reflects an
*abandoned* idea — keep 20D obs (→ 23D trajectory).

> [!IMPORTANT]
> **The one-sentence reason:** to align a box you must KNOW where the box and target are. There
> are only two ways to know: **see it** (camera → FiLM) or **be told it** (state vector). The
> non-visual path has no camera — so the box/target pose *must* live in the state. That is the
> 14 extra dimensions the 9D layout throws away.

---

## 1. The core principle: "where is the box?" must enter somehow

The aligning task = push a box (random start pose) to a target (random pose). The policy cannot
push toward something it cannot perceive. Box and target poses change every context
(`aligning_sim.py`: `env.reset(context=ctx_pool[context])`, 30 contexts), so they are **not**
memorizable constants — they must be observed at run time.

```
        HOW DOES THE POLICY LEARN THE BOX/TARGET POSE?
        ┌─────────────────────────────┬──────────────────────────────┐
        │  VISUAL path                 │  NON-VISUAL path              │
        ├─────────────────────────────┼──────────────────────────────┤
        │  camera image → ResNet →     │  NO camera.                   │
        │  128-D latent → FiLM cond    │  Box/target pose must be      │
        │  the box/target are SEEN     │  PUT INTO THE STATE VECTOR    │
        │  → state can stay robot-only │  → state must be 20-D         │
        └─────────────────────────────┴──────────────────────────────┘
```

The visual FiLM and the 20-D state are **two solutions to the same problem** (delivering box/target
info to the model). They are alternatives — you need exactly one. Visual uses the camera; non-visual
uses the state. Strip the box/target from the state *and* have no camera = the model is blind.

---

## 2. Proof from D3IL's own datasets (three cases, not opinions)

D3IL builds the state differently for each case — and the difference is exactly box/target info:

| Task / mode | Code | `input_state` contents | obs dim | Box/target in state? |
|---|---|---|---|---|
| **Avoiding** (non-visual) | `avoiding_dataset.py:57` | `[des_xy, c_xy]` | **4** | ❌ no (not needed — see §3) |
| **Aligning** (non-visual) | `aligning_dataset.py:77` | `[des_c_pos, c_pos, box_pos, box_quat, tgt_pos, tgt_quat]` | **20** | ✅ **yes** |
| **Aligning** (visual) | `aligning_dataset.py:244` | `des_c_pos` only (camera carries the rest) | 3 | ❌ no — camera does it |

The non-visual aligning state literally concatenates `push_box_pos`, `push_box_quat`,
`target_box_pos`, `target_box_quat`. D3IL itself decided you need them. We did not invent 20D — we
inherited it correctly.

---

## 3. Why avoiding can use a small state but aligning cannot

You said: *"for avoiding it's fine, but aligning NO."* Exactly right, and here's the precise reason:

| | Avoiding | Aligning |
|---|---|---|
| What varies per episode | only the robot's chosen path (multimodality) | the **box pose** and **target pose** (randomized) |
| Obstacle / object layout | **FIXED** across all episodes | **RANDOM** across contexts |
| Can the policy memorize it? | ✅ yes — fixed columns are baked into the weights | ❌ no — must be observed each episode |
| Therefore state needs | robot position only (4D) | robot + box + target (20D) |

Avoiding's obstacles are at fixed locations, so the network learns them implicitly and the state
only needs the robot's own position. Aligning's box/target move every context, so there is nothing
to memorize — the pose is information that **must be supplied at inference**. No camera ⇒ state.

---

## 4. The exact dimensions

**20-D observation (non-visual aligning):**

| Component | Dim | Source |
|---|---|---|
| `des_c_pos` | 3 | robot desired Cartesian pos (FM's own integrated target) |
| `c_pos` | 3 | robot actual Cartesian pos (sensor) |
| `push_box_pos` | 3 | **box position** ← the whole point |
| `push_box_quat` | 4 | **box orientation** |
| `target_box_pos` | 3 | **target position** |
| `target_box_quat` | 4 | **target orientation** |
| **total obs** | **20** | |

**Trajectory tensor** fed to the temporal U-Net = `action(3) | obs(20)` = **23-D**.
(`visual_unet.py` non-visual branch: `transition_dim = action_dim + obs_dim = 3 + 20 = 23`.)

Compare the **visual** trajectory = `action(3) | [des_c_pos(3), c_pos(3)]` = **9-D**, with the box/
target delivered separately through the camera/FiLM channel.

---

## 5. What "recover to 9D" would actually do (the bug we avoided)

Forcing non-visual aligning to the 9D layout = `action(3) | des_c_pos(3) | c_pos(3)` means the
state contains **only the robot's own position** and **zero information about the box or target**.
Consequences:

- The model sees identical inputs for "box at A, target at B" and "box at C, target at D" → it
  cannot condition its push on the actual scene → it can only output an averaged/blind motion.
- No camera to compensate (that's the whole "non-visual" premise).
- Result: the policy is structurally incapable of aligning — an **information bottleneck**, not a
  tuning problem. No amount of training fixes a missing input.

This is why 9D-for-non-visual was the real bug, and 20D is the fix.

---

## 6. Bottom line / parity table

| Pipeline | Camera? | Box/target info via | obs dim | trajectory dim |
|---|---|---|---|---|
| Visual aligning | ✅ | FiLM (camera latent) | 6 (robot only) | **9** |
| **Non-visual aligning** | ❌ | **state vector** | **20** (robot + box + target) | **23** |
| Avoiding (non-visual) | ❌ | n/a (obstacles fixed) | 4 (robot only) | 6 |

**Keep non-visual aligning at obs_dim = 20 (trajectory 23D).** It is the D3IL-correct, physically
necessary design. The 9D layout is valid only for the *visual* path, where the camera supplies what
the 14 missing state dims otherwise would.

---

## 7. D3IL *visual* aligning vs our 9D+FiLM — and should we try 20D×H8?

You asked: does D3IL's **visual** aligning actually use **20D + image (FiLM)**, like a richer
version of our 9D? **No — the opposite.** Peeked directly at the D3IL repo:

### 7.1 What D3IL visual aligning actually uses (verified)

| Thing | D3IL visual aligning | File |
|---|---|---|
| State `obs_dim` | **3** — `robot_des_pos` **only** (desired EE xyz). No `c_pos`, no box, no target. | `configs/aligning_vision_config.yaml:52` |
| `action_dim` | 3 (`Δrobot_des_pos`) | `aligning_vision_config.yaml:53` |
| `window_size` | 8 (obs **context** window — past frames, NOT a plan horizon) | `aligning_vision_config.yaml:55` |
| Box / target in state? | **NO** — the lines that add `push_box_*`, `target_box_*` are **commented out** | `aligning_dataset.py:228–238` |
| Where box/target come from | **dual cameras** (`bp-cam` + `inhand-cam`) | `Aligning_Img_Dataset` |
| How image+state combine | **early fusion**: `obs_encoder({agentview_image, in_hand_image, robot_ee_pos})` → one fused vector → conditions an **action-level** DDPM | `ddpm_vision_agent.py:75–89` |

> **The camera REPLACES the 20D box/target state.** In non-visual aligning you must hand-feed
> `[box_pos｜box_quat｜target_pos｜target_quat]` (14 of the 20 dims, §2). In visual aligning D3IL
> **deletes** those 14 dims and lets the ResNet read box/target off the pixels — so the explicit
> state collapses to just `robot_des_pos` (3D). Vision is the *substitute* for the 14 box/target
> dims, not an *addition* to them.

### 7.2 Three-way comparison

| | D3IL non-visual | **D3IL visual** | **Our FM-PCC visual (9D+FiLM)** |
|---|---|---|---|
| Explicit state | **20D** (robot+box+target) | **3D** (`des` only) | **6D** (`[des｜c_pos]`) inside a 9D traj |
| Box/target info from | the 20D state | **camera** | **camera** (FiLM) |
| Image conditioning | — | early fusion into obs_encoder | **"fake FiLM"** (additive bias in the U-Net) |
| Denoised object | **action** (3D), MLP, no horizon | **action** (3D), MLP, ctx window 8 | **9D trajectory** `[act｜des｜c_pos]` over **H8** (Janner) |
| Why the extra `c_pos`? | n/a | not present | we added `c_pos` so the **PCC projection** has a real-position channel to bind |

So our 9D = **D3IL-visual's 3D (`des`) + `c_pos` (added for the projector)**, lifted into Janner's
H8 trajectory container with FiLM in place of D3IL's early fusion. We did not "miss" a 20D+image
design — **D3IL deliberately does not use one**, because the camera already carries box/target.

> [!IMPORTANT]
> **"des only" (D3IL-visual) vs "`[des｜c_pos]`" (elsewhere) is not a contradiction — it depends on
> the variant, and the difference is the whole point.**
> - **D3IL *visual* aligning: `des` only (3D).** It has **no real-position channel in the state at
>   all** — `c_pos`/box/target live in the *pixels*. So for D3IL-visual, "condition on `p_des` vs
>   real `p`" **isn't even a choice**: only the command `des` is in the state; the camera *is* its
>   "real `p`."
> - **`[des｜c_pos]` appears in (a) D3IL *non-visual* (4D/20D) and (b) OUR FM-PCC visual + UAV.** In
>   vision, **we re-imported `c_pos` from D3IL's non-visual format** purely so the DPCC projector has
>   a numeric real-position channel to bind (`ParityAligningDataset`:
>   *"des_c_pos alone would project on command targets instead of real end-effector positions"*).
>
> **So the `p_des`-vs-`p` tension is NOT in D3IL's visual design — *we* introduced it into vision by
> re-adding `c_pos` to feed the projector.** Our 9D visual is a **hybrid**: D3IL-visual's `des` (3D)
> **+** `c_pos` borrowed back from D3IL-non-visual (3D). D3IL never shipped that combination in
> vision. (See [CRITIQUE_three_layer_absurdity.md](../../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/data_example_anlysis/CRITIQUE_three_layer_absurdity.md) §6.2d for why `des` exists at all — the gamepad/command-delta origin.)

### 7.2b "Keep the *command* `des`, throw away the *actual* `c_pos`?? That's backwards!" — no, it's optimal

First, a correction that trips everyone up: **D3IL visual does NOT drop `des`. It keeps `des` (3D)
and drops `c_pos`, box, and target.** The instinct "surely you keep the *real* position and drop
the *command*" feels right but is exactly wrong here. Why keeping `des` and dropping `c_pos` is the
**minimal-correct** state:

The state vector's only job is to supply what the policy **cannot get elsewhere**. With a camera:

| Quantity | In the pixels? | Needed by the action? | Verdict |
|---|---|---|---|
| `c_pos` (actual EE pos) | **Yes** — camera sees the arm | **No** — action is `Δdes`, not `Δc_pos` | **doubly droppable** → drop |
| box / target pose | **Yes** — camera sees them | No | drop |
| `des` (internal setpoint) | **No** — a camera can't see an internal command variable | **Yes** — `des += Δdes` needs the baseline | **doubly required** → keep |

So `des` is kept for **two** independent reasons (invisible to the camera **and** it is the
accumulator the `Δdes` action integrates onto), and `c_pos` is dropped for **two** reasons (the
camera already provides it **and** the action never needs it). That is *minimal-sufficient*, not
backwards.

> [!IMPORTANT]
> **Why it *feels* backwards is itself the lesson.** "Surely real `c_pos` > command `des`" is the
> right instinct for a *plant state* — but D3IL's world is **command-centric**: the demos are
> gamepad teleoperation and the action is the command delta `Δdes` (§"turtles down to the gamepad",
> CRITIQUE §6.2d). So **the command `des` is the privileged channel by construction**, and the
> camera's job is to fill in everything *else* (`c_pos`, boxes). This is harmless on a tight-tracking
> arm (`des ≈ c_pos`, so keeping `des` ≈ keeping real). It is the **same command-centric DNA** that
> turns into the category error on a lagging drone (`p_des ≠ p`). Keeping `des` over `c_pos` is not a
> bug — it is the visual-domain fingerprint of the command-centric paradigm.

### 7.3 So… should we try 20D × H8 (+image)?

Two different answers, don't conflate them:

- **For non-visual aligning:** 20D×H8 is exactly right — that *is* the fix this whole doc argues for
  (no camera ⇒ box/target must be in state). ✅
- **For visual aligning:** 20D+image is **redundant in principle** — the camera and the explicit
  box/target state cover the same information, so you'd be giving the model the answer twice and
  defeating the point of "align from pixels." ⚠️

> [!TIP]
> **But 20D×H8+image is a genuinely useful *ablation* — and it directly tests the FiLM_V2 worry.**
> Our visual conditioning is "fake FiLM" (additive bias; see the FiLM_Upgrade docs), suspected to
> be a weak channel. If you feed box/target **explicitly** as state (20D) *and* keep the image,
> you **bypass** the weak FiLM path. Then:
> - if **20D+image ≫ 9D+image** → the FiLM visual channel was the bottleneck (motivates FiLM_V2 /
>   cross-attention), and the model *can* align when given clean box/target info;
> - if **20D+image ≈ 9D+image** → the bottleneck is elsewhere (projector, horizon, self-reference),
>   not the visual conditioning.
> Either outcome is informative. So: **don't ship 20D+image as the design, but do run it as a
> diagnostic** to localize whether "fake FiLM" is what's holding visual aligning back.

### 7.4 One caveat on the "8"s

D3IL visual's `window_size=8` is an **observation-context** window (8 past frames fused), *not* a
planning horizon. Our **H8** is a Janner **trajectory horizon** (8 future steps denoised). Same
number, different role — don't read the matching `8` as evidence they're the same mechanism (§6.2a:
D3IL has no time-axis trajectory; the horizon is purely Janner's).

---

## 8. References

| File | Line | Shows |
|---|---|---|
| `d3il/environments/dataset/aligning_dataset.py` | 64–77 | non-visual aligning state = robot + **box + target** = 20D |
| `d3il/environments/dataset/aligning_dataset.py` | 244 | visual aligning state = `des_c_pos` only (camera carries box) |
| `d3il/environments/dataset/avoiding_dataset.py` | 57 | avoiding state = `[des_xy, c_xy]` = 4D, no obstacles (fixed layout) |
| `config/aligning-d3il-visual.py` | 738–740 | `ddpm_encdec_vision_nonvisual`: `obs_dim=20`, `action_dim=3` |
| `config/aligning-d3il-visual.py` | 351–352, 424–425 | visual blocks: `obs_dim=6` (robot only, camera does the rest) |
| `*/models/visual_unet.py` | non-visual branch | `transition_dim = action_dim + obs_dim = 3 + 20 = 23` |
| `d3il/simulation/aligning_sim.py` | reset/contexts | box/target pose randomized per context → must be observed |
