# Cameras in D3IL/MuJoCo and Their Use in Visual-Aligning DPCC / FM

**Audience:** anyone wiring camera data into FM-PCC visual variants.
**Scope:** what the two camera streams are, where they come from in the D3IL
sim, how each task wires them up (or doesn't), what the on-disk layout looks
like, and exactly how the visual-aligning DPCC/FM pipeline consumes them.

---

## 1. The Two Camera Streams

Visual-aligning (and any future visual-avoiding) expects **two RGB streams** per
timestep, stored in two parallel folders:

```
…/all_data/images/
    bp-cam/      env_000_00/0.png, 1.png, …      ← fixed external "cage" view
    inhand-cam/  env_000_00/0.png, 1.png, …      ← wrist-mounted view
```

Both are 96×96 PNGs encoded **BGR uint8** (cv2.imwrite convention). The dataset
loader reads them with `cv2.imread` and immediately swaps to RGB before
normalizing (`sequence.py:166-167`).

| Stream | Mounting | Naming | Movement |
|---|---|---|---|
| `bp-cam` | Cage-fixed external camera | "BP Cage Camera" | Static — same pose every frame |
| `inhand-cam` | End-effector mounted | Wrist / "in-hand" camera | Moves with the gripper |

These two views are deliberately complementary: the cage cam gives global scene
context (box positions, target), the wrist cam gives near-field interaction
detail (contact, fine alignment).

---

## 2. Where Each Camera Lives in MuJoCo

### 2.1 `bp_cam` — `BPCageCam` (env-level)

Defined **per task env**, identically in both aligning and avoiding:

```python
# d3il/environments/d3il/envs/gym_avoiding_env/.../avoiding.py:20
# d3il/environments/d3il/envs/gym_aligning_env/.../aligning.py:34
class BPCageCam(MjCamera):
    def __init__(self, width=1024, height=1024, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width, height,
            init_pos =[1.05, 0, 1.2],
            init_quat=[0.683, 0.183, 0.183, 0.683],  # ~30° tilt down at robot
        )
```

- **Pose**: 1.05 m in front of the robot base, 1.2 m up, tilted ~30° toward
  the workspace.
- **Native resolution**: 1024×1024 (we render at 96×96 for the dataset).
- **Mounting**: as a free MuJoCo object added to the scene
  (`self.scene.add_object(self.bp_cam)`), so it never moves.
- **Name origin**: D3IL leftover. The docstring just says "Cage camera"; "BP"
  appears across the codebase as a project-internal prefix (likely
  "Bird's-eye-Perspective" or initials). Not meaningful to consumers.

### 2.2 `inhand_cam` — `MjInhandCamera` (robot-level)

Defined on the **robot**, not the env:

```python
# d3il/environments/d3il/d3il_sim/sims/mj_beta/MjRobot.py:62
self.inhand_cam = MjInhandCamera(self.add_id2model_key("rgbd"))
```

`MjInhandCamera` (`MjCamera.py:165`) defaults to **96×96** and is *not* a free
object — it references a `<camera>` element with id `"rgbd"` baked into the
robot's Panda MJCF. So the cam transform follows the end-effector's transform
automatically; no Python plumbing is needed once the XML is in place.

**Key consequence:** *every* D3IL robot ships an `inhand_cam`. Whether a given
task exposes it depends on the task env file, not on the robot.

### 2.3 Both cameras use the same capture API

```python
# d3il/environments/d3il/d3il_sim/core/Camera.py:141
def get_image(self, width=None, height=None, depth=True, denormalize_depth=True) -> np.ndarray
```

- Default `depth=True` returns a **tuple** `(rgb_img, depth_img)` — useful for
  point clouds, **wrong** for plain RGB collection.
- For PNG collection, always call `get_image(width=W, height=H, depth=False)`
  → single `(H, W, 3)` RGB uint8 array.
- Output is **RGB**; convert with `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)`
  before `cv2.imwrite` (matches `aligning.py:211-212`).

---

## 3. Per-Task Wiring (Critical Asymmetry)

This is the part that bit us during avoiding collection.

### 3.1 Aligning env — **both cameras exposed**

```python
# d3il/environments/d3il/envs/gym_aligning_env/.../aligning.py:174-189
self.bp_cam     = BPCageCam()
self.inhand_cam = robot.inhand_cam            # ← re-export from robot
self.scene.add_object(self.bp_cam)
self.cam_dict = {
    "bp-cam":     CamLogger(scene, self.bp_cam),
    "inhand-cam": CamLogger(scene, self.inhand_cam),
}
```

D3IL's expert-collection pipeline (with CamLogger active) therefore wrote both
folders, and the published aligning dataset has real wrist views in
`inhand-cam/`.

### 3.2 Avoiding env — **only bp_cam exposed**

```python
# d3il/environments/d3il/envs/gym_avoiding_env/.../avoiding.py:82-87
self.bp_cam = BPCageCam()
self.scene.add_object(self.bp_cam)
self.cam_dict = {"bp-cam": CamLogger(scene, self.bp_cam)}
# NOTE: no self.inhand_cam assignment, no inhand-cam logger
```

The robot **still owns** `env.robot.inhand_cam` (`MjRobot.py:62`), but the env
never re-exports it, and no D3IL collection step ever wrote
`avoiding/.../inhand-cam/` to disk. That's why our replay+capture script
initially shipped with `inhand-cam/` as a *duplicate of bp-cam* — "Option B
placeholder" in the plan MD. The real wrist view is reachable via
`env.robot.inhand_cam.get_image(...)`; it just isn't standard for the avoiding
task.

---

## 4. On-Disk Format (canonical, set by aligning)

```
d3il/environments/dataset/data/<task>/all_data/
├── state/                            ← symlink or copy of per-episode .pkl
├── images/
│   ├── bp-cam/
│   │   ├── env_000_00/0.png, 1.png, … T-1.png
│   │   ├── env_001_00/…
│   │   └── …
│   └── inhand-cam/
│       ├── env_000_00/0.png, …
│       └── …
├── train_files.pkl                   ← list of .pkl names for training split
└── eval_files.pkl                    ← list for eval split
```

- One PNG per **timestep**, named by integer index (no zero-padding).
- Number of PNGs in `env_X/` must equal `len(state['robot']['des_c_pos']) - 1`
  (the replay loop runs `T = len(des_c_pos) - 1` steps).
- Image encoding: **BGR uint8**, 96×96.
- The two folders are aligned by **episode name and timestep index** — there
  is no manifest; the dataset just zips the two sequences.

---

## 5. How Visual-Aligning DPCC / FM Consumes Both Cameras

### 5.1 Dataset (`fm_visual_aligning/datasets/sequence.py`)

Per-episode arrays held in memory:

```python
self.bp_cam_imgs     = []   # list of (T_img_i, C, H, W) float tensors
self.inhand_cam_imgs = []
...
self.bp_cam_imgs.append(    self._load_images(data_dir, 'bp-cam',     file_name))
self.inhand_cam_imgs.append(self._load_images(data_dir, 'inhand-cam', file_name))
```

`_load_images` (sequence.py:154-167) does for each PNG:
`cv2.imread → cv2.cvtColor(BGR→RGB) → /255.0 → CHW float tensor`.

Each `__getitem__` returns the **window-start frame** (single timestep) of both
streams:

```python
{
    'trajectories': (T, 9),                       # actions + states
    'conditions':   {0: anchor_obs},              # DPCC apply_conditioning
    'primary_img':  bp_cam_imgs[ep][start],       # (C, H, W)  ← bp-cam
    'wrist_img':    inhand_cam_imgs[ep][start],   # (C, H, W)  ← inhand-cam
}
```

### 5.2 Diffusion engine (`visual_gaussian_diffusion.py:55-64`)

The two image tensors are packed into the `cond` dict the U-Net expects:

```python
primary_img = conditions['primary_img'].unsqueeze(1)   # (B, 1, C, H, W)
wrist_img   = conditions['wrist_img'].unsqueeze(1)     # (B, 1, C, H, W)
cond = {
    0: anchor_obs,                                  # DPCC obs anchor (state)
    'visual': (primary_img, wrist_img, obs_seq),    # tuple consumed by VisualUNet
}
```

### 5.3 Vision encoder (`visual_unet.py:30-103`)

`VisualUNet` builds a `MultiImageObsEncoder` with **two named RGB streams**:

```python
shape_meta = {
    'obs': {
        'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'},   # ← bp-cam
        'in_hand_image':   {'shape': [3, 96, 96], 'type': 'rgb'},   # ← inhand-cam
    }
}
obs_encoder_cfg = OmegaConf.create({
    '_target_': '…MultiImageObsEncoder',
    'shape_meta': shape_meta,
    'rgb_model':  {'_target_': '…get_resnet', 'input_shape': [3,96,96], 'output_size': 64},
    'share_rgb_model': False,         # ← two independent ResNet backbones
    'use_group_norm':  True,
    'imagenet_norm':   True,
})
```

Per-frame forward:

```python
# visual_unet.py:92-103
def encode_visual(self, bp_imgs, inhand_imgs):
    B, T, C, H, W = bp_imgs.shape
    obs_dict = {
        'agentview_image': bp_imgs.reshape(B*T, C, H, W),       # bp-cam → ResNet_A
        'in_hand_image':   inhand_imgs.reshape(B*T, C, H, W),   # inhand-cam → ResNet_B
    }
    features = self.obs_encoder(obs_dict)        # (B*T, 128)   [64 from each cam, concatenated]
    return features.view(B, T, -1).mean(dim=1)   # (B, 128)     [mean-pool over window]
```

So the **128-D visual latent** is `concat(bp_cam_feat_64, inhand_cam_feat_64)`,
mean-pooled across the time window.

### 5.4 FiLM injection into the temporal U-Net

The 128-D latent is fed as `cond_dim=128` into `UNet1DTemporalCondModel`
(`visual_unet.py:76-85`, `unet1d_temporal_cond.py:123-125`). At every ResNet
block in the trajectory U-Net, this latent is projected and used as **FiLM
scale+shift** parameters that modulate the residual stream. That is the only
path through which image content influences the denoiser — the trajectory
itself (`x`) remains 9-D `[action(3) | des_c_pos(3) | c_pos(3)]`.

```
┌──────────────────────┐    ┌────────────┐
│ bp-cam     96×96×3   │──▶ │ ResNet_A   │──┐
└──────────────────────┘    │ out 64-D   │  │
                            └────────────┘  │
                                            ├─▶ concat (128-D) ─▶ pool over T_win
┌──────────────────────┐    ┌────────────┐  │       │
│ inhand-cam 96×96×3   │──▶ │ ResNet_B   │──┘       │
└──────────────────────┘    │ out 64-D   │          │
                            └────────────┘          ▼
                                              cond_dim=128 → FiLM γ,β
                                                          │
trajectory x (B, T, 9) ─▶ UNet1D ResBlocks ◀──────────────┘
                                │
                                ▼
                            ε prediction (DDPM) / velocity (FM)
```

### 5.5 What this means for the DPCC contract

- **State conditioning is unchanged**: `apply_conditioning(x, cond)` still pins
  `x[:, 0, action_dim:] = anchor_obs`. The images do **not** replace the obs
  anchor — they are a *parallel* conditioning signal injected via FiLM.
- **Trajectory channel count is fixed at 9** in visual mode (`TRANSITION_DIM =
  9`); the box/target/state-of-the-world enters only through the image
  embeddings + the 6-D obs anchor at step 0. That is the design assumption
  behind the entire visual aligning architecture.

---

## 6. Implications for Visual Avoiding (Why `inhand-cam` Was Initially Bogus, Now Fixed)

Our collection script in `collect_visual_avoiding_data/` follows the aligning
on-disk format. The avoiding env doesn't expose `self.inhand_cam`, so an
initial version (**Option B placeholder**) wrote the bp-cam frame into both
folders so the pipeline at least *ran*.

**Now switched to Option A.** The wrist view lives on `env.robot.inhand_cam`
(`MjRobot.py:62`) and is already attached to the scene by `MjScene.py:65` for
every registered robot. The collection script now reads it directly:

```python
inhand = env.robot.inhand_cam.get_image(width=resolution, height=resolution, depth=False)
inhand = cv2.cvtColor(inhand, cv2.COLOR_RGB2BGR)
```

`save_frames` writes the two streams to their own folders. **Zero D3IL files
were modified.** Open question — *is the wrist view actually useful for an
avoiding task?* — see §7.3 below.

---

## 7. Why Both Cameras? (Design Rationale + D3IL Provenance)

### 7.1 Why `bp-cam` exists alongside `inhand-cam`

The two streams are deliberately **complementary**, not redundant:

| Stream | What it sees | What it's good for | What it misses |
|---|---|---|---|
| `bp-cam` (cage, fixed third-person) | Whole tabletop, all objects, both targets, obstacles, robot body | Global scene state, mode commitment ("which box → which target"), long-horizon planning | Fine contact, near-field geometry under the gripper |
| `inhand-cam` (wrist, robot-mounted) | Whatever is directly under the end-effector | Fine alignment, grasp verification, contact-rich phases | Anything outside the wrist FOV — the rest of the world |

Without `bp-cam`, the policy would have to *infer* global scene state from a
narrow, moving wrist view — much harder and ambiguous (multiple global states
project to the same wrist image). Without `inhand-cam`, fine alignment becomes
guesswork at the resolution the cage cam can offer at distance. Standard
third-person + first-person split, used across imitation-learning robotics.

For the aligning task specifically:
- `bp-cam` answers: "where are the two push-boxes, where are their targets,
  which box should I grab first?"
- `inhand-cam` answers: "am I aligned with the box edge right now, am I about
  to push at the right contact point?"

### 7.2 D3IL itself uses both cameras (we inherited the design)

This **isn't an FM-PCC invention**. Every D3IL vision agent consumes the same
two-stream `obs_dict`:

```python
# Identical structure across all five D3IL vision agents:
obs_dict = {
    "agentview_image": agentview_image,   # ← bp-cam      (B*T, 3, 96, 96)
    "in_hand_image":   in_hand_image,     # ← inhand-cam  (B*T, 3, 96, 96)
}
```

Verified locations in `d3il/agents/`:

| File | Line | Agent |
|---|---|---|
| `bc_agent.py` | 41-53 | Behaviour cloning |
| `bet_agent.py` | 35-44, 66-75 | BeT |
| `act_vision_agent.py` | 38-50, 68-80 | ACT |
| `gpt_bc_vision_agent.py` | 33-45 | GPT-BC |
| `ddpm_encdec_vision_agent.py` | 36-41 | DDPM encoder–decoder |

FM-PCC visual aligning (`fm_visual_aligning/models/visual_unet.py:30-103`)
keeps **the exact same key names** (`agentview_image`, `in_hand_image`) so the
shared `MultiImageObsEncoder` can be reused as-is. The dataset just renames
the **output side** to `primary_img` / `wrist_img` for clarity in the diffusion
engine; the encoder still receives the D3IL keys.

So:
- The two-cam recipe is **D3IL's baseline design**, not ours.
- Our FiLM injection of the resulting 128-D embedding is the
  diffusion-policy-style adaptation; D3IL's own vision agents use the same
  embeddings differently (e.g. concatenated into the transformer token stream
  for ACT/BeT), but the **camera plumbing is identical**.

This also explains why the aligning env exposes `inhand_cam` at the env level
while avoiding does not: the D3IL maintainers only published image-based
baselines for *aligning*, so they didn't bother re-exporting the wrist cam in
the avoiding env. The cam still exists on the robot — it just was never on
their data-collection path.

### 7.3 Opinion: would `inhand-cam`-only be better than two-stream? (per-task)

Honest take, not just rubber-stamping the intuition that "wrist cam feels more
useful because it's closer to the action." Per task:

#### Visual **aligning** — keep both. Inhand-only would likely fail.

The aligning task has **multi-modal expert demos**: the demonstrator picks
*which* push-box to align first (left or right), and the policy must commit to
one mode at episode start. That mode-commitment signal lives in the **global
scene**: which box is where, which target is where. The wrist cam, pointed
straight down from ~12 cm above the EE, only sees a small patch of table —
likely **not** both boxes at once at t=0. Without the cage view, the policy
has no way to look at the whole scene and decide. Drop bp-cam → expect mode
collapse or wrong-box selection.

Conversely, dropping inhand-cam should hurt fine-alignment at contact but not
break the high-level decision. So for aligning: **bp-cam is the load-bearing
stream; inhand-cam is the precision booster.**

#### Visual **avoiding** — bp-cam-only is probably the right choice; inhand-only would be the *worst* of the three.

This is the counter-intuitive one. Reasoning:

1. **The task is non-contact.** The gripper hovers ~12 cm above the table and
   translates in XY between fixed obstacle pegs to reach a goal line. There is
   no contact event for the wrist cam to disambiguate. The wrist cam sees
   "table moving past" — mostly featureless wood, with pegs sliding briefly
   in/out of frame only when the EE is directly over them.
2. **Mode commitment is even harder than aligning.** The avoiding task has
   *more* expert modes (multiple homotopy classes of paths through the peg
   field, demonstrator-dependent), and *all* of them are determined by the
   path the gripper takes *around* the pegs ahead. The wrist cam **cannot see
   the pegs ahead** until the gripper is on top of them — by which time it's
   too late to choose a homotopy class. Bp-cam sees the whole peg layout from
   frame 0.
3. **Goal location is far from the gripper.** The goal line is at the far end
   of the table; the wrist cam can't see it until the EE is almost there. The
   bp-cam sees it from start to finish.
4. **Wrist-cam adds little novel content for this task.** Because the EE keeps
   a fixed quaternion `[0,1,0,0]` (top-down) and fixed z, the wrist image is
   basically a translated crop of the same world the cage cam already sees.
   Information-theoretically, the marginal value of the wrist cam over the
   cage cam is small.

So if you wanted to **simplify** the (future, hypothetical) visual-avoiding
model, **drop the wrist cam, not the cage cam.** That's the opposite of the
user's hunch — but the hunch was based on the aligning-task intuition
("contact = wrist matters"), which doesn't transfer to a non-contact task.

#### Why we still collect both for avoiding right now

Three pragmatic reasons:
- **Disk layout symmetry with aligning** — same loader works, no per-task
  branch needed.
- **Cost is trivial** — second `get_image` call adds milliseconds per step;
  total avoiding collection is still under 5 s of compute per episode.
- **Optionality is cheap before training** — keeping the wrist stream on disk
  costs ~50 MB total and lets a future ablation flip a flag instead of
  re-running collection. Removing it later is one `rm -rf`; adding it later
  is another full SLURM run.

#### Quick recommendation table

| Task | Best single-cam (if forced to pick one) | Drop without much loss | Drop and break the model |
|---|---|---|---|
| Aligning | bp-cam | — | inhand-cam (loses fine alignment) **or** bp-cam (loses mode commit) — both meaningful |
| Avoiding | bp-cam | inhand-cam (low marginal info) | bp-cam (no global view, no goal, no peg layout) |

#### Bottom line

The user's intuition "bp-cam feels ineffective" is **plausible for aligning**
(where the wrist contact is dramatic and load-bearing) but **inverted for
avoiding** (where there is no contact and the cage view is the only stream
that sees the whole peg layout + goal). For our data-collection-only practice
scope this is moot, but if a real visual-avoiding model is ever built, the
recommended ablation order is: start with **bp-cam-only**, then add inhand-cam
only if a measurable failure mode demands it.

---

## 8. Reference: File / Line Map

| Concern | File | Line |
|---|---|---|
| BPCageCam definition (avoiding) | `d3il/.../gym_avoiding/envs/avoiding.py` | 20 |
| BPCageCam definition (aligning) | `d3il/.../gym_aligning/envs/aligning.py` | 34 |
| bp_cam wired into env (avoiding) | `d3il/.../avoiding.py` | 82-87 |
| bp_cam + inhand_cam wired (aligning) | `d3il/.../aligning.py` | 174-189 |
| Robot-level `inhand_cam` definition | `d3il/.../mj_beta/MjRobot.py` | 62 |
| MjInhandCamera class | `d3il/.../mj_beta/MjCamera.py` | 165 |
| `Camera.get_image` (tuple-vs-array gotcha) | `d3il/.../core/Camera.py` | 141 |
| Aligning's RGB→BGR convention | `d3il/.../aligning.py` | 211-214 |
| Visual aligning dataset load | `fm_visual_aligning/datasets/sequence.py` | 102-167 |
| `cond['visual']` packing | `fm_visual_aligning/models/visual_gaussian_diffusion.py` | 41-64 |
| Two-stream encoder + FiLM wiring | `fm_visual_aligning/models/visual_unet.py` | 30-129 |
| FiLM projection in UNet | `fm_visual_aligning/models/unet1d_temporal_cond.py` | 123-125 |
