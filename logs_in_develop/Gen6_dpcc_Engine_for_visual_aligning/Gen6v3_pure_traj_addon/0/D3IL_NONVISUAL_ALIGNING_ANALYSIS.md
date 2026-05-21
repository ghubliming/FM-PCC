# D3IL Non-Visual Aligning Analysis & DPCC Investigation (Aligning vs Avoiding)

## Executive Summary
This document merges and unifies two critical investigations into the state-only (non-visual) aligning and avoiding pipelines within the `D3IL` and `DPCC` frameworks. 

1. **Non-Visual is NOT Simple (Hidden Complexity):** While the state-only aligning task appears simple, there is a significant mismatch between the training dataset (20D state) and the evaluation loop (17D environment state padded to 20D using the previous action) that exposes hidden complexity, potential distribution shift, and shape/dimension tracking bugs.
2. **DPCC Diffuser vs. D3IL (Aligning vs. Avoiding):** `DPCC` works cleanly with D3IL *avoiding* because DPCC's dataset adapter explicitly converts D3IL pickles into the small 4D state and 2D action arrays DPCC expects. However, Aligning's non-visual path is not a straightforward drop-in. We must either:
   - **Option A (Recommended, minimal invasive):** Implement a DPCC adapter for aligning similar to `d4rl.py`'s `avoiding-d3il` branch (handling the 17D -> 20D padding and 20D trajectory representation).
   - **Option B (Larger change):** Fix D3IL environment/eval to consistently expose `desired_pos` at runtime.

---

## Table of Contents
- [Part I: D3IL Non-Visual Aligning Task Analysis](#part-i-d3il-non-visual-aligning-task-analysis)
  - [1. Aligning Dataset/Loader: State-Only Version](#1-aligning-datasetloader-state-only-version)
  - [2. Aligning Environment: `get_observation()` Logic](#2-aligning-environment-get_observation-logic)
  - [3. Agent Handling: Training & Inference Paths](#3-agent-handling-training--inference-paths)
  - [4. Evaluation Loop: The Critical Mismatch](#4-evaluation-loop-the-critical-mismatch)
  - [5. Comparison: Visual vs. Non-Visual Symmetry](#5-comparison-visual-vs-non-visual-symmetry)
  - [6. Root Cause: Why Non-Visual is NOT Simple](#6-root-cause-why-non-visual-is-not-simple)
  - [7. Specific Code Evidence](#7-specific-code-evidence)
  - [8. Original Analysis Conclusion & Recommendations](#8-original-analysis-conclusion--recommendations)
- [Part II: DPCC vs. D3IL Investigation (Aligning vs. Avoiding)](#part-ii-dpcc-vs-d3il-investigation-aligning-vs-avoiding)
  - [1. How DPCC Uses the avoiding-d3il Dataset](#1-how-dpcc-uses-the-avoiding-d3il-dataset)
  - [2. D3IL Native Avoiding Implementation](#2-d3il-native-avoiding-implementation)
  - [3. D3IL Aligning Implementation (State-Only and Image-Enabled)](#3-d3il-aligning-implementation-state-only-and-image-enabled)
  - [4. Aligning vs. Avoiding: Structural Comparison](#4-aligning-vs-avoiding-structural-comparison)
  - [5. How to Make Non-Visual Aligning Work in DPCC/Gen6](#5-how-to-make-non-visual-aligning-work-in-dpccgen6)
  - [6. Final Recommendation & Action Plan](#6-final-recommendation--action-plan)
  - [7. References (Code Locations)](#7-references-code-locations)

---

## Part I: D3IL Non-Visual Aligning Task Analysis

### 1. Aligning Dataset/Loader: State-Only Version

#### Aligning_Dataset (Non-Visual)
* **File:** [d3il/environments/dataset/aligning_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/aligning_dataset.py#L18)

```python
class Aligning_Dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 20,           # ← FIXED: 20D STATE VECTOR
            action_dim: int = 2,
            max_len_data: int = 256,
            window_size: int = 1,
    ):
```

##### Data Loading & Processing
```python
# Load state data
rp_data_dir = sim_framework_path("environments/dataset/data/aligning/all_data/state")
state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

for file in state_files:
    with open(os.path.join(rp_data_dir, file), 'rb') as f:
        env_state = pickle.load(f)

    # BUILD 20D STATE OBSERVATION
    robot_des_pos = env_state['robot']['des_c_pos']          # 3D
    robot_c_pos = env_state['robot']['c_pos']                # 3D
    push_box_pos = env_state['push-box']['pos']              # 3D
    push_box_quat = env_state['push-box']['quat']            # 4D
    target_box_pos = env_state['target-box']['pos']          # 3D
    target_box_quat = env_state['target-box']['quat']        # 4D
    
    # Total: 3+3+3+4+3+4 = 20D
    input_state = np.concatenate(
        (robot_des_pos, robot_c_pos, push_box_pos, push_box_quat, 
         target_box_pos, target_box_quat), 
        axis=-1
    )

    # ACTION: Velocity of desired position
    vel_state = robot_des_pos[1:] - robot_des_pos[:-1]  # 2D (x, y) only!
    
    zero_obs[0, :valid_len, :] = input_state[:-1]
    zero_action[0, :valid_len, :] = vel_state
    zero_mask[0, :valid_len] = 1
```

##### `__getitem__` Return Format
```python
def __getitem__(self, idx):
    i, start, end = self.slices[idx]

    obs = self.observations[i, start:end]        # Shape: [T, 20]
    act = self.actions[i, start:end]             # Shape: [T, 2]
    mask = self.masks[i, start:end]              # Shape: [T]

    return obs, act, mask
```

> [!NOTE]
> **Non-visual dataset is straightforward:**
> * Returns: `(obs, action, mask)` tuples
> * Observation: 20D state vector (robot + boxes)
> * Action: 2D velocity
> * No preprocessing beyond concatenation

---

#### Aligning_Img_Dataset (Visual Reference)
For comparison, the visual version returns:
```python
def __getitem__(self, idx):
    i, start, end = self.slices[idx]
    
    obs = self.observations[i, start:end]        # Shape: [T, 20]
    act = self.actions[i, start:end]             # Shape: [T, 2]
    mask = self.masks[i, start:end]              # Shape: [T]
    
    bp_imgs = self.bp_cam_imgs[i][start:end]     # Shape: [T, 3, 96, 96]
    inhand_imgs = self.inhand_cam_imgs[i][start:end]  # Shape: [T, 3, 96, 96]

    return bp_imgs, inhand_imgs, obs, act, mask
```

The state observation is **identical** in both—no visual-specific preprocessing.

---

### 2. Aligning Environment: `get_observation()` Logic

* **File:** [d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py](file:///workspaces/FM-PCC/d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py)

```python
class Robot_Push_Env(GymEnvWrapper):
    def __init__(self, ..., if_vision: bool = False):
        self.if_vision = if_vision
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(8, ))
        # NOTE: observation_space claims 8D, but actual obs is 20D!
```

#### `get_observation()` Method
```python
def get_observation(self) -> np.ndarray:
    robot_pos = self.robot_state()  # 3D vector (TCP position)

    if self.if_vision:
        # VISUAL PATH: Return 3 items
        bp_image = self.bp_cam.get_image(depth=False)
        bp_image = cv2.cvtColor(bp_image, cv2.COLOR_RGB2BGR)
        
        inhand_image = self.inhand_cam.get_image(depth=False)
        inhand_image = cv2.cvtColor(inhand_image, cv2.COLOR_RGB2BGR)
        
        return robot_pos, bp_image, inhand_image  # ← Returns TUPLE
    
    else:
        # NON-VISUAL PATH: Build 20D state vector
        box_pos = self.scene.get_obj_pos(self.push_box)
        box_quat = self.scene.get_obj_quat(self.push_box)
        
        target_pos = self.scene.get_obj_pos(self.target_box)
        target_quat = self.scene.get_obj_quat(self.target_box)
        
        # Total: 3 + 3 + 4 + 3 + 4 = 17D (NOT 20D!)
        env_state = np.concatenate(
            [
                robot_pos,           # 3D
                box_pos,             # 3D
                box_quat,            # 4D
                target_pos,          # 3D
                target_quat          # 4D
            ]
        )
        
        return env_state.astype(np.float32)  # ← Returns single ARRAY
```

> [!WARNING]
> **PROBLEM DETECTED:**
> * **Training dataset expects:** 20D observation from `env_state`
>   * `robot_des_pos` (3D) + `robot_c_pos` (3D) + `push_box_pos` (3D) + `push_box_quat` (4D) + `target_box_pos` (3D) + `target_box_quat` (4D)
>   
> * **Simulation returns:** 17D observation from `get_observation()`
>   * `robot_pos` (3D) + `box_pos` (3D) + `box_quat` (4D) + `target_pos` (3D) + `target_quat` (4D)
>   * **Missing:** `robot_c_pos` and `robot_des_pos` are conflated with a single `robot_pos`.

---

### 3. Agent Handling: Training & Inference Paths

#### Base Agent: `train()` Method Dispatch
* **File:** [d3il/agents/base_agent.py](file:///workspaces/FM-PCC/d3il/agents/base_agent.py#L75)

```python
def train(self):
    if self.model.visual_input:
        self.train_vision_agent()
    else:
        self.train_agent()
```

#### State-Only Agent: `ddpm_encdec_vision_agent.py`
* **File:** [d3il/agents/ddpm_encdec_vision_agent.py](file:///workspaces/FM-PCC/d3il/agents/ddpm_encdec_vision_agent.py#L175)

##### `train_agent()` - Non-Visual Training
```python
def train_agent(self):
    for num_epoch in tqdm(range(self.epoch)):
        if not (num_epoch+1) % self.eval_every_n_epochs:
            test_mse = []
            for data in self.test_dataloader:
                if self.goal_condition:
                    state, action, mask, goal = data
                    mean_mse = self.evaluate(state, action, goal)
                else:
                    state, action, mask = data      # ← 20D state vector
                    mean_mse = self.evaluate(state, action)
                test_mse.append(mean_mse)
            
            train_loss = []
            for data in self.train_dataloader:
                if self.goal_condition:
                    state, action, mask, goal = data
                    batch_loss = self.train_step(state, action, goal)
                else:
                    state, action, mask = data      # ← 20D state vector
                    batch_loss = self.train_step(state, action)
```

##### `train_vision_agent()` - Visual Training
```python
def train_vision_agent(self):
    for data in self.train_dataloader:
        bp_imgs, inhand_imgs, obs, action, mask = data
        
        bp_imgs = bp_imgs.to(self.device)
        inhand_imgs = inhand_imgs.to(self.device)
        
        obs = self.scaler.scale_input(obs)
        action = self.scaler.scale_output(action)
        
        action = action[:, self.obs_seq_len - 1:, :].contiguous()
        obs = obs[:, :self.obs_seq_len].contiguous()
        
        state = (bp_imgs, inhand_imgs, obs)
        batch_loss = self.train_step(state, action)
```

> [!NOTE]
> **Key Difference:**
> * Non-visual trains on raw 20D state vectors.
> * Visual trains on tuple `(bp_imgs, inhand_imgs, obs)`.

##### `predict()` - Single Method, Two Paths
```python
@torch.no_grad()
def predict(self, state: torch.Tensor, goal=None, if_vision=False):
    
    if if_vision:
        # VISUAL INFERENCE
        bp_image, inhand_image, des_robot_pos = state
        
        bp_image = torch.from_numpy(bp_image).unsqueeze(0)
        inhand_image = torch.from_numpy(inhand_image).unsqueeze(0)
        des_robot_pos = torch.from_numpy(des_robot_pos).unsqueeze(0)
        
        des_robot_pos = self.scaler.scale_input(des_robot_pos)
        
        self.bp_image_context.append(bp_image)
        self.inhand_image_context.append(inhand_image)
        self.des_robot_pos_context.append(des_robot_pos)
        
        bp_image_seq = torch.stack(tuple(self.bp_image_context), dim=1)
        inhand_image_seq = torch.stack(tuple(self.inhand_image_context), dim=1)
        des_robot_pos_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
        
        input_state = (bp_image_seq, inhand_image_seq, des_robot_pos_seq)
    
    else:
        # NON-VISUAL INFERENCE: 20D→17D MISMATCH HERE
        obs = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        obs = self.scaler.scale_input(obs)
        self.obs_context.append(obs)
        input_state = torch.stack(tuple(self.obs_context), dim=1)
```

---

### 4. Evaluation Loop: The Critical Mismatch

* **File:** [d3il/simulation/aligning_sim.py](file:///workspaces/FM-PCC/d3il/simulation/aligning_sim.py#L63)

#### Visual Evaluation Path ✅ (Correct)
```python
if self.if_vision:
    env_state, bp_image, inhand_image = obs
    bp_image = bp_image.transpose((2, 0, 1)) / 255.
    inhand_image = inhand_image.transpose((2, 0, 1)) / 255.
    
    des_robot_pos = env_state[:3]
    done = False
    
    while not done:
        pred_action = agent.predict(
            (bp_image, inhand_image, des_robot_pos), 
            if_vision=self.if_vision
        )
        pred_action = pred_action[0] + des_robot_pos
        pred_action = np.concatenate((pred_action, [0, 1, 0, 0]), axis=0)
        obs, reward, done, info = env.step(pred_action)
        
        des_robot_pos = pred_action[:3]
        robot_pos, bp_image, inhand_image = obs
        
        bp_image = bp_image.transpose((2, 0, 1)) / 255.
        inhand_image = inhand_image.transpose((2, 0, 1)) / 255.
```

#### Non-Visual Evaluation Path ❌ (PROBLEMATIC)
```python
else:
    # BUG: robot_state() returns 3D, but obs is 17D (from get_observation)
    pred_action = env.robot_state()  # 3D TCP position
    done = False
    while not done:
        # MISMATCH: Concatenating 3D action with 17D obs = 20D
        # But agent was trained on 20D observations from dataset!
        obs = np.concatenate((pred_action[:3], obs))  # ← 3D + 17D = 20D!
        
        pred_action = agent.predict(obs)
        pred_action = pred_action[0] + obs[:3]
        
        pred_action = np.concatenate((pred_action, [0, 1, 0, 0]), axis=0)
        obs, reward, done, info = env.step(pred_action)
```

#### Hidden Complexity Breakdown
Line-by-line what happens in the state-only path:

1. **Initial reset:**
   ```python
   obs = env.reset()  # Returns 17D: robot_pos + boxes (no robot_des_pos or robot_c_pos)
   ```
2. **First iteration:**
   ```python
   pred_action = env.robot_state()  # 3D TCP position
   obs = np.concatenate((pred_action[:3], obs))  # 3D + 17D = 20D ✓
   ```
3. **Agent prediction:**
   ```python
   pred_action = agent.predict(obs)  # Expects 20D! ✓
   pred_action = pred_action[0] + obs[:3]  # pred_action[0] is 2D, obs[:3] is 3D?
   ```
4. **Next iteration:**
   ```python
   obs = env.step(pred_action)  # Returns 17D again
   obs = np.concatenate((pred_action[:3], obs))  # But now pred_action might be 7D!
   ```

---

### 5. Comparison: Visual vs. Non-Visual Symmetry

#### Data Loading
| Aspect | Visual | Non-Visual |
|:---|:---|:---|
| **Dataset returns** | `(bp_imgs, inhand_imgs, obs, act, mask)` | `(obs, act, mask)` |
| **Observation dim** | 20D (same as non-visual) | 20D |
| **Action dim** | 2D | 2D |
| **Image processing** | High-level (loading from disk) | N/A |

#### Training
| Aspect | Visual | Non-Visual |
|:---|:---|:---|
| **Method** | `train_vision_agent()` | `train_agent()` |
| **Input** | `(bp_imgs, inhand_imgs, obs)` tuple | flat 20D array |
| **Encoding** | Image encoder + state encoder | Direct state encoder |
| **Complexity** | Higher (vision + state) | Lower (state only) |

#### Evaluation Loop
| Aspect | Visual | Non-Visual |
|:---|:---|:---|
| **obs_reset** | 17D from env | 17D from env |
| **State construction** | `(img, img, robot_pos)` | Concatenate action + obs |
| **Input shape** | Tuple, fully specified | Array, unclear structure |
| **Dim mismatch** | None | 3D action manually concat'd |
| **Training/eval parity** | High ✓ | **Low ❌** |

---

### 6. Root Cause: Why Non-Visual is NOT Simple

#### The Hidden Complexity
* **Training observes:**
  ```text
  obs_state = [robot_des_pos(3), robot_c_pos(3), box_pos(3), box_quat(4), 
               target_pos(3), target_quat(4)] = 20D
  action = [vx, vy] = 2D (computed as des_pos differences)
  ```
* **Evaluation observes:**
  ```text
  env_state = [robot_pos(3), box_pos(3), box_quat(4), target_pos(3), 
               target_quat(4)] = 17D
  Current obs = [pred_action_prev[0:3](3), env_state(17)] = 20D
  ```

#### Critical Issues
1. **On paper:** Non-visual appears symmetric (all state-based, no vision).
2. **In practice:** 
   * Training dataset has 20D state built from recorded episode data (`des_pos` + actual `pos`).
   * Evaluation loop concatenates the **previous action** (reused as position) with observation.
   * This creates a **learned implicit dependency**: actor learns to treat first 3 dims as previous robot command.
3. **Dimension tracking is obscure:** Who's tracking what the first 3D of obs mean?
4. **Action space confusion:** During evaluation, `pred_action = pred_action[0] + obs[:3]` assumes `pred_action` has shape `[1, 2]`, but shape could be different.

---

### 7. Specific Code Evidence

#### Dataset State Composition (20D)
```python
# From aligning_dataset.py:66-72
robot_des_pos = env_state['robot']['des_c_pos']    # 3D
robot_c_pos = env_state['robot']['c_pos']          # 3D
push_box_pos = env_state['push-box']['pos']        # 3D
push_box_quat = env_state['push-box']['quat']      # 4D
target_box_pos = env_state['target-box']['pos']    # 3D
target_box_quat = env_state['target-box']['quat']  # 4D
# Total: 20D ✓
```

#### Simulation State Composition (17D)
```python
# From aligning.py:226-235
robot_pos = self.robot_state()      # 3D
box_pos = self.scene.get_obj_pos(self.push_box)   # 3D
box_quat = self.scene.get_obj_quat(self.push_box) # 4D
target_pos = self.scene.get_obj_pos(self.target_box)      # 3D
target_quat = self.scene.get_obj_quat(self.target_box)    # 4D
# Total: 17D ❌ (missing robot_c_pos and robot_des_pos split)
```

#### Evaluation Loop Construction (20D via concatenation)
```python
# From aligning_sim.py:104-107
if not self.if_vision:
    pred_action = env.robot_state()  # 3D
    obs = np.concatenate((pred_action[:3], obs))  # 3D action + 17D obs = 20D
```

---

### 8. Original Analysis Conclusion & Recommendations

#### **Is non-visual aligning simple?**
**NO.** While it *appears* simple compared to vision-based aligning, there is **significant hidden complexity**:
1. **Data/Eval Mismatch:** Training dataset provides 20D state with both desired and actual robot positions, while evaluation provides 17D state without that distinction.
2. **Implicit Structure:** The evaluation loop implicitly uses the previous action as filler for the missing "desired position" dimension, but this coupling is not documented.
3. **Potential Bugs:**
   * Shape confusion: `pred_action = pred_action[0] + obs[:3]` assumes specific batch dimensions.
   * The agent must learn what the first 3 dims of its input actually represent during evaluation.
   * If agent trained on `des_pos` + `c_pos` but tested on `prev_action` + `c_pos`, there's a distribution shift.
4. **Not Symmetric:** Visual evaluation is well-structured (clear tuple of images + state). Non-visual evaluation constructs observations through array concatenation with implicit semantics.

#### **Recommendations:**
- [ ] Map dimensions explicitly in non-visual evaluation loop.
- [ ] Add assertions checking observation shapes match expected dimensions.
- [ ] Document what each dimension of the 20D observation represents during evaluation.
- [ ] Consider whether training on actual dataset (`des_pos` + `c_pos`) translates to evaluation logic (`prev_action` + `c_pos`).

---
---

## Part II: DPCC vs. D3IL Investigation (Aligning vs. Avoiding)

### 1. How DPCC Uses the avoiding-d3il Dataset

* **Key File:** [dpcc/diffuser/datasets/d4rl.py](file:///workspaces/dpcc/diffuser/datasets/d4rl.py)

#### Relevant Excerpt (DPCC):
```python
# dpcc/diffuser/datasets/d4rl.py
elif env == 'avoiding-d3il' or env == 'd3il-avoiding':
    data_directory = 'environments/dataset/data/avoiding/data'
    data_dir = sim_framework_path(data_directory)
    state_files = os.listdir(data_dir)

    for file in state_files:
        with open(os.path.join(data_dir, file), 'rb') as f:
            env_state = pickle.load(f)

            robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
            robot_c_pos = env_state['robot']['c_pos'][:, :2]

            input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)

            vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
            valid_len = len(vel_state)

        episode_data = {
            'observations': input_state[:-1],
            'actions': vel_state,
            'rewards': np.zeros(valid_len),
            'terminals': np.concatenate((np.zeros(valid_len-1), np.array([1])))
        }

        yield episode_data
```

#### Notes:
* DPCC reads the D3IL pickles and explicitly constructs a 4D `observations` (concatenation of desired and current robot pos, 2D each) and 2D `actions` (= velocity deltas).
* These are produced as Numpy arrays and later fed into DPCC's `SequenceDataset` which normalizes and concatenates `[actions, observations]` into `trajectories`.
* Files [dpcc/diffuser/datasets/sequence.py](file:///workspaces/dpcc/diffuser/datasets/sequence.py) and normalization are built to accept this shape.

---

### 2. D3IL Native Avoiding Implementation

* **Key File:** [d3il/environments/dataset/avoiding_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/avoiding_dataset.py)

#### Relevant Excerpt (D3IL):
```python
# d3il/environments/dataset/avoiding_dataset.py
robot_des_pos = env_state['robot']['des_c_pos'][:, :2]
robot_c_pos = env_state['robot']['c_pos'][:, :2]
input_state = np.concatenate((robot_des_pos, robot_c_pos), axis=-1)
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
# then padded into zero_obs / zero_action arrays and stored as tensors
```

#### Notes:
* D3IL's `Avoiding_Dataset` produces padded tensors `(B, T, obs_dim)` and `(B, T, action_dim)`, and `__getitem__` returns `(obs, act, mask)`.
* Structure and temporal semantics are identical to what DPCC expects after DPCC's adapter.

---

### 3. D3IL Aligning Implementation (State-Only and Image-Enabled)

* **Key Files:** [d3il/environments/dataset/aligning_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/aligning_dataset.py)

#### Relevant Excerpt:
##### State-Only Path (Aligning_Dataset):
```python
# Aligning_Dataset
robot_des_pos = env_state['robot']['des_c_pos']
robot_c_pos = env_state['robot']['c_pos']
push_box_pos = env_state['push-box']['pos']
push_box_quat = env_state['push-box']['quat']
target_box_pos = env_state['target-box']['pos']
target_box_quat = env_state['target-box']['quat']

input_state = np.concatenate((robot_des_pos, robot_c_pos, push_box_pos,
                              push_box_quat, target_box_pos, target_box_quat), axis=-1)
vel_state = robot_des_pos[1:] - robot_des_pos[:-1]
# valid_len = len(input_state) - 1
# padded into zero_obs, zero_action
```

##### Image-Enabled Path (Aligning_Img_Dataset)
Returns `(bp_imgs, inhand_imgs, obs, act, mask)`; `__getitem__` returns these five elements.

#### Important Evaluation-Time Quirk (Distribution Mismatch)
* In several D3IL evaluation scripts (simulation/eval loops) the environment's `get_observation()` returns a 17D `env_state` (it contains `robot_pos`, box positions, quaternions, target positions/quaternions — but no explicit `desired_pos` channel).
* The aligning dataset expects 20D `input_state` that includes `robot_des_pos` and `robot_c_pos` (two different position signals).
* To reconcile this at evaluation time, some eval code concatenates the previous predicted action (first 3 dims) with the environment state to produce a 20D vector: e.g., `obs = np.concatenate((pred_action[:3], obs))`.

> [!WARNING]
> This means training-time observations and evaluation-time observations are assembled differently (training uses `desired_pos`, eval uses `prev_action`), creating a semantic mismatch in the first 3 dims.

---

### 4. Aligning vs. Avoiding: Structural Comparison

* **Observations:**
  * **Avoiding:** 4D (`desired_pos[:2]` + `current_pos[:2]`) → produced and consumed uniformly by both D3IL and DPCC adapter.
  * **Aligning:** 20D (`robot_des_pos` + `robot_c_pos` + `push_box_pos` + `push_box_quat` + `target_box_pos` + `target_box_quat`).
* **Actions:**
  * **Avoiding:** 2D velocity (`robot_des_pos` difference)
  * **Aligning:** 2D velocity (`robot_des_pos` difference)
* **Data Pipeline:**
  * **Avoiding:** consistent between dataset and environment; DPCC adapter reads pickles and makes exactly the small arrays DPCC expects.
  * **Aligning:** dataset produces 20D padded tensors; environment/eval sometimes supplies 17D and relies on previous-action padding to reach 20D.
* **Images:**
  * **Avoiding:** no images in native D3IL avoiding dataset.
  * **Aligning:** has both a state-only dataset and an image-enabled dataset; visual version returns `(bp_imgs, inhand_imgs, obs, act, mask)`.

---

### 5. How to Make Non-Visual Aligning Work in DPCC/Gen6

#### Option A — Adapter Approach (Recommended, Minimal Invasive)
* Implement a DPCC adapter for `aligning-d3il` similar to `d4rl.py`'s `avoiding-d3il` branch:
  * Read Aligning pickles.
  * Construct `input_state` the same way as D3IL's `Aligning_Dataset` (20D).
  * Construct `vel_state = robot_des_pos[1:] - robot_des_pos[:-1]` as actions.
  * Yield `{'observations': input_state[:-1], 'actions': vel_state, ...}` so DPCC's `SequenceDataset` can consume it unchanged.
* Additionally, ensure evaluation code (when running in sim) uses the same 20D conditioning. If the sim returns 17D, replicate D3IL's eval padding (prepend previous action) to match training.

#### Option B — Fix D3IL (Larger Change)
* Modify D3IL environment/eval to consistently expose `desired_pos` at runtime (so no previous-action hack is needed). This is cleaner but more invasive.

#### Option C — Modify Gen6 Model
* Modify Gen6 model to accept mixed condition semantics (not recommended).

---

### 6. Final Recommendation & Action Plan

1. **Successful Baseline:** DPCC's diffuser successfully uses `avoiding-d3il` because DPCC explicitly adapts the raw pickles into small state/action arrays.
2. **Path to Aligning:** Aligning non-visual **can** work, but only if we perform the same adapter step for aligning and carefully match training vs. evaluation semantics (handle the 17D→20D padding). 
3. **Action:** Implementing a DPCC adapter for aligning (**Option A**) is the fastest path and keeps Gen6/DPCC code simple.

---

### 7. References (Code Locations)

* **DPCC Adapter:** [dpcc/diffuser/datasets/d4rl.py](file:///workspaces/dpcc/diffuser/datasets/d4rl.py)
* **DPCC Dataset Wrapper:** [dpcc/diffuser/datasets/sequence.py](file:///workspaces/dpcc/diffuser/datasets/sequence.py)
* **D3IL Avoiding Dataset:** [d3il/environments/dataset/avoiding_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/avoiding_dataset.py)
* **D3IL Aligning Dataset:** [d3il/environments/dataset/aligning_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/aligning_dataset.py)

---
*Merged and updated successfully on 2026-05-18.*
