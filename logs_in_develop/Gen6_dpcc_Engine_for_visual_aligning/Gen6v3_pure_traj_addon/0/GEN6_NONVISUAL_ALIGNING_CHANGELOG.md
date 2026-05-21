# 📝 Gen6 Non-Visual (State-Only) Aligning Pipeline Changelog

This document serves as a comprehensive changelog and architectural design record of the modifications made to support the **State-Only (Non-Visual)** training and evaluation pipeline within the `ddpm_encdec_vision` (FM-PCC) framework.

---

## 🏗️ Architectural Overview & Design Decisions

To study state-only baseline performance (e.g., DDPM-ACT equivalent) against visual models, the following architectural challenges were successfully reconciled:

1. **State Dimension Mismatch (17D Simulator vs. 20D Dataset Model):**
   * *Challenge:* The aligning simulator returns a 17-dimensional state vector (missing the 3D desired TCP pose), while the training dataset uses a 20-dimensional observation vector (desired TCP pos + current TCP pos + box/target dimensions).
   * *Solution:* Implemented an active **17D-to-20D Prepending Adapter** inside the evaluation agent's rollout loop. It tracks a mental desired robot position (`self.mental_robot_pos`) starting from the initial simulator TCP pos, prepends it to the 17D environment state to build the 20D vector before feeding it to the model, and then integrates predicted actions (velocities) into it in a closed-loop fashion.

2. **Transition Dimension Generalization:**
   * *Challenge:* The temporal 1D U-Net backbone was hardcoded to expect visual proprioceptive states (`obs_dim=3`) concatenated with spatial actions.
   * *Solution:* Generalized the UNet instantiation to dynamically size the transition dimension as `action_dim + obs_dim` (yielding 22 in state-only mode instead of 6 in visual mode).

3. **Conditioning & Loss Signature Flexibility:**
   * *Challenge:* The standard training loop outputs a 5-item batch for visual tasks, while the state-only dataset yields a 3-item batch.
   * *Solution:* Rewrote the diffusion engine's `loss()` signature using Python variable arguments (`*args`) to dynamically parse, cat, and conditioning-wrap batches of varying lengths.

---

## 🛠️ Detailed File Changes

### 1. Model Configurations
* **File Name:** [aligning-d3il-visual.py](file:///workspaces/FM-PCC/config/aligning-d3il-visual.py)
* **Changes:** Appended non-visual configs to the end of the `base` dictionary literal using dynamic Python dict expansion to avoid copy-pasting the long visual baseline config.
```python
# ─── Gen6 State-Only Non-Visual Configuration Appends ────────────────────────
base['ddpm_encdec_vision_nonvisual'] = {
    **base['ddpm_encdec_vision'],
    'action_dim': 2,
    'obs_dim': 20,
    'if_vision': False,
    'prefix': 'ddpm_encdec_vision_nonvisual/',
}

base['plan_ddpm_encdec_vision_nonvisual'] = {
    **base['plan_ddpm_encdec_vision'],
    'prefix': 'f:plans/ddpm_encdec_vision_nonvisual/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}/',
    'diffusion_loadpath': 'f:ddpm_encdec_vision_nonvisual/H{horizon}',
}
```

---

### 2. Multi-Stage Training Script
* **File Name:** [train_ddpm_encdec_vision.py](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/train_ddpm_encdec_vision.py)
* **Changes:** Modified dataset class instantiation and config parameters to load the flat `Aligning_Dataset` instead of `Aligning_Img_Dataset` when `if_vision` is `False`.
```python
    # 1. Dataset
    if_vision = getattr(args, 'if_vision', True)
    obs_dim = 3 if if_vision else getattr(args, 'obs_dim', 20)
    
    if if_vision:
        from d3il.environments.dataset.aligning_dataset import Aligning_Img_Dataset
        dataset_cls = Aligning_Img_Dataset
    else:
        from d3il.environments.dataset.aligning_dataset import Aligning_Dataset
        dataset_cls = Aligning_Dataset
        
    dataset_config = utils.Config(
        dataset_cls,
        savepath=(args.savepath, 'dataset_config.pkl'),
        data_directory='environments/dataset/data/aligning/train_files.pkl',
        device='cpu',
        obs_dim=obs_dim,
        action_dim=args.action_dim,
        window_size=args.horizon,
        max_len_data=args.max_path_length
    )
    dataset = dataset_config()
```
```python
    # 3. Diffusion (Engine: Dynamic Observation Dimension)
    from ddpm_encdec_vision.models.visual_gaussian_diffusion import VisualGaussianDiffusion
    diffusion_config = utils.Config(
        VisualGaussianDiffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=obs_dim, # Scaled dynamically
        action_dim=args.action_dim,
...
```

---

### 3. Dynamic U-Net Backbone
* **File Name:** [visual_unet.py](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_unet.py)
* **Changes:** Conditionally disabled the ResNet-based image encoder and dynamically resized the sequence channels based on `if_vision`.
```python
        self.if_vision = getattr(config, "if_vision", True)
        
        # 1. Instantiate Vision Encoder
        if self.if_vision:
            # ... ResNet MultiImageObsEncoder setup ...
            latent_dim = 128
        else:
            self.obs_encoder = None
            latent_dim = 0
```
```python
        # Dynamic transition dimension supporting both visual proprioception (3D) and non-visual state (20D)
        obs_dim = getattr(config, 'obs_dim', 3 if self.if_vision else 20)
        transition_dim = config.action_dim + obs_dim

        self.backbone = backbone_class(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            cond_dim=latent_dim,
            dim=getattr(config, "dim", 128),
            dim_mults=getattr(config, "dim_mults", (1, 2, 4, 8)),
            returns_condition=getattr(config, "returns_condition", False),
            condition_dropout=getattr(config, "condition_dropout", 0.1),
            use_cond_projection=self.if_vision,  # Enable FiLM conditioning only for visual embeddings
        ).to(self.device)
```

---

### 4. General Purpose Diffusion Engine
* **File Name:** [visual_gaussian_diffusion.py](file:///workspaces/FM-PCC/ddpm_encdec_vision/models/visual_gaussian_diffusion.py)
* **Changes:** Overrode `loss` to accept dynamic parameter lengths and handle state-only snapping during trajectory rollouts in `forward`.
```python
    def loss(self, *args):
        """
        Supports both vision (5 arguments) and state-only (3 arguments) batches.
        """
        if getattr(self.model, 'if_vision', True):
            # Vision mode batch: bp_imgs, inhand_imgs, obs, act, mask
            bp_imgs, inhand_imgs, obs, act, mask = args
            x = torch.cat([act, obs], dim=-1)
            cond = {
                'visual': (bp_imgs, inhand_imgs, obs),
                0: obs[:, 0] # First-frame state for snapping
            }
        else:
            # State-only mode batch: obs, act, mask
            obs, act, mask = args
            x = torch.cat([act, obs], dim=-1)
            cond = {
                0: obs[:, 0] # First-frame state for snapping
            }
```
```python
    def forward(self, cond, *args, **kwargs):
        """
        Inference: Triggers the stochastic DDPM denoising loop (p_sample_loop).
        """
        if getattr(self.model, 'if_vision', True):
            # 1. Handle vision-specific cond unpacking
            # ... visual tuple extracts & snapping ...
        else:
            # 2. State-only cond unpacking: cond is {0: obs_seq}
            if 0 in cond and isinstance(cond[0], torch.Tensor):
                obs_seq = cond[0]
                new_cond = {0: obs_seq[:, -1]}
            else:
                new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
```

---

### 5. Evaluation Loop & Agent Rollout Adapter
* **File Name:** [eval_ddpm_encdec_vision.py](file:///workspaces/FM-PCC/ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py)
* **Changes:** Implemented the 17D->20D preprocessing adapter, integrated actions ($v_x, v_y$) directly back into the mental map in closed loop, dynamically sliced actions based on loaded model configuration, and bypassed image/video logging or DPCC constraint projectors in state-only mode.
```python
        if if_vision:
            # ... visual preprocessing ...
        else:
            # ─── STATE-ONLY INFERENCE PATH ───
            # 1. 17D to 20D Mismatch Adapter: Prepend desired pos
            if self.mental_robot_pos is None:
                self.mental_robot_pos = state[:3].copy()  # Starts at actual TCP pos
            
            # Prepend 3D desired position to 17D state -> 20D
            obs_20d = np.concatenate((self.mental_robot_pos, state))
            
            # Record real position for diagnostics
            self.history_real_pos.append(state[:3].copy())
            if self.last_predicted_pos is not None:
                err = np.linalg.norm(state[:2] - self.last_predicted_pos[:2])
                self.curr_rollout_tracking_errors.append(err)
                
            # 2. Sequence conditioning assembly
            obs_torch = torch.from_numpy(obs_20d).to(self.device).float().unsqueeze(0)
            if self.scaler is not None:
                obs_torch = self.scaler.scale_input(obs_torch)
                
            self.des_robot_pos_context.append(obs_torch)
            while len(self.des_robot_pos_context) < self.window_size:
                self.des_robot_pos_context.appendleft(obs_torch)
                
            obs_seq = torch.stack(tuple(self.des_robot_pos_context), dim=1)
```
```python
            # Dynamic action dimension slice supporting both visual proprioceptive (3D) and non-visual (2D) actions
            action_dim = getattr(self.model, 'action_dim', 3 if if_vision else 2)
            action_trajectory = trajectory[[which_trajectory], :, :action_dim]
```
```python
        # Update Mental Map (Open-Loop accumulation)
        if if_vision:
            self.mental_robot_pos += next_action_np.squeeze(0)
        else:
            self.mental_robot_pos[:2] += next_action_np.squeeze(0)
```
```python
        # Main evaluation loop: determine vision availability & adjust sim + reference video generation
        sim_vision = getattr(diffusion_model.model, 'if_vision', True) if diffusion_model is not None else True
        # ...
        if sim_vision:
            generate_expert_reference(save_path, n_rollouts=3)
        # ...
        sim = Aligning_Sim(seed=seed, device=args.device, render=False, n_cores=1,
                          n_contexts=n_contexts, n_trajectories_per_context=n_trajectories, if_vision=sim_vision,
                          eval_on_train=args_cli.eval_on_train,
                          max_episode_length=getattr(args, 'max_episode_length', 400))
```
