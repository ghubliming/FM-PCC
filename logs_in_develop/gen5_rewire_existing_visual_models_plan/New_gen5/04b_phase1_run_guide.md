# Phase 1 Execution Guide: How to Run the Visual DDPM Baseline

This guide explains how to launch training and evaluation for the freshly wired D3IL visual control baseline inside the FM-PCC repository.

## 1. Directory Context
Ensure your terminal is situated at the root of the FM-PCC repository:
```bash
cd /workspaces/FM-PCC
```

## 2. Launching Training
To kick off a training run using the newly bridged D3IL DDPM components, execute the testing script with your preferred random seed.

```bash
# Run a single training seed (e.g. seed 5)
python ddpm_encdec_vision_test/train_ddpm_encdec_vision.py --seed 5
```

### Expected Behavior (Training):
- The script will initialize `Aligning_Img_Dataset` pointing to `environments/dataset/data/aligning/train_files.pkl`.
- `VisualDiffusionBridge` will instantiate ResNet image encoders and the DDPM core.
- Logs and checkpoints will stream to: `logs/aligning-d3il-visual/ddpm_encdec_vision/...`

## 3. Launching Evaluation
Once a model has been trained and a checkpoint is available, you can evaluate the agent's performance in simulation.

```bash
# Run evaluation for the corresponding seed
python ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py --seed 5
```

### Expected Behavior (Evaluation):
- The script initializes `Aligning_Sim(if_vision=True)`.
- It loads the best DDPM visual checkpoint.
- It will simulate the rollouts using `Aligning_Sim` and `VisualAgentWrapper`.
- Metrics (Success rate, Entropy, Mean Distance) will be printed to standard output and simultaneously synced to Weights & Biases (W&B) under the project `aligning-vision`.

## Next Steps
- Verify the training loss decreases monotonically.
- Ensure the evaluation script correctly loads `state_best.pt` and connects to the MuJoCo visual rendering environments without shape mismatch errors.
