# action_weight Code Trace in FM-PCC/DPCC (Config -> Training Loss -> Gradients)

Date: 2026-04-12

## 1) What this note answers

You already know where to set `action_weight` in config.
This note shows exactly how that value flows through code and where it actually changes training behavior.

## 2) Single-path answer (short version)

`config/avoiding-d3il.py` -> `Parser.read_config(...)` -> `args.action_weight` -> training script builds `diffusion_config` with `action_weight=args.action_weight` -> diffusion model `__init__` calls `get_loss_weights(...)` -> `loss_weights[0, :action_dim] = action_weight` -> `WeightedLoss.forward(...)` computes `(per_element_mse * weights).mean()` -> optimizer updates UNet with weighted gradients.

That is the full causal path.

## 3) Exact code path with files

## Step A: config defines the value

File:
- `config/avoiding-d3il.py`

Examples in this file include:
- `base['diffusion']['action_weight'] = 10`
- `base['flow_matching']['action_weight'] = 10`
- `base['flow_matching_v2']['action_weight'] = 1`
- `base['flow_matching_v3']['action_weight'] = 1`

So the value depends on which experiment block you train.

## Step B: parser loads that block into args

File:
- `diffuser/utils/setup.py`

`Parser.parse_args(experiment=...)` calls:
- `read_config(args, experiment)`

`read_config(...)` imports `config.avoiding-d3il`, selects:
- `base[experiment]`

Then writes each key into `args`, including `args.action_weight`.

## Step C: training script selects experiment block

Files and selected experiment key:
- DDPM: `scripts/train.py` uses `parse_args(experiment='diffusion', ...)`
- FM: `FM_test/train_FM.py` uses `parse_args(experiment='flow_matching', ...)`
- FM v2: `FM_v2_test/train_FM_v2.py` uses `parse_args(experiment='flow_matching_v2', ...)`
- FM v3: `FM_v3_test/train_FM_v3.py` uses `parse_args(experiment='flow_matching_v3', ...)`
- FM U-Net v2: `FM_Unet_v2_test/train_FM_Unet_v2.py` uses `parse_args(experiment='flow_matching_unet_v2', ...)`

So changing the number in config only affects runs that use that experiment key.

## Step D: args.action_weight is passed into diffusion constructor

In each training script, `diffusion_config` is built with:

```python
action_weight=args.action_weight,
loss_discount=args.loss_discount,
```

Then:

```python
diffusion = diffusion_config(model)
```

That call instantiates the configured diffusion class with the chosen `action_weight`.

## Step E: diffusion model builds the weight matrix W

Files:
- `diffuser/models/diffusion.py` (DDPM)
- `flow_matcher/models/diffusion.py` (FM)

Both use the same logic:

```python
loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
```

Inside `get_loss_weights(...)`:

1. Start with per-dimension ones.
2. Apply optional observation-dim multipliers from `weights_dict`.
3. Build temporal discounts (`discount ** horizon_index`, normalized by mean).
4. Outer-product to get matrix `W[h, d]`.
5. Override first action cells:

```python
loss_weights[0, :self.action_dim] = action_weight
```

This line is the core mechanism.

## Step F: weighted loss is actually computed

Files:
- `diffuser/models/helpers.py`
- `flow_matcher/models/helpers.py`

`WeightedLoss.forward(pred, targ)` does:

```python
loss = self._loss(pred, targ)                   # elementwise MSE
weighted_loss = (loss * self.weights).mean()    # weighted objective
a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
```

So:
- `weighted_loss` (logged as `diffusion_loss`) is weight-sensitive.
- `a0_loss` is explicitly de-weighted (raw first-action error view).

## Step G: trainer backpropagates this weighted objective

File:
- `diffuser/utils/training.py`

Train loop calls:

```python
loss, infos = self.model.loss(*batch)
loss.backward()
self.optimizer.step()
```

Since `loss` came from weighted objective, gradients are weight-shaped.

## 4) Math of what action_weight changes

Let elementwise error be:

$$
e_{b,h,d} = \hat{y}_{b,h,d} - y_{b,h,d}
$$

Total objective used by optimizer:

$$
\mathcal{L} = \frac{1}{BHD} \sum_{b,h,d} W_{h,d} e_{b,h,d}^2
$$

With first-action override:

$$
W_{0, d} = action\_weight \quad \text{for } d \in [0, action\_dim-1]
$$

Gradient scaling effect:

$$
\nabla_\theta \mathcal{L} = \frac{1}{BHD} \sum_{b,h,d} 2 W_{h,d} e_{b,h,d} \nabla_\theta e_{b,h,d}
$$

So larger `action_weight` gives proportionally larger gradient pressure on those first-action cells.

## 5) FM vs DDPM: same weighting pipe, different target

Weighting pipeline is the same.
The main difference is target definition in `p_losses(...)`:

- DDPM (`diffuser/models/diffusion.py`): target is noise `epsilon` (when `predict_epsilon=True`).
- FM (`flow_matcher/models/diffusion.py`): target is velocity `v_target = x_start - x_base`.

So identical `action_weight` can still produce different training behavior because prediction targets differ.

## 6) Where action_weight does NOT apply

`action_weight` is training-only in this stack.
It is not used directly in:
- sampling functions (`p_sample`, `p_sample_loop`)
- evaluation scripts
- projection/candidate ranking logic

Inference only uses the trained weights already produced during training.

## 7) Practical checklist when changing action_weight

1. Edit the correct experiment block in `config/avoiding-d3il.py`.
2. Run the matching training script for that block.
3. Retrain from scratch (or start a new run path); old checkpoints keep old weighting effects.
4. Compare runs mainly with:
   - `a0_loss` (raw first-action error)
   - downstream eval metrics (success/collision/projection cost)
5. Do not compare absolute `diffusion_loss` across different weight matrices as a quality metric.

## 8) One-line summary

In FM-PCC/DPCC, `action_weight` is injected from config into diffusion model construction, converted into the first-row action entries of the loss weight matrix, and affects training only through weighted MSE gradients.

## 9) Q&A (Usage Scope)

Question:
- "So only in training, not in sampling/eval, is this confirmed?"

Answer:
- Yes, confirmed for this codebase.
- `action_weight` is used directly in training-time loss construction and gradient updates.
- `action_weight` is not read directly in sampling/eval code paths.
- Inference/eval is still affected indirectly because model weights were learned under that training objective.

Direct code anchors:
- Training-time weighting construction: `diffuser/models/diffusion.py`, `flow_matcher/models/diffusion.py` (`get_loss_weights`)
- Weighted objective application: `diffuser/models/helpers.py`, `flow_matcher/models/helpers.py` (`WeightedLoss.forward`)
- Gradient update path: `diffuser/utils/training.py` (`loss.backward()` / `optimizer.step()`)
- Sampling path (no direct `action_weight` read): `diffuser/models/diffusion.py`, `flow_matcher/models/diffusion.py` (`p_sample`, `p_sample_loop`)