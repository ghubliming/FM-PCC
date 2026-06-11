# DPCC Legacy Rewire for Visual Avoiding (Markdown-Only Playbook)

This file is the exact manual rewiring sequence for running legacy DPCC on visual avoiding.

Target behavior:
- Keep baseline DPCC path unchanged (`avoiding-d3il`)
- Add parallel visual path (`avoiding-d3il-visual`)
- Run train/eval/load via separate visual scripts

## 0. Scope

This playbook updates only these areas in `dpcc`:
1. Dataset env aliasing
2. New visual config file
3. New visual projection yaml
4. New visual train/eval/load scripts

No baseline script is replaced.

## 1. Create visual files by copying baseline

Run from repo root:

```bash
cd /workspaces/dpcc

cp config/avoiding-d3il.py config/avoiding-d3il-visual.py
cp config/projection_eval.yaml config/projection_eval_visual.yaml

cp scripts/train.py scripts/train_visual.py
cp scripts/eval.py scripts/eval_visual.py
cp scripts/load_results.py scripts/load_results_visual.py
```

## 2. Rewire dataset loader for visual env id (required)

Edit:
- `diffuser/datasets/d4rl.py`

Find:

```python
elif env == 'avoiding-d3il' or env == 'd3il-avoiding':
```

Replace with:

```python
elif env in ('avoiding-d3il', 'd3il-avoiding', 'avoiding-d3il-visual', 'd3il-avoiding-visual'):
```

Why:
- Without this change, training with `avoiding-d3il-visual` will hit `NotImplementedError`.

## 3. Rewire visual train script

Edit:
- `scripts/train_visual.py`

Change this line:

```python
exp = 'avoiding-d3il'
```

to:

```python
exp = 'avoiding-d3il-visual'
```

Everything else can stay the same.

Reason:
- `config: str = 'config.' + exp` will then auto-load `config/avoiding-d3il-visual.py`.

## 4. Rewire visual eval script

Edit:
- `scripts/eval_visual.py`

Change yaml load path:

```python
with open('config/projection_eval.yaml', 'r') as file:
```

to:

```python
with open('config/projection_eval_visual.yaml', 'r') as file:
```

No other hardcoded exp value is needed in this file; it reads `exps` from yaml.

## 5. Rewire visual results loader script

Edit:
- `scripts/load_results_visual.py`

Change yaml load path:

```python
with open('config/projection_eval.yaml', 'r') as file:
```

to:

```python
with open('config/projection_eval_visual.yaml', 'r') as file:
```

Change exp:

```python
exp = 'avoiding-d3il'
```

to:

```python
exp = 'avoiding-d3il-visual'
```

## 6. Rewire visual projection yaml

Edit:
- `config/projection_eval_visual.yaml`

### 6.1 Set experiment id list

Change:

```yaml
exps: [
    'avoiding-d3il',
]
```

to:

```yaml
exps: [
    'avoiding-d3il-visual',
]
```

### 6.2 Duplicate avoiding keys for visual id

Add visual keys by copying values from existing avoiding entries.

Required maps to include `avoiding-d3il-visual`:
- `halfspace_constraints`
- `obstacle_constraints`
- `bounds`
- `ax_limits`

Minimal rule:
- Start with exactly the same numeric values as `avoiding-d3il`.

## 7. Rewire visual training config

Edit:
- `config/avoiding-d3il-visual.py`

Yes, you can directly carry DPCC parameters into this visual config, and this is the recommended path for stable behavior.

Recommended approach:
1. Keep `base['diffusion']` and `base['plan']` exactly aligned with baseline DPCC.
2. Add explicit visual overrides under `avoiding_d3il_visual` so you can tune visual-only runs later without touching baseline.

Copy-ready override block (append near bottom of file):

```python
avoiding_d3il_visual = {
    'diffusion': {
        'horizon': 8,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'predict_epsilon': True,
        'returns_condition': False,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'n_train_steps': 1e5,
    },
    'plan': {
        'horizon': 8,
        'n_diffusion_steps': 20,
        'batch_size': 4,
        'max_episode_length': 200,
        'diffusion_epoch': 'best',
    },
}
```

Why this helps:
- You lock in known DPCC hyperparameters for visual avoiding.
- You can later tune only `avoiding_d3il_visual` without changing `avoiding_d3il` baseline.

Important check:
- Ensure file defines `base = { ... }` with `diffusion` and `plan` sections.

## 8. Smoke test commands

Run in order:

```bash
cd /workspaces/dpcc

python scripts/train_visual.py
python scripts/eval_visual.py
python scripts/load_results_visual.py
```

If you want fast check first, temporarily set in `config/projection_eval_visual.yaml`:
- `seeds: [0]`
- `n_trials: 1`

## 9. Validation checklist

- [ ] `scripts/train_visual.py` uses `exp = 'avoiding-d3il-visual'`
- [ ] `scripts/eval_visual.py` loads `projection_eval_visual.yaml`
- [ ] `scripts/load_results_visual.py` loads `projection_eval_visual.yaml`
- [ ] `scripts/load_results_visual.py` uses `exp = 'avoiding-d3il-visual'`
- [ ] `diffuser/datasets/d4rl.py` accepts visual env aliases
- [ ] `config/projection_eval_visual.yaml` has visual keys in all required maps

## 10. Failure map

`NotImplementedError` during dataset load:
- Missing alias update in `diffuser/datasets/d4rl.py`

`ModuleNotFoundError: config.avoiding-d3il-visual`:
- Missing `config/avoiding-d3il-visual.py`

`KeyError: 'avoiding-d3il-visual'` in eval:
- Missing visual key in one of yaml maps

`FileNotFoundError` for `.npz` in load results:
- `eval_visual.py` was not run or used different seeds
