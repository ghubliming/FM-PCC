# CHANGELOG — fix_3: seed + n_trials moved to config file, CLI override with source logging

**Date**: 2026-06-28
**Parent**: [../fix_2_Config/CHANGELOG_fix2_config_split.md](../fix_2_Config/CHANGELOG_fix2_config_split.md)

---

## Problem

`parse_args()` in `eval_fm_uav.py` had **hardcoded defaults** for two key run-quantity params:

```python
p.add_argument('--seed',     type=int, default=5,  ...)   # ← wrong: UAV trains at seed 6
p.add_argument('--n-trials', type=int, default=20, ...)   # ← magic number, not in any config
```

Two bugs:
1. **Wrong default seed** — `DEFAULT_SEEDS = [6]` in `train_fm_uav.py`, but eval defaulted to 5. Every run without `--seed` evaluated the wrong checkpoint silently.
2. **Orphaned magic numbers** — `seed` and `n_trials` were invisible to the config files, making them impossible to version-control or batch-sweep without touching source.

The pattern reference (`projection_eval.yaml` for `avoiding-d3il`) puts BOTH in the yaml:
```yaml
seeds: [6, 7, 8, 9, 10]
n_trials: 2
```

---

## Fix

### Where they now live: `config/uav_projection.yaml`

```yaml
seed: 6       # trained-model checkpoint seed (matches train_fm_uav.py DEFAULT_SEEDS[0])
n_trials: 20  # closed-loop rollouts per scene
```

Rationale for yaml (not plan block): these are **run-quantity** params, not model params —
`projection_eval.yaml` has `seeds` and `n_trials` there for the same reason. The plan block
in `uav.py` holds model-identity and eval-control scalars (`batch_size`, `control_hz`, etc.).

### Priority: CLI > file, with console source logging

```python
# main() — before sys.argv is stripped
_seed_from_cli   = args.seed     is not None
_trials_from_cli = args.n_trials is not None

args.seed     = args.seed     if _seed_from_cli   else int(_proj_defaults.get('seed', 6))
args.n_trials = args.n_trials if _trials_from_cli else int(_proj_defaults.get('n_trials', 20))

print(f'[ eval ] seed={args.seed}     (source: {"--seed CLI"     if _seed_from_cli   else yaml_path})')
print(f'[ eval ] n_trials={args.n_trials}  (source: {"--n-trials CLI" if _trials_from_cli else yaml_path})')
```

Console output example — file default:
```
[ eval ] seed=6      (source: config/uav_projection.yaml)
[ eval ] n_trials=20 (source: config/uav_projection.yaml)
```

Console output example — CLI override:
```
[ eval ] seed=7      (source: --seed CLI)
[ eval ] n_trials=5  (source: --n-trials CLI)
```

---

## Files changed

| File | Change |
|---|---|
| `config/uav_projection.yaml` | Added `seed: 6` and `n_trials: 20` under new `# Eval run config` section |
| `FM_v3_uav_test/eval_fm_uav.py` | `parse_args()`: both defaults → `None`; updated help strings. `main()`: quick yaml read + resolve + source-print before sys.argv strip |

### `parse_args()` change

```python
# Before
p.add_argument('--seed',     type=int, default=5,  help='Trained-model seed to load.')
p.add_argument('--n-trials', type=int, default=20, help='Closed-loop rollouts per scene.')

# After
p.add_argument('--seed',     type=int, default=None,
               help='Trained-model checkpoint seed. Default: seed from config/uav_projection.yaml.')
p.add_argument('--n-trials', type=int, default=None,
               help='Rollouts per scene. Default: n_trials from config/uav_projection.yaml.')
```

---

## Backward compatibility

- **No retrain, no checkpoint rename needed.**
- Omitting `--seed` on the cluster now loads seed **6** (correct) instead of 5 (wrong silent bug).
- Passing `--seed 5` or `--n-trials 10` on the CLI works exactly as before.
- Fallback hardcoded defaults (`seed=6`, `n_trials=20`) remain inside `main()` in case `uav_projection.yaml` is missing — so the script degrades gracefully.

---

## Verification

- `py_compile` passes: `eval_fm_uav.py`, `config/uav.py`
- `uav_projection.yaml` parses: `seed=6`, `n_trials=20`, `threshold=0.5`

---

## How to revert

```bash
git checkout -- config/uav_projection.yaml FM_v3_uav_test/eval_fm_uav.py
```
