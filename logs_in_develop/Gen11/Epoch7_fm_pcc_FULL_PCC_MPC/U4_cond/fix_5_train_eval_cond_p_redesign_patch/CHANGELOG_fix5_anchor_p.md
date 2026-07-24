# CHANGELOG — fix_5: Anchor-P Integration (real-position grounding, no retrain)

**Date**: 2026-06-28
**Parent**: [../fix_4_CLI_yaml/CHANGELOG_fix4_CLI_yaml.md](../fix_4_CLI_yaml/CHANGELOG_fix4_CLI_yaml.md)
**Design**: [DESIGN_fix5_anchor_p_integration.md](DESIGN_fix5_anchor_p_integration.md)

---

## Problem

The dynamics constraint enforced `p_des[t+1] = p_des[t] + action[t]` (Euler in commanded
space). With PID tracking lag, `p_des` drifts unboundedly from the real drone position `p`.
The FM's next obs `[p_des | p | v]` then has `|p_des - p|` growing beyond training
distribution → spiral failure.

The constraint was "dreaming": projecting action feasibility from a commanded position the
drone is not at. When tracking is perfect (`p_des = p`) the DPCC formulation is consistent.
The fix generalises this: make EVERY step behave like the perfect-tracking case.

---

## Fix

Two coupled eval-only changes under one config toggle `anchor_to_p`:

| | Default (`anchor_to_p=False`) | Anchored (`anchor_to_p=True`) |
|---|---|---|
| Dynamics constraint dims | `p_des` (3,4,5) | real `p` (6,7,8) |
| Integration | `p_des = p_des + action` | `p_des = p + action` |
| Output folder tag | *(none)* | `_anchorP` appended |
| Retrain needed | No | No |

**DPCC collapse validation**: when `p_des = p` (no lag), `p_des + action = p + action`
— identical. The fix is a strict generalisation of the intended DPCC behaviour.

---

## Files Changed

### `config/uav.py`

- **Removed** `reanchor_alpha` and `lead_gain` from `plan_flow_matching_v3_uav` block
  (superseded by `anchor_to_p`).
- **Added** `'anchor_to_p': False` to `plan_flow_matching_v3_uav` block.
- **Cleaned** `cond_mode` training block comment (removed abandoned `real_p` retrain path).
- **Cleaned** `args_to_watch` comment (same).

### `FM_v3_uav_test/eval_fm_uav.py`

**`load_pcc_config()`**
- Removed: `cfg['reanchor_alpha']`, `cfg['lead_gain']`
- Added: `cfg['anchor_to_p'] = bool(getattr(plan_args, 'anchor_to_p', False))`

**`setup_dpcc_projector()`**
- Added `anchor_to_p=False` parameter.
- Dynamics `deriv` binding: when `anchor_to_p=True` → dims `[6,0],[7,1],[8,2]` (real `p`);
  else → dims `[3,0],[4,1],[5,2]` (p_des, original).

**`rollout_one()`**
- Removed: `cond_mode`, `reanchor_alpha`, `lead_gain` parameters.
- Added: `anchor_to_p=False` parameter.
- Obs: always 9D `[p_des|p|v]` (removed dead `real_p` branch that built 6D obs).
- Integration:
  ```python
  if anchor_to_p:
      p_des = p + action      # grounded to real position
  else:
      p_des = p_des + action  # default free-running
  ```

**`_run_variant()`**
- Removed: `cond_mode`, `reanchor_alpha`, `lead_gain` reads and the old multi-branch
  eval_tag logic.
- Added: `anchor_to_p = bool(config.get('anchor_to_p', False))`
- Eval tag: `eval_tag = '_anchorP' if anchor_to_p else ''`
- Updated `setup_dpcc_projector()` call: added `anchor_to_p=anchor_to_p`.
- Updated `rollout_one()` call: replaced old knobs with `anchor_to_p=anchor_to_p`.

**Module docstring** updated to reflect dual integration formula.

---

## U4 Reverts

The following U4 (previous session) additions are removed as superseded:

| Removed | Reason |
|---|---|
| `reanchor_alpha` (blend approach) | `anchor_to_p` is strictly correct; blending was an approximation |
| `lead_gain` | Was for `real_p` retrain mode only |
| `cond_mode='real_p'` obs/integration branch in `rollout_one` | Retrain path abandoned; anchor_to_p achieves same grounding without it |
| `real_p` commentary in training block comments | No longer the intended path |

`cond_mode` in `args_to_watch` and both config blocks is **kept** — it is baked into
checkpoint folder names (`..._condp_des/`) and must remain for correct checkpoint loading.

---

## How to Use

Toggle in `config/uav.py` → `plan_flow_matching_v3_uav`:

```python
'anchor_to_p': True,   # enable grounding
```

Git sync, then:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh corridor "6"
```

Output lands in: `.../plans/.../diffuser_anchorP/` (vs `.../diffuser/` for default).
Both run from the same checkpoint — no checkpoint moves, no retrain.

---

## Verification

- `grep reanchor FM_v3_uav_test/eval_fm_uav.py` → empty ✓
- `grep lead_gain FM_v3_uav_test/eval_fm_uav.py` → empty ✓
- `grep real_p FM_v3_uav_test/eval_fm_uav.py` → empty ✓
- `anchor_to_p` appears at: docstring, `load_pcc_config`, `setup_dpcc_projector` (sig +
  constraint branch), `rollout_one` (sig + integration), `_run_variant` (read + tag +
  both call sites) ✓
