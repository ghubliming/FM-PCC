# fix_1 — `ValueError: too many values to unpack (expected 3)`

**Crash:** All 6 seeds failed at the FIRST simulation eval call (epoch 20), so NO best-checkpoint
was ever saved. Training completed 200 epochs but best_success was never updated → saved
checkpoint = last epoch, not best simulation-success. U3's core fix was entirely bypassed.

---

## Root Cause

`aligning_sim.py` was intentionally modified in **Gen6V4 FIX8** (commit `7ba1f07`) to pass a
**4-tuple** `(bp_image, inhand_image, des_robot_pos, robot_pos)` to `agent.predict()`.

The Gen6V4 DPCC agent (`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py:1429`)
was updated in that same commit to unpack all 4 elements — it uses `robot_pos` internally
to track the actual robot position.

However, the d3il standard agent `ddpm_encdec_vision_agent.py` was **never updated** —
it still expected exactly 3 elements:

```python
# ddpm_encdec_vision_agent.py:340 — OLD (CRASH with 4-tuple):
bp_image, inhand_image, des_robot_pos = state   # ValueError when state has 4 elements
```

Gen6V4 worked because it uses its own DPCC agent (4-tuple aware).
The d3il baseline broke because it uses the standard d3il agent (3-tuple only).

---

## Files Changed

### `d3il/simulation/aligning_sim.py`
**Unchanged** — the 4-tuple `(bp_image, inhand_image, des_robot_pos, robot_pos)` is kept.
Reverting it would break Gen6V4's DPCC agent which needs `robot_pos`.

### `d3il/agents/ddpm_encdec_vision_agent.py` line 340

```python
# before (CRASH when sim passes 4 elements):
bp_image, inhand_image, des_robot_pos = state

# after (safe — takes first 3 regardless of tuple length):
bp_image, inhand_image, des_robot_pos = state[0], state[1], state[2]
```

This is the minimal fix: `ddpm_encdec_vision_agent` never used `robot_pos`; indexing
instead of unpacking means a 4-element tuple works without touching `aligning_sim.py`.

---

## Why Gen6V4 Still Works

Gen6V4 DPCC agent at `eval_visual_aligning_dpcc.py:1429`:
```python
bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state  # C4: unpack actual robot_pos
```
`aligning_sim.py` still passes 4 elements → Gen6V4 unaffected.

---

## Impact Summary

| Component | Before fix_1 | After fix_1 |
|---|---|---|
| d3il baseline eval | CRASH — ValueError at epoch 20 | Works — unpacks state[0:3] |
| Gen6V4 DPCC eval | Works — unpacks all 4 | Works — aligning_sim unchanged |

---

## Resubmit All Seeds

```bash
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/run_all_seeds_d3il_baseline.sh
```
