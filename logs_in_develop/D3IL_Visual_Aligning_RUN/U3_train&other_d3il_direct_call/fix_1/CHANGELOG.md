# fix_1 — Two `ValueError: too many values to unpack` bugs

**Crash:** All seeds failed at the FIRST simulation eval call (epoch 20). U3's core
simulation-success checkpoint saving never ran.

---

## Bug A — `aligning_sim.py` passes 4-tuple; standard agent expects 3

**File:** `d3il/agents/ddpm_encdec_vision_agent.py` line 340

`aligning_sim.py:102` passes `(bp_image, inhand_image, des_robot_pos, robot_pos)` — 4 elements.
This was added in Gen6V4 FIX8 for the DPCC agent which uses `robot_pos`. The standard d3il agent
was never updated.

```python
# before (CRASH — strict 3-unpack):
bp_image, inhand_image, des_robot_pos = state

# after (safe — index, ignores 4th element):
bp_image, inhand_image, des_robot_pos = state[0], state[1], state[2]
```

`aligning_sim.py` stays unchanged (4-tuple preserved for Gen6V4 DPCC agent).

---

## Bug B — `test_agent()` returns 4 values; our code unpacked 2

**File:** `d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py` line 58

`aligning_sim.test_agent()` returns `(success_rate, mode_encoding, successes, mean_distance)` — 4 values.

```python
# before (CRASH — expected 2):
successrate, _ = train_sim.test_agent(agent)

# after:
successrate, _, _, _ = train_sim.test_agent(agent)
```

Only `successrate` is used for checkpoint selection; the other 3 are discarded.

---

## Impact

Both bugs hit the same call site (`test_agent` at epoch 20). Bug A was fixed first,
Bug B surfaced immediately after. After both fixes, the epoch loop can run to completion
and save best-success checkpoints as U3 intended.

---

## Resubmit

```bash
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh ddpm_encdec_vision 42 200 all paper
```
