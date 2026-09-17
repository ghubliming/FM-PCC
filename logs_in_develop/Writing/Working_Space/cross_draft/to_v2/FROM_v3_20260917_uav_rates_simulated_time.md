# FROM v3 → v2 · 2026-09-17 · "33 Hz plan / 100 Hz inner" is simulated time, not a real-time rate

**Where in v2:** `thesis_v2.tex` rate table (≈ line 1739–1740: *plan rate … 33 Hz*, *inner rate … 100 Hz*) and
the equation near line 1808 (`f\sidx{plan} = 33 Hz, Δt\sidx{phys} = 0.01 s`).

**What the code does** (`mix_uav_test/eval_mix_uav.py:1609–1735`): physics step `dt = 0.01 s`; the planner is
queried every `decim = round(1/(dt·33)) = 3` physics steps (0.03 s); the tracker is called at every physics step.
The demonstrations were downsampled by the same factor (`uav_expert_data_collect/dataset_writer.py:73`,
`DATASET_HZ = 33`). **The simulation is stepped only after the planner returns** (lock-step).

**Why the wording misleads:** measured wall-clock per plan step in the reported UAV cells is 9–4878 ms (median
98 ms); only 21 % of cells would even fit a real 30 ms period. Nothing in this work runs at 33 Hz in real time.
"33 Hz" is the recording rate of the demonstrations, inherited by the planner — not a control-design rate, and
exactly 1/0.03 s = 33.3 Hz.

**Suggested:** state intervals of simulated time — *"a plan every third physics step (0.03 s of simulated time,
the recording interval of the demonstrations); the controller at every 0.01 s physics step; the simulation waits
for the planner"*. v3 §5 `tab:platforms` now says this (v3.18).
