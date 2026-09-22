# U18 · L1 result (2026-09-22, job 26077) — the planner in the loop with the drone matches the table

Live = FM K20, seed 6, 20 trials (same torch trial seeds as the Panda `msg20trials` cell), quadrotor plant 36×, clock
1 Hz, no feed-forward. Turbo = replay of the Panda's stored setpoints (pilot 26076, **pre-fix4**). Panda = the table.
Log: [`pilot_s36/18_01_48_live_l1_26077.log`](pilot_s36/18_01_48_live_l1_26077.log).

| cell | S&C Panda / turbo / **live** | violating steps P / T / **L** | steps P / T / **L** | live ms/step |
| :-- | :-- | :-- | :-- | --: |
| top-right `diffuser` | 0.00 / 0.00 / **0.00** | 31.6 / 32.1 / **32.5** | 67.7 / 68.0 / **68.4** | 169 |
| top-right `dpcc-r-t` | 0.95 / 0.90 / **0.95** | 0.0 / 0.0 / **0.0** | 73.2 / 72.0 / **73.3** | 316 |
| top-left `diffuser` | 0.05 / 0.00 / **0.00** | 10.8 / 12.9 / **12.3** | 67.7 / 68.0 / **68.4** | 171 |
| top-left `dpcc-r-t` | 0.95 / 0.70 / **1.00** | 0.3 / 0.0 / **0.0** | 78.8 / 74.7 / **75.6** | 468 |
| both `diffuser` | 0.05 / 0.00 / **0.00** | 12.2 / 16.4 / **15.3** | 67.7 / 68.0 / **68.4** | 171 |
| both `dpcc-r-t` | 1.00 / 0.85 / **1.00** | 0.0 / 0.0 / **0.0** | 61.9 / 61.7 / **62.3** | 535 |

Plant, live, 120 episodes: 0 contacts, 0 divergence, all ended by crossing the line; tracking error 0.19 m, setpoint
gap p95 0.007 units (Panda's own 0.03–0.13); |v| ≤ 0.55 m/s. Live timing = the table's (K20 network + projector).

**Conclusion.** No divergence from the loop: live S&C is within 0.05 of the Panda on every cell; violating steps and
steps-to-goal within the Panda's spread. Turbo's shortfalls on the projected cells (0.70 / 0.85 / 0.90) are the
pre-fix4 replay artefacts (parked short of the line, arena aborts), not loop effects — live confirms the fix4 reading.

**What follows.** The plant is validated in both modes. Either turbo (after the fix4 re-run) or live can carry the
pillars table; live at DPCC's protocol (groups A–E, ≈ 4 GPU-hours) is the cleaner story ("the models evaluated with
the quadrotor as the plant") and is affordable — author's call. Turbo remains the free way to cover every stored
cell (the 20-episode extended protocol included).

Not downloaded yet: the live npz cells themselves
(`logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/H8_Dmodels…aw10/H8_K20_…_msguavpv2s36live20/6/results/`),
needed for the paired per-episode view and any DA. The live sidecar file was overwritten per geometry (only the last
40 episodes kept) — fixed in `factory.py` (one file per close).
