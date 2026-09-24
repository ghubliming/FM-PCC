# TO v3 — R44b + C0 ready: Table 6.15 complete, Table 6.16 three of four rows; 🔴 a correction to the violation sentence

**2026-09-24 · from the DA side (R44 run chat).** Nothing in `v3/` was touched.
- **Analysis of record:** `Data_Analysis/DA_in_Paper/analysis/DA_20260924_scurve_R44bc_projection_controller.md`, script
  `scurve_r44_raw.py`; a row was added to `analysis/INDEX.md`.
- **Data:** raw result folders only (md5-verified fetch). Jobs 26195 (B1) and 26196 (C0) are clean, and B1's unprojected
  cell reproduces A7 flight for flight.
- **Still pending:** C1, the MuJoCo MPC row of the projected plans (rule random), and Figure 6.9's rebuild. Both come in
  one more short round.

## 1 · Table 6.15 `tab:uav-scurve-projection` (FM, nfe 1, cascaded geometric)

```latex
    none      & ---                   & 9/10 & $23.0 \pm 5.4$  & 0.571 & 1/10 & 8.9 \\
    \addlinespace[3pt]
    per-step  & random                & 5/10 & $71.3 \pm 66.6$ & 0.572 & 3/10 & 157.1 \\
    per-step  & cumulative cost       & 5/10 & $74.7 \pm 95.9$ & 1.007 & 3/10 & 163.8 \\
    per-step  & temporal consistency  & 5/10 & $37.2 \pm 19.6$ & 0.817 & 5/10 & 135.9 \\
    \addlinespace[3pt]
    endpoint  & ---                   & \multicolumn{5}{l}{--- no guiding step at $\nfe=1$} \\
```
- **Dataref:** tag `p23scproj`, job 26195, `dpcc-{r,c,t}-tightened`. B1's own unprojected cell equals A7 (8.8 ms/step);
  the row above keeps A7's 8.9.
- S&C 0/10 in every row.
- **For the `\hole`:** per-step projection does **not** keep the nine crossings. Every rule drops to 5/10, aborts rise
  from 1 to 3–5, and violating steps rise to 37–75 per flight. Random and cumulative cost each have two flights that drop
  to the floor after crossing. The time per step goes from 8.9 to 136–164 ms (the projection is 126–155 ms of it).

## 2 · Table 6.16 `tab:uav-controller` — three rows (ten flights each)

```latex
    none      & cascaded geometric & 9/10 & $23.0 \pm 5.4$  & 0.571 & 1/10 \\
    none      & MuJoCo MPC         & 9/10 & $41.5 \pm 18.4$ & 0.297 & 0/10 \\
    \addlinespace[3pt]
    per-step  & cascaded geometric & 5/10 & $71.3 \pm 66.6$ & 0.572 & 3/10 \\
    per-step  & MuJoCo MPC         & \multicolumn{4}{l}{pending (R44c)} \\
```
- **Per-step rule: random.** The three rules tie at 5/10 successes. Random and cumulative cost abort 3, temporal
  consistency 5; random then has fewer violating steps (71.3 against 74.7). Put this tie-break in the dataref, since the
  caption says "the rule that crosses the finish line most often".
- **MuJoCo MPC row:** tag `p23scmjpc`, job 26196, environment `FMPCC_mjx`.
- Beside the table, if useful: under MuJoCo MPC **all ten flights end within 0.3 m of the goal point** (cascaded: 9); the
  one MPC flight that is not a success is scored unsafe for contact (fraction 0.096).

## 3 · 🔴 Correction: the sentence after Table 6.14 and the provisional conclusion

My R44a note gave you the mechanism "the flights follow their plans with a mean tracking error of 0.30–0.33 m while the
route clears the boundaries by 0.121 m". **That mechanism is wrong. The numbers are right.** The flight paths show:
- **0.30 m is the setpoint's lead:** the commanded position lies about 0.3 m ahead of the vehicle along the path.
  Sideways, the cascaded geometric controller stays **within about 1 cm of the commanded path** on every flow model. On FM
  nfe 1 it departs at most 11 cm, at the corner.
- **The commanded path itself cuts the second inside corner of the crossover**, at (0.5, 0.3). It passes 8–19 cm inside
  the corner's keep-out (0.05 m corner plus the 0.31 m rotor reach) on **every flight of every configuration**. That corner
  is where the flown violations are (FM nfe 1: 223 of 230 violating steps, at x 0.18–0.24). The flown path never comes more
  than 1.4 cm closer to the corner than the command does.
- The demonstrated route clears the constraints by 0.121 m, so **the plans cut a corner the demonstrations clear**.

Suggested replacement: "No flight is free of violations, and the reason is in the plans: every commanded path cuts the
second inside corner of the crossover, passing 8 to 19 cm inside its keep-out on every flight, where the demonstrated route
clears it by 0.121 m. The cascaded geometric controller follows the commanded path to within about a centimetre on average;
the $0.30$ to $0.33$\,m by which a flight trails its commanded position is the lead of that position along the path, not a
sideways error." The same fix applies to the provisional conclusion's "so that no flight is free of violations" clause.

## 4 · Facts for `sec:res:uav:controller` and the conclusion (storyline: the controller on extreme trajectories)

1. **Where the controller is the limit: stability at the second turn.** 54 of the 57 lost flow-model flights of Table 6.14
   invert, 48 of them at x 0.2–1.2 m (median 0.38 m), just past the crossover as the route turns into the second straight.
   The baseline inverts at x 0.33–0.43 m on 7 of its 8.
2. **The 10-vs-10 on the selected plans (unprojected).** MuJoCo MPC removes the inversion (0/10 against 1/10) and brings
   every flight within 0.3 m of the goal point (mean distance 0.297 against 0.571 m). Successes stay 9/10. It does **not**
   remove violations: it follows the commanded path less closely (5 cm mean, up to 29 cm), and its plans, re-generated
   from its own states, cut the corner deeper. Violating steps rise from 23.0 to 41.5 per flight, and contact appears
   (fraction 0.035 against 0.001).
3. **Per-step projection cannot reach the corner cut as configured.** On this scene it binds the plan's measured position,
   as DPCC does, not the commanded setpoint the vehicle is sent to. The commanded path still passes 8–9 cm inside the
   corner. UAV-corridor binds the setpoint (`sec:setup:protocol:uav`: "constrains the commanded position in addition to
   the measured one"); UAV-s-curve does not. One sentence can say so.
4. **Cost, 10 vs 10 on FM nfe 1** (if you want Table 6.17 on the selected configuration instead of the pilot): the
   control loop 14.0 against 135.8 ms, the planner 8.9 against 10.4 ms, the controller and the simulator step **5.2
   against 125.4 ms** (pilot: 6.5 against 125.8).
5. **The pilot text and Figure 6.9** can go once C1 lands. I rebuild the figure on FM nfe 1: unprojected and per-step,
   each under both controllers, ten flights per panel, with the commanded path drawn beside the flown one.

## 5 · Open, the author's call

A setpoint-bound per-step run (the corridor's `-bounds_free-pdes-tightened`) on FM nfe 1, about 1 h, would show whether
the projector removes the corner cut once it binds what is commanded (DA §7). It is not needed if the section states
DPCC's binding as the configuration.
