## Immediate next steps (clean thread, actionable)

If you want a concrete starting checklist:

1. **Pick and lock in a MuJoCo quadrotor base.**  
   - Evaluate `gym_multirotor` or similar MuJoCo UAV envs and choose one with clean XML and Python API. [github](https://github.com/adipandas/gym_multirotor)
   - Fix state and action conventions (e.g., 6‑DoF pose + linear/angular velocities; actions as body‑frame accelerations or velocity targets).

2. **Define the MuJoCo obstacle world.**  
   - Start with 2–3 canonical layouts (straight corridor, S‑curve corridor, pillar field) using simple geoms.  
   - Implement a function that, given MuJoCo state, returns obstacle half‑spaces / signed distances \(\mathcal{Z}_f^t\) compatible with DPCC’s requirement.

3. **Mirror UAV‑Flow trajectory statistics.**  
   - From the UAV‑Flow sim/real data, fix your target: typical horizon \(T\), sampling rate, distance per episode, typical constant‑altitude behavior. [openreview](https://openreview.net/forum?id=fJMWkaT2HX)
   - Implement a simple expert controller in MuJoCo that generates similar “nice” paths (e.g., corridor following, around‑pillar) so you have supervised trajectories for FM.

4. **Integrate DPCC with the MuJoCo quadrotor.**  
   - Take DPCC’s dynamics interface (the prediction model inside the projection loop) and swap it to use your MuJoCo quadrotor dynamics (or a learned/linearized approximation). [github](https://github.com/ralfroemer99/dpcc)
   - Validate: given a candidate action chunk, roll it out in MuJoCo and check DPCC’s projection keeps you within acceleration and obstacle constraints.

5. **Prototype the FM policy in a non‑visual setting.**  
   - Before throwing in RGB/Depth, train FM on state‑only trajectories in the MuJoCo env; make sure FM+DPCC closes the loop and flies without crashing at your target Hz.  
   - This is your “minimal FM‑DPCC for drones in MuJoCo” milestone.

6. **Add the visual encoder and constraint extraction.**  
   - Render depth or segmentation from MuJoCo (or a side renderer) and plug in a light encoder (or X‑IL style FiLM‑ResNet) to produce both state proxy and obstacle constraints.  
   - Verify that changing obstacles or layouts shifts \(\mathcal{Z}_f^t\) correctly and that DPCC reacts.

7. **Only then worry about UAV‑Flow alignment and fancy experiments.**  
   - Use UAV‑Flow dataset for two things: (i) qualitative trajectory shape comparison, (ii) potentially pre‑training an FM policy on their trajectories before fine‑tuning in MuJoCo. [openreview](https://openreview.net/forum?id=fJMWkaT2HX)

If you restructure your notes around these seven steps, you’ll have a clean, linear “project spine” instead of the current tangled literature+idea dump. For your thesis proposal or HackMD, you can literally organize sections as:

- Existing wheels (DPCC, FM‑MPC, UAV‑Flow, MuJoCo UAV)
- New contributions (MuJoCo quadrotor‑FM‑DPCC with visual constraints)
- Implementation plan (steps 1–7 above)

If you want, paste your next draft of the project outline and I can edit it into a thesis‑ready “Problem–Gap–Method–Scope” structure.
