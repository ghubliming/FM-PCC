# FROM v3 → v2 · 2026-09-18 · v3.32 · which velocity-setpoint policy the experiments use, and why

**What v2 has.** §4.8 defines the cascaded geometric controller and all three velocity-setpoint policies in
`eq:method:dep:vdes` — increment-as-mean-velocity (the default), brake-to-rest ("stop-and-go"), and
constant-speed — and says the choice "is a recorded experimental factor, not an implementation detail". It
also records that `pid` in the run tags is an artefact token for the cascaded controller.

**What is missing.** v2 never says **which** policy the reported experiments use, and gives no reason for
it. Chapter 6 needs that: every quadrotor result in the thesis is flown with **brake-to-rest**
($\dot{p}\sidx{des} = 0$), and the s-curve caveat turns on it (`tab:uav-controller`). Without a sentence in
§4.8 the results chapter either repeats the definition or leaves the reader thinking "brake-to-rest" is a
different controller from the cascaded one — it is the same controller with one setpoint choice.

**Suggested addition to §4.8, after `eq:method:dep:vdes`** (the material exists in the repo):

- Every reported quadrotor result uses **brake-to-rest**; the other two policies are implemented and were
  measured, but are not the configuration of record.
- The reason the first policy is not used: $\Delta p_t / (n\sidx{dec}\Delta t\sidx{phys})$ divides by an
  assumed constant interval between plan queries, so it is **timing-sensitive** — jitter in when the plan
  arrives changes the commanded speed — and it is a finite-difference estimate of a velocity the expert
  collection produced analytically. Source: `Gen11/Epoch8_UAV_Mjpc_thrust_control/U3_v_des_Patch/
  PLAN_pid_const_v.md` §"Problem with Current `pid` `v_des`".
- What brake-to-rest costs, which v3 §6.3 then builds on: with $\dot{p}\sidx{des} = 0$ the loop needs a
  standing position error to move at all, so a tracking lag is structural. Measured on UAV-pillars: under a
  coherent plan the vehicle realises 0.90 of the commanded displacement with lag below 0.65 m; under an
  incoherent one it realises 0.38 and the setpoint runs up to 2.86 m ahead. Source:
  `Gen15/Study/STUDY_20260901_mf_unguided_failure_uav_pillars.md` §5–5.2.

**What v3 did meanwhile.** `tab:uav-controller` now names the row "cascaded geometric, brake-to-rest"
against "MuJoCo MPC", its caption points at `eq:method:dep:vdes` and `sec:method:mjpc`, and
`sec:res:uav:controller` states once that every result uses the brake-to-rest policy. v3 states the choice;
it does not argue it, which is §4.8's job.
