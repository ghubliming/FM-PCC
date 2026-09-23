# FROM v2 → v3 · 2026-09-23 · v2.26 · five findings from the audit check that live in Chapters 5–6

The author had a ChatGPT audit of v2.25 written (`v2/audit from chatgpt/AUDIT_v2.25_against_v3.70_2026-09-23.md`)
and asked v2 to check it. Its §8 holds the verdicts. Six of the confirmed points reach into v3's chapters; each
below names the file and line it rests on. None is a request to change code.

## 1. 🔴 The instantaneous-velocity sampler starts from σ = 0.5 and was trained at σ = 1

`flow_matcher_v3/models/diffusion.py:164` (and `:241`): `x = 0.5 * torch.randn(shape)`. Its `p_losses`
(`:284–297`) draws `x_base = torch.randn_like(x_start)` — σ = 1. The same pair is in
`mix_visual_aligning/models/fm_diffusion.py:164/284` and `flow_matcher_v3_uav/models/diffusion.py:184/354`.
MeanFM (`mf_diffusion.py:204`, "sigma=1.0 to match q_sample training noise"), CI-MeanFM and the endpoint
sampler (`init_noise_scale=1.0`) run at σ = 1. The repo already knows: `hardflow_projection.py:52–70` (fix_4)
calls the 0.5 start "an out-of-distribution tau=0 state" and made σ a required argument for that reason.

So the **FM arm is sampled off its training prior**, and **FM and the two average-velocity models do not share
a prior scale**. Neither undoes a measurement. Both belong in Chapter 5's sampler description (one sentence
each) and in Chapter 7 wherever an FM-vs-MeanFM difference is read as a property of the objective. The
audit found only the diffusion baseline's `0.5` factors (`diffuser/models/diffusion.py:158,168`, plus
`clip_denoised`), which are Diffuser's and DPCC's own and belong in the baseline paragraph.

## 2. The endpoint sampler runs MeanFM / CI-MeanFM as instantaneous-velocity models

`flow_matcher_v3_meanflow/sampling/hardflow_projection.py:838–863`: `_velocity_batch` calls
`_predict_velocity(traj, cond, t, h=torch.zeros_like(t))` — the average-velocity head at **h = 0** — for
*both* the Euler transport `X_ref = X + V·dt` and the lookahead. The plain sampler (`mf_diffusion.py:275–279`)
queries `h = 1/K`. The code comment's "u(x,t,0) = v(x,t) EXACTLY" holds for the ideal field, not for the
network's two outputs, which are trained on different slices of the (τ, h) domain.

Consequence: every endpoint-vs-per-step cell on MeanFM or CI-MeanFM — the MeanFM K3 cell (T5), the
corridor and pillars endpoint rows — compares **two samplers on one checkpoint**, not two projection points
on one sampler. Chapter 5's "constraints and solver are those of iterate projection, and only the point of
projection differs" (v2 says it too; it will be fixed there) needs the clause; Chapter 6's reading of those
cells should not attribute the difference to the projection point alone.

## 3. The two per-step gates do not fire the same number of times

Baseline (`diffuser/models/diffusion.py:175–192`): `t` descends K−1…0, projects when `t <= ηK` → at K=20,
η=0.5 that is t ∈ {0,…,10}: **11** projections. Flow samplers (`flow_matcher_v3/models/diffusion.py:173–191`):
`loop_idx >= int((1−η)K)` → 10…19: **10**. At K=2, η=0.5: 2 vs 1. The flow gate was matched to DPCC's
floor rounding (Gen12 fix 8) but not to its `≤` polarity, so the counts differ by exactly one whenever ηK is
an integer. Wherever Chapter 5 says the projection methods or the models "share the activation threshold",
that is true of the value and not of the count of solves; `tab:eval` and the protocol text should say which.

## 4. One guiding step: v2's rule and v3's corridor reading disagree — author's decision

v2 §4.5.4 (`sec:method:degenerate`) labels `n_guide = 1` "no attributable effect" and its reporting rule says
such cells are marked and never averaged. v3 reads the corridor's endpoint cells from K=3 at η=0.5
(`06_results.tex:1428–1432`), where `n_act = 2` and `n_guide = 1`, and §6.1.5 (`:640–642`) counts "one guiding
step at K=2" as a valid configuration. Either v2 relabels the row "one guided step: limited evidence,
reported as such" (v2's recommendation — it is what v3 already does) or v3 stops reading K=3 endpoint cells.
v2 will not change the rule until the author says which.

## 5. NFE ≠ K under endpoint guidance

`hardflow_projection.py:955–998`: per candidate, `NFE = K + n_act − 1 = K + n_guide` (the terminal lookahead is
skipped). K=20, η=0.5 → 29; K=3, η=1 → 5. Chapter 6's cost axis is the time to compute one action, so no table
moves; but any Chapter 5 sentence that equates the step budget with the number of network evaluations needs
"for the unguided and per-step samplers".

## 6. UAV-pillars: Chapter 4 now describes the loop; Chapter 5 should print its constants

v2.26 §4.6.3 describes the pillars loop from `uav_avoiding_bridge/frame.py` and `plant.py` (similarity map,
altitude held, mapped setpoint tracked for one control period by the cascaded geometric controller with a
rate-limited reference and no feed-forward, measured position mapped back, contact = failure), values in a
`\srcnote` only. Chapter 5's protocol (`tab:protocol`, `sec:setup:protocol:uav`) should carry them:
**scale 36, altitude 1.0 m, one setpoint per 1 s = 100 physics steps of 0.01 s, reference speed ≤ 1 m/s,
velocity feed-forward off, contact terminates the episode** (`plant.py:42–44, 56–63, 200–215`). The corridor's
0.03 s interval does not apply on that scene, and `tab:eval`'s per-scene rows should not imply it does.

**Also in v2.26, for the sync:** §1.4 is now two stages (1–4 on the benchmark, 5–6 beyond it; item 6 rewritten
for pillars-v2 / corridor-v3 / s-curve-caveat); §4.6.3 rebuilt; `eq:method:env:switched` restated as the
per-replan wall selection of `eval_mix_uav.py:1826–1837`; new label `eq:method:env:pillarsmap`;
`tab:embodiments`' third column is "UAV-corridor, UAV-s-curve".

**Written by:** Claude (Claude Code) · 2026-09-23 · FM-PCC dev container. Every file and line above was read
in this session. No code changed. **Not compiled.**
