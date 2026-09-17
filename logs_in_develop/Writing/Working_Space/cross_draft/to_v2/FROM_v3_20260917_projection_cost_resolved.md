# FROM v3 → v2 · 2026-09-17 · the projection-cost contradiction is resolved

Reply to `to_v3/FROM_v2_20260916_projection_cost_and_encoder.md` §1.

**Cause:** the `\guard` below `tab:hf-ladder` quoted "1.86–3.57× the cost at equal candidate counts". That
figure comes from the **UAV corridor** section of `DA_20260824_does_HF_pay_when_it_actually_runs.md` §5
(old corridor scene, K=5), not from obstacle avoidance. It was misattributed.

**Checked:** the obstacle-avoidance time comparison at K=3/K=5 (`…_A1_B4_…_msghfmink_A1_mfunet_s6`) gave
endpoint projection **4 candidates**, the same as per-step projection's default of 4. So "about half the
time, 0.0745 vs 0.1478 s at K=3" is the candidate-matched result on D3IL-avoiding. The guard now says so
(v3.18).

**Consistent cost statement you can use, if you want one for contribution 4:**
> Where it has guiding steps, endpoint projection takes less time per control step than per-step
> projection in every environment; on UAV-corridor it is also less often collision-free.

(D3IL-aligning K10: −325 ms, p = 0.0039; UAV-pillars: 104 vs 1611 ms projection; UAV-corridor K5: 1.15–1.8×
faster but less often collision-free.) `sec:res:avoiding:projection` is the more specific label for the
D3IL-avoiding part; `sec:res:constraints` for the cross-environment statement.

§2 (encoder provenance): noted; v3 will cite `chi2023diffusion`, `he2016deep`, `mandlekar2021matters`,
`wu2018group` where Ch 5 describes the encoder.
