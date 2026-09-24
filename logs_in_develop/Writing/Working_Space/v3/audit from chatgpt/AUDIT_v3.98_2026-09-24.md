# Audit of the current v3.98 thesis draft

Prepared by **ChatGPT (Codex), 2026-09-24**. **Review only; no proposed fixes have been applied.** This Markdown report is the deliverable.

## 1. Snapshot, scope and limits

| Item | Audited state |
| --- | --- |
| Draft | `logs_in_develop/Writing/Working_Space/v3/` |
| Content revision | **v3.98**; the subsequent **v3.98a** entry records the copy of Chapters 7–8 and the appendix to v4, without changing these sources |
| Inherited foundation | **v2.27**, synced 2026-09-23 |
| Repository HEAD | `c885010bd18fb0816d1bba3dc317b06b52c420f7` |
| Working tree at audit start | Clean |
| Primary review | Chapter 5 setup, Chapter 6 results and synthesis, appendix, and current bundle behavior |
| Earlier audit consulted | `v2/audit from chatgpt/AUDIT_v2.25_against_v3.70_2026-09-23.md`, including its later feedback, exclusions and implementation record |

The scope follows the author's request: **no full verification of experimental data**. I checked argument structure, definitions, equations relevant to interpretation, displayed table/prose consistency, selected implementation paths, and a few targeted official-analysis records. I did not reconstruct every batch, verify every checkpoint or historical code version, rerun experiments, or certify every measurement. Source-code findings establish what the inspected implementation does; they do not prove the provenance of every historical result.

No training, evaluation, pipeline, or LaTeX compilation ran. Mechanical source checks are reported separately from scientific findings. This is not a rendered-PDF layout review or a new literature audit.

**Ownership:** Chapters 7–8 are v4's work. Their known placeholders and stale conclusions are not raised as unfinished v3 tasks. The appendix was also copied to `v4/FROM_v3_v3.98_20260924_201920/`; appendix findings below describe the currently included v3 content and are handoff issues for its next owner. This audit does not reopen the previous audit's dropped additions, request statistical tests, or prescribe new experiment campaigns.

### Location convention

All line numbers refer to this snapshot. Short names below mean:

| Short name | File under `logs_in_develop/Writing/Working_Space/v3/` |
| --- | --- |
| Setup | `chapters/05_setup.tex` |
| Results | `chapters/06_results.tex` |
| Appendix | `chapters/09_appendix.tex` |
| Twenty | `chapters/app_ntrial20_feasible.tex` |
| Method | `chapters/04_method.tex` |

Other paths are repository-relative. Snapshot SHA-256 values:

```text
05_setup.tex              5dc5a477a7a9f75089781d048b551bf17db1c2b9e46d3bbc5aa65595b43dff21
06_results.tex            88565545cf8a9b90399e20fe5117982c7b26643ff6dde440e1faef07bcb30717
09_appendix.tex           1e11470b4722f03d8de233f6720cd720ac5143c2c4da445fd76d4b678ea663bf
app_ntrial20_feasible.tex ddde881204c5781a89b79325d15d8f0c00c19bc78a34c09f0c761a24155b1f14
```

## 2. Overall assessment

**V3 has a coherent experimental story and useful evidence, but several headline claims and metric descriptions are stronger than, or different from, the comparisons actually reported.** The main work is to reconcile the existing text with its own tables and evaluation definitions. Most findings can be addressed without changing any measured number.

The central avoiding result survives this audit in a precise form: **CI-MeanFM at K=1 strictly improves on the displayed DPCC K=20 baseline in control steps and planning time while retaining 30/30 S&C.** MeanFM's displayed 29/30 row is a reliability–cost trade-off, not that same dominance result. The visual results support better positional progress and constraint outcomes for particular configurations, while reporting no completed in-position alignment in the final comparison. The UAV scenes provide distinct transfer, projection and controller observations; they should retain those distinct scopes in the synthesis.

### What should be preserved

- The sequence of proving the engine replacement on avoiding, extending to visual alignment, and examining transfer to the quadrotor is understandable and appropriate.
- The matched temporal U-Net comparison is much stronger evidence than a headline built around a larger transformer. Keep it central, with the training-recipe qualifications already disclosed.
- The diffusion K=2 comparison is a useful additional check on the K=20 headline. The large planning-time difference is not merely an arithmetic comparison of twenty steps against one.
- The distinction between collision with an undeclared pillar and violation of a declared constraint is valuable. Keep both outcomes visible.
- The alignment section eventually states the 0/10 in-position result and the projection/progress trade-off. Those limits should also govern its opening and summary language.
- The corrected distinction between setpoint lead and cross-track error, and the distinction between projected measured position and commanded position, are useful technical explanations.
- The UAV-corridor conclusion generally acknowledges residual violations and projector trade-offs. The thesis does not turn the data collection period into an unsupported real-time deadline.
- The new twenty-episode appendix correctly limits itself to complete matching low-K campaigns. Its count tables are useful supplementary evidence; the obsolete block is a separate integration problem.

## 3. Findings that affect the interpretation

**Priority meanings:** high = changes the meaning of a main claim, metric, control comparison or released result; medium = a substantive but more local inconsistency or unsupported inference. **Confirmed** means established by the displayed sources or inspected code, not by a full data audit. **Interpretation** means the evidence does not establish the stated causal/general claim.

| ID | Priority | Finding |
| --- | --- | --- |
| F01 | High | Tightening used by the planner is incorrectly described as the execution-scoring boundary |
| F02 | High | Dominance claims discard a success/violation difference that the tables retain |
| F03 | High | Alignment's percentage is not the median per-context share of distance closed |
| F04 | High | Plain and tightened unprojected alignment runs are treated as identical controls |
| F05 | Medium | Zero endpoint-guidance steps are confused with identical samplers; K=2 is also miscounted |
| F06 | High | The controller comparison uses the same planner in closed loop, not the same fixed plans |
| F07 | Medium | The 24× cost claim changes its timing denominator and scope |
| F08 | High | UAV finish-line crossing, successful scoring and episode termination are conflated |
| F09 | Medium | Corridor equivalence claims omit failed cells and extrapolate beyond the shared grid |
| F10 | Medium | Pillars transfer is described as unchanged or guaranteed beyond the observations |
| F11 | Medium | General setup tables retain incorrect margins and the old universal UAV configuration |
| F12 | Medium | Disclosed training differences are incorrectly excluded from interpretation of the comparison |
| F13 | Medium | Geometric illustrations and data simplicity are used to establish more than they measure |
| F14 | High, integration | Retired appendix results remain the target of current references and lose their warning in the clean bundle |

### F01 — Planning tightening and execution scoring are different sets

**Confirmed.** Setup:166–173 says unprojected violations are scored against the tightened boundary, as part of a thesis-wide convention. The inspected evaluators instead score against nominal constraints, with the appropriate body offset for the spatial UAV scenes.

Evidence:

- `scripts/eval.py:330–343` checks `constraint_list_polytopic_not_tightened` and the raw obstacle radii.
- `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:433–446,661–678` follows the nominal-boundary scoring convention.
- `mix_visual_aligning_test/eval_mix_visual_aligning.py:1519–1530` explicitly calls execution constraint checking with enlargement **0.0**.
- `mix_uav_test/eval_mix_uav.py:811–846` scores the flown position against raw geometry plus rotor reach, excluding planning margin and tightening.
- `uav_avoiding_bridge/scoring.py:91–110` uses the nominal avoiding geometry for pillars.

This affects the meaning of S&C and violation-free, not just a parameter footnote. Tightening is intended to create a buffer between a projected plan and the nominal execution boundary; the current sentence erases that distinction.

**Correction:** state that projected configurations use the tightened feasible set, while execution metrics check the nominal constraints, with the scene-specific vehicle offset where applicable. Distinguish plan-residual diagnostics from execution metrics. This finding does not call for rescoring all experiments.

### F02 — Strict Pareto claims do not follow after allowing a reliability difference

**Confirmed.** Results:697–703,2402–2406,2465–2477 says the average-velocity models Pareto-dominate DPCC. Its own table gives:

| Configuration | S&C | Control steps | Planning ms/step |
| --- | ---: | ---: | ---: |
| DPCC diffusion, K=20, c | 30/30 | 70.1 | 553.4 |
| CI-MeanFM, K=1, t | 30/30 | 59.2 | 18.1 |
| MeanFM, K=1, t | 29/30 | 58.6 | 18.3 |

CI-MeanFM supports the strict claim. MeanFM trades one successful constrained episode for lower costs. Saying this explicitly in the following sentence does not make the preceding plural dominance statement correct.

The cost-frontier definition explains how the overstatement arises: Results:456–460 admits S&C within **0.05** of the panel maximum, then claims a less reliable configuration cannot appear as a favorable trade. At 30 episodes that rule explicitly admits 29/30 beside 30/30. It defines a **cost frontier within a reliability tolerance**, not dominance on all three measured quantities.

Two related uses need the same qualification:

- Results:626–633 excludes every K=3 endpoint point as dominated by K=1 at “the same success.” The comparison spans a four-seed K=3 table and five-seed K=1 results, including MeanFM 1.000 versus 0.967. Higher planning cost alone does not establish dominance of every point on success, steps and time.
- Results:1825–1846 groups corridor rows into violation bands and then calls the baseline dominated. The grouping is a legitimate presentation choice, but the exact violation differences remain: e.g. hump diffusion 0.0 versus FM 0.2 or 1.8 violating steps. The statement is dominance **on steps and planning time within that band**, not constraint-preserving dominance.

**Correction:** lead with CI-MeanFM's strict avoiding result; describe MeanFM's observed trade-off. Label tolerance/band frontiers by the dimensions they actually compare. Describe K=1 as the selected low-cost operating point without claiming an unsupported universal dominance over all endpoint configurations. No new statistical test is required.

### F03 — Alignment's distance percentage is misdefined

**Confirmed.** Setup:734–737 defines the share of each episode's initial distance closed, `(d_init − d_final)/d_init`, with zero meaning no movement towards the target. Results:898–905 calls the plotted value the median of that share over contexts.

But Results:911 states the implemented plotted quantity:

```text
100 × (1 − median(final distance) / 0.4530)
```

Here **0.4530 m is the mean initial distance across contexts**. A ratio of a median final distance to a mean initial distance is not the median of paired per-context progress fractions. It does not have the claimed no-motion reference.

The displayed table exposes the issue without a full data audit: Results:1203 reports diffusion under cumulative-cost projection with **0/10 moved**, median distance **0.456 m**, and **−1%**. A targeted read of the ten source initial distances gave mean **0.452980 m** and median **0.455746 m**. Even leaving every box unmoved produces approximately **−0.61%** under the draft's normalization. A negative displayed value therefore need not mean the boxes moved farther away.

**Correction:** either retain the existing values and name them as final median distance normalized by the mean initial distance, dropping the exact “0% = unmoved” interpretation, or calculate the paired per-context fractions before taking their median. The second option changes the percentages and figure axis, so it is a subsequent data/figure edit, not something silently done in this audit. The distance-in-metres comparisons remain usable.

### F04 — The unprojected alignment reference is not invariant to the tightened evaluation configuration

**Confirmed.** Results:769–771 says the constraint set has no bearing on an unprojected plan. Results:1284–1295 then calls the plain unprojected K=20 MeanFM point the projected point's “twin,” reading the projection cost from **84% at 190 ms** to **61% at 266 ms**.

The chapter itself provides a better matched reference in Results:1085: the tightened configuration with projection off has median **0.090 m**, **80%** under the current normalization, and **172.5 ms**. The plain unprojected model table/figure uses approximately **0.0741 m** and **190.5 ms**. These are different evaluation records, not interchangeable copies of the same outcome.

Results:1236–1242 also explains why changing the constraint configuration can affect an unprojected evaluation: tightening triggers the box–obstacle guard in one context, which is never attempted and is counted violation-free and unmoved. Thus “projection switched off” does not establish that every other evaluator behavior is unchanged.

**Correction:** use the same tightened evaluator's `none` row as the control when attributing changes to projection, or explicitly label the broader configuration change. Keep the exclusion disclosure and report both the ten-context bookkeeping and nine attempted contexts consistently. The relative ordering between similarly treated projected rows can remain informative; “comparisons are unaffected” should not be extended to the plain/tightened comparison.

### F05 — Terminal-only projection does not prove identical plans, and K=2 is described inconsistently

**Confirmed.** Setup:1425–1428 says K=2, η=0.5 endpoint and per-step projection “compute the same plan” and any difference would be noise. Setup:1431–1434 limits the average-field sampler difference to cases with guiding steps.

However, the endpoint implementation supplies `h=0` on **every transport step**, including when there are no guiding steps: `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:838–862,955–960`. The ordinary average-velocity sampler uses the finite interval: `flow_matcher_v3_meanflow/models/mf_diffusion.py:278–279`. The endpoint diagnostic itself qualifies equivalence by solver/variable scope at `hardflow_projection.py:904–907`. Zero nonterminal guidance does not eliminate these differences.

Separately, Results:680–685 correctly gives **zero** guiding steps at K=2, η=0.5, but Results:747–748 says K=2 has **one**. A terminal projection step is being called a guiding step in the conclusion.

**Correction:** retain the low-K degeneracy result: these settings cannot demonstrate the benefit of nonterminal endpoint guidance. Extend the existing sampler qualification to all endpoint configurations. Reserve an identical-rollout statement for the specific checked implementation/configuration; do not promote it into algorithmic equivalence. Correct the conclusion's count with its threshold stated.

### F06 — The controller experiment changes a closed-loop system, not the tracking of fixed identical plans

**Confirmed.** Results:2195–2202,2227–2229,2328–2341,2423–2425 and2488–2490 repeatedly says “keeping the plan” or “the same plans” under two controllers.

The evaluator instead constructs observations from the current measured position, replans, updates the setpoint, and executes the chosen controller: `mix_uav_test/eval_mix_uav.py:1836–1845,1861,1889,1904–1911`. A controller changes later states and therefore later model inputs and generated plans. Results:2205 acknowledges this directly: the plans generated from the MPC-controlled states cut the corner deeper.

The official `Data_Analysis/DA_in_Paper/analysis/DA_20260924_scurve_R44bc_projection_controller.md:123–124,151–153` also reports different commanded-path clearances, **−0.076 m versus −0.203 m**.

**Correction:** say the same trained planner and sampling/projection configuration are evaluated under two controllers **in closed loop**. The measured comparison supports a controller-dependent system outcome. It does not isolate tracking performance on replayed identical command sequences. No fixed-plan replay experiment is required to state the existing result correctly.

### F07 — Twenty-four times cheaper refers to a residual, not total per-step cost

**Confirmed.** Results:2308–2310 gives two different ratios:

| Timing quantity, unprojected comparison | Cascaded geometric | MuJoCo MPC | Ratio |
| --- | ---: | ---: | ---: |
| Whole-loop elapsed ms/step | 14.0 | 135.8 | 9.7× |
| Controller-plus-simulator residual ms/step | 5.2 | 125.4 | 24.1× |

Results:2331–2344,2423–2425,2528–2530 repeatedly describes the second ratio simply as cost per control step, in prose covering both projected and unprojected comparisons. The source note at Results:2320–2322 gives the **projected** whole-loop costs as **159.4 versus 348.6 ms**, about **2.19×**.

The residual is not timed around the controller call. Results:2323–2326 says so, but then claims its difference isolates the controller. Different execution environments and resource interaction are acknowledged at Results:2316–2322, so that stronger isolation claim is not established.

**Correction:** name the measured quantity and configuration: 24× for the controller-plus-simulation residual in the unprojected comparison; 9.7× for that comparison's total loop. Retain the observation that the cascaded implementation is cheaper, without treating the residual as an isolated controller benchmark or transferring its ratio to projected runs.

### F08 — Finish-line crossing is neither the complete UAV success rule nor the termination event

**Confirmed.** Results:2007–2010 says a flight ends by crossing the finish line, reaching the limit, or inversion. Results:2018 defines success simply as crossing. This conflicts with both the evaluator and Setup:759–775.

In `mix_uav_test/eval_mix_uav.py`, crossing the wall-end line is latched at :1933–1936, proximity to the route goal is separately latched at :1940–1941, and early stopping uses the latter at :2002–2008. Crossing the finish line can therefore be followed by more flight. The reported success is `crossed_line and safe` at :2051; `safe` additionally checks contact, altitude and abort conditions at :2015–2040.

This distinction matters to the counts and to the window over which violating steps are accumulated. Results:2202–2204 itself recognizes a flight close to the goal that fails because of unsafe contact.

**Correction:** define success consistently as a finish-line crossing that also passes the stated controlled-flight checks. State the actual stopping conditions separately. Avoid relabeling the existing success counts as raw crossings. Clarify that latching a crossing does not stop the rollout or discard later safety checks.

### F09 — Corridor similarity is generalized beyond the eligible rows

**Confirmed.** Results:1963–1971 says the projected objectives differ by at most **1.6 violating steps** at a shared budget and projection method. But the hump K=2 per-step rows at Results:1756,1759,1762 read **17.9, 19.8 and 0.0**, with success **0.1, 0.2 and 0.0**. The sentence needs the restriction to the successfully traversing configurations on which the claimed similarity is based. The stalled zero-violation row is not evidence of equivalence.

Results:2524–2527 then recommends “any flow-based objective” while describing the K=1 to20 trade. The grid at Results:1646–1648 tests MeanFM/CI-MeanFM at **K=1,2,3**, whereas **K=5,20 are FM-only** in this corridor comparison.

**Correction:** limit the similarity observation to jointly tested configurations that complete the passage, and describe the higher-budget extension as an FM result. The simple-data interpretation can remain, but neither objective equivalence over the entire ladder nor untested high-budget performance follows from this grid.

### F10 — Pillars' empirical transfer is overstated as unchanged behavior or a tracking bound

**Confirmed.** Results:1411 says “Before projection the vehicle changes nothing.” Results:1436–1446 instead shows raw MeanFM K=2 success falling from **30/30 to24/30**, and CI-MeanFM K=2 from **30/30 to27/30**; nine of120 unprojected flights collide. Violating-step means also change, e.g. **15.5 to19.9**. The broad pattern transfers, but the numerical behavior is not unchanged.

Results:1597–1608 describes tracking as staying within about **0.007** in avoiding coordinates. Its source note at :1614–1616 identifies the underlying statistic as **the mean across flights of each flight's 95th-percentile gap**, **0.0066**. `Data_Analysis/DA_in_Paper/analysis/pillars_v2_live.py:159–163` confirms that aggregation. It is not a maximum or a deterministic tracking bound.

Similarly, “what the per-step projector guarantees … it guarantees on UAV-pillars” at Results:1483–1488 turns an observed result into a general execution guarantee. The important empirical result is zero declared-constraint violations on the tested projected flights **before termination**, including flights terminated by contact with undeclared pillars.

**Correction:** describe the broadly preserved violation pattern and the additional collision failures; name the tracking percentile; state the observed zero-violation result on the tested flights. Preserve the distinction between declared constraints and omitted obstacles without claiming a universal closed-loop guarantee.

### F11 — Setup tables contain conflicting margins and retain the old universal UAV description

**Confirmed.** Several generic rows have not caught up with the scene-specific definitions that are already correct elsewhere:

| Item | Conflicting statement | Current evidence / correction |
| --- | --- | --- |
| Visual tightening | Setup:1201–1202 and1269 give 0.025 m for alignment/all environments | Setup:240–243,317–330 and `config/visual_aligning_eval.yaml:152` give **0.03 m** for alignment |
| UAV action and interval | Setup:57–61,100–104 describe every quadrotor as spatial increments at0.03 s | Pillars uses the planar avoiding planner, fixed altitude and **1 s** interval: Setup:374–379,1367–1372; `uav_avoiding_bridge/plant.py:42–44` |
| UAV training | Setup:1114–1128 labels a 12D-plan/9D-state column “UAV, all scenes”; :1137–1139 generalizes the layout | Restrict this to corridor/s-curve; pillars inherits the avoiding planner/training configuration |
| Inflation | Setup:662–665,781–783 generalize the0.31 m scoring offset to every quadrotor scene | Setup:450–452,487–491 separately says pillars has no added inflation band; `uav_avoiding_bridge/frame.py:13–24` uses radial reach0.36 m for transfer scaling |
| Spread | Setup:833 calls every avoiding table multi-seed; :840–843 calls all UAV scenes single-seed | Main avoiding and pillars use training-seed spread; named one-seed diagnostics, alignment, corridor and s-curve follow their stated context/flight conventions |

**Correction:** qualify the shared tables by scene and retain the specific pillars exception throughout. State the coordinate system for pillars' tightening: **0.025 in avoiding coordinates corresponds to0.90 m in the scaled arena**. These are setup-document corrections, not missing experiments.

### F12 — Training recipes remain part of the comparison

**Confirmed interpretive overstatement.** Setup:1164–1169 discloses that the average-velocity models on avoiding use four times the batch size, five times the learning rate, and EMA weights, unlike FM and diffusion. Setup:1171–1174 then says how each objective was brought to convergence is a property of training “and not of the comparison.”

The disclosed differences do not invalidate the measured system comparison. They do limit an explanation that attributes all gains exclusively to the objective. Matching a backbone is valuable; it does not turn different recipes, priors and checkpoint treatments into an objective-only intervention.

**Correction:** retain the architecture-matched result and existing recipe table, and describe the measured comparison as the trained configurations under those recipes. Remove the dismissal of their relevance. This finding does not request retraining or a larger hyperparameter study.

### F13 — Geometric illustrations and simple data do not establish the stronger causal statements

**Interpretation; confirmed evidence gap.** Three related inferences need narrower wording:

1. Setup:307–311 says the alignment demonstrations respect neither constraint. The described supporting picture at Setup:547–554 uses straight segments between **box centres**, whereas the constraint is imposed on the **end effector** (Setup:236–237,317–318). A context-level straight segment is not a recorded end-effector demonstration, and the quoted crossing count does not establish compliance with both constraint families.
2. Setup:630–638 moves from a feasible s-curve reference route to “nothing in the data has to be corrected,” a plan “never broken,” and a categorical inability of the scene to order projectors. Setup:658–660 explicitly says the figure shows generator references, not actual flights; the acceptance procedure also tolerates some contact. A feasible reference neither certifies every demonstration nor makes learned samples feasible.
3. Results:2350–2355 and2480–2484 treats simple generated data as the demonstrated reason FM leads. The grid establishes model ordering on this route. It does not vary data complexity while controlling the other differences, including the sampler initialization and closed-loop effects.

**Correction:** describe what the illustrations actually show, and keep data simplicity as an interpretation consistent with the results. Retain s-curve's role as a controller case study. No demonstration-wide verification is needed simply to avoid claiming that a geometric proxy proves actual compliance or causality.

### F14 — Retired appendix results are still reachable as current evidence

**Confirmed integration problem; partly already recorded in the v4 handoff.** The valid new twenty-episode section at Twenty:23–36 excludes incomplete K=20 and unmatched CI-MeanFM campaigns. The older section remains inside `deadblock` at Appendix:290–550, including the incomplete baseline and obsolete tables.

Three issues remain in the current build:

- Setup:1337–1338 and Results:759–761 still direct the general twenty-episode discussion to **`app:avoiding-twenty-episode`**, the retired block, rather than **`app:avoiding-twenty`**, the replacement. Setup:917–922 intentionally points to the old incomplete baseline record, so that reference needs its own decision rather than a blind global rename.
- The `deadblock` starts at Appendix:290 **before** the current UAV-corridor trajectory section at :292–328. That active figure is therefore inside the “dead twenty-episode” wrapper too. This is a scope error in the annotated copy.
- `bundle/make_bundle.py:203–221` deliberately makes `deadblock` print normally in the clean variant and hides `outdated`/`flawed`. Consequently, the clean copy contains both the retired and replacement twenty-episode sections without the old block's warning banner. This behavior is documented and intentional for layout inspection; it is still unsuitable as a representation of which experimental results are current.

**Correction for integration:** route current supplementary-result references to the curated section; separate the active corridor figure from the retired block; exclude withdrawn material from a submission build or retain an unmistakable archival status. Preserve the old material in the repository if desired. The author's pending decision about dropping the retired block remains a handoff issue, not authorization for this audit to delete it.

## 4. Smaller corrections and prose cleanup

These are concrete local issues, not requests to expand the thesis.

| Location | Issue | Suggested treatment |
| --- | --- | --- |
| Results:778 and final alignment summary | “One model moves the box to its target” reads as successful completion, while the final comparison reports0/10 in position | Say it moves the box substantially **towards** the target; keep the measured distance and completion distinction |
| Results:1236–1242 versus1270–1278; Appendix:275–278 | The frozen alignment case is called context8 in the guard and context7 in the displayed-context discussion; raw episode position, stored id and prose order differ | Use the figure's displayed numbering consistently, with the raw-id mapping only in provenance |
| Results:1241–1242 | “Whether the chapter states it is the author's call” is in a `guard`; the clean bundle prints guard text | Remove this editorial instruction from thesis prose while preserving the actual exclusion caveat |
| Setup:1229 | Endpoint lookahead is described as starting from the projected state | `hardflow_projection.py:994–1006` evaluates `X_ref` before the NLP projection; describe that order correctly |
| Setup:1261–1263 | “One more projection whenever ηK is integral” omits endpoint-threshold cases | Qualify to interior thresholds; atη=1 both projectK times. The inherited method already gives the bounded formulas |
| Setup:162 | Every avoiding obstacle is assigned radius0.025 | `uav_avoiding_bridge/frame.py:34–41` gives the first radius0.03 and the other five0.025; reconcile with the scene definition |
| Results:2290–2291 | MPC is called the controller that flies the scene, versus one that does not | This is stale pilot wording: the final raw pair is9/10 versus9/10, and the projected pair favors the cascaded controller |
| Appendix:837–868 | Old s-curve captions claim six evaluated variants and all-zero outcomes, while the tables contain projected cells marked “not run” | Remove/archive the obsolete tables or make captions describe the retained content |
| Appendix:1043–1069 | The corpus overview still includes retired-model/coverage descriptions and a universal claim that later batches reproduce every earlier value | Update it to the final tables and actual source versions during the appendix handoff; avoid treating a batch timestamp as proof of invariant results |

The main prose problem is repetition of an overstrong sentence at several levels: subsection conclusion, environment conclusion, §6.4 item, and final synthesis. Correcting only the sentence beside a table will leave the same error in the thesis headline. Apply eventual claim corrections to all cited locations together. Captions should define the quantities actually drawn; editorial requests and reasons for organizing the draft belong in provenance or the audit.

## 5. Potentially suspicious data: checks worth targeting, not established errors

### D01 — MeanFM's single main-table failure conflicts with the expected nested evaluation

**High-priority provenance question; not resolved by this audit.** The official supplementary analysis already records this:

`logs_in_develop/Writing/Working_Space/ntrial20_appendix_20260924/DA_20260924_ntrial20_feasible.md:262–288`.

For **seed7 / top-right-hard / temporal consistency**, the two-episode MeanFM K=1 andK=2 record has **1/2** success, while the twenty-episode record has **20/20**. The analysis reports the same checkpoint and intended episode-seeding scheme, so the smaller evaluation should be nested by design. The discrepancy recurs across additional K settings. It also notes duplicate configuration snapshots suggesting a possible older result folder, but that explanation is unproven.

This is important because the single failure determines whether the main MeanFM row is29/30 or30/30. **Do not silently change it to30/30, or use a subset-range calculation as proof that the discrepancy is harmless.** The targeted next check is the identity and timestamps of the two result folders and episode0–1 trajectories for that seed/geometry; it does not require a full corpus verification or a new campaign. Until resolved, use the printed29/30 in claims, as in F02.

### D02 — The twenty-episode campaign cannot validate every main-text model

**Coverage caution, largely handled correctly in the new appendix.** Twenty:33–36 and the official census exclude incomplete K=20 campaigns and the unmatched twenty-episode AlphaFlow/SiT configuration. The matching CI-MeanFM twenty-episode evaluation covers seed6 only; the supplementary analysis records **57/60** for its K=1 temporal-consistency configuration there, while the main comparison has30/30 across five seeds with two episodes per geometry.

Those are different coverage and episode counts. Neither is automatically wrong, and the single-seed result is not a replacement for the five-seed row. It does mean “the larger evaluation confirms every model” would be unsupported. Keep the replacement appendix's explicitly model-limited scope and do not recover a CI-MeanFM claim from the retired block.

### D03 — A violation-free alignment count contains one unattempted context

**Disclosed scoring artifact, not evidence of fabricated data.** Results:1236–1242 says one context is frozen by the evaluator and counted violation-free. Thus10/10 is nine attempted contexts plus one guard case; 1/10 can mean zero successful constraint-safe attempts among the other nine.

Keep the denominator convention visible and consistent, particularly in summaries of endpoint projection. Do not treat this as task completion or as ten active successes. The context-label mismatch is listed in §4; the more substantive plain/tightened comparison issue is F04.

### D04 — Some timing differences warrant restricted claims, not automatic rejection of the measurements

The avoiding FM K=20 random row has **852.6±624.9 ms** in Appendix:184, while other rules are much lower. Results:623–624 also acknowledges per-step times around64 versus146–148 ms across different model evaluations. Such variation can reflect solver workload, failures and shared-machine effects. The thesis should compare the stated metric and matched configurations, and avoid causal conclusions from a small timing difference alone.

Conversely, the surprising diffusion K=2 cost is already accompanied by an input-to-projector explanation in Results:709–714. Its being surprising is not sufficient grounds to label it corrupt. This audit does not establish that any of these timing records are wrong.

## 6. Mechanical and integration checks actually performed

| Check | Result and limit |
| --- | --- |
| Existing `tools/check.py` | Passed:14 files,290 labels,53 cited keys/53 bibliography entries,41 unique figures. No missing references or reported structural errors |
| Independent recursive input/reference/citation scan | Passed:15 source files,293 labels, no unresolved references, duplicate labels or missing citation keys, including `app_ntrial20_feasible.tex` |
| Current bundle verification | `bundle/make_bundle.py --verify` passed for the newest full-clean bundle:14 inlined content sources,0 collapsed, byte-faithful to the tree |
| Figure-file availability | All41 referenced figure names have PNGs;21 also have SVGs; none has a PDF in the draft |
| Figure copies versus official store |61 existing PNG/SVG copies are byte-identical to a matching file in `Data_Analysis/DA_in_Paper/figures`; the remaining PNG is the inherited reproduced endpoint illustration, not a newly detected missing result figure |
| Compilation and rendered layout | Not performed; no claim of successful typesetting, page fit, final numbering or visual legibility |

The existing checker reads only the master's direct inputs (`tools/check.py:48–64`), so its “all checks pass” does **not** cover the newly nested twenty-episode file. The independent recursive check above covers reference/citation integrity for that file in this snapshot. The checker should eventually recurse, but no currently broken label is alleged.

The six counted `hole` calls are in the deliberately deferred Chapters7–8 and reproducibility appendix. They are not six missing experimental results. Likewise, a source checker cannot detect F14: a reference can resolve perfectly while pointing to a retired section. Byte-faithful bundling proves copying fidelity, not scientific consistency.

No mandatory vector-conversion task is revived here: the previous audit left figure-format delivery advice optional. Availability of a PNG also does not certify publication-size readability.

## 7. Recommended order for a later revision

1. **Fix definitions and controlled-comparison wording:** F01, F03–F08. These decide what the numbers mean and what was held fixed.
2. **Reconcile the conclusions with the tables:** F02, F09–F10, F12–F13, updating every repeated occurrence in §6.4. Preserve the supported CI-MeanFM, visual-progress and controller-system results.
3. **Correct setup tables and local inconsistencies:** F11 and §4. These are mostly short source edits.
4. **Complete appendix integration with its owner:** F14 and the stale appendix captions/corpus overview. Retain the author's distinction between working archives and current evidence.
5. **Investigate D01 before upgrading the MeanFM headline.** The other watch items can remain explicitly scoped; none authorizes invented values or demands full data verification.

This is an audit recommendation, not an implementation checklist already accepted by the author. No thesis, figure, bibliography, code, configuration, bundle, changelog, cross-draft inbox, master index, Claude file or v4 file was changed by this review. No commit was made.

**Signed:** ChatGPT (Codex), auditor,2026-09-24.

---

## Subsequent reviewer feedback

Reserved for the project review workflow: append signed, evidence-based verdicts on the finding IDs below this point. No second-reviewer verdict or agreement is implied by this audit.
