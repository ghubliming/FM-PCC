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

## 8. Feedback from the v3 owner — Claude (Fable 5.1, Claude Code), 2026-09-24

**What was done.** Every finding was checked against the snapshot the audit hashes (commit `c885010b`,
whose `05_setup.tex` / `06_results.tex` SHA-256 match §1) and against the current HEAD (`65bca136`,
v3.99, which is the snapshot plus the seventeen appendix re-pointings of the v3.99 pass), against the
cited code at its current lines, against the official analysis notes, and — for F03 — against the batch
CSV the alignment tables are built from. **Nothing was applied**: no chapter, code, config, figure,
bundle, INBOX or v4 file changed; nothing compiled; nothing committed. Verdicts are below, the evidence
in §8.2–8.3, the disagreements in §8.4, and who would own each fix in §8.5.

**Line convention.** The audit's line numbers are those of `c885010b`. Setup is unchanged through line
917 and shifted by −2 from line 923 on; Results is shifted by +3 from line 134, +6 from line 414, +5
from line 762, +6 from line 2059 and +7 from line 2160 (only the seventeen v3.99 passages differ). Where
it matters, "HEAD n" gives the current line.

### 8.1 Verdicts

| ID | Verdict | In one line |
| --- | --- | --- |
| F01 | **Confirmed** | All five evaluators score violations on the nominal set (UAV: nominal surfaces ⊕ 0.31 m); the projector alone sees the tightened set. One sentence, Setup:172–173, is wrong; nothing else in Ch 5/6 or in the synced Ch 4 copy repeats it. No rescoring. |
| F02 | **Confirmed** (by the thesis' own rule) | CI-MeanFM K=1 strictly dominates the baseline; MeanFM K=1 (0.967) is a trade-off under the definition at Results:456–460. The plural "the average-velocity models Pareto-dominate" stands at four places. K=3: only MeanFM is not "at the same success". Corridor: "dominated" is within a violation band; the conclusion already says "trade". |
| F03 | **Confirmed**, magnitude measured | The printed percent is 100·(1 − median final / 0.4530) (the DA that filled the table says so). Recomputed from the CSV: initial distances mean 0.452980 m, median 0.455746 m — the audit's numbers exactly. Paired per-context medians differ from the printed values by 0–3 points; an unmoved cell prints −1 % instead of 0 %. |
| F04 | **Confirmed** | The tightened evaluation freezes one context (flag set on one rollout of every tightened cell); its unprojected row reads 0.0902 m / 80 % / 172.5 ms, the plain-set run 0.0741 m / 84 % / 190.5 ms. The grey "twin" ring and the "no bearing" sentence use the plain-set run. |
| F05 | **Confirmed, and stronger than stated** | The project's own analysis of 2026-08-24 records that at K=2, η=0.5 the two arms are bit-identical **on FM only** and **differ on MeanFM**. The code passes h=0 on every transport step. Setup:1425–1428 is false for the two average-velocity models; Results:663 names no model. The K=2 guiding-step count is 0 at η=0.5 and 1 at η=1.0 — both appear, the threshold is missing at 747–748. |
| F06 | **Confirmed** | The evaluation re-plans from the measured state every control step; the plans under the two controllers are different plans. The table title, caption and five prose passages say "the same plans". |
| F07 | **Confirmed** | Loop 135.8/14.0 = 9.7×; controller-plus-simulator residual 125.4/5.2 = 24.1×; projected loop 348.6/159.4 = 2.2×. "A twenty-fourth of the cost per control step" (2331–2344, §6.4) names the wrong quantity; 2528–2530 ("the controller's cost") is right. |
| F08 | **Confirmed** | No exit on the finish-line latch: a flight ends on abort, on reaching the goal point, or at the cap. Setup:759–775 already says so; Results:2007–2010 and the Table 6.13 caption contradict it. |
| F09 | **Confirmed**, with the arithmetic | Among jointly tested cells whose flights reach the end of the corridor the spread is 0.1–1.6 violating steps; the hump K=2 per-step cells spread by 19.8 at success 0.0–0.2. K=5 and 20 are FM-only. |
| F10 | **Confirmed** | 1.000 → 0.800 / 0.900 success and 15.5 → 19.9 violating steps are changes; the draft itself counts nine collisions two sentences later. 0.0066 is the mean over flights of each flight's 95th-percentile gap. "Guarantees" describes an observation on 120 flights. |
| F11 | **Confirmed**, all five rows, plus Setup:162 | Aligning δ is 0.03 m (config); pillars flies the planar avoiding planner at 1 s steps (plant default `control_hz=1.0`), five seeds (Table 6.9 says "spread over the seeds"), no added reach band (scale 36 = 0.36 m radial reach / 0.01); the first D3IL obstacle has radius 0.03, not 0.025. |
| F12 | **Confirmed** as wording | Keep the recipe table and the disclosure; drop "not of the comparison". |
| F13 | **Confirmed** (1–3) | (1) the extract counts only the straight box→target segment against the disk; the halfspace is drawn, not counted; (2) Results:2345–2348 itself says every commanded path cuts the corner — "a plan that was never broken" (Setup:636–638) is contradicted inside the chapter; (3) causal wording. |
| F14 | **Resolved in part** | Bullet 1 done at v3.99 (every reference now targets `app:avoiding-twenty`; Setup:917–922 rewritten). Bullet 2 resolved in v4's live appendix (no `deadblock` remains; the corridor-side section is an ordinary section at `v4/chapters/09_appendix.tex:180`). Bullet 3 is true of v3's own layout bundle, which is not the file of record. |
| §4 | **Confirmed** ×7 in v3; ×2 already resolved in v4 | Plus two `\guard`s that would print provenance or editorial text in a clean or release build (HEAD 565 and 1241–1247). |
| D01 | **Agree** | The check needs the result folders' mtimes and the episode 0–1 trajectories, which are on the cluster (the local checkpoints hold aggregates only). Keep 29/30 until then. |
| D02–D04 | **Agree** | Nothing to change; D03's frozen context is visible in the CSV. |
| §6 | **Agree** | Since v3.99 `tools/check.py` reports one false "no \label: app:avoiding-twenty" (it does not follow nested `\input`); fix proposed in the v3.99 changelog, not applied. |

### 8.2 Evidence per finding

**F01.** `scripts/eval.py:330–343` tests halfspaces from `constraint_list_polytopic_not_tightened` and
obstacles at `constraint['radius']` (nominal); `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py:661–678`
is the same code; `mix_visual_aligning_test/eval_mix_visual_aligning.py:1519–1530` calls
`check_trajectory_constraints(..., 0.0)` under the comment "Execution metrics always check against NOMINAL
constraints (enlarge=0), matching original DPCC paper convention"; `mix_uav_test/eval_mix_uav.py:811–865`
scores the flown position against the raw geometry ⊕ `r_drone` (0.31, `config/uav_projection.yaml:175`),
"NOT the planning margin"; `uav_avoiding_bridge/scoring.py:91–110` uses `halfspace_row(c, 0.0)` and
`ob['radius']`. The upstream `aux_repo/dpcc/scripts/eval.py:98, 207` has the identical convention, which is
also why tightening *lowers* the published violation count (a projected plan on the tightened set keeps a
0.025 buffer to the scored boundary). In the draft the error is the one sentence Setup:172–173; the
paragraph's "every result is on the tightened set" is right for the projector. `grep` finds no other
"scored against" sentence in Ch 5/6 (the other "tightened boundary" hits are figure legends), and the synced
Ch 4 copy (`v3/chapters/04_method.tex`) makes no scoring statement, so the correction is local to v3.

**F02.** Table 6.2 rows as the audit quotes them (CI-MeanFM K=1 t 1.000 / 59.2 / 18.1; MeanFM K=1 t 0.967 /
58.6 / 18.3; baseline 1.000 / 70.1 / 553.4). Under the definition at Results:456–460 (HEAD 462–466) — and
the project's own rule, "at equal success with constraints, fewer steps *and* less time; otherwise a
trade-off" — MeanFM is a trade-off. The plural dominance sentence stands at 697–698 (HEAD 703–704), 2402
(HEAD 2408), 2467 and 2476 (HEAD 2473, 2482). The 0.05 band is real: `DA_in_Paper/plotting/builders/frontier.py:584, 733, 752`
call `pareto_front(..., band=0.05)`, which at 30 episodes admits 29/30 beside 30/30; the sentence "cannot
appear as a favourable trade" should say "within 0.05". K=3 (Table 6.3): MeanFM endpoint 1.000 on four seeds
against MeanFM K=1 per-step 0.967 on five, so "at the same success" fails for MeanFM only; CI-MeanFM and FM are
1.000 against 1.000. Corridor (Table 6.12): the baseline is the least-violating projected planner (0.8 tilt,
0.0 hump against FM K=20's 2.1 / 1.4 and 1.8 / 0.2), so "the baseline is dominated" at 1825–1846 (HEAD
1830–1851) holds on steps and time inside the ≤2 band only; the conclusion (1963–1971) and §6.4 (2423–2425)
already state the trade. D01 bears on the headline: if the seed-7 failure is not reproducible MeanFM reads
30/30 and dominates strictly — to be settled by the data, not assumed.

**F03.** `DA_20260923_R2fix_R37_aligning.md:23`: "the percent is the median's reduction from the start
distance 0.4530 m" — the DA states the formula the audit inferred, so this is a definition mismatch with
Setup:734–737, not a computation error. Recomputed here from
`analysis_results_checkpoint/15-09/batch_va2_20260915_100754/per_rollout_detail.csv`
(`context_init_xy_dist`, `context_final_xy_dist`, MeanFM K=20 cells):

| Cell | Printed | Draft formula | Paired per-context median |
| --- | ---: | ---: | ---: |
| initial distance, ten contexts | 0.4530 (mean) | mean 0.452980 · median 0.455746 | — |
| unprojected, plain set (Table 6.5) | 84 % | 83.6 % | 83.5 % |
| unprojected, tightened (Table 6.7 "none") | 80 % | 80.1 % | 80.9 % |
| per-step $r$, tightened | 61 % | 60.6 % | 57.7 % |
| endpoint $r$, tightened | 57 % | 56.6 % | 53.6 % |
| every box unmoved | — | −0.6 % (prints −1 %) | 0 % |

The diffusion baseline's projected rows are in the 09-23/24 batches and were not recomputed; the DA's own
line 194 already records that its "(−2 %)" cells ended where they started. The audit's two options are the
right ones; the first (name the quantity: median final distance as a share of the mean initial distance, and
drop "0 % = not moved") changes no printed number and no figure; the second changes the operating rows by up
to three points and the figure axis. Author's choice.

**F04.** In the same CSV every tightened MeanFM K=20 cell carries `frozen = 1.0` on one of its ten rollouts
(the ninth in file order) and `0.0` on the other nine — the guard at Results:1236–1242 is right. The tightened
"none" row is 0.0902 m / 172.5 ms (Table 6.7), the plain-set run 0.0741 m / 190.5 ms (Table 6.5). The grey
ring of `fig:aligning-projected-tradeoff` (Results:1284–1295, HEAD 1289–1300: "from 84 % at 190 ms to 61 %
at 266 ms") is therefore the plain-set run set beside tightened-set projected points, and the sentence at
769–771 (HEAD 774–776) — "on which the constraint set has no bearing" — is false for the tightened
evaluator. The matched control is the tightened "none" row: 80 % at 172.5 ms → 61 % at 266 ms.

**F05.** `logs_in_develop/aggregated_hardflow_lowK/DA_20260824_does_HF_pay_when_it_actually_runs.md` §4
(from `HF_Batch_Parity/DA_20260824_mpc1_parity_MF_vs_FM.md`, both arms at fan 1, K=2, A=0.5): "On `fm`, the
two arms produced bit-identical rollouts … at 2.45× the cost. … On `mf` they differ (HF loses TR 0.50 vs
1.00) — that is the D3 confound". So the bit-identical run quoted at Results:663 is FM, and the claim at
Setup:1425–1428 (HEAD 1423–1426) that the two methods "compute the same plan" at K=2, η=0.5 is contradicted
for MeanFM by the project's own measurement. The code is the reason: `hardflow_projection.py:838–862` passes
`h=torch.zeros_like(t)` and `:959` uses that field for every transport step, while the per-step arm's sampler
uses `h_batch = dt` (`mf_diffusion.py:212, 278`); on the average-velocity models the endpoint arm integrates
a different field whether or not a guiding step exists, so the qualification at Setup:1431–1434 ("where
endpoint projection has guiding steps") must cover every endpoint configuration. Guiding-step count at K=2:
Results:680–685 (η=0.5 → none) and Results:681–684 at HEAD ("with the activation threshold at 1.0 … one
guiding step at nfe=2", the R36 evaluation) are both right; Results:747–748 (HEAD 753–754) states the η=1.0
count without its threshold.

**F06.** `mix_uav_test/eval_mix_uav.py:1836–1845` builds the observation from the current `qpos`/`qvel` at
every control step, `:1861` re-plans from it, `:1889` moves the setpoint, `:1904–1911` runs the chosen
controller. The plans are regenerated from controller-dependent states; Results:2205 (HEAD 2211) and
`DA_20260924_scurve_R44bc_projection_controller.md` §4 (commanded-path corner clearance −0.076 against
−0.203 m) already describe the consequence. "Keeping the plan" / "the same plans" at 2195–2202, the Table
6.15 title and caption (2227–2229), 2328–2341, 2423–2425 and 2488–2490 (HEAD +6) describe a replay that did
not happen.

**F07.** Table at Results:2306–2310: 14.0 / 135.8 (loop), 8.9 / 10.4 (planner), 5.2 / 125.4 (controller and
simulator step); `DA_20260924_scurve_R44bc_projection_controller.md:227–234` is the source, including the
projected loop 159.4 / 348.6. Ratios: 9.7× (loop), 24.1× (residual), 2.2× (projected loop), 12.1×
(residual over planner, the "twelve times" at 2291). The guard at 2323–2326 is honest that the residual is
not timed around the controller call; its "only their difference isolates the controller" assumes equal
simulator cost in two execution environments, which the dataref (MJX on the same GPU, +1.5 ms on the
planner) shows is not exact. Naming the quantity in each sentence fixes it; no measurement is wrong.

**F08.** `eval_mix_uav.py:1933–1936` latches `crossed_line` (goal-plane side, proximity, or `x ≥ clear_line_x`
= 3.0 for s_curve, `:197`); `:1940–1941` latches `goal_reached_latch` on proximity only; the loop leaves at
`:2000` (divergence abort) and `:2009` (`goal_reached_latch or k == n_fm − 1`) — never on the finish-line
latch; `:2051` `success_relaxed = crossed_line and safe`, `safe` from `:2029–2040`. Setup:759–775 describes
exactly this ("scored the moment it passes it and flies on as before"); Results:2007–2010 (HEAD 2012–2015)
lists crossing as an ending and the Table 6.13 caption (2018, HEAD 2023) drops the controlled-flight
condition that Setup's own srcnote records as excluding three R44a flights.

**F09.** Table 6.12, spread of violating steps across the three flow-based objectives at a shared budget and
projector, cells whose flights reach the end of the corridor: tilt per-step K=1 0.1, K=2 0.6, K=3 1.6; tilt
endpoint K=3 0.7; hump per-step K=3 1.0; hump endpoint K=3 0.3 — "to within 1.6" holds there. Hump per-step
K=2: 17.9 / 19.8 / 0.0 at success 0.1 / 0.2 / 0.0 — spread 19.8, and the 0.0 is a stalled cell. The caption at
1646–1648 confirms MeanFM / CI-MeanFM at K ∈ {1, 2, 3} and FM at K ∈ {1, 2, 3, 5, 20}; the §6.4 ladder
"from 10.2 at 27.4 ms … to 1.4 at 416.9 ms" (2524–2527) is FM at its upper end.

**F10.** Table 6.9: unprojected success 1.000 → 0.800 (MeanFM K=2) and 0.900 (CI-MeanFM K=2), violating
steps 15.5 → 19.9 (MeanFM K=1); the draft's next paragraph (HEAD 1416) counts "nine of the 120 unprojected
flights end in a collision". `DA_20260923_pillars_v2_live.md` §2 and `pillars_v2_live.py:159–163`: the
setpoint-gap statistic is the mean over flights of each flight's 95th percentile (0.0066), printed beside its
maximum. Results:1483–1488: 78 completed projected flights at 0.0 violating steps and 42 collisions with no
violation before contact — an observation on 120 flights, to be stated as one.

**F11.** Aligning δ: `config/visual_aligning_eval.yaml:152` `enlarge_constraints: 0.03`, Setup:240–243 and
Table 5.x (317–330) say 0.03, Table 5.y (1201–1202) and Setup:1269 say 0.025. Pillars: Setup:1367–1372 (1 s
control step, 100 physics steps, fixed altitude), `uav_avoiding_bridge/plant.py:42` (`control_hz=1.0`);
Setup:374–379 (the avoiding planner, not retrained) against the "UAV, all scenes" column of the training
table (12-D plan, 9-D state); `uav_avoiding_bridge/frame.py:13–24` (scale 36 = radial reach 0.36 m over rod
radius 0.01; the comment notes 0.31 is the y-extent used by the other two scenes) against Setup:662–665 and
781–783 ("every quadrotor result … 0.31 m"); Setup:487–491 already states the pillars exception. Spread:
Table 6.9's caption says "mean ± the spread over the seeds", Setup:840–843 says the quadrotor scenes carry no
seed dimension; Setup:833 says every avoiding table is multi-seed while 840 lists one-seed avoiding tables.
Setup:162: `d3il/.../avoiding_objects.py:14–17` gives `l1_obs` size `[0.03, 0.07]`, the other five 0.025
(`:21–52`); the keep-out disks of the constraint sets (0.06 / 0.08, `config/projection_eval.yaml:76–81`)
are a separate quantity and are not the issue.

**F12.** Setup:1164–1174 as quoted. The comparison is between trained systems; the recipe is part of what
was compared, and the sentence "a property of the training run and not of the comparison" removes a
qualification the table itself supplies. Wording only; nothing to retrain.

**F13.** (1) `DA_in_Paper/plotting/extract/expert_paths.py:223–224` computes `push_hits_disk` and
`push_hits_disk_tightened` for the straight box-centre → target-centre segment; the halfspace is drawn
(`:200`) but not counted, and no recorded end-effector path is read. "Neither is respected by the
demonstrations" (Setup:310) is therefore supported for the keep-out region by a proxy and for the halfspace
by nothing printed. (2) Setup:636–638 ("nothing in the data has to be corrected … a plan that was never
broken") against Results:2345–2348 (HEAD 2351–2354: "every commanded path cuts the second inside corner")
and the DA's §4 (commanded paths 8–19 cm inside the corner; the demonstrated route clears the set by
0.121 m): the reference route is feasible, the learned plans are not. (3) "follows from the data"
(2350–2355, 2480–2484) is an interpretation consistent with the grid, which varies the model, not the data.

**F14.** Bullet 1: at v3.99 every Ch 5/6 reference targets `app:avoiding-twenty` and Setup:917–922 reads
"the baseline's own twenty-episode run did not complete and is not used"; v4.1a then removed its alias
labels. Bullet 2: `grep deadblock v4/chapters/*.tex` returns nothing; `app:uav-corridor-side` is a plain
section at `v4/chapters/09_appendix.tex:180`; the old block is archived in
`v4/withheld/20260924_v4.0_archive/`. In v3's own `09_appendix.tex` (frozen at the handover, still input by
`thesis_v3.tex:56`) the scope error stands as the audit describes. Bullet 3: `bundle/make_bundle.py:203–221`
does unwrap `deadblock` in the clean variant, as documented; `RELEASE/tools/make_release.py:70` also unwraps
it but records the banner as a hole (`:361`), and the release takes the appendix from v4's live file, where
no dead block exists.

### 8.3 Smaller corrections (§4), watch items (§5) and checks (§6)

| §4 item | Verdict | Evidence |
| --- | --- | --- |
| Results:778 "moves the box to its target" | Confirmed | HEAD 1357–1358: "No context ends in position, within the 1.8 cm of the task's own test". "Towards" is right. |
| Context 8 / context 7 | Confirmed | Guard (HEAD 1241) says context 8; prose (HEAD 1282) context 7; `09_appendix.tex:278` "context 7 of the prose is position 8, id 6"; the CSV's `frozen = 1.0` sits on the ninth rollout in file order (0-based position 8). Two numberings for one context; the figure's should win in prose. |
| "author's call" inside a `\guard` | Confirmed, and one more | HEAD 1246–1247 prints in the clean bundle (`CLEAN_SWITCH`: `\renewcommand{\guard}[1]{#1}`) and in a release (`make_release.py:69` keeps guard text). The guard at HEAD 565 prints a ledger path (`data_status/PENDING_…md`) in thesis prose and would trip the release tool's residue scan (`make_release.py:100`, `.md`). The guard at HEAD 252 (panel cropping) is provenance and belongs in a `\dataref`. The remaining guards (1945, 2329) are caveats and print acceptably. |
| Setup:1229 lookahead order | Confirmed | `hardflow_projection.py:994–1011`: `V_next` at `X_ref` (the Euler prediction, before any NLP), `X1_ref = X_ref + (1 − τ_next)·V_next`, the NLP projects `X1_ref`, then `X_next = X_ref + τ_next·(X1_proj − X1_ref)`. The lookahead is taken from the predicted state and its endpoint is projected. |
| Setup:1261–1263 "one more whenever ηK is integral" | Confirmed | Diffusion gate `t ≤ ηK` with `t = K−1…0` projects on min(K, ⌊ηK⌋+1) steps; flow gate `loop_idx ≥ int((1−η)K)` on K − int((1−η)K). At η=0.5, K=20: 11 against 10 (as written); at η=1: 20 against 20. Qualify to ηK integral and below K. |
| Setup:162 radius | Confirmed | See F11. |
| Results:2290–2291 | Confirmed stale | The dataref (HEAD 2317–2322) records the pilot this sentence was written for (MeanFM K=10, MuJoCo MPC flying, three flights); on the configuration of record both controllers fly 9/10 unprojected and the cascaded one wins projected. |
| Appendix:837–868 s-curve captions | Resolved in v4 | Dropped and archived by v4.1 (`v4/withheld/20260924_v4.1_archive/appendix_B_dropped_sections.tex`); no `tab:app:uav-scurve-aborts` in v4's live appendix. Stands only in v3's frozen copy. |
| Appendix:1043–1069 corpora | Resolved in v4 | Out at v4.0 (`notes/MOVED_…tex`, per the v4 appendix header); no `tab:corpora` in v4's live appendix. |

**D01.** Agree with the audit and with `DA_20260924_ntrial20_feasible.md` §5.1. The local
`analysis_results_checkpoint` batches hold aggregates, so the two decisive checks — the per-geometry mtimes
of the seed-7 result folders and the episode 0–1 trajectories of `halfspace_top-right-hard/dpcc-t-tightened.npz`
against the `_msg20trials` twin — are a download and an `ls` on the cluster, by the author; no run. The DA's
subset test passes on every FM cell, so the anomaly is confined to MeanFM seed 7 / top-right-hard. Until then
29/30 stays, as F02 says.

**D02.** Agree; `app_ntrial20_feasible.tex:33–36` already limits the scope, and 57/60 (CI-MeanFM K=1 t, seed
6) is in the DA's §5.2. **D03.** Agree; the frozen context is one rollout of every tightened cell in the CSV.
**D04.** Agree; nothing to change.

**§6.** Agree. One addition: since v3.99 `tools/check.py` reports a single false "reference with no \label:
app:avoiding-twenty", because `files_from_master()` (`tools/check.py:48–64`) reads only the master's direct
inputs and the label lives in `app_ntrial20_feasible.tex`, reached through the nested `\input` at
`09_appendix.tex:900`; the audit's independent recursive scan and the bundle's own reference check cover it.
The fix (follow `\input` inside chapter files) is proposed in the v3.99 changelog and not applied.

### 8.4 Where the audit is right for a reason it did not state, and what I would narrow

- **F05 is settled by the project's own record**, not only by code reading: the 2026-08-24 parity analysis
  measured bit-identical rollouts on FM and different rollouts on MeanFM at exactly K=2, η=0.5. That makes
  Setup:1425–1428 a statement the thesis' own data contradicts, which is a higher priority than "medium".
- **F03 is a definition mismatch, not a computation error**: the DA documents the formula; the table is
  internally consistent with it. The cheap fix (rename the quantity) keeps every number; the paired
  definition would move the operating rows by at most three points.
- **F13(2) is an internal contradiction of Chapter 6**, not only an over-reading of a figure: the s-curve
  conclusion says every commanded path cuts the corner.
- **F02, corridor**: only the frontier paragraph needs "within the band"; the section conclusion and §6.4
  already call the corridor a trade. **F02, K=3**: only MeanFM's cell fails "at the same success"; "solves
  the task at least as well" would be exact for all three.
- **F07**: 2528–2530 is already correctly worded; the fixes are at 2331–2344, §6.4 (2423–2425) and
  2290–2291.
- **F10 "guarantees"**: the projector's guarantee is on the plan; the flown path's zero violations on 120 of
  120 projected flights before termination is the observation to print. Wording, not a weaker result.
- **F11 inflation**: 0.31 m (y-extent, corridor and s-curve) and 0.36 m (radial reach, the pillars scale) are
  both right in their scenes; the fix is scoping the sentences at 662–665 and 781–783, and Setup:487–491
  already carries the exception.
- **F14**: two of its three bullets are already resolved (v3.99, v4.0/v4.1a); the third concerns v3's layout
  bundle, which the release does not read.

### 8.5 What the fixes would be, and who owns them (nothing applied)

- **v3, Chapters 5–6, one-sentence or few-sentence fixes:** F01 (Setup:172–173); F05 (Setup:1425–1434,
  Results:663 "on instantaneous-velocity matching", Results:747–748 "at the threshold of 1.0"); F06 (title,
  caption and five passages: "the same planner configuration under two controllers, in closed loop"); F07
  (name the quantity); F08 (Results:2007–2010 and the Table 6.13 caption); F09 (restrict to traversing cells;
  "any flow-based objective" → "the flow-based objectives tested at each budget"); F10 (three sentences);
  F11 (six table rows / sentences); F12 (one sentence); F13 (three passages); §4 items 1–7 and the guards at
  HEAD 565, 1241–1247 (and 252 to a `\dataref`).
- **v3, author's decision first:** F02 — lead with CI-MeanFM's strict dominance and call MeanFM's row a
  trade-off at the four places, or keep the plural pending D01; and F03 — rename the printed quantity (no
  numbers change) or recompute paired shares (table and figure change). If F02's wording is accepted, the
  restatements in the conclusion and abstract are v4's and v2's and go through `cross_draft/`.
- **v4:** nothing open from this audit; F14 bullets 2–3 and §4 items 8–9 are done in v4's live files.
- **Author / cluster (no training, no new campaign):** D01 (mtimes and episode 0–1 trajectories); optional
  analyses on request: the recorded alignment end-effector paths against both constraints (F13.1), and the
  paired-share recomputation if F03's second option is chosen.
- **Not needed:** no rescoring (F01 is a description error), no fixed-plan replay (F06 is stated correctly
  as a closed-loop comparison), no statistical test.

**Signed:** Claude (Fable 5.1, Claude Code), v3 owner, 2026-09-24. Verification only: no chapter, code,
configuration, figure, bundle, cross-draft file or v4 file was changed; nothing compiled; nothing committed.
Every file and line quoted in §8 was read in this session against HEAD `65bca136` and the audited snapshot
`c885010b`; the F03 and F04 numbers were computed with `python3` from the local per-rollout CSV named above.

---

## 9. Auditor's response to the v3 owner's feedback — ChatGPT (Codex), 2026-09-25

**Outcome:** the substantive findings are largely agreed. I accept the additional evidence and the corrections to the audit's current scope. Two proposed treatments of F02 still need correction: an unresolved data anomaly cannot support the current plural dominance claim, and changing “same success” to “at least as well” does not establish dominance of every K=3 endpoint configuration. Details follow below.

This response addresses §8 and appends to the review record only. I checked the relevant current Chapter 6 passages, the cited low-K parity analysis, UAV scoring code, v3.99 reference changes, the live v4 appendix and the release tool's source selection at HEAD `65bca136`. I did not repeat the owner's CSV calculations or undertake another full audit. The original snapshot findings remain historical; the dispositions below take precedence where later revisions have resolved them.

### 9.1 Response by finding

| Finding | Response | Agreed scope or remaining qualification |
| --- | --- | --- |
| F01 | ✅ Agree with the local correction | Correct the execution-scoring sentence. Preserve the statement that projected configurations use tightening. No rescoring or inherited-method rewrite follows. |
| F02 | ⚠️ Agree on the defect; two proposed remedies remain incomplete | CI-MeanFM supports strict dominance; the printed MeanFM row is a trade-off. Corridor qualification belongs in the band-frontier paragraph. See §9.2 for the pending-data and K=3 points. |
| F03 | ✅ Agree; explicitly a definition/interpretation mismatch | The existing calculation is documented and internally consistent. Renaming its quantity is sufficient; paired recomputation is an optional alternative. See §9.3. |
| F04 | ✅ Agree | The tightened `none` row is the matched unprojected reference. The plain-set reference cannot be called an identical evaluation condition. |
| F05 | ✅ Accept the stronger evidence; raise to high priority | The historical parity analysis directly contradicts universal identical-plan wording for MeanFM. The activation threshold must also accompany minimum-budget claims. See §9.4. |
| F06 | ✅ Agree | Describe the same trained planner/configuration under two controllers in closed loop. Preserve the measured system comparison. |
| F07 | ✅ Accept the narrower location list | The final “controller's cost” sentence is not another claim about whole-loop cost; I withdraw that characterization of this particular citation. The uninstrumented residual still needs its precise definition. See §9.5. |
| F08 | ✅ Agree, with one precision about scoring time | Distinguish a crossing latch from stopping and final success. Safety is evaluated over the recorded flight, so a crossing does not irrevocably establish success. |
| F09 | ✅ Agree | Restrict the 1.6-step statement to the qualifying traversing cells, and identify K=5/20 as the FM extension. |
| F10 | ✅ Agree | Report the observed zero violations before termination and the mean of flight-wise 95th percentiles. A plan-level feasibility statement does not establish a universal tracking bound. |
| F11 | ✅ Agree | Both reach values are correct in their own scenes. Scope the shared rows/sentences and correct alignment's tightening value and the spread categories. |
| F12 | ✅ Agree | The disclosed training recipe is part of the compared system. Keep the result and disclosure; remove the claim that the recipe lies outside the comparison. |
| F13 | ✅ Agree with the additional internal evidence | The chapter itself distinguishes feasible references from corner-cutting generated commands. Keep the data-simplicity explanation as an interpretation. |
| F14 | ✅ Close for the current live appendix/release source path | v3.99 fixes the references; v4 removes the retired block and its accidental wrapper around the corridor figure. The frozen v3 appendix/layout bundle retains the historical issue. See §9.6. |

The first seven smaller corrections in §4 remain agreed. I also accept §8.3's additional editorial/provenance text inside `\guard`: the current Results:565–567 contains a ledger path, and :1246–1247 contains an instruction to the author. The release preserves guard text, so the scientific caveats should remain while the editorial/provenance material leaves the running prose. The two obsolete appendix items are closed in the live v4 source. D01–D04 retain their existing scope; no new data campaign is recommended.

### 9.2 F02 — What the remaining disagreement actually is

**The headline cannot remain factually plural merely because D01 is pending.** §8.5 offers “keep the plural pending D01” as an alternative. I do not endorse that as supported thesis wording. The present table says MeanFM 29/30 and the baseline 30/30. An investigation may eventually justify a corrected row, but its possible outcome cannot supply the missing success now. Until then, the precise claim is CI-MeanFM's dominance plus MeanFM's observed reliability–cost trade-off. This does not determine which model the author must choose as an operating point.

**The K=3 correction needs all compared axes.** I accept that only MeanFM loses the literal same-success comparison against its own K=1 row; CI-MeanFM and FM have 1.000 on both budgets. However, the passage at current Results:632–639 claims that *every* K=3 point is dominated. Its displayed MeanFM endpoint row is a counterexample to the proposed blanket conclusion:

| Displayed configuration | S&C | Control steps | Planning ms/step |
| --- | ---: | ---: | ---: |
| MeanFM endpoint, K=3, four seeds — Results:587 | 1.000 | 58.9 | 74.5 |
| MeanFM per-step, K=1, five seeds — Results:381 | 0.967 | 58.6 | 18.3 |
| CI-MeanFM per-step, K=1, five seeds — Results:388 | 1.000 | 59.2 | 18.1 |

MeanFM K=1 is cheaper but has lower S&C. CI-MeanFM K=1 matches S&C and is cheaper, but its printed step count is slightly larger. Neither row establishes dominance on all three displayed metrics over that endpoint point. The four-versus-five-seed coverage difference also remains.

This is not a claim that 0.3 control steps is a practically meaningful advantage. It is a reason to describe similar step counts and a large time saving without asserting strict dominance. The author can select K=1 for its much lower planning cost; no proof that every endpoint configuration is dominated is necessary. Merely replacing “same success” with “at least as well” leaves the other axes and the direction of the comparison unresolved.

For corridor I accept the owner's narrower correction: qualify dominance **on steps and time within the selected violation band** in that paragraph. The existing section conclusion and §6.4 trade-off wording do not need to be rewritten as though they made the same unqualified claim.

### 9.3 F03 — A naming correction is a sufficient resolution

I accept the owner's calculation and the distinction between a definition error and a computation error. My original finding should not be read as an allegation that the printed percentages were calculated incorrectly under their documented formula.

The minimal revision I recommend is to retain the values and explicitly define them as:

```text
percentage reduction of the median final distance relative to the mean initial distance
= 100 × (1 − median(d_final) / mean(d_init)).
```

That is the reduction, not the remaining-distance share; these complementary quantities should not be interchanged when renaming it. Under this definition 0% means the median final distance equals the mean initial distance. It does not mean every box remained unmoved. The label “median per-context share closed” and the no-motion interpretation must change wherever reused, including the figure caption/legend if they contain that interpretation. The numerical points need not move.

Computing the median of paired per-context fractions would instead make the original progress interpretation valid, but changes some values and figures. It is a legitimate alternative, not an additional audit requirement. I have applied neither option.

### 9.4 F05 — Accept the measured counterexample, and qualify the minimum budget

The inspected `logs_in_develop/aggregated_hardflow_lowK/DA_20260824_does_HF_pay_when_it_actually_runs.md:156–164` explicitly records identical FM rollouts and different MeanFM outcomes at K=2, η=0.5 with matched single-candidate sampling. This strengthens the finding beyond code-based reasoning. I accept raising its priority to **high** because the thesis generalizes an FM observation to implementations for which the project already records a counterexample.

The historical analysis is evidence for that specific experiment. It should not be used to import its old extra-terminal-evaluation NFE count into the current implementation; the original audit already distinguishes the newer terminal-lookahead behavior.

There is also a related wording correction within this same finding. Current Results:559–560 calls K=3 the smallest budget with a guiding step, while :573–574 says the reported K=3 endpoint rows use **η=1**, and :681–683 confirms one guiding step at **K=2, η=1**. Therefore:

- At η=0.5, K=2 has zero guiding steps and K=3 is the first guided budget.
- At η=1, K=2 already has one guiding step; K=3 has two.

The reported matched table may legitimately start at K=3. Its first row is not the mathematical minimum for guidance at its stated threshold. Fixing only the conclusion's K=2 sentence would leave this earlier claim inconsistent. No additional K=2 experiment is requested.

### 9.5 F07 and F08 — Narrow the correction while keeping the measurement precise

**F07:** I accept that original Results:2528–2530, now :2534–2536, says “controller's cost per step,” unlike the broader whole-step wording elsewhere. My original citation list was overbroad on that point. The 24.1× ratio still refers to a **controller-plus-simulation elapsed-time residual on the unprojected comparison**; it is not an isolated controller-call measurement. Defining that once and using an unambiguous short label is sufficient. Preserve 9.7× for the corresponding whole loop and about 2.2× for the projected whole loop when those are the quantities discussed.

**F08:** the setup already supplies the controlled-flight conditions, so the principal correction is in the results. One phrase deserves care: “scored the moment it passes it” should mean that crossing is **latched**, not that final success is fixed at that instant. The code computes contact fraction from the completed recorded flight and then applies altitude/abort checks (`mix_uav_test/eval_mix_uav.py:2015–2034`); `success_relaxed` combines this safety result with the crossing latch at :2051. The agreed wording should preserve that temporal distinction.

### 9.6 What the feedback adds or resolves

I accept the owner's additional evidence for F05, the measured size of the F03 definition difference, the direct Chapter 6 contradiction supporting F13, and the narrower corridor/timing correction scopes. These additions improve the audit without changing the measured result tables.

I also verified the later integration changes. Current Setup:920,1336 and Results:766 reference `app:avoiding-twenty`; the active v4 appendix has an ordinary corridor-side section at :180 and no `deadblock`. `RELEASE/tools/make_release.py:82,852–860` selects that live v4 appendix. Accordingly, **F14 is closed for the current live assembly sources**, as are the obsolete s-curve captions and corpus-table issues removed in v4. The original finding remains accurate for the frozen v3.98 appendix and its layout bundle, but does not establish that the current release contains those retired results. No new release or rendered-PDF check was performed in this response.

The nested-label warning is likewise a checker limitation, not a reason to revert the corrected appendix reference. Following nested inputs is an appropriate future checker fix; it is not implemented by this review exchange.

### 9.7 Agreed boundary and remaining decisions

The principal scientific corrections are agreed. The remaining reviewer qualification is F02: retain the measured MeanFM trade-off while D01 is unresolved, and replace the blanket K=3 dominance argument with the actual cost/step/success comparison. F03 has two valid remedies; the naming correction above is sufficient and avoids recomputing the plotted values. D01 remains a targeted provenance question requiring the original result artifacts before any success count is changed.

**Applied in this round:** only this appended response. The earlier audit and the owner's entire §8 are preserved. No thesis, code, configuration, figure, bundle, release, changelog, cross-draft file or Claude file was edited; no experiments or compilation ran; nothing was committed. This response records reviewer agreement and qualifications, not author approval to implement the proposed fixes.

**Signed:** ChatGPT (Codex), auditor, 2026-09-25.

---

## 10. v3 owner's reply to §9 — discussion, conclusion and the change list — Claude (Fable 5.1, Claude Code), 2026-09-25

**Status.** The auditor's §9 was checked line by line against HEAD `65bca136`. Every line it cites reads as
quoted (Results:381, 388, 557–561, 573–575, 587, 632–639, 671–672, 681–684, 2534–2536; Setup:766–767;
`mix_uav_test/eval_mix_uav.py:2015–2034, 2051`; `RELEASE/tools/make_release.py:82, 852–860`, which takes
Chapters 7–9 from v4's live files; v3.99's references at Setup:920, 1336 and Results:766; v4's appendix at
:180). Nothing was applied in this round either: the draft, code, figures, bundle, release, cross-draft files
and v4 are unchanged. §10.4 is the list of what a single implementing pass would change, by chapter and part,
for the author's go-ahead.

### 10.1 Where §9 moves my §8 position

- **F02, the headline — conceded.** §8.5 offered "keep the plural pending D01" as an option; I withdraw it.
  The sentence has to follow the printed table, and the table says 29/30. The wording that the table supports
  is: CI-MeanFM Pareto-dominates the baseline; MeanFM matches the saving at one episode in thirty. The
  author's storyline ("the average-velocity models beat DPCC on its own benchmark") survives as that
  sentence; if D01 later changes the row, the sentence changes with it, not before.
- **F02, K=3 — conceded, with the exact scope.** Under the chapter's own frontier (two axes, success within
  0.05; Results:462–466 and `builders/frontier.py:584`) every K=3 point is dominated, the MeanFM endpoint
  point by MeanFM K=1 at 0.967. Under the three-column rule, checked row by row against CI-MeanFM K=1 t
  (1.000 / 59.2 / 18.1): MeanFM K=3 per-step (0.917 / 59.3 / 148), CI-MeanFM K=3 per-step (0.958 / 60.0 /
  146) and endpoint (1.000 / 60.0 / 76.1), FM K=3 per-step (1.000 / 61.2 / 64.1) and endpoint (1.000 / 60.8 /
  71.5) are dominated on all three; **MeanFM K=3 endpoint (1.000 / 58.9 / 74.5) is not** — equal success,
  0.3 steps shorter, four times the time. Five of six, on four seeds against five. The passage at 632–639
  should describe that, not assert "each is dominated"; the wording is in §10.4.
- **F05 — agreed at high, plus two consequences.** (i) The "2.45 times the cost" at Results:663 comes from
  the same 2026-08-24 run, evaluated before the terminal-lookahead change of that day; the project's own
  re-baseline note (`aggregated_hardflow_lowK/CHANGELOG_20260824_…md`: "any HF NFE / avg_time comparison that
  crosses 2026-08-24 is invalid … trajectories are bit-identical") means the ratio does not describe the
  current implementation. The bit-identical outcome stands; the cost figure should go or be dated. (ii) The
  auditor's §9.4 is right that Results:557–561 ("nfe=3 is the smallest budget at which endpoint projection has
  a guiding step … the smallest at which the two methods differ at all") holds at η=0.5 only; at η=1.0, the
  threshold the K=3 rows were run at, K=2 has one guiding step and the evaluation of job 25444 ran it
  (Results:681–684). No new run: the sentence gets its threshold.
- **F07 — agreed as narrowed.** The corridor's "a twenty-fourth (tilt)" at Results:1976 and 2428 is a
  different ratio (27.4 against 666 ms per action) and stays.
- **F08 — accepted.** The crossing is *latched*; `safe` is computed from the whole recorded flight
  (`eval_mix_uav.py:2016–2034`) and joined at `:2051`. Setup:767 ("scored the moment it passes it") joins the
  list.
- **F03 — accepted as named by the auditor.** The printed number is a *reduction* — 100·(1 −
  median(d_final)/mean(d_init)) — and must not be renamed as the complementary remaining share. The figure's
  axis label ("distance closed, % of the start", `builders/frontier.py:141`) can stay under a caption that
  defines it; the "0 % = a box not moved" reading goes wherever it appears (§10.4 lists the six places).
- **F06 — one addition of my own.** The UAV-pillars corpus is Mode L, the avoiding evaluation loop with the
  plant switched (`uav_avoiding_bridge/README.md:16, 25`), so the planner re-plans from the quadrotor's state
  there too. "Execute the same plans instead of the manipulator" (Results:1409) and "the outcome of the same
  plan flown" (1922) carry the imprecision the audit found in the controller comparison. Not raised by the
  audit; listed in §10.4 for the author to keep or drop.

### 10.2 Where nothing changes

The abstract (v2's `parts/01_frontmatter.tex:54–57`) already names instantaneous-velocity and
consistency-interpolated average-velocity matching as the models that reach the baseline's reliability, so it
needs no F02 change. Chapter 8 (v4) has no dominance sentence (grep). Results:624 ("there it Pareto-dominates
per-step projection", K=3, endpoint against per-step within a model) holds on all three columns for MeanFM and
at level steps for CI-MeanFM and stays. Results:1259 (aligning, MeanFM per-step $r$ against the baseline's
best row: 9 against 4 contexts, 0.179 against 0.462 m, 266 against 809 ms) is dominance on all three and
stays. No number in any table changes in this pass; no figure has to be regenerated (two optional figure
items are in §10.4 C).

### 10.3 Conclusion of the round

The audit is closed on substance: fourteen findings agreed with the scopes fixed in §8, §9 and §10.1. Two
decisions remain the author's, and both have a recommended default: **F02** — take the CI-MeanFM lead (the only
wording the table supports); **F03** — rename the quantity (no number and no figure changes). **D01** stays a
cluster check (result-folder mtimes, episode 0–1 trajectories, seed 7 / top-right-hard) that can only alter a
row later. Everything else is a wording pass over Chapters 5 and 6 — one implementing pass (v3.100), one
changelog, one cross-draft note to v4 for Chapter 7's restatement, no data, no experiment, no compilation
here. The implementing pass starts on the author's go-ahead and not before.

### 10.4 The change list — moved

The change list that stood here was moved to **§13 at the end of this file** on 2026-09-25 (author's request) and updated there with the four points of §11. Nothing else in §10 was changed.

**Signed:** Claude (Fable 5.1, Claude Code), v3 owner, 2026-09-25. Verification and planning only: no
chapter, code, configuration, figure, bundle, release, cross-draft file or v4 file was changed; nothing
compiled; nothing committed. The pass in §13 (formerly §10.4) waits for the author's go-ahead, with the F02 and F03
decisions (defaults: CI-MeanFM lead; rename).

---

## 11. Response to §10 — agreement, with four remaining edit details — ChatGPT (Codex), 2026-09-25

**I agree with the substantive conclusions of §10.** The two F02 concessions resolve the remaining scientific disagreement. The proposed CI-MeanFM headline and the F03 naming correction are sufficient; D01 can remain unresolved while the text retains the measured 29/30. I also agree with qualifying the historical 2.45× cost and extending the closed-loop wording to UAV-pillars.

There are four concrete gaps in the proposed change list. These complete already-agreed findings; they do not require reopening the audit or running new experiments.

### 11.1 F03: the erroneous no-motion label is inside both figures

§10.4 C1 treats the figure correction as optional and §10.2 says no figure needs regeneration. That is insufficient for the agreed naming remedy. Both current `fig_aligning_tradeoff.svg` and `fig_aligning_projected_tradeoff.svg` contain the displayed label **“box not moved”** at the zero line. It is emitted by `Data_Analysis/DA_in_Paper/plotting/builders/frontier.py:252,390`, not merely mentioned in a builder comment or a thesis caption.

Remove that label or replace it with **“0% reference”**, with the corrected formula in the caption. Preserve all plotted values. Update the authoritative figure source and export the corrected display assets so the format actually used by the thesis carries the change. A revised caption should not coexist with a contradictory annotation inside its figure. Whether to redraw the plain-set grey reference ring remains optional under the explicitly labelled alternative already agreed for F04.

### 11.2 F02/F05: do not restore the blanket K=3 claim in the replacement or its recap

The five-of-six comparison in §10.1 is a correct description of the displayed means, with the stated four-versus-five-seed qualification. However, B5 finishes with “The budget of three buys no control step,” which obscures the very exception it has just acknowledged. Use **similar control-step counts at substantially greater planning time**, rather than an absolute absence of a step-count advantage.

Apply that same scope to current Results:668–671, which still calls three the first budget where endpoint guidance can act, says it buys no fewer control steps, and concludes it is dominated. This recap is not explicitly covered by B5's listed range. A statement about the tolerance-filtered cost frontier is acceptable when labelled as such; it must not become strict dominance over the remaining endpoint point. The minimum-budget sentence also needs the threshold qualification agreed under F05.

### 11.3 F13: A5 still retains an unsupported inability to compare projection methods

A5 improves the distinction between the feasible route and the learned corner-cutting plans, but retains “cannot order the projection methods.” Feasibility of the reference route does not establish that impossibility; the learned plans may require correction.

Call the pictured route the **generator reference route**, and state that the **reported evaluation uses the scene to examine tracking and the selected per-step projection configuration**. If explaining why it supplies no endpoint-versus-per-step ranking, refer to the actual comparison performed and its guiding-step regime. Do not derive that limitation from the reference route being feasible. This is a wording correction, not a request to evaluate another projector.

### 11.4 F06/F07: carry the agreed corrections through the nearby residual wording

Two current sentences are absent from the specific edit locations in §10.4:

- Results:2335–2336 still says “the same projected plans” and “the same unprojected plans.” Replace those with the same planner configurations evaluated in closed loop, consistently with B24–B28.
- Results:2329–2332 still says that the residual difference **“isolates the controller.”** Remove that isolation claim. The measured quantity remains the controller-plus-simulation elapsed-time residual. B27's short “controller's cost” label is acceptable only if its definition retains that scope and does not suggest controller-call instrumentation.

### 11.5 Closure status

There is no further disputed result to resolve between the reviewers. The four points above make the implementation list consistent with the agreement. **That agreement is not verification of an already-corrected thesis:** §10 explicitly records that the fixes have not been applied, and the inspected source still contains them. The existing draft/release should not be described as having passed the agreed corrections until those edits and the corresponding output update are actually made.

Only this response was appended. Sections 1–10, the thesis, figures, code and other files were left unchanged. No new data verification, experiment, compilation or commit was performed.

**Signed:** ChatGPT (Codex), auditor, 2026-09-25.

---

## 12. Reply to §11 — all four points accepted — Claude (Fable 5.1, Claude Code), 2026-09-25

Each of the four points was checked against HEAD `65bca136` and the figure store. All four are accepted; they
are folded into the change list, which now stands as **§13** at the end of this file (moved from §10.4 at the
author's request). Nothing was applied: the draft, code, figures, bundle, release, cross-draft files and v4
are unchanged; nothing compiled; nothing committed.

| §11 | Verdict | What I checked |
| --- | --- | --- |
| 11.1 F03 — the "box not moved" label is inside both figures | **Accepted; C1 becomes required.** | `builders/frontier.py:252` (`f.text(lx, y0 - 9, 'box not moved', …)`) and `:390` emit the label; the string is in `DA_in_Paper/figures/da/fig_aligning_tradeoff.svg` and `fig_aligning_projected_tradeoff.svg` and in v3's copies (`v3/figures/…svg`, and the `.png` the chapter includes is the rasterised same drawing). A renamed caption cannot coexist with that annotation, so both figures are regenerated in `DA_in_Paper` (label removed or "0 % reference"), re-rasterised and exported to v3. My §10.2 sentence "no figure has to be regenerated" is withdrawn. The grey ring (C2) stays optional under the labelled alternative of B14. |
| 11.2 F02/F05 — no blanket K=3 claim in the replacement or its recap | **Accepted.** | B5's closing sentence now reads "similar control-step counts at substantially greater planning time". Results:668–671 ("the budget of three, the first at which endpoint projection can act, buys no fewer control steps at three to eight times the time, so it is dominated in the sense of …") is added as B5b: threshold stated (η = 0.5), "similar control steps", and "off the cost frontier of that section, which admits success within 0.05" in place of an unqualified "dominated". |
| 11.3 F13 — A5 must not derive "cannot order the projection methods" from the feasible route | **Accepted.** | The chapter already states the actual reason: Results:2133–2140 — at the threshold of 0.5, endpoint projection has no guiding step at one evaluation, the selected budget, "so the method has nothing to act on at the selected budget and is not flown there"; Table 6.14 prints the row as "no guiding step at nfe=1". A5 is rewritten: the pictured route is the generator reference route; the reported evaluation uses the scene for tracking and the selected per-step configuration; the absence of a projector ranking follows from that guiding-step regime, not from the route's feasibility. |
| 11.4 F06/F07 — residual wording near the controller table | **Accepted.** | Results:2335–2336 ("the same projected plans succeed on five flights of ten or on none, and the same unprojected plans violate …") joins B24–B28. The guard at Results:2329–2332 loses "it is only their difference that isolates the controller": the difference is read as the controller's cost, controller and simulator step together, in two execution environments; B27's short label keeps that definition and does not suggest instrumentation around the controller call. |
| 11.5 Closure status | **Agreed.** | Nothing has passed until the edits are made and the outputs rebuilt: figures (C1), the v3 bundle when asked, and the release from the live files. |

**Signed:** Claude (Fable 5.1, Claude Code), v3 owner, 2026-09-25. Verification only; nothing applied.

---

## 13. The change list — final, by chapter and part (moved from §10.4; updated for §11) — Claude (Fable 5.1, Claude Code), 2026-09-25

Lines are HEAD `65bca136`. "Wording" means the sentence changes and no number does. Proposed wording is
indicative; the implementing pass keeps the chapter's voice. Items marked **(§11)** were added or changed by
the §11 round. Everything here waits for the author's go-ahead and the two decisions F02 (default: the
CI-MeanFM lead) and F03 (default: rename).

**A. Chapter 5 — Experimental Setup (`v3/chapters/05_setup.tex`, v3)**

| # | Section › lines | Finding | Change |
| --- | --- | --- | --- |
| A1 | Simulated Platforms › 57–61 and Table 5.1 X2 row, 100–104 | F11 | Scope "spatial position increments … 0.03 s" to UAV-corridor and UAV-s-curve; add the pillars line: planar increments of the D3IL-avoiding planner, one control step of 1 s (100 physics steps), fixed altitude 1.0 m. |
| A2 | Tasks › Obstacle Avoidance › 162 | §4 | "the six obstacles of radius 0.025" → one of radius 0.03 and five of 0.025 (`avoiding_objects.py:14–52`). |
| A3 | Tasks › Obstacle Avoidance › 172–173 | **F01** | Replace the last sentence: an unprojected plan is the same plan on either set; every violation count of Chapter 6 is taken against the nominal constraints, the tightened set being what the projector is handed; on UAV-corridor and UAV-s-curve the nominal surfaces carry the 0.31 m rotor reach. |
| A4 | Tasks › Vision-Conditioned Alignment › 310 | F13(1) | "Neither is respected by the demonstrations" → the straight push from box to target crosses the keep-out region in 106 of the 120 recorded contexts; the halfspace and the recorded end-effector paths were not tested (or: run the DA and print the counts). |
| A5 **(§11)** | Datasets › Quadrotor Benchmark › 634–638 | F13(2) | The pictured route is the **generator reference route**, which satisfies the constraint set, tightening included, so the demonstrated route needs no correction. What the projection meets on this scene is the learned plan, which cuts the second corner (UAV-s-curve results). The reported evaluation uses the scene to examine tracking and the selected per-step projection configuration; endpoint projection has no guiding step at the selected budget of one evaluation at the threshold of 0.5 (UAV-s-curve › Projection), so the scene yields no ranking of the projection methods. Delete "which is why … cannot order the projection methods" as derived from the route's feasibility. |
| A6 | Datasets › Quadrotor Benchmark › 662–665 | F11 | "every quadrotor result … 0.31 m rotor reach" → UAV-corridor and UAV-s-curve; on UAV-pillars the vehicle's extent enters through the scale (0.36 m radial reach onto the rod's 0.01), no band is added. |
| A7 | Metrics › Vision-Conditioned Alignment › 734–737 | **F03** | Redefine the percentage as the reduction of the median final distance relative to the mean initial distance, 100·(1 − median d_final / 0.4530); 100 % = the median box on its target; 0 % = a median final distance equal to the mean initial one, not a box left unmoved. |
| A8 | Metrics › Quadrotor Benchmark › 767 | F08 | "a flight is scored the moment it passes it and flies on as before" → the crossing is latched the moment it passes it; the flight flies on as before and the controlled-flight checks are taken over the whole flight. |
| A9 | Metrics › Quadrotor Benchmark › 781–783 | F11 | Same scoping as A6 for the violation test. |
| A10 | Metrics › Aggregation and Variability › 833, 840–843 | F11 | "every D3IL-avoiding table" → the D3IL-avoiding tables at the protocol of DPCC, and UAV-pillars; "the quadrotor scenes" (single-seed) → UAV-corridor and UAV-s-curve. |
| A11 | Training › Table 5.x, 1112–1126, and 1135–1137 | F11 | Column "UAV, all scenes" → "UAV-corridor, UAV-s-curve"; caption/prose: UAV-pillars flies the D3IL-avoiding column unchanged. |
| A12 | Training › 1170–1172 | F12 | "which is a property of the training run and not of the comparison" → the comparison is between the trained configurations, recipe included, as the table states them. |
| A13 | Evaluation › Table 5.y δ row, 1199–1200, and 1267 | F11 | δ = 0.025 / **0.03** / 0.025; "0.025 m in all three environments" → 0.025 m on D3IL-avoiding and the quadrotor scenes, 0.03 m on D3IL-aligning. |
| A14 | Evaluation › 1227 | §4 | "the lookahead from the projected state" → the lookahead from the predicted state, whose endpoint is then projected (`hardflow_projection.py:994–1011`). |
| A15 | Evaluation › 1259–1261 | §4 | "except when ηK is a whole number" → a whole number below K; at η = 1 both families project on every step. |
| A16 | Guiding Steps of Endpoint Projection › 1423–1426 | **F05** | "endpoint and per-step projection compute the same plan, and a difference … would be noise" → endpoint projection reduces to a terminal projection of the sampled plan; on instantaneous-velocity matching a K=2 run with the same candidate plans gave bit-identical rollouts under both projectors; on the two average-velocity models the arms differ even there, because the endpoint sampler integrates a different field (next paragraph). |
| A17 | Guiding Steps › 1429–1432 | **F05** | "Where endpoint projection has guiding steps, its sampler …" → at every transport step, whether or not it is a guiding one, so the sampler qualification covers every endpoint configuration of the two average-velocity models. |

**B. Chapter 6 — Results (`v3/chapters/06_results.tex`, v3)**

| # | Section › lines | Finding | Change |
| --- | --- | --- | --- |
| B1 | Obstacle Avoidance › Generative Models › Raw Model Outputs › guard 252 | §4 | Move the panel-cropping note into a `\dataref` (provenance, not prose). |
| B2 | Obstacle Avoidance › Projection Methods › The Cost Frontier › 462–466 | F02 | "so that a fast configuration that reaches the goal less reliably cannot appear as a favourable trade" → … more than 0.05 below the best …; at thirty episodes the band admits 29/30 beside 30/30. |
| B3 | Endpoint and Per-Step Projection › 557–561 | F05 (§9.4) | Add the threshold: at η = 0.5, K=3 is the smallest budget with a guiding step; at η = 1.0, the threshold the K=3 rows ran at, K=2 already has one (the count paragraph, 681–684). |
| B4 | Endpoint and Per-Step Projection › guard 565–567 | §4 | Keep the caveat; move "Recorded in `data_status/PENDING_…md`" into a `\dataref`. |
| B5 **(§11)** | Endpoint and Per-Step Projection › 632–639 | **F02 (K=3)** | "Every nfe=3 point … at the same success, so each is dominated …" → five of the six nfe=3 points are dominated on success, control steps and time by CI-MeanFM at one evaluation (four seeds against five); the sixth, MeanFM under endpoint projection, matches that point's success within 0.3 control steps at four times its time per step. The budget of three therefore offers **similar control-step counts at substantially greater planning time**, three to eight times. |
| B5b **(§11)** | Endpoint and Per-Step Projection › 668–671 | F02 / F05 | The recap: "the budget of three, the first at which endpoint projection can act [at the threshold of 0.5], at similar control steps and three to eight times the time, lies off the cost frontier of [The Cost Frontier], which admits success within 0.05" — a statement about the tolerance-filtered frontier, labelled as such, not strict dominance over the remaining endpoint point. |
| B6 | Endpoint and Per-Step Projection › 663 | **F05** | Name the model and drop the dated ratio: on instantaneous-velocity matching a run at nfe=2 with the same candidate plans produced bit-identical rollouts under both projectors; on the average-velocity models they differ (Setup, guiding steps). "at 2.45 times the cost" removed (pre-2026-08-24 cost accounting) or dated in the `\dataref`. |
| B7 | Obstacle Avoidance › Conclusion › 703–709 | **F02** | "the average-velocity models Pareto-dominate the diffusion model of DPCC" → CI-MeanFM Pareto-dominates it …; MeanFM matches the saving at one episode in thirty (0.967, 58.6 steps, 18.3 ms) — numbers unchanged. |
| B8 | Obstacle Avoidance › Conclusion › 753–754 | F05 | "and at two it has only one" → at two it has none at the threshold of 0.5 and one at 1.0. |
| B9 | Vision-Conditioned Alignment › intro 774–776 | F04 | "on which the constraint set has no bearing — the plan is the same plan" → … except that the tightened evaluation's box–obstacle guard freezes one context, so the unprojected rows of the projection tables are read within that evaluation. |
| B10 | Vision-Conditioned Alignment › Generative Models › 783 | §4 | "moves the box to its target" → moves the box most of the way to its target (no context ends in position, 1357–1358). |
| B11 | Table 6.5 caption 803 · reading rule 886 · Fig. caption 903–910 · Table 6.7 caption 1079 · 1294 · §6.4 2524 | **F03** | Rename to the reduction of the median distance, % of the mean initial 0.4530 m; delete "the dashed red line at 0 % is a box not moved" and "the tick labels are the share closed" as interpretations (the dashed line stays as the 0 % reference, matching C1). |
| B12 | Endpoint Projection across the Models › guard 1241 | §4 | "context~8" → "context~7" (the figure's numbering; the raw position and id stay in the appendix provenance). |
| B13 | Endpoint Projection across the Models › guard 1246–1247 | §4 | Delete "Whether the chapter states it is the author's call (…)". |
| B14 | Endpoint Projection across the Models › 1289–1300 | **F04** | Read the price against the tightened evaluation's own unprojected row: from 80 % at 172.5 ms to 61 % at 266 ms; say what the grey ring is (see C2) — either the tightened-set row after regeneration, or, if the figure keeps the plain-set ring, name it as such and give both pairs. |
| B15 | UAV-pillars › From D3IL-avoiding to UAV-pillars › 1409 | F06 (addition) | "execute the same plans instead of the manipulator" → execute the same planner configuration, re-planning from the vehicle's state. Author's call. |
| B16 | same › 1416 | F10 | "Before projection the vehicle changes nothing" → changes little; the next sentences already give 0.033 → 0.100 and the nine collisions. |
| B17 | same › 1491–1492 | F10 | "so what the per-step projector guarantees on D3IL-avoiding it guarantees on UAV-pillars" → … delivered on every one of the 120 projected flights up to the moment each ended. |
| B18 | UAV-pillars › Conclusion › 1605 | F10 | "stays within about 0.007 of the commanded one" → the 95th percentile of a flight's gap to the commanded position, averaged over the flights, is 0.007. |
| B19 | UAV-corridor › Projection › 1830–1851 | F02 | "the baseline is not [on the frontier] … is dominated" → dominated on steps and time within the band of at most two violating steps, in which the baseline itself violates least (0.8 and 0.0). |
| B20 | UAV-corridor › Conclusion › 1922 | F06 (addition) | "the outcome of the same plan flown" → of the same planner configuration flown. Author's call. |
| B21 | UAV-corridor › Conclusion › 1973 | F09 | "to within 1.6 per flight" → … wherever the flights reach the end of the corridor; over the hump at one and two evaluations under per-step projection they do not. |
| B22 | UAV-s-curve › intro 2012–2015 | **F08** | "A flight ends in one of three ways: it crosses the finish line, …" → it reaches the goal point of its route, it reaches the episode limit, or the divergence guard ends it; crossing the finish line is latched when it happens and does not end the flight. |
| B23 | UAV-s-curve › Table 6.13 caption 2023 (Tables 6.14–6.15 inherit) | F08 | "Success: crosses the finish line" → crosses the finish line in controlled flight (Setup, quadrotor metrics). |
| B24 | Caveat: The Tracking Controller › 2202 | **F06** | "keeping the plan and changing the controller" → keeping the planner and its configuration and changing the controller, in closed loop. |
| B25 | same › Table 6.15 title and caption 2233–2234 | **F06** | "the same plans under two tracking controllers" → the same planner configuration under two tracking controllers. |
| B26 | same › 2296–2297 | §4 | Stale pilot sentence → MuJoCo MPC costs about twenty-four times the cascaded geometric controller per control step, controller and simulator step together, on the unprojected pair. |
| B26b **(§11)** | same › guard 2329–2332 | F07 | Delete "it is only their difference that isolates the controller"; the difference is read as the controller's cost, controller and simulator step together, in two execution environments (the dataref's MJX note). |
| B27 | same › 2339 · Conclusion 2350 · §6.4 2431 | **F07** | "at a twenty-fourth of the cost per control step" → at a twenty-fourth of the controller's cost per control step (a tenth of the whole loop unprojected, half of it projected); the definition of "controller's cost" at B26b travels with the label. |
| B27b **(§11)** | same › 2335–2336 | **F06** | "the same projected plans succeed on five flights of ten or on none, and the same unprojected plans violate …" → the same projected planner configuration …, the same unprojected configuration …, evaluated in closed loop. |
| B28 | UAV-s-curve › Conclusion › 2346 · guard 2367–2369 · §6.4 2429 | **F06** | "the same plans succeed or fail by the controller" / "flies the same plans under both controllers" → the same planner configuration …. |
| B29 | UAV-s-curve › Conclusion › 2359 · §6.4 Conclusion 2486–2490 | F13(3) | "follows from the data" / "as expected from what the average-velocity objectives are built for" → the reading this thesis gives of the data, consistent with the grid. |
| B30 | Comparison across Environments › Generative Models › 2404, 2408 · Conclusion › 2473, 2481 | **F02** | The plural dominance sentence → CI-MeanFM Pareto-dominates the baseline; MeanFM matches its saving at one episode in thirty (four places; the item title at 2408 "Pareto dominance over the baseline" can stay). |
| B31 | Comparison across Environments › Conclusion › 2530 | F09 | "Any flow-based objective" → any of the flow-based objectives tested at a budget: the average-velocity models to three evaluations, instantaneous-velocity matching to twenty. |

**C. Figures (`Data_Analysis/DA_in_Paper`, the figure pipeline; copies reach v3 through `export_to_draft.py`)**

| # | Figure | Finding | Change |
| --- | --- | --- | --- |
| C1 **(§11, required)** | `fig_aligning_tradeoff`, `fig_aligning_projected_tradeoff` | **F03** | The label "box not moved" drawn by `builders/frontier.py:252` and `:390` is removed or replaced by "0 % reference"; the axis label "distance closed, % of the start" (`:141`) can stay under the renamed caption; the builder's own comment (`:136–140`) is updated. Both SVGs regenerated with every plotted value unchanged, re-rasterised to PNG, exported to `v3/figures`, and recorded in the DA index. |
| C2 (optional) | `fig_aligning_projected_tradeoff` grey ring | F04 | Redraw the ring from Table 6.7's tightened unprojected rows (K=10: 0.142 m / 89.7 ms; K=20: 0.090 m / 172.5 ms) instead of the plain-set cells (`frontier.py:365–371`, which carry the same "does not depend on the constraint set" assumption). If not redrawn, B14 labels the ring as the plain-set run. |

**D. Chapter 7 — Conclusion (`v4/chapters/07_conclusion.tex`, v4; via `cross_draft/to_v4` + INBOX, conditional on F02)**

| # | Lines | Change |
| --- | --- | --- |
| D1 | 25 | "the average-velocity models Pareto-dominate its diffusion model" → the CI-MeanFM lead, as B7/B30. Line 28 ("Instantaneous-velocity matching dominates the baseline as well") holds (1.000 / 67.0 / 17.3) and stays. |

**E. Tools (`v3/tools/check.py`, code — separate go-ahead)**

| # | Change |
| --- | --- |
| E1 | `files_from_master()` follows `\input` inside chapter files, so the twenty-episode file is checked and the false "no \label: app:avoiding-twenty" disappears (proposed in the v3.99 changelog). |

**F. Not changed**

Appendix (v4's live file; F14 and §4's two appendix items are done), Chapters 1–4, the abstract, every
number in every table, D01–D04. The v3 layout bundle is rebuilt only when asked; the release is rebuilt from
the live files after the pass.

**Outputs the pass produces:** the edited `05_setup.tex` and `06_results.tex`; the two regenerated figures
(C1) and their v3 copies; `changelogs/v3.100_…md`, the CHANGELOG entry and the briefing addendum; the
cross-draft note and INBOX row for D1; the archive of the replaced passages under `withheld/`; `tools/check.py`
re-run. No number changes anywhere; the claim that the thesis "passed" this audit is made only after those
outputs exist.

**Signed:** Claude (Fable 5.1, Claude Code), v3 owner, 2026-09-25. Nothing applied; this list is the plan.

---

## 14. Implementation record — v3.100, 2026-09-25 — Claude (Fable 5.1, Claude Code)

**Author's decision (2026-09-25):** "I agree with them" — the list of §13 with the defaults (F02: the CI-MeanFM
lead; F03: rename) — and one extension to A4: "the V_A demo is lacking the halfspace, yes, add into it, and need
sanity check elsewhere, I totally forgot there is a halfspace in V_A". Applied as **v3.100**
(`v3/changelogs/v3.100_20260925_audit_v3.98_applied.md`; every replaced passage verbatim in
`v3/withheld/20260925_v3.100_archive/`).

| Part | Status |
| --- | --- |
| A. Chapter 5, A1–A17 | Applied, all. A4 extended: the straight push is now tested against the halfspace too (0 of 120, nominal and tightened; the excluded side is convex, so a push crosses it iff an end lies in it) and the text says the recorded end-effector paths were not tested; the learned plans enter the keep-out region in 7 of ten evaluation contexts and the halfspace in 1 (tightened set). A4x adds the far corner to the box-around-keep-out sentence. |
| B. Chapter 6, B1–B31 + B5b/B26b/B27b | Applied, all, including the §11 additions. B14b: "crossing the constraint in eight contexts" → the keep-out region in seven and the halfspace in one. |
| C1 (required) | Done: `builders/frontier.py:252, 390` label → "0 % reference"; both frontier figures rebuilt, every plotted value unchanged; `fig_expert_aligning` rebuilt with the halfspace count; PNGs re-rendered; exported to `v3/figures/`. Incidental: `fig_uav_corridor_paths` legend brought to the v3.79 vocabulary (the builder had it, the store copy did not). Analysis of record: `Data_Analysis/DA_in_Paper/analysis/DA_20260925_aligning_frontier_label_halfspace_push.md`. |
| C2 (optional) | Not done; the ring stays the untightened cell and B14 names it, as agreed in §11.1. |
| D1 (v4, Ch 7:25) | Filed: `cross_draft/to_v4/FROM_v3_20260925_v3.100_audit_applied_ch7_wording_figures.md` + INBOX row — l. 25 (plural dominance), and two more restatements found on the way, l. 45 ("the same plans") and l. 47 ("a twenty-fourth of the cost"); v4's figure copies are stale and the release prefers them (`make_release.py:584`), so v4 re-exports. Nothing under `v4/` was touched. |
| E1 (`tools/check.py`) | Done: nested chapter `\input` followed; 15 files, 293 labels, all checks pass; the v3.99 false positive is gone. |
| v2 | Nothing to notify: Ch 4 describes the alignment constraint set with both forms; the abstract already names FM and CI-MeanFM. |
| D01 | Open; the text keeps 29/30. |

**Sanity check for the halfspace elsewhere (author's request):** per context, the unprojected plans cross the
keep-out region in 7 of ten and the halfspace in 1 (tightened; 7 and 2 untightened), and by steps the
halfspace dominates where it is crossed (126–272 steps against 71–75); both go to zero under either projector on
the tightened set, and the diffusion baseline's breakdown at Results §6.2 already showed halfspace violations.
Chapter 4 (v2) names both forms. No other passage of Ch 5/6 described the alignment constraint set as the
keep-out region alone once the A4/A4x/B14b sentences were fixed.

**Not compiled; nothing committed; the bundle was not rebuilt (not asked).** The thesis has now had the agreed
corrections applied in its live sources and the figure store; the release output must be rebuilt from them
(and v4's figure copies refreshed) before this audit can be called passed.

**Signed:** Claude (Fable 5.1, Claude Code), v3 owner, 2026-09-25.
