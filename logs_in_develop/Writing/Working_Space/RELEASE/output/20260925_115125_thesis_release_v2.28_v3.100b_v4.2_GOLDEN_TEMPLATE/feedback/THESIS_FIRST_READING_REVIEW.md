# Thesis review: first reading of the supplied release

**Reviewed:** *Flow Matching Predictive Control with Constraints*, PDF `Flow_Matching_Predictive_Control_with_Constraints - 2026-09-25T140912.673.pdf`, 185 PDF pages, and its accompanying thesis source in this release directory.

**Scope:** Only this thesis release was consulted. No repository instructions, user guides, memory, chat history, other releases, implementation code outside the release, or external literature were read. Mathematical and factual criticism below concerns what can be checked from the thesis itself; it is not an independent reproduction of the experiments or verification of the cited papers.

**Locations:** Page numbers below are the thesis’s **printed page numbers**. From the Introduction onward, add **8** to obtain the PDF viewer page number. Source filenames refer to `latex/chapters/`, unless stated otherwise. Line numbers refer to this supplied release.

## Overall assessment

The clearest contribution is the comparison of flow-based planners with the inherited diffusion controller on obstacle avoidance. The common backbone, explicit projection procedures, and analysis of when endpoint projection actually guides sampling provide a useful scientific core. The thesis also discloses several limitations that many experimental accounts leave hidden.

The current document nevertheless needs substantive revision before submission. Its principal problem is **a mismatch between the strength of the summaries and the narrower evidence underneath them**. Alignment progress is sometimes described as task completion; selected results from small evaluations become broad rankings; and plausible explanations become established mechanisms. Repeated summaries amplify these issues. There are also demonstrable mathematical qualifications, contradictory numerical statements, broken references, and visible PDF layout defects.

The strongest impression of AI-assisted writing comes from repeated rhetorical templates and confident explanatory closure, rather than any individual word. This is a stylistic assessment; the text cannot establish who or what wrote it.

### Fix first

| Priority | Issue | Main location |
|---|---|---|
| Critical | Reconcile alignment “seven/two successes” with the later zero-completion result; remove the internal audit identifier. | §6.2.1, p.103; §6.2.3, p.110 |
| Critical | Correct or qualify the claim that 0.31 m inflation covers the quadrotor footprint. | §4.6.3, p.46; §5.6.3, p.80; Fig. A.1 |
| Critical | Repair overlapping figures/tables and broken equation references. | pp.29–31, 61, 67, 122, 131 |
| High | Make seen-context evaluation, zero alignment completion, and abstention visible in the main claims. | Abstract; §§5.3.2, 6.2; Tables 6.6–6.8 |
| High | Separate observed comparisons from uncertainty, configuration selection, and causal attribution. | §5.4.4; §§6.1–6.4; Conclusion |
| High | Remove the repeated conclusion layers and correct statements invalidated by additional table rows. | §6.4; Chapter 7; numerical errata below |

## 1. Content, evidence, and experimental interpretation

### C1. Alignment success has incompatible meanings

**Location:** §6.2.1, p.103, `06_results.tex:770–772`; §6.2.3, p.110, lines 1116–1119; §6.4.3, p.140.

The first passage says MeanFM “succeeds in seven” contexts and CI-MeanFM in two, ending with **`(CLOSURE_20260907 R.4)`**. Later, the thesis says no context finishes within the task’s 1.8 cm position tolerance for the discussed unprojected and selected projected configurations. The earlier phrase “any configuration” might refer to a different population, but no reader-accessible evidence establishes that distinction.

**Fix:** Define success once, identify exactly which configurations produce the seven/two counts, and show those results in a numbered table. If they cannot be substantiated, remove the claim. Removing only the audit identifier leaves the scientific problem unresolved. Replace “the model that solves the task does so at twenty” in §6.2.2 with a statement about reduced position error.

### C2. Restore the original alignment completion metric

**Location:** §5.4.2, p.68; §5.5.2, p.73, `05_setup.tex:793–803`; Tables 6.5–6.8.

The original task requires position within 1.8 cm **and** orientation within 8.6°. The explanation that the thesis changes the network/output does not justify discarding task completion as a metric. Different architectures can still be evaluated against the same physical objective. Distance is valuable as a secondary measure when success is zero.

**Fix:** Report original task success, position-only success, final translation error, and—where available—orientation error. For the configurations already reported as missing the position tolerance, original task success must also be zero. If orientation was not recorded, state that explicitly. Reserve “alignment success,” “solves,” and “delivered” for a stated tolerance; otherwise use “progress towards the target.”

### C3. The alignment result is an evaluation on seen contexts

**Location:** §5.3.2, pp.62–63, `05_setup.tex:449–483`; §6.2 opening, p.99, `06_results.tex:627–628`.

The dataset description identifies 60 training and 60 held-out contexts, but the experiments use ten training contexts. This can establish comparative behaviour on those contexts; it does not establish generalisation to unseen box/target configurations. Separately, checkpoint selection uses a random tenth of training windows. If windows overlap within a demonstration, this can share frames between training and validation. The window stride and split boundaries are needed to assess that risk.

**Fix:** Use demonstration- or context-disjoint validation, and evaluate the available held-out contexts. If no further experiments are possible, consistently say “ten training-set contexts,” including in the abstract, contributions, conclusion, and limitations. Describe the window stride and whether any trajectory contributes to both training and validation.

### C4. A rejected context is being counted as a clean execution

**Location:** §6.2.2, p.108, `06_results.tex:1020–1026`; Tables 6.6–6.8; Appendix B.2, pp.154–155.

The tightened evaluation never attempts context 7 because its box overlaps the enlarged obstacle. It nevertheless contributes one violation-free observation to every row. Consequently, “10/10 violation-free” includes **9/9 attempted executions plus one rejected context**. Likewise, 9/10 becomes 8/9 among attempted contexts, and the baseline’s 4/10 becomes 3/9.

The disclosure is useful, but arrives after the reader has already encountered the headline counts. “Comparisons between rows are unaffected” is too broad: subtracting the same observation preserves the ordering of clean-context counts, but not their interpretation as execution reliability, and including the frozen observation can affect medians and timing summaries.

**Fix:** Keep the full evaluation population visible, and add an explicit rejected/abstained count. Report outcomes conditional on attempted execution alongside it. Recompute paired distances and timing on the same attempted contexts where the purpose is to compare execution. Introduce the guard in Experimental Setup. Do not silently discard the rejected case or describe it as successful constrained task execution.

### C5. The abstract needs the limitations that change its meaning

**Location:** `latex/pages/abstract.tex:16–25`; Introduction contribution 5, pp.3–4.

“Dominates the diffusion model again” is liable to be read as successful visual alignment or successful transfer. The detailed experiment instead concerns progress and constraint outcomes on ten training contexts, one training seed, with zero position-tolerance completions for the highlighted configurations. All platforms are simulated, and the discussion states that no loop ran in real time.

**Fix:** Replace some of the abstract’s training-target detail with the main measured result and its scope. For example:

> On ten training-set alignment contexts, MeanFM reduced final position error relative to the diffusion baseline. At the selected budget, neither the unprojected policy nor its selected projected configuration completed alignment within the position tolerance. All evaluations used simulation.

Keep the strong obstacle-avoidance result, but attach the evaluated baseline and measured endpoints to it.

### C6. The CI-MeanFM plateau is overstated and its explanation is untested

**Location:** §6.2.1, pp.101–103, `06_results.tex:700–769`; §6.2.3, p.110; Table 6.5.

“Does not improve when it is given a larger budget” and “spends time and buys nothing” conflict with the reported median falling from **0.3897 m at K=20 to 0.1791 m at K=100**. The subsequent statement that “the mechanism is in the training target” goes beyond what this budget sweep identifies. Optimisation, conditioning, sampling choices, and training variation have not been separated from target construction.

**Fix:** State the actual range: no observed improvement from K=2 to K=20, followed by improvement at K=100. Present the target-error account as a hypothesis. An ablation of the consistency target or direct error diagnostics would be needed to establish that mechanism. Update every repeated version of this explanation in the summaries.

### C7. “Human versus generated demonstrations” is a hypothesis, not an isolated cause

**Location:** Abstract ending; §§6.3.2–6.4, especially pp.126–127 and 136–139; Chapter 7, pp.142–143.

The thesis repeatedly explains the model ordering through human versus simple generated demonstrations. But demonstration source changes together with task, vehicle, controller, dimensionality, and sampling configuration. “The generative model does not matter” and “the objectives cannot be told apart” also sound stronger than a small, single-training-seed comparison establishes.

**Fix:** Report task-dependent observed rankings first. Move the explanation to Discussion: “The simpler demonstration distribution may contribute, but its effect was not isolated.” A within-task experiment varying route diversity would test it. Replace equivalence language with a numerical description of the observed differences at the shared budgets.

### C8. Architecture matching does not isolate the training objective

**Location:** Table 5.8, p.77; §5.6.2, `05_setup.tex:973–988`; §5.6.3, lines 1123–1128; §4.5.3, p.42.

The configurations differ in more than their loss: on avoiding, the average-velocity models use a fourfold larger batch, fivefold higher learning rate, and EMA. FM is trained with noise standard deviation 1 but sampled at 0.5, whereas the average-velocity models use 1 at both stages. These differences are disclosed, but the summaries sometimes attribute outcomes to the objective alone.

There is a second confound in the projection comparison: the endpoint sampler queries the average-velocity models at **zero interval**, while their ordinary sampler uses the **finite step interval**. Thus a fixed checkpoint does not make this a pure change of projection location.

**Fix:** Describe the existing results as comparisons of trained configurations and implemented sampler–projection combinations. For stronger causal claims, prioritise a matched FM noise-scale ablation and a comparison that holds the field query fixed while changing projection. Report tuning effort or convergence evidence when defending different training settings. The present choices do not invalidate a system comparison; they limit its interpretation.

### C9. Correct the uncertainty argument and qualify the Pareto claims

**Location:** §5.4.4, pp.70–71; Tables 6.1–6.3; cost-frontier discussions.

The assertion that between-seed SD is “the larger of the two spreads available and therefore the conservative one” is not generally true. SD describes dispersion; it is not automatically uncertainty in a mean or difference. Averaging episodes within a seed can reduce dispersion relative to the individual episode outcomes.

The dominant runtime differences are large, but one episode out of thirty, one context out of ten, or a few tenths of a millisecond cannot support equally strong reliability rankings. Table 6.3 also changes replication between its K=3 and K=10 blocks and omits variability despite the general reporting promise.

**Fix:** Remove the “larger/conservative” sentence. Define the aggregation precisely. Use paired differences and uncertainty intervals over the appropriate independent units; do not treat several episodes from one trained model as several independent training replications. Label the existing fronts as **fronts of observed means**. Add uncertainty to the projection comparison and state its seed/episode exceptions beside the relevant blocks.

### C10. Selection and confirmation are currently intertwined

**Location:** §6.1 opening, p.84; Table 6.12, p.122; §6.3.3 selection of the s-curve configuration.

The best rule/configuration is selected using the same outcomes reported as its evidence. This is understandable for exploratory analysis, but the selected result benefits from searching the grid. With ten flights per cell, the distinction is material.

**Fix:** Call these exploratory selections on the evaluated grid. State the selection criterion before presenting the chosen result. Freeze the selected configuration and confirm it on independent episodes or training seeds before describing it as a validated operating point. Where no confirmation is available, retain the full grid in the appendix and qualify the conclusion.

### C11. The UAV success definition materially changes the baseline result

**Location:** §5.4.3, p.69; Table 6.12, p.122.

Corridor success is crossing the end of the walls rather than the route’s goal line 0.8 m farther on. For the selected diffusion rows, the stated counts change from **0/10 at the goal to 10/10 at the exit under tilt**, and from **3/10 to 10/10 under hump**. This is disclosed, but relegating such a large difference to prose/caption makes the main comparison harder to assess. “Success” also allows the stated nonzero wall-contact fractions; it is not synonymous with collision-free completion.

**Fix:** Put both corridor traversal and goal-reaching counts in the main table, using the same timeout, and label them distinctly. Explain whether the exit criterion was fixed before evaluation. A common longer-timeout sensitivity test would help separate slow progress from failure. Keep completion and collision/constraint outcomes distinct throughout.

### C12. Controller timing is not a controller-only measurement

**Location:** Tables 6.15–6.16, pp.134–135; §6.3.3; Chapter 7 summary.

The “twenty-four times” controller-cost comparison is obtained from whole-loop time minus planner time. The residual includes simulation and other overhead, and the two controller configurations use different MuJoCo/MJX environments. The caveat exists, but the causal headline remains stronger than the instrumentation.

**Fix:** Call this “controller-and-simulator overhead” consistently, including the conclusion. To attribute the difference to the controller itself, instrument its calls and use a matched execution environment. Retain the explicit statement that simulation waits for planning; planner latency alone is not a demonstrated real-time closed-loop result.

## 2. Mathematics and technical specification

### M1. The 0.31 m footprint inflation is not a general clearance bound

**Location:** §4.6.3, p.46, `04_method.tex:1477–1482`; §5.6.3, p.80, `05_setup.tex:1076–1086`; Fig. A.1, p.152.

The thesis gives rotor centres `(±0.14, ±0.18) m` and rotor-disk radius `0.13 m`. It obtains `0.31 m` from the largest **axis-aligned** reach and applies that offset to walls, planes, roofs, and disks. Even for a level vehicle at zero yaw, the support in the horizontal diagonal direction is

\[
h\!\left(\frac{(1,1)}{\sqrt2}\right)
=\frac{0.14+0.18}{\sqrt2}+0.13
\approx0.3563\ \mathrm m.
\]

That exceeds both `0.31 m` and `0.31 + 0.025 = 0.335 m`. Zero yaw does not make an oblique boundary axis-aligned. Fig. A.1 itself shows rotor areas outside the dashed 0.31 m circle.

**Fix:** Use a support-function offset for each relevant surface normal and vehicle orientation, or a conservative enclosing bound. For the stated level rotor footprint, an enclosing radius is

\[
\sqrt{0.14^2+0.18^2}+0.13\approx0.3580\ \mathrm m.
\]

Full 3D body/attitude clearance needs its own bound. Until geometry and results are rechecked, qualify the claim that centre feasibility gives vehicle feasibility. This finding does **not** establish the cause of any particular recorded collision.

### M2. Tracking error and dynamics mismatch are different quantities

**Location:** §4.3.4, p.23, after Eq. (4.11), `04_method.tex:351–354`.

The statement that the residual tracking error “is precisely the” model mismatch `w_t` is incorrect under the stated setpoint update. Define

\[
e_t=p_{\mathrm{des},t}-p_t,
\qquad p_{\mathrm{des},t+1}=p_{\mathrm{des},t}+\Delta p_t.
\]

Then the measured-position mismatch is

\[
w_t^p=p_{t+1}-p_t-\Delta p_t=e_t-e_{t+1}.
\]

**Fix:** Explain that it is the **change in tracking error**, not the tracking error itself. Use the same definition when discussing the mismatch bound. Conversely, a large constant lag does not by itself prove that this increment-level bound is violated.

### M3. The mismatch-bound calibration needs evidence

**Location:** §§4.3.1 and 4.5.2, pp.20 and 39; §5.6.3, p.80; Discussion §8.1.

The method says the mismatch bound is estimated from unconstrained rollouts, but Setup presents margins of 0.025 m inherited from DPCC and 0.03 m for alignment without documenting the corresponding calibration for each plant. The validity of the tightening argument depends on the particular constrained state, norm, model, and execution loop.

**Fix:** Supply the error definition, coordinate units, number of calibration rollouts, observed maximum or quantile, and exceedance rate on evaluation. Otherwise describe the margins as empirical parameter choices and make the safety argument conditional. The manipulator assertion that tracking stays within the margin also needs the stated measurement behind it.

### M4. The full-state feasibility definition overstates the specified UAV constraints

**Location:** Eq. (4.4), p.17; Table 4.3; §4.6.3, p.46.

Eq. (4.4) imposes dynamics on the full state. The UAV plan state includes measured velocity, but the detailed implementation constrains only the commanded and measured position channels. No evolution constraint is specified for the velocity channels.

**Fix:** Introduce a map selecting the state channels subject to the model constraints, and identify the remaining channels as auxiliary predictions/conditioning variables. Define the mismatch bound on that same space. Also give Eq. (4.4) explicit index ranges: dynamics should run only over consecutive stored states, as the later implementation already specifies, and initial-state anchoring should be stated.

### M5. Specify whether keep-out geometry is a disk, cylinder, or sphere

**Location:** Eq. (4.72), p.44; §4.6.2, p.45; §4.6.3, p.46.

The equation `||p_i − o||² ≥ ρ²` becomes a spherical exclusion for a three-dimensional position, while later passages call the obstacles horizontal disks. With a two-dimensional centre and three-dimensional position, the subtraction is not defined.

**Fix:** For a horizontal disk extruded vertically, write

\[
\|P_{xy}p_i-o_{xy}\|_2^2\ge\rho^2.
\]

If a sphere is intended, state that explicitly. This is a specification ambiguity with mathematical consequences, not a verified implementation defect.

### M6. Keep sampling steps, network evaluations, and projection solves distinct

**Location:** Table 4.1; §§4.5.3–4.5.4, p.42; §§5.6.3–5.6.4, pp.78–83; results captions.

The method correctly gives `NFE = K + n_guide` for endpoint projection, yet much of the prose calls K the number of network evaluations. For example, `K=20, η=0.2` means three guiding steps and **23** field evaluations under the stated endpoint algorithm, versus 20 in the ordinary sampler. Equal K is therefore not equal NFE for those comparisons.

**Fix:** Use K consistently for sampling steps. Add actual NFE and projection-solve counts to the core projection comparison, or provide a short accounting table. Retain wall-clock comparisons. Correct claims of equal network-evaluation budget where only K is matched. Also preserve the diffusion gate’s extra inclusive projection step: at K=20 and η=0.5 it gives eleven solves, not ten.

### M7. Increasing the horizon alone does not activate future switched walls earlier

**Location:** Eq. (4.74), p.46; “The planning horizon,” §8.3, p.148.

The wall set is selected from the **current** position, `W(x_t)`, then applied across the horizon. It does not depend on predicted future positions or horizon length. The future-work claim that a longer horizon “would hand the projector its constraints earlier” does not follow for these switched walls under the written rule.

**Fix:** Distinguish extending the horizon from changing constraint activation. Earlier anticipation would require selecting relevant walls from a predicted/reachable corridor or otherwise revising the switching rule. Explain how such a change avoids imposing an irrelevant wall over the entire trajectory.

### M8. Separate target equivalence from objective equivalence

**Location:** §4.4.3, p.32, Eqs. (4.38)–(4.39); §4.4.4, p.33; `04_method.tex:809–881`.

At zero interval, the average-velocity **target** reduces to the instantaneous velocity. The implemented loss still has two heads, adaptive reweighting, and its own sampling distribution, so saying the objective “becomes” FM is too strong without additional qualifications. Similarly, the carefully qualified small-alpha residual relationship in §4.4.4 should not become an unqualified claim of identical objectives elsewhere.

**Fix:** Say “the target reduces to the instantaneous-velocity target.” State the additional assumptions needed for equality of losses or gradients. Give the adaptive-weight constant and precise curriculum parameters needed to reproduce the displayed implementation.

### M9. Complete the controller and solver specification where it affects interpretation

**Location:** Eq. (4.13), pp.24–25; demonstration generation, p.48; projection solver description, pp.36–37; Appendix C.

Demonstrations use analytic velocity and acceleration feed-forward. Evaluation explicitly sets desired velocity to zero but does not clearly state the desired acceleration used in the displayed controller. The nonconvex solver may return its last iterate on non-convergence; that important disclosure is not followed by solver tolerances or failure/residual statistics.

**Fix:** State evaluation acceleration and yaw conventions, and distinguish using the same controller equations from using the same reference inputs. Report SLSQP iteration limits, termination tolerances, non-convergence frequency, returned feasibility residuals, and what is executed after failure. A small diagnostic table would separate sampler errors, numerical projection failures, and tracking errors more effectively than further narrative about which component “decides” the result.

## 3. Direct consistency corrections

These are comparatively inexpensive edits, but they should be checked against the final tables rather than patched independently.

| Location | Current problem | Correction |
|---|---|---|
| §4.4.2–4.4.3, pp.29, 30, 31 | Three rendered equation references read `(??)`. | Repair the labels/references and inspect the rebuilt PDF. The FM equation has two labels worth checking. |
| §5.6 opening, p.75; `05_setup.tex:906–907` | “Every model” uses the same checkpoint at every budget. | Limit this to flow models; §6.1.1 says diffusion budgets are separately trained. |
| §6.1 opening, p.84; `06_results.tex:11–12` | “Every comparison” uses five training seeds and two episodes. | State exceptions: Table 6.3 uses four or one seed, and an additional episode-count exception. |
| §6.1.1, p.85; lines 38–39 | No raw row exceeds 0.100 S&C. | FM at K=20 is **0.133** in Table 6.1. |
| §6.1.1, p.86; lines 85–86 | Raw violating steps are described as 15.5–19.0. | Table 6.1 contains **20.4** for diffusion at K=2. |
| §6.1.1, p.86; lines 98–99 | The reference diffusion projection runs ten times. | Its inclusive gate gives **eleven** at K=20, η=0.5, as Setup correctly explains. |
| §6.2.1, p.101; lines 729–730 | “Four of the ten configurations” lie on the frontier. | The figure’s stated subset contains **13** configurations: 15 table rows minus the two alpha=0.05 rows. |
| §6.2.2, p.103; lines 776–779 | MeanFM comes closest at K=20; all following projected results use that model/budget. | K=100 has the smaller raw median; K=20 is a selected cost–outcome operating point. Later tables include other models/budgets. |
| §6.2.2, p.108; lines 1027–1028 | CI-MeanFM and FM “do not move the box.” | Table 6.5 reports only **2/10 and 4/10 unmoved** at K=20. Say their median progress is small. |
| §6.1.2, p.89; lines 256–257 | Avoiding is the only setting ahead on every axis. | Reconcile with the same kind of dominance claim for alignment on p.108. |
| Fig. 6.2 caption, p.93 | Temporal consistency is described as triangles. | The plotted markers and legend use **squares**. Also define the hollow markers in the caption/legend. |
| Front matter, PDF pp.1–3 | Author, supervisor, advisor, and date are TODO entries. | Complete the submission metadata. |
| Appendix B, p.153 and running headers | “Extended Results (long data: web link and repository code link).” | Rename to “Extended Results”; put genuine supplement information in ordinary text. |

The introduction’s broad diffusion-versus-flow distinction also needs a consistency edit. Abstract line 5 and Introduction pp.1–2 imply that diffusion cannot vary inference steps without retraining, while Related Work p.12 discusses shortened implicit diffusion sampling. Restrict the opening claim to **the inherited DPCC implementation evaluated here**; do not present it as a universal property of diffusion.

## 4. Academic writing and conspicuously formulaic prose

### W1. Replace repeated aphorisms with bounded findings

The following are unusually polished, slogan-like formulations when repeated across a technical thesis:

| Location | Wording | More precise alternative |
|---|---|---|
| Introduction, p.1 | “buys expressiveness with computation” | “Generative trajectory models require repeated network evaluations during planning.” |
| Introduction, p.1 | “structural, and the structure is what makes it addressable” | Specify the inherited denoising schedule and the evaluated inference procedure. |
| Results, p.139 | “The price of a projection is set by what the projector is handed.” | “Projection time differed substantially between the evaluated intermediate samples and terminal plans.” |
| Results, p.139 | “A task saturated at one evaluation hides what a task that rewards more evaluations exposes.” | “The budget sweep separated the models more clearly on alignment than on avoiding.” |
| Conclusion, p.142 | “a plant that holds nothing by itself” | “an underactuated quadrotor requiring feedback stabilisation” |
| Discussion, p.145 | “The results stop at three things” | Delete; the section headings already identify limitations, deployment, and future work. |

An occasional metaphor is not a problem. The persistent pattern is: memorable contrast, confident explanation, restatement of the same result. That cadence makes the document sound manufactured and sometimes hides the limits of the evidence.

### W2. Reduce counted scaffolding and reader coaching

**Locations:** `04_method.tex:199,265,351,470,669,764,817`; §6.1.1, pp.85–86.

Examples include “Three things follow,” “Two properties follow,” “Two consequences matter,” “Three pieces define it,” and “Four properties.” Results then uses “Read down the success column first,” “Read the next column,” and “The last column prices generation alone.”

**Fix:** Keep lists where the items are genuinely distinct. Delete most introductory counting and tell the reader the finding directly. One synthesis paragraph can replace the three instructions for reading Table 6.1:

> Without projection, the flow-based configurations reached the goal in all reported avoiding episodes, but success with constraint satisfaction remained low. Projection therefore accounts for most of the improvement in constraint satisfaction. At the selected budgets, it also contributes a substantial part of planning time.

### W3. Use literal technical verbs

**Locations:** Results throughout; Discussion pp.146–147; Conclusion pp.141–143.

The repeated vocabulary of “reads,” “prices,” “hands,” “buys,” “carries,” “the baseline of record,” and “the node of record” makes straightforward relationships indirect. A checkpoint is evaluated, a solver receives a candidate, and a measurement includes or excludes an operation.

**Fix:** Replace these expressions according to their actual meaning. In particular, “the better projector on the constraint” should name the endpoint: more violation-free contexts, fewer violating steps, or smaller violation depth. Those are different claims.

### W4. Shorten the long contribution and RQ paragraphs

**Location:** Introduction §1.4, pp.2–4; Chapter 7, pp.141–143; Discussion pp.146–147.

Contribution 6 combines platform provenance, coordinate transfer, scene construction, controller design, and several result interpretations in one long item. The RQ1 answer then repeats nearly every task, several exact numbers, and multiple exceptions. This obscures the main answer.

**Fix:** Use three or four contributions, explicitly distinguishing inherited objectives from the thesis’s implementation, comparison, and benchmark work. Give each RQ a direct bounded answer followed by two or three supporting sentences. Move detailed numerical sequences to one synthesis table. Put the important limitation in the same paragraph as the claim it qualifies.

### W5. Reduce model-name repetition and overclaiming verbs

Define FM, MeanFM, and CI-MeanFM once, with their relationship to the cited methods, then use those labels consistently. Repeating “consistency-interpolated average-velocity matching” several times in one paragraph makes the prose harder to parse without adding precision.

Replace “proves,” “the mechanism is,” “the model does not matter,” and unqualified “dominates” where the evidence supports only an observed comparison or interpretation. Do not replace them with vague hedging everywhere: attach the configuration, metric, and evaluation scope instead.

### W6. Preserve the distinction between the displayed percentage and actual per-context progress

**Location:** §5.4.2, p.68; §§6.2.1–6.2.2; Figs. 6.3–6.4.

The defined percentage is `100 × (1 − median(final distance) / mean(initial distance))`. It is not the median of per-context percentage improvements, and zero on that scale does not mean that every box was unmoved. Setup explicitly acknowledges this, but later prose calls it “the line at which the box was not moved at all” and says the model “closes 84% of the distance.”

**Fix:** Prefer the absolute median distance in the headline. If reporting typical percentage progress, compute the paired per-context improvement and summarise that. Otherwise consistently call the existing number a normalised aggregate distance, and avoid interpreting it as a typical trajectory’s completed fraction.

## 5. Specific material to shorten, merge, or move

The thesis’s length is not by itself the objection. The clearest removable material is repeated interpretation and duplicated data. The main body runs to p.149; Method, Setup, and Results occupy approximately 33, 35, and 57 pages respectively.

| Material | Action | What to preserve |
|---|---|---|
| Six contributions, Introduction pp.2–4 | Reduce to three or four short contributions. Remove the miniature results narratives. | What this thesis implemented, analysed, compared, and built; clear attribution of inherited methods. |
| Background vs Related Work, pp.6–15 | Remove repeated diffusion/flow/projection descriptions. | Background defines concepts; Related Work identifies the comparison gap. |
| Arm/quadrotor contrast, `02_background.tex:130–140` | Merge the immediately repeated explanation into one paragraph. | Why feedback and tracking make aerial transfer a meaningful test. |
| MeanFM development, pp.30–34 | Remove repeated target/sampler displays; move inherited derivation detail if needed. | One derivation, one exact implemented loss per model, one sampler comparison. |
| §4.6 environments and §5.2 task descriptions, pp.43–61 | Cross-reference shared scene descriptions. | Method: variables and constraints. Setup: numerical geometry and protocol. |
| Tables 6.6–6.8, pp.105–109 | Keep the full cross-model comparison in the body; move the complete threshold/rule ladder to the appendix. | Main trade-off and one informative lower-budget comparison. The MeanFM K=20 block is currently repeated. |
| Table 6.11, p.120 | Combine the two rescorings into one row per model/budget with separate tilt/hump violation columns, after confirming which timing fields are shared. | Distinct constraint outcomes without repeating identical success and step data. |
| Task conclusions plus §6.4, pp.136–140 | Keep short task takeaways; replace §6.4’s repeated numerical tour, extra conclusion, and configuration list with one synthesis table. | Findings that genuinely compare tasks, including where transfer does not hold. |
| Chapter 7 Summary plus RQ answers | Answer the RQs once. | Main effect size, decisive qualification, and contribution. |
| Discussion candidate-selection paragraph, p.147 | Reduce its many counts and timings to one implication and one representative example. | Serial projection is a runtime target; the parallel runtime estimate is hypothetical. |
| Appendix B.1–B.4 full rule grids | Retain as supplementary evidence; improve formatting. | These are useful for checking the selected main-text rows. |

The sentence opening §6.4 explicitly says it “adds no new result.” That is a particularly clear invitation to shorten it. The same avoiding comparison—59.2 versus 70.1 control steps, 18.1 versus 553.4 ms—and the “one episode in thirty” qualification recur too often.

Move **Discussion before Conclusion**, or combine them into a final chapter that ends with concise RQ answers. The current order places the strongest closing claims before the limitations that materially qualify them.

## 6. Tables: layout, information density, and notation

The basic table style is sound: restrained rules and no unnecessary vertical grid. Most large results tables use approximately **9 pt** main entries in an approximately **11 pt** document, as measured from the PDF text. They do not all need a font change. The main problems are overfull floats, repetitive entries, very long captions, and prose packed into cells.

| Table / page | Concrete issue | Fix |
|---|---|---|
| **6.12 / p.122** | **Actual footer collision.** The final FM K=20 single-candidate row occupies the page-number region; the bottom of the table extends past the body. | Split tilt/hump into separate tables or a continued table. Shorten the caption. Do not solve this by shrinking the entire table further. |
| **5.8 / p.77** | A long caption precedes a multi-level header with awkward breaks such as “MeanFM, CI-MeanFM” and task names. Explanations compete with numerical settings. | Put shared defaults and tensor-shape explanations in a short note/text paragraph. Use a compact setting matrix with deliberate header line breaks. |
| **5.9 / p.79** | Several cells are paragraphs. The body-inflation definition consumes much of the table, and the controller name breaks into many fragments. | Move definitions into the preceding text; keep parameter, unit, and value in the table. Widen the UAV column after removing the prose. |
| **6.2 / p.91** | Dense full grid plus a long caption; symbols and bolding carry several different messages. | Retain a compact main comparison and put the full grid in the appendix if space is needed. Define selection/baseline/bold conventions once and apply them consistently. |
| **6.3 / p.95** | Different replication protocols share a small table without uncertainty. | Separate the protocol blocks visibly and report the appropriate spread or paired intervals. |
| **6.6 / p.105** | Caption promises mean ± sample SD, but rows contain means without the stated SD. | Add the promised values or correct the caption. Remove rows duplicated elsewhere. |
| **6.7–6.8 / pp.106, 109** | Repeated data; nine columns combine counts, means, medians, percentages, and timing. | Keep one full main comparison. Replace embedded median percentages with a separate figure/text interpretation if they crowd the table. |
| **6.11–6.12; B.1–B.4** | Repeated `1.000 ± 0.000` for ten-flight binary outcomes adds width and apparent precision. | Use `10/10` for observed counts. Report uncertainty separately where a reliability claim depends on it. |
| **6.14–6.16 / pp.132–135** | Several small tables repeat the same selected model/budget and controller conditions. | Consider one compact controller comparison with a timing panel, keeping outcome and execution-environment differences explicit. |
| **C.1 / p.167** | Detailed cache/hardware inventory is easier to find than exact experiment identifiers. | Trade low-value inventory detail for seed/context/checkpoint/version identifiers needed to reproduce the results. |

Use decimal-aligned numerical columns, consistent precision for the same metric, and unambiguous headings such as **Planner time [ms/action]** rather than repeatedly redefining “ms/step.” A dash should have one defined meaning per table; do not interchange “not run,” “not applicable,” and “no successful flight.”

Many captions repeat the backbone, candidate count, protocol, rules, aggregation, marker meanings, and interpretive conclusion. Retain the essential sample unit and exceptions; move stable definitions to a shared note. For example, Table 6.2 could start:

> D3IL-avoiding with per-step projection: mean ± between-seed SD over five training seeds, two episodes per geometry, and three geometries. Four candidates; tightened constraints. Rules: random (r), projection cost (c), and temporal consistency (t). ▶ Selected configuration; ● reference baseline. Metric definitions: §5.4.1.

Any further bolding convention can be a short table note. Captions should identify what is shown and how to read it, without repeating the entire argument.

## 7. Figures: rendered-page inspection

These comments concern the supplied PDF at its final page size. The labels embedded in raster images do not expose reliable original font sizes; “small” below is a visual judgement, not an invented point-size measurement. Several figures are readable, especially Fig. 5.6 and Fig. A.1. The defects are concentrated in particular multi-panel figures and exports.

| Figure / page | Observed problem | Concrete fix |
|---|---|---|
| **4.1 / p.41** | Twelve panels across two rows leave the shared legend and step labels much smaller than body text. The figure is understandable mainly through its long caption. | Use a larger/landscape presentation or fewer representative stages with proper “adapted from” attribution if modifying the reproduced figure. Explain the mechanism with a short local schematic if the source image must remain unchanged. |
| **5.3 / p.53** | Three narrow panels make repeated tick labels and the legend small; some internal constraint labels sit close to boundaries. | Increase plot typography, use shared axes where possible, and move geometric definitions to the caption. Keep the three comparisons. |
| **5.6 / p.57 and 5.10 / p.64** | Both occupy substantial space with the same geometry. Fig. 5.10’s subtitle is visibly **cut off on the right**. | Retain the ten-context view in the body; consider moving the 120-context overlay to the appendix. Re-export the cropped subtitle within the image canvas. |
| **5.8 / p.61** | Seven panels are packed into a tall figure; the **caption crosses the footer rule and reaches the page-number region**. Small internal annotations are hard to read. | Split the planar/pillars and spatial/corridor explanations, or use a deliberate two-page presentation. Shorten the caption and reserve vertical space for it. |
| **5.11 / p.67** | The third legend line is **truncated at the right edge**; caption lines cross the footer, with the page number among the caption text. | Shorten “vehicle to scale” legend wording; move its caveat to the caption. Re-export with a bounded legend and reduce/rearrange the three-panel height. |
| **6.1 / p.87** | Nine small plots have extremely small ticks. The mostly empty K=20 row consumes height, while the caption and surrounding prose both explain what is not executed. | Use a compact 2×4 grid for K=1/2 and a separate baseline inset. Share axes, enlarge ticks, and explain raw versus executed paths once. |
| **6.2 / p.93** | Point labels overlap in the dense right-hand clusters; hollow points are not adequately explained; caption says triangles while the figure uses squares. | Correct the shape description. Define hollow eligibility, label fewer points, and move detailed geometry-specific panels to the appendix if the aggregate panel answers the main question. |
| **6.3 / p.102** | Several low-progress labels overlap near the zero line. The y-axis presents percentage ticks on a logarithmic transform of remaining distance, which needs a long explanation. | Plot median final distance [m] on a log axis, or percentage improvement on a linear axis. Use direct labels only for key configurations. |
| **6.4 / p.111** | The large “better” arrow occupies substantial plot area. The grey unprojected reference comes from the unguarded evaluation, while the projected points use the guarded one. CI-MeanFM also changes colour from nearby plots. | Plot the matched guarded reference as the principal comparator, with any unmatched reference explicitly secondary. Remove the large arrow and use the same model colours throughout. |
| **6.5 / p.117** | Six small trajectory panels plus several legend meanings make local path differences difficult to inspect. | Retain the geometry comparison, but enlarge the key failure region or move the full grid to the appendix and show one representative comparison in the body. |
| **6.6 / p.125** | Four panels combine model colours, two projection shapes, hollow eligibility, budget numbers, rings, staircases, and shaded groups. Several budget labels overlap. | Separate the primary violation/time frontier from the within-group step comparison. Define groups numerically and label only representative/frontier points. |
| **6.7 / p.131** | **Severe overflow:** the image reaches the footer; the caption begins in the page-number area and extends far into the bottom margin. A five-panel arrangement leaves an empty sixth slot. | Use a compact comparison with selected panels and put the remaining plans on a second page/appendix. Budget the combined image-plus-caption height, not image height alone. |
| **6.8 / p.133** | Four trajectory panels are useful, but subplot subtitles and legend text are small. | Enlarge labels and identify panels as controller × projection, reducing repeated descriptive text. |
| **A.1 / p.152** | Good-sized labels, but the dashed inflation circle visibly fails to enclose the rotors. | Retain the figure and use it to correct/qualify the geometry claim in M1. |
| **B.1–B.3 / pp.154–156** | B.1 is small for four trajectory panels. B.2 and B.3 have long captions repeating methods and setup. | Increase B.1 panel size; shorten captions to condition, encoding, and specific exceptions. Keep the useful path evidence. |

For the thesis-generated plots, use vector export where possible and choose typography at the **final printed dimensions**. A practical target is roughly 8–9 pt or larger for essential tick/legend text, checked on the actual page. Fewer panels and fewer annotations are usually better than enlarging the complete raster until it overflows.

The repeated boxed “better” arrows can usually be removed. Axis labels and a short caption statement convey the direction, freeing room for data and readable labels. Also avoid relying only on red/green distinctions for trajectory outcomes: line style or explicit markers can carry the same distinction.

## 8. Missing evidence and the most useful additions

These additions would contribute more than another summary paragraph or complete grid in the main body:

1. **An alignment outcome table by context:** initial/final position error, position success, orientation result if available, constraint outcome, and attempted/rejected status. This would resolve several current ambiguities at once.
2. **A held-out alignment evaluation with additional independent training seeds**, or a clear reduction of the transfer/generalisation claims if it cannot be supplied.
3. **A solver/mismatch diagnostic table:** convergence rate, returned feasibility residual, tracking-model error, and margin exceedance. This is necessary to interpret why a projected plan can still violate a constraint in execution.
4. **A compact reproducibility table:** exact seed identifiers, context IDs/order, L/C/R flight allocation, selected checkpoint identifiers, code revision, and precise versions of the secondary MuJoCo/JAX/MJX environment. “MuJoCo 3.x” and “the job log records the revision” are not sufficient identifiers in the thesis as delivered.
5. **A short internal-validity paragraph:** training-context evaluation, unequal training/sampling settings, endpoint sampler changes, selection on the evaluation grid, and limited independent replication. Several are already disclosed elsewhere; gather them where the claims are assessed.
6. **A clearly separated observation–interpretation–test sequence** for the two central explanations: why CI-MeanFM plateaus over part of the ladder, and why rankings change across demonstration types. An observed pattern, a plausible account, and a discriminating experiment should not be written as one established result.

## 9. Suggested revision sequence

1. Resolve the contradictory success statements, metric definitions, footprint bound, and equation/specification issues.
2. Decide which stronger claims can receive additional evidence and which must be narrowed. Propagate the decision into the abstract, contributions, Results, and Conclusion together.
3. Remove duplicate numerical tours and merge the overlapping alignment tables. Preserve the full experimental record in the appendix.
4. Rewrite formulaic passages into literal findings, using short model labels and explicit metrics.
5. Repair the named figure/table exports and all draft artifacts. Rebuild and inspect the final PDF, especially printed pp.29–31, 61, 64, 67, 93, 122, and 131.

The revision should preserve the measured obstacle-avoidance improvement and the useful negative findings on alignment and aerial execution. Those limitations make the thesis scientifically more informative when stated directly; they should shape the main argument rather than appear only after repeated claims of dominance.
