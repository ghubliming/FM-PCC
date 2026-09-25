Written: 2026-09-25 21:32:22 UTC
Updated: 2026-09-25 21:50:08 UTC — explicit conclusion and deeper thesis-only recheck added.

# Round 2 — feedback on the response, decisions, and v5.7 execution

> Round-one feedback directory: [GOLDEN_TEMPLATE/feedback](/workspaces/FM-PCC/logs_in_develop/Writing/Working_Space/RELEASE/output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/feedback)

Response assessed: [Claude/author answer and A–D decisions](/workspaces/FM-PCC/logs_in_develop/Writing/Working_Space/RELEASE/output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_to_THESIS_FIRST_READING_REVIEW_20260925_1310_v2.28_v3.100b_v4.2.md).

Current release: `20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX`.

This is a targeted second reading, **not a new full audit**. I read the authorized response and relevant old/new thesis LaTeX. Following the request for a clearer conclusion and a deeper recheck, I additionally compared current protocol definitions, controller descriptions, numerical tables and their summaries, and checked literal LaTeX references. I did not consult the guides, code, data, logs, analyses, or changelogs cited by the response. Their contents and the response’s claims about them remain unverified here. The new directory contains thesis LaTeX and logo PDFs, but no compiled thesis PDF. Consequently, I can assess source changes, not certify the new page count, figure legibility, or final float placement. Locations below refer to the **new release’s source lines**; chapter filenames are under `latex/chapters/`.

## My verdict

### Clear conclusion: what must still be fixed?

**Yes, there are still must-fix items from the earlier feedback. Your revisions substantially improve the thesis, but I would not yet call the current wording final. Keep the completed A/B/C corrections and leave further D experiments closed. The remaining necessary work is to make the claims agree with the evidence and definitions already in the thesis.**

Before calling the text final, close these five groups, detailed in Section 4 below:

1. **Scope the abstract:** state the simulation/evaluation scope, make the alignment result conditional on the evaluated training contexts, and acknowledge that the selected alignment configurations achieve no strict task completions. Bound claims about diffusion to the inherited implementation and the tested comparisons.
2. **Remove unconditional safety claims:** the adopted tracking margins are not calibrated guarantees, and the UAV's per-axis inflation does not establish clearance for the entire rotor footprint at every surface orientation.
3. **Keep the alignment denominator honest in summaries:** ten scored contexts include one guarded, unattempted context. Remove the blanket statement that comparisons are unaffected and distinguish clean attempted episodes from the reported aggregate.
4. **Correct literal inconsistencies:** the closest unprojected median is at K=100, not K=20; the hump uses switched constraints; sampling steps and network evaluations are different when endpoint guidance is active.
5. **Narrow explanatory conclusions:** the experiments support observed differences among tasks and trained configurations. They do not isolate demonstration provenance as the cause or establish equivalence between objectives.

**Optional, not conditions for accepting this revision:** changing the chapter order, renaming all models, adding seeds or significance tests, replacing every figure, and extensive further shortening. Final PDF inspection is still required to close the original layout/reference issues; no compiled thesis PDF is available in this release directory for that check.

**The deeper pass adds necessary corrections:** reconcile the controller inputs and flight-termination rules; delete an unsupported projected-loop speedup; repair four overbroad or ambiguous summary statements against the printed tables/protocol; and specify which episodes contribute to the Steps averages. Section 6 supplies the evidence and minimal remedies. It also separates small local corrections and optional writing cuts. These findings do not require reopening the experimental programme.

**The work is a meaningful improvement. The conservative revision strategy mostly makes sense, and stopping further D work is defensible.** The strongest changes are the removal of an unsupported result, correction of mathematical descriptions, and clearer disclosure of what the evaluations actually measure. These improve scientific credibility more than a wholesale stylistic rewrite would.

The remaining problem is consistency between the revised details and the headline claims. Several qualifications now appear in Setup or Method while stronger sentences survive in the abstract, conclusions, or nearby prose. A short final wording pass would address this without changing the experiments or reopening D.

I would describe v5.7 as **substantively corrected, selectively shortened, and more transparent, but not yet fully consistent in its claims**. I would not describe it as a comprehensively redesigned or visually verified final version.

## 1. What the response gets right—and where its reasoning needs adjustment

### The sensible decisions

Separating format defects, editorial choices, factual corrections, and additional experiments was useful. Removing the seven/two alignment-success sentence rather than commissioning another analysis was a sound choice: it eliminates a claim the thesis did not substantiate. Keeping the useful existing results while stating their limitations is also a legitimate alternative to expanding the experimental programme.

The response is right that several disclosures already existed in the first release. My first review was often asking for those disclosures to reach the relevant summaries, rather than claiming they were absent everywhere. It is reasonable to preserve six contributions, the chapter order, mechanism-based terminology, and a descriptive comparison without adding statistical tests. None of those choices is inherently academically unacceptable.

The response also appropriately revised its initial shortening estimates downwards. Moving material into an appendix improves the main reading path but does not remove it from the document. That distinction is more useful than a large promised page reduction.

### “Declined” and “factually wrong” should not share a verdict

The response’s red-cross category combines two different judgments: a criticism is incorrect, or the author chooses not to implement its remedy. For example, retaining the chapter order is an editorial decision; declining intervals is a scope decision; keeping a draft appendix heading is a production decision. Those decisions do not establish that the underlying criticism was false.

This matters particularly for evidence limitations. “We will not run more seeds” is a valid stopping decision. It does not establish reliability across seeds. “We will keep the existing scoring geometry” is also valid, but it does not turn that geometry into a general whole-body clearance bound.

**Suggested interpretation of the response:** some items are fixed, some are accepted limitations, and some are optional editorial suggestions declined. No additional administrative document is needed; just avoid treating all three as scientifically settled.

### A few parts of my first review should be narrowed

- My generic suggestion to remove “proves” was not an actionable example: the response reports no such occurrence, and I am not treating it as an outstanding defect.
- I would not insist on moving Discussion before Conclusion, changing all model names, rebuilding every figure, or adding a new validity section. Existing sections can carry the necessary qualifications.
- The first review already recognized that the later dynamics definition supplied the horizon indexing. Repeating the range in the introductory definition is a clarity improvement, not a missing mathematical definition throughout the thesis.
- I am not alleging that the corridor endpoint was concealed or selected improperly. The reason and alternative goal counts are disclosed. Keeping those counts in nearby prose is an acceptable presentation choice for this limited revision.

Two substantive distinctions remain. A different network does not make the same physical task-success criterion meaningless, although it can make published numerical results unsuitable as a directly matched baseline. Also, symmetric best-rule selection is a fairer comparison than tuning only one model, but it does not create independent confirmation. The new Setup sentence now handles that second distinction well.

## 2. What I could confirm in the shipped LaTeX

| Work | Targeted finding | Assessment |
|---|---|---|
| **A: equation-reference source** | The FM loss has one label at `04_method.tex:651`, and the relevant references point to it. | The identified source defect is addressed. A compiled PDF is still needed to confirm rendered references. |
| **A: text/table contradictions** | The corrected raw-result bounds, thirteen-configuration wording, and diffusion projection-count explanation are present in the revised source. | Real corrections, not merely status updates in the answer. |
| **A: float handling** | The formerly tall figures now have explicit height limits; the corridor table caption is shorter. | Sensible source changes. Final fit and embedded figure-text fixes are not verified without the new PDF. |
| **B: contributions and result presentation** | Introduction contribution items no longer repeat their detailed outcome paragraphs. The alignment ladders and complete raw-plan grid are in the appendix; the controller results are consolidated. | Useful, restrained editing that preserves the study’s structure. |
| **B: shortening** | Several repeated passages are reduced, but §6.4 and Chapter 7 still provide multiple summaries. | Improvement, with modest compression rather than a wholesale shortening. The actual page saving is unverified. |
| **C: unsupported alignment count** | The seven/two claim and its internal audit identifier are gone from the results passage. | Correctly closed by deletion; no recount is needed to support a deleted claim. |
| **C: CI-MeanFM budget claim** | `06_results.tex:737–755` now distinguishes K=2–20 from K=100 and labels the target explanation as untested. | Material improvement. Some surrounding rhetoric remains stronger than necessary. |
| **C: experimental scope** | `05_setup.tex:456–458` discloses overlapping checkpoint-selection windows; `1005–1009` discloses uncalibrated margins and selection on the evaluated grid; `1117–1119` explains the frozen context. Training-context scope appears in Chapter 7 and Discussion. | These are important, useful additions. |
| **C: mathematics/specification** | The mismatch is described as the change in tracking error; the zero-interval target is distinguished from the loss; the keep-out equation uses horizontal coordinates; unconstrained velocity channels and solver/controller settings are specified. | The principal local corrections are implemented. Some downstream claims still need the same qualifications. |
| **D4: one reported diagnostic** | `06_results.tex:467–468` reports 0.1–1.1% non-converged endpoint solves over the six cells. | The addition is present and appropriately small. Its underlying numerical provenance cannot be independently checked from thesis material alone. |

For D4, I would make one tiny wording change: replace “The endpoint solves also converge” with **“Endpoint non-convergence was uncommon.”** The following percentage already conveys the result precisely. It does not establish the missing per-step solver comparison, and need not claim to.

## 3. Stopping D: what that decision permits

I would **not reopen D as a compulsory experimental programme** for this revision. The thesis can present a useful empirical comparison with the evidence already reported, provided its conclusions retain these boundaries:

| Work not pursued | Defensible claim with the current evidence |
|---|---|
| Further alignment counts / nine-context re-aggregation | Report the existing ten-context summaries as including the guard, state that nine contexts were attempted, and retain the zero-completion statement for the specified selected configurations. Do not extrapolate that statement to every unreported configuration. |
| Re-scoring with a larger enclosing footprint | Results concern the declared 0.31 m planning/scoring convention. They do not establish clearance of the complete rotor footprint in every direction. |
| Margin-exceedance analysis | The margins are chosen/inherited parameters; the theoretical bound remains conditional. Tracking lag is not a substitute measurement of increment-level model mismatch. |
| Held-out contexts, more seeds, or matched sampler ablations | Conclusions concern the evaluated contexts and trained configurations. Cross-task explanations remain hypotheses rather than isolated causes. |
| Longer corridor timeout or controller-only timing | Corridor traversal and the separately disclosed goal-reaching results remain distinct; timing refers to controller-and-simulator overhead in the evaluated environments. |

“A re-analysis might change numbers of record” is not, by itself, a scientific reason to reject it. The stronger justification is that the thesis deliberately retains a specified evaluation convention and narrows its claims accordingly. That is the version of the no-D decision I support.

Nor do I require p-values. A descriptive thesis can report counts and observed means. It should identify its rankings as observed results rather than imply established population equivalence or reliability. A single sentence defining that scope is enough; this does not require an uncertainty-analysis project.

## 4. The small closing pass I still recommend

### 1. Bring the abstract into line with the revised body

**Location:** `latex/pages/abstract.tex:5,20–25`.

The abstract is unchanged from the first release. It still presents the diffusion schedule statement broadly, says alignment “dominates” without naming the evaluated scope, and says simple demonstrations “make” FM sufficient. The body now contains qualifications the abstract omits.

Respecting an author-owned abstract explains why it was held, but does not resolve the reader-facing mismatch. This is the highest-value remaining edit. A bounded replacement for the alignment claim would be:

> On ten training-set alignment contexts in simulation, analytic average-velocity matching reduced final position error relative to the diffusion baseline. The selected projected configuration recorded no violations, including one context held by the evaluation guard, but did not complete alignment within the position tolerance.

Also scope the diffusion statement to the **inherited DPCC sampler**, and describe the UAV ordering as an observed result rather than a demonstrated effect of demonstration simplicity. These changes need no new data.

### 2. Remove the blanket safety statements that survive the new qualifications

**Locations:** `05_setup.tex:1002–1007`; `04_method.tex:1481–1487`; `08_discussion.tex:8–14`.

Method now admits the diagonal footprint shortfall, but Setup still says inflation makes “the vehicle, not only its centre point,” clear the surface. Discussion still says the manipulator bound holds because the end effector tracks within the margin, while Setup now says the margins were not calibrated. The corrected identity also distinguishes tracking error from its one-step change.

**Minimal fix:** state that 0.31 m is the offset used for planning and scoring, with the directional limitation described in Method. Describe the tightening guarantee as conditional on the required mismatch bound. Remove the unsupported blanket clearance and verified-bound assertions. You do not need the additional 4.8 cm sentence or a re-scoring exercise to make that distinction clear.

### 3. Keep the guard visible when summarising “all ten”

**Locations:** `06_results.tex:937–943`; `07_conclusion.tex:28–30,83–87`.

Adding the guard to Setup and captions is a substantial improvement. However, “Comparisons between rows are unaffected” still survives, and the conclusion still says “every context” or “all ten” without reminding the reader that one was never attempted.

**Minimal fix:** replace the broad comparison sentence with:

> Including the same guarded context preserves the ordering of violation-free counts; the reported distances and timings also include that context.

In the conclusion, add **“including one context held by the guard”** to the selected ten-context result. Keep the existing numbers and denominators. This addresses the interpretation without undertaking D2.

### 4. Finish three local corrections marked as closed or optional

| Current passage | Why it is still inconsistent | Small correction |
|---|---|---|
| `06_results.tex:862`: the model that “comes closest to the target does so at twenty” | The same table gives MeanFM 0.0673 m at K=100 versus 0.0741 m at K=20. Changing “solves” to “comes closest” did not fully fix Ci-6. | “The selected model is evaluated at its K=20 operating point.” |
| `08_discussion.tex:113`: “always-active … the tilt and the hump” | Method describes the hump’s two halves as acting over their own x spans under the switching rule (`04_method.tex:1476–1478`). The horizon fix introduced a remaining classification conflict. | Say “constraints already included in the optimisation” and distinguish all span-switched constraints; omit the conflicting list. |
| `06_results.tex:1904`: “twenty evaluations, endpoint projection” | Setup explicitly gives NFE = K + guiding steps. At K=20, η=0.2, that is 23 field evaluations. Declaring the convention elsewhere does not make the two counts interchangeable. | Use “K=20 sampling steps” in this sentence and other endpoint summaries. No new accounting table is necessary. |

### 5. Keep the cross-task storyline, but stop short of causal closure

**Locations:** `07_conclusion.tex:37–49,94–101`; `06_results.tex:1833–1837,1889–1896`.

Replacing “the model does not matter” with “the objectives cannot be told apart” is only a partial improvement: the second formulation can still suggest equivalence. Similarly, the revised CI-MeanFM paragraph acknowledges an untested account but then returns to “as this one does.”

You can preserve the intended storyline with a bounded sentence:

> The model ordering differed across the evaluated tasks. The simpler generated demonstrations may contribute to this pattern, but their effect was not separated from the task, controller, and sampling configuration.

For the corridor, “the observed results did not establish a clear model ranking at the shared budgets” is sufficient. For Pareto comparisons, one definition such as **“dominance refers to the reported sample means on the evaluated configurations”** avoids requiring a chapter-wide rewrite or statistical tests.

## 5. Quick assessment of writing, length, and presentation

The writing is less repetitive where the edits actually removed material: the contributions are more focused, some method scaffolding is reduced, and the main results no longer carry every diagnostic figure and ladder table. Those are worthwhile changes. I support retaining your voice rather than systematically replacing every metaphor or long method name.

The formulaic impression is reduced only partly. In particular, §6.4 still contains a recap, a projection recap, a further conclusion, three general findings, and another supported-configuration list before Chapter 7. The bold aphorism “A task saturated at one evaluation hides…” also remains at `06_results.tex:1891`. This is not proof of AI authorship; it is still a conspicuous repetitive editorial pattern.

I would not require another structural pass now. If you want one further cut, remove one of §6.4’s repeated closing layers while preserving the unique evidence. Otherwise accept the remaining length as an editorial choice. Do not claim the original 11–14-page reduction was achieved: the response itself acknowledges a much smaller change, and no new compiled PDF is available here to settle it.

For layout, the source remedies are plausible, especially the height caps and moved figures. They do not establish that captions now fit, embedded raster labels are readable, or the merged controller table is well proportioned. The final compiled-page inspection remains a production check, not a request for new experiments or another full audit. Intentional metadata and appendix placeholders can remain during drafting, but need removal for submission.

## 6. Additional findings from the requested deeper recheck

These are additional to the five closing groups in Section 4. “Must fix” here means a contradiction, unsupported numerical claim, or consequential specification gap in the thesis as written. It does **not** mean the experiments have been shown to be wrong. I distinguish corrections introduced or exposed by editing from issues that simply remain in the current text.

### A. Must reconcile: what velocity reference does the UAV controller receive?

**Locations:** `04_method.tex:446–466` versus `05_setup.tex:1128–1130`.

Method says the velocity reference is synthesised rather than supplied by the plan, then explicitly says the reported cascaded controller uses **zero velocity feed-forward**. Setup says the corridor and s-curve controller receives **“the planned position and velocity.”** These describe different controller inputs. Predicted velocity channels in a plan do not settle which reference is actually passed to the controller.

This matters because velocity feed-forward changes tracking lag—the very effect used to explain the UAV results. The added acceleration/yaw detail is useful, but this sentence has not yet achieved a consistent specification.

**Fix:** reconcile Setup with Method. If Method describes the evaluated cascaded controller correctly, use:

> The cascaded geometric controller receives the accumulated position setpoint, with zero velocity and acceleration feed-forward and a zero yaw reference.

Keep the separate MuJoCo MPC interface scoped to its own comparison. The thesis alone does not let me decide which conflicting description reflects execution.

### B. Must remove: the projected-loop speedup has no reported measurement

**Locations:** `06_results.tex:1710–1711,1726–1727,1770–1771`.

The text says the cascaded controller takes “a tenth of the whole loop unprojected, **half of it projected**.” The merged table explicitly says loop timing was measured on the **unprojected pair**; both projected loop entries are `---`. The projected MuJoCo MPC planner time is also missing, so the displayed data cannot reconstruct the projected ratio.

**Fix:** delete “half of it projected.” Keep the measured unprojected comparison: 14.0 versus 135.8 ms per executed step. The approximately 24-fold remainder comparison must retain its existing qualification—controller **and simulator** together, on that unprojected pair. No additional timing experiment is necessary.

This is a concrete problem left behind by consolidation: a table can be improved while nearby prose still claims a quantity it no longer substantiates.

### C. Must reconcile: crossing a finish line does not consistently terminate a flight

**Locations:** `05_setup.tex:1026–1029` versus `05_setup.tex:631–633,658–662` and `06_results.tex:1574–1578`.

The generic episode definition says the run ends when the scene's finish line is crossed. The UAV metric and s-curve protocol instead say that crossing is latched, the flight continues, and later contact, altitude and abort checks can affect final success. The latter distinction is consequential: a finish-line crossing alone is not the reported success criterion.

**Fix:** make termination explicitly task-specific. For example:

> On UAV-corridor and UAV-s-curve, crossing the finish line is recorded without ending the flight. The flight continues until its task-specific termination condition; success also requires the whole-flight controlled-flight checks defined in the metrics section.

Retain the stated goal-point, timeout and abort conditions where applicable. Avoid a generic sentence that overrides them.

### D. Must repair: four summaries misstate or blur their evidence

| Current claim and location | Counterevidence inside this thesis | Minimal correction |
|---|---|---|
| “Every flow-based row is cheaper than every diffusion row at equal success with constraint satisfaction” (`06_results.tex:407–408`). | FM, K=20, cumulative-cost selection: S&C 1.000 at **476.7 ms** (`06:287`). Diffusion, K=2, the same rule: S&C 1.000 at **211.4 ms** (`06:294`). | Limit the sentence to the **low-budget flow operating points**. Their advantage survives this correction; the universal claim does not. |
| The diffusion model loses success when “run away from its trained budget” (`07_conclusion.tex:58–59`). | Results explicitly says **every diffusion budget is a separately trained checkpoint** (`06_results.tex:30–34,181–194`). Thus these rows do not demonstrate degradation from changing the sampling budget of one trained diffusion model. | Say that the **separately trained one-step diffusion configuration** attains lower S&C. Preserve the inference-time budget flexibility claim for the flow checkpoints. |
| At K=20, MeanFM is “the only model that moves the box in every context” (`07_conclusion.tex:20–24`; also `06_results.tex:620–623`). | CI-MeanFM with α_end=0.05 also has **0/10 unmoved** at K=20, although its median error is much worse (`06_results.tex:657–660`). | Delete “only,” or explicitly restrict the comparison to the main α_end=0.2 CI configuration. The strong median-error result needs no exclusivity claim about motion. |
| The s-curve RQ1 sentence ends “with the fewest violating steps” after listing three models' crossing counts (`07_conclusion.tex:73–75`). Its attachment is ambiguous. | At K=1, FM has **23.0** violating steps, MeanFM **18.5**, and CI-MeanFM **23.3** (`06_results.tex:1599–1607`). If the phrase qualifies FM's leading result, it is false; if it qualifies the immediately preceding MeanFM result, name that model. MeanFM aborts 7/10 flights versus FM's 1/10, so its lower count does not establish a safer full traversal. | Retain FM's **most successful flights, 9/10**. Delete the dangling “fewest” phrase, or explicitly attribute the lower recorded count to MeanFM together with its abort count. |

These are sentence-level repairs with numerical consequences for interpretation. They are higher priority than further stylistic polishing. In particular, “every,” “only,” and “fewest” need checking against **all configurations actually printed**, including the ablation rows.

### E. Must clarify: the population behind the Steps column

**Locations:** `05_setup.tex:573–588,672–690`; `06_results.tex:60–74,1333–1364`.

The thesis calls Steps the number needed to reach the goal, discusses a cited comparison over constraint-satisfying episodes, and elsewhere describes averages over episodes. It does not state unambiguously whether each benchmark's Steps average includes all episodes, successful episodes only, or constraint-satisfying successes only, nor how failed episodes enter it.

The distinction cannot be dismissed as cosmetic. The unprojected diffusion K=2 row prints **61.7 Steps despite S&C=0.000** (`06:73`), so that row is not averaged only over constraint-satisfying successes. In the corridor table, a row with no crossings has `---`, while the MeanFM K=2 row with one crossing prints 387.0 (`06:1357–1358`), indicating a different conditioning from a simple all-flight average.

**Fix:** add one explicit definition for each benchmark: which episodes contribute, whether timeout/abort lengths are included, and what `---` means. Make clear whether the avoiding raw and projected tables use the same population. This specifies the existing evaluation; it does not request another metric or experiment. I cannot infer the correct implementation from the thesis alone, and I am not asserting that its numbers were computed incorrectly.

### F. Small local corrections to include in the same edit pass

These have lower impact than A–E, but their remedies are short.

| Location | Issue | Fix |
|---|---|---|
| `06_results.tex:331–335`; `07_conclusion.tex:13–16` | Non-dominance is described using strictly better values on **both** costs; the conclusion glosses dominance as “ahead on every reported axis” although S&C ties. Also, the frontier admits a 0.05 S&C band, so it is not literally an equal-success comparison. | Define dominance as **no worse on either cost and strictly better on at least one**, within the stated eligible set. For the three-metric headline, say “equal S&C, fewer control steps and less planning time.” This is a definition correction; I have not established that the plotted frontier is numerically wrong. |
| `05_setup.tex:1153` versus `04_method.tex:1088,1295–1297` | Setup simplifies the guiding-step count to ceil(ηK)−1. Method permits η=0 and correctly retains a terminal projection; the simplified expression would give **−1** guiding steps there. | Use **max(ceil(ηK), 1)−1**, or explicitly restrict the shortened expression to η>0. None of the positive-threshold examples changes. |
| `06_results.tex:83–86` | “None of them arrives legally” is stronger than a table with nonzero S&C, up to 0.133. | “None achieves consistently constraint-satisfying arrival.” |
| `06_results.tex:1712–1713` | After merging the controller tables, “random aborts least and violates least of those” has an ambiguous denominator. Random has 71.3 violating steps, while the preceding prose gives a best count of 37.2 across rules (`06:1657–1661`). The old caption explicitly limited the violation comparison to **rules that abort least** (old `06_results.tex:1767–1768`). | Restore that qualification: “Among the rules with the fewest aborts, random selection has the fewest violating steps.” Do not imply it minimises violations across all three rules. |

### G. Further writing and presentation feedback

**Two concrete cuts would help more than replacing individual words:**

1. **Raw-plan explanation, `06_results.tex:106–126`.** The image-generation analogy appears twice, and the text repeatedly tells the reader that this is the model's output rather than its executed path. Much of the plotting protocol is also in the caption. Keep the necessary distinction once; for example:

   > The figure plots the predicted measured-position channels of four eight-waypoint candidate plans at every fourth control step. These are generated plans, not executed paths: only the selected candidate's first action is applied before replanning.

   Retain any remaining unique protocol detail in the caption. The repeated analogy and reader coaching can go.

2. **Controller closing paragraphs, `06_results.tex:1765–1787`.** The paragraph starting “What the comparison shows is that the controller matters” is immediately followed by a Conclusion starting “UAV-s-curve shows that in this framework the controller matters.” Both repeat the same success counts, violation comparison and cost ordering. Delete the first closing paragraph and keep one corrected conclusion, with the measured timing scope from B above.

These are concrete examples of the remaining **formulaic, AI-like editorial pattern**: repeated announcements of the lesson, categorical summaries, and adjacent restatements of the same finding. They are not evidence of who wrote the text. The fix is to remove repetition and retain the qualified result, not to substitute more ornate synonyms.

**Tables:** the corridor tables still print success as `1.000 ± 0.000`, `0.100 ± 0.316`, etc. (`06_results.tex:1274–1299,1357–1364`), although their sample size is ten flights and Setup provides a count convention (`05_setup.tex:688–690`). Printing **10/10, 1/10, 2/10** would make the outcome immediately readable and save width; retain mean ± SD for continuous quantities. This is a presentation improvement using already stated counts.

The merged controller table has eight columns and two text columns (`06_results.tex:1717–1727`). Inspect that specific table at normal PDF viewing size. If space is tight, group rows by Projection and define shorter controller names in the caption before reducing the font. For the moved s-curve plan figure, the 0.70-text-height cap (`09_appendix.tex:339`) limits its size on the page but does not establish that embedded labels are readable. Without the compiled PDF, I cannot honestly mark either item visually passed or failed.

**Positive check:** a source scan of 631 literal reference targets across the current LaTeX found **no undefined literal references and no duplicate literal labels**. This supports the equation-reference repair and the relocation of figures/tables. It does not replace compilation or inspect citations, page breaks, or image text.

**Completion criterion for this round:** close the factual and specification corrections in Sections 4 and 6A–E, make the short local repairs in 6F, and inspect the compiled thesis. The cuts in 6G remain optional. The existing results can support a defensible descriptive thesis without undertaking D; the remaining work is to state that evidence consistently.
