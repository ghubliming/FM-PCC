# CHANGELOG — `Working_Space/v3`

Every change to the v3 draft, its figure pipeline or its inheritance machinery gets an entry here,
newest first. Format follows [`../v2/CHANGELOG.md`](../v2/CHANGELOG.md): what changed · why · what it
is sourced from · what it left open.

**Rules for this file**

- One entry per working pass, not per edit.
- **Every number injected into the draft names its evidence of record** — a batch directory and the
  DA that published it, or a repo file with line numbers. "From a summary table" is not a source:
  the readiness ledger is a *pointer*, and what gets cited is what it points at.
- Mechanical checks (`python3 tools/check.py`) are re-run after every pass and their result is
  recorded. **There is no TeX toolchain in this container, so "checked" never means "compiled".**
- A pass that absorbs a v2 change records which v2 revision it absorbed, from
  `inherited/SYNC_STATE.json`.
- **Every individual changelog under `changelogs/` is signed at the end** --- who wrote it, on what
  date, and that it was not compiled. A pass written by an AI agent says so by name and model, so that
  a later reader knows which entries to re-check against the data rather than trust. (Author's
  instruction, 2026-09-21.)

---

## v3.64 — 2026-09-22 · §6.3 rebuilt in the order of §6.1/§6.2 (before projection → after projection → the two projectors); the alignment threshold is compressed because the budget is high → [`changelogs/v3.64_20260922_uav_prepost_structure_threshold_compressed.md`](changelogs/v3.64_20260922_uav_prepost_structure_threshold_compressed.md)

- **§6.2.2:** the threshold paragraph is re-headed and re-led — the DPCC default $\eta=0.5$ is set for small budgets and is not runnable at $\nfe=20$/$100$ (10 / 50 solves per candidate per step), so the threshold is *compressed* to a roughly constant number of guiding steps; `tab:va-threshold` is the ladder it was chosen on. No number changed.
- 🔴 **§6.3 is three parts now:** 6.3.1 the models *before* projection (**new** `tab:uav-corridor-raw`, `tab:uav-pillars-raw`), 6.3.2 the same cells *after* per-step projection (`tab:uav-corridor` projected-only again, `tab:uav-pillars`, best table, frontier, paths; s-curve kept whole), 6.3.3 per-step vs endpoint (`tab:uav-pillars-endpoint` moved beside `tab:uav-corridor-projection`). Labels Chapters 5 and 8 cite stay where the meaning is. No run, no cell changed.
- `check.py`: all mechanical checks pass, 6 927 lines. **No bundle built.**

## v3.63 — 2026-09-22 · UAV-pillars falls back to `pillars_hg`; the enlarged campaign (U17) failed and is abandoned → [`changelogs/v3.63_20260922_pillars_fallback_hg_u17_abandoned.md`](changelogs/v3.63_20260922_pillars_fallback_hg_u17_abandoned.md)

- **Info box acted on:** the DA note of 2026-09-22 — `pillars_xl` is all zero in 94 cells, every projector routes into the forbidden centre lane; author decision: fall back. INBOX row ✅.
- 🔴 **The withheld `pillars_hg` section is back in §6.3** (prose, three tables, flown-path figure), with the v3.62 *success* wording and its caveat first; the v3.56 `pillars_xl` templates, `\hole`s and `\todofigure` are gone (`\todofigure` = 0). §6.3 intro, frontier paragraph, projection paragraph (six aggregates re-verified) and UAV conclusion read `pillars_hg` again.
- **Chapter 5 back to the 0.12 m pillar** (`tab:uav-scenes`, srcnote, two captions, protocol row), and **§5.3.3 tells the attempt**: no demonstrated route satisfied the enlarged set; in evaluation the flights *succeeded* while violating at every column, nothing was collision-free, the scene ranked nothing.
- Figures: pillars drawn at 0.12 m again; `expert_paths.json` re-extracted (4/4 clean); `fig_constraints_uav`, `fig_expert_uav`, `fig_uav_pillars_paths` rebuilt and exported.
- 🔒 **§6.4 untouched (author lock):** its three summary tables still lack a pillars row; the rows are in the changelog, ready for when it is unlocked.
- `check.py`: all mechanical checks pass, 6 761 lines, 40 figures.
- **Second round, same entry:** Fig 6.7 says *after projection* on the page and draws Fig 6.6's unprojected MeanFM cells as grey rings; the **threshold study** (§6.2.2 prose) gets `tab:va-threshold`; the pillars `\hole` is two *pending* K=1 rows → **R32**; 🔴 **Table 6.10's metric double-checked**: S&C reproduces; under the evaluator's strict goal-reached criterion MeanFM leads (0.90 vs FM 0.40) because six FM flights cross and never settle — both criteria now printed, strict one defined in Ch 5. **No bundle for the second round.**
- **Third round, same entry:** appendix Tables kept, their duplicated captions cut to what the copy adds; **every row marked ▶ or • is bold on every cell** (17 rows in Ch 6, 6 in the appendix; two pillars diffusion rows get the dot); **Table 6.5 keeps diffusion at K=20 only** (K=100 row and three pending rows removed, Fig 6.6 follows, R16 narrowed to six flow-model evaluations); ledger §0 gets a running addendum, nothing deleted.
- **Bundles (closing v3.63):** `bundle/output/thesis_v3_20260922_105444_new.zip`, `…105445_new_clean.zip`, `…105445_new_nonotes.zip` — 44/44 figures render, cross-references resolve, `--verify` byte-faithful; supersede the `101932/101933` set of the first round.

## v3.62 — 2026-09-22 · Fig 6.3 on a broken step axis, Table 6.5 as the whole ladder with *pending* cells, alignment frontiers as the share closed, Fig 6.8 with nine contexts → [`changelogs/v3.62_20260922_broken_axis_full_ladder_closed_share_nine_paths.md`](changelogs/v3.62_20260922_broken_axis_full_ladder_closed_share_nine_paths.md)

- **Read with v3.61 and v3.60**; four author items on the v3.61 result.
- **Fig 6.3:** log axis tried and rejected (range under ×2, nothing moved). The step axis is **broken at 80**, the solved band takes 74 % of the height; the *better* key moves to the legend strip.
- 🔴 **Table 6.5 prints the ladder $\{1,2,10,20,100\}$** for all four models, *pending* in 11 of 20 cells; the v3.61 appendix copy is removed. **R16 widened** (ledger §20): 8 flow evaluations + 3 diffusion trainings.
- **Figs 6.6, 6.7:** axis is now the **share of the distance closed, linear, 100 % = delivered**, the table's bracketed number. 🔴 Fig 6.6 had been **missing the CI-MeanFM and FM K=100 rows** of its own table since v3.60; both drawn now, frontier unchanged.
- **Fig 6.8:** all nine moving contexts drawn; the middle panel's one violating path is now in view. The unmoved context (7) is left out, and the caption says why.
- **Second round, same entry:** Fig 6.9's success axis is **log** in flights of twelve with a floor for zero, and its frontier no longer rings zero-success or 0.1 ms "wins"; **Table 6.8 prints every cell before and after projection**; 🔴 the quadrotor metric is **success** everywhere (defined once in Ch 5; 47 "passed" occurrences replaced in Ch 6, the appendix and the figure legends); **Fig 6.12 marks crashed or aborted flights with a cross**, from flags the extract now records (`uav_paths.json` regenerated, otherwise identical).
- Source-safety check on the author's note: v3.61 is in HEAD (`999152f1`), the working-tree diff is exactly this changelog's edits, all draft files are LF in index and on disk; `core.autocrlf = true` is the only thing that could rewrite them (author's call).
- `check.py`: all mechanical checks pass, 6 611 lines. v2.25 not absorbed.
- **Bundles** (closing the update, on the author's word): `bundle/output/thesis_v3_20260922_094334_new.zip`, `…094335_new_clean.zip`, `…094335_new_nonotes.zip` — 43/43 figures render, cross-references resolve, `--verify` byte-faithful; supersede the v3.61 `085914/085915` set.

## v3.61 — 2026-09-22 · concise tables restored (full ones in the appendix), alignment frontiers on a log-percent axis, UAV-s-curve as a matched grid → [`changelogs/v3.61_20260922_tables_concise_frontiers_percent_scurve_grid.md`](changelogs/v3.61_20260922_tables_concise_frontiers_percent_scurve_grid.md)

- **Read with v3.60**; this pass amends its §1, §12 and §27 on the author's review.
- ✅ **v2.25's note was right:** planner and scorer are both offset by $0.31$ m; every evaluated scene sets `planning_inflation: {…, margin_base: 0.0}` (`uav_projection.yaml:360/376/438/513`, read at `eval_mix_uav.py:924/1017/1279`). `tab:eval` and its `\srcnote` corrected; INBOX row closed; v3's own note to v2 carries a correction banner.
- Fig 5.1 caption: the appendix pointer now says what is where.
- 🔴 **Tables 6.1, 6.2, 6.5 are the concise pre-v3.60 tables again** (restored from `git show HEAD`); the K=5/K=10 and K=2/K=10 rows live in `tab:app:avoiding-raw-full`, `tab:app:avoiding-dpcc-full`, `tab:app:va-models-full`. Rule: a main table maximises the budgets all models share; the appendix carries every evaluated budget.
- Fig 6.6 has its *better* key back; **Figs 6.6 and 6.7 use the final distance as a percentage of the start on a log axis**, not metres; Fig 6.7's legend and caption say why its cheapest point is hollow.
- 🔴 **UAV-s-curve is a matched grid with the missing cells printed as *pending*** — unprojected at $\{1,2,3,5\}$ (7 of 13 cells pending), projected at $\{3,5\}$ in the corridor's shape (5 of 6 cells pending); the ragged per-model records moved to `app:uav-scurve-budgets`. Ledger **R31** (§18) lists the cells; R29 superseded.
- `check.py`: all mechanical checks pass, 6 606 lines. v2.25 not absorbed.
- **Bundles** (built after the six items, on the author's word): `bundle/output/thesis_v3_20260922_085914_new.zip`, `…085915_new_clean.zip`, `…085915_new_nonotes.zip` — 43/43 figures render, cross-references resolve, `--verify` byte-faithful; supersede the v3.60 `205646/205651` set.

> 📄 **Start here for v3.60:** [`BRIEFING_20260922_v3.60.md`](BRIEFING_20260922_v3.60.md) --- the
> questions answered, the three things worth knowing, the four decisions waiting on the author, and the
> writing rules. Five minutes. This entry below is the full record.

## v3.60 — 2026-09-21 · Chapter 5: the two quadrotor margins, the twenty-episode row moves out, and the D3IL policy sizes → [`changelogs/v3.60_20260921_ch5_margins_success_and_params.md`](changelogs/v3.60_20260921_ch5_margins_success_and_params.md)

- 🔴 **`tab:eval` was merging two different knobs.** `Tightening … $0.025 + 0.31$` is now two rows:
  **tightening** $0.025$\,m (`uav_projection.yaml:152`, DPCC's value, only for tightened variants) and
  **body inflation** $0.31 + 0.02$\,m (`:175`, always on, applied before the projector sees a surface) ---
  and the $0.02$ planning pad, absent from the draft entirely, is now printed. The configuration says in
  its own words at `:876` that the two must stay distinguishable.
- **$0.31$ is not a radius**, it is the largest per-axis **rotor reach** at zero yaw: rotors at
  $(\pm 0.14, \pm 0.18)$\,m with radius $0.13$ give $0.31$ across and $0.27$ along
  (`quadrotor_modified.xml:43--46,17`; the same constant as `trajectories.py:44`). Renamed at all five
  Chapter 5 and appendix sites. 🆕 A verification `\srcnote` also records what the draft had never said:
  the planner is offset by $0.31+0.02$ ($+\,0.025$ tightened) while the execution scorer uses $0.31$ alone.
- 🔴 **A violation is a property of the path, not of the airframe.** `fig:expert-uav` drew the vehicle to
  scale *and* coloured it by its own overlap, over surfaces that already carry the reach --- the test drawn
  twice. The silhouette is now neutral and named as a scale reference; only the path line is green or red.
  New paragraph before the figure states the rule for every quadrotor result.
- **`tab:target` loses its twenty-episode row** (author: *"this is suspicous results"*), moved intact to
  the appendix as 🆕 `tab:target-twenty`, daggered and captioned as not resting on the coverage the other
  rows do. 🔴 Its `\dataref` had named the *twenty-episode* corpus for a row computed from the 19-09 one;
  corrected, and cross-checked against `tab:avoiding-dpcc-protocol`'s baseline of record.
- **§5.5.2:** the *"ResNet encoder of Diffusion Policy"* attribution is out --- D3IL instantiates
  robomimic's `VisualCore`/`ResNet18Conv` --- and *"obstacle avoidance from the state"* carries
  **(D3IL-avoiding)**.
- ✅ **"well under a million parameters" checked and it holds.** D3IL publishes no counts, so the heads were
  computed from the configs and `act_vae.py`: **0.25\,M** (BESO, DDPM-GPT, 4 layers at $d=72$),
  **0.35\,M** (DDPM-ACT) and **0.47\,M** (VAE-ACT) --- none reaches half a million, against $\approx 22$\,M
  of unshared perception. The old wording also mis-read as *two* layers for one model and *four* for the
  other; both are encoder 2 + decoder 4.
- **What D3IL scores as `success` is now stated exactly** --- box centre within $0.018$\,m **and**
  orientation within $0.048\pi \approx 8.6^{\circ}$, both at once, as a $0/1$ count --- together with the
  fact that **this thesis never uses it**: alignment is scored by final planar distance, never thresholded,
  with no orientation term.
- 🔴 **Bundle gap closed.** v3.59's changelog and `BUNDLE_LOG.md` name final bundles `…_192520_*` that are
  **not on disk**; the newest were `155342/155343`, built before the Figure 5.2 hotfix and missing
  `fig_render_avoiding_start.png`, which `05_setup.tex:174` includes.

**Continued, same pass --- Chapter 6 §6.1.**

- 🔴 **The §6.1 introduction said the twenty-episode evaluation was reported in this section.** It is in
  the appendix. 🆕 **§6.1.2.8 is removed** --- the last main-text holding of that evidence --- along with
  the conclusion's *"repeats the comparison at twenty episodes … does not change its ordering"* and
  §6.1.2.1's *"A second evaluation leaves that unchanged."* §6.1 is now at the protocol of \ac{DPCC} from
  first sentence to last, and a source comment warns against reintroducing a twenty-episode result there.
- 🟠 **The flagged *"$553.4$\,ms at twenty"* is the step budget, not the episode count** --- twenty
  denoising steps --- so the number stands. The *phrasing* was the fault, in a chapter where "twenty
  episodes" is a live idea: now **"at its own twenty denoising steps"**. Every other "twenty" in §6.1.1.1
  was checked; all are step-budget statements, all correct.
- 🔴 **Figure 6.1 was described as something it is not.** The text said the plans *"form a narrow band
  around the executed path"* --- **the executed path is not in the panel**; the crop is the diagnostics'
  last column, which draws plans and constraints only. From `scripts/eval.py`: the fan is drawn at
  **every fourth** control step (`plot_samples_every = H/2 = 4`, `:279`) although **every** step is
  stored (`:278`), with **four** candidates (`:411`, fan default 4 at `avoiding-d3il.py:71`), each an
  **eight-waypoint** curve of the **measured planar position** channels (`:413`,
  `projection_eval.yaml:20`), green at **waypoint 0**. Two new paragraphs and a rewritten caption say so,
  and then the receding horizon: of the eight waypoints **exactly one** is carried out (`:348--356`), so
  the plan is never executed as drawn --- *which is what makes it worth drawing*, being the network's
  output before the projector repairs it and the controller smooths it, **the analogue of sample fidelity
  for an image generative model**. 🆕 A `\srcnote` carries every line number.
- **Two removals in §6.1.2.1.** The stray *"The diffusion result at $\nfe=10$ is in Table B.1"*, which
  nothing in the passage needed; and 🟡 **the whole resolution paragraph** --- it restated the cell
  composition already printed twice, claimed a $1/30$ granularity **the printed number does not have**
  (the cells are a mean over seeds of per-geometry means), leaned on the quarantined twenty-episode
  evidence, and closed with an unnamed *"separate evaluation batch … gives the same ordering"*. The
  yellow-alert pattern: arguing with an objection nobody raised.

**Continued again --- Chapter 6 §6.1, second round.**

- 🔴 **The K=5 / K=10 census.** Rule applied: a budget enters the table only if it carries the full
  protocol. **MeanFM at 5 and 10 and diffusion at 10 are full** (5 seeds × 3 geometries × 2 episodes) and
  are now **nine new rows** across Tables 6.1 and 6.2 --- diffusion 10 promoted out of the appendix.
  **FM at 5 is seed 6 only, and FM at 10 does not exist at $T{=}0.5$**, so neither is claimed. 🟠 The
  trap: FM *does* have a full five-seed $\nfe=5$ result, but only in the quarantined **twenty-episode**
  campaign. Not used. The figures already drew these points, so the tables and figures now rest on the
  same cells and the excuse sentence in §6.1.2.3 is deleted. New gap **R28** in the ledger.
- 🆕 **A finding from the promotion:** the diffusion baseline at **ten** denoising steps reaches $1.000$
  S\&C in $68.7$ steps at $321.7$\,ms --- same satisfaction, fewer steps, $58\,\%$ of the time of its own
  twenty-step configuration. It earns bold by the table's own rule. The baseline of record is unchanged
  (it is what \ac{DPCC} publishes), but the flow-based advantage is now stated as $30\times$ against the
  published configuration and $18\times$ against the cheapest diffusion one measured here.
- **Figures 6.5 and Table 6.3 are multi-seed, not seed-6-only** --- censused because the author asked.
  Figure 6.5 is **4 seeds (7--10)**, both methods matched; 🟠 seed 6 is the one *absent*, despite the run
  tag reading `_s6`. Table 6.3 is **5 seeds (6--10) × 3 geometries × 2 episodes = 30 episodes**; 🔴 its
  caption said *"15 episodes per cell"*, but 15 is the **cell** count. Both corrected. ✅ All eight
  "ahead of N of 3" counts re-derived and correct; two cells gained the missing *"level with"*.
  🆕 `tab:app:hf-ladder-detail` publishes the numbers behind them, with time deliberately omitted
  (one candidate against four).
- **§6.1.2.7 moved to §6.1.2.2**, beside the table whose rule column it explains. 🔴 Found while moving:
  its numbers ($98.0$/$72.0$ steps, $0.927$/$0.943$) were **twenty-episode** figures --- the last of that
  evidence in main text. Replaced with the protocol's own ($72.4$/$97.2$ against $58.6$/$59.4$, $0.933$
  against $0.967$).
- **§6.1.2.8** confirmed already removed in the first round. **§6.2 has no threshold study** --- it uses
  $0.2$, $0.4$ and $0.5$, and the only $0.05$ there is $\alpha_{\mathrm{end}}$, a different knob;
  question returned to the author rather than answered by invention.

**Continued again --- Chapter 6 §6.2.**

- **§6.2.1.2 renamed.** *"Consistency-Interpolated Average-Velocity Matching"* named a model where every
  other subsubsection names what is studied. Now **Consistency Interpolation across the Step Budget**.
- 🆕 **Two alignment cells were on disk and unpublished.** CI-MeanFM at $\nfe=100$ ($0.1791$\,m,
  $902.2$\,ms) and FM at $\nfe=100$ ($0.4247$\,m, $1425.0$\,ms) are now in Table 6.5, making $\nfe=100$
  the second budget at which **all four models** are measured. §6.2.1.1 reads the pair: same ordering at
  $20$ and $100$; a hundred evaluations more than halve the median for diffusion and for the
  consistency-interpolated model, and buy instantaneous-velocity matching nothing. $\nfe=10$ stays
  analytic-only (R16). **Every alignment result in this thesis is training seed 6** --- now said plainly
  in the ledger.
- 🆕 **The threshold discussion is measured, not argued.** Analytic average-velocity matching at
  $\nfe=100$ was run at **two activation thresholds**: $\eta=0.5$ costs $15{,}218 \pm 3{,}885$\,ms per
  control step, $\eta=0.1$ costs $1{,}195 \pm 180$ --- **a factor of $12.7$ for $6$\,mm of final
  distance**. A new paragraph opens §6.2.2 with this, and says what a lower threshold gives up. The
  argument previously rested on arithmetic alone.
- **The endpoint-projection merge: censused, and blocked by one gap.** All three flow models carry both
  projectors on the tightened set; 🔴 **the diffusion baseline carries neither** (R2), so the four-model
  pre/post-projection comparison the author described cannot be built. `tab:va-projection-models` already
  *is* the merged table for what exists; §6.2.2.1 now states the limit with a `\guard`. 🟠 Left for the
  author: `tab:va-projection` is a strict subset of it and could be deleted, but that rewrites §6.2.2's
  narrative, so it was not done unasked.
- Ledger gains §14: the full alignment census (which model × budget exists, projected and unprojected),
  R2 sharpened, R16 confirmed, threshold measurement recorded.

**Continued again --- Chapter 6 §6.3 (quadrotor).**

- **Brake-to-rest has its reason now**: the tracker wants position, velocity and acceleration feedforward,
  the demonstrations had all three analytically, and a planner emits one position increment. Dividing it by
  the planning interval feeds latency jitter into the commanded speed; a constant magnitude discards the
  length the plan asked for. Zero feedforward makes the tracker a pure position regulator.
- 🔴 **Table 6.8 is projected and now says so** in its first line and its caption; the baseline's
  unprojected numbers ($12/12$ crossed, $49.8$ violating steps) are given beside it.
- 🔴 **The diffusion $0/12$ is correct; Figure 6.10 was the defect.** `success_relaxed = crossed_line and
  safe`, and the finish line is the plane through the expert endpoint **normal to the route's final
  approach heading** --- the routes turn downward on exit, the baseline holds its lane, runs the full
  396-step limit and ends $0.51$\,m short, *while reaching the same $x$ as the passing flights*. The dash
  encoding was present in the SVG but **twelve near-coincident flights filled in each other's gaps**, so
  the bundle rendered solid. Not-passed flights are now dotted on a thinner stroke with a much larger
  hollow end mark; re-rendered and inspected. No path or count changed. 🟠 Also found: the diffusion row
  uses `dpcc-t-bounds_free-pdes-tightened` where the flow rows use `dpcc-t-tightened` --- disclosed in a
  `\guard`, logged **R30**.
- **Why no cheaper diffusion on the corridor**: its step count is fixed at training, so each budget is a
  training run, and at the budget it was trained for it already fails the goal on every flight.
- ✅ **Why the corridor ladder is $\{1,3,5\}$ --- the author's hypothesis was right.** At $\eta=0.5$,
  $\nfe=1$ and $2$ leave only the terminal step, so endpoint projection is degenerate; **$\nfe=3$ is the
  smallest budget at which it has a guiding step**. 🟠 This makes UAV-pillars' planned $\{1,2,5\}$
  inconsistent --- recommendation in the ledger to move R23 to $\{1,3,5\}$.
- 🔴 **S-curve budgets were ragged.** `tab:uav-scurve` ran three models at no shared budget; the census
  found **`af|K2` on disk and unpublished**, so it is now **matched at $\nfe=2$** (FM $7/10$, CI-MeanFM
  $1/10$, MeanFM $0/10$, baseline $0/10$ at twenty), with the odd budgets moved to a second block labelled
  *not a comparison*. `tab:uav-scurve-aborts` **cannot** be matched --- one budget per model --- and now
  says outright that it is not a model comparison. New items **R29**, **R30**.

**Closing the pass --- the lock, the ledger, the bundles.**

- 🔒 **§6.4 *Comparison across Environments* is locked** with a banner comment: it restates §6.1--§6.3, so
  it moves last. If a cell there disagrees with its source section, the source section is what to check.
  Nothing in §6.4 was touched.
- **The ledger is now `PENDING_20260922_all_lacking_runs.md`** --- `git mv`, not a rewrite. Nine live
  pointers updated (six in the draft, plus both INDEXes, the runbook, the 09-16 ledger, the cross-draft
  note, the plotting request and the fetch ledger). Changelogs v3.44--v3.59 deliberately keep the old name;
  they are a record of what was true then, and the new header states the rename.
- 🆕 **A new §0, *State of play, 2026-09-22*,** heads the ledger: the five items the v3.60 census **closed
  without a run**, the open list by priority, and what is **struck but not deleted**. The standing rule is
  now written down --- nothing is deleted; a retired item keeps its reason and the version that retired it.
- 🆕 **§16 specifies UAV-pillars in this file**, as asked, superseding the older pillars PENDING/RUNBOOK
  for thesis purposes. 🔴 **Ladder corrected $\{1,2,5\} \rightarrow \{1,3,5\}$**: at $\eta=0.5$ a budget
  of 1 or 2 leaves only the terminal sampling step, so `tab:uav-pillars-projection` could never be filled;
  $\nfe=3$ is the smallest budget at which endpoint projection does its own arithmetic. Same cost, three
  budgets either way. With the run matrix, what it fills, and three things that must not happen.
  🟠 The `\hole` at `sec:res:uav:pillars` still reads $\{1,2,5\}$ --- a commissioning decision, flagged
  rather than silently changed.
- 🆕 **§17** carries **R28**, **R29** and **R30**, each marked owed or not owed, and states that **R23 is
  the only open item that turns a visible `\hole` and a `\todofigure` into a result**.
- `check.py` 6354 lines, 265 labels, 39 figures, all mechanical checks pass. Final v3-only bundles
  `205646_new`, `205651_new_clean`, `205651_new_nonotes` (8649 KB); all cross-references resolve, 43/43
  figures renderable, and `--verify` reports 9 inlined + 4 collapsed sources byte-faithful.
  **Not compiled.**

## v3.59 — 2026-09-21 · Platform figures, timing scope and result markers → [`changelogs/v3.59_20260921_platforms_timing_and_markers.md`](changelogs/v3.59_20260921_platforms_timing_and_markers.md)

- Figure 5.1 again contains only the Panda and X2 renders; the X2 dimensions/footprint now live in the Reproducibility appendix, with a v4 keep-it-there note. Figure 5.2 pairs authentic MuJoCo start and goal-passed views; the 96-demonstration trajectory panel is gone. Both low-resolution stills are tracked as D12 for future print-resolution cluster renders. Gen15's $0.03$ s UAV planner interval and per-physics-step controller are verified and distinguished from the manipulator's IK route. The corridor's $14^\circ$ halfspace is verified in the $x$--$y$ plane; s-curve and scene-figure wording now describe controller stress rather than rank difficulty.
- Removed the Chapter 5 significance-test metacommentary. Table 5.8 now prints plan and state tensor shapes, with exact RGB and visual-latent shapes in its caption. §6.1.2.8 gives the measured result without a defensive coverage sentence; its incomplete K20 census remains in the appendix and source comment.
- Figure 6.3 is explicitly a **planner-time** frontier, not full executed-step time. Matched full-step timing for a total-time frontier is missing and tracked as D11; the existing UAV controller timing remains in Table 6.15. Table 6.7 has the selected-row triangle and an explicitly unevaluated diffusion-baseline dot. Checks and bundles are in the individual entry. Not compiled.
- §5.4.4 is now *Aggregation and Variability*. Figure 6.8 shows the same selected protocol episode across its four cells, with direct START/END labels and the overlaid second paths removed. The 2/2 outcomes per cell remain stated separately. The final v3-only bundles and checks are recorded in the individual entry.

## v3.58 — 2026-09-21 · Chapter 5 protocol, tensor layouts and sampling laws → [`changelogs/v3.58_20260921_ch5_protocol_shapes_sampling.md`](changelogs/v3.58_20260921_ch5_protocol_shapes_sampling.md)

- Kept the real, route-balanced twelve-flight UAV-corridor protocol; moved the twenty-episode avoiding protocol out of Table 5.10 and linked its appendix section from the prose. Figure 5.6 now marks its lower panels as $x$--$y$ views with $z$ omitted.
- Table 5.8 now gives each environment's input/output tensor shapes and corrects UAV plan width to 12. Table 5.9 identifies 2D versus 3D projected positions. The appendix defines all three training-time sampling laws, including FM's reflected Beta draw; the earlier v2 handover with the opposite sign is superseded.
- Replaced three Chapter 4 equation references that appear as `??` in v3-only PDFs with surviving section references. v2 and v4 handover notes record the notation correction and preservation of the appendix mathematics. Structural checks pass; not compiled.

## v3.57 — 2026-09-21 · Figure consolidation, missing-budget cells, and controller timing → [`changelogs/v3.57_20260921_figures_budgets_controller_time.md`](changelogs/v3.57_20260921_figures_budgets_controller_time.md)

- Figure 5.2's X2 dimension drawing is now the right panel of Figure 5.1, beside the Panda and X2 renders; the separate float and its references are removed.
- Tables 6.1–6.2 add measured FM $\nfe=20$ at five seeds and three geometries; diffusion $\nfe=2$ is explicitly pending at that protocol. Table 6.5 now shows measured MeanFM $\nfe=10$, identifies the three absent $\nfe=10$ cells, and includes the measured MeanFM/diffusion $\nfe=100$ tail. R26 and R16 are reopened in the pending-run ledger.
- Figure 6.6 includes both $\nfe=100$ points; the old direction key was removed because it covered them. Figure 6.8 shows two representative $x$--$y$ paths per panel, with $z$ omission and full-count scope explicit in the caption. Figure 6.12 uses green for passed and red for not passed.
- Table 6.15 now prints the recorded full executed-step time beside controller outcomes: 94.9 and 215.6 ms/step unprojected; 2660.1 ms/step on the projected random-selection MJPC arm. The unmatched projected cascaded-controller time remains blank. The existing controller-cost table retains its planner/controller decomposition.
- Figures rebuilt in the DA store and exported to v3; structural check and v3-only bundles recorded in the individual changelog. Not compiled.

## v3.56 — 2026-09-21 · Alignment Pareto alternative, pillars scaffold, and Chapter 5–6 hotfix → [`changelogs/v3.56_20260921_alignment_pareto_pillars_scaffold.md`](changelogs/v3.56_20260921_alignment_pareto_pillars_scaffold.md)

- §6.1.1 now states the actual unprojected goal-arrival counts: the three flow-based models reach 30/30 at both budgets; diffusion reaches 28/30 and 29/30, so “all four at 100%” is not supported.
- Figure 6.7 now shows a projected alignment outcome–cost Pareto alternative. The former per-context Figure 6.7 asset is retained in the DA store; all points use the same ten contexts and tightened constraint set, with violation-free coverage shown explicitly.
- UAV-corridor calls its test-time constraint a **sloping halfspace** and its marked comparison the **diffusion model of DPCC**. The shared naming table follows.
- UAV-pillars has a live section between corridor and s-curve, two pending table templates, and a planned executed-path figure. R23 is a visible gap again; no old-geometry result was transferred.
- Figure 5.3 now uses a MuJoCo frame with the Panda beyond the goal line and a factual caption. §6.1.1 puts raw Table 6.1 and its discussion before the plan montage. The defensive step-budget paragraph is removed. §6.1.2.8 keeps a short twenty-episode result; its four detailed tables are in the appendix, with a v4 handover note. Figure 6.7's MeanFM/CI-MeanFM points are dark blue/plum rather than adjacent blues.
- Structural check passes; v2.24 inheritance unchanged. The final v3-only bundles are `104909_new`, `104912_new_clean`, `104916_new_nonotes`; the last verifies byte-faithful. The intentional pillars placeholder remains visible. Not compiled.

## v3.55 — 2026-09-21 · §6.1 is reported at DPCC's protocol; the twenty-episode evaluation is quarantined → [`changelogs/v3.55_20260921_ch6_dpcc_protocol.md`](changelogs/v3.55_20260921_ch6_dpcc_protocol.md)

- 🔴 **Correction of a misread.** *"20 vs 2"* meant the **episode count**, not the step budget. §6.1 is
  now reported end to end at **DPCC's own protocol** (5 seeds × 2 episodes per geometry) and everything
  resting on twenty episodes is one subsection at the end, before the conclusion — which is what v3.52's
  own census argued for: every $\nfe=20$ twenty-episode cell is short of seeds or geometries, every
  $\nfe\in\{1,2\}$ cell is complete.
- **New order in §6.1.2:** the executed-path figure becomes its own subsubsection after *At the Protocol
  of DPCC*; *The Cost Frontier* and *Step Budget* are rebuilt at two episodes; the old *Extended
  Evaluation* moves to the end as 🆕 **The Twenty-Episode Evaluation**, absorbing the budget comparison
  as its closing paragraph.
- **The frontier and the ladder now read the corpus the TABLES are computed from** (the committed
  `19-09-…/batch_avoiding_combined_20260919_132703`, not the older `temp/2508` drop), so a figure and the
  table beside it are the same evaluation. **Every point carries all five seeds and all three
  geometries**, so the ladder's dashed-hollow baseline and dotted single-seed series are gone — and
  🆕 **CI-MeanFM enters both figures as a full five-seed series**, four models instead of three, its
  $\nfe=1$ point ($59.2$ steps, $18.1$\,ms) on the frontier.
- **`fig:avoiding-paths` rebuilt at two episodes with no new fetch**: `scripts/eval.py:297--299` fixes
  episode $i$ by $i$, so the first two episodes of a twenty-episode artefact **are** the two a
  two-episode run produces. $2/2$ reach the goal in every panel, all violation-free.
- 🔴 **"Flagship" is out** — *"THE FLAGSHIP IS JARGON. NEVER MENTION IN THESIS."* Banned in the naming
  table §6, with the replacements: **operating point** and **baseline of record**.
- **Fig. 6.1's row labels** lose the `\raisebox` that lifted them above their panels.
- 🟡 **Two rewrites.** *"because the tracking controller filters the plan"* → the two things that
  actually stand between plan and motion, both from §5.1: only the first action of a plan is executed,
  and it reaches the joints through IK and a joint-space PD controller. And *"Three things have to be
  read into that matrix …"* → four sentences of fact.
- **`tab:coverage-k20` is new**: v3.52's $\nfe=20$ census promoted from a `\guard` to a table in the
  quarantined subsection. §6.1's conclusion, `tab:avoiding-conclusion`, §6.4's summary and Chapter 5
  §5.6.3.1 all follow the protocol change.
- Plumbing: `Corpus.csv()` falls back to `.gz`, loaders open through `open_csv()`, and `load_exact()`
  gained a `backbone` filter (the folder name does not carry the backbone — the 2026-09-17 pooling bug).
- 🔴 **Left alone:** `08_conclusion.tex` still leads with the twenty-episode $0.993$/$0.983$. Still true,
  still reported, but it is the quarantined evidence; Chapter 8 is frozen by author instruction.
- `check.py` 260 labels, 39 figures, all mechanical checks pass. All three bundles `101003_*` verify
  byte-faithful. **Not compiled.**

## v3.54 — 2026-09-21 · a third note level for the bundle: `--no-notes` → [`changelogs/v3.54_20260921_no_notes_bundle.md`](changelogs/v3.54_20260921_no_notes_bundle.md)

- **Tooling only; no thesis file changed.** `--clean-notes` sets `\submissiontrue`, which the inherited
  preamble wires to `\srcnote` and nothing else — so the violet `\guard` and grey `\dataref` survived it.
  `bundle/make_bundle.py` now has three note levels: default (all visible), `--clean-notes`
  (`\srcnote` hidden), and 🆕 **`--no-notes`** (`\srcnote`, `\dataref` and `\guard` hidden). Suffix
  `_nonotes`, so the three variants never collide.
- **`\hole` and `\provisional` are never hidden** — they mark missing work, and a build that hides them
  lies about how finished the draft is.
- **A hidden `\guard` is still a limit of the result**, so `--no-notes` prints what it hid
  (*18 guards, 38 datarefs*) and says the build reads tidier than the evidence is. The counts strip
  comments first and now agree with `check.py` exactly.
- **Left open:** splitting `\guard` into the caveat that ships and the internal note that does not, so
  no switch has to choose between them — a pass over all eighteen uses, for when the drafting macros go
  to zero.
- `README.md` carries both note flags. `thesis_v3_20260921_090310_new_nonotes.zip` verifies
  byte-faithful; the v3.53 pair is unchanged. **Not compiled.**

## v3.53 — 2026-09-20 · Chapter 6 figures, and the tone rule gets its second half → [`changelogs/v3.53_20260920_ch6_figures_and_tone.md`](changelogs/v3.53_20260920_ch6_figures_and_tone.md)

- **`fig:avoiding-paths` re-scoped to the flagship models** (author): out go four different models at
  three different budgets, in come **MeanFM and CI-MeanFM at $\nfe=1$ and $\nfe=2$**. All four cells
  were already in the staged drop — an `AVOIDING_PANELS` edit and a re-run, no fetch. The pairing is now
  exact: these four cells **are** the four panels of the two average-velocity columns of
  `fig:raw-plans`, with the projector switched on, so the two figures are a matched before-and-after.
  $19/19/17/20$ of $20$ reach the goal; **all eighty episodes violation-free**.
- 🔴 **Two red-alert sentences out of that caption**: *"drawn like … and like the quadrotor scenes' …"*
  (a cross-reference dressed as reassurance) and *"top-right-hard is drawn **because** it is the
  geometry that separates the models"* (a justification for a design choice). The `\guard` lost the
  same clause and keeps only the limit.
- 🔴 **`fig:avoiding-tradeoff`'s budget list was wrong.** It claimed every model at
  $\nfe\in\{1,2,5,10,20\}$; **no model has all five** — MeanFM has $1,2,5,10,20$, FM has $1,2,5,20$,
  the baseline has $20$ alone. Listed per model now. The **numbers check out**: MeanFM $\nfe=1$ $t$,
  FM $\nfe=20$ $c$ and the baseline all match `tab:state-headline` to the printed digit, so figure and
  tables are the same evaluations; $\nfe=5,10$ are real measurements the tables do not list because the
  tables are locked to the operated budgets.
- 🟡 **Self-reasoning removed near `fig:k-ladder`**: *"does not recover the gap, because what the budget
  buys back sits in the projector rather than in the denoiser"* — nothing measured that split. Now
  *"leaves the baseline at thirteen times the cost."*
- **The style hint gains its second half**: 🔴 RED (the prompt in the thesis) and 🟡 YELLOW
  (self-reasoning — a *because* the evaluation did not produce, an argument against an absent objector),
  plus **the tone**: write as the person who ran the experiments, state it, do not defend it; *a limit
  stated plainly is strength, a limit argued around is not*. Re-registered in the naming table §8 rule 9,
  `DRAFT_OWNERSHIP.md`, both READMEs and memory.
- **The orientation `\hole` in §6.2 is gone** — v3.51 took the metric out of Chapter 5 and this was its
  last mention in the draft. hole 17 → 16.
- **`PENDING_20260920_all_lacking_runs.md` swept, nothing deleted**: D2, the diffusion `both-hard`
  artefact, R25/R26/R18 and the K20 re-run are struck through and marked *NO LONGER NEEDED* with the
  reason and the v3 version (new §11); the coverage audit matches. Open and unchanged: R2, D10c,
  R7/R14, R9, R15, D8/D9.
- `check.py` 258 labels, 39 figures, all mechanical checks pass. Bundles `221030_new` /
  `221030_new_clean` (6740 KB) verify byte-faithful. **Not compiled.**

## v3.52 — 2026-09-20 · Chapter 6 §6.1: the unprojected table becomes a mirror, the pending grid is closed by decision → [`changelogs/v3.52_20260920_ch6_results_pass.md`](changelogs/v3.52_20260920_ch6_results_pass.md)

- **`tab:avoiding-raw-models` is now a row-for-row mirror of `tab:avoiding-dpcc-protocol`.** It was one
  seed, one geometry, mixed budgets and a stray $\alpha_{\mathrm{end}}=0.05$ row. The corpus of record
  already carried the **unprojected arm** (`variant = diffuser`) of exactly the projected cells at **five
  seeds × three geometries**; it had never been extracted. The two tables now differ in the projector and
  in nothing else. New: every model reaches the goal (S $=1.000$) and **none satisfies the constraints**
  (S\&C $0.000$–$0.100$, $15.5$–$19.0$ violating steps) — the section's premise, measured; and the
  unprojected time column prices generation alone, so the projector's share follows by subtraction
  (≈ $8$ of MeanFM's $18.3$\,ms, ≈ $374$ of the baseline's $553.4$\,ms).
- 🔴 **A misattributed number fixed.** "$28.0$ violating control steps … at its own twenty steps" is the
  baseline at **$\nfe=1$**; at twenty the same cell reads $18.5$. Both are now given with their budget.
- **Fig. 6.1 is guaranteed to precede Table 6.1**: the float moved to the top of §6.1.1.1 (it was
  declared five paragraphs after its first reference), placement relaxed to `[!htbp]`, and a
  `\FloatBarrier` set between the prose and the table.
- **§6.1.1's introduction stops repeating Chapter 5** — three paragraphs of model names and budget
  semantics cut to one, 217 words → 96, with pointers to `sec:setup:baselines` and `tab:eval`.
- 🔴 **The $\nfe=20$ extended campaigns are truncated, and now say so.** Counted cell by cell: the
  baseline has *both-hard* for **seed 6 alone** under $r$ and $c$, **not at all** under $t$, and none in
  its unprojected arm; MeanFM has **no *both-hard*** at that budget; FM has 2–3 of 5 seeds. Every
  $\nfe\in\{1,2\}$ cell is complete. The $^{\dagger}$ footnote is rewritten and a `\guard` carries the
  census. The author's suspicion was right.
- **The pending rows are gone** (four in `tab:state-headline`, one in `tab:avoiding-dpcc-protocol`,
  both `\hole`s; hole 19 → 17) and 🆕 **§6.1.2.7 `sec:res:avoiding:budget20`** answers their question
  from existing data: each flow model at $\nfe=2$ against $\nfe=20$, **matched to the cells both budgets
  cover**. **No model improves at twenty** (S\&C $1.000\to0.970$, $0.990\to0.955$, $1.000\to0.950$) and
  the cost rises $17$–$37\times$, two of the three then costing more per action than the baseline. R25,
  R26 and R18 marked *ruled out on cost (author)*.
- **Chapter 5 §5.6.3.1** gains the sentence that makes the marking readable: an unmarked results cell
  carries all five seeds and all three geometries.
- New analysis of record `DA_in_Paper/analysis/avoiding_unprojected_and_budget20.py` (three blocks:
  the unprojected mirror, the matched budget comparison, the coverage census), registered in
  `analysis/INDEX.md`; `PENDING_20260920_all_lacking_runs.md` §10 records the decision and the census.
  `check.py` 258 labels, 39 figures, all mechanical checks pass. Bundles `220209_new` /
  `220210_new_clean` (6733 KB) verify byte-faithful. **Not compiled.**

## v3.51 — 2026-09-20 · Chapter 5 author pass: the prompt leaves the thesis → [`changelogs/v3.51_20260920_ch5_author_pass.md`](changelogs/v3.51_20260920_ch5_author_pass.md)

- 🆕 **A standing writing rule, on the author's instruction: _the prompt is not thesis text._** He found
  his own request rephrased inside the Fig. 5.9 caption (*"both translucent **so that** the excluded
  region stays readable underneath ... **which is why** satisfying the constraints can only be the
  projection's doing"*). New file `Writing/Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md`:
  the rule, the three shapes it goes wrong in (prompt rephrasing · the justification tail · the
  done-claim), worked before/after pairs from this draft, and a caption checklist. Registered in the
  naming table (**§8 rule 9**), `DRAFT_OWNERSHIP.md` (binding on v2/v3/v4), `Auxiliary/README.md`, this
  draft's `README.md`, a new `Writing_Hints/README.md`, and the Chapter 5 header comment. The caption
  itself now states the colour rule and points at `tab:avoiding-geometries` for the count.
- **Fig. 5.3 is two panels and no longer argues.** The claim about what Fig. 5.9 shows is gone; the new
  right-hand panel is the *task finished* — the measured end-effector path of all 96 demonstrations,
  every one starting at $(0.525,-0.280)$ and ending beyond the goal line at $y=0.35$ (smallest
  per-episode maximum $y$ is $0.403$, so 96 of 96 cross), in 60–106 recorded steps. `fig_env_avoiding`
  existed in the store, unused; its in-plot title was removed (naming table §6) and `_scene_panel` now
  reserves top margin only when a heading is passed.
- **The action box is out of the alignment constraint set** (author): an `action_bounds: 'auto'` bound
  fitted to the demonstrations carries no value a reader can check. Gone from `tab:aligning-constraints`,
  `tab:va-vs-d3il`, the `combined-5` and D3IL-avoiding sentences, the UAV-corridor protocol line and
  three `\srcnote`s. Ch 4 still defines it → filed to v2.
- **The orientation metric is gone from §5.4.2** — no result reports it; the `\srcnote` now says why
  (the recorded box angle uses a convention that differs between runs). One `\hole` in Ch 6 is the last
  mention left in the draft.
- **The quadrotor `\hole` the author could not parse is dropped.** It was not a defect: it asked for the
  flown episodes instead of the reference paths, and the prose above `fig:expert-uav` already states
  that the references are the conservative drawing. hole 20 → 19.
- **§5.5.2 states how large the D3IL policies are** — heads of four layers at width 72 (BESO, DDPM-GPT)
  and encoder–decoder transformers at width 64 (DDPM-ACT, VAE-ACT), well under a million parameters
  each, behind **two ResNet-18 trunks of ≈11 M** that this thesis adopts unchanged, with its 4.0 M
  temporal U-Net in place of the head. Sourced from D3IL Table 9 **and** the repo configs; the `\srcnote`
  records that D3IL publishes no parameter counts.
- **§5.5.3 cites the original quadrotor task** in the paragraph that describes it, and opens *This
  benchmark* with the principle it shares with D3IL-avoiding and DPCC — train on demonstrations,
  introduce constraints they do not satisfy at test time, score both finishing and satisfying.
- **§5.6.4 flagged undecided** with an `% OPEN` comment recording what must move if it is dropped.
- **Chapters 7 and 8 were not opened** (author instruction). `sync_v2.py status` clean at v2.24;
  `check.py` 39 figures, hole 19, all mechanical checks pass. Bundles `214426_new` / `214426_new_clean`
  (6731 KB, 41 figures) verify byte-faithful. **Not compiled.**

## v3.50 — 2026-09-20 · the two missing path figures, and three Chapter 5 hotfixes → [`changelogs/v3.50_20260920_paths_figures_and_ch5_hotfix.md`](changelogs/v3.50_20260920_paths_figures_and_ch5_hotfix.md)

- 🆕 **The executed-path figures of the two manipulator benchmarks are built** --- asked for four times
  and blocked on a builder that was never written. New pipeline, the same split the quadrotor figures
  use: `extract/exec_paths.py` (numpy, reads the gitignored drops) → `data/exec_paths.json` →
  `builders/exec_paths.py` (standard library). Two panels were factored out
  (`avoiding.geometry_panel`, `scenes.aligning_constraint_panel`) so a path is never drawn over a
  *redrawn* constraint set --- what lies under it is the geometry the projector was given.
  - **`fig:avoiding-paths`** (§6.1.2.2): four panels of 20 episodes --- MeanFM, CI-MeanFM and FM at one
    network evaluation, the diffusion baseline at twenty --- under per-step projection with temporal
    consistency on `top-right-hard`, seed 6. **All 80 episodes are violation-free**; 19/17/17/20 reach
    the goal. The plans entering it are the near-scribbles of Fig. 6.1's $\nfe=1$ row; between the two
    figures sits the projector. `\guard`: one seed, one geometry.
  - **`fig:aligning-paths`** (§6.2): the ten contexts under no projection, per-step and endpoint.
    Violation-free **2/10, 9/10, 10/10 --- the $\nfe=20$ rows of `tab:va-projection`, reproduced**, an
    independent check that the artefacts and the published table describe the same evaluation. Measured
    while building and now in the prose: unprojected, **seven** contexts enter the keep-out region and an
    eighth leaves the halfspace; under either projector neither is entered at all, and the one context
    per-step projection does not keep clean breaches the **action box** instead. 🔴 `\guard`: **it draws
    the end effector, not the box** --- the constrained quantity here is the planned end-effector
    position, but the quadrotor figures draw the body, and the box pose is not logged (**D10c**).
  - Both `\hole`s asking for a counterpart of `fig:uav-corridor-paths` are gone. **hole 22 → 20**;
    36 → 38 figures.
- **Fig. 5.9 colours the demonstrations by whether they satisfy the geometry** --- red where a
  demonstration crosses the panel's constraint set, green where it satisfies it at every step, both
  translucent so the excluded region stays readable. The green lines number 0, 1 and 2, which is what
  `tab:avoiding-geometries` prints; the builder **asserts** its colouring against that published count,
  so the two cannot drift. Legend to two rows, caption rewritten.
- **Fig. 5.4 moved after the scene and camera figures** (author): the subsection now reads task →
  cameras → the ten contexts it is evaluated on. `fig:aligning-contexts` renumbers 5.4 → 5.6. Order
  only; no wording changed.
- **Table 5.4's caption says what the table is.** "in the two forms D3IL-avoiding also uses" was a
  cross-reference dressed as emphasis; the caption now names the constraint set and stops.
- **Verification.** `tools/check.py`: 14 files, 256 labels, 53 citations, 38 figures; hole 20,
  provisional 3, guard 16, srcnote 57, dataref 36, todofigure 0; all mechanical checks pass.
  `sync_v2.py status` clean at v2.24. Three figures rebuilt, rendered and inspected. Bundles rebuilt:
  `thesis_v3_20260920_195809_new` and `…_195810_new_clean`, both verified byte-faithful. **NOT COMPILED** --- no TeX toolchain here.

---

## v3.49 — 2026-09-20 · Chapter 8 returns, the plan matrix is complete, MuJoCo MPC gets a price → [`changelogs/v3.49_20260920_chapter8_figures_and_controller_cost.md`](changelogs/v3.49_20260920_chapter8_figures_and_controller_cost.md)

- 🆕 **§6.3.3 now prices the two tracking controllers** (`tab:uav-controller-cost`). On the unprojected
  arm a control step costs $94.9$\,ms under the cascaded geometric brake-to-rest controller and
  $215.6$\,ms under MuJoCo MPC; the planner's share is the same in both ($88$--$90$\,ms), so the
  **$119$\,ms between them, about twenty times, is the controller**, which solves an optimisation every
  control step where the other evaluates a formula. By itself that is more per control step than
  generating the plan and about seven times what the D3IL-avoiding operating point spends on a whole
  action. It buys a traverse and not a result: neither controller produces a flight that passes the goal
  within the constraints on this scene. Source `Report_20260909_MJPC_vs_PID_s_curve` §5.4--5.5; elapsed
  flight time from the job records ÷ executed control steps. `\guard`: a decomposition, not a timer
  around the controller call; the remainder row carries the simulator step, so only the *difference*
  isolates the controller; one seed, ten flights against three. Stated in the caption as a **different
  quantity** from the planner time Chapter 5 defines.
- 🆕 **Chapter 8 is built again** (author, 2026-09-20): a thesis must not be missing a chapter while v4
  has not started. Ownership is unchanged --- v3 drafts it concisely, v4 refines. 🔴 Building it exposed
  that it was stale: **every UAV-pillars claim is gone** (the scene has been withheld from Results since
  v3.41, and a built Chapter 8 would have contradicted Chapter 6), the summary and the three RQ answers
  were rewritten against §6.4 as it now stands, and **"wall-clock time"** went with them. `sec:conc:future`
  is untouched. Handover: `to_v4/FROM_v3_20260920_chapter8_is_built_again.md`.
- **§6.1.2.1 rewritten**: 9 paragraphs / 5966 characters → 7 / **4636**, with the two self-narrating
  paragraphs gone and a fixed order (what is held fixed → the marks → the baseline reproduces → the two
  models that clear it → the operating point and the rule → the resolution → what it does not mean).
  No fact, `\autoref`, table, caption, `\dataref` or `\hole` was dropped. One number was: "$0.867$ at
  $\nfe=20$" is in no table of this draft, so no reader could check it.
- **Figures stop printing protocol on the page.** Fig. 6.5 carried "*unprojected, untightened · seed 6 ·
  median over the ten contexts*" inside the drawing; the same defect was in Figs. 6.2, 6.3 and 6.7 and was
  fixed as a class. Titles and protocol subtitles removed, margins retightened, the *both-hard* panel's
  single-seed baseline now said explicitly in the caption, `fig:k-ladder`'s legend "CI-MeanFM, seed 6" →
  "CI-MeanFM (one seed)", and Fig. 6.5's "box not moved" label moved off the diffusion point it was
  overprinting. `frame()`'s docstring and the canonical translation table now carry the rule.
- **Fig. 6.6 names its model**: `MeanFM, K = 20` beside the projection legend --- §6.2 discusses three
  generative models and the figure named none.
- 🆕 **`fig:raw-plans` is complete and the draft has no `\todofigure` left.** D10d landed: the panel comes
  from the complete sibling campaign `…aw10_thres0.5` (same checkpoint, $\nfe=20$, scene, seed 6,
  `both-hard`), because the run the request named died mid-variant; `thres0.5` is a *projection* threshold
  and this is the *unprojected* arm, so no projector ran in either campaign --- stated in the figure's
  `\guard`. Prose updated ("three diffusion panels"), and the new panel is read in the text.
- 🔴 **Disclosure that came with it**: the same truncation leaves `tab:state-headline`'s diffusion
  $\nfe=20$ row under temporal consistency averaged over **two** geometries --- the only such cell in
  either avoiding table. Said in the `$^{\dagger}$` footnote. Not a flattered baseline, and one evaluation
  re-run repairs it and fills D10a's diffusion cell.
- **Verification.** `tools/check.py`: 14 files, 5645 lines, 254 labels, 53 citations, 36 figures; hole 22,
  provisional 3, guard 14, srcnote 57, dataref 34, **todofigure 0**; all mechanical checks pass.
  `sync_v2.py status` clean at v2.24. Five figures rebuilt, rendered and inspected --- **no data value
  changed**. Bundles rebuilt: `thesis_v3_20260920_161735_new` and `…_new_clean` (6247 KB each), both verified byte-faithful (9 inlined + 4 collapsed sources). **NOT COMPILED** --- no TeX toolchain here.

---

## v3.48 — 2026-09-20 · figure captions stop explaining themselves; the baseline joins the plan matrix → [`changelogs/v3.48_20260920_figure_captions_and_direction_key.md`](changelogs/v3.48_20260920_figure_captions_and_direction_key.md)

- **The v3.46 table rule now applies to figures.** All twenty figure captions of Chapters 5 and 6 were
  measured and the long ones rewritten: the worst fell from **1833 to 800** characters and the median
  from **858 to 531**. Thirteen were rewritten, seven were already concise. Nothing was dropped --- every
  sentence that left a caption is in the paragraph beside the figure, and three of them read better there,
  most of all `fig:expert-uav`, whose caption had been carrying the argument for drawing the reference
  rather than the flight.
- 🆕 **Fig. 6.1 gains the diffusion baseline at its own budget.** The matrix had $\nfe=1$ and $\nfe=2$
  only, so the baseline of record appeared nowhere in it. A third row carries $\nfe=20$, the budget it was
  trained for. **The panel is a download that has not been fetched**, so the cell carries the one
  `\todofigure` in the draft, with the run folder, seed, geometry, variant and crop written into it.
  Tracked as **D10d**; specified in `REQUEST_20260916_cluster_fm_plan_panels.md`.
- **The preferred direction is now a legend element.** `_dirarrow` draws the arrow inside a bordered box
  with its label, scaled to each figure's font factor. Two placements had to move with it: UAV-corridor's
  key went to the top right, where the baseline point is not, and each D3IL-avoiding panel gained 26 %
  y-headroom, because at the old limits the box landed on the diffusion points in two panels of four.
- **Chapter 5 credits both works for the step metric.** Control steps are cited to \ac{DPCC}
  \parencite[Sec.~6.1]{romer2025diffusion} *and* to the endpoint-projection work, which reports
  *Total Steps (Safe Trials)* on this same D3IL task \parencite[Sec.~VII-A]{li2025hardflow} --- read in
  the paper, not from a summary.
- 🔴 **"Wall clock" leaves the thesis.** It describes how the measurement is taken, not what is reported:
  the defined name is **the time to compute one action** (§5.4.1), written **time per control step** on
  axes. Five occurrences replaced; the one that meant an episode's duration became **how long a single run
  may go on**. Two rows added to the translation table, the second banning batch vocabulary
  (`rollout`, `trial`, "row") from inside a figure. v2's abstract and v4's Chapter 8 still carry one
  occurrence each and are raised in `cross_draft/INBOX.md`.
- **Fig. 6.4 rebuilt.** Title and batch-vocabulary subtitle deleted, font scaled to print size, axes
  reading *step budget K* and *time to compute one action*; its protocol moved into the caption.
- Four figures rebuilt, re-rendered, visually inspected and exported; no number in any figure changed.
  `tools/check.py` passes (13 files, 5,514 lines, 249 labels, 53 citations, 35 figures) with
  `todofigure 1` by design, and v2 inheritance is clean at v2.24. The bundles stamped `20260920_152315`
  and `20260920_152316` verify byte-faithful. **Not compiled locally.**

---

## v3.47 — 2026-09-20 · raw versus projected results, locked budgets, and a complete pending ledger → [`changelogs/v3.47_20260920_ch6_raw_projected_restructure.md`](changelogs/v3.47_20260920_ch6_raw_projected_restructure.md)

- **§6.1.1 is raw output only; §6.1.2 contains every projected result.** *The Plans Themselves* becomes
  *Raw Model Outputs*, the meta-explanatory smoothness subsection is removed, and endpoint/per-step
  projection is folded into the same projection-method hierarchy.
- **Removed two redundant result tables:** the mixed-protocol seed-spread table and the single-seed
  selected-budget table, plus the per-geometry cheapest-cell table and their nearby search prose.
- **Table 6.2 locks the model comparison to $\nfe=1,2$** plus the native diffusion baseline at
  $\nfe=20$. Diffusion $\nfe=2$ is now visibly pending in the table; $\nfe=10$ moves to the Appendix.
  The extended table likewise prints the missing CI-MeanFM $\nfe=1,2$ rows directly.
- **§6.1.3 gains the final model--projection combination table** for the extended MeanFM operating
  point, the CI-MeanFM result at DPCC's protocol and the baseline of record.
- Removed redundant Fig. 5.4; Fig. 5.5 already draws the ten contexts on the same alignment constraint
  plane. Removed the rejected plan-smoothness `\hole`.
- **Fig. 6.6 is larger and contains less text**; the captions carry its marker semantics and protocol.
  All Pareto/frontier direction arrows now use dark slate instead of red. Four canonical figures were
  rebuilt, raster previews refreshed, visually checked and exported.
- Table 5.8 audit found one Method gap: FM's Beta transport-time law lacks a mathematical definition.
  A dated v3.47 note assigns the formula and placement to v2 Chapter 4; v4 needs no action.
- **`PENDING_20260920_all_lacking_runs.md` is now the full current ledger.** R25 records the
  five-seed CI-MeanFM extended evaluation; R26 the four missing diffusion-$\nfe=2$ trainings; R27 the
  optional diffusion endpoint-projection evaluation. Smoothness is retired and the completed raw-plan
  panel marked closed.
- `tools/check.py` passes (13 files, 5,503 lines, 249 labels, 53 citations, 35 figures), and v2
  inheritance is clean at v2.24. The v3-only annotated and clean bundles stamped
  `20260920_132552` verify byte-faithful; their two expected unresolved references point into the
  deliberately collapsed Chapter 4. **Not compiled locally.**

---

## v3.46 — 2026-09-20 · **KEY UPDATE** — every averaged number in Chapter 6 carries its spread; the removed unit-scale note turns out to be a real flaw → [`changelogs/v3.46_20260920_KEY_UPDATE_spread_and_flaws.md`](changelogs/v3.46_20260920_KEY_UPDATE_spread_and_flaws.md)

- 🔴 **Mean $\pm$ spread, in seven tables of Chapter 6.** The convention is defined once in a new §5.4.4
  (`sec:setup:metrics:spread`): **multi-seed** results carry the sample sd **across the five training
  seeds** (a claim about a model, not about one training run, and the larger of the two spreads);
  **single-seed** results carry the sd across episodes, contexts or flights; **counts and medians carry
  none**. A $\pm$ here is a spread, never a confidence interval and never a test. New DA:
  `analysis/avoiding_table_spread.py`, whose point estimates reproduce the published table exactly at
  DPCC's protocol; the four extended-protocol $\nfe=20$ rows keep the published geometry-mean and carry a
  $\dagger$, because *both-hard* is short of seeds there. Three sentences were sharpened by it — most of
  all the unprojected step column, where every mean carries a spread **wider than the distance between
  them**.
- 🔴 **The `\provisional` the author deleted was a real flaw.** The ODE sampler inherited from \ac{DPCC}
  draws the initial noise at **half** unit scale while **instantaneous-velocity matching trains at unit
  scale**; MeanFM and CI-MeanFM sample correctly and the baseline is self-consistent, so the mismatch is
  FM's alone, in all three environments. Impact assessed per environment in
  `FLAW_20260920_known_flaws.md` §B: **nearly none on D3IL-avoiding** (FM is at the ceiling), **possibly
  large on D3IL-aligning and the quadrotor**, where FM is the weakest model and this is a live
  explanation. **No retraining needed** to fix. The action-weight flaw gained the author's *tiny effect*
  assessment with the evidence for it. File renamed `FLAW_20260920_known_flaws.md`.
- **The sixty-context item is retired**, in the draft and in three ledgers, with the reason recorded:
  D3IL's sixty contexts score an image policy with no projection in the loop, so they cost it almost
  nothing; every alignment episode here carries a projection solve at every control step.
- 🆕 **`fig:aligning-outcomes`** — the ten contexts individually, start distance against end distance, one
  panel per projection method, built from the committed CSV with **no fetch**. It shows what the median
  hides: the constraint column turns over almost completely while the distances are redistributed ---
  three contexts become much worse, two clearly improve, and one is never moved under any method.
- **The path figures, answered precisely.** D3IL-avoiding is a **download and nothing more** —
  `scripts/eval.py` already writes the executed end-effector path on every evaluation, it was simply
  never fetched (8 cells, <10 MB). D3IL-aligning's **box** path is **not** a download: the box pose is
  logged at no step and even its end point is missing from every thesis cell, so that one needs one line
  added to the evaluation and three re-runs. Both specified cell by cell in
  `PENDING_20260920_all_lacking_runs.md` §5a.
- **The quadrotor demonstrations are generated and flown, and §5.4.3 now says so**: a reference path per
  passage class, every episode flown in MuJoCo by the same cascaded geometric controller the evaluation
  uses, kept only if it stayed airborne and inside its contact budget. Raised for v2.
- **§6.1.1 leads with the numbers**: *Smoothness of the Plans* moved from first to last and was retitled
  *Why the Projected Numbers Separate the Models So Little*.
- Table 5.8's transport-time row fits its column (three defined tokens instead of a sentence per cell);
  Table 5.9's step-budget row loses its italics.
- **Hotfix, same pass: tables stop explaining themselves.** A table now says what it is and what its
  marks mean; the reasoning sits in the paragraph beside it. Across all 33 tables of v3 the worst caption
  fell from **1699 to 656** characters and the worst data cell from **273 to 168**. Nothing was dropped:
  what left a caption was already in the prose or was moved there, and in five places it reads better for
  it — most of all `tab:eval`, whose five difficult rows now get a paragraph, and
  `tab:avoiding-dpcc-protocol`, whose caption had been carrying the chapter-wide definitions of the
  operating point and the baseline of record.
- 🔴 **Error caught during that re-read:** the $\pm$ columns above had never landed on
  `tab:avoiding-raw-models` — the edit was in a script that aborted before writing and the retry dropped
  it. Applied, and the other six tables re-checked cell by cell.
- **Continuation hotfix: the alignment protocol is now genuinely ten contexts throughout.** Table 6.10
  now reports `ms/step` as mean $\pm$ sample sd and every row is the same ten contexts, as Chapter 5
  specifies. The audit caught the same hidden 10/30 mixture in Table 6.9 and Fig. 6.5; the analysis and
  frontier builder now restrict every raw superset to the ten shared contexts before reducing it, and
  the affected values and prose were recomputed.
- **Fig. 6.6 rebuilt as one readable panel:** ten context groups, three adjacent stems each, with a
  compact legend and no miniature repeated axes. The counts and context-level statements were checked
  directly against the committed rollout CSV.
- **Every Pareto/frontier plot now carries an in-panel red `better` arrow.** D3IL-avoiding points toward
  fewer steps and less time; D3IL-aligning toward less distance and less time; UAV-corridor toward more
  S\&C and less time. The arrows were visually checked after raster export and moved clear of labels and
  frame edges.
- Final verification: `analysis/va_results.py` reports ten contexts for every thesis cell;
  `tools/check.py` passes all mechanical checks (13 files, 5688 lines, 251 labels, 53 citations,
  36 referenced figures); the v2 inheritance status is clean. The new-sections review bundle
  `thesis_v3_20260920_120252_new` was built and verified byte-faithful. **Not compiled locally.**

---

## v3.45 — 2026-09-20 · hotfix: §6.1 reshaped to match §6.2 and §6.3; the projection tables get their missing models; no $\nfe=100$ on alignment → [`changelogs/v3.45_20260920_ch6_structure_hotfix.md`](changelogs/v3.45_20260920_ch6_structure_hotfix.md)

- 🔧 **§6.1 is now Generative Models → Projection Methods → Conclusion**, the shape §6.2 and §6.3 already
  had. The results that sat in a fourth subsection *after* the projection discussion are folded back into
  the generative-model subsection, which now reads the same four models twice — unprojected, then under
  the per-step projector. `sec:res:avoiding:models` moved with the results, `sec:res:avoiding:plans` was
  retired and its references re-pointed; no reference broke.
- **Table 6.8 locked to the section's budget set.** It had been searching every budget that exists, which
  is how a $\nfe=5$ MeanFM cell entered a section locked to $\{1,2\}$. Recomputed: every other cell is
  unchanged, and MeanFM on *top-right-hard* does not reach $1.00$ within its budgets — best **0.95 at
  $\nfe=1$**, one episode of twenty short. The cell is now a dash carrying that value and the paragraph
  under the table was rewritten around it.
- **No $\nfe=100$ in the alignment results** — removed from `tab:va-models`, the prose and
  `fig:aligning-tradeoff` (rebuilt; the frontier is unchanged, that point was dominated anyway).
  ⚠️ It is the diffusion baseline's *own* configured chain length and the only diffusion configuration
  that moves the box, so a `\guard` records that it exists, what it costs and why it is not reported.
- **The `2$^{\ast}$` marker is gone** from `tab:va-projection`; its three caveats are plain sentences in
  the caption.
- **`tab:va-projection-models` gets MeanFM**, which it was missing entirely — all three flow-based models
  now, at $\nfe=2$, $10$ and $20$. The fifteen-of-eighteen count is unchanged; the "moves the box"
  sentence was recomputed over the same eighteen (more in four, level in seven, fewer in seven).
- **Chapter-wide model-coverage audit.** Three gaps found and stated where they occur:
  D3IL-avoiding's projector study is MeanFM alone (endpoint projection was never run on FM or the
  baseline there); UAV-corridor now prints the baseline's per-step row with dashes for endpoint; the
  s-curve re-run covers the flow models only. 🔴 **Endpoint projection has never been run on the diffusion
  baseline on any scene** — recorded in `PENDING_20260920_all_lacking_runs.md` §3b as the one gap worth a
  run.

---

## v3.44 — 2026-09-20 · cost frontiers for D3IL-aligning and UAV-corridor; Chapter 5's tables explained rather than enumerated; §6.1's figures → [`changelogs/v3.44_20260920_frontiers_and_ch5_tables.md`](changelogs/v3.44_20260920_frontiers_and_ch5_tables.md)

- 🆕 **Two new cost frontiers**, built from the corpora of record. `fig:aligning-tradeoff`: median final
  box-to-target distance against ms/step — three non-dominated configurations, **all three MeanFM**, and
  the baseline at $\nfe=100$ ($1526.6$\,ms) is beaten on both axes by MeanFM at $\nfe=10$ ($0.1194$\,m
  at $99.2$\,ms). `fig:uav-corridor-tradeoff`: S&C against ms/step — four non-dominated points, all the
  two average-velocity models at $\nfe=3,5$, cheapest $97.7$\,ms; the baseline is rightmost at $625$\,ms
  and $0.00$. Both are **outcome against cost**, not cost against cost like `fig:avoiding-tradeoff`,
  because on neither task does the control-step count measure arrival — the captions say so.
  **Pillars** gets none (withheld from the draft); **s-curve** gets none and the draft says why: every
  configuration there scores $0.00$, so a frontier would rank failures by price.
  New: `plotting/builders/frontier.py`, two corpora and their cell registries in `sources.py`. All 8
  alignment and all 30 corridor cells reproduce their published tables exactly.
- **Fig 5.3** loses its demonstration panel: `fig:constraints-avoiding` already draws the same 96
  demonstrations, over the constraint geometries.
- **Table 5.8**: Diffusion and FM in separate columns (their settings *are* identical on D3IL-avoiding —
  checked — but the table no longer says so by merging them).
- **The optimiser `\guard` is now main text**: the settings differ because the training targets differ,
  and each model was tuned for the convergence of its own loss curve; a second paragraph says what is
  held fixed and therefore what the comparison rests on.
- **Table 5.9** stops enumerating $\eta$ per budget and explains instead: what an **episode** is and that
  the episode limit is a hard stop on run time; why the activation threshold is **lowered** as the budget
  rises (ten solves per candidate per control step is not affordable, and what is wanted is a fixed
  number of guiding steps, not a fixed share); why the quadrotor's tightening adds the vehicle radius.
- **Table 5.11 deleted**; §5.6.4 gives the closed form once and reads it on two settings — the one that
  has three guiding steps and the one that has none.
- **Fig 6.1** marks the MeanFM and CI-MeanFM columns with bold and an arrow, and gives every column a
  one-line summary of what its panels show.
- **Fig 6.2 → `tab:avoiding-raw-models`**: five counts and five step figures were a bar chart; they are
  now five rows.
- 🔍 **The $\nfe=1,2$ rows were challenged and re-checked.** They stand, in the corpus of record *and*
  in an earlier independent batch. §6.1.3.1 gains the explanation: a cell at DPCC's protocol is 30
  episodes, the extended protocol at 300 still puts $\nfe=1$ at least level with $\nfe=20$, and the
  reason is that this benchmark is **saturated** — not that fewer solver steps make a better plan, which
  the plans themselves and `fig:aligning-tradeoff` both contradict.
- **The two missing flown-path figures** (the counterparts of Fig 6.6/6.7 for avoiding and aligning) are
  named as missing in both `\hole`s, recorded in `PENDING_20260920_all_lacking_runs.md` §5a, and added
  to the cluster fetch request. Neither is a run; both are downloads.

---

## v3.43 — 2026-09-20 · Chapter 5's parameter tables; §6.1 reassembled as generative model → projection → results; the action loss weight deleted from the thesis → [`changelogs/v3.43_20260920_ch5_tables_and_ch6_reassembly.md`](changelogs/v3.43_20260920_ch5_tables_and_ch6_reassembly.md)

- 🔴 **The action loss weight is not 10 everywhere, and it is now gone from the draft.** D3IL-aligning:
  diffusion 10, every flow model 1. **UAV, all three scenes: every arm 1, the diffusion baseline
  included** — so that arm is not DPCC's configuration, as `config/uav_mix.py:497-503` itself declares.
  MeanFM / CI-MeanFM: the key is kept for folder naming and **never reaches the loss** (FIX-3,
  `flow_matcher_v3_meanflow/models/mf_diffusion.py:49-52`), so `_aw10` in those paths is a naming
  artefact. Author's call: no time to retrain — **recorded, not fixed**, in
  [`../data_status/FLAW_20260920_known_flaws.md`](../data_status/FLAW_20260920_known_flaws.md),
  and the parameter deleted from §5.5.1, `tab:train` (row, guard clause and srcnote token),
  `tab:state-models`, `fig:avoiding-raw-models` and the §6.3 lead-in. Nothing in the draft now claims the
  models are matched on it. `04_method.tex:1437` still does — raised for v2 in `cross_draft/INBOX.md`.
- 🔧 **§6.1 reassembled into pipeline order: generative model → projection → results.** The old
  "6.1.1 Generative Models" held only full-pipeline results. Blocks unchanged in content, reordered:
  6.1.1 **The Generative Models** (the plans, unprojected) · 6.1.2 **Projection Methods** ·
  6.1.3 **The Models under Per-Step Projection** (DPCC protocol, extended, cost frontier, step budget) ·
  6.1.4 Conclusion. **Every label travelled with its block**, so the four references from Chapters 4, 5
  and 8 still land correctly; the lead-in, the two subsection openings and four direction-sensitive
  references were rewritten.
- **Fig 6.3** (`fig:raw-plans`) columns reordered to the house order **MeanFM · CI-MeanFM · FM ·
  Diffusion**, with caption and prose updated.
- **Table 5.8**: `Diffusion steps` row deleted; transport-time sampling written out per column
  (Beta$(1.5,1)$ for FM, logit-normal $P\sidx{mean}=-0.4$, $P\sidx{std}=1.0$ for the average-velocity
  models, uniform over the twenty denoising steps for diffusion).
- **Table 5.9**: the step budget now says *ODE solver steps* for the flow-based models and denoising
  steps for diffusion; the activation threshold carries the values actually run (**the author was right
  that alignment is not all 0.5**: $0.5$ at $\nfe=2$, $0.4$ at $\nfe=10$, $0.2$ at $\nfe=20$, plus
  $\eta=1.0$ for endpoint projection at the avoiding budget floor); low-level control named as inverse
  kinematics + a joint-space PD controller (D3IL) and inverse kinematics + a cascaded geometric PID
  tracker (quadrotor), with MuJoCo MPC flagged as the §6.3.4 exception.
- **Table 5.10**: the `Seeds` column gave seed *indices* (`6--10`, `6`); it now gives counts — 5 training
  seeds on D3IL-avoiding, 1 elsewhere.
- **Table 5.11 moved to Chapter 6.** `tab:tiers` is a result, so it now opens §6.1.3.2 *Extended
  Evaluation*; Chapter 5 keeps one sentence and a pointer. Label unchanged.
- **§5.6.4 rewritten** around a new table, `tab:guiding`: the closed form
  $n\sidx{guide} = \lceil \eta\nfe \rceil - 1$ and then every $(\nfe,\eta)$ pair Chapter 6 uses,
  with its projected and guiding steps. Chapter 5's table count is unchanged — one moved out, one added.
- `tools/check.py`: 13 files, 5387 lines, 249 labels, 53 citations, 35 figures — all mechanical checks
  pass. Drafting macros: hole 19, provisional 5, guard 14, todofigure 0. **Not compiled.**

---

## v3.42 — 2026-09-19 · **KEY UPDATE** — the D3IL-avoiding tables are locked; §6.1 restructured; Fig 6.3 complete; the alignment contexts figure and the full projector studies → [`changelogs/v3.42_20260919_KEY_UPDATE_avoiding_tables_locked.md`](changelogs/v3.42_20260919_KEY_UPDATE_avoiding_tables_locked.md)

- 🔴 **Erratum.** The "Diffusion, $\nfe=5$" row of `tab:avoiding-dpcc-protocol` was **not the diffusion
  baseline**: it reproduces from `plans/flow_matching_v3_ode_selectable/…H8_K5_Mmidpoint_Dmodels.diffusion.GaussianDiffusion`,
  the **flow** model under its pre-26-May class name (`cac7cc6a` renamed it `FlowMatchingODE`), midpoint
  solver, action weight 1. Row deleted. Enumerating `/plans/diffusion/` gives the baseline at $\nfe=1$,
  $10$ and $20$ only; those three cells reproduce the published numbers exactly. `fig:k-ladder` and the
  §6.1.1.4 prose were already right and are unaffected.
- **Both avoiding tables locked** to one model $\times$ budget set: MeanFM 1, 2 · CI-MeanFM 1, 2 · FM 1,
  2, 20 · Diffusion 1, 10, 20. FM $\nfe=5$ is seed 6 only and $\nfe=10$ does not exist, so both are
  dropped; diffusion $\nfe=2$ exists at seed 6 only (jobs 25965/25966) and is **not tabled**. Missing
  cells carry `\hole` blocks under each table, per the author's "mark as pending".
  `tab:state-headline` cut to the same set: MeanFM 5/10/20 and FM 5 removed.
- **`fig:raw-plans` is complete** — `\todofigure` → `\includegraphics{fig_raw_plans_diffusion_K2}`;
  **`todofigure` is now 0**. Caption rewritten (eight panels from three campaigns; the two diffusion
  panels are two trained checkpoints, which is the chapter's asymmetry in one picture). New paragraph
  records that the $\nfe=2$ unprojected arm still scores 0.00 S&C with 20.5 violating steps — coherence,
  not feasibility. Also fixed a pre-v3.41 mechanism sentence that survived the k-ladder correction here.
- **§6.1 restructured.** §6.1.1 → "Generative Models under Per-Step Projection", stating that every
  number in it is a full-pipeline result; new §6.1.2 "Plans before Projection" now holds
  `Smoothness of the Plans` and the former `Plans before Projection` (renamed `The Plans Themselves`).
  All labels unchanged.
- §6.1.1 now names the field: **three flow-based models, two of them average-velocity** (MeanFM,
  CI-MeanFM), plus the baseline, all on the same 4.0 M U-Net.
- **`tab:seed-spread` gains CI-MeanFM** at $\nfe=1$: $1.000 \pm 0.000$, $60.4 \pm 0.9$ steps,
  $18.2 \pm 0.3$ ms, seeds 6–10 — marked `$^{\ddagger}$` because its five seeds are at the **DPCC**
  protocol, not the extended one.
- **`tab:state-lowk`** rule column switched to `$r$`/`$c$`/`$t$`; the caption no longer re-defines them.
- **Endpoint projection**: §6.1.3's closing paragraph now says directly that it is not needed here
  because $\nfe=1$/$2$ already leave no shortfall to close, cites the guiding-step sections and the
  2.45$\times$ identical-rollout run, and ends by naming where it *is* tested.
- **Figs 6.1/6.2 needed no builder change** — `avoiding.py` already draws `KS = [1,2,5,10,20]` and the
  baseline's `{1,10,20}`. Captions updated to name the budget set and to explain why the scatter carries
  points the locked table does not. **No DA code touched this pass.**
- New ledger entry: [`../data_status/PENDING_20260919_avoiding_tables_locked.md`](../data_status/PENDING_20260919_avoiding_tables_locked.md).
- **Second pass, same changelog** — four further requests:
  - **New figure `fig:aligning-contexts`** after Fig 5.4 in §5.2.2: the ten contexts the evaluation
    replays, box start as a filled mark and target as a hollow one over the two draw regions, with **one
    context drawn as the real 0.10 m footprints at their recorded yaw**. New builder in
    `plotting/builders/scenes.py`; contexts recovered from the corpus of record and stored as
    `sources.ALIGNING_CONTEXTS` — their mean distance is 0.4530 m, the figure §6.2 already quotes.
  - **`tab:va-projection` gains its $\nfe=2$ block** (MeanFM, both projectors, all three rules).
    **$\nfe=1$ does not exist on this task for any model.** Three caveats stated in the caption: 30
    episodes not 10, no guiding step at threshold 0.5, and the interior-point rather than the
    sequential-quadratic backend.
  - **New §6.2.3.1 and `tab:va-projection-models`** — the full endpoint-vs-per-step study across the
    models (CI-MeanFM at $\nfe=2$ and $\nfe=20$, FM at $\nfe=20$). Counted over all **18** matched
    comparisons: endpoint projection ahead in **15**, per-step in **3**, every exception under temporal
    consistency. Stated as "the better projector wherever it has a guiding step, consistently rather than
    largely, at a comparable price" — not as a clean sweep, which the data does not support. Cost
    reported both ways: 0.6–0.9× at $\nfe=10$/$20$, about 3× at $\nfe=2$.
  - **§6.3.1.1** now says where the corridor's projector comparison lives and why per-step is held fixed
    in that subsection.
  - **New `tab:uav-scurve-projection`** — the six s-curve projection variants printed individually
    instead of collapsed into zeros. Per-step projection is the worst arm on all three surviving
    measures and costs **3610 ms per control step** at $\nfe=20$, twenty-one times the plan it repairs;
    that is the justification, now written into the text, for the re-run giving it one selection rule
    rather than three.
- `tools/check.py` after both passes: 13 files, 5345 lines, 248 labels, 53 citations, 35 figures; hole
  19, provisional 5, guard 14, **todofigure 0**. All mechanical checks pass. Not compiled.

---

## v3.41 — 2026-09-19 · new corpora of record; UAV-pillars deleted; `tab:avoiding-dpcc-protocol` complete; the guiding-step convention moves to Chapter 5 → [`changelogs/v3.41_20260919_new_corpora_pillars_removed_guiding_steps.md`](changelogs/v3.41_20260919_new_corpora_pillars_removed_guiding_steps.md)

- **Corpora of record** move to `19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703` and
  `batch_uav_20260919_111701`; every previously published number is reproduced exactly, so the batches add
  rows and revise none. Alignment is unchanged.
- **`tab:avoiding-dpcc-protocol` is complete**, the `\hole` removed: FM at $\nfe=1$ (`dpcc-c`) and
  CI-MeanFM at $\nfe=1$ (`dpcc-t`) both hold S&C 1.000 in fewer control steps than the pinned baseline at
  32.0× and 30.6× less compute per step — architecture-matched, and the only place in the thesis a
  flow-based model is ahead of the baseline on every axis at once. CI-MeanFM now has five seeds at this
  protocol, seed 6 only at the extended one.
- **UAV-pillars is deleted from the draft**, not withheld: section, label, summary rows, projection
  paragraph and figure all removed, Chapter 5 adjusted so the scene is introduced as one that yields no
  result. The reason moves into the §6.3 opening. Text preserved at `v3/withheld/`.
- **New `tab:uav-corridor-projection`** (endpoint projection under every rule; per-step still wins, endpoint
  degrades with the budget, no $\nfe=1$ row is possible) and **new `tab:uav-scurve-aborts`** (the re-run is
  all-zero because the vehicle inverts, on the unprojected plan too). `fig:uav-scurve-paths`'s caption
  corrected: the drone flips, it does not stall.
- **`fig:raw-plans` is seven of eight** — four downloaded panels drafted; the eighth is impossible, and the
  caption says so.
- **§6.1.2.1 keeps its label but loses its mathematics**: the equation was a duplicate of §4.10 and the
  reporting convention now lives in new **§5.5.4**; the results section reports only the measured counts.
- **Hotfix, same pass.** (a) **s-curve**: checked the prior analyses — the inversion is attitude-domain,
  and the only gain presets that exist scale the *position* loop by $\pm20\,\%$ and cannot reach the
  attitude loop, so it is a **velocity-setpoint** limit, not a gain-tuning one; **status quo claims kept**,
  two sentences added to say which knob is implicated. The untried cheap test (`pid_const_v` on s-curve,
  one env var) has never been run on that scene. (b) **`fig:expert-uav` and `fig:constraints-uav` redrawn
  against `pillars_xl`** — enforced keep-out 0.35 m with the 0.12 m physical pillar inside it; the pillars
  panel inverts from **4/4 clean to 0/4**, each route crossing by 0.15 m, and the centre channel closes.
  Found while doing it: the `pillars_xl` **verification cell has already run** (`u7xlchk`) — unprojected
  MeanFM at $\nfe=5$ scores 0.00 with ~51 violating steps against 0.90 on the old set, now reported in
  §6.3. (c) **Fig 6.3's last panel needs a run, and "impossible" was too strong** — two plan routes differ
  on whether $K$ is a training or a sampling choice; new `PENDING_20260919_fig63_diffusion_K2_panel.md`
  (ledger R24) gives both options. (d) That uncovered a real error: `fig:k-ladder`'s diffusion budgets are
  **three separately trained checkpoints**, not one model read at three step counts, and the draft
  described the wrong experiment. Corrected in three places; the claim sharpens rather than weakens.
- Checks: 13 files, 243 labels, 33 figures, all mechanical checks pass; 17 stale PNG companions
  regenerated before export, and the two rebuilt scene figures checked by eye. Not compiled. Nothing
  committed.

## v3.40 — 2026-09-18 · UAV-pillars withheld: the scene tests the constraint its own demonstrations were built to satisfy → [`changelogs/v3.40_20260918_pillars_withheld.md`](changelogs/v3.40_20260918_pillars_withheld.md)

- The demonstrated channels are **derived from** the clearance the projection enforces (`trajectories.py:48`: `_Y_L = 0.6 - 0.12 - rotor reach - safety = -1.11`, against a required $|y|\ge1.03$). The plans are feasible before projection, which is why the scene could not order the models.
- `sec:res:uav:pillars` is **withheld**; its three tables and its figure are kept verbatim at `withheld/20260918_uav_pillars_section.tex`. The roadmap, the projection comparison, the quadrotor conclusion and the three summary tables read *withheld*; the quadrotor claim now rests on UAV-corridor. 30 → 29 figures, no orphaned references.
- New `data_status/PENDING_20260918_pillars_geometry_redesign.md` (evidence, the `pillars_xl` radius-0.35 ladder, the download-only figures, the jobs to cancel) and `SLURM_RUNBOOK_20260919_pillars_enlarged.md` (config change, one-cell verification, 10-driver matrix). Ledger row R23; the 18-Sep runbook marks groups C/D/E superseded.
- No code written. `check.py` passes; bundles rebuilt. Not compiled.

## v3.39 — 2026-09-18 · UAV-pillars loses its winner; the expert figures move together; a wrong claim about Fig 6.3 withdrawn → [`changelogs/v3.39_20260918_pillars_no_winner_expert_figs_download_correction.md`](changelogs/v3.39_20260918_pillars_no_winner_expert_figs_download_correction.md)

- **`tab:uav-pillars-best` was the remaining flaw**: one best-of-eleven row per model with a triangle on FM, while the diffusion baseline has **no endpoint row on this scene** — three models on eleven configurations against a baseline with five. Rebuilt as the best configuration **within each block, at every evaluated budget** (`pillars_grid.py::best_per_block`). **No selected configuration is marked, and the text says the scene supports none.** The conclusion, `tab:summary-models`, `tab:summary-combinations` and the projection summary all follow.
- The avoiding demonstrations-versus-constraints figure moved into the datasets section, so the three environments make the same statement in the same place: Figures 5.9, 5.10, 5.11.
- `fig:expert-uav`: the s-curve vehicle was at the arena edge (the whole first passage ties on clearance) — ties now break **toward the obstacle**, putting it at the corner where the route leaves the passage. The three corridor vehicles no longer pile up: each goes to the tightest moment of its own route that is clear of the ones already drawn, and a violating route is never shown at a clean moment.
- **Correction:** the claim that all five missing panels of `fig:raw-plans` need cluster evaluations was wrong — it confused *not on this container* with *never run*. **Four are a download** from named run folders (FM K1/K2 seeds 6–10, CI-MeanFM K1/K2 seed 6); the fifth, diffusion at K=2, **cannot exist**, since the baseline's step count is fixed at training. Fixed in the hole, the caption, the request doc and ledger R5.
- `check.py` passes; bundles rebuilt. Not compiled.

## v3.38 — 2026-09-18 · hotfix: the stale-PNG bug that hid two passes of figure work → [`changelogs/v3.38_20260918_hotfix_stale_pngs_vehicle_marks.md`](changelogs/v3.38_20260918_hotfix_stale_pngs_vehicle_marks.md)

- **A figure ships as `.svg` and `.png`, and LaTeX renders the PNG.** The builders write only the SVG, so a rebuilt figure whose PNG was not regenerated kept printing the OLD drawing while the bundle reported it as rendered. That is why `fig:constraints-aligning` still showed the fixed-size box marker: the correct 0.10 m footprint has been in the SVG since v3.36 and never reached the page. **13 figures were stale**, some since v3.33; all regenerated and re-exported.
- `export_to_draft.py` now **refuses to export** when a referenced figure's PNG is older than its SVG, printing the pairs and the command to refresh each. The failure mode cannot recur silently.
- `fig:expert-uav`: the X2 is now drawn **to scale at each route's tightest moment** (rotor disks + body inside the 0.31 m inflation disk), chosen by minimum clearance, red where the body overlaps. The corridor routes now show the violation as the body sitting inside the slide, instead of a line crossing a line.
- The baseline mark is **bold as well as dotted** — v3.37 left the row in normal weight. Applied in Tables 6.1 and 6.2 and, for consistency, to the diffusion rows of `tab:uav-corridor` and `tab:uav-pillars-best`.
- `fig:raw-plans`: an explicit `\hole` in the draft names the five missing panels and says they need five cluster **evaluations** — a panel is the plan fan an evaluation writes as it runs and exists in no artefact on disk.
- §6.3.1.2 was recomputed from the CSV in v3.36 and is in the current draft; `analysis/pillars_grid.py` reproduces every cell. The changelog lists where to check each part.
- `check.py` passes; bundles rebuilt. Not compiled.

## v3.37 — 2026-09-18 · hotfix: simpler quadrotor panels, a baseline mark, two figures made readable → [`changelogs/v3.37_20260918_hotfix_panels_marks_colours.md`](changelogs/v3.37_20260918_hotfix_panels_marks_colours.md)

- `fig:constraints-uav`: the corridor and s-curve panels were shaded twice over, so the free channel was the hardest thing to see. All three panels now read like the pillar one — **the obstacle drawn once, one grey inflation band, tightened dashed, free space white** — with a faint wash for the rest of a halfspace. The shared panel is used by `fig:expert-uav` and the three flown-path figures, which improve with it. Caption rewritten.
- New `\baselinemark` (filled circle) beside `\selectedmark` (triangle): the circle marks **the baseline of record** (Diffusion $\nfe=20$, cumulative projection cost) in `tab:avoiding-dpcc-protocol` and `tab:state-headline`; the triangle stays this thesis's operating point. The bold that used to sit on the baseline row of Table 6.2 is gone, so bold means one thing again.
- New `sources.ENGINE_COLOUR_DISTINCT` — blue / orange / teal / near-black — used by `fig:avoiding-tradeoff` and `fig:avoiding-raw-models` **only**, because three models in one scatter cannot be three shades of one slate hue. Marker shape still carries the selection rule.
- `fig:avoiding-raw-models` rebuilt at font scale 1.45 on a larger canvas: category labels 7.8 → 10.5 pt, value 11.5 → 12.5, row spacing 11 → 19 px, top margin opened and the subtitle shortened so nothing collides. It is legible at `\linewidth` now.
- No number in the draft changed. `check.py` passes; bundles rebuilt. Not compiled.

## v3.36 — 2026-09-18 · Chapter 8 handed to v4; the box to scale; expert data against the constraints; UAV-pillars recomputed → [`changelogs/v3.36_20260918_ch8_out_box_geometry_expert_data_pillars_redo.md`](changelogs/v3.36_20260918_ch8_out_box_geometry_expert_data_pillars_redo.md)

- **Chapter 8 is no longer built by v3** (author instruction): the `\input` is commented out and the manifest marks the file *own, not built*. The file is untouched for v4; `check.py` passes without it.
- `fig:constraints-aligning` drew the box as a fixed-size marker. It is a **0.10 m square**, larger than the 0.06 m keep-out radius beside it; box and target are now drawn to scale, at their recorded yaw, with the centre marked.
- §5.2.2 now states that the **context is drawn per episode** and that the observation is 6D — commanded and measured end-effector position, **no box or target pose** — so the cameras are the only source of where the box is. Sent to v2, whose Ch 4 implies otherwise and still calls the constraint set a workspace box.
- `fig:raw-plans` is **not waiting on a download**: a panel is the plan fan an evaluation saves as it runs, so the five missing ones need five cluster evaluations. Caption and ledger row R5 say so.
- New `fig:expert-uav` and `fig:expert-aligning`: what the expert data does against the constraints, for the two environments that lacked it. The quadrotor reference paths are clean on pillars (4/4) and s-curve (1/1) but **cross the slide on all three corridor lanes** — a test-time constraint, like D3IL-avoiding's. The direct push crosses the alignment keep-out region in **106 of 120** contexts.
- **UAV-pillars recomputed from the batch CSV** (`analysis/pillars_grid.py`). The numbers were right and the models are architecture-matched, but one mean over two kinds of configuration hid the result: over the seven per-step configurations at K=5, MeanFM and FM are **level at 0.586**; FM's lead is entirely the endpoint block (0.950 vs 0.700), which acts on a plan already collision-free before projection; at K=2 the order reverses. Now two tables covering **every evaluated budget**, with MeanFM/FM at K=1 marked as a hole.
- `check.py` passes; bundles rebuilt. Not compiled.

## v3.35 — 2026-09-18 · the v2.24 sync, and the three quadrotor path figures land → [`changelogs/v3.35_20260918_v2.24_sync_and_uav_path_figures.md`](changelogs/v3.35_20260918_v2.24_sync_and_uav_path_figures.md)

- Inherited half advanced to **v2.24** (`sync_v2.py merge`, four files fast-forwarded). Chapter 4's reorder costs v3 nothing — every reference to `sec:method:deployment` still resolves — but **RQ1 was reworded in v2**, so its answer in §8.1 is rewritten on the new axes: the level is reached at fewer solver steps and less planning time, and the budget orders the three objectives only where the environment separates them.
- Three flown-path figures placed: `fig:uav-corridor-paths` (new), `fig:uav-pillars-paths` (new) and `fig:uav-scurve-paths` (was a `\todofigure`). Three `\hole`s and one `\todofigure` closed; 25 → 28 figures.
- **The figures' goal criterion was wrong and is fixed.** The extract marked "reached" from `success_strict`, the criterion dropped in v3.28, so the panels contradicted their own tables (corridor read 8/12 where the table reads 12/12). `extract/uav_paths.py` now reads `success_relaxed`; after the re-extract every panel count matches its table cell.
- Captions written against the drawings: the s-curve panels are the unprojected plan (the comparison is of the controller), the four pillars panels are **not** under the same projection, and the corridor panels all share the configuration `tab:uav-corridor` reports.
- Still open: the D3IL-aligning box paths (staged before a naming bug was fixed) and `fig:raw-plans`. Ledger D10 is 🟠.
- `check.py` passes; bundles rebuilt. Not compiled.

## v3.34 — 2026-09-18 · hotfix: seed table moved to the results, a cell-by-cell data audit, mean percentages → [`changelogs/v3.34_20260918_hotfix_seedspread_audit_percentages.md`](changelogs/v3.34_20260918_hotfix_seedspread_audit_percentages.md)

- `tab:seed-spread` moved out of the evaluation parameters into §6.1.1.2 where it is used; §5.6.3 keeps one plain sentence.
- New `data_status/PENDING_20260918_verified_data_audit.md`: what was checked, what exists after all, what is confirmed missing, and that nothing has landed since the 15-09 checkpoint — so every open ledger row is still open. Ledger rows R20–R22 added for the quadrotor.
- `tab:va-models` gives the percentage for the mean as well as the median.
- Largest confirmed gap: the quadrotor endpoint-projection matrix, and the diffusion baseline has no endpoint row on any scene.
- `check.py` passes; bundles rebuilt. Not compiled.

## v3.33 — 2026-09-18 · hotfix: the real alignment constraints, the quadrotor's size, a rebuilt evaluation section → [`changelogs/v3.33_20260918_hotfix_constraints_dimensions_eval.md`](changelogs/v3.33_20260918_hotfix_constraints_dimensions_eval.md)

- The alignment constraint figure drew the workspace box; `combined_5` actually enforces a halfspace and a circular keep-out region. Figure, table and text rebuilt from the configuration; the box is now only the plotted extent.
- New `fig:x2-dimensions`: the Skydio X2 to scale, with the 0.54 × 0.62 m footprint, the 0.228 m rotor-centre distance and the 0.31 m inflation circle.
- §5.6.3 rebuilt around one aggregated `tab:protocol`; the per-environment subsections keep their labels and lose the scattered numbers.
- Table 6.1 marks its four missing rows in the table itself; the long hole is one line now.
- The alignment percentage is inverted: it is the share of the initial distance closed, so 100 % is best (MeanFM 0.0741 = 84 %, baseline 0.4140 = 9 %).
- `check.py` passes; bundles rebuilt. Not compiled.

## v3.32 — 2026-09-18 · Chapter 6: tables filled, the cost frontier explained, the pillars result read flight by flight → [`changelogs/v3.32_20260918_ch6_tables_pareto_and_controller_naming.md`](changelogs/v3.32_20260918_ch6_tables_pareto_and_controller_naming.md)

- Table 6.1 gains the baseline at K=1/5/10 (it collapses to 0.60–0.67 at one evaluation); Table 6.2 gains FM at K=5/20 and MeanFM at K=5/10/20, which existed in the batch but were not shown. Remaining gaps are marked must-have (`\hole`) or acceptable (`\guard`), with ledger rows R18/R19.
- §6.1.1.2 and §6.2.1.2 rewritten finding-first; the cost frontier gets its own subsection explaining the non-dominance study.
- Figure 6.2 gains CI-MeanFM (seed 6, own series); Table 6.4 names the rule tokens; the alignment tables carry the distance as a percentage of the initial 0.4530 m, defined in §5.4.2.
- The pillars result checked flight by flight: FM's 1.00 holds, but six of ten flights run to the episode limit, while MeanFM flies nine in ~420 steps and fails one outright — both now stated.
- "Brake-to-rest" named as the cascaded geometric controller's zero velocity setpoint, not a separate controller; the missing rationale sent to v2.
- `check.py` passes; bundles rebuilt, 24/24 figures. Not compiled.

## v3.31 — 2026-09-18 · Chapter 5: metrics before baselines, the alignment constraint set, a training/evaluation split → [`changelogs/v3.31_20260918_ch5_structure_constraints_and_metrics.md`](changelogs/v3.31_20260918_ch5_structure_constraints_and_metrics.md)

- Metrics now come before baselines, each metric with its unit; §5.6 is split into Compute Environment, Training and Evaluation, the per-environment protocols becoming subsubsections of Evaluation.
- New `tab:aligning-constraints` and `fig:constraints-aligning` give the alignment task the constraint set the other two environments already had.
- Table 5.3 widened, Figure 5.7 made legible (own text scale, two-row legend), Table 5.5 given the ± spread the published row has, `tab:train` marks the DPCC baseline column.
- §5.2's opening rewritten, the 0.025 margin attributed to DPCC, D3IL's policies named in §5.4.2, and the quadrotor rate numbers verified line by line against the code.
- MuJoCo is cited everywhere but never introduced: §5.1 now back-references §2.5 and v2 was asked to add the clause.
- `check.py` passes; bundles rebuilt, 24/24 figures. Not compiled.

## v3.30 — 2026-09-17 · Results chapter traversed against the pending-data ledger → [`changelogs/v3.30_20260917_results_ledger_traversal.md`](changelogs/v3.30_20260917_results_ledger_traversal.md)

- Matched every `\hole`, `\provisional`, `\todofigure` and `\guard` in Chapter 6 to a ledger row; added **R12–R17** for six gaps that had none, two of which carry no marker in the draft at all (the baseline's missing pillars configurations, and the pillars budget ladder stopping at K=2).
- Closed **D3** (the MuJoCo MPC overhead claim was removed in v3.28, not verified); marked **D10** as needing the cluster; rewrote stale section numbers as labels.
- Added a coverage table to the ledger mapping all 26 draft locations to their rows. No thesis prose changed.

## v3.29 — 2026-09-17 · v2.23 absorbed, three RQs, captions cleaned, smoothness section, trajectory figures flagged → [`changelogs/v3.29_20260917_v2_sync_captions_smoothness_and_trajectory_holes.md`](changelogs/v3.29_20260917_v2_sync_captions_smoothness_and_trajectory_holes.md)

- Absorbed **v2.23** (`sync_v2.py merge`) and closed the open v2 item: three RQs instead of four, with the budget question folded into RQ1; Ch 8 answers renumbered.
- Table 6.1's caption no longer explains what was excluded; the remaining gap is recorded in the new `data_status/PENDING_20260917_dpcc_protocol_rows_table61.md`, including the `Folder_Name` backbone trap.
- Table 6.4 reordered to MeanFM, CI-MeanFM, FM, Diffusion.
- §6.1.1.4 is now "Smoothness of the Plans" — the same evidence, stated as what the section shows rather than what the metrics miss.
- Trajectory figures flagged: `fig:uav-scurve-paths` placeholder after Table 6.12 plus holes for pillars, corridor and alignment, with `REQUEST_20260917_trajectory_figures.md` naming the cluster runs to download (ledger D10).
- `check.py` passes; bundles rebuilt, 23/23 figures. Not compiled.

## v3.28 — 2026-09-17 · Chapter 6: missing protocol rows recovered, caveat reasoning, goal-passed criterion → [`changelogs/v3.28_20260917_ch6_protocol_rows_metrics_and_goal_passed.md`](changelogs/v3.28_20260917_ch6_protocol_rows_metrics_and_goal_passed.md)

- Table 6.1 filled with MeanFM at K=1/K=2 over five seeds and two episodes: the rows were in the batch all along, hidden because `Folder_Name` does not carry the backbone.
- §6.1 intro trimmed to a roadmap and names the candidate-selection rules as v2 defines them; §6.1.1 separates the diffusion and ODE meanings of K (sent to v4).
- The Table 6.3 guard is now a result; the caveat names the inverse kinematics and reads DPCC's Table 2 as the size of the downstream margin.
- §6.2.1 leads with the finding; Table 6.6 marks its selected row; every model table is ordered MeanFM, CI-MeanFM, FM, Diffusion.
- UAV success is now "the goal passed" (crossing the finish line, as on D3IL-avoiding). **On UAV-pillars this swaps the lead: FM 0.718 against MeanFM 0.627, with MeanFM shorter by 120 control steps.** Chapter 8 and the recap tables follow.
- `check.py` passes; bundles rebuilt, 23/23 figures. Not compiled.

## v3.27 — 2026-09-17 · Chapter 5: platform facts, constraint figure, incentive, metric definition → [`changelogs/v3.27_20260917_ch5_platforms_constraints_metrics.md`](changelogs/v3.27_20260917_ch5_platforms_constraints_metrics.md)

- `tab:platforms`: IK wording checked against the controller code and abbreviated, robot named *Franka Emika Panda* with its cylindrical rod end effector (not a gripper), table widened, self-attacking real-time clause removed.
- §5.2.2 gains the incentive for the alignment environment; §5.4.2 carries D3IL's own success rates (0.359 from images against 0.950 on avoiding from the state); §5.4.3 loses a stray baseline sentence.
- `tab:uav-scenes` is now a constraint table without the workspace box, and new Fig 5.7 draws the three aerial constraint sets like the avoiding one.
- §5.5.1 defines the time metric as DPCC's "time to compute one action" with no caveat; the machine caveat moved to the appendix and the timing caveat to v4.
- Fig 5.5 re-framed from the rollout where the in-hand box is complete and centred.
- `check.py` passes (23 figures); bundles rebuilt. Not compiled.

## v3.26 — 2026-09-17 · MuJoCo wording, Ch 5 holes, no statistics, two caveats, UAV restructure → [`changelogs/v3.26_20260917_mujoco_holes_caveats_uav_restructure.md`](changelogs/v3.26_20260917_mujoco_holes_caveats_uav_restructure.md)

- MuJoCo named as the simulator throughout Ch 5 (Fig 5.5 labels, captions); D3IL = task files, data, cameras.
- Filled the holes on the UAV workspace box and tightening, on demonstration counts, gains and splits, and on the UAV controller; rewrote §5.4.3 (original MuJoCo MPC task).
- Removed every p-value and test name (Ch 5, Ch 6, Ch 8 draft); new translation rules 7–8.
- New caveat subsections: DPCC metrics and smoothness (§6.1), tracking controller (§6.3). §6.2.2 shows endpoint projection as the D3IL-aligning method (`tab:va-projection`).
- §6.3 restructured around all four models per scene with UAV-pillars as the main test; §6.4 is a three-table recap.
- `check.py` passes; bundles `thesis_v3_20260917_161355_new` and `_new_clean`, 22/22 figures. Not compiled.

## v3.25 — 2026-09-17 · thesis-style visuals, Chapter 5 audit, and clean review bundle → [`changelogs/v3.25_20260917_thesis_visuals_ch5_and_clean_bundle.md`](changelogs/v3.25_20260917_thesis_visuals_ch5_and_clean_bundle.md)

- Reworked the Chapter 5 opening and task terminology, filled the alignment workspace bounds, audited
  the UAV demonstration method against v2, and gave Table 5.7 a relaxed dedicated float page.
- Simplified Figures 6.1, 6.2, 6.4 and 6.5 to thesis-style plots, changed the model/constraint palette
  to dark slate, and made selected table configurations unmistakable with a bold black triangle.
- Added a dated v3.25 handoff for v2's Section 1.2 and added an optional clean bundle mode that hides
  source notes while keeping holes visible.
- `check.py` passes; both `thesis_v3_20260917_144454_new.tex/.zip` and
  `thesis_v3_20260917_144454_new_clean.tex/.zip` contain 22/22 rendered figures and verify
  byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.24 — 2026-09-17 · consistent model labels in every figure and table → [`changelogs/v3.24_20260917_figure_table_abbreviations.md`](changelogs/v3.24_20260917_figure_table_abbreviations.md)

- Standardised every v3 figure and table body on **Diffusion**, **FM**, **MeanFM** and **CI-MeanFM**,
  including the Figure 6.3 matrix, appendix mappings, and all four generated Chapter 6 charts.
- Section 5.3 and the affected figure captions explicitly map each short label to its full scientific
  name; running prose continues to use the full mechanism names.
- Updated the canonical translation table and official plotting vocabulary, rebuilt and visually
  checked the four affected charts, and re-exported the figure store to v3.
- `check.py` passes; `thesis_v3_20260917_110958_new.tex/.zip` contains 22/22 rendered figures and
  verifies byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.23 — 2026-09-17 · monochrome table emphasis → [`changelogs/v3.23_20260917_monochrome_table_emphasis.md`](changelogs/v3.23_20260917_monochrome_table_emphasis.md)

- Replaced the red main-configuration styling in Chapter 6 with bold monochrome emphasis for grayscale
  and accessibility robustness. Table 6.2 names its selected operating point directly because bold
  already denotes the best rule per model there.
- Audited the remaining colour commands: they are drafting diagnostics that must be removed before
  submission, not final-text styling; figure colours remain data encodings.
- `check.py` passes; `thesis_v3_20260917_110526_new.tex/.zip` contains 22/22 rendered figures and
  verifies byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.22 — 2026-09-17 · CI-MeanFM table-label hotfix → [`changelogs/v3.22_20260917_ci_meanfm_hotfix.md`](changelogs/v3.22_20260917_ci_meanfm_hotfix.md)

- Replaced the rejected `$\alpha$-MeanFM` table label with the author-approved **CI-MeanFM**:
  consistency-interpolated MeanFM. Full prose remains **consistency-interpolated average-velocity
  matching**.
- Updated Section 5.3, all affected result tables, the canonical translation table and the prior layout
  entry so the rejected label is not left as current guidance.
- `check.py` passes; `thesis_v3_20260917_105825_new.tex/.zip` has 22/22 rendered figures and verifies
  byte-faithful. Not compiled because this container has no TeX toolchain.

## v3.21 — 2026-09-17 · compact UAV floats and result-table model labels → [`changelogs/v3.21_20260917_compact_floats_and_table_labels.md`](changelogs/v3.21_20260917_compact_floats_and_table_labels.md)

- Reduced the footprint of Table 5.3, Figure 5.6 and Table 5.4; the UAV scene figure now occupies
  78% of the text width, and both tables use compact spacing and permissive float placement before the
  existing section barrier.
- Defined the table-only labels **Diffusion**, **FM**, **MeanFM** and **CI-MeanFM** alongside the
  full mechanism names in Section 5.3 and recorded them in the canonical translation table.
- Replaced long model cells throughout the v3 results tables, split the non-wrapping headers in Table
  6.7, and tightened Tables 6.7, 6.8 and 6.10 to prevent overlap.
- `check.py` passes; the new-only bundle `thesis_v3_20260917_104554_new.tex/.zip` contains 22/22
  rendered figures and verifies byte-faithful. Not compiled: this container has no TeX toolchain.

## v3.20 — 2026-09-17 · mechanism names, Chapter 5 layout, and obstacle-avoidance emphasis → [`changelogs/v3.20_20260917_naming_ch5_layout_ch6_highlights.md`](changelogs/v3.20_20260917_naming_ch5_layout_ch6_highlights.md)

- Applied the author-approved names **instantaneous-velocity matching**, **analytic average-velocity
  matching**, and **consistency-interpolated average-velocity matching** throughout v3-owned prose,
  tables, captions, appendix mappings and the affected official figure; updated the canonical
  translation table and sent the Chapter 4 rename to v2.
- Clarified MuJoCo as the common simulator, D3IL as the Panda integration, and the X2 lineage through
  MuJoCo Menagerie and the MuJoCo MPC task patch. D3IL-avoiding is now explicitly the exact DPCC
  benchmark used for controlled validation.
- Kept each §5.2 task's figures inside its subsection, reduced their footprint, and transposed the
  oversized Table 5.5 into a compact protocol-by-metric layout.
- Marked the main §6.1 configurations in red and revised its conclusion: the
  consistency-interpolated objective does show the remembered single-seed edge over the analytic
  objective, but it is not statistically resolved; the analytic objective carries the five-seed result.
- Rebuilt and visually checked `fig_avoiding_raw_models`, re-exported v3, passed `check.py`, and built
  and byte-verified both the full and v3-new-sections-only 22-figure bundles.

## v3.19 — 2026-09-17 · D3IL-avoiding: DPCC's protocol first, every selection rule shown

- 🔴 **Protocol corrected.** DPCC's released evaluation runs `n_trials: 2` for each of five training seeds (`aux_repo/dpcc/config/projection_eval.yaml`, `scripts/eval.py`) — the paper's "five training seeds and ten test seeds" is **10 episodes per geometry**, not 50 as §5.5.1 said. §5.5.1 rewritten: 10 episodes = the protocol of DPCC (primary), 100 = the extended evaluation (stronger evidence).
- **§6.1 restructured** (author): opening states the two protocols; *Generative Models* now has (a) **At the Protocol of DPCC** — new `tab:avoiding-dpcc-protocol` with all three rules $r/c/t$; (b) **Extended Evaluation** — `tab:state-headline` rebuilt with all rules for diffusion K20, flow matching K1/K2, MeanFlow K1/K2, best rule in bold; then α-Flow via `tab:state-models`.
- **Results.** At DPCC's protocol the baseline reproduces the paper: DPCC-C tightened **S&C 1.000 in 70.1 steps** (published 0.98 in 69.0); flow matching at the same K20 reaches 1.000 in 63.2 steps at 477 vs 553 ms. Extended, at each model's best rule: MeanFlow K1 $t$ 0.993 / **61.0** steps / 18.1 ms; flow matching K1 $c$ 0.997 / 67.5 / 18.9; diffusion $c$ 0.983 / 69.0 / 563.5 → **MeanFlow ahead of both; flow matching level with diffusion on the path at 1/30 of its time.** The rule matters most for MeanFlow ($c$ picks stalled 98-step plans at K2).
- `tab:target` gains a *protocol of DPCC* column (1.000 / 70.1 / 0 violations / 0.553 s).
- 🔴 **Data gap, written as placeholder:** flow matching K1/K2, MeanFlow **U-Net** and α-Flow **U-Net** were never evaluated at 5 × 2 (the 5-seed 2-episode MeanFlow/α-Flow runs are DiT/SiT; U-Net α-Flow exists for seed 6 only). *Pending* rows + `\hole`; cluster commands in `DA_in_Paper/plotting/REQUEST_20260917_avoiding_dpcc_protocol.md`; ledger R8 raised to 🔴.
- Removed a truncated sentence ("the diffusion baseline at") and a duplicated flow-matching paragraph from the old §6.1.1.
- New `DA_in_Paper/analysis/avoiding_rules_by_protocol.py` + `INDEX.md` row. v2 notified (protocol wording; a new 14.9 cm table in its Ch 4).

---

## v3.18 — 2026-09-17 · v2.17 sync; projection-cost fix; UAV rates in simulated time; α-Flow naming

**Sync.** v2.16 → **v2.17** (7 inherited files, all fast-forward, merged by name). Acting on v2.17's notes:
- 🔴 removed v3's duplicate `tabularx` / `\newcolumntype{L}` / `secnumdepth` — the inherited preamble now defines them, and a second `\newcolumntype{L}` is a LaTeX error;
- 🔴 **projection-cost contradiction** (v2's `to_v3` note): the `\guard` below `tab:hf-ladder` quoted "1.86–3.57× the cost at equal candidates" — that is **UAV-corridor** data (DA_20260824 §5), misattributed. The D3IL-avoiding K3/K5 comparison gave both methods **4 candidates** (`…_A1_B4_…hfmink…`), so "about half the time" stands. Guard corrected; resolution + a consistent cost sentence sent to v2; inbox item ✅.

**"33 Hz plan / 100 Hz controller" (author: "99 % fake").** Checked in code: *not invented, but misleading*. Physics dt 0.01 s; planner every `decim = 3` steps (0.03 s), the demonstrations' recording interval (`dataset_writer.py:73`); tracker every physics step (`eval_mix_uav.py:1609–1735`). But the evaluation is **lock-step** — the simulation waits for the planner, which takes 9–4878 ms wall clock (21 % of cells under 30 ms). `tab:platforms` now states intervals of simulated time and says so; `tab:uav-demos` "(33 Hz)" → "(0.03 s)". v2's Ch 4 rate table/equation flagged in `cross_draft`.

**Naming.** Read from the papers: MeanFlow is the paper's own name (average velocity) — kept; "mean flow matching" is not a term. What we call "consistency training" is **α-Flow** (a family between trajectory flow matching α = 1 and MeanFlow α → 0; α = consistency step ratio), and "Consistency Training" is a *different* published method (Song et al.). Renamed across Ch 5, 6, 8, appendix: **α-Flow**, final value **α_end = 0.2 / 0.05** (was "floor"); headings via `\texorpdfstring`; one definition of all three flow models at §5.3; figure labels (`fig:avoiding-raw-models`) rebuilt. Translation table §2 and rule 1 revised; v2 notified (10 occurrences, §4 title); memory updated.

**Tooling.** 🔴 The rename's first pass wrote BEL control characters (Python expanded `\a` in `\alpha`); caught by reviewing the diff, repaired (46). `check.py` now fails on control characters — verified with an injected one.

check.py pass; bundle `thesis_v3_20260917_*_new.zip`, 22/22.

---

## v3.17a — 2026-09-16 · hotfix: remaining environment prefixes; D3IL credit on the camera figure

- **Prefixes, second pass** (paragraph-level scan, catches names split across lines): Ch 5 intro now introduces D3IL-avoiding, D3IL-aligning, UAV-corridor/-pillars/-s-curve; the UAV scene description list; §5.3.3 "as on D3IL-aligning"; `tab:train` caption; Fig 5.4 caption; Ch 6 intro; D3IL-avoiding conclusion; D3IL-aligning budget text; RQ2. Object descriptions ("walls and pillars", "pillar pairs") left as they are.
- **Fig 5.5 (`fig:aligning-cameras`)**: caption credits D3IL — "Adapted from D3IL (appendix, Fig. 8): the camera set-up and the two views are D3IL's; the frames are rendered in the alignment environment of this thesis". Panel labels and §5.1.2 text use D3IL's names, *front view* and *in-hand view*, replacing "fixed overhead / wrist-mounted camera".
- check.py pass; bundle 22/22.

---

## v3.17 — 2026-09-16 · seed policy: five seeds shown once, single seed stated once

Follows the author's multi-seed assessments (`Gen14/Campaign_20260916_va_5seed_assessment/`, `Gen15/Campaign_20260916_uav_5seed_training/`): no further seeds for D3IL-aligning (cost) or UAV (cluster disk).

- **§5.5 intro:** one sentence — D3IL-avoiding over five seeds; D3IL-aligning and UAV with seed 6 "within the storage and compute available to this work". No further justification.
- **§5.5.1:** new `tab:seed-spread` + paragraph, over seeds 6–10 on the two geometries every seed covers: MeanFlow K1 S&C 0.995 ± 0.011, 62.1 ± 1.8 steps, 18.0 ± 0.2 ms; flow matching K2 68.6 ± 2.2 steps; diffusion 73.8 ± 11.3 steps; diffusion/MeanFlow time 28–33× in every seed. Script `DA_in_Paper/analysis/avoiding_seed_spread.py`, `INDEX.md` row.
- **Removed** single-seed caveats in Ch 6 (six `\guard`/`\provisional` sentences and one conclusion sentence) and Ch 8; non-seed content of those caveats kept. Neutral protocol statements ("seed 6" in captions and tables) kept.
- Ledger: R1 and R4 closed ❌ with reasons; seed policy recorded. `cross_draft/to_v4` findings updated so the Discussion does not re-open it.
- check.py pass; bundle 22/22.

---

## v3.16b — 2026-09-16 · environment prefixes where environments appear together

- Defined once at the start of §5.1: **D3IL-avoiding, D3IL-aligning, UAV-corridor, UAV-pillars, UAV-s-curve** (UAV spelled out).
- Applied wherever several environments are shown or compared together: §6.4 prose and all three summary tables, the UAV conclusion and cross-scene consistency-training text, Ch 8 (summary and RQs), `tab:train` / `tab:eval` headers, §5.5.3, `tab:uav-scenes`, `tab:uav-demos`, Fig 5.4 panel labels, appendix `tab:corpora`. Single-environment sections keep plain names; object descriptions ("corridor walls") unchanged; `\autoref`/`\label`/`\texttt`/`\dataref` untouched.
- Recorded in the translation table; v2 notified in `cross_draft/INBOX.md`. check.py pass; bundle 22/22.

---

## v3.16a — 2026-09-16 · review of v3.16 (Codex) + isolated platform renders

- **Review: v3.16 holds.** Platform specs re-derived from `quadrotor_modified.xml` (1.325 kg, rotor centres 0.228 m, 0.54×0.62 m, 0–13 N) and `panda.xml` (wrist camera); camera frames are from our own expert-replay GIF; the Ch 6 conclusions match `va_results.py` / `uav_results.py`; check.py and bundle pass.
- 🔴 **Fixed a silent defect of the restructure:** the new `\subsubsection` parts were unnumbered under `scrbook` defaults, so 13 `\autoref`s would print the parent subsection's number. `secnumdepth` raised in `parts/00_preamble_v3.tex`; logged for v2 in `cross_draft/INBOX.md`.
- **Fig 5.1** now shows the Panda and the X2 on their own — MuJoCo renders of the model files with the background removed by the segmentation pass (`PLATFORM_RENDERS`, `prep/render_mujoco_scenes.py`). This also removes one use of the D3IL README GIF.
- Fixed `render_mujoco_scenes.py --check`, which always reported the pillars render stale (nested tuples). 5/5 current.
- Bundle `thesis_v3_20260916_142003_new.zip`: 22/22 figures render.
- ⚠️ Still open: Figs 5.2 and 5.3 use crops of D3IL's own README GIF, captioned "authentic simulator frame", without credit.

---

## v3.16 — 2026-09-16 · platforms, alignment camera inputs, and two-part result structure → [`changelogs/v3.16_20260916_platforms_cameras_results_structure.md`](changelogs/v3.16_20260916_platforms_cameras_results_structure.md)

- Chapter 5 now introduces the Panda and Skydio X2 with source-derived specifications and shows the two
  actual $96\times96$ alignment observations from one expert-demonstration instant.
- Every Chapter 6 environment now follows **Generative Models → Projection Methods → Conclusion**, with a
  final table of supported model--projection operating points.
- Obstacle avoidance explicitly keeps per-step projection at its one/two-evaluation operating point;
  alignment and UAV state their supported combinations and evidence limits.
- The s-curve section now isolates the tracking-controller effect and distinguishes the flyable
  unprojected plan from the unresolved projected stack.

---

## v3.15 — 2026-09-16 · §6.2 alignment and §6.3 quadrotor rewritten from the 15-09 data → [`changelogs/v3.15_20260916_va_and_uav_results.md`](changelogs/v3.15_20260916_va_and_uav_results.md)

- Every §6.2–6.3 number recomputed by two scripts in `DA_in_Paper/analysis/`; analyses of record reproduced exactly.
- Alignment: MeanFlow ahead of both; flow matching ≈ diffusion; consistency training level at K2, behind at K20. Evaluated on **training** contexts — disclosed.
- Quadrotor: corridor and pillars written in full (MeanFlow > flow matching > consistency training on pillars); s-curve as a listing with the controller discussion.
- 🔴 Five false statements in the previous draft corrected (diffusion violation-freedom on alignment; pillars "left the constraint set"; MF>FM>diffusion on alignment; ≥2 guiding steps; pooled ablations).

---

## v3.14a — 2026-09-16 · plan matrix and Codex handover

- **`fig:raw-plans` is now a 2×4 matrix**: MeanFlow, flow matching, diffusion, consistency training × $\nfe\in\{1,2\}$. Three panels exist (MeanFlow K1/K2, diffusion K1); the other five are visible `\todofigure` placeholders, registered in `PLANNED`, with the cluster runs in `DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md`. Laid out 2×4 rather than 4×2, which would not fit on a page.
- `tools/check.py`: panel grids of minipages are now measured by their declared widths (the character estimate had flagged this figure as 35 cm wide); verified to warn at an oversized width.
- **Plan** for §6.2 alignment and §6.3 quadrotor results (first written as a Codex handover, reassigned to v3 by the author): `plans/PLAN_20260916_VA_and_UAV_results.md` — sources, expected story, known traps (held-out split, an unexplained `d3il_baseline` engine in the alignment data, consistency training below flow matching on pillars, the partially invalid MPC-vs-PID report), rules and deliverables.

---

## v3.14 — 2026-09-16 · MuJoCo renders, §5.3 baselines, §5.5 parameters → [`changelogs/v3.14_20260916_mujoco_renders_baselines_protocol.md`](changelogs/v3.14_20260916_mujoco_renders_baselines_protocol.md)

- Fig. 5.4 top row: real MuJoCo renders with the generator's reference path; 4 SVG-only figures given PNGs (18/18 render).
- Fig. 6.3 flow matching: no plan data off the cluster; exact run request written, panels registered as planned.
- §5.3.2 D3IL reference and `tab:va-vs-d3il`; §5.3.3 MuJoCo MPC quadrotor task as precedent.
- §5.5 `tab:train` / `tab:eval` from the training logs. 🔴 Training is **not** uniform across models (batch, lr, action weight, EMA) — disclosed in a `\guard`.

---

## v3.13 — 2026-09-15 · authentic environment views, figure explanations, raw-plan source boundary → [`changelogs/v3.13_20260915_environment_renders_and_figure_explanations.md`](changelogs/v3.13_20260915_environment_renders_and_figure_explanations.md)

- **Chapter 5 now shows all three environments.** Authentic D3IL frames cover obstacle avoidance and
  alignment; authentic rollout frames cover corridor v2, pillars and s-curve. Alignment and UAV views
  are paired with readable geometry reconstructions whose exact context/MJCF sources are audited.
- **Figures 6.1 and 6.2 are explained in the prose.** Figure 6.2 no longer mislabels its smaller
  five-seed by two-episode diffusion sample as the published DPCC protocol.
- **Figure 6.3 now includes MeanFlow at $K=2$.** Naïve flow matching at $K=1$ and $K=2$ cannot be
  rendered from this checkout: neither diagnostic panels nor saved plan states exist locally. Both
  cluster render sources are registered as `PLANNED`; no panel was fabricated from aggregate rows.
- Full figure store rebuilt and 18 referenced figures exported; mechanical checks pass with no
  `\todofigure`. No TeX toolchain is installed, so this pass was not compiled.

## v3.12 — 2026-09-15 · consistency training into the results, the protocol made exact, two figures rebuilt → [`changelogs/v3.12_20260915_af_results_protocol_and_figures.md`](changelogs/v3.12_20260915_af_results_protocol_and_figures.md)

- **Consistency training was missing from the results.** Built in from `Report_20260903_AF_UNet`: the
  new four-model Table 6.2 (`tab:state-models`) and a new figure `fig:avoiding-raw-models` — goal
  reached with the projection off, 20/20 against MeanFlow's and flow matching's 17/20 and the baseline's
  12/20. 🔴 Caught while building it: averaging each model over the seeds it happens to have reads
  0.97/0.97/0.92 instead of 0.85/0.85/0.60, because only the consistency-training folders are seed-6-only.
- 🔴 **§5.5.1: ten episodes is not "the published number".** The paper is five training seeds x ten test
  seeds = **50 rollouts per geometry**; our two settings are 5x2 = 10 (smaller) and 5x20 = 100 (twice).
  Set out explicitly, and the same wrong label removed from `sources.py`.
- **Figure 6.3**: the flow-matching and consistency-training plan panels have never been rendered
  (`Report_20260903` section 8, all eight still placeholders). Caption now says which models it shows;
  both missing panels registered in `PLANNED` with their plan directories.
- **Figure 6.4 rebuilt**: banned vocabulary on the canvas ("arm", "genuine", "degenerate", "not
  citable"), "lower is better" in an axis label, a truncated subtitle, and 🔴 a semantic error — both
  series were drawn hollow at K=2, though degeneracy applies only to endpoint projection.

---

## v3.11 — 2026-09-15 · v2 checked (unmoved), §6.1.1 renamed, smoothness argued with the baseline's own margin, endpoint projection at low budget → [`changelogs/v3.11_20260915_smoothness_margin_and_lowK_endpoint.md`](changelogs/v3.11_20260915_smoothness_margin_and_lowK_endpoint.md)

- **v2 has not moved** — still v2.16, both hashes identical to `SYNC_STATE.json`; nothing to merge. Read
  it anyway: it leaves a `\hole` at `:332` for a result sentence from Chapter 6, now supplied in
  `handover/HANDOVER_20260915_for_v2_result_sentences.md`. ✅ **v2 is not missing the low-budget
  endpoint-projection point** — Ch. 1 `:331`, Ch. 3 `:590` and §4.5.4 `:1594–1641` all carry it; what it
  lacks is the result, which it explicitly delegates to v3.
- **§6.1.1 "Generative Models against the Baseline" → "Flow Matching against Diffusion"**: the baseline
  is a diffusion model, so the old title implied a distinction that does not exist.
- **§6.1.3 → "Smoothness of the Plans before Projection"**, now with a number behind the claim that the
  control stack hides plan quality: DPCC's own Table 2 — a dynamics model wrong by a factor of four
  still satisfies the constraints in 0.77 of episodes. Cited as `\textcite{romer2025diffusion}` and read
  off the PDF this pass.
- **The $\nfe=10$ endpoint-projection evaluation existed and was missing from the draft.** Added as
  `tab:hf-ladder` with the guiding-step count per budget, the measured solve counts confirming zero at
  $\nfe\le2$, the bit-identical $\nfe=2$ parity run at 2.45× cost, and a `\guard` for the candidate-count
  mismatch.
- **Headline stated**: this task's operating point is one network evaluation, MeanFlow is at $0.993$
  there, and endpoint projection has no step to act on at that budget — in §6.1's opening, §6.1.4's
  close, §6.4.2 and RQ3.

---

## v3.10 — 2026-09-15 · the plans figure cut to its panel, "genuine steps" renamed, checkpoint data recorded → [`changelogs/v3.10_20260915_raw_plans_cut_and_guiding_steps.md`](changelogs/v3.10_20260915_raw_plans_cut_and_guiding_steps.md)

- **Where the lines in the obstacle-avoidance figures come from**, settled by diffing against
  `/workspaces/aux_repo/dpcc/`: the 96 curves are recorded D3IL demonstrations, the constraint geometry
  is byte-identical to DPCC's config, the goal line and obstacles come from a helper identical to
  DPCC's. No model output, nothing invented — the evidence for Chapter 5's "1:1" claim.
- **Figure 6.3 rebuilt.** It was shipping a 3000x1000 six-panel diagnostic dashboard; the panel the
  section argues from was ~12 mm wide. Now cut to that panel, with the cut **declared** in
  `VENDORED_CROP` and applied by a new `plotting/prep/crop_vendored.py`; `make_figs.py` refuses to fall
  back to the uncut image. 🔴 Its caption claimed "every plan" — the renderer draws every fourth control
  step, up to four candidates each; corrected.
- **"Genuine steps" renamed to guiding steps** ($n\sidx{guide}$), §6.1.5 retitled *Steps at Which
  Projection Guides Sampling*; Chapter 8 and the appendix follow; recorded in the translation table.
  Chapter 4 is v2's and is written up in `handover/HANDOVER_20260915_for_v2_ch4_naming.md`, not edited.
- **`analysis_results_checkpoint/15-09` recorded** as the data source of record (committed, all three
  environments); no figure re-pointed, and its long-format schema noted as the obstacle to doing so.
- Fixed the bundler's figure accounting: it counted files, not figures, and warned about placeholders
  that were not placeholders. This build: 7 figures, 7 render, 0 placeholder.

---

## v3.9 — 2026-09-15 · quadrotor dataset values, tables that fit, all three geometries, avoiding figures → [`changelogs/v3.9_20260915_tables_geometries_avoiding_figures.md`](changelogs/v3.9_20260915_tables_geometries_avoiding_figures.md)

- Synced v2.15 → **v2.16**; §5.2.3 now holds the quadrotor demonstration values (`tab:uav-demos`), read
  from the collector code. *Cubic fillets* corrected to circular arcs of radius 0.45 m.
- **Table 6.1's overflow fixed**: non-wrapping `\multicolumn` notes about 20 cm wide. All 13 tables are
  now `tabularx` at `\linewidth`; notes moved into captions; `check.py` gained a width warning.
- **All three geometries** in every obstacle-avoidance table, and a new `tab:avoiding-geometries` in
  Chapter 5. 🔴 The claim that MeanFlow and consistency training are also *faster per episode* than flow
  matching holds only on top-left-hard; corrected in the text and in `tab:summary-models`.
- **Obstacle-avoidance figures built**: the scene with its 96 demonstrations, the three constraint
  geometries, and the control-steps-against-time grid. New extraction step and an SVG preview renderer in
  `DA_in_Paper/plotting`.

- **For v2:** `tools/check.py`'s new table-width guard reports one row in Chapter 4's notation table
  (`04_method.tex:106`): the `\multicolumn{3}{l}{\emph{deployment ... so the collisions are visible}}`
  subheading is about 18.6 cm of unwrapping text against a text width of about 14.7 cm, so it likely runs
  into the margin. A `p{}`/`tabularx` column or a shorter subheading fixes it.
- **For v2:** v3's tables use `tabularx`; `\usepackage{tabularx}` and the `L` column type live in
  `parts/00_preamble_v3.tex` and must be folded into `settings.tex` when the drafts merge.

---

## v3.8 — 2026-09-14 · figures and plotting moved to `Data_Analysis/DA_in_Paper` → [`changelogs/v3.8_20260914_figures_move_to_DA_in_Paper.md`](changelogs/v3.8_20260914_figures_move_to_DA_in_Paper.md)

- `DA_in_Paper` is now the official store: `analysis/INDEX.md` (17 thesis results, links checked),
  `plotting/` (moved from `v3/plots/` + `v3/tools/svg2pdf.sh`; output byte-identical), and
  `figures/{da,demo,env,schematic}/`.
- New `plotting/export_to_draft.py`: `v3/figures/` now holds only the four figures v3 includes, as
  copies with `EXPORTED.md`.
- Concise pointer note in `data_status/`.

---

## v3.7 — 2026-09-14 · Chapters 5–8 restructured by environment; Discussion handed to v4 → [`changelogs/v3.7_20260914_ch5-8_restructure.md`](changelogs/v3.7_20260914_ch5-8_restructure.md)

Full KP-by-KP record in the file above.
- **Synced first:** v2.9 → **v2.15**, inherited files merged one by one.
- **Ch 5:** rebuilt as five sections × the three environments, in study order (DPCC's benchmark
  unchanged → two environments built for this thesis). DPCC's published baseline row sits next to v3's
  reproduction. The caveats, the regime taxonomy and the global definition of "better" are deleted.
- **Ch 6:** one section per environment, plus 6.4 answering both comparisons per environment.
  Quadrotor follows corridor (new U16 results) → pillars → s-curve.
- **Ch 7:** headings only, now v4's; the old prose is in `handover/`.
- **Ch 8:** concise.
- **Shared notes:** 12 translation rows; `DRAFT_OWNERSHIP.md` revised.

- **For v2:** no label v2 references was removed. Where they now sit:
  - `sec:res:fewstep` → *Subsection* 6.1.2 *Step Budget*;
  - `sec:res:constraints:degenerate` → *Subsection* 6.1.5 *Genuine Steps*;
  - `sec:res:constraints` → *Subsection* 6.4.2 *Projection Methods*.

  `\autoref` will print "Subsection" for these.
- **For v2:** `sec:disc:interpretation` and `sec:disc:negative` are now headings with no text (Ch 7 is
  v4's). Ch 4 line ~506 cites `sec:disc:negative` for the consistency-training result; that result is
  in `sec:res:summary:models` (6.4.1).
- **For v2:** two of Ch 4's `\hole`s point at `sec:setup:protocol` and are now answered in its
  subsections: the sampling-prior question in `sec:setup:protocol:avoiding`, and which controller
  produced the UAV results in `sec:setup:protocol:uav`.
- **For v2:** v2.13 asked for one result sentence per contribution. Not yet — the alignment and
  quadrotor results rest on one training seed.

---

## v3.6 — 2026-09-14 · the default bundle builds only the new sections

**Asked for:** since v3 does not touch the v2 sections in normal editing, the default bundle should
build only the new sections and mark the v2 ones in the `.tex` as v2 — if that premise is right.

**The premise is right, with one qualification that the tool now enforces.** All seven inherited
files are byte-identical to `inherited/v2_base/` today. But `02_background` and `04_method` carry the
`merge` policy — v3 is *allowed* to edit them (the planned `sec:bg:fewstep` change) — so "untouched"
is true now, not guaranteed. The bundler therefore checks it on every build instead of assuming it.

### What changed in `bundle/make_bundle.py`

- **Default is now new-sections-only; `--full` builds everything.** Files are named `_new` / `_full`.
- **A v2 chapter collapses to a skeleton:** its numbered `\chapter`/`\section` headings with their
  labels, plus a grey `\fmpccvtwo` box — *"v2 SECTION — not built in this bundle"* — naming the file,
  its length and the v2 revision it was inherited at (read from `SYNC_STATE.json`). The macro is
  injected into the preamble only when something was collapsed.
- **Guard 1 — only v2 content chapters are candidates.** Ownership comes from `tools/sync_v2.py`'s
  `POLICY`, imported rather than restated, so the bundler cannot disagree with the sync tool.
  `parts/` is always inlined: the document does not compile without the preamble, the front matter
  (`\mainmatter`, the ToC) or the back matter (the acronym list).
- **Guard 2 — collapse only if byte-identical to the v2 baseline.** An edited chapter is inlined in
  full and the build prints `inlined IN FULL because v3 has edited them: …`.
- **`--verify` understands collapsed chapters:** no text to diff, so it compares the source's SHA-256
  against the one recorded in the banner.
- `--list`, `--prune` and the `--verify` default order bundles by modification time, since a `_full`
  and a `_new` built in the same second would otherwise sort by name.

### Why the skeleton keeps headings rather than dropping the chapters

Dropping Chapters 1–4 outright would renumber everything: *Experimental Setup* would become Chapter 1,
and all nine `\autoref`s from v3 chapters into v2 ones would print `??`. Every one of those nine is a
`sec:` reference, so keeping the numbered headings with their labels resolves all of them. Starred
headings move no counter and are dropped.

### Verified, not assumed

| check | result |
| :-- | :-- |
| new-sections-only build | 10 files inlined, 4 v2 chapters collapsed (1 821 lines not built); 2 347 lines against 4 108 |
| cross-references in the new-only build | **all resolve** |
| numbered headings, new vs full | **58 and 58, identical sequence** — so every chapter and section number matches |
| `--verify`, new-only bundle | 9 inlined byte-faithful + 4 collapsed unchanged |
| `--verify`, full bundle | 13 inlined byte-faithful |
| **guard test** — appended one line to `02_background.tex` | `--verify` on the old bundle: `DIFF … collapsed v2 section; the source has changed since the build`. Rebuild: `inlined IN FULL because v3 has edited them: chapters/02_background.tex`, the other three still collapsed. File restored, identical to baseline again |

**Use `--full` for anything that has to be the whole thesis** — the Overleaf upload of record, a
supervisor copy, and submission. **Not compiled.**

---

## v3.5 — 2026-09-12 · raw plan quality: the half of the headline that was missing

**Asked for:** the smoothness of the raw network output matters — a chaotic, low-quality plan can be
masked downstream by the IK/controller, while MeanFlow holds the same or better quality — with the
figures from `Report_20260819_MF_UNet` (and the consistency target, if applicable). Write it into the
tex if absent, and into a note so it is not lost. Skip either if already done.

**Checked first, as asked: neither was done.** The tex mentioned smoothness exactly once, in
`sec:disc:limitations`, and only to say no metric exists — the *argument* was absent. v2's tex: zero.
The notes: one clause inside the readiness ledger's list of limits. So both jobs were needed.

### Why this was a real gap, not a nice-to-have

Every number in \autoref{ch:results} is measured after a projector **and** a tracking controller,
and both are low-pass. Without this section a reader can reasonably ask whether the $31\times$ is
bought by degrading the plan. The evidence says the opposite, and it is the missing half of the
headline: **the MeanFlow engine is not merely cheaper at one evaluation — its plans are usable at one
evaluation, where the baseline's are not.**

### 1 · `sec:res:state` — a new subsection, *The plans themselves, before any projection*

With `fig:raw-plans`, two panels side by side at one network evaluation, projector off: MeanFlow's
replans form a tight ribbon around the executed path; the diffusion baseline's are a high-frequency
scribble across the workspace. Matched backbone size, matched budget, and on this cell matched
compute — 0.0097 against 0.0094\,s per step, three per cent apart.

Quantitatively: **6.0 unprojected violations against 28.0, a factor of 4.7**, at 58.0 against 56.5
steps. A second evaluation brings MeanFlow to 12.0 and tightens the ribbon, so the one-evaluation
panel is the low end of a trend, not a lucky sample.
\dataref{Report\_20260819\_MF\_UNet \S7}

**The load-bearing sentence:** the baseline's *commanded* traces are smooth, but they are smooth
because the controller integrates the scribble away, not because the plan was good.

The consistency target's raw arm is reported too — 20/20 goals at one evaluation against MeanFlow's
and flow matching's 17/20 and the baseline's 12/20 at its full budget — with a `\guard` stating that
this is three episodes on one seed, that **the margin does not survive projection**, and that the raw
arm therefore supports *the few-step objectives draw better plans than diffusion* and **not** an
ordering among them. That keeps it consistent with the standing negative result.
⚠️ The successes-only step basis is named, because the toolchain has two `n_steps` definitions and
mixing them is a real error.

### 2 · `sec:disc:threats` — the general form

New subsection, *Downstream metrics can hide the generator*: an evaluation that measures only what
survives the control stack measures **the stack's tolerance** as much as the planner's quality.
Reporting the unprojected arm costs one configuration per cell. This also bounds the aerial entry —
with a tracking error of 0.30--0.49\,m, that stack absorbs more than any scene leaves as clearance.

### 3 · Two smaller consequences

`sec:setup:metrics` now says *why* stage 1 of the funnel is evaluated on the unprojected arm, and
that the reason applies beyond the all-fail regime. `sec:disc:limitations` was upgraded from "no
smoothness metric is reported" to naming the specific missing measurement — **jerk, path length or
curvature over the saved plan files, which needs no new runs**, only a pass over plans already on
disk. It is now the cheapest outstanding measurement in the project.

### 4 · Vendored figures, with provenance enforced

These four panels come from the evaluation's own diagnostics and cannot be rebuilt from a CSV, so
they are **copied, not redrawn**. `plots/sources.py` gains a `VENDORED` registry — destination name,
source path, and a provenance string — and `make_figs.py` copies them and prints them in their own
table in `figures/MANIFEST.md`.

🚨 **The registry exists because of the v3.3 near-miss.** All four were checked against
`/workspaces/aux_repo/` and are ours; the rule is now explicit in the registry's own header, since
`figures/avoiding*.png` at the repo root are byte-identical to the baseline authors' copies.

### 5 · The note

`Auxiliary/NOTES_plan_quality_and_smoothness.md`, registered in `Auxiliary/README.md`: the claim, the
two tables, the ladder warning, the `n_steps`-basis trap, the owed measurement, the figure-provenance
rule, and the standing caveats (unprojected arms are unconstrained, so raw violation counts are never
a safety result; the raw gap and the projected gap are different sizes).

### 🔴 A bug I introduced, caught by the tooling rather than by reading

The new figure was written with `\fmpccgraphic` in the **source**. That macro is injected by
`bundle/make_bundle.py` and does not exist in a direct build of `thesis_v3.tex`, so the bundle would
have compiled while the split draft failed — the worst way for this to break, because the artefact
that gets uploaded still works.

`make_bundle.py --verify` flagged it: it un-rewrites `\fmpccgraphic` to `\includegraphics` before
diffing, so a source that already said `\fmpccgraphic` came back as a mismatch. `check.py` also
under-counted the figures, since it looks for `\includegraphics`. Both symptoms, one cause.

Fixed to `\includegraphics`, and **`check.py` now fails on any bundler-only macro appearing in a
source file** — tested by reintroducing it, which the guard catches by name and file.

### Checks

`tools/check.py`: 4 019 lines, 146 labels, **4 figure references**, 15 `\hole`, 20 `\guard`,
20 `\dataref`; all references and citations resolve; braces and environments balanced.
6 generated + 4 vendored figures. Both bundles rebuilt; `--verify` byte-faithful. **Not compiled.**

---

## v3.4 — 2026-09-12 · absorbed v2.7–v2.9, and applied their critique to v3's own chapters

**Asked for:** two jobs. (1) Sync v2's updates into v3 and move the version marker. (2) See whether
v2's new critiques — the *"bootstrapped target is jargon, not the scientific name"* one in
particular — also apply to the sections v3 wrote.

### Job 1 · Sync

Six files moved in v2 (`01_frontmatter`, `99_backmatter`, `01_introduction`, `02_background`,
`04_method`, `bibliography.bib`), all **fast-forwards** — v3 had edited none of them. Merged clean.
Absorbed v2.7, v2.8 and v2.9; baseline stamped at **v2.9**. 3 939 lines; 28 bibliography entries,
all cited.

🔴 **The stamp was wrong on the first attempt, and the tool was at fault.** `v2_version()` took the
*first* `## v2` heading, and v2's changelog is not strictly newest-first — a stale `v2.6` entry sits
above `v2.9`. v3 would have recorded itself as inheriting from v2.6 while in fact holding v2.9. The
parser now takes the **highest** version, and `status` prints a note when the changelog is out of
descending order. A version marker that can be silently wrong is worse than no marker.

One thing the merge could have broken and did not: v2.7 **deleted the `\acro{PID}` declaration**
(the controller is not a PID). No v3 chapter used `\ac{PID}`, so nothing dangled.

### Job 2 · v2's critique, applied to Chapters 5–8

**A · The coinage. `bootstrap` 21 → 0.** v2.8 established that *"bootstrapped target"* was this
draft's own coinage and hid the method: it is ordinary **consistency training** — the identity closed
by a **finite difference at an intermediate transport time, taken from a stop-gradient copy of the
network**, where MeanFlow closes it with an analytic derivative. v3 was still using the retired term
21 times across five chapters, in prose, in a results table, and in `tab:names`.

Renamed to **the consistency target** throughout, and the two places that state the *mechanism* were
rewritten to v2's account rather than merely relabelled — the old text explained the failure as a
*"field bias of order $(1-\alpha)$"*, which was my own formulation of it. It now reads: a
finite-difference target closed with the network's own stop-gradient output **inherits a fraction of
that network's error**, at a rate set by $\alpha$; the inherited term does not shrink as the interval
shrinks, so it is budget-independent, and the objective admits self-consistent fields that are wrong.
`plots/sources.py` renamed with it, so figure legends and prose cannot diverge.

**B · The controller's name.** v2.7 established that `CascadedPID` is a misnomer on three counts (no
integral term anywhere; the inner loop is a coordinate-free $SO(3)$ error, not a scalar loop; and
"cascaded PID" implies nested scalar loops). v3 said *"cascaded geometric controller"*; now
**"the cascaded geometric tracking controller"**, matching v2 exactly.

**C · The tone.** v2.9's finding — the prose had been arguing with an imagined examiner — applied to
v3's chapters at least as much. Swept to zero:

| pattern | before | after |
|---|---|---|
| `"worth stating / worth naming"` · `"It is worth"` | 4 | **0** |
| `"is not a defect"` · `"is not a post-hoc excuse"` | 3 | **0** |
| `"must not be"` · `"does not claim"` · `"makes no claim"` | 3 | **0** |
| `"is what makes"` (justifying form) | 8 | **0** |
| `"rather than"` | 58 | **48** (survivors are factual contrasts: *CPU-limited rather than accelerator-limited*) |

Representative rewrites, keeping every fact and dropping the defending:

- *"The taxonomy is not a post-hoc excuse for weak cells. It is diagnostic, it is measured rather than
  asserted … and it is used in three chapters."* → *"The regime is measured, not assigned: an
  all-pass scene is one where the violation counter reads zero on the unprojected arm."*
- *"Retiring it as a ranking scene is the correct action, and keeping it as the demonstration … is a
  stronger use of it than a failed ranking would have been"* → *"It is kept as the demonstration of
  where the control stack runs out."*
- *"That is not a defect of the experiment — both arms are reported — but it is a reason …"* → *"Both
  arms are reported, and the comparison is a pair rather than an ordering."*
- *"Two claims explicitly not made"* → *"Two results that carry no claim"*, with the defending
  sentences deleted and the numbers left to speak.
- One real redundancy surfaced by the sweep: the sentence explaining that the halfspaces are absent
  from the demonstrations appeared **twice**, 130 lines apart. One survives.

### Not done

v2 §2.4 still carries *"No mathematics is given for it in this draft"* for the consistency target,
so the deferral v3.0 logged as *"first item of v3.1"* is still open **in v2**, which owns that
chapter. v2.8 did add the mechanism to Engine 3, so the inconsistency is smaller than it was. The
one-way rule holds: that edit belongs in v2 and syncs down.

### Checks

`tools/check.py`: 14 files, 145 labels, all references and citations resolve, braces and environments
balanced, `bootstrap` 0. `sync_v2.py status`: clean at v2.9. Both bundles rebuilt;
`make_bundle.py --verify`: 13 inlined sources, byte-faithful. **Not compiled.**

---

## v3.3 — 2026-09-11 · the environments and their constraint sets, shown rather than only described

**Asked for:** the setup section should carry an image and a description of the environment *and* the
constraint set, for `avoiding`, `aligning` and each of the three aerial scenes, with placeholders
marked TODO where no image exists — plus a demonstration of the different constraints.

**Found:** the chapter described all three environments and showed nothing. It also under-described
the constraint sets, which are the more important half — the projector is inherited and held fixed,
so **the geometry is the part that had to be designed**, and it was getting one paragraph.

### 🔴 The finding that changes what may be shipped

`figures/avoiding.png`, `figures/avoiding_constraints.png` and `figures/avoiding_data.png` exist in
this repository and are exactly the panels the chapter needs. **All three are byte-identical to the
baseline authors' own copies in `aux_repo/dpcc/figures/`** (`cmp` verified); they arrived with the
"Add DPCC Code" commit. They are the baseline's artefacts, not ours, and shipping them as this
thesis's environment figures would be the visual form of the error the naming rules exist to prevent.

**The remedy is cheap and is recorded in the figure specs:** the script that produces them,
`scripts/visualize_data_constraints.py`, is in this repository and regenerates both panels from our
own configuration and dataset. It needs numpy/matplotlib and the vendored environment, so it is a
**cluster job**, not something runnable here. Re-running it makes the figure ours.

### Added

- **`\todofigure[height]{spec}`** in `parts/00_preamble_v3.tex`: a red-bordered box for a figure that
  is *planned but does not exist*. Distinct from `\fmpccgraphic`'s fallback, which means "this file
  should be here and is not". It carries the **specification** — what the panel must show and where
  the asset comes from — in the draft itself, so the gap is visible on every read and whoever renders
  it does not have to guess. Counted by `tools/check.py`.
- **Five specified figures** in `sec:setup:tasks`: the state-based benchmark, the vision-conditioned
  task (specified to show *the policy's own two camera streams*, not a third-person render, because
  what the network sees is what distinguishes the entry), the three aerial scenes at a common scale
  with the vehicle drawn to scale, the three halfspace geometries with the tightened margin drawn as
  a ring, and the scene-clearance-against-tracking-error panel.
- **A proper `Constraint sets` subsection**, replacing the three-sentence one: `tab:constraint-families`
  giving the five families and marking which three are ours; the ablation rule; tightening written up
  as an **experimental factor** with the measured claim that it is a larger lever than the choice of
  constraint arm; and the honest-geometry defect — 0.000/0.060/0.120\,m of clearance against a
  0.30--0.49\,m tracking error — with the gate that could not tell "safe" from "exactly on the edge".

### Two holes this pass makes visible rather than fixes

- 🔴 **The vision-conditioned feasible set is undefined in prose** — not here and not in the
  methodology sources — although every table on that entry is measured against it. Now an explicit
  `\hole` that names it as **the largest single gap in the methodology of this thesis**.
- The numeric tightening margin, in metres, for both embodiments, and what the projector-threshold
  sweep varies. Configuration values, not derived quantities.

### Checks

`tools/check.py`: 3 871 lines, 143 labels, all references resolve, all citations resolve, braces and
environments balanced. Drafting macros now 14 `\hole`, 5 `\provisional`, 18 `\guard`, 35
`\srcnote`, 18 `\dataref`, **5 `\todofigure`**.

🔴 **A second tooling bug, caught by the tooling:** `make_bundle.py --verify` undid the
`\\includegraphics` → `\\fmpccgraphic` rewrite on whole lines, while the rewrite itself only touches
the part of a line TeX executes. The moment this pass added a *comment* mentioning `\\fmpccgraphic`
by name, verify corrupted that comment and reported a mismatch against an untouched file. The two
now share one `split_comment` helper, so the inverse is a real inverse.

🔴 **A counter bug fixed in passing:** `check.py` counted drafting macros with a bare `\\macro\{`
pattern, which does not match `\todofigure[0.3\textwidth]{...}`. It reported **zero** planned
figures while five existed — a pre-submission check that silently hides exactly what it exists to
find. Now matches the optional-argument form.

---

## v3.2 — 2026-09-11 · template-conformance audit

**Asked for:** sanity check that the TUM template is still preserved in v3.

**Answer: yes.** `Template_DONT_CHANGE/` is untouched (clean `git status`, last touched by
`c721f7d4`), and the `\ifstandalone`/`\else` merge bridge survives verbatim in both
`parts/00_preamble.tex` and `parts/01_frontmatter.tex` — v3 inherits them byte-for-byte and
`sync_v2.py status` confirms no drift. v3 adds **no packages**.

### One real divergence found, and fixed

`settings.tex:44-52` capitalises the `\autoref` names through babel's language hook. The standalone
branch never loads `settings.tex`, so a standalone build fell back to hyperref's own lowercase
defaults: **"section 5.2" where the merged template build says "Section 5.2"**, across 133 `\autoref`
calls. Mirrored into `parts/00_preamble_v3.tex` using the *same* `\addto\extrasamerican` hook — a
plain `\providecommand` in the preamble would be overwritten when babel selects the language at
`\begin{document}` — and guarded by `\ifstandalone` so the merged build still takes `settings.tex`'s
copy and it never fires twice.

### 🔴 Known, inherited from v2, NOT fixed: the standalone build is not PDF/A

`settings.tex` loads `\usepackage[a-2u]{pdfx}` and the template ships `main.xmpdata`; the standalone
branch loads neither. So **a bundle built here is not PDF/A-2u**, and if the submission requires it
the final PDF must come from the template path, not from the Overleaf bundle. This is a v2 divergence
that v2's own README does not list among its declared ones. Left as-is deliberately: adding `pdfx`
would change the standalone build substantially and risks breaking the Overleaf compile that v3.1
exists to enable. Flagged rather than silently carried.

Also absent from the standalone branch, and harmless because the draft uses none of them: `listings`,
`lstautogobble`, `scrhack`, `tikz`, `pgfplots`, `pgfplotstable`, `caption`, `ifthen`, `pagecolor`,
the TUM corporate colours and the `\BeforeTOCHead` PDF bookmark. All arrive with `settings.tex` on
merge.

### The merge path, restated for the split layout

v2's bridge assumed one file. v3 is fourteen — but `bundle/make_bundle.py` reconstructs exactly that
single-file shape, so the bridge is intact and in fact simpler to execute than before: **flatten, set
`\standalonefalse`, drop the bundle at the root of a template copy.** Its `\else` branch then pulls
`settings.tex`, `pages/cover`, `pages/title` and the rest from the template, which is why those
`\input`s are deliberately left unresolved in the bundle. `parts/00_preamble_v3.tex` folds into
`settings.tex` at that point, together with v2's own `amsmath`/`amssymb`/`amsthm` addition.

### Checks

`tools/check.py` passes; `make_bundle.py --verify`: 13 inlined sources, byte-faithful. Both bundles
rebuilt.

---

## v3.1 — 2026-09-10 · a flattened build, because the split draft would not compile

**Asked for:** the split draft cannot be compiled on a remote Overleaf. Write a tool — not an
LLM-driven copy-paste — that aggregates the parts into one full `.tex`, timestamped, in a subfolder
under `v3/`, with the tool living in that subfolder. Write it and run it.

**Added:** `bundle/make_bundle.py`, plus `bundle/README.md` and `bundle/BUNDLE_LOG.md`.

### What it does

Recursively inlines every `\input` whose target exists, wrapping each in `BEGIN`/`END` banners
carrying the source path and its SHA-256 prefix, and writes
`bundle/thesis_v3_<YYYYMMDD_HHMMSS>.tex` plus a `.zip` holding that file, both bibliography
resources and `figures/` — an Overleaf upload in one artefact.

**An `\input` whose target does not exist is left verbatim, and that is correct rather than a
fallback.** The inherited preamble and front matter carry `\input{settings}` and
`\input{pages/cover}` inside the `\ifstandalone … \else` branch, for the day the draft is merged
into the TUM template; `\standalonetrue` is set, so LaTeX never reads them. Six such lines survive
in the bundle and all six are in dead branches — checked, not assumed.

### The figure problem, and how it is handled

The figures are SVG and `\includegraphics` cannot read SVG; no converter exists in this container.
The tool rewrites every `\includegraphics` to `\fmpccgraphic` and injects that macro, resolving
`.pdf` → `.png` → *(mode)* → placeholder. Default mode draws a framed box naming the missing file, so
**the document always builds**; `--svg-package` adds `\usepackage{svg}` for hosts with Inkscape,
Overleaf among them. Opt-in rather than default because it is a property of the build host, not of
the document. A real `.pdf`/`.png` wins in every mode, so running `tools/svg2pdf.sh` and rebuilding
upgrades the figures with no source change.

🔴 **Ordering constraint, recorded because it is easy to get backwards:** the rewrite must run
*before* the shim is injected. The shim's own body calls `\includegraphics`; injecting first would
rewrite those calls too and make `\fmpccgraphic` infinitely recursive.

### Verification, which is the part that matters

`--verify` extracts every inlined source back out of a bundle, undoes the one transformation the tool
applies, and diffs against the tree. **All 13 inlined sources round-trip byte-for-byte.** Tested in
both directions: appending one line to a chapter made it report `DIFF … first difference at source
line 127`, and reverting restored `byte-faithful throughout`. Run against an *older* bundle it
answers a different and equally useful question — *was this built from what is on disk now?*

The build path additionally refuses to write at all unless: no resolvable `\input` remains, no
content was lost, braces balance, environments balance, and `\documentclass`, `\begin{document}`
and `\end{document}` each appear exactly once.

### Two bugs found by running it rather than by reading it

- 🔴 **The placeholder would have failed on exactly the filenames it exists to print.** It set the
  missing name in `\texttt{figures/#1}`, and every generated figure name contains underscores
  (`fig_avoiding_k_ladder`); a bare `_` in text mode is a subscript ⇒ *"Missing $ inserted"*. Now
  `\texttt{\detokenize{...}}`, with a comment saying why it is not decorative.
- **Two runs in the same second collided** and the second silently overwrote the first, defeating the
  point of stamping them. A bundle is now never overwritten: the tool suffixes instead.

### Result

14 source files, 3 648 source lines → **3 719 output lines, 228 KB**, 90 KB zipped. Two bundles
built, one per figure mode. **Still not compiled** — there is no TeX toolchain here, the tool checks
structure rather than typesetting, and it says so on every run. The first real build is the author's.

---

## v3.0 — 2026-09-10 · the workspace, the figure pipeline, and the experiments

**Asked for:** branch a v3 off v2 that can be worked on **in parallel** with it, so that a later v2
change can be identified and pulled down; give it a changelog like v2's; give it figure code kept as
a template that takes current data and emits current plots, so that when data updates the plots
update; and write the experiments — the state-based `avoiding` entry in full, the bones filled, and
what can be written for the other two entries.

**Branched from** v2 at `v2.6 — 2026-09-09 · the compute environment, and the first prose written into
Chapter 5`, recorded in `inherited/SYNC_STATE.json` together with the file digests.

### 1 · The parallel-work machinery

v2 is one 2 151-line file, which makes parallel work impossible: any v2 edit conflicts with any v3
edit. v3 is therefore v2 **split by chapter**, with a merge base and a tool.

- **`tools/split_v2.py`** splits `thesis_v2.tex` on *content markers* — `\begin{document}`,
  `\mainmatter{}`, each unstarred `\chapter{}`, `\appendix{}`, `\addchap{Abbreviations}` — never on
  line numbers, so it keeps working as v2 grows. It asserts that every input line lands in exactly
  one output file, and it **refuses to run** if v2 adds a chapter whose label the manifest does not
  know, rather than inventing a filename.
- **`inherited/v2_base/`** is the merge base: v2 as it stood at the last sync. **`tools/sync_v2.py`**
  re-splits the current v2 on demand and reports, per file, whether v2 moved and whether v3 moved;
  `merge` runs `git merge-file` with the base and advances the baseline **only on a clean merge**, so
  a conflicted file can simply be redone.
- **Per-file policy** in `inherited/MANIFEST.md`: `inherit` (v3 never edits — fast-forwards),
  `merge` (both edit), `own` (v3 wrote it; the baseline exists so drift is *visible*, not silent).
- **The dependency is one-way.** `tools/` reads `../v2/` and never writes to it. A fix belonging in
  both drafts is made in v2 and synced down.

**Tested end to end, not merely written.** In a scratch copy: v2 edited an `inherit` chapter and a
`merge` chapter; v3 independently edited the same `merge` chapter elsewhere. `status` classified both
correctly, `merge` fast-forwarded the first and three-way-merged the second with **both** edits
present and **zero** conflict markers, and re-stamped the baseline.

**No temporary aggregation sections were needed.** The split is clean at chapter granularity, so
every chapter has exactly one owner. `sec:bg:fewstep` is the single place where v3 would write inside
an inherited chapter, which is why `02_background.tex` carries the `merge` policy — and v3.0 did not
in fact write there (see *Not done*).

### 2 · The bibliography, split for the same reason 🆕

Raised mid-pass by the author: the `.bib` needs the same treatment, and v3 should record *which*
version of v2's bibliography it holds.

- `bibliography.bib` stays **inherited byte-for-byte**; v3 never appends to it. Its policy is
  therefore `inherit`, not `merge`, so it is a permanent fast-forward.
- `bibliography_v3.bib` is new and holds v3's own entries. `parts/00_preamble_v3.tex` registers it
  with a second `\addbibresource`, guarded by `\ifstandalone` for the same reason v2 guards its own:
  the template's `settings.tex` already registers a resource in the merged build.
- **Why not one file:** two live drafts appending at the end of the same file is the worst conflict
  shape there is, and it would recur on every pass.
- **The version stamp:** `sync_v2.py` reads the newest `## v2.x` heading from v2's changelog and
  writes it, with SHA-256 prefixes of both v2 files, into `inherited/SYNC_STATE.json`. `status`
  prints it, and `stamp` **refuses** to record a version while any baseline file differs from v2 —
  so the stamp cannot become a lie.
- `tools/check.py` reports a key defined in *both* `.bib` files. Biber would too; catching it here is
  cheaper.

### 3 · The figure pipeline

`plots/`, stdlib-only, plus `figures/` and a generated manifest.

- **`sources.py` is the only file containing a path.** Corpus entries mirror `DATASTATUS §10` row for
  row and carry the *protocol* and the readiness *grade*, not just the directory. **When data lands,
  editing this one file rebuilds every figure with its subtitle and provenance updated.**
- **`fmpcc_svg.py`** is carried over from
  `Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/make_figs.py`, so the thesis figures
  are drawn by the same code as the reports of record. No matplotlib: this container has no
  scientific Python stack, and a figure script that could not run where the writing happens would
  guarantee that figures drift from text.
- **`figures.py`** — one builder per figure, each returning `None` when its corpus is absent so a
  partial checkout builds what it can. **`make_figs.py`** writes `figures/MANIFEST.md` recording
  which corpus each figure came from, so a stale figure is a visible fact.
- **Six figures built** from data on disk: four cost/quality frontiers (aggregate + three
  geometries), the step-budget ladder, and the constraint-arm cost crossover.
- SVG is the committed artefact; `tools/svg2pdf.sh` converts where the document is built. The
  draft uses extension-less `\includegraphics`, so it needs no change either way.

#### 🔴 The aggregation rule, and a contradiction it would have caused

Entry-1 cells must be aggregated **per geometry first, then across geometries** — never as a flat
mean over `(seed × geometry)` cells. The two differ whenever a geometry carries a different seed
count, which is exactly the case for the pinned baseline, whose `both-hard` cell is seed-6 only.

Geometry-mean reproduces the DAs of record **exactly**: baseline 0.983 / 69.0 steps / 0.5635 s
against DA_20260827 §10.1's 0.983 / 69.0 / 564 ms; MeanFlow K1 0.993 / 61.0 / 0.0181 against its
0.993 / 61.0 / 18.1 ms; and the baseline at the published protocol 1.000 / 70.1 / 0.5534 against
DA_20260906 §3's 1.000 / 70.13 / 0.5534. **A flat cell-mean gives 0.977 / 72.5 / 544 ms** and would
have put the figures in silent contradiction with the text. `geometry_mean` takes the geometry column
index as a *required* argument, because inferring it from the key length picked the wrong column for
one of the two loaders and produced an empty figure rather than a wrong one only by luck.

### 4 · Chapter 5 — Experimental Setup

The compute-environment subsection v2 wrote is preserved **verbatim**. Around it:

- **Three entries, organised by what each adds** — the state-based benchmark, then a
  vision-conditioned task that adds harder control *and* harder perception, then an aerial embodiment.
- **Why the aerial entry is a different problem, not a harder one**: fully actuated and
  quasi-statically stable against underactuated and open-loop unstable; kinematic against dynamic
  feasibility; one loop at planner rate against a cascaded multi-rate one.
- **The regime taxonomy** — all-pass / discriminating / all-fail — defined once, measured (an
  all-pass scene is one where the unprojected arm's violation counter reads zero), and used by three
  chapters.
- **The pinned baseline** as `tab:target`, with the coverage asymmetry stated: the transport rows are
  300 episodes and the baseline 220, because its hardest geometry has one seed. 🔴 **This corrects a
  summary line in DA_20260827 §10.1**, which says 300 a side; the batch has 11 baseline cells, not 15.
- **`\autoref{def:improvement}`** — Pareto dominance with quality as a gate rather than a third
  objective — plus the funnel for all-fail regimes, and the explicit statement that the aerial
  per-step budget is never a pass/fail criterion.
- **The two protocol tiers**, with the measured cost of the difference as `tab:tiers`: a 1.00 at ten
  episodes means "≥ 0.90 at ±0.10", untightened mid-range cells move by up to 0.35 in both
  directions, tightened ones are stable to ~0.08. The consequence is a methodological point, not a
  formality: **the published protocol's resolution is coarser than most differences the field reports
  on this benchmark.**

### 5 · Chapter 6 — Results

**Entry 1 in full.** `tab:state-headline`: MeanFlow at a single network evaluation against the pinned
baseline at twenty — 0.993 / 61.0 steps / 18.1 ms against 0.983 / 69.0 / 563.5, architecture-matched
at 4.0 M. The compute margin and the quality margin are read differently *in the text*: 31× is far
outside noise, 0.993 against 0.983 is *inside* the across-seed standard error and is therefore the
**gate** of `def:improvement`, not its margin.

- **The flow-matching result is written as a cost claim**, `tab:state-fm`, with the matched-budget
  control that decomposes the 21× into ≈1.4× of engine and ≈15× of budget. The quality version is
  refuted on Entry 2 and is withdrawn in `sec:disc:negative`.
- **The low-budget pair** is a *pair*: three reasons in the text forbid the ladder — p = 0.231, a
  wash over all six rules, and **two floors trained with the better quoted**, a selection effect a
  single seed cannot absorb.
- **A defect reported as one:** the cumulative-cost rule stalls on the flagship (98.0 steps at K2,
  72.0 at K1, against ~61 elsewhere) and on the bootstrapped target, but not on the baseline —
  so the sign tracks the projection-cost landscape, not the architecture.
- **`sec:res:fewstep`** carries `fig:k-ladder`, with the baseline's published-protocol low-budget
  cells drawn **dashed and hollow** as a separate series. Mixing 10-episode and 100-episode cells
  into one solid line would have been the most misleading thing that figure could do.
- **`sec:res:constraints`** states the citable claim **per entry and never pools**, and
  `sec:res:constraints:degenerate` gives the genuine-step condition, confirmed by the shipped code at
  run time (solve counts 1 : 1.51 : 2.52 against a predicted 1 : 1.5 : 2.5).
- **Entries 2 and 3 written to grade.** Entry 2's engine result is decisive and is written as such;
  its projector claim selects **one** operating point (K = 10, 0/9, p = 0.0039) and explicitly
  refuses the K = 20 safety pairing, which is one discordant rollout at p = 1.0000. One table is
  **held**. Entry 3's chapter is written and its **ranking is not banked** — every ordering sentence
  carries seed 6, n = 10 — while the embodiment argument, the regime taxonomy and the
  zero-unsafe-rollouts-in-210 result stand independently of it.

### 6 · Chapters 7, 8 and the appendix

Discussion written in full: where Goal A and Goal B pull against each other and why the budget is
really a constraint-enforcement decision; the bootstrapped target as a **negative result with a
derived mechanism** and with **both α floors named**; the withdrawn quality claim; task saturation as
a general threat to validity, argued from this work's own corridor scene, where the projector spends
155–202 ms/step removing violations that never existed.

Conclusion written with the RQ answers settled at the strength the evidence supports; headline
numbers and the abstract left for last, on purpose. The appendix gains `tab:names` — the **reverse**
map from thesis names to the tokens in released artefacts, which `NAMING §8` records as owed and
which the results chapters make urgent, since they use mechanism names exclusively — and
`tab:corpora`.

### Not done, on purpose

- 🔴 **`sec:bg:fewstep` still says the bootstrapped target's mathematics is "deliberately deferred".**
  `DATASTATUS §7` asks for that to be reversed now that the fallback has fired and the mechanism *is*
  the contribution. v3.0 leaves Chapter 2 byte-identical and writes the mechanism where the result is
  (`sec:disc:negative`). **First item of v3.1.** Adding a pointer instead of the derivation would have
  been worse than leaving it.
- **v2's `\hole` in `sec:method:engine` about the sampling prior is answered from the protocol side
  only**, as a `\provisional`: the flow-matching numbers were produced by the shipped sampler and
  were *not* re-run at matched scale, the bias direction is unestablished, no result is attributed to
  it, and the flagship MeanFlow rows are unaffected. `04_method.tex` is left byte-identical so it
  stays a clean fast-forward.
- **Entry-2 and Entry-3 figures**, the abstract, the headline numbers, and the cross-entry synthesis
  table. Reasons in `README.md`.
- **Dataset counts and splits** are a `\hole` in `sec:setup:data`. Book-keeping, not blocked — read
  them off the dataset files rather than a dev log.

### Verification done in this pass

- Every Entry-1 number was **recomputed from the batch CSVs** and matched against the DA of record
  before being written: the target, the flagship at K1/K2, the per-geometry flow-matching cells
  (1.00/1.00 vs 1.00/0.95; 65.5/71.6 vs 70.0/77.6 steps; 1.8/1.9 vs 39.1/40.2 s/ep), the
  published-protocol budget ladder (0.667 → 1.000 → 1.000), and the cumulative-cost stall (98.0).
- Entry-2 and Entry-3 headline numbers were **spot-checked against their closure DAs directly**, not
  taken from the readiness ledger's transcription: the −0.3744 m / 0/10 / p = 0.0020 pairing, the
  −324.96 ms / 0/9 / p = 0.0039 latency sweep, the 0.635 / 0.359 / 0.235 aerial means, and the
  0-of-210 against 43-of-270 safety tally. All four reproduce.

### Mechanical checks

`python3 tools/check.py`: 14 files, 3 634 lines, 137 labels, 26 distinct citations of 26 bib
entries, 2 figure references — **dangling references: none · duplicate labels: none · missing bib
keys: none · keys in both `.bib` files: none · unbalanced environments: none · brace delta: 0 in
every file · `\includegraphics` targets: all present**. Drafting macros: 12 `\hole`, 5
`\provisional`, 17 `\guard`, 35 `\srcnote`, 18 `\dataref`. `python3 tools/sync_v2.py status`: clean —
no inherited file has moved. **Still not compiled**, and `check.py` says so on every run.

Two bugs were found and fixed in the tooling by these checks rather than by inspection: a
`[^%]*` character class in the input scanner matched across newlines under `re.MULTILINE` and
silently loaded only 5 of 13 files; and the brace counter subtracted escaped braces once instead of
excluding them from both counts, reporting a −2 delta per `\{…\}` pair on two untouched inherited
chapters.
