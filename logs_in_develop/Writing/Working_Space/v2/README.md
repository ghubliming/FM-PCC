# v2 — the mathematics, with citations

> 🚦 **Draft ownership:** v2 writes **Ch 1–4**, abstract, preamble, acronyms, `bibliography.bib` — nothing under `v3/` or `v4/`, and not v2's own Ch 5–8 placeholders. Rules: [`../DRAFT_OWNERSHIP.md`](../DRAFT_OWNERSHIP.md).

**Created:** 2026-09-08 · **File:** [`thesis_v2.tex`](thesis_v2.tex) · **Bibliography:** [`bibliography.bib`](bibliography.bib)
**Change history:** [`CHANGELOG.md`](CHANGELOG.md) — *every pass gets an entry there; current summary below; historical notes are explicitly marked*
**Built from:** [`../v1/thesis_v1.tex`](../v1/thesis_v1.tex) (copied, then extended — v1 is untouched)
**Governed by:** [`../TARGET_20260905_thesis_claim_ladder.md`](../TARGET_20260905_thesis_claim_ladder.md)
**Apparatus source:** [`../../Auxiliary/Methodology_Sources/`](../../Auxiliary/Methodology_Sources/README.md)
 — incl. [`AUX_compute_environment.md`](../../Auxiliary/Methodology_Sources/AUX_compute_environment.md), the machine of record
**Placement plan:** [`../../Auxiliary/NOTES_paper_map.md`](../../Auxiliary/NOTES_paper_map.md)

---

## What v2 is

**Current revision: v2.28 (2026-09-24), the cross notes up to v3.99 / v4.1 applied (Claude).** The abstract and the
§1.4 result sentences follow the author's storyline (`../GUIDE_20260924_results_storyline_author.md`) and v3.99's
conclusions; the outline follows v4.1's order (Conclusion, then Discussion); "MuJoCo MPC" replaces `\ac{MJPC}`;
PD is declared; §3.4 states HardFlow's DPCC setup and baseline. v2 delivers only the `.tex`; the aggregated
release is built by v4 after v3 and v4 have synced v2.28.
[v2.28 changelog](changelogs/v2.28_20260924_cross_notes_v3.99_v4.1.md).

**Base: v2.27 (2026-09-23), audit corrections by ChatGPT (Codex).**
V2 owns the abstract and Chapters 1–4. All three flow objectives are formulated in Chapter 4,
including the consistency-interpolated loss and its branches. The average-velocity field uses
start time and interval length; K counts sampling steps, separately from NFE.

The abstract and contribution statements use the available v3.75 results. The outline, inline
system schematic and component-provenance table are complete; there are no `\hole` calls in
the owned abstract/Chapters 1–4. The bibliography remains at 53 entries. The v2.26 quadrotor-scene
section is preserved unchanged. Detailed noise-scale discussion, the redundant numerical endpoint
configuration table and loss-monitoring advice are excluded as agreed with the author.

See [the signed v2.27 changelog](changelogs/v2.27_20260923_ChatGPT_audit_TODOs.md) for the finding-by-finding
record and validation. v3 synced v2.27 at v3.78 and v4 carries it through v3.98a; neither has synced v2.28 yet.

### Historical revision notes

The entries and early-draft inventory below describe their own versions, not current TODOs.
The current summary above and the latest changelog take precedence.

- **v2.0** — the engine mathematics: **DPCC's diffusion engine → flow matching → MeanFlow.**
  α-Flow carries **no mathematics**; it is named, cited and demoted to an ablation, and its
  formulation is deliberately deferred.
- **v2.1** — conformance with the TUM template (audit table below).
- **v2.2** — the **apparatus** mathematics: DPCC's first-order Euler dynamics model and its
  projector rows, the manipulator IK chain, the visual-aligning 3-D instantiation, and the aerial
  PID / MJPC stack — all written on top of that Euler trajectory.
- **v2.3** — every bibliography entry re-verified against the PDF in `aux_repo/PAPERS` it came from,
  linked to that file, and the transcribed equations pinned to the source's own equation numbers.
- **v2.4** — the visual U-Net written out (block, latent, both conditioning mechanisms, the two-time
  input), and the naming trap it exposed: **the shipped conditioning is not FiLM**, though the flag
  says `film_mode`. Rule now lives in `Auxiliary/NOTES_naming_and_rebuild.md`.
- **v2.5** — arm C's **published** experimental setup (H16, 8 actions per plan, IPOPT, fitted linear
  dynamics) recorded against ours (H8, 1 action per plan, SLSQP, Euler) in `tab:hardflow-setup`,
  with the wall-clock consequence spelled out.
- **v2.6** — the dropped terminal cost `C` and what its removal costs; and the naming pass —
  arms are now **iterate projection** / **in-ODE endpoint projection**, engine 3 is **the
  bootstrapped target**. Canonical table: `Auxiliary/Naming/NAMING_20260910_master_table.md`.
- **v2.7** — the **visual** contribution and the **arm → UAV** extension are pitched in the abstract
  (two new paragraphs) and split into their own contributions (6 → 7); and the UAV controller gets
  its real name — **cascaded geometric tracking controller**, since `CascadedPID` has no integral
  term and a non-scalar inner loop (`Remark rem:pidname`).
- **v2.8** — *"bootstrapped target"* was invented jargon; engine 3 is a **consistency target** and is
  named after the family it belongs to. The Euler surrogate is contrasted with SafeFlowMPC's
  exact-kinematics NLP (`Remark rem:eulermodel`). Method reordered bottom-up into **control
  substrate → generative model → projection mechanism**.
- **v2.9** — abstract **481 → 278 words**; and the prose stops justifying itself. Defensive
  constructions (*"does not claim"*, *"must not be"*, *"worth stating"*, *"rather than left to be
  discovered"*) removed throughout — no content lost, only the arguing.
- **v2.10** — a real abstract (**173 words**); "better" no longer defined, metrics stated in place;
  **"arm A/B/C" replaced by unguided / iterate projection / endpoint projection** (69 hits);
  self-talk and chapter-summary meta removed, Related Work rewritten as direct prose.
- **v2.11** — ~~numbers in the abstract~~ — reverted in v2.12.
- **v2.12** — abstract rewritten on the pattern of the reference papers (DPCC, HardFlow, SafeFlowMPC,
  Diffuser): one paragraph, **no numbers**, qualitative results only.
- **v2.13** — Chapter 1 overhaul; see `changelogs/v2.13_20260914_chapter1.md`.
- **v2.14** — Chapters 2–4 restructured (concepts · papers · mathematics); see `changelogs/v2.14_20260914_chapters2-4_restructure.md`.
- **v2.15** — *α-Flow* / *HardFlow* named only in Related Work (mechanism name + citation elsewhere); the evaluation told as two stages — DPCC's benchmark unchanged first, then the alignment task and quadrotor benchmark built for this thesis; see `changelogs/v2.15_20260914_names_and_two_stages.md`.
- **v2.16** — the "not free choices" remarks deleted; one visual conditioning, named **feature-wise conditional biasing** (no FiLM); §4.5.3 states our configuration only; §4.7.3 gains the UAV expert-demonstration method; see `changelogs/v2.16_20260914_conditioning_endpoint_config_uav_demos.md`.
- **v2.17** — visual encoder traced D3IL → Diffusion Policy (ResNet-18, spatial softmax, GroupNorm) in §2.6/§4.3.6; all six cross-draft inbox items closed; see `changelogs/v2.17_20260916_encoder_provenance_and_inbox.md`.
- **v2.21** (2026-09-17) — v3 inbox from v3.18/v3.19/v3.20/v3.25: mechanism names for the three generative models in abstract + Ch 1–4; UAV intervals as simulated time; contribution 4 cost sentence; §1.2 forward roadmap; `n_guide` table → `tabularx`; see `changelogs/v2.21_20260917_v3_inbox_mechanism_names_uav_intervals.md`.
- **v2.22** (2026-09-17) — HardFlow Alg. 1 + Fig. 11 cited; α-Flow target written out from its Alg. 1/2 (source stored in aux_repo); RQs 4 → 3; §2.4 few-step framed inside flow matching; FiLM named in §2.6; Chapter 3 loop/plan/backbone comparison (`tab:related-loops`); `tab:embodiments` → `tabularx`; quadrotor controller compressed; see `changelogs/v2.22_20260917_sources_rqs_related_work_and_uav_compression.md`.
- **v2.23** (2026-09-17) — MuJoCo named as the simulator (D3IL supplies task files, demonstrations, cameras); the Franka Emika Panda and its rod end effector named; `\ac{IK}` confirmed available for v3; see `changelogs/v2.23_20260917_mujoco_simulator_and_robot_naming.md`.
- **v2.24** (2026-09-18) — Chapter 4 reordered to overview → plant dynamics incl. control → generative models → projection → environments (`Low-Level Control` merged, both labels kept); RQ1 names its comparison axes; capacity column in `tab:related-loops` ($4.0$ M, sources report none); §4.3.1 "deliberately crude" rewritten as fact; `tab:embodiments` short forms expanded; MuJoCo introduced in §2.5 and the brake-to-rest setpoint policy stated (v3.31, v3.32); see `changelogs/v2.24_20260918_ch4_reorganised_rq1_axes_and_capacity.md`.
- **v2.25** (2026-09-21) — the → v2 inbox queue emptied: rotor reach not "vehicle radius" and not the tightening (v3.60), the plan holds $H$ transitions with $n=Hd$ and the dynamics-row ranges to match (v3.58), the action bound defined once in the §4.6 intro (v3.51), "wall-clock" gone (v3.48), the action weight no longer claimed as shared (v3.43), the alignment box pose stated as unobserved and the constraint set corrected to the evaluated geometry (v3.36), v3.46 needing no edit. Translation table re-swept (*projector*, *genuine*, defensive meta-prose). §2.7 says what an open-loop unstable platform costs a planner. **First figure in v2**: HardFlow's Fig. 11 in §4.5.3, CC BY 4.0, so `figures/` and `\graphicspath` are new. Table 4.1 stays in the main text per the formatting rules. `CROSS_STATE.json` records that v2 is current against **v3.60**; see `changelogs/v2.25_20260921_inbox_translation_uav_incentive_hardflow_figure.md`.
- **v2.26** (2026-09-23) — the ChatGPT audit (`audit from chatgpt/`) checked against code and v3.71 and answered in its §8 (18 confirmed, 3 qualified, none rejected; F12 widened: the FM sampler's prior is `0.5·randn` against σ = 1 training); §1.4 regrouped into *On the benchmark of DPCC* (1–4) and *Beyond it* (5–6) with item 6 rewritten for the current quadrotor benchmark; §4.6.3 rebuilt as two constructions (corridor/s-curve with the twelve-channel plan and flown demonstrations; UAV-pillars = the avoiding planner flown through the similarity map `eq:method:env:pillarsmap`); `eq:method:env:switched` restated as the per-replan wall selection; see `changelogs/v2.26_20260923_audit_feedback_contributions_two_stages_uav_scenes_rebuilt.md`.
- **v2.6** — the **compute environment**: the cluster, node, CPU/GPU and software stack every number
  was produced on, plus the two caveats that follow from the machine. First prose written into
  Chapter 5 — a deliberate, scoped exception to the bone rule (see below).

| chapter | v1 | v2 |
|---|---|---|
| 1 Introduction | filled | filled + citations |
| 2 Background | filled, prose only | **filled + all engine mathematics** (§2.2 diffusion, §2.3 FM, §2.4 MeanFlow, §2.5 constrained sampling) |
| 3 Related Work | filled, prose only | filled + citations |
| 4 Method | filled, prose only | **filled + notation table, problem formalisation, the engine delta, the Euler dynamics model, the projection NLP with its dynamics rows, the genuine-step boundary, and a full closed-loop deployment section (IK · cascaded geometric PID · MJPC)** |
| Abstract | sketched | unchanged (rewrite last) |
| 5 Setup · 6 Results · 7 Discussion · 8 Conclusion | bone | bone, **except** one written subsection: the compute environment in `sec:setup:protocol` (see below) |
| Appendix A–C | bone | `app:repro` opened with the compute-environment section and `tab:compute` |

Line count 610 → 2118. 49 numbered equations plus 6 multi-line blocks, 5 tables (notation, embodiment
instantiation, arm C published-vs-ours, the inline genuine-step regime table, and the compute
environment), 26 bibliography entries, all cited; `\srcnote` pointers into the code throughout and 5
remaining `\hole`s. No dangling `\ref`, no missing bib key, no uncited entry, no acronym
declared-but-unused except the template's own `TUM`.

### The one exception to "Chapters 5–8 stay bone"

`sec:setup:protocol` now ends with a written **Compute environment** subsection, and `app:repro`
carries the matching `tab:compute`. This is deliberate and narrow. The bone rule exists because the
*results* are unsettled; **the machine is not a result** — 2 × AMD EPYC 7282 (32 cores / 64 threads),
8 × RTX A5000 24 GB, one GPU per job, SLURM 21.08.5 — and none of it changes when the run queue
clears. A comment block at the insertion point states the exception and its scope. Everything else
in Chapters 5–8 is untouched.

Two caveats travel with it, both about the apparatus rather than any claim: the node is **shared**
(though GPUs are exclusive — no oversubscription, no preemption, so contention is CPU/memory/IO
only), and **32 physical cores serve 8 GPUs**, while the eval loop's two costs — MuJoCo rollout and
per-step SLSQP — are both CPU-side. Evaluation is plausibly CPU-limited rather than GPU-limited,
which is the stronger of the two and holds even on an idle node.

---

## The sourcing rule this draft observes

**Nothing in the mathematics is invented.** Every equation is either

- **transcribed from a paper** in `/workspaces/aux_repo/PAPERS/`, re-derived into this thesis's
  notation, with the notation change stated where the source differs (most often the direction
  of transport time); or
- **transcribed from this repository's code**, with a `\srcnote` naming the file and the
  function, because for an implementation claim the code *is* the source.

Where paper and code disagree, the disagreement is written down rather than smoothed over —
the implemented method is stated precisely and experiment settings are left to Chapter 5.
There is no current prior-scale remark in v2, following the author’s scope decision.

### Equation provenance

| thesis equation | what it is | source |
|---|---|---|
| `eq:bg:ddpm:forward` … `eq:bg:ddpm:loss` | DDPM forward process, closed-form marginal, reverse chain, ε-loss | DPCC §4 (pp. 3–4); originally [ho2020denoising] |
| `eq:bg:ddpm:x0`, `eq:bg:ddpm:mean` | standard clean-sample estimate and posterior mean; implementation clips the estimate | `aux_repo/dpcc/diffuser/models/diffusion.py:97–117` |
| `eq:bg:fm:ode` … `eq:bg:fm:cfm` | flow, continuity equation, marginal field, FM and CFM losses, Thm 1 / Thm 2 | Lipman et al. §3 (FM.pdf pp. 3–4); cross-read against HardFlow's Background (`PAPERS/Recommand_Paper/HF/main.tex:203–262`) |
| `eq:bg:fm:path` | linear (OT displacement) path, σ_min = 0 | Lipman et al. Eq. (20)–(21), read data-at-one |
| `eq:bg:fm:euler`, `eq:bg:fm:loss` | what this repo trains and integrates | `flow_matcher_v3/models/diffusion.py:125–158, 273–299` |
| `eq:bg:mf:def` … `eq:bg:mf:loss` | average velocity, consistency, **MeanFlow Identity**, JVP expansion, sg-target | MeanFlow §4.1, Eqs. (3)–(11) (MeanFlow.pdf pp. 3–5) |
| adaptive loss, logit-normal (r,t) | design decisions adopted unchanged | MeanFlow §4.3 (p. 7); logit-normal citation = Esser et al. 2024, verified against MeanFlow's own ref [11] |
| `eq:bg:mpc:proj`, `eq:bg:mpc:dyn` | model-based projection and its cost | DPCC Eq. (9), (9a)–(9b) |
| `eq:method:dpcc:thm`, `:step` | guidance derivation → constrained denoising step | DPCC Thm 1 and Eq. (16) |
| `eq:method:dpcc:tighten` | constraint tightening | DPCC Thm 2, Eq. (17) |
| `eq:method:plant` … `eq:method:feasible` | plant, novel constraints, plan variable, feasible set | DPCC §3 and §5.1, re-notated |
| `eq:method:engine:iface` | the three-line engine delta | the three `p_mean_variance` implementations, side by side |
| `eq:method:engine:mfid`, `:mftgt`, `:mfloss`, `:mfsample` | **start-anchored** MeanFlow identity, JVP tangents (v, +1, −1), dual loss, interval-jump sampler | `flow_matcher_v3_meanflow/models/mf_diffusion.py:189–300, 353–473` — this is *our* derivation of the identity under the data-at-one convention, and the sign check against the source is recorded in that function's docstring |
| `eq:method:proj:obj` … `:obs` | the QP/NLP actually solved: Q, A/b, C/d, quadratic obstacle rows, SLSQP | `aux_repo/dpcc/diffuser/sampling/projection.py:70–155` |
| `eq:method:proj:gate`, `:nact` | activation schedule, both index directions | `flow_matcher_v3/models/diffusion.py:174–183` (which also records what the guarded form changed) |
| `eq:method:degen:ngen` | the **genuine-step boundary** | `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:584–646` |
| `eq:method:sel:rules` | the three selection rules | DPCC §5.4 |
| `eq:method:dpcc:euler` | the **first-order Euler dynamics model**, `s_{t+1} = s_t + [aᵀ aᵀ]ᵀ t_s + w_t`, and `t_s = 1` because the action is a displacement | DPCC §6.1 (p. 8); `aux_repo/dpcc/config/projection_eval.yaml:14–16` |
| `eq:method:proj:deriv`, `:derivnorm` | the dynamics rows as written and **as implemented** (conjugated by the normaliser) | `aux_repo/dpcc/diffuser/sampling/projection.py:344–402` |
| `eq:method:dep:setpoint`, `:obs` | the setpoint recursion and the fed-back observation, shared by all three environments | `aux_repo/dpcc/scripts/eval.py:236–241`; `mix_uav_test/eval_mix_uav.py:1550` |
| `eq:method:dep:taskerr`–`:dls` | **the IK mechanism**: weighted damped-least-squares differential IK with null-space posture regularisation and SVD clipping, 3 iterations/step → joint PD | `d3il/.../controllers/IKControllers.py:134–323`; gains from `.../Config/mujoco_controller_config.gin:6–37` |
| `eq:method:dep:varows` | the six visual-aligning dynamics pairs; plan width 9, layout `(a, p_des, p)` | `mix_visual_aligning_test/eval_mix_visual_aligning.py:272–279` |
| `eq:method:dep:multirate` | plan every 0.03 s / physics 0.01 s of simulated time, `n_dec = 3` (v2.21) | `mix_uav_test/eval_mix_uav.py:1610,1728`; `uav_expert_data_collect/dataset_writer.py:31,73`; `quadrotor_modified.xml:4` |
| `eq:method:dep:plant` | the quadrotor rigid-body model | Lee et al. Eqs. (2)–(5) (`Drone/PID_Control_UAV.pdf` pp. 2–3) |
| `eq:method:dep:outer`–`:alloc` | cascaded **geometric** control: position PD → thrust vector → attitude extraction → SO(3) attitude error → moment → rotor allocation with thrust-first saturation. Each step cites the equation it realises — errors (17)–(18), thrust (23), attitude (22)–(23), attitude error (21), moment (20), allocation (1) — and the **three deviations** (frame convention, diagonal gains, dropped attitude feed-forward) are stated | `uav_env_test/flight_controller.py:33–155` against Lee et al. pp. 2–5 |
| `eq:method:dep:vdes` | the **three** shipped velocity-setpoint policies | `mix_uav_test/eval_mix_uav.py:1551–1563, 1822–1832` |
| `eq:method:dep:mjpc` | the tracker as an instantiation of the paper: objective (3), weighted-residual cost (4), risk-neutral transform (5), spline plan §3.3, and the search itself — **Algorithm 4** — with warm-start per Algorithm 1 | `mix_uav_test/mjpc_tracker.py:70–155` against Howell et al. pp. 2–6 |
| arm C's formulation | pinned to the **reversed receding-horizon** instance, not just the continuous statement | HardFlow Problems 1 and 5 (`HF/main.tex:280, 438`); port header `hardflow_projection.py:1–40` |

### How to trace a citation back to the paper

Every entry in `bibliography.bib` carries one of two markers:

- **`file = {...}`** — a path relative to `/workspaces/aux_repo/PAPERS/`. Title, authors, year and
  arXiv id were read off **that copy's own title page** (2026-09-08). These are the papers this
  thesis actually read: `romer2025diffusion`, `janner2022planning`, `jia2024towards`,
  `lipman2023flow`, `geng2025meanflow`, `zhang2025alphaflow`, `li2025hardflow`,
  `yang2025safeflowmatcher`, `lee2010geometric`, `howell2022predictive` — 10 of 25.
- **`% NO LOCAL COPY`** — harvested from `PAPERS/Recommand_Paper/HF/reference.bib` per
  `NOTES_paper_map.md`, **not independently verified**. 15 of 25. Check these against the publisher
  record before submission.

`file` is not printed by the alphabetic style; it exists so a citation resolves to the exact PDF in
one step. 39 of 85 citations carry an equation- or section-level locator
(`\parencite[Eq.~(21)]{lee2010geometric}`); the rest are positioning citations where a locator
would be wrong.

> Why the markers exist: the first α-Flow entry was written from memory and was wrong in title and
> in every author, and the first `lee2010geometric` entry invented a CDC venue the preprint does not
> have. Both were caught only by opening the PDF.

> A note on why that matters: the first draft of the α-Flow entry was written from memory and was
> **wrong in title and in every author**. It was corrected against `AlphaFlow.pdf` p. 1. Treat
> every `[OWN]` entry as unverified until someone opens the PDF.

---

## Historical template-conformance record

The counts below belong to the original conformance pass; current mechanical results are in the
v2.27 changelog. Checked file by file against `main.tex`, `settings.tex` and `chapters/01_introduction.tex`, and
brought into line where the template has a stated convention.

**Now conformant:**

| item | template says | v2 |
|---|---|---|
| cross-references | `settings.tex:44–52` defines `\chapterautorefname`/`\sectionautorefname`/`\subsectionautorefname` — i.e. the template is built for `\autoref` | 42 × `\autoref`, 0 × bare `\ref` (the one exception is `Appendix~\ref{app:repro}`, because no `\appendixautorefname` is defined and `\autoref` would render "Chapter C") |
| citations | `01_introduction.tex` demonstrates `\parencite{}` | 65 × `\parencite`, 0 × `\cite` |
| citation locators | biblatex postnote | the 7 places that said `(\cite{key}, Theorem~1)` now say `\parencite[Theorem~1]{key}` — this is also what the I6 rule about page numbers in citations needs |
| acronyms | declared in the preamble, used with `\ac{}`; `printonlyused` means an undeclared-but-unused acronym never prints | 7 acronyms declared and each reached by `\ac{}` at its true first prose use; the 8 that the prose never abbreviates were deleted rather than left to print nothing |
| tables | `\begin{table}[htpb]`, `\caption[short]{long}` | matched |
| front matter | `pages/cover` → `\frontmatter{}` → title/disclaimer/acknowledgments/abstract | the `\standalonefalse` branch now copies that order exactly |
| bibliography | `settings.tex:68` already registers the file (`\bibliography{bibliography}` is biblatex's legacy form of `\addbibresource`) | **bug fixed:** v2 first added `\addbibresource` unconditionally, which would have registered the resource twice in the merged build. It is now inside the standalone branch only |
| biblatex options | `backend=biber, style=alphabetic, maxnames=4, minnames=3, maxbibnames=99, giveninits, uniquename=init` | copied verbatim into the standalone branch |
| document class | `scrbook`, one-sided, 11pt, a4, `listof=totoc`, `bibliography=totoc` | identical |

**Still divergent — deliberately, and each for a reason:**

1. **One file with `\ifstandalone`, not `main.tex` + `settings.tex` + `chapters/` + `pages/`.**
   This is a working draft that has to be readable and diffable as a single artefact. The
   `\standalonefalse` branch is the bridge: flip it, drop the file into a copy of the template as
   `chapters/`, and nothing else changes.
2. **`amsmath`, `amssymb`, `amsthm` are added.** The template ships **no maths package at all**.
   Nothing in Chapters 2 and 4 can be typeset without them. Carry these into `settings.tex` when
   merging.
3. **`\hole` and `\srcnote`** are drafting macros in colour. Not template, and flagged for
   deletion before submission.
4. **Label prefixes stay `ch:` / `sec:` / `eq:` / `tab:`**, where the template's single example uses
   `chapter:introduction`. Kept on purpose: `TARGET_20260905` §7 maps every goal to a section by
   these exact label names, and renaming them would break the governing document.
5. **The 17 new subsections are starred (`\subsection*`)**, so they stay out of the table of
   contents; the template numbers subsections. This was a judgment call and it is **yours to
   reverse** — the bone planned Chapters 2 and 4 at section granularity, and numbering these would
   add a structural level the outline never budgeted. If you want them numbered and in the ToC:
   `sed -i 's/\\subsection\*/\\subsection/g' thesis_v2.tex`.
6. **`\eqref` is used for equations.** The template never typesets an equation, so it has no
   convention here; `\eqref` is the amsmath standard and is what the 71 equation references use.

---

## Decisions v2 takes that v1 did not

1. **The notation collision is resolved** (`NOTES_tum_formatting_rules.md` flagged it; open
   question 4). Table 4.1 in `sec:method:formalisation` fixes one convention for the thesis:
   control time `t`, plan `x` (not DPCC's `τ`), transport time `τ ∈ [0,1]` **data-at-one**,
   interval `h = τ − r`, step index `k`, budget `K`, activation threshold `η`, fan `B`. Every
   reproduced equation is mapped into it, and every mapping is stated at the point of use.
   `NOTES_notation_decisions.md` still does not exist; Table 4.1 is now the de-facto record and
   that note should be written from it.
2. **`τ` is data-at-one, and the two families count in opposite directions.** The diffusion index
   counts *down* to the data, the transport index counts *up*. This is stated once and used
   consistently; it is also why `eq:method:engine:iface` needs the remark about the reverse index.
3. **Experimental sampling scales belong in Chapter 5.** V2 distinguishes the standard
   background construction from the evaluated sampler without adding the detailed noise-scale
   discussion. The audit preserves its code evidence separately.
4. **The MeanFlow identity is derived in the start-anchored form**, not copied. The source anchors
   at the interval's end and runs time the other way; the sampler here queries the start. The
   derivation, the tangents `(v, +1, −1)` and the sign check are all in `sec:method:engine`.
5. **`eq:method:proj:gate` presents η = 0 as a special case, not a separate baseline.** That is
   what makes "post-processing" and "per-step projection" comparable rather than two code paths.
6. **The consistency-interpolated objective is formalised.** `sec:method:alphaflow` includes
   the target, explicit zero-interval override, zero-ratio analytic branch and branch-weighted
   loss. The numerical-target limit is distinguished from the scaled-gradient correspondence.
7. **One label was added to the bone**: `sec:res:constraints:degenerate` (a subsection of Results),
   because `sec:method:degenerate` referenced it in v1 and the reference dangled. No prose was
   written there.

---

## Build

```bash
latexmk -pdf thesis_v2.tex          # or: pdflatex ; biber thesis_v2 ; pdflatex ×2
```

> ⚠️ **Not compiled.** This container has no TeX distribution (`pdflatex`, `latexmk` and `biber`
> are all absent), so the file has **never been run through LaTeX**. What *was* checked
> mechanically: environment begin/end balance, brace balance, every `\ref`/`\eqref` resolves to a
> `\label`, every `\cite` key exists in `bibliography.bib`, and no bib entry is uncited. Syntax
> errors inside math mode would not be caught by any of that. **First build on a machine with TeX
> and fix what falls out.**

`\standalonetrue` still builds this file on its own; the biblatex options in the standalone branch
are copied verbatim from `Template_DONT_CHANGE/settings.tex`, so merging into a copy of the
template changes nothing.

---

## Remaining work outside this audit revision

- Author, supervisor, advisor, submission date and German-title confirmation remain author inputs.
- Compile and inspect PDF layout in a TeX environment; only source checks have run here.
- V3 owns experimental settings/results, its pending corridor and s-curve comparisons, and the next
  authorized inheritance sync. V4 owns discussion and final conclusion refinement.
- The original Chapter 5–8/appendix placeholders are unchanged, including their compute-accounting
  hole. They are not v2 writing TODOs.
- The additional v3.73 question about post-contact termination on the *manipulator* remains open.
  It is outside the agreed audit corrections; the existing scene section was preserved as requested.
- Source-note visibility and final metadata/layout checks remain part of submission preparation.
  Optional figure-format advice is not a prerequisite for closing this audit pass.
