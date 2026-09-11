# CHANGELOG — `Working_Space/v2`

Every change to `thesis_v2.tex` / `bibliography.bib` gets an entry here, newest first.
Format: what changed · why · what it is sourced from · what it left open.

**Rules for this file**
- One entry per working pass, not per edit.
- Every injected equation names its source — a paper (file + page/equation) or a repo file
  (path + line range). "From memory" is never an acceptable source; see the `v2.0` note on the
  α-Flow entry for why.
- Mechanical checks (env balance, braces, `\ref`→`\label`, `\cite`→bib key, `\acro`↔`\ac`) are
  re-run after every pass and their result is recorded. **There is no TeX toolchain in this
  container, so "checked" never means "compiled".**

---

## v2.6 — 2026-09-09 · the compute environment, and the first prose written into Chapter 5

**Asked for:** collect the cluster/hardware facts and put them somewhere the thesis can use.

**Where it went:** a `Compute environment` subsection at the end of `sec:setup:protocol`, and a
`Compute environment` section with `tab:compute` in `app:repro`. Source of record, carrying the raw
command output verbatim, is the new
[`Auxiliary/Methodology_Sources/AUX_compute_environment.md`](../../Auxiliary/Methodology_Sources/AUX_compute_environment.md).

### 🔴 This breaks the "Chapters 5–8 stay bone" rule, deliberately

The banner comment above `\chapter{Experimental Setup}` says nothing there is written until the run
queue clears. That rule exists because the *results* are unsettled. **The machine is not a result.**
It is a property of the hardware, fully verifiable today, and it will not change when the queue
clears. Writing it now costs nothing and saves reconstructing it from ~1000 dev-log MDs later. A
comment block at the insertion point states this exception and its scope; **everything else in
Chapters 5–8 remains bone.**

### The facts, and where each came from

| fact | source |
|---|---|
| $2\times$ AMD EPYC 7282, 32 physical cores / 64 threads, 1.5–2.8 GHz, 128 MiB L3 | `lscpu`, cluster job 25567, 2026-09-09 |
| 252 GiB RAM, `TmpDisk=0` (no node-local scratch), 2 NUMA domains, SLURM 21.08.5, Ubuntu 5.15.0 | `scontrol show node i6-gpu-1`, 2026-09-09 |
| $8\times$ NVIDIA RTX A5000 24 GB, driver 530.41.03, **1 GPU per job** | `sinfo`; the `GPU INFO:` header in all 777 header-bearing batch logs |
| driver + node constant across the project | header identical in the earliest (2026-04-29) and latest (2026-09-02) logs |
| Python 3.10, torch 2.2.2+cu121, numpy 1.26.4, scipy 1.13.1, mujoco 2.3.7, diffusers 0.31.0, transformers 4.41.2 | `Slurm_Codes/logs/2026-04-29/verify_env_job_19741_153833.log` (job 19741, git `630dd13`) |
| casadi 3.6.5, torchdiffeq 0.2.3, hydra 1.1.1, omegaconf 2.1.1 | `requirements.txt` |

### The two caveats, and why they are in Setup rather than Discussion

Both concern the measurement apparatus, not the validity of a claim, so they sit with the protocol:

1. **The node is shared** — six of eight GPUs and 42 of 64 threads were committed to other users at
   snapshot time. But `OverSubscribe=NO` and `PreemptMode=OFF`: **a job holds its GPU exclusively**.
   Contention is CPU/memory/IO only. That distinction is worth stating precisely, because it is the
   strongest available defence of the timing numbers.
2. **32 physical cores serve 8 GPUs — ~4 cores per GPU** — and both halves of the eval loop (MuJoCo
   rollout, per-step SLSQP) run on the CPU. Evaluation throughput here is plausibly **CPU-limited,
   not A5000-limited**. This is the stronger caveat of the two: it holds on an idle node as well.

Together these are why the draft calls the wall-clock figures *indicative* and refuses to present
them as a hardware benchmark — consistent with the standing rule that **`budget_ms` / 33 Hz is a
data-rate artefact, not a real-time pass/fail criterion** (TARGET §6.5).

### 🔴 Correction carried in from the source note

An earlier draft of the AUX note recorded the projector's solver as **IPOPT**. It is not.
`hardflow_projection.py:82` resolves `slsqp` by default and the inherited DPCC projector calls
`scipy.optimize.minimize(method='SLSQP')` (`aux_repo/dpcc/diffuser/sampling/projection.py:138`);
CasADi/IPOPT is a *selectable alternative* backend. `tab:compute` states both, in that order. This
also agrees with `tab:hardflow-setup` from v2.5, which already had IPOPT on the *published* side and
SLSQP on ours — the two tables would otherwise have contradicted each other.

### Not done, on purpose

- **Total compute is a `\hole`.** Timestamp arithmetic over the batch logs gives order $10^{3}$
  GPU-hours over ~5 months, but it parses only 285 of 777 header-bearing jobs and is a floor.
  `sacct` replaces it; until then no figure goes in print.
- **`nvidia-smi` was not run.** It would have added the driver API version and PCIe width — two
  facts the thesis never uses — at the cost of holding a card from a contended pool. Dropped, not
  deferred.
- **cuDNN version** and confirmation that the conda env has not drifted since 2026-04-29 remain
  open. One CPU-only `srun` closes it (§5.3 of the AUX note). Until then the draft claims only that
  the versions were *verified at the outset*, not that each produced every number.
- **No acronym was declared for SLURM.** It is used as a proper noun and never expanded, matching
  how the draft treats the other tool names.

### Mechanical checks

Re-run after the pass: dangling `\ref`/`\autoref`/`\eqref` → **none**; unbalanced environments →
**none**; missing bib keys → **none**; brace delta over both inserted regions → **0**. Line count
2011 → 2118. One new table (`tab:compute`), one new `\hole`, no new citation, no new acronym.
**Still not compiled** — no TeX toolchain in this container.

**Label hygiene:** the new headings carry **no** `\label`. Both are starred, and a `\label` on a
starred heading binds to the enclosing counter, so `\autoref` would print a wrong number — the
first draft of this pass made exactly that mistake with `\label{sec:setup:compute}` and it was
removed. The appendix cross-reference points at `sec:setup:protocol` instead. This matches the file:
none of the other 38 starred subsections is labelled.

---

## v2.9 — 2026-09-11 · abstract halved; the prose stops arguing with a referee

**Asked for:** the abstract is too long; and the tone is wrong — the chapters should *tell the story*
(*"we use this, because …, or even without the because"*), not justify. No *"why not that, why not
this"*.

### Abstract: 481 → 278 words

Rewritten as four short paragraphs that say what was built, in order: engine swap → observation
change → embodiment change → how it is measured, plus one paragraph of methodological results. Every
defensive clause is gone. The `\hole` now carries a hard budget (**keep under ~300 words** once the
headline numbers land) instead of the previous ~550.

### The tone problem, and what was actually wrong

The draft had accumulated a habit of pre-empting objections in the body text: explaining why a choice
was *not* wrong, saying what the thesis *does not* claim, and flagging that something was being stated
*"rather than left to be discovered."* That is arguing with an imagined examiner, and it belongs in a
rebuttal, not a method chapter. Removed across five passes. **No content was dropped — every fact,
number, caveat and deviation is still there; only the defending is gone.**

| pattern | before | after |
|---|---|---|
| `"does not claim"` | 2 | **0** |
| `"must not be …"` | 2 | **0** |
| `"worth stating / worth naming / worth drawing"` | 4 | **0** |
| `"is not a defect / not a concession"` | 2 | **0** |
| `"easy to conflate"`, `"left to be discovered"`, `"is stated here so that"` | 3 | **0** |
| `"is what makes / is what licenses"` (justifying form) | 9 | 3 (all plain explanation in Background) |
| `"rather than"` | 50 | 27 (the survivors are factual contrasts: *joint space rather than task space*) |

Representative rewrites:

- *"The comparison is worth drawing because it locates this thesis honestly rather than flatteringly …
  **Neither is strictly better, and this thesis does not claim its choice is.** What it claims is …"*
  → *"The two designs sit at opposite ends of one trade. … This thesis takes the second end: one
  projector, held fixed across two embodiments, is what RQ4 is about. `γ` is where the cost is
  booked."*
- *"Saying this plainly is the difference between a contribution and an accusation."* → deleted; the
  sentence before it already states the fact.
- *"Matching the engines to each other was judged more important than matching this thesis to the
  source; the alternative would have been to …"* → deleted. The configuration is stated; the
  road-not-taken is not.
- *"The name is the mechanism rather than the source's, for two reasons that … makes good on below:
  … would over-describe it; and … would assert a contribution that is not there."* → *"What runs here
  solves no objective, and the solver it uses is a backend flag, so the arm is named after its
  mechanism."*
- Remark titles: *"The surrogate is a choice, and there is a stricter alternative"* → *"A stricter
  alternative to the surrogate"*.

### One factual error caught while sweeping

The aerial section announced *"the **two** places where the implementation departs from"*
\parencite{lee2010geometric} and then listed **three** deviations. Fixed to three. Same for the
HardFlow setup delta: *"Three of these rows change what the arm is"* → **four** (the dropped terminal
cost was added in v2.6 and the count was never updated).

### Still open

The draft is impersonal throughout (*"this thesis"*, passive). The request's phrasing (*"we use
this"*) would also read well in first-person plural, which is common at I6 — but switching person is a
whole-document decision, so it was **not** done here. One `sed` if wanted.

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · missing bib keys 0 · uncited entries 0 ·
acronyms declared-not-used: TUM only`. 2274 → 2220 lines, abstract 278 words. **Not compiled.**

---

## v2.8 — 2026-09-11 · invented jargon removed, Method reordered into three blocks

**Asked for:** (1) *"bootstrapped target" is invented jargon — name the real scientific
methodology*; (2) the first-order Euler model is an interesting point — contrast it with SafeFlowMPC,
and check whether they train directly on motors; (3) reorder Method into **control substrate →
generative model (with visual as a sub-topic) → projection mechanism**, using de-facto names
throughout.

### 1 · "Bootstrapped target" was my coinage. Removed.

It was **not a term of art** and it hid what the method actually is. Checked the source paper:

- `AlphaFlow.pdf` p. 1 — the MeanFlow objective *"decomposes into two parts: **trajectory flow
  matching** and **trajectory consistency**"*.
- p. 3 — lays out the **consistency-model** family explicitly: CM \[Song et al. 2023\], discrete-time
  **Consistency Training (CT)** with `f_θ⁻ := stopgrad(f_θ)`, and **Consistency Trajectory Models**
  (Shortcut model, MeanFlow).
- p. 6 — α anneals *"from 1 to 0"*, giving *"a family of models in the interpolation between
  trajectory flow matching and MeanFlow"*; they call it a **curriculum learning strategy**.

⇒ The real name is a **consistency target**: the identity closed by a **finite difference at an
intermediate transport time**, taken from a **stop-gradient copy of the network**, where MeanFlow
closes it with an **analytic derivative**. That is ordinary consistency training, not something this
thesis invented a word for.

Applied: §2.4 subsection → *"A consistency target on the same objective"*; §4 → *"Engine 3 — the
consistency target"*; the failure mode now stated plainly (*a finite-difference target closed with
the network's own stop-gradient output inherits a fraction of that network's error*) instead of
being named. `song2023consistency` added to the bibliography (`NO LOCAL COPY`, flagged).
**Zero occurrences of "bootstrap" remain in the tex.** Naming table §1 corrected, with a standing
warning: *do not coin a word for it.*

### 2 · The Euler model vs SafeFlowMPC — checked, and the answer is "close, but not motors"

**Not** direct motor/torque training. Verified in the released code
(`aux_repo/SafeFlowMPC/`): the flow-matching model generates **`n_horizon × 7` joint-configuration
trajectories** (`train_imitation_learning.py`, `max_delta_q = 0.1`) for a KUKA 7-DoF arm — joint
*positions*, not torques. What **is** true, and is the sharper contrast: the NLP carries the
**analytic forward kinematics, the Jacobian and ten per-link collision positions** as compiled
CasADi functions (`safe_flow_mpc/RobotModel/`, `iiwa.urdf`), solved with Acados at **every
flow-matching step**. No surrogate, no mismatch ball — the constraint is imposed on the true
kinematic chain.

Added as `Remark rem:eulermodel`, and written as a **located trade-off, not a concession**: exact
feasibility on one robot's real chain, at the cost of a program built around that robot and a
planner whose output space is its joints — versus an end-effector plan and a projector that transfer
unchanged from a planar arm to a spatial one to a quadrotor, which is what makes RQ4 askable, at the
cost of feasibility holding only for the surrogate. **Neither is strictly better and the thesis does
not claim its choice is**; what it claims is that the surrogate is *why* one projector can be held
fixed across two embodiments, and that `γ` is where the cost is booked.
`oelerich2026safeflowmpc` added (verified against the PDF title page: Oelerich, Ebmer, Hartl-Nesic,
Kugi; ICRA 2026; arXiv 2602.12794).

### 3 · Method chapter reordered — bottom-up, three blocks

Common ground first (**Starting Point → Problem Formalisation → System Overview**), then:

| block | sections | what it is |
|---|---|---|
| **1 · The control substrate** | `sec:method:deployment`, retitled **"The Control Substrate: From Plan to Command"** | first-order Euler model (**moved here** from Starting Point — it is the substrate's model, not the engine's), the shared increment→setpoint skeleton, manipulator differential IK, the 3-D visual instantiation, the aerial geometric cascade, and MJPC |
| **2 · The generative model** | `sec:method:engine`, `sec:method:backbone` | the three engines; **visual conditioning as the sub-topic**, where it belongs |
| **3 · The projection mechanism** | `sec:method:proj`, `sec:method:hardflow`, `sec:method:degenerate`, `sec:method:selection` | iterate projection, endpoint projection, the well-posedness condition, the candidate machinery |

The chapter summary now states the ordering and *why*: each block is bolted onto the one before, so
the substrate has to be on the page first. Block boundaries are marked with banner comments in the
source; no new sectioning level was invented (the template numbers subsections, and adding a level
would change the outline the bone budgeted).

De-facto names throughout — no section or subsection title now carries a brand or a coined term.

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · duplicate labels 0 · missing bib keys 0 ·
uncited entries 0 · acronyms declared-not-used: TUM only · "bootstrap": 0`.
2209 → 2274 lines; 28 bibliography entries. **Not compiled.**

---

## v2.7 — 2026-09-11 · the two transfer contributions get pitched, and the UAV controller gets its real name

**Asked for:** the abstract and the main focus should also carry (a) the **visual** contribution and
(b) the **arm → UAV** extension, including *how* we control it — and the real technical name for the
"UAV PID". Both were worth a pitch and neither was getting one.

### The gap

The visual work and the whole aerial control stack were sharing **half of one sentence** —
contribution 5, *"Two constructed benchmark environments … and a low-level controller"*. The abstract
mentioned them only as a list of three settings the engine was tried on. A reader would have taken
both as evaluation venues rather than contributions.

### Abstract — now pitches four things, in order

Was 3 paragraphs; now 5, with two new middle ones. ~480 words (the `\hole` carries a length budget:
keep under ~550 once headline numbers land, and trim paragraph 1, not the transfer paragraphs).

- **P2, modality.** The contribution is *not* the encoder (vendored unchanged) but the **interface** —
  perception compressed to a fixed-width latent, so the backbone never sees an image and *"only the
  engine changed"* becomes checkable rather than assumed. And that boundary is a **requirement, not a
  convenience**: the average-velocity objective differentiates through its own network, so an encoder
  inside that derivative is both wrong and ruinous. A few-step engine and a visual observation are
  **incompatible without it**.
- **P3, embodiment.** Opens on the question a generative planner usually avoids — *a flow model does
  not fly an aircraft, so what closes the loop?* — and answers it: planner emits a position increment
  and never an actuator command → increment accumulates into a setpoint → **cascaded geometric
  tracking controller** (position PD → thrust vector + attitude → geometric attitude tracking on
  `SO(3)` → static thrust/moment allocation) → four rotor commands, faster than the planner replans.
  Closes on why writing the manipulator and aerial chains as **one structure one layer apart** makes
  RQ4 structural rather than two case studies.

### Contributions — 6 → 7

Contribution 5 split into two standalone items, each with its own argument rather than a noun phrase.
The section header now says **which** items are performance claims (2), constructions (2) and
methodological/negative (3), so the list does not read as seven wins.

### 🚨 The UAV controller's real name — `CascadedPID` is a misnomer on three counts

Checked the code for this pass (`uav_env_test/flight_controller.py:65-70, 120-135`):

1. **There is no integral term anywhere.** The complete gain set is `Kp_pos`, `Kd_pos`, `Kp_att`,
   `Kp_omega` — all proportional or derivative. It is at most a **PD** cascade.
2. **The inner loop is not a scalar loop at all** — a coordinate-free `SO(3)` error plus a gyroscopic
   feed-forward. No arrangement of scalar PID channels reproduces it.
3. **"Cascaded PID" implies nested scalar loops**; this is a two-level **geometric** cascade closed by
   a **static allocation matrix**.

**Thesis name: "the cascaded geometric tracking controller"**, after the lineage it implements
\parencite{lee2010geometric}. `pid` is an artefact tag only. Applied throughout the tex (the
`\acro{PID}` declaration is deleted — it was asserting the wrong thing) and written up as
**`Remark rem:pidname`**, the fourth false name this draft has had to correct.

Recorded in `Auxiliary/Naming/NAMING_20260910_master_table.md` §6 (now the fourth 🚨 row),
`Naming/README.md` (three → four false names) and `AUX_uav_control_stack.md` (new §"It is not a PID").

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · missing bib keys 0 · uncited entries 0 ·
acronyms declared-not-used: TUM only · bare "PID" appears only inside the remark that discusses it`.
2152 → 2209 lines. **Not compiled.**

---

## v2.6 — 2026-09-10 · the dropped terminal cost, and names that were claims

**Asked for:** (1) put the dropped terminal cost `C` into the HardFlow difference section; (2)
organise the naming — one MD in its own subfolder, three columns (repo code │ original paper │ what
it really is / thesis) — and (3) apply it to the tex.

### 1 · The dropped terminal cost `C`

`tab:hardflow-setup` had the *fact* since v2.5; the **consequence** was missing. Added as a new
bullet in *What differs from the published configuration*, making two points:

- **Why dropping it is required, not a defect.** Arm B's projection is pure proximity with no goal
  term. An arm C carrying a goal-seeking objective would beat it on task metrics for a reason
  unrelated to where the program is posed. Dropping `C` is what makes the arm comparison *a
  comparison*.
- **The upside, now stated:** with `C` gone the objective is a pure quadratic prox, whose minimiser
  is weight-independent — so arm C's per-step program **is exactly `Π_S(x̂₁)`**, arm B's operator at
  a different point. *"Identical constraints, identical solver, different linearisation point"*
  becomes literally true. What distinguishes arm C is the lookahead and the damped pull-back, **not
  any optimisation of an objective**.
- **The cost, now stated:** the source's headline on this benchmark is *fewer steps to target*, and
  that is `C`'s doing — in their own table **every projection-only baseline lengthens the path**
  (58.7 → 63.4–67.2) and only the arm carrying `C` shortens it (→ 52.5). **Arm C as run here has no
  mechanism that shortens paths.** A null task-performance result must therefore not be reported as
  evidence against the published method.

### 2 · `Auxiliary/Naming/` — the consolidated table

New folder, two files. `NAMING_20260910_master_table.md` is **canonical**; where it and any other
note disagree, it wins. Three columns throughout — **① code token** (what you meet in `config/`, a
checkpoint path, a CSV `variant` string; never in prose) │ **② paper name** (first mention + citation
only; blank = it is ours) │ **③ what it really is → thesis name**. Seven tables: engines, constraint
arms, selection rules, backbone/conditioning, geometry & protocol, environments & control, plus the
five standing rules.

Consolidates `NOTES_method_naming.md`, `NOTES_naming_and_rebuild.md`, `Rebuild_repo` §5 (🔴 unbuilt,
never cited), `AUX_visual_aligning_env.md` §1.3, `FALLBACK_…` §2.1 and `DATASTATUS_…` §9.11. Both
existing notes are kept as *rationale* and now carry a pointer at the top; `Auxiliary/README.md` and
`NOTES_workspace_layout.md` index the folder.

**Three names that were claims, and were false:**

| was | is |
|---|---|
| `film_mode='v1'` → "FiLM conditioning" | **concatenated conditioning** — FiLM with `γ ≡ 0` (fixed in v2.4) |
| "HF-SLSQP" / "HardFlow" for arm C | **in-ODE endpoint projection** — brand says nothing, and a *solver* in a method name asserts the solver is the contribution |
| §-title "In-Loop Trajectory **Optimisation**" | **In-ODE Endpoint Projection** — our port optimises no objective (see §1 above) |

### 3 · Applied to `thesis_v2.tex`

| site | change |
|---|---|
| `sec:method:proj` title | → **"Constraint Enforcement I: Per-Step Iterate Projection"** |
| `sec:method:hardflow` title | → **"Constraint Enforcement II: In-ODE Endpoint Projection"** |
| arm C opening | now names the mechanism, credits the source *and its mode* in the same sentence, and says **why** the name is not theirs and not the solver's |
| RQ3, `sec:intro:scope`, `sec:bg:mpc`, contributions | arms named **iterate projection** / **endpoint projection** — one adjective each, because the two algorithms differ by exactly one thing |
| engine 3 | "curriculum variant" → **"the bootstrapped target"** (the brand hides the mechanism, and the mechanism *is* the negative result) |
| `sec:rel:safety` | the idea of projecting a predicted endpoint is credited as the source's, explicitly |
| degeneracy table & rule | "in-loop optimisation" → "in-ODE projection" / "the endpoint-projection arm" |
| naming remark's `\srcnote` | repointed at the canonical master table |

Nothing was renamed in the evidence: CSVs, DAs, checkpoint paths and the run ledger keep the
artefact tokens forever (rule 5) — renaming them would break every existing cross-reference.

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · missing bib keys 0 · uncited entries 0`.
2011 → 2151 lines. **Not compiled.**

---

## v2.5 — 2026-09-09 · arm C's published setup is H16/8, ours is H8/1 — now stated

**Asked for:** check whether the methodology records that HardFlow's own experiments run
**H16 / T8**, not **H8 / T1** like ours — and if not, add it.

**It did not.** The draft described arm~C's *algorithm* faithfully (Problem 5, the reversed
receding-horizon instance) and named the two deliberate departures in the program itself, but it
said nothing about the **experimental configuration**, which differs on five axes. A reader would
have assumed the numbers were comparable to the paper's table. They are not.

### Verified, both sides

| | as published | as run here |
|---|---|---|
| plan length | **16 steps** | **8 steps** |
| actions executed per plan | **8** | **1** (replan every env. step) |
| step budget | 10, fixed | swept, 1–20 |
| solver | IPOPT | SLSQP |
| dynamics rows | `s_{i+1} = A s_i + B a_i + c`, least-squares fit | Euler, `t_s = 1` |
| terminal cost `C` | squared distance to target | dropped |
| activation | second half of the steps | second half (`η = 0.5`) ✅ **the one match** |

Sources — paper: `PAPERS/Recommand_Paper/HF/main.tex:1242` (*Experiment Details*, Robotic
Manipulation) states *"The trajectory horizon is H=16 and the replanning horizon is T=8"*, `N=10`
discretisation steps, IPOPT, and the least-squares `A,B,c` fit. Cross-checked against the released
code: `aux_repo/HardFlow/run_scripts/eval_hardflow_new.sh:14,20` (`horizon=16`, `replan_steps=8`)
and `run/eval.py:390-391` (the replan-every-`T`-actions loop, with an assertion
`replan_steps < horizon`). Ours: `config/avoiding-d3il.py:318` (`horizon: 8`) and
`aux_repo/dpcc/scripts/eval.py:203-241`, which calls the policy inside the per-timestep loop and
executes one action.

### Added to `sec:method:hardflow`

New subsection *"What differs from the published configuration"*, with `tab:hardflow-setup` and
three arguments:

1. **Plan length and replanning cuts both ways, and the draft now says so.** A 16-step plan gives
   the endpoint optimisation more room — the geometry is imposed over sixteen future states, not
   eight — so the arm is asked an *easier* question there. But executing 1 action instead of 8 means
   the sampler-plus-projector stack runs **eight times more often per environment step** here, which
   inflates every wall-clock figure relative to the published ones. **A per-replan timing and a
   per-step timing must never be tabulated together.** The choice is defended: matching the engines
   to each other was judged more important than matching this thesis to the source.
2. **Dynamics rows and solver are arm B's**, deliberately — giving the two arms different feasible
   sets or different solvers would turn RQ3 from a comparison of algorithms into a comparison of
   apparatus.
3. **The activation convention is the one row that agrees, and it is the load-bearing one.** The
   published config is `N=10` activating on the second half ⇒ `n_act = 5`, `n_gen = 4` by
   `eq:method:degen:ngen` — **comfortably inside the admissible regime**. So the degeneracy this
   thesis characterises is *not* a defect of the published experiment; it appears once the budget is
   pushed below what that experiment used, which is exactly what a budget sweep does. Written
   explicitly, because that is the difference between a contribution and an accusation.

A `\hole` now forbids merging any quoted number from the source into a table with ours.

### Also

`wachter2006implementation` (IPOPT) added to the bibliography — `NO LOCAL COPY`, named only to
record the solver difference. 26 entries.

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · missing bib keys 0 · uncited entries 0`.
1932 → 2011 lines; 3 tables. **Not compiled.**

---

## v2.4 — 2026-09-09 · the visual U-Net, and a naming trap it exposed

**Asked for:** add the visual U-Net math to the paper; and wire `logs_in_develop/Rebuild_repo/` into
`Writing/` — flagged by the user as *temporary, dynamic, not fixed, **not a reliable source, under
building*** — because it carries one insight worth having: the naming, with the visual U-Net as the
example.

### Added to `sec:method:backbone`

Was three paragraphs of prose; now carries the architecture.

| equation | what |
|---|---|
| `eq:method:backbone:block` | the residual temporal block — `B₂(B₁(h) + (W·φ(z))·1ᵀ) + Ph`. Channels `(d, w, 2w, 4w, 8w)`, three stride-2 stages, horizon zero-padded to a multiple of 8 and cropped back. **Conditioning enters as a per-channel additive bias, constant along the horizon** — the fact the naming remark turns on |
| `eq:method:backbone:latent` | two independent ResNet-18 towers → 64-D each → concat → mean-pooled over the `T_win` window **before** trajectory padding → `e ∈ R¹²⁸`. Stated as *the entire* perception↔generation interface |
| `eq:method:backbone:concat` | **concatenated conditioning** (the shipped default): `z = [φ_τ(τ) ; φ_vis(e)]` |
| `eq:method:backbone:film` | **affine/FiLM conditioning** (the ablation): `(1+γ(e))⊙h + β(e)`, zero-initialised so training starts at the identity |
| `eq:method:backbone:twotime` | the two-time input: `z = [φ_τ(τ) + φ_h(h) ; φ_vis(e)]` — `h` is **added to the time channel** before the visual concat, and the order is not cosmetic |

The JVP short-circuit is now argued rather than asserted: the encoder is pre-evaluated outside the
differentiated closure because its tangent is **identically zero** (`e` depends on none of
`x_r, r, h`) and forward-mode AD through the vendored ResNet is not guaranteed to exist — and it is
deliberately *not* under `no_grad`, because the encoder trains end-to-end and freezing it would
change what is learned.

Source: `models/unet1d_temporal_cond.py:53-82, 205-235`, `unet1d_temporal_film.py:38-92`,
`visual_unet.py:64-141`, `visual_unet_twotime.py:1-45`, `unet1d_twotime_cond.py:279-316`;
widths from `config/aligning-d3il-visual.py:245-246` (`dim=32`, `dim_mults=(1,2,4,8)`).

### 🔴 The trap this surfaced — the draft was calling it FiLM, and it is not

`film_mode='v1'` — **the default, and the source of every reported visual number** — is
`UNet1DTemporalCondModel`: the latent is projected and **concatenated** with the time embedding,
reaching the trajectory as one additive per-channel bias. That is `eq:method:backbone:film` with
`γ ≡ 0`. **Only `v2` is FiLM.** The code knows (`visual_unet.py:64-97` comments the branches *"Fake
FiLM"* / *"True FiLM"*); the misleading token is in the artefacts — every config block, every
checkpoint tag `filmv1`, every eval log.

**v2.2/v2.3 repeated the misnomer** ("conditions the backbone by FiLM", `sec:method:deployment`).
Fixed. Added `Remark` *"The default is not FiLM, and the code's name for it says otherwise"*, and the
thesis now names both arms by **mechanism**, never by flag value.

### Wiring `Rebuild_repo` into `Writing/` — as insight, not as a source

The user's constraint is taken literally: **it is not a reliable source and it is still being
built**, so nothing in the thesis rests on it and the thesis does not cite it. `grep Rebuild
thesis_v2.tex` returns nothing, by design.

New: **`Auxiliary/NOTES_naming_and_rebuild.md`** — the writing-side rule, with a red box stating
that `Rebuild_repo/` is never cited, never treated as done, and never pointed at from a chapter.
Every claim in it was **re-checked against the code** and carries the code line; the rebuild doc is
credited only as *where the question was first asked*. Holds the worked example, the full
artefact→thesis translation table (`diffuser`/`dpcc-c`/`filmv1`/`fm,mf,af`), and the standing rule:
**a name is a claim — if a flag asserts something the code does not do, the thesis must not repeat
it.**

Wired in from: `Auxiliary/README.md` (index + the rule), `NOTES_workspace_layout.md`
(cross-references, with the red warning), `NOTES_open_questions.md` (new items 12 and 13 — the
framing decision, and the missing `app:repro` translation table).

### Corrected in `Methodology_Sources/`

`AUX_visual_aligning_env.md` §1.3 said *"Conditioning is FiLM"* — it was propagating the misnomer
into every chapter written from it. Rewritten, hole closed, diagram label fixed, and the parameters
of record extended with the backbone width, the padding rule and the two conditioning rows.

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · missing bib keys 0 · uncited entries 0`.
1821 → 1932 lines; 49 numbered equations. **Not compiled.**

---

## v2.3 — 2026-09-08 · citations verified against the local PDFs, and pinned to equations

**Asked for:** make the citations full and linked to the papers in `aux_repo/PAPERS` that were
actually read — and specifically do not skimp on the UAV PID-control paper's math or the MJPC math.

### Every entry re-verified against its own title page

`bibliography.bib` was rewritten. Each entry now carries one of two markers:

- **`file = {...}`** — a path relative to `/workspaces/aux_repo/PAPERS/`. Title, authors, year and
  arXiv id were read off **that copy's own title page** (pypdf, 2026-09-08). 10 of 25 entries.
- **`% NO LOCAL COPY`** — harvested from `PAPERS/Recommand_Paper/HF/reference.bib` per
  `NOTES_paper_map.md`, not independently verified. 15 of 25.

arXiv identifiers read off the PDFs (previously absent or from memory):
`romer2025diffusion` 2412.09342 · `janner2022planning` 2205.09991 · `jia2024towards` 2402.14606 ·
`lipman2023flow` 2210.02747 · `geng2025meanflow` 2505.13447 · `zhang2025alphaflow` **2510.20771** ·
`li2025hardflow` 2511.08425 · `yang2025safeflowmatcher` 2509.24243 ·
`lee2010geometric` **1003.2005** · `howell2022predictive` 2212.00541.

**One entry was wrong and is fixed.** `lee2010geometric` was given
`booktitle = {IEEE Conference on Decision and Control (CDC)}` in v2.2 — invented. The PDF is the
arXiv preprint **1003.2005v4**, and the same authors' CDC 2010 paper carries a *different* title
(*Geometric tracking control of a quadrotor UAV on SE(3)*). The entry now cites the preprint that
was actually read, and the `\hole` on it is removed. `howell2022predictive`'s arXiv id was typed
from memory in v2.2 and turned out correct — verified, `\hole` removed. **Both `\hole`s in the
bibliography are gone.**

### The UAV PID math, tied to the paper's equations

`sec:method:deployment` no longer name-drops the source. Added `eq:method:dep:plant` — the rigid-body
model of `[lee2010geometric, Eqs. (2)–(5)]` — and every step of the cascade now names the equation it
realises: errors `(17)–(18)`, thrust magnitude `(23)`, attitude construction `(22)–(23)`, attitude
error `(21)`, moment `(20)`, allocation `(1)`.

**Three deviations from the source are now stated rather than left to be found:**

1. **Frame convention.** The paper writes the plant with `e₃` pointing *down* and thrust along `−b₃`;
   this implementation is z-up with thrust along `+b₃`, so the position/velocity terms carry the
   opposite sign. Same law — but the gains are not transferable between the two statements.
2. **Diagonal gains.** `K_p`, `K_d` are diagonal here (a stiffer vertical channel); the paper proves
   stability for *scalar* `k_x`, `k_v`. The per-axis split is an engineering choice **not covered by
   that proof**.
3. **The attitude feed-forward is dropped.** Setting `ω_des ≡ 0` reduces `e_Ω` to `ω` and kills the
   `−J(Ω̂ RᵀR_c Ω_c − RᵀR_c Ω̇_c)` term of `(20)`. Legitimate only while the commanded attitude turns
   slowly — and `R_des` is driven by a setpoint the **generative model** produces, so its rate is not
   under the controller's control. **The paper's almost-global exponential stability (Prop. 1)
   therefore does not transfer verbatim, and the thesis now says so instead of implying it.**

### The MJPC math, tied to the paper's algorithm

`eq:method:dep:mjpc` is rewritten as an instantiation of the paper rather than an ad-hoc cost:
the finite-horizon objective `[Eq. (3)]`; the weighted-residual running cost `[Eq. (4)]` with our
two residuals `r₁ = p − p_des`, `r₂ = ṗ` and weights `(1, w_v)`; risk-neutral `R = 0` in the risk
transform `[Eq. (5)]`, so it is the identity; the spline parameterisation `[§3.3]` with the
zero-order-hold interpolant, which at one knot per step degenerates to the raw action sequence; and
the search itself — **Algorithm 4**: draw `N−1` samples `θᵢ ~ N(θ, σ²)`, roll out all `N` *including
the incumbent*, keep the argmin. Warm-starting between control steps is `[Alg. 1]`. Defaults given:
`N = 16`, `H = 0.3 s`, 5 iterations, `σ = 0.3`, `w_v = 0.1`.

The one deliberate departure — contacts disabled in the planner's model copy — is now argued rather
than noted: it keeps the constraint mechanism a single identifiable component, so swapping the
tracker cannot silently change how constraints are enforced.

### Arm C pinned to the right formulation

`sec:method:hardflow` cited `[Problem 1]` (the continuous optimal-control statement). It now also
names what is actually implemented: the **reversed receding-horizon instance,
`[li2025hardflow, Problem 5]`** — Euler reference step, optimise over the *predicted endpoint* so the
constraint stays undistorted by the network, then map back. Verified against the port's own header
(`hardflow_projection.py:1–40`, mode `hardflow_new`). The two deliberate departures (our constraint
list instead of the source's hard-coded geometry; terminal cost `C(·)` dropped) are stated as what
makes the arm comparison controlled.

### Locators everywhere else

17 further citations gained verified equation/section locators — MeanFlow `(3)`, `(4)–(6)`, `(7)–(8)`,
`(9)–(11)`, `Tab. 1a`, `§4.3`; Lipman `(5)`, `§2`; DPCC `(9)`, `§3`, `§6.1`. **39 of 85 citations now
carry a locator**; the remaining 46 are positioning citations in Related Work and the introduction,
where a locator would be wrong.

### Checks

`env balanced · braces 0 · $ parity even · dangling refs 0 · missing bib keys 0 · uncited entries 0`.
1806 → 1821 lines; 25 bibliography entries, 10 with a local PDF. **Not compiled.**

---

## v2.2 — 2026-09-08 · the Euler model, the IK chain, and the two embodiment stacks

**Asked for:** the first-order Euler math and the IK mechanism from the DPCC paper; the visual
aligning 3-D and the UAV PID / MJPC chains, written on top of that Euler trajectory.

### Added

| where | what | source |
|---|---|---|
| `sec:method:dpcc` → *The dynamics model: first-order Euler* | `eq:method:dpcc:euler` — `s_{t+1} = s_t + [aᵀ aᵀ]ᵀ t_s + w_t`, plus the three consequences: the model is deliberately crude and `γ` absorbs the rest; **`t_s = 1` because the action is a displacement, not a velocity**; both the commanded and the measured channel are anchored | DPCC §6.1 (p. 8); `aux_repo/dpcc/config/projection_eval.yaml:14–16`; `config/uav_projection.yaml:125–127` |
| `sec:method:proj` → *The dynamics rows, written out* | `eq:method:proj:deriv` (the plain row) and `eq:method:proj:derivnorm` (**the row as implemented, conjugated by the normaliser** — dropping the affine offset `Σ` silently biases every dynamics row) | `aux_repo/dpcc/diffuser/sampling/projection.py:344–402` |
| `sec:method:deployment` | **rewritten from 3 paragraphs to a full section.** Shared skeleton `eq:method:dep:setpoint`/`:obs`; `tab:embodiments`; the manipulator IK chain; the visual-aligning 3-D instantiation; the aerial stack; the MJPC alternative | see below |
| `tab:notation` | deployment block appended, with the three collisions resolved explicitly | — |
| `bibliography.bib` | `lee2010geometric`, `howell2022predictive` (both `[OWN]`, both with a `\hole` on the venue/id) | `PAPERS/auxiliary_papers/Drone/PID_Control_UAV.pdf` p. 1; `PAPERS/auxiliary_papers/Mujuco/Mujuco_MPC.pdf` p. 1 |
| acronyms | `IK`, `PID`, `MJPC` declared and `\ac{}`-ed at first prose use (in the section opener, deliberately *before* `tab:embodiments`, so the expansion does not land inside a table cell) | — |

### The three chains, and where each equation comes from

- **Manipulator (`avoiding-d3il`, `aligning-d3il-visual`).** The eval forms the absolute setpoint
  from the increment and the *commanded* channel (`aux_repo/dpcc/scripts/eval.py:236–238`), then a
  weighted damped-least-squares differential IK with null-space posture regularisation and SVD
  clipping (`eq:method:dep:taskerr`–`:dls`) runs `n_ik = 3` iterations per step and hands a joint
  target to a joint-space PD law.
  Source: `d3il/.../controllers/IKControllers.py:134–323` +
  `d3il/.../controllers/Config/mujoco_controller_config.gin:6–37`.
  **All gains are now in the thesis** — see the note under *Holes closed*.
- **Visual aligning 3-D.** `eq:method:dep:varows`: plan width `d = 9`, layout
  `(a, p_des, p)`, and the six dynamics pairs `{(3,0),(4,1),(5,2)} ∪ {(6,0),(7,1),(8,2)}` — the
  three-dimensional writing of the DPCC four-row pattern. Perception sits *above* the plan, so the
  control chain below the setpoint is identical to the state benchmark's; that is what licenses
  reading a result change as a modality change.
  Source: `mix_visual_aligning_test/eval_mix_visual_aligning.py:272–279`; `config/aligning-d3il-visual.py:446`.
- **Aerial.** Multi-rate loop `eq:method:dep:multirate` (`n_dec = 3` at 33 Hz plan / 100 Hz physics);
  outer position PD → thrust vector `eq:method:dep:outer`; attitude extraction from the thrust
  direction and commanded yaw `eq:method:dep:att`; **geometric SO(3) attitude error**
  `eq:method:dep:inner` with the gyroscopic term; thrust/torque allocation `eq:method:dep:alloc`
  with the thrust-first saturation rule; the three `v_des` synthesis policies `eq:method:dep:vdes`;
  and the sampling-based tracker's cost `eq:method:dep:mjpc`.
  Source: `uav_env_test/flight_controller.py:33–155`; `mix_uav_test/eval_mix_uav.py:1454–1455,
  1551–1563, 1822–1832`; `mix_uav_test/mjpc_tracker.py:70–155`;
  `d3il/.../quadrotor/quadrotor_modified.xml:4` (timestep 0.01 s).

### Holes closed in `Auxiliary/Methodology_Sources/`

- `AUX_uav_control_stack.md` §5 said **"the PID gains are nowhere in the logs"**. They are in the
  code: `Kp_pos = (4,4,8)`, `Kd_pos = (3,3,4)`, `Kp_att = (70,70,4)`, `Kp_ω = (2.5,2.5,1)`,
  `u_hover = mg/4`, `u_max = max(2u_hover, 6)`, `T_floor = 0.1mg`, `g = 9.81`. All are now in
  `sec:method:deployment`. The AUX file has been updated to point at the source.
- `v_des` synthesis is documented "as a plan, not as shipped" in the same file. The shipped code has
  **three** policies, not one; all three are now written as `eq:method:dep:vdes`.

### Still open after this pass

- The MJPC controller's status is written with a `\hole`: the apparatus notes say every reported UAV
  number comes from the cascade and that MJPC is built-but-not-a-source, and that must be confirmed
  against the run ledger before it is claimed in print.
- `ω_des = 0` in `eq:method:dep:inner` is stated as a modelling choice ("acceptable only because the
  tracked setpoints are slow"). It is a candidate threat-to-validity and is not yet in `sec:disc:threats`.
- Yaw policy at the trajectory layer is still undocumented upstream; `ψ_des` appears in the
  equations but nothing says how it is chosen.

### Checks

`env balanced · braces 0 · dangling refs 0 · missing bib keys 0 · uncited bib entries 0 ·
acronyms declared-not-used: TUM only (inherited from the template)`.
1414 → 1734 lines; 43 numbered equations, 6 aligned blocks, 3 tables. **Not compiled.**

---

## v2.1 — 2026-09-08 · template conformance

**Asked for:** "the writing style is strict same as the LaTeX template yes?" — it was not.
Audited `thesis_v2.tex` against `Template_DONT_CHANGE/{main.tex, settings.tex,
chapters/01_introduction.tex}`.

### Fixed — a real bug

`settings.tex:68` already registers the bibliography (`\bibliography{bibliography}` is biblatex's
legacy form of `\addbibresource`). v2 added `\addbibresource` unconditionally, which would have
registered the resource **twice** in the merged build. It now lives inside the `\ifstandalone`
branch only.

### Brought into line

- `\ref` → `\autoref` (43 → 42; `settings.tex:44–52` defines the autoref names, so the template is
  built for it). The one survivor is `Appendix~\ref{app:repro}` — no `\appendixautorefname` is
  defined, so `\autoref` would render "Chapter C".
- `\cite` → `\parencite` (65), the form the template's own chapter demonstrates.
- 7 locator citations `(\cite{key}, Theorem 1)` → `\parencite[Theorem~1]{key}`, which is also what
  the I6 rule on page numbers in citations requires.
- Acronyms: 15 declared and **0 used** — with `printonlyused` the Abbreviations page would have come
  out empty. Now 7 declared, each reached by `\ac{}` at its true first prose use; the 8 the prose
  never abbreviates were deleted. Long forms set in sentence case where they expand mid-sentence.
- Tables: `[htpb]` and `\caption[short]{long}`.
- Front matter: the `\standalonefalse` branch now copies `main.tex`'s order exactly
  (`pages/cover` → `\frontmatter{}` → title / disclaimer / acknowledgments / abstract).

### Left divergent, deliberately

Single file with `\ifstandalone`; `amsmath`/`amssymb`/`amsthm` added (**the template ships no maths
package at all** — carry these into `settings.tex` on merge); the `\hole`/`\srcnote` draft macros;
label prefixes `ch:`/`sec:` kept because `TARGET_20260905` §7 addresses sections by those names;
starred `\subsection*` — **a decision left to the author**, one `sed` either way.

---

## v2.0 — 2026-09-08 · the mathematics and the citations

**Asked for:** copy v1 into v2 and inject the DPCC diffusion math → FM math → MeanFlow math (α-Flow
deferred), academically, with citations, sourced from the papers in `aux_repo/PAPERS` and the repo's
own code.

### Added

- **Background.** Diffusion (`eq:bg:ddpm:forward`–`:mean`), flow matching
  (`eq:bg:fm:ode`–`:euler`), MeanFlow (`eq:bg:mf:def`–`:loss`), and the constrained-sampling
  vocabulary (`eq:bg:mpc:proj`–`:dyn`).
- **Method.** The constrained denoising step (`eq:method:dpcc:thm`/`:step`), tightening, the problem
  formalisation, the three-line engine delta (`eq:method:engine:iface`), the **start-anchored**
  MeanFlow derivation as this repo implements it (`eq:method:engine:mfid`–`:mfsample`), the
  projection NLP, and the genuine-step boundary (`eq:method:degen:ngen`).
- `bibliography.bib`, 23 entries, biblatex `style=alphabetic` + biber copied verbatim from the
  template's `settings.tex`.
- `tab:notation` — resolves the `t`/`x`/`τ`/`u` collision flagged in
  `NOTES_tum_formatting_rules.md` (open question 4).

### Recorded as findings, not smoothed over

- **Remark 4.1, the prior-scale discrepancy.** The inherited sampler starts from `N(0, ¼I)` and
  injects *half* the posterior standard deviation; naive FM inherits the ½ prior at sampling while
  training on `σ = 1`; MeanFlow uses `σ = 1` at both ends. Carries a `\hole` demanding that
  `sec:setup:protocol` state whether the reported FM numbers were re-run at matched scale.
- α-Flow is cited and demoted to an ablation with **no mathematics**, per TARGET §8 kill criterion 1.

### Correction worth remembering

The first α-Flow bibliography entry was written **from memory and was wrong in title and in every
author**. Corrected against `AlphaFlow.pdf` p. 1 (Zhang, Siarohin, Menapace, Vasilkovsky, Tulyakov,
Qu, Skorokhodov — Snap Inc. / U. Michigan). This is why every `[OWN]` entry in the bib carries a
verify note, and why the sourcing rule at the top of this file exists.
