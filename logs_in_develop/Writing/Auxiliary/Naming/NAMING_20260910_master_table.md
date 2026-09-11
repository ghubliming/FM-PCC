# NAMING MASTER — code name │ paper name │ what it really is → thesis name

**Created:** 2026-09-10 · **Type:** the single translation table for the whole thesis
**Status:** 🟢 **canonical.** Where this file and any other note disagree, this file wins.
**Applied to:** [`../../Working_Space/v2/thesis_v2.tex`](../../Working_Space/v2/thesis_v2.tex) (v2.6)

**Consolidates:** [`../NOTES_method_naming.md`](../NOTES_method_naming.md) (the method-name argument
and the author's position) · [`../NOTES_naming_and_rebuild.md`](../NOTES_naming_and_rebuild.md) (the
code-flag traps) · `logs_in_develop/Rebuild_repo/CONCEPT_unified_rebuild.md` §5 (🔴 unbuilt scratch
doc — **never cited**, see that note) · [`../Methodology_Sources/AUX_visual_aligning_env.md`](../Methodology_Sources/AUX_visual_aligning_env.md) §1.3
· [`../../Working_Space/fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md`](../../Working_Space/fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md) §2.1
· [`../../Working_Space/data_status/DATASTATUS_20260910_v3_entry_readiness.md`](../../Working_Space/data_status/DATASTATUS_20260910_v3_entry_readiness.md) §9.11

> **How to read the three columns.**
> **① code** — the token you will actually meet, in `config/`, in a checkpoint path, in a CSV
> `variant` string, in a folder name. **Never appears in thesis prose.**
> **② paper** — what the originating paper calls it. Used at **first mention, with a citation**, and
> then retired. Blank means *there is no source — it is ours*.
> **③ what it really is → thesis name** — the mechanism, and the words the thesis uses from the
> second mention on.
>
> 🔴 **A name is a claim.** If ① or ② asserts something the code does not do, the thesis must not
> repeat it. Three entries below are exactly that case and are flagged 🚨.

---

## 1. Generative engines

| ① code | ② paper | ③ what it really is → **thesis name** |
|---|---|---|
| `ddpm`, `diffusion`, `diffuser`, `GaussianDiffusion` | DDPM \[Ho et al. 2020\]; Diffuser \[Janner et al. 2022\] | the inherited denoising engine, cosine schedule, `K` fixed at training time → **"the diffusion engine"** / **"the DPCC baseline"**. 🚨 **never bare `diffuser`** — it means three different things (Janner's method, the ancestor codebase, *and* our no-projection row) |
| `fm`, `FMv3ODE`, `flow_matcher_v3` | Flow Matching \[Lipman et al. 2023\]; Rectified Flow \[Liu et al. 2023\] | the **instantaneous-velocity** target; the `α = 1` endpoint of the target axis → **"flow matching"** (already descriptive — keep) |
| `mf`, `MeanFlowODE`, `MeanFlowEngine` | MeanFlow \[Geng et al. 2025\] | the **analytic average-velocity** target via a JVP; the exact `α → 0` limit → **"MeanFlow"** (keep — published *and* descriptive; gloss once as *the average-velocity objective*) |
| `af`, `AlphaFlowODE`, `alphaflow` | α-Flow / AlphaFlow \[Zhang et al. 2025\] | 🚨 **not a family, one target.** `u_tgt = α·v + (1−α)·u_next` with `u_next` a frozen forward pass — a finite difference closed with the model's **own** output; `0 < α < 1`, the interior of the same axis → first mention **"α-Flow \[cite\]"**, thereafter **"the bootstrapped target"**. The brand hides the mechanism, and **the mechanism is the whole negative result** |
| `imf`, `iMF` | Improved MeanFlow | refuted variant → appears only as a **negative result** in `sec:disc:negative` |
| `unet` · `dit` · `sit` · `mf_dit` | U-Net \[Janner et al.\]; DiT; SiT | backbone choice. Only `unet` is architecture-matched; transformer rows are **confounded secondary evidence** and carry parameter counts |

> **The axis framing is the naming fix.** `fm`, `af`, `mf` are not three methods — they are
> `α = 1`, `0 < α < 1`, `α = 0` on one target axis. Say that once and the engine chapter stops being
> a shopping list.

## 2. Constraint arms

| ① code | ② paper | ③ what it really is → **thesis name** |
|---|---|---|
| *(no projector; `diffuser` variant row)* | — | no constraint enforcement → **arm A, "unguided"** |
| `dpcc`, `dpcc-r/c/t`, `pcc` | DPCC per-step projection \[Römer et al. 2025\] | `Π_S` applied to the sampler's **current iterate**, which at early transport times is not yet a plan → **arm B, "per-step iterate projection"** (short form **IP**) |
| `hardflow_sls`, `hardflow_new`, `HF`, **"HF-SLSQP"** | HardFlow \[Li et al. 2025\], mode `hardflow_new` = their Problem 5, *reversed receding-horizon* | 🚨 `Π_S` applied to the **predicted clean endpoint**, then a τ-damped pull-back. With the optional terminal cost `C` dropped, the NLP **is exactly `Π_S(x̂₁)`** — **a projection, not an optimisation** → **arm C, "in-ODE endpoint projection"** (short form **EP**) |

🚨 **Three things are wrong with "HF-SLSQP".** It leans on a brand (says nothing); it puts a
**solver** in a method name (asserts the solver is the contribution — it is not, and the activation
gate is solver-agnostic); and "HardFlow" over-attributes, since what runs here is *our*
implementation of *one* of their four modes on *our* constraint stack, with `C` removed.

🚨 **`CascadedPID` is the third false name** (§6) — no integral term, and the inner loop is geometric,
not scalar. Recorded in `thesis_v2.tex` as `Remark rem:pidname`.

🚨 **"In-Loop Trajectory Optimisation" was also wrong** — it was the v2 section title until
2026-09-10. Our port optimises no objective. Retitled **"In-ODE Endpoint Projection"**.

**Why the contrastive pair works:** the two arms differ in *exactly one thing* — the point handed to
an identical operator. `iterate` / `endpoint` says that in one adjective each, and it makes the
degeneracy result readable from the names alone: at the terminal step the endpoint **is** the
iterate, so EP collapses to IP.

**Credit line to use at first mention of arm C:** *"…the endpoint projection introduced by
\parencite{li2025hardflow} (their `hardflow_new` mode), re-derived here on our constraint set."*

## 3. Candidate selection

| ① code | ② paper | ③ what it really is → **thesis name** |
|---|---|---|
| `-r` | — (the convention DPCC argues against) | pick one at random → **random** |
| `-c` | DPCC §5.4 *Cumulative Projection Cost* | pick the candidate the projector altered least → **cumulative projection cost** |
| `-t` | DPCC §5.4 *Temporal Consistency* | pick the candidate closest to last step's plan → **temporal consistency** |

**Never lettered in prose.** Table headers may use the letters *after* definition.

## 4. Backbone and visual conditioning

| ① code | ② paper | ③ what it really is → **thesis name** |
|---|---|---|
| `VisualUNet` | — (ours) | ResNet vision encoder → 128-D latent → 1-D temporal U-Net → **"the vision-conditioned temporal U-Net"** |
| `VisualUNetTwoTime` | — (ours) | same, with the `h` interval input → **"its two-time variant"** |
| `film_mode='v1'`, `filmv1`, `UNet1DTemporalCondModel` | — (borrows FiLM \[Perez et al. 2018\] **incorrectly**) | 🚨 **NOT FiLM.** Latent projected and **concatenated** with the time embedding ⇒ one **additive per-channel bias**, constant along the horizon; i.e. FiLM with `γ ≡ 0` → **"concatenated conditioning"** |
| `film_mode='v2'`, `UNet1DTemporalFiLMModel` | FiLM \[Perez et al. 2018\] | the real thing: per-block `γ` scale + `β` shift, `h ← (1+γ)⊙h + β`, zero-initialised → **"affine (FiLM) conditioning"** |

🚨 **The trap in full.** The *default* arm — the source of **every reported visual number** — is
selected by a flag whose name contains "FiLM", and that word then propagates into every config
block, every `filmv1` checkpoint tag and every eval log **for the arm that is not FiLM**. The code
itself is honest (`visual_unet.py:64-97` comments the branches *"Fake FiLM"* / *"True FiLM"*); the
misleading token is in the artefacts. Verified: `unet1d_temporal_cond.py:53-82, 205-235` (additive)
vs `unet1d_temporal_film.py:38-92` (affine).

## 5. Geometry, protocol and budget

| ① code | ② paper | ③ what it really is → **thesis name** |
|---|---|---|
| `tightened`, `-tightened` | DPCC Thm 2, `S̃ = S ⊖ B_γ` | constraint sets shrunk by the model-mismatch ball → **"tightened geometry"** (an experimental factor, **not** a filename suffix) |
| `u7hg`, "honest geometry" | — (ours) | the corrected scene feasibility geometry → **"the corrected geometry"**; the defect and its correction are a **methodology strength**, written up in `sec:disc:threats` |
| `model_free`, `bounds_free`, `geo_free` | — (ours) | constraint-**family ablations** → named by what they switch off. 🚨 **never ranked against legal arms** — they win by not enforcing |
| `A`, `activation_threshold`, `diffusion_timestep_threshold` | the "second half of the steps" heuristic \[Li et al.\]; \[Römer et al.\] | the transport fraction on which the projector fires → **`η`**, *"the activation threshold"* |
| `n_genuine` | — 🔑 **ours, not theirs** | active steps that are **not** the terminal one → **"genuine steps"**. Say whose it is: the published config sits at `n_gen = 4`, comfortably admissible; the degeneracy appears only below it |
| `K`, `n_diffusion_steps`, `flow_steps_v3` | NFE | the step budget → **`K`**, *"step budget"*. 🔑 the asymmetry is visible in the key itself: **training-time** for diffusion, **inference-time** for the transport family |
| `budget_ms`, `33 Hz`, `real_time_OVER` | — | a **data-rate artefact** plus cluster latency → 🚨 **never** a real-time pass/fail criterion anywhere |

## 6. Environments, embodiment and control

| ① code | ② paper | ③ what it really is → **thesis name** |
|---|---|---|
| `avoiding-d3il` | *Avoiding* \[Jia et al. 2024\] | 2-D planar state manipulator; DPCC's own benchmark → **"the state-based manipulation benchmark"** |
| `aligning-d3il-visual` | *Aligning* \[Jia et al. 2024\], vision pipeline theirs | 3-D, two RGB views; **the pairing is ours** → **"the vision-conditioned manipulation task"** |
| `uav`, `uav_mix`, `mix_uav` | — (ours; Skydio X2 from `mujoco_menagerie`) | quadrotor embodiment → **"the aerial task"** |
| `empty` · `corridor` · `s_curve` · `pillars` · `corridor_ball` | — (ours) | scene names are already descriptive → keep as-is, define the geometry once |
| `pid`, `CascadedPID`, run tag `pid` | geometric tracking control on SE(3) \[Lee et al. 2010\] | 🚨 **a misnomer on three counts.** (i) **No integral term exists** — the only gains are `Kp_pos`, `Kd_pos`, `Kp_att`, `Kp_omega` (`flight_controller.py:65-70`), so it is at most a **PD** cascade. (ii) The inner loop is not a scalar loop at all: a coordinate-free `SO(3)` error plus a gyroscopic feed-forward, which no arrangement of scalar PID channels reproduces. (iii) "cascaded PID" implies nested scalar loops; this is a **two-level geometric cascade** closed by a **static allocation matrix**. → **"the cascaded geometric tracking controller"** (position PD → thrust vector + attitude → geometric attitude tracking on `SO(3)` → thrust/moment allocation) |
| `pid_stopgo`, `pid_const_v` | — (ours) | velocity-setpoint synthesis policies → **"brake-to-rest"** / **"constant-speed"** |
| `mjpc`, MJX | MuJoCo MPC / *Predictive Sampling* \[Howell et al. 2022\] | zero-order sampling planner over spline knots → **"the sampling-based predictive tracker"** |

---

## 7. The five standing rules

1. **First mention = published name + citation. Every later mention = the mechanism name.**
2. **Never let a code tag into prose** — `hardflow_sls-r`, `dpcc-c-tightened`, `filmv1`, `af`, `mf`,
   `u7hg`. Table headers may use short forms *after* definition.
3. **Never put a solver in a method name.** SLSQP vs IPOPT is a row in `app:repro`, not an identity.
4. **Never invent a name for someone else's method.** A renamed method cannot be found by a reader.
   Rename *our instantiation*; credit *their idea* in the same sentence.
5. **Translate at the thesis boundary, never in the evidence.** CSVs, DAs, checkpoint paths and run
   ledgers keep the artefact tokens forever — renaming them breaks every existing cross-reference.

## 8. 🔴 The one thing this file does not yet discharge

**`app:repro` owes the reverse table** — thesis name → the exact tokens a reader will meet in the
released logs and checkpoints. Columns ① and ③ here are half of it; the other half is the run ledger
(`Slurm_Codes/logs/important_runs/important_runs.md`) plus the checkpoint-tag spellings. **Rename in
prose without shipping that table and no reader can match a results row to a checkpoint.**
Tracked as [`../NOTES_open_questions.md`](../NOTES_open_questions.md) item 13.
