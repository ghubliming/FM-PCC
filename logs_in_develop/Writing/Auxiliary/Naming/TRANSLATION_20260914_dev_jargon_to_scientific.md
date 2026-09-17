# TRANSLATION — dev jargon of this repo → its scientific name

**Created:** 2026-09-14 · **Last extended:** 2026-09-17 by v3.20 (generative-objective names, §2 and §8) · **Status:** 🟢 **canonical translation list for all drafts** (v2, v3, v4).
**Supersedes as the working list:** [`NAMING_20260910_master_table.md`](NAMING_20260910_master_table.md)
— every row of that table is carried over below (§2–§7), and the new rows are marked 🆕. The old file
is kept for its argument, not as a second list.

**How to use.** Left: the token you meet in code, configs, run tags, CSVs, dev logs and chat.
Middle: the name the source paper uses, if any. The restricted brands identified in §8 appear only
in Related Work; elsewhere the paper is credited by citation. Right: the scientific name — what the
thesis writes.

| verdict | meaning |
|---|---|
| ✅ | the dev term **is** the scientific term — keep it |
| 🔁 | rename in the thesis; the artefacts keep the old token |
| 🚫 | never in thesis prose (process vocabulary, internal shorthand) |
| 🚨 | the dev name asserts something false |

Artefacts (CSVs, DAs, checkpoints, run ledgers) **keep** their tokens; the translation happens only
in thesis text.

---

## 1. 🆕 Terms that are already correct — keep them

| dev term | verdict | evidence / note |
|---|---|---|
| **backbone** (the U-Net / DiT network) | ✅ | Standard. DPCC p. 8: *"We use a 1D U-Net … as the diffusion model backbone"*; DiT p. 1: *"a convolutional U-Net architecture as the de-facto choice of backbone"*; MeanFlow p. 8: *"B/4 backbone"*. |
| **NFE** (number of function evaluations) | ✅ | Standard in the diffusion/flow literature; MeanFlow and α-Flow both report *1-NFE / 2-NFE*. Defined at first use in Ch. 1 and formally in §2.3 (v2.13). |
| **JVP** (Jacobian–vector product) | ✅ | MeanFlow §4.1. |
| **receding horizon**, **MPC** | ✅ | Standard control terms. |
| **ODE**, **ODE solver**, **Euler step** | ✅ | Standard. |
| **SLSQP**, **IPOPT** | ✅ | Solver names — but never inside a method name (§3). |
| **U-Net**, **DiT**, **SiT**, **ResNet-18** | ✅ | Architecture names. |
| **EMA** (weight average) | ✅ | Exponential moving average of weights; standard. |
| **inpainting** (conditioning on the current state) | ✅ | DPCC §6.1, Janner et al. |
| **MuJoCo**, **MuJoCo Menagerie** | ✅ | Simulator and model collection; cite them. |

## 2. Generative models

| dev jargon | paper name | scientific name → thesis | |
|---|---|---|---|
| `ddpm`, `diffusion`, `diffuser`, `GaussianDiffusion` | DDPM [Ho 2020]; Diffuser [Janner 2022] | the denoising diffusion model of DPCC, cosine schedule, `K` fixed at training → **"the diffusion model"** / **"the DPCC baseline"**. Never bare *diffuser*: it means Janner's method, the ancestor codebase *and* the unprojected row | 🔁🚨 |
| `fm`, `FMv3ODE`, `flow_matcher_v3` | Flow Matching [Lipman 2023; FM Guide, Lipman 2024 — FAIR at Meta]; Rectified Flow [Liu 2023] | matches the velocity at one transport time → **"instantaneous-velocity matching"**; **FM** in all figures and tables | 🔁 |
| `mf`, `MeanFlowODE`, `MeanFlowEngine` | MeanFlow [Geng 2025] | predicts velocity averaged over an interval; its regression target is constructed analytically through a Jacobian--vector product → **"analytic average-velocity matching"**; **MeanFM** in all figures and tables | 🔁 |
| `af`, `AlphaFlowODE`, `alphaflow`, "alpha flow", "consistency target", "consistency training" | α-Flow [Zhang 2025] | predicts average velocity using a target interpolated by the consistency step ratio between instantaneous velocity and a stop-gradient network prediction → **"consistency-interpolated average-velocity matching"**; **CI-MeanFM** in all figures and tables. Write the final ratio as **`α_end = 0.2`**, never *floor*. *Consistency Training* is a different published method. Source LaTeX (Def. 1, Alg. 1 training, Alg. 2 schedule) stored at `PAPERS/auxiliary_papers/DGM/AlphaFlow_src/` (v2.22) | 🔁 |
| `imf`, `iMF` | Improved MeanFlow | refuted variant → only as a negative result | 🔁 |
| `unet` · `dit` · `sit` · `mf_dit` | U-Net; DiT; SiT | backbone choice; only `unet` is architecture-matched to the baseline | ✅ |
| 🆕 **DGM** | — | **deep generative model** — spell out; the abbreviation is not used in the thesis | 🔁 |
| 🆕 **SOTA** | — | **state of the art** — only when a source says so, and cite it (MeanFlow: *"outperforming previous state-of-the-art one-step diffusion/flow models"*; α-Flow: *"new state-of-the-art results"*) | 🔁 |
| 🆕 "engine" | — | **generative model** (or the model's name). Removed from all thesis prose in v2.14; kept only inside `\label` names | 🔁 |

> `fm`, `mf` and `af` are one target-construction axis: **instantaneous-velocity matching**,
> **analytic average-velocity matching**, and **consistency-interpolated average-velocity matching**.
> These mechanism names, not the paper brands or artefact tokens, are used outside Related Work. After
> their first definition, all figures and tables use **Diffusion**, **FM**, **MeanFM** and
> **CI-MeanFM** consistently; their captions or surrounding descriptions retain the full-name mapping.

## 3. Projection and constraint enforcement

| dev jargon | paper name | scientific name → thesis | |
|---|---|---|---|
| *(no projector)*, `diffuser` variant row | — | no constraint enforcement → **"unguided"** | 🔁 |
| `dpcc`, `dpcc-r/c/t`, `pcc`, 🆕 "PCC", 🆕 "the old naive DPCC projector" | DPCC's *model-based projection* [Römer 2025] — its central contribution | `Π` applied to the sampler's **current iterate** → **"per-step (iterate) projection"**. Never *naive* or *old* | 🔁 |
| `hardflow_sls`, `hardflow_new`, `HF`, "HF-SLSQP" | HardFlow [Li 2025]; implemented = their Algorithm 1 (Problems 2–5 lead to it), *reversed receding-horizon* | `Π` applied to the **predicted clean endpoint**, then a damped pull-back; with terminal cost `C` dropped the NLP **is exactly `Π(x̂₁)`** → **"endpoint projection"** everywhere in the body, with `\parencite{li2025hardflow}` in place. 🚫 The name *"HardFlow"* only in Related Work (§8 rule 1, v2.15). 🚨 no solver in the name | 🔁 |
| 🆕 **arm A / arm B / arm C**, "the two arms" | — | **unguided / iterate projection / endpoint projection**, "the two projection methods". Clinical-trial vocabulary; names nothing | 🔁 |
| 🆕 **projector** | — | **projection** (the operation) / **projection method** (the algorithm) | 🔁 |
| 🆕 **S&C** | — | **success with constraint satisfaction** (the episode reaches the goal with no violation) | 🔁 |
| 🆕 **Target** (DA sense), **flagship** | — | **the baseline configuration** / **the main configuration** | 🚫 |
| `A`, `T`, `activation_threshold`, `diffusion_timestep_threshold` | the *second-half* heuristic [Li]; [Römer] | fraction of the sampling steps on which projection is active → **activation threshold `η`** | 🔁 |
| `n_genuine`, 🆕 "genuine steps", `n_gen` | — (ours) | 🚨 *genuine* is a value word: it says a step is real without saying what it does. The quantity counts active steps that are **not** the final one — the only steps where a later step still exists for the network to react to, so the only steps where projection steers the sample being generated instead of correcting a finished one. → **"guiding steps"**, the **guiding step count** $n\sidx{guide}$. Artefacts keep `n_genuine`. *(v3, 2026-09-15; §6.1.5 heading was "Genuine Steps" → "Steps at Which Projection Guides Sampling". Ch. 4 §4.5.4 changed in v2.17: title, `n\sidx{guide}`, prose.)* | 🔁 |
| 🆕 degenerate / thin / admissible (tiers), 🆕 "three regimes" | — | the three cases of $n\sidx{guide}$: $0$ (no projection runs inside the ODE at all) / $1$ / $\ge 2$. Name them by the count, not by a tier word; *regime* and *tier* are on the banned list. Define once in §4.5.4 (`sec:method:degenerate`). *(Ch. 4 changed in v2.17: the table names the three cases by the count.)* | 🔁 |
| `post_processing` | DPCC's *Post-Processing* baseline | **projection after sampling** | 🔁 |
| `model_free`, `bounds_free`, `geo_free` | — (ours) | constraint-family ablations, named by what they switch off; never ranked against full-constraint rows | 🔁 |
| `tightened`, `-tightened` | DPCC Thm 2, `S̃ = S ⊖ B_γ` | constraint sets shrunk by the model-mismatch ball → **"tightened constraints"** | 🔁 |

## 4. Candidate selection

| dev jargon | paper name | scientific name → thesis | |
|---|---|---|---|
| `-r` | — | **random selection** | 🔁 |
| `-c` | DPCC §5.4 *Cumulative Projection Cost* | **cumulative projection cost** | 🔁 |
| `-t` | DPCC §5.4 *Temporal Consistency* | **temporal consistency** | 🔁 |
| 🆕 **fan**, fan width, `B`, `batch_size` (eval) | — | **number of candidate plans `B`** | 🔁 |

## 5. Network, conditioning and training

| dev jargon | paper name | scientific name → thesis | |
|---|---|---|---|
| `VisualUNet` | — (ours) | **vision-conditioned temporal U-Net** | 🔁 |
| `VisualUNetTwoTime` | — (ours) | its two-time variant (takes the interval `h`) | 🔁 |
| 🆕 "visual layer", `MultiImageObsEncoder`, `get_resnet`, "dual ResNet" | Diffusion Policy visual encoder [Chi 2023, §3.2], adopted by D3IL [Jia 2024, p. 6] | **visual encoder**: one ResNet-18 [He 2016] per camera view, spatial-softmax pooling (32 keypoints) [Mandlekar 2021] + linear layer → 64 features, GroupNorm [Wu & He 2018], trained end-to-end; two views → 128-D latent. Code: D3IL's vendored robomimic `VisualCore`. (v2.17) | 🔁 |
| `film_mode='v1'`, `filmv1`, `UNet1DTemporalCondModel`, "FiLM", "fake FiLM", ~~"concatenated conditioning"~~, "the shipped default" | *feature-wise conditional bias* — FiLM paper [Perez 2018, §3]: *"concatenating conditioning information with fully-connected layer input amounts to a feature-wise conditional bias"* (verified, `VLA/FiLM.pdf` p. 3) | 🚨 **not FiLM**: latent concatenated with the time embedding ⇒ additive per-channel bias, constant in time → **"feature-wise conditional biasing"** with `\parencite[Sec.~3]{perez2017film}`. The only conditioning the thesis has (v2.16) | 🔁🚨 |
| `film_mode='v2'`, `UNet1DTemporalFiLMModel` | FiLM [Perez 2018] | 🚫 **never described as something this thesis has** (author, v2.16): no reported run uses it. **Amended (author, 2026-09-17, v2.22):** the *name* FiLM is introduced once, in the background section on vision-conditioned policies, as the published mechanism whose bias half we use — `\ac{FiLM} \parencite[Sec.~3]{perez2017film}`. Method prose still says **feature-wise conditional biasing** and never presents a scaling variant as ours | 🚫 |
| 🆕 **shipped**, "shipped default", "shipped threshold" | — | release vocabulary → name the value (*η = 0.5*) or say nothing | 🚫 |
| 🆕 **Two facts that are not free choices**, "not cosmetic", "by construction rather than by hope", "rather than by omission" | — | defensive meta-prose — delete, state the fact | 🚫 |
| 🆕 `aw`, `action_weight` | Diffuser/DPCC loss weighting | **action loss weight** | 🔁 |
| 🆕 `H8`, `horizon` | — | **planning horizon of 8 steps** | 🔁 |

## 6. Protocol, budget and analysis vocabulary

| dev jargon | paper name | scientific name → thesis | |
|---|---|---|---|
| `K`, `n_diffusion_steps`, `flow_steps_v3` | NFE | **step budget `K`** = NFE. Training-time key for diffusion, inference-time key for flow matching | ✅ |
| 🆕 `n_trials`, seeds, "5×20" | DPCC's *trials* | **episodes per seed**; state seeds × episodes | 🔁 |
| `budget_ms`, `33 Hz`, `real_time_OVER` | — | data-rate artefact + cluster latency — 🚨 never a real-time criterion | 🚫 |
| `u7hg`, "honest geometry" | — (ours) | **corrected scene geometry** | 🔁 |
| 🆕 **Pareto / "better"** | — | never defined globally in the thesis; state the metrics compared where a claim is made | 🚫 |
| 🆕 ladder, rung, kill criterion, fallback, gate, DA, cell, row | — | internal planning vocabulary | 🚫 |
| 🆕 Gen11, U7, Fix_N, Epoch, campaign, mission, job id | — | development history — only in `app:repro` if at all | 🚫 |
| 🆕 **funnel**, "three-stage funnel", "Stage 1 / 2 / 3" (v3, V_A and UAV analyses) | — | a comparison order, not a method. Write what is compared: *"the generative models are compared on the final box-to-target distance of plans executed without projection; constraint violations and wall-clock time are compared separately"*. (v3.7) | 🚫 |
| 🆕 **regime**, "all-pass", "discriminating", "all-fail" (scene / task taxonomy) | — | coined labels. State the fact instead: *"every configuration satisfies the constraints"*, *"no configuration satisfies the full constraint set in every context"*. (v3.7) | 🚫 |
| 🆕 **Entry 1 / 2 / 3**, "the three entries" | — | the environment names: **obstacle avoidance**, **vision-conditioned alignment**, **quadrotor benchmark**. (v3.7) | 🔁 |
| 🆕 **Tier 1 / Tier 2** (protocol tiers) | DPCC §6.1: *five training seeds and ten test seeds* | **the evaluation protocol of DPCC** (5 training seeds, 10 test episodes per geometry) / **the protocol of this thesis** (5 training seeds × 20 episodes). (v3.7) | 🔁 |
| 🆕 **strict S&C**, **crossed-line S&C**, `cfree`, "crossed" (UAV) | — | **success with constraint satisfaction**, with success = *reaching within 0.30 m of the goal point* or *crossing the finish line*; `cfree` = **collision-free flight**. (v3.7) | 🔁 |
| 🆕 `untouched`, `frozen` (V_A) | — | **the box is not moved**. (v3.7) | 🔁 |
| 🆕 `min_xy_dist`, `context_final_xy_dist`, MIN / median (V_A) | — | **final distance between box and target in the table plane**, summarised by its **minimum** and **median** over the contexts. Box angle: **final orientation of the box relative to the target**. (v3.7) | 🔁 |
| 🆕 "reproducibility floor" (~0.4 m, V_A) | — | **run-to-run variation** of repeated evaluations. (v3.7) | 🔁 |
| 🆕 "controller-limit case" (s_curve) | — | say what is observed: *"the tracking controller does not follow the plans through the turns"*. (v3.7) | 🚫 |
| 🆕 "the slide", `corridor_v2_slide`, `cv2s` | — | **the slide**: a sloping test-time halfspace in the corridor, defined once in Ch 5 (`tab:uav-scenes`); `corridor_v2` is the corridor with walls 1.90 m apart. (v3.7) | ✅ |
| 🆕 "the consistency target", "consistency training", "floor 0.2" | — | **consistency-interpolated average-velocity matching**; where the final consistency step ratio is reported, write **`α_end = 0.2`** (§2, v3.20). | 🔁 |
| 🆕 "headline", "the foundation", "the paper's foundation", "held table" | — | planning vocabulary. (v3.7) | 🚫 |

## 7. Environments, embodiment and control

| dev jargon | paper name | scientific name → thesis | |
|---|---|---|---|
| `avoiding-d3il`, "avoiding" | *Avoiding* [Jia 2024] | **obstacle-avoidance task** (planar, state-based; DPCC's benchmark) | 🔁 |
| `aligning-d3il-visual`, `V_A` | *Aligning* [Jia 2024] | **vision-conditioned alignment task** (built from D3IL) | 🔁 |
| 🆕 "robot arm + IK" | — | **position-controlled manipulator with differential inverse kinematics** and a joint-space PD law | 🔁 |
| 🆕 "instable / unstable dynamic system" | — | **underactuated, open-loop unstable** system (four rotor inputs, six pose degrees of freedom; hover needs continuous stabilisation) | 🔁 |
| `uav`, `uav_mix`, `mix_uav` | — (ours) | **quadrotor** (Skydio X2 from MuJoCo Menagerie) | 🔁 |
| 🆕 "UAV thrust" | — | **rotor thrust commands** (collective thrust and body moments, allocated to four rotors) | 🔁 |
| 🆕 "pure state form" | — | **from state observations** | 🔁 |
| `empty` · `corridor` · `s_curve` · `pillars` · `corridor_ball` | — (ours) | scene names are descriptive — keep, define the geometry once | ✅ |
| `pid`, `CascadedPID` | geometric tracking control on SE(3) [Lee 2010] | 🚨 **not a PID**: no integral term, the inner loop is geometric on `SO(3)`, closed by a static allocation matrix → **"cascaded geometric tracking controller"** | 🔁🚨 |
| `pid_stopgo`, `pid_const_v` | — (ours) | velocity-setpoint policies → **brake-to-rest** / **constant-speed** | 🔁 |
| `mjpc`, MJX, 🆕 "mujoco baseline" | MuJoCo MPC, *Predictive Sampling* [Howell 2022] | **sampling-based MuJoCo predictive control (MJPC)**; planner = Predictive Sampling | 🔁 |
| 🆕 `decim`, `n_dec` | — | **control-rate ratio** (physics steps per planning step) | 🔁 |
| 🆕 `p_des`, `v_des` | — | **position setpoint**, **velocity setpoint** | 🔁 |

---

### 🆕 Environment names when several appear together (v3.16b, 2026-09-16)

| context | name |
| :-- | :-- |
| several environments shown or compared together (tables, summaries, conclusions, RQ answers) | **D3IL-avoiding**, **D3IL-aligning**, **UAV-corridor**, **UAV-pillars**, **UAV-s-curve** |
| inside one environment's own section | plain names are fine: *the corridor*, *the pillars*, *obstacle avoidance* |

Defined once at the start of v3 §5.1. `UAV` is spelled out there; it is not yet in the acronym list (v2).

## 8. Rules

1. **The thesis names a method by its mechanism, and credits the paper by an in-place citation.**
   The published brands *MeanFlow*, *α-Flow* and *HardFlow* appear only in Related Work (Ch. 3,
   outside *Positioning*) and in explicit provenance or artefact mappings. Elsewhere write
   **instantaneous-velocity matching**, **analytic average-velocity matching**,
   **consistency-interpolated average-velocity matching**, and **endpoint projection**, with the
   source credited by `\parencite`. *Flow matching* remains the established name of the transport
   family and part of the fixed thesis title; *DPCC* stays as the baseline name. *(Author,
   2026-09-17, v3.20; supersedes both the v3.18 brand-name exception and the v2.20 objective names.)*
2. No dev token in thesis prose. Table headers may use short forms after definition.
3. No solver in a method name.
4. Do not rename someone else's method; name *our instantiation*, credit *their idea*.
5. Translate at the thesis boundary; artefacts keep their tokens.
6. When a draft adds a term here, it says so in its own CHANGELOG (`DRAFT_OWNERSHIP.md`, shared material).
7. **MuJoCo is the simulator; D3IL is a source of task files and data.** Every environment runs in MuJoCo.
   D3IL supplies the Panda model and task files, the demonstrations and the camera placements of the two
   manipulation tasks. Write *rendered by MuJoCo*, *a MuJoCo frame*, *the task files / demonstrations of
   D3IL*; never *D3IL simulator view*, *D3IL's simulation* or a figure label *(D3IL)* on a MuJoCo render.
   The environment prefixes *D3IL-avoiding* / *D3IL-aligning* stay (they name the task origin).
   *(Author, 2026-09-17, v3.26.)*
8. **No statistical-test reporting in the thesis.** No $p$-values, no test names (sign, permutation,
   Fisher), no "significant". Those are internal assessments. Report the evidence the reader can see:
   counts over contexts, episodes or flights (*closer in nine of ten contexts*, *12/12 against 0/12*),
   margins, and the sample size. Recap sections restate who is best, without statistics.
   *(Author, 2026-09-17, v3.26.)*
