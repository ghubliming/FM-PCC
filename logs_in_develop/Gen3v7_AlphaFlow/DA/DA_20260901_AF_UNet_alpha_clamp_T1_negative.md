# DA 2026-09-01 — AF-UNet on `avoiding-d3il`: goal reached & steps, `diffuser` arm

**Jobs:** `25251` train (clamp 0.05, seed 7, 4 h 17 m, clean) · `25254` eval (clean) · `25253` eval control (**crashed**)
**Logs:** `temp/3008/2026-08-31/{18_16_05_train_alphaflow_25251, 18_19_42_eval_alphaflow_25253, 18_19_54_eval_alphaflow_25254}.log`
**Data:** `temp/3008/batch_avoiding_combined_20260901_093057/candidates_multidimensional_aggregated.csv`
**Task:** `avoiding-d3il`, H8, U-Net `freq_dim=32` (4.0 M params)

*Rewritten from scratch 2026-09-01 on the metric below. The previous version accumulated five layers
of amendment and was unreadable; nothing of substance is dropped, only the layering.*

---

## 0. The objective

> ## 🎯 **Make α-Flow beat MeanFlow on the U-Net. Nothing else is in scope.**
>
> **Success criterion — `diffuser` arm, seed 6, `n_trials = 20`, top-right-hard, K = 1:**
> **goal reached > 0.85 AND steps < 59.70** (MF-UNet's seed-6 numbers, §2.2).
>
> Everything in this DA is scored against that one line. The DPCC and naive-FM columns are context,
> not targets. The clamp forensics (§3), the epoch A/B and the schedule overlay are **diagnostics
> that only matter if they change what we train next** — §5 states which ones do.

**The single most important fact for this objective (§3.1): α-Flow ends on MeanFlow's objective **by
design** — verified against upstream — so its claim is that the *curriculum* reaches a better model
than direct MeanFlow training. The claim is well-posed. What breaks it on our U-Net is that the
curriculum's only signal, the probe `dt = α·h ≈ 0.0013–0.013`, is below the resolution of a time
embedding with ~4 frequencies. The plan follows from fixing **that**, not the schedule.**

---

## 1. The metric

**`diffuser` arm only** — the unprojected rollout. No MPC, no DPCC, no HardFlow. This is the raw
network output, which is the only thing that speaks to "is AF's U-Net better than MF's U-Net".

**Two numbers, in this order:**

| | column | meaning |
|---|---|---|
| **1. primary** | `n_success` | **goal reached or not.** Fraction of episodes that reached the target. |
| **2. secondary** | `n_steps` | **how many steps it took.** Only meaningful *alongside* success. |

**Read them as a pair, never separately.** A failed episode stalls at ≈ 30 steps, so a low step
count next to a low success rate is a **stall, not efficiency**. Every AF row below with `steps ≈ 30`
is an agent that stopped moving.

**Only `top-right-hard` discriminates.** `top-left-hard` and `both-hard` are 1.00 success for every
model on this task, so they carry no information and are omitted. All tables below are TR.

S&C (`collision_free_completed`) and violation counts are **not used** anywhere in this DA.

---

## 2. Results — everything on the `diffuser` arm, top-right-hard

### 2.1 The models that have real statistics (5 seeds × 20 trials = 100 episodes)

| model | backbone | params | K | **goal reached** | **steps** |
|---|---|---|---|---|---|
| **MF-UNet** | U-Net | 4.0 M | 1 | **0.97** | **61.5** |
| **MF-UNet** | U-Net | 4.0 M | 2 | **0.99** | 63.5 |
| **MF-UNet** | U-Net | 4.0 M | 5 | **0.99** | 63.9 |
| **MF-UNet** | U-Net | 4.0 M | 10 | 0.97 | 65.0 |
| **MF-UNet** | U-Net | 4.0 M | 20 | 0.97 | 65.4 |
| naive FM | U-Net | 4.0 M | 1 | 0.97 | 66.0 |
| naive FM | U-Net | 4.0 M | 2 | 0.99 | 63.2 |
| naive FM | U-Net | 4.0 M | 5 | 0.99 | 75.5 |
| **DPCC baseline** (GaussianDiffusion, aw10) | U-Net | 4.0 M | 20 | 0.92 | 70.6 |

**Two results fall straight out of this table, and neither has been written down before:**

1. **MF-UNet Pareto-dominates the DPCC baseline on the unprojected arm.** At **K = 1**: reaches the
   goal *more* often (0.97 vs 0.92) in *fewer* steps (61.5 vs 70.6) with **20× fewer network
   evaluations**. Same backbone, same parameter count — this is the architecture-matched comparison
   ([[architecture-matched-beat-is-the-strong-claim]]), and it is a clean win by
   [[pareto-definition-of-good]].
2. **MF-UNet ≥ naive FM at every K, strictly better at K = 1 and K = 5.** K=1: same success, 4.5
   fewer steps. K=5: same success, 11.6 fewer steps. K=2 is a tie. This clears the
   [[benchmark-hierarchy-who-beats-whom]] requirement that MeanFlow beat naive FM.

### 2.2 The seed-6 bar — what AF has to beat, and why seed 6 is the right seed

**Seed choice: 6** (user, 2026-09-01). Matched seed-6, `n_trials = 20` references already exist for
every competitor (`candidates_multidimensional_raw.csv`, same arm, same halfspace):

| model | K | **goal reached** | **steps** |
|---|---|---|---|
| **MF-UNet** | **1** | **0.85** | **59.70** ← **the bar** |
| MF-UNet | 2 | 0.95 | 61.35 |
| MF-UNet | 5 | 0.95 | 62.80 |
| MF-UNet | 10 | 0.85 | 64.15 |
| MF-UNet | 20 | 0.90 | 64.40 |
| naive FM | 1 | 0.85 | 62.45 |
| naive FM | 2 | 0.95 | 63.05 |
| naive FM | 5 | 0.95 | 71.15 |
| naive FM | 20 | 1.00 | 67.70 |
| DPCC baseline | 20 | **0.60** | 57.25 *(low steps ⇒ failures, not efficiency — §1)* |

> ### 🎯 **AF-UNet beats MF-UNet iff, at K = 1, it reaches the goal more than 0.85 of the time in fewer than 59.70 steps.**

**Seed 6 is the better choice and it is worth saying why.** It is the only seed where success is
**not at ceiling**: every model scores 1.00 on seed 7, so seed 7 can only discriminate on steps.
Seed 6 discriminates on **both axes at once**, and it is the seed where the DPCC baseline actually
struggles (0.60). A win here is a real win; a win on seed 7 is a tiebreak.

*(For reference, the seed-7 bar is MF-UNet K = 1 at 1.00 / 60.70 steps, naive FM 1.00 / 71.30, DPCC
1.00 / 82.30. Kept only as a secondary check.)*

### 2.2.1 ⚠️ Why the recent AF work sat on seed 7 — and what seed 6 costs

Not arbitrary, and not a free switch. **AF-UNet's seed 6 in the baseline tree is a stale pre-Fix_8
253 M checkpoint.** Report §2.2 documents it from the train log:

```
[ AFTrajectoryModel ] backbone=unet  unet_width(freq_dim)=32  params=4.0M   ← the model BUILT
[ train ] Seed 6 already reached 100000 steps — skipping                    ← the weights ON DISK
```

`--auto-resume` found an old 100 k checkpoint in the `_bbunet_` tree and skipped seed 6; Gen3v6
cleared its tree first, Gen3v7 did not. That is why seeds 7–10 are called "the 4 valid seeds of
24389" everywhere, why job 25251 trained seed 7, and why the `AF_SEEDS` patch comment names it.

**Consequences for seed-6 work:**

| what | seed 6 available? |
|---|---|
| **New training runs** (§5 step 0, `af_alpha_end`) | ✅ **yes, no obstacle** — a new `_ae` value writes to its own tree, so `--auto-resume` has nothing stale to find |
| Eval of the **T1 `_ac0.05` tree** | ❌ **no seed 6 exists** — job 25251 ran `AF_SEEDS='7'` only. Needs a 4.2 h retrain |
| Eval of the **baseline `_rf0.5` tree** | ❌ **must not be used** — the seed-6 weights are the stale 253 M model. Needs `rm -rf …/6/` then a 4.2 h retrain (report §10.6 says exactly this) |

So: **seed 6 for everything we train from here on; the three existing-checkpoint evals can only run
at seed 7** unless we spend two retrains. Both are listed in §5.

### 2.3 AF-UNet — every row is `n_trials = 2`

| model | K | epoch | clamp | **goal reached** | **steps** | episodes |
|---|---|---|---|---|---|---|
| AF-UNet | 1 | best | 0.005 | **0.00** ⚠️ | 29.5 *(stall)* | 2 |
| **AF-UNet (T1)** | 1 | **latest** | **0.05** | **1.00** | 62.0 | 2 |
| AF-UNet | 2 | best | 0.005 | 0.80 | 55.2 | 10 |
| AF-UNet (T1) | 2 | latest | 0.05 | 0.50 | 47.5 | 2 |
| AF-UNet | 5 | best | 0.005 | **0.00** ⚠️ | 30.5 *(stall)* | 2 |
| AF-UNet | 10 | best | 0.005 | **0.00** ⚠️ | 31.0 *(stall)* | 2 |

**The K = 1 row is the interesting one.** The baseline model **never reached the goal** (0/2, stalled
at 29.5 steps). The T1 model **reached it every time** (2/2, 62.0 steps — the same step count as
MF-UNet's 61.5). That is the same difference the human eye saw in the plans (§4), now visible in the
success column.

**But n = 2.** Two episodes. This is an observation, not a measurement, and it is confounded (§4.2).

### 2.4 ⛔ The gap that blocks the whole question

> **AF-UNet has never been evaluated at `n_trials = 20`.** Every AF-UNet number above is
> `n_trials = 2`; the `_msg20trials` α-Flow folders in the batch are **SiT**, not U-Net
> (`DA_20260817_AF_SiT_ntrials20_K1_K2.md`).
>
> MF-UNet has 100 episodes at every K. AF-UNet has 2, or 10 at K = 2. **So "is AF's U-Net better or
> worse than MF's" cannot be answered from existing data at any K.** This is the largest gap in the
> study — larger than the clamp question — and it closes with one eval job (§5 step 1).

---

## 3. What T1 actually did to training

T1 raised `af_alpha_clamp` 0.005 → 0.05, intended to lengthen the pure-MeanFlow tail.

**It did not do only that.** The clamp is **symmetric** — `af_diffusion.py:472-476`:

```python
if ratio < clamp_value:          ratio = 0.0     # → pure MeanFlow
elif ratio > 1.0 - clamp_value:  ratio = 1.0     # → pure FM
```

With α(p) = σ(−25(p − 0.5)), the snap points are p = 0.5 ∓ ln((1−c)/c)/25:

| clamp | α = 1 head ends | α = 0 tail starts | **steps with a genuine bootstrap (0 < α < 1)** |
|---|---|---|---|
| 0.005 (baseline, 24389) | 28 827 | 71 173 | **42 346** |
| **0.05 (T1, 25251)** | **38 222** | **61 778** | **23 556 (−44 %)** |

So T1 is **not** "the same run with a longer MeanFlow tail" — it is a run with **44 % less α-Flow**.
18 790 steps of genuine bootstrap were replaced by the two degenerate endpoints of the homotopy:
plain FM at the head (α = 1 ⇒ `u_tgt = v` for *every* h, i.e. the wrong answer for the averaged
field), pure MeanFlow at the tail (α = 0 ⇒ the Gen3v6 JVP target, unmodified — α-Flow is switched
**off**).

**Training-side field quality got worse** (matched seed 7, matched final step, α = 0 on both sides so
the targets are byte-identical):

| seed-7 final-step `val/` | clamp 0.005 | clamp 0.05 | MF-UNet |
|---|---|---|---|
| `raw_mse_u` | 6.86 | **15.78** | **1.90** |
| `per_dim_rms_u` | 0.336 | **0.418** | **0.199** |
| `h_mse` b0 / b1 / b2 / b3 | — | 3.48 / **23.6** / **41.1** / 5.08 | — |
| `raw_mse_v` | — | 3.88 *(healthy)* | — |

The damage is localised to `h > 0` (buckets b1/b2) and to `u`, not `v` — the instantaneous field is
fine, the averaged field is not.

**So the two sides disagree:** the training metric says T1 is 2.3× worse; the rollout at K = 1 says
T1 went from never reaching the goal to always reaching it. Both are real. §5 resolves it.

### 3.0 The complete list — everything α-Flow changes relative to MeanFlow

Read off the two loss bodies: `flow_matcher_v3_meanflow/models/mf_diffusion.py:381-478`
(`_p_losses_meanflow`) and `flow_matcher_v3_alphaflow/models/af_diffusion.py:529-745`
(`compute_u_target` + `_p_losses_alphaflow`). **Five differences, nothing else.**

| # | | **MF (Gen3v6)** | **AF (Gen3v7)** | tunable? |
|---|---|---|---|---|
| **1** | **the `u` target** | always `u_tgt = v + h·du/dr`, one JVP | **α = 1** → `u_tgt = v` (plain FM)  ·  **0 < α < 1** → `u_tgt = (dt·v + (h−dt)·u_next)/h`, `dt = α·h`  ·  **α = 0** → *the identical MF JVP target* | via α only |
| **2** | **extra network call** | none — the JVP primal is reused | one **extra `no_grad` forward** at `(x_r + dt·v, r + dt, h − dt)` to get `u_next` | no |
| **3** | **α schedule** | — | `sigmoid`, 1.0 → 0.0, **γ = 25**, `clamp 0.005`, over `[0, n_train_steps]` | ✅ `af_alpha_scheduler`, `af_alpha_gamma`, `af_alpha_init/end`, `af_alpha_init/end_step`, `af_alpha_clamp` |
| **4** | **target clamp** | none | `af_clamp_utgt = 4.0`, applied **only** to the bootstrapped branch — the FM anchors and the JVP target are untouched | ✅ `af_clamp_utgt` |
| **5** | **per-sample loss weight** | 1.0 for every sample | 1.0 for FM anchors and JVP samples; **α** for bootstrapped samples (`w_br`, `af_diffusion.py:736-738`) | via α only |

**Identical in both** — so none of these can explain a difference: the `v` head as a full second loss;
`ratio_fm` / `meanflow_data_proportion` = **0.5** in both; the `logit_normal` min/max `(t, r)` pair;
`action_weight = 10`; the adaptive loss; the backbone; the parameter count.

**Consequence.** AF's entire distinctive content is rows 1–5, and rows 1, 2 and 5 are *driven by α*.
So **the only levers that change what α-Flow does — without touching the network — are the α
schedule (row 3) and `af_clamp_utgt` (row 4).** That is the whole search space under a frozen
backbone, and §5 works it.

### 3.1 Is the working AF setup actually just MeanFlow? — **at the endpoint yes, and that is upstream's own design**

The deployed checkpoint is `latest` = **step 80 000**. Decomposing the 80 000 steps that produced
those weights, under clamp 0.05:

| steps | α | what the network was actually trained on | share |
|---|---|---|---|
| 0 – 38 222 | **1.0** | `u_tgt = v` — **plain Flow Matching** (gate G1) | 47.8 % |
| 38 222 – 61 778 | 0 < α < 1 | `u_tgt = α·v + (1−α)·u_next` — **genuine α-Flow** | 29.4 % |
| 61 778 – 80 000 | **0.0** | the Gen3v6 JVP target, **unmodified** — **pure MeanFlow** (gate G2) | 22.8 % |

The last 18 222 gradient steps before the saved checkpoint were pure MeanFlow;
`af_diffusion.py:552` routes `alpha <= 0.0` into a branch whose comment reads *"Gen3v6's
`_p_losses_meanflow` body, UNMODIFIED"*.

**✅ Checked against upstream, 2026-09-01 — this is faithful, not a porting artefact:**

- `aux_repo/alphaflow/src/training/loss.py` splits the batch on **`mask_c = (dt == 0)`**, and
  `dt = (t − t_next)·α`, so **α = 0 routes to `_compute_mean_velocity_c`** — which is a `jvp` call
  computing `velocity_cfg − (t − t_next)·du/dt`. That **is** the MeanFlow target.
- **Every** upstream schedule ends at zero: `infra/experiments/experiments-alphaflow.yaml:155,160,165`
  all carry `end_value: 0`, and `configs/loss/alphaflow.yaml:41-42` defaults to `0.0 → 0.0`.

> ### So α-Flow's claim is a **curriculum** claim, not a different-objective claim.
> It ends on MeanFlow's objective **on purpose**. The paper's assertion is that the path
> **FM → bootstrapped → MeanFlow** reaches a *better* model than training MeanFlow directly.
> Same destination, better route.
>
> **This means "AF beats MF" is exactly the paper's claim and is perfectly well-posed at
> `af_alpha_end = 0`.** We are not "doing nothing" by annealing to zero — we are running the method
> as specified. ⚠️ **An earlier draft of this DA proposed `af_alpha_end = 0.05` to "keep AF as AF";
> that is withdrawn — it would deviate from the paper, not implement it.**

**What this does say about our situation.** AF-UNet's problem is not that it becomes MeanFlow — it is
that **the curriculum lands it in a worse basin than direct MeanFlow training does.** We pay the
curriculum's cost and collect none of its benefit. And there is a mechanical reason to expect exactly
that on this backbone (report §9.1): the curriculum's only distinctive signal is the bootstrap probe
`dt = α·h ≈ 0.0013–0.013`, and the U-Net's time embedding is `SinusoidalPosEmb(dim)` with `dim = 32`
(`unet1d_temporal_cond.py:114,121`) — the frequency count is **tied to the channel width**, giving
~4 resolving frequencies on [0,1] against the SiT's ~32. Upstream runs α-Flow on large latent-video
DiTs, where that probe is easily visible. **On a 4-frequency time code the middle phase of the
curriculum teaches the U-Net almost nothing, while still moving its weights.**

**That is the thing to fix, and it is what §5 step 1 does.**

**It also answers "why did AF-UNet suddenly work?"** — `best` is selected on
`test_loss ≈ 0.75 + 0.25·α`, which structurally prefers a **mid-homotopy α ≈ 0.01–0.02** checkpoint,
i.e. a model caught mid-curriculum. `latest` is the model **after** the curriculum finished. T1 has
twice as much of that final phase as the baseline would (18 222 steps vs 8 827). AF-UNet looks good
when it is evaluated at the **end** of its curriculum rather than in the middle of it.

## 4. The plans, looked at by eye

### 4.1 The observation

Inspected on the cluster by the user (qualitative, unblinded, n = 1 observer), `diffuser` arm:

| | plans path (under `logs/avoiding-d3il/plans/flow_matching_v3_alphaflow/H8_D…_ag25.0_rf0.5/`) |
|---|---|
| **A** (T1) | `H8_K1_Meuler_T0.5_A0.5_B4_D…AlphaFlowODE_msgac05_latest` |
| **B** (baseline) | `H8_K1_Meuler_T0.5_D…AlphaFlowODE` |

> **Verdict (user): A is smoother than B and reaches MF-UNet level. A win.**

§2.3 corroborates it on the success column at the same K: 0/2 → 2/2.

### 4.2 What differs between A and B — four things

| # | difference | A | B | drives the result? |
|---|---|---|---|---|
| 1 | **checkpoint epoch** | `latest` = step 80 000, **α = 0** | `best` = the **mid-homotopy α ≈ 0.009–0.023** model | **prime suspect** |
| 2 | **`af_alpha_clamp`** | 0.05 | 0.005 | possible — but §3 shows it made the field 2.3× worse |
| 3 | eval config | `A0.5_B4` | none | **no** — HardFlow settings never touch the unprojected arm, and at K ≤ 2 every HardFlow arm is auto-disabled by the degeneracy guard |
| 4 | seeds | 7 | 7 | no |

Difference 2 is measured and points the wrong way; 3 and 4 are inert. **That leaves the epoch.** If
`latest` really beats `best`, then §4.3 of the report is not a footnote: **every AF-UNet number ever
recorded came from a handicapped mid-homotopy checkpoint, and AF vs MF has never been measured
fairly.** `best` is selected on `test_loss ≈ 0.75 + 0.25·α`, which structurally prefers mid-homotopy.

### 4.3 Two operational faults found on the way

- **`latest` is always step 80 000, never 100 000.** `save_freq = n_train_steps // 5` = 20 000 and
  the loop ends at 99 999, so `step % 20000 == 0` never fires at 100 k
  (`utils/training.py:81,203-205`). Any text describing "the 100 k endpoint" describes a model that
  was never saved.
- **The baseline tree has no numbered checkpoints at all.** Job 25253 died with
  `state_-1.pt`: `get_latest_epoch` (`utils/serialization.py:27-37`) returns −1 when zero numeric
  `state_*.pt` files match — only `state_best.pt` survives under `..._ag25.0_rf0.5/7/`. So
  `AF_EPOCH=latest` is unavailable for the baseline. **`/data` was at 100 % (27 G free of 7.0 T)** —
  a plausible cause and an operational risk on its own. Needs a human `ls` on the cluster.
- **Path-hygiene bug (code change, not made).** `_af_clamp_tok` is applied to `diffusion_loadpath`
  (`config/avoiding-d3il.py:1640`) but **not** to the plans savepath — which is why A and B share a
  parent directory. They stayed separate only because `FMPCC_RUN_MSG` supplied a distinct leaf.
  **Without it, a re-clamped eval silently overwrites the baseline's plans.**

---

## 5. The plan — one goal, ordered by whether it moves the goal

**Seed 6 throughout. `n_trials = 20`. `diffuser` arm. K = 1 is the decision point** (K 2/5/10/20
collected in the same job for free).

### 🔒 Hard constraint (user, 2026-09-01)

> **The U-Net does not change. The parameter count does not change — 4.0 M, exactly.**

That removes report §10.4 (`time_embed_freq_dim`, +0.06 M) and §10.5 (`E_t + E_r`) from the plan.
**An earlier draft of this section proposed the +0.06 M width change; it is withdrawn.** Per §3.0 the
remaining search space is exactly two things: **the α schedule** and **`af_clamp_utgt`** — plus one
infrastructure fix that is not a method change at all.

### 🎯 Step 1 — **ENABLE α-FLOW.** `AF_ALPHA_END > 0` (**config only, zero params — WIRED**)

**The premise, and it is correct:** every AF run this project has ever deployed is a MeanFlow model.
`af_alpha_end = 0.0` anneals α to exactly zero, and `af_diffusion.py:552` routes `alpha <= 0.0` into
Gen3v6's MeanFlow JVP body, unmodified. **The bootstrapped target has never trained a single weight
of any checkpoint we have evaluated.** So α-Flow's actual objective is untested on this task — the
comparison "AF vs MF" has to date been "MF-with-a-curriculum vs MF".

`AF_ALPHA_END` sets the terminal α, so the discrete branch stays live to the last step:

| `af_alpha_end` | α at 68 k | **α at 80 k (`latest`)** | α at 100 k | probe `dt = α·E[h]`, `E[h] ≈ 0.25` | deployed model is |
|---|---|---|---|---|---|
| **0.0** (today) | 0.011 | **0.0006** | 0.0000 | 0.0001 | ❌ **MeanFlow** |
| **0.05** | 0.060 | **0.0505** | 0.0500 | 0.0126 | ✅ α-Flow |
| 0.1 | 0.110 | 0.1005 | 0.1000 | 0.0251 | ✅ α-Flow |
| **0.2** | 0.209 | **0.2004** | 0.2000 | 0.0501 | ✅ α-Flow, 4× the probe |

**Run 0.05 and 0.2.** 0.05 is the minimal change that switches AF on — α lands just above the old
`clamp 0.005`, so the snap never fires. 0.2 exists because of the resolution argument (§3.1): a
0.0126 probe may still be under the U-Net's ~4-frequency time code, and 0.2 makes it 4× larger. If
0.05 is dead and 0.2 is alive, that *is* the resolution result, measured rather than argued.

**Upstream sanction.** Every upstream recipe uses `end_value: 0`, but upstream also ships
`discrete_training: true` (`aux_repo/alphaflow/src/training/loss.py:421-426`) whose entire purpose is
to floor α instead of snapping to zero — i.e. upstream provides a switch for exactly this intent.
`AF_ALPHA_END` reaches the same state with no code port. **Not an invented knob.**

**Status: wired 2026-09-01.** `config/avoiding-d3il.py:129` reads `AF_ALPHA_END` (default 0.0 ⇒
byte-identical to today) and feeds both the train block (`:952`) and the plan block (`:1599`), which
must match or eval finds no checkpoint. `af_alpha_end` is an **unconditional** `args_to_watch` token
(`'ae'`), so each value already trains into its own `_ae<val>` tree — no `--auto-resume` collision,
which matters at seed 6. Both sbatch scripts now echo the value.

**One command per arm — `alphaflow_pipeline.sh` chains train → eval with `afterok`**, and
`submit.sh` submits with `--export=ALL`, so the whole `AF_*` set is baked in at submit time and the
dependent eval carries it hours later:

```bash
# ⚠️ /data was at 100 % (27 G free of 7.0 T). Free space first — these write weights AND plans.
# ⚠️ Seed 6 in the BASELINE tree is the stale pre-Fix_8 checkpoint (§2.2.1). A new _ae tree is
#    unaffected, but never point a seed-6 run at `_ae0.0_`.

# --- arm A: AF ON, minimal (alpha -> 0.05) ---
AF_BONE=unet AF_ALPHA_END=0.05 AF_SEEDS="6" AF_EPOCH=latest \
  AF_NTRIALS=20 AF_FLOW_STEPS="1 2 5 10 20" FMPCC_RUN_MSG=afon005_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/alphaflow_pipeline.sh

# --- arm B: AF ON, 4x probe (alpha -> 0.2) ---
AF_BONE=unet AF_ALPHA_END=0.2 AF_SEEDS="6" AF_EPOCH=latest \
  AF_NTRIALS=20 AF_FLOW_STEPS="1 2 5 10 20" FMPCC_RUN_MSG=afon02_s6 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/alphaflow_pipeline.sh
```

**`AF_NTRIALS` is what makes the pipeline safe** (wired 2026-09-01,
`FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py`, mirroring the `AF_SEEDS` block).
`n_trials` lives in `config/alphaflow_projection_eval.yaml` and is read at **job runtime** — a `sed`
at submit time would be read by the dependent eval **4 h later**, whatever the file said by then,
and two arms could not queue together. With the env var both arms go in at once and the yaml is
never touched.

`AF_EPOCH=latest` is set on the pipeline (harmless during training, required for the eval): `best`
selects on `test_loss ≈ 0.75 + 0.25·α`, i.e. a mid-curriculum checkpoint (§3.1).

**Verify AF is actually on** before trusting any result — three log/W&B checks, all already emitted:

| signal | ❌ AF off (today) | ✅ AF on |
|---|---|---|
| `[ train ] AF_ALPHA_END=` banner | `0.0` | `0.05` / `0.2` |
| `val/alpha` at the end | `0.0` | ≈ `0.05` / `0.2` |
| **`train/discrete_frac`** | **`0.0`** — no sample took the bootstrap branch | **> 0** |
| savepath | `…_ae0.0_…` | `…_ae0.05_…` |

`discrete_frac = 0.0` was exactly what job 25251 logged, and it is the machine-readable proof that
the run deployed MeanFlow.

**Decide on §0's criterion:** seed 6, K = 1, `diffuser` — **> 0.85 goal reached and < 59.70 steps.**
Screen `val/raw_mse_u` against MF's 1.90 first. ~4.2 h per train, ~15 min per eval.

### Step 1b — deploy the model the schedule actually produces (**zero params, zero method change**)

`utils/training.py:81` sets `save_freq = n_train_steps // 5` = **20 000**, and the loop ends at
99 999, so `step % 20000 == 0` never fires at 100 k. **The last checkpoint is step 80 000 — the
step-100 000 model is never saved**, so we always deploy 20 % short of the schedule's end. With
`AF_ALPHA_END > 0` this matters much less (α is already at its floor by 80 k — see the table above),
which is another reason to do step 1 first. Still worth fixing: it also widens step 2's knob.

### Step 2 — `af_alpha_gamma` 25 → 20 (**config only, upstream key, zero params**)

γ controls how fast α crosses the useful middle of the homotopy. At γ = 25 the network spends only
**17 578 steps** with α ∈ [0.1, 0.9] — the only regime where the bootstrap probe `dt = α·h` is large
enough for a 4-frequency time code to see it. Lowering γ buys more of it, but it also pushes the
α = 0 snap later, and the snap must land **before the saved checkpoint** or we deploy a
mid-curriculum model:

| γ | α = 1 head ends | α = 0 tail starts | bootstrap steps | steps with α ∈ [0.1, 0.9] | MF steps before `latest` = 80 k |
|---|---|---|---|---|---|
| **25** (today) | 28 827 | 71 173 | 42 346 | 17 578 | 8 827 |
| **22** | 25 940 | 74 060 | 48 121 | 19 975 (+14 %) | 5 940 |
| **20** | 23 533 | 76 466 | 52 933 | **21 972 (+25 %)** | 3 534 |
| 18 | 20 593 | 79 407 | 58 814 | 24 413 | 593 ⚠️ |
| 15 | 14 711 | 85 289 | 70 577 | 29 296 | **−5 289** ❌ deploys mid-curriculum |

**γ = 20 is the most that is safe today.** If step 1b lands (final step saved), the last column
becomes `100 000 − tail start` and γ = 15 opens up as well — **so run step 1 first; it widens this
knob.** `_ag` is already a path token, so each γ gets its own tree (mandatory at seed 6).

Needs `AF_ALPHA_GAMMA` wired like `AF_ALPHA_CLAMP` (`config/avoiding-d3il.py:113-116`), or edit
`af_alpha_gamma` at `:942` and the eval block — **code change, needs a go-ahead**.

### Step 3 — `af_ratio_fm` 0.5 → 0.25 (**config only, upstream key, zero params**)

The one curriculum parameter upstream itself sweeps — `experiments-alphaflow.yaml` uses
**0.25 / 0.5 / 0.75 / 1.0**; we are pinned at 0.5. It sets what fraction of the batch is an `h = 0`
FM anchor, so 0.25 moves half of that weight onto the `h > 0` field — which is precisely where
AF-UNet is weak (`h_mse` b1 = 23.6, b2 = 41.1 vs b0 = 3.48, §3). Already has an `_rf` path token.
**Note this is not an AF-only knob** — MF's `meanflow_data_proportion` is the same quantity at the
same value, so if 0.25 helps, the MF control must be re-run at 0.25 before any claim.

### Step 4 — `af_clamp_utgt` 4.0 (**config only, upstream key, zero params**)

The only AF-specific knob left (§3.0 row 4). `train/clamp_frac` was **0.0** in 25251, so the clamp
never fired and loosening it changes nothing. **Parked — measured inert, not untested.**

### Step 5 — the one free diagnostic (no GPU)

W&B overlay of `raw_mse_u` and `h_mse_b0..b3` for 25251 vs 24389-s7 on the step axis, with the four
snap points marked (**28 827 / 38 222** head release, **61 778 / 71 173** tail snap). It is free and
it predicts E1's sign: if T1 fell behind during the extended α = 1 **head**, then bootstrap steps
build the field and E1 should help; if it only fell behind after the **tail** snap, the α = 0 phase
is the damage and E1 should help for a different reason. Either way it sharpens step 2's decision.

### Not in scope for the objective

These are diagnostics for the record, not moves toward beating MF. **Do not spend GPU on them.**

| item | why it is parked |
|---|---|
| `n_trials = 20` re-eval of the existing AF-UNet checkpoints (baseline / T1) | tells us how far behind AF *was*; the goal is where it *goes*. Seed 6 is unavailable for both trees anyway (§2.2.1) |
| the epoch A/B on the T1 tree (`ac05_best`) | subsumed by step 1, which runs both epochs on a model we actually care about |
| `AF_ALPHA_CLAMP=0.0005` and any other α-schedule tuning | §3.1 — the schedule is the paper's and is not the defect; the backbone is |
| upstream `discrete_training` port | it prevents the α = 0 endpoint, which §3.1 shows is upstream's intended endpoint. Wrong direction |
| `time_embed_freq_dim` (+0.06 M) and `E_t + E_r` | 🔒 forbidden — the U-Net and the parameter count are frozen |
| retraining AF-UNet seed 6 in the **baseline** tree | costs 4.2 h to measure a configuration we are abandoning |

### Blocking on a human

- **`/data` at 100 %** (27 G free of 7.0 T). Step 1 trains and writes plans. Free space first.
- `ls -la logs/avoiding-d3il/flow_matching_v3_alphaflow/H8_D…_ag25.0_rf0.5/7/` — only needed if the
  parked items are ever revived.
- **Path-hygiene fix** (§4.3): `_af_clamp_tok` is not applied to the plans savepath. `_ae` **is**
  applied to both, so step 1 is unaffected — but the bug is still there for clamp runs.

---

## 6. Confidence

| claim | confidence | basis |
|---|---|---|
| MF-UNet Pareto-dominates the DPCC baseline on the unprojected arm at K = 1 | **high** | 100 episodes each, matched backbone and params, both axes better (§2.1) |
| MF-UNet ≥ naive FM at every K | **high** | 100 episodes each (§2.1) |
| Raising the clamp hurts the trained field | **high** | matched seed/step/target, `raw_mse_u` 6.86 → 15.78 (§3) |
| T1 cut genuine-bootstrap steps by 44 %, not merely lengthened the tail | **high** | `af_diffusion.py:472-476` + the α banner: 25251 prints α(30 k) = 1.000 where raw σ(5) = 0.9933 |
| T1 reaches the goal at K = 1 where the baseline stalls | **observed, n = 2** | §2.3 — real but not a measurement |
| …caused by the **epoch** (`latest` > `best`) | **medium** | the only one of §4.2's four differences not ruled out or measured-contrary |
| …caused by the **clamp** | **low** | §3 measures the clamp making the field worse |
| The deployed AF checkpoint's last 18 222 steps are pure MeanFlow | **high** | §3.1 — `alpha <= 0.0` routes to the unmodified Gen3v6 MeanFlow body (`af_diffusion.py:552`) |
| Ending at α = 0 is upstream's design, not our porting artefact | **high** | `loss.py` routes `dt == 0` to the JVP MeanFlow target; `experiments-alphaflow.yaml:155,160,165` all set `end_value: 0` |
| α-Flow's claim is a curriculum claim (same objective, better path) | **high** | follows directly from the row above |
| We deploy AF 20 % short of its schedule (69 % of the α = 0 phase discarded) | **high** | `utils/training.py:81,203-205` — `save_freq = 20000`, loop ends at 99 999, so 100 k is never saved |
| AF's entire method-level surface vs MF is 5 items, 2 of them tunable without touching the net | **high** | §3.0, read off both loss bodies |
| The U-Net's ~4-frequency time code is what denies AF its advantage | **medium** | report §9.1's resolution argument + `unet1d_temporal_cond.py:114,121` tying frequency count to `dim = 32`; untested, and 🔒 now untestable directly — the backbone is frozen. Steps 2–3 attack it indirectly by enlarging the probe instead of the resolution |
| AF-UNet is better or worse than MF-UNet on goal-reached | **unknown** | ⛔ §2.4 — never run at `n_trials = 20`. The bar is exact: at seed 6, K = 1, > 0.85 goal reached and < 59.70 steps (§2.2) |
