*Gen14 · visual-aligning · batch `batch_va2_20260831_100336` (2026-08-31 10:03:36, 21 candidates).
Two new runs landed since the funnel report. They are unrelated experiments and are reported
separately below.*

**Conventions used in both parts.** Distance is `context_final_xy_dist` — raw box→target XY in
metres. "0-viol" is `collision_free_completed`, which is identical to
`constraint_exec_zero_violation` in all 7 220 rollout rows. `mean_dist_per_rollout` is **not used
anywhere** — it is `0.5·(pos_dist_3D + rot_err/π)`, not a distance. `avg_time_ms` is **per replan
step**, not per rollout. Tests are exact two-sided sign test and exact McNemar, pure stdlib.

**Pairing.** Fingerprinting each rollout by `(box_init_xy, target_xy)` shows the 10-context runs
are an exact **prefix subset** of the 30-context runs (`5 ⊆ 7`, `15 ⊆ 16`, `15 ∩ 17 = 10`). Every
comparison below is paired per context, never mean-vs-mean across different context sets. The
n=10 vs n=30 gap is not a confound.

---

# Part 1 — The α-Flow enable run (`AFconst0p05`): complete, and it did not work

**Verdict: the U10 repair did exactly what it promised to the loss and nothing good to the robot.**
Validation `raw_mse_u` fell **8.504 → 2.626** (3.2×). Across 320 exactly-paired rollouts distance
did not move (`p = 0.67`) and constraint satisfaction **fell 0.444 → 0.284**, significant at
**`p = 7.0e-6`**.

## 1.1 The setup — everything is identical except one scalar

This is a **one-knob experiment**. Both arms are the same architecture, the same training budget,
the same sampler, the same K, the same projection threshold, the same contexts, the same seed.

### Shared, identical in both arms

| | value | where verified |
|---|---|---|
| engine | `af` — `VisualAlphaFlow` / `AlphaFlowEngine` (Gen3v7) | both folder names |
| ML bone | `unet` — `VisualUNetTwoTime`, FiLM `v1`, `dim=32`, `dim_mults=(1,2,4,8)` | `filmv1` in both paths |
| horizon / dims | `H=8`, `action_dim=3`, `obs_dim=6`, `if_vision=True` | `H8`, `VTrue` in both paths |
| training budget | `n_train_steps=100 000`, `bs=64`, `grad_accum=2`, `lr=2e-4`, `ema=0.995` | `steps1000_bs64` in both paths |
| time schedule | `t_schedule=logit_normal`, `p_mean=-0.4`, `p_std=1.0` | `tslogit_normal` in both paths |
| α-Flow internals | `af_ratio_fm=0.5`, `af_adp_eps=1e-3`, `af_clamp_utgt=4.0`, `dual_head=True`, `interval_cfg=False` | job 25241 args dump |
| loss / weighting | `l2`, `action_weight=1`, `a1.5_b1.0` | `a1.5_b1.0_aw1` in both paths |
| seed | **6** | both |
| **K (sampler steps)** | **`flow_steps_v3 = 2`** for BOTH | `H8_K2_Meuler` in both eval folders |
| **T (projection)** | **0.5** for BOTH | `T0.5` in both eval folders; `run_config.csv` |
| MPC batch | `mpc_batch_size = 4` | `mpc4` in both eval folders |

**On K specifically**, because it is easy to lose: **K is an eval-time knob, not a training one.**
The checkpoint stores `flow_steps_v3 = 100`; the eval overrides it down:
`[ config->pkl ] INFO flow_steps_v3: train=100 -> eval=2`. Both arms did this identically, so both
were sampled at **K = 2** — 2 NFE per replan. Nothing in Part 1 varies K.

### The one thing that differs

| | cand 7 — shipped | cand 5 — **the new run** |
|---|---|---|
| `af_alpha_scheduler` | `sigmoid` | **`constant`** |
| `af_alpha_init` | `1.0` | **`0.05`** |
| `af_alpha_end` | `0.0` | **`0.05`** |
| `af_alpha_gamma` | `25.0` | `25.0` (inert under `constant`) |
| `af_alpha_clamp` | `0.005` | `0.005` (never reached) |
| resulting α(t) | 1.0 → 0.0, **snapping to exactly 0 at ≈71.2 %** | **flat 0.05 for all 100 000 steps** |
| path tag | `…_afschsigmoid` | `…_afschconstant_AFAFconst0p05` |

α is a single scalar that sets which target the loss regresses:
**α = 1 ⇒ pure Flow Matching · 0 < α < 1 ⇒ interpolated · α = 0 ⇒ MeanFlow, bit-identical.**
So the shipped arm trains FM for its first ~29 %, interpolates for ~42 %, then trains **MeanFlow**
for its last ~29 %. The new arm sits at 0.05 — near-MeanFlow but never *at* it — the whole run.

**Why "identical" is checkable rather than asserted:** the eval-side `run_config.csv` for the two
candidates differs in **only** the identity columns (`Candidate`, `FullPath`, `savepath`,
`diffusion_loadpath`) and the variant list. Every functional column — threshold, horizon,
`mpc_batch_size`, engine, `if_vision` — is equal. On the training side, every watched key is
encoded in the directory name, and the two names share the prefix
`H8_D…VisualAlphaFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Eaf_tslogit_normal` character for
character; only the α suffix differs. That is what U10's path key was built to guarantee.

### Two asymmetries that are NOT the knob

- **Arm C is off for cand 5.** `[ eval ] arm C (HardFlow) OFF — set HFFM_VARIANTS to enable`. So
  cand 5 has 16 variants and cand 7 has 19. No comparison below uses a HardFlow row.
- **10 contexts vs 30.** Handled by exact pairing (see the preamble); all 320 pairs are the same
  10 contexts.

## 1.2 Did it run? Yes — this is the resubmit of the pipeline that died on 08-29

The first attempt (pipeline 25190) never trained: its gate stage 25206 failed `G0` on the
unregistered `sampling/projection.py` graft, and `afterok` cancelled train 25207 and eval 25208.
After that gate was fixed, the identical command was resubmitted:

| job | stage | outcome |
|---|---|---|
| 25239 | pipeline submit | `alpha schedule: SCHED=constant INIT=0.05 END=0.05` · `🔴 PATH KEY -- '_AF<tag>' tree` |
| 25240 | gates | **17/17 PASS**, including `G0: PASS` — the gate that killed the first attempt |
| 25241 | train | 100 000 steps, 21:29 → ~02:04 (≈4.6 h), `Job completed successfully` |
| 25242 | eval | `Job completed successfully` |

All three at `GIT REV: 938641c`. **320 rollouts** = 16 variants × 2 geometries × 10 contexts,
nothing truncated.

## 1.3 What U10 predicted

`Gen14/U5` §3 measured what the α→0 snap costs:

| step | α | val `raw_mse_u` |
|---:|---:|---:|
| 70 000 | 0.006693 | **2.657** ← best the anneal ever reached |
| 71 000 | 0.005220 | 2.911 |
| **72 000** | **0.0** | **8.504** — and 6.3–8.6 for the rest of the run |

## 1.4 Training: the prediction was right

Job 25241, final W&B summary:

| | α → 0 (shipped) | **α = 0.05 const** |
|---|---:|---:|
| val `raw_mse_u` | 8.504 | **2.626** |
| train `raw_mse_u` | — | 1.657 |
| val `discrete_frac` | 0.000 | 0.502 |
| `clamp_frac` | — | 0.000 |

**3.2× better than the α→0 endpoint, and slightly better than the 2.657 the anneal touched before
the cliff.** `clamp_frac = 0.0` says the ±4 `u_target` clamp never fired — α = 0.05 is in the
well-behaved interpolating regime, not at a numerical edge. `discrete_frac ≈ 0.5` confirms the
two-time branch is genuinely active rather than collapsed to pure FM.

The α→0 snap was a real defect and U10 removed it.

## 1.5 Task: it did not transfer, and constraints got significantly worse

320 rollouts, paired context-by-context, 16 variants × 2 geometries × the same 10 contexts:

| pooled over all 320 pairs | α = 0.05 const | α → 0 anneal |
|---|---:|---:|
| distance (m) | 0.3642 | **0.3202** |
| 0-violation rate | **0.284** (91/320) | **0.444** (142/320) |
| `avg_time_ms` | 53.5 | 43.4 |

| test | result |
|---|---|
| distance, exact sign test | 128 / 136 · **`p = 0.67`** → no difference |
| 0-viol, exact McNemar | 38 vs **89** discordant · **`p = 7.0e-6`** → **const is worse** |

Per-cell on the headline variants (paired, n = 10 each; **0v** = 0-violation rate):

| geo | variant | dist const | dist anneal | Δ | 0v const | 0v anneal |
|---|---|---:|---:|---:|---:|---:|
| combined_5 | `diffuser` | 0.3507 | 0.2394 | +0.111 | 0.00 | 0.20 |
| combined_5 | `dpcc-r` | 0.4671 | 0.3657 | +0.101 | 0.30 | 0.50 |
| combined_5 | `dpcc-t` | 0.3035 | 0.3137 | −0.010 | 0.20 | 0.40 |
| combined_5 | `dpcc-c` | 0.4098 | 0.2891 | +0.121 | 0.10 | 0.20 |
| tightened | `diffuser` | 0.2755 | 0.2164 | +0.059 | 0.10 | 0.30 |
| tightened | `dpcc-r` | 0.3366 | 0.4012 | −0.065 | 0.70 | 0.80 |
| tightened | `dpcc-t` | 0.4037 | 0.3345 | +0.069 | 0.60 | **1.00** |
| tightened | `dpcc-c` | 0.4365 | 0.3392 | +0.097 | 0.40 | 0.70 |

The distance column is noise — signs go both ways, no cell significant. **The 0-viol column is
not: the anneal is better in 8 of 8.** That drives the pooled `p = 7.0e-6`.

## 1.6 Reading it

A 3.2× improvement in the velocity-field regression target produced **zero** distance improvement
and a **significant regression** in constraint satisfaction. `raw_mse_u` is not a proxy for rollout
quality on this task — the same conclusion
`Gen3v7/Study/STUDY_why_af_sit_works_unet_not_and_mf_unet_works.md` reached from the other
direction.

Stated plainly: **one seed, one α value.** This does not establish that α-Flow cannot work here.
What it *does* settle is that **the α→0 snap is not the reason α-Flow underperforms on
visual-aligning** — that hypothesis is dead, because removing the snap left the task numbers where
they were and made the constraint numbers worse.

## 1.7 Next, if anything

```bash
# upstream's anneal that never reaches zero — ~4.6 h of training
MIX_AF_ALPHA_END=0.02 MIX_AF_ALPHA_CLAMP=1e-4 \
  ./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/mix_visual_aligning/mix_visual_aligning_pipeline.sh af 6
```

Adding `HFFM_VARIANTS` to the α-const eval would also close the one asymmetry above.

---

# Part 2 — K = 100 and late-stage projection (`T = 0.1` / `0.05`)

**Three findings.**

1. **On quality, K = 100 is worth it — and it is the sampler, not the projector, doing the work.**
   Pooled over the two engines where it moves at all (MeanFlow, Diffusion), the unguided arm gets
   the box closer at high K in **37 of 58 decided contexts, `p = 0.048`**. Normalised progress on
   MeanFlow goes **0.024 → 0.679**. Flow Matching shows nothing.
2. **But K = 100 is not MeanFlow doing MeanFlow, and the reason it wins is not step count.** At
   K = 100 the sampler queries `u(x, t, h = 1/K = 0.01)`, and `u → v` as `h → 0` — it is an
   ordinary FM Euler ODE. It wins because **half of every training batch is pinned at `h = 0`**,
   so `h = 0.01` carries **54 %** of the training mass while `h = 0.5` (what K = 2 asks for)
   carries **3.5 %** and `h = 1.0` (K = 1) carries **none**. See §2.5 — this reframes §2.4.
3. **On cost it is brutal, and 94 % of it is the NLP, not the network.** `dpcc-r` at K=100 T=0.5
   costs 15 218 ms per replan; `T = 0.1` cuts it to 1 195 ms — 12.7× — at the same 0-violation
   rate. The job still died at 23.45 h of a 24 h wall with 184 of 760 rollouts.

## 2.1 What the knobs do

**K** (`flow_steps_v3`) is the number of ODE / NFE steps per replan. **T**
(`diffusion_timestep_threshold`) is the fraction of the ODE **tail** over which the projector runs —
solved when `tau_next >= 1 − T` — so projector calls per replan ≈ `T·K`:

| T | projector calls/replan at K=100 | τ range solved |
|---:|---:|---|
| 0.5 (default) | 50 | 0.50 → 1.00 |
| **0.1** | **10** | 0.90 → 1.00 |
| **0.05** | **5** | 0.95 → 1.00 |

Each projector call runs one SLSQP problem per MPC batch element (`mpc_batch_size = 4`, `dof = 66`),
so 50 calls/replan is **200 SLSQP solves per replan step**, and a rollout is ~400 replan steps.

HardFlow's `activation_threshold` is `null` in the YAML = **inherit** `diffusion_timestep_threshold`,
so arm C matched arm B. The log confirms the inheritance landed: at `T = 0.1` the first arm-C solve
reported is `non-converged SLSQP solve at tau=0.910` — exactly the first step past `1 − T = 0.90`.

## 2.2 Did it run? Partly — job 25216, `GIT REV: 81e9ea7`

The U11 override resolved through both places it has to reach, and `run_config.csv` confirms
`diffusion_timestep_threshold = 0.1` on all 18 of its config rows:

```
[ eval ] --proj-threshold: diffusion_timestep_threshold 0.5 -> 0.1  (source: cli --proj-threshold)
[ eval ]   applied to BOTH the projector config AND the results folder key T
[ eval ]   projection budget: 50 -> 10 projector call(s) per replan at K=100
```

Then it ran out of wall. The log ends mid-sentence inside `hardflow_sls-t`, context 1, with no
`JOB END` line.

## 2.3 Is the data full? No — 24.2 % of it

| | rollouts |
|---|---|
| planned (2 passes × 2 geos × 19 variants × 10 contexts) | 760 |
| delivered | **184 (24.2 %)** |
| 18 variants at full 10 | 180 |
| `hardflow_sls-t` | **4 / 10** |
| `combined_5-tightened`, all 19 variants | **0 / 190** |
| entire `T = 0.05` pass | **0 / 380** |

`T0.05` appears **zero times** in `discovery_manifest.json`, `run_config.csv` and
`va2_aggregated_long.csv`. It does not exist. All `T = 0.1` figures below are `combined_5`, n = 10,
seed 6.

**The K = 100 vs low-K comparison in §2.4 does not depend on that truncated run.** It uses the
already-complete `T = 0.5` corpus at n = 30 per arm, which is why it can carry a significance claim
the `T = 0.1` cells cannot.

---

## 2.4 🟢 Is K = 100 worth it on quality alone? **Yes — for MeanFlow and Diffusion**

Ignoring time entirely. Three engines, each with a high-K and a low-K run at **matched `T = 0.5`**,
paired per context, n = 30 for the unguided arm.

### 2.4.1 Distance — the unguided arm, which is the sampler with no projector in the way

| engine | K high | K low | dist high | dist low | Δ | closer | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MeanFlow** | 100 | 2 | **0.2200** | 0.4192 | **−0.199 m (−48 %)** | 19/10 | 0.136 |
| **Diffusion** | 100 | 20 | **0.2167** | 0.3901 | **−0.173 m (−44 %)** | 18/11 | 0.265 |
| Flow Matching | 100 | 20 | 0.3471 | 0.3373 | +0.010 (nothing) | 9/13 | 0.523 |

Neither engine clears `p < 0.05` alone at n = 30. **Pooled they do:**

| pooled sign test, unguided arm | closer at high K | p |
|---|---:|---:|
| MeanFlow + Diffusion | **37 / 21** | **0.048** |
| all three engines | 46 / 34 | 0.219 |

Two independent engines, same direction, ~45 % of the residual distance removed. Flow Matching
dilutes the three-engine pool to nothing because **for `fm`, K does not help at all** — a real
engine-dependent split, consistent with `DA_20260826_Gen14_K_sampler_steps_MF_AF_FM_diffusion.md`.

### 2.4.2 The clearest framing — normalised progress `1 − final/init`

Every context starts ~0.455 m from target, so progress is comparable across arms.
`untouched` = fraction of rollouts where the box moved < 5 mm.

| arm | K | T | arm type | **progress** | untouched |
|---|---:|---:|---|---:|---:|
| MeanFlow | 2 | 0.5 | unguided | **0.024** | 0.03 |
| MeanFlow | 2 | 0.5 | `dpcc-r` | 0.457 | 0.07 |
| MeanFlow | 100 | 0.5 | unguided | **0.509** | 0.10 |
| MeanFlow | 100 | 0.5 | `dpcc-r` | 0.236 | 0.27 |
| MeanFlow | 100 | **0.1** | unguided | **0.679** | **0.00** |
| MeanFlow | 100 | **0.1** | `dpcc-r` | **0.542** | 0.20 |
| Diffusion | 20 | 0.5 | unguided | 0.147 | 0.37 |
| Diffusion | 100 | 0.5 | unguided | **0.524** | 0.17 |
| Diffusion | 100 | 0.5 | `dpcc-r` | 0.506 | 0.32 |
| Flow Matching | 20 | 0.5 | unguided | 0.270 | 0.43 |
| Flow Matching | 100 | 0.5 | unguided | 0.243 | 0.40 |

### 2.4.3 🔴 What this actually says — where the competence lives

Read the MeanFlow rows as a group:

- **At K = 2 the generative model does almost nothing.** Unguided progress is **0.024** — the box
  effectively stays where it started. The `dpcc-r` row is **0.457**. *At low K, essentially all of
  MeanFlow's apparent competence is the MPC projector.*
- **At K = 100 the generative model does the work.** Unguided progress is **0.509–0.679**, and at
  `T = 0.1` **every single rollout moved the box** (untouched 0.00, the only arm in the corpus with
  that).
- **And then heavy projection undoes it.** At K = 100 with `T = 0.5`, `dpcc-r` progress **falls
  from 0.509 to 0.236** — the 50 projector calls drag the trajectory *away* from the goal. At
  `T = 0.1` the same arm keeps **0.542** of the sampler's 0.679.

So K = 100 is not a marginal accuracy tweak. **It is the difference between a policy whose
competence comes from the physical brakes and one whose competence comes from the generative
brain** — which is the thesis this project exists to test. And the two knobs interact: raising K
only pays if you *also* stop over-projecting it.

### 2.4.4 What K = 100 does NOT buy — state this whenever quoting the above

| axis | verdict |
|---|---|
| **constraint satisfaction** | **no gain, if anything worse.** Pooled 0-viol over the three unguided arms: 8 vs 13 discordant, **`p = 0.38`**. MeanFlow unguided 0.20 @K100 vs 0.33 @K2; at `T = 0.1` it is 0.10 vs 0.60 (`p = 0.062`, n = 10). |
| **task success** | **still ~0.** Best `n_success` anywhere is MeanFlow K100 `T=0.1` unguided at **0.20 (2/10)**; the only non-zero *success-and-constraints* cell in the whole corpus is MeanFlow K2 `dpcc-r` at **0.07 (2/30)**. The task is not solved at any K. |
| **Flow Matching** | no effect at all (9/13, `p = 0.52`). |
| **cost** | 24–33× slower per replan — see §2.5. |

**Honest summary: K = 100 gets the box roughly twice as far toward the target and does not help it
obey the constraints.** Under the Pareto rule this is a *trade-off*, not a clean win — but it is a
trade-off on a real, pooled-significant quality axis, which the funnel report (written before this
data existed) could only treat as an unaffordable outlier.

---

---

## 2.5 🔬 Why does K = 100 work? It is **not** integration accuracy

The obvious objection: MeanFlow exists to sample in 1–2 steps. Running it at 100 NFE looks like it
defeats the point, and if 100 Euler steps are what helps, plain Flow Matching should do just as
well. The code says otherwise, and it says why.

### 2.5.1 At K = 100 MeanFlow **is** running as a Flow-Matching ODE

`mix_visual_aligning/models/mf_diffusion.py::p_sample_loop` (`Meuler` = `legacy_euler`):

```python
dt      = 1.0 / max(flow_steps, 1)
h_batch = torch.full((batch_size,), dt, ...)
for i in range(total_steps):
    tau = loop_idx / max(flow_steps, 1)
    velocity = self._predict_velocity(x, cond, t_i, h=h_batch, returns=returns)
    x = x + velocity * dt
```

The interval width the network is asked about is **`h = dt = 1/K`**:

| K | h queried | what `u(·,t,h)` means there |
|---:|---:|---|
| 1 | 1.00 | average velocity over the **whole** path — true one-step MeanFlow |
| 2 | 0.50 | two half-path jumps |
| **100** | **0.01** | average velocity over a 1 %-wide window ≈ **the instantaneous velocity** |

And the MeanFlow identity, quoted from the loss docstring:

> `(t − r)·u(z_r, r, t) = z_t − z_r` … **At the r==t anchor (h=0) this reduces to
> `u_target = v_inst` — the FM velocity — which grounds the field.**

So as `h → 0`, `u → v`. **K = 100 is an ordinary FM Euler integration that happens to be reading
its velocity out of a MeanFlow network.** It throws away MeanFlow's entire selling point. Whatever
K = 100 is measuring, it is *not* MeanFlow doing few-step generation.

### 2.5.2 The real mechanism: K selects *how well-trained* the queried `h` is

```python
fm_mask = torch.rand(B, device=device) < self.meanflow_data_proportion   # = 0.5
r       = torch.where(fm_mask, t, r)      # FM anchors: h=0 ⇒ u_target = v_inst
h       = t - r
```

**Half of every batch is pinned at exactly `h = 0`.** The other half draws `h = |τ₁ − τ₂|` from two
independent `sigmoid(N(−p_mean, p_std))` draws, which concentrates near 0 as well. Sampling that
distribution (400 k draws, `p_mean=−0.4`, `p_std=1.0`, `meanflow_data_proportion=0.5`):

| h band | training mass |
|---|---:|
| **h = 0 exactly** | **0.499** ← the forced FM anchors |
| 0.00 ≤ h < 0.02 | **0.526** |
| 0.02 ≤ h < 0.10 | 0.106 |
| 0.10 ≤ h < 0.30 | 0.211 |
| 0.30 ≤ h < 0.45 | 0.095 |
| 0.45 ≤ h < 0.55 | 0.035 |
| 0.55 ≤ h ≤ 1.00 | 0.027 |

Now overlay what each K actually asks for:

| K | h queried | training mass within ±0.02 (±0.05 for K≤5) |
|---:|---:|---:|
| 1 | 1.000 | **0.0000** — never trained there at all |
| **2** | 0.500 | **0.035** |
| 5 | 0.200 | 0.106 |
| 10 | 0.100 | 0.051 |
| 20 | 0.050 | 0.053 |
| **100** | 0.010 | **0.540** |

**K = 100 queries the single most-trained value in the model; K = 2 queries one of the least — a
15× gap in coverage, and K = 1 is off the support entirely.**

That reframes §2.4. For `mf` and `af`, the K-sweep is **not** purely measuring "more integration
steps ⇒ less discretisation error" — it is also measuring **how much training the `h` your step
size implies actually received.** It explains MeanFlow's numbers directly: unguided progress
**0.024 at K = 2** (a barely-trained query — the sampler contributes nothing and DPCC carries the
arm) and **0.509–0.679 at K = 100** (the best-trained query).

**⚠️ But h-coverage alone is falsified as a complete explanation — by α-Flow.** `af` uses the
*identical* sampler (`af_diffusion.py:267-268`, `h = dt = 1/N`) and the *identical* forced-anchor
fraction (`af_ratio_fm = 0.5`), so its h-coverage curve is the same. Its K-response is the
**opposite**:

| engine | unguided K100 | unguided K2 | closer | p |
|---|---:|---:|---:|---:|
| MeanFlow | **0.2200** | 0.4192 | 19/10 → K100 | 0.136 |
| α-Flow (α→0) | 0.3598 | **0.2589** | **7/22 → K2** | **0.008** |

So there are at least **two competing effects** as K rises, and which one wins is empirical:

- **(a) h-coverage improves** — the query walks toward the 50 %-mass atom at `h = 0`. Helps.
- **(b) Euler error accumulates** — 100 sequential extrapolations of a learned field compound any
  systematic bias. Hurts, and hurts more the worse the field is.

MeanFlow's field is good enough that (a) wins; the α→0 α-Flow field is damaged (§1.3) and (b) wins,
solidly. This also explains why `fm` shows nothing: **`fm` has no `h` argument at all**
(`fm_diffusion.py::p_losses(x_start, cond, t)` is single-time), so (a) does not exist for it and
only (b) and ordinary discretisation gain remain — and they cancel.

**Three engines, three different meanings of K:**

| engine | what K is | K20→K100 |
|---|---|---|
| `fm` | pure ODE discretisation, no `h` | **saturated** — 0.3373 → 0.3471, 9/13, `p = 0.52` |
| `diffusion` | denoising steps, no `h` | **not saturated** — 0.3901 → 0.2167 |
| `mf` / `af` | discretisation **and** the queried `h` | model-dependent, both directions observed |

### 2.5.3 Then why not just use Flow Matching? Because MeanFlow's v is better

The fair test: both engines at **100 NFE**, same Euler sampler, same T, paired per context.

| variant | MeanFlow K100 | FlowMatching K100 | Δ | closer | p |
|---|---:|---:|---:|---:|---:|
| `diffuser` (unguided) | **0.2200** | 0.3471 | **−0.127** | 18/11 | 0.265 |
| `dpcc-r` | **0.3501** | 0.4507 | **−0.101** | **7/1** | **0.070** |

Identical compute, identical sampler — **the field itself is better.** Three code-level reasons,
all verifiable:

1. **MeanFlow's h = 0 branch *is* the FM loss.** `mf_diffusion.py:` `v_inst = x_start - x_base`.
   `fm_diffusion.py:290:` `v_target = x_start - x_base`. The same expression. So on half its batch
   MeanFlow trains on exactly Flow Matching's objective.
2. **The other half regularises it.** Those samples regress `u` over finite intervals against
   `u_target = v_inst + h·(JVP)`, forcing the *same network* to stay consistent with its own
   integrals across scales — supervision plain FM never receives.
3. **Adaptive loss weighting.** MeanFlow applies `err / sg((err + eps)^p)` (`mf_adp_eps = 0.01`,
   `mf_adp_p = 1.0`); the `fm` arm's `p_losses` has no such term.

The striking part: MeanFlow saw **half as many pure-FM gradient samples** as the `fm` arm and still
produced the better velocity field.

### 2.5.4 🟡 Is K = 100 actually necessary, or does K = 10–20 already get there?

**Honest answer: unknown. There is no MeanFlow run at any K between 2 and 100 anywhere in the
corpus** — a grep over the whole batch returns `H8_K2` and `H8_K100` and nothing else. The gap
between §2.4's two data points is 50×, and every intermediate claim would be interpolation.

What the corpus *does* say about intermediate K, on the engines that have it:

| engine | K20 | K100 | verdict |
|---|---:|---:|---|
| Flow Matching | 0.3373 | 0.3471 | **K20 already saturated** (9/13, `p = 0.52`) |
| Diffusion | 0.3901 | **0.2167** | **not saturated at K20** — K100 still buys 44 % |

So the two engines that *do* have intermediate K disagree with each other, which is exactly why
MeanFlow cannot be guessed.

**What the h-model predicts.** Since the `h = 0` atom holds 50 % of the training mass and the
network is continuous in `h`, the natural coverage measure is simply the distance from that atom,
`|h − 0| = 1/K`:

| K | h = 1/K | distance from the h=0 atom | cumulative training mass ≤ h |
|---:|---:|---:|---:|
| 2 | 0.500 | 0.500 | 0.958 |
| 5 | 0.200 | 0.200 | 0.748 |
| 10 | 0.100 | 0.100 | 0.632 |
| **20** | **0.050** | **0.050** | 0.567 |
| 50 | 0.020 | 0.020 | 0.527 |
| **100** | **0.010** | **0.010** | 0.514 |

**By K = 20 you have already closed 91 % of the distance from K = 2 to K = 100** (0.50 → 0.05 of a
0.50 → 0.01 span). On this model **the intuition is right: K ≈ 10–20 should capture most of the
benefit and K = 100 should be deep in diminishing returns.** The median of the continuous half of
the training h-distribution is 0.201 — i.e. **K ≈ 5** sits at the median interval width the model
was actually trained on.

**Why this matters more than it looks.** §2.6 shows the sampler costs `10.4 + 8.98·K` ms per
replan, and at K = 100 that ~900 ms is the floor no projection knob can touch:

| K | sampler ms/replan | vs K=100 |
|---:|---:|---:|
| 5 | 55 | 16× cheaper |
| 10 | 100 | 9× cheaper |
| 20 | **190** | **4.8× cheaper** |
| 100 | 908 | — |

If K = 20 delivers K = 100's quality, the whole §2.4 result stops being unaffordable: 190 ms of
sampler plus a `T = 0.1` projector budget lands near 250 ms/replan — the same order as the K = 2
arms, with the generative model actually doing the work.

**The experiment is nearly free and should be run before anything else in this document.** The
unguided arm alone answers it — no projector, no NLP, so cost is just `steps × rollouts × sampler`:

| K | est. wall for 10 unguided rollouts |
|---:|---:|
| 5 | ~4 min |
| 10 | ~7 min |
| 20 | ~13 min |
| 50 | ~31 min |
| **all four** | **≈ 55 min** |

against the ~1 h that K = 100's unguided arm already consumed. K = 2 and K = 100 exist, so four
intermediate points complete the curve for under an hour of GPU.

**And it is a real test, not a formality**, because §2.5.2's α-Flow counter-example means the
h-coverage prediction can fail. Two outcomes, both informative:

- **Curve saturates by K ≈ 10–20** ⇒ h-coverage explains it, K = 100 is waste, and the affordable
  configuration exists.
- **Curve keeps climbing to K = 100** ⇒ effect (a) is not the mechanism, and the gain is ordinary
  integration accuracy after all — in which case MeanFlow has no advantage over a well-trained FM
  and §2.5.3's claim weakens.

A second cheap test, same class: **evaluate the α-const checkpoint (cand 5) at K = 100.** It has the
best velocity field in the corpus by training loss (`raw_mse_u = 2.626`, §1.4) but was only ever
sampled at K = 2. Under the (a)/(b) split it should benefit *most* from K = 100. One eval job, no
retraining.

### 2.5.5 What this does and does not license

**Do not report this as "MeanFlow at 100 NFE".** It is not few-step generation and claiming it as a
MeanFlow result invites the obvious rebuttal. The defensible claim is:

> **The MeanFlow objective is a better *trainer* for a Flow-Matching velocity field than the naive
> FM objective, on this task. Harvest it with a standard many-step ODE.**

**Not yet isolated.** `mf` and `fm` are two training runs differing in three ways at once (the h=0
half-batch, the JVP consistency term, the adaptive weight). This data cannot say which one earns
the 0.127 m. The clean ablation is one knob:

- `meanflow_data_proportion = 1.0` ⇒ every sample is `h = 0` ⇒ the mf codepath trains **pure FM**,
  same network, same optimiser, same schedule. Compare against 0.5 at K = 100.

It is currently hard-coded at `config/aligning-d3il-visual.py:1564` with no env override, so this
needs a small knob in the U10/U11 style before it can be swept without a path collision.

**And it predicts something checkable:** if the mechanism is h-coverage rather than step count,
then a MeanFlow trained with `meanflow_data_proportion` lowered (say 0.1) should *improve* at
K = 1–2 and *degrade* at K = 100. No run in the corpus tests this.

## 2.6 🔴 Where the time actually goes

**The unguided `diffuser` arm is a direct measurement of the sampler**: identical network and ODE,
zero projector calls. Subtracting it from a projected arm at the same K isolates the NLP. All
figures paired on the same 10 contexts.

| arm | K | T | projector calls | **ms/replan** |
|---|---:|---:|---:|---:|
| `diffuser` (unguided) | 2 | — | 0 | **28.4** |
| `diffuser` (unguided) | 100 | — | 0 | **924.6** / 892.4 |
| `dpcc-r` | 2 | 0.5 | 1 | 49.8 |
| `dpcc-r` | 100 | **0.1** | 10 | **1 194.8** |
| `dpcc-r` | 100 | 0.5 | 50 | **15 217.8** |
| `hardflow_sls-r` | 100 | 0.1 | 10 | 1 308.9 |
| `hardflow_new-r` | 2 | 0.5 | 1 | 191.3 |

### Stage 1 — the NN / ODE

From the two unguided points, `t = 10.4 ms fixed + 8.98 ms per NFE`:

| K | sampler ms/replan | of which network forwards |
|---:|---:|---|
| 2 | 28.4 | 18 ms (2 NFE) |
| 100 | ~908 | **898 ms (100 NFE) — 99 % of it** |

**K=2 → K=100 costs ~880 ms per replan step, and it is essentially pure NFE count.** The fit is
linear with ~10 ms fixed overhead — no hidden per-call cost, just 50× the network evaluations.

### Stage 2 — the NLP solve, and this is where K = 100 actually died

| arm | total | − sampler | = NLP | calls | **ms per projector call** |
|---|---:|---:|---:|---:|---:|
| `dpcc-r` K=2 T=0.5 | 49.8 | 28.4 | 21.4 | 1 | **21.4** |
| `dpcc-r` K=100 **T=0.1** | 1 194.8 | 924.6 | 270.2 | 10 | **27.0** |
| `dpcc-r` K=100 **T=0.5** | 15 217.8 | 892.4 | **14 325.4** | 50 | **286.5** |
| `hardflow_sls-r` K=100 T=0.1 | 1 308.9 | 924.6 | 384.3 | 10 | 38.4 |
| `hardflow_new-r` K=2 T=0.5 | 191.3 | 28.4 | 162.9 | 1 | 162.9 |

**A late solve costs ~21–27 ms and is stable across K.** A solve at `T = 0.5` averages **286 ms** —
10.6× more. The marginal cost isolates which ones are expensive:

| slice | cost |
|---|---|
| the 40 extra calls at T=0.5 (τ 0.50 → 0.90) | 15 217.8 − 1 194.8 = **14 023 ms over 40 = 350.6 ms/call** |
| the 10 late calls at T=0.1 (τ 0.90 → 1.00) | 270.2 ms over 10 = **27.0 ms/call** |
| **ratio** | **13.0×** |

### So: what killed K = 100?

**Not the network.** The ODE at K=100 is ~900 ms/replan whether you project or not. At `T = 0.5`
it is **6 %** of the 15 218 ms bill.

**The NLP, and specifically the early-τ solves.** 94 % of a `dpcc-r` K=100 T=0.5 replan step is
SLSQP, and the 40 calls at τ < 0.9 account for 14 023 of those 14 325 ms — **92 % of the whole
replan step goes to solves that `T = 0.1` simply does not make.** Early in the ODE the trajectory
is still noisy and far from the feasible set, so SLSQP iterates far longer; near τ = 1 it is nearly
feasible and converges in a fraction of the time.

That is both why the T knob is a 12.7× win rather than 5×, **and** — with §2.4.3 — why those early
solves were never worth making: they are the expensive ones *and* the ones that drag progress from
0.509 down to 0.236.

**Caveat:** the circuit breaker is not distorting these numbers. At `T = 0.1` it tripped in 1 of 10
rollouts for `dpcc-c`/`dpcc-t` only (2.4 skipped steps) and never for `dpcc-r`, `diffuser` or arm C.
The `T = 0.5` run predates the breaker telemetry columns (`-1`), so its 15 218 ms is un-throttled or
an under-count, never an over-count.

## 2.7 The T = 0.1 result

| variant | metric | T = 0.1 | T = 0.5 | change |
|---|---|---:|---:|---|
| `dpcc-r` | dist (m) | **0.2048** | 0.3331 | −0.128 (`p = 0.73`, n = 10) |
| `dpcc-r` | `avg_time_ms` | **1 195** | 15 218 | **12.7× faster** |
| `dpcc-r` | 0-viol | 0.20 | 0.20 | unchanged |

The distance gain is not significant at n = 10. The time gain is not a statistical claim at all —
it is the arithmetic of §2.6.

**The control that validates the sweep:** `diffuser` never calls the projector, so it **must** be
invariant to `T` — and it is: 0.1440 m at T=0.1 vs 0.1361 m at T=0.5 on the same 10 contexts, and
924.6 vs 892.4 ms. The override moved the projector and nothing else.

Best cell in the whole `T = 0.1` run is arm C:

| variant | dist (m) | 0-viol | ms |
|---|---:|---:|---:|
| `hardflow_sls-r` | **0.1218** | **0.60** | 1 309 |
| `dpcc-r` | 0.2048 | 0.20 | 1 195 |
| `diffuser` (unguided) | 0.1440 | 0.10 | 925 |

The only cell that improves distance *and* triples 0-viol over the unguided arm. **n = 10, single
seed, one geometry — a lead, not a result.**

## 2.8 Against MeanFlow K2

| variant | dist K100/T0.1 | dist K2/T0.5 | ms K100 | ms K2 |
|---|---:|---:|---:|---:|
| `diffuser` | **0.1440** | 0.3676 | 925 | **28** |
| `dpcc-r` | **0.2048** | 0.3028 | 1 195 | **50** |
| `dpcc-t` | 0.2713 | 0.2721 | 1 462 | **50** |

K100 is closer on both (7/3, `p = 0.34` at n = 10) and **24–33× slower**, and §2.6 says that gap is
now dominated by the 898 ms of ODE, which no projection knob can touch — **only cutting K can move
it further.**

So the funnel's *ranking* stands on a time budget, but §2.4 changes what the ranking means: K2's
place there was earned by its projector, not its generative model. If the deliverable is "Flow
Matching as the generative engine", K = 100 is the configuration that demonstrates it and K = 2 is
the configuration that hides it behind DPCC.

## 2.9 What is missing, and what it would cost

| gap | size | why |
|---|---|---|
| `hardflow_sls-t` @ T=0.1 | 6 of 10 rollouts | wall clock |
| `combined_5-tightened` @ T=0.1 | 190 rollouts | wall clock |
| entire `T = 0.05` pass | 380 rollouts | never started |
| K=100 `T=0.1` at n = 30 | — | would move §2.4.1's `p = 0.048` claim onto the *cheap* K100 config |
| **`mf` K-curve at K = 5/10/20/50, unguided** | **≈ 55 min total** | **§2.5.4 — the highest-value run in this document; decides whether §2.4 is affordable** |
| `af` α-const (cand 5) evaluated at K = 100 | 1 eval job | §2.5.4 — best field in the corpus, never sampled above K=2 |
| `meanflow_data_proportion` ablation | 1 training run | the §2.5.3 attribution test — currently hard-coded, needs a knob |
| seeds | seed 6 only | — |

**The arithmetic says it never could have fit.** Summing the per-variant wall times the log reports
itself: 18 completed variants took **83 880 s = 23.30 h**, mean **1.29 h each**, plus 530 s into the
19th — **23.45 h against a 24 h limit**. At the delivered rate the full sweep needs **≈ 97 h, about
4.1× the wall.**

The fix is to stop evaluating 19 projection variants when five carry every claim. One job per
`(T, geometry)` restricted to `{diffuser, dpcc-r, dpcc-t, hardflow_sls-r, post_processing}` is
**≈ 6.5 h**, and four such jobs complete the sweep:

```bash
for T in 0.1 0.05; do
  MIX_PROJ_T="$T" HFFM_VARIANTS="hardflow_new-r" \
    ./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6 all 100
done
```

⚠ As submitted this still evaluates the full variant list and **will truncate again** unless the
variant set is narrowed first. That narrowing is a code change to the eval's variant list and is
**not** made here.

Priority given §2.4: the highest-value missing run is **K=100 `T=0.1` at n = 30**, not `T = 0.05`.
§2.6 shows `T = 0.05` can save at most ~135 ms of a 1 195 ms replan (~11 %), so it is a quality
question — do five solves still hold the constraints? — while n = 30 at `T = 0.1` is what turns the
best cell in this document into a defensible claim.

# Provenance

| artefact | value |
|---|---|
| batch | `batch_va2_20260831_100336` — 21 candidates, 2026-08-31 10:03:36 |
| Part 1 jobs | 25239 pipeline · 25240 gates (17/17) · 25241 train · 25242 eval — all `938641c` |
| Part 1 failed predecessor | 25190 → 25206 (`G0` FAIL, cancelled 25207/25208) |
| Part 2 job | 25216 — `81e9ea7`, truncated at 23.45 h / 24 h |
| Part 2 failed predecessors | 25191, 25215 — both `ValueError: MIX_PROJ_T='' is not a float` |
| distance metric | `context_final_xy_dist` — raw box→target XY, metres |
| 0-viol metric | `collision_free_completed` (≡ `constraint_exec_zero_violation`, 7 220/7 220 rows) |
| timing metric | `avg_time_ms` — **per replan step**; rollouts are ~400 steps |
| pairing key | `(box_init_xy, target_xy)` rounded to 1e-4 |
| tests | exact two-sided sign test; exact McNemar — pure stdlib, no SciPy |

⚠ **Context-count provenance gap.** The eval reads `n_contexts` from
`config/visual_aligning_eval.yaml`, whose **committed** value has been `3` since 2026-05-26 (it was
`30` before that). These runs delivered **10** and **30** rollouts per variant, so the YAML on the
cluster is edited outside git and the context count of any given run is **not reproducible from the
repo**. It is recoverable only from the run's own rollout count. Worth pinning before the next
sweep.
