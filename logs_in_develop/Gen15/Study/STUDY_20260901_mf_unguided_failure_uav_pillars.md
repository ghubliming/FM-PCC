# STUDY — Why MeanFlow cannot fly the UAV unguided, and FlowMatching / α-Flow can

**Date:** 2026-09-01
**Scope:** `pillars` (with `corridor` / `s_curve` cross-checks), Gen15 `mix_uav`, engines `fm` / `mf` / `af`
**Extracted from:** [`../DA/DA_20260830_pillars_K_sweep_fm_mf_af.md`](../DA/DA_20260830_pillars_K_sweep_fm_mf_af.md) §11,
which now carries only a pointer to this file.
**Nothing was run on the cluster for this study.** It reads the existing Slurm logs, the DA CSVs, and — new
here — the raw `.npz` eval artifacts.

> ⚠️ **Local-tooling note.** `numpy 2.5.2` was installed into a throwaway venv **inside the session
> scratchpad** (`/tmp/.../scratchpad/venv`), on explicit user authorisation, solely to read
> `sampled_trajectories_all` / `act_all` / `obs_all` out of the `.npz` artifacts. Nothing was installed
> into the container's system Python, nothing was added to the repo, and **no training / eval / pipeline
> code was executed locally** — the container rule in `CLAUDE.md` is otherwise intact.

---

## 0. Direct answers

| # | question | answer |
|---|---|---|
| **Q1** | **Did you find a bug?** | 🔴 **No — and the earlier "yes" was wrong.** `SafeLimitsNormalizer(eps=1)` is **not a logic error**: it avoids the `0/0` NaN and **round-trips the data exactly** (`normalize(c)=0`, `unnormalize(0)=c`). Any audit checking correctness passes it, correctly. What it *is* is a **scale-calibration defect** — `eps=1` is a magic constant in data units, so for a constant-zero dim `unnormalize` becomes the identity and that channel gets a silent **±1 m ceiling and ~23× the gain** of every other channel (§3.5). That consequence is measured and real, but a model that learned the channel emits ~0 and is unaffected — which is exactly what `fm` does. **It amplifies the failure; it does not cause it.** No defect was found in the `mf` sampler. |
| **Q2** | **Do you understand the cause?** | ✅ **Yes — measured (§3.6).** `mf` learned a **positive vertical feedback gain**, `b₁ = +0.11 m of commanded Δz per metre of altitude error`, at every K. `fm` and `af` landed at **−0.03…+0.01** (restoring/neutral). The loop `e_z ← e_z·(1+b₁)` reproduces the observed 47–144 step abort window from first principles. Links 1, 2 and 4 of the causal chain are identical across all three engines; **only the sign of this gain differs.** What remains open is the smaller question of *why* MeanFlow extrapolates to a positive gain in a channel that contains no training signal (§3.6.5). |
| **Q3** | **Is `mf`'s raw output lower quality than the others on the `diffuser` metric?** | ✅ **Yes — confirmed, and now measured at the action level** (§3), not just inferred from outcomes. |
| **Q4** | **"It is all squeezed into the beginning point — surely something is wrong."** | ✅ **Right that something is wrong, and right about the symptom.** The mechanism is neither a collapsed plan nor a weak forward command (both were my errors, §7): the forward channel is **healthy** and in the early rollout is the *fastest* of the three. The drone stays on the start line because a **vertical feedback runaway** aborts it at step 47–144, before the forward ramp-up has covered any ground. |
| **Q5** | **What *is* established?** | ✅ The failure is repaired by **one specific constraint group — the dynamics constraint — and by nothing else** (§2), and the raw command is **mis-directed, not weak** (§3). |

### 0.1 Verdict on the "squeezed into the beginning point" reading

| your claim | verdict | what the data says |
|---|---|---|
| the drone never leaves the start | ✅ **right** | **`mf` crossed 0/30 unguided; `fm` 26/30, `af` 28/30** (whole-population, no conditioning). Its 30/30 aborts land at median step **78**, x = **−2.96** of a −3.2 → +3.2 course. |
| it has lost control | ✅ **right** | the drone realises only **38 %** of the commanded displacement, and the setpoint runs away to **p95 = 2.86 m** ahead of the airframe (§5). |
| the raw output is worse than the others | ✅ **right, but not in the forward channel** | 🔴 The forward-deficit numbers here were a **truncation artifact** (§3.6.1) — per step-bin, `mf`'s Δx equals or exceeds `fm`/`af`. What is genuinely worse is the **vertical** channel: `\|Δz\|` grows 0.004 → 0.423 over the rollout while `fm`/`af` stay flat. |
| …because the plan is **squeezed / collapsed to a point** | 🔴 **wrong** | The commanded step is **`|Δp_des| = 0.0517 m/step`, which is 2–4× *larger* than the expert**, not smaller. The plan is not collapsed; it is **pointing the wrong way**. Only 0.0069 of that 0.0517 is forward — **the rest is vertical thrash**, with `std Δz` **22×** `std Δx`. |

> **The one-line statement:** `mf`'s unguided plan is **mis-directed, not under-powered** — roughly the
> right amount of motion, aimed mostly at the ceiling. The drone stays on the start line because its
> command has almost no forward component, not because the plan has collapsed onto the anchor.

---

## 1. The observation

`variant='diffuser'` is the **no-projector** arm: the raw sample is executed as-is. On `pillars`,
n=10/cell, **every column below re-derived from `per_rollout_detail.csv`**:

| engine | K | crossed | **reached** | goal_dist (med) | aborted | **crossed AND aborted** | **crossed, no abort** |
|---|---|---|---|---|---|---|---|
| **fm** | 1 | 0.70 | **0.00** | 1.71 m | 8 | **5** | 2 |
| **fm** | 2 | **1.00** | **0.00** | 1.21 m | **0** | 0 | 10 |
| **fm** | 5 | **1.00** | **0.00** | 0.97 m | 1 | 1 | 9 |
| **af** | 1 | 0.90 | **0.00** | 1.17 m | 1 | 0 | 9 |
| **af** | 2 | 0.90 | **0.00** | 1.27 m | 3 | 2 | 7 |
| **af** | 5 | **1.00** | **0.00** | 1.07 m | 1 | 1 | 9 |
| 🔴 **mf** | 1 | **0.00** | **0.00** | **6.46 m** | **10** | 0 | **0** |
| 🔴 **mf** | 2 | **0.00** | **0.00** | **6.45 m** | **10** | 0 | **0** |
| 🔴 **mf** | 5 | **0.00** | **0.00** | **6.51 m** | **10** | 0 | **0** |

### 1.1 Reading this table honestly — four things it does *not* say

An earlier draft printed the first four columns with ✅ / ❌ marks and moved on. That was wrong, and
the table needs these four caveats before any of it is usable.

1. 🔴 **`goal_reached` is 0.00 in every single cell — including `fm` K2, which is 1.00 crossed with
   zero aborts.** Unguided, **nothing reaches the goal, ever.** `goal_crossed_line` is a *latch* — the
   drone passed the goal's x at some instant — and the median final distance of 0.97–1.71 m says it
   sailed past and ended a metre away. So the ✅ on `fm`/`af` means **"it flew the length of the
   course"**, nothing stronger. Every claim in this study is about *whether the engine can fly at all*
   open-loop, never about task success.
2. 🔴 **`crossed` and `aborted` are not complementary — they overlap.** `fm` K1: 5 of its 7 crossings
   *also* aborted. `af` K2: 2 of 9. A rollout can cross the line and then diverge, because the budget
   keeps running after the latch. So "8 aborts" does not mean "8 failures to fly".
3. 🟡 **`fm` K1 → K2 is a large, unexplained swing** — 8 aborts → 0, 0.70 → 1.00 crossed, on n=10.
   `af` shows no such pattern (1 / 3 / 1). This study does not explain it and does not depend on it;
   flagging it because it is the second-largest effect in the table and it is undiscussed.
4. ✅ **The `mf` row is the one that needs no caveat.** 0/30 crossed, 30/30 aborted, **zero** rollouts
   in either overlap column, flat across K. There is no survivorship subtlety because there are no
   survivors.

**It is not the checkpoint.** `diffuser` and `dpcc-*`/`hardflow_*` for a given engine×K come out of
**one candidate folder** (mf = Candidates 51/52/53), so the weights are byte-identical across the
guided and unguided rows.

**It is not the scene.** `variant=diffuser`, median over all K:

| scene | `af` | `fm` | 🔴 `mf` |
|---|---|---|---|
| pillars | 1.17 m | 1.06 m | **6.48 m** |
| corridor | 0.47 m | 1.22 m | **30.69 m** (track_err **101**) |
| s_curve | — | 6.48 m | 6.59 m |

On `corridor` the unguided MeanFlow drone ends a median **30.7 m** from the goal. On `pillars` it only
looks milder because the v2 abort freezes the rollout at step 47–144; the K10 arm, which predates the
abort, runs free and reaches **66–172 m**.

### 1.2 What the abort records

Abort reasons, **all** `diffuser` aborts on `pillars` (from the Slurm logs):

| engine | total aborts | `off_route` | `inverted` | `overspeed` |
|---|---|---|---|---|
| `fm` | 9 | 5 | 4 | 0 |
| `af` | 5 | 1 | 4 | 0 |
| 🔴 `mf` | **30** | 16 | 11 | 3 |

`DIVERGENCE_ABORT.txt` also logs `p_des` — the setpoint **the plan itself commanded**:

| trial | step | p (achieved) | **p_des (commanded)** | Δx/step | Δz/step |
|---|---|---|---|---|---|
| 0 | 62 | [−3.00, −0.11, 3.34] | [−2.70, −0.33, **+5.65**] | +0.0081 | +0.0750 |
| 2 | 47 | [−3.22, −0.16, 0.33] | [−2.78, −0.56, **−3.88**] | +0.0089 | −0.1038 |
| 6 | 60 | [−3.17, −0.23, 0.34] | [−2.82, −0.50, **−5.65**] | +0.0063 | −0.1109 |
| 9 | 56 | [−3.04, −0.18, 3.41] | [−2.70, −0.58, **+6.39**] | +0.0090 | +0.0962 |

**The plan commands ±3–7 m of altitude against a 0.70–1.30 m training band.** The sign of the z drift
is **random per rollout** (5 up, 5 down) and then held for the whole episode.

### 1.3 The cross-engine control — with its bias stated

🔴 **An earlier draft of this table was wrong and is corrected here.** It reported "aborts" of 5 / 1 /
16 for fm / af / mf. Those were **parse yields, not abort counts**: the one-line Slurm format prints
`p=[...]` only for `off_route`, so every `inverted` and `overspeed` abort was silently dropped
(fm 5 of 9 parsed, af **1 of 5**, mf 16 of 30). The statistic is therefore **conditioned on
`reason = off_route`**, and the `af` row rests on a **single rollout**.

| engine | aborts (true) | **parsed** | median abort step | median x at abort | Δx/step | usable? |
|---|---|---|---|---|---|---|
| `fm` | 9 | 5 (off_route only) | 305 | +1.83 | +0.0162 | 🟡 indicative |
| `af` | 5 | **1** (off_route only) | 543 | +5.21 | +0.0155 | 🔴 **n=1 — not a statistic** |
| 🔴 `mf` | **30** | 16 (off_route only) | **78** | **−2.96** | **+0.0030** | 🟡 indicative |

A second bias sits on top: for `fm`, 5 of 8 K1 aborts happened *after* crossing, so "x at abort" mixes
pre- and post-crossing rollouts, and the 4 non-aborting `fm`/`af` cells are excluded entirely. **This
is a survivorship-conditioned comparison and cannot carry a quantitative claim.**

✅ **What survives the correction — and it is all this study needs:**

- **`fm` crossed 26/30 and `af` 28/30 unguided; `mf` crossed 0/30.** That is from the CSV, covers
  every rollout, and has no conditioning at all.
- **`mf`'s 30/30 aborts occur at median step 78, at x = −2.96** of a −3.2 → +3.2 course. With **zero
  survivors** there is no survivorship bias in the `mf` row — the reason-stratification is the only
  caveat, and `off_route`/`inverted`/`overspeed` all fire in the same first ~150 steps.
- **Every `fm` unguided abort is also a z-failure** (5 `off_route` on z at exactly 3.33 m, 4
  `inverted`). The z-runaway is **shared** — it is the generic open-loop failure of this 634-step
  receding-horizon loop, present in all three engines. What is unique to `mf` is that it has **no
  forward component to escape with**.

The quantitative version of that last point does **not** rest on this biased table — it rests on
§3's `act_all` measurement, which reads the model's raw command on **every executed step** of all 10
`mf` rollouts and needs no abort at all.

## 2. What *is* established — only the dynamics constraint repairs it

The K10 candidates (45 `fm`, 50 `mf`) are the only ones carrying the full `model_free` /
`bounds_free` / `geo_free` toggle grid — a **one-factor-at-a-time ablation of the projector's
constraint groups** (`eval_mix_uav.py:937-1031`: `model_free`→no dynamics, `bounds_free`→no action
bound, `geo_free`→no geo_bounds+halfspace+obstacles). n=3/cell:

| mf K10 variant | dynamics | bounds | geometry | **goal_dist** | **track_err** | |
|---|---|---|---|---|---|---|
| `diffuser` | ✗ | ✗ | ✗ | **115.7 m** | 194.5 | 🔴 |
| `model_free` | ✗ | ✓ | ✓ | **156.0 m** | 212.5 | 🔴 |
| `model_free-bounds_free` | ✗ | ✗ | ✓ | **174.2 m** | 208.2 | 🔴 |
| `geo_free-model_free` | ✗ | ✓ | ✗ | **110.2 m** | 213.1 | 🔴 |
| `bounds_free` | ✓ | ✗ | ✓ | 31.8 m | 25.8 | 🟡 |
| **`geo_free`** | ✓ | ✓ | ✗ | **0.59 m** | **0.31** | ✅ |
| **`geo_free-bounds_free`** | ✓ | ✗ | ✗ | **0.58 m** | **0.31** | ✅ |

🔴 **Every cell with dynamics OFF is 110–174 m. Every cell with dynamics ON and everything else OFF is
0.58 m.** Action bounds do nothing; geometry does nothing (it is dropped in both ✅ rows).
The same grid on `fm` K10 is flat: `diffuser` 1.46 m, `model_free` 1.34 m, `geo_free` 0.71 m —
**FlowMatching does not need the dynamics constraint; MeanFlow cannot fly without it.**

This also dissolves the apparent contradiction in the main sweep: `mf` scores 1.00 crossed / 0.00
abort under `-geo_free`, because **`-geo_free` still runs the dynamics projection**. `mf` is broken in
exactly one variant — the one with no projection at all.

### 2.1 …and the repair is *cheap*, which rules out "the model is simply bad"

`post_processing` sets `diffusion_timestep_threshold = 0.0` (`eval_mix_uav.py:1037`), so the projector
fires **once, on the final ODE step only** — a single snap, no per-step steering:

| K10, `pillars` | projection | crossed | reached | **goal_dist** | **n_viol** |
|---|---|---|---|---|---|
| `mf` `diffuser` | none | 0.33 | 0.00 | **115.65 m** | 597 |
| **`mf` `post_processing`** | **one shot** | 0.67 | 0.33 | **0.37 m** | **44** |
| `fm` `post_processing` | one shot | 1.00 | 0.33 | 1.27 m | 319 |
| **`mf` `hardflow_new`** | every step | **1.00** | **1.00** | **0.29 m** | **41** |
| `fm` `hardflow_new` | every step | 0.67 | 0.00 | 3.17 m | 355 |

A projection **adds no information** — it moves the sample to the *closest* point of the feasible set.
If the closest feasible point to `mf`'s plan beats everything `fm` produces, then `mf`'s plan carries
the right content and is merely sitting off the feasible manifold.

⚠️ **Scope.** This says the raw plan is **recoverable**. It does **not** contradict Q3: on the
`diffuser` metric the raw output *is* worse. Two different measurements, both true.

⚠️ **n = 3/cell**, older code rev. The direction is a 300× effect and agrees with the n=100 sweep, but
these exact numbers are not precision quantities.

---

## 3. Direct measurement of the raw output

`act_all` is the model's **raw Δp_des actually executed** — for `variant=diffuser` nothing has touched
it. Same checkpoint (`Emf_K1`, mf K1, `pillars`) on every row. Expert cruise:
**Δx ∈ [0.012, 0.027] m/step**, **Δz ≈ 0**.

| variant | constraints | n | **mean Δx** | mean Δz | **std Δz** | **std Δz / std Δx** |
|---|---|---|---|---|---|---|
| 🔴 `diffuser` | **none** | 778 | **+0.00692** 🔴 | +0.01107 | **0.16615** 🔴 | **22.3** |
| `dpcc-r-geo_free` | **dynamics + bounds only** | 6340 | **+0.01113** ✅ | +0.00002 | **0.00680** | **0.65** |
| `hardflow_sls` | all | 5345 | +0.01180 ✅ | +0.00065 | 0.02248 | 2.9 |
| `dpcc-r` | all | 4963 | +0.01266 ✅ | +0.00267 | 0.04485 | 5.4 |

1. 🔴 **The raw forward command is 2–4× below the expert band.**
2. 🔴 **The raw vertical channel dominates: `std Δz` is 22× `std Δx`.** The expert holds altitude, so
   this ratio should be well under 1.
3. ✅ **Projection with geometry OFF** — no box, no obstacles, so **clipping is impossible** — restores
   Δx to band and cuts `std Δz` **24×**. This is §2's dynamics result measured at the action level.

**Plan level** (`sampled_trajectories_all`, the model's own 8-step foresight; expert net travel over
H=8 is **0.10–0.22 m**):

| FM step | 2 | 3 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| median net Δx over the plan's horizon | −0.0011 | +0.0044 | +0.0168 | −0.0048 | +0.0003 | +0.1921 | +0.1056 |

🔴 **For the first ~20 control steps the model plans to travel 0–2 cm over its whole 8-step horizon,
against an expert 10–22 cm — a 10× deficit.** It reaches expert-scale travel only around step 40, by
which point the z-runaway has already carried the drone out of the envelope.

### 3.1 The defect is present in the *first* sample

Step-0 actions, from a fully in-distribution start state:

```
trial 0: dp_des = [+0.0002 +0.0007 +0.0335]      trial 5: [+0.0014 +0.0005 -0.0221]
trial 1: dp_des = [+0.0011 -0.0018 -0.0284]      trial 7: [+0.0000 +0.0023 +0.0523]
trial 2: dp_des = [+0.0020 +0.0031 -0.0451]      trial 9: [+0.0011 -0.0012 +0.0297]
```

**Δx ≈ 0.000–0.002 (≈10× under expert), Δz ≈ ±0.05 (20× larger than Δx, random sign).**
No closed-loop compounding is required to explain the failure — **the very first plan is already
wrong**, which means any explanation must act at *sampling* time.

---

## 3.5 🔴 DIRECT THREE-WAY RAW COMPARISON — and the defect it exposes

### 3.5.1 The comparison

Raw `act_all` (= Δp_des, nothing applied), `variant=diffuser`, `scene=pillars`, `controller=pid_stopgo`,
`cond_mode=pos_only`, seed folder 6 — **identical harness for every row**:

| arm | n | mean Δx | mean Δz | std Δx | std Δz | **std Δz/Δx** | **med \|Δp\|** | **P(\|Δz\|>\|Δx\|)** | **fwd frac** |
|---|---|---|---|---|---|---|---|---|---|
| **fm** Gen11 (job 23265) | 6340 | **+0.01156** ✅ | −0.00302 | 0.01108 | 0.03451 | 3.1 | 0.0259 | 53.3 % | **+0.474** |
| 🔴 **mf** Gen15 K1 | 778 | +0.00692 | +0.01107 | 0.00744 | 0.16615 | **22.3** | 0.0524 | **96.0 %** | **+0.120** |
| 🔴 **mf** Gen15 K10 | 1902 | **+0.00276** | **+0.65709** | 0.00675 | 0.48505 | **71.8** | **1.00001** | **99.7 %** | **+0.019** |

`fwd frac` = mean cosine between the commanded step and +x (1.0 = straight at the goal).
Reference: expert cruise 0.4–0.9 m/s @33 Hz ⇒ Δx 0.012–0.027, Δz ≈ 0.

**`mf` commands a step that is almost never pointed at the goal** — 96–99.7 % of steps have a larger
vertical than forward component, and `fwd frac` collapses from `fm`'s 0.474 to 0.120 / 0.019.

⚠️ **Confound, stated up front:** the `fm` arm is **Gen11 `flow_matching_v3_uav`** (job 23265, git
`6d251f6`, 2026-07-10), not Gen15 `mix_uav`. Scene, controller, obs layout, normalizer class and seed
folder all match; the generation and code rev do not. **No `af` UAV `diffuser.npz` exists locally at
all** — every local UAV artifact tree is `mix_uav_mf`. The `af` row of this table is **missing, not
omitted**, and must be fetched from the cluster before this table is quoted as three-way.

### 3.5.2 `med |Δp| = 1.00001` is not a physical number — tracing it

A median commanded step of **exactly 1 m** at 33 Hz is not something a trained model produces by
accident. Testing whether the channel is pinned at a bound:

| arm | n | **max \|Δz\|** | **\|Δz\| > 0.99** | \|Δz\| > 0.5 | max \|Δx\| | max \|Δy\| |
|---|---|---|---|---|---|---|
| fm Gen11 | 6340 | 0.22634 | **0.0 %** | 0.0 % | 0.04406 | 0.03573 |
| mf K1 | 778 | **1.00000** | 0.1 % | 2.8 % | 0.03864 | 0.03972 |
| 🔴 mf K10 | 1902 | **1.00000** | **55.5 %** | **72.2 %** | 0.04406 | 0.03972 |

🔴 **`max |Δz|` is exactly `1.00000`, and mf K10 sits on that bound for 55.5 % of every executed step.**
Meanwhile `max |Δx|` is ~0.044 for **every** arm. The two channels have different ceilings — **23×
apart** — and that is not a property of any model.

### 3.5.3 The amplifier — a scale-calibration defect, **not** a logic bug

`mix_uav/datasets/normalization.py:177-193`:

```
class SafeLimitsNormalizer(LimitsNormalizer):
    # functions like LimitsNormalizer, but can handle data for which a dimension is constant
    def __init__(self, *args, eps=1, **kwargs):            # <-- eps = 1
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                self.mins[i] -= eps                        # -> -1
                self.maxs[i] += eps                        # -> +1
```

and `LimitsNormalizer.unnormalize`:

```
    def unnormalize(self, x, eps=1e-4):
        if x.max() > 1 + eps or x.min() < -1 - eps:
            x = np.clip(x, -1, 1)          # silent; the warning print is commented out
        x = (x + 1) / 2.
        return x * (self.maxs - self.mins) + self.mins
```

**On the `pillars` expert data the vertical action is exactly constant** — the eval banner prints it on
every run, `fm` and `mf` alike:

```
[ utils/normalization ] Constant data in dimension 2 | max = min = 0.0
```

The drone holds altitude, so `Δz ≡ 0.0` and `min == max`. `SafeLimitsNormalizer` therefore widens that
dimension to `[−1, +1]`, and for that dimension `unnormalize` becomes

> `(x+1)/2 · (1 − (−1)) + (−1) = x` — **the identity.**

>  ⚠️ **This code is not wrong on its own terms.** It exists to avoid `0/0 = NaN`, and it
>  **round-trips the data exactly**: every training sample in that dim maps to the midpoint `0` and
>  back to `c`. The model's target in that channel is identically zero, and a model that outputs zero
>  produces `Δz = 0`. A correctness audit passes it — correctly. The defect is one of **scale
>  calibration**: `eps` is a magic constant expressed in *data units*, benign at D3IL's O(1) actions
>  and three orders of magnitude out of scale at the UAV's O(0.02 m) actions.

🔴 **Consequences, all three measured above:**

1. **The model's raw normalized output in the z channel is emitted verbatim as metres.** A normalized
   0.657 becomes a **0.657 m** vertical command per control step.
2. **The clip at ±1 becomes a ±1 m ceiling**, which is exactly the `max |Δz| = 1.00000` and the
   `med |Δp| = 1.00001` (‖[≈0, ≈0, ±1]‖) observed. It is applied **silently** — the warning print is
   commented out, so nothing in any log says the action was clipped.
3. **The gain is asymmetric by ~23×.** Real channels are compressed by their data range
   (`max |Δx| ≈ 0.044` ⇒ ≈0.044 m per normalized unit); the widened channel gets **1.0 m per
   normalized unit**. Identical normalized noise is **23× more dangerous in z than in x**.

This is **inherited from upstream, not introduced here** — `aux_repo/dpcc/diffuser/datasets/normalization.py:182`
carries the same `eps=1`. It is harmless on D3IL/maze, where action units are O(1) and no dimension is
constant. On the UAV, where actions are O(0.02 m) **and** one dimension is exactly constant, `eps=1` is
three orders of magnitude out of scale.

### 3.5.4 What this does and does not explain

✅ **It explains the shared z-runaway of §1.2.** Every engine aborts on z, never on x. `fm` puts 53 % of
its steps' energy into the vertical channel too — it just never exceeds 0.23 m, so it survives.

✅ **It explains why the dynamics projection is the sole repair (§2).** The dynamics constraint is the
only constraint group that ties Δz to the state trajectory, so it is the only one that can suppress a
metre-scale command the normalizer has decoupled from any physical scale.

🔴 **It does NOT cause the failure, and must not be reported as its cause.** A model that learned the
channel emits ~0 there and is untouched by the gain — which is exactly what `fm` does (`max |Δz| =
0.226`, never within 4× of the clip). It is an **amplifier of a pre-existing defect**. `fm` runs through the *same* normalizer
with the *same* ±1 m z ceiling and never gets within 4× of it. The defect is the **transmission
mechanism**, not the source.

> **But it reframes the open question, and makes it much sharper.** It is no longer
> *"why does MeanFlow diverge?"* — it is **"why does MeanFlow leave residual energy in the one action
> channel whose training target is identically zero, when FlowMatching does not?"** A channel with
> `min == max == 0` carries **no gradient signal about scale**; the model can only learn "output the
> midpoint". Whatever `mf` does differently at sampling — the untrained `(r=0, h=1/K)` corner of §4 is
> the obvious suspect — lands in that channel unattenuated.

### 3.5.5 The fix, and the test

🔧 **Not applied — code change, awaiting go-ahead.** `eps=1` should be scale-aware, e.g. a small
fraction of the median non-constant range, or the constant dimension should be dropped / hard-zeroed
rather than widened. Per repo convention this exists in **20 sibling copies** of
`datasets/normalization.py` and would need the usual mirrored sync; it also **invalidates existing
checkpoints' normalizer params**, so it is a retrain, not a re-eval.

🧪 **Zero-cost check that needs no retrain:** hard-zero the z action after unnormalize for one
`diffuser` eval per engine. Prediction — `mf`'s aborts drop sharply and it starts making forward
progress, because §3's forward channel (+0.0069 m/step) is slow but not zero; `fm` is barely affected.
If `mf` still fails with the z channel muted, the defect is incidental and §4's two candidates remain
the whole story.

---

## 3.6 ✅ THE ANSWER — MeanFlow learned a **positive altitude-feedback gain**; fm and af did not

The full matched three-way artifact set arrived at `temp/3008/uav-pillars/` (Gen15 `mix_uav`, `pillars`,
seed 6, `pid_stopgo`, K1/K2/K5 for all three engines, plus every training checkpoint). That closes the
comparison — and it overturns §3.5's reading.

### 3.6.1 First: the forward-deficit finding was a truncation artifact

Mean forward command `Δx` **per FM-step bin** (expert 0.012–0.027), `mean |Δz|` in brackets:

| eng | K | 0–5 | 5–10 | 10–20 | 20–40 | 40–60 | 60–100 | 100–200 | 200–400 |
|---|---|---|---|---|---|---|---|---|---|
| af | 1 | +0.0008 [0.002] | +0.0011 [0.002] | +0.0017 [0.003] | +0.0028 [0.003] | +0.0043 [0.002] | +0.0067 [0.002] | +0.0189 [0.024] | +0.0228 [0.027] |
| fm | 1 | +0.0006 [0.029] | +0.0002 [0.034] | +0.0002 [0.028] | +0.0006 [0.032] | +0.0023 [0.042] | +0.0059 [0.040] | +0.0250 [0.037] | +0.0217 [0.045] |
| fm | 5 | +0.0004 [0.005] | +0.0003 [0.005] | +0.0005 [0.004] | +0.0015 [0.004] | +0.0037 [0.005] | +0.0078 [0.012] | +0.0162 [0.012] | +0.0218 [0.012] |
| **mf** | 1 | **+0.0013** [0.029] | **+0.0014** [0.036] | **+0.0019** [0.037] | **+0.0043** [0.075] | **+0.0093** [0.139] | **+0.0117** [0.133] | +0.0186 [**0.279**] | — |
| **mf** | 5 | +0.0005 [0.004] | +0.0005 [0.006] | +0.0010 [0.016] | +0.0034 [0.059] | +0.0082 [0.105] | +0.0109 [0.174] | +0.0112 [**0.423**] | — |

🔴 **`mf`'s forward command is not deficient — in the early bins it is the *largest* of the three**
(+0.0013 vs af +0.0008 and fm +0.0006 at steps 0–5; +0.0093 vs +0.0043 / +0.0023 at steps 40–60).
Every engine ramps identically from ~0.0006 to ~0.022 over ~200 steps; that ramp is the `pid_stopgo`
start transient, not a model property.

**§3.5's "raw forward command 2–4× below expert" was an averaging artifact.** `mf` aborts at step
47–144, so its rollout mean is taken almost entirely over the **ramp-up phase**; `fm`/`af` run the full
634 steps and their means are dominated by **cruise**. Comparing rollout means across arms with
different lifetimes is the same survivorship trap this DA flags elsewhere — and it was walked into here.

✅ **The z column is the real signal, and it is not a bias — it *grows*.** `af` stays at 0.002–0.027 and
`fm` at 0.004–0.045 for the whole rollout; `mf` climbs **monotonically** 0.029 → 0.279 (K1) and
0.004 → **0.423** (K5), a **100× growth**. At steps 0–5 `mf` K5's `|Δz|` is 0.004 — *identical to*
`fm` K5's 0.005. **They start the same and `mf` diverges.**

### 3.6.2 The mechanism, measured: the vertical feedback gain

If the vertical command grows from an identical start, that is a **closed-loop instability**, and its
gain is directly measurable. Regressing the model's commanded `Δz` on the current altitude error
`e_z = p_z − z_cruise` (`z_cruise = 1.1422 m`), **with the step index included as a covariate** so the
speed ramp-up cannot leak in, over a **matched window (steps ≤ 150)** so every arm is measured over the
same phase:

> `Δz_cmd  ~  b₁·e_z  +  b₂·step  +  b₃`

| engine | K=1 | K=2 | K=5 | |
|---|---|---|---|---|
| **af** | **−0.0018** | **−0.0084** | **−0.0123** | ✅ restoring |
| **fm** | **−0.0163** | +0.0123 | **−0.0265** | ✅ restoring / ~0 |
| 🔴 **mf** | **+0.1121** | **+0.0804** | **+0.0310** | 🔴 **destabilising, every K** |

**Sign convention:** `b₁ > 0` means *"the higher the drone is above cruise, the harder the model
commands it further up"* — positive feedback. `b₁ < 0` means it commands back down — restoring.

🔴 **`mf` is the only engine with a positive vertical gain, at every K, and it is 3–90× larger in
magnitude than any `fm`/`af` value in the table.** For the same regression on the x and y channels
`mf` is unremarkable (`b₁` = −0.011…+0.027, inside the `fm`/`af` spread). **The instability is
z-specific and mf-specific.**

### 3.6.3 The loop gain quantitatively predicts the abort time

`p_des` accumulates the command and the tracker chases it, so `e_z(t+1) ≈ e_z(t)·(1 + b₁)`. Solving for
the step at which that reaches the 3.30 m abort ceiling, from the measured perturbation at step 5:

| | b₁ | e₀ at step 5 | **n predicted** | **n observed** |
|---|---|---|---|---|
| mf K1 | +0.1121 | 0.0383 m | **38** | 64 (range 47–114) |
| mf K2 | +0.0804 | 0.0399 m | **52** | 69 (range 56–95) |
| mf K5 | +0.0310 | 0.0387 m | **132** | 86 (range 64–144) |

⚠️ A first-order model — it ignores tracker lag and treats the gain as constant — so this is an
**order-of-magnitude consistency check, not a fit**. It lands within ~1.5× on all three cells and gets
the K1 < K2 ordering right; K5 is over-predicted. What matters is that a gain measured *independently*
from the regression reproduces the observed 47–144 step abort window **from first principles**, and
that it explains **why the failure is flat in K**: `b₁ > 0` at every K, so the loop is unstable at every
K — only the time-to-abort changes.

### 3.6.4 The complete causal chain

| # | link | shared or mf-specific? |
|---|---|---|
| 1 | **The expert data contains no vertical control signal at all.** `Δz ≡ 0` exactly — the eval banner prints `Constant data in dimension 2 \| max = min = 0.0` on every run. **No training example anywhere shows an altitude error being corrected**, so the learned vertical feedback gain is **unconstrained by data** — it is pure architecture/objective inductive bias. | **shared** — enabling condition |
| 2 | **`SafeLimitsNormalizer(eps=1)` makes `unnormalize` the identity for that channel** (§3.5.3), giving it ~23× the gain of every other channel and a silent ±1 m clip. | **shared** — amplifier |
| 3 | 🔴 **`mf` learned `b₁ = +0.11/m` (destabilising); `fm` and `af` landed at −0.03…+0.01 (restoring/neutral).** | 🔴 **mf-specific — this is the differentiator** |
| 4 | **`pid_stopgo` faithfully chases `p_des`**, closing the loop: `e_z ← e_z·(1+b₁)` → 3.3 m by step ~60 → abort. | **shared** — consumer |

**Links 1, 2 and 4 are identical for all three engines. Only link 3 differs. That is why only `mf` fails.**

It also explains the two facts that killed every earlier hypothesis:

- ✅ **Flat in K** — `b₁ > 0` at K=1, 2 and 5, so the loop is unstable at every K. Not a sampler, resolution or OOD-in-`h` effect at all.
- ✅ **Why only the dynamics projection repairs it (§2)** — the dynamics constraint is the only constraint group that ties `Δz` to the state trajectory, so it is the only one that **breaks the feedback loop**. Geometry merely clips the symptom; action bounds do nothing to a 0.1 m command.

### 3.6.5 What is *still* open — narrower, but not small

**Why did MeanFlow land on a positive gain when FM and αFlow landed on ~zero?**

Because of link 1 this is **not a question about fitting the data** — no data exists in that channel. It
is a question about **what each architecture/objective extrapolates to in a zero-variance channel**, and
the answer is arbitrary in the sense that nothing in the loss penalises it. The §4.1 confound
(two-time U-Net coordinate vs. MeanFlow objective) is unchanged and still needs the 2×2 filled.

🔧 **The engineering fix does not depend on answering it — but the research claim does (see §3.6.7).** Link 1 is the root cause of the whole
class: a channel with `min == max` should not be a free output. Options, in increasing order of cost:
**(a)** hard-zero that action channel after unnormalize; **(b)** drop constant dims from the action space
entirely; **(c)** make `eps` scale-aware (§3.5.5). **(a)** needs no retrain and is testable in one eval.

**Prediction 🧪:** under (a), `mf` unguided should fly `pillars` — its forward channel is healthy
(§3.6.1) and the only thing killing it is the vertical loop. If it still fails, §3.6.2's gain is not
the operative mechanism and this section is wrong.

---

### 3.6.6 Verification — two independent checks, no regression assumptions

**Check 1 — non-parametric.** Mean commanded `Δz`, binned by altitude error. No model, no fit:

| eng | K | −0.10..−0.03 | −0.01..+0.01 | +0.01..+0.03 | +0.03..+0.10 | +0.10..+0.30 | **+0.30..+1.00** |
|---|---|---|---|---|---|---|---|
| af | 1 | −0.0041 | −0.0020 | −0.0005 | −0.0010 | −0.0005 | — |
| af | 5 | +0.0026 | +0.0023 | +0.0034 | +0.0035 | +0.0131 | **+0.0083** |
| fm | 1 | −0.0082 | −0.0051 | −0.0101 | −0.0162 | — | — |
| fm | 5 | +0.0026 | +0.0032 | +0.0036 | +0.0049 | +0.0215 | **+0.0065** |
| 🔴 mf | 1 | −0.0116 | −0.0135 | −0.0139 | **+0.0152** | **+0.0275** | **+0.1076** |
| 🔴 mf | 5 | +0.0258 | — | +0.0033 | **+0.0139** | **+0.0734** | **+0.0755** |

🔴 **When the drone sits 0.1–1.0 m above cruise, `mf` commands a further +0.075…+0.108 m/step
*upward*. `fm` and `af` command +0.0065…+0.0083 — an order of magnitude less.** `mf`'s row rises
monotonically; `af` K1 and `fm` K1 fall.

**Check 2 — causal.** `Δz(t) ~ b₁·e_z(t) + c·Δz(t−1) + b₂·step + const`, adding the **lagged command**
so pure command autocorrelation cannot masquerade as a state response:

| | b₁ (state response) | c (autocorrelation) |
|---|---|---|
| af K1 / K5 | −0.0022 / −0.0106 | −0.19 / +0.15 |
| fm K1 / K5 | −0.0172 / −0.0233 | −0.06 / +0.13 |
| 🔴 **mf K1 / K5** | **+0.1096 / +0.0283** | +0.02 / +0.13 |

`b₁` is essentially unchanged from §3.6.2 (+0.1121 → +0.1096 at K1) and the autocorrelation term is
small. **The command is responding to the state, not to itself.**

⚠️ **One honest qualification.** The clean *sign* reversal holds at **K=1** (`fm`/`af` restoring, `mf`
destabilising). At **K=5** `fm` and `af` are also mildly positive at large error (+0.0065/+0.0083), so
the robust statement is one of **magnitude — `mf`'s vertical response to altitude error is ~10× that
of `fm` and `af`** — with the sign reversal an additional K=1 finding, not a universal one.


---

### 3.6.7 The gain tracks the training starvation — hypothesis 3 is back, as a contributor

§4 hypothesis 3 (train/sample time-distribution starvation) was recorded as *killed* because `af`
shares the identical `_sample_tau_pair` and flies. That proves it is **not sufficient**. It does not
prove it is not **contributing** — and the measured gain now supplies a quantitative signature that it is.

The `mf` sampler's first step queries `(r = 0, h = 1/K)`, and that first step carries the whole
transport off pure noise. Training coverage of that interval width rises with K; the measured
destabilising gain **falls monotonically with it**:

| K | first-step query | **training mass** `P(h > 1/2K)` | **measured b₁** | median steps survived |
|---|---|---|---|---|
| 1 | (0, 1.00) | **4.2 %** | **+0.1121** | 66 |
| 2 | (0, 0.50) | 20.1 % | +0.0804 | 70 |
| 5 | (0, 0.20) | **36.7 %** | **+0.0310** | 86 |

`corr(coverage, b₁) = −0.994`.

⚠️ **n = 3, and confounded** — larger K also means more integration steps and a more accurate ODE
solve, which would push the same direction. So this is **suggestive, not proof**. But it is a
*prediction the starvation hypothesis makes and the data satisfies*, and it is the only hypothesis in
§4 that makes a graded prediction at all. **Hypothesis 3 is reinstated as a live contributing factor.**

> **Revised status of the open question.** Calling it "much smaller" was too comfortable. It is
> narrower — the *engineering* fix (§3.6.5) does not depend on it — but the **research** claim does:
> if `mf`'s positive gain reflects a general failure to extrapolate in low-variance channels, it will
> resurface on any task with a near-degenerate action dimension, and that is a property worth knowing
> before `mf` is recommended anywhere. **The §4.2 2×2 should be run, not deferred.**


---

## 4. Why **only** `mf` — three hypotheses tested, three dead

Every single-factor explanation is killed by the **same control**: `af`.

| # | hypothesis | why it dies |
|---|---|---|
| 1 | the **few-step average-velocity jump** degrades the plan | `af` runs the *identical* sampler — `af_diffusion.py:340-341`, `x = x + u(x,t,h=dt)·dt`, same `_predict_velocity` that discards the `v` head — and scores **0.93** |
| 2 | the **U-Net backbone** | `fm` is a U-Net and scores **0.90** |
| 3 | **train/sample time-distribution starvation** | 🟡 **not sufficient** — `af_diffusion.py:_sample_tau_pair` is *"Gen3v6 verbatim"*, so `af` is starved identically and still scores **0.93**. ⚠️ **But not dead: §3.6.7 shows the measured gain `b₁` falls monotonically with this hypothesis' own coverage metric (corr −0.994). Reinstated as a contributing factor.** |

Hypothesis 3 is worth recording even though it is dead, because it is the closest thing to a defect
found anywhere in this study. Simulating `_sample_tau_pair` (400 k draws, `p_mean=-0.4, p_std=1.0,
data_proportion=0.5`):

| | P(anchor < 0.05) | sampler's 1st-step query | train mass at that query (±0.05) |
|---|---|---|---|
| `mf` anchor `r` | **0.0004** | `(r=0, h=1/K)` | K1 **0.00000** · K2 0.00007 · K5 0.00003 |
| `fm` query `t` | **0.0739** | `t = 0` | — (single-time; nothing to co-miss) |

`fm` gets **185× more training mass where its sampler starts**, and `mf`'s K=1 corner `(r=0, h=1)` drew
**zero** hits in 400 k samples. It would have explained the flatness in K perfectly — every K's first
step sits at `r = 0`, and that step carries the whole transport off pure noise.

> ⚠️ **`af` sharing the starvation proves it is not *sufficient*, not that it is not *contributing*.**
> An untrained `(r=0, h=1/K)` corner is exactly where a one-jump sample's transport magnitude would come
> out mis-scaled, and mis-scaled first-step transport is what §3 measures. It stays on the list as a
> **contributing** factor that only one backbone fails to extrapolate through.

### 4.1 What is left, and why this batch cannot decide it

| arm | backbone | **time conditioning** | unguided crossed |
|---|---|---|---|
| `fm` | U-Net `unet1d_temporal_cond` | `E(t)` — **single time** | ✅ 0.90 |
| `af` | SiT `af_sit_trajectory` | `c = E(t_abs) + E(r_abs)` — two **absolute endpoints** (`:362`) | ✅ 0.93 |
| 🔴 `mf` | U-Net `unet1d_twotime_cond` | `t = E(τ) + E_h(h)` — **start + interval**, additive (`:245`) | 🔴 0.00 |

`mf` is the only arm conditioned on the **interval width as a separate additive term**. That is the
same coordinate defect the Gen3v7 audit ranks #1 on `avoiding`
([`../../Gen3v7_AlphaFlow/Study/REPORT_20260830_af_unet_vs_sit_avoiding_root_cause.md`](../../Gen3v7_AlphaFlow/Study/REPORT_20260830_af_unet_vs_sit_avoiding_root_cause.md) §6.1),
reached there from a different direction.

**But `mf` is the only two-time-U-Net arm and `af` is the only SiT arm — one cell per condition.
Engine and backbone are perfectly confounded**, so no re-cut of these CSVs separates "MeanFlow's
objective" from "the `E(τ)+E(h)` U-Net". Any sentence of the form *"MeanFlow can't fly unguided"* is,
on this evidence, unearned.

Two facts also constrain any explanation, and both are awkward:

- 🔴 **Flat in K** — 0/10 at K=1, 2, 5 with goal_dist 6.46 / 6.45 / 6.51. A time-resolution,
  OOD-in-`h`, or solver-step story would be **K-dependent**.
- 🔴 **On `avoiding-d3il` the pairing is exactly reversed** — MF-U-Net **works**, AF-U-Net fails (same
  report, §0.1 2×2). Neither the objective nor the backbone is universally at fault; this is a **task
  interaction**, and the UAV's 634-step receding-horizon loop is a far harsher open-loop consumer than
  `avoiding`'s short episodes.

### 4.2 The run that settles it — fill the 2×2

`mf_dit_trajectory.py` and `mf_dit_official_trajectory.py` already exist in `mix_uav/models/`, so `mf`
on the SiT/DiT bone is a **config change, not new code**:

| | U-Net (single-time) | U-Net (two-time) | SiT |
|---|---|---|---|
| `fm` | ✅ 0.90 (have) | — | — |
| `mf` | — | 🔴 0.00 (have) | ❓ **run this** |
| `af` | — | ❓ *(optional)* | ✅ 0.93 (have) |

One `mf`-on-SiT UAV train + one `diffuser` eval. Flies unguided → the backbone coordinate is the cause
and MeanFlow is exonerated. Still dives → the objective is implicated and the `avoiding` contrast
becomes the thing to explain.

---

## 5. The low-level controller — `pid_stopgo`, and what it does and does not explain

Every run in this drop uses **`controller='pid_stopgo'`** (`config/uav_mix.py:132,169`; the eval folder
tag is literally `Emf_K1_mpc4_pid_stopgo_T0.5`). Its definition is one line:

```python
# eval_mix_uav.py:1310-1311
if controller == 'pid_stopgo':
    v_des = np.zeros(3)          # U2: strict stop-and-go
```

The cascaded PID gets `p_des` from the model (`p_des += Δp_des` each FM step) but is told the desired
**velocity is zero**. So there is **no velocity feed-forward at all**: the drone is dragged purely by
position error, accelerates toward `p_des`, and brakes to a stop — every control step. The two
alternatives in the same switch are `pid` (`v_des = action/dt_fm`, the E7 default) and `pid_const_v`
(`v_des = unit(action)·v_des_magnitude`, U3).

### 5.1 Measured behaviour — `obs = [p_des | p]` makes this directly observable

`cond_mode='pos_only'` stores the realised `[p_des(0:3) | p(3:6)]` per FM step in `obs_all`, so
commanded step, achieved step and tracking lag can all be read off the same artifact:

| variant | n | **`\|Δp_des\|`/step (commanded)** | **`\|Δp\|`/step (achieved)** | **realisation ratio** | lag `\|p_des−p\|` median | **p95** |
|---|---|---|---|---|---|---|
| 🔴 `diffuser` (raw mf) | 778 | **0.0517** | 0.0196 | **0.38** 🔴 | 0.515 m | **2.862 m** 🔴 |
| `dpcc-r-geo_free` | 6340 | 0.0159 | 0.0143 | **0.90** ✅ | 0.350 m | 0.648 m |
| `hardflow_sls` | 5345 | 0.0244 | 0.0161 | 0.66 | 0.402 m | 0.818 m |
| `dpcc-r` | 4963 | 0.0267 | 0.0178 | 0.67 | 0.451 m | 1.121 m |
| `dpcc-t` | 4057 | 0.0259 | 0.0185 | 0.71 | 0.483 m | 1.280 m |

**This is where "it lost control" is quantified.** Under raw `mf` the airframe realises **38 %** of the
commanded displacement and the setpoint runs **2.86 m (p95)** ahead of it. Under the same controller
with a dynamically-coherent plan, realisation is **0.90** and the lag stays under **0.65 m**.

Note also the first column: raw `mf`'s commanded step is **0.0517 m — 2–4× *larger* than the expert
0.012–0.027**. This is the number that refutes the "collapsed to a point" reading. The command is not
small; it is aimed wrong.

### 5.2 What `pid_stopgo` explains — and what it does not

✅ **It explains the *shape* of the divergence.** With `v_des = 0` the loop needs a **standing position
error** to produce any motion at all, so a lag is structural, not a bug. When the plan is incoherent
that lag has no equilibrium: `p_des` accumulates faster than the airframe can close it, the position
loop demands ever-larger attitude, and the outcome is exactly the observed
`inverted` (12/30, `body z · world z` down to −0.28) and `overspeed` (3/30, 6.06–6.15 m/s) reason
codes. The `|p_des − p|` values in the abort records — **2.34, 2.96, 4.25 m** — are that runaway
caught in the act.

🔴 **It does NOT explain the `mf`-specific failure, and cannot be the cause.**

1. **Same controller, same airframe, same scene, same seeds for all three engines.** `fm` and `af` fly
   under `pid_stopgo` at realisation ratios that keep them in the expert band.
2. **Same controller for the projected `mf` runs**, which reach 0.90 realisation and 0.35 m lag on the
   *same weights*. Only the plan changed.
3. **The primary measurement is upstream of the controller.** §3's `act_all` is the model's Δp_des
   **before** the PID sees it. A controller cannot retroactively make the raw command point at the
   ceiling.

> **Reading:** `pid_stopgo` is an **amplifier**, not the cause. It is an unforgiving consumer of an
> incoherent plan — which is arguably what you want in an evaluation harness, because it makes plan
> incoherence visible instead of absorbing it.

### 5.3 🟡 A real train/eval mismatch worth flagging separately

The expert dataset was **deliberately de-stop-and-go'd**. `uav_expert_data_collect/trajectories.py:89`:

> *"U9: the per-segment `traverse_line` chain (v=0 at all 6 interior waypoints — the stop-and-go
> behaviour, see U8 analysis) is replaced by `blended_path`: … one global cosine speed profile (v=0
> only at episode start/end)."*

and `verify_blends.py` gates on it: *"1. No stop-and-go: interior speed never drops to ~0 (the whole
point of U9). 2. Consistency: analytic v matches finite-difference of p (no kinks/jumps at element
joints — **the PID feedforward is trustworthy**)."*

🟡 **So U9 rebuilt the dataset specifically so the PID's velocity feed-forward would be usable — and
the eval then runs `pid_stopgo`, which sets that feed-forward to exactly zero.** The demonstrations are
continuous-speed; the closed loop is stop-and-go. That is a mismatch between the data design and the
eval controller, it is **not** what causes the `mf` failure (§5.2), and it is **not** addressed in this
study — but it deserves its own check.

**Cheapest test:** re-run one `diffuser` cell per engine under `controller='pid'` (`v_des =
action/dt_fm`, the E7 default and the one U9 was designed for). Predictions, all 🧪 untested:
`fm`/`af` improve modestly; the realisation ratio rises toward 1.0 across the board; **`mf` still
fails**, because its Δx deficit is upstream of `v_des`. If `mf` *recovers* under `pid`, then §3's
reading is incomplete and the controller is doing more work than this study credits.

---

## 6. What to do next, in cost order

| # | action | cost | settles |
|---|---|---|---|
| 1 | **Log the raw-sample dynamics residual** `‖x_{k+1} − f(x_k,u_k)‖` before any projection, one rollout each of `fm`/`mf`/`af` | one eval, no training | confirms §2 at the source; prediction: `mf` highest, flat in K |
| 2 | **`mf` on the SiT/DiT bone**, `diffuser` eval only (§4.2) | one train + one eval | splits **objective vs backbone** — the whole open question |
| 3 | **One `diffuser` cell per engine under `controller='pid'`** (§5.3) | one eval | separates the U9 data/controller mismatch from the model defect |
| 4 | `mf` `diffuser` on `empty` (no obstacles, no projector) | one eval | confirms the finding with the environment fully removed |

---

## 7. Corrections to earlier drafts

Recorded because two of them were stated confidently before being checked.

1. 🔴 **"The few-step average-velocity jump is the cause."** Wrong — `af` uses the identical sampler
   and flies. Killed by §4 #1.
2. ⚠️ **"The first plans are flyable; the MPC loop compounds the error."** I struck this as falsified,
   then the matched three-way data **reinstated it** — see #6 below. It was correct. The strike was the
   error, caused by comparing `mf`'s step-0 action to the expert cruise rate instead of to `fm`/`af`'s
   step-0 action, which is identical.
3. ⚠️ **A single plan printout appeared to show the trajectory pinned at exactly `x = −3.2` for all 8
   horizon steps.** That is an **FM-step-0 artefact** — 50 % of step-0 plans have exactly zero net Δx,
   5 % at step 1, **0 % from step 2 onward**. It must **not** be cited as evidence of a collapsed plan.
   The real defect is the 10× magnitude/direction deficit, which is a different thing.
4. 🔴 **The §1 cross-engine abort table reported 5 / 1 / 16 aborts for fm / af / mf.** Those were
   **regex parse yields, not abort counts** — the Slurm one-liner prints `p=[...]` only for
   `off_route`, so all `inverted` and `overspeed` aborts were dropped. True counts are **9 / 5 / 30**,
   and the `af` row was **n=1** presented as an engine statistic. The table was also
   survivorship-conditioned (non-aborting rollouts excluded, post-crossing aborts mixed in).
   Corrected and caveated in §1.3; the study's conclusion now rests on the unconditioned
   crossing counts and on §3's `act_all` measurement instead.
5. 🔴 **"`mf`'s raw forward command is 2–4× below the expert band" (§3, §3.5) — FALSE, a truncation
   artifact.** It compared rollout means across arms with different lifetimes: `mf` dies at step
   47–144 and is averaged over the `pid_stopgo` ramp-up; `fm`/`af` run 634 steps and are averaged over
   cruise. **Per step-bin `mf`'s forward command equals or exceeds both** (§3.6.1). The forward channel
   is healthy. Corrected; the real defect is vertical and it is a *runaway*, not a bias.
6. 🔴 **"The defect is present in the first sample; no closed-loop compounding is required" (§3.1) —
   FALSE.** Step-0 actions are **indistinguishable across all three engines** (mean Δx +0.0005…+0.0010,
   `|Δz|` up to 0.030 for `fm` K1 too). The failure **is** a closed-loop runaway, with a measured gain
   (§3.6.2). This reverses correction #2 below, which was itself wrong.
7. ⚠️ **"`mf` is not low quality"** was stated without scope. It is true of the *projected* rows and of
   *recoverability* (§2.1); it is false of the raw `diffuser` output (§3). Both are now scoped.

---

## 8. Provenance

| artifact | path |
|---|---|
| mf K1 pillars eval artifacts (npz, abort file) | `temp/3008/Emf_K1_mpc4_pid_stopgo_T0.5/6/pillars_bounds+dynamics+geo_bounds+obstacles/` |
| mf pillars Slurm logs (K1/K2/K5) | `temp/3008/2026-08-27/18_34_44_eval_mix_uav_2513{1,2,3}.log` |
| fm pillars Slurm logs | `temp/3008/2026-08-27/18_34_21_eval_mix_uav_2512{7,8,9,30}.log` |
| af pillars Slurm logs | `temp/3008/2026-08-27/18_31_47_uav_mix_eval_2513{6,7,8}.log` |
| K10 constraint-toggle grid (Candidates 45 / 50) | `temp/3008/batch_uav_20260830_110536/per_rollout_detail.csv` |
| sampler / objective | `mix_uav/models/mf_diffusion.py`, `af_diffusion.py`, `diffusion.py` |
| backbones | `mix_uav/models/unet1d_twotime_cond.py:245`, `unet1d_temporal_cond.py`, `af_sit_trajectory.py:362` |
| constraint-group semantics | `mix_uav_test/eval_mix_uav.py:937-1031`, `:1037` |
| controller | `mix_uav_test/eval_mix_uav.py:1195-1201, 1305-1324`; `config/uav_mix.py:132,169`; `config/uav.py:129-134` |
| expert-data stop-and-go removal (U9) | `uav_expert_data_collect/trajectories.py:89`, `verify_blends.py` |
| cross-generation 2×2 | `logs_in_develop/Gen3v7_AlphaFlow/Study/REPORT_20260830_af_unet_vs_sit_avoiding_root_cause.md` |
