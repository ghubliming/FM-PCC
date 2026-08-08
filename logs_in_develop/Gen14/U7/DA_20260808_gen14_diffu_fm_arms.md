# DA — the two new Gen14 arms (`diffusion`, `fm`): do they still behave like Gen6V4 and Gen7?

**Date:** 2026-08-08
**Question:** two fresh Gen14 pipelines were launched — `engine=diffusion` (the Gen6V4 arm)
and `engine=fm` (the Gen7 reference arm). Do they reproduce the behaviour of the archived
Gen6V4 and Gen7 runs?
**Answer, in one line:** **no, in both directions and for opposite reasons.** The
`diffusion` arm is *much healthier* than the archived Gen6V4 run (which the 08-06 DA already
called dead), and the `fm` arm is *much worse* than either archived Gen7 run — 77 % of its
unprojected rollouts lose the arm entirely.

**Data:** `temp/2026-08-07/batch_va2_20260808_105342/` — one DA_VA_v2 batch, 11 candidates,
202 units, **3275 rollouts**, all from `logs/aligning-d3il-visual/plans` on i6-gpu-1, seed 6.
**Raw run material:** `temp/2026-08-07/` (pipeline / gate / train / eval logs for jobs
24338–24346, plus the two candidate zips).
**Figures:** `figs/fig1_0808_divergence.png`, `figs/fig2_0808_vs_old_generations.png`,
`figs/fig3_0808_cost.png`
**Script:** `da_20260808_gen14_diffu_fm.py` (scratchpad venv — this container has no project env)
**Predecessor:** `DA_20260806_gen14_mfaf_vs_gen7_gen6v4.md` — same batch tooling, one batch
earlier, and the source of the "Gen6V4 is a dead run" finding this document tests.

> The batch CSVs live under `temp/`, which is **gitignored** — local only.

> **Follow-up (same day):** both arms' K defaults were off-parity with the generation they
> exist to reproduce — see §4 and §7 item 1. Both are now **K=20**; see
> `CHANGELOG_20260808_K20_diffu_fm.md`. The `diffusion` arm needs a retrain to take it.
>
> **Scope note on §2.** "The `diffusion` arm is healthier than archived Gen6V4" is strictly a
> claim about *aliveness* — the arm follows the commanded trajectory and the box moves at all.
> It is **not** a claim that the policy is good. Watching the rollouts, both new arms still look
> like the old Gen6V4/Gen7 failures, and the numbers agree: **0 goal successes in 101 rollouts
> across the two arms**, exactly as for Gen6V4 and Gen7. Nothing in this batch solves the task.

### The two new runs

| | `engine=diffusion` | `engine=fm` |
|---|---|---|
| pipeline job | 24338 (gates 24339, train 24340, eval 24341) | 24343 (gates 24344, train 24345, eval 24346) |
| git rev | `bc9b93f` | `bc9b93f` |
| batch candidate | **7** | **8** |
| checkpoint tree | `mix_visual_aligning_diffusion/…_aw10_VTrue_steps1000_bs64_filmv1_Ediffusion` | `mix_visual_aligning_fm/…_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Efm` |
| training | 100 epochs × 1000 steps = 100 k steps, `status=completed` | same, `status=completed` |
| eval loads | `state_best.pt` — best-val step **98 000** | `state_best.pt` — best-val step **89 000** |
| K (NFE) | `n_diffusion_steps` = **100** DDPM steps | `flow_steps_v3` = **100** Euler steps |
| eval outcome | **CANCELLED, 24 h TIME LIMIT**, at item 2/32 | **CANCELLED, 24 h TIME LIMIT**, at item 2/32 |
| rollouts landed | `diffuser` 30/30, `dpcc-r` 19/30 | `diffuser` 30/30, `dpcc-r` 22/30 |

Both trainings finished cleanly. Both **evals died on the 24 h cap after 2 of 32 planned
items** — `diffuser` (unprojected) complete, `dpcc-r` truncated. Item 1 alone cost 5.3 h
(diffusion) / 5.0 h (fm); the eval's own ETA for item 2 was ~28.6 h / ~25.9 h. A full 32-item
K=100 sweep is not reachable inside one job.

Data quality is clean on what did land: **0 frozen rollouts, 0 circuit-breaker trips,
`npz_complete=1.0`** for both candidates. The failure mode below is not a stuck simulator.

---

## 0. TL;DR

1. **The `diffusion` arm does NOT reproduce the archived Gen6V4 run — it is far better.**
   Archived Gen6V4: 87 % of rollouts diverge (peak tracking error > 1 m), 77 % never move the
   box, median box displacement **0.000 m**. New `diffusion` arm: 43 % diverge, 13 % never
   move the box, median displacement **0.269 m**, median tracking error **0.053 m**. This is
   the **direct confirmation of the 08-06 DA §2 finding**: the DDPM engine is not what was
   broken in Gen6V4; that specific archived run was. See §2.
2. **The `fm` arm does NOT reproduce Gen7 — it is far worse.** On the unprojected `diffuser`
   variant it diverges in **23 of 30 contexts (77 %)** with a median peak tracking error of
   **2.84 m**, against Gen7-c3 at 27 % / 0.054 m and Gen7-c4 at 20 % / 0.050 m. Its box
   displacement median collapses to **0.031 m** (Gen7-c4: 0.273 m) and its violated-step count
   triples (236 vs 83). The arm the config calls "THE REFERENCE ARM … expects bit-identical
   training" is the sickest run in the batch apart from Gen6V4. See §3.
3. **This is not a K artefact alone.** All four Gen14 engines ran the same 30 train contexts at
   K=100: `mf` 20 % diverged, `af` 30 %, `diffusion` 43 %, `fm` **77 %**. Within `mf`/`af`,
   going K=2 → K=100 moves divergence by only 7–10 points. K explains part of the gap to
   Gen7's K=20, not a 77 % rate. See §4.
4. **The projector hides the symptom and exposes a second one.** Under `dpcc-r`, `fm`'s
   divergence drops to **0 %** — but **55 % of its rollouts never move the box** and its
   violated-step count is the worst of the four arms (125). Projected `fm` is not tracking
   badly; it is inert. See §3.2.
5. **Still zero goal successes for both new arms** — 0/49 (`diffusion`), 0/52 (`fm`), against
   1–2 for `mf`/`af` K=100 on the same contexts. So the headline "these engines do not solve
   the task" is unchanged from the 08-06 DA; what changed is *which* arm is pathological.
6. **Cost rules out K=100 as an operating point regardless.** 1.4–1.5 s/replan unprojected,
   7.7–8.5 s/replan with `dpcc-r`, against 27 ms / 53 ms for the K=2 arms. See §5.

---

## 1. What is comparable

### 1.1 The clean part

All six Gen14 candidates (`diffusion`-K100, `fm`-K100, `mf`-K100/K2, `af`-K100/K2) ran the
**same 30 train contexts**, verified exactly: box init xy, target xy and box angle match at
every `rollout_idx` to **0.000e+00**. The two variants both new arms produced — `diffuser` and
`dpcc-r`, geometry `combined_5` — are shared by all of them, and `mpc_batch_size=4` and
`diffusion_timestep_threshold=0.5` are identical across every candidate in the batch
(`run_config.csv`). So **§3 and §4 are a genuinely controlled engine comparison**: same
contexts, same env, same MPC pool, same constraint YAML, same K, only the generative engine
differs.

### 1.2 The confounded part

The cross-generation comparison inherits every limitation of the 08-06 DA, unchanged:

| | split | contexts | vs Gen14's train ctx |
|---|---|---|---|
| Gen14 (all arms) | train | 30 | — |
| Gen6V4, Gen7-c3, Gen7-c4 | **test** (their n=30 data) | 30 | **maxΔ = 1.715e+02 — different draw** |
| Gen6V4, Gen7-c1…c4 | train | **3** | maxΔ = 0.000e+00 — same contexts |

So a cross-generation claim is either 30-context-but-unpaired-across-splits, or paired-but-n=3.
§2 and §3 quote the 30-context form because the effects there are 3–8× in size and the
n=3 paired form (§6) points the same way; neither is a controlled comparison. Additional
confounds, per checkpoint identity:

- **K.** Gen14 `diffusion` K=100 vs archived Gen6V4 K=20. Gen14 `fm` K=100 vs Gen7 K=20.
- **FiLM.** Gen14 arms are `filmv1`; archived Gen6V4 and Gen7-c3 are `filmv2` (Gen7-c4 is v1).
- **`max_path_length`.** Gen14 `fm` is `steps1000`; Gen7-c4 (the filmv1 sibling) is `steps900`
  — a different checkpoint.
- **Weights.** All Gen14 arms load from their own tree; nothing is shared with Gen6V4/Gen7.

`mpc_batch_size` is **not** a confound here. The 08-06 DA §1.3 flagged Gen14's `mpc4` against
`plan_visual_aligning_dpcc`'s configured `mpc_batch_size: 1`, but the archived Gen6V4 *run* in
this batch is itself `…_mpc4_filmv2` and `run_config.csv` reports `mpc_batch_size=4` for it.
The config value and the archived run disagree; the two runs being compared do not.

---

## 2. The `diffusion` arm vs archived Gen6V4 — the engine is fine

Unprojected `diffuser` variant, each generation on the 30-context data it actually has:

| | n | **diverged** (peak err > 1 m) | **box never moved** | median box displacement | median peak tracking err | violated steps | mean dist |
|---|---|---|---|---|---|---|---|
| **Gen14 `diffusion` K=100** (train) | 30 | **43 %** | **13 %** | **0.269 m** | **0.053 m** | 113 | 0.334 |
| Gen6V4 archived K=20 (test) | 30 | 87 % | 77 % | 0.000 m | 3.248 m | 233 | 0.429 |
| Gen6V4 archived K=20 (train ctx 0–2) | 6 | 100 % | 100 % | 0.000 m | 3.472 m | 283 | 0.382 |

The archived Gen6V4 run's signature — the box literally never moves and the commanded
position sits 3.2–3.5 m from the arm — **does not appear** in the new run. The new arm pushes
the box a median 0.27 m and tracks to 5 cm in the median rollout.

**What this settles.** The 08-06 DA concluded that Gen6V4-in-that-batch was "a broken run, not
a weak baseline" and listed three candidate causes (checkpoint, eval wiring, config drift).
Re-running the same DDPM engine through the current Gen14 eval, from a fresh checkpoint, at
K=100/filmv1, produces a live policy. **The engine and the current eval wiring are exonerated;
the fault was in the archived Gen6V4 artefact** (its checkpoint or the state of the eval at the
time it was produced). Gen6V4 still needs its own re-run before it can be quoted as a baseline
— this run is the Gen14 arm, not Gen6V4.

**What this does not settle.** 43 % divergence is still not a healthy number, and the new arm
scores **0 goal successes in 49 rollouts**. "Better than a dead run" is the whole claim.

---

## 3. The `fm` arm vs Gen7 — the reference arm is the problem

### 3.1 Unprojected (`diffuser`), 30 contexts each

| | n | **diverged** | **box never moved** | median displacement | median peak tracking err | violated steps | constraint sat rate |
|---|---|---|---|---|---|---|---|
| **Gen14 `fm` K=100** (train) | 30 | **77 %** | **40 %** | **0.031 m** | **2.840 m** | **236** | **0.411** |
| Gen7-c3 filmv2 K=20 (test) | 30 | 27 % | 20 % | 0.041 m | 0.054 m | 172 | 0.571 |
| Gen7-c4 filmv1 K=20 (test) | 30 | 20 % | 20 % | 0.273 m | 0.050 m | 83 | 0.793 |
| *(same-batch reference)* Gen14 `mf` K=100 | 30 | 20 % | 10 % | 0.333 m | 0.050 m | 78 | 0.804 |

The failure is **bimodal, not a shifted mean**. Per context, the `fm` arm's peak tracking error
is either ~0.03–0.05 m (7 contexts) or **2.2–3.8 m** (23 contexts) — nothing in between
(*fig1*, right panel). In the diverged branch the violated-step count is 260–342 out of 400,
i.e. the commanded trajectory is outside the constraint set for most of the episode.

A correlate worth recording, though it is an outcome and not a cause: **divergence essentially
never occurs in `mode_encoding=0` rollouts** — across the whole batch, 0/9 for `fm`, 0/26 for
`diffusion`, 0/9 `af`-K100, 0/28 `af`-K2, 0/52 `mf`-K2 (the lone exception is `mf`-K100 at
3/23). Every diverged `fm` rollout is a mode-1 rollout.

### 3.2 Projected (`dpcc-r`) — symptom swapped, not cured

| | n | diverged | **box never moved** | median displacement | violated steps | sat rate | mean dist |
|---|---|---|---|---|---|---|---|
| Gen14 `fm` K=100 | 22 | **0 %** | **55 %** | **0.000 m** | **125** | 0.688 | 0.389 |
| Gen14 `diffusion` K=100 | 19 | 5 % | 32 % | 0.265 m | 63 | 0.844 | 0.303 |
| Gen14 `mf` K=100 | 11 | 0 % | 27 % | 0.261 m | 51 | 0.872 | 0.368 |
| Gen14 `af` K=100 | 11 | 0 % | 9 % | 0.241 m | 14 | 0.966 | 0.294 |

The DPCC projection pulls the commanded trajectory back into the reachable set — `fm`'s
divergence goes to zero — but the resulting policy touches the box in fewer than half its
rollouts and still logs the highest violation count of the four arms. Note both `dpcc-r`
columns for the new arms are **truncated samples** (19 and 22 of 30 contexts), so they are not
paired against the 11-rollout `mf`/`af` columns; treat the ordering as indicative.

### 3.3 Is the training suspect?

Training completed and the loss curves converged, but the two arms' held-out first-action loss
behaves differently: `fm` test `a0_loss` plateaus at **0.115–0.122** from epoch 56 onward while
its train `a0_loss` keeps falling to 0.02–0.03; `diffusion` ends at test 0.0055 / train 0.0036.
**These numbers are not comparable across arms** — the DDPM arm regresses ε at a discrete `t`,
the FM arm regresses a velocity at a Beta-sampled continuous `t`
(`mix_visual_aligning/models/helpers.py:188` is shared, the targets are not). The *within-arm*
train/test spread (7.5× for `fm`, 1.5× for `diffusion`) is the only readable signal, and it is
suggestive of the FM arm generalising worse on `a0`, not proof. Overall test loss was flat
(0.0159 → 0.0165 over the last 40 epochs), so this is not a diverging-training story.

---

## 4. K is not the explanation

All four engines on the identical 30 train contexts, `diffuser`, K=100:

| engine | diverged | unmoved | median peak err | violated steps | mean dist | goal successes |
|---|---|---|---|---|---|---|
| `mf` | **20 %** | 10 % | 0.050 | **78** | 0.311 | 1 |
| `af` | 30 % | 10 % | 0.082 | 112 | 0.416 | 1 |
| `diffusion` | 43 % | 13 % | 0.053 | 113 | 0.334 | 0 |
| **`fm`** | **77 %** | **40 %** | **2.840** | **236** | 0.372 | 0 |

And the K sensitivity measured *inside* the flow arms, same contexts, same variant:

| | K=100 | K=2 | Δ |
|---|---|---|---|
| `af` diverged | 30 % | 20 % | −10 pts |
| `mf` diverged | 20 % | 13 % | −7 pts |

Raising K from 2 to 100 costs the flow arms 7–10 points of divergence. Gen7 ran at K=20, one
step of that ladder — nowhere near enough to bridge 20 % (Gen7-c4) to 77 % (`fm` K=100). The
`fm` arm is an outlier among engines at fixed K, which is exactly the comparison the Gen14
frame was built to make.

**The decisive missing experiment is a Gen14 `fm` run at K=20**, matching Gen7's NFE. Until it
exists, "the Gen7 arm regressed" and "K=100 is uniquely bad for this particular checkpoint"
are not separated.

---

## 5. Cost

ms/replan, `combined_5`, the two items that landed (timing does not depend on split):

| variant | `diffusion` K100 | `fm` K100 | `mf` K100 | `af` K100 | `mf` K2 | `af` K2 | Gen6V4 | Gen7-c3 |
|---|---|---|---|---|---|---|---|---|
| `diffuser` | 1527 | 1426 | 893 | 902 | **28** | **27** | 337 | 335 |
| `dpcc-r` | 8516 | 7700 | 14988 | 16419 | **56** | **53** | 1705 | 11407 ‡ |

‡ Gen7-c3's `dpcc-r` cell is n=3 (train ctx 0–2); every other cell is n≥11. All rows pool
both splits, which is legitimate here and only here — cost does not depend on the context draw.

Nothing at K=100 is within two orders of magnitude of the 33 ms / 30 Hz budget. The K=2 arms
remain the only configuration in the family that is (*fig3*). The K=100 arms are diagnostic
runs, not candidate operating points.

---

## 6. The n=3 paired basis, for completeness

Train contexts 0–2, `combined_5`, the only ground shared with Gen6V4/Gen7 — **n=3, the
08-05 noise floor scales to roughly ±0.43 m on distance at this n, so read directionally only**:

| metric, `diffuser` | `diffusion` K100 | Gen6V4 | `fm` K100 | Gen7-c3 | Gen7-c4 |
|---|---|---|---|---|---|
| mean dist [m] | 0.407 | 0.382 † | 0.271 | 0.722 | 0.303 |
| peak tracking err [m] | 1.718 | 3.213 | 1.265 | 0.290 | 1.165 |
| box displacement [m] | 0.221 | **0.000** | 0.243 | 0.439 | 0.165 |
| violated steps | 194 | 283 | 120 | n/a | n/a |

† Gen6V4's distance is its initial box offset — the box did not move (08-06 DA §2).

On 3 contexts the `fm` arm looks unremarkable; on 30 it is the sickest arm in the batch. That
is the whole reason the 30-context within-Gen14 comparison (§3, §4) is the load-bearing one and
the cross-generation rows are context.

---

## 7. What to do next

1. **Re-run Gen14 `fm` at K=20** (`--flow-steps 20`), same 30 train contexts, `diffuser` only.
   One item, ~5 h. This is the single experiment that decides whether the reference arm
   regressed or whether K=100 is the problem. Everything else waits on it.
2. **Split the K=100 evals into per-variant jobs**, or cap them at `diffuser`. Two 24 h jobs
   bought 2 of 32 items and one truncated variant each; the same GPU-hours at K=2 covered 19
   variants × 2 geometries.
3. **Do not quote either new arm as a Gen6V4 or Gen7 replacement.** The `diffusion` arm is
   evidence about the *engine*, not a Gen6V4 baseline (different K, FiLM, checkpoint, split).
   Gen6V4 and Gen7-c3 still need their own re-runs on the current eval — the 08-06 DA's items
   2 and 3, still open.
4. **Add the divergence rate (`peak tracking error > 1 m`) to DA_VA_v2 alongside `unmoved_%`.**
   It is what separated the `fm` arm from the rest here, it is one line, and both the 08-06 and
   this DA had to compute it by hand.

**No code was changed for this DA.**

---

## 8. Reproduction

```bash
# all tables and figures in this document (scratchpad venv with pandas/numpy/matplotlib)
python logs_in_develop/Gen14/U7/da_20260808_gen14_diffu_fm.py \
       temp/2026-08-07/batch_va2_20260808_105342/per_rollout_detail.csv \
       logs_in_develop/Gen14/U7/figs
```

Source logs for the two runs, all under `temp/2026-08-07/`:

| file | what |
|---|---|
| `00_23_19_mix_visual_aligning_pipeline_24338.log` | diffusion pipeline submit |
| `00_23_19_train_mix_visual_aligning_24340.log` | diffusion training (100 epochs, completed) |
| `00_23_19_eval_mix_visual_aligning_24341.log` | diffusion eval (cancelled, 24 h cap) |
| `00_23_51_mix_visual_aligning_pipeline_24343.log` | fm pipeline submit |
| `00_23_51_train_mix_visual_aligning_24345.log` | fm training (100 epochs, completed) |
| `00_23_51_eval_mix_visual_aligning_24346.log` | fm eval (cancelled, 24 h cap) |
| `CAND_7__…_Ediffusion.zip`, `CAND_8__…_Efm.zip` | the two candidates' result trees |
