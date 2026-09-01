# DA 2026-09-01 — T1: raising `af_alpha_clamp` 0.005 → 0.05 on AF-UNet — **negative result**

**Jobs:** `25251` (train), `25254` (eval, chained `afterok`), `25253` (eval control — **failed**)
**Logs:** `temp/3008/2026-08-31/{18_16_05_train_alphaflow_25251, 18_19_42_eval_alphaflow_25253, 18_19_54_eval_alphaflow_25254}.log`
**Batch:** `temp/3008/batch_avoiding_combined_20260901_093057/`
**Task:** `avoiding-d3il`, H8, UNet@`freq_dim=32` (4.0 M params), seed 7
**Tests:** §9/§10 of [`../Study/REPORT_20260830_af_unet_vs_sit_avoiding_root_cause.md`](../Study/REPORT_20260830_af_unet_vs_sit_avoiding_root_cause.md) — T1 (the clamp experiment, report §7.5/§10.3) and F2 (the free `latest` re-eval, report §10.2)

---

## 0. Verdict — **T1 refuted: raising the clamp made the field 2.3× worse.**
### *Corrected 2026-09-01: the clamp is symmetric, so T1 cut the bootstrap at **both** ends. The α = 0 tail is no longer the only suspect — see §0.2.*

T1's hypothesis was that AF-UNet's rough `h > 0` field is under-trained because the α → 0 snap
leaves too few pure-MeanFlow steps to repair it. Raising the clamp lengthens that tail by 33 %.

On the **only target-legitimate cross-run metric** (α = 0 on both sides ⇒ byte-identical targets,
same seed, same split, same final step) the field got **2.3× worse**, not better:

| seed-7 final-step `val/` | clamp 0.005 (baseline, 24389) | **clamp 0.05 (25251)** | change |
|---|---|---|---|
| `raw_mse_u` | 6.86 | **15.78** | **2.3× worse** |
| `per_dim_rms_u` | 0.336 | **0.418** | 1.24× worse |
| *(MF-UNet reference)* | *1.90 / 0.199* | — | AF now **8.3×** MF |

This is a **useful negative**: it kills the "the tail is too short" hypothesis outright.
`INSIGHT_Gen3v7_first_train_curve` §1 had already recorded the field *degrading* during the α = 0
tail on the DiT (`h_mse_b3` → 269); report §7.5 flagged that T1 could therefore go either way. It
went the wrong way.

⚠️ **But the direction of the causal arrow is NOT settled.** §0.2 shows the clamp acts on both ends
of the schedule, so T1 also removed 9 394 steps of bootstrap at the α = 1 head. Two readings survive
the data and they recommend opposite experiments. Read §0.2 before acting on this DA.

**Do not spend seeds 6–10 × `n_trials = 20` on this configuration.**

### 0.1 Scope of the verdict — what to abandon, what stays open

The negative result above is narrow. It kills **one axis**, not the method.

**Abandon: the α schedule as a tuning knob on U-Net.** T1 was the last cheap lever on that axis and
it moved backwards. Further clamp/γ/horizon sweeps on AF-UNet are not justified — no seed spend.

**Do not abandon: α-Flow on U-Net as a question.** Gate G2 still holds — **AF ⊇ MF**, so MF's own
weights are a feasible point of AF's optimisation problem. A method that *contains* a working method
and still loses 8.3× (`raw_mse_u` 1.90 vs 15.78) has a broken **training signal**, not a broken
architecture. That is diagnosable, and the diagnosis is cheap.

**What remains open, cheapest first:**

| # | item | cost | why it is still live |
|---|---|---|---|
| 1 | **Checkpoint-selection defect** (report §4.3) | free (selection rule only) | `test_loss ≈ 0.75 + 0.25·α` ⇒ every AF-UNet `state_best.pt` is a **mid-homotopy α ≈ 0.01–0.02** model. Every AF-UNet *rollout* number on record was produced by that checkpoint, not by a trained endpoint. This confound sits under all AF-UNet rollout results and is removable without a GPU. |
| 2 | **Upstream `discrete_training`** (§6) | 1 training run | Floors α at `clamp_value` instead of snapping to exactly 0 — the direct test of the inverted hypothesis (that the snap *is* the damage event). Upstream-provided knob, not one we introduce. |

**Stopping rule.** If `discrete_training` does not move `val/raw_mse_u` off ~12–15 toward MF's 1.90,
declare AF-UNet a **clean negative**, write it up as such, and keep α-Flow on the SiT backbone where
it works. That is *one* bounded run, not a campaign.

**The caveat that argues for the cap.** The benchmark is not resolving the family difference at all:
mean violations 13.2 / 14.3 / 15.5 for ac05 / AF-baseline / MF (§4), echoing the report's K = 20
result (15.27 vs 15.50). So even a fully repaired AF field may not convert into a benchmark claim.
The remaining run is worth doing **for the mechanistic understanding**, not for a number in a table.

### 0.2 Correction — the clamp is **symmetric**, so T1 shortened the bootstrap at both ends

`flow_matcher_v3_alphaflow/models/af_diffusion.py:472-476`:

```python
if ratio < clamp_value:          ratio = 0.0     # snap to pure MeanFlow
elif ratio > 1.0 - clamp_value:  ratio = 1.0     # snap to pure FM   ← ALSO clamp-driven
```

The `elif` is the branch this DA initially overlooked. Raising the clamp 0.005 → 0.05 does **not**
only move the α → 0 snap earlier; it moves the α → 1 release **later** by the same amount. With
α(p) = σ(−25(p − 0.5)), the two snap points are p = 0.5 ∓ ln((1−c)/c)/25:

| clamp | α = 1 head ends | α = 0 tail starts | **steps with a genuine bootstrap (0 < α < 1)** |
|---|---|---|---|
| 0.005 (baseline, 24389) | 28 827 | 71 173 | **42 346** |
| **0.05 (T1, 25251)** | **38 222** | **61 778** | **23 556 (−44 %)** |

So T1 is **not** "the same run with a longer MeanFlow tail". It is a run with **44 % less α-Flow**:
18 790 steps (18.8 % of training) that were genuine bootstrap in the baseline were replaced by the
two *degenerate endpoints* of the homotopy — plain FM at the head, plain MeanFlow at the tail.

**Two readings now survive the 6.86 → 15.78 result, and they point opposite ways:**

| reading | mechanism | the experiment it implies |
|---|---|---|
| **(a) The α = 0 snap is the damage event** | pure-MF training on a bootstrap-initialised net diverges; more of it is worse | port upstream `discrete_training` — floor α at `clamp_value`, never reach 0 (§6) |
| **(b) The bootstrap is what builds the field** | 0 < α < 1 is the only regime that trains `u` at `h > 0` toward a self-consistent target; cutting it 44 % starves the field | **lower** the clamp 0.005 → 0.0005 — the same knob turned the *other* way, giving 51 630 bootstrap steps (+22 %) |

Reading (b) is the more parsimonious one and it is the *favourable* reading for α-Flow, so it must
not be assumed away. Note also that during the α = 1 head, `u_tgt = v` for **every** h (gate G1) —
i.e. the head actively teaches `u(z, r, h) ≈ v` at large h, which is the wrong answer for the
averaged field and has to be unlearned later. T1 gave the net 9 394 **more** steps of that. That is
a concrete harm mechanism located at the head, entirely independent of the tail.

**Consequence for §0.1:** the "abandon the α schedule as a tuning knob" line was written under the
one-ended reading and is hereby **narrowed** — abandon *raising* the clamp; **lowering** it is now a
live, zero-code, one-run test (the `AF_ALPHA_CLAMP` env + `_ac` path token already exist).

---

## 1. Provenance — the runs did what they were told

**25251 (train), clean.** `JOB START Mon Aug 31 22:13:58 UTC → JOB END Tue Sep 1 02:30:59 UTC`
(4 h 17 m), `Training complete for all seeds`. `[ train ] seeds='7'`, `bbunet`, and the savepath
carries the new token:

```
.../AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.0_ag25.0_rf0.5_ac0.05/7
```

The `_ac0.05` token is the `**_af_clamp_key` / `_af_clamp_tok` guard added to
`config/avoiding-d3il.py` on 08-31 doing its job: the run trained into its **own** tree instead of
colliding with 24389's and being silently skipped by `--auto-resume`. Baseline runs keep the bare
`..._rf0.5` path, so the default is unchanged.

**The α schedule moved as designed.** Printed banner:

```
[ train ] alpha schedule: sigmoid 1.0 -> 0.0 over [0, 100000] gamma=25.0 clamp=0.05
[ train ]   step :       0   10000   20000   30000   40000   50000   60000   70000   80000
[ train ]   alpha:   1.000   1.000   1.000   1.000   0.924   0.500   0.076   0.000   0.000
```

With α(p) = σ(−γ(p − 0.5)), γ = 25, the snap fires at α < clamp:

| clamp | α = 1 head ends | α = 0 tail starts | pure-MF steps | pure-FM steps | bootstrap steps |
|---|---|---|---|---|---|
| 0.005 (baseline) | 28 827 | 71 173 | 28 827 | 28 827 | 42 346 |
| **0.05 (T1)** | **38 222** | **61 778** | **38 222** (+33 %) | **38 222** (+33 %) | **23 556** (−44 %) |

Consistency check against the banner: α(60 k) = 0.076 > 0.05 (not yet snapped ✅), α(70 k) = 0.000 ✅,
and α(30 k) prints **1.000** although raw σ(5) = 0.9933 — the head snap firing, which is the direct
log evidence for §0.2 (the baseline at 30 k would have printed 0.993).

**End-of-run state is coherent with α = 0:** `val/alpha 0.0`, `train/clamp_frac 0.0`,
`train/discrete_frac 0.0`, `train/fm_frac 0.46875` (= `af_ratio_fm 0.5`). The engine was live and in
its MeanFlow-identical regime, as gate G2 requires.

**25254 (eval), clean.** 7 minutes, `Evaluation completed successfully`. Loaded EMA weights from the
`_ac0.05` tree, `AF_SEEDS override: seeds from env = [7]`, K ∈ {1, 2}. All K ≥ 2 HardFlow arms were
correctly auto-DISABLED by the low-K degeneracy guard (`n_genuine=0` at A = 0.5).

---

## 2. The control (F2) failed — and the failure is itself a finding

Job 25253 died in 5 seconds:

```
FileNotFoundError: .../AlphaFlowODE_aw10_bbunet_..._rf0.5/7/state_-1.pt
```

`state_-1.pt` is not a real filename. `get_latest_epoch()`
(`flow_matcher_v3_alphaflow/utils/serialization.py:27`) globs `state_*`, parses the numeric label,
and returns `-1` when **nothing numeric matches**. `state_best.pt` throws `ValueError` on `int()` and
is skipped. So:

> **The baseline AF-UNet tree (`..._rf0.5`, seed 7) contains no numbered checkpoints — only
> `state_best.pt`.** `AF_EPOCH=latest` has never been available for the AF-UNet baseline, which is
> why every AF-UNet number in the record comes from `best`.

The trainer writes numbered states at `step % (n_train_steps // 5) == 0` → 20 k/40 k/60 k/80 k
(`flow_matcher_v3_alphaflow/utils/training.py:81,203-205`), so 24389 *should* have written four of
them. Either they were pruned, or the writes failed. **The train log header shows `/data` at 100 %:**

```
[ DISK ] /dev/md2p1      7.0T  7.0T   27G 100% /data   <- repo/logs
```

⚠️ **Operational risk, needs a human check on the cluster** — a 100 %-full `/data` can silently cost
checkpoints on any running job, not just this one:

```bash
ls -la logs/avoiding-d3il/flow_matching_v3_alphaflow/H8_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_ae0.0_ag25.0_rf0.5/7/
```

**Consequence for the report:** §10.2 ("the free `latest` re-eval") is **not free and not currently
runnable** for AF-UNet. §4.3's checkpoint-selection defect (`state_best.pt` is always the
mid-homotopy α ≈ 0.009–0.023 model) therefore stands unmitigated on the baseline, and there is no
`latest`-vs-`latest` control for T1's rollout numbers.

---

## 3. Where the damage is — the `h > 0` field, exactly as predicted

25251 final-step `val/` bucket errors (`h_mse_b0..b3`, quartiles of h):

| bucket | h range | `val/h_mse` | `train/h_mse` |
|---|---|---|---|
| b0 | h ≈ 0 | 3.48 | 3.76 |
| b1 | | **23.61** | 7.25 |
| b2 | | **41.14** | 2.92 |
| b3 | h ≥ 0.6 | 5.08 | 2.56 |

`b0` (the plain-FM corner, h ≈ 0) is fine and matches train. `b1`/`b2` blow out by 8–14× between
train and val. This is the report §6/§7 signature: the **large-`h` field is the part α-Flow's
finite-difference target under-pins**, and the α = 0 tail did not repair it. Note also
`val/raw_mse_v = 3.88` — the *instantaneous* velocity head is healthy; only the averaged field `u`
is damaged, which localises the defect to the MeanFlow/α-Flow target and not to the backbone.

---

## 4. Rollouts — no signal, and confounded. Do not cite.

`diffuser` arm (unprojected), S&C per halfspace, and mean constraint violations:

| model | epoch | K=1 S&C (TR/TL/both) | K=2 S&C (TR/TL/both) | K=2 mean viol | episodes |
|---|---|---|---|---|---|
| MF-UNet (5 seeds) | best | 0.00 / 0.10 / 0.00 | 0.00 / 0.10 / 0.00 | 15.5 | 30 |
| AF-UNet baseline (5 seeds) | best | — | 0.00 / 0.10 / 0.10 | 14.3 | 30 |
| **AF-UNet `ac05_latest`** | latest | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 13.2 | **6** |

Three reasons this table settles nothing:

1. **No power.** 1 seed × `n_trials = 2` = 6 episodes. `diffuser` S&C is 0.00 everywhere for every
   model, so the arm does not discriminate at this n regardless.
2. **Not matched.** The T1 eval ran the `A0.5_B4` config (MPC fan, HardFlow arms); the 5-seed
   baselines are the older no-`A`/`B` config.
3. **Not epoch-matched.** T1 is `latest`; both baselines are `best`, and §2 says a `latest` baseline
   cannot currently be produced.

Mean violation counts (13.2 / 14.3 / 15.5) also fail to separate the families — the same
non-separation the report §9.2 already recorded at K = 20 (15.27 vs 15.50). **The training-side
proxy in §0 is the real read.**

**Also worth recording:** `latest` resolves to **step 80 000**, never 100 000. The loop ends at step
99 999, so `step % 20000 == 0` never fires at 100 k. Every "`latest`" AF/MF checkpoint in this repo
is the 80 k checkpoint. Any DA phrasing "the 100 k endpoint" is describing a model that was never
saved and never rolled out.

---

## 5. What this does to the report's hypothesis ladder

*The report itself is finished and is deliberately left untouched; this table is the erratum, and
it lives here.*

| report item | status after this DA |
|---|---|
| §7.5 / §10.3 the clamp experiment (T1) | ✅ **run (25251), refuted.** Raising the clamp made the field 2.3× worse. **Do not run the `{0.005, 0.05, 0.15}` sweep as written.** |
| §7.5 / §10.3 *"its floor is MF-UNet quality"* | ⛔ **false — the claimed floor does not exist.** Two independent reasons, both worth carrying forward: **(i)** the argument treats the clamp as one-ended, but `af_diffusion.py:472-476` snaps to `1.0` when `ratio > 1 − clamp` as well, so raising it cuts genuine-bootstrap steps 42 346 → 23 556 (−44 %) instead of merely extending the MF tail (§0.2); **(ii)** *"after the snap α-Flow **is** MeanFlow"* is true but does not imply an MF floor — MF-from-scratch and MF-on-a-bootstrap-initialised net are different optimisation problems, and the tail inherits whatever basin the α > 0 phase left. The rider *"plus a pure-FM warm-up MeanFlow never gets"* is likely a **liability, not a bonus**: during α = 1, `u_tgt = v` for every h (gate G1), so the head teaches `u(z,r,h) ≈ v` at large h — the wrong answer for the averaged field, to be unlearned later. |
| §9.1's 8× resolution argument (the basis for choosing 0.05) | **not decided by this run.** It predicted the *direction* of a good clamp move; the move failed, but §0.2 shows the manipulation was not the clean one the argument assumed. The argument survives, its recommended sign does not. |
| §7.3 singular-vs-floored asymmetry | **unchanged** (was "strengthened" in this DA's first draft — retracted). It would be strengthened only under §0.2 reading (a); reading (b) leaves it untouched. §6 step 2 decides. |
| §10.2 the free `latest` re-eval (F2) | 🔴 **blocked** — `get_latest_epoch` → −1, i.e. no numbered checkpoints in the baseline tree (§2). |
| §4.3 checkpoint-selection defect | unchanged, and now unavoidable: `best` is the only AF-UNet baseline epoch that exists on disk. |
| §5 code audit / gates G1, G2 | unaffected — 25251 re-confirms α = 0 ⇒ `fm_frac`, `clamp_frac`, `discrete_frac` all behave. The audit did not cover the **schedule's** upper clamp branch, which is where §0.2's miss lived. |

---

## 6. Next — the hypothesis is now inverted

The live question is no longer *"how do we lengthen the α = 0 tail?"*. Per §0.2 there are two:
**"is the snap to exactly zero the damage event?"** (reading a) and **"does the bootstrap phase build
the field, so that shortening it starves it?"** (reading b). The two are separable — see step 2.

Upstream α-Flow already has the knob that tests this. `aux_repo/alphaflow/src/training/loss.py:421-426`:

```python
if current_ratio < cfg.clamp_value:
    current_ratio = 0.0
    if "discrete_training" in cfg and cfg.discrete_training:
        current_ratio = cfg.clamp_value       # floor at clamp, never snap to 0
```

`discrete_training` **floors α at `clamp_value` instead of snapping it to 0** — the model never
enters the pure-MeanFlow regime, so the degradation phase never starts. We have **not** ported this
mode. It is an upstream-provided hyperparameter, not an invented one, and it is a ~10-line port to
`flow_matcher_v3_alphaflow/models/af_engine.py`.

Ordered, cheapest first:

1. **Free** — check the baseline checkpoint directory (§2 command) and confirm whether any numbered
   AF-UNet checkpoint survives. If one does, F2 becomes runnable and the whole `latest` comparison
   reopens.
2. **Free, and now the decisive step** — W&B overlay of `h_mse_b0..b3` and `raw_mse_u` for 25251 vs
   24389-s7 on the step axis, with **all four** snap points marked: **28 827 / 38 222** (head
   release) and **61 778 / 71 173** (tail snap). This discriminates §0.2's two readings directly:
   * curves track each other until each run's own **tail** snap, then T1 degrades ⇒ **reading (a)**,
     the α = 0 snap is the damage event ⇒ do step 3a.
   * T1 is already behind by ~40 k, i.e. during the extended α = 1 **head**, before either tail snap
     ⇒ **reading (b)**, the bootstrap builds the field ⇒ do step 3b.
   * both ⇒ the two endpoints are each harmful and the bootstrap is doing real work; 3b first.
3a. **1 training run, needs a code change** — port `discrete_training` (α floored at `clamp_value`,
   never 0). ~10 lines in `af_diffusion.py`'s `_get_ratio`.
3b. **1 training run, ZERO code change** — `AF_ALPHA_CLAMP=0.0005`, the same knob turned the other
   way: bootstrap 42 346 → 51 630 steps (+22 %), head and tail each 9 284 steps shorter. The env
   wiring and the `_ac` path token already exist, so this is submit-and-wait. If reading (b) is
   right this is the *cheapest possible* confirmation, and it is a pure hyperparameter move — no
   invented knob.

Screen 3a/3b on `val/raw_mse_u` and `h_mse_b1/b2` against 24389-s7's **6.86**. Only spend rollout
seeds on a run that beats it.

**Not worth running:** more seeds on `ac05`, and any further *raising* of the clamp.

---

## 7. Confidence ledger

| claim | confidence | basis |
|---|---|---|
| T1 made the field worse | **high** | matched seed, matched metric, matched final step, α = 0 on both sides ⇒ identical targets (gate G2) |
| Raising the clamp is a dead axis | **high** | 2.3× worse on the matched metric, whatever the mechanism |
| The α = 0 tail *specifically* is what degrades the field | **low** | ⚠️ confounded: §0.2 — the clamp is symmetric, so T1 also lost 9 394 bootstrap steps at the α = 1 head. `INSIGHT_Gen3v7_first_train_curve` §1 points the same way but on a different backbone. Step 2 of §6 resolves this for free |
| The bootstrap phase (0 < α < 1) is what builds the `h > 0` field | **untested, plausible** | §0.2 reading (b); T1 cut it 44 % and the field got 2.3× worse — consistent but not isolating |
| Baseline AF-UNet has no numbered checkpoints | **high** | `get_latest_epoch` returned −1, which requires zero numeric `state_*` matches |
| Disk pressure caused it | **low** | `/data` at 100 % is suggestive, not evidence; needs the `ls` |
| T1 rollouts say anything | **none** | 6 episodes, unmatched eval config, unmatched epoch |
