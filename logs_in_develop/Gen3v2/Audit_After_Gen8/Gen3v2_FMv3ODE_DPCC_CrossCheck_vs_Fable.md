# Gen3v2 (state-based FMv3ODE / DPCC-Diffuser) — Cross-Check Against the Gen9 Fable Audit

**Auditor:** Claude Opus 4.8
**Date:** 2026-06-13
**Question asked:** Fable's `U3_audit_Fable.md` audited the **visual avoiding** pipeline
(`diffuser_visual_avoiding/`, `fm_visual_avoiding/`). Are those 10 findings specific to the
visual code, or do they also hit the **state-based standard avoiding** stack — the DPCC
Diffuser baseline (`diffuser/`, `scripts/eval.py`) and the Gen3v2 selectable-ODE Flow
Matcher (`flow_matcher_v3_ode_selectable/`)?

**Method:** Every finding re-checked by reading the actual state-based source. File:line
citations below are verified, not inferred from the visual audit. Scope read:
`scripts/eval.py`, `diffuser/utils/{serialization,training}.py`,
`diffuser/sampling/policies.py`, `diffuser/datasets/sequence.py`,
`diffuser/models/unet1d_temporal_cond.py`,
`flow_matcher_v3_ode_selectable/{models/diffusion.py, models/unet1d_temporal_cond.py,
sampling/policies.py}`, `config/avoiding-d3il.py`, `config/projection_eval.yaml`.

---

## Verdict table

| # | Fable finding (visual) | State-based / Gen3v2 status | Why |
|---|---|---|---|
| **B1** | RGB/BGR channel swap at eval | **N/A — clean by construction** | No images anywhere in the state pipeline |
| **B2** | Render-resolution mismatch (96 vs 1024) | **N/A — clean by construction** | No image rendering |
| **B3** | `trajectory_selection` dropped → variants identical | **CLEAN — state code is the correct ancestor** | `Policy` implements it; `eval.py` wires it |
| **B4** | Seed mismatch train vs eval YAML | **SHARED (ops-level), externally handled** | Same `projection_eval.yaml seeds:[6..10]`, same crash surface |
| **B5** | Dead 6-obstacle config; wrong YAML consumed | **N/A — single source of truth, by design** | State config has no parallel `constraint_list`; eval builds from `projection_eval.yaml` |
| **B6** | EMA trained+saved, never evaluated | **SHARED bug — confirmed** | `serialization.py:75` returns raw `trainer.model` |
| **B7** | Window-level split → leaky validation | **SHARED bug — confirmed** | `train_test_split=0.9` + stride-1 windows + `random_split` |
| **B8** | Final 20% of training never checkpointed | **SHARED bug — confirmed** | `save_freq=n//5`, no final save in `train()` |
| **B9** | `test()` leaves model in `eval()` mode | **SHARED but benign — confirmed** | Triggered (split<1); harmless (GroupNorm + manual CFG mask) |
| **B10** | Dead ODE-solver knobs in visual FM | **INVERTED — knobs are LIVE here** | Gen3v2's entire purpose; `torchdiffeq.odeint` consumes them |

**Headline:** The four **eval-domain** findings that dominate Fable's report — B1, B2, B3,
B5 — are **visual-only**. They are either physically impossible without images (B1, B2) or
are regressions the visual rewrite introduced relative to the state-based code, which is the
correct reference (B3, B5). The findings that **do** carry over are the four **Trainer /
serialization** items (B6, B7, B8, B9), because the visual stack inherited
`diffuser/utils/training.py` and `serialization.py` almost verbatim. B10 is *inverted*:
the dead knobs in the visual FM are exactly the feature Gen3v2 exists to provide, and it
is wired correctly.

---

## Findings that DO carry over (shared training/serialization ancestor)

### B6 — SHARED: EMA weights saved but never evaluated

`diffuser/utils/training.py` maintains and saves EMA in every checkpoint:
```python
self.ema_model = copy.deepcopy(self.model)          # :55
...
'ema': self.ema_model.state_dict(),                 # :227 (save) / :245 (save_best)
```
But `diffuser/utils/serialization.py:75` returns the **raw online** weights, and the EMA was
dropped from the experiment namedtuple — the original line is still commented out at `:9`:
```python
# DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
DiffusionExperiment = namedtuple('Diffusion', 'dataset model diffusion trainer epoch losses')
...
return DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, losses)  # :75
```
`scripts/eval.py:91-93` consumes `diffusion_experiment.diffusion` → raw weights. **Every
state-based DPCC and Gen3v2 FM number was measured on noisier raw weights, never EMA.**
Because *all* baselines in the repo share this, FM-vs-DPCC comparisons are internally
consistent — but absolute performance is understated for every model.

**Fix (no retrain — EMA already in checkpoints):** evaluate `trainer.ema_model`. Worth a free
A/B re-eval; identical to Fable's B6 fix, applied one layer lower in the stack.

### B7 — SHARED: window-level train/test split is leaky

`config/avoiding-d3il.py` sets `'train_test_split': 0.9` for every avoiding block
(`:112, 160, 209, 265, 322, 378, 432, 855`). With split < 1, `training.py:74-82` does:
```python
train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_test])
```
over **dataset windows**, and `sequence.py:make_indices` builds those windows at **stride 1**:
```python
for start in range(max_start):       # :77 — every offset, fully overlapping
    end = start + horizon
    indices.append((i, start, end))
```
Adjacent windows from one episode share `horizon−1` frames, so almost every held-out test
window has near-duplicate twins in train. Test loss ≈ train loss; `state_best.pt` (selected on
this test loss at `training.py:148-150`) carries little generalization signal. **Same leak,
same severity as Fable's B7** — and unlike the visual finding, `state_best` *is* produced here
(split<1), so `diffusion_epoch:'best'` selection is actively used and actively unreliable.

**Fix:** split at the **episode** level before windowing. Requires a retrain to produce a
meaningful `state_best`.

### B8 — SHARED: last 20% of training is never checkpointed

`training.py:60` `save_freq = n_train_steps // 5`; saving at `training.py:135` fires only when
`self.step % save_freq == 0` → steps 0/20k/40k/60k/80k for a 100k run. `train()` /
`train_epoch()` end the loop (`:184` increments `self.step`) with **no terminal save**. The
final 20k steps live only in `state_best.pt` if it happened to fire late. **Identical to
Fable's B8.**

**Fix:** `self.save(self.step)` at the end of `Trainer.train()`.

### B9 — SHARED but benign: `test()` never restores train mode

`training.py:199-217` `test()` calls `self.model.eval()` (`:200`) and never `self.model.train()`.
Because the avoiding config runs split<1, `test()` *is* invoked (`:145`), so from the first
eval tick onward training proceeds with the model in `eval()` mode. **Verified benign for the
state stack:**
- The temporal UNet has no `BatchNorm`/`nn.Dropout` module that responds to `.eval()`.
- The only "dropout" is **classifier-free-guidance conditioning dropout** —
  `unet1d_temporal_cond.py:199-202` samples a manual `Bernoulli` mask gated on the
  `use_dropout` *argument*, not on `self.training`:
  ```python
  if use_dropout:
      mask = self.mask_dist.sample(...).to(returns_embed.device)
      returns_embed = mask*returns_embed
  ```
  `model.eval()` does not touch this. (For avoiding, `returns_condition` is off anyway.)
- Normalisation is GroupNorm-family — mode-invariant.

So it is latent, exactly as in the visual code, but the *reason* differs: visual was benign via
the GroupNorm encoder swap; here it is benign because the CFG dropout is a manual mask and the
net has no mode-dependent layers. **Becomes a real bug the instant anyone adds `nn.Dropout`/
`BatchNorm`.** Fix: `self.model.train()` at the end of `test()`.

### B4 — SHARED (ops-level), externally handled

`scripts/eval.py:36-44` reads `config/projection_eval.yaml`, whose active line is
`seeds: [6,7,8,9,10]` (`:7`). The state eval then calls
`utils.load_diffusion(..., str(args.seed), ...)` (`:91`) per seed. If any of those seed dirs
was never trained, the load raises `FileNotFoundError` mid-run and the cross-seed aggregate is
never written — the **same failure surface** Fable flagged. Per-seed `--seed` fan-out
(`eval.py:42-44`, the sbatch pattern) sidesteps it. As in the visual audit, this is resolved
operationally by manual seed alignment on the remote, not a code bug. No state-specific action.

---

## Findings that do NOT carry over

### B1, B2 — N/A by construction (no images)

The state pipeline conditions on proprioceptive state only. There is no collection-time
`cv2.imwrite`, no `get_image`, no resize. The entire RGB/BGR + render-resolution failure
class cannot exist. Clean structurally, not by patch.

### B3 — CLEAN: the state code is the *correct* reference the visual code regressed from

Fable's B3 is that `VisualAgent.predict` ignores `trajectory_selection` and always executes
batch sample 0, making `dpcc-r/c/t` variants mechanically identical. The state stack is where
this feature **actually lives and works**:
- `scripts/eval.py:209-215` computes `trajectory_selection` from the variant name **and passes
  it into `Policy(...)`** (`:214-215`).
- Both `diffuser/sampling/policies.py:65-69` and
  `flow_matcher_v3_ode_selectable/sampling/policies.py:59-65` implement the actual selection —
  `temporal_consistency` and `minimum_projection_cost` branches consuming
  `infos['projection_costs']`.

So per-variant DPCC numbers from the **state-based** eval are meaningful. The visual rewrite
dropped the wiring; the ancestor is sound.

### B5 — N/A: single source of truth, by design

Fable's B5 is a *two-source* problem unique to the visual config: `avoiding-d3il-visual.py`
defines a `constraint_list` of 6 exact obstacles that the eval ignores, while a purpose-built
`visual_avoiding_eval.yaml` is consumed only for one threshold. The state config has **no such
parallel list** — a grep of `config/avoiding-d3il.py` finds no `constraint_list`. The state
eval builds constraints **exclusively** from `projection_eval.yaml`
(`eval.py:58-65, 124-139`): halfspace + bounds + obstacle + dynamics. There is one source and
the eval reads it. The state baseline deliberately reproduces the **DPCC-paper ablation
geometry** — that is the intended experiment, not a dead-config bug. Clean.

### B10 — INVERTED: the "dead" ODE knobs are Gen3v2's whole reason to exist

Fable's B10 notes `VisualFlowMatching.__init__` accepts
`ode_solver_backend_v3/method/rtol/atol/step_size` and **discards them** — the visual FM has
only legacy Euler. In **Gen3v2** these knobs are fully live:
`flow_matcher_v3_ode_selectable/models/diffusion.py`
```python
self.ode_solver_backend_v3 = str(ode_solver_backend_v3)     # :57
self.ode_solver_method_v3   = str(ode_solver_method_v3)      # :58
... rtol/atol/step_size stored                               # :59-61
...
use_torchdiffeq = self.ode_solver_backend_v3 == 'torchdiffeq'   # :190
...
odeint_kwargs = {'method': self.ode_solver_method_v3, ...}      # :221-234
x = torchdiffeq_odeint(..., **odeint_kwargs)                    # :242-246
```
with a guard that errors clearly if `torchdiffeq` is missing (`:191-194`) and correct handling
of fixed-step vs adaptive methods (`:233-237`). **This is the inversion: what is decorative in
the visual FM is the load-bearing feature in Gen3v2, and it is wired correctly.** The
sub-points of B10 (`mpc_batch_size`, the FM-eval pkl-config banner) are visual-eval-specific
config-precedence concerns and do not map onto the Parser-driven state eval.

---

## Net answer to the question

> **Is Fable pointing at problems only for the visual avoiding pipeline, or also for the
> state-based FMv3ODE / DPCC standard avoiding?**

Split cleanly in two:

1. **The eval-side headline findings (B1, B2, B3, B5) are visual-only.** Two are physically
   impossible without images; the other two are *regressions the visual rewrite introduced
   away from the state-based code*, which remains the correct reference. The state-based
   avoiding eval is not affected by any of them.

2. **The training/serialization findings (B6, B7, B8, B9) are shared**, because the visual
   stack inherited `diffuser/utils/training.py` + `serialization.py` nearly verbatim. These
   are real in the state-based DPCC/Gen3v2 stack too: raw-weights-not-EMA at eval, leaky
   window-level validation (and here `state_best` is actively used), the lost final 20% of
   training, and the latent eval-mode leak. B4 is a shared ops concern, externally handled.

3. **B10 is inverted:** the ODE-solver feature that is dead code in the visual FM is exactly
   Gen3v2's purpose, and it is implemented correctly.

### Recommended action order (state-based, all no-retrain except where noted)

> **Read with the provenance section below first.** B6/B7/B8/B9 are inherited *verbatim* from
> the published DPCC code, so the "fixes" here apply to **our new models (Gen3v2 FM, drone)**,
> **not** to the DPCC baseline column — that must stay faithful to `/workspaces/dpcc`. On the
> baseline, treat each item as a *labelled ablation*, never a silent change.

1. **B6 — switch eval to EMA** (`serialization.py:75` → `trainer.ema_model`). Free A/B, likely
   several success-rate points across *all* state baselines; do this before quoting any
   absolute DPCC/Gen3v2 numbers.
2. **B8 — add a terminal `self.save(self.step)`** so the final 20k steps are not silently lost.
3. **B9 — add `self.model.train()`** at the end of `test()` (cheap insurance before anyone adds
   a mode-dependent layer).
4. **B7 — episode-level split** on the *next* retrain; until then treat `diffusion_epoch:'best'`
   as "latest-ish", not as trustworthy model selection.
5. B4 seeds — confirm trained seed set matches `projection_eval.yaml` (already handled manually
   on the remote).

**No re-collection and no architectural change is implied for the state stack** — these are
trainer/serialization hygiene fixes plus one EMA eval flag.

---

## Provenance cross-check against the published DPCC reference (`/workspaces/dpcc`)

**Added 2026-06-13.** The question behind this section: *if the shared findings (B6–B9)
also exist in the original published DPCC code, then they are not regressions we introduced
— and "fixing" them may actually be the wrong move, because that code produced the
published baseline numbers.* I diffed our stack against the upstream repo at
`/workspaces/dpcc`.

### Result — every shared finding is present **verbatim** in the published code

| # | Published DPCC (`/workspaces/dpcc`) | Same as ours? | Evidence |
|---|---|---|---|
| **B6** | `serialization.py` returns raw `trainer.model`; EMA saved, never loaded | **byte-identical file** | `diff` reports `serialization.py` IDENTICAL; `:9-10,75` match exactly |
| **B7** | `train_test_split: 0.9` + `random_split` over stride-1 windows | **identical behaviour** | config `:61`, `training.py:76`, `sequence.py:81` `for start in range(max_start)` |
| **B8** | `save_freq=n//5`; `train()` has **no terminal save** | **identical behaviour** | original `train()` body is just `for epoch: train_epoch(...)` — no final `self.save` |
| **B9** | `test()` calls `.eval()`, **no** `self.model.train()` restore | **identical behaviour** | `grep self.model.train()` → no match in upstream `training.py` |
| **B3** | `eval.py:155-161` computes **and passes** `trajectory_selection`; `policies.py:65-69` implements it | **identical (correct)** | confirms B3 is a visual-only regression, not present upstream |

### Did *we* change any of it?

No. `diff` of `diffuser/utils/training.py` (ours vs upstream) touches **only** resume /
loss-merge / progress-bar additions (hunks around `:180-197` and `:264-344`). A targeted diff
for every bug-relevant line — `save_freq`, `% self.save_freq`, `self.model.eval()`,
`self.model.train()`, `def test`, `def save`, `def train`, terminal `self.save(...)` —
returns **empty**: none of those lines were modified. `serialization.py` (the B6 carrier) is
**byte-identical**. So B6/B7/B8/B9 were inherited unchanged from the published paper; we did
not introduce them and we did not "almost-fix" them.

### What this means — reframing B6/B7/B8/B9

The user's hypothesis is correct, and it changes the disposition of these four findings:

1. **They are not our bugs.** They are properties of the published DPCC implementation. The
   paper's reported results were *produced with* raw-weights eval (B6), the leaky window
   split + `state_best` selection (B7), the 80k-not-100k checkpoint (B8), and the eval-mode
   training tail (B9).

2. **"Matches the paper" ≠ "provably correct" — but it does mean comparability.** For any
   experiment whose purpose is **comparing against the DPCC baseline** (the entire FM-vs-DPCC
   thesis axis), these should be **kept as-is**. Silently "fixing" B6 (switch to EMA) or B7
   (episode-level split) would make our DPCC column no longer reproduce the published numbers,
   and would bias the FM-vs-DPCC comparison in whichever direction the fix happens to help.

3. **The correct way to touch them is a labelled ablation, never a silent change.** If we want
   to know whether EMA eval helps, run it as an explicit `dpcc-ema` variant alongside the
   unmodified baseline — do not overwrite the baseline. Same for an episode-level-split retrain:
   it is a *separate* model, reported as such.

4. **Demote severity in the verdict table accordingly.** B6/B7/B8 were written up above with
   "Fix" recommendations as if they were defects in *our* code. Against the published
   reference they are better read as **"inherited baseline behaviour — change only as a
   declared ablation."** B9 remains a genuine latent footgun (it would bite the moment a
   `BatchNorm`/`nn.Dropout` layer is added), but it is harmless today *and* harmless in the
   paper, so it is not evidence of anything we broke.

### The one place this does **not** restrain us

Our **new** code — the Gen3v2 selectable-ODE Flow Matcher, the drone FM-PCC, anything that is
*not* reproducing a DPCC paper number — has no comparability obligation to the upstream repo.
There, B6/B7/B8 fixes are free to adopt (EMA eval, episode-split, terminal save), and B9
should simply be fixed. The constraint is only on the **DPCC baseline column** that must stay
faithful to `/workspaces/dpcc`.

> **Bottom line of the provenance check:** B6, B7, B8, B9 are upstream-published behaviour,
> inherited byte-for-byte (B6) or line-for-line (B7/B8/B9), not defects we introduced. For
> the DPCC baseline they should be preserved for comparability; deviations belong in clearly
> labelled ablation variants. Only B1/B2/B3/B5 — all **visual-only** — represent actual
> divergence, and those live entirely in the visual rewrite, not in this state-based stack.

---

## Does this mean the published DPCC repo itself has bugs?

**Added 2026-06-13.** Direct answer to the follow-up question. Short version: **mostly no —
and where there is one genuine weakness, it is mild and does not invalidate the paper's
conclusions.** Finding imperfections in research code is normal; it is not the same as
finding errors that change results. Here is the honest per-finding verdict, judged against
**DPCC's actual eval path**, which I traced: `scripts/eval.py:59` loads
`epoch=args.diffusion_epoch`, and `config/avoiding-d3il.py:93` sets
`'diffusion_epoch': 'best'` with `'train_test_split': 0.9`, `'n_train_steps': 1e5`.

| # | Is it a bug *in DPCC*? | Does it affect DPCC's published numbers? | Honest verdict |
|---|---|---|---|
| **B3** | No — wired correctly upstream | — | Not a bug at all |
| **B8** (no final save) | Technically yes (step==100k save never fires) | **No** — DPCC evals `'best'`, not `'latest'`; `'best'` is saved independently throughout training | Inert for the paper; a real but harmless code wart |
| **B9** (eval-mode leak) | Technically a smell | **No** — the net has no `BatchNorm`/`nn.Dropout` module that responds to `.eval()`; gradients identical | Latent footgun, zero practical effect |
| **B6** (raw weights, not EMA) | Not a *bug* — a deliberate-looking choice | **Yes**, but in the **conservative** direction (EMA usually *helps*, so their numbers if anything under-state) | Nonstandard but defensible; not wrong |
| **B7** (leaky split drives `'best'`) | **This is the one real methodological weakness** | **Yes** — the checkpoint behind the numbers was selected by a test loss that ≈ train loss | Genuine but mild; see below |

### The single finding with real teeth — B7, and why it is still mild

DPCC selects the reported model with `diffusion_epoch='best'`, and `state_best.pt` is chosen
by the lowest test loss (`training.py:148-150`). Because the train/test split is over
**stride-1 overlapping windows** (`random_split` of `make_indices`), nearly every held-out
window has near-duplicates in train, so the test loss is a near-copy of the train loss. The
consequence is **not** that the model is bad — it is that "best on validation" is really
"a low-train-loss late checkpoint." In practice `'best'` lands on a converged checkpoint that
behaves like a slightly-cherry-picked `'latest'`.

Why this does **not** invalidate the paper:
- All variants (diffuser, dpcc-r/c/t, …) use the **identical** selection protocol, so the
  FM-vs-DPCC and ablation **comparisons remain fair** — any bias is common-mode.
- The selected model is still a fully-trained network; the leak weakens *model selection*, it
  does not corrupt *training* or *evaluation* of whatever model is chosen.
- The effect size is small: with a converged loss curve, "lowest test loss" and "a late
  checkpoint" pick nearly the same weights.

So B7 is a legitimate "I would not do it this way" methodological note about the published
code — **not** a result-breaking error. It is exactly the kind of thing that is fine for the
paper's relative claims and would only matter if someone leaned hard on the absolute "we do
principled validation-based selection" framing.

### So, to be precise about the claim

I am **not** saying "the DPCC paper is buggy / its results are wrong." I am saying:
- Of the four findings that exist in DPCC, **three (B6, B8, B9) are either deliberate,
  inert on its eval path, or practically zero-effect** — i.e. not defects that touch results.
- **One (B7) is a real but mild methodological weakness** that affects *model selection*, is
  common-mode across all variants, and does not change the paper's qualitative conclusions.
- The eval-side findings that *would* have been serious (B1 channel swap, B2 resolution, B3
  dropped selection, B5 wrong constraint source) **do not exist in DPCC at all** — they were
  introduced later by the **visual rewrite**.

### Why this matters for *us* (the actual takeaway)

This is the reassuring part of the cross-check: the state-based DPCC/Gen3v2 stack we depend on
is **as sound as the published baseline**, because it *is* the published baseline plus our
resume/logging additions. The serious problems Fable found are confined to the visual avoiding
code. For the baseline, "matches DPCC" is the goal, and B7's selection quirk is something we
**inherit on purpose** for comparability — not something to unilaterally "fix."

---

## Caveats on this cross-check

- Line numbers were read on `2026-06-13` against the working tree on branch `update_into_FM`.
  Verify before quoting if the files have since moved.
- I did **not** execute anything (Docker = AI-coding only, no Python runtime); conclusions are
  from source reading and the constraint algebra Fable already re-derived, not from a re-run.
- B9's benign verdict assumes the avoiding UNet contains no `nn.Dropout`/`BatchNorm` beyond the
  manual CFG mask inspected here. That held for both `diffuser` and `flow_matcher_v3_ode_selectable`
  UNets read today; re-confirm if the architecture changes.
- **Repo ≠ paper.** Claims about "DPCC's published numbers" assume the `/workspaces/dpcc`
  working tree is the code that produced the paper's tables. It may be a later/edited commit,
  and the paper may have used different settings (e.g. `'latest'` instead of `'best'`, or a
  different `train_test_split`) than the config defaults read here. The B6/B7 result-impact
  claims are therefore *conditional on the config as checked in*, not a statement about what
  the authors ran for publication. Treat the "affects published numbers" column as "affects a
  run made with this repo's current config," and downgrade confidence accordingly.
- "Mild / non-invalidating" for B7 is a reasoned judgement about effect size on a converged
  loss curve, not a measured one — I did not run a best-vs-latest A/B. If the loss curve were
  non-monotonic or under-trained, the leak could matter more than stated.
