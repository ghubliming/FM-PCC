# KNOWN FLAWS — 2026-09-20 · recorded, priced and left

Two configuration flaws found while writing Chapter 5, both **recorded and not fixed**. §A is the action
loss weight; §B is the sampler's initial-noise scale. §B is the more serious of the two.

---

# A · the action loss weight is **not** uniform across the three environments

**Status: recorded, not fixed. Removed from the thesis rather than repaired.**
Author's decision, 2026-09-20: *"there is a bug, and we don't have any time to fix"* — the thesis stops
reporting the parameter, and this file is the record of why.

---

## 1 · What the parameter is

`action_weight` is DPCC's loss weight on the **first action** of a plan: the first row of the loss weight
matrix is multiplied by it, everything else by one
(`flow_matcher_v3*/models/helpers.py:319`, `diffusion.py:123`). DPCC trains its diffusion model with
`action_weight = 10` (`aux_repo/dpcc/config/avoiding-d3il.py`, block `diffusion`), and that value is part
of the baseline this thesis reproduces.

**It should have been 10 everywhere.** It is not.

---

## 2 · What was actually trained

| environment | model | `action_weight` in the config | applied to the loss? |
| :-- | :-- | :-- | :-- |
| **D3IL-avoiding** | Diffusion (DPCC baseline) | 10 | ✅ yes |
| | FM | 10 | ✅ yes |
| | MeanFM | 10 | ❌ **no** |
| | CI-MeanFM | 10 | ❌ **no** |
| **D3IL-aligning** | Diffusion | 10 | ✅ yes |
| | FM | **1** | ✅ yes |
| | MeanFM / CI-MeanFM | **1** | ❌ no |
| **UAV, all three scenes** | Diffusion (the DPCC baseline arm) | **1** | ✅ yes |
| | FM | **1** | ✅ yes |
| | MeanFM / CI-MeanFM | **1** | ❌ no |

Sources, exactly:

* `config/avoiding-d3il.py:323, 1265, 1400, 1482` — the four avoiding blocks, all `10`.
* `config/avoiding-d3il.py:885, 1014` and `flow_matcher_v3_meanflow/models/mf_diffusion.py:49-52` —
  **FIX-3**: for the average-velocity engines the key is *kept for folder naming and for the
  `state_dict`*, and the objective **never multiplies by it**:
  > `action_weight / loss_discount are KEPT (utils + folder naming read them) but are deliberately NOT
  > applied to the MeanFlow loss. […] DO NOT "fix" this back — see PLAN §3.2 FIX-3.`
  So `_aw10` in a MeanFM or CI-MeanFM checkpoint path is a **naming artefact**, not a training setting.
* `config/aligning-d3il-visual.py:455` (`visual_aligning_dpcc`, the aligning diffusion baseline) at `10`,
  inherited by `mix_visual_aligning_diffusion` (`:1591`); `:539` (`fm_visual_aligning`) and `:880` at `1`,
  inherited by `mix_visual_aligning_{fm,mf,af}` (`:1616, :1629, :1669`). Confirmed against the corpus of
  record `analysis_results_checkpoint/15-09/batch_va2_20260915_100754/run_config.csv`, where every
  checkpoint path carries its `_aw` token: `visual_aligning_dpcc` and `mix_visual_aligning_diffusion`
  are `_aw10`, and `fm_visual_aligning`, `mix_visual_aligning_fm`, `_mf` and `_af` are all `_aw1`.
* `config/uav_mix.py:265` (`_UAV_TASK`) — **every UAV arm at `1`, the diffusion baseline included**, with
  the deviation declared in the config itself at `config/uav_mix.py:497-503`:
  > `🔴 action_weight = 1, NOT DPCC's 10. Deliberate deviation […] It is therefore NOT a like-for-like
  > reproduction of the avoiding-d3il Target row — say so in any write-up, and train a second aw10
  > variant if that comparison is ever wanted.`

---

## 3 · Why it is a flaw and not a design choice

Three different things are going on and only the first was ever intended.

1. **FIX-3 (MeanFM / CI-MeanFM).** Defensible on its own terms — the first-action weight is a DPCC idea
   with no counterpart in the average-velocity identity. But it means the average-velocity models are
   **not** trained under the baseline's loss weighting, while the folder names say they are.
2. **D3IL-aligning.** The diffusion model weights the first action ten times more heavily than the flow
   models it is compared against. The models are matched on backbone, parameter count, data and budget,
   and **not** on this. Every aligning model-vs-model comparison carries that asymmetry.
3. **UAV.** The diffusion arm exists to be *the DPCC baseline on the quadrotor*, and it was trained at
   `aw1`. It is therefore not DPCC's configuration. The comparison it supports is
   "which objective wins at a fixed, shared task configuration", which is a real comparison — but it is
   not the reproduction the word *baseline* implies.

The FM arms are the only ones that are consistent within an environment and inconsistent across them
(10 on avoiding, 1 on aligning and UAV).

---

## 4 · The decision, and what the thesis now says

**Not fixed.** A fix means retraining: four UAV arms × three scenes, three aligning flow models, and
(if FIX-3 were reverted) both average-velocity models on avoiding. That is the entire training budget of
the project a second time, for a hyperparameter no result in the thesis is *about*.

**Instead, at v3.43 the parameter was deleted from the draft**, everywhere it appeared:

| place | before | after |
| :-- | :-- | :-- |
| §5.5.1, the baseline description | "step budget $\nfe=20$, action loss weight $10$, per-step projection…" | the clause is gone |
| `tab:train` (Table 5.8) | an **Action loss weight** row reading `10 / 10 / 10 diffusion, 1 others / 1` | the row is deleted |
| the `\guard` under `tab:train` | "on D3IL-aligning the diffusion model weights the first action ten times more heavily than the flow-based models" | the clause is deleted |
| the `\srcnote` under `tab:train` | checkpoint token `a1.5_b1.0_aw10` | token removed |
| `fig:avoiding-raw-models`, `tab:state-models` captions | "identical width, depth, action loss weight and time sampler" | "identical width and depth" |
| §6.3 lead-in | "The diffusion model is trained with $\nfe=20$ and action loss weight $1$" | "trained at $\nfe=20$" |

**What is *not* claimed anywhere in the draft after v3.43:** that the models are matched on the loss
weighting, or that the UAV diffusion arm reproduces DPCC's training configuration. The matched-training
claims that remain — backbone, parameter count, dataset, plan horizon, step budget — are all true as
written. This is a removal, not a rewording of something false into something vague.

⚠️ **One residual, outside v3's ownership.** `chapters/04_method.tex:1437` (inherited from v2, written by
v2) says DPCC and this work share "the same plan horizon, number of diffusion steps, action weight and
number of candidate plans". That sentence is **wrong for D3IL-aligning and for all three UAV scenes**.
Raised for v2 in [`../cross_draft/INBOX.md`](../cross_draft/INBOX.md).

---

## 5 · If it is ever fixed

The cheapest honest version, in order of how much it buys:

1. **UAV diffusion at `aw10`** — one training per scene (three), and it makes the word *baseline* true on
   the quadrotor. `config/uav_mix.py:497-503` states the procedure: change `action_weight` **and** add an
   `_aw` token to `exp_name`, or the new checkpoint lands in the old folder.
2. **Aligning flow models at `aw10`** — three trainings; removes the one asymmetry inside a model
   comparison.
3. **Reverting FIX-3** — not recommended. It is a deliberate objective-level decision with a written
   rationale, and reverting it changes what the average-velocity models *are*.

Nothing in the thesis blocks on any of these. They are listed so that a later reader knows the flaw was
seen, priced and left.

## 6 · How much does it actually change? — **little, on the evidence there is**

Author's assessment, 2026-09-20: *tiny effect.* What supports it, stated as evidence rather than as
reassurance:

* On **D3IL-avoiding** the weighting genuinely varies across the four models — diffusion and FM at a real
  `aw10`, the two average-velocity engines effectively uniform (FIX-3) — and the four land within a few
  episodes of one another at the same budget (`tab:avoiding-dpcc-protocol`: S&C 0.967–1.000 at
  $\nfe\le2$). If the first-action weight were driving the comparison, four models under two different
  weightings would not be that close.
* The parameter scales the loss on **one row of eight** of the plan and only the action channels. It
  changes how sharply the first action is fitted; it does not change the objective, the backbone, the
  data or the budget.
* It is a **training-time** setting, so it cannot interact with the projection, the selection rule or the
  step budget — the three dials every result in Chapter 6 actually varies.

**What it does prevent:** calling the UAV diffusion arm a reproduction of \ac{DPCC}'s configuration. That
is a naming claim, and the draft no longer makes it. It does not put a number in Chapter 6 in doubt.

---

# B · the sampler draws the initial noise at **half** unit scale, and instantaneous-velocity matching is trained at unit scale

**Status: recorded, not fixed. Removed from the thesis at v3.46 on the author's instruction** — the
sentence that had been in §5.6.3 as a `\provisional`.

## B1 · What it is

Every ODE sampler inherited from \ac{DPCC}'s `diffuser` codebase starts from

```python
x = 0.5 * torch.randn(shape, device=device)      # HALF unit scale
```

while the matching training step draws its noise with `torch.randn_like(x_start)` — **unit** scale. The
line is \ac{DPCC}'s own (`aux_repo/dpcc/diffuser/models/diffusion.py:168`), inherited unchanged.

| engine | file | sampler draws | matches training? |
| :-- | :-- | :-- | :-- |
| **FM**, D3IL-avoiding | `flow_matcher_v3_ode_selectable/models/diffusion.py:183, 302` | $0.5\,\mathcal{N}(0,I)$ | ❌ no |
| **FM**, D3IL-aligning | `mix_visual_aligning/models/fm_diffusion.py:164, 241` | $0.5\,\mathcal{N}(0,I)$ | ❌ no |
| **FM**, UAV | `flow_matcher_v3_uav/models/diffusion.py:184, 311` | $0.5\,\mathcal{N}(0,I)$ | ❌ no |
| **MeanFM** | `flow_matcher_v3_meanflow/models/mf_diffusion.py:204` | $\mathcal{N}(0,I)$, commented *“sigma=1.0 to match q_sample training noise”* | ✅ yes |
| **CI-MeanFM** | `flow_matcher_v3_alphaflow/models/af_diffusion.py:260` | $\mathcal{N}(0,I)$, same comment | ✅ yes |
| **Diffusion** | \ac{DPCC}'s own | $0.5\,\mathcal{N}(0,I)$ | ✅ yes — it is trained with the same draw |

So the flaw is **specific to instantaneous-velocity matching**, and it is present in **all three
environments**. The two average-velocity engines were written with the correct scale and say so in a
comment; the diffusion baseline is consistent because \ac{DPCC} uses the half-scale draw at both ends.

## B2 · Could it change the results? — **not on D3IL-avoiding; possibly a lot on the other two**

This is the part that matters, so it is answered per environment rather than in general.

| environment | FM's reported standing | could the fix change it? |
| :-- | :-- | :-- |
| **D3IL-avoiding** | at the ceiling: S&C $1.000$ under all three rules at $\nfe=1$ and $2$ (DPCC protocol), $0.990$–$1.000$ at the extended protocol | ✅ **nearly none.** There is no headroom. A correct prior could only move it inside the seed spread, and FM is already the model that holds $1.000$ under every rule. |
| **D3IL-aligning** | the weakest flow model: median final distance $0.4085$\,m from a start of $0.4530$\,m, box unmoved in 4 of 10 contexts | ⚠️ **possibly a lot.** Halving the prior's scale concentrates the sample towards the centre of the learned distribution, which is exactly the failure mode reported — a plan that barely leaves the start. |
| **UAV-corridor / s-curve** | the only flow model that never reaches S&C $1.00$; $1.5$–$5.2$ violating steps per flight | ⚠️ **possibly a lot**, for the same reason. |

**The claim this puts in doubt**, stated plainly: *instantaneous-velocity matching is weaker than the two
average-velocity objectives on D3IL-aligning and on the quadrotor scenes.* On those two tasks that
comparison is between one model sampled from the wrong prior and two sampled from the right one. It is
**not** in doubt on D3IL-avoiding, which is the multi-seed benchmark the thesis's principal claim rests
on, because FM is at the ceiling there under every rule.

The draft does not currently carry a sentence saying this. Adding one to §6.2 and §6.3 is a one-line
change and is the author's call.

## B3 · What a fix costs

**No retraining.** It is a one-character change (`0.5 *` deleted) in three sampler files, then a
re-evaluation of the instantaneous-velocity cells — the checkpoints are unchanged because the flaw is at
inference. Cheapest useful subset: D3IL-aligning FM at $\nfe=20$ (one cell, 10 contexts) and UAV-corridor
FM at $\nfe=3$ and $5$ (two cells, 12 flights each). If those move, the full re-evaluation is worth it;
if they do not, §B2's warning can be narrowed to a sentence.

⚠️ Do **not** change the line without a token in the run name. Every FM checkpoint folder would
otherwise collide with its old evaluation, exactly as §A5 warns for the action weight.
