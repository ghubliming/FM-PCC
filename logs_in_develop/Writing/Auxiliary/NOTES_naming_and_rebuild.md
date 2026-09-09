# Naming — what the thesis calls things, and why the code's names are not it

**Thesis home:** every chapter. Enforced hardest in `sec:method:backbone`, `sec:setup:baselines`
and every results table.
**Authority:** the code (line references below).
**Where the question was first asked (NOT a source):** [`logs_in_develop/Rebuild_repo/`](../../Rebuild_repo/CONCEPT_unified_rebuild.md) §5, §8 — unbuilt, unstable, see the box.

> 🔴 **`Rebuild_repo/` IS NOT A RELIABLE SOURCE. It is a scratch document for a rebuild that is
> still being built, and it changes without notice.** Its own header says *"CONCEPT / IDEAS — not a
> finalized plan"*. Nothing in it is implemented, verified, or agreed.
>
> **Therefore:**
> - **Never cite it in the thesis.** Not in a footnote, not in `app:repro`, not as evidence for
>   anything.
> - **Never treat a rename it lists as done.** The code is the authority; that document is a wish.
> - **Never copy its file tree or its registry sketch into `sec:method:*`.** That architecture does
>   not exist.
> - **Do not point a thesis chapter at it.** Chapters must not depend on a file that moves.
>
> **What it is good for is one thing only:** it is the single place in the repo where somebody sat
> down and asked *what are these things actually called, and is the name true?* That question is
> worth importing; its answers are not evidence. So every claim below has been **re-checked against
> the code**, and the code line is given. This file is the stable writing-side rule; `Rebuild_repo/`
> is only where the question was first asked. When the rebuild moves, re-read it, re-check against
> the code, and update **this** file.

---

## 1. Why this matters for the writing, not just for the code

An implementation identifier that has hardened into a label can smuggle a claim into a paper. The
worked example is below and it is not hypothetical — it was live in the v2 draft until 2026-09-09.

**The rule the thesis follows: name things by mechanism, not by flag value.** If the reader cannot
recover what the code does from the name, the name is wrong for a thesis even if it is fine for a
`--flag`.

Second-order consequence: **the run tags and folder names in every log, checkpoint path and DA table
still use the old names.** The thesis therefore has to carry a small translation table
(`app:repro`), or a reader trying to match a table row to a checkpoint will fail. Do not silently
rename in prose and leave the artefacts unmapped.

---

## 2. 🔴 The worked example — `film_mode='v1'` is not FiLM

| | flag | class in code | what it actually computes | thesis name |
|---|---|---|---|---|
| default, **every reported visual number** | `film_mode='v1'` | `UNet1DTemporalCondModel` | visual latent projected, **concatenated** with the time embedding; one **additive per-channel bias**, constant along the horizon | **concatenated conditioning** |
| opt-in ablation | `film_mode='v2'` | `UNet1DTemporalFiLMModel` | latent → per-channel **scale γ and shift β**, `h ← (1+γ)⊙h + β`, zero-initialised | **affine (FiLM) conditioning** |

Verified in the code, 2026-09-09: `mix_visual_aligning/models/visual_unet.py:64-97` (the branch);
`models/unet1d_temporal_cond.py:53-82` (the additive bias) and `:205-235` (the concatenation);
`models/unet1d_temporal_film.py:38-92` (the γ/β head, zero-initialised).

FiLM is *by definition* the affine map with a learned multiplicative term. The default has
`γ ≡ 0`, so it is not FiLM. But the flag is called `film_mode`, so the word "FiLM" appears in every
config block, every checkpoint tag (`filmv1`) and every eval log for the arm that **is not FiLM**.

The code itself is honest about it — `visual_unet.py` comments the two branches as *"Fake FiLM:
additive bias via time-embed concat"* and *"True FiLM: per-block γ scale + β shift"*. The hazard is
that the comment lives in the source and the misleading token lives in the artefacts.

**Handled in the thesis by:** `sec:method:backbone` writing both mechanisms out as equations, and
`Remark` *"The default is not FiLM, and the code's name for it says otherwise"* stating the
discrepancy explicitly. Tables report the arm by mechanism.

---

## 3. The rest of the naming audit, as it bears on the writing

The question comes from `CONCEPT_unified_rebuild.md` §5.3–§5.6; the left column below was checked
against the code and the configs, and only rows that survived that check are kept. **Left column =
what you will see in logs, tags and configs. Right column = what the thesis says.** The right column
is *this file's* decision, not the rebuild document's — that document proposes new *code* names,
which is a separate matter the thesis does not depend on.

| in the artefacts | in the thesis | note |
| :-- | :-- | :-- |
| `VisualUNet` | vision-conditioned temporal U-Net | rebuild proposes the class name `VisionTrajectoryUNet` |
| `VisualUNetTwoTime` | its two-time variant | `VisionTrajectoryUNetTwoTime` |
| `film_mode='v1'` / `filmv1` | concatenated conditioning | see §2 |
| `film_mode='v2'` | affine (FiLM) conditioning | see §2 |
| `diffuser`, `diffusion` | the diffusion engine / the baseline | rebuild proposes the key `ddpm`; **`diffuser` is also the name of the ancestor codebase and of Janner et al.'s method** — three meanings, one token. Never write bare "diffuser" in the thesis |
| `dpcc`, `dpcc-r/c/t` | the projection arm, and its three selection rules | rebuild proposes `pcc-*`. In the thesis the rules are named (random / temporal consistency / cumulative projection cost), never lettered |
| `fm`, `mf`, `af` | flow matching, average-velocity (MeanFlow), curriculum (α-Flow) | fine as table headers **once defined**; never as prose |

## 4. Open, and owned by the author

- **The framing question** (raised as Q1 in the rebuild scratch doc) — concatenated conditioning as
  the method with affine as an ablation, or both presented as equals? This is a **thesis framing
  decision, not a code decision**, and it is the author's, not the rebuild's. The v2 draft takes the first
  option (concatenated = the method, affine = an ablation) because it is what every reported number
  used. If that changes, `sec:method:backbone` and every visual table change with it.
- **The translation table for `app:repro`** does not exist yet. It is §3 of this file plus the
  checkpoint-tag spellings, and it has to be built from the actual run ledger, not from this note.
- **Does the thesis mention the rebuild at all?** **No.** It is unbuilt and unstable; a thesis
  cannot rest on it. If it ever ships, one sentence in `sec:conc:future` is the ceiling — and that
  sentence would describe what was built, not what this document proposed.
