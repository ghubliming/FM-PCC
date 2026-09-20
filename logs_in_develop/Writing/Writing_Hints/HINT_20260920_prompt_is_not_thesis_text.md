# The prompt is not thesis text

**Created:** 2026-09-20 · **Author's instruction** (v3.51, extended v3.53) · **Applies to:** every
draft — v2, v3, v4 — and to every AI agent writing into them, **Claude Code in particular.**
**Status:** standing rule. Binding on every caption, table description and paragraph.

---

## Two alerts, in the author's own words

| | what it is | how he calls it |
| :-- | :-- | :-- |
| 🔴 **RED** | the **prompt** is in the thesis — the instruction restated, justified, or claimed as done | *"this is my prompt, why it is inside the thesis? this is self talking and prompt rephasing!"* · *"RED ALERT of AI SLOP! in description"* |
| 🟡 **YELLOW** | **self-reasoning** — a *because* the writer invented to explain a measurement, or an argument pre-empting a reader who never objected | *"yellow alert! it is self reasoning! AI yellow alert slop!"* |

Both are the same failure seen twice: **the writer is in the sentence.** Take the writer out and what
is left is the thesis.

## The rule

> **What the author asked for goes into the text as a *fact about the thesis object*, never as a
> restatement of the request, never as a justification for having done it, and never as a claim that
> it was done.**

A reader of the thesis has not seen the prompt, the changelog, or the session. Every sentence must be
checkable by that reader against the figure, the table or the data in front of them. A sentence that
is only true *because of how the work was requested* does not belong in the thesis.

## 🔴 RED ALERT — the three shapes the prompt takes

### 1. Prompt rephrasing

The author says *"mark the violating lines red, others green, all light opaque"*, and the caption comes
back carrying the instruction:

> ❌ *"... both translucent so that the excluded region stays readable underneath."*

The translucency is visible. Saying it, and saying what it is for, is the prompt talking. What the
reader needs is the **rule that assigns a colour**, because that is what they cannot see:

> ✅ *"A demonstration is drawn red where it crosses the geometry of its panel and green where it
> satisfies it at every recorded step."*

### 2. The justification tail — *"which is why ..."*

> ❌ *"That leaves 0, 1 and 2 green lines of the 96, which is why satisfying the constraints can only be
> the projection's doing."*

The count is a fact and it belongs (or better: it is in the table already, and the caption points
there). The clause after the comma is the writer arguing with the reader about what the figure proves.
Claims are made in the running text, once, where they can be qualified. A caption states what is drawn.

> ✅ *"The number of green lines in each panel is the last column of Table 5.2."*

### 3. The done-claim, and emphasis nobody asked for

> ❌ *"Constraints of the quadrotor scenes, in the two forms D3IL-avoiding also uses."*

*"also uses"* is the agent showing the author that the cross-task consistency it was asked for was
delivered. The table is a table of constraint values:

> ✅ *"The constraint set of each quadrotor scene: halfspaces and circular keep-out regions."*

Same family: *"as requested"*, *"note that"*, *"importantly"*, *"it is worth emphasising"*, *"this
clearly shows"*, *"now correctly"*, and any sentence whose subject is the thesis rather than the work
(*"this section now covers ..."*).

## 🟡 YELLOW ALERT — self-reasoning

A measurement does not come with its own explanation. When a sentence states a number and then a
*because*, ask which of the two the evaluation produced. If only the number, the *because* is the
writer reasoning aloud, and it belongs in the thesis only where the thesis can show the mechanism.

> ❌ *"Cutting the budget from twenty to two does not recover the gap, **because what the budget buys
> back sits in the projector rather than in the denoiser**."*

Nothing measured the split between denoiser and projector at that budget. The clause reads as an
explanation and is a guess, and a reader who knows the field will spot that faster than anyone.

> ✅ *"Cutting the budget from twenty to two leaves the baseline at thirteen times the cost."*

The same family, all yellow:

- **Defending a choice nobody challenged** — *"which is why it is drawn beside the small-budget rows
  rather than left out of them"*, *"so that a fast configuration cannot appear as a favourable trade"*
  where the criterion has already been stated.
- **Explaining away a weak result** instead of reporting it — *"the difference is small, but this is
  expected because ..."*. Report the difference and its size. If it needs a caveat, that is what
  `\guard` is for, and a guard states a **limit of the evidence**, never a rescue of the claim.
- **Telling the reader what to conclude** — *"which is the argument for the projector rather than
  against the model"*, *"what this shows is that ..."*. Show it; the conclusion section concludes.
- **Narrating the writing** — *"three things have to be read into that matrix"*, *"one worked example
  is enough"*.

A *because* is allowed when the thesis owns the mechanism: it is defined in Chapter 4, or measured and
reported in the sentence next to it. *"The projector runs on $\lceil \eta K \rceil$ sampling steps,
one at the flow-based budget and ten at the baseline's"* is a because the protocol carries. *"What the
budget buys back sits in the projector"* is not.

## The tone the author wants

> *"you just fact truth telling, and story telling, you are the boss, others not known is they stupid,
> this is the right scientific/academic tone I prefer, not the self explosion suicide way!"*

Write from the position of the person who ran the experiments, because that is who is writing.

- **State it.** The measurement is the authority. *"No episode of the eighty touches the excluded
  region."* Not *"it appears that"*, *"we would argue"*, *"this suggests, although"*.
- **Tell it as a story, not as a defence.** What was run, what came out, what follows. A paragraph
  that spends half its length anticipating objections has changed genre.
- **Do not argue with a reader who is not there.** Anticipated objections are *self explosion*: they
  raise a doubt the reader did not have and then answer it weakly. If a limit is real, state it once,
  flatly, in a `\guard` — a limit stated plainly is strength; a limit argued around is not.
- **Do not apologise for the work.** *"only one training seed"*, *"unfortunately"*, *"we could not"* →
  *"one training seed"*, and the number.
- **Never explain the obvious to the reader.** They are an examiner in this field.

This is not licence to overclaim. Authority means the sentence says exactly what the data supports and
stops — no hedging, and nothing added.

## What a caption is allowed to contain

1. **What the object is** — one noun phrase.
2. **What each visual element means** — colour, line style, marker, shading, axis, panel.
3. **The numbers a reader cannot recover from the drawing** — counts, units, the geometry drawn, the
   model and budget a panel belongs to, where the data came from.
4. **A pointer** to the table or section that carries the claim.

Nothing else. In particular: no argument, no comparison verdict, no rationale for a design choice, no
account of what changed since the last version. Those live in the running text, the `\srcnote`, or the
changelog — three different places, none of them the caption.

## Related standing rules

- **No text inside the plot that is not thesis vocabulary.** No seed numbers, no batch or job ids, no
  `unprojected` / `aw10` / `thres0.5`, no in-plot titles or subtitles restating the protocol — that
  belongs in the caption. Table: `Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md` §6.
- **A panel heading may identify its panel** (*"MeanFM, K = 1"*, *"top-right-hard"*) and nothing more.
- **No p-values, no significance tests** anywhere in the thesis; counts only.
- Prose style, brand names, storytelling over justification: `../Auxiliary/README.md` and the
  `thesis-prose-style` memory.

## Check before writing any caption, table description or result paragraph

- [ ] 🔴 Would this sentence still be written if the author had never said anything? If not, cut it.
- [ ] 🟡 Is there a *because*, a *which is why*, a *so that*? Did the evaluation produce it, or did I?
- [ ] 🟡 Am I answering an objection nobody made, or telling the reader what to conclude? Cut it.
- [ ] Does any clause explain *why* something was drawn or chosen that way? Move it or cut it.
- [ ] Does it claim, praise or reassure? Cut it.
- [ ] Does it hedge or apologise where a plain number would do? Cut the hedge, keep the number.
- [ ] Can a reader who has only this page verify every number in it, or is it pointed somewhere?
- [ ] Does it tell the reader how to read every colour, style and panel in the figure?
