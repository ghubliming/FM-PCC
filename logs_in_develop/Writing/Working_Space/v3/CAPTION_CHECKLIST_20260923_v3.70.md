# Caption checklist — v3 Ch 5/6, captions over 100 words

**Created:** 2026-09-23 (v3.70), on the author's item 7: *"Sanity check if such long description for thesis is bad practice … write a new MD … as a checklist, and fix them one by one."*

## The rule applied

- **Consensus** (thesis and journal guides alike): a caption makes the figure or table readable on its own — what it is, how to read each element, the numbers the drawing cannot show, a pointer to the claim. Interpretation, protocol and argument belong in the running text. The TUM template's own example captions are one sentence. The house rule already says the same: `Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md` §"What a caption is allowed to contain".
- **Threshold used here:** 100 words (about seven lines of caption text in the template). Length itself is not the fault; every caption over the threshold in this draft was over it because it restated the protocol of Table 5.10, explained a number, or carried drafting talk. Captions between 80 and 100 words were read and left (legend material: colours, markers, column definitions).
- **Where the cut text went:** to the paragraph beside the float when the text did not already say it; deleted when it did, or when it was drafting talk ("retained", "withdrawn at v3.67", "for the reason set out before the table", "so the comparison can be read in one place", "explained in the text").

## Checklist

| # | label | float | before | after | status | moved to the text / cut |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| 1 | `fig:env-uav` | Fig 5.7 | 105 | 96 | ✅ | arena size cut (Table 5.4 has it) |
| 2 | `fig:constraints-uav` | Fig 5.8 | 190 | 95 | ✅ | "the mapped scale already carries the vehicle's extent, so no rotor-reach band … tightening 0.90 m" → paragraph before the figure; per-panel plane description cut (paragraph had it) |
| 3 | `tab:avoiding-raw-models` | Table 6.1 | 133 | 87 | ✅ | "two of the three rules are defined on the projection" cut; budgets pointer kept |
| 4 | `fig:raw-plans` | Fig 6.1 | 170 | 98 | ✅ | bold header cut; "only the first waypoint … was carried out" is in the text |
| 5 | `tab:avoiding-dpcc-protocol` | Table 6.2 | 153 | 82 | ✅ | K2-time explanation cut (§6.1.2.1 text + caveat subsection have it); "retained" cut |
| 6 | `fig:avoiding-tradeoff` | Fig 6.2 | 271 | 94 | ✅ | budget list per model cut (§6.1.2.1 text); "both rules at each budget, frontier over all of them, ringed point = better rule" → paragraph before the figure; the `%` K2 note moved after `\end{figure}` |
| 7 | `tab:va-models` | Table 6.5 | 160 | 93 | ✅ | thirty-context restriction cut (§6.2.2 text and §5.6.3.2 have it); "K = 20 is the budget at which all four have one" cut |
| 8 | `fig:aligning-tradeoff` | Fig 6.3 | 166 | 92 | ✅ | reading rules paragraph in the text already; "84 % and 85 % are two points" cut |
| 9 | `tab:va-threshold` | Table 6.6 | 122 | 65 | ✅ | "K = 100 pair is the same model at two thresholds", "withdrawn at v3.67", "chapter's projected results at K = 20, η = 0.2" cut (text has all three) |
| 10 | `tab:va-projection` | Table 6.7 | 104 | 72 | ✅ | "so more is better", "no endpoint row, for the reason set out before the table" cut |
| 11 | `tab:va-projection-models` | Table 6.8 | 114 | 66 | ✅ | "so the comparison can be read in one place" cut |
| 12 | `fig:aligning-projected-tradeoff` | Fig 6.4 | 192 | 89 | ✅ | "counterpart of success with constraint satisfaction", "cannot be on the frontier whatever their distance", "one selection rule is drawn … other two in the table" cut (text has them) |
| 13 | `tab:uav-scurve` | Table 6.13 | 109 | 67 | ✅ | "no flight is collision-free so S&C is 0.00" (text has it); "rows are not a model comparison" → §6.3.3 text |
| 14 | `fig:uav-scurve-paths` | Fig 6.9 | 102 | 51 | ✅ | the guard's step range is in the text; "one flight of the left panel reached the finish line's x but made contact" → paragraph after the figure |
| 15 | `tab:uav-controller-cost` | Table 6.15 | 128 | 66 | ✅ | "the 119 ms between them is the controller" cut (text prices it) |

Float numbers are as of v3.70 and shift if floats are added. Word counts by `tools`-style split of the caption body (`\caption[…]{…}`), macros included.

## Left under the threshold, read and kept (80–100 words)

`fig:platforms` 93, `tab:train` 90, `tab:target` 80, `tab:avoiding-projectors` 98, `tab:uav-pillars-raw` 98, `tab:uav-corridor-raw` 94, `tab:uav-corridor` 86, `tab:uav-corridor-projection` 81 — legend and column definitions; nothing to move.

## Check

`python3.14 tools/check.py` passes after the pass. Not compiled — no TeX toolchain here.

## Addendum v3.71 (2026-09-23) — the s-curve rebuild

New or rewritten captions, all under the threshold: `tab:uav-scurve` 75, `tab:uav-scurve-projection` 47 (new),
`fig:uav-scurve-paths` 50, `tab:uav-controller` 62, `tab:uav-controller-cost` 66, `fig:expert-uav` 72.

## Addendum v3.72 (2026-09-23) — UAV-pillars landed

`tab:uav-pillars-raw` 92 (was 98 with the pending note), `tab:uav-pillars-geo` 32 (new), `fig:uav-pillars-paths` 63 (was a `\todofigure` placeholder).

## Addendum v3.73 (2026-09-23)

`tab:eval` 41 words (gained the UAV-pillars sentence; was 17).

## Addendum v3.75 (2026-09-23) — the s-curve grid

`tab:uav-scurve` 80, `tab:uav-scurve-projection` 51, `tab:uav-controller` 71, `tab:app:uav-scurve-unprojected-extra` 37 — all under the threshold.

## Addendum v3.76 (2026-09-23) — aligning cells filled

`tab:va-threshold` 75, `tab:va-projection-models` 77, `fig:aligning-projected-tradeoff` 92 (was 103 after the "what is drawn" clause; the arrow clause dropped — Fig 6.3's caption explains it).

## Addendum v3.77 (2026-09-23)

`fig:aligning-projected-tradeoff` 79 words (all seven cells drawn; "diffusion" in the colour list).

## Addendum v3.79 (2026-09-24)

`tab:uav-pillars-raw` 84, `tab:uav-pillars-geo` 32, `fig:uav-pillars-paths` 69 words.

## Addendum v3.80 (2026-09-24) — corridor filled, two trims

`tab:uav-corridor-raw` 87, `tab:uav-corridor` 96 (now defines the best rule), `tab:uav-corridor-projection` 79,
`tab:uav-corridor-altitude` 69 (new), `fig:uav-corridor-side` 89 (new); `tab:va-models` 84; trimmed:
`tab:avoiding-projectors` 102 → 96 (one rule sentence merged), `fig:aligning-tradeoff` 105 → 99 (rings/staircase clause
shortened). `tab:uav-scurve` 80, `tab:uav-scurve-projection` 52, `tab:uav-controller` 74.

## Addendum v3.81 (2026-09-24)

`tab:uav-pillars-raw` 87 (two rows per configuration), `tab:va-projection-models` 91 (the diffusion sentence now gives
the reason).

## Addendum v3.82 (2026-09-24)

`tab:uav-corridor` 100 (rewritten: goal-point counts added, wording shortened), `tab:uav-corridor-projection` 96.

## Addendum v3.83 (2026-09-24) — corridor rebuilt

`tab:uav-corridor-raw` 78, `tab:uav-corridor` 89, `fig:uav-corridor-tradeoff` 97 (new), appendix
`tab:app:uav-corridor-tilt-ps` 82, `-tilt-ep` 63, `-hump-ps` 95, `-hump-ep` 63. `tab:uav-corridor-projection` removed.

## Addendum v3.84 (2026-09-24)

`fig:env-uav` 95 (finish lines; trimmed from 115), `tab:uav-corridor-raw` 90, `tab:uav-corridor` 83,
`fig:uav-corridor-tradeoff` 79 (after projection only), `tab:uav-scurve` 92, `tab:uav-scurve-projection` 52,
`tab:uav-controller` 74; `fig:uav-scurve-paths` gains an `\outdated` clause.

## Addendum v3.85 (2026-09-24)

`fig:uav-corridor-tradeoff` 99 (2×2, groups I–III), `fig:uav-pillars-paths` 69 (back in Ch 6).

## Addendum v3.88 (2026-09-24)

`tab:uav-corridor-altitude` archived (author). The §6.4 tables were rebuilt: `tab:summary-models` 40,
`tab:summary-projection` 36, `tab:summary-combinations` 11.
