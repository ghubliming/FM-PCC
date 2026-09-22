# RECAP — tightened against untightened: what the thesis actually reports (v3.66, 2026-09-22)

> **Superseded at v3.67 (same day).** The author decided: **tightened only, untightened withdrawn**. Chapter 5 §5.2.1 now
> states the convention and the reason; Table 6.6 and the s-curve projected cells are blank pending their tightened
> re-runs (R37, R38); the pillars plain columns stay only inside the disabled block. §1–§3 below describe the state
> *before* that decision and are kept for the record; §2's table is still the map of where each set was used.

Written on the author's question: *"How to deal with tightened / non-tightened data? This feels impossible —
add ×2 data for every demo? What is the de facto data in the thesis now? If all tightened, just say so once."*

## 1 · The short answer

**De facto, every projected result that ranks something in Chapter 6 is on the tightened set** (D3IL-avoiding, D3IL-aligning, UAV-corridor). Nothing needs to be run twice. The two quadrotor scenes that rank nothing are the exceptions: UAV-s-curve's projected rows and UAV-pillars' endpoint rows are the *plain* variants, and pillars' per-step table prints both sets (see §2).
The untightened set appears in exactly three kinds of place, each of them on purpose, and none of them a headline:

1. **Unprojected plans** (Table 6.1, Table 6.5, Fig 6.1, Fig 6.6, Table 6.9/6.10 and every "no projection" row).
   The plan does not depend on the set; the set only decides how a violation is *scored*. These rows say "untightened"
   because that is the set the scorer used, not because a second run exists. Nothing to add.
2. **The activation-threshold ladder of D3IL-aligning** (Table 6.6). It is a cost study under random selection on
   the untightened set, and it is read for the cost columns only. Now carries both projectors (v3.66).
3. **The tightening result on UAV-corridor** (§6.3.3, one paragraph): on the v2 corpus the *untightened* per-step
   configuration scored 0.00 at every budget against 1.00 tightened, i.e. the DPCC margin *is* the constraint
   satisfaction there. It is a result about the projector and needs one untightened arm per model at one budget,
   not a full second grid. The corridor-v3 plan keeps it that way (`data_status/PENDING_20260922_corridor_v3_run.md`).

Chapter 5 now states the convention once, in the constraint-set paragraph of §5.2.1 (D3IL-avoiding), in bold:
*"unless a table says otherwise, every projected result of Chapter 6 is on the tightened set"*, followed by the
three exceptions above. Nothing else in Chapter 5 has to change.

## 2 · Section by section — which set each result is on

| where | what | set | note |
| :-- | :-- | :-- | :-- |
| §6.1.1 Table 6.1, Fig 6.1 | models unprojected | untightened (scoring only) | plan independent of the set |
| §6.1.2 Table 6.2 (DPCC protocol), Fig 6.2 paths, Fig 6.3 frontier, Fig 6.4 K-ladder | per-step projection, all rules | **tightened** | the benchmark of record |
| §6.1.2.6 Table 6.3 (new `tab:avoiding-projectors`) | endpoint vs per-step, matched candidates | **tightened** | v3.66; the old single-candidate ladder, which carried an untightened column, is archived |
| appendix `tab:app:hf-ladder-detail` | the archived ladder's numbers | both columns | supplementary only; not cited by a headline |
| §6.2.1 Table 6.5, Fig 6.6 | models unprojected | untightened (scoring only) | — |
| §6.2.2 Table 6.6 | threshold ladder, random rule, both projectors | untightened | cost study; the guard says so |
| §6.2.2 Table 6.7, Table 6.8, Fig 6.7, Fig 6.8 | projected alignment, all rules, both projectors | **tightened** | the diffusion baseline's projected cell exists on the untightened set only → printed in prose, *pending* (R2) in the table |
| §6.3.1 Tables 6.9/6.10 | UAV unprojected | untightened (scoring only) | corridor blank (R33); pillars flawed |
| §6.3.2 corridor Table 6.11 (blank) | per-step + endpoint | **tightened** | the corridor's stack is `-bounds_free-pdes-tightened` |
| §6.3.2 pillars Table 6.12 / 6.13 | per-step, plain **and** tightened; best-of table takes whichever is best | both | the one table that prints both, because on that scene the plans are feasible before projection and the two sets separate the rules; block is flawed/disabled |
| §6.3.3 pillars Table 6.17 | endpoint `hardflow_sls{,-r,-c,-t}` | **untightened** (plain) | flawed/disabled; if the scene is re-evaluated, run the tightened endpoint arm too |
| §6.3.2 s-curve Tables 6.14/6.15 | unprojected; projected `dpcc-t`, `hardflow_sls{,-r,-c,-t}` | **untightened** (plain variants) | all 0.00, ranks nothing; caption now says so (v3.66); no tightened twin owed |
| §6.3.3 corridor (blank) | r/c/t both projectors + the tightening paragraph | tightened, plus one untightened arm | see 1.3 |
| §6.4 (locked) | summaries | tightened | quotes the archived corridor numbers |

## 3 · What the author does NOT need to run

- No untightened twin of any tightened headline. The convention sentence in Chapter 5 covers it.
- No tightened twin of any unprojected row. There is no such thing: an unprojected plan is the same plan.
- The only untightened runs on any ledger are (a) the one-budget-per-model untightened arm inside the corridor-v3
  grid, for the tightening paragraph, and (b) R34, endpoint projection at the uncompressed threshold on the
  alignment ladder, which is optional and only fills the last cell of Table 6.6.

## 4 · Where a reader could still be confused, and what was done

- Table 6.5's caption says "untightened constraint set" for unprojected plans. Kept — it is the scorer's set — but
  Chapter 5's sentence now explains why that word appears under unprojected rows.
- Table 6.12 (pillars) prints "plain" and "tight." columns side by side. Kept inside the flawed block; when the
  scene is re-evaluated the author can decide whether the plain block survives.
- The archived 6.1.2.6 ladder had "Untightened / Tightened" columns whose conclusions disagreed with each other.
  That is one reason it left the chapter.

Claude (Fable 5.1, Claude Code) · 2026-09-22 · read from the draft and the corpora; nothing compiled.
