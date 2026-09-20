# TO v4 — Chapter 8 is in v3's build again, and its text changed

**2026-09-20 · from v3 (v3.49) · supersedes the standing parts of
[`FROM_v3_20260918_chapter8_handover.md`](FROM_v3_20260918_chapter8_handover.md).**

## What changed

1. **`\input{chapters/08_conclusion}` is uncommented in `thesis_v3.tex`.** It was commented out on
   2026-09-18. The author's instruction on 2026-09-20 is that the thesis must not be missing a chapter
   while v4 has not started. **Ownership did not change**: `DRAFT_OWNERSHIP.md` still says v3 keeps a
   concise version that follows Chapter 6 and v4 refines the prose and writes `sec:conc:future`.
   `inherited/MANIFEST.md` now reads **own** rather than *own, not built*.

2. **The chapter was stale and is now consistent with Chapter 6.** Building it exposed claims Chapter 6
   had withdrawn:
   - **Every UAV-pillars sentence is gone.** The scene has been withheld from Results since v3.41
     (§6.3 says "no number from the scene is used anywhere"), and the conclusion was still ranking
     models on it in the summary, RQ1 and RQ2.
   - The summary and the three RQ answers were rewritten against §6.4 as it now stands: three
     environments yield a result, UAV-s-curve yields none, endpoint projection is selected on
     D3IL-aligning alone.
   - **"wall-clock time" is gone** (v3.48 banned it; the thesis name is *the time to compute one
     action*, and *time per control step* on an axis). This closes the → v4 INBOX row for it.

3. `sec:conc:future` is untouched — still a `\hole` pointing at `future_work/`. It is yours.

## What v4 should know

- The two `\hole{Refine in v4.}` markers are still there, one after the summary and one after the RQ
  answers. They mark prose to lift, not facts to re-derive.
- Every number in the chapter is a restatement of a Chapter 6 number. If Chapter 6 moves, this chapter
  moves with it; it introduces nothing of its own.
- `tools/check.py` now counts 14 files; all of Chapter 8's cross-references resolve.

Changelog: `v3/changelogs/v3.49_20260920_chapter8_figures_and_controller_cost.md`.
