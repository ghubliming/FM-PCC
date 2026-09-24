# v4 — the final piece: Discussion, Conclusion, Appendix, and the assembled thesis

> 🚦 **Draft ownership:** v4 writes **Ch 7 Discussion, Ch 8 Conclusion and the appendix**, plus its own
> preamble additions, `bibliography_v4.bib`, `bundle/` and `tools/` — never its copies of Ch 1–6, nothing
> under `v2/` or `v3/`. Rules: [`../DRAFT_OWNERSHIP.md`](../DRAFT_OWNERSHIP.md).

**Created:** 2026-09-24 (v4.0) · **Master file:** [`thesis_v4.tex`](thesis_v4.tex) · **Change history:** [`CHANGELOG.md`](CHANGELOG.md)
**Branched from:** [`../v3/`](../v3/README.md) at **v3.98** (2026-09-24 20:19, the handover in
[`FROM_v3_v3.98_20260924_201920/`](FROM_v3_v3.98_20260924_201920/README.md)), which carries **v2.27** (2026-09-23) —
see [`inherited/SYNC_STATE.json`](inherited/SYNC_STATE.json), stamped by `tools/sync_v3.py`.
**Storyline every conclusion follows:** [`../GUIDE_20260924_results_storyline_author.md`](../GUIDE_20260924_results_storyline_author.md) ·
insights [`../INSIGHTS_20260924_results_per_environment.md`](../INSIGHTS_20260924_results_per_environment.md)
**Vocabulary:** [`../../Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md`](../../Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md) ·
**style:** [`../../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md`](../../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md)

---

## What v4 is

v2 wrote the argument and the mathematics (Ch 1–4); v3 wrote the experiments (Ch 5–6). **v4 closes the
thesis**: the Conclusion with its answers to the research questions (Ch 7), the Discussion with the
limits, the deployment view and the future work (Ch 8), and the appendix — and it assembles the whole
document, marking the v2 and v3 revisions it is built on. The endings are short by the author's rule
(v4.1): about five pages together, nothing that repeats Chapter 6.

| chapter | owner | state in v4 |
| :-- | :-- | :-- |
| 1 Introduction · 2 Background · 3 Related Work · 4 Method | v2 (via v3) | inherited byte-for-byte; **collapsed in the working bundle** |
| 5 Experimental Setup · 6 Results | v3 | inherited byte-for-byte; built in full |
| 7 Conclusion | **v4** | summary and the answers to the research questions (v4.0 text; Chapter 7 since v4.1) |
| 8 Discussion | **v4** | limitations · towards deployment (incl. the candidate machinery) · future work (v4.1; the v4.0 Interpretation, Negative Results and Threats sections are dropped and archived) |
| Appendix A–C | **v4** | A *Supplementary Material* (sampling laws, quadrotor dimensions) · B *Extended Results*, the five sections the author keeps (long-data switch on the corridor rules) · C *Compute Environment* (v4.1; the dropped sections are archived) |

Every number in Ch 7–8 restates one of Ch 6; a `\dataref` at the end of each part names the tables and
sections it comes from. Nothing in v4 introduces a new result.

## Workflow

```bash
python3 tools/sync_v3.py status      # has v3 (or v2 through it) moved? which files need a merge?
python3 tools/sync_v3.py diff        # what exactly changed upstream
python3 tools/sync_v3.py merge       # three-way-merge it in, advance the baseline, re-stamp
python3 tools/check.py               # labels, citations, environments, braces, figures, nested inputs, the appendix switch
python3 ../../../../Data_Analysis/DA_in_Paper/plotting/export_to_draft.py v4   # copy in the figures v4 uses
python3 bundle/make_bundle.py        # BOTH variants: _new (v2 collapsed, v3+v4 built) and _full_clean
python3 bundle/make_bundle.py --verify   # prove the newest bundle matches the tree, byte for byte
```

**The bundle rule for v4 (author, 2026-09-24): built without v2 but with v3 + v4.** The default annotated
bundle collapses Ch 1–4 to their headings and builds Ch 5–9 in full; `--full` collapses nothing,
`--v4-only` also collapses Ch 5–6, `--appendix-short` hides the long-data tables. Every bundle header and
every collapsed box name the v3 and v2 revisions built on. Details: [`bundle/README.md`](bundle/README.md).

**Never compiled here** — the container has no TeX toolchain. `check.py` and `--verify` are mechanical.

## Layout

```
thesis_v4.tex          master: preamble, \input order, the versions built on
parts/
  00_preamble.tex      v2's, INHERITED through v3 -- never edit
  00_preamble_v3.tex   v3's, INHERITED -- never edit
  00_preamble_v4.tex   v4-only: the appendix long-data switch (\ifappendixfull), \longdata, bib registration
  01_frontmatter.tex   v2's (abstract), INHERITED
  99_backmatter.tex    v2's acronym list, MERGE policy (v4 may declare more; note to v2)
chapters/01..06        INHERITED (v2 through v3; v3)
chapters/07_conclusion.tex, 08_discussion.tex, 09_appendix.tex   v4's
chapters/app_long/     the data record the appendix inputs inside \ifappendixfull (UAV-corridor rules)
chapters/app_ntrial20_feasible.tex   the twenty-episode section, used as is (author, v3.92)
bibliography.bib / bibliography_v3.bib   INHERITED · bibliography_v4.bib  v4's own (empty at v4.0)
inherited/
  v3_base/             the merge base: v3 as it stood at the last sync. Never edit by hand.
  SYNC_STATE.json      which v3 (and, through it, which v2) that is
  MANIFEST.md          per-file policy and owner
tools/                 sync_v3.py (status / diff / merge / stamp), check.py
bundle/                make_bundle.py, README.md, BUNDLE_LOG.md; output/ is gitignored build output
figures/               COPIES exported from DA_in_Paper (EXPORTED.md lists them)
notes/                 writing-only material: the appendix parts moved out of the thesis, the
                       abbreviation audit, the open items
withheld/              text dropped from the thesis, kept for the record (the twenty-episode dead block;
                       the v4.0 Interpretation / Negative Results / Threats sections; four appendix sections)
FROM_v3_v3.98_20260924_201920/   the handover copy from v3 -- untouched
notes.txt, REMINDER_20260914_glossary_appendix.md   the author's and v2's notes -- untouched
```

## State (v4.1a)

- Synced to **v3.99**, which carries **v2.27**; v2.28 reaches v4 once v3 has merged it.
- `tools/check.py`: 17 files, 7628 lines, 276 labels, 53 citations, 41 figures; **all mechanical checks
  pass**. Drafting marks: 1 `\hole` (v4's: the web link of the long-data section), 1 `\provisional`
  and 19 `\guard` (v3's plus two of v4's), 2 `\flawed` (v3's), 1 `\longdata` (v4's), 0 `\outdated`.
- Open items the author decides: [`notes/OPEN_20260924_v4_open_items.md`](notes/OPEN_20260924_v4_open_items.md).
- Cross-draft: [`../cross_draft/INBOX.md`](../cross_draft/INBOX.md).
