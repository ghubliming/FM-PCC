# Orchestra — minor and cross-linked changes across v2 / v3 / v4, TODO distribution, releases on request

**Created:** 2026-09-25 (job O001) · **Authority:** the author · **Record:** [`CHANGELOG.md`](CHANGELOG.md) (one row per job, date-time + the v2 / v3 / v4 revisions) and [`jobs/`](jobs/) (one file per job) · **Snapshot:** [`STATE.md`](STATE.md) · **Tool:** [`tools/orchestra.py`](tools/orchestra.py)

> **Start of every Orchestra session:** `python3 tools/orchestra.py status`, then `cd ../v5 && python3 tools/absorb.py status`,
> then read `../cross_draft/INBOX.md` (`## → v5` first) and `../DRAFT_OWNERSHIP.md`. Nothing under `Orchestra/` is thesis text.

---

## ⚡ Advance Orchestra — since 2026-09-25 (job O003): the thesis lives in `../v5`

**The author (2026-09-25):** the major parts of the thesis are set, so the split v2 / v3 / v4 flow described below is
**kept but used less**. The whole thesis is now **one draft, [`../v5`](../v5/README.md)** — the aggregate of v2.28 ·
v3.100b · v4.2, initialised 2026-09-25 13:43 (v5.0) — and the Orchestra **edits it directly**: no owner chat to notify,
no inherited copy, no sync tool, no bundle. **The release is the build:** `cd ../v5 && python3 tools/make_release_v5.py`
writes `../RELEASE/output/<stamp>_thesis_release_ORCH_v5.N[_TAG]/`, marked as the Orchestra's; **the date-time is a
build's identity**, the v5 revision is information; no build is ever deleted. **Every update of v5 is a `## v5.N` entry**
in `../v5/CHANGELOG.md` (v5.0 = init with the v2 / v3 / v4 revisions, v5.1 = the inboxes resolved, …) with a detailed
MD in `../v5/changelogs/`. The runbook of the Advance flow is **[`../v5/README.md`](../v5/README.md)**; every job,
Advance or legacy, is still a row in [`CHANGELOG.md`](CHANGELOG.md) — kind `advance` = an edit in v5, `absorb` = a change
made in a legacy draft merged into v5 (`../v5/tools/absorb.py`).

**What "kept" means for the legacy flow:** when the author takes a big job to a v2 / v3 / v4 chat, the owner edits its
own files as before and adds a row under `## → v5` in `../cross_draft/INBOX.md`; the Orchestra absorbs it into v5
(`absorb.py status / diff / merge` against `../v5/inherited/materials/`) and records a new v5.N. The rules and the
procedure below still govern an Orchestra edit *inside* a legacy draft (kinds `edit`, `todo`, `sync`); they do not
apply to v5, which the Orchestra owns. Rows that were open on 2026-09-25 were resolved in v5.1 and are marked `🔀 v5.1`.

---

## What the Orchestra is (author, 2026-09-25)

v2, v3 and v4 are each maintained by **their own chat** (the owner chats; the map is
[`../DRAFT_OWNERSHIP.md`](../DRAFT_OWNERSHIP.md)). The Orchestra is a **separate chat** that:

1. **makes minor changes, or changes that are cross-linked between chapters of different drafts, directly inside
   the drafts' own folders** (`../v2`, `../v3`, `../v4`), updates the draft's `CHANGELOG.md`, and rebuilds the
   bundle for v3 / v4;
2. **leaves a message in the cross-draft inbox for every draft it changed** (`../cross_draft/to_<draft>/` + a row in
   `../cross_draft/INBOX.md`), so the owner chat is not confused by a change it did not make;
3. **runs a RELEASE only when the author asks**, marked as run by the Orchestra (`../RELEASE`);
4. when there is no direct edit, **distributes heavy jobs to v2 / v3 / v4 as separate TODO lists** (one per draft,
   delivered through the same inbox);
5. keeps **its own changelog** with the date-time of every job and the current v2 / v3 / v4 revisions.

**Big changes are not the Orchestra's:** the author takes them to the owner chat directly. If a request turns out
big while working (a section rewritten, a new result, a restructuring, a figure), the job becomes a TODO
distribution (rule 8), not an edit.

## What it is not — in the legacy flow (for v5 the Orchestra *is* the owner; see above)

- **Not a fourth draft.** It owns no chapter, writes no prose of its own, holds no thesis text.
- **Not the merger.** The whole thesis is assembled by `../RELEASE/tools/make_release.py` from the owners' live files.
- **Not a syncer by default.** v3 syncs from v2 and v4 from v3 *only when the author asks* (ownership rule); the
  Orchestra runs a sync only as a job the author asked for (kind `sync`), never as a side effect.
- **Not a committer.** Nothing is committed; the author commits.

## Where things live

| path | what | who reads it |
| :-- | :-- | :-- |
| `Orchestra/README.md` | this runbook | the Orchestra chat, the author |
| `Orchestra/CHANGELOG.md` | one row per job: opened → closed, kind, **v2 · v3 · v4 after the job**, what, notes left, release | the author |
| `Orchestra/STATE.md` | the last snapshot (`status --write`): revisions, sync chain, open INBOX rows, last release | anyone; regenerated, never edited |
| `Orchestra/jobs/O###_<stamp>_<slug>.md` | the full record of a job (asked · scope check · done · checks · messages · release · closed) | the author, the owner chats through the links |
| `Orchestra/todo/TODO_<stamp>_O###_<slug>.md` | a distributed job: the master list with one section per draft and the cross links | the Orchestra; the owners get their section as a cross note |
| `Orchestra/templates/` | the three templates the tool fills (job, cross note, TODO master) | the tool |
| `../cross_draft/to_<draft>/FROM_Orchestra_<date>_O###_<topic>.md` + `../cross_draft/INBOX.md` | **the only channel to an owner chat** | the owner chats, at their session start |
| `../<draft>/CHANGELOG.md` | where an Orchestra edit is recorded *inside* the draft, as a revision entry tagged `(Orchestra O###)` | the owner chat, the sync tools, the release tool |
| `../RELEASE/` | the release tool, its changelog, every build kept under `output/` (legacy builds and the `_ORCH_v5.N` builds from v5) | the author |
| `../v5/` | **the thesis** since 2026-09-25: `CHANGELOG.md` (`## v5.N`), `changelogs/`, `tools/check.py`, `tools/make_release_v5.py`, `tools/absorb.py`, `inherited/` (the init record and the merge base) | the Orchestra chat, the author |

## The rules (binding on the Orchestra chat)

1. **Ownership first.** Before touching a file, the scope check of the job template: the file is in the owner's
   *owns* column of `../DRAFT_OWNERSHIP.md` — v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05_setup.tex`,
   `06_results.tex`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07_conclusion.tex`,
   `08_discussion.tex`, `09_appendix.tex`, `chapters/app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`.
   **Never** an inherited copy (v3's Ch 1–4, v4's Ch 1–6, `inherited/`, inherited `parts/` and `.bib`): that change is
   made upstream and reaches the copies through the owners' sync tools.
2. **Smallest footprint inside a draft:** the owned file(s) and one entry in its `CHANGELOG.md`. The Orchestra never
   edits a draft's `README.md`, `CROSS_STATE.json`, `inherited/SYNC_STATE.json`, `inherited/MANIFEST.md`, `bundle/README.md`,
   `notes/`, `withheld/`, `audit from chatgpt/` — the owner refreshes those at its next pass (the cross note says so).
3. **The revision entry** goes into the draft's `CHANGELOG.md`, newest first, in the draft's own format, written by
   `python3 tools/orchestra.py bump <draft> --job O### --title "…" --body-file …`:
   - **v3 and v4: the next letter** of the current revision (v3.100b → v3.100c; v4.2 → v4.2a) — the pattern the
     owners use for small passes; `v4/tools/sync_v3.py` and the release tool read the letter.
   - **v2: the next number** (v2.28 → v2.29) — `v3/tools/sync_v2.py` reads **no** letter suffix, so a `v2.28a` would be
     invisible to v3's sync while the file had moved.
   - Heading: `## v3.100c — 2026-09-25 · <what> (Orchestra O###) → [link to the job file]`. Body: **Changed** (file,
     lines / labels) · **Why** (the author's words) · **Checked** (check.py, bundle, dry-run; *not compiled*) · **Left
     to the owner** · **Signed** (Orchestra, model, job, date). The owner's rules for its changelog hold (v3: every
     number names its evidence of record; v4: every number restates Chapter 6 with a `\dataref`).
4. **Checks and bundles, every time.** v3 / v4: `python3 tools/check.py` and `python3 bundle/make_bundle.py` (+ `--verify`)
   in the draft folder — the author's bundle rules hold (v3's bundle ends after Ch 6; v4's collapses v2 and builds v3 + v4).
   v2 has no bundle and no check tool (v2 delivers only the `.tex`): the check is `python3 ../RELEASE/tools/make_release.py --dry-run`,
   which assembles the whole thesis in memory and reports labels, citations, acronyms, environments, braces, holes and bugs.
   **Nothing is ever compiled here** (no TeX toolchain); "checked" never means "compiled".
5. **One message per draft changed** — `python3 tools/orchestra.py note --to <draft> --job O### --topic <slug> --item "…" --body-file …`
   writes `../cross_draft/to_<draft>/FROM_Orchestra_<date>_O###_<topic>.md` and inserts the INBOX row
   `| ⏳ | Orchestra O### · <date> | **<item>** | [note] |` newest-first under `## → <draft>`. The note lists every
   file and line touched, the revision entry written, and what the owner still has to do (usually: read, refresh its
   README / state files, mark the row ✅ with its revision). A draft that was *not* edited but is affected (a label it
   references, a term it also uses) gets an FYI note too. The owner closes the row; the Orchestra never marks its own
   rows ✅.
6. **A cross-linked change is listed in full before it is made:** every place the label / term / number / claim
   appears, in which draft and file. Places inside the job's scope are edited (each in its owner's live file, each
   with its own entry and note); places outside it become TODO items (rule 8). A renamed or removed `\label` is
   always called out by name in every note: it breaks references in the other drafts and in the release.
7. **Release only on the author's word.** Advance flow: `cd ../v5 && python3 tools/make_release_v5.py --dry-run`, then
   `python3 tools/make_release_v5.py --job O### --note "Orchestra O###: <why>"` (folder `…_thesis_release_ORCH_v5.N`,
   `RELEASE/CHANGELOG.md` entry marked Orchestra / v5). Legacy flow, from `../RELEASE`:
   ```bash
   python3 tools/make_release.py --dry-run                                             # first: holes, bugs, the page estimate
   python3 tools/make_release.py --tag ORCH_O### --note "Orchestra O###: <why this build>"   # the build, marked as the Orchestra's
   ```
   The `--tag` puts the job into the build's folder name, the `--note` line lands in `RELEASE/CHANGELOG.md` and in
   the build's notes MD; the Orchestra's own row links the build. Read `../RELEASE/README.md` before the first build
   of a session: every build is kept, outputs are never edited, a content problem goes to the owner (rule 5), then
   rebuild. Report the page estimate against the 60–200 limit (and the 60–80 orientation), the HOLES and the BUGs.
8. **TODO distribution** (no direct edit): `python3 tools/orchestra.py new-todo "<title>" --job O### --to v2,v3,v4`
   makes the master file in `todo/` (one `## → <draft>` section each, plus the cross-link table); the Orchestra fills
   it, then delivers each section as a cross note with `note --from-todo <master> --to <draft> …` (one note and one
   INBOX row per draft). Items name file / label / what / why / evidence and say which item in another draft they must
   agree with. The owner closes its INBOX row; the Orchestra mirrors the closure in the master's status table.
9. **What the text must obey when the Orchestra writes a sentence:** the binding style rule
   [`../../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md`](../../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md)
   (the request is never restated, justified or claimed as done in the thesis), the naming table
   `../../Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md`, the author's storyline guide
   `../GUIDE_20260924_results_storyline_author.md`, and the draft's own README. Figures are never drawn or edited in a
   draft (`Data_Analysis/DA_in_Paper/` only). `../../Template_DONT_CHANGE/` is read-only. `../RELEASE/output/` is never edited.
10. **Every job is a row.** `new-job` opens it, `close-job` closes it with the revisions *after* the job; the job file
    is filled while working, not afterwards from memory. The report to the author names: revisions before → after,
    files and lines, checks, notes left, release (if any), and what was left out and why.

## The procedure (runbook)

```bash
cd logs_in_develop/Writing/Working_Space/Orchestra
python3 tools/orchestra.py status                        # 1. where are we; then read ../cross_draft/INBOX.md and ../DRAFT_OWNERSHIP.md
python3 tools/orchestra.py new-job "<title>" --kind edit  # 2. O### opened: jobs/O###_<stamp>_<slug>.md + an open CHANGELOG row
#                                                          3. scope check in the job file (rule 1); big → --kind todo (rule 8)
#                                                          4. the edit, in the owner's live file only; record file/lines in "Done"
#                                                          5. checks + bundle (rule 4); results into "Checks"
python3 tools/orchestra.py bump v3 --job O### --title "<what>" --body-file <body.md>     # 6. the draft's CHANGELOG entry (rule 3)
python3 tools/orchestra.py note --to v3 --job O### --topic <slug> --item "<inbox text>" --body-file <note.md>   # 7. the message (rule 5)
python3 tools/orchestra.py close-job O### --summary "<one line>" --notes v3              # 8. the row closed with the revisions after
python3 tools/orchestra.py status --write                                              # 9. STATE.md refreshed
```

Body files for `bump` and `note` are written to the session scratchpad (they are copied into the draft's
CHANGELOG and the cross note; nothing else keeps them). Variants:

| kind | what differs |
| :-- | :-- |
| `edit` | the sequence above; one `bump` + one `note` **per draft changed**, FYI notes for affected drafts |
| `todo` | steps 4–7 replaced by `new-todo` → fill the master → `note --from-todo` per draft |
| `release` | only on the author's word: steps 4–7 replaced by rule 7; `close-job … --release <build folder>` |
| `sync` | only on the author's word: `cd ../v3 && python3 tools/sync_v2.py status / diff / merge` (or `../v4 … sync_v3.py`); read the upstream `For v3:` / `For v4:` lines first; the owner's changelog gets a `bump` entry saying what was absorbed and which revision the state file now carries; note to the owner |
| `check` | a report only (status, dry-run, a question answered); no file of a draft touched; still a row |

## Conventions

- **Job id** `O###` (three digits, sequential; the tool picks the next), file `jobs/O###_<YYYYMMDD_HHMM>_<slug>.md`.
- **INBOX row** `| ⏳ | Orchestra O### · YYYY-MM-DD | **<item>** | [`FROM_Orchestra_…md`](to_<draft>/FROM_Orchestra_…md) |`,
  newest first; FYI rows say so in the item text, as the owners do (`⏳ (FYI — nothing asked of v3 now)`).
- **Signature** on every entry and note: `Orchestra (Claude <model>, Claude Code), O### · <date>` — an AI-written
  pass says so by name and model, as the drafts' changelog rules require.
- **Release tag** `ORCH_O###`; release note `Orchestra O###: <why>`.
- **Dates** absolute (`2026-09-25`), never "today"; times in the repo's local clock as `tools/orchestra.py` prints them.

## State at init (2026-09-25 12:19 — the live snapshot is `STATE.md`)

- Revisions: **v2.28** (2026-09-24) · **v3.100b** (2026-09-25) · **v4.2** (2026-09-25).
- Sync chain lags twice, both waiting for the author's word: v3 carries v2.27 (v2.28 is ready, INBOX → v3); v4 carries
  v3.100 / v2.27 (v3.100b archived v3's Ch 7–9 copies; INBOX → v4 asks v4 to retire four `sync_v3.py` rows).
- Open INBOX rows: → v2 none · → v3 two (one FYI) · → v4 two.
- Last release: `RELEASE/output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE` (estimate ~179 pages,
  inside 60–200; the compiled predecessor had 185). The release dry-run reports one standing BUG for v2: `\getDoctype`
  carries `in \getDegree`, which the template's pages append themselves; the tool patches it in the release copy.
- No draft file was touched by the init; the three owner chats got an FYI note (O001) describing this protocol.
