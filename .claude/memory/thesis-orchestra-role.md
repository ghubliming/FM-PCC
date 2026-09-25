---
name: thesis-orchestra-role
description: "The Orchestra chat: since 2026-09-25 13:43 (v5.0, job O003) it edits the WHOLE thesis in Working_Space/v5 — the aggregate of v2.28 · v3.100b · v4.2 ('Advance Orchestra'); v2/v3/v4 are legacy sources, kept and used less; one ## v5.N per pass in v5/CHANGELOG.md; releases from v5 via v5/tools/make_release_v5.py (folder <stamp>_thesis_release_ORCH_v5.N, date-time = identity, never deleted); legacy changes absorbed with v5/tools/absorb.py; runbooks v5/README.md + Orchestra/README.md; the legacy minor-edit / TODO flow inside v2/v3/v4 is kept for big jobs the author routes there"
metadata:
  node_type: memory
  type: project
  originSessionId: c237bedd-9a5c-4d2e-aa3b-e1623286bf8e
  modified: 2026-09-25T14:40:00.000Z
---

**Since 2026-09-25 (author): the thesis lives in `logs_in_develop/Writing/Working_Space/v5/` and the Orchestra chat
edits it directly** ("the Advance Orchestra"; job O003 = init, O004 = v5.1). v5 was initialised at 13:43 from the live
files of v2.28 (Ch 1–4, abstract, preamble, acronyms, `bibliography.bib`, split from the monolith), v3.100b (Ch 5–6,
preamble_v3, bib_v3) and v4.2 (Ch 7–9 + `app_long/`, preamble_v4, bib_v4) — byte-identical to the golden release's
sources; `v5/inherited/INIT_STATE.json` + `MANIFEST.md` record date-time, revisions and SHA-256, `inherited/materials/`
keeps the copies as the merge base. The author's words: "no more v234, all together … the build is the RELEASED folder …
RELEASE version use majorly datetime … dont delete old built latex … changelog from v5.0 (init with v2,3,4 version) to
v5.1…". The audit answer of O002 is parked ("DONT care the Audit for now").

**Why:** the major parts of the thesis are set, so three owner chats with a one-way sync cost more than they give; one
folder, one changelog, one build path. The legacy flow (minor edits inside v2/v3/v4 with tagged changelog entries +
inbox notes, TODO distribution) is **kept but used less** — only when the author takes a big job to an owner chat.

**How to apply (every Orchestra session):** `cd Working_Space/Orchestra && python3 tools/orchestra.py status` (shows
v2·v3·v4·v5, what v5 carries, INBOX incl. `## → v5`, builds) → `cd ../v5 && python3 tools/absorb.py status` (has a legacy
draft moved? merge before editing the same file) → `new-job "<title>" --kind advance` (or `absorb`) → edit v5's files
directly (the drafts' sourcing rules travel with the chapters; prompt-is-not-thesis-text; naming table; storyline guide)
→ `python3 tools/check.py` + `python3 tools/make_release_v5.py --dry-run` → `bump v5 --job O### --title … --body-file …
--link changelogs/v5.N_<date>_<slug>.md` (v5 = next NUMBER, one revision per pass) + write that MD, signed → release
**only when the author asks**: `python3 tools/make_release_v5.py --job O### --note "…"` (imports RELEASE's tool; writes
`RELEASE/output/<stamp>_thesis_release_ORCH_v5.N/`, prepends a marked entry to RELEASE/CHANGELOG.md) → `close-job` →
`status --write`. A row the Orchestra resolves in v5 is marked `🔀 v5.N` in `cross_draft/INBOX.md`; legacy owners
announce their changes under `## → v5`. Never compile, never commit, never edit v2/v3/v4 from v5. **Tool-editing lesson
(2026-09-25):** never `open(path,'w', newline='\\n')` with an escaped newline — it truncates the file before raising;
patch scripts assert every anchor and write last. Related: [[thesis-release-builds]], [[thesis-draft-ownership]],
[[master-thesis-writing-tum]], [[thesis-prose-style]].
