---
name: thesis-orchestra-role
description: "The \"Orchestra\" chat (since 2026-09-25) — a separate role for MINOR and CROSS-LINKED thesis changes across v2/v3/v4 made directly in the drafts' folders, TODO distribution to the owner chats, and RELEASE builds only on the author's word; runbook Working_Space/Orchestra/README.md, tool tools/orchestra.py, its own CHANGELOG with date-time + v2/v3/v4 revisions"
metadata:
  node_type: memory
  type: project
  originSessionId: c237bedd-9a5c-4d2e-aa3b-e1623286bf8e
  modified: 2026-09-25T12:21:39.560Z
---

Since 2026-09-25 the author runs an **Orchestra** chat beside the v2 / v3 / v4 owner chats
(`logs_in_develop/Writing/Working_Space/Orchestra/`, job O001 = init). Its job, in the author's words:
manage minor changes (or cross-linked changes between chapters of different drafts) **directly inside
`v2/`, `v3/`, `v4/`**, update their CHANGELOG (and rebuild the v3 / v4 bundles), and — because those drafts
are normally maintained by their own agent — **leave a message in `cross_draft/to_<draft>/` + an INBOX row for
every draft changed**; run a RELEASE (tagged `ORCH_O###`, `--note "Orchestra O###: …"`) **only when the
author asks**; when there is no direct edit, distribute heavy jobs as **separate TODO lists per draft**
(master in `Orchestra/todo/`, delivered as cross notes); keep `Orchestra/CHANGELOG.md` with one row per job
(date-time, v2 · v3 · v4 revisions after the job). Big changes go to the owner chats directly.

**Why:** three owner chats edit fixed chapters ([[thesis-draft-ownership]]); a fourth hand editing without
a trace would confuse them, so every Orchestra edit is a tagged revision entry in the owner's changelog plus
an inbox note, and the Orchestra never touches READMEs, state files, inherited copies or figures.

**How to apply:** when the author addresses the Orchestra (or asks for a small cross-draft fix, a TODO
split, or a release "by the Orchestra"), read `Orchestra/README.md` first and follow its runbook:
`python3 tools/orchestra.py status` → `new-job` → edit only owned files → check.py + bundle (v2: release
`--dry-run`) → `bump <draft>` (v3/v4 next letter, **v2 next number** because `sync_v2.py` reads no letter)
→ `note --to <draft>` → `close-job` → `status --write`. Never compile, never commit, never mark its own
INBOX rows ✅. Related: [[thesis-release-builds]], [[master-thesis-writing-tum]], [[thesis-prose-style]].
