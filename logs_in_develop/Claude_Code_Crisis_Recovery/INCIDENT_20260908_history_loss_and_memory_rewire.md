# Incident 2026-09-08 — Claude Code reinstall: all chat history lost, memory silently unwired

**Status:** closed (recovery assets in place) · **Data loss:** conversations only, no repo work lost
**Companion:** [`RUNBOOK_claude_code_env_recovery.md`](RUNBOOK_claude_code_env_recovery.md) — do this next time
**Gated dossier (only on explicit user request):** `/workspaces/FM-PCC/.claude/CRISIS_RECOVERY_2026-09-07_lost_history.md`

---

## 0. TL;DR

🔴 Claude Code was reinstalled on **2026-09-08 ~11:42**. All 55 session transcripts were wiped, and
the **entire working day of 2026-09-07 is unrecoverable** — the last backup predates it by ~37 h.

🔴 A second, **silent** failure: the reinstall replaced the memory **symlink** with an *empty real
directory*, so Claude started the next session with **zero memories** while everything looked normal.

🟢 **Nothing of the actual work was lost.** All Sep-7 output was committed (3 commits), the user's
result drops are in `temp/0609/II/`, and the memory *content* survived — **because it is git-tracked**.

🟡 And the backups were never protecting memory in the first place (§3.3).

---

## 1. Timeline

| when | what |
|---|---|
| 2026-09-06 22:20 | last history backup → `.claude_history_backup/2026-09-06_22-20/` (55 sessions) |
| 2026-09-07 all day | **normal, productive work** in ≥3 parallel chats → commits `9bda071c`, `d6e2a857`, `fd594d8d` |
| 2026-09-08 ~11:42 | **Claude Code reinstalled** → `~/.claude/projects/-workspaces-FM-PCC/` recreated empty |
| 2026-09-08 11:43 | user snapshots the post-wipe state → `2026-09-08_11-43/` (contains nothing useful) |
| 2026-09-08 12:20 | user opens a session and asks for a self-diagnosis |
| 2026-09-08 12:23 | **memory symlink repaired**; 20 memories visible again |
| 2026-09-08 ~12:30 | recovery dossier + gated memory written |

**The gap: 2026-09-06 22:20 → 2026-09-08 11:42 ≈ 37 h of conversation, permanently gone.**

---

## 2. What broke

### 2.1 Transcripts — total loss, expected
`~/.claude/projects/-workspaces-FM-PCC/*.jsonl` were removed by the reinstall. They live **only**
there and in the manual `.claude_history_backup/` snapshots. No snapshot exists for Sep 7.

### 2.2 The memory symlink — total loss, **silent and dangerous**
Per `CLAUDE.md`, memory is supposed to be:

```
~/.claude/projects/-workspaces-FM-PCC/memory  ->  /workspaces/FM-PCC/.claude/memory
```

After the reinstall it was an **empty real directory** instead:

```bash
$ [ -L ~/.claude/projects/-workspaces-FM-PCC/memory ] && echo symlink || echo "REAL DIR"
REAL DIR
$ ls -A ~/.claude/projects/-workspaces-FM-PCC/memory | wc -l
0
```

This is the dangerous one: **there is no error message.** The harness reads an empty directory and
Claude simply behaves as if no project memory ever existed — no "no auto-commit" rule, no
"Python-only-on-cluster" rule, no benchmark hierarchy, no thesis context. It fails *open*, not loud.
Repaired by re-creating the link exactly as `CLAUDE.md` prescribes.

### 2.3 The backup script never backed up memory *(new finding)*
`.claude_history_backup/HOW_TO_BACKUP` does `cp -r "$SRC"/. "$DST/"`. GNU `cp -r` copies a symlink
**as a symlink**, so every snapshot's `memory` entry is just a pointer at the live repo folder:

```
2026-09-04_15-04 : memory -> /workspaces/FM-PCC/.claude/memory
2026-09-05_12-47 : memory -> /workspaces/FM-PCC/.claude/memory
2026-09-06_22-20 : memory -> /workspaces/FM-PCC/.claude/memory
2026-09-08_11-43 : memory   (real dir, EMPTY — it faithfully captured the broken state)
```

So the archive holds **zero historical memory snapshots**, and the newest one would "restore"
emptiness. Memory survived this incident purely because `.claude/memory/` is committed to git.

---

## 3. What is fine — do not re-fix these

| ✅ | evidence |
|---|---|
| **Memory content** — 19 memories + `MEMORY.md` | all git-tracked; `diff` against the pre-crisis state is clean |
| **`CLAUDE.md`** and its rebuild instructions | intact, and its symlink command was correct — it just had not been re-run |
| **All Sep-7 work** | 3 commits on `update_into_FM`; `git log` is the reliable record |
| **Cluster-side runs** | untouched — Slurm and the remote repo know nothing about this |
| **User result drops** | `temp/0609/II/` (Sep-7 sbatch logs 25486–25500 + batch CSVs) intact |
| **Pre-Sep-6 transcripts** | 55 sessions in `2026-09-06_22-20/`, readable with `jq` |
| **`.claude/settings.local.json`** | intact |

---

## 4. What was rewired / added

| # | action | artefact |
|---|---|---|
| 1 | Restored the memory symlink | `~/.claude/projects/-workspaces-FM-PCC/memory -> /workspaces/FM-PCC/.claude/memory` |
| 2 | Wrote the **gated recovery dossier** — incident, damage table, the 2 key chats, a chat↔MD↔topic bridge table, the user's verbatim mission note, resume procedure | `.claude/CRISIS_RECOVERY_2026-09-07_lost_history.md` (299 lines) |
| 3 | Added a **gated memory** pointing at it — opens *only* on an explicit "resume from the Claude Code crisis" | `.claude/memory/crisis-recovery-lost-chat-2026-09-07.md` |
| 4 | Indexed it | one line in `.claude/memory/MEMORY.md` |
| 5 | This incident record + runbook | `logs_in_develop/Claude_Code_Crisis_Recovery/` |

### Why the dossier is *gated*
It is a large, one-off, historical document. Loaded by default it would pollute every future session
with dead context and stale run state. The memory entry therefore carries an explicit instruction to
open it **only** when the user says *"resume from the Claude Code crisis"* / *"continue the lost
chat"*. Normal work must never touch it.

### How the lost day was reconstructed without transcripts
1. `git log --since` → the 3 commits and every file they added.
2. `git log --diff-filter=A -- '*.md'` → the MDs created that day; those MDs *are* the lost chats'
   conclusions, written by them.
3. grep each surviving transcript for each MD basename → attribute Sep-4→6 MDs to sessions by
   mention count; **every Sep-7 MD scores 0 everywhere**, which both proves the backup boundary and
   forces topic-based inference for that day.
4. `temp/0609/II/` filenames → which Slurm jobs actually returned results.

---

## 5. Still open

- **Slurm 25514** — "mjpc solver vs old pid solver on the worst failing `s_curve`, raw NN + PCC /
  HF-SLSQP, fewer trials". Submitted on Sep 7; **no logs delivered** (`find temp/ -name "*25514*"`
  → nothing). This is the only unfinished thread of the lost UAV chat. Its enabling fix — the mjpc
  env-detection bug + `UAV_MIX_CONTROLLER` — landed the same day
  (`logs_in_develop/Gen15/U10/CHANGELOG_20260907_mjpc_controller_override.md`).
- **`MASTER_TEST_HISTORY.md` is not updated** for this incident (per the standing rule that the
  master index is never self-edited). Offered, not done.

---

## 6. Root causes

1. **Backup cadence is manual and irregular.** Snapshots exist for 08-14, 09-04, 09-05, 09-06 — a
   reinstall between snapshots loses everything since the last one, and there is no warning.
2. **The transcript store is outside git and gitignored.** `.claude_history_backup/` is 638 MB
   (`.gitignore:37`), so it is machine-local and a container rebuild removes it too — the backup
   shares the failure domain of the thing it backs up.
3. **The memory link is an install-time artefact, not a repo artefact.** Anything that recreates
   `~/.claude/projects/<slug>/` silently breaks it, and the failure is invisible.
4. **`cp -r` on a symlink** meant the memory backups were pointers, never snapshots (§2.3).

---

## 7. Recommendations (user's call — nothing here was done automatically)

| priority | recommendation |
|---|---|
| **high** | Make the memory-symlink check the *first* thing after any rebuild/reinstall — §1 of the runbook. |
| **high** | Back up history **more often than the work is valuable**: a snapshot costs seconds; the gap here was 37 h. |
| medium | Fix `HOW_TO_BACKUP` to dereference (`cp -rL`) *or* drop `memory` from it and rely on git — but say which, so nobody trusts a phantom backup. |
| medium | Keep the snapshots off this machine (external drive / remote), since a rebuild deletes them. |
| low | Consider a `.devcontainer` `postCreateCommand` that re-creates the symlink automatically, so it cannot be forgotten. |

> The `.devcontainer/devcontainer.json` is where a `postCreateCommand` would go. Not edited — that
> is a code/config change and needs an explicit go-ahead.
