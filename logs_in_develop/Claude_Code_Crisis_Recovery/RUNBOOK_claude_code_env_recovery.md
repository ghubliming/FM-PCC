# Runbook — Claude Code was reinstalled / the container was rebuilt / memory looks wrong

**Use this when:** Claude "forgot" the project rules, chat history vanished, or the devcontainer was
rebuilt. Written after the [2026-09-08 incident](INCIDENT_20260908_history_loss_and_memory_rewire.md).
Work top to bottom — **§1 first, always.** It is 10 seconds and it is the failure that hides.

---

## §1 — Check the memory symlink (do this FIRST)

```bash
ls -la ~/.claude/projects/-workspaces-FM-PCC/memory
```

| what you see | meaning |
|---|---|
| `lrwxrwxrwx … memory -> /workspaces/FM-PCC/.claude/memory` | ✅ fine, go to §2 |
| `drwxr-xr-x … memory` (a real directory) | 🔴 **broken** — Claude is running with no project memory and will not tell you |

**Repair (from `CLAUDE.md`):**
```bash
rm -rf ~/.claude/projects/-workspaces-FM-PCC/memory
ln -s /workspaces/FM-PCC/.claude/memory ~/.claude/projects/-workspaces-FM-PCC/memory
ls -A ~/.claude/projects/-workspaces-FM-PCC/memory | wc -l   # expect 21+ (20 memories + MEMORY.md)
```

⚠️ **Never "restore" memory from `.claude_history_backup/`.** Those `memory` entries are symlinks,
not snapshots, and the newest one is an empty directory. **Git is the source of truth for memory:**
```bash
cd /workspaces/FM-PCC && git status --short .claude/memory   # should be clean or show only intended edits
git checkout .claude/memory                                  # only if something was actually clobbered
```

## §2 — Snapshot whatever history still exists, before anything else touches it

```bash
bash /workspaces/FM-PCC/.claude_history_backup/HOW_TO_BACKUP
```
Do this **even if it looks empty** — a snapshot of the broken state is evidence, and it costs nothing.
Then check what you actually have:
```bash
ls -la /workspaces/FM-PCC/.claude_history_backup/
ls /workspaces/FM-PCC/.claude_history_backup/<newest>/*.jsonl | wc -l
```

## §3 — Measure the damage: how much time was lost?

```bash
# newest transcript surviving in the last GOOD backup
cd /workspaces/FM-PCC/.claude_history_backup/<last-good-backup>
for f in *.jsonl; do echo "$(jq -rc 'select(.timestamp).timestamp' "$f" | tail -1)  $f"; done | sort | tail -5
```
The newest timestamp there vs. now = **the lost window**. Everything inside it exists only as
artefacts (git, `temp/`), never as conversation.

## §4 — Confirm what is *fine*, so you don't re-fix healthy things

```bash
cd /workspaces/FM-PCC
git status                 # uncommitted work still on disk?
git log --oneline -10      # the work itself is almost always intact
ls temp/                   # user-provided result drops are untouched
cat CLAUDE.md | head -30   # project rules intact
```
Historically: **the repo work survives; only the conversation dies.**

## §5 — Reconstruct the lost window from artefacts

Transcripts are gone, but the lost chats *wrote things*. In priority order:

```bash
# 1. what was committed during the gap
git log --since="<gap start>" --pretty=format:"%h %ad %s" --date=format:"%m-%d %H:%M" --stat

# 2. which MDs were CREATED in the gap — these carry the lost reasoning
git log --since="<gap start>" --diff-filter=A --name-only \
        --pretty=format:"@@ %h %ad %s" --date=format:"%m-%d %H:%M" -- '*.md'

# 3. which results actually came back (filename match, NOT content grep —
#    a bare 5-digit id matches numbers inside CSV cells and gives false positives)
find temp/ -name "*<slurm_id>*"
```

## §6 — Attribute surviving transcripts to topics (which chat was doing what)

```bash
cd /workspaces/FM-PCC/.claude_history_backup/<last-good-backup>
# time span + first/last user message per session
for f in *.jsonl; do
  echo "## ${f%.jsonl}  ($(du -h "$f" | cut -f1))"
  jq -rc 'select(.type=="user") | [.timestamp,
    ((.message.content // "") | if type=="array"
      then (map(select(.type=="text").text)|join(" ")) else . end)] | @tsv' "$f" 2>/dev/null \
    | grep -vP '\t\s*$' | grep -v '<command' | sed -n '1p;$p'
done
```
Then bridge **chat ↔ MD ↔ topic** by grepping each transcript for each new MD basename; the session
with the most mentions wrote it. **Zero hits across all sessions = written after the backup closed**
— that is how you prove where the gap starts.

## §7 — Write the recovery assets

1. A **gated dossier** under `.claude/` (git-tracked, so it survives the next rebuild).
2. A **gated memory** file whose body says *open only when the user explicitly asks*, plus one line
   in `MEMORY.md`. Gate it — otherwise stale run state leaks into every future session.
3. An incident record + any runbook updates in **this folder**.

## §8 — Ask the user only what cannot be reconstructed

Everything derivable from git / `temp/` / the MDs, derive. Ask only for: results that were never
delivered, and decisions that were made verbally inside the lost window.

---

## Prevention checklist

- [ ] Snapshot history **before** any container rebuild, reinstall, or Claude Code update — `HOW_TO_BACKUP`.
- [ ] After **every** rebuild, run §1 before doing any work.
- [ ] Remember `.claude_history_backup/` is gitignored (638 MB) and **dies with the container** —
      copy it somewhere else if the history matters.
- [ ] Anything that must outlive a rebuild belongs in **git**, i.e. under `.claude/` or
      `logs_in_develop/` — not in `~/.claude/`.

## Known-good reference state (2026-09-08)

| item | value |
|---|---|
| memory files | 20 memories + `MEMORY.md` in `/workspaces/FM-PCC/.claude/memory` |
| symlink | `~/.claude/projects/-workspaces-FM-PCC/memory -> /workspaces/FM-PCC/.claude/memory` |
| last good history backup | `.claude_history_backup/2026-09-06_22-20/` — 55 sessions |
| known permanent gap | 2026-09-06 22:20 → 2026-09-08 11:42 |
