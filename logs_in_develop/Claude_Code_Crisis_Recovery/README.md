# Claude_Code_Crisis_Recovery

Tooling-side incidents — Claude Code's own environment (history, memory, symlinks), not the research
code. Opened after the 2026-09-08 reinstall wiped all chat history.

| file | what it is | when to read it |
|---|---|---|
| [`INCIDENT_20260908_history_loss_and_memory_rewire.md`](INCIDENT_20260908_history_loss_and_memory_rewire.md) | the 2026-09-08 incident: what broke, what was fine, what was rewired, what is still open | to understand what happened, or before changing the backup/symlink setup |
| [`RUNBOOK_claude_code_env_recovery.md`](RUNBOOK_claude_code_env_recovery.md) | **do-this-next-time playbook** — 8 numbered steps with commands | the moment history vanishes, memory looks wrong, or the container is rebuilt |

**One-line summary of 2026-09-08:** a reinstall deleted every transcript (losing all of 2026-09-07,
~37 h beyond the last backup) and silently replaced the memory symlink with an empty directory. No
repo work was lost — it was all committed — and memory survived only because `.claude/memory/` is in
git. The backups turned out never to have contained memory at all.

**The single most important check after any rebuild:**
```bash
ls -la ~/.claude/projects/-workspaces-FM-PCC/memory   # must be a symlink into /workspaces/FM-PCC/.claude/memory
```

**Related, but deliberately separate:** `/workspaces/FM-PCC/.claude/CRISIS_RECOVERY_2026-09-07_lost_history.md`
is the *gated* recovery dossier for the lost conversations themselves — it is opened **only** when
the user explicitly asks to resume from the crisis. The files in this folder are the opposite: an
ordinary, always-readable engineering record.
