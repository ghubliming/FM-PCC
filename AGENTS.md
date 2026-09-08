# Codex project instructions

Use the existing Claude Code documentation as the source of project guidance. Read the [memory index](.claude/memory/MEMORY.md) at the start of a task, then open the linked instructions relevant to the work. Consult [CLAUDE.md](CLAUDE.md) for repository background; where it conflicts with a more specific memory instruction, use the memory instruction. Current user instructions take precedence over project guidance.

Keep Claude Code's files unchanged, including `CLAUDE.md` and everything under `.claude/`. Do not modify Claude settings, memory, or symlinks as part of Codex setup. This file maps existing guidance without duplicating it.

## Working rules

- [Local execution and cluster restrictions](.claude/memory/docker-no-python-cluster-only.md)
- [Commit and attribution rules](.claude/memory/no-auto-commit-no-coauthor.md)
- [Code edits and documentation scope](.claude/memory/no-unrequested-code-edits.md)
- [Development logs and repository navigation](.claude/memory/fmpcc-dev-logs-navigation.md)
- [Changelogs after coding tasks](.claude/memory/changelog-after-coding-tasks.md)
- [Slurm entrypoints and batch execution](.claude/memory/slurm-sbatch-is-real-entrypoint.md)
- [Configuration conventions](.claude/memory/config-folder-convention.md)
- [Master history editing restrictions](.claude/memory/dont-self-edit-master-test-history.md)
- [Archived and abandoned code](.claude/memory/archived-codes-is-dead-code.md)
- [URLs and artifact delivery](.claude/memory/no-unrequested-urls-or-artifacts.md)

## Research and analysis guidance

- [Pareto comparisons](.claude/memory/pareto-definition-of-good.md)
- [Architecture-matched comparisons](.claude/memory/architecture-matched-beat-is-the-strong-claim.md)
- [Benchmark hierarchy](.claude/memory/benchmark-hierarchy-who-beats-whom.md)
- [Data analysis baseline target](.claude/memory/da-target-is-best-baseline-variant.md)
- [MeanFlow upstream references](.claude/memory/meanflow-family-upstreams.md)
- [Visual transformer upstream references](.claude/memory/visual-transformer-refs-auxrepo.md)
- [UAV timing interpretation](.claude/memory/uav-budget-ms-not-a-goal.md)
- [HardFlow low-K degeneracy](.claude/memory/hardflow-low-K-degeneracy.md)
- [Master's thesis writing](.claude/memory/master-thesis-writing-tum.md)

## Claude Code chat history

Claude Code chat history is separate from Codex project guidance. Do not load chat history or Claude-specific recovery notes during routine work. Only when the user explicitly asks for chat history, consult the [Claude Code chat history backup](.claude_history_backup/) at `/workspaces/FM-PCC/.claude_history_backup`. Keep the backup unchanged.
