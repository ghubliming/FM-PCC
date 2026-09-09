# Update MASTER_TEST_HISTORY.md From Git History and Development Notes

You are working inside the repository root.

## Objective

1. Update:

   `FM-PCC/logs_in_develop/MASTER_TEST_HISTORY.md`

   This file is a chronological daily development history.

   Your task is to append new history entries starting from the current end of the file. Do NOT rewrite existing history under any circumstances unless the user explicitly requests modifications to existing content. Preserve all existing content.

2. Maintain Daily Claude History Backup:

   Overwrite and update the latest Claude history into:
   `/workspaces/FM-PCC/.claude_history_backup/AUTO_BACKUP`

---

## Sources to Analyze

### 1. Git History

Inspect all commits that occurred after the latest date already recorded in `MASTER_TEST_HISTORY.md`.

For each relevant commit, collect:

* commit hash
* commit message
* commit date/time
* files changed
* summary of actual code changes (from diff, not only commit message)

Commands that may help:

```bash
git log --stat
git log --name-only
git show <commit>
git diff
```

### 2. Development Notes

Read all markdown files under:

```text
FM-PCC/logs_in_develop/
```

including nested folders if present.

Pay special attention to:

* testing notes
* implementation notes
* bug investigation notes
* results analysis notes
* data analysis notes
* architecture notes
* feature planning notes (code-related only)

Use these notes to enrich and explain the development history rather than merely listing commits.

### 3. Exclusions — Writing & Non-Code Notes

**IGNORE** all of the following — they are NOT part of the development history:

* The entire `FM-PCC/logs_in_develop/Writing/` folder and all its subfolders
* Any notes related to thesis writing, paper writing, manuscript drafting, or publication preparation
* Any notes under `Writing_Hints/`, `Working_Space/`, or `Auxiliary/` that deal with document composition
* Files like `thesis_bone.tex`, `thesis_bone_broad.tex`, `NOTES_paper_map.md`, `tum_i6_thesis_submission_reference.md`, `complete-thesis-guide-reference.md`

Only **code changes** and **experimental results analysis** belong in `MASTER_TEST_HISTORY.md`.

---

## Update Requirements

### Determine Current Cutoff

First:

1. Read `MASTER_TEST_HISTORY.md`
2. Identify the latest recorded date/entry
3. Only generate history AFTER that point

Avoid duplicate entries.

---

### Create New Daily Sections

Append new sections in the same style and structure already used in `MASTER_TEST_HISTORY.md`.

---

### Content Quality Rules

Do NOT simply copy commit messages.

Instead:

* group related commits together
* infer the actual feature work
* explain why changes were made
* summarize implementation progress
* mention important refactors
* mention testing activities and their quantitative results
* mention failed approaches if documented in notes
* mention architectural decisions if documented
* include key numerical results (metrics, p-values, speedups, error magnitudes)

Produce a meaningful engineering narrative focused on **code and results**.

Bad:

```markdown
- Commit: fix bug
- Commit: update code
```

Good:

```markdown
- Fixed state synchronization issue in the PCC processing pipeline that caused stale test results after configuration updates.
- Refactored validation logic into reusable modules, reducing duplicated checks across multiple execution paths.
```

---

## Cross-Validation

Before writing:

1. Verify each claimed activity is supported by either:

   * git diff
   * commit history
   * development notes

2. Merge duplicate information appearing in multiple sources.

3. Prefer actual code changes over commit-message wording.

---

## Final Output

Directly modify:

```text
FM-PCC/logs_in_develop/MASTER_TEST_HISTORY.md
```

Append the new sections after the current ending section.

Do not create a new file.

Do not truncate existing content.

Maintain chronological ordering.

---

## Claude History Auto-Backup

Run the following script to overwrite and refresh the Claude history backup (no LLM reasoning required):

```bash
bash /workspaces/FM-PCC/.claude_history_backup/AUTO_BACKUP/auto_backup.sh
```

---

## Cleanup Requirements

After successfully updating the history file and backing up Claude history, you MUST explicitly delete any temporary files, intermediate text dumps, or processing scripts you created during your research and analysis steps to keep the workspace clean.

---

## Last Known State (Cutoff for Next Run)

The following commit is the **most recent one already captured** in `MASTER_TEST_HISTORY.md`.
When running the next auto-update, only process commits **strictly after** this entry.

| Field | Value |
|---|---|
| **Commit hash / Date** | `97eec8a6` / `2026-09-08` |
| **Commit message / Scope** | `Robost Dev Container Setup, Claude Tested, Codex not; Add Codex Dev Toolchain` |
| **Commit date** | 2026-09-08 13:52:20 UTC |
| **What was recorded** | Entries from September 8, 2026:<br>1. **DevContainer Crash, Silent Memory Disconnection & Hardened Environment Recovery** (`INCIDENT_20260908_history_loss_and_memory_rewire.md`, `RUNBOOK_claude_code_env_recovery.md`, `14fe9d03`): Reinstall wiped 37h of sessions (55 JSONL files); diagnosed and fixed silent failure of memory symlink replaced by empty dir; restored symlink to git-tracked `.claude/memory/`; hardened `.devcontainer/devcontainer.json` mounts (`/home/vscode/.claude`, `/home/vscode/.codex`) and `postCreateCommand`.<br>2. **Codex Multi-Agent Integration, Automated Backup & Strict CSV-Only DA Policy** (`AGENTS.md`, `.claude/memory/da-requires-csv-never-from-logs.md`, `97eec8a6`): Onboarded Codex with `AGENTS.md` mirroring Claude rules without duplicate drift; enacted strict policy forbidding DA generation from raw sbatch logs without batch CSVs; established single-line Claude auto-backup script and ignored `/.codex_history_backup/`.<br>3. **Gen15 Five Missions Campaign: Wave Execution Status & Scheduling Forensics** (`MASTER_20260908_five_missions_campaign.md`, `RUNSTATUS_20260908_wave_25486_25514_status.md`, Jobs 25486–25514): Audited 7 wrappers; K=1,2 completed clean; manual scancel on 25500 (fm s_curve K20, 6/8) and 25501 (af pillars K5, 5/17); 25502/25503 running; Mission 5 (25514 mjpc vs pid) blocked by CPU QOS limits; 25542 malformed with newline in `UAV_MIX_VARIANTS`.<br>4. **Gen15 Mission 2 DA: Corridor Saturation, AF vs MF Equivalence & Monotonic Projector Mechanics** (`DA_20260908_T2_af_unet_corridor_K1_K2.md`, Batch `batch_uav_20260908_153947`): Evaluated 1263 units under `u7hg`; S&C saturated at 1.000 across all 20 af cells and 10 mf cells; af ≡ mf at K=2 (5/10 vs 5/10 win rate, mean |Δ| = 0.83 steps, timing within 3%); fm degraded on -t variants (S&C 0.800, 3.6–4.7× step dispersion); K=1→2 budget flat (+92% time, negligible gain); tightening constraints monotonically cuts path length and increases tracking error across all engines (-geo_free delivers ~99% quality at 1/14 cost).<br>5. **Thesis Claim Ladder Audit & Campaign-Wide Baseline Gap Identification**: Proved corridor holds only 1 of 3 rungs (af > mf tie; mf > fm weak; fm > diffusion untestable); uncovered that diffusion baseline under u7hg is missing campaign-wide (only C75/C91 exist under pre-U7 geometry), identifying an urgent need for an s_curve K=20 u7hg diffusion rerun. |
