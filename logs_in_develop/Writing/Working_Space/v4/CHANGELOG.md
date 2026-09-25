# CHANGELOG — `Working_Space/v4`

Every change to the v4 draft, its bundle or its inheritance machinery gets an entry here, newest first.
Format follows [`../v3/CHANGELOG.md`](../v3/CHANGELOG.md): what changed · why · what it is sourced from ·
what it left open.

**Rules for this file**

- One entry per working pass, not per edit.
- Ch 7–8 introduce no number of their own: **every number restates one of Chapter 6**, and the pass that
  writes it names the Chapter 6 table or section (a `\dataref` in the text).
- Mechanical checks (`python3 tools/check.py`) are re-run after every pass and their result is
  recorded. **There is no TeX toolchain in this container, so "checked" never means "compiled".**
- A pass that absorbs a v3 change records which v3 revision it absorbed, from
  `inherited/SYNC_STATE.json`, and the v2 revision v3 carries.
- **Every individual changelog under `changelogs/` is signed at the end** — who wrote it, on what date,
  and that it was not compiled. A pass written by an AI agent says so by name and model.
- Legacy notes (the author's `notes.txt`, older instructions) are **verified against the current draft
  before use** (author, 2026-09-24: "maybe outdated! Only use when need/correct"); the changelog says
  which were applied and which were superseded.

---

## v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt → [`changelogs/v4.2_20260925_audit_v4.1a_applied.md`](changelogs/v4.2_20260925_audit_v4.1a_applied.md)

- **Absorbed:** v3.100's `05_setup.tex` and `06_results.tex` (`tools/sync_v3.py merge`, fast-forward; stamp
  v3.100, carrying **v2.27** — v2.28 is still not in v3). Six figure files re-exported (`export_to_draft.py v4`).
- **Audit applied in full** (`audit from chatgpt/AUDIT_v4.1a_2026-09-25.md` §15, agreed by the auditor in
  §12–14): Ch 7 C1–C8 (CI-MeanFM dominates, MeanFM one in thirty; aligning claims scoped to the unprojected
  K = 20 comparison and medians; the order scoped to the tasks with human demonstrations; corridor tie on
  traversing flights; s-curve order; $\nfe$; RQ2 without the 2.45× claim), Ch 8 D1–D11 (populations named for
  every restated statistic; the candidate paragraph; future-work items as tests), appendix E1–E7 (rotor reach
  scope; B.1/B.2 guards; nominal scoring set; B.4 rules on traversing cells and the latched crossing; B.5
  aggregate-agreement guard for D01; C the MuJoCo MPC environment), M1–M3.
- **Author's notes:** the Extended Results heading reads "Extended Results (long data: web link and repository
  code link)" — every `\hole`/`\longdata`/`\guard` kept; Future Work gains *The planning horizon* (H = 8, one
  executed step per replan; earlier constraints and a farther endpoint against more network and projection cost;
  untested).
- **Checked:** `tools/check.py` 17 files, 7788 lines, 276 labels, 53 of 53 citations, 41 figures, all pass.
  Bundles `thesis_v4_20260925_110246_new` / `_full_clean` byte-faithful, headers v3.100 / v2.27. Release: in the
  individual changelog. **Not compiled.** Nothing committed.
- **Cross-draft:** INBOX v3.100 row closed; FYI note to v3 (K/NFE wording in Ch 6 unresolved; the 18.1 ms
  candidate-study record). Signed: Claude (Fable 5.1, Claude Code), 2026-09-25.

---

## v4.1a — 2026-09-24 · Synced to v3.99 (carrying v2.27); alias labels removed; bundle rebuilt

- **Absorbed:** v3.99's `05_setup.tex` and `06_results.tex` (`tools/sync_v3.py merge`, fast-forward; stamp
  v3.99, which carries **v2.27**). v3.99 re-pointed every Ch 5/6 reference at moved or archived appendix
  content (the v4.0 and v4.1 notes).
- **Alias labels removed** from `chapters/09_appendix.tex`, as the v3.99 note asks: the five on the
  Extended Results heading and `app:avoiding-twenty-episode`. No reference targets them any more.
- **Not yet in v4: v2.28** (Ch 1 outline, "MuJoCo MPC" in Ch 2/4, PD declared). v3 has not merged it
  (`v3/tools/sync_v2.py status`: v2 moved, 7 files), and v2 reaches v4 only through v3. After v3's
  `sync_v2.py merge`: `python3 tools/sync_v3.py merge`, then rebuild.
- **Checked:** `tools/check.py` 17 files, 7628 lines, 276 labels, 53 of 53 citations, 41 figures; all pass
  in both switch states. Bundles `thesis_v4_20260924_220757_new` / `_full_clean` rebuilt, both
  byte-faithful; headers read v3.99 / v2.27. Not compiled; nothing committed.
  Signed: Claude (Opus 5.5, Claude Code), v4 · 2026-09-24.

## v4.1 — 2026-09-24 · Conclusion = Ch 7 (summary, RQ answers); Discussion = Ch 8 (limitations, deployment, future work); §7.1–7.3 dropped and archived; Isaac Sim → [`changelogs/v4.1_20260924_conclusion_ch7_discussion_ch8.md`](changelogs/v4.1_20260924_conclusion_ch7_discussion_ch8.md)

- **Order and content (author):** `chapters/07_conclusion.tex` holds the v4.0 Summary and RQ answers
  unchanged; `chapters/08_discussion.tex` holds Limitations (one seeds sentence added), Towards
  Deployment (plus one paragraph on the candidate machinery, the -r/-t/-c discussion) and Future Work
  (the simulator is **NVIDIA Isaac Sim**; the hole is gone). Interpretation, Negative and Inconclusive
  Results and Threats to Validity are dropped as duplicates of Ch 6 and archived verbatim in
  `withheld/20260924_v4.1_archive/`.
- `tools/sync_v3.py`: policy `watch` for v3's frozen `07_discussion`/`08_conclusion`; the renamed v4
  files are `own`. The appendix `\srcnote` that pointed at the dropped §Threats now points at §Towards
  Deployment.
- **Checked:** `tools/check.py` passes (20 files, 7969 lines, 285 labels; 1 `\hole`, 20 `\guard`,
  4 `\longdata`); both bundles rebuilt and byte-faithful. **Not compiled.** Nothing committed.
- **Appendix, second round (points 9–11):** A = *Supplementary Material* (A.1 sampling laws unchanged,
  A.2 the quadrotor dimensions figure, moved out of Reproducibility); B keeps the five sections the
  author named (the `\longdata` marker stays on UAV-corridor's rules); the other four are archived; C =
  *Compute Environment*. Chapter 6's references to the archived labels are bridged by alias labels;
  v3 notified of every index change (`cross_draft/to_v3/FROM_v4_20260924_v4.1_appendix_index_changes.md`).
  `check.py`: 17 files, 7634 lines, 282 labels, all pass; bundles rebuilt and byte-faithful.
- **Abbreviations (point 12): no change.** The `_new` bundle lists DPCC alone because Ch 1–4, where the
  other ten `\ac{}` calls are, are collapsed; the full build lists all eleven by itself (author: no
  aggregation needed).
- **Cross-draft:** v2 must reword Ch 1's outline sentence (Ch 7 concludes, Ch 8 discusses); the v3
  note's pointer target is `sec:disc:practice`. Signed: Claude (Fable 5.1, Claude Code), 2026-09-24.

## v4.0 — 2026-09-24 · v4 initialised on v3.98 / v2.27: Ch 7 written, Ch 8 rewritten, appendix restructured, tools and bundle → [`changelogs/v4.0_20260924_init_ch7_ch8_appendix.md`](changelogs/v4.0_20260924_init_ch7_ch8_appendix.md)

- **Branch.** `v4/` built like `v3/`: Ch 1–6, parts and both `.bib` files copied byte-for-byte from v3 at
  **v3.98** (which carries **v2.27**); merge base in `inherited/v3_base/`, stamped in `SYNC_STATE.json`.
  `tools/sync_v3.py` (status/diff/merge/stamp against `../v3`), `tools/check.py` (nested inputs, the
  appendix switch), `bundle/make_bundle.py` (**v2 collapsed, v3 + v4 built**; versions in every header).
- **Ch 7 Discussion** written: interpretation (three steps; where the saving comes from; budget and
  guiding steps; candidate plans and the selection rule; what the demonstrations decide; the plant and
  its controller), negative and inconclusive results, threats to validity (incl. what the time per action
  contains), limitations.
- **Ch 8 Conclusion** rewritten from the author's general conclusion and the three steps: summary, RQ
  answers, *Towards Deployment*, future work. v3's stale 20-09 draft is not reused.
- **Appendix**: four pure-data sections marked `\longdata` at their head with their tables behind
  `\ifappendixfull` (default on); the twenty-episode dead block dropped (alias label kept); the
  corridor-side figure freed from that block; Reproducibility reduced to the X2 figure and the compute
  facts — the "GPUs per job, held exclusively" row, the qualification paragraphs, the artefact-name map
  and the corpora of record are out (kept in `notes/`).
- **Legacy notes verified**: points 2–4 and 6–8 of the init prompt applied as far as the data support
  them (corridor projectors = trade-off per v3.96); point 5 audited (`notes/AUDIT_…`); from `notes.txt`
  the selection-rule discussion and the timing question are answered from Ch 6 data, the Omniverse
  remedy is future work behind a `\hole`, and "drop the derivations" is superseded by the author's
  keep instruction (v3.58).
- **Checked:** `tools/check.py` passes (20 files, 8302 lines, 294 labels, 53 citations, 41 figures;
  2 `\hole`, 1 `\provisional`, 22 `\guard`, 2 `\flawed`, 4 `\longdata`). Bundles `thesis_v4_<stamp>_new`
  and `_full_clean` built and byte-faithful (`--verify`). **Not compiled.** Nothing committed.
- **Cross-draft:** notes to v3 (re-point the twenty-episode references; the `app:repro` pointer in
  Ch 5 §5.6.1) and v2 (abbreviations); the → v4 inbox rows closed at v4.0. `DRAFT_OWNERSHIP.md` gains
  v4's row. Signed: Claude (Fable 5.1, Claude Code), 2026-09-24.
