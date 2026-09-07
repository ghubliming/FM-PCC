# U19 — Path Audit Map: model-family tabs + SEL as a second candidate selector

**Scope:** `Data_Analysis/Visualizer/index.html` (DAv3, the `avoiding` explorer) — `SCIENTIFIC_SUITE_v3.16 → v3.17`.
**New test:** `Data_Analysis/Visualizer/test_audit_families_offline.py` (stdlib only — runs in the AI container).
**Not touched yet:** `Visualizer_VA_v2`, `Visualizer_UAV_v1`, `Visualizer_Visual_Aligning` — DAv3 first, per request.

---

## Why

The Path Audit Map is the one table that lists **every** candidate in a batch — 130+ rows on the
recent `batch_avoiding_combined_*`. It was one undivided wall sorted by candidate id, so the
meanflow runs, the alphaflow runs, the hardflow arms and the diffusion baselines were interleaved
with no way to read one model family at a time.

Second problem: the audit map is where you actually *decide* what to plot (it is the only place
the run path, the seed warnings and the last-run stamp are visible), but the selection lived
360 px away in the sidebar's "6. Candidates" list — a flat `CAND_1 … CAND_130` with no paths.
Choosing "the five meanflow runs" meant reading the audit map, memorising ids, then hunting
them in the sidebar.

## What changed

### 1. Model family, parsed off the run path

The family is **read, not configured**. Every run folder already states its model in the
directory directly below `plans/`:

```
.../logs/avoiding-d3il/plans/flow_matching_v3_meanflow(Bf_Fix5)/<train_cfg>/<eval_cfg>
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

`_model_folder()` finds that segment; `_model_family()` classifies it with an **ordered** rule
list into: `DIFFUSION / DPCC`, `FM`, `MF`, `iMF`, `AF`, `HARDFLOW`, `DRIFTING`, `OTHER`.

Order is load-bearing:

- `imeanflow` contains `meanflow` → **iMF is tested before MF**.
- `flow_matching_v3_*` contains `flow_matching` for *every* family → **the generic FM rule is
  last**, as the fallthrough for "a flow model with no further label"
  (`ode_selectable`, `flow_matching_v3_uav`, `fm_visual_*`, `mix_*_fm`).
- the short tokens (`_mf`, `_af`, `fm_`) are anchored on separators, so `hardflow` is not read
  as `af` and `fm_visual_avoiding` is not read as nothing.

Two fallbacks, both needed by real batches:

- `plans` is matched by **prefix**, because the UAV batches version the plans dir itself
  (`plans(Bf_U8)`, `plans(Bf_DC-FIX)`). A bare `== 'plans'` dropped every one of those to OTHER.
- With no `plans/` segment at all (the `_DA_VA_BRIDGE_d3il_baseline` folders), the model name is
  taken one level above the leaf run dir; if *that* still says nothing, the model **class** token
  (`models.visual_mf_diffusion.VisualMeanFlow` → `VisualMeanFlow`) is tried.
  The full path is deliberately **never** classified — it carries the training module name
  (`...visual_gaussian_diffusion.VisualFlowMatching`) and would file half the flow runs under
  diffusion.

### 2. Tabs over the audit map

A tab bar above the table: `ALL · n`, then one tab per family **present in this batch**, each with
its row count. A family the batch never ran gets no (dead) tab.

Implementation is **one table, rows hidden by tab** — not one table per family:

- 130+ rows each carrying two checkboxes and a ZIP button; duplicating them into an ALL pane plus
  a family pane doubles that DOM for nothing.
- more importantly it keeps a candidate as **one row with one SEL checkbox**, so no candidate can
  ever show two disagreeing selection states.

The open tab lives on `window._auditTab` (JS), not in Python, because Python re-renders this table
wholesale on every highlight toggle — it reads the value back to re-hide the rows it just emitted,
so the open tab survives a redraw. Switching batches to one that lacks the open family falls back
to `ALL` rather than rendering an all-hidden table.

New **Model** column: the family label (colour-coded) over the raw folder name, so the grouping is
auditable in place — you can see *why* a row landed on the tab it did.

### 3. SEL — the second candidate selector

New leading column of checkboxes. It is a **second view of sidebar "6. Candidates"**, not a second
filter: `toggle_audit_select()` drives the very same `.cand-check` box and then calls
`trigger_plot()`. Anything else would give the page two candidate selections that could disagree,
with no way to tell which one the plot used.

Kept honest in both directions:

- **audit → sidebar**: the tick writes the sidebar box, then replots.
- **sidebar → audit**: `sync_audit_selection()` runs at the top of `trigger_plot()` (right next to
  the existing `sync_variant_presets()`), recomputing every SEL box *from* the sidebar. So sidebar
  `[ALL]` / `[NONE]` and hand-ticked sidebar boxes show up here with no re-render.
- **re-render**: Python renders SEL from `get_checked("cand-check")` — live state, never a
  remembered copy that could go stale.

Selected rows get a light blue `.sel-row` tint, so "what is on the plot" is readable at a glance
down a 130-row table.

Three bulk links, scoped to the **open tab** — which is the point of the tabs:

- `[SELECT ALL SHOWN]` / `[DESELECT ALL SHOWN]` — tick every MF run, or every diffusion baseline,
  in one click.
- `[CLEAR EVERY CANDIDATE]` — ignores the tab, so a stray selection hiding behind another tab can
  still be cleared.

---

## Files

| File | Change |
|---|---|
| `Data_Analysis/Visualizer/index.html` | CSS `.audit-tabs/.audit-tab/.fam-tag/.fam-dir/.sel-box/.audit-note/.audit-bulk/tr.sel-row`; Python `FAMILY_ORDER`, `FAMILY_LABELS`, `FAMILY_COLORS`, `_FAMILY_RULES`, `_model_folder`, `_model_class`, `_classify`, `_model_family`, `_active_audit_tab`, rewritten `render_path_map`; JS `window._auditTab`, `showAuditTab`, `_candBox`, `toggle_audit_select`, `sync_audit_selection`, `select_audit_visible`, `sync_audit_selection()` call in `trigger_plot`; version → v3.17 |
| `Data_Analysis/Visualizer/test_audit_families_offline.py` | **new** — asserts the parser over every batch |

## Verification (in-container)

- **`test_audit_families_offline.py`** — 16 hand-written fixtures for the shapes that are easy to
  get wrong (`imeanflow` vs `meanflow`, `hardflow` vs `af`, `plans(Bf_U8)`, the DA_VA bridge
  folders), then **472 distinct run paths across 39 batches** under `Data_Analysis/analysis_results/`.
  **PASS — nothing falls through to OTHER.** Family → folder breakdown:

  | Tab | folders |
  |---|---|
  | DIFFUSION / DPCC | 8 (`diffusion`, `visual_*_dpcc`, `mix_*_diffusion`, `_DA_VA_BRIDGE_d3il_baseline*`) |
  | FM | 7 (`ode_selectable`, `flow_matching_v3_uav`, `fm_visual_*`, `mix_*_fm`) |
  | MF | 7 (`flow_matching_v3_meanflow*`, `mix_*_mf`) |
  | iMF | 17 (`flow_matching_v3_imeanflow*`) |
  | AF | 5 (`flow_matching_v3_alphaflow*`, `mix_*_af`) |
  | HARDFLOW | 4 (`flow_matching_v3_hardflow*`) |
  | DRIFTING | 1 |

  The test **reads the parser out of `index.html`** rather than copying it, so it can never
  certify code the page does not run. Re-run it whenever a new model folder appears on the cluster.
- **Syntax**: the `<script type="py">` block AST-parses; the JS block passes `node --check`.
  Scanned for backslashes inside f-string *expressions* — a `SyntaxError` on the CPython 3.11 that
  pyodide 2024.1.1 ships, which would kill the whole page silently. One was found and removed
  (`row_cls` is now precomputed).
- **Wiring**: a throwaway DOM shim under node exercised the JS — tab filtering, active-tab marking,
  `window._auditTab` persistence, audit→sidebar, sidebar→audit, tab-scoped bulk select, and
  global clear. 9/9.

Not verifiable here (no pandas/matplotlib in this container): the live PyScript render.
**Run on cluster / any machine with the science stack** — open the page against a
`batch_avoiding_combined_*` batch and confirm the tabs appear with their counts and SEL round-trips
with the sidebar.

## Follow-ups (not done)

- Port to `Visualizer_VA_v2` and `Visualizer_UAV_v1` — both carry a `render_path_map` of the same
  shape, so the helpers drop in unchanged. (`Visualizer_Visual_Aligning` is older and uses
  `FullPath`/`FolderName` columns; it needs the column names adjusted.)
- `logs_in_develop/MASTER_TEST_HISTORY.md` not updated — say the word.
