# DA — 2026-09-25 · D3IL-aligning: the frontier zero-line label, the straight push against the halfspace, and which constraint the learned plans cross

**Why:** the ChatGPT audit of v3.98 (`Working_Space/v3/audit from chatgpt/AUDIT_v3.98_2026-09-24.md`, F03 and
F13.1, applied at v3.100) and the author's note of 2026-09-25: "the V_A demo is lacking the halfspace — add
it, and sanity-check elsewhere; I totally forgot there is a halfspace in V_A". Three things, no new run.

## 1 · The alignment frontier's zero line was labelled "box not moved"; it is a "0 % reference"

The percentage printed beside every alignment median (Tables 6.5, 6.7, 6.8; the outcome axis of
`fig_aligning_tradeoff` and `fig_aligning_projected_tradeoff`) is, and always was,

    100 · (1 − median(d_final) / mean(d_init)),   mean(d_init) = 0.4530 m over the ten contexts

(`DA_20260923_R2fix_R37_aligning.md:23`; `plotting/builders/frontier.py::_pct_closed`). That is a reduction of
the *median* final distance relative to the *mean* initial distance, not the median of per-context shares
closed. The two are close for the operating rows and differ where a box does not move, because the ten
initial distances are not equal (`per_rollout_detail.csv`, `context_init_xy_dist`):

| quantity | value |
| :-- | --: |
| initial distance, ten contexts: mean / median | 0.452980 m / 0.455746 m |
| every box left unmoved, under the printed formula | −0.6 % (prints −1 %) |
| MeanFM K20 unprojected, untightened: printed / paired per-context median | 83.6 % / 83.5 % |
| MeanFM K20 unprojected, tightened (`none` row): printed / paired | 80.1 % / 80.9 % |
| MeanFM K20 per-step *r*, tightened: printed / paired | 60.6 % / 57.7 % |
| MeanFM K20 endpoint *r*, tightened: printed / paired | 56.6 % / 53.6 % |

**Change:** the thesis names the quantity as it is computed (Ch 5 metrics; Ch 6 captions), and the label the
builder drew at the zero line — `'box not moved'`, `builders/frontier.py:252` (unprojected) and `:390`
(projected) — now reads `'0 % reference'`. Both figures were rebuilt; **every plotted value is unchanged**
(token diff against HEAD: one `<text>` node each). Not done: the paired recomputation (a legitimate
alternative; it would move the operating rows by up to three points and the figure axis).

## 2 · The straight push from box to target never crosses the halfspace

`extract/expert_paths.py` tested the direct push (box centre → target centre) against the keep-out disk only.
It now tests it against the halfspace as well (`push_hits_halfspace`, `push_hits_halfspace_tightened`). The
excluded side of a halfspace is a halfplane, convex, so a segment enters it iff one of its ends lies in it;
the tightened boundary is the nominal line moved toward the allowed side by 0.03 m, as the figure draws it.

| split | contexts | push crosses the keep-out region (tightened) | push crosses the halfspace (tightened) |
| :-- | --: | --: | --: |
| train | 60 | 51 (60) | 0 (0) |
| test | 60 | 55 (59) | 0 (0) |
| **all** | **120** | **106 (119)** | **0 (0)** |

The keep-out counts are those the thesis already printed. `fig_expert_aligning` now says so in its subtitle
("… 106 direct pushes cross the keep-out region, none the halfspace"); its colouring is unchanged. What this
does **not** establish: whether a demonstration's *end effector* entered the far corner while working around
the box. The recorded end-effector paths are on the cluster (`d3il/environments/dataset/data/aligning/` holds
the contexts and file lists only) and were not tested; the thesis says so (Ch 5, datasets).

## 3 · Which constraint the learned plans cross (sanity check the author asked for)

Per context, `analysis_results_checkpoint/15-09/batch_va2_20260915_100754/per_rollout_detail.csv`, MeanFM
K20 (`H8_K20_Meuler_T0.2_…VisualMeanFlow_VTrue_mpc4_filmv1_Emf`), columns
`constraint_exec_{halfspace,obstacle,bounds}_viol_count` and `constraint_exec_zero_violation`:

| set | variant | contexts with halfspace viol. | keep-out viol. | action-bound viol. | halfspace or keep-out | violation-free | violating steps hs / keep-out / bound |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| untightened | none (`diffuser`) | 2 | 7 | 3 | 9 | 1 | 272 / 71 / 306 |
| untightened | per-step *r* | 3 | 3 | 4 | 6 | 3 | 345 / 26 / 153 |
| untightened | endpoint *r* | 3 | 3 | 2 | 5 | 4 | 328 / 8 / 127 |
| tightened | none (`diffuser`) | 1 | 7 | 2 | 8 | 2 | 126 / 75 / 120 |
| tightened | per-step *r* | 0 | 0 | 1 | 0 | 9 | 0 / 0 / 27 |
| tightened | endpoint *r* | 0 | 0 | 0 | 0 | 10 | 0 / 0 / 0 |

Reading: by contexts the keep-out region is the constraint the unprojected plans cross (7 of 10, both sets);
the halfspace is crossed in one context on the tightened set and two on the untightened, but for many steps
(the end effector stays in the far corner once it is there: 126–272 steps against 71–75 in the keep-out). Both
go to zero under either projector on the tightened set; what remains there is the action bound. The chapter's
"crossing the constraint in eight contexts of ten" (tightened, unprojected) is 7 keep-out + 1 halfspace, disjoint
contexts. **The thesis now prints "seven … and the halfspace in one"** (Ch 5 datasets; Ch 6 §6.2 projected
trade-off) and defines the alignment constraint set with both forms wherever it names one.

## 4 · Incidental, from the full store rebuild

`make_figs.py` was run in full (a partial run rewrites `MANIFEST.md` with the subset). Seventeen SVGs differed
from HEAD only by the generated `clipPath id` values and were restored from git. One more differed in content:
`fig_uav_corridor_paths.svg`, whose builder already carried the v3.79 vocabulary (*violation-free flight*,
*violated a constraint*, "12/12 violation-free") while the stored copy still read *collision-free* / *entered
an obstacle* / *clean*. The rebuilt copy is kept and its PNG re-rendered; no data changed.

## 5 · Files

- `plotting/extract/expert_paths.py` (halfspace test, counts, note), `data/expert_paths.json` (regenerated)
- `plotting/builders/expert.py` (subtitle), `plotting/builders/frontier.py` (label ×2, the axis comment)
- `figures/env/fig_expert_aligning.{svg,png}`, `figures/da/fig_aligning_tradeoff.{svg,png}`,
  `figures/da/fig_aligning_projected_tradeoff.{svg,png}`, `figures/da/fig_uav_corridor_paths.{svg,png}`;
  `figures/MANIFEST.md` (full rebuild, 2026-09-25). PNGs by `svg/preview_png.py --scale 3`, the store's scale.
- `analysis/INDEX.md` row for `fig:expert-aligning` updated. Draft copies: `export_to_draft.py v3`; **v4 holds
  its own copies and the release prefers them (`RELEASE/tools/make_release.py:584`) — v4 re-exports.**

Reproduce: `PYTHONPATH=/workspaces/FM-PCC python3.14 plotting/extract/expert_paths.py && python3.14
plotting/make_figs.py`, then the four `preview_png.py` calls and `export_to_draft.py <draft>`.

Signed: Claude (Fable 5.1, Claude Code) · 2026-09-25 · local `python3.14` only (no cluster job, no training or
evaluation run); not compiled; nothing committed.
