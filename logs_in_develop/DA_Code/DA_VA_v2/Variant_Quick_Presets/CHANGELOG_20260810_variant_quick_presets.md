# U16 — Quick presets over "5. Variants"

**Date:** 2026-08-10
**Scope:** both HTML viewers — `Data_Analysis/Visualizer/index.html` (DAv3) and
`Data_Analysis/Visualizer_VA_v2/index.html` (regenerated from it)
**Status:** written + tested here (stdlib harness, 102/102); **the full page test
still to be run on the cluster** (needs pandas + matplotlib)
**Follows:** `../Highlight_and_Seed_Coverage/CHANGELOG_20260809_candidate_highlight_and_seed_coverage.md`

---

## Why

A batch carries ~18–26 variants, and the comparison actually run over and over is
a handful of them: the diffusion baseline, the DPCC projection arms, the HardFlow
arms. Ticking those eight boxes by hand — and, sooner or later, mis-ticking one —
was the routine cost of every read. Worse, the mis-tick is silent: a chart
missing `dpcc-t-tightened` looks exactly like a chart where that arm has no data.

## What was added

Three checkboxes above the variant list, sharing its border so they read as one
control rather than a second, competing filter:

| preset | what it ticks |
|---|---|
| **DPCC + HF** | `diffuser` + every `dpcc-{r,c,t}` and `hardflow_new*` arm, tightened or not |
| **DPCC + HF (tightened)** | `diffuser` + only the `-tightened` `dpcc-{r,c,t}` and `hardflow_new-{r,c,t}` arms |
| **DPCC (tightened)** | `diffuser` + only the `-tightened` `dpcc-{r,c,t}` arms. No HardFlow. |

`diffuser` is in all three on purpose: it is the unconstrained generator every
projection arm is read *against*, and it has no tightened twin.

Ticking a preset ticks its variants in the main list below. It is a **shortcut
for those boxes, not a second filter** — what gets plotted is always exactly what
the main list shows, so there is no hidden state and nothing new to reason about
when a bar is missing. Unticking clears the same set.

### The exclusions are the point

Deliberately **not** taken by any preset:

* `dpcc-c-tightened-dt0p25`, `-dt0p5`, `-dt2p0`, `-dt4p0` — a timestep-scaling
  **sweep**, not an arm of the comparison. Folding them in puts four
  near-identical bars beside the one being read.
* `gradient`, `post_processing`, `model_free`, `geo_free`, `bounds_free` and
  their combinations — different baselines, not the DPCC/HardFlow comparison.

Against the real variant lists this comes out as:

```
DAv3 (avoiding, 24 variants)
  DPCC + HF               14   diffuser, dpcc-{c,r,t}[-tightened],
                               hardflow_new[-{c,r,t}][-tightened]
  DPCC + HF (tightened)    7
  DPCC (tightened)         4
  never in any preset          4 dt sweeps + gradient/post_processing/model_free (x2)
```

### Membership is a rule, not a list

```python
PRESET_ARM_RE = r'^(?P<fam>dpcc|hardflow_new)(?:-(?P<sel>[rct]))?(?P<tight>-tightened)?(?P<extra>-.+)?$'
```

A name matching the family with **no extra suffix** is an arm; anything with a
trailing `-…` is a sweep and is dropped. So a batch that gains an arm gets it for
free, and a batch that lacks one is never left with a checkbox that selects
nothing.

That last part matters on **DA_VA_v2**, where tightening lives on the *geometry*
axis rather than in the variant name. There the two tightened presets have no
members, so they are **not rendered at all** — and the panel says so, pointing at
`4. Geometry Focus` instead. A batch with no projection arms hides the panel
entirely. No dead checkbox in any case.

### Indeterminate state

A preset box states a **fact about the main list**, so it is recomputed from that
list on every redraw rather than remembered:

* all its members checked → ticked
* some → the grey indeterminate mark
* none → clear

Hand-unticking one member therefore drops the preset to indeterminate instead of
leaving it claiming a selection that is not there.

## How it is wired

Membership is computed in **Python** at populate time and baked into each preset's
`data-members` attribute; the toggle and the sync are **pure JS**. So ticking a
preset is instant and works even while the PyScript engine is still warming up,
and there is no async round-trip for what is a DOM operation.

`sync_variant_presets()` is called at the top of the JS `trigger_plot()` wrapper,
which every path — a main checkbox, `[ALL]`/`[NONE]`, a preset toggle — already
goes through. No new call sites, and `bulkSelect` did not have to be touched (it
is an anchor for `build_from_dav3.py`).

## Files touched

```
Data_Analysis/Visualizer/index.html                  U16 CSS, sidebar slot, VARIANT_PRESETS +
                                                     _preset_members + render_variant_presets,
                                                     toggle/sync JS
Data_Analysis/Visualizer_VA_v2/index.html            REGENERATED (34 edits, 2488 lines)
Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py   U16 checks; document stub now caches
                                                     elements so rendered markup can be read back
Data_Analysis/Visualizer_VA_v2/test_page_offline.py  U16 checks against the real batch
Data_Analysis/DA_Code_v3/README.md                   preset section
Data_Analysis/DA_VA_v2/README.md                     preset section
```

`build_from_dav3.py` needed **no** change — every anchor it relies on survived.
No pipeline, CSV or config change: viewer-only.

## Testing

**Ran here (container, stdlib only):**

```bash
python3 Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py    # 102/102 PASS
```

The U16 checks run `_preset_members` against a variant list built to contain
exactly what must be excluded (dt sweeps, gradient/post_processing/model_free,
geo_free/bounds_free) and assert the three memberships **element for element**,
that nothing leaks, that the presets nest
(`DPCC tightened ⊆ DPCC+HF tightened ⊆ DPCC+HF`), and that `diffuser` alone never
constitutes a preset. Then the rendered control: three boxes for a full DAv3
batch with the members in `data-members`; **one** box plus the "not offered by
this batch → use 4. Geometry Focus" note for a VA-style batch with no
`-tightened` names; and a hidden panel for a batch with no projection arms.

Membership was also verified directly against the real
`DA_Code_v3/config.py::DEFAULT_PROJECTION_VARIANTS` (24 variants) and
`DA_VA_v2/config.py::VARIANT_ORDER` (26 variants) — output reproduced above.

The two pages are still asserted **byte-identical** on `_preset_members` and
`render_variant_presets`, and to carry the same JS hooks.

**Still to run on the cluster:**

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py <batch_va2_dir>
```

Its `[U16 …]` section checks the panel renders one box per non-empty preset for
*that* batch, that every member is a real variant checkbox value (an orphan
member would tick nothing), that nothing leaks, and that a preset selection
actually draws.

Then, by eye in a browser: tick **DPCC + HF**, confirm the right boxes light up
in the list below and the plot redraws; hand-untick one member and confirm the
preset drops to the grey indeterminate mark.

## Known limitations

* The presets are **additive** shortcuts, not radio buttons: ticking *DPCC + HF*
  then *DPCC (tightened)* leaves the union selected. That is the honest reading
  of "these boxes tick those boxes"; use `[NONE]` first for an exact set.
* On a DA_VA_v2 batch the tightened presets are absent because tightening is a
  geometry, not a variant. There is no preset that switches
  `4. Geometry Focus` — a variant shortcut has no business moving a different
  axis without saying so.
* Membership is fixed at populate time. A preset ticks whatever variants existed
  when the batch was synced; `SYNC_SOURCE` rebuilds it.
