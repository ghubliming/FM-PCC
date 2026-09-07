# CHANGELOG — Gen15 U5: projection-variant slimming, `-geo_free` composites, HardFlow fan spread

**Date:** 2026-08-27
**Scope:** `config/uav_projection.yaml` · `config/uav_mix.py` · `mix_uav/sampling/hardflow_projection.py`
**Status:** written here, **not run** — validate on the cluster (i6-gpu-1)
**Motivation:** the next pillars sweep needs (a) fewer, more meaningful variant slots,
(b) a projector that fixes dynamics + action bounds while ignoring scene geometry, and
(c) both HardFlow fan sizes present in one job.

---

## 0. Summary

| # | change | file |
|---|---|---|
| 1 | `projection_variants` cut 20 → 10; U8 ablation variants retired as standalone rows | `config/uav_projection.yaml` |
| 2 | New `dpcc-{r,c,t}-geo_free` — DPCC with the geometric group OFF | `config/uav_projection.yaml` |
| 3 | `hardflow_variants` 3 → 7; bare `hardflow_new` (B=1) restored; `-geo_free` siblings added | `config/uav_mix.py` |
| 4 | **`resolve_hf_batch_size` now strips constraint-toggle suffixes** (correctness fix) | `mix_uav/sampling/hardflow_projection.py` |

Every replaced value is **commented out in place, not deleted**.

Variant slots per eval job: **23 → 17** (10 yaml + 7 HardFlow).

---

## 1. Why the U8 ablation variants are retired

`geo_free`, `bounds_free`, `model_free` and their composites were **study instruments** for
`U_8_new_projection_var_upgrade` — they answered "which constraint family does the work?".
That question is answered, and running them as standalone rows spent **13 of 20 variant slots**
per job on rows no benchmark table reports.

Removed as standalone entries: `gradient`, `gradient-tightened`, `post_processing`,
`post_processing-tightened`, `model_free`, `model_free-tightened`, `bounds_free`,
`bounds_free-tightened`, `geo_free`, `geo_free-bounds_free`, `geo_free-model_free`,
`model_free-bounds_free`, `model_free-bounds_free-tightened`.

Kept: `diffuser` (the unprojected reference — the denominator every DA needs), the six
`dpcc-*` rows, and the three new `dpcc-*-geo_free` rows.

The old list is preserved verbatim as a commented block directly above the active one; restoring
it is an uncomment.

---

## 2. The new "dynamics + action bound only" projector

**Requirement:** a projector that corrects the dynamics and the action bound but enforces
nothing about walls, pillars or the workspace box.

**Implementation: `dpcc-{r,c,t}-geo_free` — a composed variant name, no new code path.**

The three geometry gates in `setup_dpcc_projector` are substring tests:

```python
if 'geo_bounds' in ctypes and 'geo_free' not in variant:   # eval_mix_uav.py:927
if 'halfspace'  in ctypes and 'geo_free' not in variant:   # eval_mix_uav.py:986
if 'obstacles'  in ctypes and 'geo_free' not in variant:   # eval_mix_uav.py:995
```

so appending `-geo_free` to a selector name already produces exactly the requested projector.
Resolved families, verified by simulating the gates:

| variant | pillars | corridor |
|---|---|---|
| `dpcc-c` | geo_bounds, bounds, dynamics, obstacles | geo_bounds, bounds, dynamics, halfspace, obstacles |
| **`dpcc-c-geo_free`** | **bounds, dynamics** | **bounds, dynamics** |
| `hardflow_new-c` | geo_bounds, bounds, dynamics, obstacles | geo_bounds, bounds, dynamics, halfspace, obstacles |
| **`hardflow_new-c-geo_free`** | **bounds, dynamics** | **bounds, dynamics** |

### 2.1 Why a composed suffix and not a brand-new token

A new token (`-nogeo`, say) would have to be mirrored into **two** eval scripts —
`mix_uav_test/eval_mix_uav.py` (Gen15) *and* `FM_v3_uav_test/eval_fm_uav.py` (Gen11) — because
both read this shared yaml. Any drift between them would mean Gen11 silently building the FULL
constraint set inside a folder named `-nogeo`: wrong data under a right-looking name, the worst
failure mode available.

Reusing `-geo_free` makes the two generations identical **by construction**, and it is the
documented intent of U8 — `FM_v3_uav_test/eval_fm_uav.py:815` already spells out
`dpcc-c-geo_free-bounds_free` as the composition pattern.

The alternative the request also floated — a second `geo_constraint_variants` entry, i.e. a new
`geo_tag` folder — was **not** taken. It would fork the output path
(`…/6/<geo_tag>/<variant>/`), so the geometry-off rows would land under a different `geo_tag`
than the full-stack rows and stop being directly comparable in one table. The variant-name axis
keeps every row of one scene under one `geo_tag`, which is what the DA aggregator groups on.

### 2.2 Violations are still scored against the FULL geometry

`_exec_constraint_violations(obs_traj, config)` (`eval_mix_uav.py:442`) checks the **flown** path
against the **raw** scene geometry ⊕ `r_drone`, using the full config, and is
**variant-independent**. So a `-geo_free` row with `n viol = 0` is a real, fully-scored clean
flight — it means the **generator** produced a legal path without being told the geometry. That is
the claim these rows exist to test.

### 2.3 No `-tightened` siblings

`enlarge_constraints` only shifts geo_bounds / halfspace / obstacles surfaces. `-geo_free`
removed all three, so `dpcc-c-geo_free-tightened` would be byte-identical to
`dpcc-c-geo_free`. Same rule U8c applied to the original `geo_free*` rows.

---

## 3. HardFlow variant list — both fan sizes, and geometry-off siblings

```python
'hardflow_variants': [
    'hardflow_new',                                        # B=1, full stack
    'hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t',  # B=4, full stack
    'hardflow_new-r-geo_free',                             # B=4, geometry OFF
    'hardflow_new-c-geo_free',
    'hardflow_new-t-geo_free',
]
```

**Bare `hardflow_new` is back.** `B4_PARITY` (2026-08-20) removed it because at B=4 it was
byte-identical to `-r`. It returns deliberately as the **B=1** arm — upstream-faithful HardFlow,
which asserts `batch == 1`. It is a *different experiment* from `-r`, not a rename.

Fan sizes are resolved from the **name**, at runtime:

| variant | B |
|---|---:|
| `hardflow_new` | **1** |
| `hardflow_new-r` / `-c` / `-t` | **4** |
| `hardflow_new-{r,c,t}-geo_free` | **4** |
| every `dpcc-*` (arm B) | **4** |

Arm B never routes through `resolve_hf_batch_size`; it takes `mpc_batch_size` (4) directly at
`eval_mix_uav.py:1750`. Unchanged. The per-variant fan is printed at `eval_mix_uav.py:1953` as
`(B=…)`, so it is checkable in the batch log.

---

## 4. 🔴 The correctness fix in `resolve_hf_batch_size`

**This one is not cosmetic and is the reason change 3 could not ship alone.**

The selector test was:

```python
for _suffix in ('_train_set', '-tightened'):
    if name.endswith(_suffix): name = name[:-len(_suffix)]
...
if name.endswith(('-r', '-c', '-t')):
    return max(1, int(configured_batch))
return 1
```

`'hardflow_new-r-geo_free'` ends in `'e'`. It would have fallen through to the bare-name branch
and run at **B=1**, while its `dpcc-r-geo_free` counterpart ran at **B=4** — silently
reintroducing exactly the fan mismatch `B4_PARITY` was written to kill, in the one place nobody
would look. Both arms loop *serially* over the candidate fan around their CPU solve, so a 4×
mismatch scales projection cost near-linearly and voids every arm-B-vs-arm-C wall-clock claim.

Fix — a named suffix list, stripped **repeatedly** so composites resolve:

```python
_TOGGLE_SUFFIXES = ('_train_set', '-tightened', '-geo_free', '-bounds_free', '-model_free')
```

Any future constraint-group toggle that can follow a selector **must** be added here.

### 4.1 Verification (run locally, stdlib only)

`resolve_hf_batch_size` lifted out of the module and exercised directly:

```
OK  hardflow_new                          B=1  (expected 1)
OK  hardflow_new-r                        B=4  (expected 4)
OK  hardflow_new-c                        B=4  (expected 4)
OK  hardflow_new-t                        B=4  (expected 4)
OK  hardflow_new-r-geo_free               B=4  (expected 4)
OK  hardflow_new-c-geo_free               B=4  (expected 4)
OK  hardflow_new-t-geo_free               B=4  (expected 4)
OK  hardflow_new-r-tightened              B=4  (expected 4)
OK  hardflow_new-geo_free                 B=1  (expected 1)
OK  hardflow_new-c-geo_free-bounds_free   B=4  (expected 4)
OK  dpcc-c correctly raises (arm B never routed here)
```

`_selection_for` lifted and exercised the same way — every new name routes to the right rule:

```
dpcc-r-geo_free          -> random
dpcc-c-geo_free          -> minimum_projection_cost
dpcc-t-geo_free          -> temporal_consistency
hardflow_new             -> random
hardflow_new-r-geo_free  -> random
hardflow_new-c-geo_free  -> minimum_projection_cost
hardflow_new-t-geo_free  -> temporal_consistency
```

`hardflow_new-c-geo_free` routes correctly because `_selection_for` tests `'-c-' in variant`
(`eval_mix_uav.py:1045`), which the appended suffix preserves. Had the test been
`endswith('-c')` only, this would have silently fallen back to `random`.

**Not verified locally:** the yaml does not parse-check here (no PyYAML in this container), only
a structural check — exactly one active `projection_variants` list, brackets balanced, 10 items,
no uncommented remnant of the old block. **Confirm on the cluster** that
`config/uav_projection.yaml` loads and that the eval prints all 17 variants.

---

## 5. Constraint-matching between arms B and C is preserved

`eval_mix_uav.py:1596-1601` builds HardFlow's NLP by calling `setup_dpcc_projector(...,
variant, return_constraint_list=True)` — the **same** function, the **same** variant string, the
**same** `constraint_list` the DPCC `Projector` consumes. So `-geo_free` gates both arms
identically and `dpcc-c-geo_free` vs `hardflow_new-c-geo_free` remains a valid head-to-head. This
was the Gen12 port's first design rule and U5 does not weaken it.

---

## 6. Blast radius

- **Gen11 (`flow_matching_v3_uav`) is affected.** `config/uav.py:30` reads the same yaml, so
  Gen11's next eval also runs the 10-variant list. Safe because `-geo_free` already exists in
  Gen11's gating code with identical semantics (`eval_fm_uav.py:836/894/904`) — but any Gen11
  run after this commit is **not** variant-comparable to a Gen11 run before it.
- **`hardflow_variants` and the batch-size patch are Gen15-only.** Sibling generations
  (`flow_matcher_v3_meanflow`, `flow_matcher_v3_hardflow`, `flow_matcher_v3_alphaflow`,
  `mix_visual_aligning`, `mix_visual_avoiding`) each carry their own
  `hardflow_projection.py` with the **unpatched** `endswith(('-r','-c','-t'))` test. They are
  untouched, per the copy-modify sibling convention. **If any of them ever adds a toggle suffix
  after a selector, it inherits the B=1 bug** — sync `_TOGGLE_SUFFIXES` first.
- **Existing result folders are unaffected.** No variant was renamed; retired names simply stop
  being generated. Old `geo_free` / `gradient` / `post_processing` folders on disk stay readable
  and will still be picked up by the aggregator as historical rows at a different rev.

---

## 7. Cost

17 variants instead of 23 (−26 %), but the mix is heavier: the 4 new B=4 HardFlow rows are the
most expensive kind, and bare `hardflow_new` at B=1 is the cheapest. Net per-job wall time is
expected to be **roughly flat to modestly higher**, not lower — the win here is that every slot
now feeds a benchmark row instead of a retired ablation. Measure on the first pillars job before
sizing the rest.

---

## 8. What to check on the first cluster run

1. `config/uav_projection.yaml` loads (no PyYAML check was possible locally).
2. The eval banner at `eval_mix_uav.py:2028` lists **10** yaml variants, and the HardFlow line
   at `:450` reports **+7**.
3. `(B=1)` appears for `hardflow_new` and `(B=4)` for every `-r`/`-c`/`-t` row, **including the
   `-geo_free` ones** — this is the change-4 regression, and the batch log is where it shows.
4. `dpcc-c-geo_free` folders appear under the **same** `geo_tag` as `dpcc-c`
   (`…/6/<scene>_bounds+dynamics+geo_bounds+…/dpcc-c-geo_free/`), not a new one.
5. `[hardflow] … sel=minimum_projection_cost` prints for `hardflow_new-c-geo_free`.
