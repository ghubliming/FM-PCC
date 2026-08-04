# Does upstream DPCC have this bug? — Yes. We inherited it, we did not create it.

**Date:** 2026-08-04
**Companion to:** `CHANGELOG_Gen0_Fix2_dpcc_threshold_wiring.md`
**Upstream examined:** `/workspaces/aux_repo/dpcc`, `origin = https://github.com/ralfroemer99/dpcc.git`,
HEAD `0ee3bee` "Update README.md", Fri 20 Jun 2025

**Verdict in one line:** the threshold wiring **and** the `post_processing` baseline both
existed, both worked, and both were **removed by upstream commit `7f09d3a` on 2 Dec 2024** in a
cleanup that looked behaviour-preserving and was not. FM-PCC copied the post-cleanup code
verbatim and has now restored both — see `CHANGELOG_Gen0_Fix2_dpcc_threshold_wiring.md`.

---

## 0. Straight answers

**Q: Is DPCC's projection code wrong?**
**No.** The projector, the SLSQP solve, the constraint handling and the gate itself are all
correct. Nothing about *how* DPCC projects is broken.

**Q: Is the paper wrong?**
**No.** Its definition of post-processing — "modifying them after the last denoising step,
usually by solving an optimization problem" — matches its own pre-cleanup code exactly:
`threshold = 0` fires the gate only at `t = 0`, i.e. one optimization on the finished sample.
The paper, the argument and the published numbers are consistent. It is the code at HEAD that
drifted away from the paper, not the paper from reality.

**Q: If someone downloads DPCC today and changes `diffusion_timestep_threshold: 0.5` to
something else, will it work?**
**No. Nothing will happen.** The run will be identical to 0.5 in every respect except the
savepath name, which will show the new number. No error, no warning. That is the whole defect:
**one config key in that repo is inert.**

**Q: Are DPCC's published results wrong?**
**No.** Those were produced by the pre-cleanup code, which wired it. And every result at 0.5 is
correct either way, because 0.5 is what the default is.

**Q: Was it us?**
**No.** We copied `scripts/eval.py` after the cleanup. Same characters.

**Q: Are OUR models affected?**
**Not any more.** Two live paths were missing it — `scripts/eval.py` and
`diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:380` (found during this audit).
**Both fixed under `[Gen0fix2]`.** Every live eval path in the repo now forwards the value.
Five superseded early-April generations still omit it and are deliberately left frozen.
Full per-file evidence in §10–§12.

**Q: What about `post_processing`?**
**Restored, following the paper.** The DPCC paper defines post-processing as "modifying them
after the last denoising step, usually by solving an optimization problem" — which is exactly
`threshold = 0` in their gate, and exactly the line `7f09d3a` deleted. Without it the arm runs
the full in-loop schedule and is a byte-identical duplicate of `dpcc-r`, so the baseline meant
to represent *the alternative to* DPCC was running DPCC. Now fixed in all ten affected FM-PCC
eval scripts (§12), together with the gate change that lets `T = 0` actually fire (§12a).

---

## 1. The two questions

1. **Did we break it?** No.
2. **Does upstream have it?** Yes, at HEAD, today.

Both are settled by inspection, not inference.

## 2. Upstream at HEAD is character-identical to ours

Upstream has exactly **one** `Projector(` call site in the entire repository:

```python
# aux_repo/dpcc/scripts/eval.py:151-152
projector = Projector(horizon=args.horizon, transition_dim=trajectory_dim, action_dim=action_dim,
                      goal_dim=diffusion.goal_dim, constraint_list=constraints, normalizer=dataset.normalizer,
                      gradient=gradient, gradient_weights=[1, 0.5, 2], variant=diffuser_variant,
                      dt=delta_t, cost_dims=None, device=args.device, solver='scipy')
```

Ours (pre-fix, `FM-PCC/scripts/eval.py:205-206`) was the same characters, indented one level
deeper because we added a seed loop. No `diffusion_timestep_threshold` in either.

Exhaustive attribute trace, identical in both trees:

| role | location |
|---|---|
| default value `0.5` | `diffuser/sampling/projection.py:8` |
| only writer | `diffuser/sampling/projection.py:15` (constructor) |
| reader (gradient path) | `diffuser/models/diffusion.py:179` |
| reader (projection path) | `diffuser/models/diffusion.py:186` |

`grep "projector\."` returns only `.project`, `.gradient`, `.compute_gradient` — nothing mutates
the object after construction. There is no path from YAML to gate in either repo.

## 3. It used to work — and here is the commit that broke it

```
commit 7f09d3ab8713aeee45a2a9b0d56824630e0d884a
Author: ralfroemer99 <ralf-roemer@gmx.de>
Date:   Mon Dec 2 11:48:12 2024 +0100

    Removed unused configs
```

Diff on `scripts/eval.py` (upstream), abbreviated to the relevant hunks:

```diff
 # Constraint projection
-diffusion_timestep_threshold = config['diffusion_timestep_threshold']
 constraint_types = config['constraint_types']

             for variant_idx, variant in enumerate(projection_variants):
-                threshold = diffusion_timestep_threshold if not 'post_processing' in variant else 0
-                threshold = 0.25 if '0p25' in variant else threshold
                 gradient = True if 'gradient' in variant else False

                 # Create projector
                 projector = Projector(horizon=args.horizon, ..., normalizer=dataset.normalizer,
-                                        diffusion_timestep_threshold=threshold, gradient=gradient, ...)
+                                        gradient=gradient, ...)
```

Pre-cleanup source for reference (`git show 7f09d3a^:scripts/eval.py`, lines 130-131, 155):

```python
threshold = diffusion_timestep_threshold if not 'post_processing' in variant else 0
threshold = 0.25 if '0p25' in variant else threshold
...
projector = Projector(..., diffusion_timestep_threshold=threshold, gradient=gradient, ...)
```

**Three features died in that commit, not one.**

## 4. What each removed line did

### 4a. `diffusion_timestep_threshold = config[...]` → the YAML knob

Read the YAML value and forwarded it. Without it the projector takes its `0.5` default, so the
YAML entry became decorative. **This is Gen0 Fix2.**

### 4b. `... if not 'post_processing' in variant else 0` → the `post_processing` baseline

This is how `post_processing` was implemented: threshold `0` makes the gate
`t <= 0 * n_timesteps` true only at `t = 0`, i.e. **project the final denoised sample only** —
the textbook definition of post-processing, and exactly the semantics of the
`# self.only_last = only_last` vestige still commented out at
`diffuser/sampling/projection.py:14` in both repos.

With the line gone, `post_processing` matches no branch in the script and falls through to the
`dpcc-r` configuration. It is now **a byte-identical duplicate of `dpcc-r`** — verified by
`sha256(obs_all.npy)` across all three envs and both suffixes, in our DPCC baseline runs *and*
in all four `temp/0408/FMv3ODE` runs.

`post_processing` and `post_processing-tightened` are **still listed** in upstream HEAD's
`config/projection_eval.yaml`, under the comment `# Table 1:`. So the public repository at HEAD
cannot reproduce that row of the paper's Table 1 as a distinct method.

### 4c. `threshold = 0.25 if '0p25' in variant else threshold` → half of the Table 2 ablation

The `dt0p25` block that sets `delta_t = 0.25 * dt` survives (`eval.py:140-147`, both repos), so
the `dt*` variants still get their timestep change. What they lost is the paired threshold
override to `0.25`. Whether Table 2's variants were meant to carry both is a question for the
authors; the code before the cleanup applied both, the code after applies one.

## 5. Why this passed unnoticed for 20 months

At the moment of the cleanup, `config/projection_eval.yaml:26` already read:

```yaml
diffusion_timestep_threshold: 0.5
```

which is **exactly the constructor default**. Verified at `7f09d3a^`, at `7f09d3a`, and at HEAD.
(It had been `0.2` earlier and was raised to `0.5` in `c62a6ea`, 30 Oct 2024 — a month before
the cleanup.)

So for `dpcc-*`, `gradient`, `model_free` and `diffuser` the removal was **exactly
behaviour-preserving**, which is presumably why it read as dead code. It was not
behaviour-preserving for `post_processing` (needed `0`) or the `0p25` variants (needed `0.25`).

And the knob stays invisible until someone sets the YAML to something other than `0.5`. As far
as this repo's history shows, **we are the first to do that.** The result was three of our jobs
at θ ∈ {0.05, 0.1, 1.0} producing byte-identical trajectories.

## 6. Are the published DPCC results wrong?

**No evidence of that, and it is not the claim here.**

- The paper's results were produced by the pre-cleanup code, which wired the threshold and
  implemented `post_processing` correctly. The cleanup came afterwards, and the yaml value at
  that time made it a no-op for the arms that dominate the tables.
- Anything run at θ = 0.5 — before or after the cleanup — is correct **and** correctly labeled,
  because 0.5 is what both the config and the default say.

What *is* true, and checkable by anyone:

- **Upstream HEAD cannot vary `diffusion_timestep_threshold`.** Editing the YAML changes nothing.
- **Upstream HEAD cannot reproduce `post_processing`** as a distinct method; it silently returns
  `dpcc-r`.

That is a latent regression in a public research repo, not a defect in its published numbers.
Worth reporting upstream; the fix is the three lines their own history already contains.

## 7. What this means for FM-PCC

| claim | status |
|---|---|
| We introduced the bug | **False** — copied verbatim from post-`7f09d3a` upstream |
| The DPCC projection math is corrupt | **False** — projector, solver and gate are all correct |
| Our θ=0.5 DPCC results are wrong | **False** — correct and correctly labeled |
| Our θ≠0.5 DPCC results are wrong | **True** — they ran at 0.5; the savepath tag is wrong |
| Our `post_processing` columns are a separate method | **False** — duplicates of `dpcc-r`, everywhere, all generations |
| FMv3ODE / Gen12-HF / visual / UAV paths affected | **False** — those forward the value explicitly |

Two of these are inherited from the same upstream commit. Fix2 repairs the first
(§4a). The second (§4b) is left as a decision, with upstream's own pre-cleanup line as the
reference implementation:

```python
threshold = args.diffusion_timestep_threshold if 'post_processing' not in variant else 0
```

Restoring that one line would make `post_processing` a real method again in FM-PCC — and would
be the natural thing to send upstream as a patch.

## 8. Reproduction commands

```bash
cd /workspaces/aux_repo/dpcc

# the call site at HEAD, no threshold argument
sed -n '151,152p' scripts/eval.py

# the only writer of the attribute
sed -n '8p;15p' diffuser/sampling/projection.py

# the commit that removed the wiring
git show 7f09d3a -- scripts/eval.py

# the working version, before the cleanup
git show 7f09d3a^:scripts/eval.py | sed -n '130,131p;155p'

# the yaml has said 0.5 throughout
git show 7f09d3a^:config/projection_eval.yaml | grep diffusion_timestep_threshold
git show HEAD:config/projection_eval.yaml    | grep diffusion_timestep_threshold
```

---

## 9. How to check any script in ten seconds

There is exactly **one** thing to look at. The gate never reads the config; it reads the
projector object:

```python
# <generation>/models/diffusion.py   (every generation, without exception)
… t <= projector.diffusion_timestep_threshold * self.n_timesteps          # DPCC / diffusion
… int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3) # FM v3 line
… loop_idx >= (1.0 - projector.diffusion_timestep_threshold) * …           # HardFlow
```

and the projector attribute has exactly **one writer** — the constructor
(`sampling/projection.py:15`), fed by a parameter that **defaults to 0.5** (`:8`). Nothing
anywhere mutates a projector after construction (`grep "projector\."` returns only `.project`,
`.gradient`, `.compute_gradient`).

Therefore:

> **A script honours the threshold if and only if its `Projector(...)` call passes
> `diffusion_timestep_threshold=`. If the keyword is absent from that call, the script runs at
> 0.5 no matter what the YAML says.**

One-liner that answers it for the whole repo:

```bash
python3 - <<'EOF'
import os,re
SKIP=('Archived_Codes','aux_repo','.git','third_party')
for dp,dn,fn in os.walk('.'):
    dn[:]=[d for d in dn if d not in SKIP and not d.startswith('.')]
    for f in (x for x in fn if x.endswith('.py')):
        p=os.path.join(dp,f); src=open(p,errors='replace').read()
        for m in re.finditer(r'(?<![\w.])Projector\s*\(', src):
            i=m.end(); d=1
            while i<len(src) and d:                       # walk to the matching ')'
                d += (src[i]=='(') - (src[i]==')'); i+=1
            call=src[m.start():i]
            if '"""' in call or "'" == src[m.end()]: continue   # skip docstring mentions
            print(('OK  ' if 'diffusion_timestep_threshold' in call else '**NO**'),
                  f"{p}:{src[:m.start()].count(chr(10))+1}")
EOF
```

Grepping the *file* for the string is not enough — several files mention it in a watch list or
a docstring while the call site still omits it. The call must be parsed.

---

## 10. Does FMv3ODE have this bug? — No, and here is the exact code

`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`:

```python
 54:  diffusion_timestep_threshold = config.get('diffusion_timestep_threshold', 0.5)
      #                              ^ read from config/projection_eval.yaml

241:  projector = Projector(horizon=args.horizon, …, device=args.device, solver='scipy',
242:                        diffusion_timestep_threshold=diffusion_timestep_threshold)
      #                     ^ forwarded -- this line is what upstream 7f09d3a deleted
```

and the gate that consumes it:

```python
# flow_matcher_v3_ode_selectable/models/diffusion.py:207-208
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
```

Closed loop: YAML → local → constructor → attribute → gate.

**Empirically confirmed, same benchmark, same day** (`temp/0408/FMv3ODE/`, K=20, seed 6):

| arm | θ = 0.5 | θ = 0.1 | θ = 0.05 |
|---|---|---|---|
| `dpcc-t-tightened` | 0.449 | **0.189** | **0.180** |
| `dpcc-c-tightened` | 0.473 | **0.189** | **0.177** |
| `diffuser` (`projector = None`, control) | 0.176 | 0.174 | 0.170 |

The projected arms collapse ~2.4×; the control is flat. That is what a live threshold looks
like. Contrast the DPCC baseline at θ ∈ {0.05, 0.1, 1.0}: byte-identical trajectories, 39/39
cells, three times over.

---

## 11. Every live FM-PCC eval path — audited at the call site

Parsed from the balanced `Projector(...)` call, not grepped from the file. `n = 16` real call
sites (docstring mentions excluded).

### Wired — the threshold reaches the gate

| generation / path | call site | value passed | source of that value |
|---|---|---|---|
| **Gen0 DPCC baseline** | `scripts/eval.py:222` | `args.diffusion_timestep_threshold` | `plan` block, `config/avoiding-d3il.py:842` — **`[Gen0fix2]`, this fix** |
| **Gen12 HardFlow** | `FM_v3_hardflow_test/eval_FM_v3_hardflow.py:339` | `dpcc_threshold` | `:76-77`, YAML + `DPCC_THRESHOLD` env override — `[Gen12fix8]` |
| FMv3 ODE-selectable | `FM_v3_ode_selectable_test/…:241` | `diffusion_timestep_threshold` | `:54` |
| FMv3 MeanFlow | `FM_v3_meanflow_test/…:346` | `diffusion_timestep_threshold` | config |
| FMv3 iMeanFlow | `FM_v3_imeanflow_test/…imeanflow.py:278` | `diffusion_timestep_threshold` | config |
| FMv3 iMF (ODE variant) | `FM_v3_imeanflow_test/…ode_selectable.py:275` | `diffusion_timestep_threshold` | config |
| FMv3 AlphaFlow | `FM_v3_alphaflow_test/…:383` | `diffusion_timestep_threshold` | config |
| FMv3 Drifting | `FM_v3_drifting_test/…:272` | `diffusion_timestep_threshold` | config |
| FMv3 UAV | `FM_v3_uav_test/eval_fm_uav.py:761` | `threshold` | `:752` |
| Visual aligning — DPCC | `diffuser_visual_aligning_test/…:279` | `threshold` | `:269` |
| Visual aligning — FM | `fm_visual_aligning_test/…:286` | `threshold` | `:276` |
| Visual aligning — iMF | `imf_visual_aligning_test/…:167` | `threshold` | `:157` |
| **Gen14 Visual-Mix** | `mix_visual_aligning_test/…:316` | `threshold` | `:297` |
| Visual avoiding — FM | `fm_visual_avoiding_test/…:393` | `diffusion_timestep_threshold` | config |

### NOT wired

| path | call site | status |
|---|---|---|
| **Visual avoiding — DPCC** | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:380` | ⚠️ **LIVE AND BROKEN** — see §11a |
| FM gen 1 | `FM_test/eval_FM.py:121` | older generation, rollback only |
| FM gen 2 | `FM_v2_test/eval_FM_v2.py:121` | older generation |
| FM gen 3 | `FM_v3_test/eval_FM_v3.py:121` | older generation |
| FM UNet v2 | `FM_Unet_v2_test/eval_FM_Unet_v2.py:121` | older generation |
| FM hp-tune | `FM_hp_tune_test/eval_FM_hp_tune.py:121` | older generation |
| projector unit test | `diffuser_visual_aligning_test/test_projector_b1.py:57` | standalone, harmless |

The five older FM generations are only wrong if someone runs them with a YAML threshold ≠ 0.5.
Per the copy-modify convention they are **not** edited by Fix2 — flagged, pending a call on
whether those generations are still active.

### 11a. New finding: the visual-avoiding DPCC baseline has the identical bug

```python
# diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:380-386
projector = Projector(
    horizon=args.horizon, transition_dim=trajectory_dim,
    action_dim=action_dim, goal_dim=diffusion.goal_dim,
    constraint_list=constraints, normalizer=proj_normalizer,
    gradient=gradient, gradient_weights=[1, 0.5, 2],
    variant=diffuser_variant, dt=delta_t, cost_dims=None,
    device=args.device, solver='scipy')
#                      ^ ends here -- no diffusion_timestep_threshold
```

Same symptom, same mechanism, and **the savepath is tagged with `T` anyway** —
`config/avoiding-d3il-visual.py:242` builds
`H{horizon}_K{flow_steps_v3}_M{…}_T{diffusion_timestep_threshold}_D{diffusion}/`. So this path
produces `T`-labelled folders that all ran at 0.5, exactly like the Gen0 baseline did.

Note `:191` lists `'diffusion_timestep_threshold'` in `_SAMPLING_OVERRIDE_KEYS`, which is about
reconciling the eval config against the checkpoint's stored diffusion config. That is a
**different consumer** and does not help: no model class anywhere in this repo stores the
threshold — verified by repo-wide grep, every gate reads `projector.diffusion_timestep_threshold`.

**Fixed under `[Gen0fix2]`** — hunks 3 & 4 in the changelog. Wiring mirrors what the FM sibling
in the same generation already had (`fm_visual_avoiding_test/eval_fm_visual_avoiding.py:140`,
`:400`), reading `config/visual_avoiding_eval.yaml`. Any earlier `T ≠ 0.5` result from this
path is mislabeled the same way the Gen0 ones were and needs re-running.

---

## 12. The second axis: which paths restored `post_processing`

Independently of the threshold, `post_processing` needs the upstream line from §4b. Checking
which of our scripts carry it:

> **Correction (2026-08-04, after the gate audit):** "the script wires it" is **not** the same
> as "the arm works". Wiring `threshold = 0.0` only produces one final projection if the gate
> that consumes it fires at `T = 0`. Two of the three gate forms in this repo do; the third
> returns **zero** projections, which turns `post_processing` into `diffuser`. The `handled?`
> column below therefore has a second half. Full analysis: `GATE_ARITHMETIC_audit.md`.

| path | script wires it? | gate fires at T=0? | net |
|---|---|---|---|
| Visual aligning — DPCC | ✅ `:269` | ✅ form C | ✅ 1 final projection |
| Visual aligning — iMF | ✅ `:157` | ✅ form A (terminal guard) | ✅ 1 final projection |
| FMv3 UAV | ✅ `:752` | ✅ form A | ✅ 1 final projection |
| **Visual aligning — FM** | ✅ `:276` | ❌ **form B** | ❌ **0 projections — equals `diffuser`** |
| **Gen14 Visual-Mix** | ✅ `:297` | per engine: C / A / A / **B** | ✅ `diffusion`,`mf`,`af` — ❌ **`fm` arm** |
| **Gen0 DPCC baseline** | ❌ | no branch — falls through to `dpcc-r` |
| FMv3 ODE-selectable | ❌ | no branch |
| Gen12 HardFlow | ❌ | no branch |
| FMv3 MeanFlow / iMF / AlphaFlow / Drifting | ❌ | no branch |
| Visual avoiding — FM and DPCC | ❌ | no branch |

> **The visual line independently re-derived upstream's pre-cleanup logic** — `threshold = 0.0
> if 'post_processing' in variant else …` is line-for-line the semantics of
> `7f09d3a^:scripts/eval.py:130`, rediscovered rather than copied. Those five paths report a
> genuine `post_processing` baseline.
>
> **Every avoiding-family path does not**, so their `post_processing` and
> `post_processing-tightened` columns are duplicate `dpcc-r` columns. Confirmed byte-identical
> (`sha256(obs_all.npy)`) in the Gen0 baseline runs and in all four `temp/0408/FMv3ODE` runs.

> **Decision (2026-08-04, revised): restore it, following the paper.** The paragraph the paper
> devotes to post-processing makes it the *contrast case* — the approach that ignores the data
> likelihood and therefore drifts off-distribution — against which in-loop projection is argued.
> With the branch missing, that baseline **is** in-loop projection. The row meant to represent
> the alternative was running the method itself, which is not a naming quibble but a broken
> comparison.

Applied in all ten affected FM-PCC eval scripts, in upstream's own pre-cleanup form:

```python
threshold = 0.0 if 'post_processing' in variant else <yaml threshold>
```

### 12a. The gate had to change too

Restoring the branch alone would not have worked on every path. Under the bare-float gate
(form B, §11), `threshold = 0` yields **zero** projections, not one — so `post_processing`
would have become `diffuser` rather than `dpcc-r`. Five packages were converted to the guarded
form (`flow_matcher_v3`, `flow_matcher_v3_hardflow`, `fm_visual_aligning`, `fm_visual_avoiding`,
`mix_visual_aligning/models/fm_diffusion.py`), which is what the rest of the FMv3 line already
used. Behaviour is unchanged at every threshold used to date — see
`GATE_ARITHMETIC_audit.md` and changelog §2c.

That change also removes a Gen14 comparability defect: the four Visual-Mix engines were running
**three different gate arithmetics** (`diffusion`→C, `mf`/`af`→A, `fm`→B) in an experiment whose
premise is that only the generator differs.

---

## 13. Summary table — what is true where

| | threshold honoured | `post_processing` = one final projection |
|---|---|---|
| **upstream DPCC, pre-`7f09d3a`** | ✅ | ✅ |
| **upstream DPCC, HEAD (today)** | ❌ | ❌ |
| **FM-PCC, before Fix2** | ❌ Gen0 + Gen9-DPCC; ✅ elsewhere | ❌ everywhere except visual-aligning DPCC/iMF and UAV |
| **FM-PCC, after Fix2** | ✅ **every live path** | ✅ **every live path** |
| FM-PCC frozen generations (v1, v2, v3, UNetv2, hp-tune) | ❌ frozen on purpose | ❌ frozen on purpose |

**Net after Fix2:** every live FM-PCC eval path forwards the threshold *and* implements
`post_processing` as the paper defines it. Upstream HEAD does neither. The five remaining
omissions are superseded generations left frozen deliberately — the shared YAML now reads `1`,
so wiring them would silently change archived rollback baselines.

**What we did not touch:** `aux_repo/dpcc` (read-only reference), and form C's `+1` — DPCC's
gate yields `floor(T·K)+1` where the FM line yields `K − int((1−T)·K)`. That is DPCC's published
behaviour; changing it would break comparison with the paper. The operating rule instead is
**match runs on `n_active`, never on `T`**.

**Worth sending upstream.** The fix for `ralfroemer99/dpcc` is the three lines its own history
already contains, at `7f09d3a^:scripts/eval.py:130-131,155`. As it stands, HEAD cannot vary
`diffusion_timestep_threshold` and cannot reproduce the `post_processing` row of Table 1, while
still listing that variant in `config/projection_eval.yaml` under `# Table 1:`.
