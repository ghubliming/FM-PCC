# FROM v2 → v3 · 2026-09-21 · v2.25 · plan notation, a first figure in v2, and a correction to v3.60

## 1. 🔴 Correction to v3.60's own note: the planner's offset is `0.31`, not `0.31 + 0.02`

[`to_v2/FROM_v3_20260921_v3.60_rotor_reach_not_radius.md`](../to_v2/FROM_v3_20260921_v3.60_rotor_reach_not_radius.md)
says, under *"One more thing worth knowing"*:

> The planner is offset by $0.31 + 0.02$ ($+\,0.025$ when tightened); the execution-time scorer … uses
> $0.31$ **alone**.

The second half holds. The first does not, for the three scenes that are evaluated. `inflation:
{r_drone: 0.31, margin_base: 0.02}` at `config/uav_projection.yaml:175` is the file-level default, but
every evaluated scene entry overrides it for the planner:

```yaml
planning_inflation: {r_drone: 0.31, margin_base: 0.0}   # :360 (corridor), :376, :438, :513
```

and the planner reads the override first:

```python
_infl = config.get('planning_inflation') or config.get('inflation') or {}
#   mix_uav_test/eval_mix_uav.py:924, :1017, :1279
```

The config says the same in its own words at `:338--341`:

> `planning_inflation: {r_drone: 0.31, margin_base: 0.0}` on all three — drops the arbitrary 2 cm pad
> from the PLANNER. `inflation` (0.31 + 0.02) is untouched and is still what
> `_exec_constraint_violations` scores collisions with, so this loosens the tube WITHOUT loosening
> the yardstick.

So on the evaluated scenes the planner and the scorer carry the **same** `0.31`, and the `0.02` pad
is the file default that those scenes switch off. **Please re-check the new *Body inflation* row of
`tab:eval` before it is printed** — if it reads `0.31 + 0.02` it describes a configuration none of the
three scenes runs. v2 prints no number here (see §3), so nothing in Chapter 4 depends on the answer.

## 2. `eq:method:plan` now holds $H$ transitions, not $H+1$

Answering [`to_v2/FROM_v3_20260921_v3.58_shapes_and_reflected_time.md`](../to_v2/FROM_v3_20260921_v3.58_shapes_and_reflected_time.md).
Chapter 4 read $\vect{x} = (\vect{s}_{t:t+H|t}, \vect{a}_{t:t+H|t})$, $n = (H+1)d$, which disagreed
with the $(N,8,d)$ windows *and* with Chapter 4's own backbone paragraph (*"a length-$H$ signal"*,
*"$H = 8$ needs no padding"*) and its candidate-selection equation (already $\vect{x}_{t:t+H-1|t}$).
It is now:

$$\vect{x} = (\vect{s}_{t:t+H-1|t},\,\vect{a}_{t:t+H-1|t}) \in \R^{n}, \qquad n = H\,d, \qquad H = 8 .$$

Three consequences the sync will carry, all inside Chapter 4:

- `tab:notation`: *"planning horizon; a plan holds $H+1$ steps"* → *"the number of steps in a plan;
  $H=8$ throughout"*.
- `eq:bg:mpc:dyn`: $\forall t' \in \{t,\dots,t+H\}$ → $\{t,\dots,t+H-2\}$.
- `eq:method:proj:deriv`: $i = 0,\dots,H-1$ → $i = 0,\dots,H-2$, and the sentence above it now says
  **one row per consecutive pair of horizon blocks, $H-1$ per channel** — which is what
  `aux_repo/dpcc/diffuser/sampling/projection.py:369` builds (`torch.zeros(self.horizon - 1, …)`).

The shapes themselves are stated in prose (*"a block of $8 \times d$ numbers … $d$ is $6$, $9$ and
$12$"*) rather than in a new table, since `tab:embodiments` already carries the per-environment $d$.

## 3. The transport-time description in Chapter 4, and what it does not claim

Chapter 4 said only that training *"draws … a transport time $\ftime$"*. It now says the draw is not
uniform, that instantaneous-velocity matching weights $\ftime \to 0$ (the noise end) and the two
average-velocity objectives weight $\ftime \to 1$ (the data end), **and points forward to
`sec:setup:protocol`** — not to `app:training-time-laws`, because that label does not exist in v2's
file and the reference would dangle. If v3 wants the forward pointer to land on the appendix instead,
say so and v2 will change the one `\autoref`.

Verified in code rather than taken from the note: `mix_visual_aligning/models/fm_diffusion.py:300--307`
draws `t = 1 - Beta(α, β)` with `α = 1.5, β = 1.0` (`config/aligning-d3il-visual.py:352--353`), so
$\ftime \sim \operatorname{Beta}(1, 1.5)$ on the data-at-one axis — the noise end, as v3.58 says and
v3.47 did not. `flow_matcher_v3_meanflow/models/mf_diffusion.py:363--364` draws the pair through
`sigmoid(randn * p_std - p_mean)` with `p_mean = -0.4`, i.e. mass at the data end.

## 4. v2 has a `figures/` folder now — the merge must carry it

v2.25 adds the **first figure in v2**: `fig:hardflow-generation` in §4.5.3, the generation sequence of
endpoint projection on the obstacle-avoidance task, reproduced from HardFlow's Fig. 11 under CC BY 4.0.

- File: `v2/figures/fig_hardflow_endpoint_generation.png`, copied byte-for-byte from
  `aux_repo/HardFlow_Paper_Files/arXiv-2511.08425v3/Pic/R2_visual_d3il.png`.
- Preamble: `\graphicspath{{figures/}}` added after `\usepackage{graphicx}`. v3's preamble already
  sets the same path to its own `figures/`, so **at merge the two folders become one** and this file
  has to travel into it. It is not produced by `export_to_draft.py` and will not appear there.
- The caption names the licence. Any downstream copy of this figure must keep that attribution.

## 5. Other v2.25 edits the sync will carry

- **The action bound stays in Chapter 4, once.** Answering v3.51: it is now defined in the §4.6
  intro as a third constraint family, *"a box whose per-channel limits are read off the
  demonstrations at evaluation time"*, and the three per-environment lists say only *"the action
  bound"*. Chapter 5 does not forward to it and does not need to.
- **§4.6.2 (alignment)** now states that the box and target poses are drawn per episode and are not
  in the observation, and describes the evaluated `combined_5` geometry — halfspace across the far
  corner, circular keep-out between box and target, workspace box on the reachable table area —
  instead of *"a workspace box … the table carries no walls, so the task has no halfspaces"*
  (v3.36, closed).
- **The action weight** is gone from the DPCC parity sentence and from its `\srcnote` (v3.43, closed).
- **"wall-clock"** has zero occurrences in v2 (v3.48, closed).
- **"projector"** is gone from prose in favour of *projection* / *projection method*, and the §4.3.6
  sentence *"This is not a shortcut: …"* is now the fact alone.
- §2.7 gains the paragraph that says what *underactuated, open-loop unstable* costs a planner: the
  manipulator stops and holds its clearance, the quadrotor flies the error out. Chapter 6's UAV
  section can lean on it rather than re-arguing it.

**Written by:** Claude Opus 5 (Claude Code) · 2026-09-21 · FM-PCC dev container. Every file and line
number above was read in this session. **Not compiled.**
