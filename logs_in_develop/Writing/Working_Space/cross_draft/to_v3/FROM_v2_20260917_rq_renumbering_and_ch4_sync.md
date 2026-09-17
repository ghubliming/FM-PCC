# FROM v2 → v3 · 2026-09-17 · v2.22: RQ renumbering, a removed remark, and Chapter 4 edits to sync

v2.22 changed material that v3 carries in synced copies (`v3/chapters/01_introduction.tex`,
`04_method.tex`) or answers in its own chapters.

## 1. There are now three research questions, not four

| v2.21 | v2.22 |
| :-- | :-- |
| RQ1 Generative model | RQ1, rewritten around the flow-matching concepts (velocity field, transport ODE, budget as a solver argument, average-velocity objectives) |
| RQ2 Budget | **deleted** by author decision; the budget mechanism belongs to RQ1 |
| RQ3 Constraints | **RQ2**: at a budget with guiding steps, endpoint projection versus per-step (iterate) projection on one feasible set and one solver |
| RQ4 Transfer | **RQ3**: camera observation, alignment through contact, underactuated quadrotor — stated against the simplicity of obstacle avoidance |

**For v3:** `chapters/08_conclusion.tex` answers RQ1--RQ4 and `chapters/01_introduction.tex` is the
synced copy of the old list. After the next sync, fold the former RQ2 answer (budget) into RQ1 and
renumber the remaining two. v2 also updated the two in-text references in Chapter 4
("the mechanism behind RQ2" → "the budget mechanism of RQ1"; "what RQ4 is about" → RQ3).

## 2. `rem:pidname` no longer exists

The remark "The controller is not a PID" and the three "Deviation" blocks in the quadrotor
controller section were removed (they compare our instantiation with the published one, which the
author has ruled out). `v3/chapters/04_method.tex:1293,1295` still defines and references the label;
the next `tools/sync_v2.py` run removes both together. The facts are kept as plain statements,
including that holding the commanded angular velocity at zero drops the attitude feed-forward and
with it the exponential-stability statement of `\parencite[Prop.~3]{lee2010geometric}`.

## 3. New bibliography key and new labels

- `mellinger2011minimum` (ICRA 2011) — the cascaded position/attitude structure and the fixed rotor
  allocation matrix; NO LOCAL COPY. Bibliography 52 → 53.
- New labels: `eq:method:engine:aftgt` (the consistency-interpolated training target, which fills the
  old `\hole` in `sec:method:alphaflow`) and `tab:related-loops` (Chapter 3).
- `tab:embodiments` is now `tabularx` at `\small`.

## 4. Upstream sources now stored

- α-Flow LaTeX source: `aux_repo/PAPERS/auxiliary_papers/DGM/AlphaFlow_src/` — cited as Def. 1,
  Alg. 1 (training), Alg. 2 (curriculum schedule), Thm. 1.
- HardFlow source was already at `aux_repo/PAPERS/Recommand_Paper/HF/`; Fig. 11 is the D3IL
  generation-process figure, now cited in `sec:method:projection`.
