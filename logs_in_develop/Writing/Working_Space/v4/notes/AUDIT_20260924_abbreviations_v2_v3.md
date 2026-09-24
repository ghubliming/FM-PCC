# AUDIT — abbreviations, Ch 1–9 at v2.27 / v3.98 (v4.0, 2026-09-24)

**Asked (author, v4 init, point 5):** "Abbreviations need to recheck again for v2/3."
**Method:** every `\acro` of `parts/99_backmatter.tex` against every `\ac{}` use and every bare all-caps
token of `chapters/01–09` and `app_ntrial20_feasible.tex` (comments, `\srcnote`, `\dataref`, `\texttt`,
math and labels stripped). Script in the v4.0 session; counts below are of the v3.98 copies v4 inherits.

## 1 · The list is complete

| declared | `\ac{}` uses | first use | verdict |
| :-- | --: | :-- | :-- |
| DPCC | 104 | Ch 1 | ✅ |
| NFE | 1 | Ch 1 §1.1 (formal `$\mathrm{NFE}$` in Ch 4) | ✅ |
| ODE | 2 | Ch 1 | ✅ |
| JVP | 4 | Ch 2 | ✅ |
| MPC | 3 | Ch 2 | ✅ (bare "MuJoCo MPC" 40+ times is a product name, see §2) |
| MJPC | 3 | Ch 2 | ✅ declared "MuJoCo predictive control" |
| IK | 5 | Ch 2 | ✅ (spelled out in Ch 5–6, fine) |
| NLP, SLSQP | 1, 1 | Ch 4 | ✅ |
| UAV | 1 | Ch 3 | ✅ expanded there; the 240 bare uses are the scene names *UAV-corridor* etc. |
| FiLM | 1 | Ch 2 | ✅ |
| TUM | 0 | — | prints nothing with `printonlyused`; harmless |

Every `\ac{}` in Ch 5–9 is `DPCC`; every other abbreviation those chapters use is either a table short
form defined in Ch 5 §5.4 (FM, MeanFM, CI-MeanFM, Diffusion; S&C in `sec:setup:metrics:avoiding`), a
product or paper name (MuJoCo, ResNet, SLURM, NVIDIA, D3IL, BESO, DDPM, ACT, VAE, GPT), a unit or a
common hardware token (GPU, CPU, GB, GHz, RGB), or an equation label (RQ1–3). **Nothing needs adding for
the thesis to be readable, and v4.0 leaves `99_backmatter.tex` byte-identical to v2's.**

## 2 · Two consistency findings (owners' call)

1. **MJPC vs "MuJoCo MPC".** Ch 2 and Ch 4 introduce `\ac{MJPC}` ("MuJoCo predictive control", the
   published name of the software); Ch 5–6 and every table write **"MuJoCo MPC"** (44 bare uses). Both
   name the same controller (§4.3.6 `sec:method:mjpc`). v4 follows Ch 6's "MuJoCo MPC" in Ch 7–8 so the
   results chapters read alike. One form throughout is v2's and v3's decision; the cheapest fix is
   v3 writing `\ac{MJPC}` at its first Ch 5 use, or v2 adopting "MuJoCo MPC" in Ch 2/4.
2. **PD is not declared.** "joint-space PD law/controller" appears bare 9 times (first in Ch 4 §4.3.1,
   then Ch 5 Table 5.9, Ch 6, Ch 7). A standard control term; if v2 wants it in the list:
   `\acro{PD}[PD]{proportional--derivative}` and `\ac{PD}` at the Ch 4 first use.

## 3 · Not findings

- "S&C" is defined in Ch 5 (`sec:setup:metrics:avoiding`, Table 5.7 caption) and used in tables only. ✅
- $K$ has two meanings (diffusion denoising steps fixed at training; ODE solver steps chosen at
  inference): stated in Table 4.1 (`tab:notation`, v2.25) and again at Ch 6 §6.1.1 — the v3.28 glossary
  request is met without a separate glossary (see `OPEN_20260924_v4_open_items.md`, item 8).
- The acronym list stays in the front matter (v2.24 question): the TUM template prints the `acronym`
  environment there and has no appendix slot; v4 recommends leaving it.

Claude (Fable 5.1, Claude Code), v4 · 2026-09-24.
