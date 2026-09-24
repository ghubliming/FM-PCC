---
name: thesis-v4-conclusion-rules
description: Author's rules for the thesis endings (Ch 7 Conclusion, Ch 8 Discussion, appendix), 2026-09-24 — the model ordering to state, HF helps only where the budget is large, the controller matters; no Interpretation/Negative/Threats sections ("suicide"); appendix = facts + web link for pure data; legacy notes verified before use
metadata:
  type: feedback
---

Rules the author gave at the v4 init (2026-09-24), held to the Ch 6 data at v3.98:

- **Model ordering to state:** the average-velocity (MeanFM-family) models > FM ≳ diffusion. FM and diffusion are **level on outcome** (aligning: no ordering; avoiding: both 1.000) and differ in **cost**; never write "FM is better" where nothing shows it. CI-MeanFM is **not always ahead of MeanFM**: ahead by one episode on avoiding, behind on aligning; "the analytic objective leads or holds in every environment" is the safe sentence.
- **Endpoint projection (HF) helps on the hard task that needs the budget** (D3IL-aligning, K20: better on the constraint), **not on D3IL-avoiding** (K1: no guiding step, nothing to win); on UAV-corridor it is a **trade-off, no dominance** (author v3.96 supersedes the older "or maybe UAV" note).
- **General conclusion:** human demonstrations → average-velocity models lead; hard tasks → model + projection together; simple generated demonstrations → FM is enough, objectives equivalent; the plant and the controller matter (pillars, s-curve; our cascaded geometric controller is the better one, MuJoCo MPC is not).
- **Reproducibility appendix = environment facts only.** No "how we verified", no justification paragraphs, no "GPUs per job, held exclusively" row; the artefact-name map and corpora-of-record tables are **writing material, never thesis** (`v4/notes/`).
- **Pure-data appendix sections:** keep the reading, mark the head (`\longdata`), tables behind `\ifappendixfull` → a web link later.
- **Shape of the endings (author, v4.1, 2026-09-24):** Ch 7 = *Conclusion* with Summary + RQ answers only; Ch 8 = *Discussion* with Limitations, Towards Deployment, Future Work — about five pages together. **No Interpretation, no "Negative and Inconclusive Results", no "Threats to Validity" sections**: the author called them duplicates of Ch 6 and "suicide" (bone-template sections, not a must); they are archived in `v4/withheld/20260924_v4.1_archive/`. Limitations is fine. The quadrotor-demonstration simulator in Future Work is **NVIDIA Isaac Sim**.
- **Appendix shape (author, v4.1):** A = *Supplementary Material* ("Derivations" was rejected as a misnomer): A.1 training-time sampling laws (kept), A.2 the X2 dimensions figure (not reproducibility material). B = *Extended Results* with exactly five sections the author named: D3IL-avoiding executed paths, D3IL-aligning executed paths, UAV-corridor flights along the corridor, UAV-corridor under every selection rule (keep its `\longdata` marker), D3IL-avoiding at twenty episodes; everything else archived (`v4/withheld/20260924_v4.1_archive/`). C = *Compute Environment* only. Index changes are **notified** to v3 through `cross_draft/`, never edited in v3's chapters; dangling Ch 5/6 references are bridged by alias labels on the Extended Results heading until v3 rewrites the sentences.
- **Legacy notes** (prompt points, `notes.txt`) are *maybe outdated*: verify each against the current draft before applying, and record what was applied/superseded in the changelog.

**Why:** the endings must carry the storyline ([[thesis-results-storyline-guide]]) without leveling the flow models with the baseline or overclaiming HF; the author was explicit that older notes may be stale.

**How to apply:** when writing or revising Ch 7/8 or the appendix in v4, check every claim against these lines and `Working_Space/GUIDE_20260924_results_storyline_author.md`; keep [[thesis-prose-style]] and [[pareto-definition-of-good]]. Related: [[thesis-draft-ownership]].
