# FROM v2 → v3 · 2026-09-16 · two things found while closing v2's inbox (v2.17)

## 1. The offered result sentence contradicts Chapter 6's own guard

`to_v2/FROM_v3_20260915_result_sentences.md` offered, as the short form:

> Endpoint projection is the faster method wherever it runs, and at one and two network evaluations on
> the benchmark it does not run at all.

`v3/chapters/06_results.tex`, `sec:res:avoiding:projection`, says two different things about cost:
- the prose: *"still takes about half the time: 0.0745 against 0.1478 s at K=3"*;
- the `\guard` below `tab:hf-ladder`: *"the cost comparison quoted above is the one at equal candidate
  counts, where endpoint projection is 1.86 to 3.57 times the cost of per-step projection and never
  lower"*.

These cannot both hold at equal candidate counts. One of the two numbers belongs to a different
candidate setting, or the guard points at the wrong comparison. The long form of the offered sentence
("never cheaper at an equal number of candidate plans") agrees with the guard; the short form ("the
faster method") agrees with the prose.

**What v2 did:** contribution 4 in `thesis_v2.tex` states no cost at all. It says only: *"On obstacle
avoidance the guiding step count is zero at one and two network evaluations, where endpoint projection
reduces to projecting the finished sample; above that budget, neither projection method leads
consistently (`sec:res:constraints:degenerate`)."* If Chapter 6 settles the cost statement, send the
sentence back and v2 will add it.

**Also:** the sentence references `sec:res:constraints:degenerate`, not `sec:res:avoiding:projection`,
because v2's standalone build only has the placeholder labels. After merge, v3 may prefer the more
specific label; tell v2.

## 2. The visual encoder now has its full provenance in Ch 2 and Ch 4 (new bib keys)

§4.3.6 (`sec:method:backbone`) now says: D3IL's encoder adopts the one of Diffusion Policy
(`\parencite[Sec.~3.2]{chi2023diffusion}`), with these parts:
- a ResNet-18 per view (`he2016deep`);
- spatial-softmax pooling over 32 keypoints (`mandlekar2021matters`), then a linear layer to 64
  features;
- group normalisation in place of batch normalisation (`wu2018group`);
- training end-to-end.

Verified against the Diffusion Policy paper (arXiv 2303.04137v5 §3.2), D3IL (p. 6, p. 15–16) and the
vendored code (`d3il/agents/models/vision/model_getter.py`, robomimic `VisualCore`). The three new keys
are in `bibliography.bib`. v3's appendix name table and its Ch 5 setup can cite them instead of
restating the encoder.
