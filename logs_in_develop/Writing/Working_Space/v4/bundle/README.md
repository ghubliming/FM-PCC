# `bundle/` — one flat `.tex`, built from the split v4 draft

Ported from `v3/bundle/` (v3.94) and changed in one rule, the author's for v4 (2026-09-24): **the
working bundle is built without v2 but with v3 and v4.** Every chapter v2 owns (Ch 1–4) is collapsed
to its numbered headings and a grey box; the v3-owned chapters (5–6) and v4's own (7–9) are inlined in
full. Numbering and cross-references match the full build. Every bundle header names the **v3 revision
v4 branched from and the v2 revision that v3 carries** (from `../inherited/SYNC_STATE.json`).

```bash
python3 bundle/make_bundle.py                    # BOTH variants: _new (annotated) and _full_clean (no notes)
python3 bundle/make_bundle.py --annotated-only   # just _new
python3 bundle/make_bundle.py --clean-only       # just _full_clean
python3 bundle/make_bundle.py --full             # annotated, nothing collapsed (_full)
python3 bundle/make_bundle.py --v4-only          # annotated, v2 AND v3 chapters collapsed (_v4only)
python3 bundle/make_bundle.py --appendix-short   # the appendix's LONG DATA tables hidden (\appendixfullfalse)
python3 bundle/make_bundle.py --svg-package      # render SVGs via \usepackage{svg} (Overleaf has Inkscape)
python3 bundle/make_bundle.py --verify           # prove the newest bundle matches the tree, byte for byte
python3 bundle/make_bundle.py --list | --prune N
```

| variant | file | what it is |
| :-- | :-- | :-- |
| **annotated** (default) | `thesis_v4_<stamp>_new.{tex,zip}` | v2 chapters collapsed, v3 + v4 built; every drafting mark visible (`\srcnote`, `\dataref`, `\guard`, `\hole`, `\provisional`, `\outdated`, `\flawed`, `\longdata`). **The working copy.** |
| **clean** | `thesis_v4_<stamp>_full_clean.{tex,zip}` | The whole thesis, no notes, for checking the layout: `\guard`/`\provisional` print as plain prose, every other mark prints nothing, a dead block prints plain. |

Everything else — the figure guard (`\fmpccgraphic`, placeholder boxes for SVG-only figures), the
verification before writing, `--verify` folding nested `\input`s, the Overleaf zip (pdfLaTeX + Biber) —
is v3's, unchanged. Rules: never edit a file in this folder; nothing reads a bundle back; the header of
every bundle lists its sources with SHA-256 prefixes.

**Never compiled here** — the container has no TeX toolchain. The first real build is yours.
