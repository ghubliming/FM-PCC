---
name: read-papers-with-python314-pypdf
description: How to read the PDFs in aux_repo/PAPERS — pypdf under python3.14, not the Read tool (no poppler in this container)
metadata:
  type: reference
---

The Read tool **cannot open PDFs here** — `pdftoppm`/poppler is not installed, and neither is
`pdftotext`. Extract text instead with **`python3.14`** (not `python3`, which is 3.13 and has no
site-packages): `pypdf` is installed at `~/.local/lib/python3.14/site-packages`.

```bash
python3.14 -c "
from pypdf import PdfReader
r = PdfReader('/workspaces/aux_repo/PAPERS/in Proposual/DPCC.pdf')
print(r.pages[2].extract_text())"
```

Extraction is good enough for prose and for reading equations (LaTeX-built PDFs come out
readable, if unspaced). There is **no network** for new pip installs, and no LaTeX toolchain
(`pdflatex`/`latexmk`/`biber` are all absent), so thesis `.tex` work here can be checked
structurally but never compiled.

Some papers ship better sources than the PDF: `PAPERS/Recommand_Paper/HF/` has HardFlow's full
`main.tex` **and** `reference.bib` — the bibliography bootstrap named in
[[master-thesis-writing-tum]].

**Always verify a citation against the PDF's own title page.** Entries written from memory have
been wrong in both title and authors.
