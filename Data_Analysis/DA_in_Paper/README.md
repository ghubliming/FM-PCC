# DA_in_Paper — the official data and figures of the thesis

**Live since 2026-09-14.** Everything the thesis states as a number or shows as a figure comes from
here. Analyses elsewhere in the repository remain the working record; this folder is what the thesis
uses.

```
DA_in_Paper/
├── analysis/INDEX.md     every thesis result → the analysis it is taken from, and its protocol
├── plotting/             all figure code: builders, SVG and matplotlib tooling, export to drafts
├── data/                 plotting-ready extracts (JSON) written by plotting/extract/
└── figures/              all thesis figures, by group: da/ demo/ env/ schematic/  (+ MANIFEST.md)
```

## The rule for figures

**Every thesis figure is produced in `figures/`. A draft holds copies only.**

```bash
python3 plotting/make_figs.py            # build / refresh the store
plotting/svg/svg2pdf.sh                  # SVG -> PDF (needs rsvg-convert, inkscape or cairosvg)
python3 plotting/export_to_draft.py v3   # copy the figures v3 uses into Working_Space/v3/figures/
```

- A figure is never drawn or edited inside a draft. Change it here, rebuild, export.
- The export copies only what the draft's `.tex` actually includes, and records each copy with its
  SHA-256 in `<draft>/figures/EXPORTED.md`.
- A figure copied in from elsewhere (a report, a diagnostic plot) is checked against
  `/workspaces/aux_repo/` first and registered with its provenance in `plotting/sources.py`. Images that
  must never be used — the repo-root `figures/avoiding*.png`, identical to the DPCC authors' copies — are
  listed there under `EXCLUDED`.

## The rule for numbers

A thesis number is quoted from an analysis listed in [`analysis/INDEX.md`](analysis/INDEX.md). When the
data behind a number changes, the analysis is updated first, then the index row, then the draft.

## Related

- Readiness of each environment for writing: `logs_in_develop/Writing/Working_Space/data_status/`
- Who writes which chapter: `logs_in_develop/Writing/Working_Space/DRAFT_OWNERSHIP.md`
- Names used in figures and text: `logs_in_develop/Writing/Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md`
