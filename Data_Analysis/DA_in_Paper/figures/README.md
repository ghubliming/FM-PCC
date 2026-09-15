# `figures/` — every thesis figure

Built by `../plotting/make_figs.py`; inventory in `MANIFEST.md` (generated). Drafts receive copies via
`../plotting/export_to_draft.py` — never edit a figure inside a draft.

| group | holds |
| :-- | :-- |
| `da/` | charts computed from evaluation data |
| `demo/` | raw demonstrations of behaviour: plans, rollouts, frames |
| `env/` | environments and constraint sets: renders, scene and constraint panels |
| `schematic/` | diagrams of the method |

SVG and PDF of the same figure sit side by side (`../plotting/svg/svg2pdf.sh`). A figure name is unique
across groups.
