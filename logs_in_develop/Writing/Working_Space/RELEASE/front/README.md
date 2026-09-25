# `front/` — optional author-supplied front-matter text for a release

`tools/make_release.py` reads this folder at build time. It is empty by design; nothing here is a draft.

| file | effect when present |
| :-- | :-- |
| `acknowledgments.tex` | plain LaTeX paragraphs; the Acknowledgments page is included in the release with this text (between the template's heading and its `\cleardoublepage`). Without this file the page is dropped, unless `--acknowledgments` asks for the empty page. |

The metadata printed on the cover, title and declaration pages (`\getAuthor`, `\getSupervisor`,
`\getAdvisor`, `\getSubmissionDate`, `\getTitleGer`) is **v2's** (`v2/thesis_v2.tex`, the `\newcommand*{\get…}`
block) and is filled there, not here.
