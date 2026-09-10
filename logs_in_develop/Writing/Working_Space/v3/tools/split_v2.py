#!/usr/bin/env python3
"""Split a monolithic ``thesis_v2.tex`` into the v3 part/chapter files.

Why this exists
---------------
v2 is one 2 000-line file. v3 needs the same content as *separate chapters* so
that v2 and v3 can be worked on in parallel: v2 keeps evolving Chapters 1-4,
v3 owns Chapters 5-8, and ``sync_v2.py`` can tell the two apart and merge.

The split is **content-based**, never line-based, so it keeps working as v2
grows. Markers, in order:

    \\begin{document}          end of the preamble
    \\mainmatter{}             end of the front matter
    \\chapter{...}             one file per (unstarred) chapter
    \\appendix{}               the whole appendix becomes one file
    \\addchap{Abbreviations}   start of the back matter
    \\end{document}            end of the back matter

Every input line lands in exactly one output file; the tool asserts that.

Usage
-----
    python3 tools/split_v2.py <out_dir> [--source ../v2/thesis_v2.tex]

It writes nothing outside ``out_dir`` and never touches v2.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
DEFAULT_SOURCE = os.path.normpath(os.path.join(V3, '..', 'v2', 'thesis_v2.tex'))

# chapter \label -> output basename. Explicit on purpose: if v2 ever adds a
# chapter, the tool must fail loudly rather than invent a filename that the
# manifest and thesis_v3.tex do not know about.
CHAPTER_FILES = {
    'ch:introduction': 'chapters/01_introduction.tex',
    'ch:background':   'chapters/02_background.tex',
    'ch:related':      'chapters/03_related_work.tex',
    'ch:method':       'chapters/04_method.tex',
    'ch:setup':        'chapters/05_setup.tex',
    'ch:results':      'chapters/06_results.tex',
    'ch:discussion':   'chapters/07_discussion.tex',
    'ch:conclusion':   'chapters/08_conclusion.tex',
}
PREAMBLE = 'parts/00_preamble.tex'
FRONTMATTER = 'parts/01_frontmatter.tex'
APPENDIX = 'chapters/09_appendix.tex'
BACKMATTER = 'parts/99_backmatter.tex'

RE_CHAPTER = re.compile(r'^\\chapter\{')
RE_LABEL = re.compile(r'\\label\{([^}]*)\}')


def find(lines, pred, what):
    for i, ln in enumerate(lines):
        if pred(ln):
            return i
    sys.exit(f'split_v2: marker not found: {what}')


def split(source):
    """-> list of (relative_path, list_of_lines), covering every input line."""
    with open(source) as f:
        lines = f.readlines()

    i_begin = find(lines, lambda l: l.startswith(r'\begin{document}'), r'\begin{document}')
    i_main = find(lines, lambda l: l.startswith(r'\mainmatter{}'), r'\mainmatter{}')
    i_app = find(lines, lambda l: l.startswith(r'\appendix{}'), r'\appendix{}')
    i_abbr = find(lines, lambda l: l.startswith(r'\addchap{Abbreviations}'),
                  r'\addchap{Abbreviations}')
    i_end = find(lines, lambda l: l.startswith(r'\end{document}'), r'\end{document}')

    # The back matter opens with the \microtypesetup{protrusion=false} that
    # guards the abbreviation list; pull it in so the piece is self-contained.
    i_back = i_abbr
    for j in range(i_abbr - 1, max(i_abbr - 6, i_main), -1):
        if lines[j].startswith(r'\microtypesetup{protrusion=false}'):
            i_back = j
            break

    # Chapter starts inside the main matter, in file order.
    starts = [i for i in range(i_main, i_app) if RE_CHAPTER.match(lines[i])]
    if not starts:
        sys.exit('split_v2: no \\chapter{} found in the main matter')

    pieces = [
        (PREAMBLE, lines[:i_begin]),                    # \begin{document} lives in the master
        (FRONTMATTER, lines[i_begin + 1:starts[0]]),    # includes \mainmatter{}
    ]
    bounds = starts + [i_app]
    for k, a in enumerate(starts):
        b = bounds[k + 1]
        m = RE_LABEL.search(lines[a])
        label = m.group(1) if m else None
        if label not in CHAPTER_FILES:
            sys.exit(f'split_v2: unknown chapter label {label!r} on line {a + 1}. '
                     f'Add it to CHAPTER_FILES and to thesis_v3.tex.')
        pieces.append((CHAPTER_FILES[label], lines[a:b]))
    pieces.append((APPENDIX, lines[i_app:i_back]))
    pieces.append((BACKMATTER, lines[i_back:i_end]))    # \end{document} lives in the master

    covered = sum(len(p) for _, p in pieces) + 2        # the two lines held by the master
    if covered != len(lines):
        sys.exit(f'split_v2: coverage check failed ({covered} of {len(lines)} lines)')
    return pieces


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir', help='directory to write the split into')
    ap.add_argument('--source', default=DEFAULT_SOURCE, help=f'default: {DEFAULT_SOURCE}')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    for rel, body in split(a.source):
        dst = os.path.join(a.out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w') as f:
            f.writelines(body)
        if not a.quiet:
            print(f'{len(body):5d} lines  {rel}')


if __name__ == '__main__':
    main()
