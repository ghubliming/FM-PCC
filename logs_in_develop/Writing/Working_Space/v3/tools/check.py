#!/usr/bin/env python3
"""Mechanical checks on the v3 draft. Not a compiler -- there is no TeX
toolchain in this container, so "checks pass" NEVER means "compiles".

What it does check, which is most of what actually breaks a first build:

  labels        every \\autoref / \\ref / \\eqref resolves to a \\label
  duplicates    no \\label is defined twice
  citations     every \\parencite / \\cite key exists in one of the .bib files
  bib overlap   no key defined in BOTH bibliography.bib and bibliography_v3.bib
                (biber would report it; catching it here is cheaper)
  environments  \\begin{X} / \\end{X} balance, per file
  braces        net brace delta per file is zero
  figures       every \\includegraphics target exists as .svg or .pdf
  inputs        every \\input in the master resolves to a file
  drafting      counts \\hole, \\provisional, \\guard, \\srcnote, \\dataref
                -- all five must be gone before submission

Usage:  python3 tools/check.py [--verbose]
Exit code is 1 if anything failed.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
MASTER = os.path.join(V3, 'thesis_v3.tex')
BIBS = ['bibliography.bib', 'bibliography_v3.bib']

RE_INPUT = re.compile(r'^[^%\n]*\\input\{([^}]+)\}', re.M)
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')
RE_REF = re.compile(r'\\(?:auto|eq)?ref\{([^}]+)\}')
RE_CITE = re.compile(r'\\(?:paren|foot|text|auto)?cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}')
RE_BEGIN = re.compile(r'\\begin\{([^}]+)\}')
RE_END = re.compile(r'\\end\{([^}]+)\}')
RE_GRAPHIC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
RE_BIBKEY = re.compile(r'^\s*@\w+\s*\{\s*([^,\s]+)\s*,', re.M)
DRAFT_MACROS = ['hole', 'provisional', 'guard', 'srcnote', 'dataref']


def strip_comments(text):
    """Drop TeX comments, keeping escaped \\%."""
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in text.split('\n'))


def files_from_master():
    """The master's \\input order -- the real file list, not a glob."""
    with open(MASTER) as f:
        src = f.read()
    out = [('thesis_v3.tex', src)]
    for rel in RE_INPUT.findall(src):
        path = os.path.join(V3, rel if rel.endswith('.tex') else rel + '.tex')
        if not os.path.exists(path):
            out.append((rel, None))
            continue
        with open(path) as f:
            out.append((os.path.relpath(path, V3), f.read()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    docs = files_from_master()
    missing_inputs = [n for n, s in docs if s is None]
    docs = [(n, s) for n, s in docs if s is not None]
    clean = {n: strip_comments(s) for n, s in docs}

    problems = []
    if missing_inputs:
        problems.append(('missing \\input target', missing_inputs))

    # --- labels -------------------------------------------------------------
    labels, dup = {}, []
    for n, s in clean.items():
        for lab in RE_LABEL.findall(s):
            if lab in labels:
                dup.append(f'{lab} (in {labels[lab]} and {n})')
            labels[lab] = n
    dangling = sorted({r for s in clean.values() for r in RE_REF.findall(s)} - set(labels))
    if dup:
        problems.append(('duplicate \\label', dup))
    if dangling:
        problems.append(('reference with no \\label', dangling))

    # --- citations ----------------------------------------------------------
    bibkeys, bibfrom, overlap = set(), {}, []
    for b in BIBS:
        p = os.path.join(V3, b)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for k in RE_BIBKEY.findall(f.read()):
                if k in bibkeys:
                    overlap.append(f'{k} (in {bibfrom[k]} and {b})')
                bibkeys.add(k)
                bibfrom[k] = b
    cited = set()
    for s in clean.values():
        for group in RE_CITE.findall(s):
            cited |= {k.strip() for k in group.split(',') if k.strip()}
    unknown = sorted(cited - bibkeys)
    if overlap:
        problems.append(('bib key defined in BOTH .bib files', overlap))
    if unknown:
        problems.append(('citation with no bib entry', unknown))

    # --- environments and braces -------------------------------------------
    for n, s in clean.items():
        opened, closed = RE_BEGIN.findall(s), RE_END.findall(s)
        for env in set(opened) | set(closed):
            if env == 'document':
                continue
            if opened.count(env) != closed.count(env):
                problems.append((f'unbalanced environment in {n}',
                                 [f'{env}: {opened.count(env)} begin, {closed.count(env)} end']))
        # Escaped braces must be removed from BOTH counts, not subtracted once:
        # a line carrying \{ ... \} is balanced, and subtracting 2 made it read -2.
        bare = re.sub(r'(?<!\\)\\[{}]', '', s)
        delta = bare.count('{') - bare.count('}')
        if delta:
            problems.append((f'brace delta in {n}', [str(delta)]))

    # --- figures ------------------------------------------------------------
    figdir = os.path.join(V3, 'figures')
    wanted = {g for s in clean.values() for g in RE_GRAPHIC.findall(s)}
    absent = sorted(g for g in wanted
                    if not any(os.path.exists(os.path.join(figdir, g + e))
                               for e in ('', '.svg', '.pdf', '.png')))
    if absent:
        problems.append(('\\includegraphics target not in figures/', absent))

    # --- report -------------------------------------------------------------
    total_lines = sum(s.count('\n') for _n, s in docs)
    print(f'{len(docs)} file(s), {total_lines} lines, {len(labels)} labels, '
          f'{len(cited)} distinct citations of {len(bibkeys)} bib entries, '
          f'{len(wanted)} figure(s).')

    counts = {m: sum(len(re.findall(r'\\' + m + r'\{', s)) for s in clean.values())
              for m in DRAFT_MACROS}
    print('drafting macros (all must be 0 before submission): '
          + ', '.join(f'{m} {c}' for m, c in counts.items()))
    uncited = sorted(bibkeys - cited)
    if uncited:
        print(f'note: {len(uncited)} bib entry/entries never cited '
              f'(harmless with an alphabetic style): {", ".join(uncited[:6])}'
              + (' ...' if len(uncited) > 6 else ''))
    if a.verbose:
        for n, s in docs:
            print(f'  {s.count(chr(10)):5d}  {n}')

    if not problems:
        print('\nAll mechanical checks pass. NOT COMPILED -- no TeX toolchain here.')
        return 0
    print()
    for what, items in problems:
        print(f'FAIL  {what}:')
        for i in items:
            print(f'        {i}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
