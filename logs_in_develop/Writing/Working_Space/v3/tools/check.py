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
  drafting      counts \\hole, \\provisional, \\guard, \\srcnote, \\dataref, \\flawed, \\outdated
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
DRAFT_MACROS = ['hole', 'provisional', 'guard', 'srcnote', 'dataref', 'todofigure', 'flawed', 'outdated']


def strip_comments(text):
    """Drop TeX comments, keeping escaped \\%."""
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in text.split('\n'))


def files_from_master():
    """The master's \\input order -- the real file list, not a glob -- followed into the \\input
    lines of the chapter files themselves (v3.100: chapters/app_ntrial20_feasible.tex is reached only
    through chapters/09_appendix.tex, and its labels were reported missing). A nested \\input that
    does not resolve under this draft (the template's pages/ and settings, input from parts/) belongs
    to the template and is not reported; a missing master-level \\input still is."""
    with open(MASTER) as f:
        src = f.read()
    out = [('thesis_v3.tex', src)]
    seen = set()
    queue = [(rel, True) for rel in RE_INPUT.findall(src)]          # (path, from the master?)
    while queue:
        rel, top = queue.pop(0)
        path = os.path.join(V3, rel if rel.endswith('.tex') else rel + '.tex')
        key = os.path.normpath(path)
        if key in seen:
            continue
        seen.add(key)
        if not os.path.exists(path):
            if top:
                out.append((rel, None))
            continue
        with open(path) as f:
            text = f.read()
        out.append((os.path.relpath(path, V3), text))
        if rel.startswith('chapters/'):
            queue.extend((sub, False) for sub in RE_INPUT.findall(strip_comments(text)))
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

    # --- bundler-only macros must not appear in the sources -----------------
    # \fmpccgraphic is injected by bundle/make_bundle.py and is undefined in a
    # direct build of thesis_v3.tex. Writing it into a chapter breaks the split
    # build while leaving the bundle working, which is the worst way to break.
    leaked = sorted({n for n, s2 in clean.items() if '\\fmpccgraphic' in s2})
    if leaked:
        problems.append(('bundler-only macro \\fmpccgraphic used in a source file '
                         '(use \\includegraphics; the bundler rewrites it)', leaked))

    # --- table width (warning) ----------------------------------------------
    # Table 6.1 once ran past the right margin: a plain `tabular` is as wide as its
    # widest row, and a \multicolumn{..}{l}{...} note never wraps. Estimate both and
    # Control characters. A regex replacement once turned every "\\alpha" into BEL + "lpha" (Python
    # expanded "\\a" in the replacement string); LaTeX would have failed or printed garbage, and nothing
    # else here noticed. Any byte below 0x20 other than tab/newline/CR is an error.
    for n, s2 in clean.items():
        bad = [i for i, ch in enumerate(s2) if ord(ch) < 32 and ch not in '\t\n\r']
        if bad:
            line = s2.count('\n', 0, bad[0]) + 1
            problems.append((f'control characters in {n}', [f'{len(bad)} found, first at line {line}']))
    # warn; the text width of the TUM template at 11pt is about 14.7 cm.
    TEXTWIDTH_CM, CHAR_CM = 14.7, 0.19
    warn = []
    for n, s2 in clean.items():
        for m in re.finditer(r'\\begin\{(tabular|tabularx)\}(\{[^}]*\})?\{([^}]*)\}(.*?)\\end\{\1\}', s2, re.S):
            kind, spec, body = m.group(1), m.group(3), m.group(4)
            scale = 0.83 if '\\footnotesize' in body else (0.91 if '\\small' in body else 1.0)
            ncols = sum(spec.count(c) for c in 'lcrXL') + spec.count('p{')
            for row in body.split('\\\\'):
                mc = re.search(r'\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}\s*$', row.strip(), re.S)
                text = mc.group(1) if mc else None
                if text is not None:
                    plain = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?|[{}$&]', '', text).strip()
                    if len(plain) * CHAR_CM * scale > TEXTWIDTH_CM:
                        warn.append(f'{n}: \\multicolumn text does not wrap and is about '
                                    f'{len(plain) * CHAR_CM * scale:.1f} cm wide: "{plain[:60]}..."')
                elif kind == 'tabular' and '\\begin{minipage}' in row:
                    # A panel grid: cell widths are set by the minipages, not by the text, so the
                    # character estimate is meaningless. Sum the declared minipage fractions instead.
                    fr = [float(x) for x in re.findall(r'\\begin\{minipage\}(?:\[[^\]]*\])?\{([\d.]+)\\linewidth\}', row)]
                    width = sum(fr) * TEXTWIDTH_CM + ncols * 0.14 + 1.2   # 2pt seps, one label column
                    if width > TEXTWIDTH_CM:
                        warn.append(f'{n}: panel row of minipages is about {width:.1f} cm wide '
                                    f'(text width is {TEXTWIDTH_CM} cm)')
                elif kind == 'tabular':
                    plain = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?|[{}$]', '', row)
                    widest = sum(len(c.strip()) for c in plain.split('&')) * CHAR_CM * scale + ncols * 0.42
                    if widest > TEXTWIDTH_CM:
                        warn.append(f'{n}: `tabular` row is about {widest:.1f} cm wide (text width is '
                                    f'{TEXTWIDTH_CM} cm) -- use tabularx with an L column')

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

    # The optional-argument form matters: \todofigure[0.3\textwidth]{...} does not
    # match a bare \macro{ pattern, and counting it as zero would quietly hide
    # every planned-but-unmade figure from the pre-submission check.
    counts = {m: sum(len(re.findall(r'\\' + m + r'(?:\[[^\]]*\])?\{', s))
                     for s in clean.values())
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

    for w in dict.fromkeys(warn):
        print(f'WARN  {w}')
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
