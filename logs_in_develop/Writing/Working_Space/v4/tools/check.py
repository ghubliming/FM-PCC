#!/usr/bin/env python3
"""Mechanical checks on the v4 draft. Not a compiler -- there is no TeX
toolchain in this container, so "checks pass" NEVER means "compiles".

What it does check, which is most of what actually breaks a first build:

  labels        every \\autoref / \\ref / \\eqref resolves to a \\label
  duplicates    no \\label is defined twice
  citations     every \\parencite / \\cite key exists in one of the .bib files
  bib overlap   no key defined in two .bib files (biber would report it)
  environments  \\begin{X} / \\end{X} balance, per file
  braces        net brace delta per file is zero
  figures       every \\includegraphics target exists as .svg/.pdf/.png
  inputs        every \\input, at any depth, resolves to a file (v4 reads nested
                inputs: the appendix inputs app_ntrial20_feasible and chapters/app_long/*)
  conditionals  the \\ifappendixfull ... \\else ... \\fi blocks of the appendix are read
                in the state parts/00_preamble_v4.tex sets, so the alias labels of a
                hidden section and the real labels of its tables are never both counted
  drafting      counts \\hole, \\provisional, \\guard, \\srcnote, \\dataref, \\flawed,
                \\outdated, \\longdata -- all must be gone before submission

Usage:  python3 tools/check.py [--verbose]
Exit code is 1 if anything failed.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
MASTER = os.path.join(V4, 'thesis_v4.tex')
BIBS = ['bibliography.bib', 'bibliography_v3.bib', 'bibliography_v4.bib']
SWITCH_FILE = os.path.join(V4, 'parts', '00_preamble_v4.tex')

RE_INPUT = re.compile(r'^[^%\n]*?\\input\{([^}]+)\}', re.M)
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')
RE_REF = re.compile(r'\\(?:auto|eq)?ref\{([^}]+)\}')
RE_CITE = re.compile(r'\\(?:paren|foot|text|auto)?cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}')
RE_BEGIN = re.compile(r'\\begin\{([^}]+)\}')
RE_END = re.compile(r'\\end\{([^}]+)\}')
RE_GRAPHIC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
RE_BIBKEY = re.compile(r'^\s*@\w+\s*\{\s*([^,\s]+)\s*,', re.M)
RE_COND = re.compile(r'\\ifappendixfull\b(.*?)(?:\\else\b(.*?))?\\fi\b', re.S)
# The inherited preamble and front matter keep a template-merge branch (\input{settings},
# \input{pages/...}) under \ifstandalone ... \else ... \fi; \standalonetrue is set, so LaTeX never
# reads it, and neither does this check.
RE_STANDALONE = re.compile(r'\\ifstandalone\b(.*?)(?:\\else\b(.*?))?\\fi\b', re.S)
DRAFT_MACROS = ['hole', 'provisional', 'guard', 'srcnote', 'dataref', 'todofigure', 'flawed',
                'outdated', 'longdata']


def strip_comments(text):
    """Drop TeX comments, keeping escaped \\%."""
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in text.split('\n'))


def switch_state():
    """True if parts/00_preamble_v4.tex leaves \\appendixfulltrue in force (the last setting wins)."""
    on = True
    try:
        for ln in strip_comments(open(SWITCH_FILE).read()).split('\n'):
            if '\\appendixfulltrue' in ln:
                on = True
            elif '\\appendixfullfalse' in ln:
                on = False
    except OSError:
        pass
    return on


def prune(text, on):
    """Keep the standalone branch of every \\ifstandalone block and the active branch of every\n    \\ifappendixfull block (no nesting inside either)."""
    def pick(m):
        return (m.group(1) or '') if on else (m.group(2) or '')
    text = RE_STANDALONE.sub(lambda m: m.group(1) or '', text)
    return RE_COND.sub(pick, text)


def resolve(target):
    for cand in (target, target + '.tex'):
        p = os.path.normpath(os.path.join(V4, cand))
        if p.startswith(V4 + os.sep) and os.path.isfile(p):
            return p
    return None


def collect(path, on, seen, out, missing):
    """The master's \\input tree, depth first, comments stripped and conditionals pruned."""
    rel = os.path.relpath(path, V4)
    if rel in seen:
        return
    seen.add(rel)
    with open(path) as f:
        raw = f.read()
    clean = prune(strip_comments(raw), on)
    out.append((rel, raw, clean))
    for target in RE_INPUT.findall(clean):
        child = resolve(target)
        if child is None:
            missing.append(f'{rel}: \\input{{{target}}}')
        else:
            collect(child, on, seen, out, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    on = switch_state()
    docs, missing_inputs = [], []
    collect(MASTER, on, set(), docs, missing_inputs)
    clean = {n: c for n, _r, c in docs}

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
        p = os.path.join(V4, b)
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
        problems.append(('bib key defined in two .bib files', overlap))
    if unknown:
        problems.append(('citation with no bib entry', unknown))

    # --- environments, braces, control characters -----------------------------
    for n, s in clean.items():
        opened, closed = RE_BEGIN.findall(s), RE_END.findall(s)
        for env in set(opened) | set(closed):
            if env == 'document':
                continue
            if opened.count(env) != closed.count(env):
                problems.append((f'unbalanced environment in {n}',
                                 [f'{env}: {opened.count(env)} begin, {closed.count(env)} end']))
        bare = re.sub(r'(?<!\\)\\[{}]', '', s)
        delta = bare.count('{') - bare.count('}')
        if delta:
            problems.append((f'brace delta in {n}', [str(delta)]))
        bad = [i for i, ch in enumerate(s) if ord(ch) < 32 and ch not in '\t\n\r']
        if bad:
            line = s.count('\n', 0, bad[0]) + 1
            problems.append((f'control characters in {n}', [f'{len(bad)} found, first at line {line}']))

    # --- bundler-only macro must not appear in the sources ---------------------
    leaked = sorted({n for n, s2 in clean.items() if '\\fmpccgraphic' in s2 or '\\fmpccinherited' in s2})
    if leaked:
        problems.append(('bundler-only macro (\\fmpccgraphic / \\fmpccinherited) used in a source file', leaked))

    # --- table width (warning) ----------------------------------------------
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
                    fr = [float(x) for x in re.findall(r'\\begin\{minipage\}(?:\[[^\]]*\])?\{([\d.]+)\\linewidth\}', row)]
                    width = sum(fr) * TEXTWIDTH_CM + ncols * 0.14 + 1.2
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
    figdir = os.path.join(V4, 'figures')
    wanted = {g for s in clean.values() for g in RE_GRAPHIC.findall(s)}
    absent = sorted(g for g in wanted
                    if not any(os.path.exists(os.path.join(figdir, g + e))
                               for e in ('', '.svg', '.pdf', '.png')))
    if absent:
        problems.append(('\\includegraphics target not in figures/', absent))

    # --- report -------------------------------------------------------------
    total_lines = sum(r.count('\n') for _n, r, _c in docs)
    print(f'{len(docs)} file(s), {total_lines} lines, {len(labels)} labels, '
          f'{len(cited)} distinct citations of {len(bibkeys)} bib entries, '
          f'{len(wanted)} figure(s); appendix long-data switch: {"ON (tables printed)" if on else "OFF (tables hidden)"}.')
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
        for n, r, _c in docs:
            print(f'  {r.count(chr(10)):5d}  {n}')
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
