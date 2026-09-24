#!/usr/bin/env python3
r"""Flatten the split v4 draft into ONE self-contained .tex, and package it for Overleaf.

WHY THIS EXISTS
---------------
v4 is split by chapter so that it can inherit Chapters 1-6 from v3 (and, through v3,
Chapters 1-4 from v2) while writing Ch 7-9 -- see ../inherited/MANIFEST.md. That split
is right for editing and wrong for uploading. This tool produces the flat file and an
Overleaf zip mechanically, so the flat file is never edited by hand and cannot drift.

    ../thesis_v4.tex  +  parts/  +  chapters/   ->   bundle/output/thesis_v4_<stamp>_<kind>.tex
                                                     bundle/output/thesis_v4_<stamp>_<kind>.zip

WHAT IT DOES, EXACTLY (ported from v3/bundle/make_bundle.py, v3.94)
--------------------------------------------------------------------
1. Reads ../thesis_v4.tex and recursively replaces every ``\input{X}`` whose target
   file exists with that file's bytes, wrapped in BEGIN/END banners naming the source
   and its SHA-256 prefix. An ``\input`` whose target does not exist is left verbatim
   (the template-merge branches of the inherited preamble/front matter).
2. Rewrites ``\includegraphics`` to ``\fmpccgraphic`` and injects that macro, so a
   figure that is SVG-only becomes a labelled placeholder box instead of a build failure.
3. Copies the three bibliography resources and figures/ next to the .tex, and zips them.
4. VERIFIES before writing and refuses to write on failure.
5. COLLAPSES INHERITED CHAPTERS BY OWNER. The author's rule for v4 (2026-09-24): the
   working bundle is built "without v2 but v3+v4". By default every chapter OWNED BY v2
   (Ch 1-4) that is byte-identical to its v3-base copy is collapsed to its numbered headings
   plus a grey box naming the file, the owner and the versions it came through; the v3-owned
   chapters (5-6) and v4's own (7-9) are inlined in full. Numbering and cross-references match
   the full build. ``--v4-only`` also collapses the v3-owned chapters; ``--full`` collapses nothing.
6. RECORDS THE VERSIONS BUILT ON. The header of every bundle and every collapsed box carry
   the v3 revision this v4 branched from and the v2 revision that v3 carries, read from
   ../inherited/SYNC_STATE.json (stamped by tools/sync_v3.py).

USAGE
-----
    python3 bundle/make_bundle.py                    # BOTH variants: _new (annotated) and _full_clean
    python3 bundle/make_bundle.py --annotated-only   # just _new
    python3 bundle/make_bundle.py --clean-only       # just _full_clean
    python3 bundle/make_bundle.py --full             # annotated, nothing collapsed (_full)
    python3 bundle/make_bundle.py --v4-only          # annotated, v2 AND v3 chapters collapsed (_v4only)
    python3 bundle/make_bundle.py --appendix-short   # the appendix's LONG DATA tables hidden
    python3 bundle/make_bundle.py --svg-package      # render SVGs via \usepackage{svg} (Overleaf)
    python3 bundle/make_bundle.py --no-zip | --list | --prune N | --verify [BUNDLE.tex]

THE TWO VARIANTS (author, 2026-09-24, v3.94; kept for v4)
    thesis_v4_<stamp>_new.{tex,zip}         ANNOTATED, v2 chapters collapsed, v3+v4 in full: every
                                            drafting mark visible (\srcnote, \dataref, \guard, \hole,
                                            \provisional, \outdated, \flawed, \longdata). The working copy.
    thesis_v4_<stamp>_full_clean.{tex,zip}  CLEAN, the whole thesis, no notes: for checking the layout.
                                            \guard and \provisional print their text as plain prose;
                                            the other marks print nothing; \longdata prints nothing.

Every build writes a new timestamped pair and appends one row to BUNDLE_LOG.md.
Bundles are build output: they are never edited, and nothing reads them back.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
MASTER = os.path.join(V4, 'thesis_v4.tex')
BIBS = ['bibliography.bib', 'bibliography_v3.bib', 'bibliography_v4.bib']
FIGDIR = os.path.join(V4, 'figures')
LOG = os.path.join(HERE, 'BUNDLE_LOG.md')
OUT_DIR = os.path.join(HERE, 'output')
COMPILABLE = ('.pdf', '.png', '.jpg', '.jpeg')
MAX_DEPTH = 8
BASE_DIR = os.path.join(V4, 'inherited', 'v3_base')
STATE = os.path.join(V4, 'inherited', 'SYNC_STATE.json')

# Which file belongs to whom is decided in ONE place, tools/sync_v3.py.
sys.path.insert(0, os.path.join(V4, 'tools'))
try:
    from sync_v3 import POLICY  # noqa: E402   {rel: (policy, owner)}
except ImportError as e:        # pragma: no cover
    sys.exit(f'make_bundle: cannot read the ownership policy from tools/sync_v3.py ({e})')

RE_BUNDLE = re.compile(r'thesis_v4_\d{8}_\d{6}(?:_(?:new|v4only|full)(?:_clean)?)?(?:_\d+)?\.tex')
HEADING_PREFIXES = ('\\chapter{', '\\chapter[', '\\section{', '\\section[',
                    '\\subsection{', '\\subsection[', '\\subsubsection{', '\\subsubsection[')
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')
RE_REF = re.compile(r'\\(?:auto|eq)?ref\{([^}]+)\}')
RE_INPUT_LINE = re.compile(r'^(?P<pre>[^%\n]*?)\\input\{(?P<target>[^}]+)\}(?P<post>[^\n]*)$')
RE_GRAPHIC = re.compile(r'\\includegraphics(?P<opts>\[[^\]]*\])?\{(?P<name>[^}]+)\}')
RE_BEGIN = re.compile(r'\\begin\{([^}]+)\}')
RE_END = re.compile(r'\\end\{([^}]+)\}')

GRAPHICS_SHIM = r"""
% =============================================================================
%  GENERATED BY bundle/make_bundle.py -- DO NOT EDIT IN THE BUNDLE
% -----------------------------------------------------------------------------
%  Figure guard. The thesis figures are generated as SVG (DA_in_Paper), and
%  \includegraphics cannot read SVG. Every \includegraphics call in this file was
%  rewritten to \fmpccgraphic, which falls back to a placeholder box naming the
%  file it could not find.
%
%  TO GET THE REAL FIGURES: run DA_in_Paper/plotting/svg/svg2pdf.sh on a machine with
%  rsvg-convert, inkscape or cairosvg, export to v4, then re-run bundle/make_bundle.py.
% =============================================================================
\makeatletter
\newcommand{\fmpcc@placeholder}[1]{%
  \fbox{\begin{minipage}{0.86\linewidth}%
    \centering\vspace{1.2em}%
    \textbf{[figure not available in this bundle]}\\[0.4em]
    \texttt{\detokenize{figures/#1}}\\[0.4em]
    {\footnotesize No .pdf/.png present; the generated figure is SVG.\\
     Convert in \texttt{DA\_in\_Paper}, export, then rebuild the bundle.}%
    \vspace{1.2em}%
  \end{minipage}}%
}
\newcommand{\fmpccgraphic}[2][]{%
  \IfFileExists{figures/#2.pdf}{\includegraphics[#1]{#2.pdf}}{%
  \IfFileExists{figures/#2.png}{\includegraphics[#1]{#2.png}}{%
  \IfFileExists{figures/#2}{\includegraphics[#1]{#2}}{%
  \fmpcc@placeholder{#2}}}}%
}
\makeatother
"""

GRAPHICS_SHIM_SVG = (
    GRAPHICS_SHIM
    .replace(r'\makeatletter', '\\usepackage{svg}\n\\makeatletter', 1)
    .replace(r'  \IfFileExists{figures/#2}{\includegraphics[#1]{#2}}{%',
             r'  \IfFileExists{figures/#2.svg}{\includesvg[#1]{figures/#2}}{%')
    .replace('%  TO GET THE REAL FIGURES: run DA_in_Paper/plotting/svg/svg2pdf.sh on a machine with',
             '%  THIS BUNDLE WAS BUILT WITH --svg-package: the SVGs are rendered directly\n'
             '%  by Inkscape at build time. If the host has no Inkscape, rebuild without\n'
             '%  that flag, or run DA_in_Paper/plotting/svg/svg2pdf.sh on a machine with')
)

# The CLEAN variant. \submissiontrue reaches the two inherited macros (\srcnote hidden, \hole
# warned); every other drafting mark is v3's or v4's own and is switched off here. \guard and
# \provisional keep their TEXT (thesis prose); \longdata prints nothing.
CLEAN_SWITCH = r"""
% =============================================================================
%  GENERATED BY bundle/make_bundle.py -- CLEAN: THE THESIS ALONE (layout check)
% -----------------------------------------------------------------------------
%  \srcnote and \dataref (provenance) and \hole, \outdated, \flawed and \longdata
%  (drafting markers) print nothing. \guard and \provisional print their text as
%  plain running text. A dead block prints its content without banner or grey.
%  This build does not show what is missing: the annotated _new build does.
% =============================================================================
\submissiontrue
\renewcommand{\dataref}[1]{}
\renewcommand{\hole}[1]{}
\renewcommand{\outdated}[1]{}
\renewcommand{\flawed}[1]{}
\renewcommand{\longdata}[1]{}
\renewcommand{\guard}[1]{#1}
\renewcommand{\provisional}[1]{#1}
\renewcommand{\todofigure}[2][0.30\textwidth]{\rule{0pt}{#1}}
\pillarsflawedfalse
\renewenvironment{deadblock}[1]{}{}
"""

APPENDIX_SHORT_SWITCH = r"""
% GENERATED BY bundle/make_bundle.py --appendix-short: the LONG DATA sections of the
% appendix keep their reading and hide their tables (parts/00_preamble_v4.tex).
\appendixfullfalse
"""

# Injected before \begin{document} whenever at least one chapter was collapsed.
INHERITED_SHIM = r"""
% =============================================================================
%  GENERATED BY bundle/make_bundle.py -- INHERITED CHAPTERS COLLAPSED
% -----------------------------------------------------------------------------
%  Chapters inherited unchanged from an upstream draft were collapsed to their
%  numbered headings; each carries the box below instead of its text. Rebuild
%  with --full for the complete document.
% =============================================================================
\newcommand{\fmpccinherited}[4]{%
  \par\medskip\noindent
  \fcolorbox{gray}{white}{\parbox{0.96\linewidth}{\centering\small
    \textbf{[#3 SECTION --- not built in this bundle]}\\[0.3em]
    \texttt{\detokenize{#1}}, #2 lines, inherited unchanged from #4.\\[0.2em]
    Its section headings are kept, so numbering and cross-references match the full build.\\
    Rebuild with \texttt{make\_bundle.py --full} to include its text.}}%
  \par\medskip
}
"""


def versions():
    """-> dict: v3 (short, full), v2 (short, full) from inherited/SYNC_STATE.json."""
    try:
        with open(STATE) as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}

    def short(full):
        m = re.match(r'(v\d+\.\d+[a-z]?)\D+(\d{4}-\d{2}-\d{2})', full or '')
        return f'{m.group(1)} ({m.group(2)})' if m else 'an unrecorded revision'
    v3f = st.get('v3_version', 'unknown')
    v2f = st.get('v2_version_via_v3', 'unknown')
    return {'v3': (short(v3f), v3f), 'v2': (short(v2f), v2f)}


def owner_tag(owner, ver):
    """The provenance string printed in a collapsed box, by owner."""
    if owner == 'v2':
        return f'v2 {ver["v2"][0]}, via v3 {ver["v3"][0]}'
    if owner == 'v3':
        return f'v3 {ver["v3"][0]}'
    return owner


def collapse_decision(path, collapse_owners):
    """-> (collapse?, reason)."""
    rel = os.path.relpath(path, V4)
    if not rel.startswith('chapters/'):
        return False, 'structural'
    pol = POLICY.get(rel)
    if pol is None:
        return False, 'not in POLICY'
    policy, owner = pol
    if owner not in collapse_owners or policy == 'own':
        return False, f'{owner}-owned, built'
    base = os.path.join(BASE_DIR, rel)
    if not os.path.isfile(base):
        return False, 'no v3 baseline recorded'
    with open(path, 'rb') as a, open(base, 'rb') as b:
        if a.read() != b.read():
            return False, 'EDITED IN v4'
    return True, f'unchanged {owner}'


def skeleton(path, owner, tag):
    """A collapsed chapter: banner, numbered headings, and the inherited box."""
    rel = os.path.relpath(path, V4)
    body = open(path).read().split('\n')
    heads = [ln.rstrip() for ln in body if ln.startswith(HEADING_PREFIXES)]
    if not heads or not heads[0].startswith(('\\chapter{', '\\chapter[')):
        sys.exit(f'make_bundle: {rel} does not open with a \\chapter heading; refusing to collapse it')
    out = [f'% <<<<<<<<<<<<<<<< COLLAPSED {rel}  ({sha(path)}) <<<<<<<<<<<<<<<<',
           f'% {owner} SECTION, not built. Inherited unchanged from {tag}',
           f'% Kept: its {len(heads)} numbered heading(s). Dropped: {len(body)} lines of text.',
           heads[0],
           f'\\fmpccinherited{{{rel}}}{{{len(body)}}}{{{owner}}}{{{tag}}}']
    out += heads[1:]
    out.append(f'% >>>>>>>>>>>>>>>> END COLLAPSED {rel} >>>>>>>>>>>>>>>>')
    return out, len(body), len(heads)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def resolve(target):
    for cand in (target, target + '.tex'):
        p = os.path.normpath(os.path.join(V4, cand))
        if p.startswith(V4 + os.sep) and os.path.isfile(p):
            return p
    return None


def flatten(path, seen, sources, depth=0, collapse_owners=frozenset(), collapsed=None, kept=None, ver=None):
    rel = os.path.relpath(path, V4)
    if depth > MAX_DEPTH:
        sys.exit(f'make_bundle: \\input nested deeper than {MAX_DEPTH} at {rel}')
    if rel in seen:
        sys.exit(f'make_bundle: \\input cycle -- {rel} is included twice')
    seen.add(rel)
    with open(path) as f:
        body = f.read().split('\n')
    sources.append((rel, len(body), sha(path)))
    out = []
    for line in body:
        m = RE_INPUT_LINE.match(line)
        child = resolve(m.group('target')) if m else None
        if child is None:
            out.append(line)
            continue
        crel = os.path.relpath(child, V4)
        if m.group('pre').strip():
            out.append(m.group('pre').rstrip())
        out.append('')
        if collapse_owners:
            ok, why = collapse_decision(child, collapse_owners)
            if ok:
                owner = POLICY[crel][1]
                sk, n_lines, n_heads = skeleton(child, owner, owner_tag(owner, ver))
                out.extend(sk)
                collapsed.append((crel, n_lines, n_heads, sha(child), owner))
                out.append('')
                if m.group('post').strip():
                    out.append(m.group('post').lstrip())
                continue
            if why == 'EDITED IN v4':
                kept.append(crel)
        out.append(f'% <<<<<<<<<<<<<<<< BEGIN {crel}  ({sha(child)}) <<<<<<<<<<<<<<<<')
        out.extend(flatten(child, seen, sources, depth + 1, collapse_owners, collapsed, kept, ver))
        out.append(f'% >>>>>>>>>>>>>>>> END   {crel} >>>>>>>>>>>>>>>>')
        out.append('')
        if m.group('post').strip():
            out.append(m.group('post').lstrip())
    return out


def split_comment(line):
    m = re.search(r'(?<!\\)%', line)
    return (line[:m.start()], line[m.start():]) if m else (line, '')


def rewrite_graphics(lines):
    n = 0
    out = []
    for line in lines:
        code, comment = split_comment(line)
        code, k = RE_GRAPHIC.subn(
            lambda m: r'\fmpccgraphic' + (m.group('opts') or '') + '{' + m.group('name') + '}', code)
        n += k
        out.append(code + comment)
    return out, n


def inject_shim(lines, shim):
    for i, line in enumerate(lines):
        if line.lstrip().startswith(r'\begin{document}'):
            return lines[:i] + shim.split('\n') + lines[i:]
    sys.exit(r'make_bundle: no \begin{document} found -- refusing to inject')


def strip_comments(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in text.split('\n'))


def count_macro(lines, macro):
    return len(re.findall(r'\\' + macro + r'\{', strip_comments('\n'.join(lines))))


def verify(lines, sources, expect_lines):
    text = '\n'.join(lines)
    code = strip_comments(text)
    bad = []
    left = [m.group('target') for ln in code.split('\n')
            if (m := RE_INPUT_LINE.match(ln)) and resolve(m.group('target'))]
    if left:
        bad.append(f'{len(left)} resolvable \\input still present: {", ".join(left)}')
    if len(lines) < expect_lines:
        bad.append(f'lost content: {len(lines)} output lines < {expect_lines} source lines')
    bare = re.sub(r'(?<!\\)\\[{}]', '', code)
    if (d := bare.count('{') - bare.count('}')):
        bad.append(f'brace delta {d:+d}')
    op, cl = RE_BEGIN.findall(code), RE_END.findall(code)
    for env in set(op) | set(cl):
        if op.count(env) != cl.count(env):
            bad.append(f'environment {env}: {op.count(env)} begin, {cl.count(env)} end')
    for what, n in ((r'\begin{document}', code.count(r'\begin{document}')),
                    (r'\end{document}', code.count(r'\end{document}')),
                    (r'\documentclass', code.count(r'\documentclass'))):
        if n != 1:
            bad.append(f'{what} appears {n} times, expected 1')
    return bad


def header(stamp, sources, n_graphics, shim, collapsed, ver, kind):
    w = max([len(r) for r, _n, _s in sources] + [len(c[0]) for c in collapsed])
    rows = '\n'.join(f'%   {r.ljust(w)}  {n:5d} lines  sha256:{s}' for r, n, s in sources)
    if collapsed:
        rows += ('\n%\n%  COLLAPSED to headings (inherited sections, not built in this bundle):\n'
                 + '\n'.join(f'%   {r.ljust(w)}  {n:5d} lines  sha256:{h}  ({k} headings kept; {o}: '
                              f'{owner_tag(o, ver)})' for r, n, k, h, o in collapsed)
                 + '\n%  Rebuild with --full for the complete document.')
    return f"""% =============================================================================
%  FM-PCC MASTER'S THESIS -- v4 -- FLATTENED BUILD ({kind})
% -----------------------------------------------------------------------------
%  GENERATED FILE. DO NOT EDIT.
%
%  Built {stamp} by bundle/make_bundle.py from the split sources listed below.
%  Edit those, then re-run the tool; any edit made here is lost on the next run.
%
%  VERSIONS BUILT ON (inherited/SYNC_STATE.json, stamped by tools/sync_v3.py):
%    v3: {ver['v3'][1]}
%    v2: {ver['v2'][1]}  (carried by v3)
%
%  Sources, in \\input order, with the SHA-256 prefix of each as built:
{rows}
%
%  Graphics: {n_graphics} \\includegraphics call(s) rewritten to \\fmpccgraphic
%  {shim}.
%
%  BUILD (Overleaf or local):
%    pdflatex <this file> ; biber <jobname> ; pdflatex x2
%  Needs, next to this file: bibliography.bib, bibliography_v3.bib, bibliography_v4.bib, figures/
%  -- all of which are in the .zip this tool writes alongside it.
% =============================================================================
"""


def build(args, clean=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if clean or args.full:
        collapse_owners, kind = frozenset(), 'full'
    elif args.v4_only:
        collapse_owners, kind = frozenset({'v2', 'v3'}), 'v4only'
    else:
        collapse_owners, kind = frozenset({'v2'}), 'new'
    base = f'thesis_v4_{stamp}_{kind}' + ('_clean' if clean else '')
    if os.path.exists(os.path.join(OUT_DIR, base + '.tex')):
        n = 2
        while os.path.exists(os.path.join(OUT_DIR, f'{base}_{n}.tex')):
            n += 1
        base = f'{base}_{n}'
    tex_path = os.path.join(OUT_DIR, base + '.tex')

    ver = versions()
    sources, collapsed, kept = [], [], []
    lines = flatten(MASTER, set(), sources, collapse_owners=collapse_owners,
                    collapsed=collapsed, kept=kept, ver=ver)
    src_lines = sum(n for _r, n, _s in sources)
    if collapsed:
        lines = inject_shim(lines, INHERITED_SHIM)
    counts = {m: count_macro(lines, m) for m in ('srcnote', 'dataref', 'hole', 'outdated', 'flawed',
                                                  'longdata', 'guard', 'provisional')}
    if clean:
        lines = inject_shim(lines, CLEAN_SWITCH)
    if args.appendix_short:
        lines = inject_shim(lines, APPENDIX_SHORT_SWITCH)

    n_graphics = 0
    if not args.no_graphics_shim:
        lines, n_graphics = rewrite_graphics(lines)      # rewrite first, THEN inject the shim
        lines = inject_shim(lines, GRAPHICS_SHIM_SVG if args.svg_package else GRAPHICS_SHIM)

    bad = verify(lines, sources, src_lines)
    if bad:
        print('REFUSING TO WRITE -- the flattened file failed verification:')
        for b in bad:
            print(f'  {b}')
        return 1

    mode = ('NOT guarded -- a missing figure fails the build' if args.no_graphics_shim else
            'guarded, rendering SVG via \\usepackage{svg}' if args.svg_package else
            'guarded, with a placeholder box for a figure that has no .pdf/.png')
    what = {'full': 'FULL', 'new': 'v2 COLLAPSED, v3+v4 BUILT', 'v4only': 'v2+v3 COLLAPSED, v4 BUILT'}[kind]
    body = header(stamp, sources, n_graphics, mode, collapsed, ver, what) + '\n'.join(lines)
    if not body.endswith('\n'):
        body += '\n'
    with open(tex_path, 'w') as f:
        f.write(body)

    print(f'{"CLEAN, " if clean else "ANNOTATED, "}{what} -- '
          f'{len(sources)} source file(s) inlined, {src_lines} lines -> {len(lines)} lines, {len(body) // 1024} KB')
    print(f'  built on: v3 {ver["v3"][0]}; v2 {ver["v2"][0]} (via v3)')
    if clean:
        print('  notes: none -- ' + ', '.join(f'{counts[m]} \\{m}' for m in ('srcnote', 'dataref', 'hole', 'outdated',
                                                                            'flawed', 'longdata')) + ' print nothing; '
              + ', '.join(f'{counts[m]} \\{m}' for m in ('guard', 'provisional')) + ' print as plain text')
    else:
        print('  notes: annotated -- every drafting mark visible')
    if args.appendix_short:
        print('  appendix: LONG DATA tables hidden (\\appendixfullfalse)')
    if collapsed:
        print('  collapsed to headings: '
              + ', '.join(f'{os.path.basename(r)} ({n} lines, {o})' for r, n, _k, _h, o in collapsed))
    if kept:
        print('  inlined IN FULL because v4 has edited them: ' + ', '.join(kept))
    if collapsed:
        code = strip_comments('\n'.join(lines))
        dangling = sorted(set(RE_REF.findall(code)) - set(RE_LABEL.findall(code)))
        if dangling:
            print(f'  NOTE: {len(dangling)} reference(s) point into collapsed text and will print '
                  f'"??": ' + ', '.join(dangling[:8]) + (' ...' if len(dangling) > 8 else ''))
        else:
            print('  cross-references: all resolve (chapter and section numbers match the full build)')
    print(f'  {os.path.relpath(tex_path, V4)}')

    assets = []
    for b in BIBS:
        src = os.path.join(V4, b)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(OUT_DIR, b))
            assets.append(b)
    figs = []
    if os.path.isdir(FIGDIR):
        dst = os.path.join(OUT_DIR, 'figures')
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(FIGDIR)):
            if os.path.splitext(fn)[1].lower() in COMPILABLE + ('.svg',):
                shutil.copyfile(os.path.join(FIGDIR, fn), os.path.join(dst, fn))
                figs.append(fn)
    stems = {}
    for f in figs:
        stem, ext = os.path.splitext(f)
        stems.setdefault(stem, set()).add(ext.lower())
    usable = sorted(st for st, ex in stems.items() if ex & set(COMPILABLE))
    placeholder = sorted(st for st in stems if st not in usable)
    print(f'  assets: {len(assets)} bibliography file(s), {len(figs)} figure file(s) '
          f'= {len(stems)} figure(s) ({len(usable)} render, {len(placeholder)} placeholder)')

    zip_path = ''
    if not args.no_zip:
        zip_path = os.path.join(OUT_DIR, base + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(tex_path, base + '.tex')
            for b in assets:
                z.write(os.path.join(OUT_DIR, b), b)
            for fn in figs:
                z.write(os.path.join(OUT_DIR, 'figures', fn), f'figures/{fn}')
        print(f'  {os.path.relpath(zip_path, V4)}  '
              f'({os.path.getsize(zip_path) // 1024} KB) -- upload this to Overleaf')

    with open(LOG, 'a') as f:
        f.write(f'| {stamp} | {len(sources)} | {src_lines} | {len(lines)} | '
                f'{n_graphics} | {len(usable)}/{len(figs)} | {ver["v3"][0]} / {ver["v2"][0]} | `{base}.tex` |\n')
    if placeholder and not args.svg_package:
        print(f'\nNOTE: {len(placeholder)} figure(s) ship only as SVG and render as a labelled '
              f'placeholder box:\n      ' + ', '.join(placeholder))
    return 0


RE_BANNER_BEGIN = re.compile(r'^% <{16} BEGIN (\S+)')
RE_BANNER_END = re.compile(r'^% >{16} END\s+(?!COLLAPSED\b)(\S+)')
RE_BANNER_COLLAPSED = re.compile(r'^% <{16} COLLAPSED (\S+)\s+\(([0-9a-f]{12})\)')


def cmd_verify(path):
    """Extract every inlined source back out of a bundle and diff it against the tree."""
    if not os.path.isfile(path):
        print(f'no such bundle: {path}')
        return 1
    lines = open(path).read().split('\n')
    stack, blocks = [], {}
    for i, ln in enumerate(lines):
        if (m := RE_BANNER_BEGIN.match(ln)):
            stack.append((m.group(1), i))
        elif (m := RE_BANNER_END.match(ln)):
            if not stack:
                print(f'unmatched END banner at line {i + 1}')
                return 1
            name, start = stack.pop()
            if name != m.group(1):
                print(f'banner mismatch: BEGIN {name} closed by END {m.group(1)}')
                return 1
            blocks.setdefault(name, (start + 1, i))
    if stack:
        print(f'unclosed BEGIN banner: {stack[-1][0]}')
        return 1
    collapsed = [m.groups() for ln in lines if (m := RE_BANNER_COLLAPSED.match(ln))]

    def trim(xs):
        xs = list(xs)
        while xs and not xs[0].strip():
            xs.pop(0)
        while xs and not xs[-1].strip():
            xs.pop()
        return xs

    def children_of(a, b):
        inner = sorted((ca, cb, n) for n, (ca, cb) in blocks.items() if a <= ca - 1 and cb <= b and (ca, cb) != (a, b))
        direct, reach = [], -1
        for ca, cb, n in inner:
            if ca - 1 > reach:
                direct.append((ca, cb, n))
                reach = cb
        return direct

    def folded_bundle(a, b):
        out, k, kids = [], a, children_of(a, b)
        for ca, cb, n in kids:
            begin = ca - 1
            stop = begin - 1 if begin - 1 >= k and not lines[begin - 1].strip() else begin
            out.extend(lines[k:stop])
            out.append(f'\x00INPUT {n}')
            k = cb + 2 if cb + 1 < len(lines) and not lines[cb + 1].strip() else cb + 1
        out.extend(lines[k:b])
        return out

    def folded_source(src):
        out = []
        for ln in open(src).read().split('\n'):
            m = RE_INPUT_LINE.match(ln)
            child = resolve(m.group('target')) if m else None
            if child is None:
                out.append(ln)
                continue
            if m.group('pre').strip():
                out.append(m.group('pre').rstrip())
            out.append(f'\x00INPUT {os.path.relpath(child, V4)}')
            if m.group('post').strip():
                out.append(m.group('post').lstrip())
        return out

    bad = 0
    for name, (a, b) in sorted(blocks.items()):
        src = os.path.join(V4, name)
        if not os.path.isfile(src):
            print(f'  GONE {name}  (inlined in the bundle, absent from the tree now)')
            bad += 1
            continue

        def unrewrite(ln):
            code, comment = split_comment(ln)
            return code.replace(r'\fmpccgraphic', r'\includegraphics') + comment

        got = trim(unrewrite(ln) for ln in folded_bundle(a, b))
        want = trim(folded_source(src))
        if got == want:
            print(f'  OK   {name}  ({len(want)} lines)')
        else:
            first = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            print(f'  DIFF {name}  (bundle {len(got)} lines, source {len(want)}; '
                  f'first difference at source line {first + 1})')
            bad += 1
    for name, recorded in collapsed:
        src = os.path.join(V4, name)
        if not os.path.isfile(src):
            print(f'  GONE {name}  (collapsed in the bundle, absent from the tree now)')
            bad += 1
        elif sha(src) == recorded:
            print(f'  OK   {name}  (collapsed inherited section; unchanged since the build)')
        else:
            print(f'  DIFF {name}  (collapsed inherited section; the source has changed since the build)')
            bad += 1
    print()
    print(f'{len(blocks)} inlined + {len(collapsed)} collapsed source(s); '
          + ('byte-faithful throughout.' if not bad else
             f'{bad} MISMATCH -- the sources have moved since this bundle was built.'))
    return 1 if bad else 0


def bundles():
    found, seen = [], set()
    for d in (OUT_DIR, HERE):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if RE_BUNDLE.fullmatch(f) and f not in seen:
                seen.add(f)
                found.append((os.path.getmtime(os.path.join(d, f)), f, d))
    found.sort(key=lambda item: (item[0], item[1]))
    return [(item[1], item[2]) for item in found]


def cmd_list():
    rows = bundles()
    if not rows:
        print('no bundles built yet.')
        return 0
    for f, d in rows:
        p = os.path.join(d, f)
        z = p[:-4] + '.zip'
        print(f'  {f}  {os.path.getsize(p) // 1024:4d} KB'
              + (f'  + zip {os.path.getsize(z) // 1024} KB' if os.path.exists(z) else ''))
    print(f'\n{len(rows)} bundle(s). Newest: {rows[-1][0]}')
    return 0


def cmd_prune(keep):
    rows = bundles()
    doomed = rows[:-keep] if keep > 0 else rows
    for f, d in doomed:
        p = os.path.join(d, f)
        if os.path.exists(p):
            os.remove(p)
        z = os.path.join(d, f[:-4] + '.zip')
        if os.path.exists(z):
            os.remove(z)
        print(f'removed {f}')
    print(f'{len(rows) - len(doomed)} bundle(s) kept.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true', help='collapse nothing (annotated variant)')
    ap.add_argument('--v4-only', action='store_true', help='collapse the v2 AND v3 chapters (annotated variant)')
    ap.add_argument('--appendix-short', action='store_true', help='hide the LONG DATA tables of the appendix')
    ap.add_argument('--annotated-only', action='store_true')
    ap.add_argument('--clean-only', action='store_true')
    ap.add_argument('--no-zip', action='store_true')
    ap.add_argument('--no-graphics-shim', action='store_true')
    ap.add_argument('--svg-package', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--prune', type=int, metavar='N')
    ap.add_argument('--verify', metavar='BUNDLE.tex', nargs='?', const='')
    a = ap.parse_args()

    if not os.path.exists(LOG):
        with open(LOG, 'w') as f:
            f.write('# Bundle log\n\nAppended to by `make_bundle.py`. Bundles are build output; nothing reads them back.\n\n'
                    '| built | files | source lines | output lines | graphics | figures usable | v3 / v2 built on | file |\n'
                    '| :-- | --: | --: | --: | --: | --: | :-- | :-- |\n')
    if a.verify is not None:
        target = a.verify
        if not target:
            built = bundles()
            if not built:
                print('no bundles built yet.')
                return 1
            target = os.path.join(built[-1][1], built[-1][0])
        elif not os.path.isabs(target):
            if os.path.isfile(os.path.join(OUT_DIR, target)):
                target = os.path.join(OUT_DIR, target)
            elif os.path.isfile(os.path.join(HERE, target)):
                target = os.path.join(HERE, target)
        return cmd_verify(target if os.path.isabs(target) else os.path.join(OUT_DIR, target))
    if a.list:
        return cmd_list()
    if a.prune is not None:
        return cmd_prune(a.prune)
    if a.annotated_only and a.clean_only:
        ap.error('--annotated-only and --clean-only exclude each other')
    if a.full and a.v4_only:
        ap.error('--full and --v4-only exclude each other')
    rc = 0
    if not a.clean_only:
        rc |= build(a, clean=False)
    if not a.annotated_only:
        rc |= build(a, clean=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
