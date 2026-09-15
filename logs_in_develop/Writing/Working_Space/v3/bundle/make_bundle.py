#!/usr/bin/env python3
r"""Flatten the split v3 draft into ONE self-contained .tex, and package it for Overleaf.

WHY THIS EXISTS
---------------
v3 is split by chapter so that v2 and v3 can be worked on in parallel
(see ../inherited/MANIFEST.md). That split is right for editing and wrong for
uploading: Overleaf wants a project, and a reviewer wants one file. This tool
produces both, mechanically, from the same sources -- so the flat file is never
edited by hand and can never drift from the split one.

    ../thesis_v3.tex  +  parts/  +  chapters/   ->   bundle/thesis_v3_<stamp>.tex
                                                     bundle/thesis_v3_<stamp>.zip

WHAT IT DOES, EXACTLY
---------------------
1. Reads ../thesis_v3.tex and recursively replaces every ``\input{X}`` whose
   target FILE EXISTS with that file's bytes, wrapped in BEGIN/END banners
   naming the source and its SHA-256 prefix.

   An ``\input`` whose target does NOT exist is left verbatim. This is not a
   fallback, it is the correct behaviour: the inherited preamble and front
   matter carry ``\input{settings}`` and ``\input{pages/cover}`` inside the
   ``\ifstandalone ... \else`` branch, for the day the draft is merged into the
   TUM template. ``\standalonetrue`` is set, so LaTeX never reads them, and
   inlining them would be impossible anyway.

2. Rewrites every ``\includegraphics[opts]{name}`` to ``\fmpccgraphic[opts]{name}``
   and injects the definition of that macro just before ``\begin{document}``.
   ``\fmpccgraphic`` uses a real ``\includegraphics`` when a compilable file
   (.pdf/.png/.jpg) is present and otherwise draws a labelled placeholder box.

   THE POINT: the bundle compiles even though the figures are currently SVG only
   and no SVG converter exists in this container. A missing figure becomes a
   visible box that names the missing file, not a build failure. Run
   Data_Analysis/DA_in_Paper/plotting/svg/svg2pdf.sh where a converter exists,
   export to the draft (plotting/export_to_draft.py v3), re-run this tool, and
   the real figures are picked up with no source change. ``--no-graphics-shim``
   turns the rewrite off.

3. Copies the two bibliography resources and the figures directory next to the
   .tex, and zips the lot. Both .bib files are needed: v3 deliberately keeps its
   own entries separate from v2's inherited file (see ../bibliography_v3.bib).

4. VERIFIES the result before writing it, and refuses to write on failure:
   line-count conservation, no remaining inlinable \input, brace balance,
   environment balance, and \begin{document}/\end{document} present exactly once.

5. NEW-SECTIONS-ONLY IS THE DEFAULT. v3 inherits Chapters 1-4 from v2 and
   normally never edits them, so re-typesetting them on every build only buries
   the part under review. By default each v2 content chapter is COLLAPSED: its
   text is replaced by a grey "v2 SECTION" box, and only its numbered \chapter /
   \section headings are kept -- so every chapter and section number, and every
   \autoref from a v3 chapter into a v2 one, stays identical to the full build.
   ``--full`` builds everything (use it for the complete thesis on Overleaf).

   Two guards keep "v2 sections are untouched" honest rather than assumed:
     * only chapters/ files whose policy (tools/sync_v2.py POLICY) is inherit or
       merge are candidates -- parts/ (preamble, front and back matter) are
       structural and are always inlined, or the document would not compile;
     * a candidate collapses ONLY IF it is byte-identical to its v2 baseline in
       inherited/v2_base/. If v3 has edited it, it is inlined in full and the
       build says so, because collapsing it would hide exactly that edit.

USAGE
-----
    python3 bundle/make_bundle.py                 # NEW sections only; v2 chapters collapsed
    python3 bundle/make_bundle.py --full          # everything, v2 chapters included
    python3 bundle/make_bundle.py --no-zip        # just the .tex
    python3 bundle/make_bundle.py --prune 5       # keep only the 5 newest bundles
    python3 bundle/make_bundle.py --list          # what has been built
    python3 bundle/make_bundle.py --verify        # prove the newest bundle matches the tree

Every run writes a new timestamped pair and appends one row to BUNDLE_LOG.md.
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
V3 = os.path.dirname(HERE)
MASTER = os.path.join(V3, 'thesis_v3.tex')
BIBS = ['bibliography.bib', 'bibliography_v3.bib']
FIGDIR = os.path.join(V3, 'figures')
LOG = os.path.join(HERE, 'BUNDLE_LOG.md')
COMPILABLE = ('.pdf', '.png', '.jpg', '.jpeg')
MAX_DEPTH = 8
BASE_DIR = os.path.join(V3, 'inherited', 'v2_base')
STATE = os.path.join(V3, 'inherited', 'SYNC_STATE.json')

# Which file belongs to whom is decided in ONE place, tools/sync_v2.py. Importing
# it rather than restating it means the bundler cannot disagree with the sync tool
# about what counts as a v2 section.
sys.path.insert(0, os.path.join(V3, 'tools'))
try:
    from sync_v2 import POLICY  # noqa: E402
except ImportError as e:        # pragma: no cover
    sys.exit(f'make_bundle: cannot read the ownership policy from tools/sync_v2.py ({e})')

# thesis_v3_<stamp>[_new|_full][_<n>].tex -- the unsuffixed form is pre-v3.6.
RE_BUNDLE = re.compile(r'thesis_v3_\d{8}_\d{6}(?:_(?:new|full))?(?:_\d+)?\.tex')
# Headings kept in a collapsed chapter: numbered ones only. Starred headings do
# not move any counter, so dropping them changes no number anywhere.
HEADING_PREFIXES = ('\\chapter{', '\\chapter[', '\\section{', '\\section[',
                    '\\subsection{', '\\subsection[', '\\subsubsection{', '\\subsubsection[')
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')
RE_REF = re.compile(r'\\(?:auto|eq)?ref\{([^}]+)\}')

RE_INPUT_LINE = re.compile(r'^(?P<pre>[^%\n]*?)\\input\{(?P<target>[^}]+)\}(?P<post>[^\n]*)$')
RE_GRAPHIC = re.compile(r'\\includegraphics(?P<opts>\[[^\]]*\])?\{(?P<name>[^}]+)\}')
RE_BEGIN = re.compile(r'\\begin\{([^}]+)\}')
RE_END = re.compile(r'\\end\{([^}]+)\}')

# Injected just before \begin{document}. Kept as a plain string rather than
# built up in code so that what lands in the .tex is exactly what is written
# here and can be read as TeX.
GRAPHICS_SHIM = r"""
% =============================================================================
%  GENERATED BY bundle/make_bundle.py -- DO NOT EDIT IN THE BUNDLE
% -----------------------------------------------------------------------------
%  Figure guard. The thesis figures are generated as SVG (DA_in_Paper), and
%  \includegraphics cannot read SVG. Rather than let that fail the build, every
%  \includegraphics call in this file was rewritten to \fmpccgraphic, which
%  falls back to a placeholder box naming the file it could not find.
%
%  TO GET THE REAL FIGURES: run DA_in_Paper/plotting/svg/svg2pdf.sh on a machine with
%  rsvg-convert, inkscape or cairosvg, export to v3, then re-run bundle/make_bundle.py. The
%  PDFs are picked up automatically and no source changes.
% =============================================================================
\makeatletter
% \detokenize is required, not decorative: every generated figure name contains
% underscores (fig_avoiding_k_ladder), and a bare _ inside \texttt is a subscript
% in text mode -> "Missing $ inserted". Without it the guard would fail on
% exactly the case it exists to handle.
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

# The --svg-package variant. Overleaf ships Inkscape and enables the shell
# escape that the `svg' package needs, so this renders the generated SVGs
# directly and no conversion step is required. It is OPT-IN rather than the
# default because it is a property of the BUILD HOST, not of the document: a TeX
# installation without Inkscape fails outright on it, and the default has to
# compile anywhere. \includesvg is tried only after .pdf and .png, so a
# converted figure still wins wherever one exists.
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


# Injected before \begin{document} whenever at least one chapter was collapsed.
# ASCII-only arguments: the version tag is passed as e.g. "v2.9 (2026-09-11)",
# never the changelog's own heading, whose em dash and middle dot are safe in a
# comment but not worth risking inside \texttt.
V2_SHIM = r"""
% =============================================================================
%  GENERATED BY bundle/make_bundle.py -- NEW-SECTIONS-ONLY BUILD
% -----------------------------------------------------------------------------
%  Chapters inherited unchanged from v2 were collapsed to their numbered
%  headings; each carries the box below instead of its text. Rebuild with
%  --full for the complete document.
% =============================================================================
\newcommand{\fmpccvtwo}[3]{%
  \par\medskip\noindent
  \fcolorbox{gray}{white}{\parbox{0.96\linewidth}{\centering\small
    \textbf{[v2 SECTION --- not built in this bundle]}\\[0.3em]
    \texttt{\detokenize{#1}}, #2 lines, inherited unchanged from #3.\\[0.2em]
    Its section headings are kept, so numbering and cross-references match the full build.\\
    Rebuild with \texttt{make\_bundle.py --full} to include its text.}}%
  \par\medskip
}
"""


def v2_tag():
    """-> (short ASCII tag for the TeX box, full heading for comments)."""
    try:
        with open(STATE) as f:
            full = json.load(f).get('v2_version', 'unknown')
    except (OSError, ValueError):
        full = 'unknown'
    m = re.match(r'(v\d+\.\d+)\D+(\d{4}-\d{2}-\d{2})', full)
    return (f'{m.group(1)} ({m.group(2)})' if m else 'an unrecorded revision'), full


def collapse_decision(path):
    """-> (collapse?, reason). The two guards described in the module docstring."""
    rel = os.path.relpath(path, V3)
    if not rel.startswith('chapters/'):
        return False, 'structural'
    if POLICY.get(rel) not in ('inherit', 'merge'):
        return False, 'v3-owned'
    base = os.path.join(BASE_DIR, rel)
    if not os.path.isfile(base):
        return False, 'no v2 baseline recorded'
    with open(path, 'rb') as a, open(base, 'rb') as b:
        if a.read() != b.read():
            return False, 'EDITED IN v3'
    return True, 'unchanged v2'


def skeleton(path, tag, tag_full):
    """A collapsed chapter: banner, numbered headings, and the v2 box."""
    rel = os.path.relpath(path, V3)
    body = open(path).read().split('\n')
    heads = [ln.rstrip() for ln in body if ln.startswith(HEADING_PREFIXES)]
    if not heads or not heads[0].startswith(('\\chapter{', '\\chapter[')):
        sys.exit(f'make_bundle: {rel} does not open with a \\chapter heading; refusing to collapse it')
    out = [f'% <<<<<<<<<<<<<<<< COLLAPSED {rel}  ({sha(path)}) <<<<<<<<<<<<<<<<',
           f'% v2 SECTION, not built. Inherited unchanged from {tag_full}',
           f'% Kept: its {len(heads)} numbered heading(s). Dropped: {len(body)} lines of text.',
           heads[0],
           f'\\fmpccvtwo{{{rel}}}{{{len(body)}}}{{{tag}}}']
    out += heads[1:]
    out.append(f'% >>>>>>>>>>>>>>>> END COLLAPSED {rel} >>>>>>>>>>>>>>>>')
    return out, len(body), len(heads)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def resolve(target):
    """An \\input target -> an existing path under v3, or None.

    None is a legitimate answer, not an error: see the module docstring.
    """
    for cand in (target, target + '.tex'):
        p = os.path.normpath(os.path.join(V3, cand))
        # Never follow an \input outside the v3 tree.
        if p.startswith(V3 + os.sep) and os.path.isfile(p):
            return p
    return None


def flatten(path, seen, sources, depth=0, collapse=False, collapsed=None, kept=None, tag=('', '')):
    """-> list of output lines, with every resolvable \\input replaced in place.

    With ``collapse``, a v2 chapter that passes ``collapse_decision`` becomes a
    skeleton instead; it is recorded in ``collapsed`` and NOT in ``sources``, so
    the line-conservation check still counts only what was actually inlined. A
    v2 chapter that v3 has edited is inlined and recorded in ``kept``.
    """
    rel = os.path.relpath(path, V3)
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
        crel = os.path.relpath(child, V3)
        if m.group('pre').strip():
            out.append(m.group('pre').rstrip())
        out.append('')
        if collapse:
            ok, why = collapse_decision(child)
            if ok:
                sk, n_lines, n_heads = skeleton(child, *tag)
                out.extend(sk)
                collapsed.append((crel, n_lines, n_heads, sha(child)))
                out.append('')
                if m.group('post').strip():
                    out.append(m.group('post').lstrip())
                continue
            if why == 'EDITED IN v3':
                kept.append(crel)
        out.append(f'% <<<<<<<<<<<<<<<< BEGIN {crel}  ({sha(child)}) <<<<<<<<<<<<<<<<')
        out.extend(flatten(child, seen, sources, depth + 1, collapse, collapsed, kept, tag))
        out.append(f'% >>>>>>>>>>>>>>>> END   {crel} >>>>>>>>>>>>>>>>')
        out.append('')
        if m.group('post').strip():
            out.append(m.group('post').lstrip())
    return out


def split_comment(line):
    """-> (code, comment). The part TeX executes, and the rest.

    Used by BOTH the rewrite and its inverse in cmd_verify. They have to agree
    exactly: verify once undid the rewrite on whole lines, which corrupted a
    comment that happened to mention \\fmpccgraphic by name and reported a
    mismatch against an untouched file.
    """
    m = re.search(r'(?<!\\)%', line)
    return (line[:m.start()], line[m.start():]) if m else (line, '')


def rewrite_graphics(lines):
    """\\includegraphics -> \\fmpccgraphic, outside comments. -> (lines, count)."""
    n = 0
    out = []
    for line in lines:
        code, comment = split_comment(line)
        code, k = RE_GRAPHIC.subn(
            lambda m: r'\fmpccgraphic' + (m.group('opts') or '') + '{' + m.group('name') + '}',
            code)
        n += k
        out.append(code + comment)
    return out, n


def inject_shim(lines, shim):
    """Put the figure guard immediately before \\begin{document}."""
    for i, line in enumerate(lines):
        if line.lstrip().startswith(r'\begin{document}'):
            return lines[:i] + shim.split('\n') + lines[i:]
    sys.exit(r'make_bundle: no \begin{document} found -- refusing to inject the figure guard')


def strip_comments(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in text.split('\n'))


def verify(lines, sources, expect_lines):
    """Checks that must hold for a flatten to be correct. -> list of failures."""
    text = '\n'.join(lines)
    code = strip_comments(text)
    bad = []

    # 1. Nothing inlinable left behind.
    left = [m.group('target') for ln in code.split('\n')
            if (m := RE_INPUT_LINE.match(ln)) and resolve(m.group('target'))]
    if left:
        bad.append(f'{len(left)} resolvable \\input still present: {", ".join(left)}')

    # 2. Line-count conservation. Every source line must appear; the surplus is
    #    exactly the banners and blank lines this tool adds, plus the shim.
    if len(lines) < expect_lines:
        bad.append(f'lost content: {len(lines)} output lines < {expect_lines} source lines')

    # 3. Braces, with escaped braces excluded from BOTH counts.
    bare = re.sub(r'(?<!\\)\\[{}]', '', code)
    if (d := bare.count('{') - bare.count('}')):
        bad.append(f'brace delta {d:+d}')

    # 4. Environments.
    op, cl = RE_BEGIN.findall(code), RE_END.findall(code)
    for env in set(op) | set(cl):
        if op.count(env) != cl.count(env):
            bad.append(f'environment {env}: {op.count(env)} begin, {cl.count(env)} end')

    # 5. Exactly one document.
    for what, n in ((r'\begin{document}', code.count(r'\begin{document}')),
                    (r'\end{document}', code.count(r'\end{document}')),
                    (r'\documentclass', code.count(r'\documentclass'))):
        if n != 1:
            bad.append(f'{what} appears {n} times, expected 1')

    return bad


def header(stamp, sources, n_graphics, shim, collapsed=(), tag_full=''):
    """`shim` is a human-readable description of the figure mode."""
    w = max([len(r) for r, _n, _s in sources] + [len(c[0]) for c in collapsed])
    rows = '\n'.join(f'%   {r.ljust(w)}  {n:5d} lines  sha256:{s}' for r, n, s in sources)
    if collapsed:
        rows += ('\n%\n%  NEW-SECTIONS-ONLY BUILD. Collapsed to headings -- v2 sections, not built,'
                 f'\n%  inherited unchanged from {tag_full}:\n'
                 + '\n'.join(f'%   {r.ljust(w)}  {n:5d} lines  sha256:{h}  ({k} headings kept)'
                              for r, n, k, h in collapsed)
                 + '\n%  Rebuild with --full for the complete document.')
    return f"""% =============================================================================
%  FM-PCC MASTER'S THESIS -- v3 -- FLATTENED BUILD ({'NEW SECTIONS ONLY' if collapsed else 'FULL'})
% -----------------------------------------------------------------------------
%  GENERATED FILE. DO NOT EDIT.
%
%  Built {stamp} by bundle/make_bundle.py from the split sources listed below.
%  Edit those, then re-run the tool; any edit made here is lost on the next run
%  and, worse, silently diverges from the draft everyone else is reading.
%
%  Sources, in \\input order, with the SHA-256 prefix of each as built:
{rows}
%
%  Graphics: {n_graphics} \\includegraphics call(s) rewritten to \\fmpccgraphic
%  {shim}.
%
%  BUILD (Overleaf or local):
%    pdflatex <this file> ; biber <jobname> ; pdflatex x2
%  Needs, next to this file: bibliography.bib, bibliography_v3.bib, figures/
%  -- all of which are in the .zip this tool writes alongside it.
% =============================================================================
"""


def build(args):
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    kind = 'full' if args.full else 'new'
    base = f'thesis_v3_{stamp}_{kind}'
    # Two runs inside the same second would otherwise collide and the later one
    # would silently overwrite the earlier -- which defeats the point of stamping
    # them. Suffix instead of clobbering; a bundle is never overwritten.
    if os.path.exists(os.path.join(HERE, base + '.tex')):
        n = 2
        while os.path.exists(os.path.join(HERE, f'{base}_{n}.tex')):
            n += 1
        base = f'{base}_{n}'
    tex_path = os.path.join(HERE, base + '.tex')

    sources, collapsed, kept = [], [], []
    tag, tag_full = v2_tag()
    lines = flatten(MASTER, set(), sources, collapse=not args.full,
                    collapsed=collapsed, kept=kept, tag=(tag, tag_full))
    src_lines = sum(n for _r, n, _s in sources)
    if collapsed:
        lines = inject_shim(lines, V2_SHIM)

    n_graphics = 0
    if not args.no_graphics_shim:
        # ORDER MATTERS: rewrite first, THEN inject. The shim's own body calls
        # \includegraphics; injecting it before the rewrite would turn those
        # calls into \fmpccgraphic and make the macro infinitely recursive.
        lines, n_graphics = rewrite_graphics(lines)
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
    body = header(stamp, sources, n_graphics, mode, collapsed, tag_full) + '\n'.join(lines)
    if not body.endswith('\n'):
        body += '\n'
    with open(tex_path, 'w') as f:
        f.write(body)

    print(f'{"FULL" if args.full else "NEW SECTIONS ONLY"} -- {len(sources)} source file(s) inlined, '
          f'{src_lines} lines -> {len(lines)} lines, {len(body) // 1024} KB')
    if collapsed:
        print(f'  collapsed to headings (v2, unchanged, {tag}): '
              + ', '.join(f'{os.path.basename(r)} ({n} lines)' for r, n, _k, _h in collapsed))
    if kept:
        print('  inlined IN FULL because v3 has edited them: ' + ', '.join(kept))
    if not args.full:
        # Every reference from a built section into a collapsed one must still
        # resolve, or the review build prints "??". Headings carry their labels,
        # so section references survive; anything deeper would not.
        code = strip_comments('\n'.join(lines))
        dangling = sorted(set(RE_REF.findall(code)) - set(RE_LABEL.findall(code)))
        if dangling:
            print(f'  NOTE: {len(dangling)} reference(s) point into collapsed text and will print '
                  f'"??": ' + ', '.join(dangling[:8]) + (' ...' if len(dangling) > 8 else ''))
        else:
            print('  cross-references: all resolve (chapter and section numbers match the full build)')
    print(f'  {os.path.relpath(tex_path, V3)}')

    # ---- assets next to the .tex, and the Overleaf zip ----------------------
    assets = []
    for b in BIBS:
        src = os.path.join(V3, b)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(HERE, b))
            assets.append(b)
    figs = []
    if os.path.isdir(FIGDIR):
        dst = os.path.join(HERE, 'figures')
        # Clear the staging copy first: it used to keep figures from earlier runs,
        # which made this folder look like it held figures the draft no longer uses.
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(FIGDIR)):
            if os.path.splitext(fn)[1].lower() in COMPILABLE + ('.svg',):
                shutil.copyfile(os.path.join(FIGDIR, fn), os.path.join(dst, fn))
                figs.append(fn)
    # Count FIGURES, not files. A figure is one stem; \fmpccgraphic resolves it to
    # .pdf, else .png, else a placeholder, so a stem that ships both an .svg and a
    # .png renders fine. Counting files called that figure SVG-only and printed a
    # warning about placeholders for figures that were never going to be placeholders.
    stems = {}
    for f in figs:
        stem, ext = os.path.splitext(f)
        stems.setdefault(stem, set()).add(ext.lower())
    usable = sorted(st for st, ex in stems.items() if ex & set(COMPILABLE))
    placeholder = sorted(st for st in stems if st not in usable)
    print(f'  assets: {len(assets)} bibliography file(s), {len(figs)} figure file(s) '
          f'= {len(stems)} figure(s) ({len(usable)} render, {len(placeholder)} placeholder)')
    if args.svg_package:
        print('  figure mode: \\usepackage{svg} -- SVGs render directly if the host has Inkscape')

    zip_path = ''
    if not args.no_zip:
        zip_path = os.path.join(HERE, base + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(tex_path, base + '.tex')
            for b in assets:
                z.write(os.path.join(HERE, b), b)
            for fn in figs:
                z.write(os.path.join(HERE, 'figures', fn), f'figures/{fn}')
        print(f'  {os.path.relpath(zip_path, V3)}  '
              f'({os.path.getsize(zip_path) // 1024} KB) -- upload this to Overleaf')

    with open(LOG, 'a') as f:
        if os.path.getsize(LOG) == 0 if os.path.exists(LOG) else True:
            pass
        f.write(f'| {stamp} | {len(sources)} | {src_lines} | {len(lines)} | '
                f'{n_graphics} | {len(usable)}/{len(figs)} | `{base}.tex` |\n')

    if placeholder and not args.svg_package:
        print(f'\nNOTE: {len(placeholder)} figure(s) ship only as SVG and render as a labelled '
              f'placeholder box:\n      ' + ', '.join(placeholder) +
              '\n      For real figures either re-run with --svg-package (Overleaf has Inkscape),'
              '\n      or run DA_in_Paper/plotting/svg/svg2pdf.sh + export where a converter '
              'exists and rebuild.')
    return 0


RE_BANNER_BEGIN = re.compile(r'^% <{16} BEGIN (\S+)')
RE_BANNER_END = re.compile(r'^% >{16} END\s+(?!COLLAPSED\b)(\S+)')
RE_BANNER_COLLAPSED = re.compile(r'^% <{16} COLLAPSED (\S+)\s+\(([0-9a-f]{12})\)')


def cmd_verify(path):
    """Extract every inlined source back out of a bundle and diff it.

    This is the check that matters: it proves the flattening is byte-faithful,
    not merely that it produced something that looks like TeX. The only
    transformation the tool applies -- \\includegraphics -> \\fmpccgraphic -- is
    undone before comparing, so a difference here means real content drift.

    Use it on an OLD bundle to answer "was this built from what is on disk now?".
    A mismatch then is not a bug: it means the sources moved after that bundle
    was built, which is exactly what you wanted to know.
    """
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

    bad = 0
    for name, (a, b) in sorted(blocks.items()):
        src = os.path.join(V3, name)
        if not os.path.isfile(src):
            print(f'  GONE {name}  (inlined in the bundle, absent from the tree now)')
            bad += 1
            continue
        def unrewrite(ln):
            code, comment = split_comment(ln)
            return code.replace(r'\fmpccgraphic', r'\includegraphics') + comment

        got = trim(unrewrite(ln) for ln in lines[a:b])
        want = trim(open(src).read().split('\n'))
        if got == want:
            print(f'  OK   {name}  ({len(want)} lines)')
        else:
            first = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
            print(f'  DIFF {name}  (bundle {len(got)} lines, source {len(want)}; '
                  f'first difference at source line {first + 1})')
            bad += 1
    # A collapsed chapter has no text to diff; what can be checked is that the
    # file it stands for has not changed since the bundle was built.
    for name, recorded in collapsed:
        src = os.path.join(V3, name)
        if not os.path.isfile(src):
            print(f'  GONE {name}  (collapsed in the bundle, absent from the tree now)')
            bad += 1
        elif sha(src) == recorded:
            print(f'  OK   {name}  (collapsed v2 section; unchanged since the build)')
        else:
            print(f'  DIFF {name}  (collapsed v2 section; the source has changed since the build)')
            bad += 1
    print()
    print(f'{len(blocks)} inlined + {len(collapsed)} collapsed source(s); '
          + ('byte-faithful throughout.' if not bad else
             f'{bad} MISMATCH -- the sources have moved since this bundle was built.'))
    return 1 if bad else 0


def bundles():
    """Built bundles, oldest first. Ordered by modification time: a _full and a
    _new built in the same second would otherwise sort by name, not by age."""
    rows = [f for f in os.listdir(HERE) if RE_BUNDLE.fullmatch(f)]
    return sorted(rows, key=lambda f: (os.path.getmtime(os.path.join(HERE, f)), f))


def cmd_list():
    rows = bundles()
    if not rows:
        print('no bundles built yet.')
        return 0
    for f in rows:
        p = os.path.join(HERE, f)
        z = p[:-4] + '.zip'
        print(f'  {f}  {os.path.getsize(p) // 1024:4d} KB'
              + (f'  + zip {os.path.getsize(z) // 1024} KB' if os.path.exists(z) else ''))
    print(f'\n{len(rows)} bundle(s). Newest: {rows[-1]}')
    return 0


def cmd_prune(keep):
    rows = bundles()
    doomed = rows[:-keep] if keep > 0 else rows
    for f in doomed:
        os.remove(os.path.join(HERE, f))
        z = os.path.join(HERE, f[:-4] + '.zip')
        if os.path.exists(z):
            os.remove(z)
        print(f'removed {f}')
    print(f'{len(rows) - len(doomed)} bundle(s) kept.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true',
                    help='build everything, including the chapters inherited from v2 (default: new '
                         'sections only, with unchanged v2 chapters collapsed to their headings)')
    ap.add_argument('--no-zip', action='store_true', help='write only the .tex')
    ap.add_argument('--no-graphics-shim', action='store_true',
                    help='keep \\includegraphics as-is; a missing figure then fails the build')
    ap.add_argument('--svg-package', action='store_true',
                    help='render the SVGs directly via \\usepackage{svg} (needs Inkscape on the '
                         'build host -- Overleaf has it; a plain TeX install may not)')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--prune', type=int, metavar='N', help='delete all but the N newest bundles')
    ap.add_argument('--verify', metavar='BUNDLE.tex', nargs='?', const='',
                    help='extract every inlined source back out of a bundle and diff it against '
                         'the tree (default: the newest bundle)')
    a = ap.parse_args()

    if not os.path.exists(LOG):
        with open(LOG, 'w') as f:
            f.write('# Bundle log\n\nAppended to by `make_bundle.py`. '
                    'Bundles are build output; nothing reads them back.\n\n'
                    '| built | files | source lines | output lines | graphics | figures usable | file |\n'
                    '| :-- | --: | --: | --: | --: | --: | :-- |\n')
    if a.verify is not None:
        target = a.verify
        if not target:
            built = bundles()
            if not built:
                print('no bundles built yet.')
                return 1
            target = built[-1]
        return cmd_verify(target if os.path.isabs(target) else os.path.join(HERE, target))
    if a.list:
        return cmd_list()
    if a.prune is not None:
        return cmd_prune(a.prune)
    return build(a)


if __name__ == '__main__':
    sys.exit(main())
