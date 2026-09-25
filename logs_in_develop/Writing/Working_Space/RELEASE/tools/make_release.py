#!/usr/bin/env python3
r"""make_release.py -- assemble the submission-clean thesis from the LIVE v2 / v3 / v4 drafts.

WHAT IT DOES
------------
1. Takes every chapter from its OWNER's live file, never from a bundle (bundles may be stale):
     v2/thesis_v2.tex                 -> preamble, front matter (the author's abstract), Ch 1-4,
                                         the acronym list, bibliography.bib
     v3/chapters/05_*.tex, 06_*.tex   -> Ch 5-6          (+ parts/00_preamble_v3.tex, bibliography_v3.bib)
     v4/chapters/07_*, 08_*, 09_*     -> Ch 7-9, appendix (+ nested app_* inputs, parts/00_preamble_v4.tex,
                                         bibliography_v4.bib)
2. Cleans every .tex to a submission state:
     - every comment removed (a bare line-ending ``%`` is syntax and stays);
     - drafting macros removed: \srcnote \dataref \hole \longdata \flawed \outdated \todofigure;
       \guard and \provisional unwrapped to their text; deadblock / pillarsflawed unwrapped;
       their definitions and switches (\ifsubmission, \ifpillarsflawed, \ifappendixfull, \ifstandalone)
       removed from the preambles;
     - build switches resolved statically: \ifstandalone -> the TUM-template branch (or the standalone
       branch with --standalone), \ifappendixfull -> tables printed (hidden with --appendix-short).
3. Assembles main.tex on the TUM template: settings.tex, pages/, logos/, Makefile, .latexmkrc and
   main.xmpdata are copied from Template_DONT_CHANGE (read-only; comments stripped from the .tex copies),
   pages/abstract.tex is generated from v2's abstract, ONE aggregated bibliography.bib (dev fields and
   comments removed, duplicate keys refused), figures/ with exactly one file per figure (.pdf preferred,
   then .png/.jpg).
4. Checks the result mechanically: \input tree, labels/refs, citations, acronyms, environments, braces,
   figures, leftover drafting macros/switches/comments, dev residue in prose.
5. Estimates the page count (there is no TeX toolchain here; nothing is compiled), writes
   RELEASE_NOTES_<stamp>.md into the release folder (holes, bugs, cleaning report, outline, LoF/LoT preview),
   zips the LaTeX project (notes stay outside the zip) and prepends an entry to ../CHANGELOG.md.

USAGE
-----
    python3 tools/make_release.py [--tag GOLDEN_TEMPLATE] [--appendix-short] [--standalone] [--no-zip]
                                  [--outdir DIR] [--dry-run]
    python3 tools/make_release.py --attach-pdf main.pdf [--release FOLDER]   # record the compiled page count
    python3 tools/make_release.py --rezip FOLDER                              # rebuild the zip of a folder
    python3 tools/make_release.py --list

Outputs are build products: never edit them; fix the owner's draft and rebuild. Content questions go to
the owning draft (v2: Ch 1-4 / v3: Ch 5-6 / v4: Ch 7-9) through cross_draft/, never to a release.
"""
import argparse
import datetime
import hashlib
import math
import os
import re
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(HERE)
WS = os.path.dirname(RELEASE)                      # Working_Space
WRITING = os.path.dirname(WS)                      # Writing
REPO = os.path.dirname(os.path.dirname(WRITING))   # FM-PCC
V2, V3, V4 = (os.path.join(WS, v) for v in ('v2', 'v3', 'v4'))
TEMPLATE = os.path.join(WRITING, 'Template_DONT_CHANGE')
DA_FIGS = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'figures')
OUT_DEFAULT = os.path.join(RELEASE, 'output')
CHANGELOG = os.path.join(RELEASE, 'CHANGELOG.md')
FRONT_DIR = os.path.join(RELEASE, 'front')      # optional author-supplied front-matter text (acknowledgments.tex)
LATEX_SUBDIR = 'latex'                           # the LaTeX project inside a build folder; the zip sits next to it

PAGE_MIN, PAGE_MAX = 60, 200          # the hard limits the author set (warn outside)
PAGE_GUIDE = (60, 80)                 # TUM I6 orientation for a master's thesis (information only)
# Page model, calibrated on the first compiled release (Overleaf, 2026-09-25: 185 pages at v2.28/v3.99/v4.1a).
WORDS_PER_PAGE = 370
BIB_LINES_PER_ENTRY = 4.6

# --- what is a drafting mark ----------------------------------------------------------------
REMOVE_MACROS = {'srcnote': False, 'dataref': False, 'flawed': True, 'outdated': True,
                 'longdata': True, 'hole': True, 'todofigure': True}     # True = record the text as a HOLE
UNWRAP_MACROS = {'guard': False, 'provisional': True}                    # keep the text; True = record it
UNWRAP_ENVS = {'deadblock': 1, 'pillarsflawed': 0}                       # value = number of \begin arguments
DROP_SWITCHES = ['standalone', 'submission', 'pillarsflawed', 'appendixfull']
DROP_DEFS = list(REMOVE_MACROS) + list(UNWRAP_MACROS)

# v2's chapter labels -> release file names (v2 owns exactly these; anything else in v2 is a placeholder)
V2_CHAPTERS = {
    'ch:introduction': 'chapters/01_introduction.tex',
    'ch:background': 'chapters/02_background.tex',
    'ch:related': 'chapters/03_related_work.tex',
    'ch:method': 'chapters/04_method.tex',
}
V3_CHAPTERS = ['chapters/05_setup.tex', 'chapters/06_results.tex']
V4_CHAPTERS = ['chapters/07_conclusion.tex', 'chapters/08_discussion.tex', 'chapters/09_appendix.tex']
TEMPLATE_PAGES = ['cover', 'title', 'disclaimer', 'acknowledgments']   # acknowledgments/cover are conditional, see build()
TEMPLATE_VERBATIM = ['Makefile', '.latexmkrc']

RE_IF = re.compile(r'\\(if[a-zA-Z@]*|else|fi)(?![a-zA-Z@])')
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')
RE_REF = re.compile(r'\\(?:auto|eq|c|C|page)?ref\*?\{([^}]+)\}')
RE_CITE = re.compile(r'\\(?:paren|foot|text|auto|smart)?cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}')
RE_BEGIN = re.compile(r'\\begin\{([^}]+)\}')
RE_END = re.compile(r'\\end\{([^}]+)\}')
RE_GRAPHIC = re.compile(r'\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}')
RE_INPUT = re.compile(r'^[^%\n]*?\\input\{([^}]+)\}', re.M)
RE_AC = re.compile(r'\\ac[sflp]*\*?\{([A-Za-z0-9-]+)\}')
RE_ACRO = re.compile(r'\\acro\{([^}]+)\}')
RE_BIBENTRY = re.compile(r'^@(\w+)\s*\{\s*([^,\s]+)\s*,', re.M)
RE_VERSION = re.compile(r'^## v(\d+)\.(\d+)([a-z]?)\b(.*)$')
RESIDUE = [r'\bv[234]\.\d+[a-z]?\b', r'\(author\b', r'\bauthor:', r'\bHOLE\b', r'\bTODO\b', r'\bFIXME\b',
           r'\bXXX\b', r'\bdead block\b', r'Working\\?_Space', r'DA\\?_in\\?_Paper', r'\bchangelog\b',
           r'\.py\b', r'\.md\b', r'\bsrcnote\b', r'\bdataref\b', r'\bwithheld\b',
           r'\\texttt\{[a-z]+:[a-z]']


# =============================================================================================
#  small helpers
# =============================================================================================
def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def scan_group(text, i):
    """text[i] == '{' -> index just past the matching '}' (escapes honoured)."""
    depth, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '\\':
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError('unbalanced group')


def skip_ws(text, i):
    while i < len(text) and text[i] in ' \t\n':
        i += 1
    return i


def skip_opt(text, i):
    """skip one [..] optional argument at i (no nesting of ']' inside)."""
    if i < len(text) and text[i] == '[':
        j = text.find(']', i)
        return j + 1 if j >= 0 else i
    return i


def dedent(text):
    lines = text.split('\n')
    ind = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    k = min(ind) if ind else 0
    return '\n'.join(l[k:] if l.strip() else '' for l in lines)


def tidy(text):
    """trailing whitespace off, runs of blank lines collapsed to one, single trailing newline."""
    lines = [l.rstrip() for l in text.split('\n')]
    out, blank = [], True
    for l in lines:
        if l == '':
            if not blank:
                out.append(l)
            blank = True
        else:
            out.append(l)
            blank = False
    while out and out[-1] == '':
        out.pop()
    return '\n'.join(out) + '\n'


def plain_words(text):
    """rough prose extraction for counting: math -> one token, commands and braces dropped."""
    t = re.sub(r'\$[^$]*\$', ' M ', text)
    t = re.sub(r'\\[a-zA-Z@]+\*?', ' ', t)
    t = re.sub(r'[{}\[\]&~]', ' ', t)
    return [w for w in t.split() if re.search(r'[A-Za-z0-9]', w)]


def display_title(text):
    t = re.sub(r'\\ac[sflp]*\*?\{([^}]*)\}', r'\1', text)
    t = re.sub(r'\\(?:emph|textbf|texttt|textit|enquote|mainconfig)\{([^}]*)\}', r'\1', t)
    t = re.sub(r'\\[a-zA-Z@]+\*?', '', t)
    t = re.sub(r'[{}~]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


# =============================================================================================
#  the cleaner
# =============================================================================================
class Report:
    def __init__(self):
        self.holes = []          # (kind, file, line, text)
        self.bugs = []           # str
        self.info = []           # str
        self.cleaned = {}        # file -> dict of counts
        self.residue = []        # (file, line, pattern, excerpt)
        self.comment_todos = []  # (file, text)

    def hole(self, kind, rel, line, text):
        self.holes.append((kind, rel, line, re.sub(r'\s+', ' ', text).strip()))


def strip_comments(text, rel, rep):
    """Every comment goes. A bare line-ending % (no text after it) is syntax and stays. A comment glued
    to code (``code% note``) becomes a bare %, so no space token is introduced; a comment after
    whitespace (``code  % note``) is dropped with the whitespace, which leaves the same space token."""
    out, n_full, n_trail = [], 0, 0
    for ln in text.split('\n'):
        i, n, cut = 0, len(ln), None
        while i < n:
            c = ln[i]
            if c == '\\':
                i += 2
                continue
            if c == '%':
                cut = i
                break
            i += 1
        if cut is None:
            out.append(ln)
            continue
        code, comment = ln[:cut], ln[cut + 1:]
        if re.search(r'\bTODO\b|\bFIXME\b|\bXXX\b', comment):
            rep.comment_todos.append((rel, comment.strip()))
        if code.strip() == '':
            n_full += 1
            continue
        if comment.strip() == '':
            out.append(ln)
            continue
        n_trail += 1
        out.append(code.rstrip() if code[-1] in ' \t' else code + '%')
    rep.cleaned.setdefault(rel, {})['comments'] = n_full + n_trail
    return '\n'.join(out)


def drop_definitions(text, rel, rep):
    """Remove the drafting-aid definitions and switches from a preamble."""
    n = 0
    # \newif\ifNAME, \NAMEtrue, \NAMEfalse
    for name in DROP_SWITCHES:
        text, k = re.subn(r'^[ \t]*\\newif\\if' + name + r'\b[^\n]*\n?', '', text, flags=re.M)
        n += k
        text, k = re.subn(r'^[ \t]*\\' + name + r'(?:true|false)\b[^\n]*\n?', '', text, flags=re.M)
        n += k
    # \newcommand{\NAME}[..][..]{body}  /  \newenvironment{NAME}[..]{begin}{end}
    for name in DROP_DEFS:
        pat = re.compile(r'\\(?:new|renew|provide)command\*?\s*\{\\' + name + r'\}')
        while (m := pat.search(text)):
            i = skip_ws(text, m.end())
            i = skip_ws(text, skip_opt(text, i))
            i = skip_ws(text, skip_opt(text, i))
            j = scan_group(text, i)
            text = text[:m.start()] + text[j:]
            n += 1
    for name in UNWRAP_ENVS:
        pat = re.compile(r'\\(?:new|renew)environment\*?\s*\{' + name + r'\}')
        while (m := pat.search(text)):
            i = skip_ws(text, m.end())
            i = skip_ws(text, skip_opt(text, i))
            j = scan_group(text, i)
            j = scan_group(text, skip_ws(text, j))
            text = text[:m.start()] + text[j:]
            n += 1
    rep.cleaned.setdefault(rel, {})['definitions dropped'] = n
    return text


def find_if_blocks(text, name):
    r"""-> [(start, end, true_part, else_part)] for every top-level \ifNAME ... [\else ...] \fi."""
    blocks, pos = [], 0
    while True:
        m = next((mm for mm in RE_IF.finditer(text, pos) if mm.group(1) == 'if' + name), None)
        if m is None:
            return blocks
        depth, else_at, fi_at = 0, None, None
        for t in RE_IF.finditer(text, m.end()):
            k = t.group(1)
            if k.startswith('if'):
                depth += 1
            elif k == 'else':
                if depth == 0:
                    else_at = t
            elif k == 'fi':
                if depth == 0:
                    fi_at = t
                    break
                depth -= 1
        if fi_at is None:
            raise ValueError(f'\\if{name} without \\fi')
        if else_at:
            tp, ep = text[m.end():else_at.start()], text[else_at.end():fi_at.start()]
        else:
            tp, ep = text[m.end():fi_at.start()], ''
        blocks.append((m.start(), fi_at.end(), tp, ep))
        pos = fi_at.end()


def resolve_conditionals(text, states, rel, rep):
    r"""Replace every \ifNAME ... \else ... \fi (NAME in states) by the chosen branch, nested-aware."""
    n = 0
    for name, state in states.items():
        while True:
            blocks = find_if_blocks(text, name)
            if not blocks:
                break
            s, e, tp, ep = blocks[0]
            chosen = dedent(tp if state else ep).strip('\n')
            text = text[:s] + chosen + text[e:]
            n += 1
    rep.cleaned.setdefault(rel, {})['switches resolved'] = n
    return text


def process_macros(text, rel, rep):
    """Remove / unwrap the drafting macros; lines that consisted only of a removed macro vanish."""
    names = list(REMOVE_MACROS) + list(UNWRAP_MACROS)
    pat = re.compile(r'\\(' + '|'.join(names) + r')(?![A-Za-z])')
    counts = {}
    out, pos = [], 0
    while (m := pat.search(text, pos)):
        name = m.group(1)
        i = skip_ws(text, m.end())
        i = skip_opt(text, i)
        i = skip_ws(text, i)
        if i >= len(text) or text[i] != '{':
            out.append(text[pos:m.end()])
            pos = m.end()
            continue
        j = scan_group(text, i)
        arg = text[i + 1:j - 1]
        line = text.count('\n', 0, m.start()) + 1
        counts[name] = counts.get(name, 0) + 1
        out.append(text[pos:m.start()])
        if name in REMOVE_MACROS:
            if REMOVE_MACROS[name]:
                rep.hole(name, rel, line, arg)
            out.append('\x00')
        else:
            if UNWRAP_MACROS[name]:
                rep.hole(name, rel, line, arg)
            out.append(arg)
        pos = j
    out.append(text[pos:])
    text = ''.join(out)
    # environments that only wrap text
    for env, nargs in UNWRAP_ENVS.items():
        pat_b = re.compile(r'\\begin\{' + env + r'\}')
        while (m := pat_b.search(text)):
            i = m.end()
            banner = ''
            for _ in range(nargs):
                i = skip_ws(text, i)
                j = scan_group(text, i)
                banner = text[i + 1:j - 1]
                i = j
            rep.hole(env, rel, text.count('\n', 0, m.start()) + 1, banner or '(unwrapped)')
            counts[env] = counts.get(env, 0) + 1
            text = text[:m.start()] + '\x00' + text[i:]
        text = re.sub(r'\\end\{' + env + r'\}', '\x00', text)
    lines = []
    for ln in text.split('\n'):
        if '\x00' in ln:
            bare = ln.replace('\x00', '')
            if bare.strip() == '':
                continue
            ln = bare
        lines.append(ln)
    for k, v in counts.items():
        rep.cleaned.setdefault(rel, {})[k] = v
    return '\n'.join(lines)


def clean_tex(text, rel, rep, states, preamble=False):
    text = strip_comments(text, rel, rep)
    if preamble:
        text = drop_definitions(text, rel, rep)
    text = resolve_conditionals(text, states, rel, rep)
    text = process_macros(text, rel, rep)
    return tidy(text)


def post_check(text, rel, rep):
    """Nothing dev-ish may survive in a released file."""
    bad = []
    for name in DROP_DEFS + ['fmpccgraphic', 'fmpccinherited']:
        if re.search(r'\\' + name + r'(?![A-Za-z])', text):
            bad.append(f'\\{name}')
    for name in DROP_SWITCHES:
        if re.search(r'\\if' + name + r'\b|\\' + name + r'(?:true|false)\b', text):
            bad.append(f'\\if{name}')
    for i, ln in enumerate(text.split('\n'), 1):
        k, n = 0, len(ln)
        while k < n:
            if ln[k] == '\\':
                k += 2
                continue
            if ln[k] == '%':
                if ln[k + 1:].strip():
                    bad.append(f'comment text left on line {i}')
                break
            k += 1
        if any(ord(c) < 32 and c not in '\t' for c in ln):
            bad.append(f'control character on line {i}')
    for b in bad:
        rep.bugs.append(f'{rel}: {b} survived the cleaning -- the release tool needs a fix')
    scan = re.sub(r'\\(?:label|ref|autoref|eqref|cref|pageref)\*?\{[^}]*\}', '', text)
    for i, ln in enumerate(scan.split('\n'), 1):
        for p in RESIDUE:
            if re.search(p, ln):
                rep.residue.append((rel, i, p, ln.strip()[:110]))
                break


# =============================================================================================
#  sources
# =============================================================================================
def draft_version(draft):
    """Highest '## vN.M[a]' heading of <draft>/CHANGELOG.md -> ('v3.99', 'v3.99 -- date . title')."""
    best = None
    for ln in read(os.path.join(draft, 'CHANGELOG.md')).split('\n'):
        m = RE_VERSION.match(ln)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), m.group(3))
        if best is None or key > best[0]:
            head = ln[3:].strip()
            head = re.sub(r'\s*→.*$', '', head)
            best = (key, head)
    if best is None:
        return 'v?', 'unknown'
    short = f'v{best[0][0]}.{best[0][1]}{best[0][2]}'
    return short, best[1]


def split_v2(text):
    """Content-based split of the v2 monolith (the markers of v3/tools/split_v2.py)."""
    lines = text.split('\n')

    def find(pred, what):
        for i, ln in enumerate(lines):
            if pred(ln):
                return i
        sys.exit(f'make_release: v2 marker not found: {what}')
    i_begin = find(lambda l: l.startswith(r'\begin{document}'), r'\begin{document}')
    i_main = find(lambda l: l.startswith(r'\mainmatter{}'), r'\mainmatter{}')
    i_app = find(lambda l: l.startswith(r'\appendix{}'), r'\appendix{}')
    i_abbr = find(lambda l: l.startswith(r'\addchap{Abbreviations}'), r'\addchap{Abbreviations}')
    i_end = find(lambda l: l.startswith(r'\end{document}'), r'\end{document}')
    i_back = i_abbr
    for j in range(i_abbr - 1, max(i_abbr - 6, i_main), -1):
        if lines[j].startswith(r'\microtypesetup{protrusion=false}'):
            i_back = j
            break
    starts = [i for i in range(i_main, i_app) if lines[i].startswith(r'\chapter{')]
    chapters, dropped = {}, []
    bounds = starts + [i_app]
    for k, a in enumerate(starts):
        m = RE_LABEL.search(lines[a])
        label = m.group(1) if m else f'(no label, line {a + 1})'
        body = '\n'.join(lines[a:bounds[k + 1]])
        if label in V2_CHAPTERS:
            chapters[label] = body
        else:
            dropped.append((label, bounds[k + 1] - a))
    dropped.append(('appendix stubs', i_back - i_app))
    return {
        'preamble': '\n'.join(lines[:i_begin]),
        'frontmatter': '\n'.join(lines[i_begin + 1:starts[0]]),
        'chapters': chapters,
        'backmatter': '\n'.join(lines[i_back:i_end]),
        'dropped': dropped,
    }


# =============================================================================================
#  bibliography
# =============================================================================================
def bib_entries(text):
    """-> [(key, entry_text_without_file_field)] in file order; comments outside entries dropped."""
    entries, pos = [], 0
    for m in RE_BIBENTRY.finditer(text):
        if m.start() < pos:
            continue
        i = text.index('{', m.start())
        j = scan_group(text, i)
        entry = text[m.start():j]
        entry = re.sub(r'^[ \t]*file\s*=\s*\{[^}]*\}\s*,?[ \t]*\n', '', entry, flags=re.M)
        entry = re.sub(r'^[ \t]*%[^\n]*\n', '', entry, flags=re.M)
        entries.append((m.group(2), entry.strip()))
        pos = j
    return entries


def aggregate_bib(sources, rep):
    seen, out = {}, []
    for rel, path in sources:
        if not os.path.isfile(path):
            continue
        for key, entry in bib_entries(read(path)):
            if key in seen:
                rep.bugs.append(f'bibliography key `{key}` is defined in {seen[key]} and {rel}; the first one is kept')
                continue
            seen[key] = rel
            out.append(entry)
    return '\n\n'.join(out) + '\n', seen


# =============================================================================================
#  assembly
# =============================================================================================
def reorder_template_preamble(text, rep):
    r"""nag before \documentclass; the ams packages before settings.tex (hyperref inside pdfx) so that
    equation anchors and \eqref links are patched by hyperref."""
    lines = text.split('\n')

    def take(pred, what):
        for i, ln in enumerate(lines):
            if pred(ln.strip()):
                return lines.pop(i)
        rep.info.append(f'preamble: {what} not found in v2; layout left as is')
        return None
    nag = take(lambda l: l.startswith(r'\RequirePackage') and 'nag' in l, r'\RequirePackage{nag}')
    ams = take(lambda l: l.startswith(r'\usepackage{amsmath'), r'\usepackage{amsmath,...}')
    if ams:
        for i, ln in enumerate(lines):
            if ln.strip().startswith(r'\input{settings}'):
                lines.insert(i, ams)
                break
        else:
            lines.append(ams)
            rep.info.append(r'preamble: \input{settings} not found; ams packages appended')
    if nag:
        lines.insert(0, nag)
    return '\n'.join(lines)


def fix_doctype(text, rep):
    m = re.search(r'^\\newcommand\*\{\\getDoctype\}\{([^}]*)\}', text, re.M)
    if m and r'\getDegree' in m.group(1):
        new = re.sub(r'\s+in\s+\\getDegree\s*$', '', m.group(1))
        text = text[:m.start(1)] + new + text[m.end(1):]
        rep.bugs.append(f'v2 metadata: \\getDoctype was `{m.group(1)}`; the TUM cover and title pages print '
                        f'`\\getDoctype in \\getDegree` themselves, so the degree would appear twice. '
                        f'Set to `{new}` in the release (v2 should change it for the template build).')
    return text


def fit_title_page(text, rep):
    r"""The template's title page fits its dummy titles only. With the real English title (two \huge lines)
    and the German title (three \huge lines) it overflows, and the faculty logo lands alone on the next
    page (seen in the first compiled release). The German title goes to \LARGE and the vertical gaps
    shrink; the fonts of everything else and the order of the page stay the template's."""
    subs = [(r'\vspace{10mm}', r'\vspace{6mm}', 1),
            (r'\vspace{20mm}', r'\vspace{10mm}', 1),
            (r'\vspace{15mm}', r'\vspace{8mm}', 2),
            (r'{\huge\bfseries \foreignlanguage{ngerman}{\getTitleGer{}} \par}',
             r'{\LARGE\bfseries \foreignlanguage{ngerman}{\getTitleGer{}} \par}', 1)]
    for old, new, n in subs:
        if text.count(old) != n:
            rep.info.append(f'title page: expected {n}x `{old}` in the template copy, found {text.count(old)}; '
                            f'the fit adjustment for it was skipped -- check that the title page still fits')
            continue
        text = text.replace(old, new)
    rep.info.append('title page: German title set in \\LARGE and the vertical gaps reduced (20/15/15/10 mm -> '
                    '10/8/8/6 mm) so that the long titles fit on one page; the first compiled release had the '
                    'faculty logo alone on page ii')
    return text


def metadata_holes(text, rep):
    for m in re.finditer(r'^\\newcommand\*\{\\(get[A-Za-z]+)\}\{([^}]*)\}', text, re.M):
        if 'TODO' in m.group(2):
            rep.hole('metadata', 'main.tex', text.count('\n', 0, m.start()) + 1,
                     f'\\{m.group(1)} = `{m.group(2)}` -- printed on the cover/title/disclaimer pages')


def acronym_merge(back_v2, extra_files, used, rep):
    declared = set(RE_ACRO.findall(back_v2))
    missing = sorted(k for k in used if k not in declared)
    if not missing:
        return back_v2
    extra = {}
    for rel, path in extra_files:
        if os.path.isfile(path):
            for m in re.finditer(r'^[ \t]*(\\acro\{([^}]+)\}[^\n]*)$', read(path), re.M):
                extra.setdefault(m.group(2), (m.group(1).strip(), rel))
    add = []
    for k in missing:
        if k in extra:
            add.append('  ' + extra[k][0])
            rep.info.append(f'acronym {k} is used but not declared by v2; declaration taken from {extra[k][1]}')
        else:
            rep.bugs.append(f'acronym `{k}` is used (\\ac{{{k}}}) but declared nowhere -- prints as ?? and biber/acronym warns')
    if add:
        back_v2 = back_v2.replace(r'\end{acronym}', '\n'.join(add) + '\n' + r'\end{acronym}', 1)
    return back_v2


def figure_sources():
    """name -> {ext: path}, first hit wins per (name, ext): v4, v3, v2, then the DA store."""
    dirs = [os.path.join(V4, 'figures'), os.path.join(V3, 'figures'), os.path.join(V2, 'figures')]
    for root, _d, _f in os.walk(DA_FIGS):
        dirs.append(root)
    found = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext in ('.pdf', '.png', '.jpg', '.jpeg', '.svg'):
                found.setdefault(stem, {}).setdefault(ext, os.path.join(d, fn))
    return found


# =============================================================================================
#  checks, outline, page estimate
# =============================================================================================
def collect_tree(root, master):
    r"""[(rel, text)] for main.tex and every \input reachable from it (release copies, already clean)."""
    docs, missing, seen = [], [], set()

    def visit(rel):
        if rel in seen:
            return
        seen.add(rel)
        path = os.path.join(root, rel)
        text = read(path)
        docs.append((rel, text))
        for target in RE_INPUT.findall(text):
            cand = target if target.endswith('.tex') else target + '.tex'
            if os.path.isfile(os.path.join(root, cand)):
                visit(cand)
            elif os.path.isfile(os.path.join(root, target)):
                visit(target)
            else:
                missing.append(f'{rel}: \\input{{{target}}}')
    visit(master)
    return docs, missing


def mechanical_checks(root, docs, missing_inputs, bibkeys, rep):
    text_all = '\n'.join(t for _r, t in docs)
    problems = []
    if missing_inputs:
        problems.append(('missing \\input target', missing_inputs))
    labels, dup = {}, []
    for rel, t in docs:
        for lab in RE_LABEL.findall(t):
            if lab in labels:
                dup.append(f'{lab} (in {labels[lab]} and {rel})')
            labels[lab] = rel
    refs = {r.strip() for t in (x for _r, x in docs) for grp in RE_REF.findall(t) for r in grp.split(',')}
    dangling = sorted(refs - set(labels))
    if dup:
        problems.append(('duplicate \\label', dup))
    if dangling:
        problems.append(('reference with no \\label (prints ??)', dangling))
    cited = {k.strip() for grp in RE_CITE.findall(text_all) for k in grp.split(',') if k.strip()}
    unknown = sorted(cited - set(bibkeys))
    if unknown:
        problems.append(('citation with no bibliography entry', unknown))
    for rel, t in docs:
        op, cl = RE_BEGIN.findall(t), RE_END.findall(t)
        for env in set(op) | set(cl):
            if op.count(env) != cl.count(env):
                problems.append((f'unbalanced environment in {rel}', [f'{env}: {op.count(env)} begin, {cl.count(env)} end']))
        bare = re.sub(r'(?<!\\)\\[{}]', '', t)
        if (d := bare.count('{') - bare.count('}')):
            problems.append((f'brace delta in {rel}', [f'{d:+d}']))
    for what, n in ((r'\begin{document}', text_all.count(r'\begin{document}')),
                    (r'\end{document}', text_all.count(r'\end{document}')),
                    (r'\documentclass', text_all.count(r'\documentclass')),
                    (r'\appendix', len(re.findall(r'\\appendix\b', text_all)) - text_all.count(r'\renewcommand{\appendix}')
                     - text_all.count(r'\fmpccorigappendix\appendix'))):
        if n != 1:
            problems.append((f'{what} appears {n} times, expected 1', []))
    wanted = {name for _o, name in RE_GRAPHIC.findall(text_all) if '/' not in name and '\\' not in name}
    figdir = os.path.join(root, 'figures')
    absent = sorted(n for n in wanted if not any(os.path.isfile(os.path.join(figdir, n + e))
                                                for e in ('', '.pdf', '.png', '.jpg', '.jpeg')))
    if absent:
        problems.append(('\\includegraphics target not in figures/', absent))
    used_ac = set(RE_AC.findall(text_all))
    declared = set(RE_ACRO.findall(text_all))
    if (u := sorted(used_ac - declared)):
        problems.append(('acronym used but not declared', u))
    for what, items in problems:
        rep.bugs.append(what + (': ' + ', '.join(items) if items else ''))
    return {'labels': len(labels), 'refs': len(refs), 'cited': sorted(cited), 'figures': sorted(wanted),
            'acronyms_used': sorted(used_ac), 'acronyms_declared': sorted(declared)}


def expand_inputs(text, files, depth=0):
    r"""Replace every \input line by the (cleaned) file it names, in place."""
    def repl(m):
        target = m.group(1)
        cand = target if target.endswith('.tex') else target + '.tex'
        if cand in files and depth < 8:
            return expand_inputs(files[cand], files, depth + 1)
        return m.group(0)
    return re.sub(r'^[^%\n]*?\\input\{([^}]+)\}[^\n]*$', repl, text, flags=re.M)


def outline_and_estimate(files, order):
    """Walk the chapter files in reading order (nested inputs expanded in place) ->
    outline rows, LoF/LoT rows, per-chapter estimate."""
    full = '\n'.join(expand_inputs(files[rel], files) for rel in order)
    segments = re.split(r'(?m)(?=^\\chapter(?:\[|\{)|^\\appendix\b)', full)
    rows, lof, lot, est = [], [], [], []
    ch_num, in_app = 0, False
    sec = sub = subsub = 0
    fig_n = tab_n = 0
    heading = re.compile(r'\\(chapter|section|subsection|subsubsection|addchap)(\*?)(?:\[[^\]]*\])?\{')
    caption = re.compile(r'\\caption(?:\[([^\]]*)\])?\{')
    cur = None

    def chapter_tag():
        return chr(ord('A') + ch_num - 1) if in_app else str(ch_num)

    for text in segments:
        if re.search(r'^\\appendix\b', text, re.M) and not in_app:
            in_app, ch_num = True, 0
        events = [(m.start(), 'h', m) for m in heading.finditer(text)]
        events += [(m.start(), 'f', m) for m in re.finditer(r'\\begin\{(figure|table)\}', text)]
        events.sort(key=lambda e: e[0])
        for _pos, kind, m in events:
            if kind == 'h':
                title = display_title(text[m.end():scan_group(text, m.end() - 1) - 1])
                lvl, star = m.group(1), m.group(2) == '*'
                if lvl == 'addchap' or (lvl == 'chapter' and star):
                    rows.append(('', title, 0))
                    continue
                if lvl == 'chapter':
                    ch_num += 1
                    sec = sub = subsub = 0
                    fig_n = tab_n = 0
                    rows.append((chapter_tag(), title, 0))
                    cur = {'chapter': f'{chapter_tag()} {title}', 'words': 0, 'eqlines': 0, 'fig': 0.0,
                           'nfig': 0, 'tab': 0.0, 'ntab': 0}
                    est.append(cur)
                elif lvl == 'section':
                    if star:
                        rows.append(('', title, 1))
                        continue
                    sec += 1
                    sub = subsub = 0
                    rows.append((f'{chapter_tag()}.{sec}', title, 1))
                elif lvl == 'subsection':
                    if star:
                        rows.append(('', title, 2))
                        continue
                    sub += 1
                    subsub = 0
                    rows.append((f'{chapter_tag()}.{sec}.{sub}', title, 2))
                else:
                    if star:
                        continue
                    subsub += 1
                    rows.append((f'{chapter_tag()}.{sec}.{sub}.{subsub}', title, 3))
            else:
                env = m.group(1)
                end = text.find(f'\\end{{{env}}}', m.end())
                body = text[m.end():end if end >= 0 else len(text)]
                cm = caption.search(body)
                cap = ''
                if cm:
                    long = body[cm.end():scan_group(body, cm.end() - 1) - 1]
                    cap = display_title(cm.group(1) or long)
                if env == 'figure':
                    fig_n += 1
                    lof.append((f'{chapter_tag()}.{fig_n}', cap))
                    share = 0.0
                    for opts, _name in RE_GRAPHIC.findall(body):
                        w = re.search(r'width\s*=\s*([\d.]+)\s*\\linewidth', opts or '')
                        h = re.search(r'height\s*=\s*([\d.]+)\s*cm', opts or '')
                        share += 0.30 * float(w.group(1)) if w else (float(h.group(1)) / 24.0 if h else 0.30)
                    if cur:
                        cur['fig'] += max(0.12, min(share, 0.85)) + 0.06
                        cur['nfig'] += 1
                else:
                    tab_n += 1
                    lot.append((f'{chapter_tag()}.{tab_n}', cap))
                    rows_n = body.count('\\\\')
                    if cur:
                        cur['tab'] += min(1.0, (rows_n * 1.2 + 5) / 41.0) + 0.04
                        cur['ntab'] += 1
        if cur is not None:
            t = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', ' ', text, flags=re.S)
            eq_lines = 0
            for m in re.finditer(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.*?)\\end\{\1\}', t, re.S):
                eq_lines += 1 + m.group(2).count('\\\\')
            t = re.sub(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{\1\}', ' ', t, flags=re.S)
            t = re.sub(r'\\\[.*?\\\]', ' ', t, flags=re.S)
            cur['words'] += len(plain_words(t))
            cur['eqlines'] += eq_lines
    for c in est:
        c['pages'] = c['words'] / WORDS_PER_PAGE + c['eqlines'] * 2.2 / 41.0 + c['fig'] + c['tab'] + 0.5
    return rows, lof, lot, est


# =============================================================================================
#  the build
# =============================================================================================
def build(args):
    rep = Report()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    states = {'standalone': bool(args.standalone), 'appendixfull': not args.appendix_short}
    v2s, v2full = draft_version(V2)
    v3s, v3full = draft_version(V3)
    v4s, v4full = draft_version(V4)
    name = f'{stamp}_thesis_release_{v2s}_{v3s}_{v4s}'
    if args.tag:
        name += '_' + re.sub(r'[^A-Za-z0-9_-]+', '_', args.tag)
    if args.standalone:
        name += '_standalone'
    if args.appendix_short:
        name += '_appendixshort'
    outdir = os.path.abspath(args.outdir or OUT_DEFAULT)
    build_dir = os.path.join(outdir, name)             # output/<build>/  : latex/ + zip + notes together
    root = os.path.join(build_dir, LATEX_SUBDIR)       # the LaTeX project
    files = {}        # release rel path -> text
    binaries = {}     # release rel path -> source path
    sources = []      # (draft, rel, lines, sha)

    def src(draft, path):
        sources.append((draft, os.path.relpath(path, WS), read(path).count('\n'), sha12(path)))

    # ---- v2: preamble, front matter, Ch 1-4, back matter, bibliography ------------------------
    v2_path = os.path.join(V2, 'thesis_v2.tex')
    src('v2', v2_path)
    parts = split_v2(read(v2_path))
    for label, n in parts['dropped']:
        rep.info.append(f'v2: placeholder `{label}` ({n} lines) not used -- the live text is v3\'s / v4\'s')
    for label, rel in V2_CHAPTERS.items():
        if label not in parts['chapters']:
            sys.exit(f'make_release: v2 has no chapter labelled {label}')
        files[rel] = clean_tex(parts['chapters'][label], f'v2:{rel}', rep, states)

    pre_v2 = clean_tex(parts['preamble'], 'v2:preamble', rep, states, preamble=True)
    if not args.standalone:
        pre_v2 = reorder_template_preamble(pre_v2, rep)
        pre_v2 = fix_doctype(pre_v2, rep)
    metadata_holes(pre_v2, rep)
    if (m := re.search(r'\\getTitleGer\}\{([^}]*)\}[^\n]*%[^\n]*TODO', parts['preamble'])):
        rep.hole('metadata', 'main.tex', 0, f'\\getTitleGer = `{m.group(1)}` is marked "TODO: confirm" in v2 -- '
                 'printed on the title page')

    front_raw = strip_comments(parts['frontmatter'], 'v2:frontmatter', rep)
    abstract = None
    for _s, _e, tp, _ep in find_if_blocks(front_raw, 'standalone'):
        if r'\begin{titlepage}' not in tp and len(plain_words(tp)) > 40:
            abstract = tidy(dedent(tp))
    if abstract is None:
        sys.exit('make_release: the abstract block (\\ifstandalone ... \\fi with prose) was not found in v2\'s front matter')
    front = clean_tex(front_raw, 'v2:frontmatter', rep, states)
    front = re.sub(r'^\\pagenumbering\{alph\}\n?', '', front, flags=re.M)
    ack_text = ''
    ack_src = os.path.join(FRONT_DIR, 'acknowledgments.tex')
    if os.path.isfile(ack_src):
        ack_text = tidy(strip_comments(read(ack_src), 'front/acknowledgments.tex', rep)).strip()
    want_ack = bool(ack_text) or args.acknowledgments
    if not args.standalone:
        if not want_ack:
            front = re.sub(r'^\\input\{pages/acknowledgments\}\n?', '', front, flags=re.M)
            rep.info.append('Acknowledgments page dropped: it is optional and there is no text (the template only '
                            'has a TODO there; its own compiled PDF shows it blank). Put the text in '
                            'RELEASE/front/acknowledgments.tex, or pass --acknowledgments for the empty page')
        if args.no_cover:
            front = re.sub(r'^\\input\{pages/cover\}\n?', '', front, flags=re.M)
            rep.info.append('cover page dropped (--no-cover): the title page is the first page')

    back = clean_tex(parts['backmatter'], 'v2:backmatter', rep, states)

    # ---- v3: Ch 5-6 + preamble ---------------------------------------------------------------
    for rel in V3_CHAPTERS:
        p = os.path.join(V3, rel)
        src('v3', p)
        files[rel] = clean_tex(read(p), f'v3:{rel}', rep, states)
    p = os.path.join(V3, 'parts', '00_preamble_v3.tex')
    src('v3', p)
    pre_v3 = clean_tex(read(p), 'v3:preamble', rep, states, preamble=True)

    # ---- v4: Ch 7-9 (+ nested inputs) + preamble ------------------------------------------------
    pending = list(V4_CHAPTERS)
    while pending:
        rel = pending.pop(0)
        p = os.path.join(V4, rel)
        if not os.path.isfile(p):
            sys.exit(f'make_release: v4 file missing: {rel}')
        src('v4', p)
        text = clean_tex(read(p), f'v4:{rel}', rep, states)
        files[rel] = text
        for target in RE_INPUT.findall(text):
            cand = target if target.endswith('.tex') else target + '.tex'
            if cand not in files and cand not in pending:
                pending.append(cand)
    p = os.path.join(V4, 'parts', '00_preamble_v4.tex')
    src('v4', p)
    pre_v4 = clean_tex(read(p), 'v4:preamble', rep, states, preamble=True)

    # ---- bibliography ---------------------------------------------------------------------------
    bib_src = [('v2/bibliography.bib', os.path.join(V2, 'bibliography.bib')),
               ('v3/bibliography_v3.bib', os.path.join(V3, 'bibliography_v3.bib')),
               ('v4/bibliography_v4.bib', os.path.join(V4, 'bibliography_v4.bib'))]
    for rel, p in bib_src:
        if os.path.isfile(p):
            src(rel.split('/')[0], p)
    bib_text, bibkeys = aggregate_bib(bib_src, rep)
    files['bibliography.bib'] = bib_text
    for pre in ('pre_v2', 'pre_v3', 'pre_v4'):
        t = locals()[pre]
        t2 = re.sub(r'^[ \t]*\\addbibresource\{(?!bibliography\.bib\})[^}]*\}[^\n]*\n?', '', t, flags=re.M)
        if t2 != t:
            rep.info.append(f'{pre[4:]}: \\addbibresource of a per-draft .bib removed (one aggregated bibliography.bib)')
        if pre == 'pre_v2':
            pre_v2 = t2
        elif pre == 'pre_v3':
            pre_v3 = t2
        else:
            pre_v4 = t2

    # ---- acronyms: v2's list, plus what v3/v4 declared and the text uses ------------------------
    body_all = '\n'.join(files[r] for r in files if r.endswith('.tex')) + front + abstract
    used_ac = set(RE_AC.findall(body_all))
    back = acronym_merge(back, [('v4/parts/99_backmatter.tex', os.path.join(V4, 'parts', '99_backmatter.tex')),
                                ('v3/parts/99_backmatter.tex', os.path.join(V3, 'parts', '99_backmatter.tex'))],
                         used_ac, rep)

    # ---- main.tex ---------------------------------------------------------------------------------
    main = [pre_v2.rstrip('\n'), '']
    if pre_v3.strip():
        main += [pre_v3.rstrip('\n'), '']
    if pre_v4.strip():
        main += [pre_v4.rstrip('\n'), '']
    main += [r'\begin{document}', '', r'\pagenumbering{alph}', front.rstrip('\n'), '']
    for rel in list(V2_CHAPTERS.values()) + V3_CHAPTERS + V4_CHAPTERS[:2]:
        main.append('\\input{%s}' % rel[:-4])
    main.append('')
    if not re.search(r'^\\appendix\b', files[V4_CHAPTERS[2]], re.M):
        main.append(r'\appendix{}')
        rep.info.append('\\appendix{} added by the release tool (v4\'s appendix file does not carry it)')
    main.append('\\input{%s}' % V4_CHAPTERS[2][:-4])
    main += ['', back.rstrip('\n'), '', r'\end{document}']
    files['main.tex'] = tidy('\n'.join(main))

    # ---- template files -----------------------------------------------------------------------------
    if not args.standalone:
        for fn in ('settings.tex', 'main.xmpdata'):
            files[fn] = tidy(strip_comments(read(os.path.join(TEMPLATE, fn)), f'template:{fn}', rep))
        for page in TEMPLATE_PAGES:
            if page == 'acknowledgments' and not want_ack:
                continue
            if page == 'cover' and args.no_cover:
                continue
            text = tidy(strip_comments(read(os.path.join(TEMPLATE, 'pages', page + '.tex')),
                                       f'template:pages/{page}.tex', rep))
            if page == 'title':
                text = fit_title_page(text, rep)
            if page == 'acknowledgments' and ack_text:
                text = text.replace('\\vspace{10mm}\n', '\\vspace{10mm}\n\n' + ack_text + '\n', 1)
                rep.info.append('Acknowledgments page included with the text of RELEASE/front/acknowledgments.tex')
            files[f'pages/{page}.tex'] = text
        files['pages/abstract.tex'] = '\\chapter{\\abstractname}\n\n' + abstract
        if want_ack and not ack_text:
            rep.hole('template', 'pages/acknowledgments.tex', 1,
                     'Acknowledgments page is EMPTY (--acknowledgments without RELEASE/front/acknowledgments.tex)')
        rep.info.append('template: pages/software_used.tex (the AI-tools declaration page of the template) is not '
                        'included -- the author decides whether the submission needs it')
        for fn in TEMPLATE_VERBATIM:
            binaries[fn] = os.path.join(TEMPLATE, fn)
        for fn in sorted(os.listdir(os.path.join(TEMPLATE, 'logos'))):
            binaries[f'logos/{fn}'] = os.path.join(TEMPLATE, 'logos', fn)
    else:
        files['pages/abstract.tex'] = None   # not used
        del files['pages/abstract.tex']
        for fn in TEMPLATE_VERBATIM:
            binaries[fn] = os.path.join(TEMPLATE, fn)

    # ---- figures -------------------------------------------------------------------------------------
    wanted = sorted({name for _o, name in RE_GRAPHIC.findall(body_all)})
    store = figure_sources()
    raster, missing_fig = [], []
    for fig in wanted:
        have = store.get(fig, {})
        pick = next((have[e] for e in ('.pdf', '.png', '.jpg', '.jpeg') if e in have), None)
        if pick is None:
            missing_fig.append(fig + (' (SVG only -- pdflatex cannot read it)' if '.svg' in have else ''))
            continue
        ext = os.path.splitext(pick)[1].lower()
        binaries[f'figures/{fig}{ext}'] = pick
        if ext != '.pdf':
            raster.append(f'{fig}{ext}' + (' (an SVG exists)' if '.svg' in have else ''))
    if missing_fig:
        rep.bugs.append('figure file missing for: ' + ', '.join(missing_fig))
    if raster:
        rep.hole('figures', 'figures/', 0,
                 f'{len(raster)} of {len(wanted)} figures ship as raster (.png), no PDF exists anywhere in the drafts or '
                 f'the DA store; the institute asks for vector figures. Convert the SVGs (DA_in_Paper/plotting/svg/'
                 f'svg2pdf.sh on a machine with Inkscape/rsvg) and rebuild -- the tool prefers a .pdf automatically. '
                 f'Raster-only: ' + ', '.join(raster))

    # ---- post checks on the cleaned texts ----------------------------------------------------------------
    for rel, text in files.items():
        if rel.endswith('.tex') or rel.endswith('.xmpdata'):
            post_check(text, rel, rep)

    if args.dry_run:
        print(f'DRY RUN -- would write {len(files)} text file(s) and {len(binaries)} binary file(s) to {root}')
        for b in rep.bugs:
            print('  BUG ', b)
        return 0

    # ---- write ---------------------------------------------------------------------------------------
    if os.path.exists(build_dir):
        sys.exit(f'make_release: {build_dir} exists already')
    os.makedirs(root)
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
    for rel, p in binaries.items():
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(p, dst)

    # ---- checks on the written tree --------------------------------------------------------------------
    docs, missing_inputs = collect_tree(root, 'main.tex')
    stats = mechanical_checks(root, docs, missing_inputs, bibkeys, rep)
    order = list(V2_CHAPTERS.values()) + V3_CHAPTERS + V4_CHAPTERS
    rows, lof, lot, est = outline_and_estimate(files, order)
    body_pages = sum(c['pages'] for c in est)
    n_toc = sum(1 for r in rows if r[2] <= 2)
    front_pages = (2 if args.no_cover else 3) + (1 if want_ack else 0) + math.ceil((n_toc + 6) / 38)
    n_cited = len(stats['cited'])
    back_pages = 1 + math.ceil(len(lof) * 1.6 / 41) + math.ceil(len(lot) * 1.6 / 41) + n_cited * BIB_LINES_PER_ENTRY / 41
    total = body_pages + front_pages + back_pages
    lo, hi = 0.85 * total, 1.2 * total
    verdict = 'within the 60-200 limit'
    pagewarn = False
    if total > PAGE_MAX or total < PAGE_MIN:
        verdict = f'OUTSIDE the {PAGE_MIN}-{PAGE_MAX} limit'
        pagewarn = True
    elif hi > PAGE_MAX or lo < PAGE_MIN:
        verdict = f'inside the {PAGE_MIN}-{PAGE_MAX} limit, but the uncertainty band touches it'
    guide = ('above' if total > PAGE_GUIDE[1] else 'below' if total < PAGE_GUIDE[0] else 'within')
    uncited = sorted(set(bibkeys) - set(stats['cited']))

    if pagewarn:
        os.rename(build_dir, build_dir + '_PAGEWARN')
        build_dir, name = build_dir + '_PAGEWARN', name + '_PAGEWARN'
        root = os.path.join(build_dir, LATEX_SUBDIR)

    # ---- notes ---------------------------------------------------------------------------------------
    notes = os.path.join(build_dir, f'RELEASE_NOTES_{stamp}.md')
    with open(notes, 'w', encoding='utf-8') as f:
        f.write(render_notes(name, stamp, (v2s, v2full), (v3s, v3full), (v4s, v4full), sources, files, binaries,
                             rep, stats, rows, lof, lot, est, front_pages, back_pages, total, lo, hi, verdict,
                             guide, uncited, args, wanted, raster))
    if pagewarn:
        with open(os.path.join(build_dir, 'WARNING_PAGE_LIMIT.md'), 'w') as f:
            f.write(f'# PAGE LIMIT WARNING\n\nEstimated {total:.0f} pages ({lo:.0f}-{hi:.0f}); {verdict}. '
                    f'See RELEASE_NOTES_{stamp}.md.\n')

    zip_path = ''
    if not args.no_zip:
        zip_path = make_zip(build_dir)

    # ---- changelog ---------------------------------------------------------------------------------------
    entry = [f'## {stamp} -- {name}', '',
             f'- **Built on:** v2 **{v2s}** ({v2full}) · v3 **{v3s}** ({v3full}) · v4 **{v4s}** ({v4full})',
             f'- **Mode:** {"standalone (no TUM template)" if args.standalone else "TUM template"}'
             f'{"; appendix long-data tables hidden" if args.appendix_short else ""}'
             f'{"; tag " + args.tag if args.tag else ""}',
             f'- **Output:** `output/{name}/{LATEX_SUBDIR}/` (main.tex + {len(files) - 1} text files, {len(binaries)} binary files)'
             + (f', `output/{name}/{os.path.basename(zip_path)}` ({os.path.getsize(zip_path) // 1024} KB)' if zip_path else ''),
             f'- **Estimate:** ~{total:.0f} pages ({lo:.0f}-{hi:.0f}), {verdict}; {guide} the {PAGE_GUIDE[0]}-{PAGE_GUIDE[1]} '
             f'guideline. NOT compiled (no TeX toolchain here).',
             f'- **Holes recorded:** {len(rep.holes)} · **bugs/findings:** {len(rep.bugs)} · figures {len(wanted)} '
             f'({len(raster)} raster) · bibliography {len(bibkeys)} entries, {n_cited} cited · labels {stats["labels"]}',
             f'- **Notes:** `output/{name}/RELEASE_NOTES_{stamp}.md`']
    if args.note:
        entry += [f'- **Note:** {n}' for n in args.note]
    entry.append('')
    if not args.no_log:
        prepend_changelog('\n'.join(entry))

    # ---- console ------------------------------------------------------------------------------------------
    print(f'RELEASE  {name}')
    print(f'  built on v2 {v2s} · v3 {v3s} · v4 {v4s}   ({len(sources)} source files)')
    print(f'  {len(files)} text files, {len(binaries)} binary files -> {os.path.relpath(root, RELEASE)}/')
    if zip_path:
        print(f'  zip: {os.path.relpath(zip_path, RELEASE)} ({os.path.getsize(zip_path) // 1024} KB)')
    print(f'  page estimate: ~{total:.0f} ({lo:.0f}-{hi:.0f}) -- {verdict}; {guide} the {PAGE_GUIDE[0]}-{PAGE_GUIDE[1]} guideline')
    print(f'  holes: {len(rep.holes)}   bugs/findings: {len(rep.bugs)}   residue hits: {len(rep.residue)}')
    for b in rep.bugs:
        print('  BUG  ' + b[:160])
    print(f'  notes: {os.path.relpath(notes, RELEASE)}')
    return 0


def render_notes(name, stamp, v2, v3, v4, sources, files, binaries, rep, stats, rows, lof, lot, est,
                 front_pages, back_pages, total, lo, hi, verdict, guide, uncited, args, wanted, raster):
    L = []
    w = L.append
    w(f'# Release notes -- `{name}`')
    w('')
    w(f'**Built:** {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]} by '
      f'`RELEASE/tools/make_release.py` · **Mode:** {"standalone" if args.standalone else "TUM template"}'
      f'{" · appendix long-data tables hidden" if args.appendix_short else ""}')
    w(f'**Built on:** v2 **{v2[0]}** ({v2[1]}) · v3 **{v3[0]}** ({v3[1]}) · v4 **{v4[0]}** ({v4[1]})')
    w('')
    w('**NOT COMPILED.** There is no TeX toolchain in the container; every check below is mechanical. Build with '
      '`pdflatex main.tex ; biber main ; pdflatex main ; pdflatex main` (or `make pdf`, or upload the zip to Overleaf '
      'with pdfLaTeX + Biber) and attach the PDF with `--attach-pdf` to record the real page count.')
    w('')
    w('## 1. Page estimate')
    w('')
    w(f'**~{total:.0f} pages (band {lo:.0f}-{hi:.0f}) -- {verdict}.** The estimate is {guide} the '
      f'{PAGE_GUIDE[0]}-{PAGE_GUIDE[1]}-page orientation of the institute for a master\'s thesis '
      f'(Writing_Hints/tum_i6_thesis_submission_reference.md).')
    for n in (args.note or []):
        w('')
        w(f'**Note:** {n}')
    w('')
    w('| chapter | words | eq. lines | figures | tables | est. pages |')
    w('| :-- | --: | --: | --: | --: | --: |')
    for c in est:
        w(f'| {c["chapter"]} | {c["words"]} | {c["eqlines"]} | {c["nfig"]} | {c["ntab"]} | {c["pages"]:.1f} |')
    w(f'| front matter (cover, title, disclaimer, acknowledgments, abstract, contents) | | | | | {front_pages:.1f} |')
    w(f'| back matter (abbreviations, lists, bibliography) | | | | | {back_pages:.1f} |')
    w(f'| **total** | {sum(c["words"] for c in est)} | {sum(c["eqlines"] for c in est)} | {sum(c["nfig"] for c in est)} '
      f'| {sum(c["ntab"] for c in est)} | **{total:.1f}** |')
    w('')
    w(f'Model: {WORDS_PER_PAGE} prose words per page, 2.2 lines per displayed equation line, a figure by its width '
      '(0.30 page per full width + caption), a table by its rows (1.2 lines each + 5), half a page lost per chapter start, '
      f'the front pages + contents, {BIB_LINES_PER_ENTRY} lines per bibliography entry; calibrated on the compiled first '
      'release (185 pages, 2026-09-25). Treat it as +-15 %.')
    w('')
    w('## 2. HOLES -- what the submitted PDF would lack or show')
    w('')
    if not rep.holes:
        w('none recorded')
    for i, (kind, rel, line, text) in enumerate(rep.holes, 1):
        loc = f'`{rel}`' + (f':{line}' if line else '')
        w(f'{i}. **[{kind}]** {loc} -- {text}')
    w('')
    w('## 3. Bugs and findings')
    w('')
    if not rep.bugs:
        w('none -- all mechanical checks pass')
    for b in rep.bugs:
        w(f'- {b}')
    w('')
    if rep.residue:
        w(f'### 3a. Possible dev residue in the prose ({len(rep.residue)} lines; content owners decide)')
        w('')
        w('Lines that mention a draft version, a file name, a label name in `\\texttt`, "author", "TODO" and the like. '
          'They are printed as they stand; nothing was changed.')
        w('')
        for rel, line, pat, ex in rep.residue[:80]:
            w(f'- `{rel}:{line}` (`{pat}`): {ex}')
        if len(rep.residue) > 80:
            w(f'- ... {len(rep.residue) - 80} more')
        w('')
    if rep.info:
        w('### 3b. Decisions the tool made (information)')
        w('')
        for s in rep.info:
            w(f'- {s}')
        w('')
    if rep.comment_todos:
        w(f'### 3c. TODO / FIXME found in source COMMENTS ({len(rep.comment_todos)}; removed with the comments, never printed)')
        w('')
        for rel, t in rep.comment_todos[:40]:
            w(f'- `{rel}`: {t[:140]}')
        if len(rep.comment_todos) > 40:
            w(f'- ... {len(rep.comment_todos) - 40} more')
        w('')
    w('## 4. What was cleaned')
    w('')
    w('| source | comments | switches resolved | definitions dropped | macros removed / unwrapped |')
    w('| :-- | --: | --: | --: | :-- |')
    for rel, c in rep.cleaned.items():
        rest = ', '.join(f'{k} {v}' for k, v in c.items() if k not in ('comments', 'switches resolved', 'definitions dropped'))
        w(f'| `{rel}` | {c.get("comments", 0)} | {c.get("switches resolved", 0)} | {c.get("definitions dropped", 0)} | {rest} |')
    w('')
    w('Removed outright: `\\srcnote`, `\\dataref`, `\\hole`, `\\longdata`, `\\flawed`, `\\outdated`, `\\todofigure`. '
      'Unwrapped to plain text: `\\guard`, `\\provisional` (their text is thesis prose). Switches resolved: '
      f'`\\ifstandalone` -> {"standalone" if args.standalone else "template"} branch, `\\ifappendixfull` -> '
      f'{"tables hidden" if args.appendix_short else "tables printed"}. The drafting definitions and `\\newif` switches '
      'are gone from the preamble; the presentation macros (`\\mainconfig`, `\\selectedmark`, `\\baselinemark`) stay.')
    w('')
    w('## 5. Sources (live files of the owning draft; SHA-256 prefix as built)')
    w('')
    w('| draft | file | lines | sha256 |')
    w('| :-- | :-- | --: | :-- |')
    for d, rel, n, s in sources:
        w(f'| {d} | `{rel}` | {n} | `{s}` |')
    w('')
    w('## 6. Release contents')
    w('')
    for rel in sorted(files):
        w(f'- `{rel}` ({files[rel].count(chr(10))} lines)')
    for rel in sorted(binaries):
        w(f'- `{rel}` <- `{os.path.relpath(binaries[rel], REPO)}`')
    w('')
    w(f'Figures: {len(wanted)} referenced, {len(raster)} raster-only. Bibliography: {len(stats["cited"])} keys cited; '
      + (f'{len(uncited)} entries never cited (not printed by biblatex): ' + ', '.join(uncited) if uncited else 'every entry is cited')
      + f'. Acronyms used: {", ".join(stats["acronyms_used"])}.')
    w('')
    w('## 7. Structure preview (what the table of contents will show)')
    w('')
    for num, title, lvl in rows:
        w(f'{"  " * lvl}- {num + " " if num else ""}{title}')
    w('')
    w('### List of figures (preview)')
    w('')
    for num, cap in lof:
        w(f'- Figure {num}: {cap[:120]}')
    w('')
    w('### List of tables (preview)')
    w('')
    for num, cap in lot:
        w(f'- Table {num}: {cap[:120]}')
    w('')
    w('## 8. Build')
    w('')
    w('```')
    w('pdflatex main.tex && biber main && pdflatex main.tex && pdflatex main.tex     # or: make pdf')
    w('```')
    w('Overleaf: upload the zip, compiler pdfLaTeX, main document `main.tex` (Biber runs automatically). Then:')
    w('```')
    w(f'python3.14 RELEASE/tools/make_release.py --attach-pdf /path/to/main.pdf --release output/{name}')
    w('```')
    w('')
    w('*Generated by `RELEASE/tools/make_release.py`; not compiled; content untouched -- content questions go to the '
      'owning draft through `cross_draft/`.*')
    return '\n'.join(L) + '\n'


def make_zip(build_dir):
    """output/<build>/<build>.zip holds output/<build>/latex/ at the zip root (Overleaf-ready); notes, warnings
    and attached PDFs stay next to it in the build folder."""
    build_dir = os.path.abspath(build_dir.rstrip('/'))
    root = os.path.join(build_dir, LATEX_SUBDIR)
    if not os.path.isdir(root):
        sys.exit(f'make_release: {root} not found -- not a build folder')
    zip_path = os.path.join(build_dir, os.path.basename(build_dir) + '.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for d, _dirs, fns in os.walk(root):
            for fn in sorted(fns):
                p = os.path.join(d, fn)
                z.write(p, os.path.relpath(p, root))
    return zip_path


def prepend_changelog(entry):
    head = ('# CHANGELOG -- `Working_Space/RELEASE`\n\n'
            'One entry per build of `tools/make_release.py`, newest first: when, which v2 / v3 / v4 revisions, '
            'what was produced, the page estimate, how many holes and findings. The full record of a build is the '
            '`RELEASE_NOTES_<stamp>.md` inside its output folder.\n\n---\n\n')
    marker = '---\n\n'
    if os.path.isfile(CHANGELOG):
        text = read(CHANGELOG)
        if marker in text:
            i = text.index(marker) + len(marker)
            text = text[:i] + entry + '\n' + text[i:]
        else:
            text = head + entry + '\n' + text
    else:
        text = head + entry + '\n'
    with open(CHANGELOG, 'w', encoding='utf-8') as f:
        f.write(text)


def builds():
    if not os.path.isdir(OUT_DEFAULT):
        return []
    return sorted(d for d in os.listdir(OUT_DEFAULT)
                  if os.path.isdir(os.path.join(OUT_DEFAULT, d, LATEX_SUBDIR)))


def attach_pdf(args):
    root = args.release
    if not root:
        cands = builds()
        if not cands:
            sys.exit('no release folder found')
        root = os.path.join(OUT_DEFAULT, cands[-1])
    root = os.path.abspath(root.rstrip('/'))
    name = os.path.basename(root)
    dst = os.path.join(root, name + '.pdf')
    shutil.copyfile(args.attach_pdf, dst)
    pages = None
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(dst).pages)
    except Exception as e:      # noqa: BLE001
        print(f'page count unavailable ({e}); run with python3.14 (pypdf) or count by hand')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if pages is None:
        verdict = 'page count unknown'
    elif PAGE_MIN <= pages <= PAGE_MAX:
        verdict = f'{pages} pages -- within the {PAGE_MIN}-{PAGE_MAX} limit'
    else:
        verdict = f'{pages} pages -- OUTSIDE THE {PAGE_MIN}-{PAGE_MAX} LIMIT'
        with open(os.path.join(root, 'WARNING_PAGE_LIMIT.md'), 'a') as f:
            f.write(f'\n# PAGE LIMIT WARNING (compiled PDF, {stamp})\n\n{verdict}\n')
    notes = sorted(f for f in os.listdir(root) if f.startswith('RELEASE_NOTES_'))
    if notes:
        with open(os.path.join(root, notes[-1]), 'a') as f:
            f.write(f'\n## Compiled PDF ({stamp})\n\n`{name}.pdf` attached: **{verdict}**.\n')
    prepend_changelog(f'## {stamp} -- PDF attached to `{name}`\n\n- `{name}.pdf`: **{verdict}**\n')
    print(f'attached {dst}\n  {verdict}')
    return 0


def cmd_list():
    rows = builds()
    if not rows:
        print('no releases yet')
        return 0
    for d in rows:
        p = os.path.join(OUT_DEFAULT, d)
        z = os.path.join(p, d + '.zip')
        pdfs = [f for f in os.listdir(p) if f.lower().endswith('.pdf')]
        print(f'  {d}' + (f'   zip {os.path.getsize(z) // 1024} KB' if os.path.exists(z) else '   (no zip)')
              + (f'   pdf: {", ".join(pdfs)}' if pdfs else ''))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', help='suffix for the release folder name, e.g. GOLDEN_TEMPLATE')
    ap.add_argument('--appendix-short', action='store_true', help='hide the long-data tables of the appendix')
    ap.add_argument('--standalone', action='store_true', help='build on v2\'s standalone preamble instead of the TUM template')
    ap.add_argument('--acknowledgments', action='store_true',
                    help='include the Acknowledgments page even without RELEASE/front/acknowledgments.tex (empty page)')
    ap.add_argument('--no-cover', action='store_true', help='drop the cover page; the title page comes first')
    ap.add_argument('--note', action='append', metavar='TEXT', help='a line recorded in the notes and the changelog (repeatable)')
    ap.add_argument('--no-zip', action='store_true')
    ap.add_argument('--outdir', help=f'default: {os.path.relpath(OUT_DEFAULT, RELEASE)}')
    ap.add_argument('--dry-run', action='store_true', help='assemble and check in memory, write nothing')
    ap.add_argument('--no-log', action='store_true', help='do not record the build in CHANGELOG.md (test builds)')
    ap.add_argument('--attach-pdf', metavar='PDF', help='copy a compiled PDF into a release folder and record its page count')
    ap.add_argument('--release', metavar='FOLDER', help='the release folder for --attach-pdf (default: newest)')
    ap.add_argument('--rezip', metavar='FOLDER', help='rebuild the zip of an existing build folder (output/<build>)')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        return cmd_list()
    if a.rezip:
        z = make_zip(os.path.abspath(a.rezip))
        print(f'{z} ({os.path.getsize(z) // 1024} KB)')
        return 0
    if a.attach_pdf:
        return attach_pdf(a)
    return build(a)


if __name__ == '__main__':
    sys.exit(main())
