#!/usr/bin/env python3
r"""make_release_v5.py -- build the submission-clean thesis from v5 (the aggregate) into RELEASE/output/.

Since 2026-09-25 (author) the thesis lives in Working_Space/v5 as ONE draft, edited by the Orchestra chat
("Advance Orchestra"). This tool is the v5 counterpart of ../../RELEASE/tools/make_release.py, which reads the
three legacy drafts v2 / v3 / v4. It IMPORTS that tool and reuses its cleaner, its checks, its page model, its
notes, zip and changelog writer, and differs only in WHERE the sources come from and HOW a build is named and
marked:

    sources   v5/parts/00_preamble.tex (v2's preamble: metadata, packages, notation, drafting macros),
              parts/00_preamble_v3.tex, parts/00_preamble_v4.tex, parts/01_frontmatter.tex (the abstract),
              parts/99_backmatter.tex (acronyms), chapters/01..09 (+ nested app_* inputs), the three .bib
              files, figures/ (then the DA store Data_Analysis/DA_in_Paper/figures)
    name      RELEASE/output/<YYYYMMDD_HHMMSS>_thesis_release_ORCH_<v5.N>[_TAG][_standalone][_appendixshort][_PAGEWARN]/
              -- the DATE-TIME is the identity of a build (author: "use majorly datetime to distinguish");
              ORCH marks it as the Orchestra's; the v5 revision is information, not the key
    marked    "built by the Orchestra from v5" in the folder name, in RELEASE_NOTES_<stamp>.md and in
              RELEASE/CHANGELOG.md, together with the v2 / v3 / v4 revisions v5 was initialised from
              (v5/inherited/INIT_STATE.json) and, with --job, the Orchestra job

Everything the legacy tool guarantees holds here: every build is kept and never edited or deleted, nothing is
compiled (no TeX toolchain in this container), content is never changed by a build -- a content problem is
fixed in v5, recorded in v5/CHANGELOG.md, and the release is rebuilt.

USAGE (from Working_Space/v5)
    python3 tools/make_release_v5.py --dry-run                                  # assemble + check in memory, write nothing
    python3 tools/make_release_v5.py [--job O###] [--tag TAG] [--note "..."]    # the build: output/<build>/{latex/, zip, notes}
                                     [--appendix-short] [--standalone] [--no-cover] [--acknowledgments] [--no-zip] [--no-log]
    python3 tools/make_release_v5.py --list                                     # every build in RELEASE/output/ (legacy and v5)
    python3.14 tools/make_release_v5.py --attach-pdf main.pdf [--release FOLDER] # record the compiled page count
    python3 tools/make_release_v5.py --rezip FOLDER

Python 3 stdlib only (pypdf under python3.14 for --attach-pdf), as the legacy tool.
"""
import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.dirname(HERE)                         # Working_Space/v5
WS = os.path.dirname(V5)                           # Working_Space
RELEASE = os.path.join(WS, 'RELEASE')
LEGACY_TOOL = os.path.join(RELEASE, 'tools', 'make_release.py')
INIT_STATE = os.path.join(V5, 'inherited', 'INIT_STATE.json')

_spec = importlib.util.spec_from_file_location('make_release', LEGACY_TOOL)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)                        # the cleaner, checks, page model, notes, zip, changelog

PARTS = {
    'preamble': 'parts/00_preamble.tex',
    'pre_v3': 'parts/00_preamble_v3.tex',
    'pre_v4': 'parts/00_preamble_v4.tex',
    'front': 'parts/01_frontmatter.tex',
    'back': 'parts/99_backmatter.tex',
}
CHAPTERS_1_6 = ['chapters/01_introduction.tex', 'chapters/02_background.tex', 'chapters/03_related_work.tex',
                'chapters/04_method.tex', 'chapters/05_setup.tex', 'chapters/06_results.tex']
CHAPTERS_7_9 = list(M.V4_CHAPTERS)                 # 07_conclusion, 08_discussion, 09_appendix (+ nested inputs)
BIBS = [('v5/bibliography.bib', os.path.join(V5, 'bibliography.bib')),
        ('v5/bibliography_v3.bib', os.path.join(V5, 'bibliography_v3.bib')),
        ('v5/bibliography_v4.bib', os.path.join(V5, 'bibliography_v4.bib'))]


def v5_version():
    return M.draft_version(V5)                     # highest '## v5.N' heading of v5/CHANGELOG.md


def init_state():
    try:
        with open(INIT_STATE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def carried_line(st):
    b = st.get('built_on', {})
    if not b:
        return 'v2 / v3 / v4 revisions at init unknown (inherited/INIT_STATE.json missing)'
    return ' · '.join(f'{d} **{b[d]["short"]}**' for d in ('v2', 'v3', 'v4') if d in b) + \
        f' (initialised {st.get("initialised_at", "?")}, job {st.get("job", "?")})'


def figure_sources():
    """name -> {ext: path}; v5/figures first, then the DA store (first hit wins per name and extension)."""
    dirs = [os.path.join(V5, 'figures')]
    for root, _d, _f in os.walk(M.DA_FIGS):
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
#  the build
# =============================================================================================
def build(args):
    rep = M.Report()
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    states = {'standalone': bool(args.standalone), 'appendixfull': not args.appendix_short}
    v5s, v5full = v5_version()
    st = init_state()
    name = f'{stamp}_thesis_release_ORCH_{v5s}'
    if args.tag:
        name += '_' + re.sub(r'[^A-Za-z0-9_-]+', '_', args.tag)
    if args.standalone:
        name += '_standalone'
    if args.appendix_short:
        name += '_appendixshort'
    outdir = os.path.abspath(args.outdir or M.OUT_DEFAULT)
    build_dir = os.path.join(outdir, name)
    root = os.path.join(build_dir, M.LATEX_SUBDIR)
    files, binaries, sources = {}, {}, []

    def src(path):
        sources.append(('v5', os.path.relpath(path, WS), M.read(path).count('\n'), M.sha12(path)))

    def take(rel):
        p = os.path.join(V5, rel)
        if not os.path.isfile(p):
            sys.exit(f'make_release_v5: v5 file missing: {rel}')
        src(p)
        return p

    # ---- preamble (v2's, with the TUM metadata) -------------------------------------------------
    p = take(PARTS['preamble'])
    raw_pre = M.read(p)
    pre_v2 = M.clean_tex(raw_pre, f'v5:{PARTS["preamble"]}', rep, states, preamble=True)
    if not args.standalone:
        pre_v2 = M.reorder_template_preamble(pre_v2, rep)
        pre_v2 = M.fix_doctype(pre_v2, rep)
    M.metadata_holes(pre_v2, rep)
    if (m := re.search(r'\\getTitleGer\}\{([^}]*)\}[^\n]*%[^\n]*TODO', raw_pre)):
        rep.hole('metadata', 'main.tex', 0, f'\\getTitleGer = `{m.group(1)}` is marked "TODO: confirm" in the preamble -- '
                 'printed on the title page')

    # ---- front matter: the abstract is the \ifstandalone prose block ---------------------------
    p = take(PARTS['front'])
    front_raw = M.strip_comments(M.read(p), f'v5:{PARTS["front"]}', rep)
    abstract = None
    for _s, _e, tp, _ep in M.find_if_blocks(front_raw, 'standalone'):
        if r'\begin{titlepage}' not in tp and len(M.plain_words(tp)) > 40:
            abstract = M.tidy(M.dedent(tp))
    if abstract is None:
        sys.exit('make_release_v5: the abstract block (\\ifstandalone ... \\fi with prose) was not found in parts/01_frontmatter.tex')
    front = M.clean_tex(front_raw, f'v5:{PARTS["front"]}', rep, states)
    front = re.sub(r'^\\pagenumbering\{alph\}\n?', '', front, flags=re.M)
    ack_text = ''
    ack_src = os.path.join(M.FRONT_DIR, 'acknowledgments.tex')
    if os.path.isfile(ack_src):
        ack_text = M.tidy(M.strip_comments(M.read(ack_src), 'front/acknowledgments.tex', rep)).strip()
    want_ack = bool(ack_text) or args.acknowledgments
    if not args.standalone:
        if not want_ack:
            front = re.sub(r'^\\input\{pages/acknowledgments\}\n?', '', front, flags=re.M)
            rep.info.append('Acknowledgments page dropped: it is optional and there is no text. Put the text in '
                            'RELEASE/front/acknowledgments.tex, or pass --acknowledgments for the empty page')
        if args.no_cover:
            front = re.sub(r'^\\input\{pages/cover\}\n?', '', front, flags=re.M)
            rep.info.append('cover page dropped (--no-cover): the title page is the first page')

    # ---- back matter (the acronym list, the lists, the bibliography call) ------------------------
    p = take(PARTS['back'])
    back = M.clean_tex(M.read(p), f'v5:{PARTS["back"]}', rep, states)

    # ---- chapters 1-6 ------------------------------------------------------------------------
    for rel in CHAPTERS_1_6:
        p = take(rel)
        files[rel] = M.clean_tex(M.read(p), f'v5:{rel}', rep, states)
    p = take(PARTS['pre_v3'])
    pre_v3 = M.clean_tex(M.read(p), f'v5:{PARTS["pre_v3"]}', rep, states, preamble=True)

    # ---- chapters 7-9 and every nested \input (the appendix inputs app_long/*, app_ntrial20_*) ------
    pending = list(CHAPTERS_7_9)
    while pending:
        rel = pending.pop(0)
        p = take(rel)
        text = M.clean_tex(M.read(p), f'v5:{rel}', rep, states)
        files[rel] = text
        for target in M.RE_INPUT.findall(text):
            cand = target if target.endswith('.tex') else target + '.tex'
            if cand not in files and cand not in pending:
                pending.append(cand)
    p = take(PARTS['pre_v4'])
    pre_v4 = M.clean_tex(M.read(p), f'v5:{PARTS["pre_v4"]}', rep, states, preamble=True)

    # ---- bibliography: ONE aggregated file ------------------------------------------------------
    for rel, p in BIBS:
        if os.path.isfile(p):
            src(p)
    bib_text, bibkeys = M.aggregate_bib(BIBS, rep)
    files['bibliography.bib'] = bib_text
    pres = {'pre_v2': pre_v2, 'pre_v3': pre_v3, 'pre_v4': pre_v4}
    for k, t in pres.items():
        t2 = re.sub(r'^[ \t]*\\addbibresource\{(?!bibliography\.bib\})[^}]*\}[^\n]*\n?', '', t, flags=re.M)
        if t2 != t:
            rep.info.append(f'{k[4:]} preamble: \\addbibresource of a per-draft .bib removed (one aggregated bibliography.bib)')
        pres[k] = t2
    pre_v2, pre_v3, pre_v4 = pres['pre_v2'], pres['pre_v3'], pres['pre_v4']

    # ---- acronyms: the list of parts/99_backmatter.tex against what the text uses -------------------
    body_all = '\n'.join(files[r] for r in files if r.endswith('.tex')) + front + abstract
    used_ac = set(M.RE_AC.findall(body_all))
    back = M.acronym_merge(back, [], used_ac, rep)

    # ---- main.tex ---------------------------------------------------------------------------------
    main = [pre_v2.rstrip('\n'), '']
    if pre_v3.strip():
        main += [pre_v3.rstrip('\n'), '']
    if pre_v4.strip():
        main += [pre_v4.rstrip('\n'), '']
    main += [r'\begin{document}', '', r'\pagenumbering{alph}', front.rstrip('\n'), '']
    for rel in CHAPTERS_1_6 + CHAPTERS_7_9[:2]:
        main.append('\\input{%s}' % rel[:-4])
    main.append('')
    if not re.search(r'^\\appendix\b', files[CHAPTERS_7_9[2]], re.M):
        main.append(r'\appendix{}')
        rep.info.append('\\appendix{} added by the release tool (the appendix file does not carry it)')
    main.append('\\input{%s}' % CHAPTERS_7_9[2][:-4])
    main += ['', back.rstrip('\n'), '', r'\end{document}']
    files['main.tex'] = M.tidy('\n'.join(main))

    # ---- template files (read-only source; copies only) ------------------------------------------------
    if not args.standalone:
        for fn in ('settings.tex', 'main.xmpdata'):
            files[fn] = M.tidy(M.strip_comments(M.read(os.path.join(M.TEMPLATE, fn)), f'template:{fn}', rep))
        for page in M.TEMPLATE_PAGES:
            if page == 'acknowledgments' and not want_ack:
                continue
            if page == 'cover' and args.no_cover:
                continue
            text = M.tidy(M.strip_comments(M.read(os.path.join(M.TEMPLATE, 'pages', page + '.tex')),
                                           f'template:pages/{page}.tex', rep))
            if page == 'title':
                text = M.fit_title_page(text, rep)
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
        for fn in M.TEMPLATE_VERBATIM:
            binaries[fn] = os.path.join(M.TEMPLATE, fn)
        for fn in sorted(os.listdir(os.path.join(M.TEMPLATE, 'logos'))):
            binaries[f'logos/{fn}'] = os.path.join(M.TEMPLATE, 'logos', fn)
    else:
        for fn in M.TEMPLATE_VERBATIM:
            binaries[fn] = os.path.join(M.TEMPLATE, fn)

    # ---- figures: exactly one file per figure, .pdf preferred ---------------------------------------
    wanted = sorted({nm for _o, nm in M.RE_GRAPHIC.findall(body_all)})
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
                 f'{len(raster)} of {len(wanted)} figures ship as raster (.png), no PDF exists in v5/figures or the DA '
                 f'store; the institute asks for vector figures. Convert the SVGs (DA_in_Paper/plotting/svg/svg2pdf.sh on '
                 f'a machine with Inkscape/rsvg), copy them into v5/figures and rebuild -- the tool prefers a .pdf '
                 f'automatically. Raster-only: ' + ', '.join(raster))

    # ---- post checks on the cleaned texts ----------------------------------------------------------
    for rel, text in files.items():
        if rel.endswith('.tex') or rel.endswith('.xmpdata'):
            M.post_check(text, rel, rep)

    if args.dry_run:
        print(f'DRY RUN -- would write {len(files)} text file(s) and {len(binaries)} binary file(s) to {root}')
        print(f'  from v5 {v5s} ({len(sources)} source files); v5 carries {carried_line(st)}')
        print(f'  holes: {len(rep.holes)}   bugs/findings: {len(rep.bugs)}   residue hits: {len(rep.residue)}')
        for b in rep.bugs:
            print('  BUG ', b)
        return 0

    # ---- write ---------------------------------------------------------------------------------------
    if os.path.exists(build_dir):
        sys.exit(f'make_release_v5: {build_dir} exists already')
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

    # ---- checks on the written tree, the outline and the page estimate (the legacy model) ---------------
    docs, missing_inputs = M.collect_tree(root, 'main.tex')
    stats = M.mechanical_checks(root, docs, missing_inputs, bibkeys, rep)
    order = CHAPTERS_1_6 + CHAPTERS_7_9
    rows, lof, lot, est = M.outline_and_estimate(files, order)
    body_pages = sum(c['pages'] for c in est)
    n_toc = sum(1 for r in rows if r[2] <= 2)
    front_pages = (2 if args.no_cover else 3) + (1 if want_ack else 0) + math.ceil((n_toc + 6) / 38)
    n_cited = len(stats['cited'])
    back_pages = 1 + math.ceil(len(lof) * 1.6 / 41) + math.ceil(len(lot) * 1.6 / 41) + n_cited * M.BIB_LINES_PER_ENTRY / 41
    total = body_pages + front_pages + back_pages
    lo, hi = 0.85 * total, 1.2 * total
    verdict = f'within the {M.PAGE_MIN}-{M.PAGE_MAX} limit'
    pagewarn = False
    if total > M.PAGE_MAX or total < M.PAGE_MIN:
        verdict = f'OUTSIDE the {M.PAGE_MIN}-{M.PAGE_MAX} limit'
        pagewarn = True
    elif hi > M.PAGE_MAX or lo < M.PAGE_MIN:
        verdict = f'inside the {M.PAGE_MIN}-{M.PAGE_MAX} limit, but the uncertainty band touches it'
    guide = ('above' if total > M.PAGE_GUIDE[1] else 'below' if total < M.PAGE_GUIDE[0] else 'within')
    uncited = sorted(set(bibkeys) - set(stats['cited']))
    if pagewarn:
        os.rename(build_dir, build_dir + '_PAGEWARN')
        build_dir, name = build_dir + '_PAGEWARN', name + '_PAGEWARN'
        root = os.path.join(build_dir, M.LATEX_SUBDIR)

    # ---- notes (the legacy notes with the v5 header and footer) -----------------------------------------
    notes = os.path.join(build_dir, f'RELEASE_NOTES_{stamp}.md')
    with open(notes, 'w', encoding='utf-8') as f:
        f.write(render_notes_v5(name, stamp, (v5s, v5full), st, args.job, sources, files, binaries, rep, stats, rows, lof,
                                lot, est, front_pages, back_pages, total, lo, hi, verdict, guide, uncited, args,
                                wanted, raster))
    if pagewarn:
        with open(os.path.join(build_dir, 'WARNING_PAGE_LIMIT.md'), 'w') as f:
            f.write(f'# PAGE LIMIT WARNING\n\nEstimated {total:.0f} pages ({lo:.0f}-{hi:.0f}); {verdict}. '
                    f'See RELEASE_NOTES_{stamp}.md.\n')
    zip_path = ''
    if not args.no_zip:
        zip_path = M.make_zip(build_dir)

    # ---- RELEASE/CHANGELOG.md: the entry, marked as the Orchestra's build from v5 -------------------------
    entry = [f'## {stamp} -- {name}', '',
             f'- **Built by the Orchestra from v5 {v5s}** ({v5full}) · job {args.job or "—"} · `v5/tools/make_release_v5.py`',
             f'- **v5 was initialised from:** {carried_line(st)}; what changed in v5 since is in `v5/CHANGELOG.md`',
             f'- **Mode:** {"standalone (no TUM template)" if args.standalone else "TUM template"}'
             f'{"; appendix long-data tables hidden" if args.appendix_short else ""}'
             f'{"; tag " + args.tag if args.tag else ""}',
             f'- **Output:** `output/{name}/{M.LATEX_SUBDIR}/` (main.tex + {len(files) - 1} text files, {len(binaries)} binary files)'
             + (f', `output/{name}/{os.path.basename(zip_path)}` ({os.path.getsize(zip_path) // 1024} KB)' if zip_path else ''),
             f'- **Estimate:** ~{total:.0f} pages ({lo:.0f}-{hi:.0f}), {verdict}; {guide} the {M.PAGE_GUIDE[0]}-{M.PAGE_GUIDE[1]} '
             f'guideline. NOT compiled (no TeX toolchain here).',
             f'- **Holes recorded:** {len(rep.holes)} · **bugs/findings:** {len(rep.bugs)} · figures {len(wanted)} '
             f'({len(raster)} raster) · bibliography {len(bibkeys)} entries, {n_cited} cited · labels {stats["labels"]}',
             f'- **Notes:** `output/{name}/RELEASE_NOTES_{stamp}.md`']
    if args.note:
        entry += [f'- **Note:** {n}' for n in args.note]
    entry.append('')
    if not args.no_log:
        M.prepend_changelog('\n'.join(entry))

    # ---- console -----------------------------------------------------------------------------------------
    print(f'RELEASE  {name}   (Orchestra, from v5)')
    print(f'  built from v5 {v5s}; v5 carries {carried_line(st)}   ({len(sources)} source files)')
    print(f'  {len(files)} text files, {len(binaries)} binary files -> {os.path.relpath(root, RELEASE)}/')
    if zip_path:
        print(f'  zip: {os.path.relpath(zip_path, RELEASE)} ({os.path.getsize(zip_path) // 1024} KB)')
    print(f'  page estimate: ~{total:.0f} ({lo:.0f}-{hi:.0f}) -- {verdict}; {guide} the {M.PAGE_GUIDE[0]}-{M.PAGE_GUIDE[1]} guideline')
    print(f'  holes: {len(rep.holes)}   bugs/findings: {len(rep.bugs)}   residue hits: {len(rep.residue)}')
    for b in rep.bugs:
        print('  BUG  ' + b[:160])
    print(f'  notes: {os.path.relpath(notes, RELEASE)}')
    if not args.no_log:
        print(f'  RELEASE/CHANGELOG.md: entry prepended (marked Orchestra / v5). Record the build in v5/CHANGELOG.md and the Orchestra job.')
    return 0


def render_notes_v5(name, stamp, v5, st, job, sources, files, binaries, rep, stats, rows, lof, lot, est, front_pages,
                    back_pages, total, lo, hi, verdict, guide, uncited, args, wanted, raster):
    """The legacy notes, with the header and the footer saying that the Orchestra built this from v5."""
    text = M.render_notes(name, stamp, v5, ('-', '-'), ('-', '-'), sources, files, binaries, rep, stats, rows, lof, lot,
                          est, front_pages, back_pages, total, lo, hi, verdict, guide, uncited, args, wanted, raster)
    lines = text.split('\n')
    when = f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}'
    head = [f'**Built:** {when} by the **Orchestra** with `v5/tools/make_release_v5.py` (the v5 counterpart of '
            f'`RELEASE/tools/make_release.py`, whose cleaner and checks it reuses)'
            + (f' · **Job:** {job}' if job else '')
            + f' · **Mode:** {"standalone" if args.standalone else "TUM template"}'
            + (' · appendix long-data tables hidden' if args.appendix_short else ''),
            f'**Built from v5 {v5[0]}** ({v5[1]}) -- the aggregate of the thesis. v5 was initialised from '
            f'{carried_line(st)}; every change since is a `## v5.N` entry in `v5/CHANGELOG.md`.']
    if len(lines) > 3 and lines[2].startswith('**Built:**') and lines[3].startswith('**Built on:**'):
        lines[2:4] = head
    else:                                          # the legacy layout changed: put the v5 header first
        lines[2:2] = head + ['']
    footer = ('*Generated by `v5/tools/make_release_v5.py` (Orchestra); not compiled; content untouched -- a content '
              'problem is fixed in v5, recorded in `v5/CHANGELOG.md`, and the release rebuilt.*')
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith('*Generated by'):
            lines[i] = footer
            break
    else:
        lines.append(footer)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', help='the Orchestra job this build belongs to, e.g. O004 (recorded in the notes and the changelog)')
    ap.add_argument('--tag', help='suffix for the release folder name, e.g. GOLDEN')
    ap.add_argument('--appendix-short', action='store_true', help='hide the long-data tables of the appendix')
    ap.add_argument('--standalone', action='store_true', help="build on v2's standalone preamble instead of the TUM template")
    ap.add_argument('--acknowledgments', action='store_true',
                    help='include the Acknowledgments page even without RELEASE/front/acknowledgments.tex (empty page)')
    ap.add_argument('--no-cover', action='store_true', help='drop the cover page; the title page comes first')
    ap.add_argument('--note', action='append', metavar='TEXT', help='a line recorded in the notes and the changelog (repeatable)')
    ap.add_argument('--no-zip', action='store_true')
    ap.add_argument('--outdir', help=f'default: {os.path.relpath(M.OUT_DEFAULT, V5)}')
    ap.add_argument('--dry-run', action='store_true', help='assemble and check in memory, write nothing')
    ap.add_argument('--no-log', action='store_true', help='do not record the build in RELEASE/CHANGELOG.md (test builds)')
    ap.add_argument('--attach-pdf', metavar='PDF', help='copy a compiled PDF into a release folder and record its page count')
    ap.add_argument('--release', metavar='FOLDER', help='the release folder for --attach-pdf (default: newest)')
    ap.add_argument('--rezip', metavar='FOLDER', help='rebuild the zip of an existing build folder')
    ap.add_argument('--list', action='store_true', help='every build in RELEASE/output/ (legacy and v5)')
    a = ap.parse_args()
    if a.list:
        return M.cmd_list()
    if a.rezip:
        z = M.make_zip(os.path.abspath(a.rezip))
        print(f'{z} ({os.path.getsize(z) // 1024} KB)')
        return 0
    if a.attach_pdf:
        return M.attach_pdf(a)
    return build(a)


if __name__ == '__main__':
    sys.exit(main())
