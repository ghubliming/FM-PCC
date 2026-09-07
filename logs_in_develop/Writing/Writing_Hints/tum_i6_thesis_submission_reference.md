# TUM I6 Thesis Submission Reference

This document summarizes the **Thesis Submission Guidelines** page for TUM I6 and also covers the sub-links that are visible from that page itself, so it can be used as a writing and submission reference.[page:1]

## Scope of this reference

The source page belongs to the TUM I6 internal wiki under **Public Pages → Bachelor and Master theses** and explicitly links or points to these related pages/resources: **Defense Day**, **Bachelor and Master theses**, open thesis topics, staff contacts, TUM/CIT thesis handouts, registration pages, the TUM I6 LaTeX template, and the speakers list for the next meeting.[page:1]

Because only the full content of the **Thesis Submission Guidelines** page was available here, the sections below fully cover that page and summarize the role of each visible linked resource based on how the page describes it.[page:1]

## Page structure

The page is organized into three main parts: **Starting a thesis**, **Submitting your thesis**, and **Colloquium Presentation**.[page:1]
Within those parts, it includes focused subsections on **Registration Forms**, **Citations**, **Figures**, **Math notation**, **Length**, and **CD / DVD** requirements.[page:1]

## Starting a thesis

The page says the first step is to find a topic and an advisor.[page:1]
It points students to the institute website for open thesis topics and also says students may contact staff members directly to ask about open topics and discuss details.[page:1]

### External theses

The page sets several constraints for external theses.[page:1]
A thesis topic must be provided by an employee such as a professor or Privatdozent of TUM or another Bavarian university, because topic definition is treated as a responsibility of Bavarian public administration.[page:1]
Writing a thesis is described as an educational activity rather than company work, and the page also says the intellectual property belongs to the student, although it may later be sold.[page:1]
The page further states that I6 does not supervise external theses requiring an NDA and that I6 theses may not contain a non-disclosure clause (**Sperrvermerk**).[page:1]

### Official handout

For more information on external theses and the general framework, the page refers readers to the official **Handout on Theses and Dissertations** from CIT.[page:1]
That linked handout should be treated as the higher-level official policy source when institute-level guidance and faculty regulations need to be cross-checked.[page:1]

### Registration overview

After finding an advisor and topic, the thesis must be registered at TUM, and the exact procedure depends on the degree program.[page:1]
For Informatics, registration is handled through the Informatics Infopoint, and the page notes a final-submission deadline of the 15th of each month or the first working day after if the 15th falls on a weekend or public holiday.[page:1]
For Robotics, Cognition and Intelligence, the process is described as similar, but documents must be submitted directly to Dr. Alexander Lenz, with the dedicated RCI master thesis page given as the program-specific reference.[page:1]
For other subjects, the page tells students to check their own program requirements.[page:1]

### Registration form contents

The page says the registration form can differ by subject, but for Informatics and RCI it should contain the following items.[page:1]

- Personal data.[page:1]
- Thesis topic in German and English.[page:1]
- Name of the supervisor (**Themensteller/in**), which must be a professor.[page:1]
- One or more advisors (**Betreuer/in(nen)**), for example the PhD student supervising day-to-day work.[page:1]
- Signatures of the student and the supervisor.[page:1]

### Exposé expectation

The page recommends creating an exposé of about 3 to 4 pages when starting the thesis.[page:1]
It should summarize the topic and the planned work program.[page:1]

## Registration forms

The page includes a table of visible registration-form links by field of study and degree.[page:1]
Because the formatting in the source page is slightly irregular, the practical interpretation is the following.[page:1]

| Program / field | Degree | What the page indicates |
|---|---:|---|
| Informatics | B.Sc. | A downloadable registration form is provided.[page:1] |
| Automotive Software Engineering | M.Sc. | A downloadable registration form is provided.[page:1] |
| Informatics / Information Systems | M.Sc. context implied by table layout | The page layout appears compressed here, so this row should be verified against the linked faculty pages before submission.[page:1] |
| Robotics, Cognition, Intelligence | M.Sc. | A downloadable registration form is provided.[page:1] |

For actual filing, the safest reading is to use the linked registration document for the specific program and confirm against the corresponding faculty or program page before signing and submitting.[page:1]

## Submitting the thesis

For writing, the page instructs students to use the **TUM I6 LaTeX template**.[page:1]
It says this template should align with the IN TUM thesis guidelines, but students should still verify that the template remains up to date.[page:1]
In other subjects, formatting must satisfy the individual program requirements instead.[page:1]
The page also says that when handing in the thesis, an additional copy for the advisor must be included.[page:1]

### Citations

The recommended citation style is **alphadin** when using BibTeX, or **alpha** when using biblatex/biber.[page:1]
The page gives bibliography examples in the format of short bracketed labels such as **[BM92]** and **[CM92]**, followed by full publication details.[page:1]
When quoting an author word-for-word, the page explicitly requires adding a page number in the reference.[page:1]

### Figures

The page recommends using **vector graphics** for figures and diagrams.[page:1]
It specifically mentions PGF/TIKZ and Inkscape as suitable tools, and notes that text can be integrated from LaTeX for consistent typesetting.[page:1]
It also states an unwritten rule that figures not referenced in the text are unnecessary.[page:1]
When citing a figure, the reference should be included in the figure label and should include the page number.[page:1]

### Math notation

The page contains a notation guide and asks students to use a consistent style for formulas and pseudocode.[page:1]
A central rule is to use **single letters** for variable names.[page:1]

#### Notation rules captured from the table

| Category | Style required by the page | Example from page |
|---|---|---|
| Scalars | Lowercase, italic | `a`.[page:1] |
| Functions | Regular text | `\\textrm{sin}(x)`.[page:1] |
| Units | Regular text with half-space after number | `42\\,\\textrm{Hz}`.[page:1] |
| Angles | Greek letters | `\\alpha`.[page:1] |
| Absolute value | Vertical bars | `\\mid a \\mid`.[page:1] |
| Modulo | Regular text | `\\textrm{mod}`.[page:1] |
| Vectors | Lowercase, italic, bold | `\\boldsymbol{v}`.[page:1] |
| Matrices | Uppercase, italic, bold | `\\boldsymbol{M}`.[page:1] |
| Variable indices | Indexed form | `\\boldsymbol{x}_i`.[page:1] |
| Static indices | Text-form index | `\\boldsymbol{x}_\\textrm{min}`.[page:1] |
| Unit vectors | Bold upright style | `\\textbf{e}_x`.[page:1] |
| Identity matrices | Bold upright style | `\\textbf{Id}_n`.[page:1] |
| Transpose | Superscript upright T | `\\boldsymbol{x}^\\textrm{T}`.[page:1] |
| Cross product | Multiplication cross | `\\times`.[page:1] |
| Dot product | Center dot | `\\cdot`.[page:1] |
| Sets | Uppercase italic with braces | `A = \\{ 1, 2, 3 \\}`.[page:1] |
| Sequences | Calligraphic sequence notation | `\\mathcal{A} = \\langle 1,2,3 \\rangle`.[page:1] |
| Set subtraction | Backslash notation | `A \\backslash \\{ element \\}`.[page:1] |
| Union / adding an element | Cup notation | `A \\cup \\{ element \\}`.[page:1] |

This section is especially useful as a style guide when editing notation across the whole thesis, because it goes beyond formatting and sets a uniform visual convention for symbols, indices, and operators.[page:1]

### Length

The page says there are **no fixed page limits** for bachelor’s or master’s theses.[page:1]
As a practical reference, it states that many bachelor’s theses are about **40 to 60 pages** and many master’s theses are about **60 to 80 pages**, including abstract, table of contents, lists of figures and tables, and bibliography.[page:1]
It also warns that if a thesis exceeds **100 pages**, the advisor or supervisor will most likely not read all of it.[page:1]
For binding requirements and formal limits, the page points students to the module handbook on campus.tum.de.[page:1]

### CD / DVD material

For the **official copy** submitted to the faculty or program coordinator, a CD/DVD is **not required**, though the page says students should check their program website if uncertain.[page:1]
For the advisor’s or supervisor’s copy, however, the page asks for a CD or DVD attached on the last inner page of the thesis.[page:1]
The disk should include a digital copy of the thesis including LaTeX sources, the written code, and the datasets used for evaluation results.[page:1]
The stated rule of thumb is that the storage medium should contain everything needed to reproduce the thesis results.[page:1]

## Colloquium presentation

During the thesis, students are asked to attend the monthly **I6 Defense Day**.[page:1]
The page also says students are expected to give **two presentations** of their own.[page:1]

### Required talks

1. **Initial topic presentation**: usually within the first two months, intended to introduce the student, explain the topic, and present the planned approach so that early feedback can still influence the direction of the work.[page:1]
2. **Thesis defense**: given at the end of the thesis, presenting the research results to the group; the page states that this talk is compulsory and part of the grade.[page:1]

Both talks should be held in **English**.[page:1]
A LaTeX presentation template is available in the same **TUMlatex** repository referenced earlier.[page:1]
For scheduling, the page says students should ask their advisor to add them to the speakers list for the next meeting, although that speakers-list page is visible only to I6 employees.[page:1]

### Presentation timing

The page provides fixed timeslots for presentations and questions.[page:1]

| Presentation type | Talk time | Q&A time |
|---|---:|---:|
| Initial topic presentation | 5 min | 5 min [page:1] |
| BA thesis | 15 min | 5 min [page:1] |
| Guided Research | 10 min | 5 min [page:1] |
| Interdisciplinary Project | 15 min | 5 min [page:1] |
| MA thesis | 20 min | 5 min [page:1] |

The page recommends rehearsing beforehand and says that planning about **one minute per slide** is usually a good baseline.[page:1]

## Linked resources mentioned on the page

The visible or embedded links on the page serve different purposes.[page:1]
This list explains how each one functions in practice according to the page text.[page:1]

| Linked page or resource | Role described or implied by the source page |
|---|---|
| Open thesis topics | Main place to browse available thesis topics offered by the institute.[page:1] |
| Staff page | Alternative way to contact researchers or supervisors directly about available topics.[page:1] |
| Handout on Theses and Dissertations | Official higher-level policy reference, especially relevant for external theses.[page:1] |
| Informatics thesis registration page | Program-specific registration instructions for Informatics students.[page:1] |
| RCI Master Thesis page | Program-specific registration and submission guidance for RCI students.[page:1] |
| Registration form downloads | Program-specific forms needed to register the thesis topic.[page:1] |
| TUM I6 LaTeX template / TUMlatex repository | Recommended writing and presentation template source for thesis text and slides.[page:1] |
| IN TUM thesis guidelines | Cross-check source for thesis formatting and administrative expectations.[page:1] |
| Defense Day | Monthly event students should attend while doing the thesis.[page:1] |
| Speakers list for next meeting | Internal scheduling page used by advisors or staff to add speakers.[page:1] |
| Bachelor and Master theses | Parent page category that contains this guidance page and related thesis material.[page:1] |

## Writing checklist derived from the page

The following condensed checklist turns the page into action items for drafting and submission.[page:1]

### Before writing

- Confirm the thesis topic and advisor.[page:1]
- Register using the correct program-specific process and form.[page:1]
- Prepare the topic title in German and English if required by the form.[page:1]
- Draft a 3–4 page exposé with topic summary and work plan.[page:1]

### While writing

- Use the TUM I6 LaTeX template unless the program requires something else.[page:1]
- Keep citation style consistent, preferably alphadin or alpha depending on toolchain.[page:1]
- Add page numbers for direct quotations.[page:1]
- Prefer vector figures and reference every figure in the text.[page:1]
- Follow the math-notation conventions consistently.[page:1]
- Keep length reasonable; use about 40–60 pages for BA and 60–80 pages for MA as orientation, not as a strict rule.[page:1]

### Before submission

- Verify current faculty/program rules because linked pages may change over time.[page:1]
- Prepare the official submission copy according to the program rules.[page:1]
- Prepare an extra copy for the advisor.[page:1]
- Add a CD/DVD to the advisor’s copy with thesis PDF and sources, code, and datasets needed for reproducibility.[page:1]

### Presentation planning

- Attend Defense Day during the thesis period.[page:1]
- Give the initial topic presentation early in the thesis.[page:1]
- Prepare the final defense presentation in English.[page:1]
- Rehearse using roughly one minute per slide as a timing baseline.[page:1]

## Points to verify externally

The page is a strong institute reference, but several items should still be checked against current official program pages before final submission.[page:1]
These include current registration forms, whether template rules have changed, any subject-specific formatting requirements, whether a CD/DVD is still expected by an advisor, and exact module-handbook obligations for the degree program.[page:1]

## Practical use as a writing reference

Used as a writing reference, this page mainly gives four kinds of guidance: administrative setup, formatting conventions, reproducibility expectations, and presentation obligations.[page:1]
The most actionable writing-specific rules are to use the institute LaTeX template, apply a consistent citation style, prefer vector figures, reference every figure in the text, keep notation consistent, and keep the thesis within a readable scope.[page:1]
