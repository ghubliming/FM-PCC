# Naming — the folder

One question: **is the name true?** A name is a claim, and an implementation identifier that has
hardened into a label can smuggle a claim into a thesis. Three have already tried.

| file | what it is | status |
|---|---|---|
| **[`TRANSLATION_20260914_dev_jargon_to_scientific.md`](TRANSLATION_20260914_dev_jargon_to_scientific.md)** | 🟢 **THE translation list (since 2026-09-14).** Every dev token of this repo → its scientific name, with a ✅ keep / 🔁 rename / 🚫 never verdict; includes all rows of the older table plus terms like *backbone*, *NFE*, *arm*, *fan*, *PCC*, *SOTA*, *S&C*. | **canonical; start here** |
| [`NAMING_20260910_master_table.md`](NAMING_20260910_master_table.md) | 🟡 superseded as the list; kept for its reasoning. Three columns — *code token* │ *paper name* │ *what it really is → thesis name* — for engines, constraint arms, selection rules, backbone/conditioning, geometry, protocol and embodiment. Plus the five standing rules. | reasoning |
| [`../NOTES_method_naming.md`](../NOTES_method_naming.md) | the *argument* behind the method names: the author's position, the options weighed, and why `endpoint projection` beat the alternatives | rationale — read if you want to reopen a decision |
| [`../NOTES_naming_and_rebuild.md`](../NOTES_naming_and_rebuild.md) | the *code-flag traps*, and why `logs_in_develop/Rebuild_repo/` is never cited | rationale + the standing rule on the rebuild doc |

**Where they disagree, the translation list wins.** The others are kept because they carry the
reasoning; the list carries the decision.

### The four names that were false, and are now fixed

| was | is | where it had reached |
|---|---|---|
| `film_mode='v1'` → "FiLM conditioning" | **concatenated conditioning** (FiLM with `γ ≡ 0`) | every config, every `filmv1` tag, every eval log — and `thesis_v2.tex` until v2.4 |
| "HF-SLSQP" / "HardFlow" for arm C | **in-ODE endpoint projection** | every DA and CSV variant string |
| "In-Loop Trajectory Optimisation" (§ title) | **In-ODE Endpoint Projection** — our port optimises no objective | `thesis_v2.tex` until v2.6 |
| `CascadedPID` / run tag `pid` | **cascaded geometric tracking controller** — there is **no integral term at all**, and the inner loop is a coordinate-free `SO(3)` controller, not a scalar loop | the class name, every run tag, and `thesis_v2.tex` until v2.7 |

### Applying it

`Working_Space/v2/thesis_v2.tex` is compliant as of **v2.7**; see that folder's `CHANGELOG.md`.
The artefacts (CSVs, DAs, checkpoints, run ledger) deliberately keep the old tokens — rule 5.
