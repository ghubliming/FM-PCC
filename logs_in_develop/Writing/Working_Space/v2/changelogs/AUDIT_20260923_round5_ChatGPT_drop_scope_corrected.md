# Audit correction — distinguish unnecessary additions from necessary findings

**Author:** ChatGPT (Codex)  
**Date:** 2026-09-23  
**Scope:** audit only; no thesis version increment.

The user clarified that the auditor should judge whether additional items really are unnecessary and drop them where justified.

Replaced §11 of `v2/audit from chatgpt/AUDIT_v2.25_against_v3.70_2026-09-23.md`. The earlier classification of eleven groups as dropped was too broad: some were only evidence, conditional suggestions or optional advice in the original audit.

The corrected decision drops three specific writing additions: the detailed noise-scale discussion in v2 (F12), preserving/expanding the redundant numerical endpoint configuration table (F21), and adding raw-residual monitoring advice to the thesis (part of F19). Each has an explicit reason and retains the associated factual correction. The latter two decisions apply the user's authorization to assess necessity; they are not presented as individually approved by the user.

Retained F03, F13, F17, F18, scene/control descriptions, F04 and F09 as necessary corrections. Figure-format advice remains optional, as it originally was. This correction supersedes the broad dropped-item classification in `AUDIT_20260923_round4_ChatGPT_drop_unnecessary_additions.md`; that historical log is preserved.

Checked the rewritten section against the original recommendations. No thesis, bibliography, code, configuration, v3 source, existing changelog or Claude file changed. No compilation or experiments performed.

**Signed:** ChatGPT (Codex).
