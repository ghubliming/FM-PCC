---
name: no-unrequested-code-edits
description: Never edit code or write changelogs unless the user explicitly asks — answer questions as questions
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7b99094f-7b84-4530-b9fe-470af9e71bdf
---

When the user asks a question (e.g. "do I only need to change X?"), answer it — do NOT proactively edit code, config, or write changelog MDs. The user decides when changes are made.

**Why:** 2026-07-13 during Gen3v4_imf debugging, I patched the eval script + config and wrote a changelog in response to a pure question; the user objected ("i never let you change any of the code or write changelog") and everything had to be reverted.

**Carve-out (2026-08-31, explicit): MD reports/analysis docs are NOT covered.** *"alwys update MD, Dont ask me"* — when a question extends or corrects an analysis/report MD I already own, WRITE the update into the file immediately and report what landed. Never ask "want me to fold this in?" for an MD. Code and config still need a go-ahead.

**How to apply:** for code/config, propose the exact edit (snippet + file:line) in the reply and wait for explicit go-ahead; for MDs, just do it. Related: [[no-auto-commit-no-coauthor]], [[changelog-after-coding-tasks]] (changelogs only after *requested* coding tasks).
