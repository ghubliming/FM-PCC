---
name: dont-self-edit-master-test-history
description: Never edit MASTER_TEST_HISTORY.md unless explicitly told; it is not your job
metadata: 
  node_type: memory
  type: feedback
  originSessionId: bd7818bb-1de6-4bfb-9771-e06f31cce259
---

Do NOT add entries to `logs_in_develop/MASTER_TEST_HISTORY.md` on your own initiative. Only touch it when the user explicitly commands it. Most of the time updating that master index is not your job.

**Why:** The user maintains the master index themselves; unsolicited edits are unwanted.

**How to apply:** When finishing a task (e.g. a changelog under `logs_in_develop/<gen>/`), do the changelog per [[changelog-after-coding-tasks]], but leave MASTER_TEST_HISTORY.md alone. You may *offer* to add a pointer, but never add it without being told. This generalizes: default to not self-adding to shared index/tracking files.
