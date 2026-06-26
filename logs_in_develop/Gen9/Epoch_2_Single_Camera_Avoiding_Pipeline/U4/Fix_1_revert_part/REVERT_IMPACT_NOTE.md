# Fix1 — Revert impact (concise note)

**Date:** 2026-06-14
Companion to [CHANGELOG_Fix1.md](./CHANGELOG_Fix1.md).

## TL;DR

The Fix1 revert **does make a real difference** vs. the pre-revert state — but the **general
behaviour does not change too much**.

## What does change

- **Checkpoint selection** now scores the **EMA model** in `test()` (matching what eval actually
  deploys), instead of the raw `self.model`. So *which* checkpoint is picked as "best" can differ.
- The reported test/val loss curve shifts to the EMA model's numbers.

## What does *not* change much

- Training dynamics, optimizer, data, and the model itself are untouched.
- For a converged run the EMA and raw model are close, so the selected checkpoint is usually the
  same or a near neighbour — final deployed quality is broadly similar.
- No architectural or objective change; purely a selection/measurement alignment.

## Bottom line

Material enough to fix (selection now honest about what's deployed), but **not a behaviour
overhaul** — expect a small, generally-positive shift, not a different model regime. See the
CHANGELOG_Fix1 warning section for the material vs non-material breakdown.
