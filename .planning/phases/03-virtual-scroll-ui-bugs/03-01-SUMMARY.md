---
phase: 03-virtual-scroll-ui-bugs
plan: 01
subsystem: testing
tags: [pytest, regression, virtual-scroll, hierarchy, cache]
requires: []
provides:
  - Explicit regression coverage for virtual-scroll continuity and hierarchy stability
  - Adapter-level request sequence coverage for scroll, filter, sort, expand, and collapse flows
  - Manager-level regressions for expand-all, collapse, and sibling stability
affects: [Phase 3 backend fixes, Phase 3 frontend synchronization fixes]
requirements-completed: []
completed: 2026-03-13
---

# Phase 3 Plan 01 Summary

Wave 1 added the missing regression net for Phase 3. The new tests locked down virtual-scroll identity, repeated-request stability, expand/collapse ordering, and the stale-child-row bug after `expand all`.

## Files

- `tests/test_frontend_contract.py`
- `tests/test_frontend_filters.py`
- `test_expand_all_backend.py`
- `pivot_engine/tests/test_hierarchical_managers.py`

## Outcome

- Added adapter-level regressions for hierarchical load -> virtual-scroll continuity, repeated virtual-scroll windows, and filter/sort transitions
- Added hierarchy-manager regressions for `expand all -> selective expansion`, `expand all -> collapse`, and expand/collapse/re-expand ordering
- Exposed the backend bug where the optimized full-tree cache leaked stale child rows into narrower visibility states

## Verification

`python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py pivot_engine/tests/test_scalable_pivot.py -q --tb=line`

Result after downstream fixes: `23 passed`

## Notes

- No Git commit was created for this plan in this execution environment because the repository worktree contains extensive unrelated changes.
