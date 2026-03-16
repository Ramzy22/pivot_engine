---
phase: 03-virtual-scroll-ui-bugs
plan: 02
subsystem: backend
tags: [hierarchy, virtual-scroll, cache, paging, ibis]
requires: [03-01]
provides:
  - Correct hierarchy visibility transitions after `expand all`
  - Stable visible-row counts for collapsed and selectively expanded requests
  - Backend cache behavior that no longer leaks stale child rows across request states
affects: [Phase 3 frontend synchronization assumptions, hierarchy contract tests]
requirements-completed: [BUG-07, BUG-11, BUG-12]
completed: 2026-03-13
---

# Phase 3 Plan 02 Summary

Wave 2 fixed the backend hierarchy bug exposed by the new regressions. The hierarchy manager was reusing the optimized full-tree cache even after the request stopped being `expand all`, which left stale child rows visible in collapsed and selectively expanded states.

## Files

- `pivot_engine/pivot_engine/hierarchical_scroll_manager.py`

## Outcome

- Restricted `hier_full:{spec_hash}` reuse to true expand-all requests only
- Restored correct collapsed/selective visibility after a prior expand-all request
- Brought visible-row slicing back into line with `get_total_visible_row_count(...)`

## Verification

`python -m pytest test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py -q --tb=line`

Result: `6 passed`

## Notes

- No Git commit was created for this plan in this execution environment because the repository worktree contains extensive unrelated changes.
