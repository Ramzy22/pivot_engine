---
phase: 03-virtual-scroll-ui-bugs
plan: 03
subsystem: frontend
tags: [react, virtual-scroll, cache, synchronization]
requires: [03-01]
provides:
  - Request-aware server-side row-cache invalidation
  - Less brittle row/window synchronization in the virtualized grid
  - Correctness-first behavior across sort, filter, and expansion changes
affects: [Phase 3 UI behavior, virtual-scroll stability]
requirements-completed: [BUG-07, BUG-08]
completed: 2026-03-13
---

# Phase 3 Plan 03 Summary

Wave 2 also tightened the React-side virtual-scroll behavior. The row-model hook now clears cached blocks when the effective request state changes, and the grid no longer treats harmless object-identity churn as a stale-row condition.

## Files

- `dash_tanstack_pivot/src/lib/hooks/useServerSideRowModel.js`
- `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js`

## Outcome

- Added a request-state cache key to the server-side row model so sort, filter, expansion, row-count, and data-version changes invalidate stale blocks
- Reset request-version tracking when the cache scope changes
- Replaced the strict object-identity synchronization check with logical row identity based on `_path` / total-row identity

## Verification

`python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q --tb=line`

Result: `12 passed`

## Notes

- No Git commit was created for this plan in this execution environment because the repository worktree contains extensive unrelated changes.
