---
phase: 02-data-correctness-bugs
plan: 04
subsystem: hierarchy
tags: [virtual-scroll, hierarchy, grand-total, duckdb]
requires:
  - phase: 02-data-correctness-bugs plan 02
    provides: correct totals and finalized result path
  - phase: 02-data-correctness-bugs plan 03
    provides: stable discovery/cache invalidation
provides:
  - Stable hierarchical sort behavior
  - Grand total row preserved in virtual-scroll path
  - Materialized hierarchy flow no longer trips DuckDB pending-result error
affects: [BUG-02, BUG-05, BUG-06]
requirements-completed: [BUG-02, BUG-05]
completed: 2026-03-13
---

# Phase 2 Plan 04 Summary

Wave 3 closed the remaining user-visible integration bugs. The hierarchical virtual-scroll path now honors explicit server-side sort order instead of reordering only by dimensions, and grand-total rows remain present in the virtual-scroll response. The deferred scalable/materialized-hierarchy test is now active and passing because DuckDB materialization no longer runs on a conflicting background-thread connection.

## Files

- `pivot_engine/pivot_engine/hierarchical_scroll_manager.py`
- `pivot_engine/pivot_engine/materialized_hierarchy_manager.py`
- `pivot_engine/tests/test_complete_implementation.py`
- `tests/test_frontend_contract.py`

## Verification

- `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py -q --tb=line`
- `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short`

Results:

- quick persistence subset: passing as part of the Phase 2 quick suite
- full suite: `65 passed, 12 skipped`

## Notes

- No Git commit was created for this plan in this execution environment because the repository worktree contains extensive unrelated changes.
