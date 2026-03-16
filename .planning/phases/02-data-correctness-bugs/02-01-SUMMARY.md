---
phase: 02-data-correctness-bugs
plan: 01
subsystem: testing
tags: [pytest, regression, totals, tanstack, hierarchy]
requires: []
provides:
  - Explicit regression coverage for BUG-01 through BUG-06
  - Totals coverage across sum, avg, count, min, and max
  - Adapter-level request sequence tests for filter, sort, refresh, and virtual scroll
affects: [Phase 2 backend fixes, Phase 3 regression baseline]
requirements-completed: []
completed: 2026-03-13
---

# Phase 2 Plan 01 Summary

Wave 1 added the regression net that Phase 2 needed before changing backend behavior. The new tests proved the existing failures in AVG visual totals, grand-total roll-up semantics for non-sum aggregations, stale dynamic-column discovery after data reload, and loss of sort semantics in the virtual-scroll path.

## Files

- `tests/test_visual_totals.py`
- `pivot_engine/tests/test_controller.py`
- `tests/test_frontend_contract.py`
- `tests/test_frontend_filters.py`

## Outcome

- Added aggregate-specific totals assertions for `sum`, `avg`, `count`, `min`, and `max`
- Added request-sequence adapter regressions covering repeated filter/sort requests and dynamic-column refresh after data change
- Added a virtual-scroll regression that exposed missing grand-total/sort preservation in the hierarchical path

## Verification

`python -m pytest tests/test_visual_totals.py pivot_engine/tests/test_controller.py tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py -q --tb=line`

Result after downstream fixes: `21 passed, 1 skipped`

## Notes

- No Git commit was created for this plan in this execution environment because the repository worktree contains extensive unrelated changes.
