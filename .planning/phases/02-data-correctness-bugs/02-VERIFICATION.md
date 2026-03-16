---
phase: 02-data-correctness-bugs
verified: 2026-03-13T19:00:00Z
status: passed
score: 5/5 success criteria verified
re_verification: false
---

# Phase 2 Verification Report

**Phase Goal:** Aggregated values in the pivot table are always correct and stable - grand totals, column sets, filter state, and sort state all work reliably
**Verified:** 2026-03-13T19:00:00Z
**Status:** PASSED

## Success Criteria

| # | Success Criterion | Status | Evidence |
|---|-------------------|--------|----------|
| 1 | Grand total row displays correct aggregated values for sum, avg, count, min, and max measures | VERIFIED | `tests/test_visual_totals.py` and `pivot_engine/tests/test_controller.py` now assert all five aggregation types and pass |
| 2 | Grand total row remains visible and stable during scroll, filter changes, and data refreshes | VERIFIED | `tests/test_frontend_contract.py::test_virtual_scroll_preserves_sort_and_grand_total` passes; full suite remains green after cache-clearing and hierarchical-path fixes |
| 3 | After any filter change or page refresh, all pivot columns that should exist are present in the header | VERIFIED | `tests/test_frontend_contract.py::test_dynamic_columns_refresh_after_data_change` passes |
| 4 | Applied filters remain active and correct when the user expands rows, changes sort order, or scrolls | VERIFIED | `tests/test_frontend_filters.py::test_filter_and_sort_state_survives_repeated_requests` plus hierarchical contract tests pass |
| 5 | Sort order set by the user is applied server-side and survives data refreshes | VERIFIED | `tests/test_frontend_contract.py::test_sorting`, `tests/test_frontend_contract.py::test_virtual_scroll_preserves_sort_and_grand_total`, and `pivot_engine/tests/test_controller.py::test_ibis_planner_with_sort` pass |

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| BUG-01 | SATISFIED | weighted AVG visual totals and source-based total rows are covered by passing tests |
| BUG-02 | SATISFIED | virtual-scroll grand total remains present and singular in passing regression |
| BUG-03 | SATISFIED | dynamic columns update after data change in passing regression |
| BUG-04 | SATISFIED | repeated request state remains consistent in adapter-level regressions |
| BUG-05 | SATISFIED | repeated filter/sort request regressions pass |
| BUG-06 | SATISFIED | planner/controller/hierarchical paths now preserve explicit server-side sort semantics |

## Commands Run

```text
python -m pytest tests/test_visual_totals.py pivot_engine/tests/test_controller.py tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py -q --tb=line
python -m pytest tests/test_visual_totals.py tests/test_frontend_contract.py pivot_engine/tests/test_controller.py -q --tb=line
python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q --tb=line
python -m pytest tests/test_complete_implementation.py tests/test_scalable_pivot.py -q --tb=line   (run from pivot_engine/)
python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short
```

## Notable Fixes Verified

- `ibis_planner.py`: weighted AVG visual totals and pivoted result sorting
- `controller.py`: source-of-truth total rows and cache invalidation on reload
- `hierarchical_scroll_manager.py`: virtual-scroll sort preservation and non-null grand-total generation
- `materialized_hierarchy_manager.py`: no conflicting background-thread DuckDB materialization for the scalable path

## Residual Risk

- A manual browser-level smoke test for visible flicker is still reasonable, but payload-level behavior and the project’s automated suite are green.
