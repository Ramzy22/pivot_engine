---
phase: 06-drill-through-excel-export
plan: 01
subsystem: requirements + test scaffolding
tags: [tdd, red-phase, requirements, drill-through, excel-export]
dependency_graph:
  requires: []
  provides: [tests/test_drill_through.py, tests/test_export.py, EXPORT-05 requirement]
  affects: [.planning/REQUIREMENTS.md]
tech_stack:
  added: []
  patterns: [TDD red-phase, Flask test client via app.server.test_client()]
key_files:
  created:
    - tests/test_drill_through.py
    - tests/test_export.py
  modified:
    - .planning/REQUIREMENTS.md
decisions:
  - "Set TESTING flag on app.server.config (Flask) not app.config (Dash wrapper) — Dash rejects unknown config keys"
  - "Export tests in test_export.py validate the Python controller layer directly (asyncio.run), not JS export helpers"
  - "Sort test skipped with pytest.skip() rather than xfail so it is clearly marked as future work for Plan 02"
  - "test_sort_and_filter validates endpoint accepts sort params without 500 error, not actual sort order (sort not yet implemented)"
metrics:
  duration: 3m
  completed: 2026-03-15
  tasks_completed: 2
  files_modified: 3
---

# Phase 06 Plan 01: Requirements Update and RED Test Scaffolds Summary

**One-liner:** EXPORT-05 requirement added to REQUIREMENTS.md and RED pytest scaffolds created for drill-through REST endpoint and backend data layer.

## What Was Built

### Task 1: EXPORT-05 added to REQUIREMENTS.md

Added the missing EXPORT-05 requirement (xlsx/csv 500k threshold) that existed in the ROADMAP success criteria but was absent from REQUIREMENTS.md. Three targeted edits:
1. New bullet under `### Excel Export` section
2. New row in the Traceability table
3. Coverage count updated from 57 to 58 total requirements

### Task 2: RED test scaffolds created

**`tests/test_drill_through.py`** (5 tests, all FAILED — RED state):
- `test_endpoint_returns_rows` — GET /api/drill-through returns 200 + JSON with 'rows' key (DRILL-03)
- `test_pagination` — page=0 and page=1 return non-overlapping rows, each ≤ page_size (DRILL-04)
- `test_sort_and_filter` — sort_col/sort_dir and filter=USA params accepted and applied (DRILL-05)
- `test_coordinate_filters` — row_path=North|||USA + row_fields=region,country filters DuckDB correctly (DRILL-06)
- `test_total_rows_count_in_response` — response JSON includes 'total_rows' integer key (DRILL-04 extended)

All fail because `/api/drill-through` does not exist in `dash_presentation/app.py` yet (Dash returns HTML 200, not JSON).

**`tests/test_export.py`** (2 passed + 1 skipped):
- `test_get_drill_through_data_pagination` — controller.get_drill_through_data with limit/offset returns non-overlapping pages (PASSED — controller already supports this)
- `test_get_drill_through_data_coord_filters` — equality filter {'field': 'region', 'op': '=', 'value': 'North'} applied correctly (PASSED — controller already supports this)
- `test_get_drill_through_data_sort` — SKIPPED pending Plan 02 sort extension

## Test Results

```
FFFFF..s  [100%]
5 failed, 2 passed, 1 skipped in ~5s
```

Confirming RED state for drill-through endpoint tests, GREEN for controller tests, and SKIP for future sort test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `app.config['TESTING'] = True` → `app.server.config['TESTING'] = True`**
- **Found during:** Task 2, first test run
- **Issue:** Dash's `app.config` is a custom wrapper that only accepts Dash-recognized config keys. Setting `TESTING` on it raised `AttributeError: Invalid config key`. The Flask test client requires `TESTING` set on the underlying Flask `app.server.config`.
- **Fix:** Changed `app.config['TESTING'] = True` to `app.server.config['TESTING'] = True` in the fixture
- **Files modified:** `tests/test_drill_through.py`
- **Commit:** 5a5529a

## Decisions Made

1. **Flask server config pattern:** Set `app.server.config['TESTING']` (Flask) not `app.config['TESTING']` (Dash wrapper) — Dash rejects unknown config keys at its API boundary.
2. **test_export.py focus:** Tests validate the Python `ScalablePivotController.get_drill_through_data` layer directly (via `asyncio.run`). The JS export helpers (SheetJS `buildExportAoa`, CSV blob logic) are not testable in pytest — documented in file header.
3. **Sort test uses pytest.skip():** Explicit `pytest.skip()` (not `@pytest.mark.xfail`) so the test is clearly visible as future work, not an expected failure.
4. **test_sort_and_filter RED strategy:** The test asserts the endpoint accepts sort params without crashing, not actual sort order correctness. Sort order verification is deferred to Plan 02 when `get_drill_through_data` gains a sort parameter.

## Self-Check

- [x] `tests/test_drill_through.py` — verified exists and collected by pytest
- [x] `tests/test_export.py` — verified exists and collected by pytest
- [x] `.planning/REQUIREMENTS.md` — EXPORT-05 verified present in both section and traceability table
- [x] Commits `00fbd33` (Task 1) and `5a5529a` (Task 2) both exist in git log
