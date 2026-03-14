---
phase: 04-data-input-api
plan: "01"
subsystem: testing
tags: [pytest, pyarrow, pandas, polars, ibis, duckdb, tdd]

# Dependency graph
requires: []
provides:
  - "Failing test suite (8 test items) locking the observable API contract for DataInputNormalizer"
  - "MockController stub for load_data_from_arrow call recording"
affects:
  - 04-02-data-input-implementation

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD Red phase: tests import non-existent module to lock contract before writing production code"
    - "pytest.importorskip for optional dependencies (polars, ibis)"
    - "parametrize for multi-case error validation (API-06)"

key-files:
  created:
    - tests/test_data_input.py
  modified: []

key-decisions:
  - "Tests import from pivot_engine.pivot_engine.data_input — RED state is ModuleNotFoundError (expected)"
  - "MockController records load_data_from_arrow calls; asserts Arrow Table type and row count"
  - "test_connection_string uses a real temp DuckDB file (not in-memory) so connection_string URI resolves"
  - "test_auto_detection uses inspect.signature to verify single 3-param interface (API-05)"
  - "test_unsupported_type_error parametrized with int/list/bad-dict; asserts 'Supported types' in error message"

patterns-established:
  - "RED-state files collected by pytest --collect-only emit ImportError at module level; both outcomes (error or item list) are valid RED"
  - "temp file cleanup via try/finally + os.unlink for test isolation in test_connection_string"

requirements-completed: [API-01, API-02, API-03, API-04, API-05, API-06]

# Metrics
duration: 3min
completed: 2026-03-14
---

# Phase 4 Plan 01: DataInputNormalizer Test Contract (RED) Summary

**Eight failing pytest items locking the full DataInputNormalizer API contract (API-01 through API-06) via ModuleNotFoundError on pivot_engine.pivot_engine.data_input**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-14T10:02:56Z
- **Completed:** 2026-03-14T10:05:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `tests/test_data_input.py` with 6 test functions (8 parametrized items) covering all six API requirements
- Confirmed RED state: `pytest tests/test_data_input.py -x -q` fails with `ModuleNotFoundError: No module named 'pivot_engine.pivot_engine.data_input'`
- Established MockController stub that records `load_data_from_arrow` calls for later implementation verification

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing test suite for DataInputNormalizer (RED)** - `246f15d` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `tests/test_data_input.py` - Full TDD Red-phase test suite for DataInputNormalizer; 6 functions, 8 items, all failing with ImportError

## Decisions Made

- Used `inspect.signature` to verify single 3-param interface for API-05 (auto-detection contract)
- Used real temp DuckDB file for `test_connection_string` because `duckdb://` in-memory URI cannot be opened by a second process
- Used `pytest.importorskip` for polars and ibis so tests skip gracefully if optional deps absent

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 04-02 implements `pivot_engine/pivot_engine/data_input.py` to turn these 8 failing tests GREEN
- All test assertions are concrete and deterministic; implementation has a clear contract to satisfy

---
*Phase: 04-data-input-api*
*Completed: 2026-03-14*
