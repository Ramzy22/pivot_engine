---
phase: 04-data-input-api
plan: "02"
subsystem: api
tags: [pyarrow, pandas, polars, ibis, duckdb, tdd, data-input]

# Dependency graph
requires:
  - phase: 04-data-input-api plan 01
    provides: "Failing test suite (8 items) locking DataInputNormalizer API contract"
provides:
  - "pivot_engine/pivot_engine/data_input.py — DataInputError, DataInputNormalizer, normalize_data_input"
  - "All 8 test items GREEN (API-01 through API-06)"
affects:
  - 05-dash-component
  - any phase that wires data= prop to pivot engine

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy type detection: module name prefix + attr check avoids top-level optional imports"
    - "preserve_index=False on pa.Table.from_pandas — prevents __index_level_0__ schema leakage"
    - "rechunk().to_arrow() for polars — defensive against multi-chunk DataFrames"
    - "IbisBackend connection_string path wraps ModuleNotFoundError as DataInputError with pip hint"

key-files:
  created:
    - pivot_engine/pivot_engine/data_input.py
  modified:
    - tests/test_data_input.py

key-decisions:
  - "Lazy type detection via module name + attr avoids forcing pandas/polars imports at module load"
  - "DataInputError subclasses TypeError so existing isinstance(e, TypeError) guards still catch it"
  - "Auto-fixed NamedTemporaryFile Windows bug: deleted empty placeholder before duckdb.connect()"

patterns-established:
  - "Lazy optional-dep detection: type(x).__module__.startswith('pandas') instead of try/import"
  - "Connection dict path routes through IbisBackend so all DB drivers are handled uniformly"

requirements-completed: [API-01, API-02, API-03, API-04, API-05, API-06]

# Metrics
duration: 2min
completed: 2026-03-14
---

# Phase 4 Plan 02: DataInputNormalizer Implementation (GREEN) Summary

**DataInputNormalizer with lazy pandas/polars/ibis/connection-dict/arrow detection, zero top-level optional imports, and all 8 TDD tests GREEN**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-14T12:45:24Z
- **Completed:** 2026-03-14T12:46:45Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Created `pivot_engine/pivot_engine/data_input.py` exporting `DataInputError`, `DataInputNormalizer`, `normalize_data_input`
- All 8 pytest items GREEN: pandas, polars, ibis, connection-string, auto-detection signature, and 3 parametrized unsupported-type cases
- Lazy type detection prevents import of optional pandas/polars/ibis at module load time
- Auto-fixed Windows-specific temp-file bug in `test_connection_string` blocking CI

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement data_input.py (GREEN)** - `6f9901f` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `pivot_engine/pivot_engine/data_input.py` - DataInputNormalizer, DataInputError, normalize_data_input; lazy type detection for all four input types
- `tests/test_data_input.py` - Auto-fixed NamedTemporaryFile Windows platform bug in test_connection_string

## Decisions Made

- Used `type(data).__module__.startswith("pandas")` detection to avoid top-level pandas/polars imports — matches plan spec exactly
- `DataInputError` subclasses `TypeError` so existing `isinstance(e, TypeError)` guards continue to work
- Connection dict path routes through `IbisBackend(connection_uri=...)` for uniform driver handling

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed NamedTemporaryFile Windows platform bug in test_connection_string**
- **Found during:** Task 1 (running `pytest tests/test_data_input.py -x -q` after implementation)
- **Issue:** `tempfile.NamedTemporaryFile(delete=False)` creates an empty file; DuckDB on Windows raises `IOException: not a valid DuckDB database file` when trying to connect to an existing non-DuckDB file
- **Fix:** Added `os.unlink(db_path)` after the context manager exits and before `duckdb.connect(db_path)` so DuckDB creates a fresh database at that path
- **Files modified:** `tests/test_data_input.py`
- **Verification:** `pytest tests/test_data_input.py -x -q` exits 0 with 8 passed
- **Committed in:** `6f9901f` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 platform bug)
**Impact on plan:** Required for tests to pass on Windows. The fix is minimal, correct on all platforms, and does not alter test intent.

## Issues Encountered

None beyond the auto-fixed Windows platform bug above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `normalize_data_input` is production-ready and can be wired into the Dash component's `data=` prop callback
- `DataInputNormalizer` class available for adapter patterns that need a controller-bound instance
- All six API requirements (API-01 through API-06) have GREEN tests

---
*Phase: 04-data-input-api*
*Completed: 2026-03-14*
