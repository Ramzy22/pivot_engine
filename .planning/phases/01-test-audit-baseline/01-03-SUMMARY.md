---
phase: 01-test-audit-baseline
plan: 03
subsystem: testing
tags: [pytest, importorskip, collection-errors, test-triage]

# Dependency graph
requires:
  - phase: 01-test-audit-baseline plan 02
    provides: audit_raw.txt confirming 5 collection errors with exact missing symbols
provides:
  - 5 test files with pytest.importorskip/pytest.skip guards preventing collection errors
  - Full suite collect-only now returns 63 tests with 0 ERROR lines (down from 5 errors)
affects: [01-04-fix-baseline, all downstream phases that run pytest]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "pytest.importorskip(module) for missing module-level imports"
    - "try/except ImportError + pytest.skip(allow_module_level=True) for missing class-level imports"

key-files:
  created: []
  modified:
    - pivot_engine/tests/test_advanced_planning.py
    - pivot_engine/tests/test_config_main.py
    - pivot_engine/tests/test_diff_engine_enhancements.py
    - pivot_engine/tests/test_microservices.py
    - pivot_engine/tests/test_features_impl.py

key-decisions:
  - "Used importorskip on structlog (not ScalablePivotApplication) for test_config_main.py — the actual root cause from audit_raw.txt was a missing structlog package in the import chain"
  - "Fixed test_features_impl.py even though plan 03 claimed it was already fixed by httpx — audit_raw.txt confirmed it still had a structlog collection error"

requirements-completed: [QUAL-01]

# Metrics
duration: 2min
completed: 2026-03-13
---

# Phase 1 Plan 03: Test Audit Baseline — Import Triage Summary

**5 collection errors converted to module-level skips; full suite now collects 63 tests with 0 ERROR lines (down from 5 errors)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-13T16:55:01Z
- **Completed:** 2026-03-13T16:56:29Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `pytest.importorskip` or `pytest.skip(allow_module_level=True)` guards to all 5 files that previously crashed pytest collection
- Full suite `--collect-only` now returns exactly 63 tests with zero "ERROR collecting" lines
- Collection error count: 5 before → 0 after

## Before / After

| Metric | Before (Plan 02) | After (Plan 03) |
|--------|-----------------|-----------------|
| Collection errors | 5 | 0 |
| Tests collected | 63 | 63 |
| Tests skipped at module level | 0 | 5 files |

## Guards Applied

| File | Guard Type | Missing Symbol |
|------|-----------|----------------|
| `test_advanced_planning.py` | `importorskip("pivot_engine.planner.sql_planner")` | module missing |
| `test_config_main.py` | `importorskip("structlog")` | package missing (transitive via main.py chain) |
| `test_diff_engine_enhancements.py` | `try/except ImportError + pytest.skip(allow_module_level=True)` | `MultiDimensionalTilePlanner` class missing |
| `test_microservices.py` | `importorskip("pivot_engine.pivot_microservices")` | package missing |
| `test_features_impl.py` | `importorskip("structlog")` | package missing (transitive via complete_rest_api chain) |

## Task Commits

1. **Task 1: Add importorskip guards to 5 collection-error files** - `4c3602b` (fix)
2. **Task 2: Verify full suite collects cleanly** - No commit (verification only; zero files modified)

## Decisions Made

- Used `importorskip("structlog")` for `test_config_main.py` rather than the plan's suggested `try/except` on `ScalablePivotApplication` — the actual root cause (from `audit_raw.txt`) was a missing `structlog` package, not a missing class. Using `importorskip` on the actual missing module is simpler and more accurate.
- Applied guard to `test_features_impl.py` (5th file) even though plan 03 claimed it was already fixed by `httpx` install in plan 01. Audit raw output confirmed the file still had a collection error via `structlog` chain.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_features_impl.py which plan claimed was already resolved**

- **Found during:** Task 1 (inspecting audit_raw.txt)
- **Issue:** Plan 03 stated "test_features_impl.py was the 5th collection error — it is already fixed by the httpx install in plan 01." However `audit_raw.txt` shows it still errors at collection because `structlog` is missing (not `httpx`). The httpx install in plan 01 fixed a different error path; the structlog chain was always the blocking import.
- **Fix:** Added `pytest.importorskip("structlog")` at the top of `test_features_impl.py`
- **Files modified:** `pivot_engine/tests/test_features_impl.py`
- **Committed in:** `4c3602b`

**2. [Rule 1 - Bug] Used importorskip("structlog") for test_config_main.py instead of try/except on ScalablePivotApplication**

- **Found during:** Task 1 (reading audit_raw.txt error trace)
- **Issue:** Plan specified `try/except ImportError` on `ScalablePivotApplication` — but audit_raw.txt shows the ImportError is on `structlog` (line 4 of observability.py), not on `ScalablePivotApplication` itself. A try/except on `ScalablePivotApplication` would have caught a different error level and been less precise.
- **Fix:** Used `pytest.importorskip("structlog")` which addresses the actual root cause and is simpler
- **Files modified:** `pivot_engine/tests/test_config_main.py`
- **Committed in:** `4c3602b`

---

**Total deviations:** 2 auto-fixed (both Rule 1 - actual root cause differed from plan's description)

## Issues Encountered

None — all 5 files collected as skipped on first attempt.

## User Setup Required

None.

## Next Phase Readiness

- plan 04 (fix-baseline) can proceed: full suite collects 63 tests with 0 errors
- 4 failing tests remain from plan 02 baseline (unchanged by this plan — no test logic was modified)
- No blockers for plan 04

---
*Phase: 01-test-audit-baseline*
*Completed: 2026-03-13*

## Self-Check: PASSED
