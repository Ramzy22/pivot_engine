---
phase: 01-test-audit-baseline
plan: 01
subsystem: testing
tags: [pytest, pytest-cov, httpx, sys.path, conftest]

# Dependency graph
requires: []
provides:
  - sys.path fix in conftest.py enabling pivot_engine.* imports from repo root
  - pytest-cov installed (coverage reporting unblocked)
  - httpx installed (starlette.testclient import unblocked)
  - 58 tests collectible from 4 test locations
affects: [01-test-audit-baseline, all downstream phases that run pytest]

# Tech tracking
tech-stack:
  added: [pytest-cov==7.0.0, httpx==0.28.1, coverage==7.13.4]
  patterns: [repo-root conftest.py with sys.path.insert for nested package layout]

key-files:
  created: [conftest.py]
  modified: []

key-decisions:
  - "sys.path.insert(0, pivot_engine/) in conftest.py chosen over editable install to avoid touching pyproject.toml (Phase 8 concern)"
  - "Empty git commit used for env-only Task 1 (pip install leaves no file to stage)"

patterns-established:
  - "Pattern 1: conftest.py at repo root is the canonical sys.path fix location for nested pivot_engine/pivot_engine/ layout"

requirements-completed: [QUAL-01, QUAL-02]

# Metrics
duration: 1min
completed: 2026-03-13
---

# Phase 1 Plan 01: Test Audit Baseline — Dependency & Path Fix Summary

**pytest-cov 7.0.0 and httpx 0.28.1 installed; repo-root conftest.py adds pivot_engine/ to sys.path enabling 58 tests to collect across 4 locations**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-13T16:47:05Z
- **Completed:** 2026-03-13T16:48:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Installed pytest-cov and httpx, unblocking `--cov` coverage flag and starlette.testclient imports
- Created `conftest.py` at repo root with sys.path fix for `pivot_engine/pivot_engine/` nested layout
- pytest now collects 58 tests across `tests/`, `test_expand_all_backend.py`, `test_filtering.py`, and `pivot_engine/tests/` — well above the 30-item threshold

## Task Commits

Each task was committed atomically:

1. **Task 1: Install missing packages** - `09bce31` (chore)
2. **Task 2: Create root conftest.py with sys.path fix** - `ff05bc5` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified
- `conftest.py` — sys.path.insert(0, pivot_engine/) so `from pivot_engine.controller import PivotController` resolves from repo root

## Decisions Made
- Used `sys.path.insert` in conftest.py rather than an editable install (`pip install -e`) to avoid modifying pyproject.toml, which is reserved for Phase 8
- Task 1 (pip install) produces no file changes; committed as empty git commit to preserve atomic task history

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- 5 collection errors remain after conftest.py fix: `test_advanced_planning.py` (structlog missing), `test_config_main.py`, `test_diff_engine_enhancements.py`, `test_features_impl.py`, `test_microservices.py` (pivot_microservices module missing). These are pre-existing issues outside this plan's scope; they will be addressed in plan 03 (import triage).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- pytest can now collect and the `--cov` flag is functional — plan 02 (baseline run & count) can proceed
- 5 collection errors remain due to missing modules; plan 03 will triage and fix import issues
- No blockers for plan 02

---
*Phase: 01-test-audit-baseline*
*Completed: 2026-03-13*

## Self-Check: PASSED

- conftest.py: FOUND
- 01-01-SUMMARY.md: FOUND
- Commit 09bce31 (Task 1): FOUND
- Commit ff05bc5 (Task 2): FOUND
