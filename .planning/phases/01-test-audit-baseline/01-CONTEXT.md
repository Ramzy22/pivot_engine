# Phase 1: Test Audit & Baseline - Context

**Gathered:** 2026-03-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Run every existing test (Python + JS), fix all failures, and produce a documented baseline showing what passes. No production feature code changes in this phase — only test fixes and the baseline artifact. Phase 2 cannot start if more than 50% of tests are failing after triage.

</domain>

<decisions>
## Implementation Decisions

### Failure Policy
- Goal: all tests must pass by end of Phase 1 — fix every failure found
- All tests are equally important (Python and JS treated the same)
- For each failing test, record: test name + error message (no full stack trace needed)
- Gate: if more than 50% of tests fail after initial audit, stop and triage before fixing — something systemic is wrong
- Fixing happens in Phase 1 itself, not deferred to later phases

### Test Scope
- Both Python backend and JS frontend tests are in scope
- Python: run `pytest` across `tests/` directory and all root-level `test_*.py` files
- JS: discover whether `npm test` / jest is configured — if yes, run it; if not, document that no JS tests exist
- Do NOT include debug scripts (debug_*.py, reproduce_*.py) — only files genuinely named test_* or in tests/

### Test Runner
- Python: `pytest` (standard invocation, not fail-fast)
- Python coverage: `pytest --cov=pivot_engine` via pytest-cov
- JS: `npm test` or `jest --coverage` if configured; skip if not

### Baseline Report
- Location: `.planning/phases/01-test-audit-baseline/TEST_BASELINE.md`
- Content: list of passing tests, list of failing tests with error message, total pass/fail/skip counts
- After all fixes land: re-run full suite and replace report with the clean final state
- Coverage: Python line/branch coverage per module documented; JS coverage only if jest is configured

### Coverage Threshold
- No minimum threshold for Phase 1 — just measure and document
- Coverage report is informational: flags modules with unexpectedly low coverage for awareness
- No gate on coverage percentage

### Claude's Discretion
- Exact pytest flags beyond `--cov` (e.g., `-v`, `--tb=short`)
- How to handle tests that require external services (skip or mock)
- Order in which to fix failures
- Whether to group fixes by module or fix one-by-one

</decisions>

<specifics>
## Specific Ideas

- The codebase has ~65 test files; many are in `tests/` but some are root-level `test_*.py` and `tests/` subdirectory files
- There are also `reproduce_*.py` and `debug_*.py` files at root — these are NOT tests and should be excluded
- The JS component (`dash_tanstack_pivot/`) has a webpack build; jest may or may not be wired up
- The baseline report is the Phase 1 deliverable — a future developer should be able to read it and know the exact health of the codebase at this point

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-test-audit-baseline*
*Context gathered: 2026-03-13*
