# Roadmap: DashTanstackPivot

## Overview

This is a brownfield project with a substantial existing codebase (~65 test files, 1500+ line React component, full Ibis query planning engine). The roadmap begins by establishing a verified baseline of what actually works, then systematically fixes known bugs before adding new capabilities. Every phase builds on a stable, confirmed foundation. The final phases harden the codebase and prepare it for open-source distribution via PyPI.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Test Audit & Baseline** - Run all 65 tests, document what passes, establish green baseline before any changes (completed 2026-03-13)
- [x] **Phase 2: Data Correctness Bugs** - Fix grand totals, column discovery, filter/sort state persistence (completed 2026-03-13)
- [x] **Phase 3: Virtual Scroll & UI Bugs** - Fix scroll sync, column header alignment, row group expansion, context menu (completed 2026-03-13)
- [x] **Phase 3.1: Debug Instrumentation + Grand Total Fix** [INSERTED] - Add request/response debug logging to adapter, diagnose and fix duplicate grand total row, add regression (completed 2026-03-13)
- [x] **Phase 3.2: Test Harness Hardening** [INSERTED] - Fix unguarded app import in test_frontend_contract.py, remove dead except block in app.py (completed 2026-03-14)
- [x] **Phase 4: Data Input API** - Unify data prop to accept DataFrame, polars, Ibis, or connection string with auto-detection (completed 2026-03-14)
- [x] **Phase 5: Field Zone UI** - Complete drag-and-drop sidebar with four zones, aggregation selector, bidirectional Dash props (completed 2026-03-15)
- [ ] **Phase 6: Drill-Through & Excel Export** - Cell drill-through modal (server-side via REST endpoint, not Dash callbacks) with pagination, and Excel .xlsx download
- [x] **Phase 7: Column Display & UI States** - Review and correct all column states (pinned, sorted, hidden, resized), their visual indicators, and UI defaults (completed 2026-03-15)
- [ ] **Phase 8: Code Quality Refactor** - Split 4,338-line component, add error boundaries, fix stale closures and filter duplication
- [ ] **Phase 9: Packaging, Docs & CI/CD** - PyPI publishing, README, examples, GitHub Actions

## Phase Details

### Phase 1: Test Audit & Baseline
**Goal**: The team knows exactly which tests pass, which fail, and why — before a single line of production code is changed
**Depends on**: Nothing (first phase)
**Requirements**: QUAL-01, QUAL-02
**Success Criteria** (what must be TRUE):
  1. All 65 existing test files have been executed and results are recorded
  2. A baseline report lists every passing test, every failing test, and the failure reason
  3. Test coverage percentage is measured and documented
  4. No production code has been modified during this phase
**Plans**: 4 plans

Plans:
- [ ] 01-01-PLAN.md — Install pytest-cov/httpx and create root conftest.py (path fix)
- [ ] 01-02-PLAN.md — Run full pytest audit + standalone scripts, evaluate 50% gate
- [ ] 01-03-PLAN.md — Fix 4 collection errors with importorskip guards
- [ ] 01-04-PLAN.md — Fix remaining failures, generate coverage, write TEST_BASELINE.md

### Phase 2: Data Correctness Bugs
**Goal**: Aggregated values in the pivot table are always correct and stable — grand totals, column sets, filter state, and sort state all work reliably
**Depends on**: Phase 1
**Requirements**: BUG-01, BUG-02, BUG-03, BUG-04, BUG-05, BUG-06
**Success Criteria** (what must be TRUE):
  1. Grand total row displays correct aggregated values for sum, avg, count, min, and max measures
  2. Grand total row remains visible and stable during scroll, filter changes, and data refreshes
  3. After any filter change or page refresh, all pivot columns that should exist are present in the header
  4. Applied filters remain active and correct when the user expands rows, changes sort order, or scrolls
  5. Sort order set by the user is applied server-side and survives data refreshes
**Plans**: 4 plans

Plans:
- [x] 02-01-PLAN.md - Add explicit regression coverage for totals, dynamic columns, and state persistence
- [x] 02-02-PLAN.md - Fix totals semantics and server-side sort behavior
- [x] 02-03-PLAN.md - Fix dynamic column discovery completeness and refresh consistency
- [x] 02-04-PLAN.md - Fix hierarchy/virtual-scroll persistence and run full verification

### Phase 3: Virtual Scroll & UI Bugs
**Goal**: The table renders correctly at all times — scrolled rows match server data, headers align with cells, row groups expand accurately, and menus stay on screen
**Depends on**: Phase 2
**Requirements**: BUG-07, BUG-08, BUG-09, BUG-10, BUG-11, BUG-12, BUG-13
**Success Criteria** (what must be TRUE):
  1. Scrolling through a large dataset shows no blank rows, no stale data, and no row duplication
  2. Changing filter or sort immediately invalidates the scroll cache so no stale rows appear
  3. Multi-level column headers visually span exactly the width of their child columns at every nesting depth
  4. Expanding a row group shows the correct child rows at the correct indentation; collapsing does not shift sibling rows
  5. Right-clicking any cell opens a context menu that is fully visible within the browser viewport
**Plans**: 4 plans

Plans:
- [x] 03-01-PLAN.md - Add regressions for virtual-scroll continuity and hierarchy stability
- [x] 03-02-PLAN.md - Fix backend hierarchy semantics, visible-row counts, and expand/collapse behavior
- [x] 03-03-PLAN.md - Fix frontend block-cache invalidation and stale-row synchronization
- [x] 03-04-PLAN.md - Fix grouped-header geometry, context-menu placement, and run full verification

### Phase 3.1: Debug Instrumentation + Grand Total Fix [INSERTED]
**Goal**: Diagnose and fix the duplicate grand total row (visible at top overlapping a data row AND at bottom) by adding toggleable request/response logging to the adapter layer, then fixing the root cause
**Depends on**: Phase 3
**Requirements**: BUG-14
**Success Criteria** (what must be TRUE):
  1. A debug flag enables structured logging of every incoming frontend request payload and every outgoing backend response (row data, total row presence, data slice bounds)
  2. Exactly one grand total row appears in every backend response that includes totals — never zero, never two
  3. The pivot table displays the grand total row exactly once at the correct position
  4. A regression test asserts single grand total row presence per response
**Plans**: 4 plans

Plans:
- [ ] 03.1-01-PLAN.md — Write failing regression tests for BUG-14 (grand total duplication)
- [ ] 03.1-02-PLAN.md — Add debug logging to adapter and fix grand total dedup in controller + adapter

### Phase 3.2: Test Harness Hardening [INSERTED]
**Goal**: The test suite collects without side effects and the app.py callback has no dead error-handling code
**Depends on**: Phase 3.1
**Requirements**: QUAL-05, QUAL-06
**Success Criteria** (what must be TRUE):
  1. `tests/test_frontend_contract.py` collection does not execute any data-generation or DuckDB initialization code
  2. `dash_presentation/app.py` `update_pivot_table` callback has exactly one `except Exception` block
**Plans**: 4 plans

Plans:
- [x] 03.2-01-PLAN.md - Make dash app/bootstrap import-safe and move app smoke import behind a test boundary
- [x] 03.2-02-PLAN.md - Remove dead callback except block and run focused harness verification

### Phase 4: Data Input API
**Goal**: A Python developer can pass a pandas DataFrame, polars DataFrame, Ibis table, or connection string to the same `data` prop and the component handles it automatically
**Depends on**: Phase 3
**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06
**Success Criteria** (what must be TRUE):
  1. Passing a pandas DataFrame to `data` renders a working pivot table
  2. Passing a polars DataFrame to `data` renders a working pivot table
  3. Passing an Ibis table expression to `data` renders a working pivot table
  4. Passing a connection string and table name to `data` renders a working pivot table
  5. Passing an unsupported type (e.g., a plain dict) shows a clear, actionable error message instead of crashing
**Plans**: 3 plans

Plans:
- [ ] 04-01-PLAN.md — Write failing test suite for DataInputNormalizer (TDD Red — API-01 through API-06)
- [ ] 04-02-PLAN.md — Implement DataInputNormalizer + DataInputError in data_input.py (TDD Green)
- [ ] 04-03-PLAN.md — Wire load_data() into TanStackPivotAdapter and verify full suite

### Phase 5: Field Zone UI
**Goal**: Users can interactively reconfigure the pivot table by dragging fields between four zones — Rows, Columns, Values, Filters — and the configuration round-trips to Python
**Depends on**: Phase 4
**Requirements**: FIELD-01, FIELD-02, FIELD-03, FIELD-04, FIELD-05, FIELD-06
**Success Criteria** (what must be TRUE):
  1. The sidebar shows four labeled drop zones: Rows, Columns, Values, Filters
  2. A user can drag any available field into any of the four zones and the pivot updates immediately
  3. Dropping a field into the Values zone presents an aggregation type selector (sum/avg/count/min/max)
  4. Removing a field from any zone updates the pivot immediately without a full page reload
  5. The Python callback receives the current field zone configuration as a Dash prop update
  6. Setting field zone props from Python pre-populates the sidebar zones on initial render
**Plans**: 4 plans

Plans:
- [x] 05-01-PLAN.md — Add Filters drop zone and min/max aggregation options to sidebar
- [x] 05-02-PLAN.md — Harden drag-drop with validation, duplicate prevention, empty state, and regression tests
- [x] 05-03-PLAN.md — Fix sidebar filter-chip popover anchoring and harden viewport clamping
- [ ] 05-04-PLAN.md — Pending

### Phase 6: Drill-Through & Excel Export
**Goal**: Users can investigate any aggregated cell by seeing its source rows in a modal, and can download the current pivot view as an Excel file (or CSV when the row count exceeds the safe Excel threshold)
**Depends on**: Phase 5
**Requirements**: DRILL-01, DRILL-02, DRILL-03, DRILL-04, DRILL-05, DRILL-06, EXPORT-01, EXPORT-02, EXPORT-03, EXPORT-04, EXPORT-05
**Success Criteria** (what must be TRUE):
  1. Clicking any aggregated cell opens a drill-through modal showing the underlying source rows for that cell
  2. Drill-through data is fetched via a dedicated REST endpoint (`/api/drill-through`) called directly from React — Dash callbacks are not used, so any row count is supported without serialization limits
  3. The REST endpoint applies the cell's exact pivot coordinate filters server-side (DuckDB) and returns paginated results
  4. The drill-through modal paginates server-side — each page request hits the endpoint with `?page=N&page_size=500`; no full dataset is ever sent to the browser
  5. The drill-through modal supports server-side column sorting and a text filter, both passed as query params to the endpoint
  6. Clicking "Export" downloads a .xlsx file when the visible pivot row count is ≤ 500,000 rows; when it exceeds 500,000 rows the button downloads a .csv file instead, and the UI label/icon updates to reflect which format will be produced
**Plans**: 4 plans

Plans:
- [ ] 06-01-PLAN.md — Add EXPORT-05 to REQUIREMENTS.md and write RED test scaffolds (test_drill_through.py, test_export.py)
- [ ] 06-02-PLAN.md — Add /api/drill-through Flask route + extend get_drill_through_data with sort/filter/total_rows
- [ ] 06-03-PLAN.md — Upgrade exportExcel to exportPivot with aoa_to_sheet, hierarchy indent, totals, and 500k csv/xlsx branch
- [ ] 06-04-PLAN.md — Wire cell-click drill trigger, build DrillThroughModal sub-component, add drillEndpoint prop

### Phase 7: Column Display & UI States
**Goal**: Every column state — pinned, sorted, hidden, resized — renders with correct visual indicators and sensible defaults; combined states work without visual glitches
**Depends on**: Phase 6
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06
**Success Criteria** (what must be TRUE):
  1. Pinned columns display a shadow or border separator and remain fixed during horizontal scroll
  2. Sorted columns show a visible ascending/descending indicator on the column header; the active sort column is visually distinct
  3. Hidden columns can be toggled via the column visibility panel; their visibility state persists across filter changes and data refreshes
  4. Column resize handles appear on header hover; resized widths persist during scroll and after data refresh
  5. Combined column states (e.g., a column that is simultaneously pinned, sorted, and resized) display without visual conflict or layout breakage
  6. Default column widths, row heights, and header heights are visually balanced and consistent across all data densities
**Plans**: TBD

### Phase 8: Code Quality Refactor
**Goal**: The frontend codebase is maintainable — no file exceeds ~400 lines, crashes show error UI instead of blank screens, and shared logic is not duplicated
**Depends on**: Phase 7
**Requirements**: CODE-01, CODE-02, CODE-03, CODE-04, QUAL-03, QUAL-04
**Success Criteria** (what must be TRUE):
  1. The main React component file is under 400 lines; logic is split into focused sub-components and hooks
  2. A React error boundary wraps the table — a component crash displays an error message, not a blank screen
  3. Filter logic exists in exactly one place (shared hook or utility); no duplicate implementations across components
  4. The duplicate `run_pivot_arrow()` method in controller.py is removed with no regression in existing tests
  5. Column name sanitization in the backend uses Ibis parameter binding, eliminating SQL injection risk
**Plans**: 4 plans

Plans:
- [ ] 08-01-PLAN.md — Remove duplicate run_pivot_arrow + parameterized UPDATE (QUAL-03, QUAL-04)
- [ ] 08-02-PLAN.md — Create PivotErrorBoundary, usePersistence, useFilteredData hooks (CODE-02, CODE-03, CODE-04)
- [ ] 08-03-PLAN.md — Extract PivotAppBar and SidebarPanel sub-components (CODE-01 partial)
- [ ] 08-04-PLAN.md — Extract useColumnDefs, useRenderHelpers, PivotTableBody (CODE-01 completion)

### Phase 9: Packaging, Docs & CI/CD
**Goal**: Any Python developer can `pip install dash-tanstack-pivot`, find clear documentation, and the project has automated testing and publishing
**Depends on**: Phase 8
**Requirements**: PKG-01, PKG-02, PKG-03, PKG-04, PKG-05, DOC-01, DOC-02, DOC-03, DOC-04, CI-01, CI-02, CI-03
**Success Criteria** (what must be TRUE):
  1. `pip install dash-tanstack-pivot` installs the component cleanly; `import dash_tanstack_pivot` succeeds with no missing dependency errors
  2. `pip install dash-tanstack-pivot[redis]` installs the Redis cache extra without errors
  3. The README contains a working 10-line minimal example and all Python props are documented with types and defaults
  4. At least three example Dash apps (basic DataFrame, hierarchical, SQL-connected) exist and run without errors
  5. A GitHub Actions workflow runs Python tests and JS build on every push, and publishes to PyPI on version tag push
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 3.1 → 3.2 → 4 → 5 → 6 → 7 → 8 → 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Test Audit & Baseline | 4/4 | Complete | 2026-03-13 |
| 2. Data Correctness Bugs | 4/4 | Complete | 2026-03-13 |
| 3. Virtual Scroll & UI Bugs | 4/4 | Complete | 2026-03-13 |
| 3.1. Debug Instrumentation + Grand Total Fix | 2/2 | Complete | 2026-03-13 |
| 3.2. Test Harness Hardening | 2/2 | Complete | 2026-03-14 |
| 4. Data Input API | 3/3 | Complete | 2026-03-14 |
| 5. Field Zone UI | 4/4 | Complete | 2026-03-15 |
| 6. Drill-Through & Excel Export | 3/4 | In Progress|  |
| 7. Column Display & UI States | 3/3 | Complete | 2026-03-15 |
| 8. Code Quality Refactor | 1/4 | In Progress|  |
| 9. Packaging, Docs & CI/CD | 0/TBD | Not started | - |

