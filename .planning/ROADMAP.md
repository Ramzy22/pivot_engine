# Roadmap: DashTanstackPivot

## Overview

This is a brownfield project with a substantial existing codebase (~65 test files, 1500+ line React component, full Ibis query planning engine). The roadmap begins by establishing a verified baseline of what actually works, then systematically fixes known bugs before adding new capabilities. Every phase builds on a stable, confirmed foundation. The final phases harden the codebase and prepare it for open-source distribution via PyPI.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Test Audit & Baseline** - Run all 65 tests, document what passes, establish green baseline before any changes
- [ ] **Phase 2: Data Correctness Bugs** - Fix grand totals, column discovery, filter/sort state persistence
- [ ] **Phase 3: Virtual Scroll & UI Bugs** - Fix scroll sync, column header alignment, row group expansion, context menu
- [ ] **Phase 4: Data Input API** - Unify data prop to accept DataFrame, polars, Ibis, or connection string with auto-detection
- [ ] **Phase 5: Field Zone UI** - Complete drag-and-drop sidebar with four zones, aggregation selector, bidirectional Dash props
- [ ] **Phase 6: Drill-Through & Excel Export** - Cell drill-through modal with source rows, Excel .xlsx download of current view
- [ ] **Phase 7: Code Quality Refactor** - Split 1500-line component, add error boundaries, fix stale closures and filter duplication
- [ ] **Phase 8: Packaging, Docs & CI/CD** - PyPI publishing, README, examples, GitHub Actions

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
**Plans**: TBD

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
**Plans**: TBD

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
**Plans**: TBD

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
**Plans**: TBD

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
**Plans**: TBD

### Phase 6: Drill-Through & Excel Export
**Goal**: Users can investigate any aggregated cell by seeing its source rows in a modal, and can download the current pivot view as an Excel file
**Depends on**: Phase 5
**Requirements**: DRILL-01, DRILL-02, DRILL-03, DRILL-04, DRILL-05, EXPORT-01, EXPORT-02, EXPORT-03, EXPORT-04
**Success Criteria** (what must be TRUE):
  1. Clicking any aggregated cell opens a drill-through modal showing the underlying source rows for that cell
  2. The source rows are fetched server-side using the cell's exact pivot coordinate filters
  3. The drill-through modal paginates when the source row count exceeds one page
  4. The drill-through modal supports sorting any column and applying a basic text filter
  5. Clicking "Export to Excel" downloads a .xlsx file containing the current pivot view with multi-level headers, row hierarchy, grand totals, and subtotals
**Plans**: TBD

### Phase 7: Code Quality Refactor
**Goal**: The frontend codebase is maintainable — no file exceeds ~400 lines, crashes show error UI instead of blank screens, and shared logic is not duplicated
**Depends on**: Phase 6
**Requirements**: CODE-01, CODE-02, CODE-03, CODE-04, QUAL-03, QUAL-04
**Success Criteria** (what must be TRUE):
  1. The main React component file is under 400 lines; logic is split into focused sub-components and hooks
  2. A React error boundary wraps the table — a component crash displays an error message, not a blank screen
  3. Filter logic exists in exactly one place (shared hook or utility); no duplicate implementations across components
  4. The duplicate `run_pivot_arrow()` method in controller.py is removed with no regression in existing tests
  5. Column name sanitization in the backend uses Ibis parameter binding, eliminating SQL injection risk
**Plans**: TBD

### Phase 8: Packaging, Docs & CI/CD
**Goal**: Any Python developer can `pip install dash-tanstack-pivot`, find clear documentation, and the project has automated testing and publishing
**Depends on**: Phase 7
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
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Test Audit & Baseline | 0/TBD | Not started | - |
| 2. Data Correctness Bugs | 0/TBD | Not started | - |
| 3. Virtual Scroll & UI Bugs | 0/TBD | Not started | - |
| 4. Data Input API | 0/TBD | Not started | - |
| 5. Field Zone UI | 0/TBD | Not started | - |
| 6. Drill-Through & Excel Export | 0/TBD | Not started | - |
| 7. Code Quality Refactor | 0/TBD | Not started | - |
| 8. Packaging, Docs & CI/CD | 0/TBD | Not started | - |
