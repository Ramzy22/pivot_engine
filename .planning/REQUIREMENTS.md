# Requirements: DashTanstackPivot

**Defined:** 2026-03-13
**Core Value:** A Python developer adds an enterprise-grade pivot table to any Dash app in under 10 lines of code — no JS knowledge, no database config, no performance tuning required.

---

## v1 Requirements

### Quality Baseline

- [x] **QUAL-01**: All existing tests pass before any new development begins (establish green baseline)
- [x] **QUAL-02**: Test coverage report generated and baseline documented
- [ ] **QUAL-03**: Duplicate method `run_pivot_arrow()` in controller.py removed
- [ ] **QUAL-04**: Column name sanitization secured against SQL injection
- [x] **QUAL-05**: `tests/test_frontend_contract.py` collection executes no data-generation side effects (lazy fixture, no module-scope DuckDB init)
- [x] **QUAL-06**: `dash_presentation/app.py` `update_pivot_table` has exactly one `except Exception` block (dead block removed)

### Bug Fixes — Data Correctness

- [x] **BUG-01**: Grand total rows display correct aggregated values for all measure types
- [x] **BUG-02**: Grand total rows do not disappear or flicker on scroll/filter changes
- [x] **BUG-03**: Pivot column discovery returns complete, non-sparse column set after data changes
- [x] **BUG-04**: Pivot column discovery is consistent across page refreshes and filter changes
- [x] **BUG-05**: Filter state persists across row expansion, sort changes, and viewport scroll
- [x] **BUG-06**: Sort state applies server-side and does not reset on data refresh

### Bug Fixes — Debug & Grand Total

- [x] **BUG-14**: Grand total row appears exactly once in every response — never duplicated at top and bottom simultaneously

### Bug Fixes — Virtual Scroll & UI

- [x] **BUG-07**: Virtual scroll rows stay synchronized with server data (no stale rows, no blank areas)
- [x] **BUG-08**: Block-based cache invalidates correctly on filter/sort/spec changes
- [x] **BUG-09**: Multi-level column headers align correctly over their data cells at all nesting depths
- [x] **BUG-10**: Column header span widths match the total width of child columns
- [x] **BUG-11**: Row group expansion shows correct children with correct indentation levels
- [x] **BUG-12**: Expanding/collapsing rows does not corrupt sibling row positions
- [x] **BUG-13**: Context menu renders within viewport bounds (no off-screen rendering)

### Data Input API

- [x] **API-01**: Component accepts a pandas DataFrame as `data` prop
- [x] **API-02**: Component accepts a polars DataFrame as `data` prop
- [x] **API-03**: Component accepts an Ibis table expression as `data` prop
- [x] **API-04**: Component accepts a connection string + table name as `data` prop
- [x] **API-05**: Input type is auto-detected at runtime — same prop interface for all types
- [x] **API-06**: Meaningful error message shown when unsupported input type passed

### Field Zone UI

- [x] **FIELD-01**: Sidebar contains four drop zones: Rows, Columns, Values, Filters
- [x] **FIELD-02**: User can drag any available field into any drop zone
- [x] **FIELD-03**: Dropping a field into Values zone shows aggregation selector (sum/avg/count/min/max)
- [x] **FIELD-04**: Removing a field from a zone updates the pivot immediately
- [x] **FIELD-05**: Field zone state is reflected back to Python via Dash props (bidirectional)
- [x] **FIELD-06**: Field zone initial state can be set from Python props

### Drill-Through

- [x] **DRILL-01**: Clicking any aggregated cell triggers drill-through action
- [x] **DRILL-02**: Drill-through displays a modal showing the source rows for that cell
- [x] **DRILL-03**: Source rows are fetched via a dedicated REST endpoint (`/api/drill-through`) called directly from React — no Dash callback involved, no prop serialization limit
- [x] **DRILL-04**: Drill-through modal paginates server-side (`?page=N&page_size=500`); full dataset is never sent to the browser
- [x] **DRILL-05**: Drill-through modal supports server-side column sorting and text filter, both passed as query params
- [x] **DRILL-06**: The `/api/drill-through` REST endpoint applies the cell's exact pivot coordinate filters using DuckDB and returns only the requested page

### Excel Export

- [x] **EXPORT-01**: "Export to Excel" button downloads the current pivot view as .xlsx
- [x] **EXPORT-02**: Exported file preserves multi-level column headers
- [x] **EXPORT-03**: Exported file preserves row hierarchy (indentation or grouping)
- [x] **EXPORT-04**: Grand totals and subtotals are included in export
- [x] **EXPORT-05**: "Export" button downloads .xlsx when pivot row count is ≤ 500,000; downloads .csv above that threshold; button label and icon update to reflect the format that will be produced

### Column Display & UI States

- [ ] **UI-01**: Pinned columns display a shadow or border separator and remain fixed during horizontal scroll
- [ ] **UI-02**: Sorted columns show a visible ascending/descending indicator on the header; the active sort column is visually distinct
- [x] **UI-03**: Hidden columns can be toggled via the column visibility panel; visibility state persists across filter changes and data refreshes
- [x] **UI-04**: Column resize handles appear on header hover; resized widths persist during scroll and after data refresh
- [ ] **UI-05**: Combined column states (pinned + sorted + resized simultaneously) display without visual conflict or layout breakage
- [ ] **UI-06**: Default column widths, row heights, and header heights are visually balanced and consistent across all data densities

### Code Quality & Architecture

- [ ] **CODE-01**: Main React component (`DashTanstackPivot.react.js`) split into focused sub-components (< 400 lines each)
- [ ] **CODE-02**: React error boundary wraps the table — component crash shows error UI instead of blank screen
- [ ] **CODE-03**: All `useEffect` dependencies are correct (no stale closures causing silent bugs)
- [ ] **CODE-04**: Filter logic is not duplicated across components — shared utility or hook

### Packaging & Distribution

- [ ] **PKG-01**: `pip install dash-tanstack-pivot` installs the component with zero additional config
- [ ] **PKG-02**: Optional extras: `pip install dash-tanstack-pivot[redis]` for Redis cache
- [ ] **PKG-03**: Package published to PyPI with semantic versioning
- [ ] **PKG-04**: `dash_tanstack_pivot` Python module imports cleanly with no missing dependency errors
- [ ] **PKG-05**: npm build (`npm run build`) produces a single minified JS bundle correctly

### Documentation & Examples

- [ ] **DOC-01**: README shows a minimal working example (DataFrame → pivot table in 10 lines)
- [ ] **DOC-02**: All Python props documented with types, defaults, and descriptions
- [ ] **DOC-03**: At least 3 example Dash apps covering: basic, hierarchical, and SQL-connected use cases
- [ ] **DOC-04**: CHANGELOG.md initialized

### CI/CD

- [ ] **CI-01**: GitHub Actions runs Python tests on every push
- [ ] **CI-02**: GitHub Actions runs JS build on every push
- [ ] **CI-03**: GitHub Actions auto-publishes to PyPI on version tag push

---

## v2 Requirements

### Advanced Features

- **CALC-01**: User can define calculated fields (e.g., `Margin = Revenue - Cost`)
- **CALC-02**: Calculated fields appear in Values zone and are computed server-side
- **FMT-01**: Number formatting templates per measure (currency, percentage, K/M/B)
- **FMT-02**: Conditional cell formatting rules configurable from Python
- **REAL-01**: Real-time data updates via CDC (Change Data Capture) integration
- **CHART-01**: Optional embedded spark charts in cells

### Performance

- **PERF-01**: Apache Arrow Flight transport for zero-copy large data transfers
- **PERF-02**: Distributed cache via Redis for multi-worker Dash deployments

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| OLAP / MDX calculated members | Niche enterprise requirement, deferred |
| Cross-filtering between multiple pivots | v2 feature |
| Mobile responsive layout | Desktop-first; mobile deferred |
| Goal seek / what-if analysis | Excel Solver equivalent, out of scope |
| Real-time WebSocket streaming | High complexity; v2 |
| Embedded pivot charts | Separate concern; v2 |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| QUAL-01 | Phase 1 - Test Audit & Baseline | Complete |
| QUAL-02 | Phase 1 - Test Audit & Baseline | Complete |
| BUG-01 | Phase 2 - Data Correctness Bugs | Complete |
| BUG-02 | Phase 2 - Data Correctness Bugs | Complete |
| BUG-03 | Phase 2 - Data Correctness Bugs | Complete |
| BUG-04 | Phase 2 - Data Correctness Bugs | Complete |
| BUG-05 | Phase 2 - Data Correctness Bugs | Complete |
| BUG-06 | Phase 2 - Data Correctness Bugs | Complete |
| BUG-07 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-08 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-09 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-10 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-11 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-12 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-13 | Phase 3 - Virtual Scroll & UI Bugs | Complete |
| BUG-14 | Phase 3.1 - Debug Instrumentation + Grand Total Fix | Complete |
| QUAL-05 | Phase 3.2 - Test Harness Hardening | Complete |
| QUAL-06 | Phase 3.2 - Test Harness Hardening | Complete |
| API-01 | Phase 4 - Data Input API | Complete |
| API-02 | Phase 4 - Data Input API | Complete |
| API-03 | Phase 4 - Data Input API | Complete |
| API-04 | Phase 4 - Data Input API | Complete |
| API-05 | Phase 4 - Data Input API | Complete |
| API-06 | Phase 4 - Data Input API | Complete |
| FIELD-01 | Phase 5 - Field Zone UI | Complete |
| FIELD-02 | Phase 5 - Field Zone UI | Complete |
| FIELD-03 | Phase 5 - Field Zone UI | Complete |
| FIELD-04 | Phase 5 - Field Zone UI | Complete |
| FIELD-05 | Phase 5 - Field Zone UI | Complete |
| FIELD-06 | Phase 5 - Field Zone UI | Complete |
| DRILL-01 | Phase 6 - Drill-Through & Excel Export | Complete |
| DRILL-02 | Phase 6 - Drill-Through & Excel Export | Complete |
| DRILL-03 | Phase 6 - Drill-Through & Excel Export | Complete |
| DRILL-04 | Phase 6 - Drill-Through & Excel Export | Complete |
| DRILL-05 | Phase 6 - Drill-Through & Excel Export | Complete |
| DRILL-06 | Phase 6 - Drill-Through & Excel Export | Complete |
| EXPORT-01 | Phase 6 - Drill-Through & Excel Export | Complete |
| EXPORT-02 | Phase 6 - Drill-Through & Excel Export | Complete |
| EXPORT-03 | Phase 6 - Drill-Through & Excel Export | Complete |
| EXPORT-04 | Phase 6 - Drill-Through & Excel Export | Complete |
| EXPORT-05 | Phase 6 - Drill-Through & Excel Export | Complete |
| UI-01 | Phase 7 - Column Display & UI States | Pending |
| UI-02 | Phase 7 - Column Display & UI States | Pending |
| UI-03 | Phase 7 - Column Display & UI States | Complete |
| UI-04 | Phase 7 - Column Display & UI States | Complete |
| UI-05 | Phase 7 - Column Display & UI States | Pending |
| UI-06 | Phase 7 - Column Display & UI States | Pending |
| CODE-01 | Phase 8 - Code Quality Refactor | Pending |
| CODE-02 | Phase 8 - Code Quality Refactor | Pending |
| CODE-03 | Phase 8 - Code Quality Refactor | Pending |
| CODE-04 | Phase 8 - Code Quality Refactor | Pending |
| QUAL-03 | Phase 8 - Code Quality Refactor | Pending |
| QUAL-04 | Phase 8 - Code Quality Refactor | Pending |
| PKG-01 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| PKG-02 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| PKG-03 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| PKG-04 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| PKG-05 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| DOC-01 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| DOC-02 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| DOC-03 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| DOC-04 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| CI-01 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| CI-02 | Phase 9 - Packaging, Docs & CI/CD | Pending |
| CI-03 | Phase 9 - Packaging, Docs & CI/CD | Pending |

**Coverage:**
- v1 requirements: 58 total (50 original + DRILL-06 + UI-01 through UI-06 + EXPORT-05)
- Mapped to phases: 58
- Unmapped: 0

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-15 — added EXPORT-05 (xlsx/csv 500k threshold), added DRILL-06 (REST endpoint), UI-01–UI-06 (Phase 7), renumbered Code Quality → Phase 8, Packaging → Phase 9*
