# Requirements: DashTanstackPivot

**Defined:** 2026-03-13
**Core Value:** A Python developer adds an enterprise-grade pivot table to any Dash app in under 10 lines of code — no JS knowledge, no database config, no performance tuning required.

---

## v1 Requirements

### Quality Baseline

- [ ] **QUAL-01**: All existing tests pass before any new development begins (establish green baseline)
- [ ] **QUAL-02**: Test coverage report generated and baseline documented
- [ ] **QUAL-03**: Duplicate method `run_pivot_arrow()` in controller.py removed
- [ ] **QUAL-04**: Column name sanitization secured against SQL injection

### Bug Fixes — Data Correctness

- [ ] **BUG-01**: Grand total rows display correct aggregated values for all measure types
- [ ] **BUG-02**: Grand total rows do not disappear or flicker on scroll/filter changes
- [ ] **BUG-03**: Pivot column discovery returns complete, non-sparse column set after data changes
- [ ] **BUG-04**: Pivot column discovery is consistent across page refreshes and filter changes
- [ ] **BUG-05**: Filter state persists across row expansion, sort changes, and viewport scroll
- [ ] **BUG-06**: Sort state applies server-side and does not reset on data refresh

### Bug Fixes — Virtual Scroll & UI

- [ ] **BUG-07**: Virtual scroll rows stay synchronized with server data (no stale rows, no blank areas)
- [ ] **BUG-08**: Block-based cache invalidates correctly on filter/sort/spec changes
- [ ] **BUG-09**: Multi-level column headers align correctly over their data cells at all nesting depths
- [ ] **BUG-10**: Column header span widths match the total width of child columns
- [ ] **BUG-11**: Row group expansion shows correct children with correct indentation levels
- [ ] **BUG-12**: Expanding/collapsing rows does not corrupt sibling row positions
- [ ] **BUG-13**: Context menu renders within viewport bounds (no off-screen rendering)

### Data Input API

- [ ] **API-01**: Component accepts a pandas DataFrame as `data` prop
- [ ] **API-02**: Component accepts a polars DataFrame as `data` prop
- [ ] **API-03**: Component accepts an Ibis table expression as `data` prop
- [ ] **API-04**: Component accepts a connection string + table name as `data` prop
- [ ] **API-05**: Input type is auto-detected at runtime — same prop interface for all types
- [ ] **API-06**: Meaningful error message shown when unsupported input type passed

### Field Zone UI

- [ ] **FIELD-01**: Sidebar contains four drop zones: Rows, Columns, Values, Filters
- [ ] **FIELD-02**: User can drag any available field into any drop zone
- [ ] **FIELD-03**: Dropping a field into Values zone shows aggregation selector (sum/avg/count/min/max)
- [ ] **FIELD-04**: Removing a field from a zone updates the pivot immediately
- [ ] **FIELD-05**: Field zone state is reflected back to Python via Dash props (bidirectional)
- [ ] **FIELD-06**: Field zone initial state can be set from Python props

### Drill-Through

- [ ] **DRILL-01**: Clicking any aggregated cell triggers drill-through action
- [ ] **DRILL-02**: Drill-through displays a modal/panel showing the source rows for that cell
- [ ] **DRILL-03**: Source rows are fetched server-side using the cell's pivot coordinate filters
- [ ] **DRILL-04**: Drill-through modal is paginated (handles large source sets)
- [ ] **DRILL-05**: Drill-through modal supports column sorting and basic filtering

### Excel Export

- [ ] **EXPORT-01**: "Export to Excel" button downloads the current pivot view as .xlsx
- [ ] **EXPORT-02**: Exported file preserves multi-level column headers
- [ ] **EXPORT-03**: Exported file preserves row hierarchy (indentation or grouping)
- [ ] **EXPORT-04**: Grand totals and subtotals are included in export

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
| QUAL-01 | Phase 1 - Test Audit & Baseline | Pending |
| QUAL-02 | Phase 1 - Test Audit & Baseline | Pending |
| BUG-01 | Phase 2 - Data Correctness Bugs | Pending |
| BUG-02 | Phase 2 - Data Correctness Bugs | Pending |
| BUG-03 | Phase 2 - Data Correctness Bugs | Pending |
| BUG-04 | Phase 2 - Data Correctness Bugs | Pending |
| BUG-05 | Phase 2 - Data Correctness Bugs | Pending |
| BUG-06 | Phase 2 - Data Correctness Bugs | Pending |
| BUG-07 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| BUG-08 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| BUG-09 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| BUG-10 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| BUG-11 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| BUG-12 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| BUG-13 | Phase 3 - Virtual Scroll & UI Bugs | Pending |
| API-01 | Phase 4 - Data Input API | Pending |
| API-02 | Phase 4 - Data Input API | Pending |
| API-03 | Phase 4 - Data Input API | Pending |
| API-04 | Phase 4 - Data Input API | Pending |
| API-05 | Phase 4 - Data Input API | Pending |
| API-06 | Phase 4 - Data Input API | Pending |
| FIELD-01 | Phase 5 - Field Zone UI | Pending |
| FIELD-02 | Phase 5 - Field Zone UI | Pending |
| FIELD-03 | Phase 5 - Field Zone UI | Pending |
| FIELD-04 | Phase 5 - Field Zone UI | Pending |
| FIELD-05 | Phase 5 - Field Zone UI | Pending |
| FIELD-06 | Phase 5 - Field Zone UI | Pending |
| DRILL-01 | Phase 6 - Drill-Through & Excel Export | Pending |
| DRILL-02 | Phase 6 - Drill-Through & Excel Export | Pending |
| DRILL-03 | Phase 6 - Drill-Through & Excel Export | Pending |
| DRILL-04 | Phase 6 - Drill-Through & Excel Export | Pending |
| DRILL-05 | Phase 6 - Drill-Through & Excel Export | Pending |
| EXPORT-01 | Phase 6 - Drill-Through & Excel Export | Pending |
| EXPORT-02 | Phase 6 - Drill-Through & Excel Export | Pending |
| EXPORT-03 | Phase 6 - Drill-Through & Excel Export | Pending |
| EXPORT-04 | Phase 6 - Drill-Through & Excel Export | Pending |
| CODE-01 | Phase 7 - Code Quality Refactor | Pending |
| CODE-02 | Phase 7 - Code Quality Refactor | Pending |
| CODE-03 | Phase 7 - Code Quality Refactor | Pending |
| CODE-04 | Phase 7 - Code Quality Refactor | Pending |
| QUAL-03 | Phase 7 - Code Quality Refactor | Pending |
| QUAL-04 | Phase 7 - Code Quality Refactor | Pending |
| PKG-01 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| PKG-02 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| PKG-03 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| PKG-04 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| PKG-05 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| DOC-01 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| DOC-02 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| DOC-03 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| DOC-04 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| CI-01 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| CI-02 | Phase 8 - Packaging, Docs & CI/CD | Pending |
| CI-03 | Phase 8 - Packaging, Docs & CI/CD | Pending |

**Coverage:**
- v1 requirements: 47 total
- Mapped to phases: 47
- Unmapped: 0

---
*Requirements defined: 2026-03-13*
*Last updated: 2026-03-13 after roadmap creation*
