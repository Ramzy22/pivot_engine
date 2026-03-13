# DashTanstackPivot

## What This Is

An open-source, production-grade Dash component that delivers enterprise-level pivot table capabilities — comparable to AG Grid Enterprise or Excel pivot tables — with zero configuration. Python developers `pip install` it and pass a DataFrame, Ibis table, or connection string; the component handles server-side aggregation, virtual scrolling, and all pivot mechanics automatically.

## Core Value

A Python developer should be able to add a fully functional, high-performance pivot table to any Dash app in under 10 lines of code — no JS knowledge, no database config, no performance tuning required.

## Requirements

### Validated

- ✓ TanStack Table v8 + TanStack Virtual frontend — existing
- ✓ Ibis-based backend (DuckDB, Postgres, BigQuery, Snowflake, ClickHouse, etc.) — existing
- ✓ Virtual scrolling for millions of rows — existing
- ✓ Row/column hierarchies with tree expansion — existing
- ✓ Multi-condition filtering (AND/OR) — existing
- ✓ Sorting (natural sort, case options) — existing
- ✓ Aggregations (sum, avg, count, min, max, etc.) — existing
- ✓ Cell selection, keyboard navigation, Ctrl+C copy — existing
- ✓ Smart caching (diff engine, tile-based, Redis optional) — existing
- ✓ Server-side row model with block-based data fetching — existing
- ✓ Row/column pinning — existing
- ✓ Partial field zone sidebar — existing

### Active

- [ ] Fix grand total calculation bugs (wrong values, disappearing totals)
- [ ] Fix virtual scroll + data sync (stale rows, blank areas, desync)
- [ ] Fix filter/sort state (resets unexpectedly, doesn't apply server-side)
- [ ] Fix pivot column discovery (missing/sparse columns after data changes)
- [ ] Fix multi-level column headers (misalignment over data cells)
- [ ] Fix row group display (wrong children, wrong indentation on expand)
- [ ] Auto-detect data input (DataFrame OR Ibis table OR connection string)
- [ ] Complete field zone UI (drag rows/cols/values/filters like Excel)
- [ ] Implement drill-through (click cell → modal showing source rows)
- [ ] Excel export (.xlsx download of current pivoted view)
- [ ] Calculated fields (derived measures like Revenue - Cost)
- [ ] Pre-development test suite audit (run all 65 tests, establish baseline)
- [ ] Code refactor: split 1500-line main component into focused modules
- [ ] Add React error boundaries (prevent full unmount on crash)
- [ ] Fix SQL injection risk in column name sanitization
- [ ] Fix duplicate method definition in controller.py
- [ ] PyPI packaging (pip install dash-tanstack-pivot)
- [ ] Optional extras (pip install dash-tanstack-pivot[redis], [flight])
- [ ] Comprehensive documentation + usage examples
- [ ] CI/CD pipeline (tests on push, auto-publish on tag)

### Out of Scope

- Real-time CDC / WebSocket streaming — high complexity, deferred to v2
- OLAP cube / MDX calculated members — niche, not needed for v1
- Embedded charts — separate concern, out of v1
- Mobile responsive layout — desktop-first, deferred
- Cross-filtering between multiple pivots — v2 feature
- Goal seek / what-if analysis — Excel Solver equivalent, out of scope

## Context

The codebase is already substantial (~65 test files, 1500+ line React component, full Ibis query planning engine with cost estimation, CDC infrastructure, Arrow IPC transport). The key constraint is: **verify before touching** — run all existing tests first, understand what's actually working, then build only what's genuinely missing. Do not re-implement what exists.

Known architecture:
- Frontend: `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js` (main, ~1500 lines) + Filters/, Table/, Sidebar/, hooks/, utils/
- Backend: `pivot_engine/pivot_engine/` — controller.py, planner/ibis_planner.py, backends/, tree.py, diff/diff_engine.py, cache/
- Build: npm (webpack) for JS, setuptools for Python package
- Tests: 65 test files in `tests/` and root directory

## Constraints

- **Test-first**: Every phase must start with running existing tests — no development without knowing the baseline
- **Verify before build**: Check if a feature exists before implementing it
- **Tech stack**: TanStack Table v8 + TanStack Virtual (locked), Ibis for backend (locked), Dash (locked)
- **Distribution**: pip + optional extras ([redis], [flight]) — no mandatory heavy deps
- **Compatibility**: Must work with pandas, polars, and any Ibis-supported database (auto-detect input type)
- **Open source**: Apache/MIT license, GitHub, PyPI

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| TanStack Table v8 (not AG Grid) | Open source, no license fees, better perf on huge datasets | — Pending |
| Ibis as query abstraction | Database-agnostic, compiles to SQL, safe from injection | — Pending |
| Server-side row model | Scales to millions of rows without sending all data to browser | — Pending |
| pip + optional extras | Keep core lightweight, heavy deps (Redis, Flight) opt-in | — Pending |
| Verify-before-develop constraint | 65 existing test files and substantial code — avoid duplication | — Pending |

---
*Last updated: 2026-03-13 after initialization*
