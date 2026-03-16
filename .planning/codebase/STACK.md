# Technology Stack

**Analysis Date:** 2026-03-15

## Languages

**Primary:**
- JavaScript (ES2020+) — React frontend component source (`dash_tanstack_pivot/src/`)
- Python 3.8–3.11 — Backend pivot engine and Dash application

**Secondary:**
- JSX — Used throughout React component source files
- CSS-in-JS — All styling is inline style objects or injected `<style>` tags (no external CSS files)

---

## Runtime

**Environment:**
- Node.js ≥8.11.0 (build only; not required at runtime for the Dash app)
- Python ≥3.8

**Package Manager:**
- npm ≥6.1.0 (frontend build)
- pip / setuptools (Python packages)
- Lockfile: `dash_tanstack_pivot/package-lock.json` present

---

## Frameworks

**Core (Frontend):**
- React 16.8.6 (peer dependency) — functional components with hooks throughout
- Plotly Dash (version from environment) — widget embedding via `setProps` / prop callbacks; the component is a standard Dash component

**Core (Backend):**
- Dash (`dash`) — application framework and callback system
- ibis-framework ≥4.0.0 — query builder layer over DuckDB; used in `IbisPlanner` to translate `PivotSpec` to SQL
- DuckDB ≥0.8.0 — default in-process analytical database backend
- PyArrow ≥10.0.0 — in-memory columnar format; used throughout for data exchange between layers and for the materialized hierarchy tables

**Testing:**
- pytest ≥7.0.0
- pytest-asyncio ≥0.20.0 — async test support for controller methods

**Build/Dev:**
- Webpack 5 — production bundle (`dash_tanstack_pivot/webpack.config.js`)
- Babel 7 — transpilation with `@babel/preset-react`, `@babel/preset-env`, optional chaining, nullish coalescing plugins
- `@plotly/webpack-dash-dynamic-import` — Dash-specific webpack plugin

---

## Key Dependencies

**Critical (Frontend):**
- `@tanstack/react-table` ^8.10.7 — table core model: `useReactTable`, `getCoreRowModel`, `getExpandedRowModel`, `getGroupedRowModel`, `flexRender`
- `@tanstack/react-virtual` ^3.0.0 — `useVirtualizer` for both row and column virtualization
- `xlsx` ^0.18.5 — Excel export via `XLSX.utils.json_to_sheet` / `writeFile`
- `file-saver` ^2.0.5 — browser file save trigger for Excel export
- `ramda` ^0.26.1 — utility functions (imported but usage is limited)

**Critical (Backend):**
- `duckdb` ≥0.8.0 — primary query engine; used via ibis `duckdb.connect()` or directly
- `ibis-framework` ≥4.0.0 — abstraction over DuckDB (and potentially Clickhouse) in `IbisPlanner`
- `pyarrow` ≥10.0.0 — `pa.Table` as the canonical in-process data format
- `pandas` ≥1.5.0 — `normalize_data_input` accepts pandas DataFrames; converts to Arrow
- `dash` — prop callback system is the entire IPC channel between frontend and backend

**Optional Infrastructure:**
- `fastapi` ≥0.100.0 — REST API layer (guarded by `extra != "no-api"`, not used in default Dash mode)
- `uvicorn[standard]` ≥0.20.0 — FastAPI server (same optional guard)
- Redis — `RedisCache` in `pivot_engine/pivot_engine/cache/redis_cache.py` (optional; `MemoryCache` is default)

---

## Configuration

**Environment:**
- `PIVOT_DEBUG_OUTPUT` env var — enables verbose logging in `dash_presentation/app.py`
- No `.env` file detected in committed code; secrets are not present

**Build:**
- `dash_tanstack_pivot/webpack.config.js` — Webpack production config
- `dash_tanstack_pivot/.babelrc` — Babel presets and plugins
- `pivot_engine/pyproject.toml` — Python package build config (setuptools)

---

## Platform Requirements

**Development:**
- Node.js ≥8.11.0 and npm for frontend build
- Python ≥3.8 with pip
- Install frontend: `cd dash_tanstack_pivot && npm install && npm run build`
- Install backend: `cd pivot_engine && pip install -e .`

**Production:**
- Python runtime only (the compiled `.min.js` is committed)
- DuckDB ≥0.8.0 and PyArrow ≥10.0.0 are the primary runtime dependencies
- No Node.js required at runtime
- Dash application served via `python dash_presentation/app.py` (development) or a WSGI server (production)

---

*Stack analysis: 2026-03-15*
