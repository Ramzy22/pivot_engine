# External Integrations

**Analysis Date:** 2026-03-15

## APIs & External Services

**Plotly Dash (Component Host):**
- What it's used for: The React component is embedded as a standard Dash component. All frontend↔backend communication is mediated by Dash's prop update mechanism — `setProps(...)` from the component, Python `@app.callback` on the server.
- SDK/Client: `dash` Python package; `@plotly/webpack-dash-dynamic-import` for the webpack build
- Auth: None (inherits Dash app auth if configured externally)

**TanStack Table v8 (Frontend):**
- What it's used for: Core table state model — row model, expanded model, grouped model, column pinning, column sizing, column visibility, sorting state
- SDK/Client: `@tanstack/react-table` ^8.10.7
- Auth: None (client-side library)

**TanStack Virtual v3 (Frontend):**
- What it's used for: Row and column virtualization. `useVirtualizer` with `overscan=12` for rows and `overscan=5` for columns
- SDK/Client: `@tanstack/react-virtual` ^3.0.0
- Auth: None (client-side library)

---

## Data Storage

**Databases:**
- DuckDB (primary)
  - Connection: In-process via `duckdb.connect(":memory:")` for default mode; URI configurable in `create_tanstack_adapter(backend_uri=...)`
  - Client: ibis `duckdb.connect()` wrapping a DuckDB connection, exposed as `IbisPlanner.con`
  - Usage: All analytical queries (pivot aggregations, hierarchy construction, virtual scroll slicing) execute against DuckDB

**File Storage:**
- Local filesystem only — DuckDB databases can be persisted to `.duckdb` files (several debug files visible in the repo root: `adapter_test.duckdb`, `debug_pivot.duckdb`, etc.); these are not used in the production `app.py` (which uses `:memory:`)
- Excel export writes to browser-side via `file-saver` (no server storage)

**Caching:**
- In-memory: `MemoryCache` (`pivot_engine/pivot_engine/cache/memory_cache.py`) — default for hierarchy query results
- Redis: `RedisCache` (`pivot_engine/pivot_engine/cache/redis_cache.py`) — optional; connection string not configured in default setup
- Client-side block cache: `useRowCache` hook in-memory Map, max 500 blocks, evicted LRU

---

## Authentication & Identity

**Auth Provider:** Custom (no third-party auth library detected)

**Implementation:**
- `User` dataclass and `apply_rls_to_spec` in `pivot_engine/pivot_engine/security.py` — row-level security applied to `PivotSpec` before query execution when a `User` object is passed to `handle_virtual_scroll_request`
- Session identity: `sessionId` (UUID stored in `sessionStorage`) and `clientInstance` (ephemeral UUID) are generated client-side in `DashTanstackPivot.react.js` and stamped into every viewport request. No authentication is performed with these values — they are used only for request routing and deduplication.
- `SessionRequestGate` (`pivot_engine/pivot_engine/runtime/session_gate.py`) — serializes concurrent Dash callbacks per session ID to prevent race conditions; not an auth gate

---

## Monitoring & Observability

**Error Tracking:** None (no Sentry, Datadog, or similar detected)

**Logging:**
- Python: `logging.getLogger("pivot_engine.adapter")` in `tanstack_adapter.py`; `logging.getLogger(__name__)` in `lifecycle.py`. Standard Python logging; log level and handler must be configured externally.
- JavaScript: `debugLog` function gated on `process.env.NODE_ENV !== 'production'` — logs to `console.log` with prefix `[pivot-grid]` (main component) or `[pivot-client]` (hooks). Silent in production builds.

**Observability:**
- `pivot_engine/pivot_engine/observability.py` exists but is not wired into the default `app.py`

---

## CI/CD & Deployment

**Hosting:** Not configured — application is run locally via `python dash_presentation/app.py`

**CI Pipeline:** None detected (no `.github/workflows/`, no CI config files)

---

## Environment Configuration

**Required env vars (default app):**
- None strictly required. The app falls back to safe defaults.

**Optional env vars:**
- `PIVOT_DEBUG_OUTPUT` — set to `"1"`, `"true"`, or `"yes"` to enable verbose print output in `dash_presentation/app.py`

**Secrets location:** No secrets files detected in the repository.

---

## Webhooks & Callbacks

**Incoming:**
- Dash prop callbacks — the primary IPC channel. Key inputs handled by `register_dash_pivot_transport_callback`:
  - `viewport` prop: scroll position, window range, state epoch, session/instance IDs, col windowing params
  - `filters`, `sorting`, `expanded`, `rowFields`, `colFields`, `valConfigs` props: structural state
  - `drillThrough` prop: triggers raw-record drill-through query
  - `cellUpdate` / `cellUpdates` props: cell edit write-back

**Outgoing:**
- Dash prop outputs written by the Python callback:
  - `data`: row window (list of dicts with `_path`, `depth`, `_id`, `_isTotal`, `_has_children`)
  - `dataOffset`: integer row index of the first element in `data`
  - `dataVersion` / `rowCount`: updated on every response
  - `columns`: column definition list, may include `{ id: '__col_schema', col_schema: {...} }` sentinel
  - `drillData` / `drill-data-store`: drill-through raw records for the modal

---

*Integration audit: 2026-03-15*
