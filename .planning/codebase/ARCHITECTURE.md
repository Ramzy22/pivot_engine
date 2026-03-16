# Architecture

**Analysis Date:** 2026-03-15

## Pattern Overview

**Overall:** Server-side OLAP pivot table with a React frontend widget embedded in a Plotly Dash application. The frontend and backend communicate exclusively through Dash's property callback mechanism — the React component writes to named props (`viewport`, `expanded`, `filters`, etc.) and reads from props the Python callback updates (`data`, `rowCount`, `columns`, `dataVersion`). There is no direct HTTP call from the component; all I/O is mediated by Dash.

**Key Characteristics:**
- Full server-side pagination: the component holds a sliding cache of blocks; the server owns all row data
- Hierarchical tree model: rows carry `_path` (pipe-separated dimension values), `depth`, `_id`, `_isTotal`, `_has_children`, `_is_expanded` metadata
- Dual virtualization: rows via `@tanstack/react-virtual` row virtualizer + columns via a separate column virtualizer for wide pivot tables
- Epoch + version concurrency model: every structural change bumps `stateEpoch`; every viewport request stamps a monotonically-increasing `window_seq`; stale responses are rejected client- and server-side

---

## Layers

**React Component (Dash widget):**
- Purpose: All UI rendering, user interaction, local state, and communication with the Dash backend
- Location: `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js`
- Contains: ~4072-line monolithic functional component; TanStack Table configuration; layout render (sidebar, header, body, status bar); all event handlers
- Depends on: custom hooks (`useServerSideRowModel`, `useColumnVirtualizer`, `useStickyStyles`, `useRowCache`), utility modules (`utils/styles.js`, `utils/helpers.js`), sub-components (Filters, Sidebar, Table subdirectories)
- Used by: Dash layout via `dash_tanstack_pivot` Python package

**Custom Hooks:**
- Purpose: Encapsulate the two most complex concerns — server-side block cache management and column virtualization — out of the main component
- Location: `dash_tanstack_pivot/src/lib/hooks/`
- `useServerSideRowModel.js`: owns the block cache, viewport request debouncing, inflight tracking, stale-response rejection, grand-total extraction, scroll-clamp on rowCount shrink
- `useRowCache.js`: epoch-keyed `Map<"epoch:blockIndex", block>` with LRU eviction, stale-while-revalidate, `softInvalidateFromBlock`, `pruneToRange`
- `useColumnVirtualizer.js`: wraps `@tanstack/react-virtual` horizontal virtualizer over center (unpinned) columns; exposes `visibleColRange`, spacer widths, total layout width
- `useStickyStyles.js`: computes `position:sticky` left/right offsets for pinned columns

**Python Adapter Layer:**
- Purpose: Translate the Dash prop payload from the React component into `PivotSpec` objects; route to the correct controller method; translate results back to the row dict format the component expects
- Location: `pivot_engine/pivot_engine/tanstack_adapter.py`
- Key class: `TanStackPivotAdapter` — converts `TanStackRequest` → `PivotSpec`, calls `controller.run_hierarchy_view` / `run_virtual_scroll_hierarchical`, runs `_apply_col_windowing`, returns `TanStackResponse`
- Depends on: `ScalablePivotController`, `PivotSpec`, `security.apply_rls_to_spec`
- Used by: `PivotRuntimeService` (runtime layer)

**Runtime / Dash Callback Layer:**
- Purpose: Wire Dash Input/Output callbacks; route viewport/expansion/structural intents; enforce the `SessionRequestGate` (one in-flight request per session); inject `__col_schema` sentinel into `columns` prop
- Location: `pivot_engine/pivot_engine/runtime/` (`service.py`, `dash_callbacks.py`, `session_gate.py`)
- Key classes: `PivotRuntimeService`, `SessionRequestGate`, `register_dash_pivot_transport_callback`
- Depends on: `TanStackPivotAdapter`, Dash `callback`
- Used by: `dash_presentation/app.py`

**Backend Controller:**
- Purpose: Execute OLAP pivot queries against DuckDB/Ibis; manage materialized hierarchies, progressive loading, CDC, streaming
- Location: `pivot_engine/pivot_engine/scalable_pivot_controller.py`
- Key class: `ScalablePivotController` extends `PivotController`
- Sub-components:
  - `hierarchical_scroll_manager.py` — `HierarchicalVirtualScrollManager`: PyArrow-based slicing of the visible row window for virtual scroll
  - `materialized_hierarchy_manager.py` — pre-computes and caches hierarchy aggregations
  - `planner/ibis_planner.py` — builds Ibis expressions from `PivotSpec`
  - `backends/duckdb_backend.py`, `backends/ibis_backend.py`
  - `cache/memory_cache.py`, `cache/redis_cache.py`
  - `tree.py` — `TreeExpansionManager`

---

## Data Flow

**Server-Side Viewport Request:**

1. User scrolls → `useServerSideRowModel` debounce timer fires (50ms)
2. Missing blocks identified in the row cache → `requestViewport` called
3. `setProps({ viewport: { start, end, count, window_seq, state_epoch, session_id, col_start, col_end, needs_col_schema } })`
4. Dash callback in `PivotRuntimeService` fires; `SessionRequestGate` serializes concurrent requests per session
5. `TanStackPivotAdapter.handle_virtual_scroll_request` → `controller.run_hierarchy_view` or `run_virtual_scroll_hierarchical`
6. `HierarchicalVirtualScrollManager.get_visible_rows_hierarchical` slices the PyArrow materialized table to `[start_row, end_row]`
7. Response: `TanStackResponse.data` (row dicts with `_path`, `depth`, `_id`, `_isTotal`, `_has_children`) + optional `col_schema`
8. Callback writes `data`, `dataOffset`, `dataVersion`, `rowCount`, `columns` props back to component
9. `useServerSideRowModel` data-sync effect receives new `data`/`dataVersion` → `setBlockLoaded` fills cache blocks
10. `renderedData` memo reads from cache → `tableData` → TanStack row model renders

**Expansion Toggle:**

1. User clicks expand chevron → `onExpandedChange` fires
2. Component detects expansion-only change (no structural fields changed) → calls `beginExpansionRequest` (bumps `abortGeneration`, NOT `stateEpoch`)
3. `setProps({ ...nextProps, viewport: { intent: 'expansion', ... } })` — existing cache stays valid
4. Server returns updated row window including newly-visible children
5. `pendingExpansionRef.anchorBlock` used in deferred effect: `softInvalidateFromBlock(anchorBlock + 1)` marks post-anchor blocks as `partial` for background refresh without skeleton flash

**Structural Change (filters / sort / field zone change):**

1. Any structural prop changes → `beginStructuralTransaction` bumps both `stateEpoch` and `abortGeneration`
2. `setProps({ ...nextProps, viewport: { intent: 'structural', needs_col_schema: true } })` — full cache wipe on epoch change
3. Server returns fresh data + new `col_schema` → `cachedColSchema` is updated
4. Subsequent viewport requests use `col_start`/`col_end` windowing once schema is known

**Column Windowing:**

1. `useColumnVirtualizer` computes `visibleColRange` (first/last visible center column index)
2. Component computes `colRequestStart` / `colRequestEnd` aligned to `COL_BLOCK_SIZE=20` blocks with `COL_OVERSCAN=1`
3. These are stamped into every viewport `setProps` call as `col_start`/`col_end`
4. `TanStackPivotAdapter._apply_col_windowing` strips non-window columns from each row dict
5. `useRowCache` stores `colStart`/`colEnd` per block; a col-range mismatch triggers re-fetch

---

## Key Abstractions

**`stateEpoch`:**
- Purpose: Monotonically increasing integer; separates cache entries from different structural states. Old-epoch cache entries are purged immediately on epoch change.
- Files: `DashTanstackPivot.react.js` (lines ~607-724), `useRowCache.js` (line ~37-53)

**`TanStackRequest` / `TanStackResponse`:**
- Purpose: Typed dataclass protocol between the Dash callback and the adapter layer
- Files: `pivot_engine/pivot_engine/tanstack_adapter.py` (lines ~141-166)
- Key fields on request: `table`, `columns`, `filters`, `sorting`, `grouping`, `aggregations`, `pagination`, `version`
- Key fields on response: `data` (list of row dicts), `columns`, `total_rows`, `col_schema`

**`_path` (row identity key):**
- Purpose: Pipe-separated (`|||`) string of dimension values forming the row's position in the hierarchy tree. Used as row ID in TanStack, expansion key in `expanded` dict, and foreign key for matching cache entries after expand/collapse.
- Examples: `"North"`, `"North|||USA"`, `"__grand_total__"`
- Files: `tanstack_adapter.py` (lines ~481-499), `DashTanstackPivot.react.js` (getRowId, onExpandedChange)

**`PivotSpec`:**
- Purpose: Declarative description of a pivot query — table, row dimensions, column dimensions, measures, filters, sort, limit/offset
- Files: `pivot_engine/pivot_engine/types/pivot_spec.py`

**Block Cache (`useRowCache`):**
- Purpose: Client-side sliding window cache keyed by `"epoch:blockIndex"`. Tracks block status: `loading` | `loaded` | `partial` | `error`. Implements stale-while-revalidate via `softInvalidateFromBlock` (keeps rows, marks as `partial`).
- Files: `dash_tanstack_pivot/src/lib/hooks/useRowCache.js`

---

## Entry Points

**Dash App:**
- Location: `dash_presentation/app.py`
- Triggers: `app.run(debug=True, port=8050)`
- Responsibilities: Instantiate adapter/controller, load data, define layout, wire Dash callbacks via `register_dash_pivot_transport_callback` and `register_dash_drill_modal_callback`

**Webpack Bundle:**
- Location: `dash_tanstack_pivot/src/lib/index.js` → `DashTanstackPivot.react.js`
- Built output: `dash_tanstack_pivot/dash_tanstack_pivot/dash_tanstack_pivot.min.js`

**Python Package:**
- Location: `pivot_engine/pivot_engine/__init__.py`
- Exports: `create_tanstack_adapter`, `PivotRuntimeService`, `SessionRequestGate`, `register_dash_pivot_transport_callback`, `register_dash_drill_modal_callback`, `ScalablePivotController`

---

## Error Handling

**Strategy:** Layered degradation — the adapter wraps controller calls in try/except and falls back from `run_virtual_scroll_hierarchical` → `handle_hierarchical_request` → direct `run_pivot_async`. The frontend renders `SkeletonRow` for missing cache entries and shows stale rows for `partial` blocks.

**Patterns:**
- Stale response rejection: `block.version > requestVersion` check in `useRowCache.setBlockLoaded` discards out-of-order responses
- Inflight deduplication: `inflightRequestRef` in `useServerSideRowModel` prevents issuing duplicate viewport requests for the same row/col range
- Orphan detection: loading blocks not covered by the current inflight are re-requested immediately (prevents stuck skeletons on rapid scroll)
- 10-second structural timeout: `structuralInFlight` is force-cleared after 10 seconds to prevent permanently blocked UI (`DashTanstackPivot.react.js` line ~1902)
- `SessionRequestGate`: Python-side serialization prevents concurrent Dash callbacks from racing against each other for the same session

---

## Cross-Cutting Concerns

**Session / Client Identity:**
- `sessionId`: stable per browser tab (written to `sessionStorage`), passed in every viewport request
- `clientInstance`: ephemeral UUID per component mount, used to distinguish simultaneous mounts of the same component ID
- Both are passed through `viewport` prop to the server for request routing

**Column Schema Discovery:**
- Server-side: authoritative `col_schema` embedded as a sentinel entry (`{ id: '__col_schema', col_schema: {...} }`) in `props.columns` by the runtime callback. The component extracts it in a dedicated `useEffect` and stores it in `cachedColSchema`.
- Client-side fallback: derived from row dict keys when `serverSide=false` and no schema is present

**Grand Total Row:**
- Identified by `_isTotal=true`, `_id='Grand Total'`, or `_path='__grand_total__'`
- In server-side mode with `showColTotals=true`, the grand total is extracted from the incoming data by `useServerSideRowModel` (`excludeGrandTotal=true`), held separately in `grandTotalRow`, and pinned sticky at top or bottom via `rowPinning`

**Persistence:**
- Column and row pinning state can be persisted to `localStorage` or `sessionStorage` using the `persistence` and `persistence_type` props

---

*Architecture analysis: 2026-03-15*
