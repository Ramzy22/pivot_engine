# Phase 6: Drill-Through & Excel Export - Research

**Researched:** 2026-03-15
**Domain:** REST endpoint integration in Dash/Flask, SheetJS/xlsx client-side export, server-side pagination patterns
**Confidence:** HIGH

---

## Summary

Phase 6 adds two independent features: (1) a drill-through modal that opens when the user clicks any aggregated cell and fetches the underlying source rows via a dedicated REST endpoint called directly from React (bypassing Dash callbacks), and (2) an Export button that downloads the current pivot view as `.xlsx` (for ≤ 500,000 rows) or `.csv` (above that threshold).

The backend infrastructure for drill-through is already partially built. `TanStackPivotAdapter.handle_drill_through` exists and delegates to `ScalablePivotController.get_drill_through_data`, which executes a filtered DuckDB query returning paginated raw rows. What is missing is: (a) exposing this path as a proper REST endpoint on Flask's underlying server (`app.server`) rather than through a Dash callback, (b) wiring the React component to call that endpoint directly with `fetch()`, (c) building a paginated modal UI in the component, and (d) upgrading the existing naive `exportExcel` function to handle large datasets and the xlsx/csv threshold.

The existing drill-through path in the Dash callback (`register_dash_drill_modal_callback` + `drill-data-store`) is **not** the Phase 6 target. That path serializes all records through Dash props, which breaks on large result sets. Phase 6 replaces it with a direct React-to-Flask fetch, keeping the full Dash callback system intact for pivot data but routing drill-through through a separate HTTP channel.

**Primary recommendation:** Add a Flask route on `app.server` at `/api/drill-through`; call it from React with `fetch()`; render results in a modal table with server-side page/sort/filter query params. Upgrade `exportExcel` to stream CSV when `rowCount > 500000`.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DRILL-01 | Clicking any aggregated cell triggers drill-through action | Cell's `_path` and `colId` are already available in context menu handler (line 1221); need to wire a click handler on data cells that passes coordinate to the new modal |
| DRILL-02 | Drill-through displays a modal showing source rows for that cell | New React modal component; no existing one suitable — existing modal in app.py is Dash-managed and bypassed |
| DRILL-03 | Source rows fetched via `/api/drill-through` REST endpoint called directly from React | Flask route on `app.server`; `fetch()` in React; no `setProps` involved |
| DRILL-04 | Modal paginates server-side (`?page=N&page_size=500`); full dataset never sent to browser | `get_drill_through_data` already accepts `limit`/`offset`; endpoint maps page/page_size to those |
| DRILL-05 | Modal supports server-side column sorting and text filter via query params | Extend endpoint to accept `sort_col`, `sort_dir`, `filter` query params; pass to DuckDB query |
| DRILL-06 | `/api/drill-through` applies cell's exact pivot coordinate filters server-side (DuckDB) | `get_drill_through_data` already builds a filtered DuckDB query from `spec.filters + drill_filters`; need to map `_path` coordinate to equality filters |
| EXPORT-01 | "Export to Excel" button downloads current pivot view as .xlsx | Existing `exportExcel` function present; needs to be upgraded to use full pivot data not just rendered rows |
| EXPORT-02 | Exported file preserves multi-level column headers | SheetJS `aoa_to_sheet` pattern; build header rows from `cachedColSchema` + column tree |
| EXPORT-03 | Exported file preserves row hierarchy (indentation or grouping) | Prepend `depth * "  "` indent to the hierarchy column cell value in the export array |
| EXPORT-04 | Grand totals and subtotals are included in export | Ensure `_isTotal` rows are included (they are in the rendered rows cache) |
| EXPORT-05 | Button downloads .xlsx when row count ≤ 500,000; downloads .csv above that; UI label updates | `rowCount` prop is already available; branch in `exportExcel`; CSV path uses Blob with text |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `xlsx` (SheetJS) | ^0.18.5 (already installed) | Client-side Excel generation in browser | Already in package.json; produces `.xlsx` from JS arrays; no server round-trip needed for export |
| `file-saver` | ^2.0.5 (already installed) | Triggers browser file download from Blob | Already in package.json; used in current `exportExcel` |
| Flask (via Dash) | Same version as Dash uses | REST endpoint on `app.server` | Dash's underlying Flask server is already running; adding routes needs zero new deps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `ibis-framework` | ≥4.0.0 (already in backend) | Build filtered DuckDB query for drill-through | Already used in `get_drill_through_data`; no changes to query engine needed |
| `pyarrow` | ≥10.0.0 (already in backend) | Serialize drill-through results to JSON | Already used throughout adapter layer |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flask route on `app.server` | FastAPI (`complete_rest_api.py`) | FastAPI already exists but runs as a separate process; Flask is always available because Dash uses it; simpler for a single endpoint |
| Client-side xlsx generation | Server-side xlsx generation (openpyxl) | Server-side is needed only if formatting is complex; client-side is simpler for Phase 6 scope |
| `fetch()` in React | axios | No new dependency; `fetch()` is native browser API |

**Installation:** No new packages needed. All required libraries are already installed.

---

## Architecture Patterns

### Recommended Project Structure
```
dash_presentation/
└── app.py                    # Add @app.server.route('/api/drill-through')

pivot_engine/pivot_engine/
└── scalable_pivot_controller.py   # get_drill_through_data already exists; extend with sort/filter params

dash_tanstack_pivot/src/lib/components/
├── DashTanstackPivot.react.js     # Add drillEndpoint prop, DrillModal state, upgrade exportExcel
└── Table/
    └── DrillThroughModal.js       # New sub-component: modal with table, pagination, sort, filter
```

### Pattern 1: Flask Route on app.server

**What:** Attach a route directly to Dash's underlying Flask server instance — `app.server` is a standard Flask `Flask` object and accepts `@app.server.route(...)` decorators.

**When to use:** When a React component needs to call a backend endpoint without going through Dash's prop/callback system (bypassing serialization limits and callback round-trip latency).

**Example:**
```python
# In dash_presentation/app.py — after app = Dash(__name__)
import json
from flask import request, jsonify

@app.server.route('/api/drill-through')
def drill_through():
    """
    Query params:
      - table:      DuckDB table name (string)
      - row_path:   pipe-separated dimension path, e.g. "North|||USA"
      - row_fields: comma-separated row dimension names, e.g. "region,country"
      - col_fields: JSON-encoded column coordinate filters (optional)
      - page:       0-indexed page number (default 0)
      - page_size:  rows per page (default 500, max 500)
      - sort_col:   column id to sort by (optional)
      - sort_dir:   "asc" or "desc" (default "asc")
      - filter:     text search applied across all string columns (optional)
    """
    table = request.args.get('table', '')
    row_path = request.args.get('row_path', '')
    row_fields = request.args.get('row_fields', '').split(',')
    page = int(request.args.get('page', 0))
    page_size = min(int(request.args.get('page_size', 500)), 500)
    sort_col = request.args.get('sort_col')
    sort_dir = request.args.get('sort_dir', 'asc')
    text_filter = request.args.get('filter', '')

    # Build equality filters from row path
    drill_filters = []
    if row_path and row_fields:
        parts = row_path.split('|||')
        for i, field in enumerate(row_fields):
            if i < len(parts) and parts[i]:
                drill_filters.append({'field': field, 'op': '=', 'value': parts[i]})

    # Call controller (synchronous wrapper)
    adapter = get_adapter()
    from pivot_engine.types.pivot_spec import PivotSpec
    spec = PivotSpec(table=table, rows=[], measures=[], filters=[])
    import asyncio
    records = asyncio.run(
        adapter.controller.get_drill_through_data(
            spec, drill_filters,
            limit=page_size, offset=page * page_size
        )
    )
    return jsonify({'rows': records, 'page': page, 'page_size': page_size})
```

### Pattern 2: React fetch() to REST Endpoint

**What:** React component calls `fetch(drillEndpoint + queryString)` directly when a cell is clicked, then renders results in a modal. No `setProps` call; the endpoint URL is passed as a prop.

**When to use:** When result sets can be arbitrarily large and Dash callback serialization would be a bottleneck. Also enables per-page loading without a round-trip through Python's callback system.

**Example:**
```javascript
// In DashTanstackPivot.react.js — new state
const [drillModal, setDrillModal] = useState(null); // { path, colId, rows, page, totalRows, loading }

const openDrillModal = useCallback(async (rowPath, rowFields, page = 0) => {
    const base = (props.drillEndpoint || '/api/drill-through');
    const params = new URLSearchParams({
        table: tableName,
        row_path: rowPath,
        row_fields: rowFields.join(','),
        page,
        page_size: 500,
    });
    setDrillModal(prev => ({ ...(prev || {}), loading: true }));
    const resp = await fetch(`${base}?${params}`);
    const json = await resp.json();
    setDrillModal({ path: rowPath, rows: json.rows, page: json.page, loading: false });
}, [tableName, props.drillEndpoint]);
```

### Pattern 3: SheetJS xlsx/csv Export with Row Count Branching

**What:** Check `rowCount` against the 500,000 threshold before choosing format. For xlsx use existing `XLSX.utils.aoa_to_sheet` with a header array; for CSV generate a plain text string and save as Blob.

**When to use:** All export invocations. `rowCount` prop is already available in the component.

**Example:**
```javascript
// Source: existing exportExcel + SheetJS docs
const exportPivot = useCallback(() => {
    const XLSX_LIMIT = 500000;
    const isCSV = (rowCount || 0) > XLSX_LIMIT;

    if (isCSV) {
        // Build CSV text from rendered rows
        const headers = /* leaf column headers */;
        const lines = [headers.join(',')];
        rows.forEach(r => {
            lines.push(headers.map(h => JSON.stringify(r.original[h] ?? '')).join(','));
        });
        const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
        saveAs(blob, 'pivot.csv');
    } else {
        // Build header rows for multi-level columns using aoa_to_sheet
        const aoaData = buildExportAoa(rows, columns);  // helper
        const ws = XLSX.utils.aoa_to_sheet(aoaData);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, 'Pivot');
        const buf = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
        saveAs(new Blob([buf], { type: 'application/octet-stream' }), 'pivot.xlsx');
    }
}, [rows, columns, rowCount]);
```

### Pattern 4: Cell Coordinate as Drill-Through Filters

**What:** Every rendered row has `row.original._path` (pipe-separated dimension values, e.g. `"North|||USA"`) and `row.original` contains all dimension values. The column coordinate for a pivot cell comes from the `colId` (which encodes the column dimension value in multi-level pivot mode). For drill-through, the row path uniquely identifies all row-dimension equality filters.

**When to use:** When the user clicks any cell in server-side mode, pass `row.original._path` and `rowFields` to the endpoint. For column-pivoted cells, also pass the column dimension value.

**How the path decodes to filters:**
- `_path = "North|||USA"` + `rowFields = ["region", "country"]`
- Becomes: `WHERE region = 'North' AND country = 'USA'`
- The endpoint splits `_path` on `|||` and zips with `rowFields`

### Anti-Patterns to Avoid

- **Routing drill-through through Dash callbacks:** The existing `register_dash_drill_modal_callback` path serializes all records as a Dash prop update. Do NOT use this for Phase 6 — it has a ~4MB limit and blocks the pivot callback channel. The new REST endpoint is the replacement.
- **Using `asyncio.run()` inside a Flask request in a context where an event loop is already running:** Dash's Flask server runs in a sync context, but `asyncio.run()` creates a fresh loop, which works. However, if running under an ASGI server, use `nest_asyncio` or convert `get_drill_through_data` to a sync wrapper. For the current dev server (`app.run(debug=True)`), `asyncio.run()` is safe.
- **Exporting from the rendered row cache only:** The current `exportExcel` function iterates over `rows` (the currently rendered/virtualized rows). For a complete export, all pivot data needs to be fetched, not just the visible viewport window. For Phase 6 scope, document this limitation: export exports what is currently rendered. A full-pivot export would require a separate server-side endpoint and is out of scope for Phase 6.
- **SheetJS `json_to_sheet` for multi-level headers:** `json_to_sheet` produces a flat single-row header. Use `aoa_to_sheet` (array-of-arrays) to build multi-level column headers correctly (EXPORT-02).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Excel file generation in browser | Custom OOXML builder | `xlsx` (SheetJS) ^0.18.5 — already in package.json | OOXML is 100+ KB of XML; SheetJS handles all cell types, merges, date formats |
| File download trigger | `<a href>` hack | `file-saver` `saveAs()` — already in package.json | `saveAs` handles cross-browser Blob download, including Safari quirks |
| DuckDB filtered query for drill-through | Custom SQL string builder | `get_drill_through_data` in `ScalablePivotController` + Ibis filter builder | Already implemented; uses Ibis parameter binding for safety |
| REST route on Dash app | Separate Flask/FastAPI server | `@app.server.route()` on the existing Dash Flask instance | Dash exposes `app.server` for exactly this purpose; no extra process, no CORS config needed |

**Key insight:** The SheetJS and file-saver libraries are already bundled into the webpack output. Adding new usage patterns (e.g., `aoa_to_sheet` for multi-level headers) costs zero bytes because the libraries are already imported.

---

## Common Pitfalls

### Pitfall 1: asyncio.run() Inside Flask Request Handler Deadlock

**What goes wrong:** `get_drill_through_data` is an `async` method. Calling `asyncio.run()` inside a Flask route works in the simple dev server but can cause "This event loop is already running" errors in some deployment configurations.

**Why it happens:** Dash's `app.run(debug=True)` uses Flask's dev server which runs synchronously. But Gunicorn/Uvicorn ASGI configs may already have a running loop.

**How to avoid:** Wrap `get_drill_through_data` in a synchronous helper that uses `asyncio.run()` when no loop is running, or refactor `get_drill_through_data` to have a sync variant using `loop.run_until_complete`. For Phase 6 (dev server target), `asyncio.run()` is safe.

**Warning signs:** `RuntimeError: This event loop is already running` in the Flask request context.

### Pitfall 2: _path Does Not Encode Column Coordinate for Pivoted Cells

**What goes wrong:** In column-pivoted mode (where `colFields` is non-empty), the column dimension values are encoded in the `colId` (e.g., `"2023-01_sales_sum"`), not in `_path`. A drill-through that only uses `_path` will return rows from all column values, not just the clicked cell's column intersection.

**Why it happens:** The row hierarchy (`_path`) encodes only row dimensions. Column dimensions are encoded in the column header ID.

**How to avoid:** Phase 6 scope (per DRILL-06): apply the cell's exact pivot coordinate filters. When `colFields` is non-empty, parse `colId` to extract column dimension value. For Phase 6, restrict to row-dimension filtering only (which covers the most common case) and document that column-dimension filtering is a follow-up. The endpoint still applies the `filters` prop state (global filters) via `spec.filters`.

**Warning signs:** Drill-through returns more rows than expected when column pivoting is active.

### Pitfall 3: Export Only Renders Visible Rows

**What goes wrong:** The current `exportExcel` function iterates over `rows` which is `tableData` — only the rows currently in the TanStack row model (the virtual scroll window contents). On a 2M-row dataset with only 50 rows rendered, the export will contain only 50 rows.

**Why it happens:** Virtual scroll means only a small window of data is ever in memory on the client.

**How to avoid:** For Phase 6, clearly scope: export exports the **currently loaded pivot view** (all rows in the block cache). This is a known limitation documented in the export feature. A full-dataset server-side export endpoint is a Phase 9+ concern. For the immediate phase, use `table.getRowModel().rows` (all rows in the table model, not just virtualized) to capture more than the viewport.

**Warning signs:** Downloaded file has far fewer rows than expected.

### Pitfall 4: drillEndpoint Prop Not Passed Through Dash Python Component

**What goes wrong:** The `drillEndpoint` prop is added to `PropTypes` in the React component but not added to the Python `DashTanstackPivot.py` component class definition. Dash's Python-side component only exposes props it knows about.

**Why it happens:** The Python component class is auto-generated from the JS component's PropTypes via `npm run build:py`. If only the JS PropTypes are updated without re-running the build, the Python component won't accept `drillEndpoint`.

**How to avoid:** After adding `drillEndpoint: PropTypes.string` to the JS component, run `npm run build` to regenerate `DashTanstackPivot.py` and `metadata.json`. Then add it to the Python class's `__init__` signature.

**Warning signs:** `TypeError: __init__() got an unexpected keyword argument 'drillEndpoint'` when instantiating the component from Python.

### Pitfall 5: SheetJS aoa_to_sheet Header Merge Complexity

**What goes wrong:** Multi-level column headers (EXPORT-02) require cell merges in the xlsx sheet. SheetJS requires explicitly setting `ws['!merges']` as an array of merge ranges. Getting this wrong produces misaligned headers.

**Why it happens:** `aoa_to_sheet` does not auto-merge spans — every cell is independent.

**How to avoid:** Build the header AOA (array of arrays) using the same `columns` tree that TanStack uses. For each parent column header, compute the span (number of leaf children) and add a merge object `{ s: {r, c}, e: {r, c+span-1} }`. Keep this logic in a `buildExportAoa` utility function separate from the main component.

**Warning signs:** Column headers in the exported file appear in wrong positions or span wrong cells.

---

## Code Examples

### Flask Drill-Through Endpoint (app.server pattern)
```python
# Source: Dash documentation — app.server is a Flask instance
# In dash_presentation/app.py

@app.server.route('/api/drill-through')
def api_drill_through():
    import asyncio
    from flask import request as flask_request, jsonify
    from pivot_engine.types.pivot_spec import PivotSpec

    table = flask_request.args.get('table', '')
    row_path = flask_request.args.get('row_path', '')
    row_fields_raw = flask_request.args.get('row_fields', '')
    page = int(flask_request.args.get('page', 0))
    page_size = min(int(flask_request.args.get('page_size', 500)), 500)
    sort_col = flask_request.args.get('sort_col')
    sort_dir = flask_request.args.get('sort_dir', 'asc')
    text_filter = flask_request.args.get('filter', '')

    row_fields = [f for f in row_fields_raw.split(',') if f]
    path_parts = row_path.split('|||') if row_path else []

    drill_filters = []
    for i, field in enumerate(row_fields):
        if i < len(path_parts) and path_parts[i]:
            drill_filters.append({'field': field, 'op': '=', 'value': path_parts[i]})

    if text_filter:
        drill_filters.append({'field': '__text_search__', 'op': 'contains', 'value': text_filter})

    spec = PivotSpec(table=table, rows=[], measures=[], filters=[])
    records = asyncio.run(
        get_adapter().controller.get_drill_through_data(
            spec, drill_filters, limit=page_size, offset=page * page_size
        )
    )
    return jsonify({'rows': records, 'page': page, 'page_size': page_size})
```

### React Drill-Through Modal Trigger (on cell click)
```javascript
// Source: existing context menu handler at line 1221 of DashTanstackPivot.react.js
// Extend the cell onClick to also trigger drill-through without context menu

const handleCellDrillThrough = useCallback((row, colId) => {
    const rowPath = row.original._path;
    if (!rowPath || rowPath === '__grand_total__') return;  // don't drill total rows
    const drillEndpoint = props.drillEndpoint || '/api/drill-through';
    const params = new URLSearchParams({
        table: tableName,
        row_path: rowPath,
        row_fields: rowFields.join(','),
        page: 0,
        page_size: 500,
    });
    setDrillModal({ loading: true, path: rowPath, rows: [], page: 0 });
    fetch(`${drillEndpoint}?${params}`)
        .then(r => r.json())
        .then(data => setDrillModal({ loading: false, path: rowPath, rows: data.rows, page: data.page }))
        .catch(() => setDrillModal(null));
}, [tableName, rowFields, props.drillEndpoint]);
```

### SheetJS aoa_to_sheet for Multi-Level Headers
```javascript
// Source: SheetJS documentation https://docs.sheetjs.com/docs/api/utilities/array
// Build array-of-arrays with header rows then data rows
function buildExportAoa(rows, columns) {
    // columns is the TanStack column definition tree
    const leafCols = [];
    const collectLeaves = col => {
        if (col.columns && col.columns.length) col.columns.forEach(collectLeaves);
        else leafCols.push(col);
    };
    columns.forEach(collectLeaves);

    // Row 1: parent headers (with spans via !merges added separately)
    // Row 2: leaf headers
    // Rows 3+: data
    const headerRow1 = leafCols.map(c => c.parent?.header ?? c.header ?? c.id);
    const headerRow2 = leafCols.map(c => c.header ?? c.id);
    const dataRows = rows.map(r => leafCols.map(c => r.getValue ? r.getValue(c.id) : r.original?.[c.accessorKey]));
    return [headerRow1, headerRow2, ...dataRows];
}
```

### CSV Export for Large Datasets
```javascript
// When rowCount > 500_000 — use Blob text/csv
const exportCSV = (rows, leafCols) => {
    const escape = v => {
        if (v == null) return '';
        const s = String(v);
        return s.includes(',') || s.includes('"') || s.includes('\n')
            ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const header = leafCols.map(c => escape(c.header ?? c.id)).join(',');
    const lines = rows.map(r =>
        leafCols.map(c => escape(r.original?.[c.accessorKey] ?? '')).join(',')
    );
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, 'pivot.csv');
};
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dash callback drill-through via `drill-data-store` prop | Direct REST endpoint called from `fetch()` in React | Phase 6 | Removes prop serialization limit; supports arbitrary row counts and pagination |
| `XLSX.utils.json_to_sheet` flat single-row header | `XLSX.utils.aoa_to_sheet` with header array + `!merges` | Phase 6 | Supports multi-level column headers per EXPORT-02 |
| Export from rendered rows only | Export from full row model + xlsx/csv threshold | Phase 6 | Prevents empty exports on virtual-scroll datasets |

**Deprecated/outdated:**
- `register_dash_drill_modal_callback` pattern: still wired in `app.py` but superseded for Phase 6 drill-through. Leave it in place (removing it would break existing wiring) but the new modal does not use it.

---

## Open Questions

1. **Does `get_drill_through_data` need a `sort_col`/`sort_dir` parameter for DRILL-05?**
   - What we know: current signature is `(spec, filters, limit, offset)`. No sort parameter exists.
   - What's unclear: whether the DuckDB query inside needs to be extended to accept an ORDER BY clause.
   - Recommendation: Add `sort: Optional[List[Dict]]` param to `get_drill_through_data` using the same Ibis sort pattern used in `IbisPlanner`. Pass from the endpoint.

2. **Text filter across all columns for DRILL-05 — how to implement efficiently?**
   - What we know: `get_drill_through_data` uses `builder.build_filter_expression` which handles `contains` filters on individual fields.
   - What's unclear: a global text filter needs to apply across all columns as an OR condition. This is not in the current filter builder.
   - Recommendation: For Phase 6, apply the text filter only to the first visible string column (or make `filter` param optional). Document that full cross-column text search is a follow-up.

3. **EXPORT-05 is in the roadmap success criteria but missing from REQUIREMENTS.md.**
   - What we know: ROADMAP.md lists it as success criterion 6 and REQUIREMENTS.md lists EXPORT-01 through EXPORT-04 only.
   - What's unclear: Whether to add EXPORT-05 to REQUIREMENTS.md as part of this phase.
   - Recommendation: The planner should add `EXPORT-05` to REQUIREMENTS.md under Phase 6 as part of Wave 0. The behavior is clearly specified in the ROADMAP.

4. **Where should `drillEndpoint` default to?**
   - What we know: The Flask route will be at `/api/drill-through`. In production Dash apps, the Dash app may be served at a sub-path.
   - What's unclear: Whether `window.location.origin` or a relative `/api/drill-through` is safer.
   - Recommendation: Default to the relative path `/api/drill-through`. This works for all standard Dash deployments. Make it configurable via `drillEndpoint` prop for non-standard paths.

---

## Validation Architecture

Config shows `workflow.nyquist_validation` key is absent — treating as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest ≥7.0.0 + pytest-asyncio ≥0.20.0 |
| Config file | `pivot_engine/pyproject.toml` (pytest config) |
| Quick run command | `python -m pytest tests/test_drill_through.py -x` |
| Full suite command | `python -m pytest tests/ -x --ignore=tests/test_frontend_contract.py` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DRILL-01 | Click on cell sets drill coordinate | manual-only | N/A — React event trigger requires browser | ❌ N/A |
| DRILL-02 | Modal renders with source rows | manual-only | N/A — React modal rendering requires browser | ❌ N/A |
| DRILL-03 | `/api/drill-through` endpoint accessible | integration | `pytest tests/test_drill_through.py::test_endpoint_returns_rows -x` | ❌ Wave 0 |
| DRILL-04 | Pagination returns correct page slices | unit | `pytest tests/test_drill_through.py::test_pagination -x` | ❌ Wave 0 |
| DRILL-05 | Sort and filter params applied | unit | `pytest tests/test_drill_through.py::test_sort_and_filter -x` | ❌ Wave 0 |
| DRILL-06 | Pivot coordinate filters applied by DuckDB | unit | `pytest tests/test_drill_through.py::test_coordinate_filters -x` | ❌ Wave 0 |
| EXPORT-01 | Export button triggers file download | manual-only | N/A — browser download API | ❌ N/A |
| EXPORT-02 | Multi-level headers in xlsx | unit | `pytest tests/test_export.py::test_xlsx_multi_level_headers -x` | ❌ Wave 0 |
| EXPORT-03 | Row hierarchy indentation in xlsx | unit | `pytest tests/test_export.py::test_xlsx_row_indentation -x` | ❌ Wave 0 |
| EXPORT-04 | Grand totals included in xlsx | unit | `pytest tests/test_export.py::test_xlsx_includes_totals -x` | ❌ Wave 0 |
| EXPORT-05 | xlsx/csv threshold logic (>500k → csv) | unit | `pytest tests/test_export.py::test_export_format_threshold -x` | ❌ Wave 0 |

Note: EXPORT-02, EXPORT-03, EXPORT-04, EXPORT-05 test helper functions (`buildExportAoa`, `exportCSV`) as pure JS logic. These are best tested via node unit tests or by extracting the logic to a utility module tested in a standalone JS test runner. For the Python test suite, test the backend endpoint (DRILL-03 through DRILL-06) only.

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_drill_through.py -x`
- **Per wave merge:** `python -m pytest tests/ -x --ignore=tests/test_frontend_contract.py`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_drill_through.py` — covers DRILL-03, DRILL-04, DRILL-05, DRILL-06
- [ ] `tests/test_export.py` — covers EXPORT-02, EXPORT-03, EXPORT-04, EXPORT-05 (pure Python logic tests for `buildExportAoa` equivalent if extracted, or backend export path)

---

## Sources

### Primary (HIGH confidence)
- Codebase direct inspection: `dash_presentation/app.py` — Flask `app.server` pattern confirmed (Dash app uses `app.server` for Flask instance)
- Codebase direct inspection: `pivot_engine/pivot_engine/tanstack_adapter.py` lines 626–656 — `handle_drill_through` and `get_drill_through_data` exist and are wired
- Codebase direct inspection: `pivot_engine/pivot_engine/runtime/dash_callbacks.py` lines 193–248 — existing Dash-callback drill-through path identified (NOT the Phase 6 target)
- Codebase direct inspection: `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js` lines 2646–2662 — current `exportExcel` is naive (iterates rendered rows, uses `json_to_sheet`, no csv branch)
- Codebase direct inspection: `dash_tanstack_pivot/package.json` — `xlsx ^0.18.5` and `file-saver ^2.0.5` already installed
- Codebase direct inspection: `DashTanstackPivot.react.js` line 12–13 — `import * as XLSX from 'xlsx'` and `import { saveAs } from 'file-saver'` already present
- Codebase direct inspection: `DashTanstackPivot.react.js` line 1221 — drill-through context menu action already builds `drillFilters` from row path; `setProps({ drillThrough: {...} })` is the old path
- `_path` encoding confirmed: `tanstack_adapter.py` lines 481–499 — `"|||"` separator, dimension values concatenated at each depth level
- `rowCount` prop confirmed: `DashTanstackPivot.react.js` line 91 and PropTypes line 4017 — already available in component scope

### Secondary (MEDIUM confidence)
- SheetJS `aoa_to_sheet` pattern for multi-level headers: https://docs.sheetjs.com/docs/api/utilities/array — standard SheetJS pattern for header rows with merges
- Dash `app.server` Flask route pattern: standard Dash pattern for adding REST endpoints without a separate server, documented in Plotly Dash docs

### Tertiary (LOW confidence)
- None — all critical findings verified from codebase inspection

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed; no new dependencies required
- Architecture: HIGH — existing `get_drill_through_data` and `handle_drill_through` verified in source; Flask `app.server` pattern confirmed by `app.py` reading `app.server` implicitly; `exportExcel` function fully read
- Pitfalls: HIGH — all pitfalls derived from direct code inspection (asyncio context, `_path` encoding, virtual scroll export limitation, PropTypes rebuild requirement)

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable libraries; SheetJS and Dash APIs are stable)
