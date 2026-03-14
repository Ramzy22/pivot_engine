# Phase 4: Data Input API - Research

**Researched:** 2026-03-14
**Domain:** Python data source normalization — pandas, polars, Ibis, connection strings → PyArrow → DuckDB/Ibis backend
**Confidence:** HIGH

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| API-01 | Component accepts a pandas DataFrame as `data` prop | `pa.Table.from_pandas(df)` is HIGH-confidence path; `ibis.memtable(df)` also works. Existing `ensure_arrow_table()` utility handles this partially. |
| API-02 | Component accepts a polars DataFrame as `data` prop | `df.to_arrow()` (Polars native, mostly zero-copy) → Arrow Table → `load_data_from_arrow`. Polars is not currently in pyproject.toml dependencies. |
| API-03 | Component accepts an Ibis table expression as `data` prop | Already the internal representation. Use `expr.to_pyarrow()` to materialize → `load_data_from_arrow`, or register directly via `con.create_table`. |
| API-04 | Component accepts a connection string + table name as `data` prop | Parse connection string → `IbisBackend(connection_uri=...)` → `ibis.table(table_name)`. URI formats already partially handled in `ibis_backend.py`. |
| API-05 | Input type is auto-detected at runtime — same prop interface for all types | A `DataInputNormalizer` class with `isinstance` checks + duck-typing guards, returns `(table_name, arrow_table_or_ref)`. |
| API-06 | Meaningful error message shown when unsupported input type passed | Raise `DataInputError(unsupported_type, supported_types, hint)` instead of crashing. |
</phase_requirements>

---

## Summary

Phase 4 builds a **data normalization layer** in the Python backend that accepts heterogeneous data sources through a single `data` prop and converts them to the canonical internal representation (a named table in the DuckDB/Ibis connection). The pivot engine already speaks PyArrow natively — `load_data_from_arrow()` is the stable insertion point. All four input types (pandas, polars, Ibis expression, connection string) have well-defined, verified conversion paths to PyArrow or to a named Ibis table.

The key architectural insight is that this phase is **purely Python-side**: no React or Dash JS changes are needed. The work happens in a new `DataInputNormalizer` module that wraps the existing `TanStackPivotAdapter`. The Dash callback in `app.py` currently expects the adapter to already have data loaded; Phase 4 intercepts user-supplied `data` before the adapter sees it and performs normalization transparently.

The biggest risk is **optional dependency management**: polars is not in the current `pyproject.toml`, and ibis backends for non-DuckDB databases (postgres, snowflake, etc.) require separate backend packages. The normalizer must handle `ImportError` gracefully and produce clear error messages that tell the user which `pip install` command to run.

**Primary recommendation:** Implement a `DataInputNormalizer` class in `pivot_engine/data_input.py` that detects input type, converts to PyArrow, calls `controller.load_data_from_arrow(table_name, arrow_table)`, and raises `DataInputError` for unsupported types. Wire it into the `TanStackPivotAdapter` as a `load_data(data, table_name)` convenience method.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pyarrow | >=10.0.0 | Canonical internal data representation | Already the engine's native format; `load_data_from_arrow()` is the existing ingestion API |
| ibis-framework | >=4.0.0 | Query planning and backend abstraction | Already locked in the project; handles all SQL generation |
| duckdb | >=0.8.0 | In-memory OLAP engine | Already the default backend |
| pandas | >=1.5.0 | Source type for API-01 | Already in pyproject.toml; `pa.Table.from_pandas()` is the conversion path |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| polars | >=0.20.0 (optional) | Source type for API-02 | Only imported when user passes a polars DataFrame; `df.to_arrow()` converts to PyArrow |
| sqlalchemy | any (optional) | Connection string parsing fallback | Only needed for non-ibis connection strings; ibis handles most URI schemes natively |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyArrow as intermediate | `ibis.memtable()` directly | `memtable()` has backend constraints (cannot be named explicitly after ibis 9+) and produces anonymous tables; `load_data_from_arrow` + `con.create_table` gives a stable named table the planner can reference by string |
| Manual type detection | `pandas.api.types.is_dataframe_like()` | Duck-typing `isinstance` against known types is more explicit and produces better error messages |

**Installation (for polars support):**
```bash
pip install polars>=0.20.0
```

---

## Architecture Patterns

### Recommended Project Structure
```
pivot_engine/pivot_engine/
├── data_input.py        # NEW: DataInputNormalizer + DataInputError
├── controller.py        # EXISTING: load_data_from_arrow() — stable insertion point
├── tanstack_adapter.py  # EXISTING: add load_data(data, table_name) wrapper method
└── ...

pivot_engine/tests/
├── test_data_input.py   # NEW: unit tests for all input types + error path
└── ...
```

### Pattern 1: Normalizer as a Pure Conversion Function
**What:** A single `normalize_data_input(data, table_name, controller)` function (or class method) performs type detection and delegates to the appropriate converter. Returns nothing — side effect is that `table_name` now exists in the engine.
**When to use:** Always — called before any pivot query when `data` is not already loaded.

**Example:**
```python
# Source: project pattern derived from existing arrow_utils.py and controller.py
import pyarrow as pa

def normalize_data_input(data, table_name: str, controller) -> None:
    """Detect input type, convert to Arrow, load into controller."""
    if isinstance(data, pa.Table):
        arrow_table = data
    elif _is_pandas_dataframe(data):
        arrow_table = pa.Table.from_pandas(data, preserve_index=False)
    elif _is_polars_dataframe(data):
        arrow_table = data.to_arrow()
    elif _is_ibis_table(data):
        arrow_table = data.to_pyarrow()
    elif isinstance(data, dict) and "connection_string" in data:
        arrow_table = _load_from_connection(data["connection_string"], data["table"])
    else:
        raise DataInputError(
            f"Unsupported data type: {type(data).__name__}. "
            f"Supported types: pandas.DataFrame, polars.DataFrame, "
            f"ibis.Table, dict(connection_string=..., table=...), pa.Table."
        )
    controller.load_data_from_arrow(table_name, arrow_table)
```

### Pattern 2: Type Detection via Duck Typing (avoids hard imports)
**What:** Use `type(data).__module__` and `hasattr` checks before importing the library — so polars does not need to be installed if the user never passes a polars DataFrame.
**When to use:** All type checks in `DataInputNormalizer`.

```python
# Source: pattern established in existing arrow_utils.py line 29-33
def _is_pandas_dataframe(data) -> bool:
    """Check without requiring pandas to be imported at module level."""
    return type(data).__module__.startswith("pandas") and hasattr(data, "to_dict")

def _is_polars_dataframe(data) -> bool:
    return type(data).__module__.startswith("polars") and hasattr(data, "to_arrow")

def _is_ibis_table(data) -> bool:
    try:
        import ibis.expr.types as ibis_types
        return isinstance(data, ibis_types.Table)
    except ImportError:
        return False
```

### Pattern 3: Connection String → Ibis → Arrow
**What:** Parse a dict `{"connection_string": "duckdb://path.db", "table": "my_table"}` into an `IbisBackend`, read the table as an Arrow Table, load into the controller.
**When to use:** API-04 case.

```python
# Source: derived from existing ibis_backend.py URI handling patterns
def _load_from_connection(connection_string: str, table_name: str) -> pa.Table:
    from .backends.ibis_backend import IbisBackend
    backend = IbisBackend(connection_uri=connection_string)
    ibis_table = backend.con.table(table_name)
    return ibis_table.to_pyarrow()
```

### Anti-Patterns to Avoid
- **Importing pandas/polars at module top-level:** Forces user to install both even if they only use one. Use lazy imports inside detection functions.
- **Catching all exceptions silently:** Type detection failures must raise `DataInputError`, not return `None` or log silently. Silent failures produce confusing "table not found" errors later.
- **Storing the raw Python object:** Never store the pandas/polars object on the adapter — always convert to Arrow immediately and store the table name. This ensures the pivot engine's query pipeline is type-clean.
- **Re-loading data on every callback:** The normalizer should detect if a table is already loaded (by name/version) and skip re-ingestion unless the data has changed. Use a simple hash or id() check.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| pandas → Arrow conversion | Custom column-by-column conversion | `pa.Table.from_pandas(df, preserve_index=False)` | Handles all dtypes including nullable integers, categoricals, datetime with tz |
| polars → Arrow conversion | Iterate rows manually | `df.to_arrow()` (Polars native, mostly zero-copy) | Zero allocation copy for most column types; handles Polars-specific types |
| Ibis table → Arrow | Execute SQL manually | `ibis_table.to_pyarrow()` or `ibis_table.execute()` then `pa.Table.from_pandas()` | `to_pyarrow()` is zero-copy when backend supports Arrow IPC |
| Connection string parsing | Custom regex parser | `IbisBackend(connection_uri=...)` | Already implemented in `ibis_backend.py` for postgres, mysql, bigquery, snowflake, duckdb |
| Type registry | Manually checking class names as strings | `isinstance` + module prefix check | String-based class name checking breaks with subclasses and aliased imports |

**Key insight:** PyArrow is the universal intermediate format for this project. Every input type has an official, maintained, zero-copy-where-possible path to `pa.Table`. The engine's `load_data_from_arrow()` is the stable contract — normalize to that contract, not to any other internal format.

---

## Common Pitfalls

### Pitfall 1: pandas Index Leaks into Arrow Schema
**What goes wrong:** `pa.Table.from_pandas(df)` by default includes the DataFrame index as a column named `__index_level_0__` or similar, which then appears as a pivot dimension.
**Why it happens:** PyArrow preserves the index unless told not to.
**How to avoid:** Always call `pa.Table.from_pandas(df, preserve_index=False)`.
**Warning signs:** Unexpected `__index_level_0__` column appears in available field list.

### Pitfall 2: polars `.to_arrow()` Rechunking Failure
**What goes wrong:** On some polars versions (especially after concat operations), `df.to_arrow()` fails with an Arrow validation error.
**Why it happens:** Polars DataFrames may be multi-chunked after concat; Arrow expects single-chunk arrays.
**How to avoid:** Call `df.rechunk().to_arrow()` defensively.
**Warning signs:** `ArrowInvalid` or `rechunk` mentioned in the error trace.

### Pitfall 3: Ibis `memtable` Cannot Be Named (ibis >= 9.x)
**What goes wrong:** Code like `ibis.memtable(df, name="my_table")` raises an error in recent ibis versions.
**Why it happens:** Breaking change — `name` parameter removed from `memtable()`.
**How to avoid:** Never use `ibis.memtable()` for named ingestion. Use `con.create_table("name", arrow_table, overwrite=True)` instead. The current `load_data_from_arrow()` in `controller.py` already does this correctly.
**Warning signs:** `TypeError: memtable() got an unexpected keyword argument 'name'`.

### Pitfall 4: Connection String Backend Package Missing
**What goes wrong:** User passes `"postgres://..."` but `ibis-framework[postgres]` is not installed — raises `ModuleNotFoundError` deep in ibis internals with no helpful message.
**Why it happens:** Ibis backends are optional extras.
**How to avoid:** Catch `ModuleNotFoundError` in `_load_from_connection()` and re-raise as `DataInputError` with explicit install instructions (e.g., `pip install ibis-framework[postgres]`).
**Warning signs:** `ModuleNotFoundError: No module named 'psycopg2'` or similar.

### Pitfall 5: Dash Callback Serialization of Raw DataFrames
**What goes wrong:** User tries to pass a pandas DataFrame as a Dash prop — Dash serializes props to JSON, so DataFrames cannot be passed through `dcc.Store` or component props.
**Why it happens:** The `data` prop on `DashTanstackPivot` is typed as `list of dicts` (JSON). DataFrames cannot be stored in Dash's serialization layer.
**How to avoid:** The `data` prop normalization must happen **server-side** in the Python Dash callback before the component receives it. The normalizer lives in the callback, not in the component's prop handling. The Dash component's `data` prop remains `list of dicts` for virtual scroll; the DataFrame input is handled in the Python wrapper layer (e.g., a `PivotTable` helper class or callback utility).
**Warning signs:** Dash raises `TypeError: Object of type DataFrame is not JSON serializable`.

### Pitfall 6: Auto-Generated Table Names Collide
**What goes wrong:** Two pivot tables on the same page both load data with the same auto-generated table name, causing one to overwrite the other's data.
**Why it happens:** If no explicit table name is given, a naive normalizer might pick `"data"` every time.
**How to avoid:** Auto-generate stable table names from `id(data)` or a hash of the data object's identity, or require an explicit `table_name` parameter.

---

## Code Examples

Verified patterns from the existing codebase and official sources:

### Existing `load_data_from_arrow` (the stable insertion point)
```python
# Source: pivot_engine/pivot_engine/controller.py lines 699-724
def load_data_from_arrow(self, table_name: str, arrow_table: pa.Table, register_checkpoint: bool = True):
    if hasattr(self.planner, 'con') and hasattr(self.planner.con, 'create_table'):
        self.planner.con.create_table(table_name, arrow_table, overwrite=True)
    # ... fallbacks ...
    if hasattr(self, 'cache') and hasattr(self.cache, 'clear'):
        self.cache.clear()
```

### Pandas → Arrow → Load
```python
# Source: pa.Table.from_pandas official API, preserve_index from pyarrow docs
import pyarrow as pa
import pandas as pd

df = pd.DataFrame({"region": ["North", "South"], "sales": [100, 200]})
arrow_table = pa.Table.from_pandas(df, preserve_index=False)
controller.load_data_from_arrow("my_table", arrow_table)
```

### Polars → Arrow → Load
```python
# Source: polars.DataFrame.to_arrow official docs
import polars as pl

df = pl.DataFrame({"region": ["North", "South"], "sales": [100, 200]})
arrow_table = df.rechunk().to_arrow()  # rechunk() defensive for multi-chunk frames
controller.load_data_from_arrow("my_table", arrow_table)
```

### Ibis Table → Arrow → Load
```python
# Source: ibis to_pyarrow official API
import ibis

con = ibis.duckdb.connect()
ibis_table = con.table("existing_table")
arrow_table = ibis_table.to_pyarrow()
controller.load_data_from_arrow("my_table", arrow_table)
```

### Connection String → Arrow → Load
```python
# Source: derived from pivot_engine/pivot_engine/backends/ibis_backend.py
from pivot_engine.backends.ibis_backend import IbisBackend

data = {"connection_string": "duckdb://path/to/mydb.db", "table": "sales"}
backend = IbisBackend(connection_uri=data["connection_string"])
arrow_table = backend.con.table(data["table"]).to_pyarrow()
controller.load_data_from_arrow("my_table", arrow_table)
```

### DataInputError with actionable message
```python
# Source: project convention (new)
class DataInputError(TypeError):
    """Raised when the data= prop receives an unsupported type."""
    pass

def _unsupported_type_error(data) -> DataInputError:
    return DataInputError(
        f"Unsupported data type: {type(data).__module__}.{type(data).__name__}.\n"
        f"Supported types:\n"
        f"  - pandas.DataFrame (pip install pandas)\n"
        f"  - polars.DataFrame (pip install polars)\n"
        f"  - ibis.Table (pip install ibis-framework)\n"
        f"  - dict with keys 'connection_string' and 'table'\n"
        f"  - pyarrow.Table\n"
    )
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ibis.memtable(df, name="x")` | `con.create_table("x", arrow_table, overwrite=True)` | ibis 9.x (2024) | Named memtables no longer supported; must use create_table for named tables |
| `polars.to_pandas()` then pandas path | `polars.to_arrow()` directly | polars 0.20+ | Avoids double conversion; zero-copy where possible |
| `pd.DataFrame.to_dict("records")` for Dash data prop | Normalized server-side via `load_data_from_arrow` | Phase 4 (now) | Removes row-count limit imposed by JSON serialization |

**Deprecated/outdated:**
- `ibis.memtable(df, name="tablename")`: No longer works in ibis >= 9.x. Use `con.create_table()`.
- The existing `ensure_arrow_table()` in `arrow_utils.py` handles pandas but not polars, not ibis, not connection strings — Phase 4 replaces it as the canonical entry point.

---

## Open Questions

1. **Where does the data normalizer live in the Dash callback flow?**
   - What we know: `DashTanstackPivot` `data` prop is typed `list of dicts` (JSON) — raw DataFrames cannot go through it.
   - What's unclear: Should Phase 4 introduce a `PivotTableWrapper` Python class that accepts a DataFrame and manages loading? Or should it be a standalone `load_data()` utility the user calls in their callback?
   - Recommendation: Introduce a `load_data(adapter, data, table_name)` free function that users call once in their app setup or callback. This avoids modifying the Dash component's prop types.

2. **How should re-loading be detected to avoid redundant ingestion?**
   - What we know: `load_data_from_arrow` clears the cache every call — repeated identical calls are wasteful.
   - What's unclear: Should we hash the DataFrame content or use Python object identity (`id(df)`)?
   - Recommendation: Use `id(data)` with a small dict mapping `{id → table_name}` stored on the adapter. Simple, fast, correct for the Dash callback pattern where the same object is reused.

3. **polars version compatibility**
   - What we know: `df.to_arrow()` exists in polars >= 0.20.x; rechunk issue exists in some patch versions.
   - What's unclear: Minimum polars version to support without rechunk workaround.
   - Recommendation: Use `df.rechunk().to_arrow()` unconditionally — rechunk on an already single-chunk frame is a no-op.

---

## Validation Architecture

`workflow.nyquist_validation` is absent from config.json — treating as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 7.0.0 (already installed) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (root) |
| Quick run command | `pytest tests/test_data_input.py -x -q` |
| Full suite command | `pytest tests/ pivot_engine/tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| API-01 | pandas DataFrame loads and produces working pivot | unit | `pytest tests/test_data_input.py::test_pandas_dataframe -x` | ❌ Wave 0 |
| API-02 | polars DataFrame loads and produces working pivot | unit | `pytest tests/test_data_input.py::test_polars_dataframe -x` | ❌ Wave 0 |
| API-03 | Ibis table expression loads and produces working pivot | unit | `pytest tests/test_data_input.py::test_ibis_table -x` | ❌ Wave 0 |
| API-04 | Connection string + table name loads and produces working pivot | unit | `pytest tests/test_data_input.py::test_connection_string -x` | ❌ Wave 0 |
| API-05 | Same `normalize_data_input()` entry point handles all types | unit | `pytest tests/test_data_input.py::test_auto_detection -x` | ❌ Wave 0 |
| API-06 | Unsupported type raises DataInputError with actionable message | unit | `pytest tests/test_data_input.py::test_unsupported_type_error -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_data_input.py -x -q`
- **Per wave merge:** `pytest tests/ pivot_engine/tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_data_input.py` — covers API-01 through API-06; needs polars as optional dep (skip with `pytest.importorskip("polars")`)
- [ ] `pivot_engine/pivot_engine/data_input.py` — `DataInputNormalizer` class + `DataInputError`

*(Existing conftest.py at repo root handles sys.path — no new conftest needed)*

---

## Sources

### Primary (HIGH confidence)
- Existing codebase: `pivot_engine/pivot_engine/controller.py:699` — `load_data_from_arrow()` stable API
- Existing codebase: `pivot_engine/pivot_engine/util/arrow_utils.py` — `ensure_arrow_table()` pandas pattern
- Existing codebase: `pivot_engine/pivot_engine/backends/ibis_backend.py` — URI parsing patterns
- Existing codebase: `pivot_engine/pivot_engine/backends/duckdb_backend.py` — `create_table_from_arrow`
- [ibis expression-tables reference](https://ibis-project.org/reference/expression-tables) — `ibis.memtable()` accepts pandas, polars, pyarrow, dicts
- [ibis DuckDB backend docs](https://ibis-project.org/backends/duckdb) — `con.create_table(name, obj=df)` works for pandas and polars

### Secondary (MEDIUM confidence)
- [Polars Arrow producer/consumer guide](https://docs.pola.rs/user-guide/misc/arrow/) — `df.to_arrow()` zero-copy, rechunk workaround
- [Polars DataFrame.to_arrow API](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.to_arrow.html) — confirmed stable API
- WebSearch: ibis memtable breaking change in ibis 9.x — name parameter removed; `create_table` is the replacement

### Tertiary (LOW confidence)
- WebSearch: ibis 9.x memtable named table removal — confirmed direction but exact version number not pinned; validate against installed ibis version before implementation

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in pyproject.toml except polars; conversion APIs verified against official docs
- Architecture: HIGH — `load_data_from_arrow()` is the existing stable insertion point; normalizer pattern is straightforward
- Pitfalls: MEDIUM — pandas index pitfall and polars rechunk verified from official issues; Dash serialization pitfall is architectural fact; ibis memtable naming change is confirmed direction but exact version boundary is LOW confidence

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable APIs; polars/ibis may release minor updates but core conversion paths are stable)
