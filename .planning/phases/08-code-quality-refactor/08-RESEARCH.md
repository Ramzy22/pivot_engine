# Phase 8: Code Quality Refactor - Research

**Researched:** 2026-03-15
**Domain:** React component splitting, error boundaries, stale closure hygiene, Python backend deduplication, Ibis parameter binding
**Confidence:** HIGH

---

## Summary

Phase 8 is a pure refactor — no new user-facing features. It has two independent work streams: (1) frontend JavaScript refactoring of the 4,338-line `DashTanstackPivot.react.js` and (2) backend Python cleanup in `controller.py` and `scalable_pivot_controller.py`.

The frontend has no error boundary at all today. The main component file is more than ten times the 400-line target (4,338 lines, 115 hook calls). The filter evaluation logic lives exclusively inside `DashTanstackPivot.react.js` as the `filteredData` useMemo block (lines 1285-1334) — it is not duplicated across files; the duplication risk is between the frontend condition-evaluation logic and any future server-side filter path. Sub-component infrastructure already exists (`Filters/`, `Sidebar/`, `Table/`, `hooks/`), so extraction work has natural landing zones.

The backend has a concrete, verified bug: `controller.py` defines `run_pivot_arrow()` twice at lines 657 and 683 (identical signatures, identical bodies). Python resolves this by silently using the second definition. The first definition should be deleted. The `scalable_pivot_controller.py` builds raw SQL UPDATE statements with manual string-quote-escaping instead of parameterized queries (lines 874-915, 945-996). Ibis does not natively support UPDATE, so the fix is DuckDB parameter binding via `?`-style placeholders or `$1` syntax via `con.execute(sql, [params])`.

**Primary recommendation:** Split frontend work into three sequential sub-plans (error boundary + persistence hook, filter logic extraction hook, JSX region extraction), and backend work into one sub-plan (remove duplicate + add parameterized UPDATE). Four plans total, each staying well under 400 lines of change.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CODE-01 | Main React component split into focused sub-components, each < 400 lines | File is 4,338 lines; existing sub-component dirs provide structure |
| CODE-02 | React error boundary wraps the table; crash shows error UI not blank screen | No ErrorBoundary exists anywhere in src/; must be created as class component |
| CODE-03 | All useEffect dependencies correct (no stale closures) | 29 useEffect calls found; setProps is already correctly ref-stabilized; risk area is event-listener effects that close over state without ref guards |
| CODE-04 | Filter logic in exactly one place — shared hook or utility | filteredData logic is in DashTanstackPivot.react.js only; extracting to useFilteredData hook eliminates future duplication risk |
| QUAL-03 | Duplicate run_pivot_arrow() in controller.py removed | Lines 657 and 683 both define identical methods; second definition silently wins |
| QUAL-04 | Column name sanitization uses Ibis parameter binding, eliminating SQL injection risk | update_cell and update_record use raw string interpolation; Ibis does not support UPDATE natively — use DuckDB `con.execute(sql, params)` |
</phase_requirements>

---

## Standard Stack

### Core (already in use — no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 18.x | Component model, hooks, class components for error boundaries | Already installed |
| @tanstack/react-table | 8.x | Table model — not affected by this refactor | Already installed |
| ibis-framework | 9.x | Ibis query building — `.filter()`, `.literal()` already used | Already installed |
| DuckDB (via ibis con) | in use | Raw parameterized SQL for UPDATE path | Already in use |

### No New Dependencies
Phase 8 adds zero new npm packages and zero new Python packages. React error boundaries use class components — a first-class React API requiring no library.

---

## Architecture Patterns

### Recommended Split Targets

The 4,338-line component has clear logical clusters. The extraction order matters: extract hooks first (no JSX changes), then JSX regions second (hooks already stable).

```
src/lib/
├── hooks/
│   ├── useServerSideRowModel.js     # EXISTS (581 lines)
│   ├── useStickyStyles.js           # EXISTS (132 lines)
│   ├── useColumnVirtualizer.js      # EXISTS
│   ├── useRowCache.js               # EXISTS
│   ├── usePersistence.js            # NEW: loadPersistedState / savePersistedState (lines 124-153)
│   └── useFilteredData.js           # NEW: filteredData useMemo (lines 1285-1334)
├── components/
│   ├── DashTanstackPivot.react.js   # TARGET: orchestrator only, < 400 lines
│   ├── PivotErrorBoundary.js        # NEW: React error boundary class component
│   ├── Filters/                     # EXISTS
│   ├── Sidebar/                     # EXISTS
│   └── Table/                       # EXISTS
```

### Pattern 1: React Error Boundary (class component)

**What:** React requires a class component with `componentDidCatch` + `getDerivedStateFromError` to catch render errors. Functional components cannot implement this.
**When to use:** Wrap the entire table render output so any child component crash is caught.

```jsx
// Source: React official docs - https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary
class PivotErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('[PivotErrorBoundary]', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 24, color: '#d32f2f', border: '1px solid #d32f2f', borderRadius: 4 }}>
          <strong>Pivot table error:</strong> {String(this.state.error?.message || 'Unknown error')}
        </div>
      );
    }
    return this.props.children;
  }
}
```

**Usage in DashTanstackPivot:**
```jsx
return (
  <PivotErrorBoundary>
    {/* existing table JSX */}
  </PivotErrorBoundary>
);
```

### Pattern 2: usePersistence Hook Extraction

**What:** Lines 124-153 contain `getStorage`, `loadPersistedState`, and `savePersistedState` — pure utility logic with no JSX dependency.
**When to use:** Extract to reduce DashTanstackPivot.react.js line count and make persistence logic independently testable.

```js
// hooks/usePersistence.js
export function usePersistence(id, persistence, persistence_type) {
  const getStorage = () => { ... };
  const load = (key, defaultValue) => { ... };
  const save = (key, value) => { ... };
  return { load, save };
}
```

### Pattern 3: useFilteredData Hook Extraction (CODE-04)

**What:** The `filteredData` useMemo (lines 1285-1334) is the single source of filter evaluation logic for client-side mode. Extract to a named hook so it can be imported by any component that needs to apply the same condition model.
**When to use:** Prevents future duplication if a sidebar or export path needs to evaluate filter conditions.

```js
// hooks/useFilteredData.js
// Source: extracted from DashTanstackPivot.react.js lines 1285-1334
export function useFilteredData(data, filters, serverSide) {
  return useMemo(() => {
    if (serverSide) return data || [];
    if (!data || !data.length) return [];
    return data.filter(row =>
      Object.entries(filters).every(([colId, filterGroup]) =>
        evaluateFilterGroup(row[colId], filterGroup)
      )
    );
  }, [data, filters, serverSide]);
}

// Pure function — unit-testable without React
export function evaluateFilterGroup(rowVal, filterGroup) { ... }
```

### Pattern 4: useEffect Stale Closure Audit (CODE-03)

**What:** 29 useEffect calls exist. The `setProps` stale closure is already correctly handled via `setPropsRef` (line 761). The remaining risk areas are event-listener effects that close over mutable state.

**Known safe:** setPropsRef pattern (lines 761-764) correctly uses ref to stabilize.

**Risk areas to audit:**
- Line 586-596: `handleMouseUp` closes over `isFilling`, `fillRange`, `dragStart` — all in dep array, correct.
- Line 598-610: `handleKeyDown` closes over `selectedCells` — in dep array, but creates new listener on every cell change. Pattern is correct but may re-add listeners frequently.
- Line 266-274: `handleGlobalKeyDown` — check dep array completeness.

**The standard fix for missing deps:**
```js
// BAD: stale closure — handler sees initial value forever
useEffect(() => {
  window.addEventListener('keydown', handler);
  return () => window.removeEventListener('keydown', handler);
}, []); // handler closes over stale state

// GOOD: use ref to get current value in stable handler
const stateRef = useRef(state);
useEffect(() => { stateRef.current = state; }, [state]);
useEffect(() => {
  const handler = (e) => { /* use stateRef.current */ };
  window.addEventListener('keydown', handler);
  return () => window.removeEventListener('keydown', handler);
}, []); // stable — handler reads from ref
```

### Pattern 5: Backend Duplicate Method Removal (QUAL-03)

**What:** `controller.py` has two identical `def run_pivot_arrow(self, spec)` definitions at lines 657 and 683. Python takes the second definition silently.

**Fix:** Delete lines 657-672 (the first definition). The docstring of the second definition (line 683) is already more accurate ("optimized for Arrow Flight operations"). Callers in `flight_server.py` (lines 39, 69) call `self._controller.run_pivot_arrow(spec)` — no change needed there.

**Regression guard:** Write a test asserting the class has exactly one `run_pivot_arrow` method:
```python
import inspect
methods = [m for name, m in inspect.getmembers(PivotController, predicate=inspect.isfunction) if name == 'run_pivot_arrow']
assert len(methods) == 1
```

### Pattern 6: Ibis Parameter Binding for UPDATE (QUAL-04)

**What:** `scalable_pivot_controller.py` `update_cell()` (line 945) and `update_record()` (line 874) build SQL UPDATE strings with manual escaping. The comment at line 950 explicitly acknowledges this is wrong.

**The target API:** DuckDB connection's `.execute()` accepts positional parameters:
```python
# Source: DuckDB docs - https://duckdb.org/docs/api/python/overview#querying
# BAD (current):
sql = f"UPDATE {table_name} SET {column} = '{escaped_value}' WHERE {id_col} = '{escaped_id}'"
con.execute(sql)

# GOOD (parameterized):
# Column identifiers cannot be parameterized — only values can.
# isidentifier() check for table/column names is still needed.
sql = f"UPDATE {table_name} SET {column} = ? WHERE {id_column} = ?"
con.execute(sql, [value, row_id])
```

**Critical constraint:** SQL identifiers (table name, column name) cannot be bound as parameters in any SQL dialect — only values can. The existing `isidentifier()` guard for table/column names is the correct approach for identifiers. Remove the manual quote-escaping only for VALUES.

**Ibis alternative check:** Ibis does not have a first-class `.update()` expression for DuckDB as of Ibis 9.x. The `raw_sql` / `execute(sql, params)` path via DuckDB connection is the correct approach and is already used by the codebase.

### Anti-Patterns to Avoid

- **Changing public prop interface:** Do not rename, add, or remove any Dash props during this refactor. `DashTanstackPivot.propTypes` must remain unchanged.
- **Moving state out of DashTanstackPivot:** State (`useState`) must stay in the main component. Extract only derived values (`useMemo`) and side effects (`useEffect` groups) into hooks.
- **Splitting JSX across files without clear boundaries:** Extract only self-contained render regions (Sidebar, Header, Body, StatusBar) that already have clear prop surfaces.
- **Extracting the entire return block at once:** The 4,338-line file's JSX is deeply nested. Extract one region per plan; attempt full extraction in one step risks merge conflicts and broken renders.
- **Deleting both run_pivot_arrow definitions:** Keep the second (line 683). Only delete the first (lines 657-672).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Error boundary state management | Custom try/catch in render | React class component with `getDerivedStateFromError` | React's error boundary protocol is the only way to catch render errors |
| SQL parameterization for values | Manual string escaping | `con.execute(sql, [value, row_id])` DuckDB positional params | Manual escaping is incomplete (handles `'` but not other edge cases like null bytes) |
| Filter condition evaluation | Duplicate logic in each component | `evaluateFilterGroup()` pure function in `useFilteredData.js` | Single source prevents condition evaluation drift between components |
| React hook testing | Custom test infrastructure | Jest + `@testing-library/react-hooks` (already available via React Testing Library) | renderHook from RTL is the standard approach |

**Key insight:** The hardest part of this phase is not the code changes — it is deciding where to draw the extraction boundaries without breaking the deeply-coupled state graph in the main component.

---

## Common Pitfalls

### Pitfall 1: Breaking the filteredData → useServerSideRowModel dependency chain
**What goes wrong:** `filteredData` is passed as `data` into `useServerSideRowModel` (line 1910). If the hook extraction changes referential identity (e.g., returns a new array on every render), it triggers infinite re-render loops.
**Why it happens:** `useMemo` with stable deps returns the same reference; a hook that wraps it incorrectly may not.
**How to avoid:** Ensure `useFilteredData` wraps the entire computation in a single `useMemo` with the same deps (`[data, filters, serverSide]`). Do not add intermediate state.
**Warning signs:** Browser tab freezes immediately after extraction; React DevTools shows infinite render cycles.

### Pitfall 2: useEffect dep array divergence after hook extraction
**What goes wrong:** After moving `filteredData` into a hook, any `useEffect` in the main component that lists `filteredData` in its dep array still works — but the dependency is now coming from the hook's return value, which must maintain stable identity.
**Why it happens:** Extracted hooks break referential stability if the memo is placed inside a render function instead of a hook.
**How to avoid:** Always return memoized values from custom hooks, never plain computed values.

### Pitfall 3: Python class having no method after deleting both definitions
**What goes wrong:** If an editor deletes lines 657-698 (first definition + surrounding context), it may accidentally delete `run_hierarchical_pivot_batch_load` at line 674.
**Why it happens:** The two `run_pivot_arrow` definitions are not adjacent — `run_hierarchical_pivot_batch_load` (lines 674-681) sits between them.
**How to avoid:** Delete only lines 657-672. Confirm line 673 is still `async def run_hierarchical_pivot_batch_load`.

### Pitfall 4: isidentifier() is not sufficient for all column names
**What goes wrong:** Column names like `"Order Date"` or `"Sales (USD)"` fail `isidentifier()` and get rejected, breaking the UPDATE path for real-world data.
**Why it happens:** `str.isidentifier()` only accepts Python/SQL bare identifiers; quoted identifiers are valid SQL but contain spaces or special chars.
**How to avoid:** For column identifiers, use `re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', col)` (no spaces, safe subset). Or use DuckDB's quoted identifier syntax: `f'"{col.replace(chr(34), chr(34)+chr(34))}"'` for the identifier itself. Values still go through `?` parameters.

### Pitfall 5: Error boundary not re-mounting on key change
**What goes wrong:** After a crash, the error boundary shows the error UI permanently — even after the user changes data or props — because the error state is sticky.
**Why it happens:** `getDerivedStateFromError` sets `hasError: true` but nothing resets it.
**How to avoid:** Pass `key={dataVersion}` (or similar) to `PivotErrorBoundary` so React unmounts and remounts the boundary (and clears error state) when data changes. Or add a "Retry" button that calls `this.setState({ hasError: false })`.

---

## Code Examples

### Error Boundary: Minimal Class Component
```jsx
// Source: https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary
// File: src/lib/components/PivotErrorBoundary.js
import React from 'react';

class PivotErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    if (process.env.NODE_ENV !== 'production') {
      console.error('[PivotErrorBoundary]', error, info.componentStack);
    }
  }

  render() {
    if (this.state.hasError) {
      const msg = this.state.error?.message || 'An unexpected error occurred.';
      return (
        <div style={{ padding: '16px', color: '#d32f2f', border: '1px solid #ffcdd2',
                      borderRadius: '4px', background: '#fff8f8', fontFamily: 'sans-serif' }}>
          <strong>Pivot table error</strong>
          <p style={{ marginTop: '8px', fontSize: '13px' }}>{msg}</p>
          <button onClick={() => this.setState({ hasError: false, error: null })}
                  style={{ marginTop: '8px', padding: '4px 12px', cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default PivotErrorBoundary;
```

### usePersistence Hook
```js
// hooks/usePersistence.js
import { useCallback } from 'react';

export function usePersistence(id, persistence, persistence_type) {
  const getStorage = useCallback(() => {
    if (persistence_type === 'local') return window.localStorage;
    if (persistence_type === 'session') return window.sessionStorage;
    return null;
  }, [persistence_type]);

  const load = useCallback((key, defaultValue) => {
    if (!persistence) return defaultValue;
    const storage = getStorage();
    if (!storage) return defaultValue;
    try {
      const saved = storage.getItem(`${id}-${key}`);
      return saved ? JSON.parse(saved) : defaultValue;
    } catch (e) {
      return defaultValue;
    }
  }, [id, persistence, getStorage]);

  const save = useCallback((key, value) => {
    if (!persistence) return;
    const storage = getStorage();
    if (!storage) return;
    try { storage.setItem(`${id}-${key}`, JSON.stringify(value)); } catch (e) { /* no-op */ }
  }, [id, persistence, getStorage]);

  return { load, save };
}
```

### useFilteredData Hook
```js
// hooks/useFilteredData.js
import { useMemo } from 'react';

export function evaluateFilterGroup(rowVal, filterGroup) {
  if (!filterGroup) return true;
  if (typeof filterGroup === 'string') {
    return String(rowVal).toLowerCase().includes(filterGroup.toLowerCase());
  }
  if (!filterGroup.conditions || filterGroup.conditions.length === 0) return true;
  const rStr = String(rowVal).toLowerCase();
  const passes = filterGroup.conditions.map(cond => {
    const val = cond.value;
    const vStr = String(val).toLowerCase();
    if (cond.type === 'in') return Array.isArray(val) && val.includes(rowVal);
    if (cond.type === 'contains') return rStr.includes(vStr);
    if (cond.type === 'startsWith') return rStr.startsWith(vStr);
    if (cond.type === 'endsWith') return rStr.endsWith(vStr);
    if (cond.type === 'eq' || cond.type === 'equals') return cond.caseSensitive ? String(rowVal) === String(val) : rStr === vStr;
    if (cond.type === 'ne' || cond.type === 'notEquals') return cond.caseSensitive ? String(rowVal) !== String(val) : rStr !== vStr;
    const rNum = Number(rowVal); const vNum = Number(val);
    if (!isNaN(rNum) && !isNaN(vNum)) {
      if (cond.type === 'gt') return rNum > vNum;
      if (cond.type === 'lt') return rNum < vNum;
      if (cond.type === 'gte') return rNum >= vNum;
      if (cond.type === 'lte') return rNum <= vNum;
      if (cond.type === 'between') return rNum >= vNum && rNum <= Number(cond.value2);
    }
    return true;
  });
  return filterGroup.operator === 'OR' ? passes.some(p => p) : passes.every(p => p);
}

export function useFilteredData(data, filters, serverSide) {
  return useMemo(() => {
    if (serverSide) return data || [];
    if (!data || !data.length) return [];
    return data.filter(row =>
      Object.entries(filters).every(([colId, filterGroup]) =>
        evaluateFilterGroup(row[colId], filterGroup)
      )
    );
  }, [data, filters, serverSide]);
}
```

### Removing Duplicate run_pivot_arrow (QUAL-03)
```python
# controller.py — BEFORE: two identical def run_pivot_arrow at lines 657 and 683
# ACTION: Delete lines 657-672 entirely.
# KEEP lines 683-697 (second definition with "optimized for Arrow Flight" docstring).
# KEEP lines 674-681 (run_hierarchical_pivot_batch_load) — sits between the two defs.

# Verification test:
import inspect
from pivot_engine.pivot_engine.controller import PivotController
arrow_methods = [name for name, _ in inspect.getmembers(PivotController, predicate=inspect.isfunction)
                 if name == 'run_pivot_arrow']
assert len(arrow_methods) == 1, f"Expected 1, got {len(arrow_methods)}"
```

### Parameterized UPDATE for update_cell (QUAL-04)
```python
# scalable_pivot_controller.py update_cell — BEFORE (lines 960-974):
# sql = f"UPDATE {table_name} SET {column} = '{escaped_value}' WHERE {id_column} = '{escaped_row_id}'"

# AFTER — identifiers still validated with isidentifier(), values use ? params:
if not table_name.isidentifier() or not column.isidentifier() or not id_column.isidentifier():
    raise ValueError("Invalid identifier in update request")
sql = f"UPDATE {table_name} SET {column} = ? WHERE {id_column} = ?"
# Value coercion: keep None as None (DuckDB maps Python None to NULL)
params = [value, row_id]

def execute_update():
    if hasattr(con, 'raw_sql'):
        con.raw_sql(sql, params)
    elif hasattr(con, 'execute'):
        con.execute(sql, params)
    elif hasattr(con, 'con'):
        con.con.execute(sql, params)
    else:
        raise NotImplementedError("Backend does not support parameterized SQL updates")
```

---

## What the File Split Looks Like in Practice

The 4,338-line component has its JSX return at **line 3161**. That means:
- Lines 1–3160: state declarations, hooks, handlers, memos (~3,160 lines of logic)
- Lines 3161–4338: JSX return tree (~1,177 lines of JSX)

The 29 useEffect calls and 115 hook invocations all live above line 3161. The most impactful extractions (by line count reduction) are:

| Extraction | Approx Lines Removed | Risk |
|------------|----------------------|------|
| usePersistence hook (lines 124-153) | ~30 lines | Very low — pure utility |
| useFilteredData hook (lines 1285-1334) | ~50 lines | Low — stable deps |
| Error boundary class component | +~40 lines new file, wraps existing JSX | Very low |
| Sidebar JSX to SidebarPanel component | ~400-600 lines | Medium — requires identifying prop surface |
| Header render to PivotHeader component | ~300-400 lines | Medium |
| Cell render callback (line 2871) to CellRenderer | ~80 lines | Low |

After hook extractions (~80 lines), the main file will be ~4,250 lines. To reach <400 lines, extensive JSX extraction is also required. The plan should sequence: (1) hooks + error boundary (safe, high-value), (2) sidebar extraction, (3) header extraction, (4) remaining body.

**Realistic target:** Each extracted JSX sub-component needs a clear prop interface. The Sidebar is already partially extracted (`SidebarFilterItem`, `ToolPanelSection`, `ColumnTreeItem`) but the sidebar render logic itself (lines ~3260-3400) still lives inline in the main component.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| React 16 error boundary must re-render to reset | React 18: pass `key` prop to boundary to reset on key change | React 18 | Use `key={dataVersion}` on PivotErrorBoundary |
| Manual SQL escaping for parameterized values | DuckDB `con.execute(sql, [params])` positional binding | DuckDB 0.9+ | Eliminates SQLi for value parameters |
| Monolithic component files | Custom hook extraction pattern | React 16.8 (hooks) | Standard — hooks share logic without HOCs |

**Deprecated/outdated:**
- `componentWillCatch` / `unstable_handleError`: Replaced by `componentDidCatch` + `getDerivedStateFromError`. Do not use.
- HOC-based error boundary wrappers (`react-error-boundary` library): Not needed here — a simple class component is sufficient and adds no dependency.

---

## Open Questions

1. **How far below 400 lines should CODE-01 target?**
   - What we know: Success criterion says "< 400 lines each" for sub-components. The main orchestrator file itself also needs to be < 400 lines.
   - What's unclear: Getting the main file from 4,338 to < 400 lines requires extracting nearly all JSX. This is achievable but requires 3-4 sequential plans.
   - Recommendation: Plan for a staged approach across 4 plans. Track line count after each plan; stop when the file is ≤ 400 lines.

2. **Does update_record (line 874) also need parameterized binding?**
   - What we know: QUAL-04 mentions "column name sanitization" but `update_record` also does value escaping via string interpolation.
   - What's unclear: Whether QUAL-04 scope includes `update_record` or only `update_cell`.
   - Recommendation: Fix both `update_cell` and `update_record` in the same plan — the pattern is identical and leaving one unfixed is inconsistent.

3. **Are there tests covering update_cell / update_record today?**
   - What we know: No test file in `tests/` or `pivot_engine/tests/` references `update_cell` or `update_record`.
   - What's unclear: Whether these methods are exercised at all in the current test suite.
   - Recommendation: Add a regression test before changing the implementation (TDD red-green approach, consistent with Phase 2 pattern).

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (pyproject.toml configured) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/ -x -q` |
| Full suite command | `python -m pytest tests/ pivot_engine/tests/ -q --continue-on-collection-errors` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CODE-01 | Main file < 400 lines | smoke | Assert `wc -l` in post-extraction check | ❌ Wave 0 |
| CODE-02 | Error boundary catches crash, shows error UI | unit (Jest) | `npm test -- --testPathPattern=PivotErrorBoundary` | ❌ Wave 0 |
| CODE-03 | useEffect deps correct, no stale closures | manual review + lint | `npx eslint --rule 'react-hooks/exhaustive-deps: error' src/` | ❌ Wave 0 |
| CODE-04 | Filter logic in one file | unit (Jest) | `npm test -- --testPathPattern=useFilteredData` | ❌ Wave 0 |
| QUAL-03 | Only one run_pivot_arrow method | unit | `python -m pytest tests/test_code_quality.py::test_no_duplicate_run_pivot_arrow -x` | ❌ Wave 0 |
| QUAL-04 | No raw string escaping in UPDATE paths | unit | `python -m pytest tests/test_code_quality.py::test_update_cell_parameterized -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/ -x -q` (Python tests) + `npm run build` (JS bundle sanity)
- **Per wave merge:** Full suite `python -m pytest tests/ pivot_engine/tests/ -q --continue-on-collection-errors`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_code_quality.py` — covers QUAL-03 (duplicate method assertion) and QUAL-04 (parameterized UPDATE assertion)
- [ ] `src/lib/components/PivotErrorBoundary.js` — new file, no test yet
- [ ] `src/lib/hooks/useFilteredData.js` — new file, no test yet
- [ ] `src/lib/hooks/usePersistence.js` — new file, no test yet
- [ ] eslint `react-hooks/exhaustive-deps` rule check — verify it is in `.eslintrc` or equivalent

---

## Sources

### Primary (HIGH confidence)
- React official docs (https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary) — error boundary class component API
- Direct file inspection of `controller.py` lines 657-697 — duplicate method confirmed
- Direct file inspection of `scalable_pivot_controller.py` lines 874-996 — string-interpolated UPDATE confirmed
- Direct file inspection of `DashTanstackPivot.react.js` — 4,338 lines confirmed, 29 useEffect calls confirmed, no ErrorBoundary found

### Secondary (MEDIUM confidence)
- DuckDB Python API docs (https://duckdb.org/docs/api/python/overview) — positional `?` parameter binding in `con.execute(sql, [params])`
- Ibis 9.x docs — UPDATE statement not natively supported; raw SQL path is documented approach

### Tertiary (LOW confidence)
- React hooks exhaustive-deps ESLint rule behavior — based on well-known community standard, not re-verified against current ESLint plugin version

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all findings from direct file inspection
- Architecture: HIGH — extraction targets confirmed from line-count audit; patterns from React official docs
- Pitfalls: HIGH — dependency chain and duplicate-method risks verified against actual source
- Backend fixes: HIGH — both issues confirmed at specific line numbers in source

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (stable React/DuckDB APIs; no expiry risk)
