# Phase 7: Column Display & UI States - Research

**Researched:** 2026-03-15
**Domain:** TanStack Table v8 column pinning, sorting UI, visibility state, and column sizing state in a virtualized Dash React grid
**Confidence:** HIGH

## Summary

Phase 7 is mostly a frontend state and rendering correctness phase. The good news is the current codebase already has strong foundations for UI-01 through UI-05: pinning state exists, sticky rendering exists, sorting icons exist, a visibility panel exists, and resize handlers are wired. The main planning risk is not missing capability, but inconsistent state ownership and visual rules when states combine.

The highest leverage implementation choice is to make column sizing a first-class controlled state (same as sorting/pinning/visibility), then standardize visual tokens for "active sorted", "pinned boundary", and "resize affordance". TanStack already exposes stable APIs for these (`getIsPinned`, `getStart`, `getAfter`, `getIsSorted`, `getSortIndex`, `getVisibleLeafColumns`, `getSize`, `onColumnSizingChange`). Planning should align with those APIs instead of custom offset or visibility bookkeeping.

A second planning risk is validation: the repository currently has Python tests but no browser UI test harness, while these requirements are visual and interaction-heavy. Plan for a manual verification matrix at minimum, and preferably add a lightweight browser automation slice in Wave 0.

**Primary recommendation:** Treat Phase 7 as "state ownership + visual contract hardening": control column sizing state, unify pinned/sorted/resized styling rules, and add explicit verification for combined states.

## Planning Preconditions

- `ROADMAP.md` currently shows Phase 6 at `3/4` plans complete and Phase 7 depends on Phase 6.
- No phase-specific `*-CONTEXT.md` exists for Phase 7, so constraints come from `REQUIREMENTS.md` and `ROADMAP.md`.
- `CLAUDE.md` is not present.
- `.claude/skills` and `.agents/skills` are not present.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UI-01 | Pinned columns display a separator/shadow and remain fixed on horizontal scroll | Use TanStack pinning APIs (`getIsPinned`, `getStart`, `getAfter`, edge detection) and a single boundary shadow rule; current code already has sticky sections and boundary shadows |
| UI-02 | Sorted columns show asc/desc indicator and active sort column is visually distinct | Keep `getIsSorted` icon logic, add explicit "sorted-active" header styling token (background/border/weight) |
| UI-03 | Hidden columns toggled in visibility panel persist across filter/data refresh | Keep visibility panel toggles, ensure render paths use visible APIs only, and avoid resetting `columnVisibility` on structural refreshes |
| UI-04 | Resize handles appear on header hover; resized widths persist through scroll/refresh | Gate resize handle visibility by hover/focus state; move sizing to controlled state (`columnSizing` + `onColumnSizingChange`) and optionally persist |
| UI-05 | Combined states (pinned + sorted + resized) have no visual conflicts | Define deterministic z-index and style precedence matrix for headers/cells in combined states; verify left/right boundary shadows and sort styles do not clash |
| UI-06 | Default widths/heights are balanced and consistent | Replace scattered width/height literals with centralized table density/size tokens, then apply consistently to header and body renderers |

</phase_requirements>

## Standard Stack

### Core

| Library | Version (repo) | Purpose | Why Standard |
|---------|----------------|---------|--------------|
| `@tanstack/react-table` | `^8.10.7` | Column pinning, sorting, visibility, sizing state and table models | Official v8 APIs directly cover all required column UI states |
| `@tanstack/react-virtual` | `^3.0.0` | Horizontal/vertical virtualization | Already integrated; required so pinned/sized columns remain performant |
| React | `^16.8.6` | Component state and hooks | Existing app baseline; no migration needed for this phase |

### Supporting

| Library | Version (repo) | Purpose | When to Use |
|---------|----------------|---------|-------------|
| Dash component bridge (`setProps`) | current codebase | Round-trip state to Python callback layer | Keep for sorting/pinning/visibility events and backend sync |
| Browser storage (`local` / `session`) | current codebase | Persist pinning state across reloads | Extend only if Phase 7 decides to persist visibility/sizing beyond in-memory |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Continue custom sticky offset math | TanStack pinning offset APIs (`getStart`/`getAfter`) | TanStack APIs are less error-prone under resize/grouping changes |
| Keep sizing unmanaged (internal only) | Controlled sizing state in component | Controlled state adds code but makes persistence and debugging deterministic |
| Manual-only validation | Lightweight Playwright smoke suite | Browser automation adds setup cost but materially lowers UI regression risk |

**Installation:** No production dependency changes are required for implementation.  
Optional validation hardening may add a browser test dependency (see Validation Architecture Wave 0 gaps).

## Current Code Reality (What Planner Must Assume)

1. Sorting, column pinning, row pinning, and column visibility are already in controlled state (`DashTanstackPivot.react.js`).
2. Column sizing is used (`table.setColumnSizing`, `column.getSize`) but not represented in component state or Dash props.
3. Sticky pinning is implemented through split left/center/right sections plus `useStickyStyles` offset calculations.
4. Resize handle is always rendered for resizable headers; it is not currently hover-gated.
5. Visibility controls already exist in the column sidebar (`ColumnTreeItem`) via `column.toggleVisibility()`.
6. Width/height defaults are scattered numeric literals (e.g. `250`, `150`, `130`, `60`, row heights `[32, 40, 56]`).

## Architecture Patterns

### Pattern 1: Control Every User-Visible Column State

**What:** Keep sorting/pinning/visibility controlled and add controlled `columnSizing`.

**Why:** UI-04 requires persistence stability through interactions and refresh cycles. Controlled state gives deterministic behavior and easier persistence hooks.

**Use this shape:**

```javascript
const [columnSizing, setColumnSizing] = useState({});

const table = useReactTable({
  data,
  columns,
  state: {
    sorting,
    columnPinning,
    columnVisibility,
    columnSizing,
    // existing controlled state...
  },
  onColumnSizingChange: setColumnSizing,
  columnResizeMode: 'onChange',
  enableColumnResizing: true,
});
```

### Pattern 2: Use TanStack Pinning Geometry APIs for Sticky Offsets

**What:** Use `column.getStart('left')` / `column.getAfter('right')` and edge checks (`getIsLastColumn`, `getIsFirstColumn`) to compute pinned positioning and separator shadows.

**Why:** This is the official v8 sticky-pinning pattern and avoids drift when widths change.

```javascript
const getPinStyles = (column) => {
  const pinned = column.getIsPinned();
  return {
    position: pinned ? 'sticky' : 'relative',
    left: pinned === 'left' ? `${column.getStart('left')}px` : undefined,
    right: pinned === 'right' ? `${column.getAfter('right')}px` : undefined,
    boxShadow:
      column.getIsLastColumn('left')
        ? '-4px 0 4px -4px gray inset'
        : column.getIsFirstColumn('right')
          ? '4px 0 4px -4px gray inset'
          : undefined,
  };
};
```

### Pattern 3: Visibility Must Be Applied in Both Controls and Render Paths

**What:** Toggle with column APIs, and render only visible headers/cells (`getVisibleLeafColumns`, `row.getVisibleCells`, or left/center/right visible APIs).

**Why:** UI-03 depends on hiding being a first-class state, not a display-only CSS toggle.

### Pattern 4: Define a Sort Visual Contract Beyond Icons

**What:** Keep asc/desc icon indicators from `getIsSorted`, and add an explicit visual treatment for active sorted headers (for example: subtle background tint + border accent + text weight).

**Why:** UI-02 requires sorted columns to be visually distinct, not only iconized.

### Pattern 5: Hover/Focus-Gated Resize Affordance

**What:** Keep resize hit area attached to `header.getResizeHandler()`, but render handle at low/zero opacity by default and reveal on header hover/focus.

**Why:** Satisfies UI-04 and avoids accidental resize drags.

### Pattern 6: Centralize Sizing and Density Tokens

**What:** Move default widths and heights into one configuration object and reuse in column definitions and header/body row rendering.

**Why:** UI-06 requires balanced, consistent defaults. Current numeric literals are spread out, which creates drift.

### Anti-Patterns to Avoid

- Mixing custom sticky-offset arithmetic with TanStack pinning geometry in different render paths.
- Using non-visible column APIs (`getAllLeafColumns`, `getAllCells`) in render loops for table body/header.
- Keeping column sizing unmanaged while expecting reliable persistence semantics.
- Leaving resize handles always active and invisible (hard to discover, easy to trigger accidentally).
- Treating combined-state behavior as implicit; it must be explicitly styled and verified.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pinned offset math | Manual cumulative left/right calculations across render paths | TanStack `getStart` / `getAfter` / edge helpers | Handles resize and pin order safely |
| Sort state decoding | Custom sort-direction bookkeeping | `getIsSorted`, `getSortIndex`, toggle handlers | Removes state divergence bugs |
| Visibility render filtering | Ad hoc hidden-column filtering in JSX | TanStack visible-column/visible-cell APIs | Guarantees hidden columns stay hidden everywhere |
| Resize interaction plumbing | Custom pointer-move logic | `header.getResizeHandler`, built-in sizing state | Avoids drag math edge cases |

**Key insight:** This phase is mostly "correctly using existing TanStack primitives", not new algorithm design.

## Common Pitfalls

### Pitfall 1: Column Sizing Is Not Currently Controlled

**What goes wrong:** Width changes are not represented in Dash props or storage and can be harder to keep deterministic across structural updates.

**Why it happens:** Current table state excludes `columnSizing` even though `setColumnSizing` is called.

**How to avoid:** Add controlled sizing state and include it in persistence strategy.

**Warning signs:** Resized widths unexpectedly reset after structural reconfiguration or explicit reset flows.

### Pitfall 2: Resize Handle Discoverability

**What goes wrong:** Users miss resizing because the handle is always present but visually subtle.

**Why it happens:** Handle rendering is unconditional and lacks hover/focus affordance styling.

**How to avoid:** Add hover/focus-visible styles and minimum hit target.

**Warning signs:** Users report accidental or hard-to-find resizing.

### Pitfall 3: Pinned Reorder Intent Is Partially Implemented

**What goes wrong:** Tool panel drop logic computes a target index but pin handler currently ignores it.

**Why it happens:** `handlePinColumn` signature accepts only `(columnId, side)` while drop path passes an index.

**How to avoid:** Either implement indexed insertion fully or remove dead index flow to avoid false expectations.

**Warning signs:** Drag-drop pin order appears inconsistent or non-deterministic.

### Pitfall 4: Mobile Auto-Unpin Side Effect

**What goes wrong:** Right-pinned columns are automatically cleared below 768px.

**Why it happens:** Window resize effect force-removes right pinning.

**How to avoid:** Decide intentionally whether this behavior is acceptable for UI-01 in mobile widths; document and test.

**Warning signs:** Users report pinned-right columns "disappearing" on viewport resize.

### Pitfall 5: Combined State Layering Conflicts

**What goes wrong:** Pinned shadow, sorted highlighting, selection fill, and totals styling can compete for z-index/background.

**Why it happens:** Styles are composed in several places without a formal precedence matrix.

**How to avoid:** Define and enforce style precedence for header and body cell states.

**Warning signs:** Flickering shadows, unreadable active sort styling on pinned headers, clipped indicators.

## Code Examples

### Example 1: Controlled Sizing + Optional Persistence Hook

```javascript
const [columnSizing, setColumnSizing] = useState(() => {
  if (!persistence || !id) return {};
  const raw = window.localStorage.getItem(`${id}-columnSizing`);
  return raw ? JSON.parse(raw) : {};
});

useEffect(() => {
  if (!persistence || !id) return;
  window.localStorage.setItem(`${id}-columnSizing`, JSON.stringify(columnSizing));
}, [columnSizing, persistence, id]);

const table = useReactTable({
  data,
  columns,
  state: { ...tableState, columnSizing },
  onColumnSizingChange: setColumnSizing,
  enableColumnResizing: true,
  columnResizeMode: 'onChange',
});
```

### Example 2: Sorted-Active Header Visual Contract

```javascript
const sorted = header.column.getIsSorted(); // false | 'asc' | 'desc'
const sortedStyle = sorted
  ? {
      background: theme.select,
      borderBottom: `2px solid ${theme.primary}`,
      fontWeight: 700,
    }
  : {};

<div style={{ ...styles.headerCell, ...sortedStyle }}>
  {label}
  {sorted === 'asc' ? <Icons.SortAsc /> : sorted === 'desc' ? <Icons.SortDesc /> : null}
</div>
```

### Example 3: Hover-Revealed Resize Handle

```javascript
<div
  className="header-cell"
  style={{ position: 'relative' }}
>
  {content}
  {header.column.getCanResize() && (
    <div
      onMouseDown={header.getResizeHandler()}
      onTouchStart={header.getResizeHandler()}
      className="resize-handle"
    />
  )}
</div>
```

```css
.header-cell .resize-handle {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 8px;
  opacity: 0;
  cursor: col-resize;
}
.header-cell:hover .resize-handle,
.header-cell:focus-within .resize-handle {
  opacity: 1;
}
```

## State of the Art

| Old/Fragile Approach | Recommended Current Approach | Impact |
|----------------------|------------------------------|--------|
| Custom sticky offsets only | TanStack pinning geometry APIs | Lower risk of pinning drift under resize |
| Icon-only sort indication | Icon + explicit sorted-active header style | Meets UI-02 visual distinction requirement |
| Unmanaged sizing state | Controlled `columnSizing` with optional persistence | Better UI-04 reliability and debuggability |
| Scattered sizing literals | Centralized width/height tokens | Consistent UI-06 defaults |

## Open Questions

1. Should width persistence include browser reload, or only in-session refresh/filter changes?
   - Current requirement language guarantees refresh/filter persistence, not necessarily reload persistence.
2. Is auto-unpin of right-pinned columns on narrow screens still desired behavior for Phase 7?
   - This behavior may conflict with strict interpretation of UI-01.
3. Should Phase 7 include browser automation setup, or rely on manual UAT?
   - There is currently no frontend UI test runner in the repo.

## Validation Architecture

`workflow.nyquist_validation` is not explicitly set to `false` in `.planning/config.json`, so validation is treated as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | `pytest` (Python backend/integration), no frontend UI runner currently |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| Quick run command | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -x` |
| Full suite command | `python -m pytest tests -x` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UI-01 | Pinned shadow/separator and fixed behavior on horizontal scroll | manual visual/e2e | N/A (no browser runner yet) | No |
| UI-02 | Sort icon + active sorted style | manual visual + unit/e2e preferred | N/A (no browser runner yet) | No |
| UI-03 | Visibility toggle persists across filter/data refresh | integration/e2e | N/A (no browser runner yet) | No |
| UI-04 | Hover-only resize handles, width persistence | manual visual/e2e | N/A (no browser runner yet) | No |
| UI-05 | Combined pinned+sorted+resized has no conflicts | manual visual/e2e | N/A (no browser runner yet) | No |
| UI-06 | Defaults for widths/heights remain balanced and consistent | manual visual snapshot + targeted unit checks | N/A (no browser runner yet) | No |

### Sampling Rate

- **Per task commit:** run focused Python regression command above and perform targeted manual browser checks for affected UI states.
- **Per wave merge:** run full Python suite.
- **Phase gate:** all required manual checks passed plus Python suite green.

### Wave 0 Gaps

- [ ] Add a browser-level test harness (recommended: Playwright) for UI-01 through UI-05 regression checks.
- [ ] Add Phase 7 manual verification checklist document with explicit scenarios:
  - pin left/right + horizontal scroll
  - sorted pinned column
  - hide/show around filter and data refresh
  - resize + scroll + refresh
  - combined pinned+sorted+resized
- [ ] Add utility-level tests if sizing/pinning style computation is extracted to pure helper functions.

## Sources

### Primary (HIGH confidence)

- Local code inspection:
  - `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js`
  - `dash_tanstack_pivot/src/lib/hooks/useStickyStyles.js`
  - `dash_tanstack_pivot/src/lib/components/Sidebar/ColumnTreeItem.js`
  - `dash_tanstack_pivot/package.json`
  - `.planning/REQUIREMENTS.md`
  - `.planning/ROADMAP.md`
  - `.planning/STATE.md`
- TanStack official docs:
  - https://tanstack.com/table/latest/docs/guide/column-pinning
  - https://tanstack.com/table/v8/docs/framework/react/examples/column-pinning-sticky
  - https://tanstack.com/table/v8/docs/guide/column-visibility
  - https://tanstack.com/table/v8/docs/guide/column-sizing
  - https://tanstack.com/table/v8/docs/api/features/column-sizing
  - https://tanstack.com/table/v8/docs/guide/sorting

### Secondary (MEDIUM confidence)

- None required for critical claims.

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - dependencies and versions verified from local `package.json` and official TanStack docs.
- Architecture patterns: HIGH - patterns are directly supported by TanStack guides and align with current repository code paths.
- Pitfalls: MEDIUM-HIGH - most are codebase-verified; visual conflict risk remains partly inferential until browser verification.

**Research date:** 2026-03-15  
**Valid until:** 2026-04-15 (stable APIs, but frontend behavior should be re-validated after major TanStack or component refactors)

