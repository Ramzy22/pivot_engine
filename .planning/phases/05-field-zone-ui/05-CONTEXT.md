# Phase 5: Field Zone UI - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

An interactive sidebar where users drag available fields into four labeled zones (Rows, Columns, Values, Filters). The configuration updates the pivot table immediately and round-trips to Python as a Dash prop. Creating/editing data and building the pivot engine are out of scope.

**Implementation status at context time:** Criteria 4, 5, 6 are fully implemented. Criteria 1, 2 are partially implemented (Rows/Columns/Values drag-drop works; Filters zone is missing from the drag-drop UI). Criterion 3 is implemented but missing min/max aggregation options.

</domain>

<decisions>
## Implementation Decisions

### Filters zone (HIGH priority — criterion 1 & 2 gap)
- Add Filters as a fourth drop zone in the drag-drop zones array (currently only `[rows, cols, vals]`)
- A field dropped into Filters should be treated the same way as adding a filter via the Filters tab
- Unreachable `zone.id==='filter'` conditionals in the zones loop must be made reachable (not removed)

### Aggregation types (HIGH priority — criterion 3 gap)
- Add `min` and `max` options to the Values zone aggregation dropdown
- Current options: sum, avg, count. Required: sum, avg, count, min, max

### Duplicate field prevention (MEDIUM priority — bug)
- When dropping a field into Rows or Columns, check if it already exists in the target zone before inserting
- Prevent the same field from being added twice to a single zone

### Empty state messages (MEDIUM priority — UX gap)
- Each zone that is empty should show a helper text (e.g. "Drag fields here") so users understand the zones are interactive drop targets

### Drag-drop error handling (MEDIUM priority — robustness)
- Wrap the `onDrop` handler logic in validation: check that `fieldName` is a non-empty string before mutating state
- Prevent invalid or malformed field objects from corrupting zone state

### Filter icon visual indicator (LOW priority — bug)
- The filter icon background uses `filters[label] ? theme.select : 'transparent'` — an empty string `''` is falsy
- Fix to use a more robust truthy check (e.g. check if the filter value has meaningful content)

### Claude's Discretion
- Exact helper text wording for empty zones
- Whether filter drop handling reuses `SidebarFilterItem` logic or adds a new code path
- Insertion validation approach (includes check vs Set dedup)

</decisions>

<specifics>
## Specific Ideas

- The Filters zone in the drag-drop area should feel consistent with Rows/Columns/Values — same chip UI, same remove button, same immediate pivot update
- When a field is dragged into Filters, it should behave equivalently to adding it via the Filters tab (the two paths should stay in sync)

</specifics>

<deferred>
## Deferred Ideas

- Accessibility labels and ARIA roles for drop zones — future a11y phase
- Format input field integration (the `Fmt` text field exists but is not wired to backend rendering) — future formatting phase
- Drag-drop insertion point visual polish (binary above/below calculation) — future UX polish phase

</deferred>

---

*Phase: 05-field-zone-ui*
*Context gathered: 2026-03-14*
