# Phase 07 UI State Verification Checklist

Use this checklist for final manual sign-off of column display UI behavior.
Mark each scenario as `Pass` or `Fail` and include notes/screenshots for any failure.

## Scenario Matrix

| Scenario | Expected Result | Status | Notes |
| --- | --- | --- | --- |
| UI-01 | Left/right pinned separators stay visible and correct during horizontal scroll | Pending | |
| UI-02 | Sort direction icons and active sorted emphasis remain clear and consistent | Pending | |
| UI-03 | Hide/show column choices persist through filter and data refresh cycles | Pending | |
| UI-04 | Resize handles show on hover/focus and resized widths persist through refresh | Pending | |
| UI-05 | Pinned + sorted + resized states can coexist (left and right pinned) without clipping/overlap | Pending | |
| UI-06 | Compact/normal/loose density modes keep consistent header/body/floating-filter spacing | Pending | |

## UI-01: Pinned Left/Right Separators During Scroll

Steps:
1. Pin one column to the left and one to the right.
2. Scroll horizontally through a wide column set.
3. Verify pinned columns remain fixed and center columns move beneath them.
4. Observe the boundary separator/shadow on both sides while scrolling.

Pass criteria:
- Pinned columns remain stationary.
- Boundary separator/shadow appears once per edge (no doubled borders/shadows).
- No overlap, clipping, or content bleed at pinned edges.

## UI-02: Sorting Indicators and Active Sorted Emphasis

Steps:
1. Sort a leaf column ascending, then descending.
2. Enable multi-sort (Shift + sort another column).
3. Observe icon direction and sort order index indicators.
4. Check sorted header emphasis while pinned and unpinned.

Pass criteria:
- Sort icons match active direction (asc/desc).
- Multi-sort indices are correct.
- Sorted-active header emphasis remains readable and aligned.

## UI-03: Hide/Show Persistence Through Filter and Refresh

Steps:
1. Hide at least two columns from different zones.
2. Apply a header or sidebar filter.
3. Trigger a data refresh.
4. Re-open column chooser and re-show hidden columns.

Pass criteria:
- Hidden columns stay hidden after filter and refresh.
- Re-shown columns return without layout corruption.
- Visibility state remains consistent with control panel state.

## UI-04: Resize Handle Visibility and Width Persistence

Steps:
1. Hover and focus a header cell to reveal resize handle.
2. Drag-resize multiple columns (including pinned columns).
3. Refresh data.
4. Revisit resized columns and interact with sort/filter controls.

Pass criteria:
- Resize handle appears on hover/focus and during drag.
- Dragging does not trigger unintended sorting.
- Resized widths persist after refresh.

## UI-05: Combined Pinned + Sorted + Resized States

Steps:
1. Choose one left-pinned column and one right-pinned column.
2. Resize each pinned column.
3. Apply sorting on both pinned columns (one asc, one desc if possible).
4. Scroll horizontally and vertically with floating filters enabled.
5. Repeat after toggling another non-pinned sorted column.

Pass criteria:
- Combined states render cleanly for both left and right pinned regions.
- No clipped sort icons, doubled shadows, or border artifacts.
- Header and body cells stay aligned while pinned/sorted/resized together.

## UI-06: Density and Default Dimension Consistency

Steps:
1. Switch spacing modes in order: Compact, Normal, Loose.
2. For each mode, inspect:
   - Header row heights
   - Body row heights
   - Floating filter row heights
   - Grouped header alignment
3. Refresh data in each spacing mode.

Pass criteria:
- Header/body/floating-filter heights remain internally consistent per mode.
- Column labels and controls stay vertically centered.
- No mode introduces clipping, overlap, or jitter after refresh.

## Sign-Off

- Reviewer:
- Date:
- Result: `Approved` / `Needs rework`
- Failing scenarios (if any):
