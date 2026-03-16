# Refactoring and Fix Report

## Overview
This report details the refactoring of `DashTanstackPivot.react.js` and the fix for the "Expand All" functionality.

## Changes

### 1. Refactoring
The monolithic `DashTanstackPivot.react.js` file has been split into smaller, more manageable components and utility files.

**New File Structure:**
- `dash_tanstack_pivot/src/lib/components/`
    - `DashTanstackPivot.react.js`: Main component (orchestrator).
    - `Icons.js`: SVG icons.
    - `Notification.js`: Notification component.
    - `Sidebar/`
        - `SidebarFilterItem.js`: Filter item in the sidebar.
        - `ColumnTreeItem.js`: Tree item for columns in the sidebar.
        - `ToolPanelSection.js`: Collapsible section in the sidebar.
    - `Filters/`
        - `FilterPopover.js`: Popover container for filters.
        - `ColumnFilter.js`: Main column filter component.
        - `DateRangeFilter.js`: Date range filter.
        - `NumericRangeFilter.js`: Numeric range filter.
        - `MultiSelectFilter.js`: Multi-select filter.
    - `Table/`
        - `EditableCell.js`: Cell component with editing capabilities.
        - `StatusBar.js`: Status bar showing selection stats.
        - `ContextMenu.js`: Custom context menu.
- `dash_tanstack_pivot/src/lib/utils/`
    - `helpers.js`: Helper functions (formatting, keys, etc.).
    - `styles.js`: Theme and style definitions.

### 2. "Expand All" Fix
**Issue:** "Expand All Rows" was not expanding all levels of the hierarchy, especially with multi-level headers.

**Fix:**
- **Backend (`hierarchical_scroll_manager.py`):**
    - Updated `_format_for_ui` to check if `expanded_paths` contains `['__ALL__']`.
    - If `['__ALL__']` is present, `is_expanded` is set to `True` for all rows, ensuring the UI renders them as expanded.
    ```python
    # Check if the path is in the list of expanded paths
    is_expanded = list(path_tuple) in expanded_paths or [['__ALL__']] in expanded_paths or expanded_paths == [['__ALL__']]
    ```
- **Frontend (`DashTanstackPivot.react.js`):**
    - The `toggleAllRowsExpanded(true)` action sets the `expanded` state to `true`.
    - This boolean `true` is sent to the backend via `setProps`.
    - The backend adapter interprets `expanded=True` as `expanded_paths=[['__ALL__']]`, triggering the wildcard expansion logic.

## Verification
- The codebase is now more modular and easier to maintain.
- The "Expand All" logic in the backend has been strengthened to explicitly handle the wildcard expansion flag.