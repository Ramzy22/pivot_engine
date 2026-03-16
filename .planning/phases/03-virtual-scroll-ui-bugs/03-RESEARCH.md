---
phase: 03-virtual-scroll-ui-bugs
researched: 2026-03-13
status: complete
---

# Phase 3 Research

## Goal

Plan Phase 3 well enough to fix the remaining virtual-scroll and visible UI bugs without repeating Phase 2's backend correctness work.

Phase 3 requirements: `BUG-07`, `BUG-08`, `BUG-09`, `BUG-10`, `BUG-11`, `BUG-12`, `BUG-13`.

## Current Architecture

### Backend request and hierarchy path

- `pivot_engine/pivot_engine/tanstack_adapter.py` translates TanStack requests into `PivotSpec` objects and owns the hierarchical and virtual-scroll request handlers.
- `pivot_engine/pivot_engine/hierarchical_scroll_manager.py` is the main backend implementation for hierarchical visible-row calculation, cache keys, visible-row counts, query construction, and UI row shaping.
- `pivot_engine/pivot_engine/scalable_pivot_controller.py` mostly wires the adapter/controller path into the hierarchy manager.
- General cache behavior also touches `pivot_engine/pivot_engine/controller.py` and the cache implementations under `pivot_engine/pivot_engine/cache/`.

### Frontend rendering path

- `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js` owns most of the UI behavior for server-side virtualization, header rendering, row expansion, and context menu triggers.
- `dash_tanstack_pivot/src/lib/hooks/useServerSideRowModel.js` drives block fetching and row-window assembly.
- `dash_tanstack_pivot/src/lib/hooks/useRowCache.js` caches fetched blocks by block index only.
- `dash_tanstack_pivot/src/lib/hooks/useColumnVirtualizer.js` and `dash_tanstack_pivot/src/lib/hooks/useStickyStyles.js` affect header and pinned-column layout.
- `dash_tanstack_pivot/src/lib/components/Table/ContextMenu.js` clamps menu position, but only with fixed-size heuristics.

## Existing Coverage

### Automated coverage that already exists

- `tests/test_frontend_contract.py` covers adapter-level hierarchy and virtual-scroll request contracts.
- `tests/test_frontend_filters.py` covers repeated filter/sort request sequences.
- `test_expand_all_backend.py` covers expand-all behavior in the hierarchy manager.
- `pivot_engine/tests/test_hierarchical_managers.py` and `pivot_engine/tests/test_scalable_pivot.py` cover hierarchical manager and scalable-controller behavior.

### Coverage gap

- There is no meaningful direct frontend automation for:
  - block-cache invalidation in the React hook layer
  - header span/alignment geometry
  - row expansion UI identity/indent interactions
  - context-menu viewport positioning

This means Phase 3 should preserve strong backend contract tests while also isolating the riskiest frontend logic into smaller, testable units where practical. For the remaining pure-visual behaviors, manual smoke checks are still required.

## Main Hotspots

### BUG-07 and BUG-08: virtual-scroll desync and cache invalidation

- Frontend block cache is keyed only by block index in `useRowCache.js`.
- `DashTanstackPivot.react.js` intentionally avoids clearing the block cache for `expanded` changes, which is the highest-risk stale-row path under server-side hierarchy changes.
- The render path can temporarily show stale rows while revalidation is happening, which is useful for smoothness but risky when row identity or row count changes.
- Backend cache keys are stronger than the frontend block-cache contract, so Phase 3 likely needs the frontend cache to become request-signature-aware instead of relying on selective prop-based clearing.

### BUG-09 and BUG-10: header alignment and span widths

- Header rendering is split into left, center, and right sections in the main React component.
- Group headers rely on `header.getSize()` and sticky offset math even when child columns are pinned or virtualized differently.
- The highest-risk cases are:
  - group headers spanning children across pin boundaries
  - multi-depth column groups with center-column virtualization
  - placeholder/group headers whose visual width no longer equals the sum of visible children

### BUG-11 and BUG-12: row-group expansion correctness and sibling stability

- Expansion state is derived from a mix of local `expanded`, backend `_has_children`, backend `_is_expanded`, `_path`, and fallback row IDs.
- `tanstack_adapter.py` still contains request/response shaping logic that can be fragile for non-trivial hierarchies and generic row-field combinations.
- `hierarchical_scroll_manager.py` owns visible-row counting and paging math; stale or mismatched counts can shift siblings or produce blank areas after expand/collapse.

### BUG-13: context menu off-screen rendering

- `ContextMenu.js` uses hardcoded menu width and height assumptions.
- It only clamps right and bottom overflow.
- It does not clamp top/left, measure actual rendered menu size, or recompute after layout changes.

## Planning Implications

### Suggested execution order

1. Add/strengthen regressions first, especially around request sequences and hierarchy visibility math.
2. Fix backend hierarchy semantics and visible-row accounting before touching frontend smoothness logic.
3. Fix frontend block-cache invalidation and stale-row behavior after backend semantics are stable.
4. Fix header geometry and context-menu clamping last, then run a full verification pass.

### Files most likely to change

- `tests/test_frontend_contract.py`
- `tests/test_frontend_filters.py`
- `test_expand_all_backend.py`
- `pivot_engine/tests/test_hierarchical_managers.py`
- `pivot_engine/tests/test_scalable_pivot.py`
- `pivot_engine/pivot_engine/tanstack_adapter.py`
- `pivot_engine/pivot_engine/hierarchical_scroll_manager.py`
- `pivot_engine/pivot_engine/scalable_pivot_controller.py`
- `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js`
- `dash_tanstack_pivot/src/lib/hooks/useServerSideRowModel.js`
- `dash_tanstack_pivot/src/lib/hooks/useRowCache.js`
- `dash_tanstack_pivot/src/lib/hooks/useStickyStyles.js`
- `dash_tanstack_pivot/src/lib/components/Table/ContextMenu.js`

## Validation Architecture

### Quick automated suite

- `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py pivot_engine/tests/test_scalable_pivot.py -q --tb=line`

### Full automated suite

- `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short`

### Manual-only checks that still matter

- multi-level header groups remain visually aligned with their child columns while horizontally scrolling and pinning columns
- expand/collapse does not visibly shift or duplicate siblings in the rendered grid
- right-click context menus stay fully inside the viewport near all four edges

## Recommended Plan Split

- Plan 01: add regressions for hierarchy/virtual-scroll continuity and define verification targets
- Plan 02: fix backend hierarchy visibility, row identity, and expand/collapse semantics
- Plan 03: fix frontend block-cache invalidation and stale-row synchronization
- Plan 04: fix header geometry and context-menu placement, then run full verification

## Risks

- The frontend component is large and mixes unrelated responsibilities, so Phase 3 should avoid broad refactors that belong in Phase 7.
- Header alignment bugs can be hidden by certain datasets; final verification needs a deliberately grouped/pinned dataset, not only simple tables.
- Block-cache fixes that remove stale reuse entirely may regress perceived scroll smoothness; the target is correctness first, then bounded smoothness.
