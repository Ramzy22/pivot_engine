---
phase: 05-field-zone-ui
plan: 04
status: complete
completed_at: 2026-03-15
---

# Summary: 05-04 — Viewport Reset Identity Split

## What was done

Separated the server-side cache invalidation key from the viewport reset key so that row expansion/collapse no longer scrolls the user back to the top of the table.

Previously a single `serverSideCacheKey` drove both cache wipes and `scrollTop = 0` resets, meaning every expand/collapse jumped the viewport. The fix splits these into two distinct keys:
- `serverSideCacheKey` — includes `expanded` state; invalidates block cache on structural changes
- `serverSideViewportResetKey` — excludes `expanded`; only resets scroll on sort/filter/field changes

## Result

Expansion and collapse preserve the user's scroll position. Sort, filter, and field changes still reset to the top. No stale-row regression was introduced.
