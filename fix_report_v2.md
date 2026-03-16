# Virtual Scrolling Fix Report

## Critical Bugs Fixed

### 1. Rendering Logic & Index Mismatch
**Issue:** The rendering loop was using `localIndex` to access `centerRows`, but `centerRows` was being filtered (removing totals) while `renderedData` and the virtualizer expected a full dataset. This caused `centerRows[i]` to correspond to the wrong global index.
**Fix:** 
- Removed the filter `baseData.filter(r => !r._isTotal)` in `useMemo` when `serverSide` is true.
- Added logic in the render loop to return `null` for total rows if `showColTotals` is false. This preserves the index alignment while satisfying the visual requirement.

### 2. Data Synchronization Race Condition
**Issue:** There was a race condition where `centerRows` (derived from table state) might be stale compared to the cache. The existing check was insufficient or caused unnecessary skeleton flashes.
**Fix:** 
- Implemented a strict check: `if (row && row.original !== cachedRowData) { row = undefined; }`.
- This ensures that if the Table Model hasn't caught up to the Cache (or vice versa), we show a Skeleton instead of rendering the wrong row.

### 3. Data Rejection in Cache
**Issue:** `useRowCache` was rejecting incoming data because `requestVersion` was passed as `0` (untracked) but the block had a version `> 0` (from the loading state).
**Fix:** 
- Updated `setBlockLoaded` to allow updates if `requestVersion` is `0`, assuming the caller (Dash) implicitly validates the data via `dataOffset`.

### 4. Cache Efficiency
**Issue:** `getBlock` was performing a Map delete/set operation on every read (LRU), which is expensive for a virtualizer loop running 60fps.
**Fix:** 
- Removed the LRU update on read.

## Files Modified
- `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js`
- `dash_tanstack_pivot/src/lib/hooks/useRowCache.js`
