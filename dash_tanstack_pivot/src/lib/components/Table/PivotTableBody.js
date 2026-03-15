import React from 'react';
import { isDarkTheme } from '../../utils/styles';
import SkeletonRow from '../SkeletonRow';
import StatusBar from './StatusBar';

/**
 * PivotTableBody — the main scroll container and virtual-scroll table body.
 *
 * Contains: sticky header (left/center/right), top-pinned rows, virtualized
 * center rows (with skeleton, expand/collapse loaders), bottom-pinned rows,
 * floating filters, column-loading skeletons, and the StatusBar.
 */
export function PivotTableBody({
    // Scroll container
    parentRef,
    handleKeyDown,
    rows,
    visibleLeafColumns,
    // Layout widths
    totalLayoutWidth,
    beforeWidth,
    afterWidth,
    bodyRowsTopOffset,
    stickyHeaderHeight,
    // Pinned / virtual rows
    effectiveTopRows,
    effectiveBottomRows,
    effectiveCenterRows,
    virtualRows,
    virtualCenterCols,
    // Virtualizers
    rowVirtualizer,
    // Row model
    rowModelLookup,
    getRow,
    serverSide,
    serverSidePinsGrandTotal,
    pendingRowTransitions,
    // Table instance / column groups
    table,
    leftCols,
    centerCols,
    rightCols,
    centerColIndexMap,
    visibleLeafIndexSet,
    // Display options
    rowHeight,
    showFloatingFilters,
    showColTotals,
    grandTotalPosition,
    showColumnLoadingSkeletons,
    pendingColumnSkeletonCount,
    columnSkeletonWidth,
    // Theme / styles
    theme,
    styles,
    // Render helpers
    renderCell,
    renderHeaderCell,
    // Filter interaction
    filters,
    handleHeaderFilter,
    // Status bar
    selectedCells,
    rowCount,
}) {
    return (
        <div style={styles.main}>
            <div
                ref={parentRef}
                style={{...styles.scrollContainer, overflow: 'auto'}}
                onKeyDown={handleKeyDown}
                tabIndex={0}
                role="grid"
                aria-rowcount={rows.length}
                aria-colcount={visibleLeafColumns.length}
            >
                 <div style={{width: `${totalLayoutWidth}px`, minWidth:'100%', height: `${rowVirtualizer.getTotalSize() + (effectiveTopRows.length + effectiveBottomRows.length) * rowHeight}px`, position: 'relative'}}>
                     {/* Sticky Header */}
                     <div style={{...styles.headerSticky, width: 'fit-content', display: 'flex'}} role="rowgroup">
                         {/* Left Section */}
                         <div style={{position: 'sticky', left: 0, zIndex: 4, background: theme.headerBg}}>
                             {table.getLeftHeaderGroups().map((group, level) => (
                                     <div key={group.id} style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`}}>
                                     {group.headers.map((header) => renderHeaderCell(header, level, 'left'))}
                                 </div>
                             ))}
                             {showFloatingFilters && (
                                 <div style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`, background: theme.background}}>
                                     {leftCols.map((column, idx) => (
                                         <div key={column.id} style={{...styles.headerCell, width: column.getSize(), height: rowHeight, padding: '2px 4px', borderRight: idx === leftCols.length - 1 ? `1px solid ${theme.border}` : 'none'}}>
                                             {column.id !== 'hierarchy' && (
                                                 <input
                                                     style={{width: '100%', fontSize: '11px', padding: '2px 4px', border: `1px solid ${theme.border}`, borderRadius: '2px'}}
                                                     placeholder="Filter..."
                                                     value={(filters[column.id] && filters[column.id].conditions && filters[column.id].conditions[0]) ? filters[column.id].conditions[0].value : ''}
                                                     onChange={e => handleHeaderFilter(column.id, {
                                                         operator: 'AND',
                                                         conditions: [{ type: 'contains', value: e.target.value, caseSensitive: false }]
                                                     })}
                                                     onClick={(e) => e.stopPropagation()}
                                                 />
                                             )}
                                         </div>
                                     ))}
                                 </div>
                             )}
                         </div>

                         {/* Center Section */}
                         <div style={{position: 'relative'}}>
                             {table.getCenterHeaderGroups().map((group, level) => {
                                 // Virtualize center headers: only render headers whose leaf
                                 // columns overlap the visible virtual column range, plus spacers.
                                 // centerColIndexMap and visibleLeafIndexSet are memoized above the render.
                                 const visibleHeaders = [];
                                 for (const header of group.headers) {
                                     const leafCols = header.column.getLeafColumns
                                         ? header.column.getLeafColumns()
                                         : [header.column];
                                     const centerLeafPairs = leafCols
                                         .map(lc => ({ col: lc, idx: centerColIndexMap.has(lc.id) ? centerColIndexMap.get(lc.id) : -1 }))
                                         .filter(p => p.idx >= 0);
                                     if (centerLeafPairs.length === 0) continue;
                                     const visiblePairs = centerLeafPairs.filter(p => visibleLeafIndexSet.has(p.idx));
                                     if (visiblePairs.length === 0) continue;
                                     const visWidth = visiblePairs.reduce((sum, p) => sum + p.col.getSize(), 0);
                                     visibleHeaders.push({ header, visWidth });
                                 }
                                 return (
                                     <div key={group.id} style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`}}>
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {visibleHeaders.map(({ header, visWidth }) =>
                                             renderHeaderCell(header, level, 'center', visWidth)
                                         )}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                     </div>
                                 );
                             })}
                             {showColumnLoadingSkeletons && (
                                 <div
                                     aria-hidden="true"
                                     style={{
                                         position: 'absolute',
                                         top: 0,
                                         right: 0,
                                         height: rowHeight,
                                         display: 'flex',
                                         alignItems: 'center',
                                         justifyContent: 'flex-end',
                                         gap: '8px',
                                         padding: '0 8px',
                                         pointerEvents: 'none',
                                         zIndex: 9
                                     }}
                                 >
                                     {Array.from({ length: pendingColumnSkeletonCount }).map((_, index) => (
                                         <div
                                             key={`col-header-skeleton-${index}`}
                                             style={{
                                                 width: `${columnSkeletonWidth}px`,
                                                 height: '60%',
                                                 borderRadius: '8px',
                                                 background: 'linear-gradient(90deg, #eef2fb 0%, #dbe8ff 45%, #eef2fb 100%)',
                                                 backgroundSize: '220% 100%',
                                                 border: `1px solid ${theme.border}`,
                                                 animation: 'pivot-row-loader-enter 220ms ease-out, pivot-skeleton-shimmer 1.25s ease-in-out infinite'
                                             }}
                                         />
                                     ))}
                                 </div>
                             )}
                             {showFloatingFilters && (
                                 <div style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`, background: theme.background}}>
                                     <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                     {virtualCenterCols.map(virtualCol => {
                                         const column = centerCols[virtualCol.index];
                                         if (!column) return null;
                                         return (
                                             <div key={column.id} style={{...styles.headerCell, width: column.getSize(), height: rowHeight, padding: '2px 4px'}}>
                                                 {column.id !== 'hierarchy' && (
                                                     <input
                                                         style={{width: '100%', fontSize: '11px', padding: '2px 4px', border: `1px solid ${theme.border}`, borderRadius: '2px'}}
                                                         placeholder="Filter..."
                                                         value={(filters[column.id] && filters[column.id].conditions && filters[column.id].conditions[0]) ? filters[column.id].conditions[0].value : ''}
                                                         onChange={e => handleHeaderFilter(column.id, {
                                                             operator: 'AND',
                                                             conditions: [{ type: 'contains', value: e.target.value, caseSensitive: false }]
                                                         })}
                                                         onClick={(e) => e.stopPropagation()}
                                                     />
                                                 )}
                                             </div>
                                         );
                                     })}
                                     <div style={{ width: afterWidth, flexShrink: 0 }} />
                                 </div>
                             )}
                         </div>

                         {/* Right Section */}
                         <div style={{position: 'sticky', right: 0, zIndex: 4, background: theme.headerBg}}>
                             {table.getRightHeaderGroups().map((group, level) => (
                                 <div key={group.id} style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`}}>
                                     {group.headers.map((header) => renderHeaderCell(header, level, 'right'))}
                                 </div>
                             ))}
                             {showFloatingFilters && (
                                 <div style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`, background: theme.background}}>
                                     {rightCols.map((column, idx) => (
                                         <div key={column.id} style={{...styles.headerCell, width: column.getSize(), height: rowHeight, padding: '2px 4px', borderLeft: idx === 0 ? `1px solid ${theme.border}` : 'none'}}>
                                             {column.id !== 'hierarchy' && (
                                                 <input
                                                     style={{width: '100%', fontSize: '11px', padding: '2px 4px', border: `1px solid ${theme.border}`, borderRadius: '2px'}}
                                                     placeholder="Filter..."
                                                     value={(filters[column.id] && filters[column.id].conditions && filters[column.id].conditions[0]) ? filters[column.id].conditions[0].value : ''}
                                                     onChange={e => handleHeaderFilter(column.id, {
                                                         operator: 'AND',
                                                         conditions: [{ type: 'contains', value: e.target.value, caseSensitive: false }]
                                                     })}
                                                     onClick={(e) => e.stopPropagation()}
                                                 />
                                             )}
                                         </div>
                                     ))}
                                 </div>
                             )}
                         </div>
                     </div>

                     {/* Top Pinned Rows */}
                     {effectiveTopRows.map((row, i) => {
                         const isExpandedRow = row.getIsExpanded();
                         const isLastPinnedTop = i === effectiveTopRows.length - 1;
                         const headerHeight = stickyHeaderHeight;
                         return (
                             <div
                                key={row.id}
                                role="row"
                                style={{
                                 ...styles.row,
                                 height: rowHeight,
                                 width: `${totalLayoutWidth}px`,
                                 position: 'sticky',
                                 top: headerHeight + (i * rowHeight),
                                 zIndex: 50, // Increased for top rows
                                 background: (row.original && row.original._isTotal) ? (isDarkTheme(theme) ? '#1a2e1a' : '#f0f7f0') : theme.background,
                                 borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                 boxShadow: isLastPinnedTop ? `0 2px 4px -2px ${theme.border}80` : 'none'
                             }}>
                                 {row.getLeftVisibleCells().map((cell) => renderCell(cell, i, false))}
                                 <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                 {virtualCenterCols.map(virtualCol => {
                                     const cell = row.getCenterVisibleCells()[virtualCol.index];
                                     return renderCell(cell, i, false);
                                 })}
                                 <div style={{ width: afterWidth, flexShrink: 0 }} />
                                 {row.getRightVisibleCells().map((cell) => renderCell(cell, i, false))}
                             </div>
                         )
                     })}

                     {showColumnLoadingSkeletons && (
                         <div
                             aria-hidden="true"
                             style={{
                                 position: 'absolute',
                                 top: `${bodyRowsTopOffset}px`,
                                 right: 0,
                                 height: `${Math.max(rowVirtualizer.getTotalSize(), rowHeight * 4)}px`,
                                 display: 'flex',
                                 gap: '8px',
                                 padding: '0 8px',
                                 pointerEvents: 'none',
                                 zIndex: 3
                             }}
                         >
                             {Array.from({ length: pendingColumnSkeletonCount }).map((_, index) => (
                                 <div
                                     key={`col-body-skeleton-${index}`}
                                     style={{
                                         width: `${columnSkeletonWidth}px`,
                                         height: '100%',
                                         borderRadius: '8px',
                                         background: 'linear-gradient(90deg, rgba(238,242,251,0.65) 0%, rgba(219,232,255,0.9) 45%, rgba(238,242,251,0.65) 100%)',
                                         backgroundSize: '220% 100%',
                                         border: `1px solid ${theme.border}55`,
                                         animation: 'pivot-row-loader-enter 220ms ease-out, pivot-skeleton-shimmer 1.25s ease-in-out infinite'
                                     }}
                                 />
                             ))}
                         </div>
                     )}

                     {/* Center Virtualized Rows */}
                     {virtualRows.map(virtualRow => {
                         const topOffset = bodyRowsTopOffset;

                         let row;

                         if (serverSide) {
                             // 1. Fetch Data Directly from Cache (Source of Truth)
                             const cachedData = getRow(virtualRow.index);

                             if (!cachedData) {
                                 // Data not loaded yet -> Skeleton
                                 return (
                                     <div
                                        key={`skeleton_${virtualRow.index}`}
                                        style={{
                                         ...styles.row,
                                         height: virtualRow.size,
                                         top: `${virtualRow.start + topOffset}px`,
                                         width: `${totalLayoutWidth}px`,
                                         position: 'absolute',
                                         background: theme.background,
                                         borderBottom: `1px solid ${theme.border}`,
                                         display: 'flex', alignItems: 'center'
                                     }}>
                                         <SkeletonRow style={{width: '100%'}} rowHeight={rowHeight} />
                                     </div>
                                 );
                             }

                             if (
                                 serverSidePinsGrandTotal &&
                                 (cachedData._isTotal || cachedData._path === '__grand_total__' || cachedData._id === 'Grand Total')
                             ) {
                                 return null;
                             }

                             // 2. Resolve Row Object via ID (Decoupled from Index)
                             // We reconstruct the ID exactly as getRowId does, but using the global index directly.
                             // Global Index = virtualRow.index
                             let rowId;
                             if (cachedData._isTotal || cachedData._path === '__grand_total__' || cachedData._id === 'Grand Total') {
                                 rowId = '__grand_total__';
                             } else {
                                 rowId = cachedData._path || (cachedData.id ? cachedData.id : String(virtualRow.index));
                             }

                             row = rowModelLookup.get(rowId);

                             // 3. Synchronization Check
                             // If table hasn't updated yet, table.getRow might return old data or undefined.
                             // We verify the row's data matches our cache.
                             const cachedPath = cachedData._isTotal ? '__grand_total__' : (cachedData._path || rowId);
                             const rowPath = row && row.original
                                 ? (row.original._isTotal ? '__grand_total__' : (row.original._path || row.id))
                                 : null;
                             if (row && rowPath !== cachedPath) {
                                 row = undefined; // Stale row object
                             }
                         } else {
                             // Client-side mode: simple index access
                             row = effectiveCenterRows[virtualRow.index];
                         }

                         // 4. Fallback: If row object is missing (even if we had cache), show skeleton
                         if (!row || !row.original) {
                              return (
                                 <div
                                    key={`skeleton_wait_${virtualRow.index}`}
                                    style={{
                                     ...styles.row,
                                     height: virtualRow.size,
                                     top: `${virtualRow.start + topOffset}px`,
                                     width: `${totalLayoutWidth}px`,
                                     position: 'absolute',
                                     background: (row && row.original && row.original._isTotal) ? (isDarkTheme(theme) ? '#1a2e1a' : '#f0f7f0') : theme.background,
                                     borderBottom: `1px solid ${theme.border}`,
                                     display: 'flex', alignItems: 'center'
                                 }}>
                                     <SkeletonRow style={{width: '100%'}} rowHeight={rowHeight} />
                                 </div>
                             );
                         }

                         // 5. Feature: Hide Totals if requested (Server Side only workaround)
                         if (serverSide && !showColTotals && row.original._isTotal) {
                             return null;
                         }

                          const isExpandedRow = row.getIsExpanded();
                          const pendingTransitionMode = pendingRowTransitions.get(row.id);
                          const showRowTransitionLoader = !!pendingTransitionMode;

                          // Stable key: use row path/id for loaded rows so expand/collapse
                          // does not remount rows that merely shifted index (AG Grid getRowId pattern).
                          const stableRowKey = serverSide
                              ? (row.id || String(virtualRow.index))
                              : String(virtualRow.index);

                          return (
                              <React.Fragment key={stableRowKey}>
                                  <div
                                     role="row"
                                     aria-rowindex={virtualRow.index}
                                     style={{
                                      ...styles.row,
                                      height: virtualRow.size,
                                      top: `${virtualRow.start + topOffset}px`,
                                      width: `${totalLayoutWidth}px`,
                                      background: (row.original && row.original._isTotal) ? (isDarkTheme(theme) ? '#1a2e1a' : '#f0f7f0') : theme.background,
                                      borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                      transition: rowVirtualizer.isScrolling ? 'none' : 'background-color 0.2s'
                                  }}>
                                      {row.getLeftVisibleCells().map((cell) => renderCell(cell, virtualRow.index, true))}
                                      <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                      {virtualCenterCols.map(virtualCol => {
                                          const cell = row.getCenterVisibleCells()[virtualCol.index];
                                          return renderCell(cell, virtualRow.index, true);
                                      })}
                                      <div style={{ width: afterWidth, flexShrink: 0 }} />
                                      {row.getRightVisibleCells().map((cell) => renderCell(cell, virtualRow.index, true))}
                                  </div>
                                  {showRowTransitionLoader && (
                                      <div
                                         role="row"
                                         aria-hidden="true"
                                         style={{
                                          ...styles.row,
                                          pointerEvents: 'none',
                                          height: rowHeight,
                                          top: `${virtualRow.start + topOffset + virtualRow.size}px`,
                                          width: `${totalLayoutWidth}px`,
                                          position: 'absolute',
                                          background: `linear-gradient(90deg, #f8fbff 0%, #eef5ff 50%, #f8fbff 100%)`,
                                          backgroundSize: '220% 100%',
                                          borderBottom: `1px dashed ${theme.border}`,
                                          display: 'flex',
                                          alignItems: 'center',
                                          justifyContent: 'flex-start',
                                          overflow: 'hidden',
                                          opacity: 0.95,
                                          zIndex: 18,
                                          boxShadow: `0 4px 12px -8px ${theme.border}`,
                                          animation: 'pivot-row-loader-enter 220ms ease-out, pivot-skeleton-shimmer 1.25s ease-in-out infinite'
                                      }}>
                                          <SkeletonRow style={{width: '100%', opacity: 0.45}} rowHeight={rowHeight} />
                                          <div
                                             style={{
                                              position: 'absolute',
                                              paddingLeft: `${((row.original && typeof row.original.depth === 'number' ? row.original.depth : row.depth || 0) + 1) * 24 + 8}px`,
                                              fontSize: '12px',
                                              color: theme.textSec,
                                              display: 'flex',
                                              alignItems: 'center',
                                              gap: '8px',
                                              fontWeight: 500
                                          }}
                                          >
                                              <span
                                                 aria-hidden="true"
                                                 style={{
                                                  width: '11px',
                                                  height: '11px',
                                                  border: `2px solid ${theme.primary}`,
                                                  borderTopColor: 'transparent',
                                                  borderRadius: '50%',
                                                  animation: 'pivot-spinner-rotate 0.75s linear infinite'
                                              }}
                                              />
                                              {pendingTransitionMode === 'collapse' ? 'Collapsing...' : 'Loading children...'}
                                          </div>
                                      </div>
                                  )}
                              </React.Fragment>
                          )
                      })}

                     {/* Spacer: only needed when grand total is pinned to bottom.
                          Virtual rows use position:absolute (out of flow), so without this spacer
                          the sticky bottom rows would sit at the top of the container and never
                          reach their sticky activation point. */}
                     {grandTotalPosition === 'bottom' && effectiveBottomRows.length > 0 && (
                         <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, flexShrink: 0 }} />
                     )}
                     {/* Bottom Pinned Rows */}
                    {effectiveBottomRows.map((row, i) => {
                         const isExpandedRow = row.getIsExpanded();
                         const isFirstPinnedBottom = i === 0;
                         return (
                             <div
                                key={row.id}
                                role="row"
                                style={{
                                 ...styles.row,
                                 height: rowHeight,
                                 width: `${totalLayoutWidth}px`,
                                 position: 'sticky',
                                 bottom: ((effectiveBottomRows.length - 1 - i) * rowHeight),
                                 zIndex: 50, // Increased for bottom rows
                                 background: (row.original && row.original._isTotal) ? (isDarkTheme(theme) ? '#1a2e1a' : '#f0f7f0') : theme.background,
                                 borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                 boxShadow: isFirstPinnedBottom ? `0 -2px 4px -2px ${theme.border}80` : 'none'
                             }}>
                                 {row.getLeftVisibleCells().map((cell) => renderCell(cell, i, false))}
                                 <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                 {virtualCenterCols.map(virtualCol => {
                                     const cell = row.getCenterVisibleCells()[virtualCol.index];
                                     return renderCell(cell, i, false);
                                 })}
                                 <div style={{ width: afterWidth, flexShrink: 0 }} />
                                 {row.getRightVisibleCells().map((cell) => renderCell(cell, i, false))}
                             </div>
                         )
                     })}
                 </div>
            </div>
            <StatusBar selectedCells={selectedCells} rowCount={rowCount} visibleRowsCount={rows.length} theme={theme} />
        </div>
    );
}
