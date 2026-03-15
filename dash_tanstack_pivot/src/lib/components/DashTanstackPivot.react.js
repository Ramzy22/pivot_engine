// DashTanstackPivot - Enterprise Grade Pivot Table
import React, { useMemo, useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import {
    useReactTable,
    getCoreRowModel,
    getExpandedRowModel,
    getGroupedRowModel,
    flexRender,
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';
import { themes, getStyles, isDarkTheme } from '../utils/styles';
import Icons from './Icons';
const debugLog = process.env.NODE_ENV !== 'production'
    ? (...args) => console.log('[pivot-grid]', ...args)
    : () => {};
import Notification from './Notification';
import useStickyStyles from '../hooks/useStickyStyles';
import { useServerSideRowModel } from '../hooks/useServerSideRowModel';
import { useColumnVirtualizer } from '../hooks/useColumnVirtualizer';
import SkeletonRow from './SkeletonRow';
import { formatValue, getKey, getAllLeafIdsFromColumn, isGroupColumn, hasChildrenInZone } from '../utils/helpers';
import FilterPopover from './Filters/FilterPopover';
import SidebarFilterItem from './Sidebar/SidebarFilterItem';
import ToolPanelSection from './Sidebar/ToolPanelSection';
import ColumnTreeItem from './Sidebar/ColumnTreeItem';
import ContextMenu from './Table/ContextMenu';
import EditableCell from './Table/EditableCell';
import StatusBar from './Table/StatusBar';
import DrillThroughModal from './Table/DrillThroughModal';

const getOrCreateSessionId = (componentId = 'pivot-grid') => {
    if (typeof window === 'undefined') {
        return `${componentId}-server-session`;
    }

    const storageKey = `${componentId}-client-session-id`;
    try {
        const fromStorage = window.sessionStorage.getItem(storageKey);
        if (fromStorage) return fromStorage;
    } catch (e) {
        // no-op: storage may be blocked in some browser privacy modes
    }

    let generated = null;
    if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        generated = window.crypto.randomUUID();
    } else {
        generated = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }

    try {
        window.sessionStorage.setItem(storageKey, generated);
    } catch (e) {
        // no-op
    }

    return generated;
};

const createClientInstanceId = (componentId = 'pivot-grid') => {
    if (typeof window !== 'undefined' && window.crypto && typeof window.crypto.randomUUID === 'function') {
        return `${componentId}-${window.crypto.randomUUID()}`;
    }
    return `${componentId}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
};

const loadingAnimationStyles = `
@keyframes pivot-row-loader-enter {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pivot-skeleton-shimmer {
    0% { background-position: 180% 0; }
    100% { background-position: -180% 0; }
}
@keyframes pivot-spinner-rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
`;

export default function DashTanstackPivot(props) {
    const { 
        id, 
        data = [], 
        style = {}, 
        setProps, 
        serverSide = false, 
        rowCount,
        rowFields: initialRowFields = [],
        colFields: initialColFields = [],
        valConfigs: initialValConfigs = [],
        filters: initialFilters = {},
        sorting: initialSorting = [],
        expanded: initialExpanded = {},
        showRowTotals: initialShowRowTotals = true,
        showColTotals: initialShowColTotals = true,
        grandTotalPosition = 'top',
        filterOptions = {},
        conditionalFormatting = [],
        validationRules = {},
        columnPinning: initialColumnPinning = { left: ['hierarchy'], right: [] },
        rowPinning: initialRowPinning = { top: [], bottom: [] },
        persistence,
        persistence_type = 'local',
        pinningOptions = {},
        pinningPresets = [],
        sortOptions = {},
        columnVisibility: initialColumnVisibility = {},
        reset,
        sortLock = false,
        availableFieldList,
        table: tableName,
        dataOffset = 0,
        dataVersion = 0,
        drillEndpoint = '/api/drill-through',
    } = props;



    // --- Persistence Helper ---
    const getStorage = () => {
        if (persistence_type === 'local') return window.localStorage;
        if (persistence_type === 'session') return window.sessionStorage;
        return null;
    };

    const loadPersistedState = (key, defaultValue) => {
        if (!persistence) return defaultValue;
        const storage = getStorage();
        if (!storage) return defaultValue;
        try {
            const saved = storage.getItem(`${id}-${key}`);
            return saved ? JSON.parse(saved) : defaultValue;
        } catch (e) {
            console.warn('Error loading persistence for', key, e);
            return defaultValue;
        }
    };

    const savePersistedState = (key, value) => {
        if (!persistence) return;
        const storage = getStorage();
        if (!storage) return;
        try {
            storage.setItem(`${id}-${key}`, JSON.stringify(value));
        } catch (e) {
            console.warn('Error saving persistence for', key, e);
        }
    };

        const [notification, setNotification] = useState(null);

        useEffect(() => {
            if (notification) {
                const timer = setTimeout(() => setNotification(null), 3000);
                return () => clearTimeout(timer);
            }
        }, [notification]);

        const showNotification = React.useCallback((msg, type='info') => {
            setNotification({ message: msg, type });
        }, []);

        // --- State ---

        const availableFields = useMemo(() => {
            if (availableFieldList && availableFieldList.length > 0) return availableFieldList;
            if (serverSide && props.columns) return props.columns.filter(c => c.id !== '__col_schema').map(c => c.id || c);

            return data && data.length ? Object.keys(data[0]) : [];

        }, [data, props.columns, serverSide, availableFieldList]);

        // Theme State
        const [themeName, setThemeName] = useState('light');
        const theme = useMemo(() => themes[themeName], [themeName]);
        const styles = useMemo(() => getStyles(theme), [theme]);

        const [rowFields, setRowFields] = useState(initialRowFields);
        const [colFields, setColFields] = useState(initialColFields);
        const [valConfigs, setValConfigs] = useState(initialValConfigs);
        const [filters, setFilters] = useState(initialFilters);
        const [sorting, setSorting] = useState(initialSorting);
        const [expanded, setExpanded] = useState(initialExpanded);
        const [columnPinning, setColumnPinning] = useState(() => loadPersistedState('columnPinning', initialColumnPinning));
        const [rowPinning, setRowPinning] = useState(() => loadPersistedState('rowPinning', initialRowPinning));
        const [layoutMode, setLayoutMode] = useState('hierarchy'); // hierarchy, tabular
        const [columnVisibility, setColumnVisibility] = useState(() => loadPersistedState('columnVisibility', initialColumnVisibility));
        const [columnSizing, setColumnSizing] = useState(() => loadPersistedState('columnSizing', {}));
        const [announcement, setAnnouncement] = useState("");
        const [drillModal, setDrillModal] = useState(null);
        // drillModal shape: { loading, rows, page, totalRows, path, sortCol, sortDir, filterText } | null
        const tableRef = useRef(null);

    // Reset Effect
    useEffect(() => {
        if (reset) {
            setRowFields(initialRowFields);
            setColFields(initialColFields);
            setValConfigs(initialValConfigs);
            setFilters({});
            setSorting([]);
            setExpanded({});
            setColumnPinning(initialColumnPinning);
            setRowPinning(initialRowPinning);
            setColumnVisibility({});
            setColumnSizing({});

            if (setPropsRef.current) {
                setPropsRef.current({
                    rowFields: initialRowFields,
                    colFields: initialColFields,
                    valConfigs: initialValConfigs,
                    filters: {},
                    sorting: [],
                    expanded: {},
                    columnPinning: initialColumnPinning,
                    rowPinning: initialRowPinning,
                    columnVisibility: {},
                    columnSizing: {},
                    reset: null
                });
            }
        }
    }, [reset, initialRowFields, initialColFields, initialValConfigs, initialColumnPinning, initialRowPinning]);

        // Save Persistence
        useEffect(() => {
            if (!persistence) return;
            savePersistedState('columnPinning', columnPinning);
            savePersistedState('rowPinning', rowPinning);
            savePersistedState('columnVisibility', columnVisibility);
            savePersistedState('columnSizing', columnSizing);
        }, [id, columnPinning, rowPinning, columnVisibility, columnSizing, persistence, persistence_type]);

        useEffect(() => {
            const handleResize = () => {
                if (window.innerWidth < 768 && columnPinning.right && columnPinning.right.length > 0) {
                     setColumnPinning(prev => ({ ...prev, right: [] }));
                     showNotification("Right pinned columns hidden due to screen size.", "warning");
                }
            };
            window.addEventListener('resize', handleResize);
            return () => window.removeEventListener('resize', handleResize);
        }, [columnPinning.right, showNotification]);

    const [showRowTotals, setShowRowTotals] = useState(initialShowRowTotals);
    const [showColTotals, setShowColTotals] = useState(initialShowColTotals);
    const [showRowNumbers, setShowRowNumbers] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [activeFilterCol, setActiveFilterCol] = useState(null);
    const [filterAnchorEl, setFilterAnchorEl] = useState(null);
    const [sidebarTab, setSidebarTab] = useState('fields'); // 'fields', 'columns'
    const [showFloatingFilters, setShowFloatingFilters] = useState(false);
    const [colSearch, setColSearch] = useState('');
    const [colTypeFilter, setColTypeFilter] = useState('all');
    const [selectedCols, setSelectedCols] = useState(new Set());
    const [hoveredHeaderId, setHoveredHeaderId] = useState(null);
    const [focusedHeaderId, setFocusedHeaderId] = useState(null);
    
    // Global Keyboard Shortcuts
    useEffect(() => {
        const handleGlobalKeyDown = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
                e.preventDefault();
                setSidebarOpen(prev => !prev);
            }
        };
        window.addEventListener('keydown', handleGlobalKeyDown);
        return () => window.removeEventListener('keydown', handleGlobalKeyDown);
    }, []);

    const [colorScale, setColorScale] = useState(false);
    const [spacingMode, setSpacingMode] = useState(0);
    const spacingLabels = ['Compact', 'Normal', 'Loose'];
    const rowHeights = [32, 40, 56];
    
    const [colExpanded, setColExpanded] = useState({});
    const [contextMenu, setContextMenu] = useState(null);
    const [selectedCells, setSelectedCells] = useState({});
    const [lastSelected, setLastSelected] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState(null);
    const [isFilling, setIsFilling] = useState(false);
        const [fillRange, setFillRange] = useState(null);

        // --- Data Management State ---
    const [history, setHistory] = useState([]);
    const [future, setFuture] = useState([]);

    const handleUndo = () => {
        if (history.length === 0) return;
        const previous = history[history.length - 1];
        setHistory(history.slice(0, -1));
        setFuture([selectedCells, ...future]);
        if (setProps) setProps({ undo: true, timestamp: Date.now() });
    };

    const handleRedo = () => {
        if (future.length === 0) return;
        const next = future[0];
        setFuture(future.slice(1));
        setHistory([...history, selectedCells]);
        if (setProps) setProps({ redo: true, timestamp: Date.now() });
    };

    const handleRefresh = () => {
        if (setProps) setProps({ refresh: Date.now() });
    };

    // Clipboard Paste
    useEffect(() => {
        const handlePaste = (e) => {
            if (!lastSelected) return;
            e.preventDefault();
            const clipboardData = e.clipboardData.getData('text');
            const rows = clipboardData.split(/\r\n|\n/).map(r => r.split('\t'));

            if (setPropsRef.current && lastSelected.rowIndex !== undefined && lastSelected.colIndex !== undefined) {
                const visibleLeafColumns = (tableRef.current && tableRef.current.getVisibleLeafColumns) ? tableRef.current.getVisibleLeafColumns() : [];
                const visibleRows = (tableRef.current && tableRef.current.getRowModel) ? (tableRef.current.getRowModel().rows || []) : [];

                const startRow = visibleRows[lastSelected.rowIndex];
                const startCol = visibleLeafColumns[lastSelected.colIndex];

                if (startRow && startCol) {
                    setPropsRef.current({
                        paste: {
                            startRowId: startRow.id,
                            startColId: startCol.id,
                            data: rows
                        }
                    });
                }
            }
        };

        window.addEventListener('paste', handlePaste);
        return () => window.removeEventListener('paste', handlePaste);
    }, [lastSelected]);

    // Validation helper
    const validateCell = (val, rule) => {
        if (!rule) return true;
        if (rule.type === 'regex') return new RegExp(rule.pattern).test(val);
        if (rule.type === 'numeric') return !isNaN(parseFloat(val));
        if (rule.type === 'required') return val !== null && val !== '' && val !== undefined;
        return true;
    };

    const getConditionalStyle = (colId, value) => {
        if (typeof value !== 'number') return {};
        const rules = conditionalFormatting.filter(r => r.column === colId || !r.column);
        let style = {};
        for (const rule of rules) {
            let match = false;
            if (rule.condition === '>') match = value > rule.value;
            else if (rule.condition === '<') match = value < rule.value;
            else if (rule.condition === '>=') match = value >= rule.value;
            else if (rule.condition === '<=') match = value <= rule.value;
            else if (rule.condition === '==') match = value === rule.value;

            if (match) {
                style = { ...style, ...rule.style };
            }
        }
        return style;
    };

    const handleKeyDown = (e) => {
        if (!lastSelected) return;

        const { rowIndex, colIndex } = lastSelected;
        let nextRow = rowIndex;
        let nextCol = colIndex;

        // Get visibleLeafColumns from tableRef
        const visibleLeafColumns = (tableRef.current && tableRef.current.getVisibleLeafColumns) ? tableRef.current.getVisibleLeafColumns() : [];

        if (e.key === 'ArrowUp') nextRow = Math.max(0, rowIndex - 1);
        else if (e.key === 'ArrowDown') nextRow = Math.min(rows.length - 1, rowIndex + 1);
        else if (e.key === 'ArrowLeft') nextCol = Math.max(0, colIndex - 1);
        else if (e.key === 'ArrowRight') nextCol = Math.min(visibleLeafColumns.length - 1, colIndex + 1);
        else if (e.key === 'Tab') {
            e.preventDefault();
            nextCol = e.shiftKey ? Math.max(0, colIndex - 1) : Math.min(visibleLeafColumns.length - 1, colIndex + 1);
        } else if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
             // Handled by another useEffect, but can be here too
             return;
        } else {
            return;
        }

        const nextRowObj = rows[nextRow];
        const nextColObj = visibleLeafColumns[nextCol];

        if (nextRowObj && nextColObj) {
            const key = `${nextRowObj.id}:${nextColObj.id}`;
            const val = nextRowObj.getValue(nextColObj.id);

            if (e.shiftKey && e.key.startsWith('Arrow')) {
                 const newRange = selectRange(dragStart || lastSelected, { rowIndex: nextRow, colIndex: nextCol });
                 setSelectedCells(newRange);
            } else {
                 setSelectedCells({ [key]: val });
                 setDragStart({ rowIndex: nextRow, colIndex: nextCol });
            }
            setLastSelected({ rowIndex: nextRow, colIndex: nextCol });

            // Scroll into view if needed
            if (rowVirtualizer.scrollToIndex) rowVirtualizer.scrollToIndex(nextRow);
            if (columnVirtualizer.scrollToIndex) columnVirtualizer.scrollToIndex(nextCol);
        }
    };

    const selectRange = (start, end) => {
        const rStart = Math.min(start.rowIndex, end.rowIndex);
        const rEnd = Math.max(start.rowIndex, end.rowIndex);
        const cStart = Math.min(start.colIndex, end.colIndex);
        const cEnd = Math.max(start.colIndex, end.colIndex);

        const visibleRows = table.getRowModel().rows;
        const visibleCols = table.getVisibleLeafColumns();
        const newSelection = {};

        for (let r = rStart; r <= rEnd; r++) {
            for (let c = cStart; c <= cEnd; c++) {
                const rRow = visibleRows[r];
                const cCol = visibleCols[c];
                if (rRow && cCol) {
                    newSelection[`${rRow.id}:${cCol.id}`] = rRow.getValue(cCol.id);
                }
            }
        }
        return newSelection;
    };

    const [isRowSelecting, setIsRowSelecting] = useState(false);
    const [rowDragStart, setRowDragStart] = useState(null);

    // Stop row selection on mouse up
    useEffect(() => {
        const handleMouseUp = () => {
            setIsRowSelecting(false);
            setRowDragStart(null);
        };
        window.addEventListener('mouseup', handleMouseUp);
        return () => window.removeEventListener('mouseup', handleMouseUp);
    }, []);

    const handleRowRangeSelect = useCallback((startIdx, endIdx) => {
        if (!tableRef.current) return;
        const visibleCols = tableRef.current.getVisibleLeafColumns();
        const rows = tableRef.current.getRowModel().rows;
        const min = Math.min(startIdx, endIdx);
        const max = Math.max(startIdx, endIdx);
        
        const rangeSelection = {};
        for(let i=min; i<=max; i++) {
            const r = rows[i];
            if(r) {
                visibleCols.forEach(col => {
                    rangeSelection[`${r.id}:${col.id}`] = r.getValue(col.id);
                });
            }
        }
        // Merge with existing if ctrl held? No, drag usually replaces or extends from anchor.
        // For simplicity, let's just set selection to this range.
        setSelectedCells(rangeSelection);
    }, []);

    const handleRowSelect = useCallback((row, isShift, isCtrl) => {
        if (!tableRef.current) return;
        const visibleCols = tableRef.current.getVisibleLeafColumns();
        const rowId = row.id;
        const newSelection = {};
        
        visibleCols.forEach((col) => {
            newSelection[`${rowId}:${col.id}`] = row.getValue(col.id);
        });

        if (isCtrl) {
            setSelectedCells(prev => ({...prev, ...newSelection}));
            setLastSelected({ rowIndex: row.index, colIndex: 0 });
        } else if (isShift && lastSelected) {
             const startRowIndex = lastSelected.rowIndex;
             const endRowIndex = row.index;
             const rows = tableRef.current.getRowModel().rows;
             const min = Math.min(startRowIndex, endRowIndex);
             const max = Math.max(startRowIndex, endRowIndex);
             
             const rangeSelection = {};
             for(let i=min; i<=max; i++) {
                 const r = rows[i];
                 if(r) {
                    visibleCols.forEach(col => {
                        rangeSelection[`${r.id}:${col.id}`] = r.getValue(col.id);
                    });
                 }
             }
             setSelectedCells(rangeSelection);
        } else {
            setSelectedCells(newSelection);
            setLastSelected({ rowIndex: row.index, colIndex: 0 });
        }
    }, [lastSelected]);

    const handleCellMouseDown = useCallback((e, rowIndex, colIndex, rowId, colId, value) => {
        if (e.button === 2) return; // Ignore right-click

        if (e.shiftKey) {
            e.preventDefault(); // Prevent text selection
            const start = lastSelected || { rowIndex, colIndex };
            const newSelection = selectRange(start, { rowIndex, colIndex });
            // Merge if ctrl key, else replace
            if (e.ctrlKey || e.metaKey) {
                setSelectedCells(prev => ({...prev, ...newSelection}));
            } else {
                setSelectedCells(newSelection);
            }
            return;
        }

        setIsDragging(true);
        setDragStart({ rowIndex, colIndex });
        setLastSelected({ rowIndex, colIndex });

        const key = `${rowId}:${colId}`;
        if (e.ctrlKey || e.metaKey) {
             const newSelection = { ...selectedCells };
             newSelection[key] = value;
             setSelectedCells(newSelection);
        } else {
            // Clear and start new
            setSelectedCells({ [key]: value });
        }
    }, [lastSelected, selectedCells]);

    const handleCellMouseEnter = (rowIndex, colIndex) => {
        if (isDragging && dragStart) {
             const newRange = selectRange(dragStart, { rowIndex, colIndex });
             setSelectedCells(newRange); 
        }
        if (isFilling && dragStart) {
            const rStart = Math.min(dragStart.rowIndex, rowIndex);
            const rEnd = Math.max(dragStart.rowIndex, rowIndex);
            const cStart = Math.min(dragStart.colIndex, colIndex);
            const cEnd = Math.max(dragStart.colIndex, colIndex);
            setFillRange({ rStart, rEnd, cStart, cEnd });
        }
    };

    const handleFillMouseDown = (e) => {
        e.stopPropagation();
        e.preventDefault();
        setIsFilling(true);
    };

    const handleFillMouseUp = () => {
        if (isFilling && fillRange && setProps) {
            const startValue = table.getRowModel().rows[dragStart.rowIndex].getVisibleCells()[dragStart.colIndex].getValue();
            const updates = [];
            const visibleRows = table.getRowModel().rows;
            const visibleCols = table.getVisibleLeafColumns();
            for (let r = fillRange.rStart; r <= fillRange.rEnd; r++) {
                for (let c = fillRange.cStart; c <= fillRange.cEnd; c++) {
                    const row = visibleRows[r];
                    const col = visibleCols[c];
                    if (row && col) {
                        updates.push({ rowId: row.id, colId: col.id, value: startValue });
                    }
                }
            }
            if (updates.length > 0) {
                setProps({ cellUpdates: updates });
            }
        }
        setIsFilling(false);
        setFillRange(null);
    };

    useEffect(() => {
        const handleMouseUp = () => {
            setIsDragging(false);
            setDragStart(null);
            if (isFilling) {
                handleFillMouseUp();
            }
        };
        window.addEventListener('mouseup', handleMouseUp);
        return () => window.removeEventListener('mouseup', handleMouseUp);
    }, [isDragging, isFilling, fillRange, dragStart]);
    
    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
                const keys = Object.keys(selectedCells);
                if (keys.length === 0) return;
                e.preventDefault();
                const data = getSelectedData(false);
                if (data) copyToClipboard(data);
            }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [selectedCells]);

    const toggleCol = (key) => {
        setColExpanded(prev => ({
            ...prev,
            [key]: prev[key] === undefined ? false : !prev[key]
        }));
    };

    const isColExpanded = (key) => colExpanded[key] !== false;

    const [dragItem, setDragItem] = useState(null);
    const [dropLine, setDropLine] = useState(null);
    const sessionIdRef = useRef(getOrCreateSessionId(id || 'pivot-grid'));
    const clientInstanceRef = useRef(createClientInstanceId(id || 'pivot-grid'));
    const requestVersionRef = useRef(Number(dataVersion) || 0);
    const latestDataVersionRef = useRef(Number(dataVersion) || 0);
    const stateEpochRef = useRef(0);
    const abortGenerationRef = useRef(0);
    const structuralPendingVersionRef = useRef(null);
    const latestViewportRef = useRef({ start: 0, end: 99, count: 100 });
    const [stateEpoch, setStateEpoch] = useState(0);
    const [cachedColSchema, setCachedColSchema] = useState(null);
    const colSchemaEpochRef = useRef(-1);
    const [visibleColRange, setVisibleColRange] = useState({ start: 0, end: 0 });
    const colRequestStartRef = useRef(null);
    const colRequestEndRef = useRef(null);
    const needsColSchemaRef = useRef(true);
    const [abortGeneration, setAbortGeneration] = useState(0);
    const [structuralInFlight, setStructuralInFlight] = useState(false);
    const [pendingRowTransitions, setPendingRowTransitions] = useState(() => new Map());
    const [pendingColumnSkeletonCount, setPendingColumnSkeletonCount] = useState(0);

    useEffect(() => {
        const numericVersion = Number(dataVersion);
        if (!Number.isFinite(numericVersion)) return;
        latestDataVersionRef.current = numericVersion;
        if (numericVersion > requestVersionRef.current) {
            requestVersionRef.current = numericVersion;
        }
    }, [dataVersion]);

    // Clear schema on structural change so we re-derive from fresh data
    useEffect(() => {
        if (!serverSide) return;
        setCachedColSchema(null);
    }, [stateEpoch, serverSide]);

    // Extract authoritative col_schema embedded by the server as a sentinel entry in props.columns.
    // This is more robust than inferring schema from row keys (handles windowed responses correctly).
    useEffect(() => {
        if (!serverSide || !props.columns) return;
        const schemaEntry = props.columns.find(c => c.id === '__col_schema');
        if (schemaEntry && schemaEntry.col_schema) {
            setCachedColSchema(schemaEntry.col_schema);
            colSchemaEpochRef.current = stateEpoch;
        }
    }, [serverSide, props.columns, stateEpoch]);

    // Derive schema from row keys — only used in client-side mode.
    // In server-side mode the authoritative schema always comes from the __col_schema sentinel
    // embedded in props.columns by the server.  Allowing row-key inference in server-side mode
    // risks schema drift on windowed/partial payloads (only the visible col slice is present).
    useEffect(() => {
        if (serverSide || cachedColSchema) return;
        if (!filteredData || filteredData.length === 0) return;
        const rowMetaKeys = new Set(['_id', '_path', '_isTotal', '_level', '_expanded',
            '_parentPath', '_has_children', '_is_expanded', 'depth', 'uuid', 'subRows', '__virtualIndex']);
        const ignoredIds = new Set([...rowFields, ...colFields, '_isTotal']);
        const colIds = [];
        const colIdSet = new Set();
        for (const row of filteredData) {
            if (!row) continue;
            for (const key of Object.keys(row)) {
                if (!colIdSet.has(key) && !rowMetaKeys.has(key) && !ignoredIds.has(key)) {
                    colIds.push(key);
                    colIdSet.add(key);
                }
            }
        }
        if (colIds.length > 0) {
            setCachedColSchema({
                total_center_cols: colIds.length,
                columns: colIds.map((id, i) => ({ index: i, id, size: 140 }))
            });
            colSchemaEpochRef.current = stateEpoch;
        }
    }, [serverSide, filteredData, cachedColSchema, stateEpoch, rowFields, colFields]);

    // Compute column request window and store in refs for use in field-zone effect
    const COL_BLOCK_SIZE = 20;
    const COL_OVERSCAN = 1; // extra col blocks to prefetch on each side

    const needsColSchema = !cachedColSchema || colSchemaEpochRef.current !== stateEpoch;
    const totalCenterCols = cachedColSchema ? cachedColSchema.total_center_cols : null;

    // Only window columns once we have the schema and are in server-side mode.
    // When needsColSchema is true we fetch all cols to get the schema.
    // Guard uses totalCenterCols !== null (not visibleColRange.end > 0) so that windowing
    // also activates when only column index 0 is visible (end === 0 would falsely disable it).
    const colRequestStart = (serverSide && cachedColSchema && !needsColSchema && totalCenterCols !== null)
        ? Math.max(0, (Math.floor(visibleColRange.start / COL_BLOCK_SIZE) - COL_OVERSCAN) * COL_BLOCK_SIZE)
        : null;

    const colRequestEnd = (serverSide && cachedColSchema && !needsColSchema && totalCenterCols !== null)
        ? Math.min(totalCenterCols - 1,
            (Math.floor(visibleColRange.end / COL_BLOCK_SIZE) + 1 + COL_OVERSCAN) * COL_BLOCK_SIZE - 1)
        : null;

    // Keep refs in sync for use in field-zone effect closures
    colRequestStartRef.current = colRequestStart;
    colRequestEndRef.current = colRequestEnd;
    needsColSchemaRef.current = needsColSchema;

    const beginStructuralTransaction = useCallback(() => {
        stateEpochRef.current += 1;
        abortGenerationRef.current += 1;
        const baselineVersion = Math.max(requestVersionRef.current, latestDataVersionRef.current);
        const nextVersion = baselineVersion + 1;
        requestVersionRef.current = nextVersion;

        setStateEpoch(stateEpochRef.current);
        setAbortGeneration(abortGenerationRef.current);
        setStructuralInFlight(true);
        structuralPendingVersionRef.current = {
            version: nextVersion,
            startDataVersion: latestDataVersionRef.current
        };

        return {
            stateEpoch: stateEpochRef.current,
            abortGeneration: abortGenerationRef.current,
            version: nextVersion
        };
    }, []);

    // Lightweight expansion request: clears inflight (via abortGeneration bump) but
    // does NOT change stateEpoch, so the existing cache stays valid and rows remain
    // visible instead of flashing to skeletons.
    const beginExpansionRequest = useCallback(() => {
        abortGenerationRef.current += 1;
        const newVersion = requestVersionRef.current + 1;
        requestVersionRef.current = newVersion;
        setAbortGeneration(abortGenerationRef.current);
        return {
            abortGeneration: abortGenerationRef.current,
            stateEpoch: stateEpochRef.current,
            version: newVersion
        };
    }, []);

    const setPropsRef = useRef(setProps);
    useEffect(() => {
        setPropsRef.current = setProps;
    }, [setProps]);

    const lastPropsRef = useRef({
        rowFields: initialRowFields,
        colFields: initialColFields,
        valConfigs: initialValConfigs,
        filters: {},
        sorting: [],
        expanded: {},
        showRowTotals: initialShowRowTotals,
        showColTotals: initialShowColTotals,
        columnPinning: initialColumnPinning,
        rowPinning: initialRowPinning,
        columnVisibility: {},
        columnSizing: {}
    });

    React.useEffect(() => {
        const nextProps = {
            rowFields, colFields, valConfigs, filters, sorting, expanded,
            showRowTotals, showColTotals, columnPinning, rowPinning, columnVisibility, columnSizing
        };
        const colFieldsChanged = JSON.stringify(nextProps.colFields) !== JSON.stringify(lastPropsRef.current.colFields);

        const changed = Object.keys(nextProps).some(key => {
            const val = nextProps[key];
            const lastVal = lastPropsRef.current[key];
            return JSON.stringify(val) !== JSON.stringify(lastVal);
        });

        if (setPropsRef.current && changed) {
            debugLog('Sync to Dash Triggered', nextProps);

            // Detect expansion-only: only `expanded` changed, no structural fields.
            // In that case we keep the existing cache (no stateEpoch bump) so rows
            // remain visible. A loading row appears below the expanded row via
            // pendingRowTransitions, and the viewport snaps in place.
            const structuralKeys = ['rowFields', 'colFields', 'valConfigs', 'filters', 'sorting',
                'showRowTotals', 'showColTotals', 'columnPinning', 'rowPinning', 'columnVisibility', 'columnSizing'];
            const isExpansionOnly = serverSide && structuralKeys.every(
                key => JSON.stringify(nextProps[key]) === JSON.stringify(lastPropsRef.current[key])
            );

            lastPropsRef.current = nextProps;

            if (isExpansionOnly) {
                // Cancel any pending scroll restore — the viewport stays exactly in place.
                expansionScrollRestoreRef.current = null;
                if (expansionScrollRestoreRafRef.current !== null && typeof cancelAnimationFrame === 'function') {
                    cancelAnimationFrame(expansionScrollRestoreRafRef.current);
                    expansionScrollRestoreRafRef.current = null;
                }
                const tx = beginExpansionRequest();
                const viewportSnapshot = latestViewportRef.current || { start: 0, end: 99, count: 100 };

                // Extend the row window to cover the block immediately after the anchor block.
                // When expanding a row near the END of its block (e.g. row 95 in block 0),
                // new children overflow into block N+1. Without this extension, those rows have
                // no cache entry → they flash with skeleton loaders until a follow-up fetch lands.
                // pendingExpansionRef.current is already set by onExpandedChange (same event,
                // before this effect runs), so anchorBlock is available here.
                const anchorBlockHint = pendingExpansionRef.current?.anchorBlock ?? -1;
                const expansionBlockSize = 100; // must match blockSize prop
                const extendedEnd = anchorBlockHint >= 0
                    ? Math.max(viewportSnapshot.end, (anchorBlockHint + 2) * expansionBlockSize - 1)
                    : viewportSnapshot.end;
                const extendedCount = extendedEnd - viewportSnapshot.start + 1;

                // Record the last block the expansion response will cover so the deferred
                // effect knows to start soft-invalidating from the block AFTER it, rather
                // than re-dirtying block N+1 that we just filled with fresh data.
                if (pendingExpansionRef.current) {
                    pendingExpansionRef.current.extendedToBlock =
                        anchorBlockHint >= 0 ? anchorBlockHint + 1 : -1;
                }

                setPropsRef.current({
                    ...nextProps,
                    viewport: {
                        table: tableName || undefined,
                        start: viewportSnapshot.start,
                        end: extendedEnd,
                        count: extendedCount,
                        version: tx.version,
                        window_seq: tx.version,
                        state_epoch: tx.stateEpoch,
                        session_id: sessionIdRef.current,
                        client_instance: clientInstanceRef.current,
                        abort_generation: tx.abortGeneration,
                        intent: 'expansion',
                        col_start: colRequestStartRef.current !== null ? colRequestStartRef.current : undefined,
                        col_end: colRequestEndRef.current !== null ? colRequestEndRef.current : undefined,
                        needs_col_schema: needsColSchemaRef.current && serverSide || undefined
                    }
                });
                return;
            }

            // Structural change: full transaction (new stateEpoch clears cache).
            if (serverSide && colFieldsChanged) {
                const prevCount = Array.isArray(lastPropsRef.current.colFields) ? lastPropsRef.current.colFields.length : 0;
                const nextCount = Array.isArray(nextProps.colFields) ? nextProps.colFields.length : 0;
                setPendingColumnSkeletonCount(Math.max(0, nextCount - prevCount));
            } else {
                setPendingColumnSkeletonCount(0);
            }
            const tx = beginStructuralTransaction();
            const viewportSnapshot = latestViewportRef.current || { start: 0, end: 99, count: 100 };
            setPropsRef.current({
                ...nextProps,
                viewport: {
                    table: tableName || undefined,
                    start: viewportSnapshot.start,
                    end: viewportSnapshot.end,
                    count: viewportSnapshot.count,
                    version: tx.version,
                    window_seq: tx.version,
                    state_epoch: tx.stateEpoch,
                    session_id: sessionIdRef.current,
                    client_instance: clientInstanceRef.current,
                    abort_generation: tx.abortGeneration,
                    intent: 'structural',
                    needs_col_schema: serverSide || undefined
                }
            });
        }
    }, [rowFields, colFields, valConfigs, filters, sorting, expanded, showRowTotals, showColTotals, columnPinning, rowPinning, columnVisibility, columnSizing, beginStructuralTransaction, beginExpansionRequest, serverSide, tableName]);

    useEffect(() => {
        const handleClick = () => setContextMenu(null);
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, []);

    const copyToClipboard = (text) => {
        navigator.clipboard.writeText(text);
    };

    const getSelectedData = (withHeaders = false) => {
        const keys = Object.keys(selectedCells);
        if (keys.length === 0) return null;

        const visibleRows = table.getRowModel().rows;
        const visibleCols = table.getVisibleLeafColumns();
        
        // Map indices
        const rowIdMap = {};
        visibleRows.forEach((r, i) => rowIdMap[r.id] = i);
        const colIdMap = {};
        visibleCols.forEach((c, i) => colIdMap[c.id] = i);

        let minR = Infinity, maxR = -1, minC = Infinity, maxC = -1;
        const selectedGrid = {};

        keys.forEach(key => {
            const [rid, cid] = key.split(':');
            const rIdx = rowIdMap[rid];
            const cIdx = colIdMap[cid];
            
            if (rIdx !== undefined && cIdx !== undefined) {
                minR = Math.min(minR, rIdx);
                maxR = Math.max(maxR, rIdx);
                minC = Math.min(minC, cIdx);
                maxC = Math.max(maxC, cIdx);
                if (!selectedGrid[rIdx]) selectedGrid[rIdx] = {};
                selectedGrid[rIdx][cIdx] = selectedCells[key];
            }
        });

        if (minR === Infinity) return null;

        let tsv = "";
        
        if (withHeaders) {
            const headerRow = [];
            for (let c = minC; c <= maxC; c++) {
                headerRow.push(visibleCols[c].columnDef.header);
            }
            tsv += headerRow.join("\t") + "\n";
        }

        for (let r = minR; r <= maxR; r++) {
            const rowVals = [];
            for (let c = minC; c <= maxC; c++) {
                const val = selectedGrid[r] && selectedGrid[r][c];
                rowVals.push(val !== undefined && val !== null ? String(val) : "");
            }
            tsv += rowVals.join("\t") + "\n";
        }
        return tsv;
    };

    const getPinningState = (colId) => {
        const { left, right } = columnPinning;
        if ((left || []).includes(colId)) return 'left';
        if ((right || []).includes(colId)) return 'right';
        return false;
    };

    const handlePinColumn = useCallback((columnId, side) => {
        const table = tableRef.current;
        if (!table) return;
        
        const col = table.getColumn(columnId);
        if (!col) return;

        // Get all IDs to pin/unpin (leaves + potential collapsed placeholder)
        const idsToUpdate = new Set();
        
        // 1. Add leaf IDs
        const isGroup = col.columns && col.columns.length > 0;
        const leafIds = isGroup ? getAllLeafIdsFromColumn(col) : [columnId];
        leafIds.forEach(id => idsToUpdate.add(id));

        // 2. Add collapsed placeholder ID if it's a pivot group
        if (columnId.startsWith('group_')) {
            const rawPathKey = columnId.replace('group_', '');
            idsToUpdate.add(`${rawPathKey}_collapsed`);
        }

        const idsArray = Array.from(idsToUpdate);

        setColumnPinning(prev => {
            const next = { left: [...(prev.left || [])], right: [...(prev.right || [])] };

            // Remove all relevant IDs from both sides first
            next.left = next.left.filter(id => !idsToUpdate.has(id));
            next.right = next.right.filter(id => !idsToUpdate.has(id));

            // Add to new side
            if (side === 'left') next.left.push(...idsArray);
            if (side === 'right') next.right.push(...idsArray);

            return next;
        });
    }, []);


    const handlePinRow = (rowId, pinState) => {
        setRowPinning(prev => {
            const next = { ...prev, top: [...prev.top], bottom: [...prev.bottom] };
            next.top = next.top.filter(d => d !== rowId);
            next.bottom = next.bottom.filter(d => d !== rowId);
            if (pinState === 'top') next.top.push(rowId);
            if (pinState === 'bottom') next.bottom.push(rowId);
            return next;
        });

        // Fire Pinning Event
        if (setProps) {
            setProps({
                rowPinned: {
                    rowId: rowId,
                    pinState: pinState,
                    timestamp: Date.now()
                }
            });
        }
    };

    useEffect(() => {
        setColumnPinning(prev => {
            let nextLeft = [...(prev.left || [])];
            let changed = false;

            // 1. Enforce Hierarchy Pinning
            if (layoutMode === 'hierarchy' && rowFields.length > 0) {
                if (!nextLeft.includes('hierarchy')) {
                     nextLeft = ['hierarchy', ...nextLeft];
                     changed = true;
                }
            }

            // 2. Enforce Row Number Pinning (User Request: Always utmost left)
            if (showRowNumbers) {
                if (!nextLeft.includes('__row_number__')) {
                    nextLeft = ['__row_number__', ...nextLeft];
                    changed = true;
                }
                // Ensure it is first (utmost left)
                const idx = nextLeft.indexOf('__row_number__');
                if (idx > 0) {
                    nextLeft.splice(idx, 1);
                    nextLeft.unshift('__row_number__');
                    changed = true;
                }
            } else {
                 // If hidden, remove from pinned? (Optional, but clean)
                 if (nextLeft.includes('__row_number__')) {
                     nextLeft = nextLeft.filter(id => id !== '__row_number__');
                     changed = true;
                 }
            }

            if (changed) {
                debugLog('Pinning Enforcement Triggered', nextLeft);
                return { ...prev, left: nextLeft };
            }
            return prev;
        });
    }, [layoutMode, rowFields.length, showRowNumbers]);

    // 4. FIXED: handleHeaderContextMenu with proper group detection
    const handleHeaderContextMenu = (e, colId) => {
        e.preventDefault();
        const actions = [];
        const column = table.getColumn(colId);

        if (!column) {
            return;
        }

        const { left, right } = columnPinning;

        // Determine if this is a group column
        const isGroup = isGroupColumn(column);

        // Only show sort options for leaf columns
        if (!isGroup) {
            actions.push({
                label: 'Sort Ascending',
                icon: <Icons.SortAsc/>,
                onClick: () => column.toggleSorting(false)
            });
            actions.push({
                label: 'Sort Descending',
                icon: <Icons.SortDesc/>,
                onClick: () => column.toggleSorting(true)
            });
            actions.push({
                label: 'Clear Sort',
                onClick: () => column.clearSorting()
            });

            actions.push('separator');
            actions.push({
                label: 'Filter...',
                icon: <Icons.Filter/>,
                onClick: () => setActiveFilterCol(colId)
            });
            actions.push({
                label: 'Clear Filter',
                onClick: () => handleHeaderFilter(colId, null)
            });

            actions.push('separator');
        }

        // Pin options for both leaf and group columns
        let isPinned = false;

        if (isGroup) {
            // For group columns, check if ALL leaf columns are pinned
            const leafIds = getAllLeafIdsFromColumn(column);

            const allPinnedLeft = leafIds.length > 0 && leafIds.every(id => (left || []).includes(id));
            const allPinnedRight = leafIds.length > 0 && leafIds.every(id => (right || []).includes(id));

            if (allPinnedLeft) isPinned = 'left';
            else if (allPinnedRight) isPinned = 'right';

            actions.push({
                label: 'Pin All Children Left',
                onClick: () => handlePinColumn(colId, 'left')
            });
            actions.push({
                label: 'Pin All Children Right',
                onClick: () => handlePinColumn(colId, 'right')
            });
            if (isPinned) {
                actions.push({
                    label: 'Unpin All Children',
                    onClick: () => handlePinColumn(colId, false)
                });
            }
        } else {
            // For leaf columns, check directly
            if ((left || []).includes(colId)) isPinned = 'left';
            else if ((right || []).includes(colId)) isPinned = 'right';

            actions.push({
                label: 'Pin Column Left',
                onClick: () => handlePinColumn(colId, 'left')
            });
            actions.push({
                label: 'Pin Column Right',
                onClick: () => handlePinColumn(colId, 'right')
            });
            if (isPinned) {
                actions.push({
                    label: 'Unpin Column',
                    onClick: () => handlePinColumn(colId, false)
                });
            }
        }

        actions.push('separator');
        actions.push({
            label: 'Expand All Rows',
            onClick: () => handleExpandAllRows(true)
        });
        actions.push({
            label: 'Collapse All Rows',
            onClick: () => handleExpandAllRows(false)
        });

        actions.push('separator');
        actions.push({
            label: 'Auto-size Column',
            onClick: () => autoSizeColumn(colId)
        });
        actions.push({
            label: 'Export to Excel',
            icon: <Icons.Export/>,
            onClick: exportPivot
        });

        setContextMenu({
            x: e.clientX,
            y: e.clientY,
            actions: actions
        });
    };

    const handleContextMenu = (e, value, colId, row) => {
        e.preventDefault();
        const rowId = row ? row.id : null;
        
        // Check if clicked cell is already in selection
        const key = `${rowId}:${colId}`;
        const isSelected = selectedCells[key] !== undefined;
        
        const hasSelection = Object.keys(selectedCells).length > 0;
        
        const getTableData = (withHeaders) => {
            const visibleRows = table.getRowModel().rows;
            const visibleCols = table.getVisibleLeafColumns();
            let tsv = "";
            if (withHeaders) {
                tsv += visibleCols.map(c => typeof c.columnDef.header === 'string' ? c.columnDef.header : c.id).join('\t') + '\n';
            }
            visibleRows.forEach(r => {
                const vals = visibleCols.map(c => {
                    const v = r.getValue(c.id);
                    return v !== undefined && v !== null ? String(v) : "";
                });
                tsv += vals.join('\t') + '\n';
            });
            return tsv;
        };

        const actions = [
            { label: 'Copy Table', icon: <Icons.DragIndicator/>, onClick: () => copyToClipboard(getTableData(false)) },
            { label: 'Copy Table with Headers', onClick: () => copyToClipboard(getTableData(true)) },
        ];

        if (hasSelection) {
            actions.push('separator');
            actions.push({ label: 'Copy Selection', onClick: () => {
                const data = getSelectedData(false);
                if (data) copyToClipboard(data);
            }});
            actions.push({ label: 'Copy Selection with Headers', onClick: () => {
                const data = getSelectedData(true);
                if (data) copyToClipboard(data);
            }});
        }

        actions.push('separator');
        actions.push({ label: `Filter by "${value}"`, icon: <Icons.Filter/>, onClick: () => {
            handleHeaderFilter(colId, {
                operator: 'AND',
                conditions: [{ type: 'eq', value: String(value), caseSensitive: false }]
            });
        }});
        actions.push({ label: 'Clear Filter', onClick: () => handleHeaderFilter(colId, null) });

        actions.push('separator');
        actions.push({ label: 'Drill Through', icon: <Icons.Search/>, onClick: () => {
             if (row && row.original && row.original._path && row.original._path !== '__grand_total__' && !row.original._isTotal) {
                 fetchDrillData(row.original._path, 0, null, 'asc', '');
             }
        }});

        actions.push('separator');
        if (row && serverSide && row.getCanExpand() && row.original && row.original._path && rowFields.length > 1) {
            actions.push({
                label: 'Expand All Children',
                icon: <Icons.ChevronDown/>,
                onClick: () => {
                    const rowPath = row.original._path;
                    subtreeExpandRef.current = { path: rowPath, expandedPaths: new Set([rowPath]) };
                    captureExpansionScrollPosition();
                    clearCache();
                    setExpanded(prev => {
                        const base = (prev !== null && typeof prev === 'object') ? prev : {};
                        return { ...base, [rowPath]: true };
                    });
                }
            });
        }

        actions.push('separator');
        if (rowId) {
            const isPinnedTop = rowPinning.top.includes(rowId);
            const isPinnedBottom = rowPinning.bottom.includes(rowId);

            actions.push({ label: 'Pin Row Top', onClick: () => handlePinRow(rowId, 'top') });
            actions.push({ label: 'Pin Row Bottom', onClick: () => handlePinRow(rowId, 'bottom') });
            if (isPinnedTop || isPinnedBottom) {
                actions.push({ label: 'Unpin Row', onClick: () => handlePinRow(rowId, false) });
            }
        }

        setContextMenu({
            x: e.clientX,
            y: e.clientY,
            actions: actions
        });
    };

    const filteredData = useMemo(() => {
        if (serverSide) return data || [];
        if (!data || !data.length) return [];
        
        return data.filter(row => {
            return Object.entries(filters).every(([colId, filterGroup]) => {
                if (!filterGroup) return true;
                // Handle legacy string filters if any remain
                if (typeof filterGroup === 'string') {
                    return String(row[colId]).toLowerCase().includes(filterGroup.toLowerCase());
                }
                
                if (!filterGroup.conditions || filterGroup.conditions.length === 0) return true;
                
                const rowVal = row[colId];
                const passes = filterGroup.conditions.map(cond => {
                    let val = cond.value;
                    if (cond.type === 'in') {
                        return Array.isArray(val) && val.includes(rowVal);
                    }
                    
                    const rStr = String(rowVal).toLowerCase();
                    const vStr = String(val).toLowerCase();
                    
                    if (cond.type === 'contains') return rStr.includes(vStr);
                    if (cond.type === 'startsWith') return rStr.startsWith(vStr);
                    if (cond.type === 'endsWith') return rStr.endsWith(vStr);
                    if (cond.type === 'eq' || cond.type === 'equals') return cond.caseSensitive ? String(rowVal) === String(val) : rStr === vStr;
                    if (cond.type === 'ne' || cond.type === 'notEquals') return cond.caseSensitive ? String(rowVal) !== String(val) : rStr !== vStr;
                    
                    const rNum = Number(rowVal);
                    const vNum = Number(val);
                    if (!isNaN(rNum) && !isNaN(vNum)) {
                        if (cond.type === 'gt') return rNum > vNum;
                        if (cond.type === 'lt') return rNum < vNum;
                        if (cond.type === 'gte') return rNum >= vNum;
                        if (cond.type === 'lte') return rNum <= vNum;
                        if (cond.type === 'between') {
                            const vNum2 = Number(cond.value2);
                            return rNum >= vNum && rNum <= vNum2;
                        }
                    }
                    return true;
                });
                
                if (filterGroup.operator === 'OR') return passes.some(p => p);
                return passes.every(p => p);
            });
        });
    }, [data, filters, serverSide]);

    const staticTotal = useMemo(() => ({ _isTotal: true, _path: '__grand_total__', _id: 'Grand Total', __isGrandTotal__: true }), []);
    const staticMinMax = useMemo(() => ({}), []);

    const { nodes, total, minMax } = useMemo(() => {
        return { nodes: filteredData, total: staticTotal, minMax: staticMinMax };
    }, [filteredData, staticTotal, staticMinMax]);

    const handleHeaderFilter = (columnId, filterValue) => {
        setFilters(prev => {
            const newFilters = {...prev};
            if (filterValue === null || filterValue.conditions.length === 0) {
                delete newFilters[columnId];
            } else {
                newFilters[columnId] = filterValue;
            }
            return newFilters;
        });
    };

    const autoSizeColumn = (columnId) => {
        const rows = table.getRowModel().rows;
        const sampleRows = rows.slice(0, 100); 
        let maxWidth = 0;
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        context.font = '13px Roboto, Helvetica, Arial, sans-serif'; 
        
        const column = table.getColumn(columnId);
        const header = column.columnDef.header;
        const headerText = typeof header === 'string' ? header : columnId;
        maxWidth = context.measureText(headerText).width + 40; 
        
        sampleRows.forEach(row => {
            const cellValue = row.getValue(columnId);
            const text = formatValue(cellValue); 
            const width = context.measureText(text).width + 24; 
            if (width > maxWidth) maxWidth = width;
        });
        
        maxWidth = Math.min(maxWidth, 600);
        maxWidth = Math.max(maxWidth, 60);
        
        table.setColumnSizing(old => ({
            ...old,
            [columnId]: maxWidth
        }));
    };

    const handleFilterClick = (e, columnId) => {
        e.stopPropagation();
        setActiveFilterCol(columnId);
        setFilterAnchorEl(e.currentTarget);
        if (setProps) {
            setProps({
                filters: { ...filters, '__request_unique__': columnId }
            });
        }
    };

    const closeFilterPopover = () => {
        setActiveFilterCol(null);
        setFilterAnchorEl(null);
    };

    const columns = useMemo(() => {
        // Enhanced Sorting Logic (Tree-aware + Natural + Customization)
        const customSortingFn = (rowA, rowB, columnId) => {
            try {
                // Safety check for loading rows (server-side)
                if (!rowA.original || !rowB.original) return 0;

                // 1. Special handling for grand total - it should always be at the end
                // Check multiple ways to identify the grand total
                const aIsGrandTotal = rowA.id === '__grand_total__' ||
                                     rowA.original.__isGrandTotal__ ||
                                     rowA.original._path === '__grand_total__' ||
                                     rowA.original._id === 'Grand Total';
                const bIsGrandTotal = rowB.id === '__grand_total__' ||
                                     rowB.original.__isGrandTotal__ ||
                                     rowB.original._path === '__grand_total__' ||
                                     rowB.original._id === 'Grand Total';

                // If one is grand total and the other is not, grand total goes last
                if (aIsGrandTotal && !bIsGrandTotal) return 1;
                if (!aIsGrandTotal && bIsGrandTotal) return -1;

                // If both are grand totals, they are equal
                if (aIsGrandTotal && bIsGrandTotal) return 0;

                // 2. Regular totals (but not grand total) should come after non-totals
                const aIsRegularTotal = (rowA.original && rowA.original._isTotal) && !aIsGrandTotal;
                const bIsRegularTotal = (rowB.original && rowB.original._isTotal) && !bIsGrandTotal;

                if (aIsRegularTotal && !bIsRegularTotal) return 1;
                if (!aIsRegularTotal && bIsRegularTotal) return -1;

                // Both are regular totals (not grand totals) - they can be equal for sorting purposes
                if (aIsRegularTotal && bIsRegularTotal) return 0;

                const valA = rowA.getValue(columnId);
                const valB = rowB.getValue(columnId);

                // 2. Column-Specific Customization
                const colSortOptions = (sortOptions.columnOptions && sortOptions.columnOptions[columnId]) || {};
                const isNatural = colSortOptions.naturalSort !== undefined ? colSortOptions.naturalSort : (sortOptions.naturalSort !== false);
                const isCaseSensitive = colSortOptions.caseSensitive !== undefined ? colSortOptions.caseSensitive : sortOptions.caseSensitive;

                if (isNatural) {
                    const sensitivity = isCaseSensitive ? 'variant' : 'base';
                    return new Intl.Collator(undefined, { numeric: true, sensitivity }).compare(String(valA || ''), String(valB || ''));
                }

                // Default Alphanumeric with configured sensitivity
                const defaultSensitivity = isCaseSensitive ? 'variant' : 'base';
                return String(valA || '').localeCompare(String(valB || ''), undefined, { numeric: true, sensitivity: defaultSensitivity });
            } catch (err) {
                console.error('Sorting error:', err);
                return 0;
            }
        };

        const sortingFn = serverSide ? 'auto' : customSortingFn;
        const hierarchyCols = [];

        if (showRowNumbers) {
             hierarchyCols.push({
                id: '__row_number__',
                header: '#',
                size: 50,
                enablePinning: false, // User Request: Cannot be changed
                cell: ({ row }) => (
                    <div
                        style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            background: '#f5f5f5',
                            cursor: 'pointer',
                            fontSize: '11px',
                            color: '#666',
                            borderRight: `1px solid ${theme.border}`,
                            userSelect: 'none'
                        }}
                        onMouseDown={(e) => {
                            if (e.button !== 0) return;
                            e.stopPropagation();
                            setIsRowSelecting(true);
                            setRowDragStart(row.index);
                            handleRowSelect(row, e.shiftKey, e.ctrlKey || e.metaKey);
                        }}
                        onMouseEnter={() => {
                            if (isRowSelecting && rowDragStart !== null) {
                                handleRowRangeSelect(rowDragStart, row.index);
                            }
                        }}
                    >
                        {row.index + 1 + (serverSide ? (renderedOffset || 0) : 0)}
                    </div>
                )
            });
        }

        if (layoutMode === 'hierarchy') {
            if (rowFields.length > 0) {
                hierarchyCols.push({
                    id: 'hierarchy',
                    accessorFn: row => row._id,
                    header: rowFields.join(' > '),
                    size: 250,
                    sortingFn, // Apply sort
                    cell: ({ row }) => {
                        const depth = (row.original.depth !== undefined) ? row.original.depth : (row.depth || 0);
                        // Note: We removed selectedCells from this dependency to avoid unnecessary re-renders
                        // isSelected is calculated dynamically in the renderCell function instead
                        return (
                            <div
                                style={{
                                    paddingLeft: `${depth * 24}px`,
                                    display: 'flex',
                                    alignItems: 'center',
                                    width: '100%',
                                    height: '100%'
                                    // isSelected styling will be applied in renderCell
                                }}
                            >
                                 {row.getCanExpand() && !(row.original && row.original._isTotal) ? (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            debugLog('Toggling expansion (hierarchy) for', row.id);
                                            row.getToggleExpandedHandler()(e);
                                        }}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        style={{border:'none',background:'none',cursor:'pointer',padding:0,marginRight:'6px',color:'#757575',display:'flex'}}
                                    >
                                        {row.getIsExpanded() ? <Icons.ChevronDown/> : <Icons.ChevronRight/>}
                                        {pendingRowTransitions.has(row.id) && (
                                            <span style={{fontSize: '10px', opacity: 0.75, marginLeft: '3px'}}>...</span>
                                        )}
                                    </button>
                                ) : <span style={{width:'18px'}}/>}
                                <span style={{ fontWeight: (row.original && row.original._isTotal) ? 700 : 400 }}>{row.original ? row.original._id : ''}</span>
                            </div>
                        );
                    }
                });
            }
        } else {
            rowFields.forEach((field, i) => {
                hierarchyCols.push({
                    id: field,
                    accessorKey: field,
                    header: field,
                    size: 150,
                    enablePinning: true,
                    sortingFn,
                    cell: ({ row, getValue }) => {
                        const val = getValue();
                        // Note: We removed selectedCells from this dependency to avoid unnecessary re-renders
                        // isSelected is calculated dynamically in the renderCell function instead
                        const depth = (row.original.depth !== undefined) ? row.original.depth : (row.depth || 0);

                        // Outline: Show only if current column matches depth (step layout)
                        // Tabular: Show if column is <= depth (repeat labels)
                        let showValue = true;
                        if (layoutMode === 'outline') {
                            if (i !== depth) showValue = false;
                        } else {
                            // Tabular
                            if (i > depth) showValue = false;
                        }

                        // Expander only on the active level column
                        const showExpander = (i === depth) && row.getCanExpand() && !(row.original && row.original._isTotal);

                        return (
                            <div
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    width: '100%',
                                    height: '100%',
                                    fontWeight: (row.original && row.original._isTotal) ? 700 : 400
                                    // isSelected styling will be applied in renderCell
                                }}
                            >
                                {showExpander && (
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            debugLog('Toggling expansion (mode) for', row.id);
                                            row.getToggleExpandedHandler()(e);
                                        }}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        style={{border:'none',background:'none',cursor:'pointer',padding:0,marginRight:'6px',color:'#757575',display:'flex'}}
                                    >
                                        {row.getIsExpanded() ? <Icons.ChevronDown/> : <Icons.ChevronRight/>}
                                        {pendingRowTransitions.has(row.id) && (
                                            <span style={{fontSize: '10px', opacity: 0.75, marginLeft: '3px'}}>...</span>
                                        )}
                                    </button>
                                )}
                                {showValue ? val : ''}
                            </div>
                        );
                    }
                });
            });
        }

        let dataCols = [];
        if (colFields.length === 0) {
            dataCols = valConfigs.map(c => ({
                id: getKey('', c.field, c.agg),
                accessorFn: row => row[getKey('', c.field, c.agg)] ,
                header: `${c.field} (${c.agg})`,
                size: 130,
                enablePinning: true,
                sortingFn,
                cell: info => (
                    <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'flex-end', paddingRight:'8px'}} onContextMenu={e => handleContextMenu(e, info.getValue(), info.column.id, info.row)}>
                        {formatValue(info.getValue(), c.format)}
                    </div>
                )
            }));
        } else if (serverSide) {
            const keys = new Set();
            // Prefer explicit columns from backend callback (primary, always correct)
            if (props.columns && props.columns.length > 0) {
                props.columns.filter(c => c.id !== '__col_schema').forEach(c => keys.add(c.id));
            // Fallback: use schema derived from row data (when props.columns not yet available)
            } else if (cachedColSchema && cachedColSchema.columns && cachedColSchema.columns.length > 0) {
                cachedColSchema.columns.forEach(c => keys.add(c.id));
            } else if (filteredData.length > 0) {
                filteredData.forEach(row => Object.keys(row).forEach(k => keys.add(k)));
            }

            if (keys.size > 0) {
                const ignoreKeys = new Set(['_id', 'depth', '_isTotal', '_path', 'uuid', ...rowFields, ...colFields]);
                
                // Helper to determine if a column is relevant for the grid
                const measureSuffixes = valConfigs.map(v => `_${v.field}_${v.agg}`);
                const measureIds = new Set(valConfigs.map(v => getKey('', v.field, v.agg)));
                
                const flatCols = [];
                Array.from(keys).sort().forEach(k => {
                    if (ignoreKeys.has(k)) return;

                    // Filter: Only show active measures, row totals, or pivoted measure columns
                    let isRelevant = false;
                    if (measureIds.has(k)) isRelevant = true;
                    else if (k.startsWith('__RowTotal__')) isRelevant = true;
                    else if (measureSuffixes.some(s => k.endsWith(s))) isRelevant = true;
                    
                    if (!isRelevant) return;

                    flatCols.push({
                        id: k,
                        accessorFn: row => row[k],
                        header: k,
                        size: 130,
                        sortingFn,
                        cell: info => {
                            const v = info.getValue();
                            let fmt = null;
                            if (valConfigs) {
                                for (const c of valConfigs) { if (k.includes(c.field)) { fmt = c.format; break; } }
                            }
                            return (
                                <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'flex-end', paddingRight:'8px'}} onContextMenu={e => handleContextMenu(e, v, info.column.id, info.row)}>
                                    {formatValue(v, fmt)}
                                </div>
                            );
                        }
                    });
                });

                const rowTotalCols = flatCols.filter(c => c.id.startsWith('__RowTotal__'));
                const pivotCols = flatCols.filter(c => !c.id.startsWith('__RowTotal__'));

                const buildRecursiveTree = (cols) => {
                    const root = { columns: [] };
                    cols.forEach(col => {
                        const key = col.id;
                        if (!key) return;
                        let dimStr = key;
                        let measureStr = "";
                        let matchedConfig = null;
                        if (valConfigs) {
                            for (const config of valConfigs) {
                                const suffix = `_${config.field}_${config.agg}`;
                                if (key.toLowerCase().endsWith(suffix.toLowerCase())) {
                                    matchedConfig = config;
                                    measureStr = `${config.field} (${config.agg})`;
                                    dimStr = key.substring(0, key.length - suffix.length);
                                    break;
                                }
                            }
                        }
                        if (!matchedConfig) {
                             const parts = key.split('_');
                             if (parts.length > 1) {
                                 dimStr = parts.slice(0, parts.length - 2).join('_');
                                 measureStr = parts.slice(parts.length - 2).join(' ');
                                 if (!dimStr) dimStr = "Total";
                             }
                        }
                        const dimPath = dimStr ? dimStr.split('|') : [];
                        let current = root;
                        let pathKey = '';
                        let parentCollapsed = false;
                        for (let idx = 0; idx < dimPath.length; idx++) {
                            const val = dimPath[idx].trim();
                            if (idx > 0 && !isColExpanded(pathKey)) {
                                parentCollapsed = true;
                                break;
                            }
                            const currentPathKey = pathKey ? `${pathKey}|||${val}` : val;
                            pathKey = currentPathKey;
                            let node = current.columns.find(c => c.headerVal === val);
                            if (!node) {
                                node = {
                                    id: `group_${currentPathKey}`,
                                    headerVal: val,
                                    header: (
                                        <div style={{display:'flex', alignItems:'center', gap:4, width:'100%', overflow:'hidden'}}>
                                            <span style={{flex:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}} title={val}>{val}</span>
                                            <span onClick={(e) => { e.stopPropagation(); toggleCol(currentPathKey); }} style={{cursor:'pointer', display:'flex', opacity:0.6, flexShrink:0}}>
                                                {isColExpanded(currentPathKey) ? <Icons.ColCollapse/> : <Icons.ColExpand/>}
                                            </span>
                                        </div>
                                    ),
                                    columns: [],
                                    enablePinning: true
                                };
                                current.columns.push(node);
                            }
                            current = node;
                        }
                        if (parentCollapsed) {
                             if (current.columns.length === 0) {
                                 current.columns.push({
                                    id: pathKey + "_collapsed",
                                    header: "...",
                                    size: 60,
                                    accessorFn: () => "",
                                    cell: () => <div style={{color:'#999', textAlign:'center'}}>...</div>
                                });
                             }
                             return;
                        }
                        if (isColExpanded(pathKey) || dimPath.length === 0) {
                            const newCol = { ...col, header: measureStr || col.header, enablePinning: true };
                            if (matchedConfig && matchedConfig.format) {
                                newCol.cell = info => (
                                    <EditableCell
                                        getValue={info.getValue}
                                        row={info.row}
                                        column={info.column}
                                        format={matchedConfig.format}
                                        validationRules={validationRules}
                                        setProps={setProps}
                                        handleContextMenu={handleContextMenu}
                                    />
                                );
                            }
                            current.columns.push(newCol);
                        } else if (current.columns.length === 0) {
                             current.columns.push({
                                id: pathKey + "_collapsed",
                                header: "...",
                                size: 60,
                                accessorFn: () => "",
                                cell: () => <div style={{color:'#999', textAlign:'center'}}>...</div>
                            });
                        }
                    });
                    return root.columns;
                };
                dataCols = buildRecursiveTree(pivotCols);

                if (rowTotalCols.length > 0) {
                    rowTotalCols.forEach(c => {
                         if (c.header.startsWith('__RowTotal__')) {
                             c.header = c.header.replace('__RowTotal__', 'Total ');
                         }
                                              c.cell = info => {
                                                 const config = valConfigs.find(v => c.id.includes(v.field));
                                                 return (
                                                     <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'flex-end', paddingRight:'8px', fontWeight:'bold'}} onContextMenu={e => handleContextMenu(e, info.getValue(), info.column.id, info.row)}>
                                                         {formatValue(info.getValue(), config ? config.format : null)}
                                                     </div>
                                                 );
                                              };                         dataCols.push(c);
                    });
                }
            }
        }
        if (hierarchyCols.length === 0 && dataCols.length === 0) {
             dataCols.push({ id: 'no_data', header: 'No Data', cell: () => 'No Data' });
        }

        const buildColumns = (cols) => {
            return cols.map(col => {
                if (col.columns) {
                    return {
                        ...col,
                        columns: buildColumns(col.columns)
                    };
                }
                return col;
            });
        };

        return buildColumns([...hierarchyCols, ...dataCols]);
    // filteredData is intentionally excluded: in server-side mode columns come from props.columns /
    // cachedColSchema and filteredData changes on every viewport scroll, causing the entire column
    // tree to rebuild. filteredData is used only as a last-resort fallback (client-side, no schema).
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rowFields, colFields, valConfigs, minMax, colorScale, colExpanded, serverSide, layoutMode, showRowNumbers, isRowSelecting, rowDragStart, props.columns, cachedColSchema]);

    useEffect(() => {
        const activeLeafIds = new Set();
        const stack = [...columns];

        while (stack.length > 0) {
            const columnDef = stack.pop();
            if (!columnDef) continue;

            if (Array.isArray(columnDef.columns) && columnDef.columns.length > 0) {
                columnDef.columns.forEach(child => stack.push(child));
                continue;
            }

            const leafId = columnDef.id || columnDef.accessorKey;
            if (leafId) {
                activeLeafIds.add(String(leafId));
            }
        }

        setColumnSizing(prev => {
            if (!prev || typeof prev !== 'object' || Object.keys(prev).length === 0) {
                return prev;
            }

            let changed = false;
            const next = {};
            Object.keys(prev).forEach(key => {
                if (activeLeafIds.has(key)) {
                    next[key] = prev[key];
                } else {
                    changed = true;
                }
            });

            return changed ? next : prev;
        });
    }, [columns]);

    const parentRef = useRef(null);
    const expansionScrollRestoreRef = useRef(null);
    const expansionScrollRestoreRafRef = useRef(null);
    // Tracks an in-progress "expand all children" operation.
    // { path: string, expandedPaths: Set<string> }
    // We use a ref (not state) to avoid dependency cycles in the watcher effect.
    const subtreeExpandRef = useRef(null);
    // After a single-row expansion, holds the anchor block index so we can
    // invalidate subsequent blocks once the expansion response has landed.
    // Doing it after the response avoids a concurrent viewport request that
    // would race with (and stale-reject) the expansion request.
    const pendingExpansionRef = useRef(null);
    const rowHeight = rowHeights[spacingMode] || 32;

    // Cache key: only structural changes that require a full cache wipe.
    // Expansion and rowCount are intentionally excluded — expansion uses targeted
    // block invalidation (invalidateFromBlock) so rows before the toggled node
    // stay cached, and rowCount is a derived result that changes with expansion.
    const serverSideCacheKey = useMemo(() => JSON.stringify({
        sorting,
        filters,
        rowFields,
        colFields,
        valConfigs,
    }), [sorting, filters, rowFields, colFields, valConfigs]);
    // Viewport reset key: only changes that semantically restart the user's view
    // (new sort/filter/fields). rowCount is excluded because it changes when rows
    // are expanded, which must NOT scroll back to the top.
    const serverSideViewportResetKey = useMemo(() => JSON.stringify({
        sorting,
        filters,
        rowFields,
        colFields,
        valConfigs,
    }), [sorting, filters, rowFields, colFields, valConfigs]);

    const serverSidePinsGrandTotal = serverSide && showColTotals;
    const effectiveRowCount = serverSidePinsGrandTotal && rowCount ? Math.max(rowCount - 1, 0) : rowCount;

    const captureExpansionScrollPosition = useCallback(() => {
        if (!serverSide || !parentRef.current) return;
        expansionScrollRestoreRef.current = {
            scrollTop: parentRef.current.scrollTop,
            restorePassesRemaining: 3
        };
    }, [serverSide]);

    const { rowVirtualizer, getRow, renderedData, renderedOffset, clearCache, invalidateFromBlock, softInvalidateFromBlock, grandTotalRow } = useServerSideRowModel({
        parentRef,
        serverSide,
        rowCount: effectiveRowCount,
        rowHeight,
        data: filteredData,
        dataOffset: dataOffset || 0,
        dataVersion: dataVersion || 0,
        setProps,
        blockSize: 100,
        cacheKey: serverSideCacheKey,
        excludeGrandTotal: serverSidePinsGrandTotal,
        stateEpoch,
        sessionId: sessionIdRef.current,
        clientInstance: clientInstanceRef.current,
        abortGeneration,
        structuralInFlight,
        requestVersionRef,
        tableName,
        colStart: colRequestStart,
        colEnd: colRequestEnd,
        needsColSchema: needsColSchema && serverSide,
    });

    useEffect(() => {
        if (!serverSide || !structuralInFlight) return;
        const pending = structuralPendingVersionRef.current;
        const numericVersion = Number(dataVersion);
        if (!pending || !Number.isFinite(numericVersion)) return;
        if (numericVersion > pending.startDataVersion && numericVersion >= pending.version) {
            setStructuralInFlight(false);
            structuralPendingVersionRef.current = null;
            setPendingRowTransitions(new Map());
            setPendingColumnSkeletonCount(0);
        }
    }, [dataVersion, serverSide, structuralInFlight]);

    useEffect(() => {
        if (!serverSide || !structuralInFlight) return;
        const timeoutId = setTimeout(() => {
            setStructuralInFlight(false);
            structuralPendingVersionRef.current = null;
            setPendingRowTransitions(new Map());
            setPendingColumnSkeletonCount(0);
        }, 10000);
        return () => clearTimeout(timeoutId);
    }, [serverSide, structuralInFlight, stateEpoch]);

    const tableData = useMemo(() => {
        if (serverSide) {
             const centerData = renderedData.filter(row => (
                 row &&
                 !row._isTotal &&
                 row._path !== '__grand_total__' &&
                 row._id !== 'Grand Total'
             ));
             if (showColTotals && grandTotalRow) {
                 return [...centerData, grandTotalRow];
             }
             return centerData;
        }

        let baseData = (filteredData.length ? [...nodes] : []);

        if (!showColTotals) {
            baseData = baseData.filter(r => !r._isTotal);
        }

        // Add the grand total to the end of the data array to ensure it appears at the bottom
        // Only add if not in serverSide mode and showColTotals is true
        if (!serverSide && showColTotals) {
            baseData = [...baseData, total];
        }

        return baseData;
    }, [nodes, total, filteredData, serverSide, showColTotals, renderedData, grandTotalRow]);

    const getRowId = useCallback((row, relativeIndex) => {
        if (!row) return `skeleton_${relativeIndex}`; // Handle skeleton rows
        if (row._isTotal || row._path === '__grand_total__' || row._id === 'Grand Total') return '__grand_total__';
        if (serverSide && typeof row.__virtualIndex === 'number') {
            return row._path || (row.id ? row.id : String(row.__virtualIndex));
        }
        
        // Use renderedOffset if available (from virtualizer cache), else fallback to dataOffset
        const effectiveOffset = (serverSide && renderedOffset !== undefined) ? renderedOffset : (dataOffset || 0);
        const actualIndex = serverSide ? relativeIndex + effectiveOffset : relativeIndex;
        
        return row._path || (row.id ? row.id : String(actualIndex));
    }, [serverSide, dataOffset, renderedOffset]);
    const getSubRows = useCallback(r => r ? r.subRows : undefined, []);
    const getRowCanExpand = useCallback(row => {
        if (!row.original) return false;
        // Prevent expansion of any total rows, including grand totals
        if (row.original && row.original._isTotal) return false;
        
        if (serverSide) {
             // Use server-provided flag if available for accurate child detection
             if (row.original._has_children !== undefined) return row.original._has_children;
             return (row.original.depth || 0) < rowFields.length - 1;
        }
        
        return row.subRows && row.subRows.length > 0;
    }, [serverSide, rowFields.length]);

    const getIsRowExpanded = useCallback(row => {
        if (!row.original) return false;
        if (row.original && row.original._isTotal) return false;

        if (serverSide) {
             // 1. "Expand All" mode
             if (expanded === true) return true;
             
             // 2. Explicit Local State (Optimistic)
             // We check if the key exists in the expanded object to respect user interactions
             if (expanded && Object.prototype.hasOwnProperty.call(expanded, row.id)) {
                 return !!expanded[row.id];
             }
             
             // 3. Server State (Fallback/Source of Truth)
             // If local state doesn't know about this row yet (e.g. initial load), trust the server
             if (row.original._is_expanded !== undefined) {
                 return row.original._is_expanded;
             }
        }

        // Standard Client-Side Logic
        if (expanded === true) return true;
        // Otherwise check if this specific row is expanded
        return !!expanded[row.id];
    }, [expanded, serverSide]);

    const tableState = useMemo(() => {
        // Automatically pin Grand Total to top or bottom based on grandTotalPosition prop
        let finalRowPinning = rowPinning;
        const grandTotalId = '__grand_total__';
        const pinToBottom = grandTotalPosition === 'bottom';

        // Find the actual Grand Total row in the data and get its real ID
        let actualGrandTotalRowId = null;
        if (tableData) {
            for (const row of tableData) {
                if (!row) continue;
                if (row.__isGrandTotal__ || row._path === '__grand_total__' || row._id === 'Grand Total') {
                    if (row._isTotal || row._path === '__grand_total__' || row._id === 'Grand Total') {
                        actualGrandTotalRowId = '__grand_total__';
                        break;
                    }
                }
            }
        }

        if (actualGrandTotalRowId) {
            const topWithoutGrandTotal = (rowPinning.top || []).filter(id => id !== actualGrandTotalRowId);
            const bottomWithoutGrandTotal = (rowPinning.bottom || []).filter(id => id !== actualGrandTotalRowId);

            finalRowPinning = {
                ...rowPinning,
                top: pinToBottom ? topWithoutGrandTotal : [...topWithoutGrandTotal, actualGrandTotalRowId],
                bottom: pinToBottom ? [...bottomWithoutGrandTotal, actualGrandTotalRowId] : bottomWithoutGrandTotal,
            };
        } else {
            // If GT is NOT in data, ensure it is NOT pinned (to avoid crash)
            const cleanPinning = {
                top: (rowPinning.top || []).filter(id => id !== grandTotalId),
                bottom: (rowPinning.bottom || []).filter(id => id !== grandTotalId),
            };
            if (cleanPinning.top.length !== (rowPinning.top || []).length ||
                cleanPinning.bottom.length !== (rowPinning.bottom || []).length) {
                finalRowPinning = { ...rowPinning, ...cleanPinning };
            }
        }

        return {
            sorting,
            expanded,
            columnPinning,
            rowPinning: finalRowPinning,
            grouping: rowFields,
            columnVisibility,
            columnSizing
        };
    }, [sorting, expanded, columnPinning, rowPinning, rowFields, columnVisibility, columnSizing, tableData, grandTotalPosition]);



    const handleExpandAllRows = (shouldExpand) => {
        if (serverSide) {
            captureExpansionScrollPosition();
            // Expanding/collapsing ALL rows changes every row index — full cache wipe.
            // (This path bypasses onExpandedChange so invalidateFromBlock won't run.)
            clearCache();
            setExpanded(shouldExpand ? true : {});
            return;
        }

        if (shouldExpand) {
            // Expand all rows by creating an object with all row IDs set to true
            const allRows = table.getCoreRowModel().rows;
            const newExpanded = {};

            allRows.forEach(row => {
                // Only add rows that can be expanded and are not totals
                if (row.getCanExpand() && !(row.original && row.original._isTotal)) {
                    newExpanded[row.id] = true;

                    // Also expand sub-rows recursively
                    const expandSubRows = (subRows) => {
                        subRows.forEach(subRow => {
                            if (subRow.getCanExpand() && !(subRow.original && subRow.original._isTotal)) {
                                newExpanded[subRow.id] = true;
                                if (subRow.subRows && subRow.subRows.length > 0) {
                                    expandSubRows(subRow.subRows);
                                }
                            }
                        });
                    };

                    if (row.subRows && row.subRows.length > 0) {
                        expandSubRows(row.subRows);
                    }
                }
            });

            setExpanded(newExpanded);
        } else {
            // Collapse all by setting empty object
            setExpanded({});
        }
    };

    const handleSortingChange = (updater) => {
        const newSorting = typeof updater === 'function' ? updater(sorting) : updater;
        setSorting(newSorting);

        // Fire sort event to backend
        if (setPropsRef.current) {
            setPropsRef.current({
                sorting: newSorting,
                sortEvent: {
                    type: 'change',
                    status: 'applied',
                    sorting: newSorting,
                    timestamp: Date.now()
                }
            });
        }
    };

    const table = useReactTable({
        data: tableData,
        columns,
        state: tableState,
        onSortingChange: (updater) => { handleSortingChange(updater); },
        onExpandedChange: (updater) => {
            captureExpansionScrollPosition();
            const newExpanded = typeof updater === 'function' ? updater(expanded) : updater;

            if (serverSide) {
                // Find which path was toggled so we know which block to defer-invalidate
                // after the expansion response lands (see pendingExpansionRef effect).
                // Value-diff: detect any key whose boolean value flipped (covers
                // both key-add/remove AND false→true / true→false toggles).
                const oldExp = expanded || {};
                const newExp = newExpanded || {};
                const allKeys = new Set([...Object.keys(oldExp), ...Object.keys(newExp)]);
                const changedPath = [...allKeys].find(k => !!oldExp[k] !== !!newExp[k]);
                if (changedPath) {
                    const isNowExpanded = !!(newExpanded && newExpanded[changedPath]);
                    setPendingRowTransitions(prev => {
                        const next = new Map(prev);
                        next.set(changedPath, isNowExpanded ? 'expand' : 'collapse');
                        return next;
                    });
                }

                // -1 signals "row not in viewport — do a full cache clear".
                let anchorBlock = -1;
                let expandedRowVirtualIndex = undefined;
                if (changedPath) {
                    const toggledRow = renderedData.find(r => r && r._path === changedPath);
                    if (toggledRow && typeof toggledRow.__virtualIndex === 'number') {
                        anchorBlock = Math.floor(toggledRow.__virtualIndex / 100);
                        // Record the virtual index for viewport anchor preservation.
                        // When rows are inserted/removed ABOVE the current scroll position
                        // we adjust scrollTop so the same logical rows remain in view.
                        expandedRowVirtualIndex = toggledRow.__virtualIndex;
                    }
                    // Do NOT fall back to the scroll position when the row is not in the
                    // rendered viewport.  Using the viewport block as the anchor leaves all
                    // blocks between the expanded row and the viewport with stale (shifted)
                    // row indices.  A full cache clear (anchorBlock = -1) is safer.
                }
                // Don't invalidate now — doing so fires a concurrent viewport request
                // that races with the expanded sync request and causes a stale rejection.
                // Record the anchor so the deferred effect invalidates subsequent blocks
                // once the expansion response has landed (dataVersion bump).
                pendingExpansionRef.current = { anchorBlock, expandedRowVirtualIndex, oldRowCount: rowCount };
            }

            setExpanded(newExpanded);
        },
        onColumnPinningChange: (updater) => { debugLog('onColumnPinningChange'); setColumnPinning(updater); },
        onRowPinningChange: (updater) => { debugLog('onRowPinningChange'); setRowPinning(updater); },
        onColumnVisibilityChange: (updater) => { debugLog('onColumnVisibilityChange'); setColumnVisibility(updater); },
        onColumnSizingChange: (updater) => { debugLog('onColumnSizingChange'); setColumnSizing(updater); },
        getRowId,
        getCoreRowModel: getCoreRowModel(),
        getExpandedRowModel: getExpandedRowModel(),
        getGroupedRowModel: getGroupedRowModel(),
        getSubRows,
        enableRowPinning: true, // Enable Row Pinning
        enableColumnResizing: true,
        enableMultiSort: true, // Explicitly enable multi-sort
        columnResizeMode: 'onChange',
        manualPagination: serverSide,
        manualSorting: serverSide,
        manualFiltering: serverSide,
        manualGrouping: serverSide,
        manualExpanding: serverSide,
        pageCount: serverSide ? Math.ceil((rowCount || 0) / 100) : undefined,
        getRowCanExpand,
        getIsRowExpanded,
        enableColumnPinning: true,
    });

    // Update the ref with the current table instance
    useEffect(() => {
        tableRef.current = table;
    }, [table]);

    // Accessibility & Event System Effect for Sorting
    useEffect(() => {
        if (sorting.length > 0) {
            const sortDesc = sorting[0].desc ? 'descending' : 'ascending';
            const colId = sorting[0].id;
            const col = tableRef.current.getColumn(colId);
            const colName = col ? (typeof col.columnDef.header === 'string' ? col.columnDef.header : colId) : colId;
            setAnnouncement(`Sorted by ${colName} ${sortDesc}`);
        } else {
            setAnnouncement("Sorting cleared");
        }
        
        // Fire sort event
        if (setPropsRef.current) {
             setPropsRef.current({
                sortEvent: {
                    type: 'change',
                    status: 'applied',
                    sorting: sorting,
                    timestamp: Date.now()
                }
            });
        }
    }, [sorting]); // Removed table and setProps from dependencies

    const toggleAllColumnsPinned = (pinState) => {
        const leafColumns = table.getAllLeafColumns();
        const newPinning = { left: [], right: [] };
        
        if (pinState === 'left') {
            newPinning.left = leafColumns.map(c => c.id).filter(id => id !== 'no_data');
        } else if (pinState === 'right') {
            newPinning.right = leafColumns.map(c => c.id).filter(id => id !== 'no_data');
        } else {
            if (layoutMode === 'hierarchy') {
                newPinning.left = ['hierarchy'];
            }
        }
        
        setColumnPinning(newPinning);
    };

    const activeFilterOptions = useMemo(() => {
        if (!activeFilterCol) return [];
        if (filterOptions[activeFilterCol]) return filterOptions[activeFilterCol];
        
        const col = table.getColumn(activeFilterCol);
        if (!col) return [];
        
        const unique = new Set();
        const rows = table.getCoreRowModel().rows;
        rows.forEach(row => {
            const val = row.getValue(activeFilterCol);
            if (val !== null && val !== undefined) unique.add(val);
        });
        
        return Array.from(unique).sort();
    }, [activeFilterCol, filterOptions, table]);

    const { rows } = table.getRowModel();
    const topRows = table.getTopRows();
    const bottomRows = table.getBottomRows();
    const centerRows = table.getCenterRows();
    const lastStableRowModelRef = useRef({
        topRows: [],
        centerRows: [],
        bottomRows: []
    });
    const hasRenderedData = renderedData.some(Boolean);

    useEffect(() => {
        if (!serverSide) return;
        if (centerRows.length > 0 || topRows.length > 0 || bottomRows.length > 0) {
            lastStableRowModelRef.current = { topRows, centerRows, bottomRows };
        }
    }, [serverSide, topRows, centerRows, bottomRows]);

    useEffect(() => {
        if (!serverSide) return;
        lastStableRowModelRef.current = {
            topRows: [],
            centerRows: [],
            bottomRows: []
        };
        // Expansion should still refetch server-side data, but it should not force the viewport back to the top.
        expansionScrollRestoreRef.current = null;
        if (expansionScrollRestoreRafRef.current !== null && typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(expansionScrollRestoreRafRef.current);
            expansionScrollRestoreRafRef.current = null;
        }
        if (parentRef.current) {
            parentRef.current.scrollTop = 0;
        }
    }, [serverSide, serverSideViewportResetKey, parentRef]);

    useLayoutEffect(() => {
        if (!serverSide || expansionScrollRestoreRef.current === null || !parentRef.current) return;
        if (!hasRenderedData && centerRows.length === 0 && topRows.length === 0 && bottomRows.length === 0) return;

        const restoreTarget = expansionScrollRestoreRef.current;
        if (!restoreTarget) return;

        const applyScrollRestore = () => {
            if (!parentRef.current || !expansionScrollRestoreRef.current) return;

            const nextTarget = expansionScrollRestoreRef.current;
            const targetScrollTop = nextTarget.scrollTop;
            if (rowVirtualizer.scrollToOffset) {
                rowVirtualizer.scrollToOffset(targetScrollTop);
            }
            if (Math.abs(parentRef.current.scrollTop - targetScrollTop) > 1) {
                parentRef.current.scrollTop = targetScrollTop;
            }

            if (nextTarget.restorePassesRemaining <= 1) {
                expansionScrollRestoreRef.current = null;
                expansionScrollRestoreRafRef.current = null;
                return;
            }

            expansionScrollRestoreRef.current = {
                ...nextTarget,
                restorePassesRemaining: nextTarget.restorePassesRemaining - 1
            };

            if (typeof requestAnimationFrame === 'function') {
                expansionScrollRestoreRafRef.current = requestAnimationFrame(applyScrollRestore);
            } else {
                applyScrollRestore();
            }
        };

        applyScrollRestore();

        return () => {
            if (expansionScrollRestoreRafRef.current !== null && typeof cancelAnimationFrame === 'function') {
                cancelAnimationFrame(expansionScrollRestoreRafRef.current);
                expansionScrollRestoreRafRef.current = null;
            }
        };
    }, [serverSide, hasRenderedData, centerRows.length, topRows.length, bottomRows.length, renderedOffset, dataVersion, rowVirtualizer]);

    const effectiveTopRows = (serverSide && hasRenderedData && topRows.length === 0 && centerRows.length === 0)
        ? lastStableRowModelRef.current.topRows
        : topRows;
    const effectiveCenterRows = (serverSide && hasRenderedData && centerRows.length === 0)
        ? lastStableRowModelRef.current.centerRows
        : centerRows;
    const effectiveBottomRows = (serverSide && hasRenderedData && centerRows.length === 0 && bottomRows.length === 0)
        ? lastStableRowModelRef.current.bottomRows
        : bottomRows;
    const rowModelLookup = useMemo(() => {
        const lookup = new Map();
        [...effectiveTopRows, ...effectiveCenterRows, ...effectiveBottomRows].forEach(row => {
            if (row && row.id) {
                lookup.set(row.id, row);
            }
        });
        return lookup;
    }, [effectiveTopRows, effectiveCenterRows, effectiveBottomRows]);

    // Progressive subtree expansion: each time the backend returns new data,
    // scan it for descendants of the target path that still have children and
    // haven't been expanded yet. Auto-expand them and let the cycle continue
    // until every reachable descendant is expanded.
    useEffect(() => {
        if (!serverSide || !data || !subtreeExpandRef.current) return;
        const { path, expandedPaths } = subtreeExpandRef.current;

        const toExpand = data.filter(row => {
            if (!row || !row._path || !row._has_children) return false;
            const inSubtree = row._path === path || row._path.startsWith(path + '|||');
            return inSubtree && !expandedPaths.has(row._path);
        });

        if (toExpand.length === 0) {
            subtreeExpandRef.current = null;
            return;
        }

        toExpand.forEach(row => expandedPaths.add(row._path));

        captureExpansionScrollPosition();
        clearCache();
        setExpanded(prev => {
            const base = (prev !== null && typeof prev === 'object') ? prev : {};
            const next = { ...base };
            toExpand.forEach(row => { next[row._path] = true; });
            return next;
        });
    }, [data, serverSide]); // intentionally omits captureExpansionScrollPosition/clearCache/setExpanded — stable refs

    // Deferred block invalidation after single-row expansion.
    // We wait for the expansion response to land (dataVersion bumps) before
    // invalidating subsequent blocks. This ensures only ONE backend request fires
    // (the expanded sync), with no concurrent viewport request to race against it.
    // After the anchor block is updated with fresh data, blocks beyond it are
    // deleted so they get re-fetched on next scroll (their row indices shifted).
    useEffect(() => {
        if (!serverSide || !pendingExpansionRef.current) return;
        const { anchorBlock, expandedRowVirtualIndex, oldRowCount, extendedToBlock = -1 } = pendingExpansionRef.current;
        pendingExpansionRef.current = null;
        if (anchorBlock < 0) {
            // The expanded row was not in the viewport when the user toggled it.
            // We can't know which anchor block shifted, but a hard clear causes
            // a full skeleton flash.  Soft-invalidate all blocks from 0 instead
            // so existing rows stay visible (stale-while-revalidate) until fresh
            // data lands (finding #6).
            if (softInvalidateFromBlock) softInvalidateFromBlock(0);
        } else {
            // The expansion request was extended to cover through extendedToBlock
            // (anchorBlock + 1 when the anchor is known).  Those blocks were filled
            // with fresh data by the data-sync effect, so we must NOT re-dirty them.
            // Start soft-invalidating from the first block BEYOND the fresh coverage.
            const firstStaleBlock = extendedToBlock >= 0 ? extendedToBlock + 1 : anchorBlock + 1;
            if (softInvalidateFromBlock) softInvalidateFromBlock(firstStaleBlock);
        }
        // Clear the transition loader now that the expansion response has landed.
        setPendingRowTransitions(new Map());

        // Viewport anchor preservation.
        // When rows are inserted or removed ABOVE the current scroll position, the
        // virtualizer re-layouts and the same pixel offset now shows a different
        // logical row. Compensate by shifting scrollTop so that the user continues
        // to see the same rows they were looking at before the toggle.
        if (
            parentRef.current &&
            typeof expandedRowVirtualIndex === 'number' &&
            typeof oldRowCount === 'number' &&
            rowHeight > 0
        ) {
            const rowDelta = (rowCount || 0) - (oldRowCount || 0);
            if (rowDelta !== 0) {
                // Y position of the expanded/collapsed row (uniform row heights).
                const expandedRowY = expandedRowVirtualIndex * rowHeight;
                const currentScrollTop = parentRef.current.scrollTop;
                // Only compensate when the anchor row is entirely ABOVE the viewport.
                // If it is at or inside the viewport the inserted children appear
                // naturally below it and no scroll adjustment is needed.
                if (expandedRowY + rowHeight <= currentScrollTop) {
                    const newScrollTop = Math.max(0, currentScrollTop + rowDelta * rowHeight);
                    parentRef.current.scrollTop = newScrollTop;
                    if (rowVirtualizer.scrollToOffset) {
                        rowVirtualizer.scrollToOffset(newScrollTop);
                    }
                }
            }
        }
    }, [dataVersion, serverSide, rowCount, rowHeight, parentRef, rowVirtualizer]); // fires when expansion response arrives

    // Debug effect removed (finding #10 — hot-path logging).

    const visibleLeafColumns = table.getVisibleLeafColumns();

    // 1. Row Virtualizer (Managed by useServerSideRowModel)
    const virtualRows = rowVirtualizer.getVirtualItems();
    const showColumnLoadingSkeletons = serverSide && structuralInFlight && pendingColumnSkeletonCount > 0;
    const columnSkeletonWidth = 140;
    const stickyHeaderHeight = (table.getHeaderGroups().length * rowHeight) + (showFloatingFilters ? rowHeight : 0);
    const bodyRowsTopOffset = stickyHeaderHeight + (effectiveTopRows.length * rowHeight);

    useEffect(() => {
        if (!serverSide || virtualRows.length === 0) return;
        const firstRow = virtualRows[0].index;
        const lastRow = virtualRows[virtualRows.length - 1].index;
        latestViewportRef.current = {
            start: firstRow,
            end: lastRow,
            count: Math.max(1, lastRow - firstRow + 1)
        };
    }, [serverSide, virtualRows]);

    // 2. Column Virtualizer (Extracted)
    const {
        columnVirtualizer,
        virtualCenterCols,
        beforeWidth,
        afterWidth,
        totalLayoutWidth,
        leftCols,
        rightCols,
        centerCols
    } = useColumnVirtualizer({
        parentRef,
        table
    });

    // Memoized lookup structures for the header render path.
    // centerColIndexMap: O(1) id→index lookup; only rebuilt when the column list changes.
    // visibleLeafIndexSet: O(1) membership check; only rebuilt when the virtual window shifts.
    const centerColIndexMap = useMemo(
        () => new Map(centerCols.map((c, i) => [c.id, i])),
        [centerCols]
    );
    const visibleLeafIndexSet = useMemo(
        () => new Set(virtualCenterCols.map(v => v.index)),
        [virtualCenterCols]
    );

    // O(1) colId → visible-leaf-index map for renderCell; rebuilt only when column list changes.
    const visibleLeafColIndexMap = useMemo(
        () => new Map(table.getVisibleLeafColumns().map((c, i) => [c.id, i])),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [table.getVisibleLeafColumns()]
    );

    // Sync the column virtualizer's visible range into state so useServerSideRowModel
    // can detect column window changes and trigger re-fetches.
    useEffect(() => {
        if (virtualCenterCols.length === 0) return;
        const newStart = virtualCenterCols[0].index;
        const newEnd = virtualCenterCols[virtualCenterCols.length - 1].index;
        setVisibleColRange(prev => {
            if (prev.start === newStart && prev.end === newEnd) return prev;
            return { start: newStart, end: newEnd };
        });
    }, [virtualCenterCols]);

    // Use the custom hook
    const { getHeaderStickyStyle, getStickyStyle } = useStickyStyles(
        theme,
        leftCols,
        rightCols
    );


    const getFieldZone = (id) => {
        if (id === 'hierarchy') return 'rows';
        if (rowFields.includes(id)) return 'rows';
        if (colFields.includes(id)) return 'cols';
        if (valConfigs.find(v => id.includes(v.field))) return 'vals';
        return null;
    };

    const getFieldIndex = (id, zone) => {
        if (zone === 'rows') return rowFields.indexOf(id);
        if (zone === 'cols') return colFields.indexOf(id);
        if (zone === 'vals') return valConfigs.findIndex(v => id.includes(v.field));
        return -1;
    };

    const onHeaderDrop = (e, targetColId) => {
        e.preventDefault();
        if (!dragItem) return;
        const { field, zone: srcZone } = dragItem;
        const fieldName = typeof field === 'string' ? field : field.field;
        
        // Handle dropping on the same column (no-op)
        if (fieldName === targetColId) {
            setDragItem(null);
            return;
        }

        const targetZone = getFieldZone(targetColId);
        if (!targetZone) {
            setDragItem(null);
            return;
        }

        // Check for Pinning (Drag-to-Pin)
        const targetIsPinned = getPinningState(targetColId);
        if (targetIsPinned) {
             handlePinColumn(fieldName, targetIsPinned);
        } else {
             // If dropping on unpinned, maybe unpin?
             handlePinColumn(fieldName, false);
        }

        // Reordering or Pivoting
        if (srcZone === targetZone) {
             const srcIdx = getFieldIndex(fieldName, srcZone);
             const targetIdx = getFieldIndex(targetColId, targetZone);
             if (srcIdx !== -1 && targetIdx !== -1 && srcIdx !== targetIdx) {
                 const move = (list, setList) => {
                    const n = [...list]; 
                    const [moved] = n.splice(srcIdx, 1);
                    n.splice(targetIdx, 0, moved); 
                    setList(n);
                };
                if (srcZone==='rows') move(rowFields, setRowFields);
                if (srcZone==='cols') move(colFields, setColFields);
                if (srcZone==='vals') move(valConfigs, setValConfigs);
             }
        } else {
            // Pivoting (Moving between zones)
            const targetIdx = getFieldIndex(targetColId, targetZone);
            // Remove from source
            if (srcZone==='rows') setRowFields(p=>p.filter(f=>f!==fieldName));
            if (srcZone==='cols') setColFields(p=>p.filter(f=>f!==fieldName));
            if (srcZone==='vals') setValConfigs(p=>p.filter(f=>f.field!==fieldName));

            // Insert into target
            const insert = (list, setList, item) => {
                const n = [...list];
                n.splice(targetIdx, 0, item);
                setList(n);
            };
            if (targetZone==='rows') insert(rowFields, setRowFields, fieldName);
            if (targetZone==='cols') insert(colFields, setColFields, fieldName);
            if (targetZone==='vals') insert(valConfigs, setValConfigs, {field: fieldName, agg:'sum'});
        }
        setDragItem(null);
    };

    const onDragStart = (e, field, zone, idx) => {
        setDragItem({ field, zone, idx });
        e.dataTransfer.effectAllowed = 'move';
    };
    const onDragOver = (e, zone, idx) => {
        e.preventDefault();
        // If hovering over a header, zone might be 'cols' or 'rows' derived from ID
        // For sidebar, we use dropLine logic
        if (['rows', 'cols', 'vals', 'filter'].includes(zone)) {
            const rect = e.currentTarget.getBoundingClientRect();
            const mid = rect.top + rect.height / 2;
            setDropLine({ zone, idx: e.clientY > mid ? idx + 1 : idx });
        }
    };
    const onDrop = (e, targetZone) => {
        e.preventDefault();
        if (!dragItem) return;
        const { field, zone: srcZone, idx: srcIdx } = dragItem;
        const targetIdx = (dropLine && dropLine.idx) || 0;
        const insertItem = (list, idx, item) => { const n = [...list]; n.splice(idx, 0, item); return n; };
        const fieldName = typeof field === 'string' ? field : field.field;
        if (!fieldName || typeof fieldName !== 'string') { setDragItem(null); setDropLine(null); return; }
        if (srcZone !== targetZone) {
            if (srcZone==='rows') setRowFields(p=>p.filter(f=>f!==fieldName));
            if (srcZone==='cols') setColFields(p=>p.filter(f=>f!==fieldName));
            if (srcZone==='vals') setValConfigs(p=>p.filter((_,i)=>i!==srcIdx));
            if (targetZone==='rows') setRowFields(p=>p.includes(fieldName) ? p : insertItem(p, targetIdx, fieldName));
            if (targetZone==='cols') setColFields(p=>p.includes(fieldName) ? p : insertItem(p, targetIdx, fieldName));
            if (targetZone==='vals') setValConfigs(p=>insertItem(p, targetIdx, {field: fieldName, agg:'sum'}));
            if (targetZone==='filter' && !filters.hasOwnProperty(fieldName)) setFilters(p=>({...p, [fieldName]: ''}));
        } else {
            const move = (list, setList) => {
                const n = [...list]; const [moved] = n.splice(srcIdx, 1);
                let ins = targetIdx; if (srcIdx < targetIdx) ins -= 1;
                n.splice(ins, 0, moved); setList(n);
            };
            if (targetZone==='rows') move(rowFields, setRowFields);
            if (targetZone==='cols') move(colFields, setColFields);
            if (targetZone==='vals') move(valConfigs, setValConfigs);
        }
        setDragItem(null); setDropLine(null);
    };





    const buildExportAoa = (allRows) => {
        // Use table.getHeaderGroups() so we get the real multi-level header structure
        // with correct parent/child relationships and colSpans set by TanStack.
        const headerGroups = table.getHeaderGroups();

        // Identify leaf (data) columns from the last header group, excluding
        // internal/UI-only columns that should not appear in the export.
        const SKIP_COL_IDS = new Set(['__row_number__']);
        const leafHeaders = (headerGroups[headerGroups.length - 1]?.headers ?? [])
            .filter(h => !SKIP_COL_IDS.has(h.column.id) && !h.isPlaceholder);

        const leafCount = leafHeaders.length;

        // Build one AOA row per header group.
        // For each header row, we fill a flat array of length leafCount.
        // A header with colSpan > 1 occupies that many leaf slots; placeholders fill gaps.
        const headerAoaRows = [];
        const allMerges = [];

        headerGroups.forEach((hg, rowIdx) => {
            const aoaRow = new Array(leafCount).fill('');
            let leafPos = 0;
            hg.headers.forEach(h => {
                if (SKIP_COL_IDS.has(h.column.id)) return;
                const span = h.colSpan ?? 1;
                if (!h.isPlaceholder) {
                    // Resolve header text — prefer columnDef.header string, fall back to id
                    const colDef = h.column.columnDef;
                    let headerText = '';
                    if (typeof colDef.header === 'string') {
                        headerText = colDef.header;
                    } else if (typeof h.column.id === 'string') {
                        // Strip group_ prefix and internal path separators for cleaner output
                        headerText = h.column.id
                            .replace(/^group_/, '')
                            .replace(/\|\|\|/g, ' > ');
                    }
                    aoaRow[leafPos] = headerText;
                    if (span > 1 && rowIdx < headerGroups.length - 1) {
                        // Merge across the span; row 0-indexed in the final aoa
                        allMerges.push({ s: { r: rowIdx, c: leafPos }, e: { r: rowIdx, c: leafPos + span - 1 } });
                    }
                }
                leafPos += span;
            });
            headerAoaRows.push(aoaRow);
        });

        // If there is only one header group and it looks identical to itself
        // (no real parent grouping), just keep one header row to avoid duplication.
        const dedupedHeaderRows = headerAoaRows.length > 1
            ? headerAoaRows
            : headerAoaRows;  // keep as-is for single group (flat table)

        // Build data rows — include ALL rows (totals + data rows)
        // Track max content width per column for auto-sizing.
        const colWidths = leafHeaders.map(h => {
            const colDef = h.column.columnDef;
            return typeof colDef.header === 'string' ? colDef.header.length : (h.column.id ?? '').length;
        });

        const dataRows = allRows.map(r => {
            return leafHeaders.map((h, ci) => {
                const col = h.column;
                const colId = col.id;
                const colDef = col.columnDef;

                let val;
                if (colId === 'hierarchy') {
                    // Hierarchy column: indent using spaces to reflect depth
                    const depth = r.original?.depth ?? r.depth ?? 0;
                    const label = r.original?._isTotal ? (r.original?._id ?? 'Total') : (r.original?._id ?? '');
                    val = '\u00A0\u00A0'.repeat(depth) + label;  // non-breaking spaces for Excel visibility
                } else if (typeof colDef.accessorFn === 'function') {
                    // Use accessorFn to get the value (same as TanStack does internally)
                    val = colDef.accessorFn(r.original, r.index);
                } else if (colDef.accessorKey) {
                    val = r.original?.[colDef.accessorKey];
                } else {
                    val = '';
                }

                // Normalize: undefined/null → empty string; keep numbers as numbers
                if (val === undefined || val === null) val = '';

                // Track max width for column auto-sizing
                const cellLen = String(val).length;
                if (cellLen > colWidths[ci]) colWidths[ci] = cellLen;

                return val;
            });
        });

        // Build ws['!cols'] — cap at 60 chars to avoid overly wide columns
        const wsCols = colWidths.map(w => ({ wch: Math.min(Math.max(w + 2, 8), 60) }));

        return {
            aoa: [...dedupedHeaderRows, ...dataRows],
            merges: allMerges,
            wsCols,
            headerRowCount: dedupedHeaderRows.length,
        };
    };

    const fetchDrillData = useCallback(async (rowPath, page = 0, sortCol = null, sortDir = 'asc', filterText = '') => {
        const params = new URLSearchParams({
            table: tableName,
            row_path: rowPath,
            row_fields: rowFields.join(','),
            page: String(page),
            page_size: '100',
        });
        if (sortCol) { params.set('sort_col', sortCol); params.set('sort_dir', sortDir); }
        if (filterText) params.set('filter', filterText);

        setDrillModal(prev => ({ ...(prev || { path: rowPath, rows: [], page: 0, totalRows: 0, sortCol, sortDir, filterText }), loading: true }));
        try {
            const resp = await fetch(`${drillEndpoint}?${params.toString()}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const json = await resp.json();
            setDrillModal({ loading: false, path: rowPath, rows: json.rows || [], page: json.page || 0, totalRows: json.total_rows || 0, sortCol, sortDir, filterText });
        } catch (err) {
            console.error('Drill-through fetch failed:', err);
            setDrillModal(null);
        }
    }, [tableName, rowFields, drillEndpoint]);

    const handleCellDrillThrough = useCallback((row, colId) => {
        if (!row || !row.original) return;
        const rowPath = row.original._path;
        if (!rowPath || rowPath === '__grand_total__') return;  // skip total rows
        fetchDrillData(rowPath, 0, null, 'asc', '');
    }, [fetchDrillData]);

    const exportPivot = useCallback(() => {
        const XLSX_LIMIT = 500000;
        const allRows = table.getRowModel().rows;  // full row model, not just virtual window

        const isCSV = (rowCount || 0) > XLSX_LIMIT;

        if (isCSV) {
            // CSV path — flat, no merge support needed
            // Use TanStack visible leaf columns so we match what's shown on screen,
            // and skip internal UI-only columns.
            const SKIP_CSV = new Set(['__row_number__']);
            const leafCols = table.getVisibleLeafColumns().filter(c => !SKIP_CSV.has(c.id));

            const escape = (v) => {
                if (v == null) return '';
                const s = String(v);
                return (s.includes(',') || s.includes('"') || s.includes('\n'))
                    ? `"${s.replace(/"/g, '""')}"` : s;
            };
            const header = leafCols.map(c => {
                const h = c.columnDef?.header;
                return escape(typeof h === 'string' ? h : (c.id ?? ''));
            }).join(',');
            const lines = allRows.map(r =>
                leafCols.map(c => {
                    if (c.id === 'hierarchy') {
                        const depth = r.original?.depth ?? r.depth ?? 0;
                        return escape('  '.repeat(depth) + (r.original?._id ?? ''));
                    }
                    const val = typeof c.columnDef?.accessorFn === 'function'
                        ? c.columnDef.accessorFn(r.original, r.index)
                        : (c.columnDef?.accessorKey ? r.original?.[c.columnDef.accessorKey] : '');
                    return escape(val ?? '');
                }).join(',')
            );
            const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8;' });
            saveAs(blob, 'pivot.csv');
        } else {
            // XLSX path — multi-level headers + hierarchy indent
            const { aoa, merges, wsCols } = buildExportAoa(allRows);
            const ws = XLSX.utils.aoa_to_sheet(aoa);
            if (merges.length > 0) ws['!merges'] = merges;
            if (wsCols && wsCols.length > 0) ws['!cols'] = wsCols;
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, 'Pivot');
            const buf = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
            saveAs(new Blob([buf], { type: 'application/octet-stream' }), 'pivot.xlsx');
        }
    }, [rows, columns, rowCount, table]);

    // --- Helper to Render a single Cell with useCallback ---
            const renderCell = useCallback((cell, virtualRowIndex, isVirtualRow = false) => {
                if (!cell) return null;
                
                const row = cell.row;
                const col = cell.column;
                const colIndex = visibleLeafColIndexMap.get(col.id) ?? -1;
                const isHierarchy = cell.column.id === 'hierarchy';
                const isSelected = selectedCells[`${row.id}:${cell.column.id}`] !== undefined;
                const isLastSelected = lastSelected && lastSelected.rowIndex === virtualRowIndex && lastSelected.colIndex === colIndex; // Approximate check            
            // Check for fill handle selection
            let isFillSelected = false;
            if (fillRange && dragStart) {
                 const rMin = Math.min(dragStart.rowIndex, fillRange.rEnd); // simplified for demo
                 // Precise range check would require row index mapping
                 if (virtualRowIndex >= fillRange.rStart && virtualRowIndex <= fillRange.rEnd && colIndex >= fillRange.cStart && colIndex <= fillRange.cEnd) {
                     isFillSelected = true;
                 }
            }
    
            const rowBackground = (row.original && row.original._isTotal)
                ? (isDarkTheme(theme) ? '#1a2e1a' : '#f0f7f0')
                : (isDarkTheme(theme) ? '#212121' : '#fff');
            let bg = rowBackground;
            if (isSelected) bg = theme.select;
    
            const stickyStyle = getStickyStyle(cell.column, bg);
    
            const condStyle = getConditionalStyle(cell.column.id, cell.getValue());
            
            // Fix row number ordering
            let cellContent;
            if (cell.column.id === '__row_number__' && isVirtualRow) {
                cellContent = (row.original && typeof row.original.__virtualIndex === 'number')
                    ? row.original.__virtualIndex + 1
                    : virtualRowIndex + 1;
            } else {
                cellContent = flexRender(cell.column.columnDef.cell, cell.getContext());
            }
    
            return (
                <div
                    key={cell.id}
                    role="gridcell"
                    aria-selected={isSelected}
                    onMouseDown={(e) => handleCellMouseDown(e, virtualRowIndex, colIndex, row.id, cell.column.id, cell.getValue())}
                    onMouseEnter={() => handleCellMouseEnter(virtualRowIndex, colIndex)}
                    style={{
                        ...styles.cell,
                        width: col.getSize(),
                        height: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: isHierarchy ? 'flex-start' : 'flex-end',
                        fontWeight: (row.original && row.original._isTotal) ? 700 : ((isHierarchy && row.getIsGrouped()) ? 500 : 400),
                        color: (row.original && row.original._isTotal) ? theme.text : undefined,
                        background: bg,
                        ...stickyStyle,
                        ...condStyle,
                        ...(isFillSelected ? {boxShadow: `inset 0 0 0 1px ${theme.primary}`} : {}),
                        userSelect: 'none',
                        position: stickyStyle && stickyStyle.position === 'sticky' ? 'sticky' : 'relative',
                    }}
                    onContextMenu={e => handleContextMenu(e, cell.getValue(), cell.column.id, row)}
                >
                    {cellContent}
                    {isLastSelected && Object.keys(selectedCells).length === 1 && isSelected && (
                        <div 
                            onMouseDown={handleFillMouseDown}
                            style={{
                                position: 'absolute',
                                right: 0,
                                bottom: 0,
                                width: '8px',
                                height: '8px',
                                background: theme.primary,
                                cursor: 'crosshair',
                                zIndex: 100,
                                border: `1px solid ${isDarkTheme(theme) ? '#000' : '#fff'}`,                            borderRadius: '1px'
                            }}
                        />
                    )}
                </div>
            );
        }, [selectedCells, fillRange, theme, getStickyStyle, handleCellMouseDown, handleCellMouseEnter, handleContextMenu, handleFillMouseDown, isDarkTheme]);

    // NEW: Render Header Cell for Split Sections
    // overrideWidth: when set, replaces the computed section width (used for partially-visible
    // group headers during center-column virtualization so the width matches only the visible leaves).
    const renderHeaderCell = (header, level, renderSection = 'center', overrideWidth = null) => {
        const isGroupHeader = header.column.columns && header.column.columns.length > 0;
        const isSorted = header.column.getIsSorted();
        const sortIndex = header.column.getSortIndex();
        const isMultiSort = table.getState().sorting.length > 1;
        const isResizingColumn = header.column.getIsResizing();
        const isHoveredHeader = hoveredHeaderId === header.column.id;
        const isFocusedHeader = focusedHeaderId === header.column.id;
        const isResizeHandleVisible = isResizingColumn || isHoveredHeader || isFocusedHeader;
        const isPinned = header.column.getIsPinned();
        const leafColumns = header.column.getLeafColumns ? header.column.getLeafColumns() : [header.column];
        const sectionLeafIds = new Set(
            (renderSection === 'left' ? leftCols : renderSection === 'right' ? rightCols : centerCols).map(column => column.id)
        );
        const sectionWidth = leafColumns
            .filter(column => sectionLeafIds.has(column.id))
            .reduce((sum, column) => sum + column.getSize(), 0);
        const headerWidth = overrideWidth !== null ? overrideWidth : (sectionWidth || header.getSize());

        // Calculate sticky style for pinned headers using the hook
        const stickyStyle = getHeaderStickyStyle(header, level, renderSection);

        return (
            <div key={header.id} style={{
                ...styles.headerCell,
                width: headerWidth,
                minWidth: headerWidth,
                flexShrink: 0,
                height: rowHeight,
                ...stickyStyle,
                cursor: 'pointer',
                // Position is handled by getHeaderStickyStyle or parent container
                position: stickyStyle.position || 'relative'
            }}
            role="columnheader"
            aria-sort={isSorted || 'none'}
            aria-label={`${typeof header.column.columnDef.header === 'string' ? header.column.columnDef.header : header.column.id}. Click or press Alt+Up/Down to sort.`}
            tabIndex={0}
            onKeyDown={(e) => {
                if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                    e.preventDefault();
                    header.column.toggleSorting(e.key === 'ArrowDown', e.shiftKey);
                }
            }}
            draggable={!isGroupHeader && header.column.id !== '__row_number__'}
            onDragStart={(e) => {
                if (!isGroupHeader && header.column.id !== '__row_number__') {
                    onDragStart(e, header.column.id, 'cols', -1);
                }
            }}
            onContextMenu={(e) => handleHeaderContextMenu(e, header.column.id)}
            onMouseEnter={() => setHoveredHeaderId(header.column.id)}
            onMouseLeave={() => setHoveredHeaderId(current => (current === header.column.id ? null : current))}
            onFocus={() => setFocusedHeaderId(header.column.id)}
            onBlur={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget)) {
                    setFocusedHeaderId(current => (current === header.column.id ? null : current));
                }
            }}
            onClick={header.column.getToggleSortingHandler()}>
                <div style={{
                    display:'flex',
                    alignItems:'center',
                    gap: '4px',
                    width: '100%',
                    justifyContent: header.column.id === 'hierarchy' ? 'flex-start' : 'center',
                    padding: '0 4px',
                    overflow: 'hidden',
                    minWidth: '60px'
                }}>
                <span style={{
                    overflow:'hidden',
                    textOverflow:'ellipsis',
                    whiteSpace:'nowrap',
                    flex: 1,
                    minWidth: 0
                }}>
                    {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </span>

                {!isGroupHeader && header.column.id !== 'hierarchy' && !header.isPlaceholder && (
                    <div
                        onClick={(e) => handleFilterClick(e, header.column.id)}
                        style={{
                            display:'flex',
                            alignItems: 'center',
                            padding: '2px',
                            borderRadius: '4px',
                            background: filters[header.column.id] ? theme.select : 'transparent',
                            color: filters[header.column.id] ? theme.primary : 'inherit'
                        }}
                        aria-label="Filter"
                    >
                        <Icons.Filter/>
                    </div>
                )}

                {!isGroupHeader && ({ asc: <Icons.SortAsc/>, desc: <Icons.SortDesc/> }[isSorted] || null)}
                {!isGroupHeader && isSorted && isMultiSort && (
                    <span style={{fontSize: '9px', verticalAlign: 'super', marginLeft: '1px', opacity: 0.8, fontWeight: 700}}>{sortIndex + 1}</span>
                )}

                <div
                    onClick={(e) => { e.stopPropagation(); handleHeaderContextMenu(e, header.column.id); }}
                    style={{
                        display:'flex',
                        alignItems: 'center',
                        padding: '2px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        color: theme.textSec,
                        opacity: 0.6,
                        hover: { opacity: 1, background: '#eee' }
                    }}
                    aria-label="More options"
                >
                    <Icons.MoreVert/>
                </div>
                </div>

                {!isGroupHeader && activeFilterCol === header.column.id && (
                    <FilterPopover
                    column={header.column}
                    anchorEl={filterAnchorEl}
                    onClose={closeFilterPopover}
                    onFilter={(type, val) => handleHeaderFilter(header.column.id, type, val)}
                    currentFilter={filters[header.column.id]}
                    options={activeFilterCol === header.column.id ? activeFilterOptions : []}
                    theme={theme}
                    />
                )}
                {!isGroupHeader && filters[header.column.id] && filters[header.column.id].conditions && (
                <div style={{fontSize: '10px', color: theme.primary, paddingTop: '2px', textAlign: 'center'}}>
                    {filters[header.column.id].conditions.map(c => `${c.type}: ${c.value}${c.caseSensitive ? ' (Match Case)' : ''}`).join(` ${filters[header.column.id].operator} `)}
                </div>
                )}

                {header.column.getCanResize() && <div
                    onMouseDown={(e) => {
                        e.stopPropagation();
                        header.getResizeHandler()(e);
                    }}
                    onTouchStart={(e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        header.getResizeHandler()(e);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    onDoubleClick={(e) => {
                        e.stopPropagation();
                        autoSizeColumn(header.column.id);
                    }}
                    style={{
                        position: 'absolute',
                        right: -4,
                        top: 0,
                        bottom: 0,
                        width: 10,
                        cursor: 'col-resize',
                        touchAction: 'none',
                        zIndex: 4,
                        opacity: isResizeHandleVisible ? 1 : 0.14,
                        transition: 'opacity 120ms ease',
                        background: isResizeHandleVisible ? theme.primary : 'transparent'
                    }}
                />}
            </div>
        );
    };
    
    const srOnly = {
        position: 'absolute',
        width: '1px',
        height: '1px',
        padding: 0,
        margin: '-1px',
        overflow: 'hidden',
        clip: 'rect(0, 0, 0, 0)',
        whiteSpace: 'nowrap',
        border: 0,
        pointerEvents: 'none'
    };

    // Add focus styles to interactive elements
    const focusStyle = {
        outline: `2px solid ${theme.primary}`,
        outlineOffset: '2px'
    };

    return (
        <div id={id} style={{ ...styles.root, ...style }}>
            <style>{loadingAnimationStyles}</style>
            <div style={srOnly} role="status" aria-live="polite">{announcement}</div>
            <div style={styles.appBar}>
                <div style={{display:'flex', alignItems:'center', gap:'12px'}}>
                    <button onClick={() => setSidebarOpen(!sidebarOpen)} style={{border:'none', background:'transparent', cursor:'pointer', padding:'4px', borderRadius:'4px', display:'flex', color: theme.textSec}}>
                        <Icons.Menu />
                    </button>
                    <div style={{fontWeight:500,fontSize:'16px',color:theme.primary}}>Analytics Pivot</div>
                </div>
                <div style={styles.searchBox}>
                    <Icons.Search />
                    <input style={{border:'none',background:'transparent',marginLeft:'8px',outline:'none',width:'100%', color: theme.text}} placeholder="Global Search..." onChange={e=>setFilters(p=>({...p,'global':e.target.value}))}/>
                </div>
                <div style={{display:'flex',gap:'8px'}}>
                    <button style={{...styles.btn, background: showRowNumbers ? theme.select : 'transparent'}} onClick={() => setShowRowNumbers(!showRowNumbers)}>Row #</button>
                    <button style={{...styles.btn, background: showFloatingFilters ? theme.select : 'transparent'}} onClick={() => setShowFloatingFilters(!showFloatingFilters)}>Filters</button>
                    <button style={{...styles.btn, background: showRowTotals ? theme.select : 'transparent'}} onClick={() => setShowRowTotals(!showRowTotals)}>Row Totals</button>
                    <button style={{...styles.btn, background: showColTotals ? theme.select : 'transparent'}} onClick={() => setShowColTotals(!showColTotals)}>Col Totals</button>
                    <button style={{...styles.btn, background: theme.hover}} onClick={() => setSpacingMode((spacingMode + 1) % 3)}>
                        <Icons.Spacing/> {spacingLabels[spacingMode]}
                    </button>
                    <button style={{...styles.btn, background: theme.hover}} onClick={() => setLayoutMode(prev => prev === 'hierarchy' ? 'outline' : prev === 'outline' ? 'tabular' : 'hierarchy')}>
                        {layoutMode === 'hierarchy' ? 'Hierarchy' : layoutMode === 'outline' ? 'Outline' : 'Tabular'}
                    </button>
                    <button style={{...styles.btn, background: colorScale ? theme.select : 'transparent'}} onClick={() => setColorScale(!colorScale)}>Color Scale</button>
                    
                    <div style={{width: '1px', height: '20px', background: theme.border, margin: '0 4px'}} />
                    <select value={themeName} onChange={e => setThemeName(e.target.value)} style={{...styles.btn, background: theme.hover, padding: '4px 8px', outline: 'none'}}>
                        {Object.keys(themes).map(t => <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>)}
                    </select>

                    <button style={{
                        ...styles.btn,
                        background: theme.primary,
                        color: '#fff',
                        border: `1px solid ${theme.primary}`,
                        '&:hover': {
                            background: theme.primary,
                            filter: 'brightness(1.1)'
                        }
                    }} onClick={exportPivot}>
                        <Icons.Export/> {(rowCount || 0) > 500000 ? 'Export CSV' : 'Export'}
                    </button>
                </div>
            </div>
            <div style={{display:'flex', flex:1, overflow:'hidden'}}>
                {sidebarOpen && (
                <div style={styles.sidebar} role="complementary" aria-label="Tool Panel">
                    <div style={{display: 'flex', borderBottom: `1px solid ${theme.border}`, marginBottom: '16px'}}>
                        <div 
                            onClick={() => setSidebarTab('fields')}
                            style={{
                                padding: '8px 16px', cursor: 'pointer', 
                                borderBottom: sidebarTab === 'fields' ? `2px solid ${theme.primary}` : 'none',
                                fontWeight: sidebarTab === 'fields' ? 600 : 400,
                                color: sidebarTab === 'fields' ? theme.primary : theme.textSec
                            }}
                        >Fields</div>
                        <div 
                            onClick={() => setSidebarTab('filters')}
                            style={{
                                padding: '8px 16px', cursor: 'pointer', 
                                borderBottom: sidebarTab === 'filters' ? `2px solid ${theme.primary}` : 'none',
                                fontWeight: sidebarTab === 'filters' ? 600 : 400,
                                color: sidebarTab === 'filters' ? theme.primary : theme.textSec,
                                display: 'flex', alignItems: 'center', gap: '6px'
                            }}
                        >
                            Filters
                            {Object.keys(filters).length > 0 && (
                                <div style={{width: '6px', height: '6px', borderRadius: '50%', background: '#d32f2f'}} />
                            )}
                        </div>
                        <div 
                            onClick={() => setSidebarTab('columns')}
                            style={{
                                padding: '8px 16px', cursor: 'pointer',
                                borderBottom: sidebarTab === 'columns' ? `2px solid ${theme.primary}` : 'none',
                                fontWeight: sidebarTab === 'columns' ? 600 : 400,
                                color: sidebarTab === 'columns' ? theme.primary : theme.textSec
                            }}
                        >Columns</div>
                    </div>

                    {sidebarTab === 'filters' ? (
                        <div style={{flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', padding: '8px'}}>
                            <div style={{marginBottom: '10px', display: 'flex', alignItems: 'center', background: theme.background, borderRadius: '6px', padding: '4px 8px', border: `1px solid ${theme.border}`}}>
                                <Icons.Search />
                                <input 
                                    placeholder="Search columns..."
                                    value={colSearch} 
                                    onChange={e => setColSearch(e.target.value)} 
                                    style={{border:'none', background:'transparent', marginLeft:'10px', outline:'none', width:'100%', color: theme.text, fontSize: '13px'}}
                                />
                            </div>
                            {(() => {
                                const allFields = availableFields;
                                const colsForDisplay = allFields.map(field => {
                                    const tableCol = table.getColumn(field);
                                    if (tableCol) return tableCol;
                                    return { id: field, header: field, columnDef: { header: field } };
                                });
                                const filtered = colsForDisplay.filter(col => {
                                    const header = (col.columnDef && typeof col.columnDef.header === 'string') ? col.columnDef.header : (typeof col.header === 'string' ? col.header : col.id);
                                    return String(header).toLowerCase().includes(colSearch.toLowerCase()) || col.id.toLowerCase().includes(colSearch.toLowerCase());
                                });
                                return (
                                    <div style={{display: 'flex', flexDirection: 'column'}}>
                                        {filtered.map(col => (
                                            <SidebarFilterItem
                                                key={col.id}
                                                column={col}
                                                theme={theme}
                                                styles={styles}
                                                onFilter={(val) => handleHeaderFilter(col.id, val)}
                                                currentFilter={filters[col.id]}
                                                options={[]}
                                            />
                                        ))}
                                    </div>
                                );
                            })()}
                        </div>
                    ) : sidebarTab !== 'columns' ? (
                        <>
                            {sidebarTab === 'fields' && (
                                <div>
                                    <div style={styles.sectionTitle}>Available Fields</div>
                                    {availableFields.map(f => (
                                        <div key={f} draggable onDragStart={e=>onDragStart(e,f,'pool')} style={styles.chip}>
                                            <div style={{display:'flex',gap:'6px'}}><Icons.DragIndicator/> {f}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                            {[{id:'rows', label:'Rows'}, {id:'cols', label:'Columns'}, {id:'vals', label:'Values'}, {id:'filter', label:'Filters'}].map(zone => (
                                <div key={zone.id}>
                                    <div style={styles.sectionTitle}>{zone.label}</div>
                                    {zone.id === 'rows' && rowFields.length > 0 && (
                                        <div style={{display:'flex', gap:'4px', padding:'0 4px 6px 4px'}}>
                                            <button
                                                title="Expand all hierarchy levels"
                                                onClick={() => handleExpandAllRows(true)}
                                                style={{flex:1, border:`1px solid ${theme.primary}`, background:theme.select, cursor:'pointer', padding:'3px 6px', fontSize:'11px', color:theme.primary, borderRadius:'4px', fontWeight:600}}
                                            >+ Expand All</button>
                                            <button
                                                title="Collapse all rows"
                                                onClick={() => handleExpandAllRows(false)}
                                                style={{flex:1, border:`1px solid ${theme.border}`, background:theme.background, cursor:'pointer', padding:'3px 6px', fontSize:'11px', color:theme.textSec, borderRadius:'4px'}}
                                            >- Collapse All</button>
                                        </div>
                                    )}
                                    <div style={styles.dropZone} onDragOver={e=>e.preventDefault()} onDrop={e=>onDrop(e, zone.id)}>
                                        {(zone.id==='filter' ? Object.keys(filters).filter(k=>k!=='global') : zone.id==='rows'?rowFields:zone.id==='cols'?colFields:valConfigs).map((item, idx) => {
                                            const label = zone.id==='vals' ? item.field : item;
                                            return (
                                                                                        <div key={idx} draggable onDragStart={e=>onDragStart(e,item,zone.id,idx)} onDragOver={e=>onDragOver(e,zone.id,idx)} >
                                                                                            <div style={styles.chip}>
                                                                                                {dropLine && dropLine.zone===zone.id && dropLine.idx===idx && <div style={{...styles.dropLine,top:-2}}/>}
                                                                                                <div style={{display:'flex',gap:'6px'}}><Icons.DragIndicator/> <b>{label}</b></div>
                                                                                                {zone.id === 'vals' && (
                                                                                                    <div style={{display:'flex',flexDirection:'column', gap:2}}>
                                                                                                        <div style={{display:'flex', gap:2}}>
                                                                                                            <select value={item.agg} onChange={e=>{const n=[...valConfigs];n[idx].agg=e.target.value;setValConfigs(n)}} style={{border:'none',background:'transparent',color:theme.primary,cursor:'pointer',maxWidth:'50px',fontSize:'11px'}}><option value="sum">Sum</option><option value="avg">Avg</option><option value="count">Cnt</option><option value="min">Min</option><option value="max">Max</option></select>
                                                                                                            <select value={item.windowFn || 'none'} onChange={e=>{const n=[...valConfigs];n[idx].windowFn=e.target.value==='none'?null:e.target.value;setValConfigs(n)}} style={{border:'none',background:'transparent',color:theme.primary,cursor:'pointer',maxWidth:'60px',fontSize:'11px'}}><option value="none">Norm</option><option value="percent_of_row">%Row</option><option value="percent_of_col">%Col</option><option value="percent_of_grand_total">%Tot</option></select>
                                                                                                        </div>
                                                                                                        <input placeholder="Fmt (currency)" value={item.format || ''} onChange={e=>{const n=[...valConfigs];n[idx].format=e.target.value;setValConfigs(n)}} style={{border:'1px solid #eee', fontSize:'10px', padding:'2px', width:'100%'}}/>
                                                                                                    </div>
                                                                                                )}
                                                                                                <div style={{display:'flex', gap:'4px', marginLeft:'auto', alignItems: 'center'}}>
                                                                                                    {zone.id==='filter' && (
                                                                                                        <div 
                                                                                                            onClick={(e) => handleFilterClick(e, label)} 
                                                                                                            style={{
                                                                                                                cursor:'pointer', 
                                                                                                                display:'flex', 
                                                                                                                alignItems:'center',
                                                                                                                padding: '2px',
                                                                                                                borderRadius: '4px',
                                                                                                                background: (filters[label] && (filters[label].conditions?.length > 0 || (typeof filters[label] === 'string' && filters[label].length > 0))) ? theme.select : 'transparent',
                                                                                                                color: (filters[label] && (filters[label].conditions?.length > 0 || (typeof filters[label] === 'string' && filters[label].length > 0))) ? theme.primary : 'inherit'
                                                                                                            }}
                                                                                                        >
                                                                                                            <Icons.Filter />
                                                                                                        </div>
                                                                                                    )}
                                                                                                    <span onClick={()=>{
                                                                                                        if (zone.id==='filter'){const n={...filters};delete n[label];setFilters(n)}
                                                                                                        if (zone.id==='rows') setRowFields(p=>p.filter(x=>x!==label))
                                                                                                        if (zone.id==='cols') setColFields(p=>p.filter(x=>x!==label))
                                                                                                        if (zone.id==='vals') setValConfigs(p=>p.filter((_,i)=>i!==idx))
                                                                                                    }} style={{cursor:'pointer'}}><Icons.Close/></span>
                                                                                                </div>
                                                                                                                                                {zone.id === 'filter' && activeFilterCol === label && (
                                                                                                                    <FilterPopover 
                                                                                                                        column={{header: label, id: label}} 
                                                                                                                        anchorEl={filterAnchorEl}
                                                                                                                        onClose={closeFilterPopover}
                                                                                                                        onFilter={(filterValue) => handleHeaderFilter(label, filterValue)}
                                                                                                                        currentFilter={filters[label]}
                                                                                                                        options={filterOptions[label] || []}
                                                                                                                        theme={theme}
                                                                                                                    />
                                                                                                                                                )}
                                                                                                                                    {dropLine && dropLine.zone===zone.id && dropLine.idx===idx+1 && <div style={{...styles.dropLine,bottom:-2}}/>}
                                                                                            </div>
                                                                                                                                        {zone.id ==='filter' && filters[label] && filters[label].conditions && (
                                                                                                                                            <div style={{fontSize: '10px', color: theme.primary, padding: '0 8px 4px 8px', marginTop: '-4px'}}>
                                                                                                                                                {filters[label].conditions.map(c => `${c.type}: ${c.value}${c.caseSensitive ? ' (Match Case)' : ''}`).join(` ${filters[label].operator} `)}
                                                                                                                                            </div>
                                                                                                                                        )}
                                                                                                                        </div>
                                            )
                                        })}
                                        {(zone.id==='filter' ? Object.keys(filters).filter(k=>k!=='global') : zone.id==='rows'?rowFields:zone.id==='cols'?colFields:valConfigs).length === 0 && (
                                            <div style={{opacity:0.5, fontSize:'11px', padding:'8px', textAlign:'center', pointerEvents:'none'}}>Drag fields here</div>
                                        )}
                                        <div style={{height:20}} onDragOver={e=>onDragOver(e,zone.id,(zone.id==='rows'?rowFields:zone.id==='cols'?colFields:zone.id==='vals'?valConfigs:Object.keys(filters).filter(k=>k!=='global')).length)} />
                                    </div>
                                </div>
                            ))}
                        </>
                    ) : (
                        <div style={{display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', overflow: 'hidden'}}>
                            {/* Enhanced Search Header */}
                            <div style={{display: 'flex', flexDirection: 'column', gap: '10px', padding: '8px', background: theme.headerBg, borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.08)'}}>
                                <div style={{display: 'flex', alignItems: 'center', background: theme.background, borderRadius: '6px', padding: '8px 12px', border: `2px solid ${theme.border}`, transition: 'border-color 0.2s'}}>
                                    <Icons.Search />
                                    <input 
                                        placeholder="Search columns..."
                                        value={colSearch} 
                                        onChange={e => setColSearch(e.target.value)} 
                                        style={{
                                            border:'none', 
                                            background:'transparent', 
                                            marginLeft:'10px', 
                                            outline:'none', 
                                            width:'100%', 
                                            color: theme.text, 
                                            fontSize: '13px',
                                            fontWeight: 500
                                        }} 
                                    />
                                    {colSearch && (
                                        <span 
                                            onClick={() => setColSearch('')} 
                                            style={{
                                                cursor: 'pointer', 
                                                display: 'flex', 
                                                padding: '4px',
                                                borderRadius: '4px',
                                                background: theme.hover,
                                                transition: 'background 0.2s'
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.background = theme.select}
                                            onMouseLeave={e => e.currentTarget.style.background = theme.hover}
                                        >
                                            <Icons.Close />
                                        </span>
                                    )}
                                </div>
                                
                                {/* Type Filter Pills */}
                                <div style={{display: 'flex', gap: '6px', flexWrap: 'wrap'}}>
                                    {[{"value": "all", "label": "All", "icon": "📊"},
                                        {"value": "number", "label": "Numbers", "icon": "🔢"},
                                        {"value": "string", "label": "Text", "icon": "📝"},
                                        {"value": "date", "label": "Dates", "icon": "📅"}
                                    ].map(type => (
                                        <button
                                            key={type.value}
                                            onClick={() => setColTypeFilter(type.value)}
                                            style={{
                                                padding: '6px 12px',
                                                borderRadius: '6px',
                                                border: 'none',
                                                background: colTypeFilter === type.value ? theme.primary : theme.background,
                                                color: colTypeFilter === type.value ? '#fff' : theme.text,
                                                cursor: 'pointer',
                                                fontSize: '11px',
                                                fontWeight: 600,
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '4px',
                                                transition: 'all 0.2s',
                                                boxShadow: colTypeFilter === type.value ? '0 2px 4px rgba(0,0,0,0.1)' : 'none'
                                            }}
                                            onMouseEnter={e => {
                                                if (colTypeFilter !== type.value) {
                                                    e.currentTarget.style.background = theme.hover;
                                                }
                                            }}
                                            onMouseLeave={e => {
                                                if (colTypeFilter !== type.value) {
                                                    e.currentTarget.style.background = theme.background;
                                                }
                                            }}
                                        >
                                            <span>{type.icon}</span>
                                            <span>{type.label}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* Enhanced Action Buttons */}
                            <div style={{display: 'flex', gap: '6px', padding: '0 8px'}}>
                                <button 
                                    onClick={() => table.toggleAllColumnsVisible(true)} 
                                    style={{
                                        ...styles.btn, 
                                        padding: '8px 12px', 
                                        fontSize: '11px', 
                                        flex: 1, 
                                        justifyContent: 'center',
                                        background: theme.background,
                                        fontWeight: 600,
                                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                        transition: 'all 0.2s'
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'}
                                    onMouseLeave={e => e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)'}
                                >
                                    <Icons.Visibility style={{fontSize: '14px'}} />
                                    <span>Show All</span>
                                </button>
                                <button 
                                    onClick={() => table.toggleAllColumnsVisible(false)} 
                                    style={{
                                        ...styles.btn, 
                                        padding: '8px 12px', 
                                        fontSize: '11px', 
                                        flex: 1, 
                                        justifyContent: 'center',
                                        background: theme.background,
                                        fontWeight: 600,
                                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                        transition: 'all 0.2s'
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'}
                                    onMouseLeave={e => e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)'}
                                >
                                    <Icons.VisibilityOff style={{fontSize: '14px'}} />
                                    <span>Hide All</span>
                                </button>
                                <button 
                                    onClick={() => toggleAllColumnsPinned(false)} 
                                    style={{
                                        ...styles.btn, 
                                        padding: '8px 12px', 
                                        fontSize: '11px', 
                                        flex: 1, 
                                        justifyContent: 'center',
                                        background: theme.background,
                                        fontWeight: 600,
                                        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                        transition: 'all 0.2s'
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'}
                                    onMouseLeave={e => e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)'}
                                >
                                    <Icons.Unpin style={{fontSize: '14px'}} />
                                    <span>Unpin All</span>
                                </button>
                            </div>

                            {/* Enhanced Selection Bar */}
                            {selectedCols.size > 0 && (
                                <div style={{
                                    display: 'flex', 
                                    flexDirection: 'column',
                                    gap: '8px', 
                                    background: `linear-gradient(135deg, ${theme.select}ee, ${theme.select}dd)`, 
                                    padding: '12px', 
                                    borderRadius: '8px', 
                                    border: `2px solid ${theme.primary}66`,
                                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                                    margin: '0 8px'
                                }}>
                                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                        <span style={{
                                            fontSize: '12px', 
                                            fontWeight: 700, 
                                            padding: '4px 10px', 
                                            color: theme.primary, 
                                            background: theme.background,
                                            borderRadius: '6px',
                                            boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                                        }}>
                                            {selectedCols.size} Column{selectedCols.size > 1 ? 's' : ''} Selected
                                        </span>
                                        <button 
                                            onClick={() => setSelectedCols(new Set())} 
                                            style={{
                                                border: 'none', 
                                                background: theme.background, 
                                                cursor: 'pointer', 
                                                display: 'flex', 
                                                color: theme.textSec,
                                                padding: '4px',
                                                borderRadius: '4px',
                                                transition: 'all 0.2s'
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.background = theme.hover}
                                            onMouseLeave={e => e.currentTarget.style.background = theme.background}
                                        >
                                            <Icons.Close/>
                                        </button>
                                    </div>
                                    <div style={{display: 'flex', gap: '6px', flexWrap: 'wrap'}}>
                                        <button 
                                            onClick={() => {
                                                Array.from(selectedCols).forEach(id => handlePinColumn(id, 'left')); 
                                                setSelectedCols(new Set()); 
                                            }} 
                                            style={{
                                                ...styles.btn, 
                                                padding: '6px 12px', 
                                                fontSize: '11px', 
                                                background: theme.background,
                                                flex: 1,
                                                justifyContent: 'center',
                                                fontWeight: 600,
                                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                                transition: 'all 0.2s'
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'}
                                            onMouseLeave={e => e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)'}
                                        >
                                            <Icons.PinLeft />
                                            <span>Pin Left</span>
                                        </button>
                                        <button 
                                            onClick={() => {
                                                Array.from(selectedCols).forEach(id => handlePinColumn(id, false)); 
                                                setSelectedCols(new Set()); 
                                            }} 
                                            style={{
                                                ...styles.btn, 
                                                padding: '6px 12px', 
                                                fontSize: '11px', 
                                                background: theme.background,
                                                flex: 1,
                                                justifyContent: 'center',
                                                fontWeight: 600,
                                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                                transition: 'all 0.2s'
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'}
                                            onMouseLeave={e => e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)'}
                                        >
                                            <Icons.Unpin />
                                            <span>Unpin</span>
                                        </button>
                                        <button 
                                            onClick={() => {
                                                Array.from(selectedCols).forEach(id => handlePinColumn(id, 'right')); 
                                                setSelectedCols(new Set()); 
                                            }} 
                                            style={{
                                                ...styles.btn, 
                                                padding: '6px 12px', 
                                                fontSize: '11px', 
                                                background: theme.background,
                                                flex: 1,
                                                justifyContent: 'center',
                                                fontWeight: 600,
                                                boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
                                                transition: 'all 0.2s'
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'}
                                            onMouseLeave={e => e.currentTarget.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)'}
                                        >
                                            <Icons.PinRight />
                                            <span>Pin Right</span>
                                        </button>
                                    </div>
                                </div>
                            )}

                            <div style={{flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column'}}>
                                {(() => {
                                    const allCols = table.getAllColumns().filter(c => !c.parent && c.id !== 'no_data');
                                    
                                    const filteredCols = allCols.filter(col => {
                                        if (colTypeFilter === 'all') return true;
                                        let type = 'string';
                                        
                                        // Check if it's a value column (aggregated)
                                        const config = valConfigs.find(v => col.id.includes(v.field));
                                        
                                        if (config) {
                                            type = 'number';
                                        } else if (data && data.length > 0) {
                                            // Check the first row's data for this column
                                            // Handle both flat data and nested/grouped data structures
                                            const firstRow = data[0];
                                            let val = firstRow[col.id];
                                            
                                            // If value is undefined in flat data, try to find it via accessor if possible, or skip
                                            if (val === undefined && col.accessorFn) {
                                                try {
                                                    val = col.accessorFn(firstRow);
                                                } catch (e) { /* ignore */ }
                                            }

                                            if (typeof val === 'number') {
                                                type = 'number';
                                            } else if (val instanceof Date) {
                                                type = 'date';
                                            } else if (typeof val === 'string') {
                                                if (!isNaN(Number(val)) && val.trim() !== '') {
                                                    type = 'number';
                                                } else if (!isNaN(Date.parse(val)) && val.includes('-')) {
                                                    type = 'date';
                                                }
                                            }
                                        }
                                        return type === colTypeFilter;
                                    });

                                    const leftPinned = filteredCols.filter(c => hasChildrenInZone(c, 'left'));
                                    const rightPinned = filteredCols.filter(c => hasChildrenInZone(c, 'right'));
                                    const unpinned = filteredCols.filter(c => hasChildrenInZone(c, 'unpinned'));

                                    const renderColList = (cols, sectionId) => cols.map(column => (
                                        <ColumnTreeItem 
                                            key={column.id} 
                                            column={column} 
                                            level={0} 
                                            theme={theme} 
                                            styles={styles} 
                                            handlePinColumn={handlePinColumn}
                                            colSearch={colSearch}
                                            selectedCols={selectedCols}
                                            setSelectedCols={setSelectedCols}
                                            onDrop={handleToolPanelDrop}
                                            sectionId={sectionId}
                                        />
                                    ));

                                    const handleToolPanelDrop = (colId, sectionId, targetColId) => {
                                        let targetIndex = undefined;
                                        const currentPinning = columnPinning || { left: [], right: [] };

                                        if (targetColId) {
                                            const list = sectionId === 'left' ? currentPinning.left : (sectionId === 'right' ? currentPinning.right : null);
                                            if (list) {
                                                targetIndex = list.indexOf(targetColId);
                                                
                                                // If not found, maybe it's a group?
                                                if (targetIndex === -1) {
                                                    const targetCol = table.getColumn(targetColId);
                                                    if (targetCol && isGroupColumn(targetCol)) {
                                                        const leaves = getAllLeafIdsFromColumn(targetCol);
                                                        const firstPinnedLeaf = leaves.find(id => list.includes(id));
                                                        if (firstPinnedLeaf) {
                                                            targetIndex = list.indexOf(firstPinnedLeaf);
                                                        }
                                                    }
                                                }

                                                if (targetIndex === -1) targetIndex = undefined;
                                            }
                                        }

                                        if (sectionId === 'left') handlePinColumn(colId, 'left', targetIndex);
                                        else if (sectionId === 'right') handlePinColumn(colId, 'right', targetIndex);
                                        else handlePinColumn(colId, false);
                                    };

                                    return (
                                        <>
                                            {leftPinned.length > 0 && (
                                                <ToolPanelSection 
                                                    title="Pinned Left" 
                                                    count={leftPinned.length} 
                                                    theme={theme} 
                                                    styles={styles}
                                                    sectionId="left"
                                                    onDrop={handleToolPanelDrop}
                                                >
                                                    {renderColList(leftPinned, 'left')}
                                                </ToolPanelSection>
                                            )}

                                            {rightPinned.length > 0 && (
                                                <ToolPanelSection 
                                                    title="Pinned Right" 
                                                    count={rightPinned.length} 
                                                    theme={theme} 
                                                    styles={styles}
                                                    sectionId="right"
                                                    onDrop={handleToolPanelDrop}
                                                >
                                                    {renderColList(rightPinned, 'right')}
                                                </ToolPanelSection>
                                            )}
                                            
                                            <ToolPanelSection 
                                                title="Columns" 
                                                count={unpinned.length} 
                                                theme={theme} 
                                                styles={styles}
                                                sectionId="unpinned"
                                                onDrop={handleToolPanelDrop}
                                            >
                                                {renderColList(unpinned, 'unpinned')}
                                            </ToolPanelSection>
                                        </>
                                    );
                                })()}
                            </div>
                            
                            {pinningPresets && pinningPresets.length > 0 && (
                                <div style={{padding: '8px', borderTop: `1px solid ${theme.border}`}}>
                                    <div style={styles.sectionTitle}>Pinning Presets</div>
                                    <div style={{display: 'flex', gap: '4px', flexWrap: 'wrap'}}>
                                        {pinningPresets.map((preset, i) => (
                                            <button 
                                                key={i}
                                                onClick={() => {
                                                    setColumnPinning(preset.config);
                                                    showNotification(`Applied preset: ${preset.name}`);
                                                }}
                                                style={{...styles.btn, fontSize: '11px', background: theme.headerBg}}
                                            >
                                                {preset.name}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div style={{fontSize: '11px', color: theme.textSec, fontStyle: 'italic', padding: '8px', background: theme.headerBg, borderTop: `1px solid ${theme.border}44`}}>
                                <Icons.Lock style={{ verticalAlign: 'middle', marginRight: '4px', opacity: 0.5 }} />
                                Drag columns or use pin icons to freeze areas.
                            </div>
                        </div>
                    )}
                </div>
                )}
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
                                 const headerHeight = (table.getHeaderGroups().length * rowHeight) + (showFloatingFilters ? rowHeight : 0);
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
                                 const headerHeight = (table.getHeaderGroups().length * rowHeight) + (showFloatingFilters ? rowHeight : 0);
                                 const topOffset = headerHeight + (effectiveTopRows.length * rowHeight);
                                 
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
                                             background: (row.original && row.original._isTotal) ? (isDarkTheme(theme) ? '#1a2e1a' : '#f0f7f0') : theme.background,
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
            </div>
            {contextMenu && <ContextMenu {...contextMenu} onClose={() => setContextMenu(null)} />}
            {notification && <Notification message={notification.message} type={notification.type} onClose={() => setNotification(null)} />}
            <DrillThroughModal
                drillState={drillModal}
                onClose={() => setDrillModal(null)}
                onPageChange={(newPage) => {
                    if (!drillModal) return;
                    fetchDrillData(drillModal.path, newPage, drillModal.sortCol, drillModal.sortDir, drillModal.filterText);
                }}
                onSort={(col, dir) => {
                    if (!drillModal) return;
                    fetchDrillData(drillModal.path, 0, col, dir, drillModal.filterText);
                }}
                onFilter={(text) => {
                    if (!drillModal) return;
                    fetchDrillData(drillModal.path, 0, drillModal.sortCol, drillModal.sortDir, text);
                }}
            />
        </div>
    );
};

DashTanstackPivot.propTypes = {
    id: PropTypes.string,
    table: PropTypes.string,
        data: PropTypes.arrayOf(PropTypes.object),
        setProps: PropTypes.func,
        style: PropTypes.object,
        serverSide: PropTypes.bool,
        rowCount: PropTypes.number,
        rowFields: PropTypes.array,
        colFields: PropTypes.array,
        valConfigs: PropTypes.array,
        filters: PropTypes.object,
        sorting: PropTypes.array,
        expanded: PropTypes.oneOfType([PropTypes.object, PropTypes.bool]),
        columns: PropTypes.array,
    
    showRowTotals: PropTypes.bool,
    showColTotals: PropTypes.bool,
    grandTotalPosition: PropTypes.oneOf(['top', 'bottom']),
    filterOptions: PropTypes.object,
    viewport: PropTypes.object,
    cellUpdate: PropTypes.object,
    cellUpdates: PropTypes.arrayOf(PropTypes.object),
    rowMove: PropTypes.object,
    drillThrough: PropTypes.object,
    drillEndpoint: PropTypes.string,
    conditionalFormatting: PropTypes.arrayOf(PropTypes.object),
    validationRules: PropTypes.object,
    columnPinning: PropTypes.shape({
        left: PropTypes.arrayOf(PropTypes.string),
        right: PropTypes.arrayOf(PropTypes.string)
    }),
    rowPinning: PropTypes.shape({
        top: PropTypes.arrayOf(PropTypes.string),
        bottom: PropTypes.arrayOf(PropTypes.string)
    }),
    columnPinned: PropTypes.object,
    rowPinned: PropTypes.object,
    columnVisibility: PropTypes.object,
    columnSizing: PropTypes.object,
    reset: PropTypes.any,
    persistence: PropTypes.oneOfType([PropTypes.bool, PropTypes.string, PropTypes.number]),
    persistence_type: PropTypes.oneOf(['local', 'session', 'memory']),
    pinningOptions: PropTypes.shape({
        maxPinnedLeft: PropTypes.number,
        maxPinnedRight: PropTypes.number,
        suppressMovable: PropTypes.bool,
        lockPinned: PropTypes.bool
    }),
    pinningPresets: PropTypes.arrayOf(PropTypes.shape({
        name: PropTypes.string,
        config: PropTypes.object
    })),
    sortOptions: PropTypes.shape({
        naturalSort: PropTypes.bool,
        caseSensitive: PropTypes.bool,
        columnOptions: PropTypes.object
    }),
    sortLock: PropTypes.bool,
    sortEvent: PropTypes.object,
    availableFieldList: PropTypes.arrayOf(PropTypes.string),
    dataOffset: PropTypes.number,
    dataVersion: PropTypes.number,
};

