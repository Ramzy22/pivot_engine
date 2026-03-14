// DashTanstackPivot - Enterprise Grade Pivot Table
import React, { useMemo, useState, useRef, useEffect, useCallback } from 'react';
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
        dataOffset = 0,
        dataVersion = 0
    } = props;



    // --- Persistence Helper ---
    const getStorage = () => {
        if (persistence_type === 'local') return window.localStorage;
        if (persistence_type === 'session') return window.sessionStorage;
        return null;
    };

    const loadPersistedPinning = (key, defaultValue) => {
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
            if (serverSide && props.columns) return props.columns.map(c => c.id || c);

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
        const [columnPinning, setColumnPinning] = useState(() => loadPersistedPinning('columnPinning', initialColumnPinning));
        const [rowPinning, setRowPinning] = useState(() => loadPersistedPinning('rowPinning', initialRowPinning));
        const [layoutMode, setLayoutMode] = useState('hierarchy'); // hierarchy, tabular
        const [columnVisibility, setColumnVisibility] = useState(initialColumnVisibility);
        const [announcement, setAnnouncement] = useState("");
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
                    reset: null
                });
            }
        }
    }, [reset, initialRowFields, initialColFields, initialValConfigs, initialColumnPinning, initialRowPinning]);

        // Save Persistence
        useEffect(() => {
            if (!persistence) return;
            const storage = getStorage();
            if (!storage) return;
            storage.setItem(`${id}-columnPinning`, JSON.stringify(columnPinning));
            storage.setItem(`${id}-rowPinning`, JSON.stringify(rowPinning));
        }, [id, columnPinning, rowPinning, persistence, persistence_type]);

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
        columnVisibility: {}
    });

    React.useEffect(() => {
        const nextProps = {
            rowFields, colFields, valConfigs, filters, sorting, expanded,
            showRowTotals, showColTotals, columnPinning, rowPinning, columnVisibility
        };

        const changed = Object.keys(nextProps).some(key => {
            const val = nextProps[key];
            const lastVal = lastPropsRef.current[key];
            // Use JSON.stringify for comparison instead of isEqual function
            return JSON.stringify(val) !== JSON.stringify(lastVal);
        });

        if (setPropsRef.current && changed) {
            console.log('[DEBUG] Sync to Dash Triggered', nextProps);
            lastPropsRef.current = nextProps;
            setPropsRef.current(nextProps);
        }
    }, [rowFields, colFields, valConfigs, filters, sorting, expanded, showRowTotals, showColTotals, columnPinning, rowPinning, columnVisibility]); // Remove setProps from dependencies

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
                console.log('[DEBUG] Pinning Enforcement Triggered', nextLeft);
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
            onClick: exportExcel
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
             if (setProps) {
                 // Construct drill through context
                 const drillFilters = { ...filters };
                 // Add row group filters
                 let currentRow = row;
                 while (currentRow) {
                     if (currentRow.groupingColumnId) {
                         drillFilters[currentRow.groupingColumnId] = { operator: 'AND', conditions: [{ type: 'eq', value: currentRow.groupingValue }] };
                     }
                     currentRow = currentRow.getParentRow();
                 }
                 
                 setProps({
                     drillThrough: { 
                         rowId, 
                         colId, 
                         filters: drillFilters,
                         value
                     } 
                 });
             }
        }});

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
                                            console.log('[DEBUG] Toggling expansion (hierarchy) for', row.id);
                                            row.getToggleExpandedHandler()(e);
                                        }}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        style={{border:'none',background:'none',cursor:'pointer',padding:0,marginRight:'6px',color:'#757575',display:'flex'}}
                                    >
                                        {row.getIsExpanded() ? <Icons.ChevronDown/> : <Icons.ChevronRight/>}
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
                                            console.log('[DEBUG] Toggling expansion (mode) for', row.id);
                                            row.getToggleExpandedHandler()(e);
                                        }}
                                        onMouseDown={(e) => e.stopPropagation()}
                                        style={{border:'none',background:'none',cursor:'pointer',padding:0,marginRight:'6px',color:'#757575',display:'flex'}}
                                    >
                                        {row.getIsExpanded() ? <Icons.ChevronDown/> : <Icons.ChevronRight/>}
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
            // Prioritize explicit columns from backend if available (prevents sparse data issues)
            if (props.columns && props.columns.length > 0) {
                props.columns.forEach(c => keys.add(c.id));
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
                                                     <div style={{width:'100%', height:'100%', display:'flex', alignItems:'center', justifyContent:'flex-end', paddingRight:'8px', fontWeight:'bold', background:'#fafafa'}} onContextMenu={e => handleContextMenu(e, info.getValue(), info.column.id, info.row)}>
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
    }, [filteredData, rowFields, colFields, valConfigs, minMax, colorScale, colExpanded, serverSide, layoutMode, showRowNumbers, isRowSelecting, rowDragStart, props.columns]); // Removed selectedCells to prevent infinite re-renders

    const parentRef = useRef(null);
    const rowHeight = rowHeights[spacingMode] || 32;

    const serverSideCacheKey = useMemo(() => JSON.stringify({
        sorting,
        filters,
        rowFields,
        colFields,
        valConfigs,
        expanded,
        rowCount
    }), [sorting, filters, rowFields, colFields, valConfigs, expanded, rowCount]);
    const serverSideViewportResetKey = useMemo(() => JSON.stringify({
        sorting,
        filters,
        rowFields,
        colFields,
        valConfigs,
        rowCount
    }), [sorting, filters, rowFields, colFields, valConfigs, rowCount]);

    const serverSidePinsGrandTotal = serverSide && showColTotals;
    const effectiveRowCount = serverSidePinsGrandTotal && rowCount ? Math.max(rowCount - 1, 0) : rowCount;

    const { rowVirtualizer, getRow, renderedData, renderedOffset, clearCache, grandTotalRow } = useServerSideRowModel({
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
        excludeGrandTotal: serverSidePinsGrandTotal
    });

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
            columnVisibility
        };
    }, [sorting, expanded, columnPinning, rowPinning, rowFields, columnVisibility, tableData, grandTotalPosition]);



    const handleExpandAllRows = (shouldExpand) => {
        if (serverSide) {
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
        onSortingChange: (updater) => { console.log('[DEBUG] onSortingChange'); handleSortingChange(updater); },
        onExpandedChange: (updater) => { console.log('[DEBUG] onExpandedChange'); setExpanded(updater); },
        onColumnPinningChange: (updater) => { console.log('[DEBUG] onColumnPinningChange'); setColumnPinning(updater); },
        onRowPinningChange: (updater) => { console.log('[DEBUG] onRowPinningChange'); setRowPinning(updater); },
        onColumnVisibilityChange: (updater) => { console.log('[DEBUG] onColumnVisibilityChange'); setColumnVisibility(updater); },
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
        if (parentRef.current) {
            parentRef.current.scrollTop = 0;
        }
    }, [serverSide, serverSideViewportResetKey, parentRef]);

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

    useEffect(() => {
        if (!serverSide) return;
        console.log('[pivot-client-table]', {
            renderedOffset,
            renderedDataLength: renderedData.length,
            rowCount,
            centerRows: centerRows.length,
            topRows: topRows.length,
            bottomRows: bottomRows.length,
            effectiveCenterRows: effectiveCenterRows.length,
            grandTotalPresent: !!grandTotalRow,
            centerSample: effectiveCenterRows.slice(0, 5).map(row => ({
                id: row.id,
                path: row.original ? row.original._path : null,
                isTotal: !!(row.original && row.original._isTotal)
            }))
        });
    }, [serverSide, renderedOffset, renderedData, rowCount, centerRows, topRows, bottomRows, effectiveCenterRows, grandTotalRow]);

    const visibleLeafColumns = table.getVisibleLeafColumns();

    // 1. Row Virtualizer (Managed by useServerSideRowModel)
    const virtualRows = rowVirtualizer.getVirtualItems();

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

    // Use the custom hook
    const { getHeaderStickyStyle, getStickyStyle } = useStickyStyles(
        visibleLeafColumns,
        columnPinning,
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





    const exportExcel = () => {
        const rowsToExport = rows.map(r => {
             const d = { 'Hierarchy': r.original._id };
             columns.forEach(c => {
                 const visit = col => {
                     if (col.columns) col.columns.forEach(visit);
                     else if (col.accessorKey) d[col.header] = r.getValue(col.accessorKey);
                 };
                 visit(c);
             });
             return d;
        });
        const ws = XLSX.utils.json_to_sheet(rowsToExport);
        const wb = XLSX.utils.book_new();
        XLSX.utils.book_append_sheet(wb, ws, "Pivot");
        const buf = XLSX.write(wb, { bookType:'xlsx', type:'array' });
        saveAs(new Blob([buf], {type:'application/octet-stream'}), 'pivot.xlsx');
    };

    // --- Helper to Render a single Cell with useCallback ---
            const renderCell = useCallback((cell, virtualRowIndex, isLastPinnedLeft = false, isFirstPinnedRight = false, isVirtualRow = false) => {
                if (!cell) return null;
                
                const row = cell.row;
                const col = cell.column;
                const colIndex = table.getVisibleLeafColumns().findIndex(c => c.id === col.id);
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
    
            const rowBackground = (row.original && row.original._isTotal) ? '#e8f5e9' : (isDarkTheme(theme) ? '#212121' : '#fff');
            let bg = rowBackground;
            if (isSelected) bg = theme.select;
    
            const stickyStyle = getStickyStyle(cell.column, bg);
            
            // Add boundary shadows
            if (isLastPinnedLeft) {
                stickyStyle.boxShadow = '2px 0 5px -2px rgba(0,0,0,0.2)';
                stickyStyle.zIndex = (stickyStyle.zIndex || 0) + 1;
            }
            if (isFirstPinnedRight) {
                stickyStyle.boxShadow = '-2px 0 5px -2px rgba(0,0,0,0.2)';
                stickyStyle.zIndex = (stickyStyle.zIndex || 0) + 1;
            }
    
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
                        fontWeight: ((row.original && row.original._isTotal) || (isHierarchy && row.getIsGrouped())) ? 500 : 400,
                        background: bg,
                        ...stickyStyle,
                        ...condStyle,
                        ...(isFillSelected ? {boxShadow: `inset 0 0 0 1px ${theme.primary}`} : {}),
                        userSelect: 'none',
                        position: stickyStyle && stickyStyle.position === 'sticky' ? 'sticky' : 'relative'
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
    const renderHeaderCell = (header, level, isLastPinnedLeft = false, isFirstPinnedRight = false, renderSection = 'center') => {
        const isGroupHeader = header.column.columns && header.column.columns.length > 0;
        const isSorted = header.column.getIsSorted();
        const sortIndex = header.column.getSortIndex();
        const isMultiSort = table.getState().sorting.length > 1;
        const isPinned = header.column.getIsPinned();
        const leafColumns = header.column.getLeafColumns ? header.column.getLeafColumns() : [header.column];
        const sectionLeafIds = new Set(
            (renderSection === 'left' ? leftCols : renderSection === 'right' ? rightCols : centerCols).map(column => column.id)
        );
        const sectionWidth = leafColumns
            .filter(column => sectionLeafIds.has(column.id))
            .reduce((sum, column) => sum + column.getSize(), 0);
        const headerWidth = sectionWidth || header.getSize();

        // Calculate sticky style for pinned headers using the hook
        const stickyStyle = getHeaderStickyStyle(header, level, isLastPinnedLeft, isFirstPinnedRight, renderSection);

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

                {header.column.getCanResize() && <div onMouseDown={header.getResizeHandler()} onTouchStart={header.getResizeHandler()} onDoubleClick={() => autoSizeColumn(header.column.id)} style={{position:'absolute',right:0,top:0,bottom:0,width:4,cursor:'col-resize'}}/>}
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
                    }} onClick={exportExcel}>
                        <Icons.Export/> Export
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
                                             {group.headers.map((header, idx) => renderHeaderCell(header, level, idx === group.headers.length - 1, false, 'left'))}
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
                                 <div>
                                     {table.getCenterHeaderGroups().map((group, level) => (
                                         <div key={group.id} style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`}}>
                                             {group.headers.map(header => renderHeaderCell(header, level, false, false, 'center'))}
                                         </div>
                                     ))}
                                     {showFloatingFilters && (
                                         <div style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`, background: theme.background}}>
                                             {centerCols.map((column) => (
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
                                             ))}
                                         </div>
                                     )}
                                 </div>

                                 {/* Right Section */}
                                 <div style={{position: 'sticky', right: 0, zIndex: 4, background: theme.headerBg}}>
                                     {table.getRightHeaderGroups().map((group, level) => (
                                         <div key={group.id} style={{display: 'flex', height: rowHeight, borderBottom: `1px solid ${theme.border}`}}>
                                             {group.headers.map((header, idx) => renderHeaderCell(header, level, false, idx === 0, 'right'))}
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
                                         background: (row.original && row.original._isTotal) ? '#e8f5e9' : theme.background,
                                         borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                         boxShadow: isLastPinnedTop ? `0 2px 4px -2px ${theme.border}80` : 'none'
                                     }}>
                                         {row.getLeftVisibleCells().map((cell, idx) => renderCell(cell, i, idx === leftCols.length - 1, false, false))}
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {virtualCenterCols.map(virtualCol => {
                                             const cell = row.getCenterVisibleCells()[virtualCol.index];
                                             return renderCell(cell, i, false, false, false);
                                         })}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                         {row.getRightVisibleCells().map((cell, idx) => renderCell(cell, i, false, idx === 0, false))}
                                     </div>
                                 )
                             })}

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
                                             background: theme.background,
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

                                 return (
                                     <div
                                        key={row.id}
                                        role="row"
                                        aria-rowindex={virtualRow.index}
                                        style={{
                                         ...styles.row,
                                         height: virtualRow.size,
                                         top: `${virtualRow.start + topOffset}px`,
                                         width: `${totalLayoutWidth}px`,
                                         background: (row.original && row.original._isTotal) ? '#e8f5e9' : '#fff',
                                         borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                         transition: rowVirtualizer.isScrolling ? 'none' : 'top 0.2s ease-out, background-color 0.2s'
                                     }}>
                                         {row.getLeftVisibleCells().map((cell, idx) => renderCell(cell, virtualRow.index, idx === leftCols.length - 1, false, true))}
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {virtualCenterCols.map(virtualCol => {
                                             const cell = row.getCenterVisibleCells()[virtualCol.index];
                                             return renderCell(cell, virtualRow.index, false, false, true);
                                         })}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                         {row.getRightVisibleCells().map((cell, idx) => renderCell(cell, virtualRow.index, false, idx === 0, true))}
                                     </div>
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
                                         background: (row.original && row.original._isTotal) ? '#e8f5e9' : theme.background,
                                         borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                         boxShadow: isFirstPinnedBottom ? `0 -2px 4px -2px ${theme.border}80` : 'none'
                                     }}>
                                         {row.getLeftVisibleCells().map((cell, idx) => renderCell(cell, i, idx === leftCols.length - 1, false, false))}
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {virtualCenterCols.map(virtualCol => {
                                             const cell = row.getCenterVisibleCells()[virtualCol.index];
                                             return renderCell(cell, i, false, false, false);
                                         })}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                         {row.getRightVisibleCells().map((cell, idx) => renderCell(cell, i, false, idx === 0, false))}
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
        </div>
    );
};

DashTanstackPivot.propTypes = {
    id: PropTypes.string,
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
    dataVersion: PropTypes.number
};
