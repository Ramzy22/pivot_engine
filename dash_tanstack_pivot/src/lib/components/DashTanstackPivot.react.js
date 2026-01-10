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

// --- Shared Logic ---
const getKey = (prefix, field, agg) => prefix ? `${prefix}_${field}_${agg}` : `${field}_${agg}`;

const formatValue = (value, fmt) => {
    if (value === null || value === undefined) return '';
    if (typeof value !== 'number') return value;
    if (!fmt) return value.toLocaleString();
    
    try {
        if (fmt === 'currency') return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
        if (fmt === 'accounting') return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', currencySign: 'accounting' }).format(value);
        if (fmt === 'percent') return new Intl.NumberFormat('en-US', { style: 'percent', maximumFractionDigits: 2 }).format(value);
        if (fmt === 'scientific') return value.toExponential(2);
        if (fmt.startsWith('fixed')) {
            const parts = fmt.split(':');
            const decimals = parts.length > 1 ? parseInt(parts[1]) : 2;
            return value.toFixed(decimals);
        }
    } catch (e) {
        console.warn('Format error', e);
    }
    return value.toLocaleString();
};

const Sparkline = ({ data = [], width = 100, height = 30, color = '#1976d2' }) => {
    if (!data || data.length < 2) return null;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const padding = 2;
    const innerWidth = width - padding * 2;
    const innerHeight = height - padding * 2;

    // Calculate points with padding
    const points = data.map((d, i) => ({
        x: padding + (i / (data.length - 1)) * innerWidth,
        y: padding + innerHeight - ((d - min) / range) * innerHeight
    }));

    return (
        <svg width={width} height={height} style={{ overflow: 'hidden' }}>
            <path
                d={`M ${points.map(p => `${p.x},${p.y}`).join(' L ')}`}
                fill="none"
                stroke={color}
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
};

// --- Icons ---
const Icons = {
    SortAsc: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8z"/></svg>,
    SortDesc: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M20 12l-1.41-1.41L13 16.17V4h-2v12.17l-5.58-5.59L4 12l8 8 8-8z"/></svg>,
    Export: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>,
    Search: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>,
    ChevronRight: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg>,
    ChevronDown: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M16.59 8.59L12 13.17 7.41 8.59 6 10l6 6 6-6z"/></svg>,
    DragIndicator: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="#bdbdbd"><path d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>,
    Close: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>,
    Spacing: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/></svg>,
    ColExpand: () => <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>,
    ColCollapse: () => <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M19 13H5v-2h14v2z"/></svg>,
    Filter: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/></svg>,
    Menu: () => <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>,
    MoreVert: () => <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>,
    PinLeft: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M13 12H7v1.5L4.5 11 7 8.5V10h6v2zm6.41-7.12l-1.42-1.41-4.83 4.83c-.37-.13-.77-.21-1.19-.21-1.91 0-3.47 1.55-3.47 3.47 0 1.92 1.56 3.47 3.47 3.47 1.92 0 3.47-1.55 3.47-3.47 0-.42-.08-.82-.21-1.19l4.83-4.83-1.42-1.41z"/></svg>,
    PinRight: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M11 12h6v-1.5l2.5 2.5-2.5 2.5V14h-6v-2zm-6.41 7.12l1.42 1.41 4.83-4.83c.37.13.77.21 1.19.21 1.91 0 3.47-1.55 3.47-3.47 0-1.92-1.56-3.47-3.47-3.47-1.92 0-3.47 1.55-3.47 3.47 0 .42.08.82.21 1.19l-4.83 4.83 1.42 1.41z"/></svg>,
    Unpin: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M16 9V4l1 1V3H7v1l1-1v5c0 1.66-1.34 3-3 3v2h5.97v7l1 1 1-1v-7H19v-2c-1.66 0-3-1.34-3-3z"/></svg>,
    Visibility: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>,
    VisibilityOff: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.82l2.92 2.92c1.51-1.39 2.72-3.13 3.44-5.04-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2zm4.31-.78l3.15 3.15.02-.17c0-1.66-1.34-3-3-3l-.17.02z"/></svg>,
    Group: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5s-3 1.34-3 3 1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5 5 1.34 5 3 6.34 3 8 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>,
    Lock: () => <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/></svg>
};

// --- Sort Helpers ---
const alphanumeric = (rowA, rowB, columnId) => {
    const a = rowA.getValue(columnId);
    const b = rowB.getValue(columnId);
    // Use localeCompare for natural alphanumeric sort
    // sensitivity: 'base' ignores case (default behavior often desired)
    // We can make this configurable later via sortOptions
    return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
};

const DateRangeFilter = ({ onFilter, currentFilter, theme }) => {
    const styles = getStyles(theme || { primary: '#1976d2', border: '#e0e0e0', headerBg: '#f5f5f5', text: '#212121' });
    
    const isMulti = currentFilter && currentFilter.conditions;
    const [operator, setOperator] = useState(isMulti ? currentFilter.operator : 'AND');
    const [conditions, setConditions] = useState(() => {
        if (isMulti) return currentFilter.conditions;
        if (currentFilter && currentFilter.value && !currentFilter.conditions) {
             // Handle legacy single value or specialized filter structure
             return [{ type: 'eq', value: currentFilter.value }];
        }
        return [{ type: 'between', value: '', value2: '' }];
    });

    const updateCondition = (index, key, value) => {
        const newConditions = [...conditions];
        newConditions[index][key] = value;
        setConditions(newConditions);
    };

    const addCondition = () => {
        setConditions([...conditions, { type: 'between', value: '', value2: '' }]);
    };

    const removeCondition = (index) => {
        const newConditions = conditions.filter((_, i) => i !== index);
        setConditions(newConditions);
    };

    const apply = () => {
        const validConditions = conditions.filter(c => {
            if (c.type === 'between') return c.value && c.value2;
            return c.value;
        });
        
        if (validConditions.length > 0) {
            onFilter({ operator, conditions: validConditions });
        } else {
            onFilter(null);
        }
    };

    const setPreset = (days) => {
        const e = new Date();
        const s = new Date();
        s.setDate(e.getDate() - days);
        const fmt = d => d.toISOString().split('T')[0];
        // For presets, we reset to a single condition
        setConditions([{ type: 'between', value: fmt(s), value2: fmt(e) }]);
    };

    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
            <div style={{display: 'flex', justifyContent: 'flex-end', gap: '4px'}}>
                <button onClick={() => setOperator('AND')} style={{...styles.btn, padding: '2px 6px', fontSize: '10px', background: operator === 'AND' ? theme.primary : '#eee', color: operator === 'AND' ? '#fff' : '#333'}}>AND</button>
                <button onClick={() => setOperator('OR')} style={{...styles.btn, padding: '2px 6px', fontSize: '10px', background: operator === 'OR' ? theme.primary : '#eee', color: operator === 'OR' ? '#fff' : '#333'}}>OR</button>
            </div>

            <div style={{display: 'flex', gap: '6px', flexWrap: 'wrap'}}>
                <button onClick={() => setPreset(0)} style={{...styles.btn, fontSize:'11px', padding:'4px 8px'}}>Today</button>
                <button onClick={() => setPreset(7)} style={{...styles.btn, fontSize:'11px', padding:'4px 8px'}}>Last 7d</button>
                <button onClick={() => setPreset(30)} style={{...styles.btn, fontSize:'11px', padding:'4px 8px'}}>Last 30d</button>
            </div>

            <div style={{display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px', overflowY: 'auto'}}>
                {conditions.map((cond, index) => (
                    <div key={index} style={{display: 'flex', flexDirection: 'column', gap: '4px', border: '1px solid #f0f0f0', padding: '8px', borderRadius: '4px'}}>
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                            <select value={cond.type} onChange={e => updateCondition(index, 'type', e.target.value)} style={{padding:'2px', borderRadius:'2px', border:'1px solid #ddd', fontSize: '11px', width: '100px'}}>
                                <option value="between">Between</option>
                                <option value="eq">Equals</option>
                                <option value="gt">After</option>
                                <option value="lt">Before</option>
                            </select>
                            {conditions.length > 1 && <span onClick={() => removeCondition(index)} style={{cursor: 'pointer'}}><Icons.Close/></span>}
                        </div>
                        <div style={{display: 'flex', gap: '4px', alignItems: 'center'}}>
                            <input type="date" value={cond.value} onChange={e => updateCondition(index, 'value', e.target.value)} style={{border:'1px solid #ddd', borderRadius:'4px', padding:'4px', flex: 1, minWidth: 0, fontSize: '12px'}} />
                            {cond.type === 'between' && (
                                <>
                                    <span>-</span>
                                    <input type="date" value={cond.value2 || ''} onChange={e => updateCondition(index, 'value2', e.target.value)} style={{border:'1px solid #ddd', borderRadius:'4px', padding:'4px', flex: 1, minWidth: 0, fontSize: '12px'}} />
                                </>
                            )}
                        </div>
                    </div>
                ))}
            </div>
            
            <div style={{display: 'flex', gap: '8px'}}>
                <button onClick={addCondition} style={{...styles.btn, flex: 1, justifyContent: 'center', background: '#f5f5f5', fontSize: '11px'}}>Add</button>
                <button onClick={apply} style={{...styles.btn, flex: 1, justifyContent: 'center', background: theme.primary, color: '#fff', fontSize: '11px'}}>Apply</button>
            </div>
        </div>
    );
};

const NumericRangeFilter = ({ onFilter, currentFilter, theme }) => {
    const [min, setMin] = useState((currentFilter && currentFilter.conditions && currentFilter.conditions[0]) ? currentFilter.conditions[0].value : '');
    const [max, setMax] = useState((currentFilter && currentFilter.conditions && currentFilter.conditions[0]) ? currentFilter.conditions[0].value2 : '');

    const apply = (mn, mx) => {
        if (mn !== '' && mx !== '') {
            onFilter({ operator: 'AND', conditions: [{ type: 'between', value: Number(mn), value2: Number(mx) }] });
        } else if (mn !== '') {
            onFilter({ operator: 'AND', conditions: [{ type: 'gte', value: Number(mn) }] });
        } else if (mx !== '') {
            onFilter({ operator: 'AND', conditions: [{ type: 'lte', value: Number(mx) }] });
        } else {
            onFilter(null);
        }
    };

    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: '8px'}}>
             <div style={{display: 'flex', gap: '8px', alignItems: 'center'}}>
                <input type="number" placeholder="Min" value={min} onChange={e => { setMin(e.target.value); apply(e.target.value, max); }} style={{border:'1px solid #ddd', borderRadius:'4px', padding:'4px', width: '80px'}} />
                <div style={{flex:1, height:'2px', background:'#eee', position:'relative'}}>
                     <div style={{position:'absolute', left:'0', right:'0', top:'-1px', height:'4px', background: theme.primary, opacity: 0.3}} />
                </div>
                <input type="number" placeholder="Max" value={max} onChange={e => { setMax(e.target.value); apply(min, e.target.value); }} style={{border:'1px solid #ddd', borderRadius:'4px', padding:'4px', width: '80px'}} />
            </div>
        </div>
    );
};

const MultiSelectFilter = ({ options = [], onFilter, currentFilter }) => {
    const [search, setSearch] = useState('');
    const [selected, setSelected] = useState(new Set());

    useEffect(() => {
        // Initialize from current filter if it's an 'in' type
        if (currentFilter && currentFilter.conditions && currentFilter.conditions[0] && currentFilter.conditions[0].type === 'in') {
            setSelected(new Set(currentFilter.conditions[0].value));
        }
    }, [currentFilter]);

    const filteredOptions = options.filter(o => String(o).toLowerCase().includes(search.toLowerCase()));

    const toggle = (val) => {
        const newSet = new Set(selected);
        if (newSet.has(val)) newSet.delete(val);
        else newSet.add(val);
        setSelected(newSet);
        
        if (newSet.size > 0) {
            onFilter({ operator: 'AND', conditions: [{ type: 'in', value: Array.from(newSet) }] });
        } else {
            onFilter(null);
        }
    };

    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: '8px', maxHeight: '200px'}}>
            <input 
                placeholder="Search..." 
                value={search} 
                onChange={e => setSearch(e.target.value)} 
                style={{border:'1px solid #ddd', borderRadius:'4px', padding:'4px', fontSize:'11px'}} 
            />
            <div style={{overflowY: 'auto', flex: 1, border: '1px solid #f0f0f0', borderRadius:'4px'}}>
                {filteredOptions.length === 0 ? <div style={{padding:'8px', color:'#999', fontSize:'11px'}}>No options...</div> :
                filteredOptions.map((opt, i) => (
                    <div key={i} onClick={() => toggle(opt)} style={{display:'flex', gap:'6px', padding:'4px 8px', cursor:'pointer', alignItems:'center', background: selected.has(opt) ? '#e3f2fd' : 'transparent'}}>
                        <input type="checkbox" checked={selected.has(opt)} readOnly style={{margin:0}} />
                        <span style={{fontSize:'11px'}}>{opt}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

// --- Helpers ---
const isGroupColumn = (column) => {
    return column.columns && column.columns.length > 0;
};

const hasChildrenInZone = (col, zone) => {
    const pin = col.getIsPinned();
    if (!col.columns || col.columns.length === 0) {
        return pin === zone || (zone === 'unpinned' && !pin);
    }
    return col.columns.some(child => hasChildrenInZone(child, zone));
};

const getAllLeafColumns = (col) => {
    if (!col.columns || col.columns.length === 0) return [col];
    return col.columns.flatMap(getAllLeafColumns);
};

const getAllLeafIdsFromColumn = (column) => {
    return getAllLeafColumns(column).map(c => c.id);
};

const ColumnFilter = ({ column, onFilter, currentFilter, options = [], theme, onClose }) => {
    const styles = getStyles(theme || { primary: '#1976d2', border: '#e0e0e0', headerBg: '#f5f5f5', text: '#212121' });
    const isLeaf = column.getLeafColumns
        ? column.getLeafColumns().length === 1
        : !column.columns;

    if (!isLeaf) {
        return (
            <div style={{ padding: '12px', fontSize: '12px', background: theme.background, color: theme.text }}>
                Filters are available only for value columns.
            </div>
        );
    }

    const leaf = column.getLeafColumns ? column.getLeafColumns()[0] : column;
    const colId = leaf.id.toLowerCase();
    const isDate = (leaf.columnDef && leaf.columnDef.meta && leaf.columnDef.meta.type === 'date') || colId.includes('date') || colId.includes('time');
    const isNumeric = (leaf.columnDef && leaf.columnDef.meta && leaf.columnDef.meta.type === 'number') || colId.includes('sales') || colId.includes('cost') || colId.includes('amount') || colId.includes('price');

    const [tab, setTab] = useState('condition'); // condition, values, smart

    // Auto-select tab based on available options
    useEffect(() => {
        if (options && options.length > 0 && tab === 'condition') {
            setTab('values');
        } else if (isDate && tab === 'condition') {
             setTab('date');
        }
    }, [options, isDate]);

    // --- Existing Condition Logic ---
    const isMulti = currentFilter && currentFilter.conditions;
    const [operator, setOperator] = useState(isMulti ? currentFilter.operator : 'AND');
    const [conditions, setConditions] = useState(
        isMulti ? currentFilter.conditions : [{type: 'contains', value: '', caseSensitive: false}]
    );

    const updateCondition = (index, key, value) => {
        const newConditions = [...conditions];
        newConditions[index][key] = value;
        setConditions(newConditions);
    };
    
    const addCondition = () => {
        setConditions([...conditions, {type: 'contains', value: '', caseSensitive: false}]);
    };

    const removeCondition = (index) => {
        const newConditions = conditions.filter((_, i) => i !== index);
        setConditions(newConditions);
    };

    const handleApply = () => {
        const validConditions = conditions.filter(c => {
             if (c.type === 'between') return c.value && c.value2;
             return String(c.value).trim() !== '';
        });
        
        const newFilter = {
            operator: operator,
            conditions: validConditions.map(c => {
                 let finalVal = c.value;
                 let finalVal2 = c.value2;
                 
                 if (isNumeric) {
                     if (finalVal !== '' && !isNaN(Number(finalVal))) finalVal = Number(finalVal);
                     if (finalVal2 !== '' && !isNaN(Number(finalVal2))) finalVal2 = Number(finalVal2);
                 }

                 if (c.type === 'between') {
                      return { ...c, value: [finalVal, finalVal2] };
                 }
                 return { ...c, value: finalVal };
            })
        };
        
        if (newFilter.conditions.length > 0) {
            onFilter(newFilter);
        } else {
            onFilter(null);
        }
        if (onClose) onClose();
    };

    return (
        <div style={{display: 'flex', flexDirection: 'column', gap: '8px', color: '#333'}}>
            <div style={{fontWeight: 600, fontSize: '12px', borderBottom: '1px solid #eee', paddingBottom: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                <span>Filter: {column.header}</span>
                <div style={{display: 'flex', background: '#f5f5f5', borderRadius: '4px', padding: '2px'}}>
                    <div onClick={() => setTab('condition')} style={{padding:'2px 8px', fontSize:'10px', cursor:'pointer', borderRadius:'3px', background: tab==='condition'?'#fff':'transparent', boxShadow: tab==='condition'?'0 1px 2px rgba(0,0,0,0.1)':'none'}}>Rules</div>
                    {options.length > 0 && <div onClick={() => setTab('values')} style={{padding:'2px 8px', fontSize:'10px', cursor:'pointer', borderRadius:'3px', background: tab==='values'?'#fff':'transparent', boxShadow: tab==='values'?'0 1px 2px rgba(0,0,0,0.1)':'none'}}>List</div>}
                    {isDate && <div onClick={() => setTab('date')} style={{padding:'2px 8px', fontSize:'10px', cursor:'pointer', borderRadius:'3px', background: tab==='date'?'#fff':'transparent', boxShadow: tab==='date'?'0 1px 2px rgba(0,0,0,0.1)':'none'}}>Date</div>}
                    {isNumeric && <div onClick={() => setTab('numeric')} style={{padding:'2px 8px', fontSize:'10px', cursor:'pointer', borderRadius:'3px', background: tab==='numeric'?'#fff':'transparent', boxShadow: tab==='numeric'?'0 1px 2px rgba(0,0,0,0.1)':'none'}}>Range</div>}
                </div>
            </div>
            
            <div style={{maxHeight: '300px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px'}}>
                {tab === 'values' && (
                    <MultiSelectFilter options={options} onFilter={onFilter} currentFilter={currentFilter} />
                )}

                {tab === 'date' && (
                    <DateRangeFilter onFilter={onFilter} currentFilter={currentFilter} theme={theme} />
                )}

                {tab === 'numeric' && (
                    <NumericRangeFilter onFilter={onFilter} currentFilter={currentFilter} theme={theme} />
                )}

                {tab === 'condition' && (
                    <>
                        <div style={{display: 'flex', justifyContent: 'flex-end', gap: '4px', marginBottom: '4px'}}>
                            <button onClick={() => setOperator('AND')} style={{padding: '2px 6px', fontSize: '10px', background: operator === 'AND' ? theme.primary: '#eee', color: operator === 'AND' ? '#fff' : '#333', border: 'none', borderRadius: '2px'}}>AND</button>
                            <button onClick={() => setOperator('OR')} style={{padding: '2px 6px', fontSize: '10px', background: operator === 'OR' ? theme.primary: '#eee', color: operator === 'OR' ? '#fff' : '#333', border: 'none', borderRadius: '2px'}}>OR</button>
                        </div>
                        <div style={{display: 'flex', flexDirection: 'column', gap: '12px', paddingRight: '8px'}}>
                        {conditions.map((cond, index) => (
                            <div key={index} style={{display: 'flex', flexDirection: 'column', gap: '4px', border: '1px solid #f0f0f0', padding: '8px', borderRadius: '4px'}}>
                                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                    <label style={{fontSize:'11px', color: '#666'}}>Condition {index + 1}</label>
                                    {conditions.length > 1 && <span onClick={() => removeCondition(index)} style={{cursor: 'pointer'}}><Icons.Close/></span>}
                                </div>
                                <select value={cond.type} onChange={e => updateCondition(index, 'type', e.target.value)} style={{padding:'4px', borderRadius:'2px', border:'1px solid #ddd'}}>
                                    <option value="eq">Equals</option>
                                    <option value="ne">Not Equals</option>
                                    <option value="contains">Contains</option>
                                    <option value="startsWith">Starts With</option>
                                    <option value="endsWith">Ends With</option>
                                    <option value="gt">Greater Than</option>
                                    <option value="lt">Less Than</option>
                                    <option value="between">Between (Range)</option>
                                    <option value="in">In List</option>
                                </select>
                                
                                {cond.type === 'between' ? (
                                    <div style={{display: 'flex', gap: '4px'}}>
                                        <input 
                                            placeholder="Start" 
                                            value={cond.value} 
                                            onChange={e => updateCondition(index, 'value', e.target.value)} 
                                            style={{padding:'6px', borderRadius:'2px', border:'1px solid #ddd', fontSize: '13px', width: '50%'}}
                                        />
                                        <input 
                                            placeholder="End" 
                                            value={cond.value2 || ''} 
                                            onChange={e => updateCondition(index, 'value2', e.target.value)} 
                                            style={{padding:'6px', borderRadius:'2px', border:'1px solid #ddd', fontSize: '13px', width: '50%'}}
                                        />
                                    </div>
                                ) : (
                                    <input 
                                        placeholder="Value..." 
                                        value={cond.value} 
                                        onChange={e => updateCondition(index, 'value', e.target.value)} 
                                        style={{padding:'6px', borderRadius:'2px', border:'1px solid #ddd', fontSize: '13px'}}
                                    />
                                )}
                                
                                <div style={{display:'flex', alignItems:'center', gap:'4px', marginTop:'2px'}}>
                                    <input 
                                        type="checkbox" 
                                        checked={cond.caseSensitive || false} 
                                        onChange={e => updateCondition(index, 'caseSensitive', e.target.checked)}
                                        id={`cs-${index}`}
                                    />
                                    <label htmlFor={`cs-${index}`} style={{fontSize:'11px', color:'#555', cursor:'pointer'}}>Match Case</label>
                                </div>
                            </div>
                        ))}
                        </div>
                        <button onClick={addCondition} style={{...styles.btn, justifyContent: 'center', background: '#f5f5f5'}}>Add Condition</button>
                    </>
                )}
            </div>

            <div style={{display:'flex', justifyContent: 'space-between', gap: '8px', marginTop: '8px', borderTop: '1px solid #eee', paddingTop: '8px'}}>
                <button onClick={() => { onFilter(null); if(onClose) onClose(); }} style={{padding: '4px 8px', border:'none', background:'none', cursor:'pointer', color: '#d32f2f', fontSize: '11px'}}>Clear & Close</button>
                <div style={{display: 'flex', gap: '8px'}}>
                    {onClose && <button onClick={onClose} style={{padding: '4px 8px', border:'none', background:'none', cursor:'pointer', fontSize: '11px'}}>Close</button>}
                    {tab === 'condition' && (
                        <button onClick={handleApply} style={{padding: '4px 12px', background: theme.primary, color: '#fff', border:'none', borderRadius: '2px', cursor:'pointer', fontSize: '11px'}}>
                            Apply
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

const FilterPopover = ({ column, anchorEl, onClose, onFilter, currentFilter, options = [], theme }) => {
    const [position, setPosition] = useState({ top: 0, left: 0 });
    const popoverRef = useRef(null);

    useEffect(() => {
        // Calculate position to avoid viewport overflow
        const target = anchorEl || (column instanceof Element ? column : null);
        if (!target) return;

        const rect = target.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        const viewportWidth = window.innerWidth;

        let top = rect.bottom;
        let left = rect.left;

        // Check if would overflow bottom
        if (top + 400 > viewportHeight) {
            top = rect.top - 400; // Position above
        }

        // Check if would overflow right
        if (left + 300 > viewportWidth) {
            left = viewportWidth - 320; // Adjust to fit
        }

        setPosition({ top, left });
    }, [anchorEl, column]);

    return (
        <div ref={popoverRef}
            style={{
                position: 'fixed', // Changed from absolute
                top: `${position.top}px`,
                left: `${position.left}px`,
                background: '#fff',
                border: '1px solid #ccc',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                zIndex: 1000,
                padding: '12px',
                borderRadius: '4px',
                width: '300px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                color: '#333'
            }} onClick={e => e.stopPropagation()}>
            <ColumnFilter 
                column={column} 
                onFilter={onFilter} 
                currentFilter={currentFilter} 
                options={options} 
                theme={theme} 
                onClose={onClose} 
            />
        </div>
    );
};

const themes = {
    light: {
        name: 'light',
        primary: '#1976d2',
        border: '#e0e0e0',
        headerBg: '#f5f5f5',
        text: '#212121',
        textSec: '#757575',
        hover: '#eeeeee',
        select: '#e3f2fd',
        background: '#fff',
        sidebarBg: '#fafafa'
    },
    dark: {
        name: 'dark',
        primary: '#90caf9',
        border: '#424242',
        headerBg: '#333',
        text: '#fff',
        textSec: '#b0b0b0',
        hover: '#424242',
        select: '#1e3a5f',
        background: '#212121',
        sidebarBg: '#2c2c2c'
    },
    material: {
        name: 'material',
        primary: '#6200ee',
        border: '#e0e0e0',
        headerBg: '#fff',
        text: '#000',
        textSec: '#666',
        hover: '#f5f5f5',
        select: '#e8eaf6',
        background: '#fff',
        sidebarBg: '#fafafa'
    },
    balham: {
        name: 'balham',
        primary: '#0091ea',
        border: '#BDC3C7',
        headerBg: '#F5F7F7',
        text: '#2c3e50',
        textSec: '#7f8c8d',
        hover: '#ecf0f1',
        select: '#d6eaf8',
        background: '#fff',
        sidebarBg: '#fafafa'
    }
};

const getStyles = (theme) => ({
    root: {
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'Roboto, Helvetica, Arial, sans-serif',
        height: '100%',
        background: theme.background,
        border: `1px solid ${theme.border}`,
        borderRadius: '4px',
        overflow: 'hidden',
        fontSize: '13px',
        color: theme.text
    },
    appBar: {
        height: '48px',
        borderBottom: `1px solid ${theme.border}`,
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        justifyContent: 'space-between',
        background: theme.headerBg,
        color: theme.text
    },
    searchBox: {
        display: 'flex',
        alignItems: 'center',
        background: theme.text === '#fff' ? '#424242' : '#f5f5f5',
        borderRadius: '4px',
        padding: '4px 8px',
        width: '200px'
    },
    sidebar: {
        width: '320px',
        minWidth: '320px',
        borderRight: `1px solid ${theme.border}`,
        background: theme.sidebarBg,
        display: 'flex',
        flexDirection: 'column',
        padding: '16px',
        gap: '16px',
        overflowY: 'auto'
    },
    sectionTitle: {
        fontSize: '11px',
        fontWeight: 700,
        textTransform: 'uppercase',
        color: theme.textSec,
        marginBottom: '8px'
    },
    chip: {
        background: theme.text === '#fff' ? '#424242' : '#fff',
        border: `1px solid ${theme.border}`,
        borderRadius: '4px',
        padding: '6px 8px',
        marginBottom: '6px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        cursor: 'grab',
        boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
        position: 'relative',
        color: theme.text
    },
    dropZone: {
        minHeight: '40px',
        border: `1px dashed ${theme.border}`,
        borderRadius: '4px',
        padding: '8px',
        background: 'rgba(0,0,0,0.02)'
    },
    main: {
        flex: 1,
        overflow: 'hidden',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column'
    },
    scrollContainer: {
        flex: 1,
        overflow: 'auto',
        position: 'relative'
    },
    headerSticky: {
        position: 'sticky',
        top: 0,
        zIndex: 2,
        background: theme.headerBg,
        width: 'fit-content',
        minWidth: '100%'
    },
    headerRow: {
        display: 'flex',
        width: '100%'
    },
    headerCell: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 8px',
        borderRight: `1px solid ${theme.border}`,
        borderBottom: `1px solid ${theme.border}`,
        fontWeight: 600,
        color: theme.text,
        position: 'relative',
        boxSizing: 'border-box',
        flexShrink: 0,
        minWidth: 0,
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        textOverflow: 'ellipsis'
    },
    pinned: {
        position: 'sticky',
        zIndex: 3,
        background: theme.background
    },
    pinnedLeft: {
        left: 0,
        borderRight: `1px solid ${theme.border}`
    },
    pinnedRight: {
        right: 0,
        borderLeft: `1px solid ${theme.border}`
    },
    row: {
        display: 'flex',
        position: 'absolute',
        left: 0,
        width: '100%',
        boxSizing: 'border-box'
    },
    cell: {
        display: 'flex',
        alignItems: 'center',
        padding: '0 8px',
        borderRight: `1px solid ${theme.border}`,
        borderBottom: `1px solid ${theme.border}`,
        background: theme.background,
        color: theme.text,
        overflow: 'hidden',
        whiteSpace: 'nowrap',
        boxSizing: 'border-box',
        flexShrink: 0
    },
    cellSelected: {
        background: `${theme.select} !important`,
        boxShadow: `inset 0 0 0 2px ${theme.primary}`,
        zIndex: 2,
        position: 'relative'
    },
    btn: {
        padding: '6px 12px',
        borderRadius: '4px',
        border: `1px solid ${theme.border}`,
        background: 'transparent',
        cursor: 'pointer',
        fontSize: '12px',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        fontWeight: 500,
        color: theme.text
    },
    dropLine: {
        position: 'absolute',
        height: '2px',
        background: theme.primary,
        left: 0, right: 0, zIndex: 10, pointerEvents: 'none'
    },
    expandedSeparator: {
        borderBottom: `2px solid ${theme.primary}`
    },
    toolPanelSection: {
        display: 'flex',
        flexDirection: 'column',
        borderBottom: `1px solid ${theme.border}44`,
        marginBottom: '4px',
        transition: 'all 0.3s ease-in-out'
    },
    toolPanelSectionHeader: {
        display: 'flex',
        alignItems: 'center',
        padding: '8px 12px',
        background: theme.headerBg,
        cursor: 'pointer',
        fontSize: '11px',
        fontWeight: 600,
        textTransform: 'uppercase',
        color: theme.textSec,
        userSelect: 'none',
        transition: 'background 0.2s'
    },
    toolPanelList: {
        display: 'flex',
        flexDirection: 'column',
        padding: '4px 0',
        transition: 'height 0.3s ease'
    },
    columnItem: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        fontSize: '12px',
        cursor: 'default',
        transition: 'background 0.2s, transform 0.2s',
        position: 'relative',
        userSelect: 'none'
    }
});

const ContextMenu = ({ x, y, onClose, actions }) => {
    const [adjustedPosition, setAdjustedPosition] = useState({ x, y });

    useEffect(() => {
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const menuWidth = 200;
        const menuHeight = actions.length * 32 + 20;

        let adjustedX = x;
        let adjustedY = y;

        if (x + menuWidth > viewportWidth) {
            adjustedX = viewportWidth - menuWidth - 10;
        }
        if (y + menuHeight > viewportHeight) {
            adjustedY = viewportHeight - menuHeight - 10;
        }

        setAdjustedPosition({ x: adjustedX, y: adjustedY });
    }, [x, y, actions.length]);

    return (
        <div style={{
            position: 'fixed',
            top: adjustedPosition.y,
            left: adjustedPosition.x,
            background: '#fff',
            border: '1px solid #ccc',
            boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
            zIndex: 10001, // Above notifications
            padding: '6px 0',
            borderRadius: '6px',
            fontSize: '13px',
            minWidth: '180px',
            maxWidth: '300px'
        }}>
            {actions.map((action, i) => {
                if (action === 'separator') {
                    return <div key={i} style={{height: '1px', background: '#e0e0e0', margin: '4px 0'}} />;
                }
                return (
                    <div key={i} onClick={() => { action.onClick(); onClose(); }} style={{
                        padding: '8px 16px', cursor: 'pointer', backgroundColor: '#fff', display: 'flex', alignItems: 'center', gap: '8px'
                    }} onMouseEnter={e => e.currentTarget.style.backgroundColor = '#f5f5f5'} onMouseLeave={e => e.currentTarget.style.backgroundColor = '#fff'}>
                        {action.icon && <span style={{color: '#757575'}}>{action.icon}</span>}
                        {action.label}
                    </div>
                );
            })}
        </div>
    );
};

const StatusBar = ({ selectedCells, rowCount, visibleRowsCount, theme }) => {
    const stats = useMemo(() => {
        const values = Object.values(selectedCells).map(v => parseFloat(v)).filter(v => !isNaN(v));
        const count = Object.keys(selectedCells).length;
        if (values.length === 0) return { count };
        
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        // Advanced stats
        const sqDiffs = values.map(v => Math.pow(v - avg, 2));
        const variance = sqDiffs.reduce((a, b) => a + b, 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        return { count, sum, avg, min, max, variance, stdDev };
    }, [selectedCells]);

    return (
        <div style={{ height: '32px', borderTop: `1px solid ${theme.border}`, background: theme.headerBg, display: 'flex', alignItems: 'center', padding: '0 16px', justifyContent: 'space-between', fontSize: '12px', color: theme.textSec }}>
            <div>
                {rowCount ? `Total: ${rowCount.toLocaleString()}` : 'Loading...'} 
                {visibleRowsCount && ` | Visible: ${visibleRowsCount}`}
            </div>
            <div style={{display: 'flex', gap: '16px', overflowX: 'auto'}}>
                <span>Count: {stats.count}</span>
                {stats.sum !== undefined && (
                    <>
                        <span>Sum: {stats.sum.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        <span>Avg: {stats.avg.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        <span>Min: {stats.min.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        <span>Max: {stats.max.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        <span>Var: {stats.variance.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        <span>StdDev: {stats.stdDev.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                    </>
                )}
            </div>
        </div>
    );
};

// --- Editable Cell Component ---
const EditableCell = ({ 
    getValue, 
    row, 
    column, 
    format, 
    validationRules,
    setProps,
    handleContextMenu
}) => {
    const initialValue = getValue();
    const [value, setValue] = useState(initialValue);
    const [isEditing, setIsEditing] = useState(false);
    const [error, setError] = useState(null);
    const inputRef = useRef(null);

    useEffect(() => {
        setValue(initialValue);
    }, [initialValue]);

    useEffect(() => {
        if (isEditing && inputRef.current) {
            inputRef.current.focus();
            inputRef.current.select();
        }
    }, [isEditing]);

    const validate = (val) => {
        if (!validationRules || !validationRules[column.id]) return true;
        const rules = validationRules[column.id];
        for (const rule of rules) {
            if (rule.type === 'required' && (val === null || val === '')) return false;
            if (rule.type === 'numeric' && isNaN(Number(val))) return false;
            if (rule.type === 'min' && Number(val) < rule.value) return false;
            if (rule.type === 'max' && Number(val) > rule.value) return false;
            if (rule.type === 'regex' && !new RegExp(rule.pattern).test(val)) return false;
        }
        return true;
    };

    const onBlur = () => {
        setIsEditing(false);
        setError(null);
        
        // Basic type conversion for numeric fields
        let submitValue = value;
        // Check if the column is generally numeric (based on format or current value)
        const isNumeric = typeof initialValue === 'number' || (format && (format.startsWith('fixed') || format === 'currency' || format === 'percent'));
        
        if (isNumeric && value !== '') {
             submitValue = Number(value);
        }

        if (String(submitValue) !== String(initialValue)) {
            if (validate(submitValue)) {
                 if (setProps) {
                    setProps({
                        cellUpdate: {
                            rowId: row.id,
                            colId: column.id,
                            value: submitValue,
                            oldValue: initialValue,
                            timestamp: Date.now()
                        }
                    });
                }
            } else {
                setError("Invalid value");
                // Optional: keep editing or revert. For now, revert after short delay or keep visual error
                console.warn("Validation failed for", submitValue);
                setValue(initialValue); 
            }
        }
    };

    return isEditing ? (
        <input 
            ref={inputRef}
            value={value} 
            onChange={e => setValue(e.target.value)} 
            onBlur={onBlur}
            onKeyDown={e => {
                if(e.key === 'Enter') {
                    e.preventDefault();
                    onBlur();
                }
                if(e.key === 'Escape') {
                    setIsEditing(false);
                    setValue(initialValue);
                }
                if(e.key === 'Tab') {
                     // Tab handling is complex in React Table without custom logic, relying on default behavior for now (blur)
                }
            }}
            style={{
                width: '100%', 
                height: '100%', 
                border: error ? '2px solid red' : '2px solid #2196f3',
                borderRadius: '0',
                padding: '0 4px',
                margin: 0,
                outline: 'none',
                fontFamily: 'inherit',
                fontSize: 'inherit',
                textAlign: 'right' 
            }}
        />
    ) : (
        <div 
            onDoubleClick={() => setIsEditing(true)}
            onContextMenu={e => handleContextMenu(e, initialValue, column.id, row)}
            style={{
                width: '100%', 
                height: '100%', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'flex-end', 
                paddingRight: '8px',
                cursor: 'cell',
                border: error ? '1px solid red' : '1px solid transparent'
            }}
            title={error || undefined}
        >
            {formatValue(initialValue, format)}
        </div>
    );
};

// --- Column Tool Panel Tree Item ---
const ColumnTreeItem = ({ column, level, theme, styles, handlePinColumn, colSearch, selectedCols, setSelectedCols, onDrop, sectionId }) => {
    const [expanded, setExpanded] = useState(level < 1); // Only expand root level by default
    const [isHovered, setIsHovered] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    
    const isGroup = column.columns && column.columns.length > 0;
    
    const header = column.columnDef.header;
    // Extract clean label, removing group prefix
    let label = typeof header === 'string' ? header : column.id;
    if (typeof label === 'string' && label.startsWith('group_')) {
        label = column.id.replace('group_', '').split('|||').pop() || label;
    }
    // For React elements (like the collapse button), extract the headerVal
    if (column.headerVal) {
        label = column.headerVal;
    }
    
    const pin = column.getIsPinned();
    const isVisible = column.getIsVisible();
    const isSelected = selectedCols.has(column.id);

    if (colSearch && !label.toLowerCase().includes(colSearch.toLowerCase())) {
        if (isGroup) {
            const anyChildMatches = (col) => {
                const childHeader = col.columnDef.header;
                const childLabel = typeof childHeader === 'string' ? childHeader : col.id;
                if (childLabel.toLowerCase().includes(colSearch.toLowerCase())) return true;
                if (col.columns) return col.columns.some(anyChildMatches);
                return false;
            };
            if (!anyChildMatches(column)) return null;
        } else {
            return null;
        }
    }

    const toggleSelection = (e) => {
        e.stopPropagation();
        const newSet = new Set(selectedCols);
        // User fix: When toggling a group selection, it now only selects the leaf columns (removed the parent group ID from the selection array).
        const ids = isGroup ? getAllLeafIdsFromColumn(column) : [column.id];
        
        const allSelected = ids.every(id => newSet.has(id));
        if (allSelected) {
            ids.forEach(id => newSet.delete(id));
        } else {
            ids.forEach(id => newSet.add(id));
        }
        setSelectedCols(newSet);
    };

    const toggleVisibility = (e) => {
        e.stopPropagation();
        
        if (isGroup) {
            const leafCols = getAllLeafColumns(column);
            const shouldShow = leafCols.some(c => !c.getIsVisible());
            leafCols.forEach(c => c.toggleVisibility(shouldShow));
        } else {
            column.toggleVisibility();
        }
    };

    const handlePin = (e, side) => {
        e.stopPropagation();
        handlePinColumn(column.id, side);
    };

    const onColDragStart = (e) => {
        setIsDragging(true);
        e.dataTransfer.setData('text/plain', column.id);
        e.dataTransfer.effectAllowed = 'move';
    };

    const onColDragEnd = () => {
        setIsDragging(false);
    };

    const handleItemDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const droppedColId = e.dataTransfer.getData('text/plain');
        if (droppedColId && onDrop && droppedColId !== column.id) {
             onDrop(droppedColId, sectionId, column.id);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            toggleVisibility(e);
        } else if (e.key === ' ') {
            e.preventDefault();
            toggleSelection(e);
        } else if (e.altKey && e.key === 'ArrowLeft') {
            e.preventDefault();
            handlePinColumn(column.id, 'left');
        } else if (e.altKey && e.key === 'ArrowRight') {
            e.preventDefault();
            handlePinColumn(column.id, 'right');
        } else if (e.altKey && e.key === 'ArrowDown') {
            e.preventDefault();
            handlePinColumn(column.id, false);
        }
    };

    const getPinBtnStyle = (active) => ({
        padding: '4px', 
        background: active ? theme.primary : 'transparent',
        border: 'none', 
        cursor: 'pointer', 
        borderRadius: '4px', 
        color: active ? '#fff' : theme.textSec,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        opacity: active ? 1 : 0.6,
        transition: 'all 0.2s'
    });

    return (
        <div 
            style={{ display: 'flex', flexDirection: 'column', opacity: isDragging ? 0.5 : 1 }}
            draggable={!isGroup}
            onDragStart={onColDragStart}
            onDragEnd={onColDragEnd}
            onDragOver={e => e.preventDefault()}
            onDrop={handleItemDrop}
            role="treeitem"
            aria-selected={isSelected}
            aria-expanded={expanded}
        >
            <div 
                style={{
                    ...styles.columnItem,
                    paddingLeft: `${level * 12 + 8}px`, // Reduced indentation step
                    background: isSelected ? theme.select : (isHovered ? theme.hover : 'transparent'),
                    borderLeft: pin ? `3px solid ${theme.primary}` : '3px solid transparent'
                }} 
                onMouseEnter={e => !isSelected && setIsHovered(true)} 
                onMouseLeave={e => !isSelected && setIsHovered(false)}
                tabIndex={0}
                onKeyDown={handleKeyDown}
            >
                <input 
                    type="checkbox" 
                    checked={isSelected} 
                    onChange={toggleSelection}
                    onClick={(e) => { e.stopPropagation(); toggleSelection(e); }}
                    style={{ margin: 0, cursor: 'pointer', pointerEvents: 'auto' }}
                    tabIndex={-1} 
                />

                {!isGroup && (
                    <span style={{ cursor: 'grab', display: 'flex', opacity: 0.7, marginRight: '4px' }}>
                        <Icons.DragIndicator />
                    </span>
                )}
                
                {isGroup ? (
                    <span onClick={() => setExpanded(!expanded)} style={{ cursor: 'pointer', display: 'flex', opacity: 0.7, marginRight: '4px' }}>
                        {expanded ? <Icons.ChevronDown /> : <Icons.ChevronRight />}
                    </span>
                ) : <span style={{ width: '20px' }} />}
                
                <span 
                    onClick={toggleVisibility}
                    style={{ 
                        cursor: 'pointer', 
                        display: 'flex', 
                        color: isVisible ? theme.primary : theme.textSec,
                        opacity: isVisible ? 1 : 0.5,
                        marginRight: '4px'
                    }}
                    title={isVisible ? "Hide Column" : "Show Column"}
                >
                    {isVisible ? <Icons.Visibility /> : <Icons.VisibilityOff />}
                </span>
                
                <span 
                    style={{ 
                        flex: 1, 
                        overflow: 'hidden', 
                        textOverflow: 'ellipsis', 
                        whiteSpace: 'nowrap',
                        fontWeight: isGroup ? 600 : 400,
                        cursor: isGroup ? 'pointer' : 'default',
                        color: isVisible ? theme.text : theme.textSec,
                        opacity: isVisible ? 1 : 0.6,
                        display: 'flex', alignItems: 'center'
                    }} 
                    onClick={() => isGroup && setExpanded(!expanded)}
                    title={label}
                >
                    {isGroup && <Icons.Group style={{ marginRight: '6px', fontSize: '14px', opacity: 0.8 }} />}
                    {label}
                </span>

                {!isGroup && (
                    <div className="pin-controls">
                        <button onClick={(e) => handlePin(e, 'left')}>
                            <Icons.PinLeft />
                        </button>
                        <button onClick={(e) => handlePin(e, false)}>
                            <Icons.Unpin />
                        </button>
                        <button onClick={(e) => handlePin(e, 'right')}>
                            <Icons.PinRight />
                        </button>
                    </div>
                )}
            </div>
            {isGroup && expanded && (
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    {column.columns
                        .filter(child => hasChildrenInZone(child, sectionId))
                        .map(child => (
                        <ColumnTreeItem 
                            key={child.id} 
                            column={child} 
                            level={level + 1} 
                            theme={theme} 
                            styles={styles} 
                            handlePinColumn={handlePinColumn}
                            colSearch={colSearch}
                            selectedCols={selectedCols}
                            setSelectedCols={setSelectedCols}
                            onDrop={onDrop}
                            sectionId={sectionId}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};

const SidebarFilterItem = ({ column, theme, styles, onFilter, currentFilter, options }) => {
    const [expanded, setExpanded] = useState(false);
    const hasFilter = currentFilter && (currentFilter.conditions || currentFilter.value);

    return (
        <div style={{display: 'flex', flexDirection: 'column'}}>
            <div 
                style={{
                    ...styles.columnItem,
                    cursor: 'pointer',
                    background: expanded ? theme.select : 'transparent',
                    borderLeft: hasFilter ? `3px solid ${theme.primary}` : '3px solid transparent'
                }}
                onClick={() => setExpanded(!expanded)}
            >
                <span style={{marginRight: '8px', opacity: 0.7, display: 'flex'}}>
                    {expanded ? <Icons.ChevronDown/> : <Icons.ChevronRight/>}
                </span>
                <span style={{flex: 1, fontWeight: 500, display: 'flex', alignItems: 'center', gap: '6px'}}>
                    {hasFilter && <Icons.Filter style={{fontSize: '12px', color: theme.primary}}/>}
                    {typeof column.header === 'string' ? column.header : (column.columnDef && typeof column.columnDef.header === 'string' ? column.columnDef.header : column.id)}
                </span>
            </div>
            {expanded && (
                <div style={{padding: '8px', borderBottom: `1px solid ${theme.border}44`}}>
                    <ColumnFilter
                        column={column}
                        onFilter={onFilter}
                        currentFilter={currentFilter}
                        options={options}
                        theme={theme}
                    />
                </div>
            )}
        </div>
    );
};

// --- Tool Panel Section Component ---
const ToolPanelSection = ({ title, children, items, renderItem, theme, styles, initialExpanded = true, count, onDrop, sectionId }) => {
    const [expanded, setExpanded] = useState(initialExpanded);
    const [height, setHeight] = useState(initialExpanded ? 200 : 0);
    const [isOver, setIsOver] = useState(false);
    const contentRef = useRef(null);
    const parentRef = useRef(null);

    useEffect(() => {
        if (expanded && contentRef.current) {
            const contentHeight = contentRef.current.scrollHeight;
            setHeight(Math.min(contentHeight, 300)); // Cap at 300px
        } else {
            setHeight(0);
        }
    }, [expanded, items, count]);

    const rowVirtualizer = useVirtualizer({
        count: items ? items.length : 0,
        getScrollElement: () => contentRef.current,
        estimateSize: () => 32,
        overscan: 10,
        enabled: !!items && expanded
    });

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsOver(true);
    };

    const handleDragLeave = () => {
        setIsOver(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsOver(false);
        const columnId = e.dataTransfer.getData('text/plain');
        if (onDrop && columnId) {
            onDrop(columnId, sectionId);
        }
    };

    return (
        <div
            style={{
                ...styles.toolPanelSection,
                flex: expanded ? '0 1 auto' : '0 0 auto',
                minHeight: expanded ? '40px' : '32px',
                overflow: 'hidden' // Prevent overlap
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            <div style={styles.toolPanelSectionHeader} onClick={() => setExpanded(!expanded)}>
                <span style={{ marginRight: '8px', opacity: 0.7, display: 'flex' }}>
                    {expanded ? <Icons.ChevronDown /> : <Icons.ChevronRight />}
                </span>
                <span style={{ flex: 1 }}>{title}</span>
                {count !== undefined && <span style={{ fontSize: '10px', opacity: 0.5, background: theme.hover, padding: '2px 6px', borderRadius: '10px' }}>{count}</span>}
            </div>

            <div
                ref={contentRef}
                style={{
                    ...styles.toolPanelList,
                    height: `${height}px`,
                    overflowY: expanded ? 'auto' : 'hidden',
                    transition: 'height 0.3s ease',
                    opacity: expanded ? 1 : 0
                }}
            >
                {items ? (
                    <div style={{
                        height: `${rowVirtualizer.getTotalSize()}px`,
                        width: '100%',
                        position: 'relative'
                    }}>
                        {rowVirtualizer.getVirtualItems().map(virtualRow => (
                            <div
                                key={virtualRow.index}
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: `${virtualRow.size}px`,
                                    transform: `translateY(${virtualRow.start}px)`
                                }}
                            >
                                {renderItem(items[virtualRow.index], virtualRow.index)}
                            </div>
                        ))}
                    </div>
                ) : children}
            </div>
        </div>
    );
};

const Notification = ({ message, type, onClose }) => (
    <div style={{
        position: 'fixed', // Changed from absolute
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        padding: '12px 20px',
        borderRadius: '8px',
        color: '#fff',
        fontSize: '14px',
        background: type === 'error' ? '#d32f2f' :
                   type === 'warning' ? '#f57c00' : '#323232',
        boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
        zIndex: 10000, // High z-index
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        maxWidth: '400px',
        minWidth: '200px',
        pointerEvents: 'auto'
    }}>
        <span style={{flex: 1}}>{message}</span>
        <span onClick={onClose} style={{
            cursor:'pointer',
            opacity: 0.7,
            display: 'flex',
            padding: '2px',
            borderRadius: '4px',
            background: 'rgba(255,255,255,0.1)'
        }}><Icons.Close/></span>
    </div>
);

const renderCount = { current: 0 };

export default function DashTanstackPivot(props) {
    renderCount.current++;
    console.log('[DEBUG] Render Entry', props.id, 'Count:', renderCount.current);
    try {
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
        sortLock = false
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

        const [error, setError] = useState(null);

        const showNotification = React.useCallback((msg, type='info') => {

            setNotification({ message: msg, type });

        }, []);

    

        // --- State ---

        const availableFields = useMemo(() => {

            if (serverSide && props.columns) return props.columns.map(c => c.id || c);

            return data && data.length ? Object.keys(data[0]) : [];

        }, [data, props.columns, serverSide]);

    

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

            // Visibility State
            const [columnVisibility, setColumnVisibility] = useState(initialColumnVisibility);

            // Accessibility
            const [announcement, setAnnouncement] = useState("");

            // Refs
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
    }, [lastSelected]); // Fixed: Removed setProps from dependencies

    // Validation helper
    const validateCell = (val, rule) => {
        if (!rule) return true;
        if (rule.type === 'regex') return new RegExp(rule.pattern).test(val);
        if (rule.type === 'numeric') return !isNaN(parseFloat(val));
        if (rule.type === 'required') return val !== null && val !== '' && val !== undefined;
        return true;
    };

    const exportCSV = () => {
        const rowsToExport = rows.map(r => {
             const d = {};
             columns.forEach(c => {
                 const visit = col => {
                     if (col.columns) col.columns.forEach(visit);
                     else if (col.accessorKey) d[col.header] = r.getValue(col.accessorKey);
                     else if (col.accessorFn) d[col.header] = col.accessorFn(r.original);
                 };
                 visit(c);
             });
             return d;
        });
        
        if (rowsToExport.length === 0) return;
        
        const header = Object.keys(rowsToExport[0]).join(',');
        const csv = rowsToExport.map(row => Object.values(row).map(v => `"${v}"`).join(',')).join('\n');
        const blob = new Blob([header + '\n' + csv], { type: 'text/csv;charset=utf-8;' });
        saveAs(blob, 'pivot_data.csv');
    };

    const exportJSON = () => {
        const rowsToExport = rows.map(r => r.original);
        const blob = new Blob([JSON.stringify(rowsToExport, null, 2)], { type: 'application/json' });
        saveAs(blob, 'pivot_data.json');
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

    // 1. FIXED: Helper to check if column is a group
    const isGroupColumn = (column) => {
        return column.columns && column.columns.length > 0;
    };

    // 2. FIXED: Helper to get all leaf IDs from a column (recursive)
    const getAllLeafIdsFromColumn = (column) => {
        const leafIds = [];

        const collectLeafIds = (col) => {
            if (col.columns && col.columns.length > 0) {
                // It's a group - recurse through children
                col.columns.forEach(childCol => collectLeafIds(childCol));
            } else {
                // It's a leaf - add its ID
                leafIds.push(col.id);
            }
        };

        collectLeafIds(column);
        return leafIds;
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
        if (layoutMode === 'hierarchy' && rowFields.length > 0) {
            setColumnPinning(prev => {
                if (!prev.left.includes('hierarchy')) {
                     console.log('[DEBUG] Hierarchy Pinning Triggered');
                     return { ...prev, left: ['hierarchy', ...prev.left] };
                }
                return prev;
            });
        }
    }, [layoutMode, rowFields.length]);

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
            onClick: () => table.toggleAllRowsExpanded(true)
        });
        actions.push({
            label: 'Collapse All Rows',
            onClick: () => table.toggleAllRowsExpanded(false)
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
        
        // Only select if not already selected (to allow multi-cell context actions)
        if (!isSelected && rowId) {
             setSelectedCells({ [key]: value });
        }

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

    const staticTotal = useMemo(() => ({ _isTotal: true }), []);
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

    const columns = useMemo(() => {
        // Enhanced Sorting Logic (Tree-aware + Natural + Customization)
        const customSortingFn = (rowA, rowB, columnId) => {
            try {
                // 1. Tree Data Sorting: Keep Totals at bottom (or top if needed, defaulting to bottom here)
                const aTotal = rowA.original._isTotal;
                const bTotal = rowB.original._isTotal;
                if (aTotal && !bTotal) return 1;
                if (!aTotal && bTotal) return -1;
                if (aTotal && bTotal) return 0;

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
                enablePinning: true,
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
                        {row.index + 1}
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
                                 {row.getCanExpand() && !row.original._isTotal ? (
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
                                <span style={{ fontWeight: row.original._isTotal ? 700 : 400 }}>{row.original._id}</span>
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
                        const showExpander = (i === depth) && row.getCanExpand() && !row.original._isTotal;

                        return (
                            <div
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    width: '100%',
                                    height: '100%',
                                    fontWeight: row.original._isTotal ? 700 : 400
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
            if (filteredData.length > 0) {
                const keys = new Set();
                filteredData.forEach(row => Object.keys(row).forEach(k => keys.add(k)));
                const ignoreKeys = new Set(['_id', 'depth', '_isTotal', '_path', 'uuid', ...rowFields, ...colFields]);
                const flatCols = [];
                Array.from(keys).sort().forEach(k => {
                    if (!ignoreKeys.has(k)) {
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
                    }
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
    }, [filteredData, rowFields, colFields, valConfigs, minMax, colorScale, colExpanded, serverSide, layoutMode, showRowNumbers, isRowSelecting, rowDragStart]); // Removed selectedCells to prevent infinite re-renders

    const tableData = useMemo(() => {
        let baseData = serverSide ? filteredData : (filteredData.length ? [...nodes, total] : []);
        if (serverSide && !showColTotals) {
            baseData = baseData.filter(r => !r._isTotal);
        }
        return baseData;
    }, [nodes, total, filteredData, serverSide, showColTotals]);


    const tableState = useMemo(() => ({
        sorting,
        expanded,
        columnPinning,
        rowPinning,
        grouping: rowFields,
        columnVisibility
    }), [sorting, expanded, columnPinning, rowPinning, rowFields, columnVisibility]);

    const handleSortingChange = useCallback((updater) => {
        if (sortLock) {
            showNotification('Sorting is locked.', 'warning');
            return;
        }
        setSorting(updater);
    }, [sortLock, showNotification]);

    const coreRowModelFn = useMemo(() => getCoreRowModel(), []);
    const expandedRowModelFn = useMemo(() => getExpandedRowModel(), []);
    const groupedRowModelFn = useMemo(() => getGroupedRowModel(), []);
    const getRowId = useCallback((row, relativeIndex) => row._path || (row.id ? row.id : String(relativeIndex)), []);
    const getSubRows = useCallback(r => r.subRows, []);
    const getRowCanExpand = useCallback(row => {
        if (row.original._isTotal) return false;
        if (serverSide) return row.original.depth < rowFields.length - 1;
        return row.subRows && row.subRows.length > 0;
    }, [serverSide, rowFields.length]);
    const getIsRowExpanded = useCallback(row => !row.original._isTotal && (expanded === true || !!expanded[row.id]), [expanded]);

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
        getCoreRowModel: coreRowModelFn,
        getExpandedRowModel: expandedRowModelFn,
        getGroupedRowModel: groupedRowModelFn,
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

    const parentRef = useRef(null);
    const { rows } = table.getRowModel();
    const topRows = table.getTopRows();
    const bottomRows = table.getBottomRows();
    const centerRows = table.getCenterRows();

    const visibleLeafColumns = table.getVisibleLeafColumns();
    const rowHeight = rowHeights[spacingMode];
    
    const rowVirtualizer = useVirtualizer({
        count: centerRows.length, getScrollElement: () => parentRef.current, estimateSize: () => rowHeight, overscan: 20
    });
    const virtualRows = rowVirtualizer.getVirtualItems();


    // --- Optimized Horizontal Virtualization with Sticky Support ---
    const leftCols = table.getLeftLeafColumns();
    const rightCols = table.getRightLeafColumns();
    const centerCols = table.getCenterLeafColumns();

    const columnVirtualizer = useVirtualizer({
        horizontal: true,
        count: centerCols.length,
        getScrollElement: () => parentRef.current,
        estimateSize: (index) => centerCols[index].getSize(),
        overscan: 5
    });
    const virtualCenterCols = columnVirtualizer.getVirtualItems();
    const centerTotalWidth = columnVirtualizer.getTotalSize();
    
    // Calculate spacers for virtualized center
    const [beforeWidth, afterWidth] = virtualCenterCols.length > 0 ? [
        Math.max(0, virtualCenterCols[0].start),
        Math.max(0, centerTotalWidth - virtualCenterCols[virtualCenterCols.length - 1].end)
    ] : [0, 0];

    // Add this useEffect instead for pinning validation
    /*
    useEffect(() => {
        if (!visibleLeafColumns.length) return;

        const allLeafIds = visibleLeafColumns.map(c => c.id);
        const { left = [], right = [] } = columnPinning;

        const validLeft = left.filter(id => allLeafIds.includes(id));
        const validRight = right.filter(id => allLeafIds.includes(id));

        // Check if we need to update
        if (validLeft.length !== left.length || validRight.length !== right.length) {
            console.log('[DEBUG] Pinning Validation Triggered Update', { left, right, validLeft, validRight, allLeafIds });
            setColumnPinning({
                left: validLeft,
                right: validRight
            });
        }
    }, [visibleLeafColumns, columnPinning.left, columnPinning.right]); // Only run when these change
    */

    // DEBUG HELPER: Removed logging to prevent console flooding
    useEffect(() => {
        /*
        console.log('[PINNING STATE]', {
            left: columnPinning.left,
            right: columnPinning.right,
            visibleLeafCount: visibleLeafColumns.length
        });
        */
    }, [columnPinning.left, columnPinning.right, visibleLeafColumns.length]); 

    // Helper function to check if theme is dark
    const isDarkTheme = (theme) => theme.name === 'dark' || theme.text === '#fff';

    // Border utility functions
    const getBorderColor = (theme, opacity = 1.0) => {
        const rgb = theme.border.startsWith('#')
            ? theme.border
            : '#e0e0e0'; // fallback
        return `${rgb}${Math.floor(opacity * 255).toString(16).padStart(2, '0')}`;
    };



    // --- Add custom hook for sticky styles ---
    const useStickyStyles = (visibleLeafColumns, columnPinning, theme, leftCols, rightCols) => {
        return useMemo(() => {
            const { left, right } = columnPinning;

            const getHeaderStickyStyle = (header, level = 0, isLastPinnedLeft = false, isFirstPinnedRight = false, renderSection = 'center') => {
                const isGroupHeader = header.column.columns && header.column.columns.length > 0;
                
                let shouldBeSticky = false;
                let effectivePinDirection = null;
                
                if (isGroupHeader) {
                    // Group headers become sticky when rendered in pinned sections
                    if (renderSection === 'left') {
                        shouldBeSticky = true;
                        effectivePinDirection = 'left';
                    } else if (renderSection === 'right') {
                        shouldBeSticky = true;
                        effectivePinDirection = 'right';
                    }
                } else {
                    // Leaf column - use direct pinning
                    const isPinned = header.column.getIsPinned();
                    effectivePinDirection = isPinned;
                    shouldBeSticky = !!isPinned;
                }

                if (!shouldBeSticky) return {};

                // Z-index calculation
                const baseZIndex = 500 - (level * 10);
                const isLeaf = !isGroupHeader;
                const zIndexBoost = isLeaf ? 2 : 1;

                const style = {
                    position: 'sticky',
                    zIndex: baseZIndex + zIndexBoost,
                    background: theme.headerBg,
                    top: `${level * 40}px`,
                };

                // Calculate position
                if (effectivePinDirection === 'left') {
                    if (isGroupHeader) {
                        // For group headers in left section, position at the start of leftmost pinned child
                        const leafColumns = getAllLeafColumns(header.column);
                        const pinnedLeaves = leafColumns.filter(col => col.getIsPinned() === 'left');
                        const leftmostLeaf = pinnedLeaves[0];
                        style.left = leftmostLeaf ? `${leftmostLeaf.getStart('left')}px` : '0px';
                    } else {
                        style.left = `${header.column.getStart('left')}px`;
                    }
                    if (isLastPinnedLeft) {
                        style.borderRight = `1px solid ${theme.border}`;
                        style.boxShadow = '2px 0 5px -2px rgba(0,0,0,0.2)';
                        style.zIndex = style.zIndex + 1;
                    }
                } else if (effectivePinDirection === 'right') {
                    if (isGroupHeader) {
                        // For group headers in right section, position at the end of rightmost pinned child
                        const leafColumns = getAllLeafColumns(header.column);
                        const pinnedLeaves = leafColumns.filter(col => col.getIsPinned() === 'right');
                        const rightmostLeaf = pinnedLeaves[pinnedLeaves.length - 1];
                        style.right = rightmostLeaf ? `${rightmostLeaf.getAfter('right')}px` : '0px';
                    } else {
                        style.right = `${header.column.getAfter('right')}px`;
                    }
                    if (isFirstPinnedRight) {
                        style.borderLeft = `1px solid ${theme.border}`;
                        style.boxShadow = '-2px 0 5px -2px rgba(0,0,0,0.2)';
                        style.zIndex = style.zIndex + 1;
                    }
                }

                return style;
            };

            const getStickyStyle = (column, rowBackground) => {
                const isPinned = column.getIsPinned();
                const isLeft = isPinned === 'left';
                const isRight = isPinned === 'right';

                if (!isPinned) return {};

                const style = {
                    position: 'sticky',
                    zIndex: column.id === 'hierarchy' ? 30 : 20,
                    background: rowBackground,
                };

                if (isLeft) {
                    style.left = `${column.getStart('left')}px`;
                    style.borderRight = `1px solid ${theme.border}`;
                } else if (isRight) {
                    style.right = `${column.getAfter('right')}px`;
                    style.borderLeft = `1px solid ${theme.border}`;
                }
                return style;
            };

            return { getHeaderStickyStyle, getStickyStyle };
        }, [visibleLeafColumns, columnPinning, theme, leftCols, rightCols]);
    };


    // Use the custom hook
    const { getHeaderStickyStyle, getStickyStyle } = useStickyStyles(
        visibleLeafColumns,
        columnPinning,
        theme,
        leftCols,
        rightCols
    );


    // 8. QUICK TEST: Removed logging to prevent console flooding
    useEffect(() => {
        /*
        console.log('[PINNED COLUMNS]', {
            left: columnPinning.left,
            right: columnPinning.right,
            visibleCount: visibleLeafColumns.length
        });
        */
    }, [columnPinning.left, columnPinning.right, visibleLeafColumns.length]); 

    // Calculate total layout width for scroll container
    const leftWidth = leftCols.reduce((acc, col) => acc + col.getSize(), 0);
    const rightWidth = rightCols.reduce((acc, col) => acc + col.getSize(), 0);
    const totalLayoutWidth = leftWidth + centerTotalWidth + rightWidth;

    // --- Progressive Data Loading ---
    const viewportRange = virtualRows.length > 0 
        ? `${virtualRows[0].index}-${virtualRows[virtualRows.length - 1].index}` 
        : '';

    useEffect(() => {
        if (!serverSide || !setPropsRef.current || virtualRows.length === 0) return;

        const start = virtualRows[0].index;
        const end = virtualRows[virtualRows.length - 1].index;
        
        const timer = setTimeout(() => {
             setPropsRef.current({ 
                viewport: { start, end, count: end - start + 1 } 
             });
        }, 100);

        return () => clearTimeout(timer);
    }, [viewportRange, serverSide]);


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
        if (srcZone !== targetZone) {
            if (srcZone==='rows') setRowFields(p=>p.filter(f=>f!==fieldName));
            if (srcZone==='cols') setColFields(p=>p.filter(f=>f!==fieldName));
            if (srcZone==='vals') setValConfigs(p=>p.filter((_,i)=>i!==srcIdx));
            if (targetZone==='rows') setRowFields(p=>insertItem(p, targetIdx, fieldName));
            if (targetZone==='cols') setColFields(p=>insertItem(p, targetIdx, fieldName));
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
            const renderCell = useCallback((cell, virtualRowIndex, isLastPinnedLeft = false, isFirstPinnedRight = false) => {
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
    
            const rowBackground = row.original._isTotal ? '#e8f5e9' : (isDarkTheme(theme) ? '#212121' : '#fff');
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
                        justifyContent: isHierarchy ? 'flex-start' : 'flex-end',
                        fontWeight: (row.original._isTotal || (isHierarchy && row.getIsGrouped())) ? 500 : 400,
                        background: bg,
                        ...stickyStyle,
                        ...condStyle,
                        ...(isFillSelected ? {boxShadow: `inset 0 0 0 1px ${theme.primary}`} : {}),
                        userSelect: 'none',
                        position: stickyStyle && stickyStyle.position === 'sticky' ? 'sticky' : 'relative'
                    }}
                    onContextMenu={e => handleContextMenu(e, cell.getValue(), cell.column.id, row)}
                >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
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

        // Calculate sticky style for pinned headers using the hook
        const stickyStyle = getHeaderStickyStyle(header, level, isLastPinnedLeft, isFirstPinnedRight, renderSection);

        return (
            <div key={header.id} style={{
                ...styles.headerCell,
                width: header.getSize(),
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
            draggable={!isGroupHeader}
            onDragStart={(e) => {
                if (!isGroupHeader) {
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
                    onClose={() => { setActiveFilterCol(null); setFilterAnchorEl(null); }}
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
                            {[{id:'rows', label:'Rows'}, {id:'cols', label:'Columns'}, {id:'vals', label:'Values'}].map(zone => (
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
                                                                                                            <select value={item.agg} onChange={e=>{const n=[...valConfigs];n[idx].agg=e.target.value;setValConfigs(n)}} style={{border:'none',background:'transparent',color:theme.primary,cursor:'pointer',maxWidth:'50px',fontSize:'11px'}}><option value="sum">Sum</option><option value="avg">Avg</option><option value="count">Cnt</option></select>
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
                                                                                                                background: filters[label] ? theme.select : 'transparent',
                                                                                                                color: filters[label] ? theme.primary : 'inherit'
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
                                                                                                                                                        onClose={() => setActiveFilterCol(null)}
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
                                        <div style={{height:20}} onDragOver={e=>onDragOver(e,zone.id,(zone.id==='rows'?rowFields:zone.id==='cols'?colFields:valConfigs).length)} />
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
                                    {[
                                        {value: 'all', label: 'All', icon: '📊'},
                                        {value: 'number', label: 'Numbers', icon: '🔢'},
                                        {value: 'string', label: 'Text', icon: '📝'},
                                        {value: 'date', label: 'Dates', icon: '📅'}
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
                         <div style={{width: `${totalLayoutWidth}px`, minWidth:'100%', height: `${rowVirtualizer.getTotalSize() + (topRows.length + bottomRows.length) * rowHeight}px`, position: 'relative'}}>
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
                             {topRows.map((row, i) => {
                                 const isExpandedRow = row.getIsExpanded();
                                 const isLastPinnedTop = i === topRows.length - 1;
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
                                         background: row.original._isTotal ? '#e8f5e9' : theme.background,
                                         borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                         boxShadow: isLastPinnedTop ? `0 2px 4px -2px ${theme.border}80` : 'none'
                                     }}>
                                         {row.getLeftVisibleCells().map((cell, idx) => renderCell(cell, i, idx === leftCols.length - 1, false))}
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {virtualCenterCols.map(virtualCol => {
                                             const cell = row.getCenterVisibleCells()[virtualCol.index];
                                             return renderCell(cell, i);
                                         })}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                         {row.getRightVisibleCells().map((cell, idx) => renderCell(cell, i, false, idx === 0))}
                                     </div>
                                 )
                             })}

                             {/* Center Virtualized Rows */}
                             {virtualRows.map(virtualRow => {
                                 const row = centerRows[virtualRow.index];
                                 const isExpandedRow = row.getIsExpanded();
                                 const headerHeight = (table.getHeaderGroups().length * rowHeight) + (showFloatingFilters ? rowHeight : 0);
                                 const topOffset = headerHeight + (topRows.length * rowHeight);

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
                                         background: row.original._isTotal ? '#e8f5e9' : '#fff',
                                         borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                         transition: rowVirtualizer.isScrolling ? 'none' : 'top 0.2s ease-out, background-color 0.2s'
                                     }}>
                                         {row.getLeftVisibleCells().map((cell, idx) => renderCell(cell, virtualRow.index, idx === leftCols.length - 1, false))}
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {virtualCenterCols.map(virtualCol => {
                                             const cell = row.getCenterVisibleCells()[virtualCol.index];
                                             return renderCell(cell, virtualRow.index);
                                         })}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                         {row.getRightVisibleCells().map((cell, idx) => renderCell(cell, virtualRow.index, false, idx === 0))}
                                     </div>
                                 )
                             })}

                             {/* Bottom Pinned Rows */}
                             {bottomRows.map((row, i) => {
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
                                         bottom: ((bottomRows.length - 1 - i) * rowHeight),
                                         zIndex: 50, // Increased for bottom rows
                                         background: row.original._isTotal ? '#e8f5e9' : theme.background,
                                         borderBottom: isExpandedRow ? `2px solid ${theme.primary}` : `1px solid ${theme.border}`,
                                         boxShadow: isFirstPinnedBottom ? `0 -2px 4px -2px ${theme.border}80` : 'none'
                                     }}>
                                         {row.getLeftVisibleCells().map((cell, idx) => renderCell(cell, i, idx === leftCols.length - 1, false))}
                                         <div style={{ width: beforeWidth, flexShrink: 0 }} />
                                         {virtualCenterCols.map(virtualCol => {
                                             const cell = row.getCenterVisibleCells()[virtualCol.index];
                                             return renderCell(cell, i);
                                         })}
                                         <div style={{ width: afterWidth, flexShrink: 0 }} />
                                         {row.getRightVisibleCells().map((cell, idx) => renderCell(cell, i, false, idx === 0))}
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
    } catch (e) {
        console.error('[DEBUG] Render Crash', e);
        return <div>Error in DashTanstackPivot: {e.message}</div>;
    } finally {
        console.log('[DEBUG] Render Exit');
    }
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
    sortEvent: PropTypes.object
};
