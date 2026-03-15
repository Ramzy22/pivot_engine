import React from 'react';
import Icons from './Icons';
import { themes } from '../utils/styles';

export function PivotAppBar({
    sidebarOpen, setSidebarOpen,
    themeName, setThemeName,
    showRowNumbers, setShowRowNumbers,
    showFloatingFilters, setShowFloatingFilters,
    showRowTotals, setShowRowTotals,
    showColTotals, setShowColTotals,
    spacingMode, setSpacingMode, spacingLabels,
    layoutMode, setLayoutMode,
    colorScale, setColorScale,
    rowCount, exportPivot,
    theme, styles,
    setFilters,
}) {
    return (
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
    );
}
