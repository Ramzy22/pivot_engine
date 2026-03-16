

// Mock logic from DashTanstackPivot.react.js

let colExpanded = {};
const toggleCol = (key) => { colExpanded[key] = colExpanded[key] === undefined ? false : !colExpanded[key]; };
const isColExpanded = (key) => colExpanded[key] !== false;

// Mock valConfigs
const valConfigs = [{field: 'sales', agg: 'sum'}];

// Mock Columns (Flat from backend)
// Scenario: Year > Quarter > Month
const flatCols = [
    { id: '2024|Q1|Jan_sales_sum', header: 'Jan' },
    { id: '2024|Q1|Feb_sales_sum', header: 'Feb' },
    { id: '2024|Q2|Apr_sales_sum', header: 'Apr' },
    { id: '2025|Q1|Jan_sales_sum', header: 'Jan' }
];

// Recursive Builder Logic (Copied/Adapted)
const buildRecursiveTree = (cols) => {
    const root = { columns: [] };
    cols.forEach(col => {
        const key = col.id;
        
        let dimStr = "";
        let measureStr = "";
        
        // Mock suffix match logic
        if (key.endsWith("_sales_sum")) {
            measureStr = "sales sum";
            dimStr = key.substring(0, key.length - "_sales_sum".length);
        }
        
        const dimPath = dimStr ? dimStr.split('|') : [];
        
        let current = root;
        let pathKey = '';
        let parentCollapsed = false;
        
        for (let idx = 0; idx < dimPath.length; idx++) {
            const val = dimPath[idx];
            
            // CHECK 1: Parent Expanded?
            // If idx > 0, pathKey is the parent's key from previous iteration
            if (idx > 0 && !isColExpanded(pathKey)) {
                parentCollapsed = true;
                break;
            }
            
            pathKey = pathKey ? `${pathKey}|||${val}` : val;
            
            let node = current.columns.find(c => c.headerVal === val);
            if (!node) {
                node = { 
                    id: pathKey, 
                    headerVal: val,
                    columns: [] 
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
                });
             }
             return;
        }

        // Add leaf measure if THIS node is expanded
        // pathKey is now the FULL path to the current group (e.g. 2024|Q1|Jan)
        if (isColExpanded(pathKey) || dimPath.length === 0) {
            current.columns.push({
                ...col,
                header: measureStr || col.header
            });
        } else if (current.columns.length === 0) {
             current.columns.push({
                id: pathKey + "_collapsed",
                header: "...",
            });
        }
    });
    return root.columns;
};

// --- Test Cases ---

console.log("--- Case 1: Fully Expanded ---");
colExpanded = {}; // Reset
const res1 = buildRecursiveTree(flatCols);
// Expect: 2024 -> Q1 -> Jan, Feb. 2024 -> Q2 -> Apr. 2025 -> ...
console.log(JSON.stringify(res1, (k,v) => k==='columns' ? (v.length > 0 ? v : undefined) : v, 2));

console.log("\n--- Case 2: Collapse 2024 (Root Level) ---");
colExpanded = { '2024': false };
const res2 = buildRecursiveTree(flatCols);
// Expect: 2024 -> [...]. 2025 -> ...
console.log(JSON.stringify(res2, (k,v) => k==='columns' ? (v.length > 0 ? v : undefined) : v, 2));

console.log("\n--- Case 3: Collapse 2024|Q1 (Second Level) ---");
colExpanded = { '2024': true, '2024|||Q1': false };
const res3 = buildRecursiveTree(flatCols);
// Expect: 2024 -> Q1 -> [...]. 2024 -> Q2 -> Apr.
console.log(JSON.stringify(res3, (k,v) => k==='columns' ? (v.length > 0 ? v : undefined) : v, 2));

console.log("\n--- Case 4: Deep Collapse Logic Check ---");
// Simulate user complaint: "collapse next level first column only"
// Maybe 2024 is collapsed, but loop continues for Q2?
colExpanded = { '2024': false };
// Loop trace:
// 1. 2024|Q1|Jan. 
// idx=0. val=2024. pathKey=2024. Node 2024 created.
// idx=1. val=Q1. prev pathKey=2024. isColExpanded(2024) FALSE. Break.
// 2024 gets placeholder.

// 2. 2024|Q2|Apr. 
// idx=0. val=2024. pathKey=2024. Node 2024 found.
// idx=1. val=Q2. prev pathKey=2024. isColExpanded(2024) FALSE. Break.
// 2024 has placeholder. No duplicate placeholder added.

// Result should be correct.

