

// Mock React state
let colExpanded = {};
const toggleCol = (key) => { colExpanded[key] = colExpanded[key] === undefined ? false : !colExpanded[key]; };
const isColExpanded = (key) => colExpanded[key] !== false;

const isPathExpanded = (key) => {
    if (!key) return true;
    const parts = key.split('|||');
    let currentPath = "";
    for (const part of parts) {
        currentPath = currentPath ? `${currentPath}|||${part}` : part;
        if (colExpanded[currentPath] === false) return false;
    }
    return true;
};

// Configs
const valConfigs = [{field: 'sales', agg: 'sum'}, {field: 'cost', agg: 'sum'}];
const flatCols = [
    { id: '2024|Q1_sales_sum', header: '2024|Q1_sales_sum' },
    { id: '2024|Q1_cost_sum', header: '2024|Q1_cost_sum' },
    { id: '2024|Q2_sales_sum', header: '2024|Q2_sales_sum' }
];

// Recursive Builder
const buildRecursiveTree = (cols) => {
    const root = { columns: [] };
    cols.forEach(col => {
        const key = col.id;
        
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
        
        const dimPath = dimStr ? dimStr.split('|') : [];
        let current = root;
        let pathKey = '';
        
        dimPath.forEach((val, idx) => {
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
        });

        if (isPathExpanded(pathKey) || dimPath.length === 0) {
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

console.log("--- Initial State (Expanded) ---");
const tree1 = buildRecursiveTree(flatCols);
console.log(JSON.stringify(tree1, null, 2));

console.log("\n--- Collapsing 2024 ---");
colExpanded['2024'] = false;
const tree2 = buildRecursiveTree(flatCols);
console.log(JSON.stringify(tree2, null, 2));

console.log("\n--- Collapsing 2024|Q1 ---");
colExpanded['2024'] = true; // Expand root
colExpanded['2024|||Q1'] = false; // Collapse child
const tree3 = buildRecursiveTree(flatCols);
console.log(JSON.stringify(tree3, null, 2));
