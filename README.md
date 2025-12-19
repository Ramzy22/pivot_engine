# 🚀 Scalable Pivot Engine

A high-performance, database-agnostic pivot engine designed to handle **millions of rows** with ease. Built on [Ibis](https://ibis-project.org/) and [Apache Arrow](https://arrow.apache.org/), it provides a production-grade backend for hierarchical data exploration, infinite scrolling, and real-time analytical updates.

---

## ✨ Key Capabilities

### 🏎️ Performance at Scale
- **Millions of Rows**: Optimized for large-scale datasets using vectorized operations and zero-copy data transfers via PyArrow.
- **Async Materialization**: Background pre-computation of hierarchical rollups (levels/drill-down paths) to ensure sub-second UI responsiveness.
- **Automatic Indexing**: Automatically creates database indexes on materialized views to maintain high performance during deep drill-downs.
- **Intelligent Caching**: Multi-level caching (Memory/Redis) with semantic query diffing to minimize database load.

### 🌐 Backend Agnostic
- **Powered by Ibis**: Support for 20+ backends including **DuckDB, Clickhouse, PostgreSQL, BigQuery, Snowflake, and MySQL**.
- **Consistent API**: One query language (PivotSpec) for all databases.

### 🧩 Frontend Ready
- **TanStack Table Adapter**: Native support for TanStack Table (React Table) filter/sort/grouping models.
- **Virtual Scrolling**: Built-in support for hierarchical infinite scrolling with cursor-based pagination.
- **REST & WebSocket**: Full FastAPI-based REST API and WebSockets for real-time data streaming (CDC).

---

## 🛠️ Installation

```bash
# Core installation
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

*Required: `python >= 3.9`, `duckdb`, `pyarrow`, `ibis-framework`.*

---

## 🚀 Quick Start

### 1. Initialize the Engine
```python
from pivot_engine.scalable_pivot_controller import ScalablePivotController

# Connect to any database (e.g., local DuckDB or remote ClickHouse)
controller = ScalablePivotController(
    backend_uri="duckdb://data.db",
    cache="memory",
    enable_streaming=True
)
```

### 2. Run a Hierarchical Pivot
```python
from pivot_engine.types.pivot_spec import PivotSpec

spec = PivotSpec(
    table="sales",
    rows=["region", "category", "product"],
    measures=[{"field": "amount", "agg": "sum", "alias": "total_sales"}],
    totals=True
)

# Execute (Async)
result = await controller.run_pivot_async(spec, return_format="dict")
```

### 3. Materialize for Performance
For datasets with millions of rows, trigger a background materialization job:
```python
# Start background job
job = await controller.run_materialized_hierarchy(spec)
job_id = job["job_id"]

# Check status
status = controller.get_materialization_status(job_id)
# Returns: {"status": "completed", "progress": 100, ...}
```

---

## 📡 REST API & Frontend Integration

The engine comes with a built-in FastAPI implementation that maps directly to common frontend state management.

### TanStack Table Integration
Send your TanStack state directly to the `/pivot/tanstack` endpoint:

```json
// POST /pivot/tanstack
{
  "operation": "get_data",
  "table": "sales_data",
  "columns": [{"id": "region"}, {"id": "sales", "aggregationFn": "sum"}],
  "grouping": ["region"],
  "pagination": {"pageIndex": 0, "pageSize": 100}
}
```

### Endpoints Overview
| Endpoint | Description |
| :--- | :--- |
| `POST /pivot/tanstack` | Direct TanStack Table integration. |
| `POST /pivot/virtual-scroll` | Optimized hierarchical rows for infinite scrolling. |
| `POST /pivot/materialized-hierarchy` | Trigger background pre-computation. |
| `GET /pivot/jobs/{id}` | Poll status of materialization/long-running queries. |
| `WS /ws/pivot/{id}` | WebSocket for real-time CDC updates. |

---

## 🏗️ Architecture

- **UI Layer**: TanStack / React / Vue
- **API Layer**: FastAPI + WebSockets
- **Controller**: ScalablePivotController (Orchestration)
- **Engine**: Ibis + PyArrow (Vectorized execution)
- **Storage**: Any Ibis-supported DB (DuckDB, ClickHouse, etc.)

---

## 🧪 Development & Testing

```bash
# Run the complete test suite
pytest tests/

# Verify performance features
python test_arrow_conversion.py
```

## 📄 License
MIT © 2025 Pivot Engine Team
