# Pivot Engine

A high-performance, scalable pivot engine optimized for large datasets with advanced features including real-time updates, hierarchical data processing, and TanStack integration.

## 🚀 Features

### Scalable Architecture
- **Millions of rows support** - Optimized for large datasets with virtual scrolling
- **Microservice architecture** - Decoupled services for horizontal scaling
- **Distributed caching** - L1/L2 caching with various backends
- **Tile-aware processing** - Smart chunking for virtual scrolling

### Advanced Pivot Features
- **Hierarchical data processing** - Multi-level pivot with drill-down support
- **Virtual scrolling** - Efficient rendering for large hierarchical datasets
- **Progressive loading** - Load data in chunks as needed
- **Materialized hierarchies** - Pre-computed rollups for performance
- **Intelligent prefetching** - Predictive data loading based on patterns

### Real-time Capabilities
- **Change Data Capture (CDC)** - Real-time tracking of data changes
- **Streaming aggregations** - Real-time rollup computations
- **Incremental materialized views** - Automatically updated pre-computed views
- **WebSocket updates** - Real-time UI updates

### Performance Optimizations
- **Query planning** - Cost-based optimization and plan selection
- **Query diffing** - Semantic spec diffing for intelligent caching
- **Pruning strategies** - Multiple algorithms to reduce complexity
- **Arrow-native operations** - Zero-copy data transfers
- **Async everywhere** - Fully asynchronous architecture for high concurrency
- **Optimized Arrow-to-JSON conversion** - Vectorized serialization for massive data transfer speedups
- **Arrow Flight support** - Native Arrow format for frontend integration

## 🛠️ Supported Backends

- **DuckDB** (Primary, optimized)
- **ClickHouse** (via Ibis)
- **PostgreSQL** (via Ibis)
- **MySQL** (via Ibis)
- **BigQuery** (via Ibis)
- **Snowflake** (via Ibis)
- **SQLite** (via Ibis)

## 📊 Frontend Integration

### Direct TanStack Integration (Recommended)
```python
from pivot_engine.tanstack_adapter import create_tanstack_adapter

# Create adapter that bypasses REST API
adapter = create_tanstack_adapter(backend_uri="clickhouse://user:pass@host:port/db")

# Handle TanStack requests directly
result = await adapter.handle_request(tanstack_request)
```

### REST API Integration
```python
from pivot_engine.complete_rest_api import create_realtime_api

# Create complete REST API with all endpoints
api = create_realtime_api(backend_uri="clickhouse://user:pass@host:port/db")
app = api.get_app()  # FastAPI app
```

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           CLIENT (Browser/UI)                             │
├───────────────────────────────────────────────────────────────────────────┤
│  TanStack Table/Query or any framework                                    │
│     │                                                                    │
│     └─ Direct adapter (bypasses REST) or REST API via HTTP              │
└───────────────────────────────────────────────────────────────────────────┘

                                   │ JSON over HTTP
                                   ▼

┌───────────────────────────────────────────────────────────────────────────┐
│                    SERVER (Microservice Architecture)                     │
├───────────────────────────────────────────────────────────────────────────┤
│  ScalablePivotController + TanStack Adapter                              │
│    ├─ All scalable features: CDC, streaming, caching, optimization        │
│    ├─ Materialized hierarchies, intelligent prefetching                   │
│    ├─ Async everywhere for high concurrency                               │
│    └─ Progressive hierarchical loading                                   │
└───────────────────────────────────────────────────────────────────────────┘

                                   │
                                   ▼

┌───────────────────────────────────────────────────────────────────────────┐
│                        BACKEND DATABASES                                  │
├───────────────────────────────────────────────────────────────────────────┤
│  DuckDB, ClickHouse, PostgreSQL, MySQL, BigQuery, Snowflake, SQLite     │
│     ├─ Ibis abstraction layer for backend agnosticism                    │
│     └─ Optimized queries per backend                                     │
└───────────────────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Ramzy22/pivot_engine.git
cd pivot_engine

# Install dependencies
pip install -e .
```

## 💻 Usage

### Basic Usage
```python
from pivot_engine.scalable_pivot_controller import ScalablePivotController
from pivot_engine.types.pivot_spec import PivotSpec, Measure

# Create controller
controller = ScalablePivotController(
    backend_uri="clickhouse://user:pass@host:port/database",
    enable_streaming=True,
    enable_incremental_views=True,
    tile_size=100
)

# Define pivot specification
spec = PivotSpec(
    table="sales",
    rows=["region", "product", "category"],  # Hierarchical structure
    measures=[Measure(field="sales", agg="sum", alias="total_sales")],
    filters=[],
    totals=True
)

# Execute pivot (sync)
result = controller.run_pivot(spec)

# Execute pivot (async for high concurrency)
async_result = await controller.run_pivot_async(spec)

# Get raw Arrow table for Arrow Flight (sync)
arrow_result = controller.run_pivot_arrow(spec)

# Get raw Arrow table for Arrow Flight (async)
async_arrow_result = await controller.run_pivot_arrow_async(spec)
```

### TanStack Integration
```python
from pivot_engine.tanstack_adapter import TanStackPivotAdapter, TanStackRequest, TanStackOperation

# Create adapter
adapter = TanStackPivotAdapter(controller)

# Create TanStack request
request = TanStackRequest(
    operation=TanStackOperation.GET_DATA,
    table="sales",
    columns=[
        {"id": "region", "header": "Region"},
        {"id": "total_sales", "header": "Sales", "aggregationFn": "sum", "aggregationField": "sales"}
    ],
    grouping=["region"],
    pagination={"pageIndex": 0, "pageSize": 100}
)

# Get results in TanStack format
result = await adapter.handle_request(request)
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_complete_implementation.py

# Run scalability tests
python tests/test_scalable_pivot.py
```

## 🚀 Deploy

### Production Deployment
```bash
# With FastAPI (if available)
uvicorn pivot_engine.main_complete:main --host 0.0.0.0 --port 8000

# Or using the complete engine
python -c "from pivot_engine.main_complete import create_complete_engine; import asyncio; asyncio.run(create_complete_engine())"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - See `LICENSE` file for details.

## 📞 Support

- Report issues on [GitHub Issues](https://github.com/Ramzy22/pivot_engine/issues)
- For questions, open a discussion

## 🙏 Acknowledgments

- Built with Python, DuckDB, Ibis, Arrow
- Inspired by the need for scalable pivot operations on million+ row datasets
- Thank you to all contributors and users