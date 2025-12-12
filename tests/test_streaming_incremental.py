"""
Test suite for streaming and incremental views
"""
import pytest
import asyncio
from unittest.mock import Mock
import pyarrow as pa
from pivot_engine.streaming.streaming_processor import StreamAggregationProcessor, IncrementalMaterializedViewManager
from pivot_engine.types.pivot_spec import PivotSpec, Measure


@pytest.fixture
def mock_backend():
    """Mock backend for testing"""
    backend = Mock()
    backend.execute = Mock(return_value=pa.table({"col1": [1], "col2": [2]}))
    return backend


@pytest.fixture
def stream_processor():
    """Create stream aggregation processor for testing"""
    processor = StreamAggregationProcessor()
    return processor


@pytest.fixture
def incremental_view_manager(mock_backend):
    """Create incremental materialized view manager for testing"""
    manager = IncrementalMaterializedViewManager(mock_backend)
    return manager


def test_stream_aggregation_processor_creation(stream_processor):
    """Test creation of stream aggregation processor"""
    assert stream_processor is not None
    assert stream_processor.aggregation_jobs == {}
    assert stream_processor.kafka_config == {}


@pytest.mark.asyncio
async def test_create_real_time_aggregation_job(stream_processor):
    """Test creation of real-time aggregation jobs"""
    spec = PivotSpec(
        table="test_table",
        rows=["region"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    
    job_id = await stream_processor.create_real_time_aggregation_job(spec)
    
    assert isinstance(job_id, str)
    assert job_id.startswith("agg_job_")
    assert job_id in stream_processor.aggregation_jobs


@pytest.mark.asyncio
async def test_maintain_incremental_views(stream_processor):
    """Test maintaining incremental views"""
    spec = PivotSpec(
        table="test_table", 
        rows=["region"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    
    # Test maintaining incremental views
    await stream_processor.maintain_incremental_views([spec])
    
    # Should have created jobs for the spec
    assert len(stream_processor.aggregation_jobs) >= 1


@pytest.mark.asyncio
async def test_process_stream_update(stream_processor):
    """Test processing of stream updates"""
    # Add a job first
    spec = PivotSpec(
        table="test_table",
        rows=["region"], 
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    
    await stream_processor.create_real_time_aggregation_job(spec)
    
    # Process a stream update
    record = {"region": "North", "sales": 100}
    await stream_processor.process_stream_update("test_table", record, "INSERT")
    
    # Should not raise an exception
    assert True


def test_incremental_materialized_view_manager_creation(incremental_view_manager):
    """Test creation of incremental view manager"""
    assert incremental_view_manager is not None
    assert incremental_view_manager.views == {}


@pytest.mark.asyncio
async def test_create_incremental_view(incremental_view_manager, mock_backend):
    """Test creation of incremental materialized views"""
    spec = PivotSpec(
        table="test_table",
        rows=["region", "product"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    
    # Mock the database execute call to avoid actual table creation
    mock_backend.execute = Mock(return_value=True)
    incremental_view_manager.database = mock_backend
    
    view_name = await incremental_view_manager.create_incremental_view(spec)
    
    assert isinstance(view_name, str)
    assert view_name.startswith("mv_")
    assert spec.table in incremental_view_manager.views


@pytest.mark.asyncio
async def test_update_view_incrementally(incremental_view_manager, mock_backend):
    """Test incremental view updates"""
    spec = PivotSpec(
        table="test_table",
        rows=["region"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    
    # Add view to manager (simulate it exists)
    incremental_view_manager.views["test_table"] = {
        'name': 'mv_test_table_12345',
        'spec': spec,
        'last_updated': 0,
        'dependencies': ['test_table']
    }
    
    # Test incremental update with changes
    changes = [
        {"type": "INSERT", "new_row": {"region": "North", "sales": 100}},
        {"type": "UPDATE", "old_row": {"region": "South", "sales": 50}, "new_row": {"region": "South", "sales": 75}},
        {"type": "DELETE", "old_row": {"region": "East", "sales": 25}}
    ]
    
    # This should run without errors
    await incremental_view_manager.update_view_incrementally("test_table", changes)
    
    assert True


@pytest.mark.asyncio
async def test_stream_processor_update_materialized_view_incrementally(stream_processor):
    """Test stream processor's update of materialized views"""
    # This tests the internal method that updates materialized views
    await stream_processor._update_materialized_view_incrementally("test_job", {"agg": "sum", "field": "sales"}, {"test": "data"})
    
    # Should not raise an exception
    assert True


@pytest.mark.asyncio
async def test_stream_aggregation_with_complex_spec(stream_processor):
    """Test stream aggregation with more complex pivot specification"""
    spec = PivotSpec(
        table="complex_table",
        rows=["region", "product", "category"],
        measures=[
            Measure(field="sales", agg="sum", alias="total_sales"),
            Measure(field="quantity", agg="avg", alias="avg_quantity")
        ],
        filters=[{"field": "year", "op": "=", "value": 2023}]
    )
    
    job_id = await stream_processor.create_real_time_aggregation_job(spec)
    
    assert job_id in stream_processor.aggregation_jobs
    job_config = stream_processor.aggregation_jobs[job_id]
    
    assert job_config['table'] == 'complex_table'
    assert len(job_config['measures']) == 2


@pytest.mark.asyncio
async def test_incremental_view_manager_handles_multiple_tables(incremental_view_manager, mock_backend):
    """Test incremental view manager with multiple tables"""
    specs = [
        PivotSpec(
            table="table1",
            rows=["col1"],
            measures=[Measure(field="val1", agg="sum", alias="sum_val1")],
            filters=[]
        ),
        PivotSpec(
            table="table2", 
            rows=["col2"],
            measures=[Measure(field="val2", agg="count", alias="count_val2")],
            filters=[]
        )
    ]
    
    incremental_view_manager.database = mock_backend
    
    # Create views for multiple tables
    for spec in specs:
        view_name = await incremental_view_manager.create_incremental_view(spec)
        assert view_name.startswith("mv_")
        assert spec.table in incremental_view_manager.views


@pytest.mark.asyncio
async def test_stream_processor_process_various_operations(stream_processor):
    """Test stream processor with various CRUD operations"""
    spec = PivotSpec(
        table="test_table",
        rows=["region"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    
    await stream_processor.create_real_time_aggregation_job(spec)
    
    # Test different operations
    operations = ["INSERT", "UPDATE", "DELETE"]
    
    for op in operations:
        record = {"region": "TestRegion", "sales": 100} if op != "DELETE" else {"region": "TestRegion"}
        old_record = {"region": "TestRegion", "sales": 50} if op == "UPDATE" else None
        
        stream_data = {"type": op}
        if op == "INSERT":
            stream_data["new_row"] = record
        elif op == "UPDATE":
            stream_data["old_row"] = old_record
            stream_data["new_row"] = record
        elif op == "DELETE":
            stream_data["old_row"] = record
        
        await stream_processor.process_stream_update("test_table", stream_data, op)


if __name__ == "__main__":
    pytest.main([__file__])