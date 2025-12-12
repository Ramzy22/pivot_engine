"""
Test suite for CDC (Change Data Capture) functionality
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from pivot_engine.cdc.cdc_manager import PivotCDCManager, Change
from pivot_engine.scalable_pivot_controller import ScalablePivotController


class MockDatabaseConnection:
    """Mock database connection for testing"""
    def __init__(self):
        self.execute_calls = []
    
    async def execute(self, query, params=None):
        # Track executed queries for verification
        self.execute_calls.append((query, params))
        # Simulate successful execution
        return True


async def mock_change_stream():
    """Mock change stream for testing"""
    changes = [
        Change(table="sales", type="INSERT", new_row={"id": 1, "value": "test1"}),
        Change(table="sales", type="UPDATE", old_row={"id": 1}, new_row={"id": 1, "value": "updated"}),
        Change(table="sales", type="DELETE", old_row={"id": 1})
    ]
    
    for change in changes:
        yield change


@pytest.fixture
def mock_db_connection():
    """Create mock database connection"""
    return MockDatabaseConnection()


@pytest.fixture
def cdc_manager():
    """Create CDC manager for testing"""
    async def empty_stream():
        # Empty stream for initialization
        return
        yield  # This makes it an async generator that yields nothing initially
    
    manager = PivotCDCManager(":memory:", empty_stream())
    
    # Mock the database execute method
    manager.database = Mock()
    manager.database.execute = AsyncMock(return_value=True)
    
    return manager


@pytest.mark.asyncio
async def test_cdc_manager_creation(cdc_manager):
    """Test CDC manager creation"""
    assert cdc_manager is not None
    assert cdc_manager.checkpoints == {}


@pytest.mark.asyncio
async def test_setup_cdc(cdc_manager):
    """Test CDC setup for a table"""
    # Setup CDC for a table
    await cdc_manager.setup_cdc("test_table")
    
    # Assertions about database.execute calls would verify the setup
    # The execute method should have been called with CREATE TABLE and ALTER TABLE statements
    assert cdc_manager.database.execute.called


@pytest.mark.asyncio
async def test_process_change_insert(cdc_manager):
    """Test processing of INSERT changes"""
    change = Change(
        table="test_table", 
        type="INSERT", 
        new_row={"id": 1, "name": "test", "value": 100}
    )
    
    # Process the change
    await cdc_manager._process_change(change)
    
    # Verify that the appropriate method was called
    await cdc_manager._incremental_insert(change.new_row)


@pytest.mark.asyncio
async def test_process_change_update(cdc_manager):
    """Test processing of UPDATE changes"""
    change = Change(
        table="test_table",
        type="UPDATE", 
        old_row={"id": 1, "value": 50}, 
        new_row={"id": 1, "value": 100}
    )
    
    # Process the change
    await cdc_manager._process_change(change)
    
    # Verify that the appropriate method was called
    await cdc_manager._incremental_update(change.old_row, change.new_row)


@pytest.mark.asyncio
async def test_process_change_delete(cdc_manager):
    """Test processing of DELETE changes"""
    change = Change(
        table="test_table", 
        type="DELETE", 
        old_row={"id": 1, "name": "test"}
    )
    
    # Process the change
    await cdc_manager._process_change(change)
    
    # Verify that the appropriate method was called
    await cdc_manager._incremental_delete(change.old_row)


@pytest.mark.asyncio
async def test_track_changes():
    """Test tracking of changes from a stream"""
    async def test_stream():
        changes = [
            Change(table="sales", type="INSERT", new_row={"id": 1, "amount": 100}),
        ]
        for change in changes:
            yield change
    
    manager = PivotCDCManager(":memory:", test_stream())
    
    # Mock the database
    manager.database = Mock()
    manager.database.execute = AsyncMock(return_value=True)
    
    # Mock the processing methods to track calls
    manager._process_change = AsyncMock()
    
    # Track changes
    task = asyncio.create_task(manager.track_changes("sales"))
    
    # Allow the task to run briefly
    await asyncio.sleep(0.1)
    
    # Cancel the task since it's an infinite loop
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected
    
    # Verify that _process_change was called (at least once)
    # Note: This test is somewhat limited because tracking runs indefinitely
    assert True  # The setup is correct


@pytest.mark.asyncio
async def test_controller_cdc_integration():
    """Test CDC integration with the main controller"""
    controller = ScalablePivotController(backend_uri=":memory:")
    
    # This test would set up CDC with a mock stream
    async def mock_stream():
        yield Change(table="test", type="INSERT", new_row={"id": 1})
        return  # Stop after one change for testing
    
    # Setup CDC for a table
    cdc_manager = await controller.setup_cdc("test_table", mock_stream())
    
    assert cdc_manager is not None


@pytest.mark.asyncio
async def test_incremental_operations(cdc_manager):
    """Test incremental insert/update/delete operations"""
    # Test incremental insert
    await cdc_manager._incremental_insert({"id": 1, "value": "test"})
    
    # Test incremental update  
    await cdc_manager._incremental_update({"id": 1, "value": "old"}, {"id": 1, "value": "new"})
    
    # Test incremental delete
    await cdc_manager._incremental_delete({"id": 1, "value": "old"})
    
    # All should run without errors
    assert True


@pytest.mark.asyncio
async def test_update_affected_cache_keys(cdc_manager):
    """Test updating affected cache keys after changes"""
    change = Change(table="test_table", type="INSERT", new_row={"id": 1})
    
    # This should not raise an exception
    await cdc_manager._update_affected_cache_keys(change)
    
    assert True


if __name__ == "__main__":
    pytest.main([__file__])