"""
Database Change Detection System for Pivot Engine CDC

This module provides mechanisms to detect database changes and produce change streams.
For systems that don't have native CDC, it provides polling-based change detection.
"""
import asyncio
import time
from typing import Dict, Any, AsyncGenerator, Optional, List
from dataclasses import dataclass
from pivot_engine.cdc.models import Change


@dataclass
class TableSnapshot:
    """Represents a snapshot of table data at a point in time"""
    table_name: str
    row_count: int
    checksum: str # Using string for consistency with previous implementation
    timestamp: float
    sample_data: List[Dict[str, Any]]  # Contains sample records for change detection
    max_id: Optional[int] = None
    max_updated_at: Optional[float] = None


class DatabaseChangeDetector:
    """Detects changes in database tables and generates change events"""
    
    def __init__(self, backend):
        self.backend = backend  # This is an Ibis connection object
        self.table_snapshots: Dict[str, TableSnapshot] = {}
        self.change_queues: Dict[str, asyncio.Queue] = {}
        self.running = False
        
    async def start_tracking_table(self, table_name: str):
        """Start tracking changes for a specific table"""
        # Take initial snapshot
        initial_snapshot = await self._take_snapshot(table_name)
        self.table_snapshots[table_name] = initial_snapshot
        
        # Create change queue for the table
        if table_name not in self.change_queues:
            self.change_queues[table_name] = asyncio.Queue()
        
        print(f"Started tracking changes for table: {table_name}")
    
    async def stop_tracking_table(self, table_name: str):
        """Stop tracking changes for a specific table"""
        if table_name in self.table_snapshots:
            del self.table_snapshots[table_name]
        if table_name in self.change_queues:
            del self.change_queues[table_name]
    
    async def _take_snapshot(self, table_name: str) -> TableSnapshot:
        """Take a snapshot of the table data using Ibis expressions."""
        ibis_table = self.backend.table(table_name)
        
        # Get row count
        row_count = ibis_table.count().execute()
        
        # Get a simple checksum using row count for backend agnosticism.
        checksum = str(row_count)
        
        # Get sample data
        sample_data_pyarrow = ibis_table.limit(5).to_pyarrow()
        sample_data = sample_data_pyarrow.to_pylist() if sample_data_pyarrow.num_rows > 0 else []
        
        # Capture incremental markers if available
        max_id = None
        max_updated_at = None
        
        cols = ibis_table.columns
        if 'id' in cols:
            try:
                max_id = ibis_table['id'].max().execute()
            except: pass
            
        if 'updated_at' in cols:
            try:
                max_updated_at = ibis_table['updated_at'].max().execute()
            except: pass
        
        return TableSnapshot(
            table_name=table_name,
            row_count=row_count,
            checksum=checksum,
            timestamp=time.time(),
            sample_data=sample_data,
            max_id=max_id,
            max_updated_at=max_updated_at
        )
    
    async def detect_changes(self, table_name: str) -> List[Change]:
        """Detect changes for a tracked table by comparing with previous snapshot"""
        if table_name not in self.table_snapshots:
            await self.start_tracking_table(table_name)
        
        current_snapshot = await self._take_snapshot(table_name)
        previous_snapshot = self.table_snapshots[table_name]
        
        changes = []
        
        # Compare snapshots to detect changes
        if current_snapshot.checksum != previous_snapshot.checksum:
            # Data has changed - determine what changed
            if current_snapshot.row_count > previous_snapshot.row_count:
                # Likely INSERT operations
                changes.extend(await self._detect_inserts(table_name, previous_snapshot, current_snapshot))
            elif current_snapshot.row_count < previous_snapshot.row_count:
                # Likely DELETE operations
                changes.extend(await self._detect_deletions(table_name, previous_snapshot, current_snapshot))
            else:
                # Row count is same but data changed - likely UPDATE operations
                # The snapshot method cannot granularly detect updates without row data
                changes.append(Change(
                    table=table_name,
                    type='UPDATE',
                    old_row={"_change_type": "detected_update", "_timestamp": time.time()},
                    new_row={"_change_type": "detected_update", "_timestamp": time.time()}
                ))
        
        # Update the stored snapshot
        self.table_snapshots[table_name] = current_snapshot
        
        return changes
    
    async def _detect_inserts(self, table_name: str, old_snapshot: TableSnapshot, new_snapshot: TableSnapshot) -> List[Change]:
        """Detect INSERT operations"""
        changes = []
        ibis_table = self.backend.table(table_name)
        
        # Try to fetch actual inserted rows using incremental keys
        new_rows = []
        
        if old_snapshot.max_id is not None and new_snapshot.max_id is not None:
            if new_snapshot.max_id > old_snapshot.max_id:
                try:
                    new_rows_table = ibis_table.filter(ibis_table['id'] > old_snapshot.max_id).to_pyarrow()
                    new_rows = new_rows_table.to_pylist()
                except Exception as e:
                    print(f"Error fetching incremental inserts by ID: {e}")
                    
        elif old_snapshot.max_updated_at is not None and new_snapshot.max_updated_at is not None:
            if new_snapshot.max_updated_at > old_snapshot.max_updated_at:
                try:
                    new_rows_table = ibis_table.filter(ibis_table['updated_at'] > old_snapshot.max_updated_at).to_pyarrow()
                    new_rows = new_rows_table.to_pylist()
                except Exception as e:
                    print(f"Error fetching incremental inserts by updated_at: {e}")

        # If we successfully fetched real rows
        if new_rows:
            for row in new_rows:
                changes.append(Change(
                    table=table_name,
                    type='INSERT',
                    new_row=row
                ))
        else:
            # Fallback: report the difference in row count as placeholder INSERTs
            num_inserts = new_snapshot.row_count - old_snapshot.row_count
            if num_inserts > 0:
                for _ in range(num_inserts):
                    changes.append(Change(
                        table=table_name,
                        type='INSERT',
                        new_row={"_change_type": "detected_insert_placeholder", "_timestamp": time.time()}
                    ))
        
        return changes
    
    async def _detect_deletions(self, table_name: str, old_snapshot: TableSnapshot, new_snapshot: TableSnapshot) -> List[Change]:
        """Detect DELETE operations"""
        changes = []
        
        # For simplicity, we'll report the difference in row count as individual DELETEs
        num_deletes = old_snapshot.row_count - new_snapshot.row_count
        
        for _ in range(num_deletes):
            changes.append(Change(
                table=table_name,
                type='DELETE',
                old_row={"_change_type": "detected_delete", "_timestamp": time.time()}
            ))
        
        return changes
    
    async def _detect_updates(self, table_name: str, old_snapshot: TableSnapshot, new_snapshot: TableSnapshot) -> List[Change]:
        """Detect UPDATE operations"""
        changes = []
        
        # Since row counts are the same but data changed, we have updates
        # In a real implementation, identify the specific rows that changed
        changes.append(Change(
            table=table_name,
            type='UPDATE',
            old_row={"_change_type": "detected_update", "_timestamp": time.time()},
            new_row={"_change_type": "detected_update", "_timestamp": time.time()}
        ))
        
        return changes
    
    async def get_change_stream(self, table_name: str, poll_interval: float = 1.0) -> AsyncGenerator[Change, None]:
        """Generate a stream of changes for a table using polling"""
        if table_name not in self.change_queues:
            await self.start_tracking_table(table_name)
        
        while True:
            changes = await self.detect_changes(table_name)
            for change in changes:
                yield change
            await asyncio.sleep(poll_interval)
    
    async def track_table_changes(self, table_name: str, callback_func=None) -> AsyncGenerator[Change, None]:
        """Continuously track and yield changes for a table"""
        async for change in self.get_change_stream(table_name):
            if callback_func:
                try:
                    await callback_func(change)
                except Exception as e:
                    print(f"Error in change callback: {e}")
            yield change


class DatabaseChangeProducer:
    """Produces change streams for multiple tables"""

    def __init__(self, backend):
        self.backend = backend  # This should be a DuckDBBackend instance
        self.detector = DatabaseChangeDetector(backend)
        self.table_trackers = {}
        
    async def setup_change_stream(self, table_name: str) -> AsyncGenerator[Change, None]:
        """Set up a change stream for a specific table"""
        return self.detector.track_table_changes(table_name)
    
    async def register_table_for_cdc(self, table_name: str, callback_func=None):
        """Register a table for CDC tracking"""
        tracker_task = asyncio.create_task(
            self._track_table(table_name, callback_func)
        )
        self.table_trackers[table_name] = tracker_task
        return tracker_task
    
    async def _track_table(self, table_name: str, callback_func):
        """Internal method to track a table"""
        async for change in self.detector.track_table_changes(table_name, callback_func):
            # This will continuously yield changes
            pass  # The changes are handled by the callback