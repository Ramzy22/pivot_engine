import asyncio
import time
from typing import AsyncGenerator, Any, Dict, List, Optional
import ibis
from ibis.backends.base import Backend as IbisBaseBackend
from pivot_engine.streaming.streaming_processor import IncrementalMaterializedViewManager
from pivot_engine.cdc.database_change_detector import DatabaseChangeDetector, TableSnapshot
from pivot_engine.cdc.models import Change


class PivotCDCManager:
    def __init__(self, database: IbisBaseBackend, change_stream: Optional[AsyncGenerator[Change, None]] = None):
        self.database = database
        self.change_stream = change_stream
        self.checkpoints = {}
        self.running = False
        self.change_processors = []
        self.materialized_view_managers = {}
        self.database_change_detector = DatabaseChangeDetector(database) # Use the dedicated detector

    async def setup_cdc(self, table_name: str):
        """Set up CDC for a specific table"""
        # Create a changes tracking table if it doesn't exist
        changes_table_name = f"{table_name}_changes_tracking"
        
        # Define the schema for the tracking table using Ibis types
        schema = ibis.schema([
            ('id', 'int64'), # Use BIGINT
            ('operation', 'string'), # Use VARCHAR
            ('timestamp', 'timestamp'), # Use TIMESTAMP
            ('processed', 'boolean') # Use BOOLEAN
        ])
        
        try:
            # Check if table exists before creating to replicate "IF NOT EXISTS"
            if changes_table_name not in self.database.list_tables():
                self.database.create_table(changes_table_name, schema)
        except Exception as e:
            print(f"Could not create changes tracking table '{changes_table_name}': {e}")
            pass

        # Initialize checkpoint data
        self.checkpoints[table_name] = {
            'setup_time': asyncio.get_event_loop().time(),
            'last_processed': 0,
            'active': True,
            'tracking_table': changes_table_name
        }
        await self.database_change_detector.start_tracking_table(table_name) # Start detector for this table

    def register_materialized_view_manager(self, table_name: str, manager: IncrementalMaterializedViewManager):
        """Register a materialized view manager to receive change notifications"""
        if table_name not in self.materialized_view_managers:
            self.materialized_view_managers[table_name] = []
        self.materialized_view_managers[table_name].append(manager)

    def register_change_processor(self, processor_func):
        """Register a function to process changes"""
        self.change_processors.append(processor_func)

    async def track_changes(self, table_name: str):
        """Track and process data changes in real-time"""
        if not self.running:
            self.running = True

        if self.change_stream:
            # Use predefined stream if available (e.g., for testing)
            async for change in self.change_stream:
                if change.table == table_name and self.checkpoints.get(table_name, {}).get('active', False):
                    await self._process_change(change)
                    await self._update_affected_cache_keys(change)
                    await self._update_materialized_views(change)
                    for processor in self.change_processors:
                        try:
                            await processor(change)
                        except Exception as e:
                            print(f"Error in change processor: {e}")
        else:
            # Use the DatabaseChangeDetector for polling-based changes
            async for change in self.database_change_detector.get_change_stream(table_name):
                if self.checkpoints.get(table_name, {}).get('active', False):
                    await self._process_change(change)
                    await self._update_affected_cache_keys(change)
                    await self._update_materialized_views(change)
                    for processor in self.change_processors:
                        try:
                            await processor(change)
                        except Exception as e:
                            print(f"Error in change processor: {e}")

    async def _process_change(self, change: Change):
        """Process individual changes and update pivot structures"""
        if change.type == 'INSERT':
            await self._incremental_insert(change.table, change.new_row)
        elif change.type == 'UPDATE':
            await self._incremental_update(change.table, change.old_row, change.new_row)
        elif change.type == 'DELETE':
            await self._incremental_delete(change.table, change.old_row)

    async def _incremental_insert(self, table_name: str, new_row: Dict):
        """Handle incremental insert by updating affected aggregations"""
        for manager in self.materialized_view_managers.get(table_name, []):
            try:
                await manager.process_incremental_change({
                    'table': table_name,
                    'operation': 'INSERT',
                    'old_row': None,
                    'new_row': new_row
                })
            except Exception as e:
                print(f"Error updating materialized view for insert: {e}")

    async def _incremental_update(self, table_name: str, old_row: Dict, new_row: Dict):
        """Handle incremental update by adjusting affected aggregations"""
        for manager in self.materialized_view_managers.get(table_name, []):
            try:
                await manager.process_incremental_change({
                    'table': table_name,
                    'operation': 'UPDATE',
                    'old_row': old_row,
                    'new_row': new_row
                })
            except Exception as e:
                print(f"Error updating materialized view for update: {e}")

    async def _incremental_delete(self, table_name: str, old_row: Dict):
        """Handle incremental delete by removing from aggregations"""
        for manager in self.materialized_view_managers.get(table_name, []):
            try:
                await manager.process_incremental_change({
                    'table': table_name,
                    'operation': 'DELETE',
                    'old_row': old_row,
                    'new_row': None
                })
            except Exception as e:
                print(f"Error updating materialized view for delete: {e}")

    async def _update_affected_cache_keys(self, change: Change):
        """Update affected cache keys by identifying and invalidating specific cached queries"""
        if hasattr(self.database, 'cache') and self.database.cache:
            cache = self.database.cache
            # The cache object must implement _find_and_invalidate_affected_cache_keys
            if hasattr(cache, '_find_and_invalidate_affected_cache_keys'):
                await cache._find_and_invalidate_affected_cache_keys(change.table)
            else:
                print(f"Warning: Cache backend does not support granular invalidation. Full table cache will be cleared.")
                # Fallback to clearing table cache if granular invalidation is not supported
                if hasattr(cache, 'clear_table_cache'):
                    await cache.clear_table_cache(change.table) if asyncio.iscoroutinefunction(cache.clear_table_cache) else cache.clear_table_cache(change.table)

    async def _update_materialized_views(self, change: Change):
        """Update materialized views based on changes"""
        table_name = change.table
        if table_name in self.materialized_view_managers:
            changes_list = [change]  # List of changes to process
            for manager in self.materialized_view_managers[table_name]:
                try:
                    await manager.update_view_incrementally(f"mv_{table_name}", changes_list)
                except Exception as e:
                    print(f"Error updating materialized view for {table_name}: {e}")

    def stop_tracking(self):
        """Stop the CDC tracking"""
        self.running = False

    def get_cdc_status(self, table_name: str) -> Dict[str, Any]:
        """Get the status of CDC for a specific table"""
        return self.checkpoints.get(table_name, {})