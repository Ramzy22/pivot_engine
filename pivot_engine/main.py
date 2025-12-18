"""
main.py - Main application entry point for the scalable pivot engine
"""
import asyncio
import json
from typing import Dict, Any, Optional

# FastAPI is optional for microservices
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    FASTAPI_AVAILABLE = False

from .scalable_pivot_controller import ScalablePivotController
from .config import get_config, ScalablePivotConfig
from .cdc.cdc_manager import Change


class ScalablePivotApplication:
    """Main application class that orchestrates all components"""

    def __init__(self, config: Optional[ScalablePivotConfig] = None):
        self.config = config or get_config()
        self.controller: Optional[ScalablePivotController] = None
        self.ui_service = None  # Will be imported as needed
        self.caching_service = None
        self.execution_service = None
        self.planning_service = None
        self.app = None

        self._setup_services()

    def _setup_services(self):
        """Setup all services"""
        # Create main controller
        self.controller = ScalablePivotController(
            backend_uri=self.config.backend_uri,
            cache=self.config.cache_type,
            cache_options=self.config.redis_config,
            enable_tiles=self.config.enable_tiles,
            enable_delta=self.config.enable_delta_updates,
            enable_streaming=self.config.enable_streaming,
            enable_incremental_views=self.config.enable_incremental_views,
            tile_size=self.config.tile_size
        )

        if FASTAPI_AVAILABLE:
            # Import microservices only if FastAPI is available
            from pivot_engine.pivot_microservices.ui_proxy.ui_proxy_service import UIPivotService
            from pivot_engine.pivot_microservices.caching.caching_service import CacheService
            from pivot_engine.pivot_microservices.execution.execution_service import ExecutionService
            from pivot_engine.pivot_microservices.planning.query_planning_service import QueryPlanningService

            # Create microservices
            service_config = {
                'default_ttl': self.config.default_cache_ttl,
                'l1_ttl': self.config.l1_cache_ttl,
                'cache_type': self.config.cache_type,
                'redis_config': self.config.redis_config,
                'compression': True
            }

            self.caching_service = CacheService(service_config)
            self.execution_service = ExecutionService(service_config)
            self.planning_service = QueryPlanningService(service_config)

            # Create UI service
            self.ui_service = UIPivotService(self.controller)

            # Setup FastAPI app
            self.app = self.ui_service.get_app()
            self._setup_app_routes()

    def _setup_app_routes(self):
        """Setup additional routes for the main app"""
        if not self.app or not FASTAPI_AVAILABLE:
            return

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "scalable-pivot-engine"}

        @self.app.post("/pivot")
        async def pivot_endpoint(spec: Dict[str, Any]):
            try:
                result = await self.handle_pivot_request(spec)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/hierarchical")
        async def hierarchical_endpoint(spec: Dict[str, Any],
                                       expanded_paths: Optional[list] = None,
                                       user_preferences: Optional[dict] = None):
            try:
                expanded_paths = expanded_paths or []
                user_preferences = user_preferences or {}

                result = await self.handle_hierarchical_request(spec, expanded_paths, user_preferences)
                return JSONResponse(content=result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    async def handle_pivot_request(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a pivot request"""
        # Use caching to get or compute result
        cache_key = f"pivot:{hash(str(spec))}"
        
        async def compute_pivot():
            return self.controller.run_pivot(spec)
        
        result = await self.caching_service.get_or_compute(
            cache_key, 
            compute_pivot,
            ttl=self.config.default_cache_ttl
        )
        
        return {
            "status": "success",
            "data": result,
            "cached": True  # Simplified - would check actual cache status
        }
    
    async def handle_hierarchical_request(self, spec: Dict[str, Any], 
                                        expanded_paths: list, 
                                        user_preferences: dict) -> Dict[str, Any]:
        """Handle a hierarchical pivot request"""
        # Determine if enhanced hierarchical processing is needed
        if user_preferences.get('enable_pruning', False) or user_preferences.get('use_materialized_hierarchies', False):
            # Use the enhanced hierarchical methods
            from .types.pivot_spec import PivotSpec
            pivot_spec = PivotSpec.from_dict(spec)
            
            if user_preferences.get('enable_pruning', False):
                result = await self.controller.run_pruned_hierarchical_pivot(
                    pivot_spec, expanded_paths, user_preferences
                )
            elif user_preferences.get('use_materialized_hierarchies', False):
                # First create materialized hierarchies if they don't exist
                await self.controller.run_materialized_hierarchy(pivot_spec)
                # Then get the optimized data
                result = await self.controller.materialized_hierarchy_manager.get_optimized_hierarchical_data(
                    pivot_spec, expanded_paths
                )
            else:
                result = self.controller.run_hierarchical_pivot(spec)
        else:
            result = self.controller.run_hierarchical_pivot(spec)
        
        return {
            "status": "success", 
            "data": result,
            "expanded_paths": expanded_paths
        }
    
    async def setup_cdc_for_table(self, table_name: str):
        """Setup CDC for a table to enable real-time updates"""
        if not self.config.enable_cdc:
            return {"status": "disabled", "message": "CDC is not enabled"}
        
        # Create a mock change stream for demonstration
        async def mock_change_stream():
            # In a real implementation, this would connect to a real change stream
            # such as database WAL, Kafka, etc.
            for i in range(10):  # Simulate 10 changes
                change = Change(
                    table=table_name,
                    type='INSERT',  # Could be INSERT, UPDATE, DELETE
                    new_row={'id': i, 'data': f'value_{i}'}
                )
                yield change
                await asyncio.sleep(1)  # Simulate time between changes
        
        # Setup CDC
        cdc_manager = await self.controller.setup_cdc(table_name, mock_change_stream())
        
        return {
            "status": "success",
            "message": f"CDC setup complete for table {table_name}",
            "manager_id": id(cdc_manager)
        }
    
    def get_app(self):
        """Get the FastAPI application instance"""
        if not FASTAPI_AVAILABLE:
            return None
        if not self.app:
            self._setup_services()
        return self.app if self.app else FastAPI(title="Scalable Pivot Engine")


def create_app():
    """Create and return the FastAPI application"""
    config = get_config()
    app_instance = ScalablePivotApplication(config)
    return app_instance.get_app()


# Example usage
async def main():
    """Example main function showing how to use the scalable pivot engine"""
    print("Starting Scalable Pivot Engine...")
    
    # Create the application
    app = ScalablePivotApplication()
    
    # Example: Handle a simple pivot request
    spec = {
        "table": "sales",
        "rows": ["region", "product"],
        "measures": [
            {"field": "amount", "agg": "sum", "alias": "total_sales"}
        ],
        "filters": [],
        "sort": [{"field": "region", "order": "asc"}]
    }
    
    result = await app.handle_pivot_request(spec)
    print(f"Pivot result: {result}")
    
    # Example: Setup CDC for real-time updates
    cdc_result = await app.setup_cdc_for_table("sales")
    print(f"CDC setup: {cdc_result}")
    
    # Example: Hierarchical request with pruning
    hierarchical_result = await app.handle_hierarchical_request(
        spec, 
        [["North"], ["South"]],  # Expanded paths
        {"enable_pruning": True, "pruning_strategy": "top_n", "top_n": 10}
    )
    print(f"Hierarchical result: {hierarchical_result}")


if __name__ == "__main__":
    asyncio.run(main())